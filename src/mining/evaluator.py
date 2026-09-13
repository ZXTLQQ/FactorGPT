# -*- coding: utf-8 -*-
"""多维度评价体系（山西证券《算子网格搜索 + Numba 加速 + 多维度评价》落地）。

研报把"评价"从单一 IC 拆成**四个维度**，理由是：只看 IC 的搜索一定会过拟合。
本模块逐维实现，并给出一个可比的统一评分：

==========  ====================================================================
维度          指标
==========  ====================================================================
数据质量       有效覆盖、横截面退化比例、极端值比例、有效截面天数
预测能力       RankIC 均值/标准差/ICIR/t 值/胜率、Pearson IC、
             分组多空收益及其 t 值、分组单调性、因子换手
稳定性        分段 IC（同号比例、最差段）、滚动 IC 最低值、IC 衰减与半衰期、
             因子自相关（L1~L5）
相关性        与既有因子池的最大/平均 |相关|、对因子池复合因子的**增量 IC**
==========  ====================================================================

统一评分把四维归一化到 [0,1] 后加权，再扣掉**换手成本惩罚**与**表达式复杂度
惩罚**（山西研报的"简约性"要求：同样 IC，表达式越短越可信）。

天风的六道风险闸门不在这里重算，需要时通过 ``with_risk=True`` 调 ``risk`` 模块，
把风险视角的判据并入同一份报告（``report.detail["risk"]``）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import expr as ex
from . import ops
from . import risk as _risk
from .panel import PanelData

__all__ = [
    "EvalConfig", "FactorReport", "ICScreener", "evaluate", "evaluate_expr",
    "rank_ic", "pearson_ic", "ic_stats", "quantile_stats", "turnover",
    "ic_decay", "segmented_ic", "data_quality", "pool_correlation",
    "incremental_ic", "factor_corr_matrix", "rank_reports",
]


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
@dataclass
class EvalConfig:
    """评价口径（全部可调，报告里随结果一起落盘，保证可复现）。"""

    horizons: Sequence[int] = (1, 5, 20)
    primary_horizon: int = 5
    n_groups: int = 5
    min_stocks: int = 20
    rolling_ic_window: int = 60
    n_segments: int = 4
    periods_per_year: int = 244
    w_turnover_penalty: float = 0.20
    w_simplicity_penalty: float = 0.02
    dimension_weights: Dict[str, float] = dc_field(default_factory=lambda: {
        "predictive": 0.40, "stability": 0.25,
        "quality": 0.15, "correlation": 0.20})
    # 归一化区间：metric → (下限=0 分, 上限=1 分)
    scales: Dict[str, Tuple[float, float]] = dc_field(default_factory=lambda: {
        "abs_ic": (0.0, 0.06),
        "abs_icir": (0.0, 1.20),
        "ic_win": (0.50, 0.68),
        "ls_t": (0.0, 4.0),
        "monotonicity": (0.0, 1.0),
        "seg_win": (0.50, 1.00),
        "worst_seg_ic": (-0.02, 0.01),
        "rolling_ic_min": (-0.05, 0.02),
        "ac_l1": (0.30, 0.95),
        "coverage": (0.50, 1.00),
        "degenerate": (0.30, 0.00),
        "extreme": (0.10, 0.00),
        "pool_max_corr": (0.90, 0.20),
    })

    def scale(self, name: str, value: float) -> float:
        lo, hi = self.scales[name]
        if not np.isfinite(value):
            return 0.0
        v = (float(value) - lo) / (hi - lo) if hi != lo else 0.0
        return float(min(max(v, 0.0), 1.0))


# --------------------------------------------------------------------------
# 基元：IC / 分组 / 换手
# --------------------------------------------------------------------------
def _daily_corr(factor: pd.DataFrame, fwd: pd.DataFrame, method: str,
                min_stocks: int) -> pd.Series:
    """逐日横截面相关（**向量化**，无 Python 逐日循环）。

    ``method='spearman'`` 时两侧先做横截面排名再求相关。等价于逐日
    ``corrcoef``，但一次矩阵运算算完整个面板；网格搜索要跑上万条表达式，
    这里是必须的加速点（分组检验仍走 Python 循环，因为它只在入围因子上跑一次）。
    """
    out = ops.cs_corr(factor, fwd, rank=(method == "spearman"),
                      min_stocks=max(int(min_stocks), 5))
    return out.rename(f"{method}_ic")


class ICScreener:
    """批量筛选用的 IC 计算器：预排名前瞻收益，避免逐条表达式重复排序。

    网格搜索的筛选阶段只关心"这条表达式值不值得进多维度评价"，因此只算
    RankIC 与覆盖率。把 ``cs_rank(fwd)`` 预先算好并按持有期缓存，每条表达式
    的筛选成本从 O(排序) 降到 O(一次乘加)。
    """

    def __init__(self, panel: PanelData, min_stocks: int = 20) -> None:
        self.panel = panel
        self.min_stocks = int(min_stocks)
        self._ranked_fwd: Dict[int, pd.DataFrame] = {}

    def ranked_fwd(self, horizon: int) -> pd.DataFrame:
        if horizon not in self._ranked_fwd:
            self._ranked_fwd[horizon] = ops.cs_rank(self.panel.fwd(horizon))
        return self._ranked_fwd[horizon]

    def ic(self, values: pd.DataFrame, horizon: int = 5) -> pd.Series:
        # 只对候选值排序，前瞻收益的排名已预先算好
        return _daily_corr(ops.cs_rank(values), self.ranked_fwd(horizon),
                           "pearson", self.min_stocks)

    def screen(self, values: pd.DataFrame, horizon: int = 5,
               periods_per_year: int = 244, warmup: int = 0) -> Dict[str, float]:
        st = ic_stats(self.ic(values, horizon), periods_per_year)
        st.update(data_quality(values, warmup))
        return st


def rank_ic(factor: pd.DataFrame, fwd: pd.DataFrame,
            min_stocks: int = 20) -> pd.Series:
    """逐日 RankIC（Spearman）。"""
    return _daily_corr(factor, fwd, "spearman", min_stocks)


def pearson_ic(factor: pd.DataFrame, fwd: pd.DataFrame,
               min_stocks: int = 20) -> pd.Series:
    return _daily_corr(factor, fwd, "pearson", min_stocks)


def ic_stats(ic: pd.Series, periods_per_year: int = 244) -> Dict[str, float]:
    """IC 序列的汇总统计（含 t 值与年化 ICIR）。"""
    x = ic.dropna()
    n = len(x)
    if n < 3:
        return {"ic_mean": 0.0, "ic_std": float("nan"), "icir": 0.0, "ic_t": 0.0,
                "ic_win": float("nan"), "icr_annual": 0.0, "n_days": float(n)}
    mean = float(x.mean())
    std = float(x.std(ddof=1))
    icir = mean / std if std > 0 else 0.0
    return {
        "ic_mean": mean,
        "ic_std": std,
        "icir": float(icir),
        "ic_t": float(icir * math.sqrt(n)),
        "ic_win": float((x > 0).mean()),
        "icr_annual": float(icir) * math.sqrt(periods_per_year),
        "n_days": float(n),
    }


def _mean_ignore_nan(x: np.ndarray, axis: Optional[int] = None):
    """忽略 NaN 的均值；整片全 NaN 时返回 NaN 而**不抛 RuntimeWarning**。

    ``np.nanmean`` 在空切片上会告警（``Mean of empty slice``）。分组检验里
    预热期、极窄截面必然出现空组，这是正常的数据状态而非异常，因此这里用
    有效点计数显式判断：非空走普通除法，全空给 NaN。
    """
    a = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(a)
    cnt = ok.sum(axis=axis)
    tot = np.where(ok, a, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = tot / cnt
    if np.ndim(out) == 0:
        return float(out) if cnt else float("nan")
    out = np.asarray(out, dtype=np.float64)
    out[np.asarray(cnt) == 0] = np.nan
    return out


def quantile_stats(factor: pd.DataFrame, fwd: pd.DataFrame, n_groups: int = 5,
                   min_stocks: int = 20) -> Dict[str, Any]:
    """分组收益：单调性、多空收益与多空 t 值（研报里的分组检验）。

    分组按**当日横截面排名**划分，因此天然对量纲不敏感；组收益取未来 h 日
    等权收益，多空 = 最高组 − 最低组。
    """
    rank = ops.cs_rank(factor).to_numpy(dtype=np.float64)
    y = fwd.to_numpy(dtype=np.float64)
    n_dates = rank.shape[0]
    day_ret = np.full((n_dates, n_groups), np.nan, dtype=np.float64)
    ls = np.full(n_dates, np.nan, dtype=np.float64)
    for i in range(n_dates):
        a, b = rank[i], y[i]
        m = np.isfinite(a) & np.isfinite(b)
        if int(m.sum()) < max(min_stocks, n_groups * 2):
            continue
        av, bv = a[m], b[m]
        # 用排名分箱，避免并列导致的空组
        idx = np.minimum((av * n_groups).astype(np.int64), n_groups - 1)
        for g in range(n_groups):
            sel = bv[idx == g]
            if sel.size:
                day_ret[i, g] = float(sel.mean())
        if np.isfinite(day_ret[i, 0]) and np.isfinite(day_ret[i, -1]):
            ls[i] = day_ret[i, -1] - day_ret[i, 0]

    group_mean = _mean_ignore_nan(day_ret, 0) if n_dates else np.zeros(n_groups)
    ls_v = ls[np.isfinite(ls)]
    ls_mean = float(ls_v.mean()) if ls_v.size else 0.0
    ls_t = (float(ls_v.mean() / (ls_v.std(ddof=1) / math.sqrt(ls_v.size)))
            if ls_v.size > 2 and ls_v.std(ddof=1) > 0 else 0.0)
    mono = 0.0
    if np.isfinite(group_mean).all() and np.std(group_mean) > 0:
        mono = float(np.corrcoef(np.arange(n_groups), group_mean)[0, 1])
    return {
        "group_returns": [float(g) for g in np.nan_to_num(group_mean)],
        "group_ls_mean": ls_mean,
        "group_ls_t": float(ls_t),
        "monotonicity": mono,
        "top_share": float(_mean_ignore_nan(day_ret[:, -1])) if n_dates else 0.0,
        "bottom_share": float(_mean_ignore_nan(day_ret[:, 0])) if n_dates else 0.0,
    }


def turnover(factor: pd.DataFrame, n_groups: Optional[int] = None) -> float:
    """因子换手（横截面排名变化的平均幅度，∈[0,1]，越大越贵）。

    口径：相邻两期排名绝对变化均值 / 截面宽度。这个口径对因子取值尺度不敏感，
    可与分组组合的换手成本直接挂钩。
    """
    r = ops.cs_rank(factor).to_numpy(dtype=np.float64)
    tot, cnt = 0.0, 0
    for i in range(1, r.shape[0]):
        a, b = r[i], r[i - 1]
        m = np.isfinite(a) & np.isfinite(b)
        if int(m.sum()) < 10:
            continue
        tot += float(np.abs(a[m] - b[m]).mean())
        cnt += 1
    return float(tot / cnt) if cnt else float("nan")


def ic_decay(factor: pd.DataFrame, panel: PanelData,
             horizons: Sequence[int], min_stocks: int = 20) -> Dict[str, float]:
    """IC 随持有期变化，并在 |IC| **单调衰减**时给出指数半衰期。

    ``decay_monotone`` 是必要的前提标记：若 |IC| 随持有期先降后升（典型的
    "短期反转 + 长期动量"叠加因子），半衰期没有意义，此时返回 NaN 而不是
    硬拟合出一个假的数字。
    """
    out: Dict[str, float] = {}
    for h in horizons:
        s = ic_stats(rank_ic(factor, panel.fwd(h), min_stocks))
        out[f"ic_h{h}"] = s["ic_mean"]
        out[f"icir_h{h}"] = s["icir"]
    hs = [h for h in horizons if h > 0]
    out["decay_lambda"] = float("nan")
    out["decay_halflife"] = float("nan")
    out["decay_monotone"] = float("nan")
    if len(hs) >= 2:
        y = np.array([abs(out[f"ic_h{h}"]) for h in hs])
        mono = bool(np.all(y[:-1] >= y[1:]))
        out["decay_monotone"] = 1.0 if mono else 0.0
        if mono and np.all(y > 0):
            slope = float(np.polyfit(np.log(hs), np.log(y), 1)[0])
            out["decay_lambda"] = -slope
            if slope < 0:
                out["decay_halflife"] = float(math.log(2.0) / -slope)
            else:
                out["decay_halflife"] = float("inf")
    return out


def segmented_ic(ic: pd.Series, n_segments: int = 4) -> Dict[str, Any]:
    """把 IC 序列等分成若干段（模拟不同市场环境）后的稳定性。"""
    x = ic.dropna()
    if len(x) < n_segments * 5:
        return {"segment_means": [], "seg_win": float("nan"),
                "worst_seg_ic": float("nan"), "seg_std": float("nan")}
    parts = np.array_split(x.to_numpy(dtype=np.float64), n_segments)
    means = [float(p.mean()) for p in parts if p.size]
    return {
        "segment_means": means,
        "seg_win": float(np.mean(np.array(means) > 0)) if means else float("nan"),
        "worst_seg_ic": float(np.min(means)) if means else float("nan"),
        "seg_std": float(np.std(means)) if means else float("nan"),
    }


def data_quality(factor: pd.DataFrame, warmup: int = 0) -> Dict[str, float]:
    """数据质量维度：覆盖率、退化截面比例、极端值比例、有效截面天数。

    ``warmup``（表达式的必然前置窗口，见 ``expr.lookback``）会额外给出
    ``coverage_adj``：**扣除预热期**后的覆盖率。判死一条长回看因子之前，
    应该看的是这个数而不是原始覆盖率——原始覆盖率低可能只是"数据不够长"。
    """
    arr = factor.to_numpy(dtype=np.float64)
    n, k = arr.shape
    finite = np.isfinite(arr)
    cov = float(finite.mean()) if arr.size else 0.0
    lb = int(max(0, min(warmup, n - 1)))
    cov_adj = float(finite[lb:].mean()) if arr.size and lb < n else cov
    deep = 0          # 横截面退化为常数（或有效样本过少）的天数
    extreme = 0.0
    stds: List[float] = []
    for i in range(n):
        row = arr[i][finite[i]]
        if row.size < 5 or float(row.std()) <= 1e-12:
            deep += 1
            continue
        stds.append(float(row.std()))
        extreme += float((np.abs(row) > 10.0).mean())
    return {
        "coverage": cov,
        "coverage_adj": cov_adj,
        "warmup": float(warmup),
        "degenerate": float(deep / n) if n else 1.0,
        "extreme": float(extreme / n) if n else 1.0,
        "n_valid_days": float(n),
        "mean_cs_std": float(np.mean(stds)) if stds else 0.0,
    }


def pool_correlation(factor: pd.DataFrame,
                     pool: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """与既有因子池的横截面 |相关|（Spearman 口径）。"""
    out: Dict[str, float] = {"pool_max_corr": 0.0, "pool_mean_corr": 0.0,
                             "pool_max_name": ""}
    if not pool:
        return out
    per: List[Tuple[str, float]] = []
    for name, other in pool.items():
        if other is None or name == "factor":
            continue
        r = rank_ic(factor, other, min_stocks=10).abs()
        per.append((name, float(r.mean())))
    per = [(n, v) for n, v in per if np.isfinite(v)]
    if per:
        best = max(per, key=lambda kv: kv[1])
        out["pool_max_corr"] = best[1]
        out["pool_max_name"] = best[0]
        out["pool_mean_corr"] = float(np.mean([v for _, v in per]))
    return out


def incremental_ic(factor: pd.DataFrame, pool: Dict[str, pd.DataFrame],
                   fwd: pd.DataFrame, min_stocks: int = 20) -> Dict[str, float]:
    """增量信息：因子池等权复合后的 IC，与把候选加入后的 IC 之差。

    这是研报里"相关性维度"真正关心的东西 —— 与存量因子相关高不高不重要，
    重要的是**加了它以后还有没有增量**。
    """
    keys = [k for k, v in pool.items() if v is not None]
    if not keys:
        return {"pool_ic": 0.0, "pool_plus_ic": 0.0, "incremental_ic": 0.0}
    comp = sum(ops.cs_zscore(pool[k].reindex_like(fwd)) for k in keys) / len(keys)
    pool_ic = float(rank_ic(comp, fwd, min_stocks).mean())
    both = ops.cs_zscore(comp) + ops.cs_zscore(factor.reindex_like(fwd))
    plus_ic = float(rank_ic(both, fwd, min_stocks).mean())
    return {"pool_ic": pool_ic, "pool_plus_ic": plus_ic,
            "incremental_ic": plus_ic - pool_ic}


def factor_corr_matrix(factors: Dict[str, pd.DataFrame],
                       min_stocks: int = 20) -> pd.DataFrame:
    """因子池两两横截面 |相关| 矩阵（用于发现"换个名字的同一个因子"）。"""
    names = [k for k, v in factors.items() if v is not None]
    mat = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            v = float(rank_ic(factors[a], factors[b], min_stocks).abs().mean())
            mat.loc[a, b] = mat.loc[b, a] = 0.0 if not np.isfinite(v) else v
    return mat


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
@dataclass
class FactorReport:
    """单个因子的完整体检报告。"""

    name: str
    expression: str
    metrics: Dict[str, float] = dc_field(default_factory=dict)
    scores: Dict[str, float] = dc_field(default_factory=dict)
    score: float = 0.0
    grade: str = "D"
    detail: Dict[str, Any] = dc_field(default_factory=dict)
    config: Optional[Dict[str, Any]] = None
    errors: List[str] = dc_field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_frame(self) -> pd.DataFrame:
        rows = dict(self.metrics)
        rows.update({f"score_{k}": v for k, v in self.scores.items()})
        rows["score"] = self.score
        return pd.DataFrame({"value": pd.Series(rows, dtype=float)})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "expression": self.expression,
            "score": self.score, "grade": self.grade,
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "scores": {k: float(v) for k, v in self.scores.items()},
            "errors": list(self.errors),
            "detail": {k: v for k, v in self.detail.items() if k != "factor"},
        }

    def summary(self) -> str:
        if self.errors:
            return f"[无效] {self.name}: {'; '.join(self.errors)}"
        g = self.metrics.get
        nan = float("nan")
        return (f"[{self.grade}] {self.score:.3f}  {self.name}\n"
                f"      RankIC={g('rank_ic_mean', nan):+.4f} "
                f"ICIR={g('rank_icir', nan):+.3f} "
                f"胜率={g('rank_ic_win', nan):.3f} "
                f"多空t={g('group_ls_t', nan):+.2f} "
                f"单调={g('monotonicity', nan):+.2f} "
                f"换手={g('turnover', nan):.3f} "
                f"池相关={g('pool_max_corr', nan):.2f} "
                f"分段胜={g('seg_win', nan):.2f}\n"
                f"      {self.expression}")


def _grade(score: float) -> str:
    return ("A" if score >= 0.75 else "B" if score >= 0.60
            else "C" if score >= 0.45 else "D")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def evaluate(factor: pd.DataFrame, panel: PanelData,
             name: str = "factor", expression: str = "",
             pool: Optional[Dict[str, pd.DataFrame]] = None,
             config: Optional[EvalConfig] = None,
             with_risk: bool = False,
             exposures: Optional[Dict[str, pd.DataFrame]] = None,
             node: Optional[ex.Node] = None) -> FactorReport:
    """一站式评价：四维指标 → 归一化 → 加权统一评分。"""
    cfg = config or EvalConfig()
    rep = FactorReport(name=name, expression=expression or name,
                       config={"horizons": list(cfg.horizons),
                               "n_groups": cfg.n_groups,
                               "min_stocks": cfg.min_stocks,
                               "dimension_weights": dict(cfg.dimension_weights)})
    m = rep.metrics

    q = data_quality(factor, ex.lookback(node) if node is not None else 0)
    m.update(q)
    if q["coverage_adj"] <= 0.0 or q["degenerate"] >= 0.999:
        rep.errors.append("因子几乎无有效取值（全 NaN 或逐日退化）")
        rep.detail["quality"] = q
        return rep

    h = int(cfg.primary_horizon)
    fwd = panel.fwd(h)
    ric = rank_ic(factor, fwd, cfg.min_stocks)
    m.update({f"rank_{k}": v for k, v in ic_stats(ric, cfg.periods_per_year).items()})
    pic = pearson_ic(factor, fwd, cfg.min_stocks)
    m["pearson_ic_mean"] = float(pic.dropna().mean()) if pic.notna().any() else 0.0
    m.update(quantile_stats(factor, fwd, cfg.n_groups, cfg.min_stocks))
    m["turnover"] = turnover(factor)
    m.update(_risk.autocorr(factor))
    m.update(ic_decay(factor, panel, cfg.horizons, cfg.min_stocks))
    m.update(segmented_ic(ric, cfg.n_segments))
    roll = ric.rolling(cfg.rolling_ic_window,
                       min_periods=max(10, cfg.rolling_ic_window // 3)).mean()
    m["rolling_ic_min"] = float(roll.min()) if roll.notna().any() else float("nan")
    m["rolling_ic_pos"] = (float((roll.dropna() > 0).mean())
                           if roll.notna().any() else float("nan"))

    if pool:
        m.update(pool_correlation(factor, pool))
        m.update(incremental_ic(factor, pool, fwd, cfg.min_stocks))
        m.setdefault("pool_max_corr", 0.0)
    m.setdefault("pool_max_corr", 0.0)
    m.setdefault("incremental_ic", 0.0)

    # 非数值指标（分组收益明细、池内最相关因子名等）挪到 detail，
    # 保证 metrics 是纯数值字典，可直接落表。
    for k in [k for k, v in m.items()
              if not isinstance(v, (int, float, np.integer, np.floating))]:
        rep.detail[k] = m.pop(k)

    rep.detail["ic_series"] = ric
    rep.detail["quality"] = q
    rep.detail["factor"] = factor

    # -- 维度得分 --
    s_abs_ic = cfg.scale("abs_ic", abs(m["rank_ic_mean"]))
    s_icir = cfg.scale("abs_icir", abs(m["rank_icir"]))
    win = m["rank_ic_win"]
    s_win = cfg.scale("ic_win", max(win, 1.0 - win) if np.isfinite(win) else np.nan)
    s_ls = cfg.scale("ls_t", abs(m["group_ls_t"]))
    s_mono = cfg.scale("monotonicity", abs(m["monotonicity"]))
    predictive = (0.30 * s_abs_ic + 0.25 * s_icir + 0.15 * s_win
                  + 0.20 * s_ls + 0.10 * s_mono)
    # 方向一致性：IC 与多空方向必须同号，否则是"平均出来的 IC"
    if np.isfinite(m["monotonicity"]) and np.isfinite(m["rank_ic_mean"]):
        if m["rank_ic_mean"] * m["monotonicity"] < 0:
            predictive *= 0.6

    stab = (0.35 * cfg.scale("seg_win", m["seg_win"])
            + 0.25 * cfg.scale("worst_seg_ic", m["worst_seg_ic"])
            + 0.25 * cfg.scale("rolling_ic_min", m["rolling_ic_min"])
            + 0.15 * cfg.scale("ac_l1", m.get("ac_l1", float("nan"))))
    # 质量维度按"扣除预热后的覆盖率"打分：否则长回看因子（如 TTM）会被
    # 预热期白白扣分，而预热是表达式结构决定的、不是数据质量问题。
    qual = (0.45 * cfg.scale("coverage", q.get("coverage_adj", q["coverage"]))
            + 0.30 * cfg.scale("degenerate", q["degenerate"])
            + 0.25 * cfg.scale("extreme", q["extreme"]))
    corr = 1.0 - cfg.scale("pool_max_corr", m.get("pool_max_corr", 0.0))
    corr = 0.5 * corr + 0.5 * float(min(max(
        (m.get("incremental_ic", 0.0) + 0.01) / 0.03, 0.0), 1.0))

    dims = {"predictive": predictive, "stability": stab,
            "quality": qual, "correlation": corr}
    rep.scores = {k: float(v) for k, v in dims.items()}
    wsum = sum(cfg.dimension_weights.get(k, 0.0) for k in dims) or 1.0
    base = sum(cfg.dimension_weights.get(k, 0.0) * v
               for k, v in dims.items()) / wsum
    penalties = (cfg.w_turnover_penalty * (m["turnover"] if np.isfinite(m["turnover"]) else 0.0)
                 + cfg.w_simplicity_penalty * math.log1p(node.size() if node is not None else 1))
    rep.scores["base"] = float(base)
    rep.scores["penalty"] = float(penalties)
    rep.score = float(min(max(base - penalties, 0.0), 1.0))
    rep.grade = _grade(rep.score)

    if with_risk:
        exps = exposures if exposures is not None else _risk.default_risk_exposures(panel)
        rep.detail["risk"] = _risk.risk_report(
            factor, fwd, exps, pool=pool, turnover=m["turnover"],
            min_stocks=cfg.min_stocks)
    if node is not None:
        rep.detail["node"] = node.to_dict()
        rep.detail["complexity"] = ex.complexity(node)
    return rep


def evaluate_expr(text: str, panel: PanelData,
                  pool: Optional[Dict[str, pd.DataFrame]] = None,
                  config: Optional[EvalConfig] = None,
                  with_risk: bool = False,
                  registry: Optional[Any] = None,
                  evaluator: Optional[ex.Evaluator] = None) -> FactorReport:
    """从 DSL 文本直接评价（解析 → 类型校验 → 求值 → 四维评价）。"""
    from .panel import default_registry
    reg = registry or panel.registry or default_registry()
    try:
        node = ex.parse(text)
    except ex.ExprError as exc:
        return FactorReport(name=text, expression=text, errors=[str(exc)])
    errs = ex.validate(node, reg)
    if errs:
        return FactorReport(name=text, expression=node.render(), errors=errs)
    ev = evaluator or ex.Evaluator(panel, reg)
    try:
        values = ev.run(node)
    except Exception as exc:  # noqa: BLE001 - 求值期异常也归入"无效因子"
        return FactorReport(name=text, expression=node.render(),
                            errors=[f"求值失败: {type(exc).__name__}: {exc}"])
    return evaluate(values, panel, name=node.render(), expression=node.render(),
                    pool=pool, config=config, with_risk=with_risk, node=node)


def rank_reports(reports: Iterable[FactorReport]) -> pd.DataFrame:
    """把多个报告汇总成排序表（无效因子单列，便于审计被剔原因）。"""
    rows = []
    for r in reports:
        d = {"name": r.name, "expression": r.expression, "score": r.score,
             "grade": r.grade, "valid": r.ok}
        d.update(r.metrics)
        d.update({f"score_{k}": v for k, v in r.scores.items()})
        if r.errors:
            d["errors"] = "; ".join(r.errors)
        rows.append(d)
    df = pd.DataFrame(rows)
    if "score" in df.columns and len(df):
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df
