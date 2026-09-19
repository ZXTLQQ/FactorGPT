"""多因子体系搭建（src/engine/factor_synthesis.py）。

把「因子合成」从单一 IC 加权升级为一条**可对照、可回退**的体系链路：

    linear baseline → 显式非线性 → 树模型 → contextual（状态依赖）

每一层都必须在**样本外**与上一层对照，只有增益显著才保留，否则退回上一层
（可解释性优先：能用线性说清楚就不上黑箱）。

各层要点：
- ``linear_composite``：等权 / IC 加权 / ICIR 加权 / **max IC**（max ICIR 的
  解析解 w ∝ Σ_IC⁻¹ μ_IC，协方差用 Ledoit-Wolf 收缩）/ 截面回归 / 贝叶斯收缩。
  max IC 是整个体系的 baseline，后续所有增益都以它为标尺。
- ``nonlinear_explicit``：显式引入非线性——交互项、二次项、分位哑变量，
  用前向贪心 + 验证集早停挑选，因此每一项都看得见、说得清（区别于黑箱）。
- ``tree_synthesis``：树模型合成（sklearn GBDT 可用时），捕捉线性与显式非线性
  都够不到的高阶交互；无 sklearn 时降级并说明原因，不静默失败。
- ``contextual_modeling``：contextual modeling——按市场状态（波动 regime /
  涨跌趋势 / 自定义 context 列）分别估计权重，解决「同一套权重在不同状态下
  互相抵消」的问题。
- ``build_factor_system``：串联以上各层，输出逐层增益表与最终推荐。

约定：面板为扁平 DataFrame，含 date/symbol 列、若干因子列与前瞻收益列。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from engine.ic_utils import ic_stats, panel_ic
from engine.twin_model import ledoit_wolf_shrink

try:  # 树模型为可选依赖，缺失时降级而非报错
    from sklearn.ensemble import GradientBoostingRegressor
    _HAS_SKLEARN = True
except Exception:  # noqa: BLE001
    _HAS_SKLEARN = False


@dataclass
class SynthesisResult:
    """单层合成结果。"""

    name: str
    signal: pd.Series                       # 合成信号（与面板同索引）
    ic: pd.Series                           # 全样本 IC 序列
    stats: Dict[str, float] = field(default_factory=dict)   # ic_stats（全样本）
    oos_stats: Dict[str, float] = field(default_factory=dict)
    weights: pd.Series = field(default_factory=pd.Series)   # 因子权重（线性类）
    terms: List[str] = field(default_factory=list)          # 实际使用的项（非线性类）
    extra: Dict = field(default_factory=dict)

    def summary_row(self) -> Dict:
        row = {"方案": self.name}
        row.update({f"IC_{k}": v for k, v in self.stats.items() if k in ("ic", "icir", "t_stat")})
        for k, v in self.oos_stats.items():
            if k in ("ic", "icir"):
                row[f"OOS_{k}"] = v
        return row


# ======================================================================
# 基础设施：IC 矩阵、时序切分、信号评估
# ======================================================================

def factor_ic_matrix(panel: pd.DataFrame, factor_cols: Sequence[str],
                     ret_col: str = "fwd_ret", date_col: str = "date") -> pd.DataFrame:
    """逐因子计算截面 IC 时间序列，返回 index=date / columns=因子的矩阵。"""
    if panel is None or not factor_cols:
        return pd.DataFrame()
    out: Dict[str, pd.Series] = {}
    y = pd.to_numeric(panel[ret_col], errors="coerce").to_numpy(dtype=float)
    keys = panel[date_col].to_numpy()
    for c in factor_cols:
        f = pd.to_numeric(panel[c], errors="coerce").to_numpy(dtype=float)
        out[c] = panel_ic(f, y, keys)
    return pd.DataFrame(out)


def _time_split(panel: pd.DataFrame, date_col: str, train_ratio: float = 0.7):
    """按时序切分训练/测试面板（不做随机打散，避免前视泄漏）。"""
    dates = np.sort(panel[date_col].unique())
    cut_idx = int(len(dates) * train_ratio)
    if cut_idx < 1 or cut_idx >= len(dates):
        cut_idx = max(1, min(len(dates) - 1, cut_idx))
    cut = dates[cut_idx]
    return panel[panel[date_col] < cut], panel[panel[date_col] >= cut]


def _zscore_cross_section(s: pd.Series, keys: np.ndarray) -> pd.Series:
    """按日期做截面标准化（去均值除标准差），使各因子可比。"""
    df = pd.DataFrame({"v": pd.to_numeric(s, errors="coerce"), "k": keys})
    g = df.groupby("k")["v"]
    mean = g.transform("mean")
    std = g.transform("std").replace(0, np.nan)
    return ((df["v"] - mean) / std).fillna(0.0).reset_index(drop=True)


def _evaluate_signal(panel: pd.DataFrame, signal: pd.Series, ret_col: str,
                     date_col: str, train_ratio: float = 0.7) -> Dict:
    """评估合成信号：全样本 IC 统计 + 样本外 IC 统计。"""
    sig = pd.Series(np.asarray(signal, dtype=float), index=panel.index)
    y = pd.to_numeric(panel[ret_col], errors="coerce")
    keys = panel[date_col].to_numpy()
    ic_all = panel_ic(sig.to_numpy(dtype=float), y.to_numpy(dtype=float), keys)
    tr_mask = _train_mask(panel, date_col, train_ratio)
    ic_oos = panel_ic(sig.to_numpy(dtype=float)[~tr_mask],
                      y.to_numpy(dtype=float)[~tr_mask], keys[~tr_mask])
    return {"ic": ic_all, "stats": ic_stats(ic_all), "oos_stats": ic_stats(ic_oos),
            "ic_oos": ic_oos}


def _train_mask(panel: pd.DataFrame, date_col: str, train_ratio: float) -> np.ndarray:
    dates = np.sort(panel[date_col].unique())
    cut = dates[max(1, min(len(dates) - 1, int(len(dates) * train_ratio)))]
    return panel[date_col].to_numpy() < cut


# ======================================================================
# 第一层：线性合成（含 max IC baseline）
# ======================================================================

def linear_composite(panel: pd.DataFrame, factor_cols: Sequence[str],
                     ret_col: str = "fwd_ret", date_col: str = "date",
                     scheme: str = "max_ic", train_ratio: float = 0.7,
                     shrink_ic_cov: bool = True) -> SynthesisResult:
    """线性合成：体系 baseline。

    scheme:
        equal      等权（最朴素的对照）
        ic         w ∝ 各因子平均 IC
        icir       w ∝ 各因子 ICIR（均值/标准差）
        max_ic     w ∝ Σ_IC⁻¹ μ_IC（**最大化组合 ICIR 的解析解**，体系 baseline）
        regression 截面回归系数的时序均值
        bayesian   贝叶斯收缩后的 IC 权重（样本少的因子自动降权）
    """
    if panel is None or not factor_cols:
        return SynthesisResult(name="linear", signal=pd.Series(dtype=float), ic=pd.Series(dtype=float))

    keys = panel[date_col].to_numpy()
    z = pd.DataFrame({c: _zscore_cross_section(panel[c], keys) for c in factor_cols},
                     index=panel.index)
    train_mask = _train_mask(panel, date_col, train_ratio)
    ic_train = factor_ic_matrix(panel[train_mask], factor_cols, ret_col, date_col)

    if scheme == "equal":
        w = pd.Series(1.0 / len(factor_cols), index=list(factor_cols))
    elif scheme == "bayesian":
        from engine.method_hub import bayesian_ic_weights
        w = bayesian_ic_weights(ic_train)
        w = w.reindex(factor_cols).fillna(0.0)
    elif scheme == "regression":
        coeffs = []
        y_tr = pd.to_numeric(panel.loc[train_mask, ret_col], errors="coerce")
        for date, grp in panel[train_mask].groupby(date_col):
            sub = z.loc[grp.index, factor_cols]
            yy = y_tr.loc[grp.index]
            ok = sub.notna().all(axis=1) & yy.notna()
            if ok.sum() < len(factor_cols) + 2:
                continue
            a = sub[ok].to_numpy(dtype=float)
            b = yy[ok].to_numpy(dtype=float)
            beta, *_ = np.linalg.lstsq(a, b, rcond=None)
            coeffs.append(beta)
        w = pd.Series(np.mean(coeffs, axis=0) if coeffs else np.ones(len(factor_cols)),
                      index=list(factor_cols))
    else:
        mu = ic_train.mean()
        if scheme == "ic":
            w = mu.copy()
        elif scheme == "icir":
            sd = ic_train.std().replace(0, np.nan)
            w = mu / sd
        else:  # max_ic：最大化组合 ICIR → w ∝ Σ⁻¹ μ
            cov = ledoit_wolf_shrink(ic_train.dropna()) if shrink_ic_cov \
                else ic_train.dropna().cov().to_numpy(dtype=float)
            if isinstance(cov, pd.DataFrame):
                cov = cov.to_numpy(dtype=float)
            cov = np.atleast_2d(cov)
            if cov.shape[0] != len(mu):
                cov = np.eye(len(mu))
            try:
                w = pd.Series(np.linalg.solve(cov + 1e-8 * np.eye(len(mu)), mu.to_numpy(dtype=float)),
                              index=list(factor_cols))
            except np.linalg.LinAlgError:
                w = mu.copy()
        w = w.fillna(0.0)

    if float(w.abs().sum()) > 0:
        w = w / w.abs().sum()
    signal = (z[list(factor_cols)] * w).sum(axis=1)
    ev = _evaluate_signal(panel, signal, ret_col, date_col, train_ratio)
    return SynthesisResult(name=f"linear:{scheme}", signal=signal, ic=ev["ic"],
                           stats=ev["stats"], oos_stats=ev["oos_stats"], weights=w)


# ======================================================================
# 第二层：显式非线性
# ======================================================================

def _candidate_nonlinear_terms(z: pd.DataFrame, cols: Sequence[str],
                               max_pairs: int = 12) -> Dict[str, pd.Series]:
    """构造候选非线性项：二次项、符号保持的幂、以及相关性最低的若干交互项。

    只取「与已有因子相关性较低」的交互对，避免生成一堆线性相关的冗余项。
    """
    terms: Dict[str, pd.Series] = {}
    for c in cols:
        terms[f"{c}^2"] = z[c] ** 2
        terms[f"sign({c})*|{c}|^0.5"] = np.sign(z[c]) * np.sqrt(z[c].abs())
    if len(cols) >= 2:
        corr = z[list(cols)].corr().abs()
        pairs = []
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                pairs.append((float(corr.loc[a, b]), a, b))
        pairs.sort()   # 相关性低的交互项信息量更大
        for _, a, b in pairs[:max_pairs]:
            terms[f"{a}*{b}"] = z[a] * z[b]
    return terms


def nonlinear_explicit(panel: pd.DataFrame, factor_cols: Sequence[str],
                       ret_col: str = "fwd_ret", date_col: str = "date",
                       max_terms: int = 5, train_ratio: float = 0.7,
                       min_gain: float = 0.02) -> SynthesisResult:
    """显式非线性合成：前向贪心挑选非线性项，增益不足则原样退回。

    每一步在验证集（训练段的后 30%）上评估候选项带来的 ICIR 提升，
    只保留真正有增益的项，因此最终模型形如「线性部分 + 若干可名状的非线性项」，
    每一项都能写出公式，区别于树模型的黑箱。
    """
    base = linear_composite(panel, factor_cols, ret_col, date_col, "max_ic", train_ratio)
    if base.signal.empty:
        return base

    keys = panel[date_col].to_numpy()
    z = pd.DataFrame({c: _zscore_cross_section(panel[c], keys) for c in factor_cols},
                     index=panel.index)
    train_mask = _train_mask(panel, date_col, train_ratio)
    # 训练段再切出验证段用于贪心早停
    tr_dates = np.sort(panel[train_mask][date_col].unique())
    val_cut = tr_dates[max(1, int(len(tr_dates) * 0.7))]
    fit_mask = train_mask & (panel[date_col].to_numpy() < val_cut)
    val_mask = train_mask & (panel[date_col].to_numpy() >= val_cut)
    if val_mask.sum() < 20:
        return SynthesisResult(name="nonlinear:skipped(验证段不足)", signal=base.signal,
                               ic=base.ic, stats=base.stats, oos_stats=base.oos_stats,
                               weights=base.weights)

    cands = _candidate_nonlinear_terms(z, factor_cols)
    y = pd.to_numeric(panel[ret_col], errors="coerce")
    chosen: List[str] = []
    cur = base.signal.to_numpy(dtype=float).copy()

    def _icir(mask: np.ndarray, sig: np.ndarray) -> float:
        s = ic_stats(panel_ic(sig[mask], y.to_numpy(dtype=float)[mask],
                              panel[date_col].to_numpy()[mask]))
        return float(s.get("icir", np.nan)) if np.isfinite(s.get("icir", np.nan)) else -9.9

    best_val = _icir(val_mask, cur)
    for _ in range(int(max_terms)):
        best_name, best_sig, best_score = None, None, best_val
        for name, series in cands.items():
            if name in chosen:
                continue
            cand = cur + series.to_numpy(dtype=float)
            sc = _icir(val_mask, cand)
            if sc > best_score + min_gain:
                best_name, best_sig, best_score = name, cand, sc
        if best_name is None:
            break
        chosen.append(best_name)
        cur, best_val = best_sig, best_score

    signal = pd.Series(cur, index=panel.index)
    ev = _evaluate_signal(panel, signal, ret_col, date_col, train_ratio)
    name = f"nonlinear(+{len(chosen)})" if chosen else "nonlinear:无显著增益"
    return SynthesisResult(name=name, signal=signal, ic=ev["ic"], stats=ev["stats"],
                           oos_stats=ev["oos_stats"], weights=base.weights, terms=chosen,
                           extra={"val_icir": best_val})


# ======================================================================
# 第三层：树模型合成
# ======================================================================

def tree_synthesis(panel: pd.DataFrame, factor_cols: Sequence[str],
                   ret_col: str = "fwd_ret", date_col: str = "date",
                   train_ratio: float = 0.7, n_estimators: int = 200,
                   max_depth: int = 3, seed: int = 42) -> SynthesisResult:
    """树模型合成：捕捉线性与显式非线性都够不到的高阶交互。

    无 sklearn 时不静默失败，返回降级说明（extra['degraded']）。
    """
    base = linear_composite(panel, factor_cols, ret_col, date_col, "max_ic", train_ratio)
    if base.signal.empty:
        return base
    if not _HAS_SKLEARN:
        return SynthesisResult(name="tree:降级(未安装 scikit-learn)", signal=base.signal,
                               ic=base.ic, stats=base.stats, oos_stats=base.oos_stats,
                               weights=base.weights,
                               extra={"degraded": True, "reason": "scikit-learn 不可用"})

    keys = panel[date_col].to_numpy()
    z = pd.DataFrame({c: _zscore_cross_section(panel[c], keys) for c in factor_cols},
                     index=panel.index)
    train_mask = _train_mask(panel, date_col, train_ratio)
    y = pd.to_numeric(panel[ret_col], errors="coerce")
    x_tr, y_tr = z[train_mask], y[train_mask]
    ok = x_tr.notna().all(axis=1) & y_tr.notna()
    if ok.sum() < 50:
        return SynthesisResult(name="tree:跳过(训练样本不足)", signal=base.signal,
                               ic=base.ic, stats=base.stats, oos_stats=base.oos_stats,
                               weights=base.weights)

    model = GradientBoostingRegressor(
        n_estimators=int(n_estimators), max_depth=int(max_depth),
        learning_rate=0.05, subsample=0.9, random_state=seed,
    )
    model.fit(x_tr[ok].to_numpy(dtype=float), y_tr[ok].to_numpy(dtype=float))
    pred = model.predict(z.fillna(0.0).to_numpy(dtype=float))
    signal = pd.Series(pred, index=panel.index)
    ev = _evaluate_signal(panel, signal, ret_col, date_col, train_ratio)
    imp = pd.Series(model.feature_importances_, index=list(factor_cols)).sort_values(ascending=False)
    return SynthesisResult(name=f"tree:GBDT(d={max_depth})", signal=signal, ic=ev["ic"],
                           stats=ev["stats"], oos_stats=ev["oos_stats"],
                           extra={"feature_importance": imp})


# ======================================================================
# 第四层：Contextual modeling
# ======================================================================

def _infer_context(panel: pd.DataFrame, ret_col: str, date_col: str,
                   kind: str = "vol_regime", n_bins: int = 3) -> pd.Series:
    """推断市场状态标签（context）：波动 regime / 涨跌趋势。

    vol_regime：按「截面收益横截面标准差的滚动分位」分档（高波动/中/低）；
    trend：按「截面等权平均收益的滚动累计方向」分档（上涨/震荡/下跌）。
    """
    y = pd.to_numeric(panel[ret_col], errors="coerce")
    g = y.groupby(panel[date_col].to_numpy())
    if kind == "trend":
        mkt = g.mean().sort_index()
        roll = mkt.rolling(20, min_periods=5).mean()
        lab = pd.cut(roll.rank(pct=True), bins=n_bins,
                     labels=[f"trend_{i + 1}" for i in range(n_bins)])
        mapping = lab.to_dict()
    else:
        vol = g.std().sort_index()
        roll = vol.rolling(20, min_periods=5).mean()
        lab = pd.cut(roll.rank(pct=True), bins=n_bins,
                     labels=[f"vol_{i + 1}" for i in range(n_bins)])
        mapping = lab.to_dict()
    ctx = panel[date_col].map(mapping)
    return pd.Series(ctx.astype(str).values, index=panel.index)


def contextual_modeling(panel: pd.DataFrame, factor_cols: Sequence[str],
                        ret_col: str = "fwd_ret", date_col: str = "date",
                        context: Optional[pd.Series] = None,
                        context_kind: str = "vol_regime",
                        train_ratio: float = 0.7, min_obs: int = 200) -> SynthesisResult:
    """Contextual modeling：按市场状态分别估计权重。

    核心假设：因子的有效性是状态依赖的。若用一套全局权重，不同状态下的
    相反信号会互相抵消。这里对每个 context 独立估计 max IC 权重，
    再按当期所属状态选择权重，得到「状态依赖」的合成信号。

    Args:
        context: 自定义状态标签（与面板同长度）。缺省时按 context_kind 自动推断。
        min_obs: 单个 context 的最小样本量，不足则回落到全局权重。
    """
    base = linear_composite(panel, factor_cols, ret_col, date_col, "max_ic", train_ratio)
    if base.signal.empty:
        return base
    ctx = context if context is not None else _infer_context(panel, ret_col, date_col, context_kind)
    if ctx is None or ctx.empty:
        return base
    ctx = pd.Series(np.asarray(ctx, dtype=str), index=panel.index)

    keys = panel[date_col].to_numpy()
    z = pd.DataFrame({c: _zscore_cross_section(panel[c], keys) for c in factor_cols},
                     index=panel.index)
    train_mask = _train_mask(panel, date_col, train_ratio)

    weights_by_ctx: Dict[str, pd.Series] = {}
    counts: Dict[str, int] = {}
    for lab, idx in pd.Series(np.asarray(ctx)).groupby(np.asarray(ctx)):
        m = train_mask & (ctx.to_numpy() == lab)
        counts[str(lab)] = int(m.sum())
        if m.sum() < min_obs:
            weights_by_ctx[str(lab)] = base.weights
            continue
        sub = panel[m]
        r = linear_composite(sub, factor_cols, ret_col, date_col, "max_ic", train_ratio=1.0)
        weights_by_ctx[str(lab)] = r.weights if not r.weights.empty else base.weights

    w_mat = pd.DataFrame(weights_by_ctx).T.reindex(columns=list(factor_cols)).fillna(0.0)
    # 按每期所属 context 取对应权重（同一期的 context 一致，取首个即可）
    ctx_by_date = pd.Series(np.asarray(ctx).reshape(-1), index=panel.index).groupby(
        pd.Series(keys, index=panel.index)).agg(lambda s: s.iloc[0])
    lab_arr = ctx_by_date.reindex(pd.Series(keys, index=panel.index)).to_numpy()
    w_sel = w_mat.reindex(lab_arr)
    w_sel = w_sel.fillna(base.weights).reset_index(drop=True)
    signal = pd.Series((z[list(factor_cols)].to_numpy(dtype=float)
                        * w_sel.to_numpy(dtype=float)).sum(axis=1), index=panel.index)
    ev = _evaluate_signal(panel, signal, ret_col, date_col, train_ratio)
    return SynthesisResult(name=f"contextual:{context_kind}", signal=signal, ic=ev["ic"],
                           stats=ev["stats"], oos_stats=ev["oos_stats"],
                           weights=base.weights,
                           extra={"weights_by_context": w_mat, "context_counts": counts,
                                  "context_kind": context_kind})


# ======================================================================
# 体系搭建：串联各层 + 智能选择
# ======================================================================

@dataclass
class FactorSystemResult:
    """体系搭建结果。"""

    layers: List[SynthesisResult] = field(default_factory=list)
    recommended: Optional[SynthesisResult] = None
    baseline: Optional[SynthesisResult] = None
    method_advice: Dict = field(default_factory=dict)
    gain_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        if not self.layers:
            return "无可用结果。"
        lines = [f"智能选择建议：{self.method_advice.get('primary', '-')} —— "
                 f"{self.method_advice.get('reason', '')}"]
        for r in self.layers:
            lines.append(
                f"  {r.name}: IC={r.stats.get('ic', float('nan')):.4f} "
                f"ICIR={r.stats.get('icir', float('nan')):.4f} "
                f"t={r.stats.get('t_stat', float('nan')):.2f} | "
                f"OOS IC={r.oos_stats.get('ic', float('nan')):.4f} "
                f"OOS ICIR={r.oos_stats.get('icir', float('nan')):.4f}"
            )
        if self.recommended is not None:
            lines.append(f"推荐采用：{self.recommended.name}")
        return "\n".join(lines)


def build_factor_system(panel: pd.DataFrame, factor_cols: Sequence[str],
                        ret_col: str = "fwd_ret", date_col: str = "date",
                        use_nonlinear: bool = True, use_tree: bool = True,
                        use_contextual: bool = True,
                        train_ratio: float = 0.7,
                        min_oos_gain: float = 0.10,
                        notes: Optional[List[str]] = None) -> FactorSystemResult:
    """串联各层搭建多因子体系，并按「样本外增益是否值得」给出推荐。

    推荐规则（可解释优先）：
      以 max IC 线性 baseline 为标尺，只有某层的 OOS IC 相对 baseline 提升
      超过 ``min_oos_gain``（默认 10%）才采用该层；否则退回 baseline。
      若启用 contextual，则单独比较其 OOS 表现（状态依赖权重往往收益最大）。

    Returns:
        FactorSystemResult（含逐层结果、增益表与推荐方案）。
    """
    from engine.method_hub import profile_from_ic, select_method

    res = FactorSystemResult()
    if panel is None or not factor_cols:
        return res

    # 1) 智能选择：先看清数据画像，再决定起点
    ic_mat = factor_ic_matrix(panel, factor_cols, ret_col, date_col)
    profile = profile_from_ic(ic_mat, panel[list(factor_cols)], notes=notes)
    advice = select_method(profile)
    res.method_advice = advice

    # 2) Baseline 必做（后续所有增益以它为标尺）
    baseline = linear_composite(panel, factor_cols, ret_col, date_col, "max_ic", train_ratio)
    res.baseline = baseline
    res.layers.append(baseline)
    # 同时记录等权与 IC 加权作为朴素对照
    res.layers.append(linear_composite(panel, factor_cols, ret_col, date_col, "equal", train_ratio))

    # 3) 逐层增益
    if use_nonlinear:
        res.layers.append(nonlinear_explicit(panel, factor_cols, ret_col, date_col,
                                             train_ratio=train_ratio))
    if use_tree and _HAS_SKLEARN:
        res.layers.append(tree_synthesis(panel, factor_cols, ret_col, date_col,
                                         train_ratio=train_ratio))
    if use_contextual:
        res.layers.append(contextual_modeling(panel, factor_cols, ret_col, date_col,
                                              train_ratio=train_ratio))

    # 4) 增益表与推荐
    rows = [r.summary_row() for r in res.layers]
    res.gain_table = pd.DataFrame(rows)
    base_oos = abs(float(baseline.oos_stats.get("ic", 0.0) or 0.0))
    best, best_score = baseline, base_oos
    for r in res.layers[1:]:
        oos = abs(float(r.oos_stats.get("ic", 0.0) or 0.0))
        if not np.isfinite(oos):
            continue
        # 要求相对 baseline 有实质增益，才值得牺牲可解释性
        if base_oos > 0 and oos < base_oos * (1.0 + min_oos_gain):
            continue
        if oos > best_score:
            best, best_score = r, oos
    res.recommended = best
    return res
