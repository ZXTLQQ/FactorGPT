"""选股域与持仓映射（src/engine/universe.py）。

论文 2607.23068v1 的最后一段是"从清洗后的协方差到组合权重"的端到端落地。因子体系构建
的对应终点是**选股域**：合成因子只能给股票打分，"能不能买"却由流动性、可交易性决定。

把这一步单独拿出来，原因很实在：同一个因子在"全市场"和"流动性前 300"两个域上的
IC / 换手 / 容量会系统性不同，甚至符号相反。所以"这套体系好不好"必须与"在哪个域上
评价"一起说，否则结论不可复现 —— 这也是回测里最常见的一类口径事故。

三段式，每段都可单独调用：

1. :func:`build_universe`   —— 逐日打标签：可交易 / 不可交易 + 原因
2. :func:`build_portfolio`  —— 域内 Top-N 建仓（等权或分数加权），输出持仓与换手
3. :func:`evaluate_domains` —— 同一因子在多个候选域上的表现对照

域内规则的**优先级是固定**的（历史 → 流动性 → 价格 → ST → 涨跌停），
先命中者即为剔除原因，这样"某天为什么没买这只票"永远只有一个答案。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .ic_utils import ic_stats, panel_ic

# 不可交易原因（顺序即判定优先级，界面上直接按这个顺序展示）
REASON_HISTORY = "历史不足"
REASON_LIQUIDITY = "成交额不足"
REASON_PRICE = "价格越界"
REASON_ST = "风险警示股"
REASON_LIMIT = "涨跌停"
REASONS: Sequence[str] = (REASON_HISTORY, REASON_LIQUIDITY, REASON_PRICE, REASON_ST, REASON_LIMIT)


@dataclass
class UniverseSpec:
    """一个选股域的定义。

    Attributes:
        min_history: 上市/有数据的最少交易日数（滚动计数）。
        min_amount: 日成交额下限（元）；0 表示不过滤。
        min_price/max_price: 价格区间（元）；``None`` 表示该侧不过滤。
        exclude_st: 是否剔除名称含 ST / *ST 的标的（需要 ``name`` 列）。
        max_pct_chg: 涨跌幅绝对值超过该值视为涨跌停，当日不可买入；``None`` 关闭。
        top_n: 域内建仓数量。
        score_col: 打分列。
        weighting: ``equal``（等权）或 ``score``（按分数占比，负分截断为 0）。
        label: 域的可读名称（界面展示用）。
    """

    min_history: int = 60
    min_amount: float = 2.0e7
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    exclude_st: bool = True
    max_pct_chg: Optional[float] = 9.5
    top_n: int = 50
    score_col: str = "score"
    weighting: str = "equal"
    label: str = "默认域"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label, "min_history": int(self.min_history),
            "min_amount": float(self.min_amount), "min_price": self.min_price,
            "max_price": self.max_price, "exclude_st": bool(self.exclude_st),
            "max_pct_chg": self.max_pct_chg, "top_n": int(self.top_n),
            "weighting": self.weighting,
        }


def _ensure_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """校验并规范化长表面板：必须有 ``date`` / ``symbol`` 两列。"""
    if not isinstance(panel, pd.DataFrame) or panel.empty:
        raise ValueError("panel 不能为空")
    missing = {"date", "symbol"} - set(panel.columns)
    if missing:
        raise ValueError(f"panel 缺少必要列: {sorted(missing)}")
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def _liquidity(df: pd.DataFrame) -> pd.Series:
    """取日成交额序列：优先 ``amount``，否则用 ``close``×``volume`` 现算。"""
    if "amount" in df.columns:
        return pd.to_numeric(df["amount"], errors="coerce")
    if {"close", "volume"} <= set(df.columns):
        return (pd.to_numeric(df["close"], errors="coerce")
                * pd.to_numeric(df["volume"], errors="coerce"))
    raise ValueError("panel 缺少流动性字段：需要 amount，或同时提供 close 与 volume")


def build_universe(
    panel: pd.DataFrame,
    spec: Optional[UniverseSpec] = None,
    keep_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """给每个 ``(date, symbol)`` 打上可交易标签与剔除原因。

    Args:
        panel: 长表行情面板，需含 ``date``/``symbol``；流动性用 ``amount``，或
            ``close``×``volume`` 现算；价格过滤用 ``close``；涨跌停用 ``pct_chg``
            （缺失则该条规则自动跳过）。
        spec: 选股域定义，``None`` 取默认。
        keep_cols: 额外透传的列。

    Returns:
        与 ``panel`` 同长的长表，附 ``tradable``（bool）、``reason``（不可交易原因，
        可交易时为空串）、``amount``、``hist_days`` 等列。
    """
    spec = spec or UniverseSpec()
    df = _ensure_panel(panel)
    n = len(df)

    df["amount"] = _liquidity(df)

    hist = df.groupby("symbol").cumcount() + 1
    df["hist_days"] = hist.astype(int)

    reason = np.full(n, "", dtype=object)
    tradable = np.ones(n, dtype=bool)

    def _strike(mask: np.ndarray, text: str) -> None:
        """把尚未被剔除的行标记为该原因（首次命中优先）。"""
        hit = mask & tradable
        reason[hit] = text
        tradable[hit] = False

    # 判定顺序即优先级：历史 → 流动性 → 价格 → ST → 涨跌停
    _strike((hist < int(spec.min_history)).to_numpy(), REASON_HISTORY)
    if spec.min_amount and spec.min_amount > 0:
        _strike((df["amount"] < float(spec.min_amount)).to_numpy(), REASON_LIQUIDITY)
    if "close" in df.columns and (spec.min_price is not None or spec.max_price is not None):
        close = df["close"].astype(float).to_numpy()
        bad = np.zeros(n, dtype=bool)
        if spec.min_price is not None:
            bad |= close < float(spec.min_price)
        if spec.max_price is not None:
            bad |= close > float(spec.max_price)
        _strike(bad | ~np.isfinite(close), REASON_PRICE)
    if spec.exclude_st and "name" in df.columns:
        nm = df["name"].astype(str).str.upper()
        _strike(nm.str.contains("ST", na=False).to_numpy(), REASON_ST)
    if spec.max_pct_chg is not None and "pct_chg" in df.columns:
        chg = pd.to_numeric(df["pct_chg"], errors="coerce").to_numpy(dtype=float)
        _strike(np.abs(chg) >= float(spec.max_pct_chg), REASON_LIMIT)

    out = df.assign(tradable=tradable, reason=reason)
    cols = ["date", "symbol", "tradable", "reason", "amount", "hist_days"]
    extra = [c for c in (keep_cols or []) if c in out.columns and c not in cols]
    return out[cols + extra]


def universe_summary(flags: pd.DataFrame) -> Dict[str, Any]:
    """选股域的体检报告：每日可选数量、剔除原因分布、覆盖率。"""
    if flags is None or flags.empty:
        return {"ok": False, "reason": "空标签表"}
    total = len(flags)
    trad = int(flags["tradable"].sum())
    per_day = flags.groupby("date")["tradable"].sum()
    reasons = (
        flags.loc[~flags["tradable"], "reason"]
        .value_counts()
        .reindex(list(REASONS))
        .fillna(0)
        .astype(int)
    )
    return {
        "ok": True,
        "n_rows": total,
        "n_tradable": trad,
        "coverage_pct": trad / total * 100.0 if total else 0.0,
        "n_dates": int(flags["date"].nunique()),
        "avg_per_day": float(per_day.mean()) if len(per_day) else 0.0,
        "min_per_day": int(per_day.min()) if len(per_day) else 0,
        "max_per_day": int(per_day.max()) if len(per_day) else 0,
        "reasons": {k: int(v) for k, v in reasons.items()},
        "empty_days": int((per_day == 0).sum()),
    }


def build_portfolio(
    panel: pd.DataFrame,
    flags: pd.DataFrame,
    spec: Optional[UniverseSpec] = None,
    score_col: Optional[str] = None,
) -> pd.DataFrame:
    """域内 Top-N 建仓，输出逐日持仓与权重。

    Args:
        panel: 含 ``date``/``symbol`` 与打分列的长表。
        flags: :func:`build_universe` 的输出（同 ``date``/``symbol`` 粒度）。
        spec: 选股域定义（决定 ``top_n`` 与 ``weighting``）。
        score_col: 打分列名；默认取 ``spec.score_col``。

    Returns:
        持仓长表，列含 ``date``/``symbol``/``score``/``rank``/``weight``；
        ``weight`` 每日合计为 1（可选标的不为空的日子）。
    """
    spec = spec or UniverseSpec()
    col = score_col or spec.score_col
    df = _ensure_panel(panel)
    if col not in df.columns:
        raise ValueError(f"panel 缺少打分列 {col}")
    fl = flags[["date", "symbol", "tradable", "reason"]].copy() if flags is not None else None
    if fl is None:
        raise ValueError("flags 不能为空")
    fl["date"] = pd.to_datetime(fl["date"])
    fl["symbol"] = fl["symbol"].astype(str)

    merged = df[["date", "symbol", col]].merge(fl, on=["date", "symbol"], how="left")
    merged["tradable"] = merged["tradable"].fillna(False).astype(bool)
    merged["score"] = pd.to_numeric(merged[col], errors="coerce")
    picked = merged.loc[merged["tradable"] & merged["score"].notna()].copy()
    if picked.empty:
        return pd.DataFrame(columns=["date", "symbol", "score", "rank", "weight"])

    picked["rank"] = picked.groupby("date")["score"].rank(ascending=False, method="first")
    top = picked.loc[picked["rank"] <= int(max(spec.top_n, 1))].copy()

    if str(spec.weighting).lower() == "score":
        w = top["score"].clip(lower=0.0)
        tot = w.groupby(top["date"]).transform("sum")
        top["weight"] = np.where(tot > 0, w / tot, np.nan)
        # 全为负分时分数加权无意义，退回等权，避免出现"权重全 0 的空仓日"
        fallback = 1.0 / top.groupby("date")["symbol"].transform("size")
        top["weight"] = np.where(np.isfinite(top["weight"]), top["weight"], fallback)
    else:
        top["weight"] = 1.0 / top.groupby("date")["symbol"].transform("size")

    return (top[["date", "symbol", "score", "rank", "weight"]]
            .sort_values(["date", "rank"]).reset_index(drop=True))


def portfolio_turnover(portfolio: pd.DataFrame) -> pd.Series:
    """逐日单边换手率（相对上一期持仓的权重变动一半）。

    单边口径 = ``Σ|w_t − w_{t−1}| / 2``，这样"全部换一遍"= 1.0，
    与成本模型里"双边各收一次"的写法对得上。
    """
    if portfolio is None or portfolio.empty:
        return pd.Series(dtype=float)
    wide = portfolio.pivot_table(index="date", columns="symbol", values="weight",
                                 aggfunc="sum").fillna(0.0)
    if len(wide) < 2:
        return pd.Series(dtype=float)
    diff = wide.diff().iloc[1:].abs().sum(axis=1) / 2.0
    diff.name = "turnover"
    return diff


def evaluate_domains(
    panel: pd.DataFrame,
    specs: Sequence[UniverseSpec],
    factor_col: str,
    y_col: str = "fwd_ret",
    min_count: int = 5,
) -> pd.DataFrame:
    """同一因子在多个候选域上的表现对照（IC/ICIR/可选数量/换手）。

    这是"选股域"这一步最该给出的东西：域不是越严越好，也不是越宽越好 ——
    整体 IC 高但只覆盖 30 只股票，与整体 IC 略低但覆盖 800 只，是两个完全不同的产品。
    """
    df = _ensure_panel(panel)
    if factor_col not in df.columns:
        raise ValueError(f"panel 缺少因子列 {factor_col}")
    if y_col not in df.columns:
        raise ValueError(f"panel 缺少收益列 {y_col}")

    rows: List[Dict[str, Any]] = []
    for spec in specs:
        flags_full = build_universe(df, spec)
        flags = flags_full[["date", "symbol", "tradable"]]
        merged = df[["date", "symbol", factor_col, y_col]].merge(
            flags, on=["date", "symbol"], how="left")
        in_domain = merged["tradable"].fillna(False).astype(bool)
        sub = merged.loc[in_domain]
        ic = panel_ic(sub[factor_col].to_numpy(dtype=float),
                      sub[y_col].to_numpy(dtype=float),
                      sub["date"].to_numpy(), min_count=int(min_count))
        st = ic_stats(ic)
        port = build_portfolio(df, flags_full, spec, score_col=factor_col)
        to = portfolio_turnover(port)
        rows.append({
            "domain": spec.label,
            "top_n": int(spec.top_n),
            "min_amount": float(spec.min_amount),
            "min_history": int(spec.min_history),
            "coverage_pct": float(in_domain.mean() * 100.0),
            "avg_per_day": float(sub.groupby("date").size().mean()) if len(sub) else 0.0,
            "n_ic": int(st.get("n", 0) or 0),
            "ic": st.get("ic", float("nan")),
            "icir": st.get("icir", float("nan")),
            "t_stat": st.get("t_stat", float("nan")),
            "positive_ratio": st.get("positive_ratio", float("nan")),
            "avg_turnover": float(to.mean()) if len(to) else float("nan"),
            "max_turnover": float(to.max()) if len(to) else float("nan"),
        })
    return pd.DataFrame(rows)


def preset_domains(top_n: int = 50) -> List[UniverseSpec]:
    """三档常用域（宽 / 中 / 严），供界面一键对照。"""
    return [
        UniverseSpec(min_history=20, min_amount=5.0e6, top_n=top_n * 4, label="宽域（全市场可交易）"),
        UniverseSpec(min_history=60, min_amount=2.0e7, top_n=top_n * 2, label="中域（流动性前段）"),
        UniverseSpec(min_history=120, min_amount=1.0e8, top_n=top_n, label="严域（流动性头部）"),
    ]
