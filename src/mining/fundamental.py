"""量价 × 基本面统一框架（中信建投《"逐鹿"Alpha专题报告(三十)》落地）。

研报的核心论断是：**量价因子与基本面因子不该是两套系统**。它们共用同一个
面板、同一套算子、同一个评价体系，差别只在字段来源；能跨域组合（`mul/div/
spread/ts_corr/ts_reg_resi`）与跨域中性化（`neutral(x, 基本面控件)`）之后，
才有"统一因子空间"可言。

本模块负责三件事：

1. **字段元数据**：财务科目的维度/语义/角色（``F`` 角色），注册进字段注册表
   后即可参与表达式树的静态类型检查。
2. **PIT 安装**：``install_fundamentals`` 把「公告日 + 数值」长表投影成交易日
   面板。这是财务因子唯一真正致命的风险点——只要有一处用了未来财报，回测
   再漂亮也是假的。``check_pit`` 用一份**独立参考实现**逐点复核，专门用来
   抓前视。
3. **因子库与混合模板**：TTM / 同比 / 环比 / 质量 / 杠杆 / 应计 / 估值，
   以及量价 × 基本面的组合模板（含滚动正交残差）。

关于 TTM 的实现口径：单季流量科目的 TTM = 最近四个季度单季值之和。季度报告
间隔约 63 个交易日，因此 ``ts_delay(x, 250)`` 恰好落在去年同一季度的报告上，
``add(add(x, ts_delay(x, 250)), add(ts_delay(x, 500), ts_delay(x, 750)))``
就是四个季度单季值之和。这样把"TTM"完全表达成已有算子，不引入新的数据
处理特例。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import expr as ex
from . import ops
from .panel import (
    DIM_AMOUNT,
    DIM_COUNT,
    ROLE_FUNDAMENTAL,
    SEM_GROWTH,
    SEM_LEVERAGE,
    SEM_QUALITY,
    SEM_SIZE,
    SEM_VALUATION,
    FieldMeta,
    FieldRegistry,
    PanelData,
    asof_align,
    day_delta,
)

__all__ = [
    "FUND_FACTOR_LIBRARY",
    "FUND_FIELDS",
    "MIN_HISTORY_DAYS",
    "QUARTER_DAYS",
    "TTM_QUARTER_OFFSETS",
    "YEAR_DAYS",
    "check_history",
    "check_pit",
    "install_fundamentals",
    "library_errors",
    "market_factor_library",
    "mixed_templates",
    "pit_reference",
    "register_fundamental_fields",
    "synthetic_quarterly",
]

# 交易日近似：一年 ≈ 250 个交易日、一季 ≈ 61 个交易日
YEAR_DAYS = 250
QUARTER_DAYS = 61
TTM_QUARTER_OFFSETS: Tuple[int, ...] = (0, YEAR_DAYS, 2 * YEAR_DAYS, 3 * YEAR_DAYS)

# --------------------------------------------------------------------------
# 1. 字段元数据
# --------------------------------------------------------------------------
# (name, dimension, semantics, doc)
FUND_FIELDS: Tuple[Tuple[str, str, str, str], ...] = (
    ("revenue", DIM_AMOUNT, SEM_GROWTH, "营业收入（单季流量，PIT）"),
    ("net_profit", DIM_AMOUNT, SEM_QUALITY, "归母净利润（单季流量，PIT）"),
    ("gross_profit", DIM_AMOUNT, SEM_QUALITY, "毛利（单季流量，PIT）"),
    ("cfo", DIM_AMOUNT, SEM_QUALITY, "经营活动现金流净额（单季流量，PIT）"),
    ("total_assets", DIM_AMOUNT, SEM_SIZE, "总资产（时点存量，PIT）"),
    ("total_equity", DIM_AMOUNT, SEM_SIZE, "归母净资产（时点存量，PIT）"),
    ("total_liab", DIM_AMOUNT, SEM_LEVERAGE, "总负债（时点存量，PIT）"),
    ("shares", DIM_COUNT, SEM_SIZE, "总股本（时点存量，PIT）"),
    ("market_cap", DIM_AMOUNT, SEM_VALUATION, "总市值 = 收盘价 × 总股本（日频）"),
)


def register_fundamental_fields(registry: FieldRegistry) -> FieldRegistry:
    """把财务字段元数据注册进字段注册表（无数据也可先注册，便于写表达式）。"""
    registry.register_fields(FUND_FIELDS, source="fundamental",
                             role=ROLE_FUNDAMENTAL)
    return registry


# --------------------------------------------------------------------------
# 2. PIT 安装与前视自检
# --------------------------------------------------------------------------
def pit_reference(long_df: pd.DataFrame, dates: Sequence[pd.Timestamp],
                  symbol: str, value_col: str, lag_days: int = 1) -> pd.Series:
    """**独立参考实现**：用最朴素的循环写出"某交易日可见的最新一条财报"。

    与 ``asof_align`` 的向量化实现互为对照物。两者必须逐点一致——这是
    "机制上杜绝前视"从口号变成可验证约束的地方。
    """
    rec = long_df[long_df["symbol"] == symbol].copy()
    rec["_eff"] = pd.to_datetime(rec["ann_date"]) + day_delta(lag_days)
    rec = rec.sort_values("_eff")
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    eff = rec["_eff"].to_numpy()
    vals = rec[value_col].to_numpy(dtype=np.float64)
    pos = np.searchsorted(eff, idx.to_numpy(), side="right") - 1
    out = np.where(pos >= 0, vals[np.maximum(pos, 0)], np.nan)
    return pd.Series(out, index=idx, dtype=np.float64)


def install_fundamentals(panel: PanelData, long_df: pd.DataFrame,
                         fields: Optional[Sequence[str]] = None,
                         lag_days: int = 1,
                         registry: Optional[FieldRegistry] = None) -> List[str]:
    """把「公告日 + 数值」长表按 PIT 规则装进面板。

    ``long_df`` 需含 ``symbol`` / ``ann_date`` 两列，其余数值列即财务科目
    （单季流量用单季值，存量用时点值）。``market_cap`` 由 ``close × shares``
    现算，不由财报提供。
    """
    names = list(fields) if fields else [
        c for c in long_df.columns
        if c not in ("symbol", "ann_date") and pd.api.types.is_numeric_dtype(long_df[c])]
    reg = registry or panel.registry
    meta_map = {row[0]: row for row in FUND_FIELDS}
    installed: List[str] = []
    for name in names:
        if name not in long_df.columns:
            continue
        wide = asof_align(long_df, panel.dates, lag_days=lag_days,
                          value_col=name, symbol_col="symbol",
                          date_col="ann_date")
        row = meta_map.get(name)
        meta = (FieldMeta(name, row[1], row[2], ROLE_FUNDAMENTAL,
                          "fundamental", row[3]) if row else
                FieldMeta(name, DIM_AMOUNT, SEM_QUALITY, ROLE_FUNDAMENTAL,
                          "fundamental", "外部财务科目"))
        panel.add_field(name, wide, meta)
        installed.append(name)
    if "shares" in installed and "close" in panel.fields:
        cap = panel.fields["close"] * panel.fields["shares"]
        row = meta_map["market_cap"]
        panel.add_field("market_cap", cap,
                        FieldMeta("market_cap", row[1], row[2],
                                  ROLE_FUNDAMENTAL, "fundamental", row[3]))
        installed.append("market_cap")
    return installed


def check_pit(panel: PanelData, long_df: pd.DataFrame, field: str,
              lag_days: int = 1, tol: float = 1e-9) -> Dict[str, Any]:
    """复核某字段的 PIT 投影是否有前视 / 错位。

    同时对每个标的比较面板取值与 ``pit_reference`` 的朴素实现；再单独检查
    "首次可见日之前必须为 NaN"这一条（参考实现若也写错就会同错，所以这条
    独立判据不可省）。
    """
    if field not in long_df.columns:
        raise ValueError(f"长表里没有 {field!r} 列，无法复核")
    ref_errors = 0
    early_hits = 0
    checked = 0
    lag = day_delta(lag_days)
    for sym in panel.symbols:
        if sym not in set(long_df["symbol"]):
            continue
        checked += 1
        ref = pit_reference(long_df, panel.dates, sym, field, lag_days)
        got = panel.fields[field][sym]
        diff = (ref - got).abs()
        bad = diff > tol
        bad &= ~(ref.isna() & got.isna())
        ref_errors += int(bad.sum())
        rec = long_df[long_df["symbol"] == sym]
        first_eff = pd.to_datetime(rec["ann_date"]).min() + lag
        early = got.index < first_eff
        early_hits += int(got[early].notna().sum())
    return {"field": field, "symbols_checked": checked,
            "mismatch_points": ref_errors, "lookahead_points": early_hits,
            "ok": ref_errors == 0 and early_hits == 0}


# --------------------------------------------------------------------------
# 3. 合成财报（离线演示 / 测试）
# --------------------------------------------------------------------------
MIN_HISTORY_DAYS = 3 * YEAR_DAYS + 30


def check_history(panel: PanelData) -> Dict[str, Any]:
    """检查面板长度是否够算 TTM / 同比类财务因子。

    TTM 由四个季度延迟相加构成，必然占用约 3×250 个交易日的预热期。
    数据不够长时因子会几乎全 NaN——这时该报的是"数据不够长"，
    而不是让评价体系把它判成无效因子（这也是 ``coverage_adj`` 的用途，
    两者配合使用：先看历史够不够，再看扣除预热后的覆盖率）。
    """
    n = len(panel.dates)
    ok = n >= MIN_HISTORY_DAYS
    advice = "" if ok else (
        f"面板仅 {n} 个交易日，TTM/同比类财务因子需要约 {MIN_HISTORY_DAYS} "
        f"个交易日的历史；请补充历史数据，或先用环比等短窗口财务因子")
    return {"n_days": n, "needed": MIN_HISTORY_DAYS, "ok": bool(ok),
            "advice": advice}


def _quarter_ends(start: Any, end: Any) -> pd.DatetimeIndex:
    """季度末交易日序列（兼容 pandas < 2.2 的 'Q' 别名）。"""
    try:
        return pd.date_range(start=start, end=end, freq="QE")
    except ValueError:  # pragma: no cover - 老版本 pandas
        return pd.date_range(start=start, end=end, freq="Q")


def synthetic_quarterly(panel: PanelData, seed: int = 7,
                        lag_days: int = 30,
                        quality_beta: float = 1.0,
                        value_beta: float = 0.55) -> pd.DataFrame:
    """按面板的**潜在状态**合成一份季度财报长表（含真实公告日）。

    为什么必须挂钩潜在状态：如果财报与股价完全独立，那么挖掘器挖到的基本面
    因子必然没有 IC，"统一框架"就无从验证。这里把财报质量状态接到
    ``PanelData.synthetic()`` 暴露的 ``mu``（真实预期收益状态）上：

    - 高 ``mu`` 的公司 → 高 ROE、高毛利、低应计、低杠杆（**质量**方向）；
    - 同时给它们更高的 PB/PE（**估值溢价**），于是低估值类因子方向为负、
      质量类因子方向为正，两条方向相反的可检验结构同时成立。

    公告日 = 季度结束 + ``lag_days``（默认 30 天，模拟真实披露时滞）。
    """
    if not getattr(panel, "latent", None) or "mu" not in panel.latent:
        raise ValueError(
            "面板缺少潜在状态；请用 PanelData.synthetic() 构造，或直接提供真实财报长表。")
    rng = np.random.default_rng(seed)
    dates = panel.dates
    syms = panel.symbols
    n = len(syms)
    mu = panel.latent["mu"]                      # (T, N) 真实预期收益状态
    vol_z = panel.latent.get("vol_z")
    if vol_z is None:
        vol_z = np.zeros(n)
    close = panel.fields["close"].to_numpy(dtype=np.float64)

    # 质量状态 q：用 mu 的 20 日滚动均值做横截面标准化（慢变量，季度级别稳定）
    mu_s = pd.DataFrame(mu, index=dates, columns=syms).rolling(
        20, min_periods=5).mean()
    q = ops.cs_zscore(mu_s).to_numpy(dtype=np.float64)
    q = np.where(np.isfinite(q), q, 0.0) * float(quality_beta)

    # 季度报告日：覆盖到面板首日之前 900 天，保证 750 日前置窗口拿得到数据
    qend = _quarter_ends(dates[0] - day_delta(900),
                        dates[-1] + day_delta(120))
    base_equity = np.exp(rng.normal(np.log(3e9), 0.6, size=n))
    base_lev = rng.uniform(1.6, 2.6, size=n)
    turnover0 = rng.uniform(0.14, 0.26, size=n)
    gm0 = rng.uniform(0.20, 0.36, size=n)
    roe0 = rng.uniform(0.008, 0.022, size=n)

    rows: List[Dict[str, Any]] = []
    for t in qend:
        ann = t + day_delta(lag_days)
        # 用公告日之前最后一个交易日（或首日）的状态作为该季度的基本面
        pos = int(np.searchsorted(dates.to_numpy(), np.datetime64(ann)))
        pos = min(max(pos, 0), len(dates) - 1)
        qv = q[pos]                                    # (N,)
        noise = rng.normal(0.0, 0.25, size=n)
        equity = base_equity * np.exp(0.35 * qv + noise * 0.3)
        lev = base_lev - 0.45 * qv                     # 高质量 → 低杠杆
        lev = np.clip(lev, 1.2, 4.0)
        assets = equity * lev
        revenue = assets * (turnover0 + 0.06 * qv + rng.normal(0, 0.03, size=n))
        gm = np.clip(gm0 + 0.06 * qv + rng.normal(0, 0.02, size=n), 0.05, 0.75)
        gross = revenue * gm
        roe_q = roe0 * (1.0 + 0.9 * qv) + rng.normal(0, 0.004, size=n)
        profit = equity * roe_q
        cfo = profit * (1.1 + 0.35 * qv) + rng.normal(0, 0.01 * np.abs(profit) + 1.0,
                                                      size=n)
        # 估值溢价：高质量 → 高 PB/PE（value_beta 控制溢价强度）
        pb = np.clip(1.3 + float(value_beta) * qv + rng.normal(0, 0.15, size=n),
                     0.4, 6.0)
        shares = pb * equity / np.maximum(close[pos], 1e-6)
        for i, sym in enumerate(syms):
            rows.append({
                "symbol": sym, "ann_date": ann, "report_date": t,
                "revenue": float(revenue[i]), "net_profit": float(profit[i]),
                "gross_profit": float(gross[i]), "cfo": float(cfo[i]),
                "total_assets": float(assets[i]),
                "total_equity": float(equity[i]),
                "total_liab": float(assets[i] - equity[i]),
                "shares": float(shares[i]),
            })
    df = pd.DataFrame(rows).sort_values(["symbol", "ann_date"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# 4. 因子库
# --------------------------------------------------------------------------
def _ttm(x: str) -> str:
    """单季流量 → TTM（四个季度单季值之和）。"""
    return ("add(add(%s, ts_delay(%s, %d)), add(ts_delay(%s, %d), ts_delay(%s, %d)))"
            % (x, x, YEAR_DAYS, x, 2 * YEAR_DAYS, x, 3 * YEAR_DAYS))


NET_PROFIT_TTM = _ttm("net_profit")
REVENUE_TTM = _ttm("revenue")
GROSS_PROFIT_TTM = _ttm("gross_profit")
CFO_TTM = _ttm("cfo")

FUND_FACTOR_LIBRARY: Dict[str, str] = {
    # -- 规模/存量 --
    "总资产": "total_assets",
    "净资产": "total_equity",
    "市值": "market_cap",
    # -- 流量（TTM）--
    "净利润TTM": NET_PROFIT_TTM,
    "营收TTM": REVENUE_TTM,
    "毛利TTM": GROSS_PROFIT_TTM,
    "经营现金流TTM": CFO_TTM,
    # -- 成长 --
    "营收同比": "ts_yoy(revenue, %d)" % YEAR_DAYS,
    "净利润同比": "ts_yoy(net_profit, %d)" % YEAR_DAYS,
    "净利润环比": "ts_qoq(net_profit, %d)" % QUARTER_DAYS,
    "营收环比": "ts_qoq(revenue, %d)" % QUARTER_DAYS,
    "净利润增速变化": "sub(ts_yoy(net_profit, %d), ts_delay(ts_yoy(net_profit, %d), %d))"
                     % (YEAR_DAYS, YEAR_DAYS, YEAR_DAYS),
    # -- 质量（杜邦三因子 + 现金流）--
    "ROE_TTM": f"div({NET_PROFIT_TTM}, total_equity)",
    "ROA_TTM": f"div({NET_PROFIT_TTM}, total_assets)",
    "毛利率_TTM": f"div({GROSS_PROFIT_TTM}, {REVENUE_TTM})",
    "资产周转率_TTM": f"div({REVENUE_TTM}, total_assets)",
    "现金流质量": f"div({CFO_TTM}, {NET_PROFIT_TTM})",
    "应计利润": f"div(sub({NET_PROFIT_TTM}, {CFO_TTM}), total_assets)",
    # -- 杠杆与偿债 --
    "资产负债率": "div(total_liab, total_assets)",
    "权益乘数": "div(total_assets, total_equity)",
    # -- 估值 --
    "PE_TTM": f"div(market_cap, {NET_PROFIT_TTM})",
    "PB": "div(market_cap, total_equity)",
    "PS_TTM": f"div(market_cap, {REVENUE_TTM})",
    "PCF_TTM": f"div(market_cap, {CFO_TTM})",
    "盈利收益率": f"inv(div(market_cap, {NET_PROFIT_TTM}))",
    "账面市值比": "inv(div(market_cap, total_equity))",
    # -- 标准化/中性化后的常用形式 --
    "ROE_TTM_z": f"zscore_cs(div({NET_PROFIT_TTM}, total_equity))",
    "应计利润_z": f"zscore_cs(div(sub({NET_PROFIT_TTM}, {CFO_TTM}), total_assets))",
    "EP_TTM_z": f"zscore_cs(inv(div(market_cap, {NET_PROFIT_TTM})))",
    "净利润同比_z": "zscore_cs(ts_yoy(net_profit, %d))" % YEAR_DAYS,
    "规模中性ROE": f"neutral(zscore_cs(div({NET_PROFIT_TTM}, total_equity)), size)",
}


def library_errors(registry: FieldRegistry) -> Dict[str, str]:
    """逐条解析 + 类型校验整个因子库，返回 ``{名称: 报错}``（空字典即全通过）。"""
    bad: Dict[str, str] = {}
    for name, text in FUND_FACTOR_LIBRARY.items():
        try:
            node = ex.parse(text)
        except ex.ExprError as exc:
            bad[name] = f"解析失败: {exc}"
            continue
        errs = ex.validate(node, registry)
        if errs:
            bad[name] = "; ".join(errs)
    return bad


# --------------------------------------------------------------------------
# 5. 量价 × 基本面 混合模板
# --------------------------------------------------------------------------
def mixed_templates(market_expr: str, fund_expr: str,
                    windows: Sequence[int] = (20, 60, 120),
                    groups: Sequence[int] = (5,)) -> Dict[str, str]:
    """给定一条量价因子与一条基本面因子，返回跨域组合模板。

    每一类都对应研报里的一种"统一"方式：

    - ``mul``：双排序打分（两条信号需同时成立）；
    - ``spread`` / ``sub``：相对强弱（量价强而基本面弱的股票）；
    - ``div``：以基本面为分母的强度归一；
    - ``ts_corr`` / ``ts_beta``：量价与基本面的**联动关系**（同源变量）；
    - ``ts_reg_resi``：滚动正交残差，量价因子中基本面解释不掉的部分；
    - ``neutral``：横截面线性正交（一次性回归，比滚动更稳）；
    - ``dgtw_cs``：按基本面分组后组内调整（吸收非线性关系）。
    """
    m, f = market_expr, fund_expr
    out: Dict[str, str] = {
        "双排序": f"mul(rank_cs({m}), rank_cs({f}))",
        "相对强弱": f"spread(zscore_cs({m}), zscore_cs({f}))",
        "量价减基本面": f"sub(zscore_cs({m}), zscore_cs({f}))",
        "量价除以基本面": f"div({m}, {f})",
        "横截面正交": f"neutral({m}, zscore_cs({f}))",
    }
    for w in windows:
        out[f"时序相关{w}"] = f"ts_corr({m}, {f}, {int(w)})"
        out[f"时序beta{w}"] = f"ts_beta({m}, {f}, {int(w)})"
        out[f"滚动正交残差{w}"] = f"ts_reg_resi({m}, {f}, {int(w)})"
    for g in groups:
        out[f"基本面分组{g}"] = f"dgtw_cs({m}, zscore_cs({f}), {int(g)})"
    return out


def market_factor_library() -> Dict[str, str]:
    """常用量价因子（混合模板的另一半输入）。"""
    return {
        "动量": "zscore_cs(ts_pct(close, 60))",
        "反转": "zscore_cs(ts_pct(close, 5))",
        "波动": "zscore_cs(ts_std(ret, 20))",
        "流动性": "rank_cs(amount)",
        "换手": "zscore_cs(ts_mean(turnover, 20))",
        "量价背离": "ts_corr(close, volume, 60)",
        "振幅": "zscore_cs(ts_mean(amplitude, 20))",
        "距高点": "zscore_cs(ts_max_diff(close, 120))",
        "非流动性": "zscore_cs(div(ts_std(ret, 20), ts_mean(amount, 20)))",
    }
