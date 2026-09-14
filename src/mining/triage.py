# -*- coding: utf-8 -*-
"""验收闸门：把挖掘产出接到统计检验、选股域与分层多尺度挖掘。

``src/mining`` 回答"能挖出什么"，这一层回答"挖出来的东西值不值得信"。后者需要
三样挖掘层手上没有的输入 —— IC 序列的**时序结构**、这批候选一共**试了多少次**、
以及同一因子在**不同选股域**下的表现差异。它们分别落在：

======================  ==========================================================
``engine.significance``  平稳块 bootstrap + 有效样本量 t 检验 + 选择校正 + BH FDR
``engine.universe``      逐日可交易标签（含剔除原因）与宽/中/严三档域对照
``engine.multiscale_gp`` 粗网格演化 → 失真区间诊断 → 局部加密 + 粗尺度终端代价
======================  ==========================================================

而三者的入参都是**长表**，挖掘层手上是宽表 :class:`mining.panel.PanelData`：
本模块就是那层薄适配（:func:`panel_long` / :func:`kline_long`），外加两个不容
含糊的口径：

1. ``n_trials`` 必须传**本次实际评估过的表达式数**（``SearchResult.n_evaluated``），
   否则选择校正退化成"用候选个数判候选"——把门槛压低到刚好能过；
2. 前瞻收益一律走 ``attach_forward_return``（组内 ``shift(−h)``、末尾置 NaN 而非 0）。

依赖口径与 ``report`` 一致：``engine`` 包的导入放在函数内部（:func:`_engine`），
因为 ``engine/__init__`` 会拉起遗传 / 非结构化等重模块，而 ``import mining.triage``
应当保持轻量；入参缺失时返回空结果，不抛异常。
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "KLINE_FIELDS", "panel_long", "kline_long", "ic_series_of",
    "significance_check", "significance_for_search", "quantile_domains",
    "universe_check", "domain_check", "multiscale_mine", "acceptance",
]

#: GP 算子集（``engine.genetic_enhanced._COLS``）要求的原始列。
KLINE_FIELDS: Tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount")


def _engine(name: str) -> Any:
    """按需导入 ``engine.<name>``（避免 ``import mining.triage`` 拉起重依赖）。"""
    return importlib.import_module(f"engine.{name}")


# --------------------------------------------------------------------------
# 1. 宽表 → 长表
# --------------------------------------------------------------------------
def panel_long(panel: Any, fields: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """把面板转成长表 ``date / symbol / 各字段``。

    Args:
        panel: :class:`mining.panel.PanelData`（宽表：索引为交易日、列为标的）。
        fields: 需要的字段名；``None`` 取面板已有字段。名字不存在则跳过，
            不因为"顺手多写了个字段"整段失败。

    Returns:
        长表；含 ``pct_chg``（百分数口径，由 ``close`` 现算补齐）——缺了它，
        GP 的 ``random_expr`` 一旦采样到 ``pct_chg`` 就会在求值时 KeyError。
    """
    names = list(fields) if fields else sorted(getattr(panel, "fields", {}) or {})
    wide: Dict[str, pd.DataFrame] = {}
    for name in names:
        if name in panel:
            try:
                wide[name] = panel.field(name)
            except (KeyError, ValueError):
                continue
    if "close" not in wide:
        raise ValueError("面板缺少 close 字段，无法构造前瞻收益")
    long = type(panel)._to_long(wide)          # 与 from_kline 同一入口，口径一致
    if "pct_chg" not in long.columns:
        prev = long.groupby("symbol")["close"].shift(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            long["pct_chg"] = (long["close"].astype(float) / prev - 1.0) * 100.0
    return long


def kline_long(panel: Any, horizon: int = 5,
               fields: Sequence[str] = KLINE_FIELDS) -> pd.DataFrame:
    """面板 → 带 ``fwd_ret`` 的长表（选股域与分层多尺度挖掘共用同一份）。"""
    long = panel_long(panel, fields=fields)
    return _engine("multiscale_gp").attach_forward_return(long, horizon=int(horizon))


# --------------------------------------------------------------------------
# 2. 统计显著性
# --------------------------------------------------------------------------
def ic_series_of(reports: Iterable[Any]) -> List[Tuple[str, Any]]:
    """从 ``FactorReport`` 列表里取 ``[(因子名, IC 序列), ...]``（缺序列的跳过）。"""
    out: List[Tuple[str, Any]] = []
    for rep in reports or ():
        detail = getattr(rep, "detail", None) or {}
        ic = detail.get("ic_series") if isinstance(detail, Mapping) else None
        if ic is None or len(ic) == 0:
            continue
        name = getattr(rep, "name", None) or detail.get("name") or f"factor_{len(out) + 1}"
        out.append((str(name), ic))
    return out


def significance_check(ic_map: Any, n_trials: Optional[int] = None,
                       alpha: float = 0.05, n_boot: int = 1000,
                       seed: int = 42, min_dates: int = 20) -> Dict[str, Any]:
    """对一批候选的 IC 序列做完整显著性判定。

    Args:
        ic_map: ``{因子名: IC 序列}`` 或 ``[(名字, IC 序列), ...]``。
        n_trials: 本次挖掘评估过的表达式数；缺省退化为候选个数，此时
            ``selection.ic_crit`` 只扣掉了"表里这 m 个"，务必在结果里看清该口径。
        alpha / n_boot / seed / min_dates: 转交 ``engine.significance``。

    Returns:
        ``{table, summary, markdown, warning, n_trials, n_candidates, min_dates}``；
        无候选时 ``table`` 为空表（不抛异常，界面按"无候选"展示）。
    """
    items: List[Tuple[str, Any]] = (list(ic_map.items()) if isinstance(ic_map, Mapping)
                                    else list(ic_map or ()))
    sig = _engine("significance")
    m = int(n_trials) if n_trials else max(1, len(items))
    out = sig.summarise(items, n_trials=m, alpha=float(alpha), n_boot=int(n_boot),
                        seed=int(seed), min_dates=int(min_dates))
    out["n_trials"] = m
    out["n_candidates"] = len(items)
    # 摘要里没有回传 min_dates，报告层要按实际口径渲染"样本量 ≥ N 期"，故补上
    out["min_dates"] = int(min_dates)
    return out


def significance_for_search(search: Any, top_k: int = 10,
                            **kwargs: Any) -> Dict[str, Any]:
    """直接对一次网格搜索的结果做检验（尝试次数取 ``n_evaluated``）。"""
    reports: Sequence[Any] = ()
    best = getattr(search, "best_reports", None)
    if callable(best):
        reports = list(best(int(top_k)) or ())
    items = ic_series_of(reports)
    n_trials = int(getattr(search, "n_evaluated", 0) or 0)
    out = significance_check(items, n_trials=n_trials or len(items), **kwargs)
    out["top_k"] = len(items)
    out["stop_reason"] = getattr(search, "stop_reason", None)
    return out


# --------------------------------------------------------------------------
# 3. 选股域
# --------------------------------------------------------------------------
def quantile_domains(panel: Any, top_n: int = 50,
                     quantiles: Sequence[float] = (0.20, 0.50, 0.80)) -> List[Any]:
    """按**面板自身**的成交额分位生成宽/中/严三档域。

    引擎自带的 ``preset_domains`` 用绝对成交额门槛（5e6 / 2e7 / 1e8 元），在真实
    A 股面板上是合理的；但合成面板、或只取了部分标的/部分区间的面板量纲不同，
    硬套会出现"严域 0 只可交易"——那不是域定义错了，是标尺拿错了。这里改为
    相对分位，三档在任何面板上都可比；``min_history`` 仍按 20/60/120 递增，
    "越严的域要求越长的上市历史"这一条保持不变。
    """
    uni = _engine("universe")
    long = panel_long(panel, fields=("close", "volume", "amount"))
    amount = pd.to_numeric(long.get("amount"), errors="coerce")
    if amount is None:
        amount = (pd.to_numeric(long["close"], errors="coerce")
                  * pd.to_numeric(long["volume"], errors="coerce"))
    amount = pd.to_numeric(amount, errors="coerce")
    if not amount.notna().any():
        raise ValueError("面板无成交额信息（amount 或 close×volume），无法分档")
    labels = ("宽域（成交额后 20% 分位起）", "中域（成交额 50% 分位起）",
              "严域（成交额 80% 分位起）")
    out = []
    for q, hist, label in zip(quantiles, (20, 60, 120), labels):
        out.append(uni.UniverseSpec(min_history=int(hist),
                                    min_amount=float(amount.quantile(float(q))),
                                    top_n=int(top_n), label=label))
    return out


def universe_check(panel: Any, spec: Optional[Any] = None, horizon: int = 5,
                   keep_flags: bool = True) -> Dict[str, Any]:
    """单档选股域的体检：可交易标签、每日可选数量、剔除原因分布。

    ``spec=None`` 取 :func:`quantile_domains` 的中域，这样与
    :func:`domain_check` 的对照表口径一致；``keep_flags=False`` 时不回传逐行标签
    （大面板落报告会撑爆 JSON）。
    """
    uni = _engine("universe")
    spec = spec or quantile_domains(panel)[1]
    flags = uni.build_universe(kline_long(panel, horizon=horizon), spec,
                               keep_cols=("close", "pct_chg"))
    out = uni.universe_summary(flags)
    out["spec"] = spec.as_dict()
    if keep_flags:
        out["flags"] = flags
    return out


def domain_check(panel: Any, factor: Any, specs: Optional[Sequence[Any]] = None,
                 horizon: int = 5, min_count: int = 5,
                 top_n: int = 50) -> pd.DataFrame:
    """同一因子在宽/中/严三档域下的表现对照（IC / ICIR / 覆盖 / 换手）。

    Args:
        panel: :class:`mining.panel.PanelData`。
        factor: 因子值，宽表（交易日 × 标的）或 ``MultiIndex(date, symbol)`` 序列。
        specs: 域定义；``None`` 取 :func:`quantile_domains` 三档（按面板自身成交额
            分位），需要引擎的绝对门槛三档时显式传 ``universe.preset_domains()``。
        horizon / min_count / top_n: 前瞻期、截面最少只数、每域建仓数。

    Returns:
        每域一行的对照表；域不是越严越好 —— 覆盖 30 只与覆盖 800 只是两个产品。
    """
    uni = _engine("universe")
    if isinstance(factor, pd.DataFrame):
        wide = factor.reindex(index=panel.dates, columns=panel.symbols)
        piece = type(panel)._to_long({"factor": wide})
    elif isinstance(factor, pd.Series) and isinstance(factor.index, pd.MultiIndex):
        piece = (factor.rename("factor")
                 .rename_axis(index=["date", "symbol"]).reset_index())
    else:
        raise TypeError("factor 需为宽表 DataFrame 或 MultiIndex(date, symbol) 序列")
    long = kline_long(panel, horizon=horizon).merge(piece, on=["date", "symbol"],
                                                   how="left")
    specs = list(specs) if specs else quantile_domains(panel, top_n=int(top_n))
    return uni.evaluate_domains(long, specs, factor_col="factor", y_col="fwd_ret",
                                min_count=int(min_count))


# --------------------------------------------------------------------------
# 4. 分层多尺度挖掘
# --------------------------------------------------------------------------
def multiscale_mine(panel: Any, horizon: int = 5, *, n_intervals: int = 8,
                    n_select: int = 2, n_folds: int = 1, seed: int = 42,
                    levels: Optional[Sequence[Any]] = None,
                    terminal_weight: float = 0.35,
                    **kwargs: Any) -> Dict[str, Any]:
    """在同一份挖掘面板上跑分层多尺度挖掘（粗演化 → 区间诊断 → 局部加密）。

    ``n_folds >= 2`` 时启用文献的多折细化：在已选父区间内部再做一轮
    Step 2 诊断，只有得分最差的子区间才消耗第三尺度预算。返回 ``mine()``
    的结果并补三个面板口径字段；额外 kwargs 转交
    :class:`engine.multiscale_gp.HierarchicalFactorMiner`。
    """
    ms = _engine("multiscale_gp")
    long = kline_long(panel, horizon=horizon)
    miner = ms.HierarchicalFactorMiner(
        long, y_col="fwd_ret", levels=levels, n_intervals=int(n_intervals),
        n_select=int(n_select), n_folds=int(n_folds), seed=int(seed),
        terminal_weight=float(terminal_weight), **kwargs)
    out = miner.mine()
    out["horizon"] = int(horizon)
    symbols, dates = getattr(panel, "symbols", None), getattr(panel, "dates", None)
    out["n_symbols"] = int(len(symbols)) if symbols is not None else 0
    out["n_dates_panel"] = int(len(dates)) if dates is not None else 0
    return out


# --------------------------------------------------------------------------
# 5. 一次跑齐（报告 / 界面共用入口）
# --------------------------------------------------------------------------
def acceptance(panel: Any, reports: Sequence[Any] = (), search: Any = None, *,
               factor: Any = None, horizon: int = 5, alpha: float = 0.05,
               n_boot: int = 1000, seed: int = 42, top_k: int = 10,
               top_n: int = 50, min_count: int = 5, n_intervals: int = 8,
               n_select: int = 2, do_domain: bool = True,
               do_multiscale: bool = True) -> Dict[str, Any]:
    """三项验收一次跑齐：统计显著性 / 选股域 / 分层多尺度挖掘。

    任一项的输入缺失（没有候选、没给因子、主动关闭）就跳过该项并把原因写进
    ``skipped``，而不是给一个"看起来跑过了"的空结果。
    """
    out: Dict[str, Any] = {"skipped": {}}

    # 显著性优先取"这次搜索自己选出来的候选"：多重性校正是对搜索收的税，
    # 拿手工挑的因子去套别人搜索的尝试次数，方向上是保守但口径是错的。
    if search is not None:
        sig = significance_for_search(search, top_k=top_k, alpha=alpha,
                                      n_boot=n_boot, seed=seed)
        miss = "搜索没有产生可复核的候选报告"
    else:
        sig = significance_check(ic_series_of(reports), alpha=alpha,
                                 n_boot=n_boot, seed=seed)
        miss = "没有带 IC 序列的候选因子"
    if sig.get("n_candidates"):
        out["significance"] = sig
    else:
        out["skipped"]["significance"] = miss

    if not do_domain:
        out["skipped"]["domains"] = "已关闭选股域对照"
    elif factor is None:
        out["skipped"]["domains"] = "未提供因子值"
    else:
        specs = quantile_domains(panel, top_n=int(top_n))
        out["domains"] = domain_check(panel, factor, specs=specs, horizon=horizon,
                                      min_count=min_count, top_n=top_n)
        out["universe"] = universe_check(panel, spec=specs[1], horizon=horizon,
                                         keep_flags=False)

    if do_multiscale:
        out["multiscale"] = multiscale_mine(
            panel, horizon=horizon, n_intervals=n_intervals, n_select=n_select,
            seed=seed)
    else:
        out["skipped"]["multiscale"] = "已关闭分层多尺度挖掘"
    return out
