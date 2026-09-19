"""挖掘后的深度分析入口（src/pipeline/deep_analysis.py）。

把「回测 → 多因子体系 → 统计体检 → 图表 → 大模型解读」串成一次调用，
供 UI、Agent 与脚本统一使用，避免每个调用方各拼一遍。

    from pipeline.deep_analysis import deep_analysis
    res = deep_analysis(kline, factor, factor_name="mom_rev", factor_expr="mom - rev")
    print(res["report"].to_markdown())
    print(res["report"].charts)          # 图表路径
    print(res["report"].interpretation)  # 大模型解读（LLM 不可用时自动降级）

单因子时只做回测 + 报告；传入 ``extra_factors``（多个候选因子的截面矩阵）
时会额外跑一次多因子体系搭建，给出「线性 baseline / 非线性 / 树 / contextual」
的逐层增益与推荐方案。
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import pandas as pd

from engine.backtest import FactorBacktester
from reporting.factor_report import FactorReport, generate_factor_report

logger = logging.getLogger("factor_gpt.deep_analysis")


def _to_factor_series(factor) -> pd.Series:
    """统一因子输入：Series 直接用；DataFrame 取第一列并尽力构造 (date, symbol) 索引。"""
    if isinstance(factor, pd.Series):
        return factor
    if isinstance(factor, pd.DataFrame):
        s = factor.iloc[:, 0]
        if not isinstance(s.index, pd.MultiIndex) and {"date", "symbol"}.issubset(factor.columns):
            s.index = pd.MultiIndex.from_arrays([factor["date"], factor["symbol"]])
        return s
    raise TypeError("factor 需为 Series 或 DataFrame。")


def deep_analysis(
    kline: pd.DataFrame,
    factor,
    factor_name: str = "",
    factor_expr: str = "",
    extra_factors: Optional[pd.DataFrame] = None,
    output_dir: str = "output/factor_report",
    llm_client=None,
    n_quantiles: int = 10,
    forward_periods: int = 1,
    commission: float = 0.001,
    risk_free_rate: float = 0.03,
    run_portfolio: bool = True,
    run_system: bool = False,
    top_frac: float = 0.1,
) -> Dict:
    """因子挖掘后的完整深度分析。

    Args:
        kline: 行情长表，至少含 date / symbol / close 三列。
        factor: 因子值，Series（推荐 MultiIndex(date, symbol)）或 DataFrame。
        factor_name / factor_expr: 进入大模型上下文的名称与表达式。
        extra_factors: 多因子截面矩阵（可选，用于体系搭建与相关性图）。
        output_dir: 图表与 Markdown 报告输出目录。
        llm_client: 已构造的 LLMClient；缺省自动构建，失败则降级为规则化结论。
        run_portfolio: 是否跑 A 股约束下的组合级回测（涨跌停/手续费/印花税）。
        run_system: 是否跑多因子体系搭建（仅在提供 extra_factors 时有意义）。

    Returns:
        {"metrics", "portfolio", "report", "system"}，其中 report 为 FactorReport。
    """
    fac = _to_factor_series(factor)
    bt = FactorBacktester(n_quantiles=n_quantiles, forward_periods=forward_periods,
                          commission=commission, risk_free_rate=risk_free_rate)

    metrics: Dict = {}
    try:
        metrics = bt.evaluate(kline, fac, verbose=False) or {}
    except Exception as e:  # noqa: BLE001
        logger.warning("回测失败：%s: %s", type(e).__name__, e)

    portfolio: Optional[Dict] = None
    if run_portfolio and metrics:
        try:
            portfolio = bt.realistic_portfolio(kline, fac, top_frac=top_frac)
        except Exception as e:  # noqa: BLE001
            logger.warning("组合级回测失败：%s: %s", type(e).__name__, e)
            portfolio = None

    report: FactorReport = generate_factor_report(
        metrics, portfolio=portfolio, factor_name=factor_name, factor_expr=factor_expr,
        factors=extra_factors, output_dir=output_dir, llm_client=llm_client,
    )

    system = None
    if run_system:
        system = _build_system_from_kline(kline, fac, extra_factors, forward_periods)

    return {"metrics": metrics, "portfolio": portfolio, "report": report, "system": system}


def _build_system_from_kline(kline: pd.DataFrame, factor: pd.Series,
                             extra_factors: Optional[pd.DataFrame],
                             forward_periods: int):
    """用回测面板构造多因子体系（需要至少两个因子才能体现「体系」的意义）。"""
    try:
        from engine.factor_synthesis import build_factor_system

        cols: Dict[str, pd.Series] = {"factor": factor}
        if extra_factors is not None and not extra_factors.empty:
            for c in extra_factors.columns:
                if c not in ("date", "symbol"):
                    cols[str(c)] = extra_factors[c]
        if len(cols) < 2:
            return None

        df = kline[["date", "symbol", "close"]].copy()
        df["date"] = df["date"].astype(str)
        df = df.sort_values(["symbol", "date"])
        df["ret"] = df.groupby("symbol")["close"].pct_change()
        df["fwd_ret"] = df.groupby("symbol")["ret"].shift(-max(1, int(forward_periods)))

        base = pd.Series(factor.values, index=pd.MultiIndex.from_arrays(
            [factor.index.get_level_values(0).astype(str), factor.index.get_level_values(1)]
        )) if isinstance(factor.index, pd.MultiIndex) else None
        if base is None:
            return None
        panel = df.join(base.rename("factor"), on=["date", "symbol"])
        if extra_factors is not None and not extra_factors.empty:
            ef = extra_factors.copy()
            if "date" in ef.columns:
                ef["date"] = ef["date"].astype(str)
                panel = panel.merge(ef, on=["date", "symbol"], how="left")
        panel = panel.dropna(subset=["fwd_ret"])
        factor_cols = [c for c in ("factor", *[str(c) for c in
                                               (extra_factors.columns if extra_factors is not None else [])])
                       if c in panel.columns and c not in ("date", "symbol")]
        if len(factor_cols) < 2:
            return None
        return build_factor_system(panel, factor_cols)
    except Exception as e:  # noqa: BLE001
        logger.warning("多因子体系搭建失败：%s: %s", type(e).__name__, e)
        return None


def write_report_bundle(res: Dict, output_dir: str = "output/factor_report") -> Dict[str, str]:
    """把深度分析结果落盘：Markdown 报告 + 指标 CSV，返回路径字典。"""
    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    rep = res.get("report")
    if rep is None:
        return paths
    try:
        p = os.path.join(output_dir, "factor_report.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write(rep.to_markdown())
        paths["markdown"] = p
    except Exception as e:  # noqa: BLE001
        logger.warning("Markdown 落盘失败：%s", e)
    tbl = getattr(rep, "tables", {}).get("metrics")
    if isinstance(tbl, pd.DataFrame) and not tbl.empty:
        try:
            p = os.path.join(output_dir, "metrics.csv")
            tbl.to_csv(p, index=False, encoding="utf-8-sig")
            paths["metrics_csv"] = p
        except Exception as e:  # noqa: BLE001
            logger.warning("指标 CSV 落盘失败：%s", e)
    return paths
