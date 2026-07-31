"""PART-03 研磨车间 · RPN 求值引擎 (Refinery Performance & Numerics Engine).

对现有 `FactorBacktester` 进行「可插拔式」封装，提供：
  * 指标注册表（RPN_METRICS）：支持评估指标灵活扩展
  * 算法稳定性评估模块（stability_score）
  * 多进程并行批量求值（Daily 多进程并行计算）
  * 目标函数：最大化 Alpha 收益 + 控制波动率

该模块不修改原有 `FactorBacktester`，仅在其之上做能力增强，保证向后兼容。
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest import FactorBacktester

logger = logging.getLogger("factor_gpt.rpn")

DATE = "date"
SYMBOL = "symbol"


@dataclass
class RPNConfig:
    """RPN 引擎配置。"""
    n_quantiles: int = 5
    forward_periods: int = 1          # IC 预测周期（日频）
    commission: float = 0.001
    risk_free_rate: float = 0.03
    # 目标函数中波动率（换手率）惩罚权重：最大化 Alpha 收益 + 控制波动率
    w_turnover_penalty: float = 0.2
    # 并行批量求值开关
    parallel: bool = True
    n_workers: int = 4
    # 稳定性评估参数
    stab_turnover_cap: float = 1.0    # 换手率归一化上限


# --------------------------------------------------------------------------- #
# 指标注册表：可插拔式设计，支持评估指标灵活扩展
# --------------------------------------------------------------------------- #
RPN_METRICS: Dict[str, Callable[[Dict], float]] = {}


def register_metric(name: str):
    """装饰器：将函数注册为 RPN 自定义指标（输入为 backtester 返回的 metrics 字典）。"""
    def _wrap(fn: Callable[[Dict], float]):
        RPN_METRICS[name] = fn
        return fn
    return _wrap


@register_metric("stability_score")
def stability_score(metrics: Dict) -> float:
    """算法稳定性评估：方向正确的信息比 × 正显著比例 ÷（1 + 换手惩罚）。

    反映因子在“有效（IC>0 且 ICIR 高）”、“稳健（正显著比例高）”、“低磨损（换手低）”三者的均衡。
    """
    ic = float(metrics.get("ic_mean", 0.0) or 0.0)
    icir = float(metrics.get("icir", 0.0) or 0.0)
    ic_pos = float(metrics.get("ic_pos_ratio", 0.5) or 0.5)
    turn = float(metrics.get("turnover", 0.0) or 0.0)
    cfg_cap = RPNConfig().stab_turnover_cap
    turn_norm = min(turn, cfg_cap) / cfg_cap
    direction = icir if ic > 0 else -abs(icir)
    score = direction * ic_pos / (1.0 + 2.0 * turn_norm)
    return float(np.clip(score, -3.0, 3.0))


@register_metric("objective")
def objective_default(metrics: Dict) -> float:
    """默认目标函数：最大化 Alpha（信息比） - 控制波动率（换手惩罚）。"""
    w = RPNConfig().w_turnover_penalty
    icir = float(metrics.get("icir", 0.0) or 0.0)
    ic = float(metrics.get("ic_mean", 0.0) or 0.0)
    turn = float(metrics.get("turnover", 0.0) or 0.0)
    direction = icir if ic > 0 else -abs(icir)
    return direction - w * turn


# --------------------------------------------------------------------------- #
# RPN 引擎
# --------------------------------------------------------------------------- #
class RPNEngine:
    """对 `FactorBacktester` 的增强封装：可插拔指标 + 稳定性 + 并行批量求值。"""

    def __init__(self, config: Optional[RPNConfig] = None):
        self.config = config or RPNConfig()

    # -- 单因子求值 -------------------------------------------------------- #
    def evaluate(self, factor: pd.Series, kline: pd.DataFrame) -> Dict:
        """对单个因子求值，并补充 RPN 自定义指标。

        Parameters
        ----------
        factor : 多级索引 (date, symbol) 的因子值 Series
        kline  : 行情面板 DataFrame（含 date, symbol, close 等）
        """
        bt = FactorBacktester(
            n_quantiles=self.config.n_quantiles,
            forward_periods=self.config.forward_periods,
            commission=self.config.commission,
            risk_free_rate=self.config.risk_free_rate,
        )
        metrics = bt.evaluate(kline, factor)
        metrics = self._attach_rpn_metrics(metrics)
        return metrics

    # -- 多因子并行批量求值 ------------------------------------------------ #
    def evaluate_batch(self, factors: Dict[str, pd.Series], kline: pd.DataFrame) -> Dict[str, Dict]:
        """多进程并行计算一组因子的 RPN 指标。Daily 多进程并行计算。"""
        if not self.config.parallel or len(factors) <= 1:
            return {name: self.evaluate(f, kline) for name, f in factors.items()}

        items = list(factors.items())
        args = [
            (name, f, kline, self.config.n_quantiles, self.config.forward_periods,
             self.config.commission, self.config.risk_free_rate)
            for name, f in items
        ]
        with mp.Pool(processes=min(self.config.n_workers, len(items))) as pool:
            results = pool.map(_eval_worker, args)
        return {name: m for name, m in zip([n for n, _ in items], results)}

    # -- 目标函数 / 排序 --------------------------------------------------- #
    def objective(self, metrics: Dict, w_turnover_penalty: Optional[float] = None) -> float:
        w = w_turnover_penalty if w_turnover_penalty is not None else self.config.w_turnover_penalty
        icir = float(metrics.get("icir", 0.0) or 0.0)
        ic = float(metrics.get("ic_mean", 0.0) or 0.0)
        turn = float(metrics.get("turnover", 0.0) or 0.0)
        direction = icir if ic > 0 else -abs(icir)
        return direction - w * turn

    def rank_by_icir(self, metrics_map: Dict[str, Dict]) -> List[Tuple[str, float]]:
        ranked = [(n, float(m.get("icir", 0.0) or 0.0)) for n, m in metrics_map.items()]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    # -- 内部工具 ---------------------------------------------------------- #
    def _attach_rpn_metrics(self, metrics: Dict) -> Dict:
        out = dict(metrics)
        for name, fn in RPN_METRICS.items():
            try:
                out[name] = fn(metrics)
            except Exception as e:  # 自定义指标异常不影响主流程
                logger.warning("RPN 指标 %s 计算失败: %s", name, e)
                out[name] = float("nan")
        return out


def _eval_worker(args) -> Dict:
    name, factor, kline, n_q, fwd, comm, rf = args
    try:
        bt = FactorBacktester(
            n_quantiles=n_q, forward_periods=fwd, commission=comm, risk_free_rate=rf
        )
        metrics = bt.evaluate(kline, factor)
    except Exception as e:
        metrics = {"error": str(e)}
    metrics = RPNEngine()._attach_rpn_metrics(metrics)
    return metrics
