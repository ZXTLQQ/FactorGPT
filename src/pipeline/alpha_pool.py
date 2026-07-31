"""PART-05 合金配比 · AlphaPool 合成 (Alloy Formulation)。

将精选因子合成为高质量复合因子：
  * 配方设计：类比「炼丹术」式调参，权重由各因子 ICIR 方向决定；
  * 正交化：对候选因子做序列正交（Gram-Schmidt），去除冗余、降低共线性；
  * 过拟合控制：leave-one-out 测试，检验复合因子对单一因子的依赖度；
  * 迭代优化：坐标上升微调权重以最大化复合 ICIR。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from agent.rl_search import _fast_icir
from pipeline.schema import CandidateFactor

logger = logging.getLogger("factor_gpt.alpha_pool")

DATE = "date"
SYMBOL = "symbol"


@dataclass
class AlphaPoolConfig:
    ortho: bool = True
    loo: bool = True
    iterative: bool = True
    n_iter: int = 20
    robustness: bool = True  # 过拟合检验（walk-forward / DSR / 参数稳定性）


class AlphaPool:
    """因子池合成器。"""

    def __init__(self, config: Optional[AlphaPoolConfig] = None):
        self.config = config or AlphaPoolConfig()

    # -- 正交化（Gram-Schmidt，截面独立） -------------------------------- #
    @staticmethod
    def _orthogonalize(matrix: np.ndarray) -> np.ndarray:
        """对列向量做正交化（去除因子间冗余）。matrix: (N, K)。"""
        N, K = matrix.shape
        Q = np.zeros((N, K))
        for k in range(K):
            v = matrix[:, k].copy()
            for j in range(k):
                v = v - (np.dot(Q[:, j], matrix[:, k]) / (np.dot(Q[:, j], Q[:, j]) + 1e-12)) * Q[:, j]
            norm = np.linalg.norm(v)
            Q[:, k] = v / (norm + 1e-12) if norm > 1e-12 else v
        return Q

    # -- 权重 ------------------------------------------------------------ #
    def _icir_weights(self, candidates: List[CandidateFactor]) -> np.ndarray:
        w = np.array([float(c.metrics.get("icir", 0) or 0) for c in candidates])
        # 仅保留正 ICIR 方向，按幅度加权
        w = np.where(w > 0, w, 0.0)
        s = w.sum()
        if s <= 0:
            w = np.ones(len(candidates)) / len(candidates)
        else:
            w = w / s
        return w

    # -- 合成 ------------------------------------------------------------ #
    def synthesize(self, candidates: List[CandidateFactor], kline: pd.DataFrame) -> pd.Series:
        names = [c.name for c in candidates]
        series = [c.series for c in candidates]
        common = series[0].index
        for s in series[1:]:
            common = common.intersection(s.index)
        X = np.column_stack([s.reindex(common).to_numpy(dtype=float) for s in series])
        X = np.nan_to_num(X)
        w = self._icir_weights(candidates)

        if self.config.ortho and X.shape[1] > 1:
            Q = self._orthogonalize(X)
            comp = Q @ w
        else:
            comp = X @ w

        comp_series = pd.Series(comp, index=common)
        comp_series = comp_series.groupby(level=0, group_keys=False).rank(pct=True)
        return comp_series

    # -- leave-one-out 过拟合检验 ---------------------------------------- #
    def leave_one_out(self, candidates: List[CandidateFactor], kline: pd.DataFrame) -> Dict:
        if not self.config.loo or len(candidates) < 3:
            return {"enabled": False}
        full = self.synthesize(candidates, kline)
        base_icir = _fast_icir(full, kline)
        deltas = {}
        for i, c in enumerate(candidates):
            sub = candidates[:i] + candidates[i + 1:]
            comp = self.synthesize(sub, kline)
            icir = _fast_icir(comp, kline)
            deltas[c.name] = {
                "icir_without": float(icir),
                "delta": float(icir - base_icir),
            }
        # 依赖度：去掉后 ICIR 下降最多者，对复合因子贡献最大（也可能是过拟合来源）
        most_dependent = min(deltas.items(), key=lambda kv: kv[1]["delta"])
        return {
            "enabled": True,
            "base_icir": float(base_icir),
            "per_factor": deltas,
            "most_dependent_factor": most_dependent[0],
            "most_dependent_drop": float(most_dependent[1]["delta"]),
        }

    # -- 迭代优化（坐标上升） -------------------------------------------- #
    def optimize(self, candidates: List[CandidateFactor], kline: pd.DataFrame) -> pd.Series:
        if not self.config.iterative or len(candidates) < 2:
            return self.synthesize(candidates, kline)
        names = [c.name for c in candidates]
        series = [c.series for c in candidates]
        common = series[0].index
        for s in series[1:]:
            common = common.intersection(s.index)
        X = np.nan_to_num(np.column_stack([s.reindex(common).to_numpy(dtype=float) for s in series]))
        w = self._icir_weights(candidates)
        if X.shape[1] > 1 and self.config.ortho:
            X = self._orthogonalize(X)
        best_icir, best_w = -1e9, w.copy()
        for _ in range(self.config.n_iter):
            improved = False
            for k in range(X.shape[1]):
                for delta in (0.05, -0.05):
                    trial = w.copy()
                    trial[k] = max(0.0, trial[k] + delta)
                    s_sum = trial.sum()
                    if s_sum <= 0:
                        continue
                    trial /= s_sum
                    comp = pd.Series(X @ trial, index=common).groupby(level=0, group_keys=False).rank(pct=True)
                    icir = _fast_icir(comp, kline)
                    if icir > best_icir + 1e-6:
                        best_icir, best_w, w = icir, trial.copy(), trial.copy()
                        improved = True
            if not improved:
                break
        comp = pd.Series(X @ best_w, index=common).groupby(level=0, group_keys=False).rank(pct=True)
        logger.info("AlphaPool 迭代优化后 ICIR=%.3f", best_icir)
        return comp

    # -- 过拟合检验（P0，对应 pipeline/robustness.py） ---------------------- #
    def robustness_check(
        self,
        composite: pd.Series,
        kline: pd.DataFrame,
        n_trials: int = 10,
        label: str = "composite",
    ) -> Dict:
        """对合成后的复合因子执行过拟合检验（walk-forward / DSR / 参数稳定性）。"""
        if not self.config.robustness:
            return {"enabled": False}
        from pipeline.robustness import RobustnessValidator

        validator = RobustnessValidator(forward_periods=1, n_quantiles=5)
        return {
            "enabled": True,
            **validator.validate(kline, composite, n_trials=n_trials, label=label),
        }
