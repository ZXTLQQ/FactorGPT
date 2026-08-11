"""PART-04 三级筛选 · 浮选工艺 (Flotation Screener)。

剔除冗余与低质量因子，三级过滤：
  第一级：LASSO 回归筛选 —— 剔除不显著的冗余特征（保留非零系数因子）；
  第二级：Partice 协同平台 —— 人机交互式筛选，支持可视化分析（可注入 review 回调）；
  第三级：TOP 10% 截断 —— 仅保留 ICIR 最高的前 10% 因子。

设计原则：每级可独立开关，且「人机协同」通过回调解耦，便于接入 Web UI 可视化面板。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from pipeline.schema import CandidateFactor

logger = logging.getLogger("factor_gpt.screener")

DATE = "date"
SYMBOL = "symbol"

try:
    from sklearn.linear_model import LassoCV
    _HAS_SKLEARN = True
except Exception:  # noqa: BLE001
    _HAS_SKLEARN = False


@dataclass
class ScreenerConfig:
    use_lasso: bool = True
    lasso_alpha_ratio: float = 0.7        # LassoCV cv 折数之外的松弛（保留系数阈值）
    use_human_collab: bool = True
    topk_ratio: float = 0.1              # TOP 10% 截断
    min_keep: int = 3


class Screener:
    """三级浮选筛选器。"""

    def __init__(self, config: Optional[ScreenerConfig] = None):
        self.config = config or ScreenerConfig()
        # 三级筛选审计留痕：记录每一级的进出数量与人工评审明细，供方法学报告与交付物追溯
        self.audit: Dict[str, object] = {}

    # -- 第一级：LASSO ---------------------------------------------------- #
    @staticmethod
    def _dedup_series(s: pd.Series) -> pd.Series:
        """去除 (date, symbol) 多索引中的重复项（真实行情下偶发重复，保留最后一条）。"""
        if s.index.duplicated().any():
            return s[~s.index.duplicated(keep="last")]
        return s

    def lasso_filter(self, candidates: List[CandidateFactor], kline: pd.DataFrame) -> List[CandidateFactor]:
        if not self.config.use_lasso or len(candidates) < 3:
            return candidates
        names = [c.name for c in candidates]
        series = [self._dedup_series(c.series) for c in candidates]
        common = series[0].index
        for s in series[1:]:
            common = common.intersection(s.index)
        common = common[~common.duplicated()]
        X = np.column_stack([s.reindex(common).to_numpy(dtype=float) for s in series])
        X = np.nan_to_num(X)
        kl = kline.set_index([DATE, SYMBOL])
        if kl.index.duplicated().any():
            kl = kl[~kl.index.duplicated(keep="last")]
        y = kl.groupby(level=SYMBOL)["close"].pct_change(1).reindex(common).to_numpy(dtype=float)
        mask = ~np.isnan(y)
        X, y = X[mask], y[mask]
        if X.shape[0] < 50 or _HAS_SKLEARN is False:
            # 无 sklearn 或样本不足：改用相关性冗余去重（保留与收益相关性最高者）
            return self._corr_redundancy(candidates, y, common)
        try:
            model = LassoCV(cv=5, random_state=0, max_iter=5000).fit(X, y)
            coef = model.coef_
            keep = [candidates[i] for i in range(len(candidates)) if abs(coef[i]) > 1e-6]
            logger.info("LASSO 筛选：%d → %d（保留显著因子）", len(candidates), len(keep))
            return keep if keep else candidates
        except Exception as e:  # noqa: BLE001
            logger.warning("LASSO 失败，降级相关性去重: %s", e)
            return self._corr_redundancy(candidates, y, common)

    @staticmethod
    def _corr_redundancy(candidates, y, common) -> List[CandidateFactor]:
        scores = []
        for c in candidates:
            s = c.series
            if s.index.duplicated().any():
                s = s[~s.index.duplicated(keep="last")]
            f = s.reindex(common).to_numpy(dtype=float)
            f = np.nan_to_num(f)
            if np.std(f) == 0:
                scores.append(0.0)
            else:
                scores.append(abs(np.corrcoef(f, y)[0, 1]))
        order = np.argsort(scores)[::-1]
        keep_n = max(3, len(candidates) // 2)
        return [candidates[i] for i in order[:keep_n]]

    # -- 第二级：人机协同 ------------------------------------------------ #
    def human_collab_filter(
        self,
        candidates: List[CandidateFactor],
        review_callback: Optional[Callable[[List[CandidateFactor]], List[str]]] = None,
    ) -> List[CandidateFactor]:
        if not self.config.use_human_collab:
            self.audit["human_collab"] = {"mode": "disabled", "in": len(candidates), "out": len(candidates)}
            return candidates
        # 生成可视化统计（供 Partice 协同平台展示）
        stats = self._viz_stats(candidates)
        logger.info("人机协同筛选候选：\n%s", stats)
        if review_callback is None:
            self.audit["human_collab"] = {
                "mode": "auto",  # 无人值守：本级透传，实际由 LASSO + TOP-K 决定
                "in": len(candidates), "out": len(candidates), "rejected": [],
            }
            return candidates  # 默认全保留（无人介入时）
        keep_names = {str(n) for n in (review_callback(candidates) or [])}
        kept = [c for c in candidates if c.name in keep_names]
        if not kept:
            # 人工全部剔除会使后续合成无米下炊：留痕告警并回退全保留，保证流水线不中断
            logger.warning("人机协同评审剔除了全部候选，已回退为全保留以维持流水线可用性")
            self.audit["human_collab"] = {
                "mode": "human", "in": len(candidates), "out": len(candidates),
                "rejected": [], "warning": "评审结果为空，已回退全保留",
            }
            return candidates
        rejected = [c.name for c in candidates if c.name not in keep_names]
        logger.info("人机协同筛选：%d → %d（人工剔除 %d 个）", len(candidates), len(kept), len(rejected))
        self.audit["human_collab"] = {
            "mode": "human", "in": len(candidates), "out": len(kept),
            "kept": [c.name for c in kept], "rejected": rejected,
        }
        return kept

    @staticmethod
    def _viz_stats(candidates) -> str:
        lines = ["因子\tICIR\t稳定性\t换手"]
        for c in candidates:
            m = c.metrics
            lines.append(f"{c.name}\t{m.get('icir', 0):.3f}\t"
                         f"{m.get('stability_score', 0):.3f}\t{m.get('turnover', 0):.3f}")
        return "\n".join(lines)

    # -- 第三级：TOP 10% 截断 -------------------------------------------- #
    def topk_truncation(self, candidates: List[CandidateFactor]) -> List[CandidateFactor]:
        if len(candidates) <= self.config.min_keep:
            return candidates
        ranked = sorted(candidates, key=lambda c: float(c.metrics.get("icir", 0) or 0), reverse=True)
        k = max(self.config.min_keep, int(np.ceil(len(ranked) * self.config.topk_ratio)))
        keep = ranked[:k]
        logger.info("TOP %d%% 截断：%d → %d", int(self.config.topk_ratio * 100), len(candidates), len(keep))
        return keep

    # -- 流水线 ---------------------------------------------------------- #
    def screen(
        self,
        candidates: List[CandidateFactor],
        kline: pd.DataFrame,
        review_callback: Optional[Callable[[List[CandidateFactor]], List[str]]] = None,
    ) -> List[CandidateFactor]:
        self.audit = {"input": len(candidates)}
        out = self.lasso_filter(candidates, kline)
        self.audit["lasso"] = {"in": len(candidates), "out": len(out),
                               "mode": "lasso" if (self.config.use_lasso and _HAS_SKLEARN) else "corr_fallback"}
        out = self.human_collab_filter(out, review_callback)
        n_before_topk = len(out)
        out = self.topk_truncation(out)
        self.audit["topk"] = {"in": n_before_topk, "out": len(out),
                              "ratio": self.config.topk_ratio, "min_keep": self.config.min_keep}
        self.audit["output"] = len(out)
        return out
