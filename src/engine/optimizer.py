"""
因子组合优化引擎（src/engine/optimizer.py）

提供多因子合成与筛选能力：
- ic_weighted_combine: 依据各因子历史 RankIC 的绝对值加权合成综合因子；
- orthogonalize: 用逐步回归（Gram-Schmidt 思路）对因子做正交化，去除冗余信息；
- select_top: 按评价指标（IC / ICIR）对候选因子排序筛选。

所有函数均接收「索引为 (date, symbol) 的因子 Series」或由其组成的 DataFrame。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FactorOptimizer:
    """多因子组合优化器。"""

    @staticmethod
    def ic_weighted_combine(
        factors: Dict[str, pd.Series],
        ic_dict: Dict[str, float],
        min_abs_ic: float = 0.005,
    ) -> pd.Series:
        """按 |IC| 加权合成综合因子。

        Args:
            factors: 名称 -> 因子 Series 的字典。
            ic_dict: 名称 -> 对应 RankIC（或 IC）的字典。
            min_abs_ic: 权重阈值，低于该值的因子权重置 0。

        Returns:
            合成后的综合因子 Series（截面已标准化）。
        """
        if not factors:
            raise ValueError("factors 为空")

        aligned = pd.DataFrame(factors)
        weights = pd.Series(
            {k: max(0.0, abs(ic_dict.get(k, 0.0))) for k in factors}
        )
        weights = weights.where(weights >= min_abs_ic, 0.0)
        if weights.sum() == 0:
            weights = pd.Series(1.0 / len(factors), index=factors.keys())
        weights = weights / weights.sum()

        # 截面标准化后加权
        normed = aligned.groupby(level="date").transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-12)
        )
        combined = (normed * weights).sum(axis=1)
        return combined

    @staticmethod
    def orthogonalize(factors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """对因子做逐步正交化，输出互不相关的增量因子。

        采用序列回归法：第 i 个正交因子 = 原因子对其之前所有因子的回归残差。

        Returns:
            名称 -> 正交化后因子 Series 的字典（含 _orth 后缀）。
        """
        if len(factors) < 2:
            return {f"{k}_orth": v for k, v in factors.items()}

        aligned = pd.DataFrame(factors)
        names = list(factors.keys())
        result: Dict[str, pd.Series] = {}

        for i, name in enumerate(names):
            resid = aligned[name].copy()
            if i > 0:
                prev = aligned[names[:i]]
                # 逐日截面回归取残差
                orth_parts = []
                for date, grp in aligned.groupby(level="date"):
                    y = grp[name]
                    X = grp[names[:i]]
                    mask = y.notna() & X.notna().all(axis=1)
                    if mask.sum() < 5:
                        orth_parts.append(pd.Series(np.nan, index=grp.index))
                        continue
                    Xc = X[mask].values
                    yc = y[mask].values
                    Xc = np.column_stack([np.ones(len(Xc)), Xc])
                    beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
                    pred = Xc @ beta
                    r = y[mask].copy()
                    r.iloc[:] = yc - pred
                    full = pd.Series(np.nan, index=grp.index)
                    full[mask] = r.values
                    orth_parts.append(full)
                resid = pd.concat(orth_parts)
            result[f"{name}_orth"] = resid
        return result

    @staticmethod
    def select_top(
        metrics_list: list,
        top_k: int = 3,
        key: str = "ic",
    ) -> list:
        """按指标对候选因子排序，返回前 top_k 个因子名。

        Args:
            metrics_list: 元素为 dict，需包含 'name' 与 key 指定的指标。
            top_k: 保留数量。
            key: 排序依据的指标键（如 'ic' / 'icir'）。

        Returns:
            排序后的因子名列表。
        """
        valid = [m for m in metrics_list if key in m and pd.notna(m[key])]
        valid.sort(key=lambda m: abs(m[key]), reverse=True)
        return [m.get("name") for m in valid[:top_k]]
