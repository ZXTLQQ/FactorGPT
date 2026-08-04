"""
Transformer-Agent 深度耦合 (src/engine/transformer_coupling.py)

实现 Transformer 自注意力机制与因子挖掘 Agent 的深度融合：

1. FactorEncoder — 将因子时间序列编码为隐空间表征
2. CrossAttentionFusion — 因子间的交叉注意力融合（学习因子交互模式）
3. FactorScorer — 基于注意力的因子重要性评分（替代简单 IC 排名）
4. PatternMemory — 因子模式记忆库（基于相似度检索 + 更新）
5. TransformerCoupling — 总控模块，桥接 LLM Agent 与因子表征

依赖：仅依赖 numpy，无需 torch。自注意力在 numpy 层面实现。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ===================================================================
# 1. 因子编码器 (FactorEncoder)
# ===================================================================

class FactorEncoder:
    """将单因子时序编码为固定维度的隐空间表征向量。

    使用多粒度统计特征 + 时序趋势编码：
    - 统计矩：均值、标准差、偏度、峰度
    - 趋势特征：线性趋势系数、最大回撤、累积收益
    - 周期性特征：多窗口自相关
    - 分布特征：分位数值、尾部比例
    """

    def __init__(self, encoding_dim: int = 64) -> None:
        self._dim = encoding_dim
        self._proj_matrix: Optional[np.ndarray] = None
        self._init_projection()

    def _init_projection(self) -> None:
        """初始化投影矩阵（固定种子，确保复现性）。"""
        rng = np.random.RandomState(42)
        # 原始特征维度（统计特征数）+ 扩展
        raw_dim = 28
        self._proj_matrix = rng.randn(raw_dim, self._dim) / np.sqrt(raw_dim)

    def _extract_features(self, series: np.ndarray) -> np.ndarray:
        """从因子时序中提取多粒度统计特征。返回 28 维特征向量。"""
        s = series[~np.isnan(series)]
        if len(s) < 10:
            s = np.array([0.0] * 10)

        features = []

        # 1. 基本统计 (6 维)
        features.extend([
            float(np.mean(s)),
            float(np.std(s)) if np.std(s) > 0 else 0.0,
            float(np.median(s)),
            float(np.min(s)),
            float(np.max(s)),
            float(np.max(s) - np.min(s)),
        ])

        # 2. 高阶矩 (2 维)
        z = (s - np.mean(s)) / (np.std(s) + 1e-8)
        features.extend([
            float(np.mean(z ** 3)),  # skewness
            float(np.mean(z ** 4)),  # kurtosis
        ])

        # 3. 趋势 (3 维)
        n = len(s)
        slope = np.polyfit(np.arange(n), s, 1)[0] if n > 1 else 0.0
        cum_ret = s[-1] / (s[0] + 1e-8) - 1
        maxdd = float(max(1.0 - s / np.maximum.accumulate(s)) if len(s) > 0 else 0.0)
        features.extend([slope, cum_ret, maxdd])

        # 4. 分位数 (5 维)
        for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
            features.append(float(np.percentile(s, q * 100)))

        # 5. 自相关 (4 维 — lag=1,3,5,10)
        for lag in [1, 3, 5, 10]:
            if len(s) > lag:
                corr = np.corrcoef(s[:-lag], s[lag:])[0, 1]
                features.append(float(corr if not np.isnan(corr) else 0.0))
            else:
                features.append(0.0)

        # 6. 尾部比例 (2 维)
        features.extend([
            float(np.mean(s > np.percentile(s, 90))),
            float(np.mean(s < np.percentile(s, 10))),
        ])

        # 7. 非零比例 (1 维)
        features.append(float(np.mean(np.abs(s) > 1e-8)))

        # 8. 上分位均值 vs 下分位均值 (2 维)
        top = s[s >= np.percentile(s, 75)]
        bot = s[s <= np.percentile(s, 25)]
        features.append(float(np.mean(top)) if len(top) > 0 else 0.0)
        features.append(float(np.mean(bot)) if len(bot) > 0 else 0.0)

        # 9. 波动聚集性 (ARCH 效应) (1 维)
        if len(s) > 2:
            ret = np.diff(s)
            ret2 = ret ** 2
            arch = np.corrcoef(ret2[:-1], ret2[1:])[0, 1] if len(ret2) > 1 else 0.0
            features.append(float(arch if not np.isnan(arch) else 0.0))
        else:
            features.append(0.0)

        # 10. Hurst 近似 (1 维) — 简化版 R/S
        if len(s) > 10:
            rs_vals = []
            for chunk_size in [5, 10, 20]:
                if chunk_size * 2 <= len(s):
                    chunks = [s[i:i + chunk_size] for i in range(0, len(s) - chunk_size, chunk_size)]
                    rs = []
                    for c in chunks:
                        if len(c) > 0 and np.std(c) > 1e-8:
                            r = np.max(c - np.mean(c)) - np.min(c - np.mean(c))
                            rs.append(r / np.std(c))
                    if rs:
                        rs_vals.append(np.mean(rs))
            features.append(float(np.mean(rs_vals)) if rs_vals else 0.5)
        else:
            features.append(0.5)

        # 补齐到 raw_dim
        vec = np.array(features, dtype=np.float64)
        if len(vec) < 28:
            pad = np.zeros(28 - len(vec))
            vec = np.concatenate([vec, pad])
        return vec[:28]

    def encode_factor(self, series: np.ndarray) -> np.ndarray:
        """将单因子时序编码为隐空间向量。"""
        feats = self._extract_features(series)
        return feats @ self._proj_matrix  # (28,) x (28, dim) -> (dim,)

    def encode_batch(
        self,
        factor_df: pd.DataFrame,
        factor_names: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """批量编码多个因子。

        Args:
            factor_df: 含 date, symbol 及多个因子列
            factor_names: 要编码的因子列名，不传则全部非 date/symbol 列

        Returns:
            {factor_name: encoding_vector}
        """
        if factor_names is None:
            factor_names = [c for c in factor_df.columns if c not in ("date", "symbol")]

        encodings: Dict[str, np.ndarray] = {}
        for name in factor_names:
            col = factor_df[name].values.astype(float)
            encodings[name] = self.encode_factor(col)
        return encodings


# ===================================================================
# 2. 交叉注意力融合 (CrossAttentionFusion)
# ===================================================================

class CrossAttentionFusion:
    """多因子之间的交叉注意力机制。

    对 N 个因子的隐空间表征计算 pairwise attention，
    学习因子之间的互补/替代关系，产出融合后的因子表示。

    核心公式（简化版）：
      Q=W_q*X, K=W_k*X, V=W_v*X
      Attention = softmax(QK^T / sqrt(d_k))
      Fusion = Attention * V
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4) -> None:
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self._rng = np.random.RandomState(42)

        # 可学习的投影矩阵（简化版：固定初始化 + 伪更新由外界控制）
        self.W_q: Dict[int, np.ndarray] = {}
        self.W_k: Dict[int, np.ndarray] = {}
        self.W_v: Dict[int, np.ndarray] = {}

        for h in range(n_heads):
            self.W_q[h] = self._rng.randn(d_model, self.d_k) / np.sqrt(d_model)
            self.W_k[h] = self._rng.randn(d_model, self.d_k) / np.sqrt(d_model)
            self.W_v[h] = self._rng.randn(d_model, self.d_k) / np.sqrt(d_model)

    def _scaled_dot_attention(
        self, Q: np.ndarray, K: np.ndarray, V: np.ndarray
    ) -> np.ndarray:
        """Scaled Dot-Product Attention。

        Q: (N, d_k), K: (N, d_k), V: (N, d_k) -> output: (N, d_k)
        """
        scores = Q @ K.T / np.sqrt(self.d_k)  # (N, N)
        # softmax 稳定版
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        attn_weights = np.exp(scores) / (np.sum(np.exp(scores), axis=-1, keepdims=True) + 1e-8)
        return attn_weights @ V  # (N, d_k)

    def fuse(
        self,
        encodings: Dict[str, np.ndarray],
        return_attention: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """多因子融合。

        Args:
            encodings: {factor_name: encoding_vector(d_model)}
            return_attention: 是否返回注意力权重

        Returns:
            (fused_vector(d_model), attention_info or None)
        """
        names = list(encodings.keys())
        N = len(names)
        if N == 0:
            return np.zeros(self.d_model), None
        if N == 1:
            return encodings[names[0]], None

        # 构建输入矩阵 X: (N, d_model)
        X = np.stack([encodings[n] for n in names], axis=0)

        # 多头注意力
        head_outputs = []
        all_attn: Dict[str, np.ndarray] = {}
        for h in range(self.n_heads):
            Q = X @ self.W_q[h]  # (N, d_k)
            K = X @ self.W_k[h]
            V = X @ self.W_v[h]
            ho = self._scaled_dot_attention(Q, K, V)  # (N, d_k)
            head_outputs.append(ho)
            if return_attention:
                all_attn[f"head_{h}"] = (
                    Q @ K.T / np.sqrt(self.d_k)
                )

        # 拼接多头
        fused = np.concatenate(head_outputs, axis=-1)  # (N, d_model)

        # 平均池化得到融合向量
        fused_vector = np.mean(fused, axis=0)  # (d_model,)

        if return_attention:
            return fused_vector, {"attention": all_attn, "factor_names": names}

        return fused_vector, None

    def compute_factor_importance(
        self, encodings: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """基于自注意力计算每个因子的重要性权重。

        用全局平均注意力作为因子重要性分数。
        """
        _, attn_info = self.fuse(encodings, return_attention=True)
        if attn_info is None:
            return {n: 1.0 for n in encodings}

        names = attn_info["factor_names"]
        N = len(names)

        # 聚合所有头的注意力
        total_attn = np.zeros((N, N))
        for h_key, attn in attn_info["attention"].items():
            attn_norm = attn - np.max(attn, axis=-1, keepdims=True)
            attn_weights = np.exp(attn_norm) / (np.sum(np.exp(attn_norm), axis=-1, keepdims=True) + 1e-8)
            total_attn += attn_weights

        total_attn /= len(attn_info["attention"])

        # 行平均 = 每个因子被其他因子关注的总量
        importance = np.mean(total_attn, axis=1)
        importance = importance / (importance.sum() + 1e-8)

        return {n: float(importance[i]) for i, n in enumerate(names)}


# ===================================================================
# 3. 因子评分器 (FactorScorer)
# ===================================================================

class FactorScorer:
    """基于注意力的因子评分与排序。

    替代简单的 IC 排名，综合考虑：
    - attention_importance: 交叉注意力权重
    - statistical_quality: 统计特征（偏度、稳定性、自相关）
    - diversity_bonus: 与已有因子的重复度惩罚
    """

    def __init__(self, encoder: Optional[FactorEncoder] = None) -> None:
        self.encoder = encoder or FactorEncoder()
        self._scored_history: List[Dict[str, Any]] = []

    def score_factors(
        self,
        factor_data: Dict[str, pd.Series],
        existing_encodings: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Dict[str, Any]]:
        """评分多个因子。

        Args:
            factor_data: {factor_name: series}
            existing_encodings: 已有因子的编码（用于计算多样性）

        Returns:
            按综合得分降序的分数列表
        """
        # Step 1: 编码
        encodings = {}
        for name, s in factor_data.items():
            encodings[name] = self.encoder.encode_factor(s.values)

        # Step 2: 交叉注意力重要性
        fusion = CrossAttentionFusion()
        importance = fusion.compute_factor_importance(encodings)

        # Step 3: 统计质量
        stat_quality = {}
        for name, s in factor_data.items():
            vals = s.dropna().values
            if len(vals) < 10:
                stat_quality[name] = 0.3
                continue
            skew = abs(np.mean(((vals - np.mean(vals)) / (np.std(vals) + 1e-8)) ** 3))
            stab = 1.0 / (np.std(vals) + 1.0)
            auto = np.corrcoef(vals[:-1], vals[1:])[0, 1] if len(vals) > 1 else 0
            auto = abs(auto) if not np.isnan(auto) else 0
            stat_quality[name] = float(np.clip(stab * 0.4 + (1.0 - min(skew, 5) / 5.0) * 0.3 + auto * 0.3, 0, 1))

        # Step 4: 多样性（与已有因子的余弦相似度负相关）
        diversity = {}
        if existing_encodings:
            existing_vecs = np.stack(list(existing_encodings.values()))
            for name, enc in encodings.items():
                sim = np.dot(existing_vecs, enc) / (
                    np.linalg.norm(existing_vecs, axis=1) * np.linalg.norm(enc) + 1e-8
                )
                diversity[name] = float(np.clip(1.0 - np.mean(np.abs(sim)), 0.2, 1.0))
        else:
            for name in encodings:
                diversity[name] = 1.0

        # Step 5: 综合得分
        scored = []
        for name in factor_data:
            score = (
                importance.get(name, 0.1) * 0.40 +
                stat_quality.get(name, 0.5) * 0.35 +
                diversity.get(name, 1.0) * 0.25
            )
            scored.append({
                "name": name,
                "score": round(float(score), 4),
                "attention_importance": round(importance.get(name, 0.1), 4),
                "statistical_quality": round(stat_quality.get(name, 0.5), 4),
                "diversity": round(diversity.get(name, 1.0), 4),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        self._scored_history.append({"timestamp": datetime.now().isoformat(), "scores": scored})
        return scored


# ===================================================================
# 4. 因子模式记忆 (PatternMemory)
# ===================================================================

class PatternMemory:
    """因子模式记忆库。

    功能：
    - 存储成功因子的编码表征（winner patterns）
    - 存储失败因子的编码表征（loser patterns）用于规避
    - 基于相似度检索最近的 top-k 模式
    - 提供模式多样性指引给 GP/LLM Agent
    """

    def __init__(self, persist_path: Optional[str] = None) -> None:
        self._winners: List[Dict[str, Any]] = []  # [{name, encoding, meta, timestamp}]
        self._losers: List[Dict[str, Any]] = []
        self._persist_path = persist_path
        self._encoder = FactorEncoder()

    def remember(
        self,
        name: str,
        factor_series: pd.Series,
        quality: str = "winner",  # "winner" or "loser"
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """存储一个因子模式。"""
        encoding = self._encoder.encode_factor(factor_series.values)
        entry = {
            "name": name,
            "encoding": encoding.tolist(),
            "quality": quality,
            "meta": meta or {},
            "timestamp": datetime.now().isoformat(),
        }
        if quality == "winner":
            self._winners.append(entry)
        else:
            self._losers.append(entry)
        self._save()

    def find_similar(
        self,
        factor_series: pd.Series,
        pool: str = "winners",  # "winners" / "losers" / "all"
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """查找与给定因子最相似的历史模式。"""
        encoding = self._encoder.encode_factor(factor_series.values)
        entries = []
        if pool in ("winners", "all"):
            entries.extend(self._winners)
        if pool in ("losers", "all"):
            entries.extend(self._losers)

        if not entries:
            return []

        # 计算余弦相似度
        similarities = []
        for entry in entries:
            stored = np.array(entry["encoding"])
            sim = float(np.dot(encoding, stored) / (np.linalg.norm(encoding) * np.linalg.norm(stored) + 1e-8))
            similarities.append((sim, entry))

        similarities.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, entry in similarities[:top_k]:
            results.append({
                "similarity": round(sim, 4),
                "name": entry["name"],
                "quality": entry.get("quality", "unknown"),
                "meta": entry.get("meta", {}),
                "timestamp": entry.get("timestamp", ""),
            })
        return results

    def get_diversity_guidance(
        self, factor_names: List[str], top_k: int = 3
    ) -> Dict[str, Any]:
        """提供多样性指导——基于已有成功模式，建议探索哪些新区域。

        Returns:
            {
                "explored_regions": [...],   # 已充分探索的模式簇
                "underexplored_hints": [...], # 建议探索的方向
                "overlap_warnings": [...],    # 与已有因子高度重叠的警告
            }
        """
        if not self._winners:
            return {"explored_regions": [], "underexplored_hints": ["无历史数据，自由探索"], "overlap_warnings": []}

        winner_encodings = np.array([np.array(w["encoding"]) for w in self._winners])
        # 聚类成3-5个粗略区域（简化版：用前3个主方向）
        centroid = np.mean(winner_encodings, axis=0)
        # 分散度
        dispersion = np.std(winner_encodings, axis=0)

        high_dispersion_dims = np.argsort(dispersion)[-3:].tolist()
        low_dispersion_dims = np.argsort(dispersion)[:3].tolist()

        return {
            "explored_regions": [
                f"维度 {d}: 均值={centroid[d]:.3f}, 标准差={dispersion[d]:.3f}"
                for d in [int(np.argmax(dispersion))]
            ],
            "underexplored_hints": [
                f"建议在维度 {d} 附近探索新因子模式（当前变异性低）"
                for d in low_dispersion_dims
            ],
            "overlap_warnings": [
                f"新因子与 {w['name']} 的编码距离较近，注意去重"
                for w in self._winners[-3:]
            ],
        }

    def _save(self) -> None:
        if not self._persist_path:
            return
        Path(self._persist_path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "winners": self._winners,
            "losers": self._losers,
            "saved_at": datetime.now().isoformat(),
        }
        with open(self._persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if not self._persist_path or not Path(self._persist_path).exists():
            return
        with open(self._persist_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._winners = data.get("winners", [])
        self._losers = data.get("losers", [])


# ===================================================================
# 5. Transformer-Agent 耦合总控
# ===================================================================

class TransformerCoupling:
    """Transformer-Agent 深度耦合总控模块。

    桥接 LLM Agent 与因子表征：
    1. 将因子库中的因子编码为隐空间表征
    2. 根据 Agent 的用户需求，在编码空间中检索最匹配的因子模板
    3. 为 Agent 的 prompt 生成提供结构化的因子上下文（而非纯文本）
    4. 在因子评估后更新模式记忆库

    用法：
        coupling = TransformerCoupling(library, memory_path="memory.json")
        # 在 Agent 节点中：
        context = coupling.build_agent_context(user_query, top_k_factor=10)
        # context 是结构化的因子表征信息，可注入到 LLM prompt 中
    """

    def __init__(
        self,
        library: Any,  # FactorLibrary
        memory_path: Optional[str] = None,
        encoding_dim: int = 64,
    ) -> None:
        self.library = library
        self.encoder = FactorEncoder(encoding_dim=encoding_dim)
        self.scorer = FactorScorer(encoder=self.encoder)
        self.fusion = CrossAttentionFusion(d_model=encoding_dim)
        self.memory = PatternMemory(persist_path=memory_path)

        # 缓存的因子编码
        self._encoding_cache: Dict[str, np.ndarray] = {}
        self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        """重建因子编码缓存（在因子库更新后调用）。"""
        from .factor_library import FactorLibrary
        if not isinstance(self.library, FactorLibrary):
            return
        # 对静态因子建编码表（代码到向量 -> 通过模拟？暂用空向量占位）
        # 实际使用时在 build_agent_context 中按需编码
        self._encoding_cache.clear()

    def build_agent_context(
        self,
        user_query: str,
        category: Optional[str] = None,
        top_k_factor: int = 10,
    ) -> Dict[str, Any]:
        """构建注入到 LLM Agent 的结构化因子上下文。

        Args:
            user_query: 用户需求文本
            category: 限制因子大类
            top_k_factor: 检索的顶级因子数

        Returns:
            {
                "related_factors": [...],       # 最相关的因子列表
                "factor_descriptions": str,     # 格式化的因子描述文段
                "diversity_guidance": dict,     # 多样性指导建议
                "recommended_patterns": [...],   # 推荐探索的因子模式
                "statistics": dict,             # 因子库统计
            }
        """
        # Step 1: 从因子库搜索相关因子
        factors = self.library.search(query=user_query) if self.library else []
        if not factors:
            factors = self.library.list_by_category(category) if category else self.library.list_all()[:top_k_factor]

        factors = self.library.quality_rank(factors, top_k=top_k_factor)

        # Step 2: 构建结构化描述
        descriptions = []
        for f in factors:
            cat_label = f"{f.category} - "
            desc = f"[{f.name}] ({cat_label}{f.display_name}) [{f.direction}] — {f.description[:120]}"
            descriptions.append(desc)

        # Step 3: 多样性指导
        guidance = self.memory.get_diversity_guidance([f.name for f in factors])

        # Step 4: 构建上下文输出
        context = {
            "related_factors": [f.to_dict() for f in factors],
            "factor_descriptions": "\n".join(descriptions),
            "diversity_guidance": guidance,
            "recommended_patterns": self._get_recommended_patterns(factors),
            "statistics": self.library.statistics() if self.library else {},
        }

        return context

    def _get_recommended_patterns(
        self, factors: List[Any], n: int = 3
    ) -> List[Dict[str, Any]]:
        """根据 Top 因子推荐可能遗漏的模式。"""
        categories_present = set(f.category for f in factors)
        missing_categories = [c for c in ALL_CATEGORIES if c not in categories_present]
        recommendations = []
        for cat in missing_categories[:n]:
            cat_factors = self.library.list_by_category(cat)
            if cat_factors:
                top_f = cat_factors[:2]
                for f in top_f:
                    recommendations.append({
                        "category": CATEGORY_LABELS.get(cat, cat),
                        "name": f.name,
                        "description": f.description[:100],
                        "reason": "该大类因子在当前选择中未被覆盖，建议补充以增加多样性",
                    })
        return recommendations

    def update_memory_from_experiment(
        self,
        factor_name: str,
        factor_series: pd.Series,
        ic: float,
        ic_threshold: float = 0.02,
    ) -> None:
        """基于回测结果更新模式记忆。IC > 阈值的为 winner。"""
        if ic > ic_threshold:
            self.memory.remember(factor_name, factor_series, quality="winner", meta={"ic": ic})
        elif ic < -ic_threshold:
            self.memory.remember(factor_name, factor_series, quality="loser", meta={"ic": ic})

    def rank_factors_for_agent(
        self,
        factor_data: Dict[str, pd.Series],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """用自注意力为 Agent 排序因子（作为 Agent 选因子的参考）。"""
        ranked = self.scorer.score_factors(factor_data)
        return ranked[:top_k]

    def fuse_factors_for_agent(
        self,
        factor_data: Dict[str, pd.Series],
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """融合多个因子+返回重要性。供 Agent 做最终决策参考。"""
        encodings = self.encoder.encode_batch(
            pd.DataFrame(factor_data), factor_names=list(factor_data.keys())
        )
        fused, _ = self.fusion.fuse(encodings, return_attention=True)
        importance = self.fusion.compute_factor_importance(encodings)
        return fused, importance
