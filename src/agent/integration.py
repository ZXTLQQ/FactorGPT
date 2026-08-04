"""
FactorGPT 增强集成 (src/agent/integration.py)

将新增模块（传统因子库、遗传增强、Transformer耦合、非结构化数据）深度集成到 Agent 流程中。

在 nodes.py 的 FactorAgent 声明后调用：
    from .integration import apply_enhancements
    apply_enhancements(agent)

或在 Graph 构建时：
    from .integration import build_enhanced_context
    context = build_enhanced_context(state, library, coupling)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..engine.factor_library import FactorLibrary, create_default_library, mass_produce_factors
from ..engine.traditional_factors import (
    FactorDef,
    ALL_CATEGORIES,
    CATEGORY_LABELS,
    get_all_factors,
    get_factors_by_category,
    search_factors,
)
from ..engine.transformer_coupling import TransformerCoupling
from ..engine.genetic_enhanced import EnhancedFactorEvolver, FactorCluster, EventWindow
from ..engine.unstructured_miner import (
    TextAnalyzer,
    DataUploadParser,
    AlternativeDataManager,
    UnstructuredFactorIntegrator,
)

logger = logging.getLogger(__name__)

# 全局单例
_library: Optional[FactorLibrary] = None
_coupling: Optional[TransformerCoupling] = None
_unstructured_mgr: Optional[AlternativeDataManager] = None
_text_analyzer: Optional[TextAnalyzer] = None


def get_library(persist_dir: Optional[str] = None) -> FactorLibrary:
    global _library
    if _library is None:
        _library = create_default_library(persist_dir=persist_dir)
        logger.info(f"因子库已初始化: {_library.statistics()['total']} 个因子")
    return _library


def get_coupling(library: Optional[FactorLibrary] = None) -> TransformerCoupling:
    global _coupling
    if _coupling is None:
        lib = library or get_library()
        _coupling = TransformerCoupling(lib, memory_path="data/cache/factor_memory.json")
        logger.info("Transformer-Agent耦合已初始化")
    return _coupling


def get_unstructured_manager(data_dir: Optional[str] = None) -> AlternativeDataManager:
    global _unstructured_mgr
    if _unstructured_mgr is None:
        _unstructured_mgr = AlternativeDataManager(data_dir=data_dir or "data/alternative")
    return _unstructured_mgr


def get_text_analyzer() -> TextAnalyzer:
    global _text_analyzer
    if _text_analyzer is None:
        _text_analyzer = TextAnalyzer()
    return _text_analyzer


# ---------------------------------------------------------------------------
# 知识丰富的关键词到因子映射
# ---------------------------------------------------------------------------

_KEYWORD_FACTOR_MAP: Dict[str, List[str]] = {
    "动量": ["momentum_20d", "momentum_10d", "momentum_5d", "momentum_residual_20d", "momentum_60d"],
    "反转": ["reversal_1d", "reversal_5d", "reversal_10d", "path_dependency_maxdd_20d"],
    "趋势": ["ma_cross_5_20", "ma_cross_10_60", "macd_signal", "rsi_14d", "trend_strength_20d"],
    "波动率": ["realized_vol_20d", "parkinson_vol_20d", "gk_vol_20d", "downside_vol_20d", "skewness_20d"],
    "低波": ["realized_vol_20d", "realized_vol_60d", "parkinson_vol_20d", "gk_vol_20d"],
    "换手": ["turnover_20d", "turnover_5d", "turnover_change_5d"],
    "流动性": ["turnover_20d", "amihud_illiquidity_20d", "dollar_volume_20d", "high_low_spread_20d"],
    "量价": ["price_volume_corr_20d", "vwap_deviation_5d", "up_volume_ratio_20d", "net_flow_pressure_5d"],
    "背离": ["volume_delta_price_20d", "price_volume_corr_20d"],
    "放量": ["volume_breakout_5d", "volume_price_ratio_20d", "volume_concentration_20d"],
    "成交量": ["obv_momentum_20d", "vcmf_20d", "pvo_12_26", "force_index_13d"],
    "资金流": ["obv_momentum_20d", "cmf_20d", "net_flow_pressure_5d", "up_volume_ratio_20d", "mfi_14d"],
    "突破": ["breakout_20d_high", "close_to_high_20d", "keltner_pct_20d"],
    "质量": ["trend_strength_20d", "path_dependency_maxdd_20d"],
    "风险": ["realized_vol_20d", "downside_vol_20d", "max_up_consecutive", "jump_risk_20d"],
    "技术指标": ["rsi_14d", "macd_signal", "atr_14d", "bollinger_pct_20d", "mfi_14d"],
    "涨停": ["limit_up_touch_20d", "limit_down_touch_20d"],
}


def query_to_factor_suggestions(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """根据用户需求推荐相关传统因子。

    用于注入到 Agent 的 prompt 中作为先验知识。
    """
    library = get_library()
    # 关键词匹配
    matched_names: set = set()
    for kw, names in _KEYWORD_FACTOR_MAP.items():
        if kw in query:
            matched_names.update(names)

    # 模糊搜索
    search_results = library.search(query=query)[:top_k * 2]
    matched_names.update(f.name for f in search_results)

    # 取 top_k
    factors = [library.get_factor(n) for n in matched_names if library.get_factor(n)]
    factors = [f for f in factors if f is not None]
    if len(factors) > top_k:
        factors = library.quality_rank(factors, top_k=top_k)

    return [f.to_dict() for f in factors]


def build_enriched_knowledge(query: str) -> str:
    """基于传统因子库构建先验知识文本，注入到 LLM 的 comprehension prompt 中。"""
    suggestions = query_to_factor_suggestions(query)
    if not suggestions:
        return ""

    lines = [
        "## 传统因子库先验知识（供参考模式，请据此创新）\n",
        "以下是从传统量化因子库中检索到的相关因子模式：",
    ]

    by_cat: Dict[str, List[Dict]] = {}
    for s in suggestions:
        by_cat.setdefault(s["category_label"], []).append(s)

    for cat_label, factors in by_cat.items():
        lines.append(f"\n### {cat_label}")
        for f in factors[:3]:
            lines.append(f"- **{f['display_name']}** ({f['name']}): {f['description'][:100]}")

    lines.append(f"\n以上共 {len(suggestions)} 个因子可作为模式参考。")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 增强的 Agent 状态管理
# ---------------------------------------------------------------------------

def build_enhanced_context(
    state: Dict[str, Any],
    library: Optional[FactorLibrary] = None,
    coupling: Optional[TransformerCoupling] = None,
) -> Dict[str, Any]:
    """构建增强的 Agent 上下文（调用在 Graph 或 nodes 的状态中）。

    返回一个字典供 Agent 节点注入额外信息。
    """
    if library is None:
        library = get_library()
    if coupling is None:
        coupling = get_coupling(library)

    query = state.get("requirement", "")
    context: Dict[str, Any] = {
        "library_stats": library.statistics(),
        "factor_suggestions": query_to_factor_suggestions(query, top_k=5),
        "knowledge_text": build_enriched_knowledge(query),
    }

    # Transformer 耦合上下文
    if query:
        try:
            transformer_ctx = coupling.build_agent_context(query, top_k_factor=8)
            context["transformer_context"] = transformer_ctx
        except Exception:
            pass

    return context


# ---------------------------------------------------------------------------
# 批量因子生产接口
# ---------------------------------------------------------------------------

def mass_produce_from_library(
    kline: Optional[Any] = None,
    generations: int = 8,
    windows: Optional[List[int]] = None,
    event_driven: bool = False,
) -> Dict[str, Any]:
    """一键批量因子生产 — 组合参数扩增 + GP 演化。

    Args:
        kline: 行情数据 DataFrame
        generations: GP 演化代数
        windows: 参数窗口列表
        event_driven: 是否启用事件驱动模式

    Returns:
        {"factors": [...], "stats": {...}}
    """
    library = get_library()

    # Phase 1: 参数扩增
    expanded = library.cluster_expand_all(windows=windows)
    logger.info(f"参数扩增: {len(expanded)} 个新变体")

    # Phase 2: GP 演化（如果有 kline 数据）
    gp_factors: List[Dict] = []
    if kline is not None:
        try:
            evolver = EnhancedFactorEvolver(kline, library=library)
            result = evolver.mass_produce(
                generations=generations,
                pop_per_cluster=20,
                top_k_per_cluster=8,
                auto_save=True,
                verbose=True,
            )
            gp_factors = result.get("factors", [])
            logger.info(f"GP演化: {len(gp_factors)} 个优质因子")
        except Exception as e:
            logger.warning(f"GP演化异常: {e}")

    stats = library.statistics()
    return {
        "factors": gp_factors[:30],
        "param_expanded": expanded,
        "stats": stats,
        "total_in_library": stats["total"],
    }


def analyze_unstructured_file(
    file_path: str,
) -> Dict[str, Any]:
    """分析用户上传的非结构化文件。"""
    parser = DataUploadParser()
    df, meta = parser.parse_file(file_path)

    result = {"meta": meta, "preview": df.head(10).to_dict("records")}

    # 如果含文本列，分析情感
    if "text" in meta["column_mapping"]:
        analyzer = get_text_analyzer()
        sentiments = analyzer.analyze_batch(df[meta["column_mapping"]["text"]])
        result["sentiment"] = {
            "mean": float(sentiments["sentiment"].mean()),
            "pos_hits": int(sentiments["pos_hits"].sum()),
            "neg_hits": int(sentiments["neg_hits"].sum()),
            "industries": [i for inds in sentiments["industries"] for i in inds],
        }

    # 尝试转为因子时序
    try:
        factor_df = parser.to_factor_time_series(df, meta["column_mapping"])
        result["factor_preview"] = factor_df.head(5).to_dict("records")
        result["factor_ready"] = True
    except Exception:
        result["factor_ready"] = False

    return result
