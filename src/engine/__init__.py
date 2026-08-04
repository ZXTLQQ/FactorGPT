"""FactorGPT 因子计算与回测引擎子包。

核心模块：
- backtest: 因子回测评估
- factor_builder: 沙箱执行 + 因子构造
- genetic_factors: 遗传规划因子发现（基础版）
- genetic_enhanced: 增强遗传规划因子挖掘（因子簇/事件簇/岛屿模型/批量生产）
- traditional_factors: 传统因子库（55+ 预置因子，五大方向）
- factor_library: 因子库管理器（CRUD/搜索/扩增/融合）
- unstructured_miner: 非结构化数据因子挖掘（文本/上传/另类数据）
- transformer_coupling: Transformer-Agent 深度耦合（编码/注意力/评分/记忆）
- risk_model: 风险模型归因
- tracking: 实验追踪（MLflow）
"""

from .backtest import FactorBacktester
from .factor_builder import FactorSandbox, build_pipeline, analyze_lookahead, generate_from_keywords
from .genetic_factors import GeneticFactorMiner
from .genetic_enhanced import (
    EnhancedFactorEvolver,
    FactorCluster,
    EventWindow,
    random_expr,
    eval_expr,
    expr_to_code,
)
from .traditional_factors import (
    FactorDef,
    ALL_CATEGORIES,
    CATEGORY_LABELS,
    get_all_factors,
    get_factors_by_category,
    get_factor_by_name,
    search_factors,
    get_factor_stats,
    export_all_to_dict,
)
from .factor_library import (
    FactorLibrary,
    create_default_library,
    mass_produce_factors,
)
from .unstructured_miner import (
    TextAnalyzer,
    DataUploadParser,
    AlternativeDataManager,
    UnstructuredFactorIntegrator,
)
from .transformer_coupling import (
    FactorEncoder,
    CrossAttentionFusion,
    FactorScorer,
    PatternMemory,
    TransformerCoupling,
)

__all__ = [
    # backtest
    "FactorBacktester",
    # factor_builder
    "FactorSandbox",
    "build_pipeline",
    "analyze_lookahead",
    "generate_from_keywords",
    # genetic
    "GeneticFactorMiner",
    "EnhancedFactorEvolver",
    "FactorCluster",
    "EventWindow",
    "random_expr",
    "eval_expr",
    "expr_to_code",
    # traditional_factors
    "FactorDef",
    "ALL_CATEGORIES",
    "CATEGORY_LABELS",
    "get_all_factors",
    "get_factors_by_category",
    "get_factor_by_name",
    "search_factors",
    "get_factor_stats",
    "export_all_to_dict",
    # factor_library
    "FactorLibrary",
    "create_default_library",
    "mass_produce_factors",
    # unstructured_miner
    "TextAnalyzer",
    "DataUploadParser",
    "AlternativeDataManager",
    "UnstructuredFactorIntegrator",
    # transformer_coupling
    "FactorEncoder",
    "CrossAttentionFusion",
    "FactorScorer",
    "PatternMemory",
    "TransformerCoupling",
]
