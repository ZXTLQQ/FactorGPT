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
from .factor_builder import FactorSandbox, analyze_lookahead, build_pipeline, generate_from_keywords
from .factor_library import (
    FactorLibrary,
    create_default_library,
    mass_produce_factors,
)
from .genetic_enhanced import (
    EnhancedFactorEvolver,
    EventWindow,
    FactorCluster,
    eval_expr,
    expr_to_code,
    random_expr,
)
from .genetic_factors import GeneticFactorMiner
from .traditional_factors import (
    ALL_CATEGORIES,
    CATEGORY_LABELS,
    FactorDef,
    export_all_to_dict,
    get_all_factors,
    get_factor_by_name,
    get_factor_stats,
    get_factors_by_category,
    search_factors,
)
from .transformer_coupling import (
    CrossAttentionFusion,
    FactorEncoder,
    FactorScorer,
    PatternMemory,
    TransformerCoupling,
)
from .unstructured_miner import (
    AlternativeDataManager,
    DataUploadParser,
    TextAnalyzer,
    UnstructuredFactorIntegrator,
)

__all__ = [
    "ALL_CATEGORIES",
    "CATEGORY_LABELS",
    "AlternativeDataManager",
    "CrossAttentionFusion",
    "DataUploadParser",
    "EnhancedFactorEvolver",
    "EventWindow",
    # backtest
    "FactorBacktester",
    "FactorCluster",
    # traditional_factors
    "FactorDef",
    # transformer_coupling
    "FactorEncoder",
    # factor_library
    "FactorLibrary",
    # factor_builder
    "FactorSandbox",
    "FactorScorer",
    # genetic
    "GeneticFactorMiner",
    "PatternMemory",
    # unstructured_miner
    "TextAnalyzer",
    "TransformerCoupling",
    "UnstructuredFactorIntegrator",
    "analyze_lookahead",
    "build_pipeline",
    "create_default_library",
    "eval_expr",
    "export_all_to_dict",
    "expr_to_code",
    "generate_from_keywords",
    "get_all_factors",
    "get_factor_by_name",
    "get_factor_stats",
    "get_factors_by_category",
    "mass_produce_factors",
    "random_expr",
    "search_factors",
]
