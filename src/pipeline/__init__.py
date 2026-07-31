"""因子精炼厂流水线包。

六阶段因子冶炼流水线：数据底座 → 三维生成 → RPN 评估 → 三级筛选 → AlphaPool 合成 → 方法学总结。

注意：为避免与 data 包的循环导入（refinery 依赖 data.feature_forge，
而 feature_forge 依赖 pipeline.schema），此处仅导入无外部依赖的 schema，
其余子模块请在调用处用 `from pipeline.refinery import ...` 显式导入。
"""

from .schema import OreStock, CandidateFactor, RefineryResult

__all__ = [
    "OreStock", "CandidateFactor", "RefineryResult",
]
