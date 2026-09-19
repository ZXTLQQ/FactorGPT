"""统一因子挖掘与多维度评价（FactorGPT mining layer）。

四篇卖方研报的工程化落地：

============  ====================================================================
来源            落地内容
============  ====================================================================
山西证券        ``ops``（Numba 加速算子库，同一实现既作 JIT 目标又作 NumPy 回退）
                ``gridminer``（分层算子网格搜索：预算/剪枝/去重/早停）
                ``evaluator``（四维评价：数据质量 / 预测能力 / 稳定性 / 相关性）
                ``report``（把搜索结果与评价结论落成可复核的 Markdown / JSON）
中信建投        ``expr``（强类型表达式树，量价与基本面共处同一因子空间）
                ``panel``（字段注册表 + ``asof_align`` PIT 投影，杜绝财务前视）
                ``fundamental``（TTM/YoY/QoQ、估值比率、量价×基本面混合模板）
天风证券        ``risk``（ΔR²/调整 ΔR²、系数 |t| 与 |t|>2 占比、VIF、自相关、
                拥挤度、因子收益分解与组合风险贡献）
西部证券        ``concept``（概念数量因子：CN/ACN/稀缺度/衰减/IN/AIN）
============  ====================================================================

子模块按需导入（``from mining import evaluator``），避免无关依赖被提前拉起；
``report`` 依赖最轻（只要 pandas/numpy），可独立用于渲染既有结果。
"""

from __future__ import annotations

__version__ = "1.1.0"

from . import ops, panel

__all__ = [
           "concept",
           "evaluator",
           "expr",
           "fundamental",
           "gridminer",
           "ops",
           "panel",
           "report",
           "risk",
           "split",
]
