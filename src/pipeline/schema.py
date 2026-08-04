"""因子精炼厂 · 数据契约 (Schema)。

定义六阶段流水线在阶段之间传递的标准化数据结构，确保「矿石」到「成品」的
数据流清晰、可审计、可复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class OreStock:
    """PART-01 矿石原料仓：统一数据底座。

    Attributes
    ----------
    universe      : 中证1000 成分股列表
    train_kline   : 训练集行情面板 (date, symbol, close, open, high, low, volume, ...)
    test_kline    : 测试集行情面板
    industry      : 个股→行业映射 (9 个行业分类维度)
    style         : 个股→风格映射 (3 个风格分类维度)
    raw_features  : 28 个分钟级原始特征面板
    factor_pool   : 50+ 个时序/截面因子池
    meta          : 元信息（建池期、数据来源等）
    """
    universe: List[str]
    train_kline: pd.DataFrame
    test_kline: pd.DataFrame
    industry: pd.DataFrame
    style: pd.DataFrame
    raw_features: pd.DataFrame
    factor_pool: Dict[str, pd.Series] = field(default_factory=dict)
    meta: Dict = field(default_factory=dict)


@dataclass
class CandidateFactor:
    """单个候选因子（来自 LLM 矿场 / RL 搜索 / 研磨车间向量化表征）。"""
    name: str
    source: str                       # "llm" | "rl" | "transformer" | "manual" | "pool"
    code: Optional[str] = None        # LLM 生成的因子代码
    series: Optional[pd.Series] = None
    description: str = ""
    metrics: Dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    # P1 LLM 可解释性：自然语言逻辑解释 + 学术/实务依据引用
    rationale: str = ""
    references: List[str] = field(default_factory=list)


@dataclass
class RefineryResult:
    """精炼厂最终交付物。"""
    ore: OreStock
    candidates: List[CandidateFactor] = field(default_factory=list)
    screened: List[CandidateFactor] = field(default_factory=list)
    composite: Optional[pd.Series] = None
    composite_metrics: Dict = field(default_factory=dict)
    loo_result: Dict = field(default_factory=dict)
    report_path: Optional[str] = None
    stage_trace: List[Dict] = field(default_factory=list)
    # P0 组合级回测与过拟合检验
    robustness: Optional[Dict] = None            # 过拟合检验（walk-forward/DSR/参数稳定性）
    portfolio: Optional[Dict] = None             # A 股现实约束组合回测（含净值/调仓清单）
    cost_sensitivity: Optional[Dict] = None      # 换手成本情景
    ic_by_year: Optional[Dict] = None            # 分年度 IC
    benchmark_comparison: Optional[Dict] = None  # 与中证800等权基准对比
    # P2 因子动物园
    factor_zoo: Optional[Dict] = None            # 主流因子横向对比 / 增量信息
    # P1 数据广度与多模态
    multimodal_factors: Optional[List[str]] = None  # 纳入因子池的多模态因子名
    # 评估窗口：复合因子性能指标所用的数据集，"test"=样本外 / "train"=样本内回退
    eval_set: str = "train"
    # PART-04 三级筛选审计留痕（LASSO / 人机协同 / TOP-K 各级进出数量与人工剔除明细）
    screen_audit: Dict = field(default_factory=dict)
