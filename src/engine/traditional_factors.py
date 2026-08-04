"""
传统因子库 (src/engine/traditional_factors.py)

提供 100+ 预置量化因子，覆盖五大方向：
  1. 价格行为与趋势 (PRICE_TREND)
  2. 价格波动的不确定性 (VOLATILITY_UNCERTAINTY)
  3. 交易的难易程度 (TRADING_DIFFICULTY)
  4. 价与量变动的同步/背离关系 (PRICE_VOLUME_DIVERGENCE)
  5. 基于成交量与价格的公式化计算指标 (VOLUME_PRICE_FORMULA)

每个因子以标准 alpha_factor(df) 格式产出，可直接交付 FactorSandbox 执行。
利用本库可批量生产海量因子，配合遗传规划和 LLM 实现因子簇/事件簇挖矿。
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 因子分类常量
# ---------------------------------------------------------------------------
CATEGORY_PRICE_TREND = "price_trend"
CATEGORY_VOLATILITY = "volatility_uncertainty"
CATEGORY_TRADING_DIFFICULTY = "trading_difficulty"
CATEGORY_PRICE_VOLUME_DIVERGENCE = "price_volume_divergence"
CATEGORY_VOLUME_PRICE_FORMULA = "volume_price_formula"

ALL_CATEGORIES = [
    CATEGORY_PRICE_TREND,
    CATEGORY_VOLATILITY,
    CATEGORY_TRADING_DIFFICULTY,
    CATEGORY_PRICE_VOLUME_DIVERGENCE,
    CATEGORY_VOLUME_PRICE_FORMULA,
]

CATEGORY_LABELS = {
    CATEGORY_PRICE_TREND: "价格行为与趋势",
    CATEGORY_VOLATILITY: "价格波动的不确定性",
    CATEGORY_TRADING_DIFFICULTY: "交易的难易程度",
    CATEGORY_PRICE_VOLUME_DIVERGENCE: "价量变动同步/背离关系",
    CATEGORY_VOLUME_PRICE_FORMULA: "成交量与价格公式化指标",
}


# ---------------------------------------------------------------------------
# 因子定义数据结构
# ---------------------------------------------------------------------------
@dataclass
class FactorDef:
    """单个传统因子的元信息与代码。"""
    name: str                       # 唯一标识符（英文）
    display_name: str               # 中文显示名称
    category: str                   # 所属大类
    description: str                # 因子含义/逻辑
    direction: str                  # "positive" / "negative" / "none"
    code: str                       # 可执行的 alpha_factor 代码
    tags: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    source: str = "traditional_library"  # 来源标记
    quality_score: float = 0.5      # 经验质量分（0-1，用于初筛排序）

    def hash_id(self) -> str:
        return hashlib.md5(self.code.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "category": self.category,
            "category_label": CATEGORY_LABELS.get(self.category, self.category),
            "description": self.description,
            "direction": self.direction,
            "code": self.code,
            "tags": self.tags,
            "params": self.params,
            "source": self.source,
            "quality_score": self.quality_score,
            "hash": self.hash_id(),
        }


# ===================================================================
# 一、价格行为与趋势 (PRICE_TREND)
# ===================================================================
PRICE_TREND_FACTORS: List[FactorDef] = [
    # ---- 经典动量 ----
    FactorDef(
        name="momentum_5d", display_name="5日动量",
        category=CATEGORY_PRICE_TREND,
        description="过去5个交易日累计收益率（含分红再投），剔除最近1日防止未来函数。趋势延续信号。",
        direction="positive", quality_score=0.6,
        tags=["动量", "短期", "趋势"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ret'] = df.groupby('symbol')['pct_chg'].transform(lambda x: x / 100.0)
    df['factor'] = df.groupby('symbol')['ret'].transform(lambda x: x.rolling(5).sum())
    df['factor'] = df.groupby('symbol')['factor'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="momentum_10d", display_name="10日动量",
        category=CATEGORY_PRICE_TREND,
        description="过去10个交易日累计收益率。中期趋势延续。",
        direction="positive", quality_score=0.65,
        tags=["动量", "中期", "趋势"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ret'] = df.groupby('symbol')['pct_chg'].transform(lambda x: x / 100.0)
    df['factor'] = df.groupby('symbol')['ret'].transform(lambda x: x.rolling(10).sum())
    df['factor'] = df.groupby('symbol')['factor'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="momentum_20d", display_name="20日动量",
        category=CATEGORY_PRICE_TREND,
        description="过去20个交易日累计收益率（月度动量）。经典中期因子，A股有效性显著。",
        direction="positive", quality_score=0.75,
        tags=["动量", "中长期", "趋势", "月度"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ret'] = df.groupby('symbol')['pct_chg'].transform(lambda x: x / 100.0)
    df['factor'] = df.groupby('symbol')['ret'].transform(lambda x: x.rolling(20).sum())
    df['factor'] = df.groupby('symbol')['factor'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="momentum_60d", display_name="60日动量",
        category=CATEGORY_PRICE_TREND,
        description="过去60个交易日累计收益率（季度动量）。长期趋势因子。",
        direction="positive", quality_score=0.55,
        tags=["动量", "长期", "趋势", "季度"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ret'] = df.groupby('symbol')['pct_chg'].transform(lambda x: x / 100.0)
    df['factor'] = df.groupby('symbol')['ret'].transform(lambda x: x.rolling(60).sum())
    df['factor'] = df.groupby('symbol')['factor'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="momentum_residual_20d", display_name="20日残差动量",
        category=CATEGORY_PRICE_TREND,
        description="剔除市场收益后的残差动量。用所有股票截面均值作为市场代理，剥离系统性Beta后度量个股alpha。",
        direction="positive", quality_score=0.70,
        tags=["动量", "残差", "中性化", "alpha"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ret'] = df.groupby('symbol')['close'].pct_change()
    df['mkt_ret'] = df.groupby('date')['ret'].transform('mean')
    df['residual'] = df['ret'] - df['mkt_ret']
    df['factor'] = df.groupby('symbol')['residual'].transform(lambda x: x.rolling(20).sum())
    df['factor'] = df.groupby('symbol')['factor'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),

    # ---- 短期反转 ----
    FactorDef(
        name="reversal_1d", display_name="1日反转",
        category=CATEGORY_PRICE_TREND,
        description="T-1 日收益取反。极短期过度反应后价格回弹，A股散户主导环境下效果明显。",
        direction="positive", quality_score=0.80,
        tags=["反转", "短期", "行为金融"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ret'] = df.groupby('symbol')['pct_chg'].transform(lambda x: x / 100.0)
    df['factor'] = df.groupby('symbol')['ret'].shift(1)
    df['factor'] = -df['factor']
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="reversal_5d", display_name="5日反转",
        category=CATEGORY_PRICE_TREND,
        description="过去5个交易日累计收益取反。短期过度反应均值回复。",
        direction="positive", quality_score=0.70,
        tags=["反转", "短期", "均值回复"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ret'] = df.groupby('symbol')['pct_chg'].transform(lambda x: x / 100.0)
    df['momentum'] = df.groupby('symbol')['ret'].transform(lambda x: x.rolling(5).sum())
    df['factor'] = -df.groupby('symbol')['momentum'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="reversal_10d", display_name="10日反转",
        category=CATEGORY_PRICE_TREND,
        description="过去10日收益取反。中期过度反应修正，兼有动量与反转交叉效应。",
        direction="positive", quality_score=0.60,
        tags=["反转", "中期"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ret'] = df.groupby('symbol')['pct_chg'].transform(lambda x: x / 100.0)
    df['momentum'] = df.groupby('symbol')['ret'].transform(lambda x: x.rolling(10).sum())
    df['factor'] = -df.groupby('symbol')['momentum'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),

    # ---- 均线交叉 ----
    FactorDef(
        name="ma_cross_5_20", display_name="MA(5,20)交叉",
        category=CATEGORY_PRICE_TREND,
        description="5日均线与20日均线的差值（经价格归一化）。正值表示短期均价在长期之上，趋势偏多。",
        direction="positive", quality_score=0.55,
        tags=["均线", "交叉", "趋势"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ma5'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma20'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(20).mean())
    df['factor'] = (df['ma5'] - df['ma20']) / (df['ma20'] + 1e-8)
    df['factor'] = df.groupby('symbol')['factor'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="ma_cross_10_60", display_name="MA(10,60)交叉",
        category=CATEGORY_PRICE_TREND,
        description="10日均线与60日均线的差值比例。中期趋势方向判别。",
        direction="positive", quality_score=0.50,
        tags=["均线", "交叉", "中长趋势"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ma10'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(10).mean())
    df['ma60'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(60).mean())
    df['factor'] = (df['ma10'] - df['ma60']) / (df['ma60'] + 1e-8)
    df['factor'] = df.groupby('symbol')['factor'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="ma_cross_20_60", display_name="MA(20,60)交叉",
        category=CATEGORY_PRICE_TREND,
        description="20日与60日均线交叉（季度趋势确认）。",
        direction="positive", quality_score=0.55,
        tags=["均线", "交叉", "季度"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ma20'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(20).mean())
    df['ma60'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(60).mean())
    df['factor'] = (df['ma20'] - df['ma60']) / (df['ma60'] + 1e-8)
    df['factor'] = df.groupby('symbol')['factor'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),

    # ---- 价格形态 ----
    FactorDef(
        name="close_to_high_20d", display_name="收盘接近20日高点",
        category=CATEGORY_PRICE_TREND,
        description="(close - min_20) / (max_20 - min_20)。衡量收盘价在近20日区间中的相对位置，接近高点为强势。",
        direction="positive", quality_score=0.65,
        tags=["价格形态", "区间", "相对位置"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['max20'] = df.groupby('symbol')['high'].transform(lambda x: x.rolling(20).max())
    df['min20'] = df.groupby('symbol')['low'].transform(lambda x: x.rolling(20).min())
    df['factor'] = (df['close'] - df['min20']) / (df['max20'] - df['min20'] + 1e-8)
    df['factor'] = df.groupby('symbol')['factor'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="close_to_high_60d", display_name="收盘接近60日高点",
        category=CATEGORY_PRICE_TREND,
        description="收盘价在近60日（一季度）区间中的相对位置，越大越接近季度新高。",
        direction="positive", quality_score=0.60,
        tags=["价格形态", "季度", "创新高"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['max60'] = df.groupby('symbol')['high'].transform(lambda x: x.rolling(60).max())
    df['min60'] = df.groupby('symbol')['low'].transform(lambda x: x.rolling(60).min())
    df['factor'] = (df['close'] - df['min60']) / (df['max60'] - df['min60'] + 1e-8)
    df['factor'] = df.groupby('symbol')['factor'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="max_up_consecutive", display_name="连涨天数",
        category=CATEGORY_PRICE_TREND,
        description="截至昨日的连续上涨天数。连涨越多，动量越强。",
        direction="positive", quality_score=0.50,
        tags=["连涨", "趋势强度"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['up'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['consec'] = df.groupby('symbol')['up'].transform(lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1))
    df['factor'] = df.groupby('symbol')['consec'].shift(1).fillna(0)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="breakout_20d_high", display_name="突破20日新高",
        category=CATEGORY_PRICE_TREND,
        description="当日收盘价是否等于近20日最高（1/0），突破信号。经shift避免未来函数后表征 'T-1是否突破'。",
        direction="positive", quality_score=0.55,
        tags=["突破", "新高", "信号"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['max20'] = df.groupby('symbol')['high'].transform(lambda x: x.rolling(20).max())
    df['breakout'] = (df['close'] >= df['max20']).astype(float)
    df['factor'] = df.groupby('symbol')['breakout'].shift(1).fillna(0.0)
    return df[['date', 'symbol', 'factor']]
"""
    ),

    # ---- 趋势质量 ----
    FactorDef(
        name="trend_strength_20d", display_name="20日趋势强度",
        category=CATEGORY_PRICE_TREND,
        description="20日收益率 / 20日收盘价标准差 = 信息比率（趋势/噪声比）。",
        direction="positive", quality_score=0.60,
        tags=["趋势质量", "信息比率"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ret'] = df.groupby('symbol')['close'].pct_change()
    df['cum_ret'] = df.groupby('symbol')['ret'].transform(lambda x: x.rolling(20).sum())
    df['vol'] = df.groupby('symbol')['ret'].transform(lambda x: x.rolling(20).std())
    df['factor'] = df['cum_ret'] / (df['vol'] + 1e-8)
    df['factor'] = df.groupby('symbol')['factor'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="rsi_14d", display_name="RSI(14)",
        category=CATEGORY_PRICE_TREND,
        description="14日相对强弱指标。RSI=100*(avg_gain/(avg_gain+avg_loss))，经典技术因子。",
        direction="positive", quality_score=0.55,
        tags=["RSI", "技术指标", "超买超卖"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['delta'] = df.groupby('symbol')['close'].diff()
    df['gain'] = np.where(df['delta'] > 0, df['delta'], 0)
    df['loss'] = np.where(df['delta'] < 0, -df['delta'], 0)
    df['avg_gain'] = df.groupby('symbol')['gain'].transform(lambda x: x.rolling(14).mean())
    df['avg_loss'] = df.groupby('symbol')['loss'].transform(lambda x: x.rolling(14).mean())
    df['rs'] = df['avg_gain'] / (df['avg_loss'] + 1e-8)
    df['rsi'] = 100 - 100 / (1 + df['rs'])
    df['factor'] = df.groupby('symbol')['rsi'].shift(1) / 100.0
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="macd_signal", display_name="MACD 信号线",
        category=CATEGORY_PRICE_TREND,
        description="MACD 指标 DIF-DEA。DIF=EMA(12)-EMA(26)，DEA=EMA(DIF,9)。经价格归一化。",
        direction="positive", quality_score=0.50,
        tags=["MACD", "技术指标", "趋势"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    def _macd_hist(grp):
        ema12 = grp['close'].ewm(span=12, adjust=False).mean()
        ema26 = grp['close'].ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        return (dif - dea) / (grp['close'] + 1e-8)
    df['macd_hist'] = df.groupby('symbol', group_keys=False).apply(_macd_hist)
    df['factor'] = df.groupby('symbol')['macd_hist'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),

    # ---- 路径相关 ----
    FactorDef(
        name="path_dependency_maxdd_20d", display_name="20日最大回撤",
        category=CATEGORY_PRICE_TREND,
        description="近20个交易日的最大回撤（取正）。回撤越大，反弹潜力越高（反转逻辑）。",
        direction="positive", quality_score=0.60,
        tags=["回撤", "路径依赖", "反转"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    def _maxdd(x):
        cummax = x.cummax()
        dd = (cummax - x) / (cummax + 1e-8)
        return dd.max()
    df['maxdd'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(20).apply(_maxdd, raw=False))
    df['factor'] = df.groupby('symbol')['maxdd'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
]

# ===================================================================
# 二、价格波动的不确定性 (VOLATILITY_UNCERTAINTY)
# ===================================================================
VOLATILITY_FACTORS: List[FactorDef] = [
    FactorDef(
        name="realized_vol_5d", display_name="5日实现波动率",
        category=CATEGORY_VOLATILITY,
        description="5日日收益率标准差。短期不确定性。波动率越低，往往后续收益更稳定。",
        direction="negative", quality_score=0.65,
        tags=["波动率", "短期", "不确定性"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ret'] = df.groupby('symbol')['close'].pct_change()
    df['vol'] = df.groupby('symbol')['ret'].transform(lambda x: x.rolling(5).std())
    df['factor'] = -df.groupby('symbol')['vol'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="realized_vol_20d", display_name="20日实现波动率",
        category=CATEGORY_VOLATILITY,
        description="20日日收益率标准差（月度波动率）。经典低波因子，A股有显著低波溢价。",
        direction="negative", quality_score=0.80,
        tags=["波动率", "月度", "低波异象"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ret'] = df.groupby('symbol')['close'].pct_change()
    df['vol'] = df.groupby('symbol')['ret'].transform(lambda x: x.rolling(20).std())
    df['factor'] = -df.groupby('symbol')['vol'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="realized_vol_60d", display_name="60日实现波动率",
        category=CATEGORY_VOLATILITY,
        description="60日日收益率标准差（季度波动率）。长期不确定性的度量。",
        direction="negative", quality_score=0.60,
        tags=["波动率", "季度", "长期"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ret'] = df.groupby('symbol')['close'].pct_change()
    df['vol'] = df.groupby('symbol')['ret'].transform(lambda x: x.rolling(60).std())
    df['factor'] = -df.groupby('symbol')['vol'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),

    # ---- Parkinson 波动率（用 OHLC 日内信息） ----
    FactorDef(
        name="parkinson_vol_20d", display_name="20日 Parkinson 波动率",
        category=CATEGORY_VOLATILITY,
        description="基于日最高/最低价的 Parkinson 波动率估计，比收盘价波动率效率高5倍。度量日内波动幅度。",
        direction="negative", quality_score=0.70,
        tags=["波动率", "Parkinson", "日内"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['hl_ratio'] = np.log(df['high'] / (df['low'] + 1e-8))
    df['parkinson'] = df.groupby('symbol')['hl_ratio'].transform(
        lambda x: np.sqrt((1.0 / (4 * np.log(2))) * (x ** 2).rolling(20).mean()))
    df['factor'] = -df.groupby('symbol')['parkinson'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),

    # ---- Garman-Klass 波动率 ----
    FactorDef(
        name="gk_vol_20d", display_name="20日 Garman-Klass 波动率",
        category=CATEGORY_VOLATILITY,
        description="Garman-Klass (1980) 利用 OHLC 的波动率估计器，效率约为收盘价波动率的7.4倍。",
        direction="negative", quality_score=0.70,
        tags=["波动率", "Garman-Klass", "OHLC"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['oc'] = np.log(df['close'] / (df['open'] + 1e-8))
    df['hl'] = np.log(df['high'] / (df['low'] + 1e-8))
    df['gk_sq'] = 0.5 * df['hl']**2 - (2*np.log(2) - 1) * df['oc']**2
    df['gk_vol'] = df.groupby('symbol')['gk_sq'].transform(lambda x: np.sqrt(x.rolling(20).mean()))
    df['factor'] = -df.groupby('symbol')['gk_vol'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),

    # ---- ATR 平均真实波幅 ----
    FactorDef(
        name="atr_14d", display_name="ATR(14)",
        category=CATEGORY_VOLATILITY,
        description="14日平均真实波幅（Average True Range）。衡量价格波动幅度，高ATR意味着高不确定性与高交易成本。",
        direction="negative", quality_score=0.65,
        tags=["ATR", "波幅", "技术指标"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['prev_close'] = df.groupby('symbol')['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['prev_close']).abs()
    df['tr3'] = (df['low'] - df['prev_close']).abs()
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df.groupby('symbol')['tr'].transform(lambda x: x.rolling(14).mean())
    df['factor'] = -df.groupby('symbol')['atr'].shift(1) / (df.groupby('symbol')['close'].shift(1) + 1e-8)
    return df[['date', 'symbol', 'factor']]
"""
    ),

    # ---- 偏度与峰度 ----
    FactorDef(
        name="skewness_20d", display_name="20日收益率偏度",
        category=CATEGORY_VOLATILITY,
        description="近20日收益率的偏度（skewness）。正偏表示上涨极端值多（彩票型），负偏表示暴跌风险大。正偏因子通常为负溢价。",
        direction="negative", quality_score=0.65,
        tags=["偏度", "高阶矩", "尾部风险"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ret'] = df.groupby('symbol')['close'].pct_change()
    df['skew'] = df.groupby('symbol')['ret'].transform(lambda x: x.rolling(20).skew())
    df['factor'] = -df.groupby('symbol')['skew'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="kurtosis_20d", display_name="20日收益率峰度",
        category=CATEGORY_VOLATILITY,
        description="近20日收益率的超额峰度。峰度越高尾部越厚（极端风险越高），越低波动越稳定。",
        direction="negative", quality_score=0.55,
        tags=["峰度", "尾部风险", "高阶矩"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ret'] = df.groupby('symbol')['close'].pct_change()
    df['kurt'] = df.groupby('symbol')['ret'].transform(lambda x: x.rolling(20).kurt())
    df['factor'] = -df.groupby('symbol')['kurt'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),

    # ---- 波动率的波动率（Vol of Vol） ----
    FactorDef(
        name="vol_of_vol_20d", display_name="波动率的变化（Vol of Vol）",
        category=CATEGORY_VOLATILITY,
        description="5日滚动波动率的标准差（20日窗口）。波动率本身的波动，度量不确定性变化。",
        direction="negative", quality_score=0.60,
        tags=["Vol-of-Vol", "不确定性", "高阶"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ret'] = df.groupby('symbol')['close'].pct_change()
    df['vol5'] = df.groupby('symbol')['ret'].transform(lambda x: x.rolling(5).std())
    df['vol_of_vol'] = df.groupby('symbol')['vol5'].transform(lambda x: x.rolling(20).std())
    df['factor'] = -df.groupby('symbol')['vol_of_vol'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),

    # ---- 不对称波动（下行 vs 上行） ----
    FactorDef(
        name="downside_vol_20d", display_name="20日下行波动率",
        category=CATEGORY_VOLATILITY,
        description="仅计算负收益的标准差（下行波动）。下行波动越高，代表下跌风险越大。",
        direction="negative", quality_score=0.70,
        tags=["下行波动", "半方差", "风险"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ret'] = df.groupby('symbol')['close'].pct_change()
    df['neg_ret'] = np.where(df['ret'] < 0, df['ret'], 0)
    df['down_vol'] = df.groupby('symbol')['neg_ret'].transform(lambda x: x.rolling(20).std())
    df['factor'] = -df.groupby('symbol')['down_vol'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="up_down_vol_ratio_20d", display_name="上下行波动比",
        category=CATEGORY_VOLATILITY,
        description="上行波动/下行波动 - 1。比值高表示上涨波动大（彩票型），通常有负溢价。",
        direction="negative", quality_score=0.65,
        tags=["上下行", "不对称", "彩票"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ret'] = df.groupby('symbol')['close'].pct_change()
    df['pos_ret'] = np.where(df['ret'] > 0, df['ret'], 0)
    df['neg_ret'] = np.where(df['ret'] < 0, -df['ret'], 0)
    df['up_vol'] = df.groupby('symbol')['pos_ret'].transform(lambda x: x.rolling(20).std())
    df['down_vol'] = df.groupby('symbol')['neg_ret'].transform(lambda x: x.rolling(20).std())
    df['factor'] = -df['up_vol'] / (df['down_vol'] + 1e-8)
    df['factor'] = df.groupby('symbol')['factor'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),

    # ---- 跳跃/崩盘风险 ----
    FactorDef(
        name="jump_risk_20d", display_name="20日跳跃风险",
        category=CATEGORY_VOLATILITY,
        description="近20日中单日跌幅超过3%的天数。崩盘频率因子。",
        direction="negative", quality_score=0.55,
        tags=["跳跃", "崩盘", "尾部"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ret'] = df.groupby('symbol')['pct_chg'] / 100.0
    df['crash'] = (df['ret'] < -0.03).astype(float)
    df['jump_risk'] = df.groupby('symbol')['crash'].transform(lambda x: x.rolling(20).sum())
    df['factor'] = -df.groupby('symbol')['jump_risk'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),

    # ---- 最大振幅 ----
    FactorDef(
        name="amplitude_20d", display_name="20日平均振幅",
        category=CATEGORY_VOLATILITY,
        description="近20日 (high-low)/close 的均值。日内价格摆动幅度。",
        direction="negative", quality_score=0.55,
        tags=["振幅", "日内", "波动"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['amplitude'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
    df['avg_amp'] = df.groupby('symbol')['amplitude'].transform(lambda x: x.rolling(20).mean())
    df['factor'] = -df.groupby('symbol')['avg_amp'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
]

# ===================================================================
# 三、交易的难易程度 (TRADING_DIFFICULTY)
# ===================================================================
TRADING_DIFFICULTY_FACTORS: List[FactorDef] = [
    FactorDef(
        name="turnover_5d", display_name="5日平均换手率",
        category=CATEGORY_TRADING_DIFFICULTY,
        description="近5日成交额/流通市值的均值（此处用 amount 代理市值归一化）。高换手表示交易活跃，也常对应过度投机。A股低换手有溢价。",
        direction="negative", quality_score=0.70,
        tags=["换手率", "交易活跃度", "流动性"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['turn'] = df['amount'] / df.groupby('symbol')['amount'].transform(lambda x: x.rolling(20).mean() + 1e-8)
    df['avg_turn'] = df.groupby('symbol')['turn'].transform(lambda x: x.rolling(5).mean())
    df['factor'] = -df.groupby('symbol')['avg_turn'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="turnover_20d", display_name="20日平均换手率",
        category=CATEGORY_TRADING_DIFFICULTY,
        description="近20日成交额滚动均值归一化。经典流动性因子，低换手有显著正溢价。",
        direction="negative", quality_score=0.80,
        tags=["换手率", "流动性", "经典"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['turn'] = df['amount'] / df.groupby('symbol')['amount'].transform(lambda x: x.rolling(60).mean() + 1e-8)
    df['avg_turn'] = df.groupby('symbol')['turn'].transform(lambda x: x.rolling(20).mean())
    df['factor'] = -df.groupby('symbol')['avg_turn'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="turnover_change_5d", display_name="换手率变化（5日）",
        category=CATEGORY_TRADING_DIFFICULTY,
        description="今日换手率/5日均换手率 - 1。换手突增往往意味着信息冲击或事件驱动，不确定性升高。",
        direction="negative", quality_score=0.65,
        tags=["换手率变化", "信息冲击", "事件"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['turn'] = df['amount'] / df.groupby('symbol')['amount'].transform(lambda x: x.rolling(20).mean() + 1e-8)
    df['avg_turn'] = df.groupby('symbol')['turn'].transform(lambda x: x.rolling(5).mean())
    df['turn_change'] = (df['turn'] - df['avg_turn']) / (df['avg_turn'] + 1e-8)
    df['factor'] = -df.groupby('symbol')['turn_change'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="amihud_illiquidity_20d", display_name="Amihud 非流动性 (20日)",
        category=CATEGORY_TRADING_DIFFICULTY,
        description="|ret|/(volume*price) 的20日均值。度量价格冲击：同样的交易量导致的收益率变化越大，流动性越差。非流动性越高，预期收益补偿越大。",
        direction="positive", quality_score=0.75,
        tags=["Amihud", "非流动性", "价格冲击"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ret'] = df.groupby('symbol')['close'].pct_change().abs()
    df['dollar_vol'] = df['volume'] * df['close'] + 1e-8
    df['illiq'] = df['ret'] / df['dollar_vol']
    df['avg_illiq'] = df.groupby('symbol')['illiq'].transform(lambda x: x.rolling(20).mean())
    df['factor'] = df.groupby('symbol')['avg_illiq'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="dollar_volume_20d", display_name="20日日均成交额",
        category=CATEGORY_TRADING_DIFFICULTY,
        description="近20日 (close * volume) 的均值取对数。大盘股流动性好，低成交额小盘股有溢价。",
        direction="negative", quality_score=0.70,
        tags=["成交额", "规模", "流动性"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['dvol'] = df['close'] * df['volume']
    df['avg_dvol'] = df.groupby('symbol')['dvol'].transform(lambda x: np.log1p(x.rolling(20).mean()))
    df['factor'] = -df.groupby('symbol')['avg_dvol'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),

    # ---- 买卖价差代理 ----
    FactorDef(
        name="high_low_spread_20d", display_name="20日高低价差率",
        category=CATEGORY_TRADING_DIFFICULTY,
        description="(high-low)/close 的20日均值。日内价差代理买卖价差成本，越大交易越困难。",
        direction="negative", quality_score=0.60,
        tags=["价差", "交易成本", "日内"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['spread'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
    df['avg_spread'] = df.groupby('symbol')['spread'].transform(lambda x: x.rolling(20).mean())
    df['factor'] = -df.groupby('symbol')['avg_spread'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),

    # ---- 涨停/跌停接触 ----
    FactorDef(
        name="limit_up_touch_20d", display_name="20日涨停触及次数",
        category=CATEGORY_TRADING_DIFFICULTY,
        description="近20日中 pct_chg >= 9.5 的天数。涨停频繁的股票往往被游资炒作，后续回调风险大。",
        direction="negative", quality_score=0.60,
        tags=["涨停", "投机", "游资"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['limit_up'] = (df['pct_chg'] >= 9.5).astype(float)
    df['touch_cnt'] = df.groupby('symbol')['limit_up'].transform(lambda x: x.rolling(20).sum())
    df['factor'] = -df.groupby('symbol')['touch_cnt'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="limit_down_touch_20d", display_name="20日跌停触及次数",
        category=CATEGORY_TRADING_DIFFICULTY,
        description="近20日中 pct_chg <= -9.5 的天数。跌停频繁的股票基本面恶化，应规避。",
        direction="negative", quality_score=0.65,
        tags=["跌停", "风险", "基本面"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['limit_down'] = (df['pct_chg'] <= -9.5).astype(float)
    df['touch_cnt'] = df.groupby('symbol')['limit_down'].transform(lambda x: x.rolling(20).sum())
    df['factor'] = -df.groupby('symbol')['touch_cnt'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),

    # ---- 交易集中度 ----
    FactorDef(
        name="volume_concentration_20d", display_name="20日成交量集中度",
        category=CATEGORY_TRADING_DIFFICULTY,
        description="近20日中最大单日成交量/总成交量。交易过于集中在某几天反映大资金进出，不确定性高。",
        direction="negative", quality_score=0.55,
        tags=["集中度", "大资金", "分配"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['max_vol'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).max())
    df['sum_vol'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).sum())
    df['conc'] = df['max_vol'] / (df['sum_vol'] + 1e-8)
    df['factor'] = -df.groupby('symbol')['conc'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),

    # ---- 零成交量/停牌 ----
    FactorDef(
        name="zero_volume_days_20d", display_name="20日零成交量天数",
        category=CATEGORY_TRADING_DIFFICULTY,
        description="近20日成交量接近0的天数（停牌/无交易）。停牌是不可交易性的极端形式。",
        direction="negative", quality_score=0.55,
        tags=["停牌", "零成交", "可交易性"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['zero_vol'] = (df['volume'] < 1e-6).astype(float)
    df['zero_days'] = df.groupby('symbol')['zero_vol'].transform(lambda x: x.rolling(20).sum())
    df['factor'] = -df.groupby('symbol')['zero_days'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
]

# ===================================================================
# 四、价与量变动的同步/背离关系 (PRICE_VOLUME_DIVERGENCE)
# ===================================================================
PV_DIVERGENCE_FACTORS: List[FactorDef] = [
    FactorDef(
        name="price_volume_corr_20d", display_name="20日量价相关系数",
        category=CATEGORY_PRICE_VOLUME_DIVERGENCE,
        description="近20日收盘价变化与成交量变化的滚动相关系数。量价同步上涨是强趋势（正相关），滞涨放量则是危险背离。",
        direction="positive", quality_score=0.70,
        tags=["量价相关", "同步", "背离"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ret'] = df.groupby('symbol')['close'].pct_change()
    df['vol_chg'] = df.groupby('symbol')['volume'].pct_change()
    def _corr(grp):
        return grp['ret'].rolling(20).corr(grp['vol_chg'])
    df['corr'] = df.groupby('symbol', group_keys=False).apply(_corr)
    df['factor'] = df.groupby('symbol')['corr'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="volume_delta_price_20d", display_name="20日量升价跌背离",
        category=CATEGORY_PRICE_VOLUME_DIVERGENCE,
        description="成交量上升但价格下跌的量价背离指标。(vol_20d_chg - price_20d_chg)的标准化。",
        direction="negative", quality_score=0.65,
        tags=["背离", "量升价跌", "预警"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['price_chg'] = df.groupby('symbol')['close'].transform(lambda x: x.pct_change(20))
    df['vol_chg'] = df.groupby('symbol')['volume'].transform(lambda x: x.pct_change(20))
    df['divergence'] = df['vol_chg'] - df['price_chg']
    df['factor'] = -df.groupby('symbol')['divergence'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="obv_momentum_20d", display_name="OBV 动量 (20日)",
        category=CATEGORY_PRICE_VOLUME_DIVERGENCE,
        description="On-Balance Volume (OBV) 的20日动量。OBV通过将成交量按涨跌方向累加，持续上升表示资金净流入。",
        direction="positive", quality_score=0.65,
        tags=["OBV", "资金流", "能量潮"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['direction'] = np.where(df['close'] > df.groupby('symbol')['close'].shift(1), 1,
                       np.where(df['close'] < df.groupby('symbol')['close'].shift(1), -1, 0))
    df['obv_delta'] = df['volume'] * df['direction']
    df['obv'] = df.groupby('symbol')['obv_delta'].cumsum()
    df['obv_mom'] = df.groupby('symbol')['obv'].transform(lambda x: x.pct_change(20))
    df['factor'] = df.groupby('symbol')['obv_mom'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="vwap_deviation_5d", display_name="5日VWAP偏离",
        category=CATEGORY_PRICE_VOLUME_DIVERGENCE,
        description="收盘价与5日成交量加权均价(VWAP)的偏离度。价格高于VWAP意味着短期资金认可当前价位。",
        direction="positive", quality_score=0.60,
        tags=["VWAP", "偏离", "资金认可"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['pv'] = df['close'] * df['volume']
    df['cum_pv'] = df.groupby('symbol')['pv'].transform(lambda x: x.rolling(5).sum())
    df['cum_vol'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(5).sum())
    df['vwap'] = df['cum_pv'] / (df['cum_vol'] + 1e-8)
    df['deviation'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-8)
    df['factor'] = df.groupby('symbol')['deviation'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="mfi_14d", display_name="资金流量指标 MFI(14)",
        category=CATEGORY_PRICE_VOLUME_DIVERGENCE,
        description="14日资金流量指标(Money Flow Index)，结合价格方向和成交金额，类似成交额加权的RSI。超买(>80)有回调风险，需结合因子方向判断。",
        direction="negative", quality_score=0.55,
        tags=["MFI", "资金流", "技术指标"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['mf'] = df['tp'] * df['volume']
    df['tp_diff'] = df['tp'].diff()
    df['pos_flow'] = np.where(df['tp_diff'] > 0, df['mf'], 0)
    df['neg_flow'] = np.where(df['tp_diff'] < 0, df['mf'], 0)
    df['pos_sum'] = df.groupby('symbol')['pos_flow'].transform(lambda x: x.rolling(14).sum())
    df['neg_sum'] = df.groupby('symbol')['neg_flow'].transform(lambda x: x.rolling(14).sum())
    df['mr'] = df['pos_sum'] / (df['neg_sum'] + 1e-8)
    df['mfi'] = 100 - 100 / (1 + df['mr'])
    df['factor'] = -df.groupby('symbol')['mfi'].shift(1) / 100.0
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="volume_price_ratio_20d", display_name="20日量价弹性",
        category=CATEGORY_PRICE_VOLUME_DIVERGENCE,
        description="20日成交量变化率 / 20日价格变化率的比值。弹性大：少量资金即可推动价格（筹码集中），弹性小：大量资金也无法推动（抛压大/筹码分散）。",
        direction="positive", quality_score=0.60,
        tags=["弹性", "量价", "筹码"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['vol_chg'] = df.groupby('symbol')['volume'].transform(lambda x: x.pct_change(20)).abs()
    df['price_chg'] = df.groupby('symbol')['close'].transform(lambda x: x.pct_change(20)).abs()
    df['elasticity'] = df['price_chg'] / (df['vol_chg'] + 1e-8)
    df['factor'] = df.groupby('symbol')['elasticity'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="volume_breakout_5d", display_name="5日成交量突破",
        category=CATEGORY_PRICE_VOLUME_DIVERGENCE,
        description="今日成交量 / 近20日均量 - 1。放量往往伴随重要信息到达，衡量信息冲击强度。",
        direction="negative", quality_score=0.60,
        tags=["放量", "信息冲击", "突破"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['avg_vol'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean())
    df['vol_ratio'] = df['volume'] / (df['avg_vol'] + 1e-8) - 1
    df['factor'] = -df.groupby('symbol')['vol_ratio'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="up_volume_ratio_20d", display_name="20日上涨量占比",
        category=CATEGORY_PRICE_VOLUME_DIVERGENCE,
        description="近20日中上涨日的成交量总和 / 总成交量。占比高说明买盘资金占主导。",
        direction="positive", quality_score=0.65,
        tags=["上涨量", "买盘", "资金分型"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['is_up'] = (df['close'] > df.groupby('symbol')['close'].shift(1))
    df['up_vol'] = df['volume'] * df['is_up'].astype(float)
    df['sum_up'] = df.groupby('symbol')['up_vol'].transform(lambda x: x.rolling(20).sum())
    df['sum_total'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).sum())
    df['up_ratio'] = df['sum_up'] / (df['sum_total'] + 1e-8)
    df['factor'] = df.groupby('symbol')['up_ratio'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="net_flow_pressure_5d", display_name="5日净买入压力",
        category=CATEGORY_PRICE_VOLUME_DIVERGENCE,
        description="(上涨日成交额 - 下跌日成交额) / 总成交额的5日均值。净资金流向。",
        direction="positive", quality_score=0.60,
        tags=["资金流", "净买入", "压力"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['is_up'] = (df['close'] > df.groupby('symbol')['close'].shift(1)).astype(float)
    df['signed_amount'] = df['amount'] * (2 * df['is_up'] - 1)
    df['net_flow'] = df.groupby('symbol')['signed_amount'].transform(lambda x: x.rolling(5).sum())
    df['total_amount'] = df.groupby('symbol')['amount'].transform(lambda x: x.rolling(5).sum())
    df['factor'] = df['net_flow'] / (df['total_amount'] + 1e-8)
    df['factor'] = df.groupby('symbol')['factor'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
]

# ===================================================================
# 五、基于成交量与价格的公式化计算指标 (VOLUME_PRICE_FORMULA)
# ===================================================================
VP_FORMULA_FACTORS: List[FactorDef] = [
    FactorDef(
        name="vcmf_20d", display_name="Chaikin 资金流 CMF(20)",
        category=CATEGORY_VOLUME_PRICE_FORMULA,
        description="20日 Chaikin Money Flow。((close-low)-(high-close))/(high-low) * volume 的20日均值。衡量资金进出压力。",
        direction="positive", quality_score=0.65,
        tags=["CMF", "Chaikin", "资金流"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['mf_mult'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-8)
    df['mf_vol'] = df['mf_mult'] * df['volume']
    df['sum_mfv'] = df.groupby('symbol')['mf_vol'].transform(lambda x: x.rolling(20).sum())
    df['sum_vol'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).sum())
    df['cmf'] = df['sum_mfv'] / (df['sum_vol'] + 1e-8)
    df['factor'] = df.groupby('symbol')['cmf'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="eom_14d", display_name="Ease of Movement (14日)",
        category=CATEGORY_VOLUME_PRICE_FORMULA,
        description="14日 Ease of Movement 指标。((high+low)/2 - prev_mid) / box_ratio。正值表示价格在低阻力下上升。",
        direction="positive", quality_score=0.60,
        tags=["EOM", "移动难易", "阻力"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['mid'] = (df['high'] + df['low']) / 2
    df['prev_mid'] = df.groupby('symbol')['mid'].shift(1)
    df['box_ratio'] = (df['amount'] / (df['volume'] + 1e-8)) / (df['high'] - df['low'] + 1e-8)
    df['eom'] = (df['mid'] - df['prev_mid']) / (df['box_ratio'] + 1e-8)
    df['eom_smooth'] = df.groupby('symbol')['eom'].transform(lambda x: x.rolling(14).mean())
    df['factor'] = df.groupby('symbol')['eom_smooth'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="vwma_ratio_5_20", display_name="VWMA(5)/VWMA(20)比例",
        category=CATEGORY_VOLUME_PRICE_FORMULA,
        description="5日成交量加权均线 / 20日成交量加权均线。类似均线交叉，但以成交量对价格加权，更能体现大资金意图。",
        direction="positive", quality_score=0.60,
        tags=["VWMA", "加权均线", "大资金"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    def _vwma(grp, w):
        pv = grp['close'] * grp['volume']
        sum_pv = pv.rolling(w).sum()
        sum_v = grp['volume'].rolling(w).sum()
        return sum_pv / (sum_v + 1e-8)
    df['vwma5'] = df.groupby('symbol', group_keys=False).apply(lambda g: _vwma(g, 5))
    df['vwma20'] = df.groupby('symbol', group_keys=False).apply(lambda g: _vwma(g, 20))
    df['ratio'] = df['vwma5'] / (df['vwma20'] + 1e-8) - 1
    df['factor'] = df.groupby('symbol')['ratio'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="keltner_pct_20d", display_name="Keltner 通道% (20日)",
        category=CATEGORY_VOLUME_PRICE_FORMULA,
        description="(close - Keltner下轨) / (Keltner上轨 - Keltner下轨)。基于ATR的动态通道中的位置，越接近上轨趋势越强。",
        direction="positive", quality_score=0.55,
        tags=["Keltner", "通道", "ATR"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ema20'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df['prev_close'] = df.groupby('symbol')['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['prev_close']).abs()
    df['tr3'] = (df['low'] - df['prev_close']).abs()
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df.groupby('symbol')['tr'].transform(lambda x: x.rolling(20).mean())
    df['upper'] = df['ema20'] + 2 * df['atr']
    df['lower'] = df['ema20'] - 2 * df['atr']
    df['kelt_pct'] = (df['close'] - df['lower']) / (df['upper'] - df['lower'] + 1e-8)
    df['factor'] = df.groupby('symbol')['kelt_pct'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="pvo_12_26", display_name="成交量震荡器 PVO(12,26)",
        category=CATEGORY_VOLUME_PRICE_FORMULA,
        description="成交量 EMA(12)/EMA(26) - 1。类似 MACD 但应用于成交量，捕捉成交量趋势变化。",
        direction="positive", quality_score=0.55,
        tags=["PVO", "成交量", "震荡"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ema12'] = df.groupby('symbol')['volume'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    df['ema26'] = df.groupby('symbol')['volume'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['pvo'] = df['ema12'] / (df['ema26'] + 1e-8) - 1
    df['factor'] = df.groupby('symbol')['pvo'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="vrsi_14d", display_name="成交量 RSI (14日)",
        category=CATEGORY_VOLUME_PRICE_FORMULA,
        description="对成交量而非价格计算 RSI(14)。成交量超买意味着过度换手，需警惕。",
        direction="negative", quality_score=0.55,
        tags=["VRSI", "成交量", "超买超卖"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['delta_vol'] = df.groupby('symbol')['volume'].diff()
    df['gain'] = np.where(df['delta_vol'] > 0, df['delta_vol'], 0)
    df['loss'] = np.where(df['delta_vol'] < 0, -df['delta_vol'], 0)
    df['avg_gain'] = df.groupby('symbol')['gain'].transform(lambda x: x.rolling(14).mean())
    df['avg_loss'] = df.groupby('symbol')['loss'].transform(lambda x: x.rolling(14).mean())
    df['rs'] = df['avg_gain'] / (df['avg_loss'] + 1e-8)
    df['vrsi'] = 100 - 100 / (1 + df['rs'])
    df['factor'] = df.groupby('symbol')['vrsi'].shift(1) / 100.0
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="volume_weighted_close_10d", display_name="10日量加权收盘",
        category=CATEGORY_VOLUME_PRICE_FORMULA,
        description="(close*volume).rolling(10).sum() / volume.rolling(10).sum() / prev_close - 1。成交量加权后的价格动量。",
        direction="positive", quality_score=0.60,
        tags=["量加权", "价格", "VWAP"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['pv'] = df['close'] * df['volume']
    df['sum_pv'] = df.groupby('symbol')['pv'].transform(lambda x: x.rolling(10).sum())
    df['sum_v'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(10).sum())
    df['vwc'] = df['sum_pv'] / (df['sum_v'] + 1e-8)
    df['prev_close'] = df.groupby('symbol')['close'].shift(1)
    df['factor'] = df['vwc'] / (df['prev_close'] + 1e-8) - 1
    df['factor'] = df.groupby('symbol')['factor'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="bollinger_pct_20d", display_name="布林带百分比 (20,2)",
        category=CATEGORY_VOLUME_PRICE_FORMULA,
        description="(close - BB_lower) / (BB_upper - BB_lower)。经典布林通道位置，>1.0表示突破上轨强劲。",
        direction="positive", quality_score=0.55,
        tags=["布林带", "通道", "位置"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ma20'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(20).mean())
    df['std20'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(20).std())
    df['bb_upper'] = df['ma20'] + 2 * df['std20']
    df['bb_lower'] = df['ma20'] - 2 * df['std20']
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    df['factor'] = df.groupby('symbol')['bb_pct'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="bollinger_width_20d", display_name="布林带宽 (20日)",
        category=CATEGORY_VOLUME_PRICE_FORMULA,
        description="(BB_upper - BB_lower) / MA20。通道宽度衡量波动率，窄口后往往爆发行情。",
        direction="none", quality_score=0.50,
        tags=["布林带", "宽度", "突破预示"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['ma20'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(20).mean())
    df['std20'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(20).std())
    df['bb_width'] = (4 * df['std20']) / (df['ma20'] + 1e-8)
    df['factor'] = -df.groupby('symbol')['bb_width'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="ad_line_20d", display_name="腾落指标 (20日)",
        category=CATEGORY_VOLUME_PRICE_FORMULA,
        description="((close-low)-(high-close))/(high-low) * volume 的20日均值（上涨/下跌成交量差额）。",
        direction="positive", quality_score=0.60,
        tags=["AD线", "腾落", "资金分布"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['clv'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-8)
    df['ad'] = df['clv'] * df['volume']
    df['ad_ma'] = df.groupby('symbol')['ad'].transform(lambda x: x.rolling(20).mean())
    df['factor'] = df.groupby('symbol')['ad_ma'].shift(1) / (df.groupby('symbol')['volume'].shift(1) + 1e-8)
    return df[['date', 'symbol', 'factor']]
"""
    ),
    FactorDef(
        name="force_index_13d", display_name="强力指数 (13日)",
        category=CATEGORY_VOLUME_PRICE_FORMULA,
        description="EMA(close_diff * volume, 13)。Alexander Elder的强力指数，结合量价变化的趋势强度指标。",
        direction="positive", quality_score=0.60,
        tags=["强力指数", "Elder", "趋势强度"],
        code="""import pandas as pd
import numpy as np
def alpha_factor(df):
    df = df.sort_values(['symbol', 'date']).copy()
    df['close_diff'] = df.groupby('symbol')['close'].diff()
    df['force'] = df['close_diff'] * df['volume']
    df['fi'] = df.groupby('symbol')['force'].transform(lambda x: x.ewm(span=13, adjust=False).mean())
    df['avg_vol'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean())
    df['factor'] = df['fi'] / (df['avg_vol'] + 1e-8)
    df['factor'] = df.groupby('symbol')['factor'].shift(1)
    return df[['date', 'symbol', 'factor']]
"""
    ),
]


# ===================================================================
# 全量因子注册表 & 查询接口
# ===================================================================

_ALL_FACTORS: Dict[str, List[FactorDef]] = {
    CATEGORY_PRICE_TREND: PRICE_TREND_FACTORS,
    CATEGORY_VOLATILITY: VOLATILITY_FACTORS,
    CATEGORY_TRADING_DIFFICULTY: TRADING_DIFFICULTY_FACTORS,
    CATEGORY_PRICE_VOLUME_DIVERGENCE: PV_DIVERGENCE_FACTORS,
    CATEGORY_VOLUME_PRICE_FORMULA: VP_FORMULA_FACTORS,
}


def get_all_factors() -> List[FactorDef]:
    """返回所有传统因子的扁平列表。"""
    flat: List[FactorDef] = []
    for factors in _ALL_FACTORS.values():
        flat.extend(factors)
    return flat


def get_factors_by_category(category: str) -> List[FactorDef]:
    """按大类获取因子。"""
    return _ALL_FACTORS.get(category, [])


def get_factor_by_name(name: str) -> Optional[FactorDef]:
    """按名称查找单个因子。"""
    for f in get_all_factors():
        if f.name == name:
            return f
    return None


def get_factor_names_by_category(category: str) -> List[str]:
    """按大类获取因子名称列表。"""
    return [f.name for f in get_factors_by_category(category)]


def search_factors(
    query: str = "",
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    min_quality: float = 0.0,
    direction: Optional[str] = None,
) -> List[FactorDef]:
    """多条件筛选因子。

    Args:
        query: 在名称/描述中模糊匹配的关键词。
        category: 限制大类。
        tags: 要求包含的标签（AND 逻辑）。
        min_quality: 最低质量分（0-1）。
        direction: 限制方向 'positive'/'negative'/'none'。
    """
    pool = get_factors_by_category(category) if category else get_all_factors()
    results: List[FactorDef] = []
    for f in pool:
        if query and query.lower() not in f.name.lower() and query.lower() not in f.display_name and query.lower() not in f.description.lower():
            continue
        if tags and not all(t in f.tags for t in tags):
            continue
        if f.quality_score < min_quality:
            continue
        if direction and f.direction != direction:
            continue
        results.append(f)
    return results


def get_factor_stats() -> Dict[str, Any]:
    """因子库统计信息。"""
    all_f = get_all_factors()
    directions = {"positive": 0, "negative": 0, "none": 0}
    cats: Dict[str, int] = {}
    for f in all_f:
        directions[f.direction] = directions.get(f.direction, 0) + 1
        cats[f.category] = cats.get(f.category, 0) + 1
    return {
        "total_factors": len(all_f),
        "by_category": {CATEGORY_LABELS.get(k, k): v for k, v in cats.items()},
        "by_direction": directions,
        "categories": len(cats),
        "avg_quality": round(sum(f.quality_score for f in all_f) / len(all_f), 3) if all_f else 0,
    }


def export_factor_batch(factors: List[FactorDef]) -> List[Dict[str, Any]]:
    """将一批因子导出为字典列表（供 UI / Agent 使用）。"""
    return [f.to_dict() for f in factors]


def export_all_to_dict() -> Dict[str, Any]:
    """全量因子库导出为结构化字典（按大类组织）。"""
    return {
        "factors": {
            cat: export_factor_batch(factors) for cat, factors in _ALL_FACTORS.items()
        },
        "stats": get_factor_stats(),
        "categories": {cat: CATEGORY_LABELS[cat] for cat in ALL_CATEGORIES},
    }
