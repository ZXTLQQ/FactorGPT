"""
因子构建与执行引擎（src/engine/factor_builder.py）

职责：
1. 提供一个受限的「代码沙箱」(FactorSandbox)，安全地执行 LLM 生成的因子代码；
2. 将因子输出归一化为统一的长表（date, symbol, factor）；
3. 提供因子后处理流水线：去极值(Winsorize) -> 标准化(Z-score) -> 中性化；
4. 内置常见因子的「关键词模板生成器」，在 LLM 不可用时作为兜底。

因子生成代码的契约（Prompt 中要求 LLM 遵守）：
    def alpha_factor(df: pd.DataFrame) -> pd.Series:
        # df 列：date, symbol, open, high, low, close, volume, amount, pct_chg
        # 返回一个以 (date, symbol) 为索引的 pd.Series，名称为 'factor'
        ...
也可直接返回一个包含 'factor' 列的 pd.DataFrame。
"""

from __future__ import annotations

import builtins
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 受限导入白名单（沙箱仅允许这些科学计算/统计库）
# ----------------------------------------------------------------------
ALLOWED_MODULES = {"pandas", "numpy", "scipy", "math", "statsmodels", "datetime"}


def _safe_import(name, *args, **kwargs):
    root = name.split(".")[0]
    if root not in ALLOWED_MODULES:
        raise ImportError(f"因子沙箱禁止导入模块: {name}（仅允许 {sorted(ALLOWED_MODULES)}）")
    return builtins.__import__(name, *args, **kwargs)


# 受限内置函数集合：仅暴露量化因子计算所需的最小子集
_SAFE_BUILTINS = {
    "__import__": _safe_import,  # 关键：import 语句在 __builtins__ 中查找 __import__
    "range": range, "len": len, "min": min, "max": max, "abs": abs,
    "sum": sum, "float": float, "int": int, "str": str, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set, "zip": zip,
    "map": map, "filter": filter, "sorted": sorted, "enumerate": enumerate,
    "print": print, "round": round, "pow": pow, "divmod": divmod,
    "isinstance": isinstance, "Exception": Exception, "ValueError": ValueError,
    "TypeError": TypeError, "KeyError": KeyError, "IndexError": IndexError,
    "ZeroDivisionError": ZeroDivisionError, "NotImplementedError": NotImplementedError,
    "True": True, "False": False, "None": None,
}


class FactorSandbox:
    """受限执行环境：安全地运行用户/LLM 提供的因子代码。"""

    def __init__(self) -> None:
        self.globals: Dict = {
            "__builtins__": _SAFE_BUILTINS,
            "pd": pd,
            "np": np,
        }

    def run(self, code: str, df: pd.DataFrame) -> pd.Series:
        """在沙箱中执行因子代码并返回因子 Series。

        Args:
            code: 因子计算代码（定义 alpha_factor 函数或直接给出 factor）。
            df: 输入的行情长表，至少包含 date / symbol / close 等列。

        Returns:
            以 (date, symbol) 为索引的因子 Series（名称为 'factor'）。

        Raises:
            ValueError: 代码不可执行、无因子输出或形状不合法。
        """
        local_globals = dict(self.globals)
        local_globals["df"] = df.copy()
        try:
            exec(code, local_globals)  # noqa: S102  受限沙箱，已白名单化导入与内置
        except Exception as e:  # 捕获一切执行异常并转为可读错误
            raise ValueError(f"因子代码执行失败: {type(e).__name__}: {e}") from e

        # 优先使用 alpha_factor 函数
        fn = local_globals.get("alpha_factor")
        if callable(fn):
            try:
                result = fn(df)
            except Exception as e:
                raise ValueError(f"alpha_factor(df) 调用失败: {type(e).__name__}: {e}") from e
        elif "factor" in local_globals and isinstance(local_globals["factor"], (pd.Series, pd.DataFrame)):
            result = local_globals["factor"]
        else:
            raise ValueError("因子代码未定义 alpha_factor() 函数，也未产出 'factor' 变量")

        return self._normalize(result, df)

    @staticmethod
    def _normalize(result, df: pd.DataFrame) -> pd.Series:
        """将因子计算结果对齐为 (date, symbol) 索引的 Series。

        支持两种返回约定：
        1. 返回包含 'date','symbol','factor' 列的 DataFrame（最稳健，行顺序无关）；
        2. 返回已以 (date, symbol) 为 MultiIndex 的 Series；
        3. 返回普通 Series（按位置对齐到输入 df 的 date,symbol）。
        """
        # 约定 1：DataFrame 含 date/symbol 列 —— 以真实日期-标的重建索引（抗乱序）
        if isinstance(result, pd.DataFrame) and {"date", "symbol"}.issubset(result.columns):
            series = pd.Series(
                result["factor"].values,
                index=pd.MultiIndex.from_arrays(
                    [result["date"].values, result["symbol"].values]
                ),
                name="factor",
            )
        # 约定 2：Series 且索引已是 (date,symbol) 二级 MultiIndex
        elif isinstance(result, pd.Series) and isinstance(result.index, pd.MultiIndex):
            series = result.copy()
            series.name = "factor"
        # 约定 3：普通 Series / DataFrame（按位置对齐输入 df）
        else:
            if isinstance(result, pd.DataFrame):
                if "factor" in result.columns:
                    vals = result["factor"].values
                else:
                    num_cols = result.select_dtypes(include=[np.number]).columns
                    if len(num_cols) == 0:
                        raise ValueError("因子 DataFrame 中没有可用的数值列")
                    vals = result[num_cols[0]].values
            else:
                vals = result.values
            series = pd.Series(
                vals,
                index=pd.MultiIndex.from_arrays([df["date"].values, df["symbol"].values]),
                name="factor",
            )

        # 对齐到输入数据的 (date,symbol) 全集
        # 防御：LLM 生成的因子偶发重复 (date,symbol) 行，直接 reindex 会抛
        # "cannot handle a non-unique multi-index"。先按索引去重（保留最后）再对齐。
        if series.index.duplicated().any():
            series = series[~series.index.duplicated(keep="last")]
        target = df[["date", "symbol"]].drop_duplicates()
        target = target.set_index(["date", "symbol"]).index
        series = series.reindex(target)
        series.name = "factor"
        return series


# ----------------------------------------------------------------------
# 因子后处理流水线
# ----------------------------------------------------------------------
def build_pipeline(
    factor: pd.Series,
    winsorize_pct: float = 0.01,
    standardize: bool = True,
    industry: Optional[pd.Series] = None,
    mkt_cap: Optional[pd.Series] = None,
) -> pd.Series:
    """对原始因子值做标准后处理：缩尾 -> 中性化 -> 标准化。

    Args:
        factor: 原始因子 Series，索引 (date, symbol)。
        winsorize_pct: 缩尾分位数（双侧）。
        standardize: 是否做截面 Z-score 标准化。
        industry: 行业分类 Series（索引为 symbol），可选，用于中性化。
        mkt_cap: 市值 Series（索引为 symbol），可选，用于中性化。

    Returns:
        处理后的因子 Series（索引不变）。
    """
    from data.cleaner import DataCleaner

    s = factor.copy()

    # 1) 截面缩尾（按日期）
    s = s.groupby(level="date", group_keys=False).transform(
        lambda x: DataCleaner.winsorize(x, pct=winsorize_pct)
    )

    # 2) 行业/市值中性化（若提供）：逐日对齐到 symbol 索引后再回归取残差
    if industry is not None:
        neutralized = []
        for date, grp in s.groupby(level="date"):
            syms = grp.index.get_level_values("symbol")
            ind = industry.reindex(syms)
            cap = mkt_cap.reindex(syms) if mkt_cap is not None else None
            g = grp.copy()
            g.index = syms  # 改为 symbol 索引以匹配 industry / mkt_cap
            res = DataCleaner.neutralize(g, ind, cap)
            res.index = grp.index  # 还原为 (date, symbol)
            neutralized.append(res)
        if neutralized:
            s = pd.concat(neutralized).sort_index()

    # 3) 截面标准化
    if standardize:
        s = s.groupby(level="date", group_keys=False).transform(DataCleaner.standardize)

    s.name = "factor"
    return s


# ----------------------------------------------------------------------
# 关键词模板生成器（LLM 不可用时的兜底）
# ----------------------------------------------------------------------
TEMPLATE_FACTORS: Dict[str, Dict[str, str]] = {
    "动量": {
        "name": "momentum_20",
        "desc": "过去 20 个交易日收益率（动量因子），剔除最近 1 日避免未来函数",
        "code": (
            "import pandas as pd\n"
            "import numpy as np\n"
            "def alpha_factor(df):\n"
            "    df = df.sort_values(['symbol', 'date']).copy()\n"
            "    df['ret'] = df.groupby('symbol')['close'].pct_change()\n"
            "    df['factor'] = df.groupby('symbol')['ret'].transform(lambda x: x.rolling(20).sum())\n"
            "    df['factor'] = df.groupby('symbol')['factor'].shift(1)\n"
            "    return df[['date', 'symbol', 'factor']]\n"
        ),
    },
    "反转": {
        "name": "reversal_5",
        "desc": "过去 5 个交易日收益率取反（短期反转因子）",
        "code": (
            "import pandas as pd\n"
            "import numpy as np\n"
            "def alpha_factor(df):\n"
            "    df = df.sort_values(['symbol', 'date']).copy()\n"
            "    df['ret'] = df.groupby('symbol')['close'].pct_change()\n"
            "    df['factor'] = df.groupby('symbol')['ret'].transform(lambda x: -x.rolling(5).sum())\n"
            "    df['factor'] = df.groupby('symbol')['factor'].shift(1)\n"
            "    return df[['date', 'symbol', 'factor']]\n"
        ),
    },
    "波动率": {
        "name": "volatility_20",
        "desc": "过去 20 日收益率标准差（波动率因子，越低越好）",
        "code": (
            "import pandas as pd\n"
            "import numpy as np\n"
            "def alpha_factor(df):\n"
            "    df = df.sort_values(['symbol', 'date']).copy()\n"
            "    df['ret'] = df.groupby('symbol')['close'].pct_change()\n"
            "    df['vol'] = df.groupby('symbol')['ret'].transform(lambda x: x.rolling(20).std())\n"
            "    df['factor'] = -df['vol']\n"
            "    df['factor'] = df.groupby('symbol')['factor'].shift(1)\n"
            "    return df[['date', 'symbol', 'factor']]\n"
        ),
    },
    "市值": {
        "name": "size",
        "desc": "总市值对数（规模因子）；行情中无市值时用成交额近似",
        "code": (
            "import pandas as pd\n"
            "import numpy as np\n"
            "def alpha_factor(df):\n"
            "    df = df.sort_values(['symbol', 'date']).copy()\n"
            "    df['size'] = df.groupby('symbol')['amount'].transform(\n"
            "        lambda x: np.log1p(x.rolling(5).mean()))\n"
            "    df['factor'] = -df['size']\n"
            "    df['factor'] = df.groupby('symbol')['factor'].shift(1)\n"
            "    return df[['date', 'symbol', 'factor']]\n"
        ),
    },
    "流动性": {
        "name": "liquidity_turnover",
        "desc": "用成交额代理的流动性因子（成交额越高流动性越好）",
        "code": (
            "import pandas as pd\n"
            "import numpy as np\n"
            "def alpha_factor(df):\n"
            "    df = df.sort_values(['symbol', 'date']).copy()\n"
            "    df['liq'] = df.groupby('symbol')['amount'].transform(\n"
            "        lambda x: np.log1p(x.rolling(20).mean()))\n"
            "    df['factor'] = df['liq']\n"
            "    df['factor'] = df.groupby('symbol')['factor'].shift(1)\n"
            "    return df[['date', 'symbol', 'factor']]\n"
        ),
    },
    "成长": {
        "name": "momentum_growth",
        "desc": "用价格斜率近似的成长/趋势因子（20 日线性回归斜率）",
        "code": (
            "import pandas as pd\n"
            "import numpy as np\n"
            "def alpha_factor(df):\n"
            "    df = df.sort_values(['symbol', 'date']).copy()\n"
            "    def _slope(x):\n"
            "        y = np.log(x.values)\n"
            "        y = y[~np.isnan(y)]\n"
            "        if len(y) < 5:\n"
            "            return np.nan\n"
            "        xx = np.arange(len(y))\n"
            "        return np.polyfit(xx, y, 1)[0]\n"
            "    df['slope'] = df.groupby('symbol')['close'].transform(\n"
            "        lambda x: x.rolling(20).apply(_slope, raw=False))\n"
            "    df['factor'] = df['slope']\n"
            "    df['factor'] = df.groupby('symbol')['factor'].shift(1)\n"
            "    return df[['date', 'symbol', 'factor']]\n"
        ),
    },
}


def generate_from_keywords(description: str) -> Optional[Dict[str, str]]:
    """根据中文描述中的关键词匹配内置因子模板。

    Args:
        description: 用户的因子描述文本。

    Returns:
        命中的模板字典 {name, desc, code}；未命中返回 None。
    """
    for kw, tpl in TEMPLATE_FACTORS.items():
        if kw in description:
            return tpl
    return None
