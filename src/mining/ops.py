"""统一算子库：六类算子 + Numba 加速 + 注册表。

设计来源（四篇卖方研报的工程化落地）：

- 山西证券《因子挖掘框架：基于算子网格搜索、Numba 加速的多维度评价体系》
  —— 滚动窗口算子必须索引 + 循环实现，用 Numba(nopython) 编译，禁止 pandas
  在挖掘内循环里被反复调用；本模块把全部滚动算子收敛为**两个内核**
  （一元 ``_roll_unary_impl`` / 二元 ``_roll_binary_impl``），同一份实现既作
  Numba 编译目标、又作纯 NumPy 回退（无 numba 环境行为完全一致，便于 CI）。
- 中信建投《"逐鹿"Alpha 专题报告(三十)：量价 X 基本面因子挖掘统一框架》
  —— 算子分六类（基础运算/横截面/时序/时序二元/衍生财务/中性化），每个算子
  带**维度与语义标签**，供表达式树做静态类型检查，使量价与基本面因子共处
  同一因子空间。
- 西部证券《因子手工作坊系列(7)：概念数量因子》—— 概念类因子本质是
  "计数 + 加权聚合"，复用 ``ts_*`` 算子族即可表达，无需另立运行时。

数据约定：所有面板均为 ``pandas.DataFrame``，**行=交易日、列=标的**，
时间轴向下递增；滚动算子取窗口内**最新值在最后一行**。
NaN 一律视为缺失（滚动窗口内跳过，不参与统计）。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # numba 为可选依赖：缺失时自动选择纯 NumPy 回退（结果一致，只是更慢）
    from numba import njit as _njit

    HAS_NUMBA = True
except Exception:  # pragma: no cover - 取决于运行环境
    _njit = None
    HAS_NUMBA = False

_USE_JIT = True

# ── 一元滚动内核的 kind 编码（Numba 需要整数常量，不能用字符串/枚举） ──
KIND_MEAN = 0
KIND_STD = 1
KIND_SUM = 2
KIND_MIN = 3
KIND_MAX = 4
KIND_RANK = 5
KIND_ARGMAX = 6
KIND_ARGMIN = 7
KIND_SKEW = 8
KIND_MEDIAN = 9
KIND_COUNT = 10
KIND_DECAY = 11
KIND_IR = 12
KIND_PRODUCT = 13

# ── 二元滚动内核的 kind 编码 ──
KIND_B_CORR = 0
KIND_B_COV = 1
KIND_B_BETA = 2
KIND_B_RSQ = 3
KIND_B_RESI = 4
KIND_B_INTERCEPT = 5

_UNARY_KIND = {
    "mean": KIND_MEAN, "std": KIND_STD, "sum": KIND_SUM, "min": KIND_MIN,
    "max": KIND_MAX, "rank": KIND_RANK, "argmax": KIND_ARGMAX,
    "argmin": KIND_ARGMIN, "skew": KIND_SKEW, "median": KIND_MEDIAN,
    "count": KIND_COUNT, "decay": KIND_DECAY, "ir": KIND_IR,
    "product": KIND_PRODUCT,
}
_BINARY_KIND = {
    "corr": KIND_B_CORR, "cov": KIND_B_COV, "beta": KIND_B_BETA,
    "rsq": KIND_B_RSQ, "resi": KIND_B_RESI, "intercept": KIND_B_INTERCEPT,
}


def numba_available() -> bool:
    """当前是否会走 JIT 路径（编译器存在且未被显式关闭）。"""
    return bool(HAS_NUMBA and _USE_JIT)


def disable_numba() -> None:
    """关闭 JIT（对照实验 / 排查用）；纯 NumPy 回退结果与 JIT 完全一致。"""
    global _USE_JIT
    _USE_JIT = False


def enable_numba() -> None:
    global _USE_JIT
    _USE_JIT = HAS_NUMBA


# --------------------------------------------------------------------------
# 内核实现（同一份代码 = Numba 编译目标 + 纯 NumPy 回退）
# --------------------------------------------------------------------------
def _roll_unary_impl(x: np.ndarray, w: int, kind: int, minp: int) -> np.ndarray:
    """一元滚动内核：窗口内跳过 NaN，按 kind 聚合到当前行。

    w        : 窗口长度（含当前行）
    minp     : 窗口内最少有效样本数，不足则该点为 NaN
    返回     : 与 x 同形状的 float64 数组
    """
    t_len, n_col = x.shape
    out = np.full((t_len, n_col), np.nan, dtype=np.float64)
    buf = np.empty(w, dtype=np.float64)
    for j in range(n_col):
        for i in range(t_len):
            m = 0
            for k in range(w):
                ii = i - k
                if ii < 0:
                    break
                v = x[ii, j]
                if not math.isnan(v):
                    buf[m] = v
                    m += 1
            if m < minp:
                continue
            if m == 0:
                continue
            if kind == KIND_MEAN or kind == KIND_SUM or kind == KIND_IR:
                s = 0.0
                for k in range(m):
                    s += buf[k]
                mu = s / m
                if kind == KIND_SUM:
                    out[i, j] = s
                elif kind == KIND_MEAN:
                    out[i, j] = mu
                else:  # IR = 均值 / 标准差（信息比率式时序标准化）
                    if m > 1:
                        v2 = 0.0
                        for k in range(m):
                            d = buf[k] - mu
                            v2 += d * d
                        sd = math.sqrt(v2 / (m - 1))
                        if sd > 0.0:
                            out[i, j] = mu / sd
            elif kind == KIND_STD:
                if m > 1:
                    s = 0.0
                    for k in range(m):
                        s += buf[k]
                    mu = s / m
                    v2 = 0.0
                    for k in range(m):
                        d = buf[k] - mu
                        v2 += d * d
                    out[i, j] = math.sqrt(v2 / (m - 1))
            elif kind == KIND_MIN or kind == KIND_MAX:
                best = buf[0]
                for k in range(1, m):
                    if kind == KIND_MIN:
                        if buf[k] < best:
                            best = buf[k]
                    else:
                        if buf[k] > best:
                            best = buf[k]
                out[i, j] = best
            elif kind == KIND_COUNT:
                out[i, j] = float(m)
            elif kind == KIND_PRODUCT:
                p = 1.0
                for k in range(m):
                    p *= buf[k]
                out[i, j] = p
            elif kind == KIND_RANK:
                # 当前值（窗口最新一行的原值）在窗口内的分位（0~1 含端点）
                cur = x[i, j]
                if not math.isnan(cur):
                    cnt = 0
                    for k in range(m):
                        if buf[k] <= cur:
                            cnt += 1
                    out[i, j] = cnt / m
            elif kind == KIND_MEDIAN:
                tmp = np.empty(m, dtype=np.float64)
                for k in range(m):
                    tmp[k] = buf[k]
                tmp.sort()
                if m % 2 == 1:
                    out[i, j] = tmp[m // 2]
                else:
                    out[i, j] = 0.5 * (tmp[m // 2 - 1] + tmp[m // 2])
            elif kind == KIND_SKEW:
                # 无偏（Fisher）偏度，与 pandas rolling().skew() 一致，需 m>=3
                if m > 2:
                    s = 0.0
                    for k in range(m):
                        s += buf[k]
                    mu = s / m
                    v2 = 0.0
                    v3 = 0.0
                    for k in range(m):
                        d = buf[k] - mu
                        v2 += d * d
                        v3 += d * d * d
                    sd = math.sqrt(v2 / (m - 1))
                    if sd > 0.0:
                        g1 = (v3 / m) / (sd ** 3)
                        # 无偏（Fisher-Pearson 校正）：m²/((m-1)(m-2))
                        out[i, j] = g1 * (m * m) / ((m - 1) * (m - 2))
            elif kind == KIND_ARGMAX or kind == KIND_ARGMIN:
                best = 0.0
                best_age = 0
                found = False
                for k in range(w):
                    ii = i - k
                    if ii < 0:
                        break
                    v = x[ii, j]
                    if math.isnan(v):
                        continue
                    if not found:
                        best = v
                        best_age = k
                        found = True
                    elif kind == KIND_ARGMAX:
                        if v > best:
                            best = v
                            best_age = k
                    else:
                        if v < best:
                            best = v
                            best_age = k
                if found:
                    out[i, j] = float(best_age)
            elif kind == KIND_DECAY:
                # 线性衰减加权均值（越近权重越大）：权重 ∝ (w - age)
                s = 0.0
                sw = 0.0
                for k in range(w):
                    ii = i - k
                    if ii < 0:
                        break
                    v = x[ii, j]
                    if math.isnan(v):
                        continue
                    wgt = float(w - k)
                    s += v * wgt
                    sw += wgt
                if sw > 0.0:
                    out[i, j] = s / sw
    return out


def _roll_binary_impl(x: np.ndarray, y: np.ndarray, w: int, kind: int,
                      minp: int) -> np.ndarray:
    """二元滚动内核：成对跳过 NaN，计算 x/y 的滚动统计量。"""
    t_len, n_col = x.shape
    out = np.full((t_len, n_col), np.nan, dtype=np.float64)
    for j in range(n_col):
        for i in range(t_len):
            m = 0
            sx = 0.0
            sy = 0.0
            sxx = 0.0
            syy = 0.0
            sxy = 0.0
            for k in range(w):
                ii = i - k
                if ii < 0:
                    break
                a = x[ii, j]
                b = y[ii, j]
                if math.isnan(a) or math.isnan(b):
                    continue
                m += 1
                sx += a
                sy += b
                sxx += a * a
                syy += b * b
                sxy += a * b
            if m < minp or m < 2:
                continue
            mx = sx / m
            my = sy / m
            vxx = sxx - m * mx * mx
            vyy = syy - m * my * my
            vxy = sxy - m * mx * my
            if kind == KIND_B_COV:
                out[i, j] = vxy / (m - 1)
                continue
            if vxx <= 1e-300 or vyy <= 1e-300:
                if kind == KIND_B_RESI and vyy > 0.0:
                    out[i, j] = y[i, j] - my
                continue
            if kind == KIND_B_CORR:
                c = vxy / math.sqrt(vxx * vyy)
                out[i, j] = max(-1.0, min(1.0, c))
            elif kind == KIND_B_BETA:
                out[i, j] = vxy / vxx
            elif kind == KIND_B_RSQ:
                c = vxy / math.sqrt(vxx * vyy)
                out[i, j] = max(0.0, min(1.0, c * c))
            elif kind == KIND_B_RESI or kind == KIND_B_INTERCEPT:
                beta = vxy / vxx
                alpha = my - beta * mx
                if kind == KIND_B_RESI:
                    out[i, j] = y[i, j] - (alpha + beta * x[i, j])
                else:
                    out[i, j] = alpha
    return out


_ROLL_UNARY_JIT = _njit(cache=True, nogil=True)(_roll_unary_impl) if HAS_NUMBA else None
_ROLL_BINARY_JIT = _njit(cache=True, nogil=True)(_roll_binary_impl) if HAS_NUMBA else None


def _default_minp(w: int) -> int:
    """窗口内最少有效样本：默认 ceil(w/2)（与常见滚动算子的宽松口径一致）。"""
    return max(1, math.ceil(w / 2.0))


def roll_unary(x: np.ndarray, w: int, kind: int, minp: Optional[int] = None) -> np.ndarray:
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    w = int(w)
    minp = _default_minp(w) if minp is None else int(minp)
    if _ROLL_UNARY_JIT is not None and _USE_JIT:
        return _ROLL_UNARY_JIT(x, w, int(kind), minp)
    return _roll_unary_impl(x, w, int(kind), minp)


def roll_binary(x: np.ndarray, y: np.ndarray, w: int, kind: int,
                minp: Optional[int] = None) -> np.ndarray:
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    w = int(w)
    minp = _default_minp(w) if minp is None else int(minp)
    if _ROLL_BINARY_JIT is not None and _USE_JIT:
        return _ROLL_BINARY_JIT(x, y, w, int(kind), minp)
    return _roll_binary_impl(x, y, w, int(kind), minp)


# --------------------------------------------------------------------------
# 面板级算子（DataFrame 进出，行=日期、列=标的）
# --------------------------------------------------------------------------
def _as_frame(arr: np.ndarray, like: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(arr, index=like.index, columns=like.columns)


def rolling_unary(df: pd.DataFrame, w: int, kind_name: str,
                  minp: Optional[int] = None) -> pd.DataFrame:
    if kind_name not in _UNARY_KIND:
        raise KeyError(f"未知滚动算子 kind: {kind_name}")
    return _as_frame(roll_unary(df.to_numpy(dtype=np.float64), w,
                                _UNARY_KIND[kind_name], minp), df)


def rolling_binary(dfx: pd.DataFrame, dfy: pd.DataFrame, w: int,
                   kind_name: str, minp: Optional[int] = None) -> pd.DataFrame:
    if kind_name not in _BINARY_KIND:
        raise KeyError(f"未知滚动二元算子 kind: {kind_name}")
    x, y = dfx.align(dfy, join="outer")
    return _as_frame(roll_binary(x.to_numpy(dtype=np.float64),
                                 y.to_numpy(dtype=np.float64), w,
                                 _BINARY_KIND[kind_name], minp), x)


def _time_grid(like: pd.DataFrame) -> pd.DataFrame:
    """时间轴栅格面板（列内取值 0,1,2,...），供 ts_slope / ts_rsquare 复用二元内核。"""
    n = len(like.index)
    return pd.DataFrame(
        np.repeat(np.arange(n, dtype=np.float64).reshape(-1, 1),
                  like.shape[1], axis=1),
        index=like.index, columns=like.columns)


# ---- 横截面算子 -----------------------------------------------------------
def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """横截面分位排名（0~1，忽略 NaN）。

    热点榜里它自己就占掉约 2.2s（一次搜索上千次调用）。有原生动态库时走
    ``native_kernels``（同一套语义，见其单元测试）；没有就退回 pandas，行为
    与今天完全一致——编译产物是**可选加速**，不是依赖。
    """
    from .native_kernels import native as _native
    from .native_kernels import rank_pct

    if _native() is None:
        return df.rank(axis=1, pct=True)
    arr = df.to_numpy(dtype=np.float64, copy=False)
    return pd.DataFrame(rank_pct(arr), index=df.index, columns=df.columns)


def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    from .native_kernels import native as _native
    from .native_kernels import zscore

    if _native() is None:
        return _pandas_cs_zscore(df)
    arr = df.to_numpy(dtype=np.float64, copy=False)
    return pd.DataFrame(zscore(arr), index=df.index, columns=df.columns)


def _pandas_cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """pandas 基准路径（也是原生实现的语义参照）。"""
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=1)
    return df.sub(mu, axis=0).div(sd.replace(0.0, np.nan), axis=0)


def cs_demean(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0)


def cs_scale(df: pd.DataFrame) -> pd.DataFrame:
    """L1 归一：x / Σ|x|（保留符号，控制极端值影响力）。"""
    denom = df.abs().sum(axis=1)
    return df.div(denom.replace(0.0, np.nan), axis=0)


def cs_winsorize(df: pd.DataFrame, n: float = 3.0) -> pd.DataFrame:
    """中位数 ± n×1.4826×MAD 截尾（对非正态更稳健）。"""
    med = df.median(axis=1)
    mad = df.sub(med, axis=0).abs().median(axis=1)
    width = (n * 1.4826 * mad).replace(0.0, np.nan)
    lower = med.sub(width, axis=0)
    upper = med.add(width, axis=0)
    out = df.clip(lower=lower, upper=upper, axis=0)
    return out.where(df.notna())


def cs_quantile(df: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    """横截面分箱，返回 [0,1] 的分箱强度（0=最低箱，1=最高箱）。"""
    r = df.rank(axis=1, pct=True)
    bucket = np.floor(r.to_numpy() * q)
    bucket = np.clip(bucket, 0, q - 1)
    return pd.DataFrame(bucket / (q - 1), index=df.index, columns=df.columns)


def cs_dgtw(df: pd.DataFrame, groups: pd.DataFrame, n_groups: int = 5,
            min_stocks: int = 5, standardize: bool = True) -> pd.DataFrame:
    """DGTW 市值调整：按分组变量（通常取规模暴露）分箱后**组内**去均值/标准化。

    西部证券《概念数量因子》用它剥离"小市值股票天然属于更多概念"这一
    机械相关：概念数量与市值高度相关，不调整就会把规模因子重新挖一遍。
    与 ``neutral(x, size)`` 的线性残差化不同，这里做的是**分组内**调整，
    能吸收非线性关系（小盘股概念数的截断效应）。

    ``n_groups`` 即表达式里的窗口参数：``dgtw_cs(concept_count, size, 5)``。
    """
    n = max(2, int(n_groups))
    g = groups.rank(axis=1, pct=True)
    bucket = np.clip(np.floor(g.to_numpy(dtype=np.float64) * n), 0, n - 1)
    x = df.reindex_like(groups).to_numpy(dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    for b in range(n):
        m = (bucket == b) & np.isfinite(x)
        cnt = m.sum(axis=1)
        good = cnt >= max(2, int(min_stocks) // 2)
        if not good.any():
            continue
        xm = np.where(m, x, 0.0)
        mu = np.where(cnt > 0, xm.sum(axis=1) / np.maximum(cnt, 1), 0.0)
        dev = np.where(m, x - mu[:, None], 0.0)
        if standardize:
            sd = np.sqrt((dev * dev).sum(axis=1) / np.maximum(cnt - 1, 1))
            sd = np.where(sd > 0, sd, np.nan)
            dev = np.where(m, dev / sd[:, None], 0.0)
        out = np.where(m & good[:, None], dev, out)
    return pd.DataFrame(out, index=groups.index, columns=groups.columns)


_CS2: Dict[str, Callable[[pd.DataFrame, pd.DataFrame, int], pd.DataFrame]] = {
    "dgtw_cs": cs_dgtw,
}


def cs_corr(a: pd.DataFrame, b: pd.DataFrame, rank: bool = False,
            min_stocks: int = 10) -> pd.Series:
    """逐行（横截面）相关系数，**向量化**实现。

    不用 ``np.corrcoef`` 逐日循环的原因有二：一是慢（要素 IC 要算上万次），
    二是它在某一行标准差退化为 0 时会抛 RuntimeWarning 并被 std 的 ddof 口径
    （pandas 默认 ddof=1）与我们的空值判断错位，从而漏出告警。这里统一用
    一次矩阵内积算完，退化行直接给 NaN。
    """
    from .native_kernels import corr as _corr_native
    from .native_kernels import native as _native

    n = _native()
    if n is not None and n.has_corr and not rank:
        # 融合内核：一次调用扫两遍内存，没有中间 DataFrame/临时矩阵
        out = _corr_native(a.to_numpy(dtype=np.float64),
                           b.to_numpy(dtype=np.float64), min_stocks)
        return pd.Series(out, index=a.index, name="cs_corr")
    if rank:
        a, b = cs_rank(a), cs_rank(b)
    av = a.to_numpy(dtype=np.float64)
    bv = b.to_numpy(dtype=np.float64)
    m = np.isfinite(av) & np.isfinite(bv)
    n = m.sum(axis=1)
    nn = np.maximum(n, 1)
    a0 = np.where(m, av, 0.0)
    b0 = np.where(m, bv, 0.0)
    da = np.where(m, av - (a0.sum(axis=1) / nn)[:, None], 0.0)
    db = np.where(m, bv - (b0.sum(axis=1) / nn)[:, None], 0.0)
    denom = np.sqrt((da * da).sum(axis=1) * (db * db).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (da * db).sum(axis=1) / denom
    out = np.asarray(out, dtype=np.float64)
    out[~np.isfinite(out)] = np.nan
    out[n < max(int(min_stocks), 3)] = np.nan
    return pd.Series(out, index=a.index, name="cs_corr")


# ---- 基础运算 -------------------------------------------------------------
def _safe_div(a: pd.DataFrame, b: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    a, b = a.align(b, join="outer")
    return a / b.where(b.abs() > eps)


def _align2(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return a.align(b, join="outer")


def _elem(d: pd.DataFrame, fn: Callable[..., np.ndarray]) -> pd.DataFrame:
    """逐元素套用 numpy 一元函数（log/log1p/sqrt/sign/tanh），NaN 安全。"""
    with np.errstate(invalid="ignore", divide="ignore"):
        arr = fn(d.to_numpy(dtype=np.float64))
    return pd.DataFrame(arr, index=d.index, columns=d.columns)


def _binary_np(a: pd.DataFrame, b: pd.DataFrame,
               fn: Callable[..., np.ndarray]) -> pd.DataFrame:
    """对齐后逐元素套用 numpy 二元函数（fmax/fmin/power），NaN 安全。"""
    left, right = _align2(a, b)
    with np.errstate(invalid="ignore", divide="ignore"):
        arr = fn(left.to_numpy(dtype=np.float64), right.to_numpy(dtype=np.float64))
    return pd.DataFrame(arr, index=left.index, columns=left.columns)


# --------------------------------------------------------------------------
# 算子规格与注册表
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class OpSpec:
    """算子元数据（供表达式树做静态类型检查）。

    family    : elem/elem2/cs/ts/ts2/derive/neutral
    arity     : 参数个数（不含窗口参数）
    window    : 是否需要窗口参数（int）
    dim       : 输出维度；None = 继承首个输入
    sem       : 输出语义；None = 继承首个输入
    doc       : 一句话说明
    """

    name: str
    family: str
    arity: int
    window: bool = False
    dim: Optional[str] = None
    sem: Optional[str] = None
    doc: str = ""


def _ts_slope(df: pd.DataFrame, w: int) -> pd.DataFrame:
    return rolling_binary(_time_grid(df), df, w, "beta")


def _ts_rsquare(df: pd.DataFrame, w: int) -> pd.DataFrame:
    return rolling_binary(_time_grid(df), df, w, "rsq")


def _ts_resi(df: pd.DataFrame, w: int) -> pd.DataFrame:
    return rolling_binary(_time_grid(df), df, w, "resi")


def _geom_mean(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    a, b = a.align(b, join="outer")
    return np.sign(a * b) * np.sqrt((a * b).abs())


def _harm_mean(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    a, b = a.align(b, join="outer")
    return _safe_div(2.0 * a * b, a + b)


def _spread(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """相对强弱 a/(a+b)，天然可比且对不同量纲稳健。"""
    a, b = a.align(b, join="outer")
    return _safe_div(a, a + b)


OPS: Dict[str, OpSpec] = {}


def _register(spec: OpSpec) -> OpSpec:
    OPS[spec.name] = spec
    return spec


# 基础运算（维度/语义不变）
for _name, _doc in (
    ("neg", "取负：反转因子方向"),
    ("abs", "绝对值"),
    ("log", "自然对数（要求正值）"),
    ("log1p", "log(1+x)，可处理 0 值"),
    ("sqrt", "平方根（压缩量纲）"),
    ("square", "平方（放大极端）"),
    ("sign", "符号（只保留方向）"),
    ("tanh", "双曲正切（软截尾到 ±1）"),
    ("inv", "倒数（安全除）"),
    ("clip3", "截尾到 ±3（配合 zscore 使用）"),
):
    _register(OpSpec(_name, "elem", 1, doc=_doc))

# 横截面算子（输出统一为无量纲 score）
for _name, _doc in (
    ("rank_cs", "横截面分位排名"),
    ("zscore_cs", "横截面标准化"),
    ("demean_cs", "横截面去均值"),
    ("scale_cs", "横截面 L1 归一"),
    ("winsorize_cs", "横截面 MAD 截尾（剔除极端值）"),
):
    _register(OpSpec(_name, "cs", 1, dim="score", sem=None, doc=_doc))
_register(OpSpec("quantile_cs", "cs", 1, window=True, dim="score",
                 doc="横截面分箱强度（+窗口参数即箱数）"))

# 二元横截面算子：第二个参数是分组/对照变量（Numba 不介入，纯 pandas/numpy）
_register(OpSpec("dgtw_cs", "cs2", 2, window=True, dim="score",
                 doc="按第二参数分组（如市值暴露）后组内标准化，"
                     "「+窗口参数即分组数」；DGTW 调整"))

# 时序一元算子（维度/语义继承）
for _name, _kind, _doc in (
    ("ts_mean", "mean", "滚动均值"),
    ("ts_std", "std", "滚动标准差（波动）"),
    ("ts_sum", "sum", "滚动求和"),
    ("ts_min", "min", "滚动最小值"),
    ("ts_max", "max", "滚动最大值"),
    ("ts_rank", "rank", "当前值在窗口内的分位"),
    ("ts_argmax", "argmax", "窗口最高值距今天数（0=今日）"),
    ("ts_argmin", "argmin", "窗口最低值距今天数"),
    ("ts_skew", "skew", "滚动偏度（尾部风险）"),
    ("ts_median", "median", "滚动中位数（稳健中枢）"),
    ("ts_count", "count", "窗口内有效样本数（数据质量）"),
    ("ts_product", "product", "滚动连乘"),
    ("ts_decay_linear", "decay", "线性衰减加权均值（近期权重更大）"),
    ("ts_ir", "ir", "滚动均值/标准差（时序信息比）"),
):
    _register(OpSpec(_name, "ts", 1, window=True, doc=_doc))

# 时序便捷算子（带窗口）
_register(OpSpec("ts_zscore", "ts", 1, window=True,
                 doc="滚动标准化 (x-mean)/std"))
_register(OpSpec("ts_max_diff", "ts", 1, window=True,
                 doc="x - 滚动最大值（距高点距离）"))
_register(OpSpec("ts_min_diff", "ts", 1, window=True,
                 doc="x - 滚动最小值（距低点距离）"))
_register(OpSpec("ts_slope", "ts", 1, window=True,
                 doc="窗口内 OLS 斜率（趋势强度）"))
_register(OpSpec("ts_rsquare", "ts", 1, window=True, dim="score",
                 doc="窗口内线性拟合 R²（趋势确定性）"))
_register(OpSpec("ts_resi", "ts", 1, window=True,
                 doc="当前值相对窗口趋势线的残差（偏离度）"))
_register(OpSpec("ts_delay", "ts", 1, window=True, doc="滞后 n 期"))
_register(OpSpec("ts_delta", "ts", 1, window=True, doc="x - x[n]（一阶差分）"))
_register(OpSpec("ts_pct", "ts", 1, window=True, dim="ratio",
                 doc="x/x[n] - 1（区间变化率）"))
_register(OpSpec("ts_ret", "ts", 1, window=True, dim="ratio",
                 doc="区间收益（ts_pct 别名）"))
_register(OpSpec("ts_yoy", "ts", 1, window=True, dim="growth", sem="growth",
                 doc="同比变化率（窗口通常取 250 交易日）"))
_register(OpSpec("ts_qoq", "ts", 1, window=True, dim="growth", sem="growth",
                 doc="环比变化率（窗口通常取 60 交易日）"))
_register(OpSpec("ema", "ts", 1, window=True, doc="指数移动平均"))

# 时序二元算子（窗口内两个面板的联合统计）
for _name, _kind, _dim, _sem, _doc in (
    ("ts_corr", "corr", "ratio", None, "滚动相关系数"),
    ("ts_cov", "cov", None, None, "滚动协方差"),
    ("ts_beta", "beta", "ratio", None, "滚动 beta（y 对 x 回归斜率）"),
    ("ts_reg_rsq", "rsq", "score", None, "滚动回归 R²"),
    ("ts_reg_resi", "resi", None, None, "滚动回归残差（去线性暴露）"),
):
    _register(OpSpec(_name, "ts2", 2, window=True, dim=_dim, sem=_sem, doc=_doc))

# 二元基础运算
for _name, _dim, _sem, _doc in (
    ("add", None, None, "逐元素相加（同维度方可相加）"),
    ("sub", None, None, "逐元素相减"),
    ("mul", "score", None, "逐元素相乘（信号组合）"),
    ("div", "ratio", None, "逐元素相除（结果无量纲）"),
    ("safe_div", "ratio", None, "安全除（分母接近 0 返回 NaN）"),
    ("max2", None, None, "逐元素取大"),
    ("min2", None, None, "逐元素取小"),
    ("pow2", None, None, "a 的 b 次方"),
    ("geom_mean", "score", None, "几何平均（双信号需同时高）"),
    ("harm_mean", "ratio", None, "调和平均（对低值更敏感）"),
    ("spread", "ratio", None, "相对强弱 a/(a+b)"),
):
    _register(OpSpec(_name, "elem2", 2, dim=_dim, sem=_sem, doc=_doc))


def get_op(name: str) -> OpSpec:
    if name not in OPS:
        raise KeyError(f"未注册的算子: {name}（可用: {sorted(OPS)[:12]} ...）")
    return OPS[name]


def call_op(name: str, args: List[pd.DataFrame],
            window: Optional[int] = None) -> pd.DataFrame:
    """按算子名执行（表达式树求值入口）。"""
    spec = get_op(name)
    if len(args) != spec.arity:
        raise ValueError(f"{name} 需要 {spec.arity} 个参数，收到 {len(args)}")
    w = int(window) if window is not None else None
    if spec.family == "elem":
        return _ELEM[name](args[0])
    if spec.family == "elem2":
        return _ELEM2[name](args[0], args[1])
    if spec.family == "cs":
        return cs_quantile(args[0], w) if name == "quantile_cs" else _CS[name](args[0])
    if spec.family == "cs2":
        return _CS2[name](args[0], args[1], w or 5)
    if spec.family == "ts":
        return _TS[name](args[0], w)
    if spec.family == "ts2":
        return _TS2[name](args[0], args[1], w)
    raise KeyError(f"算子族未实现: {spec.family}")


_ELEM: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "neg": lambda d: -d,
    "abs": lambda d: d.abs(),
    "log": lambda d: _elem(d, np.log),
    "log1p": lambda d: _elem(d, np.log1p),
    "sqrt": lambda d: _elem(d, np.sqrt),
    "square": lambda d: d * d,
    "sign": lambda d: _elem(d, np.sign),
    "tanh": lambda d: _elem(d, np.tanh),
    "inv": lambda d: _safe_div(
        pd.DataFrame(1.0, index=d.index, columns=d.columns), d),
    "clip3": lambda d: d.clip(-3.0, 3.0),
}

_ELEM2: Dict[str, Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]] = {
    "add": lambda a, b: _align2(a, b)[0].add(_align2(a, b)[1]),
    "sub": lambda a, b: _align2(a, b)[0].sub(_align2(a, b)[1]),
    "mul": lambda a, b: _align2(a, b)[0].mul(_align2(a, b)[1]),
    "div": lambda a, b: _safe_div(a, b),
    "safe_div": lambda a, b: _safe_div(a, b),
    "max2": lambda a, b: _binary_np(a, b, np.fmax),
    "min2": lambda a, b: _binary_np(a, b, np.fmin),
    "pow2": lambda a, b: _binary_np(a, b, np.power),
    "geom_mean": _geom_mean,
    "harm_mean": _harm_mean,
    "spread": _spread,
}

_CS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "rank_cs": cs_rank,
    "zscore_cs": cs_zscore,
    "demean_cs": cs_demean,
    "scale_cs": cs_scale,
    "winsorize_cs": cs_winsorize,
}

_TS: Dict[str, Callable[[pd.DataFrame, int], pd.DataFrame]] = {
    "ts_mean": lambda d, w: rolling_unary(d, w, "mean"),
    "ts_std": lambda d, w: rolling_unary(d, w, "std"),
    "ts_sum": lambda d, w: rolling_unary(d, w, "sum"),
    "ts_min": lambda d, w: rolling_unary(d, w, "min"),
    "ts_max": lambda d, w: rolling_unary(d, w, "max"),
    "ts_rank": lambda d, w: rolling_unary(d, w, "rank"),
    "ts_argmax": lambda d, w: rolling_unary(d, w, "argmax"),
    "ts_argmin": lambda d, w: rolling_unary(d, w, "argmin"),
    "ts_skew": lambda d, w: rolling_unary(d, w, "skew"),
    "ts_median": lambda d, w: rolling_unary(d, w, "median"),
    "ts_count": lambda d, w: rolling_unary(d, w, "count"),
    "ts_product": lambda d, w: rolling_unary(d, w, "product"),
    "ts_decay_linear": lambda d, w: rolling_unary(d, w, "decay"),
    "ts_ir": lambda d, w: rolling_unary(d, w, "ir"),
    "ts_slope": _ts_slope,
    "ts_rsquare": _ts_rsquare,
    "ts_resi": _ts_resi,
    "ts_delay": lambda d, w: d.shift(w),
    "ts_delta": lambda d, w: d - d.shift(w),
    "ts_pct": lambda d, w: _safe_div(d, d.shift(w)) - 1.0,
    "ts_ret": lambda d, w: _safe_div(d, d.shift(w)) - 1.0,
    "ts_yoy": lambda d, w: _safe_div(d, d.shift(w)) - 1.0,
    "ts_qoq": lambda d, w: _safe_div(d, d.shift(w)) - 1.0,
    "ts_zscore": lambda d, w: _safe_div(
        d - rolling_unary(d, w, "mean"), rolling_unary(d, w, "std")),
    "ts_max_diff": lambda d, w: d - rolling_unary(d, w, "max"),
    "ts_min_diff": lambda d, w: d - rolling_unary(d, w, "min"),
    "ema": lambda d, w: d.ewm(span=int(w), adjust=False).mean(),
}

_TS2: Dict[str, Callable[[pd.DataFrame, pd.DataFrame, int], pd.DataFrame]] = {
    "ts_corr": lambda a, b, w: rolling_binary(a, b, w, "corr"),
    "ts_cov": lambda a, b, w: rolling_binary(a, b, w, "cov"),
    "ts_beta": lambda a, b, w: rolling_binary(a, b, w, "beta"),
    "ts_reg_rsq": lambda a, b, w: rolling_binary(a, b, w, "rsq"),
    "ts_reg_resi": lambda a, b, w: rolling_binary(a, b, w, "resi"),
}


# --------------------------------------------------------------------------
# Numba 加速对照实验（山西证券的口径：同一算子的 NumPy 回退 vs JIT）
# --------------------------------------------------------------------------
def benchmark(rows: int = 1200, cols: int = 300, windows: Sequence[int] = (5, 20, 60),
              kinds: Sequence[str] = ("mean", "std", "rank", "skew"),
              binary_kinds: Sequence[str] = ("corr", "beta", "resi"),
              seed: int = 42, repeats: int = 2) -> List[Dict[str, Any]]:
    """对比朴素 NumPy 内核与 Numba JIT 内核的耗时（加速比）。

    返回可直接落表的字典列表；无 numba 时 ``speedup`` 为 None。
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, cols))
    x[rng.random((rows, cols)) < 0.02] = np.nan
    y = x * 0.6 + rng.standard_normal((rows, cols)) * 0.8

    def _time(fn: Callable[[], Any]) -> float:
        best = float("inf")
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best * 1000.0

    out: List[Dict[str, Any]] = []
    for w in windows:
        for k in kinds:
            kind = _UNARY_KIND[k]
            py_ms = _time(lambda: _roll_unary_impl(x, w, kind, _default_minp(w)))
            if _ROLL_UNARY_JIT is not None:
                _ROLL_UNARY_JIT(x[:50, :10], w, kind, _default_minp(w))  # 预热编译
                nb_ms = _time(lambda: _ROLL_UNARY_JIT(x, w, kind, _default_minp(w)))
            else:
                nb_ms = None
            out.append({"op": f"ts_{k}", "window": w, "rows": rows, "cols": cols,
                        "numpy_ms": round(py_ms, 2),
                        "numba_ms": None if nb_ms is None else round(nb_ms, 2),
                        "speedup": None if not nb_ms else round(py_ms / nb_ms, 1)})
        for k in binary_kinds:
            kind = _BINARY_KIND[k]
            py_ms = _time(lambda: _roll_binary_impl(x, y, w, kind, _default_minp(w)))
            if _ROLL_BINARY_JIT is not None:
                _ROLL_BINARY_JIT(x[:50, :10], y[:50, :10], w, kind, _default_minp(w))
                nb_ms = _time(lambda: _ROLL_BINARY_JIT(x, y, w, kind, _default_minp(w)))
            else:
                nb_ms = None
            out.append({"op": f"ts_{k}", "window": w, "rows": rows, "cols": cols,
                        "numpy_ms": round(py_ms, 2),
                        "numba_ms": None if nb_ms is None else round(nb_ms, 2),
                        "speedup": None if not nb_ms else round(py_ms / nb_ms, 1)})
    return out


# --------------------------------------------------------------------------
# 算子便捷入口：ops.<算子名>(...) ≡ call_op("<算子名>", ...)
# 例：ops.ts_pct(close, 60) / ops.ts_corr(a, b, window=20) / ops.add(a, b)
# 与内置名冲突的算子（abs）不生成顶层别名，避免覆盖 Python 语义。
# --------------------------------------------------------------------------
_BUILTIN_SHADOW = {"abs", "min", "max", "sum", "pow", "all", "any", "round"}


def _make_forwarder(name: str, spec: OpSpec) -> Callable[..., pd.DataFrame]:
    n_args = spec.arity

    def _forward(*args: Any, window: Optional[int] = None,
                 **kwargs: Any) -> pd.DataFrame:
        w = kwargs.pop("w", None)
        if w is not None:
            window = w
        if kwargs:
            raise TypeError(f"{name} 不接受参数 {sorted(kwargs)}")
        args = list(args)
        # 允许把窗口写成最后一个位置参数：ts_pct(close, 60)
        if spec.window and window is None and len(args) == n_args + 1:
            window = int(args.pop())
        if len(args) != n_args:
            raise TypeError(f"{name} 需要 {n_args} 个面板参数，收到 {len(args)}")
        return call_op(name, args, window)

    _forward.__name__ = name
    _forward.__qualname__ = name
    _forward.__doc__ = spec.doc
    return _forward


def _install_forwarders() -> None:
    for _name, _spec in OPS.items():
        if _name in _BUILTIN_SHADOW or not _name.isidentifier():
            continue
        if _name not in globals():
            globals()[_name] = _make_forwarder(_name, _spec)


_install_forwarders()

