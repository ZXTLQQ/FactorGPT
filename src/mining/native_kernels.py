"""横截面热核：C++ 优先，pandas 兜底，**语义必须完全一致**。

热点榜显示挖掘层有约 2.2s 的纯 ``DataFrame.rank(axis=1, pct=True)`` 开销（一次
搜索里被调用上千次），外加逐行均值/标准差的 DataFrame 开销。这些操作的共同点是
**沿标的维做一次排序或两遍归约**，而定序聚合本身在 numpy/pandas 里已经是 C 实现——
Python 侧的开销几乎全在框架调度、索引对齐与中间对象上。

所以这里把两个最热的算子下沉到 C++（``native/fg_kernels.cpp``），并保留今天
在用的 pandas 实现作为兜底：没有编译产物时行为与今天**逐位相同**，有编译产物时
只是更快。语义基准始终是 pandas：并列取平均秩、pct 分母是非空计数、NaN 不参与
且保持 NaN、标准差 ddof=1 且恰为 0 时输出 NaN。

加载顺序
--------
1. 已构建的 pybind11 模块 ``fg_native``（最快）；
2. 动态库：``FG_NATIVE_DLL`` 指定，或 ``build/native/fg_kernels.{dll,so,dylib}``
   → ctypes。**本机验证走这条路**：它不依赖 CPython ABI，所以在没有 MSVC 的
   机器上（用 zig/clang 编出来）也能把 C++ 的数值正确性验掉，不必等到 CI；
3. 都没有 → 返回 ``None``，走 pandas。

``FG_DISABLE_NATIVE=1`` 可强制回到 pandas：怀疑原生结果异常时，一行命令即可
二分定位是 C++ 的问题还是别的问题。
"""
from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

_PTR = ctypes.POINTER(ctypes.c_double)
_SIG = [_PTR, _PTR, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]


# ------------------------------------------------------------------ pandas 兜底
def pandas_rank_pct(a: np.ndarray) -> np.ndarray:
    """pandas 基准实现（今天的线上行为，语义的唯一权威）。"""
    return pd.DataFrame(a).rank(axis=1, pct=True).to_numpy(dtype=np.float64)


def pandas_zscore(a: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(a)
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=1)
    return df.sub(mu, axis=0).div(
        sd.replace(0.0, np.nan), axis=0).to_numpy(dtype=np.float64)


def pandas_corr(a: np.ndarray, b: np.ndarray, min_stocks: int = 10) -> np.ndarray:
    """``ops.cs_corr`` 的基准副本（保持独立，避免把"被测实现"拿来当基准）。"""
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
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
    return out


# ------------------------------------------------------------------ 原生加载
class _Native:
    """包住一种原生调用方式（pybind11 模块或 ctypes 动态库）。"""

    def __init__(self, kind: str, rank_fn, zscore_fn, corr_fn=None) -> None:
        self.kind = kind
        self._rank = rank_fn
        self._zscore = zscore_fn
        self._corr = corr_fn

    @property
    def has_corr(self) -> bool:
        """相关核是后加的：旧动态库里没有它，必须显式检查而不是假设存在。"""
        return self._corr is not None

    def corr(self, a: np.ndarray, b: np.ndarray, min_stocks: int = 10) -> np.ndarray:
        out = np.empty(a.shape[0], dtype=np.float64)
        self._corr(a, b, out, min_stocks)
        return out

    def rank_pct(self, a: np.ndarray) -> np.ndarray:
        out = np.empty(a.shape, dtype=np.float64)
        self._rank(a, out)
        return out

    def zscore(self, a: np.ndarray) -> np.ndarray:
        out = np.empty(a.shape, dtype=np.float64)
        self._zscore(a, out)
        return out


def _from_pybind() -> Optional[_Native]:
    try:
        import fg_native as ext  # type: ignore
    except Exception:  # noqa: BLE001 - 没构建就是没有，不是错误
        return None

    def wrap(name: str, min_count: int):
        fn = getattr(ext, name, None)
        if fn is None:
            raise AttributeError(name)

        def _call(a: np.ndarray, out: np.ndarray) -> None:
            out[...] = fn(a, min_count)

        return _call

    fn_corr = getattr(ext, "cs_corr", None)

    def _corr(a: np.ndarray, b: np.ndarray, out: np.ndarray, min_stocks: int) -> None:
        out[...] = fn_corr(a, b, min_stocks)

    try:
        return _Native("pybind11", wrap("cs_rank_pct", 1), wrap("cs_zscore", 2),
                       _corr if fn_corr is not None else None)
    except Exception as e:  # noqa: BLE001
        _log.debug("pybind11 模块可用但接口不符：%s", e)
        return None


def _dll_candidates() -> list[str]:
    env = str(os.environ.get("FG_NATIVE_DLL") or "").strip()
    cands = [env] if env else []
    d = Path(__file__).resolve().parents[2] / "build" / "native"
    if d.is_dir():
        cands += [str(p) for p in sorted(d.glob("fg_kernels*"))
                  if p.suffix.lower() in {".dll", ".so", ".dylib"}]
    return [c for c in cands if c]


def _from_ctypes() -> Optional[_Native]:
    for path in _dll_candidates():
        try:
            lib = ctypes.CDLL(str(path))
        except OSError:
            continue
        rank = lib.fg_cs_rank_pct
        rank.restype, rank.argtypes = None, _SIG
        zsc = lib.fg_cs_zscore
        zsc.restype, zsc.argtypes = None, _SIG

        def _mk(fn, min_count: int):
            def _call(a: np.ndarray, out: np.ndarray) -> None:
                arr = np.ascontiguousarray(a, dtype=np.float64)
                fn(arr.ctypes.data_as(_PTR),
                   out.ctypes.data_as(_PTR),
                   arr.shape[0], arr.shape[1], min_count)
            return _call

        corr_fn = getattr(lib, "fg_cs_corr", None)   # 旧动态库没有这个符号
        corr: Optional[object] = None
        if corr_fn is not None:
            corr_fn.restype = None
            # 注意：相关核的签名是 (a, b, out, rows, cols, min_stocks)，比另外两个
            # 多一个指针参数，不能复用 _SIG
            corr_fn.argtypes = [_PTR, _PTR, _PTR,
                                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]

            def _corr(a: np.ndarray, b: np.ndarray, out: np.ndarray,
                      min_stocks: int) -> None:
                x = np.ascontiguousarray(a, dtype=np.float64)
                y = np.ascontiguousarray(b, dtype=np.float64)
                corr_fn(x.ctypes.data_as(_PTR), y.ctypes.data_as(_PTR),
                        out.ctypes.data_as(_PTR),
                        x.shape[0], x.shape[1], int(min_stocks))

            corr = _corr
        return _Native(f"ctypes:{Path(path).name}", _mk(rank, 1), _mk(zsc, 2), corr)
    return None


_CACHE: dict[str, Optional[_Native]] = {}


def native() -> Optional[_Native]:
    """返回可用的原生实现；没有就 ``None``（调用方必须能接受）。"""
    if "obj" in _CACHE:
        return _CACHE["obj"]
    if str(os.environ.get("FG_DISABLE_NATIVE") or "").strip() in {"1", "true"}:
        obj = None
    else:
        obj = _from_pybind() or _from_ctypes()
    _CACHE["obj"] = obj
    return obj


def backend() -> str:
    """当前生效的后端名，便于回答"为什么没加速"。"""
    n = native()
    return n.kind if n else "pandas"


def reset_cache() -> None:
    """测试用：清掉已缓存的动态库句柄（换 DLL 或改环境变量后需要）。"""
    _CACHE.pop("obj", None)


# ------------------------------------------------------------------ 对外 API
def rank_pct(a: np.ndarray) -> np.ndarray:
    """逐行百分位排名；有原生走原生，否则走 pandas（语义一致）。"""
    n = native()
    if n is None:
        return pandas_rank_pct(a)
    return n.rank_pct(np.ascontiguousarray(a, dtype=np.float64))


def zscore(a: np.ndarray) -> np.ndarray:
    """逐行标准化；有原生走原生，否则走 pandas（语义一致）。"""
    n = native()
    if n is None:
        return pandas_zscore(a)
    return n.zscore(np.ascontiguousarray(a, dtype=np.float64))


def corr(a: np.ndarray, b: np.ndarray, min_stocks: int = 10) -> np.ndarray:
    """逐行相关系数；原生缺失该符号时退回 numpy（语义一致）。"""
    n = native()
    if n is None or not n.has_corr:
        return pandas_corr(a, b, min_stocks)
    return n.corr(np.ascontiguousarray(a, dtype=np.float64),
                  np.ascontiguousarray(b, dtype=np.float64), min_stocks)
