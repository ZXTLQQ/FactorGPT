"""逐日横截面 OLS 的**批量**实现（矩阵版本 + 逐日参考版本共存）。

为什么单独抽一个模块
--------------------
``scripts/profile_mining.py`` 的热点榜显示，挖掘层的墙钟主要花在两处
"逐日 Python 循环 + 每天一次 ``lstsq``/``pinv``" 上：

* ``panel.residualize_cs``（表达式 ``neutral(...)`` 与网格搜索的增量去冗余都走它）
* ``risk.cross_sectional_ols``（ΔR²、t 值、因子收益、风险贡献都建在它上面）

这两个循环每天做一次 ``np.column_stack`` + ``np.linalg.lstsq``：按天数是 Python
解释器开销，按天又有一次 numpy 调用开销，而每次求解的矩阵只有 ``k ≤ 4`` 列。
"很多次极小的线性代数" 恰好可以把**天数维批量化**——每天的正规方程
``X'X β = X'y`` 中，``X'X`` 的每个元素都只是"两个针列逐元素相乘后沿标的求和"，
用 ``einsum`` 一次算完**所有交易日**，再做一次 ``np.linalg.solve`` 的**堆叠**求解。

语义的唯一权威是 :func:`fit_cs_rowwise`（朴素逐日循环，与历史行为对齐），
批量版本 :func:`fit_cs` 必须与它对拍一致，将来的 C++ 版本也是如此。因此参考
实现不是"没人调用的遗留代码"，它是**测试基准**。

退化处理：``X'X`` 奇异（某列共线、或 happy 有效样本不足）时，堆叠 ``solve`` 会
静默给出巨大却看似合理的 β——这是最危险的一类错。所以奇异的日子一律**退回逐日
``lstsq``**，并在结果 ``ok`` 里标记。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

__all__ = ["CSResult", "fit_cs", "fit_cs_rowwise"]


@dataclass
class CSResult:
    """逐日横截面回归结果（全部按交易日序号 ``0..T-1`` 对齐）。"""

    beta: np.ndarray        # (T, k) 系数；未解出的日子为 NaN
    resid: np.ndarray       # (T, N) 残差；未参与回归的位置为 NaN
    n_valid: np.ndarray     # (T,) 每天有效样本数
    xtx_inv: np.ndarray     # (T, k, k) (X'X)^-1；未解出的日子为 NaN
    ss_res: np.ndarray      # (T,) 残差平方和
    r2: np.ndarray          # (T,) 每日 R²；ss_tot=0 的日子为 NaN
    ok: np.ndarray          # (T,) 这一天是否真的解出来了


def _stacked_columns(cols: Sequence[np.ndarray]) -> np.ndarray:
    """把 ``k`` 个 (T,N) 针列堆成逐日可用的列表（此处仅做 float64 规整）。"""
    return [np.ascontiguousarray(np.asarray(c, dtype=np.float64)) for c in cols]


def fit_cs_rowwise(y: np.ndarray, xs: Sequence[np.ndarray], add_const: bool = True,
                   min_stocks: int = 20) -> CSResult:
    """朴素逐日实现（``lstsq``）——**语义基准**：慢，但定义就是它。

    任何加速版本都必须能在数值上与它对齐；这里用 ``lstsq`` 而非 SVD 截断，
    因为历史结果由它定义，改成 pinv 会移动共线日的系数。
    """
    y_arr = np.asarray(y, dtype=np.float64)
    cols = _stacked_columns(xs)
    t, n = y_arr.shape
    k = len(cols) + (1 if add_const else 0)
    need = max(int(min_stocks), k + 2)

    beta = np.full((t, k), np.nan)
    resid = np.full((t, n), np.nan)
    xtx_inv = np.full((t, k, k), np.nan)
    ss_res = np.full(t, np.nan)
    r2 = np.full(t, np.nan)
    ok = np.zeros(t, dtype=bool)
    n_valid = np.zeros(t, dtype=np.int64)

    for i in range(t):
        yy = y_arr[i]
        parts = ([np.ones_like(yy)] if add_const else []) + [c[i] for c in cols]
        X = np.column_stack(parts)
        mask = np.isfinite(yy) & np.all(np.isfinite(X), axis=1)
        n_valid[i] = int(mask.sum())
        if n_valid[i] < need:
            continue
        Xm, ym = X[mask], yy[mask]
        try:
            b, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover - 极端退化
            continue
        if not np.all(np.isfinite(b)):  # pragma: no cover
            continue
        err = ym - Xm @ b
        row = np.full(n, np.nan)
        row[mask] = err
        resid[i] = row
        ssr = float(err @ err)
        ss_tot = float(((ym - ym.mean()) ** 2).sum())
        beta[i] = b
        try:
            xtx_inv[i] = np.linalg.pinv(Xm.T @ Xm)
        except np.linalg.LinAlgError:  # pragma: no cover
            pass
        ss_res[i] = ssr
        r2[i] = (1.0 - ssr / ss_tot) if ss_tot > 0 else np.nan
        ok[i] = True
    return CSResult(beta=beta, resid=resid, n_valid=n_valid, xtx_inv=xtx_inv,
                    ss_res=ss_res, r2=r2, ok=ok)


def _solve_days(y_arr: np.ndarray, cols: Sequence[np.ndarray], days: np.ndarray,
                mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """对 ``days`` 指定的少数日子退回逐日 ``lstsq``（奇异/共线的兜底路径）。"""
    ys = y_arr[days]
    cs = [np.asarray(c[days], dtype=np.float64) for c in cols]
    betas = np.full((len(days), len(cols)), np.nan)
    invs = np.full((len(days), len(cols), len(cols)), np.nan)
    ms = mask[days]
    for j in range(len(days)):
        yy = ys[j]
        X = np.column_stack([c[j] for c in cs])
        m = ms[j]
        Xm, ym = X[m], yy[m]
        if Xm.shape[0] < Xm.shape[1]:  # pragma: no cover - 已被 need 拦住
            continue
        try:
            betas[j], *_ = np.linalg.lstsq(Xm, ym, rcond=None)
            invs[j] = np.linalg.pinv(Xm.T @ Xm)
        except np.linalg.LinAlgError:  # pragma: no cover
            continue
    return betas, invs


def fit_cs(y: np.ndarray, xs: Sequence[np.ndarray], add_const: bool = True,
           min_stocks: int = 20) -> CSResult:
    """沿交易日维批量求解逐日横截面 OLS（结果与 :func:`fit_cs_rowwise` 一致）。

    流程：公共掩码 → 批量算 ``X'X``/``X'y`` → 堆叠 ``solve`` → **奇异日退回逐日
    ``lstsq``**。最后那一步不是保守，是必须：共线日上 ``solve`` 给出的是巨大但
    看起来正常的系数。
    """
    y_arr = np.asarray(y, dtype=np.float64)
    cols = _stacked_columns(xs)
    t, n = y_arr.shape
    if add_const:
        # 截距列用掩码本身：未参与回归的位置必须是"不存在"，而不是 1
        mask0 = np.isfinite(y_arr)
        for c in cols:
            mask0 &= np.isfinite(c)
    else:  # pragma: no cover - 调用方目前都带截距
        mask0 = np.isfinite(y_arr)
        for c in cols:
            mask0 &= np.isfinite(c)
    k = len(cols) + (1 if add_const else 0)
    if k == 0:  # pragma: no cover
        raise ValueError("至少需要一个回归列（控件或截距）")

    design = ([mask0.astype(np.float64)] if add_const else []) + cols
    n_valid = mask0.sum(axis=1)
    need = max(int(min_stocks), k + 2)
    enough = n_valid >= need

    # 单次扫描式的 X'X / X'y：k 很小，(a,b) 对级别循环即可，不物化三维数组
    zeroed = [np.where(mask0, c, 0.0) for c in design]
    XtX = np.empty((t, k, k), dtype=np.float64)
    for a in range(k):
        for b in range(a, k):
            s = np.einsum("tn,tn->t", zeroed[a], zeroed[b], optimize=True)
            XtX[:, a, b] = s
            XtX[:, b, a] = s
    y0 = np.where(mask0, y_arr, 0.0)
    Xty = np.stack([np.einsum("tn,tn->t", z, y0, optimize=True) for z in zeroed],
                   axis=1)

    beta = np.full((t, k), np.nan)
    xtx_inv = np.full((t, k, k), np.nan)
    # 奇异判定用奇异值比（k 很小，一次 SVD 很便宜）
    svals = np.linalg.svd(XtX, compute_uv=False)
    smax = svals[:, 0]
    smin = svals[:, -1]
    well_cond = np.isfinite(smax) & (smin > 1e-10 * np.maximum(smax, 1e-30))
    good_days = np.flatnonzero(enough & well_cond)
    bad_days = np.flatnonzero(enough & ~well_cond)

    if good_days.size:
        beta[good_days] = np.linalg.solve(
            XtX[good_days], Xty[good_days][..., None])[..., 0]
        xtx_inv[good_days] = np.linalg.inv(XtX[good_days])
    if bad_days.size:
        b2, inv2 = _solve_days(y_arr, design, bad_days, mask0)
        beta[bad_days] = b2
        xtx_inv[bad_days] = inv2

    ok = enough & np.all(np.isfinite(beta), axis=1)
    fitted = np.zeros_like(y_arr)
    for a in range(k):
        fitted += design[a] * beta[:, a][:, None]
    resid = np.where(mask0, y_arr - fitted, np.nan)
    resid[~ok] = np.nan

    r0 = np.where(mask0 & ok[:, None], resid, 0.0)
    ss_res = np.einsum("tn,tn->t", r0, r0, optimize=True)
    cnt = np.maximum(n_valid, 1)
    mu = y0.sum(axis=1) / cnt
    dev0 = np.where(mask0, y_arr - mu[:, None], 0.0)
    ss_tot = np.einsum("tn,tn->t", dev0, dev0, optimize=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1.0 - ss_res / ss_tot
    r2 = np.where((ss_tot > 0) & ok, r2, np.nan)
    ss_res = np.where(ok, ss_res, np.nan)
    return CSResult(beta=beta, resid=resid, n_valid=n_valid, xtx_inv=xtx_inv,
                    ss_res=ss_res, r2=r2, ok=ok)
