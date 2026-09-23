"""``mining.csreg``：批量横截面 OLS 必须与朴素逐日实现对拍一致。

批量求解是为了消掉"逐日 Python 循环 + 每天一次 lstsq"这个热点（见
``scripts/profile_mining.py`` 的热点榜），但它换来的是**数值路径变了**：

- 参考实现走 ``lstsq``（每次 SVD）；
- 批量实现走正规方程 + 堆叠 ``solve``，只在奇异日退回 ``lstsq``。

两条路径在良性数据上的差异应落在浮点噪声级别（这里用 1e-8），在**退化数据**上
必须落在同一批日子里失败——如果批量版在共线日给出了"看着正常"的系数，就是最
危险的那类错，必须由这里拦住。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mining import csreg as CR  # noqa: E402


def _frames(n_days: int, n_sym: int, seed: int, n_ctrl: int = 1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    cols = [f"s{i}" for i in range(n_sym)]
    y = pd.DataFrame(rng.normal(size=(n_days, n_sym)), index=idx, columns=cols)
    xs = [pd.DataFrame(rng.normal(size=(n_days, n_sym)), index=idx, columns=cols)
          for _ in range(n_ctrl)]
    return y, xs


def _compare(y, xs, **kw):
    ref = CR.fit_cs_rowwise(y.to_numpy(), [x.to_numpy() for x in xs], **kw)
    got = CR.fit_cs(y.to_numpy(), [x.to_numpy() for x in xs], **kw)
    return ref, got


def test_batch_matches_rowwise_single_control():
    y, xs = _frames(40, 60, seed=7)
    ref, got = _compare(y, xs, add_const=True, min_stocks=20)
    assert np.array_equal(ref.ok, got.ok), "解出的日子集合必须一致"
    assert np.array_equal(ref.n_valid, got.n_valid)
    assert np.allclose(ref.beta, got.beta, rtol=1e-8, atol=1e-9)
    assert np.allclose(ref.resid, got.resid, rtol=1e-8, atol=1e-9, equal_nan=True)
    assert np.allclose(ref.r2, got.r2, rtol=1e-8, atol=1e-9, equal_nan=True)


def test_batch_matches_rowwise_multi_controls():
    y, xs = _frames(30, 80, seed=11, n_ctrl=3)
    ref, got = _compare(y, xs, add_const=True, min_stocks=20)
    assert np.array_equal(ref.ok, got.ok)
    assert np.allclose(ref.beta, got.beta, rtol=1e-7, atol=1e-8)
    assert np.allclose(ref.resid, got.resid, rtol=1e-7, atol=1e-8, equal_nan=True)


def test_collinear_control_falls_back_to_lstsq():
    """完全共线：两条路径要给出同一组有限系数，而不是一边 NaN 一边正常。"""
    y, xs = _frames(25, 50, seed=3, n_ctrl=2)
    xs[1] = xs[0].copy()          # 与第一个控件完全共线
    ref, got = _compare(y, xs, add_const=True, min_stocks=20)
    assert np.array_equal(ref.ok, got.ok), "共线日的成败判定必须一致"
    assert np.all(np.isfinite(got.beta[got.ok]))
    assert np.allclose(ref.beta, got.beta, rtol=1e-6, atol=1e-8)


def test_insufficient_sample_days_are_rejected():
    """样本不足的日子两边都不许产出残差——漏出 NaN 之外的值会污染下游统计。"""
    y, xs = _frames(20, 40, seed=5)
    y.iloc[0:3, 3:] = np.nan       # 前 3 天只剩 3 个有效点
    ref, got = _compare(y, xs, add_const=True, min_stocks=20)
    assert not ref.ok[:3].any() and not got.ok[:3].any()
    assert np.array_equal(ref.ok, got.ok)
    assert np.isnan(got.resid[~got.ok]).all(), "未解出的日子不得留下残差"


def test_batch_is_faster_than_rowwise():
    """加速本身要有断言：只对齐不算完成，慢回去应当能在这里被发现。"""
    y, xs = _frames(300, 200, seed=13, n_ctrl=2)
    yv, xsv = y.to_numpy(), [x.to_numpy() for x in xs]
    r0 = CR.fit_cs_rowwise(yv, xsv, min_stocks=20)
    g0 = CR.fit_cs(yv, xsv, min_stocks=20)
    assert np.allclose(r0.resid, g0.resid, rtol=1e-7, atol=1e-8, equal_nan=True)

    def timed(fn, repeat: int = 5) -> float:
        fn()                      # 预热：首次调用要付 import/JIT/分配器的冷启动成本
        best = float("inf")
        for _ in range(repeat):
            t = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t)
        return best               # 取最优而不是均值：CI 是共享机器，噪声围不住均值

    t_ref = timed(lambda: CR.fit_cs_rowwise(yv, xsv, min_stocks=20))
    t_fast = timed(lambda: CR.fit_cs(yv, xsv, min_stocks=20))
    assert t_fast < t_ref * 0.9, (
        f"批量版没有体现出加速：参考 {t_ref:.3f}s vs 批量 {t_fast:.3f}s")
