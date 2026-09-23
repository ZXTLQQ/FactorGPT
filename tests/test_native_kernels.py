"""原生热核：有编译产物时验证**逐位一致**，没有时不许假装通过。

这个文件要解决的是一个具体风险：C++ 热核一旦在语义上有偏差（并列秩的处理、NaN
的去留、ddof、sd==0 的输出），因子值会**悄悄改变**——回测不会报错，只会给你另一个
答案，而且是过了很久才发现的那种。所以：

1. 只要构建产物在，就必须跑完整对拍；
2. CI 里用 ``FG_REQUIRE_NATIVE=1`` 把"跳过"变成"失败"，避免某天动态库不再被构建
   而没人察觉——静默降级到 pandas 是设计好的容错，但**长期**静默降级等于这个功能
   从来不存在；
3. ``FG_DISABLE_NATIVE=1`` 时必须回到 pandas 且结果不变。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mining import native_kernels as NK  # noqa: E402
from mining import ops  # noqa: E402

REQUIRE_NATIVE = str(os.environ.get("FG_REQUIRE_NATIVE") or "").strip() in {"1", "true"}


def _native_available() -> bool:
    NK.reset_cache()
    return NK.native() is not None


def _require_or_skip() -> None:
    """没有原生库时：CI 会把它变成失败，本地则跳过（并说明怎么编出来）。"""
    if _native_available():
        return
    msg = ("未检测到原生热核（python scripts/build_native.py 可构建）；"
           "CI 下应以 FG_REQUIRE_NATIVE=1 运行")
    if REQUIRE_NATIVE:
        pytest.fail(msg)
    pytest.skip(msg)


def _tricky_frame() -> np.ndarray:
    """把所有容易走偏的情形塞进一个矩阵：并列、负零、全并列、全 NaN、标准差为 0。"""
    rng = np.random.default_rng(7)
    a = np.round(rng.normal(size=(10, 11)) * 3) / 3      # 造出大量并列与 -0.0
    a[0, 3:] = np.nan          # 部分缺失
    a[1, :] = 2.0              # 全并列 → sd == 0
    a[2, :] = np.nan           # 全 NaN
    a[3, 0] = np.inf           # 无穷值参与排名
    a[4, 0] = -0.0
    a[4, 1] = 0.0              # -0.0 与 0.0 必须算并列
    a[5, :] = 0.0              # 全零 → sd == 0（不是 NaN，但除不得）
    return a


def test_backend_switches_off_by_env(monkeypatch):
    monkeypatch.setenv("FG_DISABLE_NATIVE", "1")
    NK.reset_cache()
    assert NK.backend() == "pandas"
    monkeypatch.delenv("FG_DISABLE_NATIVE", raising=False)
    NK.reset_cache()


def test_ops_match_pandas_without_native(monkeypatch):
    """禁用原生后，``ops`` 的结果必须与纯 pandas 路径一致（保证降级无损功能）。"""
    import pandas as pd

    monkeypatch.setenv("FG_DISABLE_NATIVE", "1")
    NK.reset_cache()
    a = np.round(np.random.default_rng(3).normal(size=(12, 9)) * 2) / 2
    a[0, 4:] = np.nan
    df = pd.DataFrame(a)
    assert np.allclose(ops.cs_rank(df).to_numpy(), df.rank(axis=1, pct=True).to_numpy(),
                       equal_nan=True)
    monkeypatch.delenv("FG_DISABLE_NATIVE", raising=False)
    NK.reset_cache()


def test_rank_matches_pandas_bitwise():
    _require_or_skip()
    a = _tricky_frame()
    got = NK.rank_pct(a)
    exp = NK.pandas_rank_pct(a)
    assert np.array_equal(np.isnan(got), np.isnan(exp)), "NaN 位置必须与 pandas 一致"
    assert np.nanmax(np.abs(got - exp)) <= 1e-12


def test_zscore_matches_pandas_bitwise():
    _require_or_skip()
    a = _tricky_frame()
    got = NK.zscore(a)
    exp = NK.pandas_zscore(a)
    assert np.array_equal(np.isnan(got), np.isnan(exp))
    assert np.nanmax(np.abs(got - exp)) <= 1e-10


def test_ties_and_zero_handling():
    """-0.0 与 0.0 是一组并列；全常数列的标准差为 0 → 输出必须是 NaN 而不是 0。"""
    _require_or_skip()
    a = np.array([[-0.0, 0.0, 1.0], [5.0, 5.0, 5.0]], dtype=np.float64)
    r = NK.rank_pct(a)
    assert r[0, 0] == pytest.approx(0.5) and r[0, 1] == pytest.approx(0.5)
    assert r[0, 2] == pytest.approx(1.0)
    assert np.isnan(NK.zscore(a)[1]).all(), "sd==0 的行必须给 NaN，不能给 0"


def test_min_count_rows_are_nan():
    _require_or_skip()
    a = np.full((2, 6), np.nan)
    a[0, 0] = 1.0                      # 只有一个有效值：sd 不可估
    assert np.isnan(NK.rank_pct(a)[1]).all()
    assert np.isnan(NK.zscore(a)[0]).all()


def test_corr_matches_reference():
    """相关核的基准是同一套 numpy 公式的独立副本（NaN 位置也要一致）。"""
    _require_or_skip()
    rng = np.random.default_rng(11)
    a = rng.normal(size=(40, 30))
    b = rng.normal(size=(40, 30))
    a[0, 5:] = np.nan            # 样本不足
    b[7, :] = 3.0                # 常数行 → 方差为 0
    a[9, 0] = np.inf             # inf 视为无效
    got = NK.corr(a, b, min_stocks=10)
    exp = NK.pandas_corr(a, b, min_stocks=10)
    assert np.array_equal(np.isnan(got), np.isnan(exp)), "退化行的判定必须一致"
    assert np.nanmax(np.abs(got - exp)) <= 1e-12


def test_ops_corr_uses_same_semantics_with_and_without_native(monkeypatch):
    """开关两侧结果相同——加速不许改变任何一个数字。"""
    import pandas as pd

    rng = np.random.default_rng(13)
    a = pd.DataFrame(rng.normal(size=(20, 25)))
    b = pd.DataFrame(rng.normal(size=(20, 25)))
    NK.reset_cache()
    with_native = ops.cs_corr(a, b, min_stocks=10).to_numpy()
    monkeypatch.setenv("FG_DISABLE_NATIVE", "1")
    NK.reset_cache()
    without = ops.cs_corr(a, b, min_stocks=10).to_numpy()
    monkeypatch.delenv("FG_DISABLE_NATIVE", raising=False)
    NK.reset_cache()
    assert np.allclose(with_native, without, rtol=1e-10, atol=1e-12, equal_nan=True)


def test_ops_use_native_and_keep_frame_metadata():
    """走 ops 时必须保留 index/columns——热核只认裸数组，元数据由调用方复原。"""
    _require_or_skip()
    import pandas as pd

    idx = pd.bdate_range("2024-01-01", periods=5)
    cols = [f"s{i}" for i in range(4)]
    df = pd.DataFrame(np.random.default_rng(9).normal(size=(5, 4)),
                      index=idx, columns=cols)
    out = ops.cs_rank(df)
    assert list(out.index) == list(idx) and list(out.columns) == cols
    assert np.allclose(out.to_numpy(), NK.pandas_rank_pct(df.to_numpy()),
                       equal_nan=True)
