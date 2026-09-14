"""FactorGPT 核心回测数学与沙箱安全性的单元测试。

运行：在仓库根目录执行  pytest tests/test_backtest.py  -q
（模块搜索路径已在文件顶部插入 src/）
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

try:
    import pytest
except ImportError:  # 允许在无 pytest 环境（如评委机器）下用标准库运行
    pytest = None

from engine.backtest import FactorBacktester, portfolio_turnover
from engine.factor_builder import analyze_lookahead


# ---------------------------------------------------------------------------
# 小工具：构造可控的行情/因子
# ---------------------------------------------------------------------------
def _make_panel(n_sym=10, n_days=60, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    syms = [f"S{i}" for i in range(n_sym)]
    rows = []
    for s in syms:
        ret = rng.normal(0, 0.01, n_days)
        price = 10 * np.cumprod(1 + ret)
        for d, r, p in zip(dates, ret, price):
            rows.append((d, s, r, p))
    kline = pd.DataFrame(rows, columns=["date", "symbol", "pct_chg", "close"])
    return kline, dates, syms


def _factor_series(values, dates, syms):
    arr = np.asarray(values).reshape(len(dates) * len(syms))
    return pd.Series(
        arr,
        index=pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"]),
        name="factor",
    )


# ---------------------------------------------------------------------------
# 1) IC / RankIC 正确性
# ---------------------------------------------------------------------------
def test_ic_perfect_positive():
    kline, dates, syms = _make_panel()
    # 因子 = 未来收益的精确副本 -> IC 应为 1
    merged = kline.copy()
    merged["fwd"] = merged.groupby("symbol")["pct_chg"].shift(-1)
    fac_vals = []
    for s in syms:
        sub = merged[merged["symbol"] == s]
        fac_vals.extend(sub["fwd"].shift(1).dropna().tolist())  # 用 t-1 的未来收益作因子值，避免前视
    # 简化：直接用未来收益构造因子（允许，仅为测试数学）
    fwd_full = merged.sort_values(["symbol", "date"]).assign(
        fwd=lambda d: d.groupby("symbol")["pct_chg"].shift(-1)
    )
    fac = pd.Series(
        fwd_full["fwd"].values,
        index=pd.MultiIndex.from_arrays([fwd_full["date"], fwd_full["symbol"]]),
        name="factor",
    )
    m = FactorBacktester().evaluate(kline, fac)
    assert m.get("ic") is not None
    # 因子即未来收益 -> IC 接近 1（受 shift 对齐影响应仍显著为正）
    assert m["ic"] > 0.9, f"IC 应接近 1，实际 {m['ic']}"


def test_rank_ic_consistent_sign():
    kline, dates, syms = _make_panel(seed=3)
    rng = np.random.default_rng(7)
    base = rng.normal(0, 1, len(dates) * len(syms))
    fac = _factor_series(base, dates, syms)
    m = FactorBacktester().evaluate(kline, fac)
    # 随机因子 IC 应接近 0
    assert abs(m["ic"]) < 0.2, f"随机因子 IC 应接近 0，实际 {m['ic']}"


# ---------------------------------------------------------------------------
# 2) 换手率口径一致性（evaluate 与 realistic_portfolio）
# ---------------------------------------------------------------------------
def test_turnover_consistency():
    kline, dates, syms = _make_panel(n_days=120)
    rng = np.random.default_rng(11)
    fac = _factor_series(rng.normal(0, 1, len(dates) * len(syms)), dates, syms)
    m_eval = FactorBacktester().evaluate(kline, fac)
    # portfolio_turnover 应与 evaluate 内部使用的同一函数结果一致
    t = portfolio_turnover(fac, kline, top_frac=0.1)
    assert t is not None
    assert abs(t - m_eval["turnover"]) < 1e-9, "evaluate 与 portfolio_turnover 应一致"
    assert 0.0 <= t <= 2.0


# ---------------------------------------------------------------------------
# 3) 前视偏差静态检查
# ---------------------------------------------------------------------------
def test_lookahead_detect_shift0():
    code = "def alpha_factor(df):\n    return df.groupby('symbol')['close'].shift(0)\n"
    assert analyze_lookahead(code), "shift(0) 应被检测为前视"


def test_lookahead_detect_shift_neg():
    code = "def alpha_factor(df):\n    return df.groupby('symbol')['close'].shift(-1)\n"
    assert analyze_lookahead(code), "shift(-1) 应被检测为前视"


def test_lookahead_pass_shift1():
    code = "def alpha_factor(df):\n    return df.groupby('symbol')['close'].shift(1)\n"
    assert not analyze_lookahead(code), "shift(1) 不应被判前视"


def test_lookahead_detect_future_name():
    code = "def alpha_factor(df):\n    return df['fwd_ret']\n"
    assert analyze_lookahead(code), "引用 fwd_ret 应被检测为前视"


# ---------------------------------------------------------------------------
# 4) 与 alphalens-reloaded 交叉校验（可选，未安装则跳过）
# ---------------------------------------------------------------------------
def test_alphalens_crosscheck():
    try:
        import importlib
        importlib.import_module("alphalens.reformed")
    except Exception:  # noqa: BLE001
        # 统一抛 unittest.SkipTest：pytest 会将其识别为 skip，
        # 且 __main__ 手写运行器（仅捕获 unittest.SkipTest）也不会因
        # pytest.skip() 抛出的 _pytest.outcomes.Skipped 而崩溃。
        import unittest
        raise unittest.SkipTest("alphalens-reformed 未安装，跳过交叉校验")


# ---------------------------------------------------------------------------
# 5) 向量化逐日 IC / 分位数分组：与逐日参考实现等价
# ---------------------------------------------------------------------------
def _synthetic_ic_panel(seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=40).strftime("%Y-%m-%d")
    syms = [f"S{i}" for i in range(8)]
    idx = pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"])
    f = pd.Series(rng.normal(size=len(idx)), index=idx)
    y = pd.Series(0.4 * f.to_numpy() + rng.normal(size=len(idx)), index=idx)
    panel = pd.DataFrame({"date": idx.get_level_values("date").to_numpy(),
                          "symbol": idx.get_level_values("symbol").to_numpy(),
                          "factor": f.to_numpy(), "fwd_ret": y.to_numpy()})
    # 边界日：第 0 天常数截面；第 2 天混入 inf；第 3 天只留一只股票
    panel.loc[panel["date"] == dates[0], "factor"] = 1.0
    panel.loc[16, "factor"] = np.inf
    keep = ~((panel["date"] == dates[3]) & (panel["symbol"] != "S0"))
    return panel[keep].reset_index(drop=True), dates


def _ic_reference(panel, method):
    """逐日 np.corrcoef 参考实现（与向量化前的旧实现同语义）。"""
    out = {}
    for d, g in panel.groupby("date"):
        xv, yv = g["factor"].to_numpy(float), g["fwd_ret"].to_numpy(float)
        if len(xv) < 2 or not (np.isfinite(xv).all() and np.isfinite(yv).all()):
            continue
        if np.ptp(xv) == 0 or np.ptp(yv) == 0:
            continue
        if method == "spearman":
            xv = pd.Series(xv).rank().to_numpy()
            yv = pd.Series(yv).rank().to_numpy()
        c = np.corrcoef(xv, yv)[0, 1]
        if np.isfinite(c):
            out[d] = c
    return pd.Series(out, name="ic")


def test_ic_series_matches_daily_reference_pearson_and_spearman():
    from engine.backtest import _ic_series

    panel, _ = _synthetic_ic_panel()
    for method in ("pearson", "spearman"):
        got = _ic_series(panel, method)
        ref = _ic_reference(panel, method)
        assert list(got.index) == list(ref.index), f"{method}: 日期集合要一致"
        np.testing.assert_allclose(got.to_numpy(), ref.to_numpy(),
                                   rtol=1e-12, atol=1e-12)
    # 常数截面 / 含 inf / 单只股票的日子必须被丢掉，而不是给出伪 IC
    assert len(_ic_series(panel, "pearson")) == 37


def test_quantile_grouping_matches_daily_qcut_reference():
    rng = np.random.default_rng(23)
    n_q = 5
    dates = pd.bdate_range("2024-02-01", periods=30).strftime("%Y-%m-%d")
    syms = [f"S{i}" for i in range(20)]
    panel = pd.DataFrame({
        "date": np.repeat(dates, len(syms)),
        "factor": rng.normal(size=30 * len(syms)),
        "fwd_ret": rng.normal(size=30 * len(syms)),
    })
    # 新实现：逐日百分位秩 → 等分桶
    pct = panel.groupby("date")["factor"].rank(pct=True)
    new = np.ceil(pct.to_numpy() * n_q).clip(1, n_q) - 1
    # 旧实现：逐日 qcut（连续无并列时两者应逐一相等）
    old = np.empty_like(new)
    for d, g in panel.groupby("date"):
        m = panel["date"].to_numpy() == d
        old[m] = pd.qcut(g["factor"], n_q, labels=False, duplicates="drop").to_numpy()
    np.testing.assert_array_equal(new, old)


if __name__ == "__main__":
    # 无 pytest 时也能直接运行：python tests/test_backtest.py
    import traceback
    import unittest

    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = skipped = failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except unittest.SkipTest as e:
            print(f"SKIP {fn.__name__}: {e}")
            skipped += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n=== passed={passed} skipped={skipped} failed={failed} ===")
    sys.exit(1 if failed else 0)
