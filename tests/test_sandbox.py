"""因子沙箱安全性与 AST 前视偏差检测测试。

覆盖两类核心防线：
  1) 沙箱隔离：禁用 os/subprocess/socket 等危险模块、危险内建（open/eval/exec）、
     import 白名单、子进程超时（防死循环卡死主进程）；
  2) AST 静态检查：shift(<=0)、未来/标签变量名、未 shift 的价格列参与运算。
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.factor_builder import FactorSandbox


def _make_df(n_dates=40, n_symbols=3, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    rows = []
    for sym in [f"{i:06d}" for i in range(1, n_symbols + 1)]:
        for d in dates:
            rows.append(
                (d.strftime("%Y-%m-%d"), sym,
                 rng.random(), rng.random(), rng.random(),
                 5.0 + rng.random() * 20.0, int(rng.integers(1e4, 1e6)),
                 rng.random() * 1e7, rng.random() * 4.0 - 2.0)
            )
    return pd.DataFrame(rows, columns=[
        "date", "symbol", "open", "high", "low",
        "close", "volume", "amount", "pct_chg"])


def _sandbox(**kw) -> FactorSandbox:
    cfg = {"engine": {"sandbox": {"subprocess": True, "timeout": 5.0,
                                  "memory_limit_mb": 0, **kw}}}
    return FactorSandbox(cfg)


LEGAL = """\
def alpha_factor(df):
    \"\"\"5 日动量，严格 shift(1) 防前视。\"\"\"
    return df.groupby("symbol")["close"].pct_change().shift(1)
"""


# ---------------------------------------------------------------- 合法代码 #
class TestLegalCode:
    def test_shifted_momentum_passes(self):
        df = _make_df()
        out = _sandbox().run(LEGAL, df)
        assert isinstance(out, pd.Series)
        assert len(out) == len(df)
        assert out.notna().any()

    def test_nan_preserved(self):
        out = _sandbox().run(LEGAL, _make_df(n_dates=20))
        assert out.isna().sum() > 0  # 首日 pct_change 为 NaN，不应被填零篡改


# ---------------------------------------------------------------- 沙箱隔离 #
class TestSandboxIsolation:
    @pytest.mark.parametrize("mod", ["os", "subprocess", "socket", "shutil"])
    def test_dangerous_imports_rejected(self, mod):
        code = f'def alpha_factor(df):\n    import {mod}\n    return df["close"].pct_change()\n'
        with pytest.raises((ValueError, ImportError)):
            _sandbox().run(code, _make_df())

    def test_import_statement_rejected(self):
        code = 'def alpha_factor(df):\n    from os import path\n    return df["close"].pct_change()\n'
        with pytest.raises((ValueError, ImportError)):
            _sandbox().run(code, _make_df())

    def test_dynamic_import_rejected(self):
        code = 'def alpha_factor(df):\n    return df["close"].pct_change() + __import__("os").pathsep\n'
        with pytest.raises((ValueError, ImportError)):
            _sandbox().run(code, _make_df())

    @pytest.mark.parametrize("builtin", ["eval", "exec", "open", "compile"])
    def test_dangerous_builtins_rejected(self, builtin):
        code = f'def alpha_factor(df):\n    {builtin}("1")\n    return df["close"].pct_change()\n'
        with pytest.raises((ValueError, NameError)):
            _sandbox().run(code, _make_df())

    def test_infinite_loop_times_out(self):
        code = 'def alpha_factor(df):\n    while True:\n        pass\n'
        with pytest.raises(TimeoutError):
            _sandbox(timeout=3).run(code, _make_df())

    def test_os_error_without_function(self):
        with pytest.raises(ValueError, match="alpha_factor"):
            _sandbox().run('# 没有定义 alpha_factor 函数\nx = 1\n', _make_df())

    def test_empty_code_rejected(self):
        with pytest.raises(ValueError):
            _sandbox().run("", _make_df())


# ---------------------------------------------------------------- 前视检测 #
class TestLookaheadDetection:
    def test_negative_shift_rejected(self):
        code = 'def alpha_factor(df):\n    return df.groupby("symbol")["close"].shift(-1)\n'
        with pytest.raises(ValueError, match="前视"):
            _sandbox().run(code, _make_df())

    def test_zero_shift_rejected(self):
        code = 'def alpha_factor(df):\n    return df.groupby("symbol")["close"].shift(0)\n'
        with pytest.raises(ValueError, match="前视"):
            _sandbox().run(code, _make_df())

    @pytest.mark.parametrize("name", ["fwd_ret", "future_ret", "next_close", "target_ret"])
    def test_future_variable_names_rejected(self, name):
        code = f'def alpha_factor(df):\n    return df["{name}"].fillna(0)\n'
        with pytest.raises(ValueError, match="前视"):
            _sandbox().run(code, _make_df())

    def test_unshifted_price_attribute_rejected(self):
        # df.close 以属性形式参与运算且未被 shift 包裹 → 前视
        code = 'def alpha_factor(df):\n    return df.groupby("symbol").close.pct_change()\n'
        with pytest.raises(ValueError, match="前视"):
            _sandbox().run(code, _make_df())

    def test_unshifted_price_in_binop_rejected(self):
        code = 'def alpha_factor(df):\n    return df.close * 2.0\n'
        with pytest.raises(ValueError, match="前视"):
            _sandbox().run(code, _make_df())

    def test_shifted_price_attribute_ok(self):
        code = 'def alpha_factor(df):\n    return df.groupby("symbol").close.pct_change().shift(1)\n'
        out = _sandbox().run(code, _make_df())
        assert isinstance(out, pd.Series)
