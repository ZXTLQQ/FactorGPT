"""方向二（工程化）与方向三（因子生成升级）的单元测试。

沿用 tests/test_backtest.py 的手写运行器风格：定义 test_* 函数，末尾统一收集执行。
覆盖：
- 实验追踪 ExperimentTracker（本地 JSONL + best/history）
- 遗传编程因子发现 GeneticFactorMiner（产出可运行代码）
- 多 LLM 路由 LLMRouter（mock client，draft 海选 + critic 精炼）
- 并行批量回测 batch_evaluate
- Optuna 超参搜索（optuna 未装时自动跳过）

运行：python tests/test_engineering.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json

import numpy as np
import pandas as pd

from engine.backtest import FactorBacktester, batch_evaluate
from engine.genetic_factors import GeneticFactorMiner
from engine.hpo import search_backtest_params
from engine.tracking import ExperimentTracker
from llm.router import LLMRouter


def _make_kline(n_days: int = 120, n_sym: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="D")
    syms = [f"S{i}" for i in range(n_sym)]
    rows = []
    for s in syms:
        ret = rng.normal(0, 0.01, n_days)
        px = 10 * np.cumprod(1 + ret)
        for d, p, r in zip(dates, px, ret):
            rows.append((d, s, p, r))
    df = pd.DataFrame(rows, columns=["date", "symbol", "close", "pct_chg"])
    df["open"] = df["close"] * (1 + rng.normal(0, 0.001, len(df)))
    df["high"] = df[["open", "close"]].max(axis=1) * 1.01
    df["low"] = df[["open", "close"]].min(axis=1) * 0.99
    df["volume"] = rng.integers(1e5, 1e6, len(df)).astype(float)
    df["amount"] = df["volume"] * df["close"]
    return df


def test_tracking_local() -> None:
    import tempfile

    d = tempfile.mkdtemp()
    tr = ExperimentTracker({"experiment_tracking": {"backend": "local", "dir": d}})
    rec = tr.log_factor(
        "f1", "def alpha_factor(df):\n    return df",
        {"ic": 0.05, "icir": 0.3}, {"k": 1}, {"t": "x"},
    )
    assert rec["name"] == "f1"
    assert tr.best("f1")["metrics"]["ic"] == 0.05
    assert len(tr.history()) == 1
    assert "f1" in tr.summary()
    print("test_tracking_local OK")


def test_genetic_mine() -> None:
    kline = _make_kline(seed=3)
    miner = GeneticFactorMiner(kline, seed=1)
    res = miner.mine(generations=4, pop_size=30, top_k=2, verbose=False)
    assert isinstance(res, list) and len(res) >= 1
    for r in res:
        assert "code" in r and "train_ic" in r
        # 生成的代码语法合法且可定义 alpha_factor
        ns: dict = {}
        exec(r["code"], ns)
        assert callable(ns["alpha_factor"])
    print("test_genetic_mine OK")


def test_router_mock() -> None:
    class Fake:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def complete(self, system: str, user: str, temperature=None) -> str:
            return json.dumps({
                "name": "f", "description": "d",
                "code": "def alpha_factor(df):\n    return df['close'].pct_change()",
                "rationale": "r", "references": [],
            })

    router = LLMRouter(draft_client=Fake("draft"), critic_client=Fake("critic"))
    out = router.complete("sys", "usr")
    parsed = json.loads(out)
    assert "code" in parsed
    code = router.generate_factor_code("sys", "usr")
    assert code and "alpha_factor" in code
    print("test_router_mock OK")


def test_batch_evaluate() -> None:
    kline = _make_kline(seed=7)
    # evaluate 约定因子索引为 (date, symbol) 二级多重索引（与 nodes 中 sandbox.run 产出一致）
    idx = pd.MultiIndex.from_arrays([kline["date"].to_numpy(), kline["symbol"].to_numpy()])
    f1 = kline.groupby("symbol")["close"].pct_change().set_axis(idx)
    f2 = kline.groupby("symbol")["volume"].pct_change().set_axis(idx)
    f1.name = "factor"
    f2.name = "factor"
    factors = {"mom": f1, "vol": f2}
    out = batch_evaluate(kline, factors, verbose=False)
    assert set(out.keys()) == {"mom", "vol"}
    for m in out.values():
        assert "ic" in m
    print("test_batch_evaluate OK")


def test_hpo() -> None:
    try:
        import optuna  # noqa: F401
    except ImportError:
        print("test_hpo SKIP (optuna not installed)")
        return
    kline = _make_kline(seed=11)
    f = kline.groupby("symbol")["close"].pct_change()
    f.name = "factor"
    params, val = search_backtest_params(kline, f, n_trials=5)
    assert "n_quantiles" in params
    print("test_hpo OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = skipped = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except SystemExit:
            skipped += 1
            print(f"{t.__name__} SKIP")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"{t.__name__} FAIL: {e}")
    print(f"\n=== passed={passed} skipped={skipped} failed={failed} ===")
    sys.exit(1 if failed else 0)
