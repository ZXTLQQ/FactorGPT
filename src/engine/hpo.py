"""超参搜索 (src/engine/hpo.py)

基于 Optuna 把因子构造 / 回测超参从手调升级为系统化搜索，目标为样本外 ICIR 或
多空 Sharpe。Optuna 为可选依赖，未安装时给出清晰提示。

典型用法：
    from engine.backtest import FactorBacktester
    from engine.hpo import search_backtest_params
    best_params, best_value = search_backtest_params(kline, factor, n_trials=30)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd


def _require_optuna():
    try:
        import optuna  # type: ignore
        return optuna
    except ImportError:
        raise ImportError(
            "超参搜索需要 optuna，请先安装：pip install optuna\n"
            "（这是可选依赖，不安装不影响其余功能）"
        )


def search_backtest_params(
    kline: pd.DataFrame,
    factor: pd.Series,
    n_trials: int = 20,
    objective_metric: str = "icir",
    seed: int = 0,
) -> Tuple[Dict[str, Any], float]:
    """搜索回测口径超参：分组数 n_quantiles、持有期 forward_periods、缩尾比例 winsorize_pct。

    对固定因子，这些超参影响分组收益结构；目标为最大化 ICIR（或 long_short_sharpe）。
    """
    optuna = _require_optuna()

    def objective(trial):
        n_q = trial.suggest_int("n_quantiles", 3, 10)
        fwd = trial.suggest_int("forward_periods", 1, 5)
        wz = trial.suggest_float("winsorize_pct", 0.005, 0.05)
        m: Dict[str, Any] = {}
        try:
            from engine.backtest import FactorBacktester
            bt = FactorBacktester(n_quantiles=n_q, forward_periods=fwd, winsorize_pct=wz)
            m = bt.evaluate(kline, factor, verbose=False)
        except Exception:
            return -1e9
        val = float(m.get(objective_metric) or 0.0)
        turn = float(m.get("turnover", 0) or 0)
        # 稳定性惩罚：换手率过高降分，避免过拟合高换手的"假 alpha"
        return val - 0.01 * turn

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return dict(study.best_params), float(study.best_value)


def optimize(
    objective_fn: Callable[[Any], float],
    n_trials: int = 20,
    direction: str = "maximize",
    seed: int = 0,
) -> Tuple[Dict[str, Any], float]:
    """通用超参搜索。objective_fn(trial) -> float，由 optuna 注入 trial。

    返回 (best_params, best_value)。示例：
        def obj(trial):
            w = trial.suggest_int("window", 5, 60)
            ...
        params, val = optimize(obj, n_trials=30)
    """
    optuna = _require_optuna()
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective_fn, n_trials=n_trials)
    return dict(study.best_params), float(study.best_value)
