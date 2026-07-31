#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kronos 预测因子演示脚本 (scripts/run_kronos_factor.py)。

把 Kronos(金融 K 线基础模型) 作为一个"预测收益因子"接入 FactorGPT 的选股框架:
对面板数据逐标的、逐交易日滚动预测未来收益, 并用真实未来收益评估 IC 与多空收益。

用法
----
    # 离线运行(stub 模式, 无需 torch/GPU, 秒级完成)
    python scripts/run_kronos_factor.py

    # 尝试加载真实 Kronos 模型(需 pip install torch transformers 并下载权重)
    python scripts/run_kronos_factor.py --real

说明
----
- 默认使用带轻度动量持续性的合成面板, 以便直观看到预测因子与未来收益的同向关系。
- 若未安装 KronosPredictor/torch 或权重下载失败, 自动降级为 stub(几何动量代理),
  并打印告警, 流水线仍可跑通(这正是离线演示的预期行为)。
- 接入真实 refinery 流水线: 在 config.yaml 置 kronos.enabled=true, 并在
  refinery 的 PART-01 之后调用 src.kronos.attach_kronos_factor(ore, cfg)。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

# 允许从仓库任意位置运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kronos import KronosForecaster  # noqa: E402
from src.llm.client import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("kronos_demo")


def build_synthetic_panel(n_symbols: int = 200, n_days: int = 400, seed: int = 42) -> pd.DataFrame:
    """构造带轻度动量持续性的合成日线面板(便于演示预测因子的有效性)。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    symbols = [f"STK{i:04d}" for i in range(n_symbols)]
    rows = []
    # 每个标的的"真实"潜在动量 alpha, 使过去上涨更易继续上涨(轻度 AR(1))
    alpha = rng.normal(0, 0.001, n_symbols)
    for s, sym in enumerate(symbols):
        rets = np.zeros(n_days)
        last = 0.0
        for t in range(n_days):
            shock = rng.normal(0, 0.02)
            # 动量持续性: 上一期收益的一部分延续 + 标的 alpha
            ret = 0.15 * last + alpha[s] + shock
            last = ret
            rets[t] = ret
        price = 20 * np.cumprod(1 + rets)
        for t in range(n_days):
            rows.append(
                {
                    "symbol": sym,
                    "date": dates[t],
                    "open": price[t] * (1 + rng.normal(0, 0.001)),
                    "high": price[t] * 1.01,
                    "low": price[t] * 0.99,
                    "close": price[t],
                    "volume": float(rng.integers(1e5, 1e6)),
                }
            )
    return pd.DataFrame(rows)


def evaluate(preds: pd.DataFrame) -> dict:
    """评估预测因子: 全样本 IC、ICIR、多空分组收益。"""
    if preds.empty:
        return {}
    # 日度 IC: 每个交易日内的 rank IC(预测收益排名 vs 真实收益排名)
    ic_by_day = []
    long_short_by_day = []
    for _, g in preds.groupby("date"):
        if len(g) < 10:
            continue
        pr = g["kronos_pred_ret"].rank()
        rr = g["fwd_ret_realized"].rank()
        if pr.std() == 0 or rr.std() == 0:
            continue
        ic = np.corrcoef(pr, rr)[0, 1]
        ic_by_day.append(ic)
        # 多空: 多预测最高组, 空预测最低组
        q = pd.qcut(g["kronos_pred_ret"].rank(method="first"), 5, labels=False)
        long_ret = g.loc[q == 4, "fwd_ret_realized"].mean()
        short_ret = g.loc[q == 0, "fwd_ret_realized"].mean()
        long_short_by_day.append(long_ret - short_ret)
    ic = np.mean(ic_by_day) if ic_by_day else float("nan")
    icir = ic / (np.std(ic_by_day) + 1e-9) if len(ic_by_day) > 1 else float("nan")
    ls = np.mean(long_short_by_day) if long_short_by_day else float("nan")
    return {
        "n_points": int(len(preds)),
        "mean_ic": float(ic),
        "icir": float(icir),
        "mean_long_short_ret": float(ls),
        "n_days": len(ic_by_day),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Kronos 预测因子演示")
    ap.add_argument("--real", action="store_true", help="尝试加载真实 Kronos 模型")
    ap.add_argument("--n-symbols", type=int, default=200)
    ap.add_argument("--n-days", type=int, default=400)
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--n-eval", type=int, default=None, help="每标的仅评估最后 N 日(None=全部)")
    args = ap.parse_args()

    cfg = load_config() if os.path.exists("config.yaml") else {}
    kcfg = dict(cfg.get("kronos", {}))
    if args.real:
        kcfg["fallback_to_stub"] = False

    forecaster = KronosForecaster(kcfg)

    logger.info("构造合成面板 (%d 标的 × %d 日)...", args.n_symbols, args.n_days)
    kline = build_synthetic_panel(args.n_symbols, args.n_days)

    logger.info("运行 Kronos 面板预测 (mode=%s)...", "real" if not forecaster.using_stub else "stub")
    preds = forecaster.predict_panel(
        kline, lookback=args.lookback, n_eval=args.n_eval
    )
    if preds.empty:
        logger.error("预测结果为空, 退出")
        return 1

    stats = evaluate(preds)
    print("\n==================== Kronos 预测因子评估 ====================")
    print(f"运行模式        : {'真实 Kronos' if not forecaster.using_stub else 'stub(几何动量代理)'}")
    print(f"模型            : {forecaster.model_name}")
    print(f"预测样本数      : {stats.get('n_points')}")
    print(f"覆盖交易日数    : {stats.get('n_days')}")
    print(f"全样本 Rank-IC  : {stats.get('mean_ic'):.4f}")
    print(f"ICIR            : {stats.get('icir'):.4f}")
    print(f"日均多空收益    : {stats.get('mean_long_short_ret'):.4%}")
    print("============================================================")
    if forecaster.using_stub:
        print("注: 当前为 stub 降级模式(未加载真实 Kronos 权重)。")
        print("    如需真实预测, 请先 `pip install torch transformers`,")
        print("    并将脚本 --real 运行 (或置 config.yaml kronos.fallback_to_stub=false)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
