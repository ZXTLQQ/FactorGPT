"""对抗式合成数据：注入已知因子结构，验证 Agent 能否「挖出」真实存在的因子。

用途：评判/自检时，证明「系统真的在挖掘因子，而非随机噪声里碰运气」。
做法：为每只股票赋予一个固定潜在特征 beta（截面可排序），并让日收益
    ret = signal_strength * beta + 噪声
则「过去收益（动量）」类因子应与 beta 正相关，从而在合成数据上取得显著 IC。
同时给出 ground_truth 因子（=beta 本身），其 IC 应≈ sign(signal_strength)，
作为「信号确实存在」的下界校准。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def build_adversarial_synthetic(
    n_symbols: int = 50,
    start: str = "2021-01-01",
    end: str = "2023-12-31",
    signal_strength: float = 0.4,
    noise: float = 0.02,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """生成注入已知截面因子结构的行情长表。

    Returns:
        kline: 长表，含 date/symbol/open/high/low/close/volume/amount/pct_chg
        ground_truth: 以 (date,symbol) 为索引的 Series，值=该股票的潜在 beta
        meta: 元信息（signal_strength, noise, seed, beta 范围等）
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="B")  # 工作日
    syms = [f"AST{i:03d}" for i in range(n_symbols)]
    beta = rng.normal(0, 1, size=n_symbols)  # 截面潜在特征

    rows = []
    for i, s in enumerate(syms):
        ret = signal_strength * beta[i] + rng.normal(0, noise, len(dates))
        price = 10.0 * np.cumprod(1.0 + ret)
        for d, r, p in zip(dates, ret, price):
            rows.append((d, s, p * (1 - 0.005), p, p * (1 + 0.005), p, 1.0, 1.0e6, r))
    kline = pd.DataFrame(
        rows,
        columns=["date", "symbol", "open", "high", "low", "close", "volume", "amount", "pct_chg"],
    )
    gt = pd.Series(
        np.tile(beta, len(dates)),
        index=pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"]),
        name="factor",
    )
    meta = {
        "signal_strength": signal_strength,
        "noise": noise,
        "seed": seed,
        "n_symbols": n_symbols,
        "beta_min": float(beta.min()),
        "beta_max": float(beta.max()),
    }
    return kline, gt, meta


def verify_recovery(kline: pd.DataFrame, ground_truth: pd.Series, threshold: float = 0.02):
    """校验注入信号是否可被 ground_truth 因子恢复（IC 达到阈值）。

    直接按截面相关计算 raw IC（绕过 backtest.evaluate 内的强中性化），
    用以证明「信号确实存在于数据中、可被因子恢复」。注意：若注入的是
    *逐股恒定* 的潜在特征，正常回测流水线的中性化会将其归零（这是真实行为），
    因此这里用 raw IC 做下界校准。

    返回 (ic, recovered: bool)。若 recovered=False，说明注入强度/噪声配比
    导致信号被淹没，应调大 signal_strength 或调小 noise。
    """
    merged = kline[["date", "symbol", "pct_chg"]].merge(
        ground_truth.rename("factor").reset_index(), on=["date", "symbol"], how="inner"
    )
    merged["fwd"] = merged.groupby("symbol")["pct_chg"].shift(-1)
    merged = merged.dropna(subset=["fwd", "factor"])
    if merged.empty:
        return 0.0, False
    corrs = merged.groupby("date").apply(lambda g: g["factor"].corr(g["fwd"]))
    ic = float(corrs.mean())
    return ic, abs(ic) >= threshold
