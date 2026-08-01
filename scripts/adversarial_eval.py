"""对抗式验证脚本：用注入已知因子结构的合成数据，检验挖掘流水线能否恢复信号。

流程：
  1) 生成注入「截面潜在因子」的合成行情（ret = strength*beta + 噪声）；
  2) 用 ground_truth 直接验证信号确实可被恢复（raw IC 达到阈值）；
  3) 用一条「基准动量因子」穿过真实沙箱 + 回测流水线，验证流水线也能取到显著 IC，
     从而证明系统「真的在挖掘因子，而非在随机噪声里碰运气」。

用法：
  python scripts/adversarial_eval.py --n-symbols 50 --strength 0.5 --noise 0.015
"""
import argparse
import sys

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from data.adversarial_synthetic import build_adversarial_synthetic, verify_recovery
from engine.factor_builder import FactorSandbox
from engine.backtest import FactorBacktester


BASELINE_MOMENTUM = (
    "def alpha_factor(df):\n"
    "    df = df.sort_values(['symbol','date']).copy()\n"
    "    df['ret'] = df.groupby('symbol')['close'].pct_change()\n"
    "    df['factor'] = df.groupby('symbol')['ret'].transform(lambda x: x.rolling(20).sum())\n"
    "    df['factor'] = df.groupby('symbol')['factor'].shift(1)\n"
    "    return df[['date','symbol','factor']]\n"
)


def main():
    ap = argparse.ArgumentParser(description="FactorGPT 对抗式因子恢复验证")
    ap.add_argument("--n-symbols", type=int, default=50)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--noise", type=float, default=0.015)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.02)
    args = ap.parse_args()

    kline, gt, meta = build_adversarial_synthetic(
        n_symbols=args.n_symbols, signal_strength=args.strength,
        noise=args.noise, seed=args.seed,
    )
    print(f"[adversarial] meta={meta}")

    # 1) ground truth 信号下界
    gt_ic, gt_rec = verify_recovery(kline, gt, threshold=args.threshold)
    print(f"[adversarial] ground_truth raw IC={gt_ic:.4f} recovered={gt_rec}")

    # 2) 基准动量因子穿过真实流水线
    sb = FactorSandbox({"engine": {"sandbox": {"subprocess": False}}})
    fac_series = sb.run(BASELINE_MOMENTUM, kline)
    m = FactorBacktester().evaluate(kline, fac_series)
    disc_ic = m.get("ic", 0.0) or 0.0
    print(f"[adversarial] 基准动量因子 流水线 IC={disc_ic:.4f} | 多空Sharpe={m.get('long_short_sharpe')}")

    ok = gt_rec and abs(disc_ic) >= args.threshold
    print(f"[adversarial] 结论：{'PASS（信号可被恢复）' if ok else 'CHECK（信号偏弱，建议调大 strength/调小 noise）'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
