"""
FactorGPT 产品模拟演示脚本（参赛作品"模拟效果"展示用）
- 使用产品自身的回测引擎 engine/backtest.FactorBacktester
- 使用产品自身的 61 个预置传统因子库 engine/traditional_factors
- 在"可复现合成数据 + 离线"下，复现单因子挖掘闭环的指标与图表产出
运行：py -3 demo_sim.py
"""
from __future__ import annotations
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd

from engine.backtest import FactorBacktester
import engine.traditional_factors as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20260808)
N_STOCKS = 300
T_DAYS = 750          # 约 3 年日频
FORWARD = 5           # 持有 5 日

# 1) 构造带"动量信号"的可复现合成行情：潜在信号 s 缓慢漂移，
#    个股未来收益部分由上一期信号驱动 -> 历史动量可预测未来收益
#    （离线演示用合成数据，信号权重较低、噪声较高，使指标接近真实量级）
dates = pd.date_range("2023-01-01", periods=T_DAYS, freq="B").astype(str)
symbols = [f"{i:06d}.SZ" for i in range(N_STOCKS)]

s = RNG.normal(0, 1, size=(T_DAYS, N_STOCKS))
# 信号自相关（缓慢漂移）
for t in range(1, T_DAYS):
    s[t] = 0.9 * s[t-1] + np.sqrt(1 - 0.9**2) * s[t]

ret = np.zeros((T_DAYS, N_STOCKS))
noise = RNG.normal(0, 0.03, size=(T_DAYS, N_STOCKS))
for t in range(1, T_DAYS):
    # 未来 5 日收益由上一期信号驱动（信号领先），权重 0.008，噪声 0.03
    lead = s[t-1]
    ret[t] = 0.008 * lead + noise[t]

# 价格序列
close = 10.0 * np.exp(np.cumsum(ret, axis=0))
# 行业与市值（用于中性化展示）
industry = (np.arange(N_STOCKS) % 10).astype(str)
mktcap = 10 ** RNG.uniform(1.5, 4.5, size=N_STOCKS)

rows = []
for j, sym in enumerate(symbols):
    for t in range(T_DAYS):
        rows.append((dates[t], sym, float(close[t, j]), industry[j], float(mktcap[j])))
kline = pd.DataFrame(rows, columns=["date", "symbol", "close", "industry", "mktcap"])

# 2) 从产品 61 因子库中选取"20 日动量"因子并实际计算
fac_def = tf.get_factor_by_name("momentum_20d")
print("=" * 64)
print("FactorGPT 单因子挖掘模拟 · 离线合成数据")
print("=" * 64)
print(f"选中因子：{fac_def.display_name} ({fac_def.name})")
print(f"方向：{fac_def.direction} | 类别：{fac_def.category}")
print(f"经济学解释：{fac_def.description}")

# 先用 pandas 计算 20 日动量作为因子值（复现 alpha_factor 的核心逻辑）
kline_sorted = kline.sort_values(["symbol", "date"]).copy()
kline_sorted["ret"] = kline_sorted.groupby("symbol")["close"].pct_change()
kline_sorted["mom20"] = kline_sorted.groupby("symbol")["ret"].transform(
    lambda x: x.rolling(20).mean()
)
factor = kline_sorted.dropna(subset=["mom20"]).set_index(["date", "symbol"])["mom20"]

# 3) 调用产品回测引擎
bt = FactorBacktester(n_quantiles=5, forward_periods=FORWARD)
metrics = bt.evaluate(kline, factor, verbose=False)

print("\n" + "=" * 64)
print("因子回测指标（产品 engine/backtest 输出）")
print("=" * 64)
for k in ["ic", "rank_ic", "icir", "ic_positive_ratio", "long_short_return",
          "long_short_sharpe", "long_short_cum_return", "max_drawdown",
          "turnover", "coverage", "n_stocks", "n_dates"]:
    v = metrics.get(k)
    print(f"  {k:22s}: {v}")

print("\n分位数分组（Q1~Q5）未来 {0} 日平均收益：".format(FORWARD))
for g, r in sorted(metrics.get("quantile_returns", {}).items()):
    print(f"  Q{g+1}: {r:+.4f}")

# 4) 生成产品标准图表（IC 序列 / 分位数收益 / 多空权益 / 分层累积）
out_dir = Path("demo_output")
out_dir.mkdir(exist_ok=True)
figs = bt.plot_metrics(metrics)
renamed = []
for i, fig in enumerate(figs, 1):
    p = out_dir / f"factorgpt_sim_{i}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    renamed.append(str(p))
print("\n" + "=" * 64)
print("产品标准化回测图表（已落盘）：")
print("=" * 64)
for p in renamed:
    print(f"  - {p}")

# 5) 另演示"因子库检索"能力（产品 RAG/传统因子库的查询接口）
filt = tf.search_factors(query="反转", category="price_trend")
print("\n" + "=" * 64)
print("因子库检索示例（keyword='反转'）：")
print("=" * 64)
for f in filt[:5]:
    print(f"  - {f.display_name} ({f.name}) | 质量分 {f.quality_score:.2f}")
print(f"\n因子库总量：{tf.get_factor_stats()['total_factors']} 个（五大方向预置）")
