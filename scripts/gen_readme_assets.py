"""
生成 README 配图（真实引擎产出，非手绘）。

使用项目自带的 src/engine/backtest.py 中的 FactorBacktester，
在「带已知信号的合成 A 股面板」上跑回测，导出 PNG 图表到 docs/assets/。

图表规范（符合量化研究行业惯例）：
- 横轴一律使用真实日期（非交易日序号），仅稀疏标注刻度；
- 纵轴带明确单位（IC 无量纲 / 收益 % / 净值）；
- 柱状图/条形图标注具体数值；
- 标题包含关键统计指标（均值 IC、ICIR、年化、Sharpe、最大回撤等）。

说明：数据为合成演示数据（signal-to-noise 受控），仅用于展示
FactorGPT 的因子评价体系与可视化风格，不代表真实选股表现。
"""

from __future__ import annotations

import os
import sys

# 确保项目根目录可被导入（脚本位于 scripts/ 下）
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from src.engine.backtest import FactorBacktester

np.random.seed(20260805)
sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linestyle"] = "--"

# FactorGPT 主题色
BLUE = "#2c7fb8"
GREEN = "#41ab5d"
PINK = "#c51b8a"
RED = "#d62728"
DARK = "#1f2d3d"
GOLD = "#d99c2b"

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")
os.makedirs(ASSET_DIR, exist_ok=True)


def build_synthetic_panel(n_stocks: int = 60, n_days: int = 600, beta: float = 0.006, noise: float = 0.02):
    """构造带已知信号的合成面板（date,symbol,close,volume,amount）与因子序列。

    下一交易日收益 = beta * 因子[t] + noise，使因子对次日收益有可控预测力。
    """
    # 交易日（连续日期）
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    date_strs = dates.strftime("%Y-%m-%d").tolist()
    symbols = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]

    rng = np.random.default_rng(7)
    # 每个因子值用 AR(1) 平滑，避免每日白噪声导致截面相关性被稀释
    rho = 0.92
    factor = rng.standard_normal((n_days, n_stocks))
    for t in range(1, n_days):
        factor[t] = rho * factor[t - 1] + np.sqrt(1 - rho**2) * factor[t]

    # 收益：ret[t] 由 factor[t-1] 驱动（lead-lag），与 evaluate 的 fwd_ret=shift(-1) 对齐
    ret = np.zeros((n_days, n_stocks))
    for t in range(1, n_days):
        sig = beta * factor[t - 1] + noise * rng.standard_normal(n_stocks)
        ret[t] = sig
    # 个股漂移，避免全市场同步
    drift = rng.normal(0.0002, 0.0003, n_stocks)
    ret = ret + drift[None, :]

    close = 10.0 * np.exp(np.cumsum(ret, axis=0))
    volume = rng.integers(1_000_000, 5_000_000, (n_days, n_stocks)).astype(float)
    amount = close * volume * rng.uniform(0.9, 1.1, (n_days, n_stocks))

    rows = []
    fseries = {}
    for ti, d in enumerate(date_strs):
        for si, s in enumerate(symbols):
            rows.append((d, s, close[ti, si], volume[ti, si], amount[ti, si]))
            fseries[(d, s)] = float(factor[ti, si])
    kline = pd.DataFrame(rows, columns=["date", "symbol", "close", "volume", "amount"])
    factor_series = pd.Series(fseries, name="factor")
    return kline, factor_series, dates, symbols


def _date_axis(ax, dates: pd.DatetimeIndex, n_ticks: int = 6):
    """设置横轴为真实日期并稀疏标注刻度。"""
    ax.set_xlim(dates[0], dates[-1])
    loc = mdates.AutoDateLocator(minticks=4, maxticks=n_ticks)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)


def save(fig, name: str):
    path = os.path.join(ASSET_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def main():
    kline, factor, dates, symbols = build_synthetic_panel(beta=0.0008, noise=0.020)

    bt = FactorBacktester(n_quantiles=5, forward_periods=1)
    m = bt.evaluate(kline, factor, verbose=False)

    # 1) IC 时间序列（横轴：日期；纵轴：IC 值；标注均值 IC / ICIR）
    ic = m["_ic_series"]
    ic.index = pd.to_datetime(ic.index)
    ic_mean = m["ic"]
    icir = m["icir"]
    pos_ratio = m.get("ic_positive_ratio", float((ic > 0).mean()))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(ic.index, ic.values, lw=0.8, color=BLUE, label="逐日 IC")
    ax.axhline(ic_mean, color=RED, ls="--", lw=1.2,
               label=f"均值 IC = {ic_mean:.4f}（ICIR = {icir:.2f}）")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_title(f"因子 IC 时间序列（IC 均值 {ic_mean:.4f}，ICIR {icir:.2f}，胜率 {pos_ratio*100:.0f}%）",
                 fontsize=12)
    ax.set_xlabel("日期", fontsize=10)
    ax.set_ylabel("IC（信息系数，逐日截面秩相关）", fontsize=10)
    _date_axis(ax, ic.index)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "ic_series.png")

    # 2) 分位数分组平均收益（横轴：分位组；纵轴：平均次日收益 %；标注数值）
    qr = m["quantile_returns"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    groups = sorted(qr.keys())
    labels = [f"Q{g+1}" for g in groups]
    vals = [qr[g] * 100 for g in groups]  # 转成 %
    colors = [GREEN if v >= 0 else RED for v in vals]
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.5, width=0.6)
    ax.axhline(0, color="black", lw=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + (0.002 if v >= 0 else -0.02),
                f"{v:.2f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_title("分位数分组平均次日收益（Q1 最低 → Q5 最高）", fontsize=12)
    ax.set_xlabel("因子分位组（按因子值截面分组）", fontsize=10)
    ax.set_ylabel("平均次日收益（%）", fontsize=10)
    save(fig, "quantile_returns.png")

    # 3) 多空对冲累计收益（横轴：日期；纵轴：累计收益 %）
    ls = m["_ls_series"]
    ls.index = pd.to_datetime(ls.index)
    eq = (1 + ls).cumprod() - 1
    total = m["long_short_cum_return"]
    sharpe = m.get("long_short_sharpe", np.nan)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(eq.index, eq.values * 100, color=PINK, lw=1.2, label="多空对冲（Q5−Q1）")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.fill_between(eq.index, 0, eq.values * 100, where=(eq.values >= 0),
                    color=PINK, alpha=0.12)
    ax.set_title(f"多空对冲累计收益（Q5−Q1，累计 {total*100:.1f}%，多空 Sharpe = {sharpe:.2f}）",
                 fontsize=12)
    ax.set_xlabel("日期", fontsize=10)
    ax.set_ylabel("累计收益（%）", fontsize=10)
    _date_axis(ax, eq.index)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "long_short.png")

    # 4) 分层累积收益曲线（横轴：日期；纵轴：累积收益 %；图例 Q1-Q5）
    qc = m["quantile_cum"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for g, s in sorted(qc.items()):
        s.index = pd.to_datetime(s.index)
        ax.plot(s.index, s.values * 100, lw=1.0, label=f"Q{g+1}")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title("分层累积收益曲线（按因子分位组，检验收益单调性）", fontsize=12)
    ax.set_xlabel("日期", fontsize=10)
    ax.set_ylabel("累积收益（%）", fontsize=10)
    _date_axis(ax, pd.to_datetime(list(qc.values())[0].index))
    ax.legend(ncol=5, fontsize=8, loc="upper left")
    save(fig, "quantile_cum.png")

    # 5) A 股现实约束组合净值（横轴：日期；纵轴：净值；标注年化/Sharpe/回撤）
    rp = bt.realistic_portfolio(kline, factor, top_frac=0.1)
    nav = rp["equity"]
    nav.index = pd.to_datetime(nav.index)
    mm = rp["metrics"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(nav.index, nav.values, color=DARK, lw=1.2, label="多因子组合净值")
    ax.axhline(1.0, color="gray", lw=0.8, ls="--", label="初始净值 = 1.0")
    ax.set_title(
        f"现实约束组合净值（T+1·涨跌停·交易成本，年化 {mm['ann_return']*100:.1f}%，"
        f"Sharpe {mm['sharpe']:.2f}，最大回撤 {mm['max_drawdown']*100:.1f}%）",
        fontsize=12,
    )
    ax.set_xlabel("日期", fontsize=10)
    ax.set_ylabel("净值（初始 1.0）", fontsize=10)
    _date_axis(ax, nav.index)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "portfolio_nav.png")

    # 6) 多因子 IC 对比（横轴：平均 IC；纵轴：因子名；标注数值）
    np.random.seed(11)
    rng = np.random.default_rng(11)
    factor_defs = {
        "动量 (20D)": 0.0012,
        "价值 (EP)": 0.0008,
        "质量 (ROE)": 0.0010,
        "低波 (IVR)": -0.0006,
        "成长 (PEG)": 0.0005,
        "规模 (市值)": -0.0004,
    }
    ics = []
    for name, b in factor_defs.items():
        _, fac2, _, _ = build_synthetic_panel(beta=abs(b), noise=0.020)
        if b < 0:
            fac2 = -fac2
        mm2 = bt.evaluate(kline, fac2, verbose=False)
        ics.append((name, mm2["ic"], mm2["icir"]))
    ics_sorted = sorted(ics, key=lambda x: x[1], reverse=True)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    names = [x[0] for x in ics_sorted]
    vals = [x[1] for x in ics_sorted]
    icirs = [x[2] for x in ics_sorted]
    colors = [GREEN if v >= 0 else RED for v in vals]
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1],
                   edgecolor="white", linewidth=0.5, height=0.6)
    ax.axvline(0, color="black", lw=0.8)
    for bar, v, iv in zip(bars, vals[::-1], icirs[::-1]):
        ax.text(v + (0.0002 if v >= 0 else -0.0002), bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}  (ICIR {iv:.2f})", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_title("多因子 IC 对比（因子体系概览，按平均 IC 排序）", fontsize=12)
    ax.set_xlabel("平均 IC（信息系数）", fontsize=10)
    ax.set_ylabel("因子名称", fontsize=10)
    ax.set_xlim(min(vals) - 0.001, max(vals) + 0.003)
    save(fig, "factor_ic_bars.png")

    # 打印关键指标摘要，便于写 README
    print("\n=== 主因子指标 ===")
    for k in ["ic", "rank_ic", "icir", "ic_positive_ratio", "long_short_sharpe", "max_drawdown", "coverage"]:
        print(f"  {k}: {m[k]:.4f}")
    print("\n=== 现实组合指标 ===")
    for k, v in mm.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
