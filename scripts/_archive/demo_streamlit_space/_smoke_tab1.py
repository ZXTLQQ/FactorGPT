"""验证 demo/app.py Tab1 修复后的因子执行链路（沙箱→后处理→回测→图表落盘）。
运行：py -3 demo/_smoke_tab1.py
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from engine.factor_builder import FactorSandbox, build_pipeline
from engine.backtest import FactorBacktester

np.random.seed(42)
n_dates, n_symbols = 200, 60
dates = pd.date_range("2022-01-01", periods=n_dates, freq="B")
symbols = [f"STK_{i:04d}" for i in range(n_symbols)]
data_rows = []
for i, sym in enumerate(symbols):
    drift = np.random.normal(0.0003, 0.0001)
    vol = np.random.uniform(0.015, 0.035)
    price = 10 + abs(np.random.normal(0, 5))
    prices = []
    for _ in range(n_dates):
        price *= (1 + np.random.normal(drift, vol))
        prices.append(price)
    for t, (date, p) in enumerate(zip(dates, prices)):
        data_rows.append({"date": date, "symbol": sym, "close": p})
df = pd.DataFrame(data_rows)

factor_code = '''
def alpha_factor(df):
    """20-day momentum factor with industry-market neutralization."""
    df = df.copy()
    df["factor"] = df.groupby("symbol")["close"].pct_change(20)
    q01, q99 = df["factor"].quantile(0.01), df["factor"].quantile(0.99)
    df["factor"] = df["factor"].clip(q01, q99)
    df["factor"] = df.groupby("date")["factor"].rank(pct=True)
    return df[["date", "symbol", "factor"]].dropna()
'''

# 与 demo/app.py 完全相同的调用方式
sandbox = FactorSandbox({"engine": {"sandbox": {"subprocess": False, "timeout": 60}}})
factor_series = sandbox.run(factor_code, df)
processed = build_pipeline(factor_series, winsorize_pct=0.01)
bt = FactorBacktester(n_quantiles=5, forward_periods=5)
metrics = bt.evaluate(df, processed, verbose=False)
assert "error" not in metrics, metrics.get("error")

charts = []
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
chart_dir = Path(tempfile.gettempdir()) / "factorgpt_demo"
chart_dir.mkdir(exist_ok=True)
figs = bt.plot_metrics(metrics)
for i, fig in enumerate(figs, 1):
    cp = chart_dir / f"demo_chart_{i}.png"
    fig.savefig(str(cp), dpi=110, bbox_inches="tight")
    plt.close(fig)
    charts.append(str(cp))

qr = metrics.get("quantile_returns", {})
top_ret = max(qr.values()) if qr else 0.0
print(f"rank_ic={metrics.get('rank_ic', 0):+.4f} icir={metrics.get('icir', 0):+.3f} "
      f"pos_ratio={metrics.get('ic_positive_ratio', 0):.1%} top_ret={top_ret:+.2%} "
      f"sharpe={metrics.get('long_short_sharpe', 0):+.2f}")
print(f"charts_saved={len(charts)} first={charts[0] if charts else 'N/A'}")
print("TAB1_CHAIN_OK")
