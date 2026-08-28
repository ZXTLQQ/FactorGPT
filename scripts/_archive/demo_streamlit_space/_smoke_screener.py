"""定向验证 screener 修复：带重复 (date,symbol) 索引的候选因子不再使 lasso_filter 崩溃。
运行：py -3 demo/_smoke_screener.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from pipeline.schema import CandidateFactor
from pipeline.screener import Screener, ScreenerConfig

rng = np.random.default_rng(1)
n_dates, n_syms = 120, 30
dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
syms = [f"SYM_{i:02d}" for i in range(n_syms)]
idx = pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"])

kline = pd.DataFrame({"close": 10.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))}, index=idx)
kline = kline.reset_index()
kline["symbol"] = kline["symbol"].astype(str)

# 构造 5 个候选因子，故意给其中一个插入重复索引行
series_list = []
for k in range(5):
    s = pd.Series(rng.normal(0, 1, len(idx)), index=idx, name=f"f{k}")
    if k == 2:
        dup = s.iloc[[0]]
        s = pd.concat([s, dup]).sort_index()
    series_list.append(s)

cands = [CandidateFactor(name=f"f{k}", source="manual", series=s) for k, s in enumerate(series_list)]

sc = Screener(ScreenerConfig(use_lasso=True, use_human_collab=False, topk_ratio=0.3, min_keep=2))
out = sc.lasso_filter(cands, kline)
print(f"lasso_filter OK: {len(cands)} -> {len(out)} candidates")
print("SCREENER_FIX_OK")
