"""端到端验证精炼厂 PART-04/05 修复：用 real_ore.pkl 缓存 + 重复索引候选因子，
确认 screener.screen -> AlphaPool.optimize/synthesize 不再抛 reindex 错误。
运行：py -3 demo/_smoke_pipeline.py
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from pipeline.schema import CandidateFactor
from pipeline.screener import Screener, ScreenerConfig
from pipeline.alpha_pool import AlphaPool, AlphaPoolConfig

CACHE = Path("data/cache/real_ore.pkl")
if not CACHE.exists():
    print("SKIP: real_ore.pkl 不存在，跳过")
    raise SystemExit(0)

with open(CACHE, "rb") as f:
    ore = pickle.load(f)

train_kline = ore.train_kline.drop_duplicates(subset=["date", "symbol"], keep="last")
print(f"train_kline: {train_kline.shape} symbols={train_kline['symbol'].nunique()}")

# 用缓存 factor_pool 构造候选，给其中一个强制插入重复索引
pool_items = list(ore.factor_pool.items())[:6]
rng = np.random.default_rng(7)
cands = []
for i, (name, s) in enumerate(pool_items):
    s = s.copy()
    if s.index.duplicated().any():
        s = s[~s.index.duplicated(keep="last")]
    if i == 2:
        dup = s.iloc[[0]]
        s = pd.concat([s, dup]).sort_index()  # 人为制造重复索引
    cands.append(CandidateFactor(name=f"pool_{i}", source="pool", series=s))

# PART-04: 三级筛选
sc = Screener(ScreenerConfig(use_lasso=True, use_human_collab=False, topk_ratio=0.3, min_keep=2))
screened = sc.screen(cands, train_kline)
print(f"PART-04 screen: {len(cands)} -> {len(screened)}")

# PART-05: AlphaPool 合成
ap = AlphaPool(AlphaPoolConfig(ortho=True, loo=False, iterative=True, n_iter=8))
comp = ap.optimize(screened, train_kline)
print(f"PART-05 optimize: composite len={len(comp)}, dup={comp.index.duplicated().sum()}")

print("PIPELINE_FIX_OK")
