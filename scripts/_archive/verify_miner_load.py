# -*- coding: utf-8 -*-
"""验证 miner 模块可加载，且新的本地数据源配置生效。"""
import os
import sys

BASE = r"E:/Qlib/qlib_factorgpt/因子库"
PROJ = r"E:/Qlib/qlib_factorgpt"
sys.path.insert(0, PROJ)
sys.path.insert(0, BASE)

import tools.miner as miner  # noqa: E402

print("miner.mine_today:", hasattr(miner, "mine_today"))
print("EVAL_START:", miner.EVAL_START)
print("EVAL_END:", miner.EVAL_END)
print("UNIVERSE:", miner.UNIVERSE)
print("DATA_SOURCE:", miner.DATA_SOURCE)
print("FWD:", miner.FWD)
print("SUBPROCESS:", miner.SUBPROCESS)

# 假设库
hypos = miner._factor_hypotheses()
print("hypotheses count:", len(hypos))
for h in hypos:
    print("  -", h["key"])

# 本地数据源模拟（不跑完整挖掘）
import data_bridge as db  # noqa: E402
k = db.load_kline(instruments=miner.UNIVERSE, start=miner.EVAL_START, end=miner.EVAL_END, source="local")
print("local kline rows:", len(k), "empty:", k.empty)
print("ALL_LOAD_OK")
