# -*- coding: utf-8 -*-
"""合并 parts 为最终离线数据文件，生成 meta.json / instruments.txt / README.md。

用法:
  python finalize_offline_data.py csi500
  python finalize_offline_data.py all
"""
import os
import sys
import json
import datetime

import pandas as pd

LIB_DIR = r"E:/Qlib/qlib_factorgpt/因子库"
OUT_DIR = os.path.join(LIB_DIR, "历史数据", "2023-2026")

pool = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("all", "csi500") else "csi500"
part_dir = os.path.join(OUT_DIR, "parts_" + pool)

parts = sorted(f for f in os.listdir(part_dir) if f.startswith("part_") and f.endswith(".parquet"))
print(f"[{pool}] 合并 {len(parts)} 个 part...")

frames = [pd.read_parquet(os.path.join(part_dir, p)) for p in parts]
bars = pd.concat(frames, ignore_index=True)
bars = bars.sort_values(["symbol", "date"]).reset_index(drop=True)
print(f"总行数: {len(bars)}, 股票数: {bars['symbol'].nunique()}, 交易日: {bars['date'].nunique()}")

# 统一文件：bars_<pool>.parquet
out_file = os.path.join(OUT_DIR, f"bars_{pool}.parquet")
bars.to_parquet(out_file, index=False)
print(f"写出 {out_file}: {os.path.getsize(out_file)//1024//1024} MB")

# instruments 清单
instr = sorted(bars["symbol"].unique())
with open(os.path.join(OUT_DIR, f"instruments_{pool}.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(instr) + "\n")

# meta
meta = {
    "pool": pool,
    "start": "2023-01-01",
    "end": "2026-08-01",
    "actual_min_date": bars["date"].min(),
    "actual_max_date": bars["date"].max(),
    "rows": int(len(bars)),
    "symbols": int(bars["symbol"].nunique()),
    "trading_days": int(bars["date"].nunique()),
    "source": "Qlib cn_data (前复权)",
    "columns": ["date", "symbol", "open", "high", "low", "close", "volume", "amount"],
    "generated_at": datetime.date.today().isoformat(),
}
meta_path = os.path.join(OUT_DIR, f"meta_{pool}.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("meta:", json.dumps(meta, ensure_ascii=False))

print(f"\n[{pool}] 完成 -> {out_file}")
