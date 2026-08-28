# -*- coding: utf-8 -*-
"""将 Qlib cn_data 中的 2023-01-01 ~ 2026-08-01 历史行情导出为离线静态数据，
落地到因子库目录下的 历史数据/2023-2026/。

特点：
  - 分批增量落盘 + 断点续传（每批写入 part_*.parquet，progress.json 记录已完成代码）
  - 带 __main__ 保护，避免 Windows multiprocessing spawn 重复执行
  - 导出完成后可用 finalize.py 合并 part 并生成 meta.json / README

用法:
  python export_offline_data.py csi500   # 导出 csi500（默认，快）
  python export_offline_data.py all      # 导出全市场（分批，慢）
  python export_offline_data.py csi500 --resume   # 续传
  python export_offline_data.py csi500 --clean    # 清空该池旧产物重来
"""
import os
import sys
import json
import time

import qlib
from qlib.data import D
import pandas as pd

QLIB_PROVIDER = r"E:/Qlib/data/cn_data"
LIB_DIR = r"E:/Qlib/qlib_factorgpt/因子库"
OUT_DIR = os.path.join(LIB_DIR, "历史数据", "2023-2026")

START = "2023-01-01"
END = "2026-08-01"  # 含 2026-07-31 最后一个交易日
FIELDS = ["$open", "$high", "$low", "$close", "$volume", "$amount"]
BATCH = 300


def main():
    pool = "csi500"
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args and args[0] in ("all", "csi500"):
        pool = args[0]
    resume = "--resume" in sys.argv
    clean = "--clean" in sys.argv

    qlib.init(provider_uri=QLIB_PROVIDER, region="cn")
    os.makedirs(OUT_DIR, exist_ok=True)
    part_dir = os.path.join(OUT_DIR, "parts_" + pool)
    if clean and os.path.isdir(part_dir):
        for f in os.listdir(part_dir):
            os.remove(os.path.join(part_dir, f))
        print(f"[{pool}] 已清空旧产物 {part_dir}")
    os.makedirs(part_dir, exist_ok=True)
    progress_path = os.path.join(part_dir, "progress.json")

    # 1) 股票池（membership 区间与目标区间重叠的代码）
    codes = []
    with open(os.path.join(QLIB_PROVIDER, "instruments", pool + ".txt"), "r", encoding="utf-8") as f:
        for ln in f:
            parts = ln.split()
            if not parts:
                continue
            if len(parts) >= 3:
                code, s, e = parts[0], parts[1], parts[2]
                if s <= END and e >= START:
                    codes.append(code)
            else:
                codes.append(parts[0])
    codes = sorted(set(codes))
    print(f"[{pool}] 目标区间内股票数: {len(codes)}")

    # 2) 断点续传
    done = set()
    if resume and os.path.exists(progress_path):
        done = set(json.load(open(progress_path, "r", encoding="utf-8")))
        print(f"续传: 已完成 {len(done)} 只")
    pending = [c for c in codes if c not in done]
    print(f"待处理: {len(pending)} 只")

    # 3) 分批处理，每批落盘一个 part parquet
    def process_batch(batch_codes, part_idx):
        t0 = time.time()
        try:
            raw = D.features(batch_codes, FIELDS, start_time=START, end_time=END)
        except Exception as e:  # noqa: BLE001
            print(f"  batch {part_idx} bulk failed: {e!r}; 逐只回退")
            sub = []
            for c in batch_codes:
                try:
                    one = D.features([c], FIELDS, start_time=START, end_time=END)
                    if not one.empty:
                        sub.append(one)
                except Exception as ee:  # noqa: BLE001
                    print(f"    {c} failed: {ee!r}")
            raw = pd.concat(sub) if sub else pd.DataFrame()
        if raw is None or raw.empty:
            return 0
        df = raw.reset_index().rename(columns={"datetime": "date", "instrument": "symbol"})
        df["date"] = df["date"].astype(str)
        df = df.rename(columns={
            "$open": "open", "$high": "high", "$low": "low",
            "$close": "close", "$volume": "volume", "$amount": "amount",
        })
        df = df[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]]
        df = df.dropna(subset=["close"])
        part_path = os.path.join(part_dir, f"part_{part_idx:04d}.parquet")
        df.to_parquet(part_path, index=False)
        dt = time.time() - t0
        print(f"  part {part_idx}: {len(df)} 行 / {df['symbol'].nunique()} 只, 耗时 {dt:.0f}s")
        return len(df)

    part_idx = 0
    existing = [f for f in os.listdir(part_dir) if f.startswith("part_")]
    if existing:
        part_idx = max(int(f.split("_")[1].split(".")[0]) for f in existing) + 1

    for i in range(0, len(pending), BATCH):
        batch = pending[i:i + BATCH]
        process_batch(batch, part_idx)
        done.update(batch)
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(sorted(done), f, ensure_ascii=False)
        part_idx += 1
        print(f"  进度: {len(done)}/{len(codes)}")

    print(f"\n[{pool}] 分批导出完成，parts 位于 {part_dir}")


if __name__ == "__main__":
    main()
