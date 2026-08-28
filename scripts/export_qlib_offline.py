# -*- coding: utf-8 -*-
"""从 Qlib cn_data 导出 FactorGPT 离线数据源（parquet）。

用法（必须用带 qlib 的 Python 运行，本机为 E:\\Qlib\\runtime\\python311\\python.exe）::

    E:\\Qlib\\runtime\\python311\\python.exe scripts/export_qlib_offline.py

产物（写入 data/offline/，已被 .gitignore 忽略）::

    data/offline/bars_csi800.parquet    # 全量日K: instrument/date/open/high/low/close/volume/amount/factor
    data/offline/constituents_csi800.json  # 指数成分股列表
    data/offline/meta.json              # 导出元信息（时间范围、股票数、交易日数）

之后在 config.yaml 设置 ``data.source: offline`` 即可让全流水线
(DataFetcher 接口) 完全离线读取本地 Qlib 数据，不触网。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "offline"
QLIB_URI = "E:/Qlib/data/cn_data"
INDEX = "csi800"
START = "2019-01-01"
END = "2026-12-31"
BATCH = 400  # 每批读取的股票数（控制内存）


def main() -> int:
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=QLIB_URI, region="cn")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    insts = D.instruments(market=INDEX)
    codes = D.list_instruments(insts, as_list=True)
    codes = sorted(codes)
    print(f"[export_qlib_offline] {INDEX} 成分股数量: {len(codes)}")

    # 日历范围
    cal = [str(d)[:10] for d in D.calendar(start_time=START, end_time=END, freq="day")]
    print(f"[export_qlib_offline] 交易日: {len(cal)} ({cal[0]} ~ {cal[-1]})")

    fields = ["$open", "$high", "$low", "$close", "$volume", "$amount", "$factor"]
    frames = []
    t0 = time.time()
    for i in range(0, len(codes), BATCH):
        batch = codes[i : i + BATCH]
        try:
            df = D.features(batch, fields, start_time=START, end_time=END, freq="day")
        except Exception as e:  # noqa: BLE001
            print(f"[export_qlib_offline] 批次 {i}-{i + len(batch)} 失败: {e}")
            continue
        if df is not None and len(df):
            frames.append(df.reset_index())
        print(
            f"[export_qlib_offline] 已处理 {min(i + BATCH, len(codes))}/{len(codes)} "
            f"({time.time() - t0:.1f}s)"
        )
        del df

    if not frames:
        print("[export_qlib_offline] 未导出任何数据，退出")
        return 1

    import pandas as pd

    bars = pd.concat(frames, ignore_index=True)

    # 列名规范化为 FactorGPT 约定
    bars = bars.rename(
        columns={
            "$open": "open",
            "$high": "high",
            "$low": "low",
            "$close": "close",
            "$volume": "volume",
            "$amount": "amount",
            "$factor": "factor",
        }
    )
    bars["date"] = bars["datetime"].astype(str).str[:10]
    bars = bars.drop(columns=["datetime"]).sort_values(["instrument", "date"]).reset_index(drop=True)

    parquet_path = OUT_DIR / f"bars_{INDEX}.parquet"
    bars.to_parquet(parquet_path, index=False)
    print(f"[export_qlib_offline] 已写入 {parquet_path}  行数: {len(bars)}")

    with open(OUT_DIR / f"constituents_{INDEX}.json", "w", encoding="utf-8") as f:
        json.dump(codes, f, ensure_ascii=False, indent=1)
    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "qlib_uri": QLIB_URI,
                "index": INDEX,
                "start": cal[0],
                "end": cal[-1],
                "trade_days": len(cal),
                "symbols": len(codes),
                "rows": int(len(bars)),
                "columns": list(bars.columns),
                "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("[export_qlib_offline] 完成。可在 config.yaml 设置 data.source=offline 启用离线数据源。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
