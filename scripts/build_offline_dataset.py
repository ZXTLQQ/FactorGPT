"""离线数据补充导出（从本地 Qlib 行情库生成 data/offline/ 补充数据集）。

在既有的 ``bars_<index>_part*.parquet`` 日K分片之外，补充以下随仓库分发的文件，
使 ``OfflineDataSource`` 具备基准指数、交易日历与多票池能力（全部离线可用）::

    data/offline/index_daily.parquet      # 主要宽基指数日线（含中证800基准）
    data/offline/trade_calendar.json      # 区间内交易日历
    data/offline/constituents_<pool>.json # csi300/csi500/csi1000 等票池成分股

用法::

    python scripts/build_offline_dataset.py --qlib-dir E:/Qlib/data/cn_data

直接读取 Qlib 原生 ``.bin``（numpy float32 + 起始日历下标），不依赖 qlib 包。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "offline"
DEFAULT_QLIB = "E:/Qlib/data/cn_data"
START = "2019-01-02"
END = "2026-08-25"

INDEXES: Dict[str, str] = {
    "SH000300": "沪深300",
    "SH000905": "中证500",
    "SH000906": "中证800",
    "SH000852": "中证1000",
    "SH000985": "中证全指",
}
POOLS = ["csi300", "csi500", "csi1000", "csi800", "csiall"]

MIN_COVERAGE = 0.95  # 指数在区间内的最小覆盖率（低于此值视为数据不完整，剔除）


def read_calendar(qlib: Path) -> List[str]:
    with open(qlib / "calendars" / "day.txt", encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def read_bin(qlib: Path, inst: str, field: str, cal: List[str]):
    p = qlib / "features" / inst.lower() / f"{field}.day.bin"
    if not p.exists():
        return [], np.array([], dtype="float32")
    arr = np.fromfile(p, dtype="<f4")
    start = int(arr[0])
    vals = arr[1:]
    return cal[start:start + len(vals)], vals


def read_instruments(qlib: Path, pool: str) -> List[str]:
    out: List[str] = []
    p = qlib / "instruments" / f"{pool}.txt"
    with open(p, encoding="utf-8") as fh:
        for ln in fh:
            parts = ln.split()
            if parts:
                out.append(parts[0].upper())
    return sorted(set(out))


def build_index_bars(qlib: Path, cal: List[str]) -> pd.DataFrame:
    rows = []
    for inst, name in INDEXES.items():
        dates, close = read_bin(qlib, inst, "close", cal)
        _, factor = read_bin(qlib, inst, "factor", cal)
        _, high = read_bin(qlib, inst, "high", cal)
        _, low = read_bin(qlib, inst, "low", cal)
        _, volume = read_bin(qlib, inst, "volume", cal)
        _, amount = read_bin(qlib, inst, "amount", cal)
        _, chg = read_bin(qlib, inst, "change", cal)  # 分数（0.0118 = +1.18%）
        # Qlib 指数 OHLC 以"千点"为单位（factor≈0.001），需折算回真实点位
        mask = (close > 0) & (factor > 0)
        scale = np.where(factor > 0, 1.0 / factor, np.nan)
        rows.append(pd.DataFrame({
            "instrument": inst,
            "index_name": name,
            "date": np.asarray(dates)[mask],
            "close": (close * scale)[mask],
            "high": (high * scale)[mask],
            "low": (low * scale)[mask],
            "volume": volume[mask],
            "amount": amount[mask],
            "pct_chg": (chg * 100.0)[mask],
        }))
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["pct_chg"] = out["pct_chg"].fillna(0.0)
    return out


def scan_bars() -> tuple:
    """扫描已分发的日K parquet，返回 (代码集合, 总行数)。"""
    insts: set = set()
    rows = 0
    for p in sorted(OUT_DIR.glob("bars_*_part*.parquet")):
        col = pd.read_parquet(p, columns=["instrument"])["instrument"]
        insts |= {str(x).upper() for x in col.unique()}
        rows += len(col)
    return insts, rows


def build_constituents(qlib: Path, available: set) -> Dict[str, List[str]]:
    """各票池成分股：仅保留日K已覆盖的代码，避免产出"有名单无行情"的空池。"""
    out: Dict[str, List[str]] = {}
    for pool in POOLS:
        if not (qlib / "instruments" / f"{pool}.txt").exists():
            continue
        out[pool] = [i for i in read_instruments(qlib, pool) if i in available]
    return out


def write_outputs(index_bars: pd.DataFrame, constituents: Dict[str, List[str]],
                  cal: List[str]) -> Dict[str, object]:
    """写出补充数据集，返回 meta.json 的增量字段。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_bars.sort_values(["instrument", "date"]).reset_index(drop=True).to_parquet(
        OUT_DIR / "index_daily.parquet", index=False)
    for pool, codes in constituents.items():
        (OUT_DIR / f"constituents_{pool}.json").write_text(
            json.dumps(codes), encoding="utf-8")
    (OUT_DIR / "trade_calendar.json").write_text(
        json.dumps(cal), encoding="utf-8")
    return {
        "indices": {
            "file": "index_daily.parquet",
            "codes": sorted(index_bars["instrument"].unique().tolist()),
            "names": {k: v for k, v in INDEXES.items()
                      if k in set(index_bars["instrument"].unique())},
            "rows": len(index_bars),
        },
        "constituents": {p: len(c) for p, c in sorted(constituents.items())},
        "calendar": {
            "file": "trade_calendar.json",
            "start": cal[0] if cal else None,
            "end": cal[-1] if cal else None,
            "trade_days": len(cal),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 data/offline/ 补充离线数据集")
    ap.add_argument("--qlib-dir", default=DEFAULT_QLIB)
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    args = ap.parse_args()

    qlib = Path(args.qlib_dir)
    full_cal = read_calendar(qlib)  # .bin 内的起始下标基于完整日历，不能裁剪
    cal = [d for d in full_cal if args.start <= d <= args.end]
    if not cal:
        raise SystemExit(f"日历为空：{qlib}/calendars/day.txt 与 [{args.start}, {args.end}] 无交集")
    print(f"交易日 {len(cal)} 天：{cal[0]} ~ {cal[-1]}")

    bars = build_index_bars(qlib, full_cal)
    bars = bars[(bars["date"] >= args.start) & (bars["date"] <= args.end)]
    # 剔除区间内存在缺失/零值挡位的指数（避免产出"断裂"基准曲线）
    n_days = bars.groupby("instrument")["date"].size()
    keep = n_days[n_days >= MIN_COVERAGE * len(cal)].index
    dropped = sorted(set(bars["instrument"].unique()) - set(keep))
    if dropped:
        print(f"跳过区间内数据不完整的指数：{dropped}")
    bars = bars[bars["instrument"].isin(keep)].reset_index(drop=True)
    print(f"指数日线 {len(bars)} 行，覆盖 {bars['instrument'].nunique()} 个指数")

    available, bar_rows = scan_bars()
    cons = build_constituents(qlib, available)
    for pool, codes in cons.items():
        print(f"成分股 {pool}: {len(codes)} 只（已对齐日K覆盖）")

    meta_patch = write_outputs(bars, cons, cal)
    meta_patch["symbols"] = len(available)
    meta_patch["rows"] = bar_rows
    meta_patch["generator"] = "scripts/build_offline_dataset.py"
    meta_path = OUT_DIR / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.update(meta_patch)
    meta["updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出 meta.json：{sorted(meta_patch)}")


if __name__ == "__main__":
    main()
