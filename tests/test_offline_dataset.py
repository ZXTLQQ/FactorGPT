"""离线数据集契约测试（随仓库分发的 data/offline/ 补充数据）。

覆盖：指数日线、交易日历、多票池成分股的物理契约，以及
``OfflineDataSource`` 对这些文件的实际读取路径——保证"离线可运行"不是文档口号。
测试只读取轻量文件（指数/日历/成分股）与 parquet 元数据，不加载 3.4M 行日K分片。
"""
import json
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OFFLINE = os.path.join(ROOT, "data", "offline")
sys.path.insert(0, os.path.join(ROOT, "src"))

from data.offline_adapter import OfflineDataSource  # noqa: E402

INDEX_FILE = os.path.join(OFFLINE, "index_daily.parquet")


def _meta() -> dict:
    with open(os.path.join(OFFLINE, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def _json(name: str):
    with open(os.path.join(OFFLINE, name), encoding="utf-8") as f:
        return json.load(f)


def _bars_parts():
    return sorted(p for p in os.listdir(OFFLINE)
                  if p.startswith("bars_") and p.endswith(".parquet"))


def test_offline_supplement_files_exist() -> None:
    for name in ("index_daily.parquet", "trade_calendar.json",
                 "constituents_csi300.json", "constituents_csi500.json",
                 "constituents_csi800.json"):
        path = os.path.join(OFFLINE, name)
        assert os.path.exists(path), f"离线补充数据缺失：{name}"
        assert os.path.getsize(path) > 0, f"离线补充数据为空：{name}"
    assert _bars_parts(), "日K分片缺失：bars_*.parquet"
    print("test_offline_supplement_files_exist OK")


def test_index_daily_schema_and_integrity() -> None:
    df = pd.read_parquet(INDEX_FILE)
    for col in ("instrument", "index_name", "date", "close", "high", "low",
                "volume", "amount", "pct_chg"):
        assert col in df.columns, f"index_daily.parquet 缺少列 {col}"
    assert not df.isna().any().any(), "index_daily.parquet 存在缺失值"
    assert (df["close"] > 0).all() and (df["high"] >= df["low"]).all(), "指数点位异常"
    meta = _meta()
    assert df["instrument"].nunique() == len(meta["indices"]["codes"]), \
        "指数数量与 meta.indices.codes 不一致"
    counts = df.groupby("instrument")["date"].size()
    assert counts.min() >= 0.95 * meta["calendar"]["trade_days"], \
        f"部分指数交易日覆盖不足：{counts.to_dict()}"
    assert str(df["date"].min())[:10] >= meta["start"], "指数日线早于声明区间"
    assert str(df["date"].max())[:10] <= meta["end"], "指数日线晚于声明区间"
    print("test_index_daily_schema_and_integrity OK")


def test_trade_calendar_matches_meta() -> None:
    cal = _json("trade_calendar.json")
    meta = _meta()
    assert len(cal) == meta["calendar"]["trade_days"] == meta["trade_days"], \
        "交易日历长度与 meta 不一致"
    assert cal[0] == meta["start"] and cal[-1] == meta["end"], "交易日历边界与 meta 不一致"
    assert cal == sorted(set(cal)), "交易日历存在重复或乱序"
    assert len(cal) > 1000, "交易日历异常偏少"
    print("test_trade_calendar_matches_meta OK")


def test_constituents_subsets_and_meta_counts() -> None:
    meta = _meta()
    pools = {p: set(_json(f"constituents_{p}.json"))
             for p in ("csi300", "csi500", "csi800")}
    for pool, codes in pools.items():
        assert codes, f"{pool} 成分股为空"
        assert all(len(c) == 8 and c[:2] in ("SH", "SZ", "BJ") for c in codes), \
            f"{pool} 成分股代码格式异常"
        assert len(codes) == meta["constituents"][pool], f"{pool} 数量与 meta 不一致"
    assert pools["csi300"] <= pools["csi800"], "csi300 不是 csi800 的子集"
    assert pools["csi500"] <= pools["csi800"], "csi500 不是 csi800 的子集"
    print("test_constituents_subsets_and_meta_counts OK")


def test_offline_adapter_reads_supplements() -> None:
    ds = OfflineDataSource({"data": {"offline": {"index": "csi800"}}})
    idx = ds.get_index_daily("000906", "2024-01-01", "2024-01-10")
    assert not idx.empty and idx["date"].is_monotonic_increasing, "指数日线读取失败"
    assert ds.last_fetch_info["source"] == "offline"
    assert ds.get_index_daily("399006").empty, "未收录指数应返回空而非联网"
    assert ds.last_fetch_info["source"] == "none"
    assert len(ds.get_trade_calendar("2024-01-01", "2024-12-31")) == 242, "2024 交易日数异常"
    assert len(ds.get_index_constituents("000300")) == _meta()["constituents"]["csi300"], \
        "按指数代码切换票池失败"
    assert len(ds.get_index_constituents("csi500")) == _meta()["constituents"]["csi500"], \
        "按票池名切换票池失败"
    print("test_offline_adapter_reads_supplements OK")


def test_meta_rows_match_parquet_parts() -> None:
    meta = _meta()
    rows = sum(pq.ParquetFile(os.path.join(OFFLINE, p)).metadata.num_rows
               for p in meta["parts"])
    assert rows == meta["rows"], f"meta.rows={meta['rows']} 与实际分片行数 {rows} 不一致"
    schema = pq.ParquetFile(os.path.join(OFFLINE, meta["parts"][0])).schema_arrow
    assert "instrument" in schema.names, "日K分片缺少 instrument 列"
    print("test_meta_rows_match_parquet_parts OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"{t.__name__} FAIL: {e}")
    print(f"\n=== offline dataset: failed={failed}/{len(tests)} ===")
    sys.exit(1 if failed else 0)
