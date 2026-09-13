"""NeoData markdown 表格解析契约测试（纯离线，不触网）。

样本取自 2026-09-13 对真实网关的抓包落盘（scripts 探针），覆盖：
表格切分/省略占位行识别、K 线映射与列契约、指数成分股的脏数据回归（旧实现会把
指数自身代码混入成分股）、分块预算控制、以及「残缺数据一律回退」的核心原则。
"""
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data.neo_adapter import (  # noqa: E402
    NeoDataSource,
    _num,
    _parse_md_tables,
)

# --- 真实抓包样本（节选） ---------------------------------------------------

KLINE_5D = """### 600519.SH 贵州茅台

市场: 沪市 | 币种: CNY | 复权: 前复权 | 接口查询时间: 2026-09-13 15:17:32 | 交易状态: 已休市

2024-01-28: 未开盘

2024-01-27: 未开盘

| 日期 | 开盘 | 最高 | 最低 | 收盘/最新 | 前收 | 涨跌额 | 涨跌幅 | 振幅 | 成交量 | 成交额 | 换手率 | 量比 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-01-31 | 1478.66 | 1485.63 | 1463.79 | 1470.57 | 1478.66 | -8.09 | -0.55% | 1.48% | 2964900.00 | 4770104891.00 | 0.2400% | -- |
| 2024-01-30 | 1515.64 | 1515.64 | 1475.66 | 1478.66 | 1515.66 | -37.00 | -2.44% | 2.64% | 3086600.00 | 5010784454.00 | 0.2500% | -- |
| 2024-01-29 | 1518.00 | 1525.00 | 1502.00 | 1515.66 | 1518.00 | -2.34 | -0.15% | 1.51% | 2258700.00 | 3425438300.00 | 0.1800% | -- |
| 2024-01-26 | 1520.00 | 1530.00 | 1510.00 | 1518.00 | 1520.00 | -2.00 | -0.13% | 1.32% | 2103400.00 | 3197817600.00 | 0.1700% | -- |
| 2024-01-25 | 1508.00 | 1525.66 | 1505.00 | 1520.00 | 1508.00 | 12.00 | 0.80% | 1.37% | 2445600.00 | 3701286500.00 | 0.1900% | -- |
"""

# 长区间：服务端在表格中间插入省略占位行（行数不随区间增长）
KLINE_TRUNCATED = """### 000001.SZ 平安银行

| 日期 | 开盘 | 最高 | 最低 | 收盘/最新 | 前收 | 涨跌额 | 涨跌幅 | 振幅 | 成交量 | 成交额 | 换手率 | 量比 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024-01-31 | 9.30 | 9.36 | 9.25 | 9.32 | 9.30 | 0.02 | 0.22% | 1.18% | 102030400.00 | 950123400.00 | 0.5300% | -- |
| 2024-01-30 | 9.35 | 9.38 | 9.28 | 9.30 | 9.35 | -0.05 | -0.53% | 1.07% | 98123400.00 | 913400200.00 | 0.5100% | -- |
| 2024-01-02 ~ 2024-01-22 | 省略中间历史行情，不包含任何具体数值，禁止根据该标记推断、补全或参与计算 | -- |
| 2024-01-01 | 9.10 | 9.15 | 9.05 | 9.12 | 9.10 | 0.02 | 0.22% | 1.10% | 76543200.00 | 700123400.00 | 0.4000% | -- |
"""

CONSTITUENTS_PARTIAL = """### 000906.SH 中证800

| 指数代码 | 指数名称 | 发布日期 | 成份证券代码 | 成份证券名称 | 权重(%) |
| :--- | :--- | :--- | :--- | :--- | ---: |
| 000906.SH | 中证800 | 20260831 | 600519.SH | 贵州茅台 | 2.654 |
| 000906.SH | 中证800 | 20260831 | 300750.SZ | 宁德时代 | 1.912 |
| 000906.SH | 中证800 | 20260831 | 601318.SH | 中国平安 | 1.503 |
| 000906.SH | 中证800 | 20260831 | 000001.SZ | 平安银行 | 0.612 |
"""

CONSTITUENTS_FULL = """| 指数代码 | 指数名称 | 成份证券代码 | 成份证券名称 | 权重(%) |
| :--- | :--- | :--- | :--- | ---: |
""" + "\n".join(
    f"| 000906.SH | 中证800 | {600000 + i}.SH | 股票{i} | 0.10 |" for i in range(60)
)

PROFIT_TABLE = """### 600519.SH 贵州茅台 利润表

| 发布日期 | 报告期 | 报表类型 | 营业收入 | 净利润 | 资产负债率 | 净资产收益率 |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| 2026-04-17 | 2025-12-31 | 2025-FY | 191_234_567_890.00 | 86_123_456_789.00 | 16.53% | 32.10% |
| 2025-04-16 | 2024-12-31 | 2024-FY | -- | -- | -- | -- |
"""


def _resp(*blocks):
    """构造 NeoData 风格的响应体。"""
    return {"data": {"apiData": {"apiRecall": [
        {"type": t, "content": c} for t, c in blocks]}}}


def _ds(**neo_cfg):
    cfg = {"base_url": "https://example.invalid/neodata"}
    cfg.update(neo_cfg)
    return NeoDataSource({"data": {"neodata": cfg}})


# --- 表格解析 ---------------------------------------------------------------

def test_parse_md_tables_basic() -> None:
    tables = _parse_md_tables(KLINE_5D)
    assert len(tables) == 1, f"应解析出 1 张表，实际 {len(tables)}"
    tb = tables[0]
    assert tb.columns[0] == "日期" and "收盘/最新" in tb.columns, "表头解析错误"
    assert len(tb.rows) == 5, f"数据行数应为 5，实际 {len(tb.rows)}"
    assert not tb.truncated, "完整表格不应被标记为截断"
    # 表格外的「未开盘」行不得混入数据行
    assert all("未开盘" not in " ".join(r) for r in tb.rows), "非交易行混入表格"
    # 两个独立表格应被切分为两张
    assert len(_parse_md_tables(KLINE_5D + "\n\n" + CONSTITUENTS_PARTIAL)) == 2, "多表切分失败"
    print("test_parse_md_tables_basic OK")


def test_parse_md_tables_marks_omitted_rows() -> None:
    tb = _parse_md_tables(KLINE_TRUNCATED)[0]
    assert tb.truncated, "含「省略中间历史行情」占位行时必须标记 truncated"
    joined = " ".join(" ".join(r) for r in tb.rows)
    assert "省略" not in joined, "省略占位行必须被丢弃，不能当成数据行"
    assert not any("~" in r[0] for r in tb.rows), "区间占位行不得进入数据行"
    # 占位行前后的真实行都应保留
    dates = [r[0] for r in tb.rows]
    assert dates == ["2024-01-31", "2024-01-30", "2024-01-01"], f"占位行两侧数据丢失：{dates}"
    print("test_parse_md_tables_marks_omitted_rows OK")


def test_num_cell_cleaning() -> None:
    assert _num("1,234.56") == 1234.56
    assert _num("-0.55%") == -0.55
    assert _num("+12.30") == 12.30
    for na in ("--", "", "-", "N/A", "无"):
        assert pd.isna(_num(na)), f"{na!r} 应解析为 NaN"
    print("test_num_cell_cleaning OK")


# --- K 线映射 ---------------------------------------------------------------

def test_map_kline_complete_window() -> None:
    ds = _ds()
    df = ds._map_kline(_resp(("统一行情查询", KLINE_5D)), "600519")
    assert df is not None and len(df) == 5, "完整窗口应解析出 5 行"
    for col in ("date", "open", "high", "low", "close", "volume", "amount",
                "pct_chg", "symbol"):
        assert col in df.columns, f"缺少标准列 {col}"
    assert df["date"].is_monotonic_increasing, "日期必须升序"
    assert df["symbol"].eq("600519").all(), "symbol 未回填"
    assert df["close"].iloc[-1] == 1470.57, "收盘价解析错误"
    assert df["pct_chg"].iloc[-1] == -0.55, "涨跌幅应保留百分数数值（-0.55）"
    assert df["volume"].iloc[-1] == 2964900.0, "成交量解析错误"
    assert set(df["date"]) == {"2024-01-25", "2024-01-26", "2024-01-29", "2024-01-30", "2024-01-31"}
    print("test_map_kline_complete_window OK")


def test_map_kline_rejects_truncated() -> None:
    ds = _ds()
    assert ds._map_kline(_resp(("统一行情查询", KLINE_TRUNCATED)), "000001") is None, \
        "含省略占位行时必须判不可用（残缺时序绝不能进回测）"
    loose = _ds(allow_partial=True)
    df = loose._map_kline(_resp(("统一行情查询", KLINE_TRUNCATED)), "000001")
    assert df is not None and len(df) == 3, "allow_partial 时应返回残缺行（研究性用途）"
    print("test_map_kline_rejects_truncated OK")


def test_map_kline_ignores_unrelated_blocks() -> None:
    ds = _ds()
    assert ds._map_kline(_resp(("板块涨跌排行", CONSTITUENTS_PARTIAL)), "600519") is None, \
        "无关表格不应被误当 K 线"
    assert ds._map_kline({}, "600519") is None, "空响应应返回 None"
    print("test_map_kline_ignores_unrelated_blocks OK")


# --- 指数成分股 -------------------------------------------------------------

def test_map_index_constituents_no_dirty_codes() -> None:
    """回归：旧实现用全文正则抓 6 位数字，会把指数代码 000906 混进成分股。"""
    ds = _ds(allow_partial=True)
    codes = ds._map_index_constituents(_resp(("指数成分及权重", CONSTITUENTS_PARTIAL)))
    assert codes == ["600519", "300750", "601318", "000001"], f"成分股解析错误：{codes}"
    assert "000906" not in codes, "指数自身代码不得混入成分股"
    print("test_map_index_constituents_no_dirty_codes OK")


def test_map_index_constituents_requires_complete_list() -> None:
    ds = _ds()
    assert ds._map_index_constituents(_resp(("指数成分及权重", CONSTITUENTS_PARTIAL))) is None, \
        "只返回前几只时必须判不可用（残缺票池会静默缩小回测池）"
    full = ds._map_index_constituents(_resp(("指数成分及权重", CONSTITUENTS_FULL)))
    assert full is not None and len(full) == 60, "完整成分股应全部解析"
    assert full[0] == "600000" and len(set(full)) == 60, "成分股去重或顺序异常"
    print("test_map_index_constituents_requires_complete_list OK")


# --- 分块预算 ---------------------------------------------------------------

def test_chunk_windows_budget_and_coverage() -> None:
    ds = _ds(max_chunk_requests=8)
    # 短窗口：2 个分块，落在预算内
    wins = ds._chunk_windows("2024-01-25", "2024-01-31", 1)
    assert wins is not None and len(wins) <= 8, f"短窗口应可分块：{wins}"
    assert wins[0][0] == "2024-01-25" and wins[-1][1] == "2024-01-31", "分块未覆盖请求区间"
    for (a, b), (c, d) in zip(wins, wins[1:]):
        assert (pd.Timestamp(c) - pd.Timestamp(b)).days == 1, "分块不连续"

    # 半年 = 约 131 个交易日 -> 27 块，超预算，必须整体放弃
    assert ds._chunk_windows("2024-01-01", "2024-06-30", 1) is None, \
        "长区间必须判定不可用（否则会打出海量请求）"
    # 标的数放大请求数
    assert ds._chunk_windows("2024-01-25", "2024-01-31", 100) is None, \
        "批量标的必须判定不可用"

    big = _ds(max_chunk_requests=200)
    wins = big._chunk_windows("2024-01-01", "2024-06-30", 1)
    assert wins is not None and len(wins) <= 200, "放宽预算后应可分块"
    covered = sum((pd.Timestamp(b) - pd.Timestamp(a)).days + 1 for a, b in wins)
    assert covered >= (pd.Timestamp("2024-06-30") - pd.Timestamp("2024-01-01")).days + 1, \
        "分块覆盖不完整"
    assert ds._chunk_windows("2024-01-31", "2024-01-01", 1) is None, "反向区间应返回 None"
    print("test_chunk_windows_budget_and_coverage OK")


def test_skip_reason_reported_in_fallback() -> None:
    """超预算时应给出「超预算」原因，而非笼统的「解析为空」（关闭回退以免触网）。"""
    ds = _ds(max_chunk_requests=8, fallback_to_legacy=False)
    out = ds.get_daily_kline(["600519"], "2020-01-01", "2024-12-31")
    info = ds.last_fetch_info
    assert info["source"] == "none", f"关闭回退时应返回空并标记 none，实际 {info}"
    assert "预算" in info["message"], f"未报告超预算原因：{info}"
    assert out is not None and out.empty, "超预算时应返回空 DataFrame"
    print("test_skip_reason_reported_in_fallback OK")


# --- 财务与固定回退 ---------------------------------------------------------

def test_map_financials_from_profit_table() -> None:
    ds = _ds()
    df = ds._map_financials(_resp(("利润表", PROFIT_TABLE)), "600519")
    assert df is not None and len(df) == 1, "应解析出最新报告期一行"
    assert df["symbol"].iloc[0] == "600519"
    assert df["报告期"].iloc[0] == "2025-12-31", "应取最新报告期"
    assert df["报表类型"].iloc[0] == "2025-FY"
    # `--` 缺失值不应写入结果
    assert all("--" != str(v).strip() for v in df.iloc[0].tolist()), "缺失值 -- 未被剔除"
    assert ds._map_financials({}, "600519") is None, "空响应应返回 None"
    print("test_map_financials_from_profit_table OK")


def test_industry_and_news_fixed_fallback() -> None:
    """行业映射与新闻无 NeoData 能力，必须稳定返回 None（诚实回退）。"""
    ds = _ds()
    assert ds._map_industry(_resp(("指数行业分布", CONSTITUENTS_PARTIAL))) is None, \
        "残缺的板块排行不得当成完整行业映射"
    assert ds._map_industry(_resp(("板块涨跌排行", CONSTITUENTS_PARTIAL))) is None
    assert ds._map_news({}) is None, "空响应应返回 None"
    print("test_industry_and_news_fixed_fallback OK")


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
    print(f"\n=== neo adapter parse: failed={failed}/{len(tests)} ===")
    sys.exit(1 if failed else 0)
