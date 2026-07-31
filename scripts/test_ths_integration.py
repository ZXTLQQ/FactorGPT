"""
同花顺 iFinD MCP 接入端到端验证

1) 握手 + 工具发现
2) 用配置的 ths_symbols 小宇宙逐只拉取日K线（真实数据）
3) 解析为标准 DataFrame 后，跑一个最简单的动量因子（20日收益率），
   计算 IC 以证明「数据 → 因子」链路可真正跑通。

用法：
    python scripts/test_ths_integration.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    from llm.client import load_config

    cfg = load_config().get("data", {})
    token = os.environ.get("THS_API_TOKEN") or cfg.get("ths_api_token", "")
    base_url = os.environ.get("THS_API_BASE_URL") or cfg.get("ths_api_base_url", "")
    symbols = cfg.get("ths_symbols") or []
    start = cfg.get("default_start_date", "2020-01-01")
    end = cfg.get("default_end_date", "2024-12-31")

    print(f"[1] 端点: {base_url}")
    print(f"    小宇宙({len(symbols)}只): {symbols}")

    from data.ths_fetcher import THSDataFetcher

    ths = THSDataFetcher(token=token, base_url=base_url)
    diag = ths.connect_and_discover()
    print(f"[2] 握手成功: server={diag['server_info']}, 工具数={diag['tool_count']}")

    # 取数（为缩短演示时间，仅取最近 ~3 个月窗口；如需全量把 start 调回）
    demo_start = "2024-01-01"
    kline = ths.get_daily_kline(symbols, start=demo_start, end=end)
    if kline is None or kline.empty:
        print("[3] 未取到任何K线，接入失败。")
        return 1

    print(f"[3] 取到 K线: {kline['symbol'].nunique()} 只, "
          f"{kline['date'].nunique()} 个交易日")
    print("\n样例(前3行):")
    print(kline.head(3).to_string(index=False))

    # 简单因子：20日动量 = 今日收盘 / 20日前收盘 - 1
    kline = kline.sort_values(["symbol", "date"]).reset_index(drop=True)
    kline["ret_20"] = kline.groupby("symbol")["close"].transform(
        lambda s: s / s.shift(20) - 1
    )
    # 次日收益（用作 IC 目标）
    kline["fwd_ret"] = kline.groupby("symbol")["close"].transform(
        lambda s: s.shift(-1) / s - 1
    )
    valid = kline.dropna(subset=["ret_20", "fwd_ret"])

    # 截面 IC（每日 corr(ret_20, fwd_ret)，再取均值）
    ic_per_day = valid.groupby("date").apply(
        lambda d: d["ret_20"].corr(d["fwd_ret"]) if len(d) > 2 else np.nan
    )
    ic_mean = ic_per_day.mean()
    print(f"\n[4] 20日动量因子 截面IC均值 = {ic_mean:.4f} "
          f"(有效样本日 {ic_per_day.notna().sum()} 天)")

    out_path = Path(__file__).resolve().parent.parent / "output" / "ths_kline_demo.csv"
    kline.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[5] 已保存演示数据: {out_path}")
    print("\n结论: 同花顺 iFinD MCP 接入可运行 [OK]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
