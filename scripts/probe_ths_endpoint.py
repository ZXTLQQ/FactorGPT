"""
同花顺 iFinD MCP —— 工具 Schema 导出 + 真实取数验证

1) 把 stock 域 10 个工具的 inputSchema 落盘为 JSON（避免控制台乱码）。
2) 真实调用 get_stock_performance，拉取一只股票的日K线，检验返回结构是否
   包含 date/open/high/low/close/volume，能直接喂给 factor-gpt 回测。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    from llm.client import load_config

    cfg = load_config().get("data", {})
    token = os.environ.get("THS_API_TOKEN") or cfg.get("ths_api_token", "")
    if not token:
        print("FAIL: no token")
        return 1

    from data.ths_fetcher import THSDataFetcher

    stock_url = "https://api-mcp.51ifind.com:8643/ds-mcp-servers/hexin-ifind-ds-stock-mcp"
    ths = THSDataFetcher(token=token, base_url=stock_url)
    diag = ths.connect_and_discover()

    out_dir = Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(exist_ok=True)

    # 1) 落盘 schema
    schema_path = out_dir / "ths_tools_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "server_info": diag["server_info"],
                "tools": diag["tools"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[OK] 工具 schema 已写入: {schema_path}")

    # 2) 真实取数：贵州茅台 600519.SH 日线
    print("\n[TEST] 调用 get_stock_performance 拉取 贵州茅台 日K线 …")
    raw = ths.call_tool(
        "get_stock_performance",
        {
            "query": "600519.SH 2024-01-01 至 2024-03-31 日线 开盘 最高 最低 收盘 成交量 成交额",
        },
    )
    raw_path = out_dir / "ths_kline_sample.txt"
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(str(raw))
    print(f"[OK] 原始返回已写入: {raw_path}（长度 {len(str(raw))} 字符）")

    # 也尝试结构化参数（部分工具如 highfreq 使用 symbols）
    print("\n[TEST] 调用 stock_highfreq_quotes 探测结构化参数返回 …")
    try:
        raw2 = ths.call_tool(
            "stock_highfreq_quotes",
            {
                "symbols": ["600519.SH"],
                "start_time": "2024-01-01",
                "end_time": "2024-01-05",
            },
        )
        with open(out_dir / "ths_highfreq_sample.txt", "w", encoding="utf-8") as f:
            f.write(str(raw2))
        print(f"[OK] highfreq 返回已写入 output/ths_highfreq_sample.txt")
    except Exception as e:
        print(f"[WARN] highfreq 调用失败（可能参数不匹配）: {e}")


if __name__ == "__main__":
    main()
