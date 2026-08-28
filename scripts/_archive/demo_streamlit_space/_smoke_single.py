"""单因子 Agent 离线兜底验证：中文需求 + 强制合成数据。
运行：py -3 demo/_smoke_single.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.environ["FACTORGPT_FORCE_SYNTHETIC"] = "1"

from llm.client import load_config
from agent.graph import FactorAgent

config = load_config()
agent = FactorAgent(config)
result = agent.run("构建一个 20 日动量因子，做行业市值中性化处理")

print("=" * 60)
print("回测指标：")
print("=" * 60)
for k, v in (result.get("metrics") or {}).items():
    print(f"  {k}: {v}")

print("\n" + "=" * 60)
print("最终报告（前 60 行）：")
print("=" * 60)
report = result.get("report", "")
for line in report.splitlines()[:60]:
    print(line)

chart_paths = result.get("state", {}).get("chart_paths") or []
print(f"\n[图表] {len(chart_paths)} 张: {chart_paths}")
print("SINGLE_AGENT_OK" if result.get("metrics") and "error" not in result["metrics"] else "SINGLE_AGENT_PARTIAL")
