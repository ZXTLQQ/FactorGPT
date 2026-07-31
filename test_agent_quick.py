"""离线集成测试：用桩 LLM + 合成数据验证因子挖掘 Agent 全流程。"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

os.environ["FACTORGPT_FORCE_SYNTHETIC"] = "1"  # 强制合成数据，跳过网络

from agent.graph import FactorAgent
from llm.client import load_config

# 桩 LLM：返回动量因子模板的 JSON，不触发任何网络请求
class StubLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, system, user, temperature=None):
        self.calls += 1
        code = (
            "import pandas as pd\nimport numpy as np\n"
            "def alpha_factor(df):\n"
            "    df = df.sort_values(['symbol','date']).copy()\n"
            "    df['ret'] = df.groupby('symbol')['close'].pct_change()\n"
            "    df['factor'] = df.groupby('symbol')['ret'].transform(lambda x: x.rolling(20).sum())\n"
            "    df['factor'] = df.groupby('symbol')['factor'].shift(1)\n"
            "    return df[['date','symbol','factor']]\n"
        )
        import json
        return json.dumps({"name": "momentum_20", "description": "20日动量", "code": code})


cfg = load_config()
agent = FactorAgent(cfg)
agent.llm = StubLLM()  # 替换为桩
agent.nodes.llm = agent.llm

res = agent.run("请构建20日动量因子")
print("LLM 调用次数:", agent.llm.calls)
print("指标:", {k: v for k, v in res["metrics"].items() if k != "quantile_returns"})
print("报告前 400 字:\n", res["report"][:400])
print("OK")

# 将结果以 UTF-8 写入文件，便于干净查看
import os
os.makedirs("output", exist_ok=True)
with open("output/agent_result.txt", "w", encoding="utf-8") as f:
    f.write(f"LLM calls: {agent.llm.calls}\n\n")
    f.write("METRICS:\n")
    for k, v in res["metrics"].items():
        if k != "quantile_returns":
            f.write(f"  {k}: {v}\n")
    f.write("\nREPORT:\n")
    f.write(res["report"])
