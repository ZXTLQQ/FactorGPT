"""
因子方法学解读（ui/methodologist.py）

为因子挖掘结果提供「方法学审查」文本：
- `get_factor_name_from_report`：从报告文本中提取因子名称；
- `run_methodologist`：调用 LLM 生成方法学解读（经济逻辑 / 过拟合风险 /
  样本外稳健性 / 应用注意点），并尊重 UI 中切换的模型配置；无 API 时降级
  为报告摘要。
"""

from pathlib import Path

import yaml

from llm.client import LLMClient

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"


def _load_llm_cfg() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        data = {}
    cfg = dict(data.get("llm", {}))
    # 若 Streamlit 会话已切换模型，则优先使用（支持通过 API Key 切换模型）
    try:
        import streamlit as st

        if "llm_cfg" in st.session_state:
            for k in ("provider", "model", "api_key", "base_url", "temperature"):
                if k in st.session_state.llm_cfg:
                    cfg[k] = st.session_state.llm_cfg[k]
    except Exception:
        pass
    return cfg


def get_factor_name_from_report(report: str):
    """从报告中提取一个简短的因子名称（首个非空行，去掉 markdown 标题符）。"""
    if not report:
        return None
    for line in report.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:30]
    return None


def run_methodologist(name: str, report: str) -> str:
    """生成因子方法学解读。使用 LLM；失败时降级为报告摘要。"""
    cfg = _load_llm_cfg()
    client = LLMClient({"llm": cfg})
    prompt = (
        f"你是一位严谨的量化方法学审查员。因子名称：{name}。\n"
        f"以下是因子研究报告：\n{report}\n\n"
        "请从方法学角度评估该因子的：\n"
        "1) 经济逻辑与可解释性；\n"
        "2) 潜在过拟合 / 前视偏差风险；\n"
        "3) 样本外稳健性；\n"
        "4) 实际应用注意点。\n"
        "用简洁中文分点回答。"
    )
    try:
        return client.complete(
            system="你是量化因子方法学专家，回答严谨、客观、可执行。",
            user=prompt,
        )
    except Exception as e:
        return (
            f"⚠️ 方法学解读需调用 LLM（当前模型配置不可用：{e}）。\n\n"
            f"**因子名称**：{name}\n\n"
            f"**报告摘要（前 600 字）**：\n\n{report[:600]}"
        )
