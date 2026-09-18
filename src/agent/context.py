"""
对话上下文（src/agent/context.py）
=================================

聊天页里真正的多轮工作是这样的：第一轮「挖个动量因子」，第二轮「把窗口改成 60 天
再跑」。第二句离开上文就是一句废话——而挖掘流水线此前是**无状态**的：
``run(user_input)`` 只拿到当前这一句，上一轮的因子代码、指标、反思意见一概不知，
于是每轮都从零重挖一遍。

本模块把「最近几轮对话」压成一段可注入 prompt 的文本，供挖掘图使用。它刻意只做
两件事：一是**消解指代**（让「换个窗口」知道换的是哪个因子），二是**带上一轮的
结果与反思**（让改进真的发生在上一版代码上，而不是重新掷骰子）。

刻意不做的事：不做语义摘要（会引入又一次 LLM 调用与不确定性），不把整份报告塞进
prompt（几千 token 会挤掉因子知识）。截断是硬性的：上下文是调味品，不是主料。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["render_dialogue_context", "summarize_agent_message"]

#: 单条消息进入上下文的最大字符数（超出尾部截断并标注）。
_MAX_CHARS_PER_MSG = 400
#: 默认带进上下文的轮数（一轮 = 用户 + 助手）。
DEFAULT_TURNS = 3


def summarize_agent_message(msg: Dict[str, Any], max_chars: int = _MAX_CHARS_PER_MSG) -> str:
    """把一条助手消息压成一句话上下文。

    挖掘结果里真正对下一轮有用的只有三样：因子名、核心指标、上一版代码的存在与否。
    完整报告有几百行，塞进去只会挤掉因子知识与契约。
    """
    agent = msg.get("agent") or {}
    if not agent and not msg.get("answer"):
        return ""
    if not agent:
        return _clip(str(msg.get("answer") or ""), max_chars)

    name = str(agent.get("factor_name") or "")
    metrics = agent.get("metrics") or {}
    parts = [f"挖出因子 {name}" if name else "完成一次挖掘"]
    keep = ("ic", "icir", "rank_ic", "annual_return", "sharpe", "max_drawdown")
    nums = [f"{k}={metrics[k]}" for k in keep
            if isinstance(metrics.get(k), (int, float))]
    if nums:
        parts.append("指标 " + ", ".join(nums[:4]))
    if agent.get("factor_code"):
        parts.append("（上一版代码见会话记录）")
    if agent.get("llm_error"):
        parts.append(f"上次生成阶段异常：{agent['llm_error'][:80]}")
    return _clip("；".join(parts), max_chars)


def _clip(text: str, max_chars: int) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + " …（略）"


def render_dialogue_context(
    history: Optional[Sequence[Dict[str, Any]]] = None,
    turns: int = DEFAULT_TURNS,
    max_chars: int = _MAX_CHARS_PER_MSG,
) -> str:
    """把最近若干轮对话渲染成供 LLM 阅读的上下文文本。

    Args:
        history: 消息列表，形如 ``[{"role": "user"|"assistant", "content": ...,
                 "answer": ..., "agent": {...}}]``（与 UI 会话状态一致）。
        turns: 带进上下文的轮数。
        max_chars: 单条消息的最大字符数。

    Returns:
        多行文本；无历史时返回空串（调用方据此决定要不要加这一段）。
    """
    if not history or turns <= 0:
        return ""
    tail: List[Dict[str, Any]] = list(history)[-turns * 2:]
    lines: List[str] = []
    for msg in tail:
        role = str(msg.get("role", ""))
        if role == "user":
            body = _clip(str(msg.get("content") or ""), max_chars)
            prefix = "用户"
        else:
            body = summarize_agent_message(msg, max_chars)
            prefix = "助手"
        if body:
            lines.append(f"{prefix}：{body}")
    return "\n".join(lines)
