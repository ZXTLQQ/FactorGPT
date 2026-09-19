"""``src/agent/context`` 多轮上下文回归测试。

挖掘流水线此前是无状态的：``run(user_input)`` 只拿到当前这一句，上一轮的因子代码、
指标、失败原因一概不知，于是第二轮「把窗口改成 60 天再跑」只能从零重挖一遍。
本文件钉住两件事：

1. **上下文能消解指代**：历史里出现过因子名/指标，下一轮 prompt 里必须看得到；
2. **上下文是调味品不是主料**：再长的报告也要被压到几百字，不能挤掉因子知识与
   生成契约；空历史、纯闲聊历史不该带出噪声。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from agent import context as CTX  # noqa: E402


def _agent_msg(name="mom_60d", metrics=None, code=True, err=""):
    return {"role": "assistant", "agent": {
        "factor_name": name,
        "metrics": metrics or {"ic": 0.031, "icir": 0.42, "sharpe": 1.1},
        "factor_code": "def alpha_factor(df): ..." if code else "",
        "llm_error": err,
        "report": "这是一份几百行的完整回测报告 " * 200,  # 必须被压掉
    }}


def test_empty_history_renders_nothing():
    assert CTX.render_dialogue_context(None) == ""
    assert CTX.render_dialogue_context([]) == ""


def test_context_carries_factor_name_and_metrics():
    history = [{"role": "user", "content": "挖一个 20 天动量因子"}, _agent_msg()]
    out = CTX.render_dialogue_context(history)
    assert "挖一个 20 天动量因子" in out
    assert "mom_60d" in out and "ic=0.031" in out


def test_report_is_compressed_not_dumped():
    out = CTX.render_dialogue_context([_agent_msg()])
    assert len(out) < 400  # 几百行报告不得整份进 prompt
    assert "回测报告" not in out


def test_llm_error_is_surfaced():
    out = CTX.render_dialogue_context(
        [_agent_msg(err="生成阶段：AuthenticationError: 401")])
    assert "401" in out


def test_plain_answer_history_is_kept_but_clipped():
    history = [{"role": "assistant", "answer": "长回答" * 500}]
    out = CTX.render_dialogue_context(history, turns=1)
    assert out.startswith("助手：") and len(out) <= 420


def test_turns_limits_how_much_history_enters():
    history = [{"role": "user", "content": f"第{i}轮需求"} for i in range(1, 7)]
    out = CTX.render_dialogue_context(history, turns=2)  # 一轮 = 用户 + 助手
    assert "第1轮需求" not in out and "第2轮需求" not in out
    assert "第5轮需求" in out and "第6轮需求" in out


def test_only_recent_messages_survive():
    out = CTX.render_dialogue_context([_agent_msg(name="old"), {"role": "user", "content": "now"}])
    assert "now" in out and "old" in out  # 两轮都在默认 3 轮窗口内


@pytest.mark.parametrize("turns", [0, -1])
def test_non_positive_turns_render_nothing(turns):
    assert CTX.render_dialogue_context([{"role": "user", "content": "hi"}], turns=turns) == ""


def test_summarize_agent_message_without_payload():
    assert CTX.summarize_agent_message({}) == ""
    assert CTX.summarize_agent_message({"role": "assistant"}) == ""


def test_summarize_prefers_answer_when_no_agent():
    assert "你好" in CTX.summarize_agent_message({"answer": "你好，我能帮你挖因子"})


def test_generate_prompt_includes_context_only_when_present():
    """端到端拼接：上下文段只在该有的时候出现，且带「在上文基础上改」的约束。"""
    from agent.nodes import FactorAgentNodes

    node = object.__new__(FactorAgentNodes)  # 跳过 __init__：只测 prompt 拼接
    base = node._build_generate_prompt("动量因子", "知识", "")
    assert "上文对话" not in base
    with_ctx = node._build_generate_prompt("动量因子", "知识", "助手：挖出因子 mom_60d")
    assert "mom_60d" in with_ctx and "另起炉灶" in with_ctx
