# -*- coding: utf-8 -*-
"""``src/agent/intent`` 意图分类回归测试。

对话入口此前是单意图的：任何输入都被当成因子需求塞进挖掘流水线，于是「你好」
也会换来一份因子报告；更糟的是 LLM 连不上时还有关键词模板兜底，报告看起来
照样完整，用户无从判断本轮到底有没有模型参与。本文件钉住三件事：

1. **挖掘流水线只为挖掘类输入启动**：问候/致谢/概念咨询都必须被分流出去；
2. **分类失败必须可见且不静默**：LLM 抛异常、返回脏 JSON，一律退化为规则兜底
   并把原因写进 ``IntentResult.error``，绝不退化成"照常跑一遍挖掘"；
3. **直接作答不许假装回测**：LLM 不可用时返回解释性文本，而不是一份报告。

测试全部离线：分类与作答都注入假 LLM（只要求 ``complete`` / ``chat`` 两个方法），
不发任何网络请求。
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from agent import intent as IT  # noqa: E402


class _FakeLLM:
    """最小 LLM 替身：记录调用参数，可按需在指定方法上抛异常。"""

    def __init__(self, reply: str = "", raise_on: str = "") -> None:
        self.reply = reply
        self.raise_on = raise_on
        self.complete_calls = []
        self.chat_calls = []

    def complete(self, system, user, temperature=None):
        self.complete_calls.append((system, user, temperature))
        if self.raise_on == "complete":
            raise RuntimeError("endpoint unreachable")
        return self.reply

    def chat(self, messages, temperature=None):
        self.chat_calls.append((messages, temperature))
        if self.raise_on == "chat":
            raise RuntimeError("endpoint unreachable")
        return self.reply


def _msg_content(msg):
    """消息可能是 LangChain 对象，也可能是缺依赖时降级的 dict——两种都要能读。"""
    return msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")


def _json_reply(intent_name, confidence=0.9, reason="测试", rewritten=""):
    """模拟模型常见输出：带代码围栏的 JSON。"""
    payload = {"intent": intent_name, "confidence": confidence,
               "reason": reason, "rewritten": rewritten}
    return "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"


# --------------------------------------------------------------------------
# 规则兜底
# --------------------------------------------------------------------------
def test_rule_greeting_is_chitchat():
    for text in ("你好", "您好！", "hi", "谢谢", "你是谁", "再见"):
        res = IT.rule_classify(text)
        assert res.intent == IT.INTENT_CHITCHAT, text
        assert res.source == "rule"


def test_rule_factor_request_is_mining():
    res = IT.rule_classify("构建一个低估值质量反转因子并回测")
    assert res.intent == IT.INTENT_MINING
    assert res.rewritten  # 挖掘类必须带出可供流水线使用的需求原文


def test_rule_concept_question_is_qa_not_mining():
    # 「ICIR」里含 "ic"，若只看关键词会被 mining 吃掉——这正是回归点。
    res = IT.rule_classify("什么是 ICIR？")
    assert res.intent == IT.INTENT_QA


def test_rule_history_carries_mining_context():
    history = [{"role": "user", "content": "帮我挖一个动量因子"},
               {"role": "assistant", "agent": {"report": "因子报告"}}]
    res = IT.rule_classify("把窗口改成 60 天再跑", history=history)
    assert res.intent == IT.INTENT_MINING


def test_rule_empty_input_is_clarify():
    assert IT.rule_classify("").intent == IT.INTENT_CLARIFY


# --------------------------------------------------------------------------
# LLM 语义分类
# --------------------------------------------------------------------------
def test_classify_parses_llm_json():
    llm = _FakeLLM(_json_reply("mining", 0.88, "明确要回测一个因子",
                               "构建低估值质量反转因子并回测"))
    res = IT.classify("来个低估值反转的", llm=llm)
    assert res.intent == IT.INTENT_MINING
    assert res.source == "llm"
    assert abs(res.confidence - 0.88) < 1e-6
    assert res.rewritten == "构建低估值质量反转因子并回测"
    assert not res.error
    assert len(llm.complete_calls) == 1


def test_classify_packs_history_into_prompt():
    llm = _FakeLLM(_json_reply("mining", 0.9, "承接上文", "60 天窗口动量因子"))
    history = [{"role": "user", "content": "挖一个动量因子"},
               {"role": "assistant", "answer": "已生成动量因子"}]
    IT.classify("换个 60 天窗口", llm=llm, history=history)
    _, user_prompt, _ = llm.complete_calls[0]
    assert "最近对话" in user_prompt and "60 天" in user_prompt


def test_classify_alias_and_confidence_clamp():
    # 模型爱用 factor / smalltalk 这类词，且置信度可能越界。
    llm = _FakeLLM(_json_reply("factor", 2.5))
    res = IT.classify("x", llm=llm)
    assert res.intent == IT.INTENT_MINING
    assert res.confidence == 1.0
    res2 = IT.classify("y", llm=_FakeLLM(_json_reply("smalltalk", -1)))
    assert res2.intent == IT.INTENT_CHITCHAT
    assert res2.confidence == 0.0


def test_classify_llm_error_falls_back_visibly():
    llm = _FakeLLM(raise_on="complete")
    res = IT.classify("什么是 ICIR？", llm=llm)
    assert res.source == "rule"
    assert res.error  # 失败原因必须留在结果里，供界面如实提示
    assert "RuntimeError" in res.error
    assert res.intent == IT.INTENT_QA


def test_classify_unparsable_json_falls_back():
    llm = _FakeLLM("我觉得这大概是个因子需求吧（未按要求输出 JSON）")
    res = IT.classify("这个指标现在还能用吗", llm=llm)
    assert res.source == "rule"
    assert res.error  # 脏输出必须留下痕迹，而不是静默当没发生过


def test_classify_unknown_intent_falls_back():
    llm = _FakeLLM(_json_reply("whatever", 0.9))
    res = IT.classify("这个指标现在还能用吗", llm=llm)
    assert res.source == "rule"
    assert res.error  # 未收录的意图值同样要留下痕迹


def test_classify_mining_without_rewritten_uses_original():
    llm = _FakeLLM(_json_reply("mining", 0.9, "要回测", ""))
    res = IT.classify("挖个动量因子", llm=llm)
    assert res.rewritten == "挖个动量因子"


def test_classify_non_mining_clears_rewritten():
    llm = _FakeLLM(_json_reply("chitchat", 0.95, "打招呼", "挖个因子"))
    assert IT.classify("你好", llm=llm).rewritten == ""


def test_classify_greeting_fast_path_skips_llm():
    llm = _FakeLLM(_json_reply("mining", 0.1))
    res = IT.classify("你好", llm=llm)
    assert res.intent == IT.INTENT_CHITCHAT
    assert llm.complete_calls == []  # 无悬念的问候不该烧一次往返


def test_classify_disabled_keeps_legacy_behaviour():
    cfg = {"intent": {"enabled": False}}
    res = IT.classify("你好", config=cfg, llm=_FakeLLM(_json_reply("chitchat", 0.9)))
    assert res.intent == IT.INTENT_MINING
    assert res.confidence == 1.0


def test_classify_empty_input():
    assert IT.classify("", llm=_FakeLLM()).intent == IT.INTENT_CLARIFY


def test_classify_result_is_serialisable():
    res = IT.classify("你好", llm=_FakeLLM())
    payload = res.as_dict()
    assert json.dumps(payload, ensure_ascii=False)
    assert payload["label"] in IT.INTENT_LABELS.values()


# --------------------------------------------------------------------------
# 直接作答
# --------------------------------------------------------------------------
def test_chat_answer_returns_model_reply():
    llm = _FakeLLM("你好！我可以帮你构建并回测因子，或解释 IC / ICIR 这类指标。")
    assert "回测" in IT.chat_answer("你好", llm=llm)
    assert len(llm.chat_calls) == 1


def test_chat_answer_failure_is_explicit_not_a_report():
    llm = _FakeLLM(raise_on="chat")
    reply = IT.chat_answer("什么是 ICIR？", llm=llm)
    assert reply.startswith("没能连上大模型")
    assert "RuntimeError" in reply
    assert "失败原因" in reply
    # 绝不能给出一份看起来正常的因子报告：没有指标表、没有回测数字。
    assert "|" not in reply and "夏普" not in reply and "ICIR" not in reply


def test_chat_answer_clarify_intent_gets_extra_instruction():
    llm = _FakeLLM("请补充一下你想做什么")
    res = IT.IntentResult(IT.INTENT_CLARIFY, 0.3, "信息不足", source="llm")
    IT.chat_answer("嗯", llm=llm, intent=res)
    system_msg = _msg_content(llm.chat_calls[0][0][0])
    assert "澄清" in system_msg


def test_chat_answer_works_without_langchain(monkeypatch):
    # 精简部署/CI 没装 langchain_core：消息退化成 dict，对话本身照常进行。
    monkeypatch.setitem(sys.modules, "langchain_core", None)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", None)
    llm = _FakeLLM("你好，我可以帮你挖因子")
    assert "可以" in IT.chat_answer("你好", llm=llm)
    # 装了 langchain 时是消息对象；没装时必须是 dict，且 role 语义不变。
    roles = [m.get("role") if isinstance(m, dict) else {"SystemMessage": "system",
                                                       "HumanMessage": "user",
                                                       "AIMessage": "assistant"}[type(m).__name__]
             for m in llm.chat_calls[0][0]]
    assert roles == ["system", "user"]


def test_chat_answer_packs_history_in_order():
    llm = _FakeLLM("接着说")
    history = [{"role": "user", "content": "什么是 ICIR"},
               {"role": "assistant", "answer": "ICIR 是 IC 的信息比率"}]
    IT.chat_answer("那 IR 呢", llm=llm, history=history)
    msgs = llm.chat_calls[0][0]
    assert len(msgs) == 4
    assert "ICIR" in _msg_content(msgs[1]) and "信息比率" in _msg_content(msgs[2])
    assert "IR" in _msg_content(msgs[3])


def test_chat_answer_empty_input():
    assert "没有" in IT.chat_answer("", llm=_FakeLLM())


# --------------------------------------------------------------------------
# 对外判据
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    (IT.INTENT_MINING, True), (IT.INTENT_QA, False),
    (IT.INTENT_CHITCHAT, False), (IT.INTENT_CLARIFY, False),
])
def test_is_mining(name, expected):
    assert IT.is_mining(IT.IntentResult(name, 0.9)) is expected


def test_default_config_shape():
    cfg = IT.DEFAULT_INTENT_CONFIG
    assert cfg["enabled"] is True and 0 <= cfg["min_confidence"] <= 1
    assert set(IT.INTENTS) == {"mining", "qa", "chitchat", "clarify"}
