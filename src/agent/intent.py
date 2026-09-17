"""
用户输入意图分类（src/agent/intent.py）
=======================================

FactorGPT 的对话入口此前是「单意图」的：任何输入都被当成因子需求塞进挖掘流水线，
于是「你好」也会换来一份耗时数十秒的因子报告（LLM 连不上时还有关键词模板兜底，
报告看起来照样完整）。本模块把「要不要启动重型流水线」这件事前置成一等公民：

===================  ==========================================================
``mining``           要构建/回测一个具体因子或策略 → 走 FactorAgent 挖掘图
``qa``               量化/因子/平台用法咨询 → 直接对话作答，不跑回测
``chitchat``         问候、致谢、与量化无关的闲聊 → 直接对话作答
``clarify``          信息不足，需要追问 → 直接对话作答并请用户补充
===================  ==========================================================

判定的主体是**一次 LLM 语义分类**（:func:`classify`）：把对话历史与当前输入一起
交给模型，要求它只输出一个 JSON 对象。两种情况下退化为规则兜底
（:func:`rule_classify`）：LLM 不可用/返回不可解析，或命中「极短问候」快路径
（省掉一次往返）。兜底不改变既有行为的方向（默认仍判 mining），但会把置信度
压低并在 ``source`` 里标 ``"rule"``，让调用方能如实告诉用户「这次是猜的」。

设计约束：
- **永不因分类而中断主流程**：LLM 异常、JSON 脏、字段缺失，一律降级为规则结果；
- **宁可 clarify，不要误启流水线**：误判 mining 的代价（几分钟空跑）远大于多问一句；
- 分类与直接作答共用同一个 :class:`~llm.client.LLMClient` 配置，UI 侧切换模型后
  分类与对话同步生效（见 ``src/ui/app.py``）。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "INTENT_MINING", "INTENT_QA", "INTENT_CHITCHAT", "INTENT_CLARIFY",
    "INTENTS", "IntentResult", "DEFAULT_INTENT_CONFIG",
    "rule_classify", "classify", "chat_answer", "is_mining",
]

INTENT_MINING = "mining"
INTENT_QA = "qa"
INTENT_CHITCHAT = "chitchat"
INTENT_CLARIFY = "clarify"

INTENTS: Tuple[str, ...] = (INTENT_MINING, INTENT_QA, INTENT_CHITCHAT, INTENT_CLARIFY)

#: 意图的中文展示名（界面直接复用，避免各处重复维护一份映射）。
INTENT_LABELS: Dict[str, str] = {
    INTENT_MINING: "因子挖掘",
    INTENT_QA: "知识问答",
    INTENT_CHITCHAT: "闲聊",
    INTENT_CLARIFY: "需要澄清",
}

DEFAULT_INTENT_CONFIG: Dict[str, Any] = {
    "enabled": True,            # false 时 classify() 直接返回 mining（完全回到旧行为）
    "min_confidence": 0.5,      # 低于此值视为「没把握」，调用方应如实提示用户
    "history_turns": 3,         # 带进分类 prompt 的最近对话轮数（处理指代/跟进）
    "temperature": 0.0,         # 分类要稳定，不要采样
    "fast_path_greeting": True,  # 极短问候直接规则判定，省一次 LLM 往返
    "timeout": 30.0,
}

#: 模型常见别名 → 标准意图（模型爱用 factor/chat/smalltalk 这类词）。
_ALIASES: Dict[str, str] = {
    "mining": INTENT_MINING, "mine": INTENT_MINING, "factor": INTENT_MINING,
    "factor_mining": INTENT_MINING, "build": INTENT_MINING, "backtest": INTENT_MINING,
    "qa": INTENT_QA, "question": INTENT_QA, "ask": INTENT_QA, "咨询": INTENT_QA, "问答": INTENT_QA,
    "chitchat": INTENT_CHITCHAT, "chat": INTENT_CHITCHAT, "smalltalk": INTENT_CHITCHAT,
    "greeting": INTENT_CHITCHAT, "闲聊": INTENT_CHITCHAT, "打招呼": INTENT_CHITCHAT,
    "clarify": INTENT_CLARIFY, "unclear": INTENT_CLARIFY, "ambiguous": INTENT_CLARIFY,
}

_GREETING_RE = re.compile(
    r"^(你好|您好|您哈|hi|hello|hey|嗨|哈喽|哈啰|早上好|中午好|下午好|晚上好|"
    r"谢谢|多谢|感谢|thanks|thx|再见|拜拜|bye|在吗|在么|你是谁|你叫什么|"
    r"你能做什么|你能干嘛|help|帮助|test|测试)[!！?？。.\s]*$",
    re.IGNORECASE,
)

_MINING_WORDS = (
    "因子", "选股", "回测", "构建", "生成", "挖掘", "挖一个", "跑一遍", "跑一次",
    "策略", "alpha", "动量", "反转", "波动率", "换手", "中性化", "市值中性", "行业中性",
    "ic", "icir", "夏普", "分层", "调仓", "合成", "权重", "优化", "编写一个", "写个函数",
    "def alpha", "组合", "信号", "因子库", "体系",
)

_QUESTION_WORDS = (
    "什么是", "什么意思", "啥意思", "为什么", "为何", "如何评价", "怎么看", "怎么算",
    "怎么理解", "区别", "差异", "解释", "是不是", "能不能", "可以吗", "注意什么",
    "?", "？", "吗",
)

#: 强疑问短语：命中它且没有构建动词时判 qa。
#: 之所以要显式排除构建动词——「什么是 ICIR」里含 "ic"，光看关键词会被 mining 吃掉。
_STRONG_QUESTION_WORDS = (
    "什么是", "什么意思", "啥意思", "为什么", "为何", "如何评价", "怎么看", "怎么算",
    "怎么理解", "区别", "差异", "解释",
)

#: 构建动词：它的出现基本意味着用户真的想要一个能跑的因子。
_BUILD_VERBS = (
    "构建", "生成", "挖掘", "挖一个", "回测", "编写", "写个", "跑一遍", "跑一次",
    "来一个", "合成", "优化一下",
)

#: 分类结果缓存：同一句话在同一会话里被反复问（UI rerun 也会重放），不重复烧 token。
_CACHE: Dict[str, "IntentResult"] = {}
_CACHE_MAX = 256


@dataclass
class IntentResult:
    """一次意图判定结果。

    Attributes:
        intent: 标准意图（见 :data:`INTENTS`）。
        confidence: 0~1 的把握程度；规则兜底一律取较低值。
        reason: 判定理由（模型给出的一句话，或兜底说明）。
        rewritten: 仅 mining 有意义——把「换个窗口」「再来一个」这类指代消解后
            补全成的独立因子需求，供挖掘流水线直接使用。
        source: ``"llm"`` / ``"rule"`` —— 用户需要知道这次判定是谁做的。
        error: LLM 调用失败原因（无失败则为空），界面据此解释为何退化为兜底。
    """

    intent: str = INTENT_MINING
    confidence: float = 0.0
    reason: str = ""
    rewritten: str = ""
    source: str = "rule"
    error: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return INTENT_LABELS.get(self.intent, self.intent)

    @property
    def confident(self) -> bool:
        return self.confidence >= float(DEFAULT_INTENT_CONFIG["min_confidence"])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent, "label": self.label, "confidence": round(self.confidence, 3),
            "reason": self.reason, "rewritten": self.rewritten,
            "source": self.source, "error": self.error,
        }


# --------------------------------------------------------------------------
# 规则兜底
# --------------------------------------------------------------------------
def _norm(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def rule_classify(text: str, history: Optional[Sequence[Dict[str, str]]] = None) -> IntentResult:
    """不依赖 LLM 的启发式判定，供 :func:`classify` 降级时调用。

    它不追求覆盖所有输入，只保证两件事：明显的问候不会被送去跑回测；
    其余情况判 mining 但**置信度压低**，由调用方决定是否提示用户。
    """
    s = _norm(text)
    if not s:
        return IntentResult(INTENT_CLARIFY, 0.9, "输入为空", source="rule")

    if _GREETING_RE.match(s):
        return IntentResult(INTENT_CHITCHAT, 0.9, "规则：纯问候/礼貌用语", source="rule")

    has_mining = any(w in s for w in _MINING_WORDS)
    has_question = any(w in s for w in _QUESTION_WORDS)
    # 上一轮在挖因子，这一轮说「再跑一次/换 60 天」——历史里带挖掘词也算数。
    if not has_mining and history:
        tail = "".join(_norm(m.get("content", "")) for m in list(history)[-2:])
        has_mining = any(w in tail for w in _MINING_WORDS)

    # 强疑问 + 无构建动词 → 咨询。放在 mining 之前：否则「什么是 ICIR」会被 "ic" 命中成挖掘。
    if any(w in s for w in _STRONG_QUESTION_WORDS) and not any(w in s for w in _BUILD_VERBS):
        return IntentResult(INTENT_QA, 0.6, "规则：强疑问短语且无构建动词", source="rule")

    if has_mining and not (len(s) <= 12 and not has_question):
        return IntentResult(INTENT_MINING, 0.6, "规则：命中因子/回测类关键词",
                            rewritten=str(text or "").strip(), source="rule")
    if has_mining:
        return IntentResult(INTENT_MINING, 0.45, "规则：关键词弱命中，把握不高",
                            rewritten=str(text or "").strip(), source="rule")
    if has_question:
        return IntentResult(INTENT_QA, 0.55, "规则：疑问句且无因子构建动词", source="rule")
    if len(s) <= 10:
        return IntentResult(INTENT_CHITCHAT, 0.4, "规则：极短输入且无任何因子线索", source="rule")
    return IntentResult(INTENT_MINING, 0.35,
                        "规则兜底：未命中明确信号，沿用既有挖掘行为", source="rule")


# --------------------------------------------------------------------------
# LLM 语义分类
# --------------------------------------------------------------------------
_CLASSIFY_SYSTEM = """\
你是 FactorGPT（量化因子研究平台）的意图分类器。

平台有一条重型流水线：因子挖掘（检索知识 → 生成因子代码 → 沙箱校验 → 回测 → 反思 → 报告），
一次要跑几十秒到几分钟。所以必须先把用户这句话分成四类之一，决定值不值得启动它：

- mining：用户要构建 / 改进 / 回测一个具体的选股因子或策略，期待得到因子代码与回测指标。
- qa：用户就量化、因子、指标或平台用法提问，想要的是解释或建议，而不是一次回测。
- chitchat：问候、致谢、与量化研究无关的闲聊。
- clarify：信息不足，无法判断，需要向用户追问。

判定要点：
1. "构建一个低估值反转因子""换个 60 天窗口再跑""帮我挖一个动量因子" → mining。
2. "什么是 ICIR""IC 高但收益为什么差""这个指标怎么算""现在该用哪类因子" → qa。
3. "你好""谢谢""你是谁" → chitchat。
4. 结合对话历史消解指代：上文在挖因子，这句说"窗口改成 60 天" → mining。
5. 拿不准就 clarify。宁可多问一句，也不要为了显得有用而误判成 mining —— 误启动流水线的代价很高。

只输出一个 JSON 对象，不要任何解释文字、不要代码围栏：
{"intent": "mining|qa|chitchat|clarify", "confidence": 0.0到1.0的小数, \
"reason": "20 字以内中文理由", "rewritten": "仅 intent=mining 时，把指代消解后补全成一句独立的因子需求；否则空字符串"}
"""

_CHAT_SYSTEM = """\
你是 FactorGPT 的对话助手，服务于量化因子研究。

1. 中文回答，简洁：默认 300 字以内，用户要求展开时再展开。
2. 只讲确定的事。涉及收益、IC、夏普这类数字时，明确说明它们需要回测验证，绝不编造具体数值。
3. 如果用户其实想要一个能回测的因子，结尾补一句：把需求写成"要构建什么样的因子、用什么数据、怎么算"，
   我就可以真正跑一遍回测并给出指标。
4. 不输出代码围栏包裹的大段实现，除非用户明确要代码。
"""


def _prompt_with_history(text: str, history: Optional[Sequence[Dict[str, str]]], turns: int) -> str:
    """把最近若干轮对话拼进分类输入，让模型能消解「换个窗口」这类指代。"""
    if not history or turns <= 0:
        return f"【用户当前输入】\n{text}"
    lines = []
    for m in list(history)[-turns * 2:]:
        role = "用户" if str(m.get("role", "")) == "user" else "助手"
        body = str(m.get("content") or "")
        if not body:
            body = (m.get("agent") or {}).get("report", "") or (m.get("answer") or "")
        body = re.sub(r"\s+", " ", body)[:200]  # 历史只作语境，截断防 prompt 膨胀
        if body:
            lines.append(f"{role}：{body}")
    ctx = "\n".join(lines)
    return f"【最近对话】\n{ctx}\n\n【用户当前输入】\n{text}"


def _resolve_llm(config: Optional[Dict[str, Any]], llm: Any):
    """优先复用调用方传入的 LLM（UI 切换模型后由 Agent 持有），否则自建。"""
    if llm is not None and hasattr(llm, "complete"):
        return llm, False
    from llm.client import LLMClient

    cfg = dict(config or {})
    ic = cfg.get("intent") or {}
    llm_cfg = dict(cfg.get("llm") or {})
    if ic.get("timeout") is not None:
        llm_cfg["timeout"] = ic["timeout"]
    cfg["llm"] = llm_cfg
    return LLMClient(cfg), True


def _parse(raw_text: str, fallback_text: str) -> Optional[IntentResult]:
    from llm.client import extract_json

    data = extract_json(raw_text or "")
    if not isinstance(data, dict):
        return None
    key = str(data.get("intent", "")).strip().lower()
    intent = _ALIASES.get(key, key if key in INTENTS else "")
    if not intent:
        return None
    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = min(1.0, max(0.0, conf))
    rewritten = str(data.get("rewritten") or "").strip()
    if intent == INTENT_MINING and not rewritten:
        rewritten = fallback_text
    return IntentResult(
        intent=intent,
        confidence=conf,
        reason=str(data.get("reason") or "").strip()[:80],
        rewritten=rewritten if intent == INTENT_MINING else "",
        source="llm",
        raw=data,
    )


def classify(
    text: str,
    config: Optional[Dict[str, Any]] = None,
    llm: Any = None,
    history: Optional[Sequence[Dict[str, str]]] = None,
) -> IntentResult:
    """判定 ``text`` 的意图。

    Args:
        text: 用户当前输入。
        config: 完整配置字典（读 ``config["intent"]`` 与 ``config["llm"]``）。
        llm: 可选，任何提供 ``complete(system, user)`` 的对象（如 UI 侧已切好模型
            的 Agent LLM / LLMRouter）；不传则按 config 自建 :class:`LLMClient`。
        history: 最近对话（``[{"role": "user"|"assistant", "content": ...}]``），用于指代消解。

    Returns:
        :class:`IntentResult`。LLM 不可用或返回不可解析时退化为 :func:`rule_classify`
        的结果，并在 ``error`` 里写明原因——调用方应把它显示给用户，而不是静默跑挖掘。
    """
    text = str(text or "").strip()
    ic = dict(DEFAULT_INTENT_CONFIG)
    ic.update({k: v for k, v in ((config or {}).get("intent") or {}).items() if v is not None})

    if not text:
        return IntentResult(INTENT_CLARIFY, 0.9, "输入为空", source="rule")
    if not ic.get("enabled", True):
        return IntentResult(INTENT_MINING, 1.0, "意图分流已在配置中关闭",
                            rewritten=text, source="rule")

    cache_key = f"{text}|{ic.get('history_turns')}|{len(history or [])}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    # 极短问候走快路径：一次 LLM 往返换一个几乎无悬念的判定，不值得。
    if ic.get("fast_path_greeting", True) and _GREETING_RE.match(_norm(text)):
        res = IntentResult(INTENT_CHITCHAT, 0.95, "快路径：纯问候/礼貌用语", source="rule")
        _CACHE[cache_key] = res
        return res

    result: Optional[IntentResult] = None
    error = ""
    try:
        client, _ = _resolve_llm(config, llm)
        raw = client.complete(
            _CLASSIFY_SYSTEM,
            _prompt_with_history(text, history, int(ic.get("history_turns", 3))),
            temperature=float(ic.get("temperature", 0.0)),
        )
        result = _parse(raw, text)
        if result is None:
            error = f"模型返回无法解析为意图 JSON：{str(raw)[:120]}"
    except Exception as e:  # noqa: BLE001 —— 分类失败绝不能挡住用户
        error = f"{type(e).__name__}: {e}"
        logger.warning("[intent] LLM 分类失败，退化为规则兜底: %s", error)

    if result is None:
        result = rule_classify(text, history)
        result.error = error

    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[cache_key] = result
    return result


def _message(role: str, content: str) -> Any:
    """构造一条对话消息。

    优先 LangChain 消息对象（与 :meth:`llm.client.LLMClient.chat` 的既有契约一致）；
    若环境没装 ``langchain_core``（离线测试、精简部署），退回 ``{"role", "content"}``
    纯字典——ChatOpenAI 同样接受这种形式。意图分流不该把 langchain 变成硬依赖：
    它是**调用模型的通道**，而不是判定意图的前提。
    """
    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        return {"system": SystemMessage, "user": HumanMessage,
                "assistant": AIMessage}[role](content=content)
    except Exception:  # noqa: BLE001 —— 缺依赖时降级为 dict，不阻断对话
        return {"role": role, "content": content}


def chat_answer(
    text: str,
    config: Optional[Dict[str, Any]] = None,
    llm: Any = None,
    history: Optional[Sequence[Dict[str, str]]] = None,
    intent: Optional[IntentResult] = None,
    turns: int = 4,
) -> str:
    """非挖掘意图的直接作答（不跑回测、不生成因子代码）。

    LLM 不可用时返回一段解释性文本（含失败原因），**不**回退到因子报告——
    「连不上模型却输出一份完整回测报告」正是本模块要消灭的错觉。
    """
    text = str(text or "").strip()
    if not text:
        return "你没有输入内容。可以问我因子/指标问题，或直接说要构建什么样的因子。"

    ic = dict(DEFAULT_INTENT_CONFIG)
    ic.update({k: v for k, v in ((config or {}).get("intent") or {}).items() if v is not None})

    messages: List[Any] = []
    try:
        head = _CHAT_SYSTEM
        if intent is not None and intent.intent == INTENT_CLARIFY:
            head += ("\n5. 本轮判定为「需要澄清」：先用一两句话说明你还缺什么信息，"
                     "再给出你认为用户最可能想要的那个方向。")
        elif intent is not None and intent.intent == INTENT_CHITCHAT:
            head += "\n5. 本轮是闲聊：简短自然地回应，并顺带一句你能帮上什么忙。"
        messages.append(_message("system", head))
        for m in list(history or [])[-turns * 2:]:
            role = str(m.get("role", ""))
            body = str(m.get("content") or "")
            if not body:
                body = str(m.get("answer") or "") or str((m.get("agent") or {}).get("report", ""))
            if not body:
                continue
            messages.append(_message("user" if role == "user" else "assistant", body[:500]))
        messages.append(_message("user", text))

        client, _ = _resolve_llm(config, llm)
        reply = client.chat(messages, temperature=float(ic.get("temperature", 0.0)) or 0.3)
        return (reply or "").strip() or "模型返回了空内容，请换个说法再试一次。"
    except Exception as e:  # noqa: BLE001
        return (
            "没能连上大模型，所以这里不会给你一份「看起来正常」的因子报告——\n\n"
            f"失败原因：`{type(e).__name__}: {e}`\n\n"
            "请在左侧「⚙️ 模型 / API 设置」填写密钥并点击「应用配置」后重试。"
        )


def is_mining(result: IntentResult) -> bool:
    """是否应当启动挖掘流水线（未启用分流时一律 True，保持向后兼容）。"""
    return result.intent == INTENT_MINING
