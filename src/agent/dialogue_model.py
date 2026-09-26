"""对话能力与上下文关联的本地可训练层（src/agent/dialogue_model.py）。

为什么需要这一层
----------------
``intent.py`` 的判定链此前只有两级：**LLM 语义分类** 和 **正则兜底**。LLM 在时
指代消解做得很好（"窗口改成 60 天" → 补全成一句独立需求）；LLM 不在时正则兜底
只能判意图，**消解不了指代**——`rewritten` 原样返回 "窗口改成 60 天"，流水线拿到
一句没有主语、没有因子的废话，于是每轮都从零重挖。也就是说：离线时多轮对话是断的。

本模块补上中间那一级，并把它做成**可训练**的（与 ``engine.multimodal_train`` 同一
思路：合成语料 → 本机训出小模型 → 落盘 → 推理优先用），两个头各有分工：

- **意图头**（4 类：mining / qa / chitchat / clarify）；
- **跟进头**（二分类：这句话**是否依赖上文**才能理解）。

为什么跟进头要单独训
--------------------
"再来一个""它 IC 怎么样""换个窗口"这三句的共同点是：**字面没有任何因子关键词**。
正则靠关键词活着，这类句子它一律看不见；而判定"要不要去上文里找指代"与"这句话是
什么意图"是两件事——先知道它是跟进句，才谈得上在上文里找 slot 把它补全。

上下文关联（指代消解）为什么用规则而不是再训一个模型
----------------------------------------------------
消解需要的是**精确抽取**（"60 天"是窗口、"中证500"是股票池），这种槽位抽取用正则
比用词袋模型可靠——模型把"60"判成窗口的概率再高，也不如一条正则说得清。所以分工
写死在这里：跟进判定交给模型，槽位填充交给规则，谁也别假装自己能做对方那件事。

产物与降级
----------
训练产物在 ``data/models/dialogue/``；未训练 / 产物缺失时 :func:`load_classifier`
返回 ``None``，调用方（``intent.py``）继续退到正则，**永不因本模块阻塞对话**。
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = "data/models/dialogue"

#: 与 ``agent.intent.INTENTS`` 保持一致（这里不 import，避免循环依赖）。
INTENT_LABELS: List[str] = ["mining", "qa", "chitchat", "clarify"]
FOLLOWUP_LABELS: List[str] = ["no", "yes"]


# ======================================================================
# 1. 对话状态：从历史里抽出可绑定的槽位
# ======================================================================
#: 因子方向词（决定"换个方向"换的是什么）。
_DIRECTION_WORDS: Tuple[str, ...] = (
    "动量", "反转", "低估值", "价值", "质量", "成长", "波动", "波动率", "流动性",
    "换手", "规模", "市值", "红利", "一致预期", "超预期", "资金流", "情绪", "杠杆",
)
#: 股票池词。
_UNIVERSE_WORDS: Tuple[str, ...] = (
    "沪深300", "沪深 300", "中证500", "中证 500", "中证800", "中证 800",
    "中证1000", "创业板", "科创板", "全市场", "csi300", "csi500", "csi800",
)
#: 窗口：如 "20日" / "60 天" / "3个月"。
_WINDOW_RE = re.compile(r"(\d{1,4})\s*(日|天|周|个月|月|年|min|分钟)")
#: 显式指代：这些词一出现，句子几乎必然在说上文里的东西。
_ANAPHORA_RE = re.compile(
    r"(改成|改为|换成|换成|调成|调到|再跑|再来|重跑|再挖|再试|重新|上一个|上一版|"
    r"上一轮|刚才|它|他|她|这个|那个|这样|同样|还是|换个|改一下|再来一个)")
#: 纯跟进句（几乎不含任何实质槽位，完全依赖上文）。
_PURE_FOLLOWUP_RE = re.compile(
    r"^(再(来|跑|挖|试)?(一个|一次|一遍)?|重(跑|新)(一个|一次)?|换个?(参数|窗口|方向|"
    r"周期|天数)?|改(一)?下?(参数|窗口|方向|周期)?|上一个|上一版|刚才那个|就这样|"
    r"同上|同上所述|它(怎么样|如何|呢)?)[。.!！?？\s]*$")


@dataclass
class DialogueState:
    """从最近对话里抽出的可绑定槽位。

    只保留"下一轮真会用得上"的四样东西：因子名、方向、窗口、股票池。
    完整报告有几百行，塞进状态只会让消解结果变得不可预测。
    """

    factor_name: str = ""
    direction: str = ""
    window: str = ""
    universe: str = ""
    last_user: str = ""
    has_mining: bool = False
    turns: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """有没有足够信息可以把一句指代补全成独立需求。"""
        return bool(self.direction or self.factor_name or self.window or self.universe)


def _find_first(text: str, words: Sequence[str]) -> str:
    for w in words:
        if w in text:
            return w
    return ""


def _norm_window(text: str) -> str:
    m = _WINDOW_RE.search(str(text or ""))
    if not m:
        return ""
    num, unit = m.group(1), m.group(2)
    if unit in ("个月", "月"):
        return f"{num}个月"
    if unit in ("min", "分钟"):
        return f"{num}分钟"
    return f"{num}{unit}"


def extract_state(history: Optional[Sequence[Dict[str, Any]]] = None,
                  turns: int = 6) -> DialogueState:
    """从对话历史里抽出 :class:`DialogueState`。

    Args:
        history: 消息列表，形如 ``[{"role": "user"|"assistant", "content": ...,
                 "agent": {...}}]``（与 UI 会话状态一致）。
        turns: 往前看几轮。

    Returns:
        槽位状态；无历史时返回全空状态（``usable`` 为 False）。
    """
    st = DialogueState()
    if not history:
        return st
    tail = list(history)[-max(1, int(turns)) * 2:]
    st.turns = len(tail)
    # 从旧到新扫，后面的覆盖前面的——"先把窗口改成 60 天，再改成 120 天"以最后为准。
    for msg in tail:
        role = str(msg.get("role", ""))
        body = str(msg.get("content") or "")
        if role == "user":
            if body:
                st.last_user = body
            probe = body
            if any(w in probe for w in ("因子", "选股", "回测", "构建", "挖掘", "策略")):
                st.has_mining = True
            d = _find_first(probe, _DIRECTION_WORDS)
            if d:
                st.direction = d
            u = _find_first(probe, _UNIVERSE_WORDS)
            if u:
                st.universe = u
            w = _norm_window(probe)
            if w:
                st.window = w
        else:
            agent = msg.get("agent") or {}
            if not agent:
                continue
            st.has_mining = True
            name = str(agent.get("factor_name") or "")
            if name:
                st.factor_name = name
            m = agent.get("metrics") or {}
            if isinstance(m, dict) and m:
                st.metrics = {k: v for k, v in m.items()
                              if isinstance(v, (int, float))}
    return st


# ======================================================================
# 2. 跟进判定 + 指代消解
# ======================================================================
def _rule_is_followup(text: str, state: DialogueState) -> Tuple[bool, str]:
    """不依赖模型的跟进判定（模型不可用时的兜底，也是模型输出的对照）。"""
    s = re.sub(r"\s+", "", str(text or ""))
    if not s:
        return False, "空输入"
    if _PURE_FOLLOWUP_RE.match(s):
        return True, "规则：纯指代句，无实质槽位"
    if _ANAPHORA_RE.search(s) and state.usable:
        return True, "规则：含指代词且上文有可绑定槽位"
    return False, "规则：未命中指代线索"


def is_followup(text: str,
                history: Optional[Sequence[Dict[str, Any]]] = None,
                state: Optional[DialogueState] = None,
                model: Optional["LocalDialogueClassifier"] = None) -> Tuple[bool, str]:
    """这句话是否需要上文才能理解。

    模型与规则是**或**的关系：任一方说是，就按跟进处理。理由是不对称代价——
    把一句独立需求误判成跟进，最多是补全时多带一点上文（无害）；把一句跟进
    误判成独立需求，流水线就拿到一句缺主语的废话，整轮白跑。

    Returns:
        ``(是否跟进, 判定理由)``。
    """
    st = state if state is not None else extract_state(history)
    rule_yes, rule_reason = _rule_is_followup(text, st)
    if not st.usable:
        # 上文无槽位可绑定，判了跟进也补不出东西——直接按独立句处理。
        return False, "上文无可绑定槽位，按独立需求处理"
    if model is not None:
        try:
            p = float(model.predict_followup(text))
            if p >= 0.5:
                return True, f"本地模型：跟进概率 {p:.2f}"
        except Exception as e:  # noqa: BLE001
            logger.debug("[dialogue] 跟进模型推理失败，按规则: %s", e)
    return rule_yes, rule_reason


def resolve_reference(text: str,
                      history: Optional[Sequence[Dict[str, Any]]] = None,
                      state: Optional[DialogueState] = None,
                      model: Optional["LocalDialogueClassifier"] = None) -> str:
    """把一句带指代的跟进补全成**独立可读**的需求，供挖掘流水线直接使用。

    补全规则（刻意保守）：

    1. 不是跟进句 → 原样返回。凭空"补全"一句本来就完整的需求只会添乱；
    2. 本句自带的槽位优先，只补本句没有的那些（"改成 60 天"里的 60 覆盖上文的 20）；
    3. 上文完全没有可绑槽位 → 原样返回，绝不编造方向或股票池。

    Returns:
        补全后的需求文本；无需补全时返回原文。
    """
    raw = str(text or "").strip()
    st = state if state is not None else extract_state(history)
    if not raw or not st.usable:
        return raw
    follow, _ = is_followup(raw, history=history, state=st, model=model)
    if not follow:
        return raw

    window = _norm_window(raw) or st.window
    universe = _find_first(raw, _UNIVERSE_WORDS) or st.universe
    direction = _find_first(raw, _DIRECTION_WORDS) or st.direction

    parts: List[str] = []
    if direction:
        parts.append(f"构建一个{window}窗口的{direction}因子" if window
                     else f"构建一个{direction}因子")
    elif st.factor_name:
        parts.append(f"在上一轮因子 {st.factor_name} 的基础上")
        if window:
            parts.append(f"把窗口调整为 {window}")
        else:
            parts.append("按下面的要求调整")
    else:
        return raw
    if universe:
        parts.append(f"股票池 {universe}")
    head = "，".join(parts)
    return f"{head}（原始要求：{raw}）"


# ======================================================================
# 3. 向量化与朴素贝叶斯（自实现，零依赖）
# ======================================================================
#: 中文按字切、英文数字按整词切（"ICIR" 切开就废了，"预计" 按字切更稳）。
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[一-鿿]")
_PAD, _UNK = 0, 1


class TextVectorizer:
    """词/字级词表 + 词袋计数。

    刻意**不复用** ``engine.multimodal_train.TextVectorizer``：那个模块在 import 时
    就会 ``import torch``，实测 2.7 秒——对话路径为了一个只做词袋计数的朴素贝叶斯
    去付一次 torch 导入，是纯粹的浪费，而且会让"离线毫秒级"这句话失去意义。
    """

    def __init__(self, max_features: int = 4000, max_len: int = 48) -> None:
        self.max_features = int(max_features)
        self.max_len = int(max_len)
        self.vocab: Dict[str, int] = {}
        self.fitted = False

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return _TOKEN_RE.findall(str(text or "").lower())

    def build(self, texts: Sequence[str]) -> "TextVectorizer":
        counts: Dict[str, int] = {}
        for t in texts:
            for tk in self.tokenize(t):
                counts[tk] = counts.get(tk, 0) + 1
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: self.max_features]
        self.vocab = {tk: i + 2 for i, (tk, _) in enumerate(top)}  # 0=pad 1=unk
        self.fitted = bool(self.vocab)
        return self

    def counts(self, texts: Sequence[str]) -> np.ndarray:
        """词袋计数矩阵 ``[N, V+2]``。"""
        v = len(self.vocab) + 2
        out = np.zeros((len(texts), v), dtype=np.float64)
        for i, t in enumerate(texts):
            for tk in self.tokenize(t):
                out[i, self.vocab.get(tk, _UNK)] += 1.0
        return out

    def to_json(self) -> Dict[str, Any]:
        return {"max_features": self.max_features, "max_len": self.max_len,
                "vocab": self.vocab}

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "TextVectorizer":
        v = cls(int(d.get("max_features") or 4000), int(d.get("max_len") or 48))
        v.vocab = {str(k): int(i) for k, i in (d.get("vocab") or {}).items()}
        v.fitted = bool(v.vocab)
        return v


class MultinomialNB:
    """多项式朴素贝叶斯 + 加一平滑，闭式解，对数域计算。"""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.classes: List[str] = []
        self.log_prior: np.ndarray = np.zeros(0)
        self.log_likelihood: np.ndarray = np.zeros((0, 0))

    def fit(self, X: np.ndarray, y: Sequence[str]) -> "MultinomialNB":
        self.classes = sorted(set(y))
        idx = {c: i for i, c in enumerate(self.classes)}
        k, v = len(self.classes), X.shape[1]
        counts = np.zeros((k, v), dtype=np.float64)
        total = np.zeros(k, dtype=np.float64)
        for row, lab in zip(X, y):
            i = idx[lab]
            counts[i] += row
            total[i] += float(row.sum())
        self.log_likelihood = np.log(counts + self.alpha) - np.log(
            (total + self.alpha * v)[:, None])
        n = np.array([sum(1 for l in y if l == c) for c in self.classes], dtype=np.float64)
        self.log_prior = np.log(n / n.sum())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.classes:
            raise RuntimeError("模型未训练")
        lp = X @ self.log_likelihood.T + self.log_prior
        lp = lp - lp.max(axis=1, keepdims=True)
        p = np.exp(lp)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> List[str]:
        return [self.classes[int(i)] for i in self.predict_proba(X).argmax(axis=1)]

    def save(self, path: Any) -> None:
        np.savez(path, classes=np.array(self.classes),
                 log_likelihood=self.log_likelihood, log_prior=self.log_prior,
                 alpha=np.array([self.alpha]))

    @classmethod
    def load(cls, path: Any) -> "MultinomialNB":
        z = np.load(path, allow_pickle=True)
        m = cls(float(z["alpha"][0]))
        m.classes = [str(c) for c in z["classes"]]
        m.log_likelihood = z["log_likelihood"]
        m.log_prior = z["log_prior"]
        return m


# ======================================================================
# 4. 本地模型：意图头 + 跟进头
# ======================================================================
class LocalDialogueClassifier:
    """加载训练产物做离线意图/跟进判定。

    两个头共用同一个词表（``TextVectorizer``）。产物缺失或版本不匹配时
    :attr:`available` 为 ``False``，调用方退到正则——本类的存在与否
    不影响对话能否进行。
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        self.vec: Optional[Any] = None
        self.intent_nb: Optional[Any] = None
        self.followup_nb: Optional[Any] = None
        self.version = ""
        try:
            meta = json.loads((self.model_dir / "vectorizer.json").read_text(encoding="utf-8"))
            self.vec = TextVectorizer.from_json(meta)
            self.intent_nb = MultinomialNB.load(self.model_dir / "nb_intent.npz")
            self.followup_nb = MultinomialNB.load(self.model_dir / "nb_followup.npz")
            self.version = str(meta.get("version") or "")
        except Exception as e:  # noqa: BLE001
            logger.debug("[dialogue] 本地对话模型不可用: %s", e)
            self.vec = None

    @property
    def available(self) -> bool:
        return self.vec is not None and self.intent_nb is not None and self.followup_nb is not None

    def _X(self, texts: Sequence[str]) -> np.ndarray:
        assert self.vec is not None
        return self.vec.counts(list(texts))

    def predict_intent(self, text: str) -> Tuple[str, float]:
        """返回 ``(意图, 置信度)``。"""
        if not self.available:
            raise RuntimeError("本地对话模型未训练")
        probs = self.intent_nb.predict_proba(self._X([str(text or "")]))[0]
        i = int(np.argmax(probs))
        return str(self.intent_nb.classes[i]), float(probs[i])

    def predict_followup(self, text: str) -> float:
        """返回「这句话依赖上文」的概率。"""
        if not self.available:
            raise RuntimeError("本地对话模型未训练")
        probs = self.followup_nb.predict_proba(self._X([str(text or "")]))[0]
        classes = [str(c) for c in self.followup_nb.classes]
        return float(probs[classes.index("yes")]) if "yes" in classes else 0.0


_CLASSIFIER_CACHE: Dict[str, Optional[LocalDialogueClassifier]] = {}


def load_classifier(config: Optional[dict] = None) -> Optional[LocalDialogueClassifier]:
    """按配置加载本地对话模型；未启用/无产物时返回 ``None``。

    按目录缓存：Streamlit 每次交互都会重建对象，重复读盘 + 重建词表没有意义。
    """
    cfg = ((config or {}).get("dialogue") or {})
    if not bool(cfg.get("enabled", True)):
        return None
    d = str(cfg.get("model_dir") or DEFAULT_MODEL_DIR)
    if d in _CLASSIFIER_CACHE:
        return _CLASSIFIER_CACHE[d]
    cli = LocalDialogueClassifier(d)
    _CLASSIFIER_CACHE[d] = cli if cli.available else None
    return _CLASSIFIER_CACHE[d]


def reset_cache() -> None:
    """清掉分类器缓存（训练完立刻生效、测试隔离用）。"""
    _CLASSIFIER_CACHE.clear()


# ======================================================================
# 5. 合成语料
# ======================================================================
_DIRECTIONS = ("动量", "反转", "低估值", "质量", "成长", "低波动", "高流动性",
               "换手率", "小市值", "红利", "资金流")
_UNIVERSES = ("沪深300", "中证500", "中证800", "创业板", "全市场", "")
_CONCEPTS = ("IC", "ICIR", "RankIC", "夏普", "换手率", "因子中性化", "市值中性化",
             "分层回测", "最大回撤", "年化收益", "信息比率", "因子衰减")

_MINING_TEMPLATES = (
    "构建一个{dir}因子", "帮我挖一个{dir}因子", "来一个{dir}策略并回测",
    "用{uni}做{dir}因子", "{dir}因子窗口设成{w}", "写一个{dir}因子的代码",
    "回测一下{dir}因子在{uni}上的表现", "给我一个{w}{dir}的选股信号",
    "把{dir}和{dir2}合成一个因子", "优化{dir}因子的权重",
)
_QA_TEMPLATES = (
    "什么是{c}", "{c}怎么算", "{c}和{c2}有什么区别", "为什么{c}高但收益差",
    "怎么理解{c}", "{c}是多少算好", "如何评价一个{c}", "这个平台的{c}在哪看",
    "{c}需要注意什么", "{dir}因子适合什么行情",
)
_CHITCHAT_TEMPLATES = (
    "你好", "您好啊", "hi", "hello", "谢谢", "多谢了", "再见", "拜拜",
    "你是谁", "你能做什么", "今天天气不错", "讲个笑话", "辛苦了", "早上好",
)
_CLARIFY_TEMPLATES = (
    "帮我看看", "弄一下", "这个怎么处理", "你说的那个", "随便来点", "嗯", "?",
    "有个事", "帮我弄弄", "看看这个",
)
#: 跟进句模板：字面没有因子关键词，必须靠上文才能落地。
#: 分两组——"再来一个/换个窗口"是把上一轮的事再做一遍（mining），
#: "它IC怎么样/上一版的夏普是多少"是问上一轮的结果（qa）。混在一起标 mining
#: 会让模型学到"带指代就是 mining"，正好是这里最不该犯的错。
_FOLLOWUP_MINING = (
    "再来一个", "再跑一次", "重新跑一遍", "换个窗口试试", "窗口改成{w}",
    "把窗口调到{w}", "换个方向", "改成{dir}", "换成{uni}", "上一版再优化一下",
    "刚才那个再改改", "换个参数", "同样的方法再来一次", "还是用上次那个",
    "上一个继续", "同上",
)
_FOLLOWUP_QA = (
    "它IC怎么样", "这个因子的夏普是多少", "上一版为什么收益差", "刚才那个怎么看",
    "它的换手率高不高", "上一轮的结果怎么理解", "这个能解释一下吗",
)
#: 固定模板（无占位符）的尾缀变化：否则 14 条问候语会变成 14 条完全相同的样本，
#: 训练/测试集里都有它,"准确率 1.0"就只是重复计数。
_FILLERS = ("", "啊", "呀", "哈", "～", "。")


def _fill(tpl: str, rnd: random.Random) -> str:
    d1, d2 = rnd.sample(_DIRECTIONS, 2)
    c1, c2 = rnd.sample(_CONCEPTS, 2)
    text = tpl.format(dir=d1, dir2=d2, uni=rnd.choice(_UNIVERSES) or "全市场",
                      w=rnd.choice(("20日", "60日", "5日", "120日", "10天")),
                      c=c1, c2=c2)
    if "{" not in tpl:
        text += rnd.choice(_FILLERS)
    return text


def synthesize_corpus(per_template: int = 40, seed: int = 20260926
                      ) -> List[Dict[str, Any]]:
    """生成可复现的合成语料（标签由生成过程确定，不存在标错还自认为对）。

    每条样本是 ``{"text", "intent", "followup"}``。``followup="yes"`` 的样本全部
    来自跟进模板——它们刻意不含因子关键词，正是正则看不见、而模型应当接住的那部分。
    末尾按文本去重：同一句话在语料里出现多次只会让评估变成重复计数。
    """
    rnd = random.Random(int(seed))
    rows: List[Dict[str, Any]] = []
    plan = [
        (_MINING_TEMPLATES, "mining", "no"),
        (_QA_TEMPLATES, "qa", "no"),
        (_CHITCHAT_TEMPLATES, "chitchat", "no"),
        (_CLARIFY_TEMPLATES, "clarify", "no"),
        (_FOLLOWUP_MINING, "mining", "yes"),
        (_FOLLOWUP_QA, "qa", "yes"),
    ]
    for tpls, intent, follow in plan:
        for tpl in tpls:
            for _ in range(max(1, int(per_template))):
                rows.append({"text": _fill(tpl, rnd), "intent": intent,
                             "followup": follow})
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for r in rows:
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        uniq.append(r)
    rnd.shuffle(uniq)
    return uniq


# ======================================================================
# 6. 训练与评估
# ======================================================================
def _split(rows: Sequence[Dict[str, Any]], test_ratio: float = 0.25, seed: int = 7
           ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    k = max(1, round(len(idx) * float(test_ratio)))
    test = [rows[i] for i in idx[:k]]
    train = [rows[i] for i in idx[k:]]
    return train, test


def _metrics(y_true: Sequence[str], y_pred: Sequence[str],
             labels: Sequence[str]) -> Dict[str, float]:
    n = max(1, len(y_true))
    acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / n
    f1s = []
    for lab in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b == lab)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != lab and b == lab)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b != lab)
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1s.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return {"acc": round(acc, 4), "macro_f1": round(sum(f1s) / max(1, len(f1s)), 4),
            "n": len(y_true)}


def _hard_node_split(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """挑出「字面无关键词」的困难样本——这是模型相对正则真正的增量所在。

    一条样本如果含 ``因子/构建/什么是/你好`` 这类提示词，正则也能判对，模型判对
    说明不了任何问题。所以单独切出**没有任何提示词**的那一批（纯跟进句几乎全在
    这里），在它上面把模型与正则并排比一次。
    """
    cues = ("因子", "构建", "生成", "挖掘", "回测", "策略", "选股", "优化", "什么是",
            "怎么", "为什么", "如何", "区别", "你好", "您好", "谢谢", "再见", "拜拜",
            "hi", "hello", "你是谁")
    out = []
    for r in rows:
        s = str(r.get("text", "")).lower()
        if not any(c in s for c in cues):
            out.append(r)
    return out


def _rule_intent_of(text: str) -> str:
    """用 ``agent.intent`` 的正则兜底判意图（困难集上的对照臂）。"""
    from agent.intent import rule_classify

    return str(rule_classify(text).intent)


def train_all(config: Optional[dict] = None,
              out_dir: Optional[str] = None,
              per_template: int = 40,
              seed: int = 20260926) -> Dict[str, Any]:
    """训练两个头并落盘，返回训练报告（含困难集上模型 vs 正则的对照）。

    只用 numpy：闭式解的朴素贝叶斯几十毫秒训完，CI 里也能跑，不依赖 torch。
    """
    cfg = ((config or {}).get("dialogue") or {})
    d = Path(out_dir or cfg.get("model_dir") or DEFAULT_MODEL_DIR)
    d.mkdir(parents=True, exist_ok=True)

    rows = synthesize_corpus(per_template=int(per_template), seed=int(seed))
    train, test = _split(rows)
    X_all = [r["text"] for r in rows]
    vec = TextVectorizer(max_features=4000, max_len=48).build(X_all)
    Xtr = vec.counts([r["text"] for r in train])
    Xte = vec.counts([r["text"] for r in test])

    intent_nb = MultinomialNB().fit(Xtr, [r["intent"] for r in train])
    follow_nb = MultinomialNB().fit(Xtr, [r["followup"] for r in train])

    y_int = [r["intent"] for r in test]
    y_fol = [r["followup"] for r in test]
    report_models = {
        "intent_nb": _metrics(y_int, list(intent_nb.predict(Xte)), INTENT_LABELS),
        "followup_nb": _metrics(y_fol, list(follow_nb.predict(Xte)), FOLLOWUP_LABELS),
    }

    hard = _hard_node_split(test)
    hard_report: Dict[str, Any] = {"n": len(hard)}
    if hard:
        hx = vec.counts([r["text"] for r in hard])
        y_true = [r["intent"] for r in hard]
        model_pred = list(intent_nb.predict(hx))
        rule_pred = [_rule_intent_of(r["text"]) for r in hard]
        hard_report["model"] = _metrics(y_true, model_pred, INTENT_LABELS)
        hard_report["rule"] = _metrics(y_true, rule_pred, INTENT_LABELS)
        hf_true = [r["followup"] for r in hard]
        hf_pred = list(follow_nb.predict(hx))
        f_tp = sum(1 for a, b in zip(hf_true, hf_pred) if a == "yes" and b == "yes")
        f_fp = sum(1 for a, b in zip(hf_true, hf_pred) if a != "yes" and b == "yes")
        f_fn = sum(1 for a, b in zip(hf_true, hf_pred) if a == "yes" and b != "yes")
        p = f_tp / max(1, f_tp + f_fp)
        r = f_tp / max(1, f_tp + f_fn)
        hard_report["followup"] = {"precision": round(p, 4), "recall": round(r, 4),
                                   "f1": round(0.0 if p + r == 0 else 2 * p * r / (p + r), 4)}

    vec_payload = vec.to_json()
    vec_payload["version"] = "dialogue-1"
    (d / "vectorizer.json").write_text(
        json.dumps(vec_payload, ensure_ascii=False), encoding="utf-8")
    intent_nb.save(d / "nb_intent.npz")
    follow_nb.save(d / "nb_followup.npz")

    report = {
        "dataset": {"rows": len(rows), "train": len(train), "test": len(test)},
        "models": report_models,
        "hard_node": hard_report,
        "config": {"out_dir": str(d), "per_template": int(per_template), "seed": int(seed)},
    }
    (d / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    reset_cache()
    return report


__all__ = [
    "DEFAULT_MODEL_DIR",
    "FOLLOWUP_LABELS",
    "INTENT_LABELS",
    "DialogueState",
    "LocalDialogueClassifier",
    "MultinomialNB",
    "TextVectorizer",
    "extract_state",
    "is_followup",
    "load_classifier",
    "reset_cache",
    "resolve_reference",
    "synthesize_corpus",
    "train_all",
]
