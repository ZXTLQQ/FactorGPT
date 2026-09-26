"""JEV（TypeSafe "System One" 决策模型）适配器。

JEV 不是聊天模型，它不写长文、只做**判断题**：喂一段状态文本 + 一组问题，
一次往返并行返回类型化答案与校准概率——

    POST https://api.typesafe.ai/v1/systemone
    {"model": "jev-latest",
     "state": "...非结构化内容...",
     "questions": {
        "data_type": {"type": "choice", "instructions": "...", "criteria": {...}},
        "sentiment": {"type": "score", "instructions": "...", "criteria": [...]},
        "factorizable": {"type": "noul", "instructions": "..."}
     }}
    → {"answers": {"data_type": {"choice": "research", "confidence": 0.97,
                                 "probabilities": {...}},
                   "sentiment": {"score": 3.0, "confidence": 0.88},
                   "factorizable": {"noul": 0.72}}}

为什么用它接非结构化数据
------------------------
用户上传的图片/研报/PDF 直接塞进挖掘 prompt 有两个问题：一是**太长**，
二是**不可判定**——LLM 会把材料里的任何话都当成可用信号，包括含前瞻信息的
（"预计下半年需求回暖"）会造成前视偏差。JEV 的价值在于把材料压成
**少数几个可分支的判定 + 概率**，代码可以据此硬分流（如
``forward_looking > 0.7 → 禁止直接做因子，只能做事件标注``），
而不是求模型"自己注意一下"。

降级策略是一个三级链，**越靠前越可信**：

1. **JEV**（需 ``TYPESAFE_API_KEY``，联网，校准概率）；
2. **本地训练模型**（``engine.multimodal_train``，由 ``scripts/train_multimodal.py``
   训出的朴素贝叶斯/Transformer，离线、毫秒级；产物不存在时自动跳过）；
3. **本地正则规则**（:meth:`_heuristic`，最糙但永不可用尽）。

任一环节不阻塞流水线，结果里的 ``engine`` 字段如实标明判定来源。
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"

# 一次调用并行问完所有问题（共享 state，成本不随问题数线性上升）
QUESTIONS: Dict[str, Dict[str, Any]] = {
    "data_type": {
        "type": "choice",
        "instructions": "这段材料主要属于哪一类金融另类数据？",
        "criteria": {
            "quote_screenshot": "行情/盘口截图或行情导出的数值表格",
            "announcement": "上市公司公告、财报或监管披露文件",
            "research": "券商研报、行业研究或投研笔记",
            "news": "新闻资讯、快讯或社交媒体文本",
            "trading_log": "自身成交/委托流水或交易日志",
            "macro": "宏观数据、政策文件或统计报表",
            "other": "以上都不是",
        },
    },
    "sentiment": {
        "type": "score",
        "instructions": "材料整体对未来价格/基本面的倾向性如何？",
        "criteria": ["强负面", "偏负面", "中性", "偏正面", "强正面"],
    },
    "alignable": {
        "type": "noul",
        "instructions": "材料中含有可对齐到具体标的（股票/期货代码）与日期的结构化信息吗？",
    },
    "forward_looking": {
        "type": "noul",
        "instructions": "材料含未来信息或事后才知道的结论（用于因子会造成前视偏差）吗？",
    },
    "factorizable": {
        "type": "noul",
        "instructions": "这段材料能直接派生出一个可用于横截面选股的因子信号吗？",
    },
    "factor_hint": {
        "type": "choice",
        "instructions": "若可因子化，最合适的构造路径是哪一种？",
        "criteria": {
            "event": "事件型：出现/未出现的哑变量或事件窗口",
            "sentiment": "情绪型：文本情感打分做截面暴露",
            "text_complexity": "文本复杂度/可读性/信息量",
            "numeric": "数值型：材料里的数字直接做因子值",
            "unusable": "不构成可用因子",
        },
    },
}

_SENTIMENT_LABELS = ["强负面", "偏负面", "中性", "偏正面", "强正面"]
_DATA_TYPE_LABELS = {
    "quote_screenshot": "行情截图/数值表",
    "announcement": "公告财报",
    "research": "研报笔记",
    "news": "新闻资讯",
    "trading_log": "交易流水",
    "macro": "宏观/政策",
    "other": "其他",
}
_HINT_LABELS = {
    "event": "事件型因子",
    "sentiment": "情绪型因子",
    "text_complexity": "文本复杂度因子",
    "numeric": "数值型因子",
    "unusable": "不可用",
}


#: 进程内共享的本地训练模型（见 :meth:`JEVClient._ensure_local`）。
#: **按 multimodal 配置分片**：缓存不带 key 的话，第一个调用方的配置就定了终身——
#: 测试里换个 ``model_dir`` 训一份迷你模型，拿到的仍是别人那份（或那个人的 None）。
_SHARED_LOCAL: Dict[str, Any] = {}


def reset_local_cache() -> None:
    """清掉共享的本地模型缓存（重新训练后立刻生效、测试隔离用）。"""
    _SHARED_LOCAL.clear()


class JEVClient:
    """JEV 判定客户端（带本地规则降级）。"""

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("jev", {}) or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.base_url = str(cfg.get("base_url") or DEFAULT_BASE_URL)
        self.model = str(cfg.get("model") or DEFAULT_MODEL)
        self.timeout = float(cfg.get("timeout") or 15.0)
        self.max_chars = int(cfg.get("max_chars") or 6000)
        self.fallback = bool(cfg.get("fallback_to_heuristic", True))
        self.api_key_env = str(cfg.get("api_key_env") or "TYPESAFE_API_KEY")
        self.api_key = os.environ.get(self.api_key_env, "").strip()
        self.last_error = ""
        self._config = config or {}
        self._local: Any = None            # 本地训练模型（惰性加载）
        self._local_loaded = False

    # ------------------------------------------------------------------
    @property
    def callable(self) -> bool:
        return self.enabled and bool(self.api_key)

    def _ensure_local(self) -> Any:
        """惰性加载本地训练模型：产物不存在/未启用时返回 None，绝不抛到调用方。

        模型在**进程内共享**：加载一次要 4 秒（torch 权重 + 词表），而 UploadIngestor
        会随 UI 每次交互重建——不共享的话，用户多传一个文件就多付一次 4 秒，
        而且这个开销完全与"这次上传了什么"无关。
        """
        if self._local_loaded:
            return self._local
        self._local_loaded = True
        key = str((self._config or {}).get("multimodal") or "default")
        if key in _SHARED_LOCAL:
            self._local = _SHARED_LOCAL[key]
            return self._local
        try:
            from engine.multimodal_train import load_local_classifier

            _SHARED_LOCAL[key] = load_local_classifier(self._config)
        except Exception as e:  # noqa: BLE001
            logger.debug("[jev] 本地模型不可用，按规则降级: %s", e)
            _SHARED_LOCAL[key] = None
        self._local = _SHARED_LOCAL[key]
        return self._local

    def ask(self, state: str, questions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """直接问一组问题，返回原始 answers 字典；不可用时抛异常由调用方降级。"""
        if not self.callable:
            raise RuntimeError(f"JEV 未启用或 {self.api_key_env} 未配置")
        payload = {
            "model": self.model,
            "state": str(state)[: self.max_chars],
            "questions": questions or QUESTIONS,
        }
        # requests 只在真要联网时才导入：它是可选项，没装不能让 `import engine.jev`
        # 直接崩掉——那样会把离线的本地模型与规则降级一起拖下水。
        try:
            import requests  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - 环境缺失
            raise RuntimeError(f"JEV 联网判定需要 requests：pip install requests（{e}）") from e
        r = requests.post(
            self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"JEV HTTP {r.status_code}: {r.text[:200]}")
        body = r.json()
        return body.get("answers") or {}

    # ------------------------------------------------------------------
    def analyze(self, text: str, filename: str = "") -> Dict[str, Any]:
        """把一段非结构化内容压成结构化判定。

        返回统一结构（无论走 JEV 还是本地规则），字段含义见模块文档；
        ``engine`` 标明判定来源，``confidence`` 是该判定的置信度。
        """
        state = f"【文件名】{filename or '未命名'}\n【内容】\n{text or ''!s}"[: self.max_chars]
        answers: Dict[str, Any] = {}
        engine = "heuristic"
        if self.callable:
            try:
                answers = self.ask(state)
                engine = "jev"
                self.last_error = ""
            except Exception as e:  # noqa: BLE001
                self.last_error = str(e)[:300]
                logger.warning("[jev] 调用失败，回退本地规则: %s", e)
        if not answers:
            # 第二级：本地训练模型（离线、无 Key 也能用；没训过就静默跳过）
            merged = self._local_answers(text)
            if merged:
                answers, engine = merged, "local-model"
        if not answers:
            if not self.fallback:
                return {"engine": "unavailable", "error": self.last_error, "summary": ""}
            answers = self._heuristic(text)
            engine = "heuristic"

        res = self._normalize(answers, engine)
        res["summary"] = self.render(res, filename=filename)
        return res

    # ------------------------------------------------------------------
    @staticmethod
    def _heuristic(text: str) -> Dict[str, Any]:
        """本地规则判定：关键词 + 正则，只求可用的粗分流，不追求精度。"""
        t = str(text or "")
        low = t.lower()
        if re.search(r"(委托|成交|平仓|开仓|手数|报单)", t) and re.search(r"\d{2}:\d{2}", t):
            data_type = "trading_log"
        elif re.search(r"(收盘价|开盘价|成交量|盘口|K线|涨跌幅)", t):
            data_type = "quote_screenshot"
        elif re.search(r"(公告|财报|年报|季报|招股|重大事项)", t):
            data_type = "announcement"
        elif re.search(r"(研报|评级|目标价|投资建议|行业研究)", t):
            data_type = "research"
        elif re.search(r"(宏观|GDP|CPI|政策|央行|统计局)", t):
            data_type = "macro"
        elif len(t.strip()) > 0:
            data_type = "news"
        else:
            data_type = "other"

        pos = len(re.findall(r"(利好|增长|超预期|上涨|回暖|扩张|买入|增持)", t))
        neg = len(re.findall(r"(利空|下滑|不及预期|下跌|萎缩|亏损|卖出|减持)", t))
        score = 2.0 if pos == neg else (3.0 if pos > neg else 1.0)
        if pos and neg:
            score = 2.0

        has_code = bool(re.search(r"\b\d{6}\b|[A-Za-z]{1,2}\d{3,4}(\.SHF|\.DCE|\.CZC)?", t))
        has_date = bool(re.search(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}", t)) or bool(
            re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", t))
        alignable = 0.85 if (has_code and has_date) else (0.5 if (has_code or has_date) else 0.15)
        fwd = 0.8 if re.search(r"(预计|预测|展望|将(?:会|要)|forecast)", t) else 0.2
        numeric = len(re.findall(r"-?\d+\.?\d*", t)) >= 8
        factorizable = 0.75 if (alignable > 0.5 or numeric) else 0.35
        hint = "numeric" if numeric and data_type == "quote_screenshot" else (
            "event" if data_type == "announcement" else (
                "sentiment" if data_type in ("news", "research") else
                ("unusable" if factorizable < 0.5 else "text_complexity")))

        return {
            "data_type": {"type": "choice", "choice": data_type, "confidence": 0.5},
            "sentiment": {"type": "score", "score": score, "confidence": 0.5},
            "alignable": {"type": "noul", "noul": alignable},
            "forward_looking": {"type": "noul", "noul": fwd},
            "factorizable": {"type": "noul", "noul": factorizable},
            "factor_hint": {"type": "choice", "choice": hint, "confidence": 0.5},
        }

    def _local_answers(self, text: str) -> Dict[str, Any]:
        """用本地训练模型补上它训过的维度，其余维度仍交给规则。

        模型只训了「材料类型 / 情绪 / 是否含前瞻信息」三件事；「可对齐」「可因子化」
        依赖标的与日期的抽取，正则比模型可靠，不拿模型硬凑——分工写死在这里，
        避免以后把没训过的维度也标成模型输出，看着高级实则瞎猜。
        """
        cli = self._ensure_local()
        if cli is None:
            return {}
        try:
            pred = cli.classify_text(text)
        except Exception as e:  # noqa: BLE001
            logger.debug("[jev] 本地模型推理失败，按规则降级: %s", e)
            return {}
        if not pred.get("data_type"):
            return {}
        base = self._heuristic(text)
        probs = pred.get("data_type_probs") or {}
        base["data_type"] = {"type": "choice", "choice": str(pred["data_type"]),
                             "confidence": round(float(max(probs.values())) if probs else 0.6, 3)}
        sent_map = {"negative": 1.0, "neutral": 2.0, "positive": 3.0}
        if pred.get("sentiment"):
            sp = pred.get("sentiment_probs") or {}
            base["sentiment"] = {
                "type": "score", "score": sent_map.get(str(pred["sentiment"]), 2.0),
                "confidence": round(float(max(sp.values())) if sp else 0.6, 3)}
        if pred.get("forward_looking") is not None:
            base["forward_looking"] = {"type": "noul",
                                       "noul": round(float(pred["forward_looking"]), 3)}
        base["_model_name"] = str(pred.get("data_type_model") or "local")
        return base

    @staticmethod
    def _normalize(answers: Dict[str, Any], engine: str) -> Dict[str, Any]:
        def _choice(k: str) -> str:
            a = answers.get(k) or {}
            return str(a.get("choice") or "")

        def _noul(k: str) -> float:
            a = answers.get(k) or {}
            try:
                return float(a.get("noul", 0.0))
            except (TypeError, ValueError):
                return 0.0

        def _score(k: str) -> float:
            a = answers.get(k) or {}
            try:
                return float(a.get("score", 2.0))
            except (TypeError, ValueError):
                return 2.0

        def _conf(k: str) -> float:
            a = answers.get(k) or {}
            try:
                return float(a.get("confidence", 0.0))
            except (TypeError, ValueError):
                return 0.0

        sent = _score("sentiment")          # 0~4
        sidx = min(4, max(0, round(sent)))  # JEV 可能返回 4.2 之类的越界值
        return {
            "engine": engine,
            "model": (DEFAULT_MODEL if engine == "jev"
                      else str(answers.get("_model_name") or "local") if engine == "local-model"
                      else "local-rule"),
            "data_type": _choice("data_type"),
            "data_type_label": _DATA_TYPE_LABELS.get(_choice("data_type"), "其他"),
            "sentiment_score": sent,
            "sentiment_label": _SENTIMENT_LABELS[sidx] if 0 <= sent <= 4 else "中性",
            "sentiment": round((sent - 2.0) / 2.0, 3),   # 归一到 -1~1
            "alignable": round(_noul("alignable"), 3),
            "forward_looking": round(_noul("forward_looking"), 3),
            "factorizable": round(_noul("factorizable"), 3),
            "factor_hint": _choice("factor_hint"),
            "factor_hint_label": _HINT_LABELS.get(_choice("factor_hint"), "不可用"),
            "confidence": round(max(_conf("data_type"), _conf("factor_hint")), 3),
            "raw": answers,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def render(res: Dict[str, Any], filename: str = "") -> str:
        """渲染成注入 prompt 的中文摘要（含可直接硬分流的判定结论）。"""
        if res.get("engine") == "unavailable":
            return f"- {filename}：JEV 不可用（{res.get('error', '')}）"
        warns = []
        if float(res.get("forward_looking", 0.0)) >= 0.7:
            warns.append("含前瞻信息，禁止直接作为当期因子值（会造成前视偏差），"
                         "只允许按披露日做事件标注")
        if float(res.get("alignable", 0.0)) < 0.4:
            warns.append("缺少可对齐的标的/日期，无法并入 date×symbol 面板")
        if res.get("factor_hint") == "unusable":
            warns.append("判定为不可因子化")
        return (
            f"- 文件：{filename or '未命名'}（判定引擎 {res.get('engine')}，"
            f"置信度 {res.get('confidence')}）\n"
            f"  类型：{res.get('data_type_label')}；情绪：{res.get('sentiment_label')}"
            f"（{res.get('sentiment')}）\n"
            f"  可对齐(date×symbol)：{res.get('alignable')}；"
            f"可因子化：{res.get('factorizable')}；构造路径：{res.get('factor_hint_label')}\n"
            + (f"  ⚠️ {'；'.join(warns)}\n" if warns else "")
        )

    def render_many(self, results: Dict[str, Dict[str, Any]]) -> str:
        if not results:
            return ""
        return ("【上传材料的结构化判定（JEV）】\n"
                + "\n".join(self.render(r, name) for name, r in results.items()))


def override_data_type(res: Dict[str, Any], data_type: str, source: str = "") -> Dict[str, Any]:
    """用更可信的来源覆盖「材料类型」判定，并同步 label 与渲染文本。

    典型场景：图片没文字，正则/JEV 都只能判成"其他"，但 CNN 从版式认出是
    K 线图——这条判定比文本线索硬，应当覆盖。**只覆盖类型这一项**，情绪/可对齐
    等维度不动（CNN 没训过，不能顺手改）。
    """
    if not res or not data_type or data_type not in _DATA_TYPE_LABELS:
        return res
    res["data_type"] = data_type
    res["data_type_label"] = _DATA_TYPE_LABELS[data_type]
    if source:
        res["data_type_source"] = source
        raw = res.get("raw") or {}
        raw["data_type"] = {"type": "choice", "choice": data_type, "confidence": 0.6,
                            "source": source}
        res["raw"] = raw
    if "summary" in res:
        res["summary"] = JEVClient.render(res, filename=_filename_of(res))
    return res


def _filename_of(res: Dict[str, Any]) -> str:
    m = re.search(r"文件：([^\n（]+)", str(res.get("summary") or ""))
    return m.group(1).strip() if m else ""


def analyze_upload(text: str, filename: str = "", config: Optional[dict] = None) -> Dict[str, Any]:
    """便捷入口：单份材料的 JEV 判定。"""
    return JEVClient(config).analyze(text, filename=filename)


__all__ = ["QUESTIONS", "JEVClient", "analyze_upload"]
