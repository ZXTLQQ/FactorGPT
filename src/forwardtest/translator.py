"""translator — 宏观观点 -> Headline Arena 方向预测 的翻译器。

Headline Arena 方向性挑战的标的为跨市场宏观资产（黄金 GC / 美债 ZN / 原油 CL /
股指 ES / 白银 SI / 铜 HG / 天然气 NG / 美元 DXY / 比特币 BTC 等），而 FactorGPT
因子研究的天然产物是 A 股截面因子。二者之间的桥接在于**宏观主题文本**：任何因子
观点（或研报/新闻摘要）只要包含利率、风格、商品方向等宏观措辞，就能被翻译成
「某资产在未来结算窗口内 看多/看空/中性 + 置信度」的概率预测，锁定到 HA 挑战上。

本模块是**确定性规则翻译**（关键词主题匹配），不调用任何外部服务，因此：
- 离线 / CI 可测（tests/test_forwardtest.py 直接覆盖）；
- 在 runner 里可叠加 LLM 翻译（LLM 输出 JSON view 后同样走 view_to_prediction
  校验，保证进入账本/提交环节的字段永远是干净的）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------- 资产元信息 #
ASSET_META: Dict[str, Dict[str, str]] = {
    "GC": {"name": "黄金", "category": "贵金属", "note": "COMEX 黄金；避险/通胀敏感"},
    "SI": {"name": "白银", "category": "贵金属", "note": "COMEX 白银；贵金属+工业属性"},
    "HG": {"name": "铜", "category": "工业金属", "note": "COMEX 铜；全球增长风向标"},
    "CL": {"name": "WTI原油", "category": "能源", "note": "供需/地缘驱动"},
    "NG": {"name": "天然气", "category": "能源", "note": "HH 天然气"},
    "RB": {"name": "RBOB汽油", "category": "能源", "note": "成品油，夏季出行旺季敏感"},
    "ZS": {"name": "大豆", "category": "农产品", "note": "CBOT 大豆"},
    "ES": {"name": "E-mini标普500", "category": "股指", "note": "美股大盘；风格/风险偏好代理"},
    "ZN": {"name": "10年期美债", "category": "利率", "note": "注意：bullish=债券价格上涨=收益率下行"},
    "DXY": {"name": "美元指数", "category": "汇率", "note": "美元兑一篮子货币"},
    "BTC": {"name": "比特币", "category": "加密货币", "note": "BTC Arena 暂停新建挑战"},
    "ETH": {"name": "以太坊", "category": "加密货币", "note": "ETH 价格事件挑战"},
}

DIRECTIONS = ("bullish", "bearish", "neutral")

DEFAULT_ASSETS = ["GC", "ES", "CL", "ZN", "SI", "HG", "DXY"]

# ------------------------------------------------------------ 主题关键词规则 #
# 每条规则：(关键词列表, 目标资产, 方向, 基础置信度, 解释)
# 关键词按「宏观措辞优先于资产字面」匹配，避免 "黄金" 同时命中多条产生抵消。
THEME_RULES: List[tuple] = [
    # ── 利率 / 债券（ZN：bullish=收益率下行=价格上行） ──
    (("加息", "鹰派", "紧缩", "货币收紧", "利率上行", "收益率上行", "美债收益率上行", "再通胀"),
     "ZN", "bearish", 0.62, "紧缩/利率上行 → 债券价格承压"),
    (("降息", "鸽派", "宽松", "货币转松", "利率下行", "收益率下行", "美债收益率下行", "流动性宽松", "降准"),
     "ZN", "bullish", 0.62, "宽松/利率下行 → 债券价格上涨"),
    (("通胀回落", "通胀降温", "CPI低于预期", "通胀下行"),
     "ZN", "bullish", 0.58, "通胀回落 → 加息预期降温 → 债券走强"),
    # ── 黄金（避险 + 通胀 + 弱美元） ──
    # 注意：不写裸词"黄金/金价"做多（会与"金价下跌"等子串冲突），
    # 明确方向短语才计多空；仅提及资产名由别名逻辑给弱中性。
    (("黄金走强", "金价走强", "黄金上涨", "金价上涨", "看多黄金", "黄金走牛"),
     "GC", "bullish", 0.60, "明确看多黄金"),
    (("金价下跌", "黄金下跌", "黄金走弱", "金价走弱", "看空黄金", "金价回落"),
     "GC", "bearish", 0.58, "明确看空黄金"),
    (("避险", "衰退", "硬着陆", "risk off", "risk-off", "恐慌", "不确定性上升", "地缘冲突", "去美元化", "滞胀"),
     "GC", "bullish", 0.62, "避险需求 → 黄金走强"),
    (("通胀回升", "通胀上行", "通胀超预期", "物价上行"),
     "GC", "bullish", 0.56, "通胀对冲需求 → 黄金受益"),
    # ── 原油 / 能源 ──
    (("油价上行", "原油走强", "看多原油", "供给收紧", "OPEC+减产", "地缘风险推升油价"),
     "CL", "bullish", 0.62, "供给/地缘驱动油价上行"),
    (("油价下行", "原油走弱", "看空原油", "需求疲软", "累库", "供给过剩", "OPEC+增产"),
     "CL", "bearish", 0.60, "需求/供给宽松驱动油价下行"),
    # ── 股指 / 风险偏好 ──
    (("risk on", "risk-on", "风险偏好回升", "风险偏好上升", "经济复苏", "软着陆", "增长超预期", "股市走强", "美股上涨", "盈利上修"),
     "ES", "bullish", 0.60, "风险偏好回升 → 股指走强"),
    (("risk off", "risk-off", "风险偏好回落", "风险偏好下降", "股市走弱", "美股下跌", "盈利下修", "盈利预警", "衰退担忧"),
     "ES", "bearish", 0.60, "风险偏好回落 → 股指承压"),
    (("风格切换", "大盘跑赢", "价值跑赢"),
     "ES", "bullish", 0.52, "风格偏防御/大盘 → 股指方向弱多"),
    (("小盘占优", "成长跑赢", "题材活跃", "流动性宽松利好成长"),
     "ES", "bullish", 0.52, "成长/题材活跃 → 风险偏好抬升（弱映射）"),
    # ── 美元 ──
    (("美元走强", "美元指数走强", "强美元", "美元升值"),
     "DXY", "bullish", 0.62, "美元走强"),
    (("美元走弱", "美元指数走弱", "弱美元", "美元贬值"),
     "DXY", "bearish", 0.60, "美元走弱"),
    # ── 加密货币 ──
    (("比特币上涨", "BTC走强", "加密货币走强", "加密牛"),
     "BTC", "bullish", 0.58, "看多 BTC"),
    (("比特币下跌", "BTC走弱", "加密货币走弱", "加密熊"),
     "BTC", "bearish", 0.58, "看空 BTC"),
]

# 因子类别 -> 可能的宏观措辞（用于把因子描述翻译成主题文本的启发式）
CATEGORY_THEME_HINTS = {
    "动量": "risk on 动能延续",
    "反转": "均值回归 不确定性上升",
    "价值": "风格切换 价值跑赢",
    "质量": "盈利确定性 风险偏好回落",
    "成长": "成长跑赢 风险偏好回升",
    "低波": "防御 风险偏好回落",
    "流动性": "流动性宽松",
    "size": "小盘占优",
}


def direction_valid(direction: str) -> bool:
    return direction in DIRECTIONS


def clean_direction(direction: str, fallback: str = "neutral") -> str:
    d = (direction or "").strip().lower()
    return d if d in DIRECTIONS else fallback


def clamp_confidence(value: float, lo: float = 0.34, hi: float = 0.95) -> float:
    """HA 三选一随机基准为 1/3，因此把置信度钳制在 (0.34, 0.95)。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 0.5
    return max(lo, min(hi, v))


def view_to_prediction(
    view: Dict[str, Any],
    challenge: Optional[Dict[str, Any]] = None,
    fallback_confidence: float = 0.5,
) -> Dict[str, str]:
    """把一个 view dict 清洗成可直接 POST /predict 的 payload。

    view 形如 {"asset": "GC", "direction": "bullish", "confidence": 0.62,
               "reasoning": "..."}。asset 可缺省（交由调用方配到具体挑战）。
    若给出 challenge，则校验资产一致并附加结算窗口上下文。
    """
    direction = clean_direction(str(view.get("direction", "neutral")))
    confidence = clamp_confidence(view.get("confidence", fallback_confidence))
    asset = str(view.get("asset") or "").upper()
    if challenge is not None:
        chal_asset = str((challenge or {}).get("asset") or "").upper()
        if chal_asset and asset and asset != chal_asset:
            raise ValueError(
                f"view.asset={asset} 与 challenge.asset={chal_asset} 不一致")
        if not asset:
            asset = chal_asset
    reasoning = str(view.get("reasoning") or "").strip()
    if challenge and reasoning:
        q = str(challenge.get("question") or "").strip()
        reasoning = f"{reasoning}｜挑战:{q[:120]}" if q else reasoning
    payload: Dict[str, str] = {"direction": direction, "confidence": str(round(confidence, 4))}
    if reasoning:
        payload["reasoning"] = reasoning
    if asset:
        payload["asset"] = asset
    return payload


def parse_theme(
    theme_text: str,
    assets: Optional[List[str]] = None,
    default_confidence: float = 0.5,
) -> List[Dict[str, Any]]:
    """对宏观主题文本做关键词匹配，返回 [view, ...]。

    - 同一资产多条信号时：按 (方向命中数 - 反向命中数) 的符号定方向，
      基础置信度随命中强度小幅上调（上限 0.9）；
    - 文本不含任何信号时返回 []（调用方决定是否补中性占位）。
    """
    text = (theme_text or "").lower()
    if not text:
        return []
    target = {a.upper() for a in (assets or DEFAULT_ASSETS)}
    # 资产名到代码的别名（如 "美债" -> ZN）
    alias = {"美债": "ZN", "债券": "ZN", "标普": "ES", "美股": "ES", "原油": "CL",
             "油价": "CL", "黄金": "GC", "金价": "GC", "白银": "SI", "美元": "DXY",
             "铜": "HG", "天然气": "NG", "比特币": "BTC", "以太坊": "ETH"}
    votes: Dict[str, Dict[str, Any]] = {}
    for keywords, asset, direction, conf, note in THEME_RULES:
        if asset not in target:
            continue
        hit = sum(1 for kw in keywords if kw.lower() in text)
        if not hit:
            continue
        cell = votes.setdefault(asset, {"bullish": 0, "bearish": 0, "total": 0,
                                        "conf": 0.0, "notes": []})
        cell[direction] += hit
        cell["total"] += hit
        cell["conf"] = max(cell["conf"], conf)
        cell["notes"].append(note)
    # 别名直连：文本明确出现资产名但无方向措辞时给弱中性提示（total>=1 才会计入 views）
    for alias_key, code in alias.items():
        if code in target and alias_key in text and code not in votes:
            votes[code] = {"bullish": 0, "bearish": 0, "total": 1,
                           "conf": default_confidence,
                           "notes": [f"文本提及 {alias_key}（无明确方向）"]}
    views: List[Dict[str, Any]] = []
    for asset, v in votes.items():
        if v["total"] == 0:
            continue
        net = v["bullish"] - v["bearish"]
        if net > 0:
            direction = "bullish"
        elif net < 0:
            direction = "bearish"
        else:
            direction = "neutral"
        strength = min(0.9, v["conf"] + 0.05 * min(v["total"] - 1, 4))
        views.append({
            "asset": asset,
            "direction": direction,
            "confidence": clamp_confidence(strength),
            "reasoning": "；".join(v["notes"][:3]) or f"主题命中 {asset}",
        })
    return views


def factor_to_theme(factor_name: str, factor_description: str, metrics: Optional[dict] = None) -> Optional[str]:
    """把因子元信息折叠成宏观主题文本（启发式，确定性）。

    只在描述/名称命中宏观措辞时返回主题（可能为空串表示"无方向信号"）；
    否则返回 None，示意调用方本轮不生成影子预测。
    """
    blob = f"{factor_name or ''} {factor_description or ''}".strip().lower()
    if not blob:
        return None
    macro_hits = [grp for _kw, _asset, _d, _c, _note in THEME_RULES
                  for grp in _kw if grp.lower() in blob]
    if not macro_hits:
        return None
    return blob
