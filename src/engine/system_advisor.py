"""AI 体系咨询引擎。

因子体系分析会把一次回测拆成几十个数字，但研究者拿到数字后真正要回答的只有几个决策
问题：这套体系能不能进组合、哪几个因子在重复配置、风险压在谁身上、下一步先改哪一处。
本模块把回测结果压缩成一份**事实表**（:func:`distill`），再让两条回答路径共用它：

* **规则路径**（:func:`build_rule_answer`）——纯本地、无外部依赖，按意图把事实翻译成
  结论与动作。它保证「没有模型可用时仍然拿得到可执行答案」，同时充当 LLM 答案的数字基准。
* **LLM 路径**（:func:`advise` 传入 ``llm``）——事实表作为唯一上下文注入提示词，并显式
  禁止引用未出现的数字，避免模型用「看起来合理但不存在」的数字编出一段像样的分析。

两条路径共用同一份 facts，所以新增指标只需改一处，不会出现「界面上有这个数、咨询里引用
不了」的错位。判据阈值与 :func:`engine.factor_system.build_findings` 同源（IC≥0.03 且
ICIR≥0.4 视为有效、|ρ|≥0.8 视为重复配置、单一因子风险占比≥40% 视为集中），否则同一份
回测会在两个地方给出互相矛盾的判词。

约定：

* 引擎层不依赖 Streamlit，也不自己创建 LLM 客户端——客户端由调用方注入，便于离线测试。
* LLM 调用失败不抛异常：退回规则答案，并把失败原因记在结果的 ``error`` 字段上。咨询窗口
  不该因为模型不可用而空白一片。
* 每条动作都带「依据」，数字全部来自事实表，不做模型外推，也不构成投资建议。
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .factor_system import WEIGHT_LABELS, build_findings

__all__ = [
    "ADVISOR_SYSTEM_PROMPT",
    "DEFAULT_INTENT",
    "INTENTS",
    "advise",
    "build_actions",
    "build_rule_answer",
    "distill",
    "facts_markdown",
    "match_intent",
    "suggest_questions",
]


# ---------------------------------------------------------------------------
# 判据阈值与常量
# ---------------------------------------------------------------------------
# 与 engine.factor_system.build_findings 保持同源，避免同一份回测出现两套判词。
IC_STRONG = 0.03
ICIR_STRONG = 0.4
IC_WEAK = 0.015
ICIR_WEAK = 0.2
CORR_REDUNDANT = 0.8
RISK_CONCENTRATED = 0.4
NOISE_HEAVY = 0.5
DECAY_KEEP_WEAK = 0.5
TURNOVER_HIGH = 0.4
VARIANCE_CUT_WORTHWHILE = 1.0
ADDITION_WORTHWHILE = 1.0
WEAK_FACTOR_ICIR = 0.1

# 成本口径取自 config.yaml 的 backtest.commission（单边手续费率），
# 与 FactorBacktester.realistic_portfolio 的 cost = Σ|Δw| × commission 一致。
COMMISSION = 0.001
TRADING_DAYS = 252

MAX_HISTORY_TURNS = 6
MAX_ACTIONS = 6

ADVISOR_SYSTEM_PROMPT = """你是 FactorGPT 的因子体系顾问，服务对象是正在搭建多因子体系的研究者。

工作方式：
1. 用户会先给你一份「体系事实表」，那是本次回测**唯一**可信的数据来源。
2. 回答必须建立在事实表上：每个结论后面跟具体数字与对应指标名。
3. 严禁使用事实表以外的数字，严禁自行估算、补齐或引用外部行情/研报数据。事实表里没有
   的指标，直接说明「本次回测未产出该指标」。
4. 事实表显示样本不足、指标缺失或口径存疑时，先指出这一点，再给结论；不确定就说不确定。
5. 涉及取舍时给出量化对比（例如「切到最小方差方案，体系波动从 0.3898 降到 0.3719，
   降 4.60%」），不要只说「建议优化权重」。

输出结构（用 Markdown，不要代码块）：
- **结论**：一句话回答用户的问题。
- **依据**：2~4 条，每条都带事实表中的数字。
- **动作**：不超过 3 条可执行建议，标注预期效果。
- **提示**：一句话风险或口径提醒（样本区间、估计误差、成本假设等）。

语气：直接、克制、像同事讨论问题。不确定时明说，不要用「可能」「或许」堆砌空话。
本回答是研究流程的内部参考，不构成投资建议。"""

INTENTS: Dict[str, str] = {
    "overview": "整体成色",
    "overfit": "过拟合与可信度",
    "spectrum": "谱清洗与估计噪声",
    "redundancy": "冗余与重复配置",
    "weight": "权重方案",
    "risk": "风险集中度",
    "decay": "信号衰减与持有期",
    "prune": "增删因子",
    "capacity": "容量与成本",
    "next": "下一步动作",
    "general": "体系概览",
}

DEFAULT_INTENT = "general"

# 意图识别：越具体的模式越靠前，命中即返回。注意「一个词只归一个意图」——
# 例如「换手」归衰减（持有期取舍），「成本」归容量，两者混用会把问题导错模板。
_INTENT_PATTERNS: List[Tuple[str, str]] = [
    ("spectrum", r"谱清洗|特征值|噪声带|噪声|随机矩阵|条件数|带内|eigen|marchenko|mp\s*带"),
    ("overfit", r"过拟合|过度拟合|过拟|可信|靠谱|真实|样本外|稳健|显著|存活|偷价|未来函数|overfit"),
    ("redundancy", r"冗余|重复|同质|共线|collinear|相关.*(高|重复)|撞车|重叠|同质化"),
    ("weight", r"权重|配权|加权|等权|最小方差|风险平价|分散化权重|weight"),
    ("capacity", r"容量|资金|规模|冲击|成本|手续费|滑点|流动性|capacity|换手成本"),
    ("decay", r"衰减|持有期|调仓|换手|周期|多长|holding|decay|turnover"),
    ("prune", r"加因子|加新因子|要不要加|该不该加|值得加|新增|纳入|剔除|删掉|删减|精简|扩建|留哪些|去掉|替换"),
    ("risk", r"风险|回撤|集中|波动|暴露|单因子.*(占比|贡献)|drawdown"),
    ("next", r"下一步|接下来|怎么改|改进|优化|建议|怎么办|从哪|优先|action|next"),
    ("overview", r"整体|概况|怎么样|如何|好不好|成色|有用|能不能用|值不值得|评价|体检|overview"),
]


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _num(v: Any, default: float = float("nan")) -> float:
    """安全取数：任何不可转成有限浮点的输入都退化为 default。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _val(v: Any, digits: int = 4, suffix: str = "", default: str = "—") -> str:
    f = _num(v)
    return default if not math.isfinite(f) else f"{f:.{digits}f}{suffix}"


def _pct(v: Any, digits: int = 1, default: str = "—") -> str:
    f = _num(v)
    return default if not math.isfinite(f) else f"{f * 100:.{digits}f}%"


def _strip_tags(text: Any) -> str:
    return re.sub(r"<[^>]+>", "", str(text or "")).strip()


def _level(ic: float, icir: float) -> str:
    if abs(_num(ic, 0.0)) >= IC_STRONG and abs(_num(icir, 0.0)) >= ICIR_STRONG:
        return "strong"
    if abs(_num(ic, 0.0)) >= IC_WEAK and abs(_num(icir, 0.0)) >= ICIR_WEAK:
        return "weak"
    return "none"


_LEVEL_TEXT = {
    "strong": "达到可以进入组合构建阶段的水准",
    "weak": "偏弱，先把信号修到能用，再谈权重与风控",
    "none": "横截面预测力基本不可用，先不要动权重",
}


def _name_of(members: Sequence[Dict[str, Any]], factor_name: str) -> str:
    """因子名 → 展示名。体系成员表里存的是代码名，展示名更贴近界面。"""
    for m in members or []:
        if str(m.get("factor_name", "")) == factor_name:
            return str(m.get("display_name") or factor_name)
    return factor_name


def _scheme_label(sol: Dict[str, Any]) -> str:
    return str(sol.get("label") or sol.get("mode") or "")


# ---------------------------------------------------------------------------
# 1. 事实表
# ---------------------------------------------------------------------------
def distill(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """把一次体系回测结果压缩成咨询用的事实表。

    事实表是咨询的唯一数据来源：规则模板直接引用它，LLM 提示词里注入它。所有值都是
    普通 Python 类型（float / int / str / list / dict），可以直接放进 session state。

    Args:
        result: :func:`engine.factor_system.analyze_system` 的返回值。

    Returns:
        ``ok=False`` 时只带 ``reason``；``ok=True`` 时含指标、结构、谱清洗、风险、
        权重对照、边际价值、衰减、维度与结论等分组字段。
    """
    if not result or result.get("error") or not result.get("composite_metrics"):
        reason = "尚未运行体系回测"
        if result:
            reason = str(result.get("error") or reason)
        return {"ok": False, "reason": reason}

    cm = result.get("composite_metrics") or {}
    corr = result.get("correlation") or {}
    spec = result.get("spectral") or {}
    spec_ok = bool(isinstance(spec, dict) and spec.get("ok"))
    sd = (spec.get("spectrum_dict") or {}) if spec_ok else {}

    members = result.get("members") or []
    factor_stats = result.get("factor_stats") or {}

    facts: Dict[str, Any] = {
        "ok": True,
        "reason": "",
        "system": str((result.get("_system") or {}).get("name") or ""),
        "n_factors": int(result.get("n_factors", 0) or 0),
        "weight_mode": str(result.get("weight_mode") or ""),
        "weight_label": WEIGHT_LABELS.get(str(result.get("weight_mode") or ""), "未知方案"),
        # 组合层指标
        "ic": _num(cm.get("ic")),
        "rank_ic": _num(cm.get("rank_ic")),
        "icir": _num(cm.get("icir")),
        "ic_win": _num(cm.get("ic_positive_ratio")),
        "sharpe": _num(cm.get("long_short_sharpe")),
        "ls_daily": _num(cm.get("long_short_return")),
        "ls_cum": _num(cm.get("long_short_cum_return")),
        "mdd": _num(cm.get("max_drawdown")),
        "turnover": _num(cm.get("turnover")),
        "coverage": _num(cm.get("coverage")),
        "n_dates": int(_num(cm.get("n_dates"), 0.0)),
        "n_stocks": int(_num(cm.get("n_stocks"), 0.0)),
        # 结构
        "mean_abs_corr": _num(corr.get("mean_abs_corr")),
        "max_abs_corr": _num(corr.get("max_abs_corr")),
        "effective_factors": _num(corr.get("effective_factors")),
        "redundant_pairs": list(corr.get("redundant_pairs") or []),
        "members": members,
        "errors": dict(result.get("errors") or {}),
    }
    facts["level"] = _level(facts["ic"], facts["icir"])
    facts["ls_annual"] = facts["ls_daily"] * TRADING_DAYS if math.isfinite(facts["ls_daily"]) else float("nan")

    # 谱清洗
    facts.update({
        "spectral_ok": spec_ok,
        "spectral_reason": str(spec.get("reason") or "") if isinstance(spec, dict) else "",
        "n_obs": int(_num(sd.get("n_obs"), 0.0)),
        "q": _num(sd.get("q")),
        "lam_low": _num(sd.get("lambda_minus")),
        "lam_high": _num(sd.get("lambda_plus")),
        "noise_ratio": _num(sd.get("noise_ratio")),
        "n_signal": int(_num(sd.get("n_signal"), 0.0)),
        "n_inband": int(_num(sd.get("n_inband"), 0.0)),
        "inband_ratio": _num(sd.get("inband_ratio")),
        "cond_before": _num(sd.get("cond_before")),
        "cond_after": _num(sd.get("cond_after")),
        "eff_before": _num(sd.get("effective_factors_before")),
        "eff_after": _num(sd.get("effective_factors_after")),
        "spectrum_notes": [],
    })
    # 清洗日志优先取 SpectrumReport 自己的摘要，保证与仪表盘同一份措辞
    report = spec.get("spectrum") if spec_ok else None
    if report is not None and hasattr(report, "summary_lines"):
        facts["spectrum_notes"] = [_strip_tags(n) for n in report.summary_lines()]
    if not facts["spectrum_notes"]:
        facts["spectrum_notes"] = [_strip_tags(n) for n in (sd.get("notes") or [])]

    bias = (spec.get("bias") or {}) if spec_ok else {}
    facts.update({
        "vol_raw": _num(bias.get("vol_raw")),
        "vol_clean": _num(bias.get("vol_clean")),
        "vol_gap_pct": _num(bias.get("vol_gap_pct")),
        "fake_div_pct": _num(bias.get("fake_div_pct")),
    })

    risk = spec.get("risk") if spec_ok else None
    facts.update({
        "risk_vol": _num(getattr(risk, "portfolio_vol", float("nan"))),
        "div_ratio": _num(getattr(risk, "diversification_ratio", float("nan"))),
        "eff_n_risk": _num(getattr(risk, "effective_n_risk", float("nan"))),
        "risk_hhi": _num(getattr(risk, "hhi", float("nan"))),
        "top_risk": _name_of(members, str(getattr(risk, "top_risk", "") or "")),
        "top_risk_pct": _num(getattr(risk, "top_risk_pct", float("nan"))),
        # 同样复用 RiskDecomposition 自己的摘要
        "risk_summary": (
            [_strip_tags(x) for x in risk.summary_lines()]
            if risk is not None and hasattr(risk, "summary_lines") else []
        ),
    })

    # 权重方案对照：以清洗后矩阵为准（样本矩阵高估分散化，别拿它选方案）
    solutions = [s for s in (spec.get("solutions_dict") or []) if s.get("success", True)]
    current = None
    for s in solutions:
        if str(s.get("mode")) == facts["weight_mode"]:
            current = s
            break
    best = None
    for s in solutions:
        if not math.isfinite(_num(s.get("variance"))):
            continue
        if best is None or _num(s.get("variance")) < _num(best.get("variance")):
            best = s
    facts["current_scheme"] = {
        "key": str(current.get("mode")) if current else facts["weight_mode"],
        "label": _scheme_label(current) if current else facts["weight_label"],
        "vol": _num(current.get("vol")) if current else facts["risk_vol"],
        "variance": _num(current.get("variance")) if current else float("nan"),
        "expected_icir": _num(current.get("expected_icir")) if current else float("nan"),
    }
    facts["best_scheme"] = {
        "key": str(best.get("mode")) if best else "",
        "label": _scheme_label(best) if best else "—",
        "vol": _num(best.get("vol")) if best else float("nan"),
        "variance": _num(best.get("variance")) if best else float("nan"),
        "expected_icir": _num(best.get("expected_icir")) if best else float("nan"),
    }
    cur_var = facts["current_scheme"]["variance"]
    best_var = facts["best_scheme"]["variance"]
    facts["variance_cut_pct"] = (
        (cur_var - best_var) / cur_var * 100.0
        if math.isfinite(cur_var) and math.isfinite(best_var) and cur_var > 0 else float("nan")
    )
    facts["same_scheme"] = bool(best and str(best.get("mode")) == facts["weight_mode"])

    # 边际价值：区分「未在体系内」（新增价值）与「已在体系内」（增配价值）
    addition = []
    for row in (spec.get("addition") or []):
        nm = str(row.get("name") or "")
        addition.append({
            "name": _name_of(members, nm),
            "factor_name": nm,
            "alpha": _num(row.get("alpha")),
            "vol_cut_pct": _num(row.get("vol_reduction_pct")),
            "corr": _num(row.get("corr_with_system")),
            "in_system": bool(_num(row.get("in_system"), 0.0) > 0),
            "weight_now": _num(row.get("weight_now")),
        })
    facts["addition"] = addition

    # 衰减曲线
    decay = []
    for row in (result.get("decay") or []):
        decay.append({
            "period": int(_num(row.get("period"), 0.0)),
            "ic": _num(row.get("ic")),
            "rank_ic": _num(row.get("rank_ic")),
            "icir": _num(row.get("icir")),
        })
    facts["decay"] = decay
    keep = float("nan")
    if len(decay) >= 2 and decay[0]["ic"]:
        keep = abs(decay[-1]["ic"]) / abs(decay[0]["ic"])
    facts["decay_keep"] = keep

    # 维度与强弱因子
    facts["dimensions"] = list(result.get("dimensions") or [])
    usable = [
        {"name": _name_of(members, k), "factor_name": k,
         "ic": _num(v.get("ic")), "icir": _num(v.get("icir"))}
        for k, v in factor_stats.items() if isinstance(v, dict) and "error" not in v
    ]
    ranked = sorted(usable, key=lambda d: abs(_num(d["icir"], 0.0)))
    facts["weak_factors"] = ranked[:3]
    facts["strong_factors"] = list(reversed(ranked[-3:]))
    facts["n_evaluated"] = len(usable)

    # 复用的诊断结论（带 HTML 标签，这里剥掉以便直接进提示词）
    facts["findings"] = [
        {"tone": str(f.get("tone", "info")), "text": _strip_tags(f.get("text"))}
        for f in build_findings(result)
    ]
    return facts


def facts_markdown(facts: Dict[str, Any]) -> str:
    """把事实表渲染成提示词用的紧凑文本（也是界面「本次咨询依据」的展示内容）。"""
    if not facts or not facts.get("ok"):
        return f"（无可用回测结果：{facts.get('reason', '未知原因') if facts else '未知原因'}）"

    lines = [
        f"体系：{facts.get('system') or '未命名'}（{facts['n_factors']} 个因子，"
        f"权重方案 {facts['weight_label']}）",
        f"样本：{facts['n_dates']} 个交易日 × {facts['n_stocks']} 只股票"
        f"｜平均覆盖 {_pct(facts['coverage'], 0)}",
        "",
        "【组合层表现（合成因子）】",
        f"IC={_val(facts['ic'])}｜RankIC={_val(facts['rank_ic'])}｜ICIR={_val(facts['icir'], 2)}"
        f"｜IC 胜率={_pct(facts['ic_win'], 0)}",
        f"多空日均收益={_val(facts['ls_daily'], 4)}（×252 近似年化 {_pct(facts['ls_annual'])}）"
        f"｜多空夏普={_val(facts['sharpe'], 2)}｜最大回撤={_pct(facts['mdd'])}",
        f"日均换手={_val(facts['turnover'], 3)}"
        f"（Σ|Δw| 口径，年化成本量级≈{_pct(facts['turnover'] * TRADING_DAYS * COMMISSION)}）",
        "",
        "【结构】",
        f"平均绝对相关={_val(facts['mean_abs_corr'], 3)}｜最大绝对相关={_val(facts['max_abs_corr'], 3)}"
        f"｜有效因子数（方差解释率口径）={_val(facts['effective_factors'], 2)}",
    ]
    if facts["redundant_pairs"]:
        pairs = "；".join(
            f"{p.get('a')} 与 {p.get('b')} ρ={_num(p.get('corr')):.2f}"
            for p in facts["redundant_pairs"][:5]
        )
        lines.append(f"|ρ|≥{CORR_REDUNDANT} 的重复对（最多 5 组）：{pairs}")
    else:
        lines.append(f"|ρ|≥{CORR_REDUNDANT} 的重复对：无")

    if facts["spectral_ok"]:
        lines += [
            "",
            "【谱清洗（随机矩阵理论，MP 噪声带）】",
            f"有效观测数 T={facts['n_obs']}｜纵横比 q=N/T={_val(facts['q'], 3)}"
            f"｜噪声带 [{_val(facts['lam_low'], 4)}, {_val(facts['lam_high'], 4)}]",
            f"信号方向（>上界）={facts['n_signal']} 个｜带内无法判定={facts['n_inband']} 个"
            f"（占 {_pct(facts['inband_ratio'], 0)}）｜明确噪声方向（<下界）占 {_pct(facts['noise_ratio'], 0)}",
            f"条件数 {_val(facts['cond_before'], 1)} → 清洗后 {_val(facts['cond_after'], 1)}"
            f"｜有效因子数 {_val(facts['eff_before'], 2)} → {_val(facts['eff_after'], 2)}",
            f"同一组权重：原始矩阵下 σ={_val(facts['vol_raw'])}，清洗后 σ={_val(facts['vol_clean'])}"
            f"（波动偏差 {_val(facts['vol_gap_pct'], 2)}%，虚假分散化 {_val(facts['fake_div_pct'], 2)}%）",
            "风险分解（清洗后矩阵）：" + ("；".join(facts["risk_summary"]) or "未产出"),
            f"权重方案对照（清洗后矩阵）：当前 {facts['current_scheme']['label']} σ={_val(facts['current_scheme']['vol'])}；"
            f"方差最小 {facts['best_scheme']['label']} σ={_val(facts['best_scheme']['vol'])}"
            f"（比当前降 {_val(facts['variance_cut_pct'], 2)}%）",
        ]
        if facts["addition"]:
            top = facts["addition"][:5]
            lines.append("边际价值排行（按纳入后体系波动下降幅度）：" + "；".join(
                f"{d['name']}{'（体系内）' if d['in_system'] else '（体系外）'}"
                f" α*={_val(d['alpha'], 2)} 降幅={_val(d['vol_cut_pct'], 2)}%"
                f" 与体系相关={_val(d['corr'], 2)}" for d in top
            ))
    else:
        lines += ["", f"【谱清洗】未执行：{facts['spectral_reason'] or '因子数不足或已关闭'}"]

    if facts["decay"]:
        lines += ["", "【IC 衰减】" + "；".join(
            f"{d['period']}日 IC={_val(d['ic'])} ICIR={_val(d['icir'], 2)}" for d in facts["decay"]
        )]
        if math.isfinite(facts["decay_keep"]):
            lines.append(f"最长持有期相对 1 日的 IC 保留比例={_pct(facts['decay_keep'], 0)}")

    if facts["dimensions"]:
        lines += ["", "【维度结构】" + "；".join(
            f"{d.get('dimension')}（{d.get('n_factors')} 个，权重 {_pct(d.get('weight'))}，"
            f"平均 ICIR={_val(d.get('mean_icir'), 2)}）" for d in facts["dimensions"][:8]
        )]

    weak = [w for w in facts["weak_factors"] if abs(_num(w["icir"], 0.0)) < WEAK_FACTOR_ICIR]
    if weak:
        lines += ["", "【疲软因子】" + "；".join(
            f"{w['name']} IC={_val(w['ic'])} ICIR={_val(w['icir'], 2)}" for w in weak
        )]
    if facts["errors"]:
        lines += ["", "【计算失败】" + "；".join(f"{k}: {v}" for k, v in facts["errors"].items())]

    if facts["findings"]:
        lines += ["", "【系统诊断结论】"] + [f"- {f['text']}" for f in facts["findings"]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. 动作清单
# ---------------------------------------------------------------------------
def build_actions(facts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把事实表翻译成带优先级的动作清单（数字越小越先做）。

    规则路径与界面共用同一份动作，避免「聊天里说了一套、页面上列了另一套」。
    """
    if not facts or not facts.get("ok"):
        return [{
            "priority": 0,
            "title": "先跑一次体系回测",
            "detail": "咨询需要回测产出的指标作为依据，当前没有可用的体系分析结果。",
            "evidence": facts.get("reason", "") if facts else "",
        }]

    actions: List[Dict[str, Any]] = []

    if facts["level"] != "strong":
        actions.append({
            "priority": 1,
            "title": "先修信号，再谈权重",
            "detail": f"合成因子 IC={_val(facts['ic'])}、ICIR={_val(facts['icir'], 2)}，"
                      f"{_LEVEL_TEXT[facts['level']]}。优先核查因子方向设置、样本区间与"
                      f"极端值处理，把 ICIR 提到 {ICIR_STRONG} 以上再进入权重优化。",
            "evidence": f"IC={_val(facts['ic'])} / ICIR={_val(facts['icir'], 2)} / "
                        f"IC 胜率={_pct(facts['ic_win'], 0)}",
        })

    if facts["redundant_pairs"]:
        p = facts["redundant_pairs"][0]
        actions.append({
            "priority": 1,
            "title": f"处理重复因子对 {p.get('a')} 与 {p.get('b')}",
            "detail": f"两者相关系数 {_num(p.get('corr')):.2f}（阈值 {CORR_REDUNDANT}），"
                      f"合计 {len(facts['redundant_pairs'])} 组重复配置。等权下它们等于同一份"
                      f"暴露被投了两次：二选一，或对其中一个做残差正交化后再入体系。",
            "evidence": f"平均绝对相关={_val(facts['mean_abs_corr'], 3)}，"
                        f"有效因子数={_val(facts['effective_factors'], 2)}（共 {facts['n_factors']} 个）",
        })

    if facts["spectral_ok"]:
        noise_heavy = (_num(facts["noise_ratio"], 0.0) >= NOISE_HEAVY
                       or _num(facts["inband_ratio"], 0.0) >= NOISE_HEAVY)
        if noise_heavy:
            actions.append({
                "priority": 2,
                "title": "用清洗后矩阵重估权重与风险",
                "detail": f"谱清洗显示 {_pct(facts['noise_ratio'], 0)} 的特征方向明确低于噪声带下界、"
                          f"{_pct(facts['inband_ratio'], 0)} 落在带内无法判定，样本相关矩阵的"
                          f"估计误差已不可忽略。样本内优化出来的权重会高估分散化，"
                          f"请在清洗后矩阵上重解一次再比较。",
                "evidence": f"q=N/T={_val(facts['q'], 3)}，噪声带 "
                            f"[{_val(facts['lam_low'], 4)}, {_val(facts['lam_high'], 4)}]，"
                            f"σ {_val(facts['vol_raw'])} → {_val(facts['vol_clean'])}"
                            f"（{_val(facts['vol_gap_pct'], 2)}%）",
            })
        if (math.isfinite(_num(facts["variance_cut_pct"]))
                and _num(facts["variance_cut_pct"]) >= VARIANCE_CUT_WORTHWHILE
                and not facts["same_scheme"]):
            actions.append({
                "priority": 2,
                "title": f"权重方案切到「{facts['best_scheme']['label']}」",
                "detail": f"清洗后矩阵下方差最小的方案不是当前的「{facts['current_scheme']['label']}」："
                          f"切换后体系波动 {_val(facts['current_scheme']['vol'])} → "
                          f"{_val(facts['best_scheme']['vol'])}，方差降 "
                          f"{_val(facts['variance_cut_pct'], 2)}%。"
                          f"切换前确认该方案的 ICIR 预期没有明显下降。",
                "evidence": f"期望 ICIR 当前 {_val(facts['current_scheme']['expected_icir'], 2)} → "
                            f"{_val(facts['best_scheme']['expected_icir'], 2)}",
            })
        if _num(facts["top_risk_pct"], 0.0) >= RISK_CONCENTRATED:
            actions.append({
                "priority": 3,
                "title": f"给 {facts['top_risk'] or '最大风险因子'} 设权重上限",
                "detail": f"该因子的风险贡献占体系 {_pct(facts['top_risk_pct'])}，"
                          f"而有效风险来源只有 {_val(facts['eff_n_risk'], 2)} 个。"
                          f"权重上限（如 25%）能把集中度压回均值附近，代价是方差略升。",
                "evidence": f"HHI={_val(facts['risk_hhi'], 3)}，分散化比率={_val(facts['div_ratio'], 3)}",
            })

    outside = [d for d in facts["addition"]
               if not d["in_system"] and _num(d["vol_cut_pct"], 0.0) >= ADDITION_WORTHWHILE]
    if outside:
        d = outside[0]
        actions.append({
            "priority": 3,
            "title": f"值得纳入的候选因子：{d['name']}",
            "detail": f"把它按 α*={_val(d['alpha'], 2)} 混入现有体系，波动可降 "
                      f"{_val(d['vol_cut_pct'], 2)}%；与体系的相关性只有 {_val(d['corr'], 2)}，"
                      f"说明它覆盖了现有因子没有的风险方向。",
            "evidence": f"候选排行第二："
                        + (f"{outside[1]['name']} 降幅 {_val(outside[1]['vol_cut_pct'], 2)}%"
                           if len(outside) > 1 else "无"),
        })

    if facts["decay"] and math.isfinite(_num(facts["decay_keep"])) \
            and _num(facts["decay_keep"], 1.0) < DECAY_KEEP_WEAK:
        actions.append({
            "priority": 4,
            "title": "缩短持有期或按换手成本重筛",
            "detail": f"IC 从 1 日的 {_val(facts['decay'][0]['ic'])} 衰减到 "
                      f"{facts['decay'][-1]['period']} 日的 {_val(facts['decay'][-1]['ic'])}"
                      f"（保留 {_pct(facts['decay_keep'], 0)}），信号属于短周期类型。"
                      f"持有期拉长会显著损失 IC，请把成本假设写进筛选目标再决定调仓频率。",
            "evidence": f"日均换手={_val(facts['turnover'], 3)}，"
                        f"年化成本量级≈{_pct(facts['turnover'] * TRADING_DAYS * COMMISSION)}",
        })

    weak = [w for w in facts["weak_factors"] if abs(_num(w["icir"], 0.0)) < WEAK_FACTOR_ICIR]
    if weak and facts["n_factors"] > 3:
        w = weak[0]
        actions.append({
            "priority": 4,
            "title": f"剔除或替换疲软因子：{w['name']}",
            "detail": f"该因子 ICIR={_val(w['icir'], 2)}（贡献接近噪声），"
                      f"保留它只会稀释权重、增加维护成本。建议先做一次剔除前后的对照回测。",
            "evidence": f"体系内共 {len(facts['weak_factors'])} 个因子 |ICIR| 偏低，"
                        f"最低 {_val(facts['weak_factors'][0]['icir'], 2)}",
        })

    if facts["errors"]:
        actions.append({
            "priority": 1,
            "title": "修掉计算失败的因子",
            "detail": f"{len(facts['errors'])} 个因子在本次回测中报错，已从体系里剔除，"
                      f"但它们仍在体系定义中占位。先修代码或移出体系，再重新评估结构指标。",
            "evidence": "；".join(f"{k}: {v}" for k, v in list(facts["errors"].items())[:3]),
        })

    actions.sort(key=lambda a: a["priority"])
    return actions[:MAX_ACTIONS]


# ---------------------------------------------------------------------------
# 3. 意图识别与规则答案
# ---------------------------------------------------------------------------
def match_intent(question: str) -> str:
    """关键词路由：把自然语言问题映射到某个意图。

    规则引擎与 LLM 路径共用它——LLM 路径用它决定注入哪一段重点提示，规则路径直接用它
    选模板。识别不出来时落到 ``general``，由模板告诉用户能问什么，而不是硬猜。
    """
    text = str(question or "").strip().lower()
    if not text:
        return DEFAULT_INTENT
    for intent, pattern in _INTENT_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return intent
    return DEFAULT_INTENT


def _answer_overview(f: Dict[str, Any]) -> str:
    lines = [
        f"**结论**：{_LEVEL_TEXT[f['level']]}。",
        "",
        f"**依据**：合成因子 IC={_val(f['ic'])}、ICIR={_val(f['icir'], 2)}、"
        f"IC 胜率 {_pct(f['ic_win'], 0)}、多空夏普 {_val(f['sharpe'], 2)}、"
        f"最大回撤 {_pct(f['mdd'])}。",
        f"结构上 {f['n_factors']} 个因子折合有效维度 {_val(f['effective_factors'], 2)} 个"
        f"（平均绝对相关 {_val(f['mean_abs_corr'], 3)}），当前权重方案「{f['weight_label']}」；"
        f"日均换手 {_val(f['turnover'], 3)}。",
    ]
    if f["spectral_ok"]:
        lines.append(
            f"谱清洗口径下，{_pct(f['noise_ratio'], 0)} 的特征方向明确低于噪声带下界、"
            f"{_pct(f['inband_ratio'], 0)} 落在带内无法判定，"
            f"即样本相关矩阵中只有 {f['n_signal']} 个方向可以算作信号。"
        )
    nexts = build_actions(f)
    if nexts:
        lines += ["", f"**最该先做的一件事**：{nexts[0]['title']}——{nexts[0]['detail']}"]
    lines += ["", f"**提示**：以上均为样本内 {f['n_dates']} 个交易日的回测结果，"
                  f"不构成投资建议。"]
    return "\n".join(lines)


def _answer_overfit(f: Dict[str, Any]) -> str:
    lines = [
        "**结论**：可信度要从三个角度同时看——统计显著性、样本量、以及相关矩阵的估计噪声，"
        f"本次回测后两项都{'成立' if f['spectral_ok'] else '无法判定（谱清洗未执行）'}。",
        "",
        f"**依据**：IC={_val(f['ic'])}、ICIR={_val(f['icir'], 2)}（多空夏普 {_val(f['sharpe'], 2)}），"
        f"样本 {f['n_dates']} 个交易日、平均覆盖 {_pct(f['coverage'], 0)}；"
        f"IC 胜率 {_pct(f['ic_win'], 0)}，"
        + ("胜率接近五成时 IC 的均值更容易是噪声。" if _num(f["ic_win"], 0.0) < 0.55
           else "胜负分布不算极端，均值 IC 有一定稳定性。"),
    ]
    if f["spectral_ok"]:
        lines.append(
            f"估计噪声方面：q=N/T={_val(f['q'], 3)}，噪声带 "
            f"[{_val(f['lam_low'], 4)}, {_val(f['lam_high'], 4)}]，"
            f"{_pct(f['noise_ratio'], 0)} 个方向低于下界（明确是估计噪声）、"
            f"{_pct(f['inband_ratio'], 0)} 个落在带内（该样本量下无法判定真假）。"
        )
        if math.isfinite(_num(f["vol_gap_pct"])):
            if _num(f["vol_gap_pct"]) > 0:
                lines.append(
                    f"同一组权重在原始矩阵下 σ={_val(f['vol_raw'])}，清洗后 σ={_val(f['vol_clean'])}，"
                    f"原始矩阵低估波动 {_val(f['vol_gap_pct'], 2)}%——"
                    f"这意味着用样本矩阵做优化会得到「样本内看着分散、样本外并不成立」的权重。"
                )
            else:
                lines.append(
                    f"同一组权重在两套矩阵下的波动几乎一样（{_val(f['vol_raw'])} vs "
                    f"{_val(f['vol_clean'])}），本样本中估计误差没有明显扭曲风险刻画。"
                )
    else:
        lines.append(f"谱清洗未执行：{f['spectral_reason'] or '因子数不足或已关闭'}，"
                     f"因此无法给出估计噪声的定量判断。")
    if f["decay"]:
        lines.append(
            f"衰减角度：IC 从 1 日的 {_val(f['decay'][0]['ic'])} 变到 "
            f"{f['decay'][-1]['period']} 日的 {_val(f['decay'][-1]['ic'])}。"
            "衰减越快，样本内表现越依赖特定调仓节奏。"
        )
    lines += [
        "",
        "**动作**",
        f"1. 用「因子体系分析 → 谱清洗与风险」页的清洗后矩阵复算一次权重，与当前结果对比。",
        f"2. 把回测区间至少再延长一倍（当前 {f['n_dates']} 个交易日）后再看 ICIR 是否还在 "
        f"{_val(f['icir'], 2)} 附近；这个指标对区间长度最敏感。",
        "3. 剔除或替换 |ICIR| 偏低的因子后重跑，观察体系 ICIR 是升还是降——升说明原来在摊薄。",
        "",
        f"**提示**：IC 与 ICIR 都是样本内统计量，{f['n_dates']} 个交易日的 ICIR 标准误"
        f"大致在 1/√T 量级，别把第三位小数当真。",
    ]
    return "\n".join(lines)


def _answer_redundancy(f: Dict[str, Any]) -> str:
    lines = [
        f"**结论**：{f['n_factors']} 个因子折合有效维度只有 "
        f"{_val(f['effective_factors'], 2)} 个，"
        + (f"存在 {len(f['redundant_pairs'])} 组 |ρ|≥{CORR_REDUNDANT} 的重复配置。"
           if f["redundant_pairs"] else "没有出现 |ρ|≥0.8 的重复对，冗余主要来自弱相关方向的堆积。"),
        "",
        f"**依据**：平均绝对相关 {_val(f['mean_abs_corr'], 3)}、最大绝对相关 "
        f"{_val(f['max_abs_corr'], 3)}。",
    ]
    if f["redundant_pairs"]:
        for p in f["redundant_pairs"][:4]:
            lines.append(f"- {p.get('a')} 与 {p.get('b')}：ρ={_num(p.get('corr')):.2f}")
    if f["spectral_ok"]:
        lines.append(
            f"谱口径：噪声带 [{_val(f['lam_low'], 4)}, {_val(f['lam_high'], 4)}]，"
            f"只有 {f['n_signal']} 个方向超过上界，"
            f"{f['n_inband']} 个方向落在带内——带内方向既不能算信号也不能算噪声，"
            f"它们的「重复」在本样本量下无法证实。"
        )
    if f["addition"]:
        ins = [d for d in f["addition"] if d["in_system"]]
        if ins:
            lines.append(
                "边际价值视角：" + "；".join(
                    f"{d['name']} 当前权重 {_pct(d['weight_now'])}，"
                    f"按 α*={_val(d['alpha'], 2)} 增配后体系波动仅降 {_val(d['vol_cut_pct'], 2)}%"
                    for d in ins[:3]
                ) + "——降幅接近 0 说明这些因子彼此替代性很强。"
            )
    lines += [
        "",
        "**动作**",
        "1. 对每组重复对做二选一：保留 ICIR 更高、维护成本更低的那一个。",
        "2. 若两个都要留，把其中一个对另一个做残差正交化后再入体系，避免同一份暴露被重复下注。",
        "3. 重跑后确认有效维度上升、平均绝对相关下降，再比较体系 ICIR 是否受损。",
        "",
        "**提示**：相关矩阵本身由有限样本估计，"
        + (f"当前 q=N/T={_val(f['q'], 3)}，" if f["spectral_ok"] else "")
        + "小样本下的高相关既可能是真重复，也可能是共同的市场暴露，剔除前先看两者的经济逻辑。",
    ]
    return "\n".join(lines)


def _answer_weight(f: Dict[str, Any]) -> str:
    cur, best = f["current_scheme"], f["best_scheme"]
    lines = [
        f"**结论**：当前权重方案是「{cur['label']}」；"
        + (f"在清洗后矩阵下它已经是方差最小的方案。" if f["same_scheme"]
           else f"在清洗后矩阵下方差最小的方案是「{best['label']}」，"
                f"切换后体系方差可降 {_val(f['variance_cut_pct'], 2)}%。"),
        "",
        f"**依据**：清洗后矩阵口径 σ 当前 {_val(cur['vol'])}，"
        + (f"最优 {_val(best['vol'])}" if not f["same_scheme"] else f"最优同为 {_val(best['vol'])}")
        + f"；期望 ICIR {_val(cur['expected_icir'], 2)}"
        + (f" → {_val(best['expected_icir'], 2)}" if not f["same_scheme"] else "")
        + "。",
    ]
    if f["spectral_ok"]:
        lines.append(
            f"风险分解：分散化比率 {_val(f['div_ratio'], 3)}，有效风险来源 "
            f"{_val(f['eff_n_risk'], 2)} 个（共 {f['n_factors']} 个因子），"
            f"HHI={_val(f['risk_hhi'], 3)}，最大风险贡献 {f['top_risk'] or '—'} 占 "
            f"{_pct(f['top_risk_pct'])}。"
        )
        lines.append(
            f"口径提醒：原始矩阵下 σ={_val(f['vol_raw'])}，清洗后 {_val(f['vol_clean'])}"
            f"（{_val(f['vol_gap_pct'], 2)}%）。选方案必须以清洗后为准，"
            f"否则等于按估计误差挑一个样本内最好看的解。"
        )
    else:
        lines.append(f"谱清洗未执行（{f['spectral_reason'] or '因子数不足或已关闭'}），"
                     f"以下关于权重方案优劣的判断缺少清洗后口径支撑。")
    cap = min(1.5 * 100.0 / max(f["n_factors"], 1), 40.0)
    lines += [
        "",
        "**动作**",
        "1. 用「权重方案对照」表横向看：等权 / 最小方差 / 最大分散化 / 风险平价 / ICIR 倾斜，"
        "同时比较波动与期望 ICIR，别只看方差。",
        f"2. 加单因子上限（等权的 1.5 倍约 {cap:.0f}%）避免优化把仓位压到个别因子上。",
        "3. 若最小方差的期望 ICIR 明显低于 ICIR 倾斜方案，说明你在用收益换波动，量一下换得值不值。",
        "",
        f"**提示**：权重优化只在协方差估计可信时才有意义；当前 "
        f"{'相关矩阵已做谱清洗，' if f['spectral_ok'] else ''}"
        f"因子数 {f['n_factors']}、有效观测 {f['n_obs'] or f['n_dates']}，"
        f"q={_val(f['q'], 3) if f['spectral_ok'] else '—'}。",
    ]
    return "\n".join(lines)


def _answer_risk(f: Dict[str, Any]) -> str:
    if not f["spectral_ok"]:
        return (f"**结论**：本次回测没有产出风险分解（{f['spectral_reason'] or '谱清洗未执行'}），"
                f"只能先给组合层风险：最大回撤 {_pct(f['mdd'])}、多空夏普 {_val(f['sharpe'], 2)}。\n\n"
                f"**动作**：在「体系回测分析」里确认因子数 ≥2 且开启谱清洗后重新运行，"
                f"才能看到风险按因子分解的结果。")
    # 年化波动由「多空年化收益 / 夏普」反推，避免再引入一个未在事实表里的指标
    sharpe, ls_annual = _num(f["sharpe"]), _num(f["ls_annual"])
    ls_vol = (ls_annual / sharpe
              if math.isfinite(sharpe) and abs(sharpe) > 1e-9 and math.isfinite(ls_annual)
              else float("nan"))
    lines = [
        f"**结论**：风险主要压在 {f['top_risk'] or '—'} 上，占体系 {_pct(f['top_risk_pct'])}，"
        + (f"超过 {_pct(RISK_CONCENTRATED, 0)} 的集中度阈值，需要干预。"
           if _num(f["top_risk_pct"], 0.0) >= RISK_CONCENTRATED
           else "分布尚可接受。"),
        "",
        f"**依据**：清洗后矩阵口径体系 σ={_val(f['risk_vol'])}，分散化比率 "
        f"{_val(f['div_ratio'], 3)}，有效风险来源 {_val(f['eff_n_risk'], 2)} 个"
        f"（共 {f['n_factors']} 个因子），HHI={_val(f['risk_hhi'], 3)}。",
        f"风险贡献与权重不是一回事：权重相同的体系里，低相关、高波动的因子会拿走更多风险。"
        f"当前最大风险贡献 {_pct(f['top_risk_pct'])} vs 等权基准 "
        f"{_pct(1.0 / max(f['n_factors'], 1))}。",
        f"组合层参照：多空最大回撤 {_pct(f['mdd'])}、年化波动 {_val(ls_vol, 2)}"
        f"（由多空年化收益 / 夏普反推，仅作量级参照）。",
    ]
    if f["addition"]:
        d = f["addition"][0]
        lines.append(
            f"若嫌集中，最低成本的做法是增配 {d['name']}（α*={_val(d['alpha'], 2)}，"
            f"波动降 {_val(d['vol_cut_pct'], 2)}%，与体系相关 {_val(d['corr'], 2)}）。"
        )
    lines += [
        "",
        "**动作**",
        f"1. 给 {f['top_risk'] or '最大风险因子'} 设权重上限，或换成与它低相关的替代因子。",
        "2. 检查该因子是否同时在多个维度里重复出现（重复暴露会放大风险贡献）。",
        "3. 回看它的风险贡献是否稳定：滚动窗口下若逐年抬升，说明集中是被动的、而非主动选择。",
        "",
        "**提示**：风险贡献由协方差矩阵估计得出，"
        + (f"当前 q=N/T={_val(f['q'], 3)}，" if f["spectral_ok"] else "")
        + "因子数接近观测数时单项占比会被高估，别对小差异过度解读。",
    ]
    return "\n".join(lines)


def _answer_spectrum(f: Dict[str, Any]) -> str:
    if not f["spectral_ok"]:
        return (f"**结论**：本次回测没有谱清洗结果：{f['spectral_reason'] or '未执行'}。\n\n"
                f"谱清洗需要至少 2 个因子，并要有一段足够长的截面序列来估计 q=N/T。"
                f"在「体系回测分析」里勾选谱清洗后重跑即可。")
    lines = [
        "**结论**：谱清洗回答的是「样本相关矩阵里有多少是估计噪声」。"
        f"本次 {f['n_factors']} 个因子中，{f['n_signal']} 个方向超过噪声带上界（可视为信号）、"
        f"{f['n_inband']} 个落在带内（本样本量下无法判定）、"
        f"{_pct(f['noise_ratio'], 0)} 个方向低于下界（明确是噪声）。",
        "",
        f"**依据**：有效观测 T={f['n_obs']}，纵横比 q=N/T={_val(f['q'], 3)}，"
        f"MP 噪声带 [{_val(f['lam_low'], 4)}, {_val(f['lam_high'], 4)}]（迹守恒，清洗不改变总方差）。",
        f"条件数 {_val(f['cond_before'], 1)} → 清洗后 {_val(f['cond_after'], 1)}；"
        f"有效因子数 {_val(f['eff_before'], 2)} → {_val(f['eff_after'], 2)}——"
        f"谱变平意味着原来那个「因子数很多」的印象部分来自估计误差。",
        f"实际影响：同一组权重在原始矩阵下 σ={_val(f['vol_raw'])}，清洗后 "
        f"{_val(f['vol_clean'])}，差 {_val(f['vol_gap_pct'], 2)}%；"
        f"虚假分散化 {_val(f['fake_div_pct'], 2)}%。",
    ]
    if f["spectrum_notes"]:
        lines += ["", "**清洗日志**"] + [f"- {n}" for n in f["spectrum_notes"][:4]]
    lines += [
        "",
        "**动作**",
        "1. 把权重优化与风险分解都切到清洗后矩阵（「谱清洗与风险」页已按此口径给出对照表）。",
        f"2. 若 q={_val(f['q'], 3)} 偏大（因子数相对样本量偏多），优先扩样本或删因子，"
        f"而不是继续加清洗强度。",
        f"3. 对 {f['n_inband']} 个带内方向不要急着下结论：它们既没被证实也没被否证，"
        f"扩样本后再判。",
        "",
        f"**提示**：噪声带的位置只由 q 决定，与因子好坏无关；"
        f"低于下界只说明「这个方向在样本里看不出信号」，不等于经济逻辑一定错。",
    ]
    return "\n".join(lines)


def _answer_decay(f: Dict[str, Any]) -> str:
    if not f["decay"]:
        return ("**结论**：本次回测没有产出 IC 衰减曲线，无法判断这个体系适合多长持有期。\n\n"
                "**动作**：在「体系回测分析」里勾选衰减分析后重跑。")
    d0, dn = f["decay"][0], f["decay"][-1]
    lines = [
        f"**结论**：信号属于{'短周期' if _num(f['decay_keep'], 1.0) < DECAY_KEEP_WEAK else '相对耐持有'}"
        f"类型——IC 从 1 日的 {_val(d0['ic'])} 变到 {dn['period']} 日的 {_val(dn['ic'])}"
        + (f"（保留 {_pct(f['decay_keep'], 0)}）" if math.isfinite(_num(f["decay_keep"])) else "")
        + "。",
        "",
        "**依据**：" + "；".join(
            f"{d['period']}日 IC={_val(d['ic'])} / ICIR={_val(d['icir'], 2)}" for d in f["decay"]
        ),
        f"成本侧：日均换手 {_val(f['turnover'], 3)}（Σ|Δw| 口径），"
        f"按单边 {_pct(COMMISSION, 2)} 计，年化交易成本量级约 "
        f"{_pct(f['turnover'] * TRADING_DAYS * COMMISSION, 2)}"
        f"（= 换手 × {TRADING_DAYS} × 手续费率，仅作量级判断）。",
        f"组合层：多空日均收益 {_val(f['ls_daily'], 4)}（年化近似 {_pct(f['ls_annual'])}），"
        f"夏普 {_val(f['sharpe'], 2)}。",
    ]
    if math.isfinite(_num(f["decay_keep"])):
        if _num(f["decay_keep"], 1.0) < DECAY_KEEP_WEAK:
            lines.append("持有期拉到最长档会损失一半以上 IC，"
                         "若成本模型没算进来，回测收益会被高估。")
        else:
            lines.append("IC 衰减平缓，适当拉长调仓周期有机会在不明显损失信号的前提下压低换手。")
    lines += [
        "",
        "**动作**",
        "1. 在回测里把手续费/印花税按真实假设打开，对比净收益与单边换手的关系。",
        "2. 用「持有期 × IC」扫描确定最优调仓频率，而不是默认日频。",
        "3. 若必须高频，优先降低无谓换手（权重平滑、缓冲区），而不是砍因子。",
        "",
        f"**提示**：衰减曲线只说明 IC 的均值随时间变化，"
        f"{f['n_dates']} 个交易日样本下端点的 IC 噪声较大，看趋势别抠单点。",
    ]
    return "\n".join(lines)


def _answer_prune(f: Dict[str, Any]) -> str:
    lines = [
        f"**结论**：体系现有 {f['n_factors']} 个因子、有效维度 "
        f"{_val(f['effective_factors'], 2)} 个，"
        + ("还有值得纳入的候选因子。" if any(
            (not d["in_system"]) and _num(d["vol_cut_pct"], 0.0) >= ADDITION_WORTHWHILE
            for d in f["addition"]) else
           "加因子的边际价值已经不大，先考虑精简。"),
        "",
    ]
    if f["addition"]:
        lines.append("**边际价值排行**（纳入后体系波动下降幅度）")
        for d in f["addition"][:5]:
            tag = "体系内" if d["in_system"] else "体系外候选"
            lines.append(
                f"- {d['name']}（{tag}）：α*={_val(d['alpha'], 2)}，"
                f"降幅 {_val(d['vol_cut_pct'], 2)}%，与体系相关 {_val(d['corr'], 2)}"
                + ("" if d["in_system"] else f"，当前权重 {_pct(d['weight_now'])}")
            )
        lines.append("降幅接近 0 表示它与体系中已有方向重复；降幅大且相关性低，才值得新增。")
    else:
        lines.append("**依据**：本次回测未产出边际价值排行（需谱清洗结果），"
                     "可用体系 ICIR 与有效维度作为间接判据。")

    lines.append("")
    if f["weak_factors"]:
        lines.append("**最弱因子**：" + "；".join(
            f"{w['name']} ICIR={_val(w['icir'], 2)}" for w in f["weak_factors"]))
    if f["strong_factors"]:
        lines.append("**最强因子**：" + "；".join(
            f"{w['name']} ICIR={_val(w['icir'], 2)}" for w in f["strong_factors"]))
    lines += [
        f"体系整体 ICIR={_val(f['icir'], 2)}，平均绝对相关 {_val(f['mean_abs_corr'], 3)}。",
        "",
        "**动作**",
        "1. 加因子：只加「体系外 + 降幅≥1% + 相关性低」的候选，加完重跑有效维度与 ICIR。",
        "2. 删因子：先剔除 |ICIR| 最低的一个做对照回测，若体系 ICIR 不降反升，说明它在摊薄权重。",
        "3. 每次只动一个因子，保留对照记录，否则无法归因。",
        "",
        "**提示**：边际价值由协方差矩阵解析求得（解析最优纳入比例 α*），"
        "受估计误差影响；加因子前先确认它在样本外区间也站得住。",
    ]
    return "\n".join(lines)


def _answer_capacity(f: Dict[str, Any]) -> str:
    annual_cost = _num(f["turnover"]) * TRADING_DAYS * COMMISSION
    lines = [
        f"**结论**：这个体系的成本敏感度由换手决定——日均换手 "
        f"{_val(f['turnover'], 3)}，年化交易成本量级约 {_pct(annual_cost, 2)}"
        f"（按单边 {_pct(COMMISSION, 2)}、年 {TRADING_DAYS} 个交易日，"
        f"口径与 realistic_portfolio 的 Σ|Δw|×费率一致）。",
        "",
        f"**依据**：多空日均收益 {_val(f['ls_daily'], 4)}、近似年化 {_pct(f['ls_annual'])}，"
        f"夏普 {_val(f['sharpe'], 2)}，最大回撤 {_pct(f['mdd'])}。"
        + (f"若成本真按上述量级发生，净收益会被削掉 "
           f"{_pct(min(annual_cost / abs(_num(f['ls_annual'], 1e-9)), 1.0), 0)} 左右。"
           if math.isfinite(_num(f["ls_annual"])) and abs(_num(f["ls_annual"], 0.0)) > 1e-9 else ""),
        f"覆盖度平均 {_pct(f['coverage'], 0)}，样本 {f['n_stocks']} 只股票——"
        f"覆盖度越低，可交易标的越少，实际容量越小。",
    ]
    if f["decay"]:
        lines.append(
            f"持有期越长，换手越低：IC 到 {f['decay'][-1]['period']} 日仍有 "
            f"{_val(f['decay'][-1]['ic'])}"
            + (f"（保留 {_pct(f['decay_keep'], 0)}）" if math.isfinite(_num(f["decay_keep"])) else "")
            + "，拉长调仓周期是压成本最直接的手段。"
        )
    lines += [
        "",
        "**动作**",
        "1. 在回测里打开手续费与印花税，比较毛/净收益，确认策略在真实成本下还成立。",
        "2. 用权重平滑或缓冲区降低无谓换手，通常比砍因子更能保住 IC。",
        "3. 容量测算需要盘口与成交额数据：按目标持仓市值与标的日均成交额估算，"
        "冲击成本随规模非线性上升，本模块没有日内数据，给不出精确容量数字。",
        "",
        f"**提示**：上表成本是量级估算，不含冲击成本、融券费用与滑点；"
        f"实盘前请用真实成交数据重估。",
    ]
    return "\n".join(lines)


def _answer_next(f: Dict[str, Any]) -> str:
    actions = build_actions(f)
    if not actions:
        return (f"**结论**：当前没有触发任何风险阈值——IC={_val(f['ic'])}、"
                f"ICIR={_val(f['icir'], 2)}、有效维度 {_val(f['effective_factors'], 2)}、"
                f"最大风险贡献 {_pct(f['top_risk_pct'])}，各项都在可接受区间。\n\n"
                f"**建议的下一步**：把回测区间外推一段做样本外检验，"
                f"并用真实成本假设复核净收益。")
    lines = [
        f"**结论**：按当前诊断，共有 {len(actions)} 件事值得处理，建议按下面顺序推进。",
        "",
    ]
    for i, a in enumerate(actions, 1):
        lines.append(f"**{i}. {a['title']}**（优先级 {a['priority']}）")
        lines.append(a["detail"])
        if a.get("evidence"):
            lines.append(f"依据：{a['evidence']}")
        lines.append("")
    lines.append(f"**提示**：以上动作全部基于样本内 {f['n_dates']} 个交易日的结果，"
                 f"每改一处都留一次对照回测，才有归因。")
    return "\n".join(lines)


def _answer_general(f: Dict[str, Any]) -> str:
    lines = [
        "**结论**：我按你给的回测结果做体系诊断。这次的问题是没听出具体指向，先把体检结论放前面。",
        "",
        f"**依据**：{f['n_factors']} 个因子、IC={_val(f['ic'])}、ICIR={_val(f['icir'], 2)}、"
        f"有效维度 {_val(f['effective_factors'], 2)}"
        + (f"、谱清洗噪声占比 {_pct(f['noise_ratio'], 0)}" if f["spectral_ok"] else "")
        + f"、最大风险贡献 {_pct(f['top_risk_pct'])}。",
        "",
        "**可以问我的**：" + " / ".join(
            f"「{q}」" for q in suggest_questions_from_facts(f, limit=5)
        ),
        "",
        "**提示**：答案只引用本次回测产出的指标；跨体系比较请先切到对应体系再问。",
    ]
    return "\n".join(lines)


_BUILDERS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "overview": _answer_overview,
    "overfit": _answer_overfit,
    "spectrum": _answer_spectrum,
    "redundancy": _answer_redundancy,
    "weight": _answer_weight,
    "risk": _answer_risk,
    "decay": _answer_decay,
    "prune": _answer_prune,
    "capacity": _answer_capacity,
    "next": _answer_next,
    "general": _answer_general,
}


def build_rule_answer(intent: str, facts: Dict[str, Any]) -> str:
    """按意图把事实表翻译成结论（离线规则路径）。"""
    if not facts or not facts.get("ok"):
        reason = (facts or {}).get("reason", "未知原因")
        return (f"**结论**：现在没有可用的回测结果（{reason}），无法给出诊断。\n\n"
                f"**动作**：先到「因子体系分析」页选中体系并点击运行，"
                f"拿到 IC / ICIR / 相关性 / 谱清洗 / 风险分解等指标后回到这里提问。")
    builder = _BUILDERS.get(intent) or _BUILDERS[DEFAULT_INTENT]
    return builder(facts)


# ---------------------------------------------------------------------------
# 4. 建议提问
# ---------------------------------------------------------------------------
def suggest_questions_from_facts(facts: Dict[str, Any], limit: int = 6) -> List[str]:
    """按诊断结果生成追问清单（界面上的快捷按钮）。"""
    if not facts or not facts.get("ok"):
        return ["这个体系整体怎么样？", "我应该先改哪一处？"]

    pool: List[str] = ["这个体系整体成色如何？"]

    if facts["redundant_pairs"]:
        p = facts["redundant_pairs"][0]
        pool.append(f"{p.get('a')} 和 {p.get('b')} 是不是重复配置？")
    if _num(facts["mean_abs_corr"], 0.0) >= 0.3:
        pool.append("哪些因子在重复下注？")
    if facts["spectral_ok"] and _num(facts["noise_ratio"], 0.0) >= NOISE_HEAVY:
        pool.append("样本相关矩阵里有多少是估计噪声？")
    if facts["spectral_ok"] and _num(facts["inband_ratio"], 0.0) >= NOISE_HEAVY:
        pool.append("带内那几个方向该不该当信号用？")
    if not facts["same_scheme"] and _num(facts["variance_cut_pct"], 0.0) >= VARIANCE_CUT_WORTHWHILE:
        # 方案标签自带口径注解（如「最小方差（谱清洗后）」），带进问题里会把路由带偏到
        # 谱清洗意图，这里去掉括号注解
        plain = re.sub(r"[（(][^）)]*[）)]", "", str(facts["best_scheme"]["label"])).strip()
        pool.append(f"换成「{plain}」权重能省多少风险？")
    if _num(facts["top_risk_pct"], 0.0) >= RISK_CONCENTRATED:
        pool.append(f"风险为什么集中在 {facts['top_risk']}？")
    if facts["decay"]:
        pool.append("这个体系适合多长的持有期？")
    if _num(facts["turnover"], 0.0) >= TURNOVER_HIGH:
        pool.append("换手和交易成本吃掉多少收益？")
    if any((not d["in_system"]) and _num(d["vol_cut_pct"], 0.0) >= ADDITION_WORTHWHILE
           for d in facts["addition"]):
        pool.append("还值得加新因子吗？加哪个？")
    if any(abs(_num(w["icir"], 0.0)) < WEAK_FACTOR_ICIR for w in facts["weak_factors"]):
        pool.append("有没有该剔除的疲软因子？")
    if facts["level"] != "strong":
        pool.append("这个体系可信吗？会不会过拟合？")
    if facts["errors"]:
        pool.append("有几个因子计算失败了，怎么办？")

    pool.append("下一步我该先做什么？")

    out: List[str] = []
    for q in pool:
        if q not in out:
            out.append(q)
    return out[: max(int(limit), 1)]


def suggest_questions(result: Optional[Dict[str, Any]], limit: int = 6) -> List[str]:
    """便捷入口：直接传体系分析结果。"""
    return suggest_questions_from_facts(distill(result), limit=limit)


# ---------------------------------------------------------------------------
# 5. 咨询主入口
# ---------------------------------------------------------------------------
def _history_block(history: Optional[Sequence[Dict[str, Any]]]) -> str:
    if not history:
        return ""
    turns = []
    for h in list(history)[-MAX_HISTORY_TURNS:]:
        role = "用户" if str(h.get("role")) in ("user", "human") else "顾问"
        content = str(h.get("content", "")).strip()
        if content:
            turns.append(f"{role}：{content}")
    if not turns:
        return ""
    return "【之前的对话（仅供理解上下文，不得作为数据来源）】\n" + "\n".join(turns) + "\n\n"


def advise(
    question: str,
    result: Optional[Dict[str, Any]],
    llm: Any = None,
    history: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """回答一个关于因子体系的问题。

    优先走 LLM（``llm`` 需提供 ``complete(system, user) -> str``）；没传客户端、或调用
    抛异常时退回规则答案，并把失败原因放在 ``error`` 里，界面据此提示用户。

    Args:
        question: 用户问题原文。
        result: 体系分析结果，透传给 :func:`distill`。
        llm: LLM 客户端，可选。
        history: 之前的对话 ``[{role, content}, ...]``，用于 LLM 路径的多轮上下文。

    Returns:
        ``{"answer", "mode", "intent", "intent_label", "facts", "actions", "error"}``；
        ``mode`` 为 ``"llm"`` / ``"rules"``。
    """
    facts = distill(result)
    intent = match_intent(question)
    actions = build_actions(facts)
    base = {
        "intent": intent,
        "intent_label": INTENTS.get(intent, ""),
        "facts": facts,
        "actions": actions,
        "error": None,
    }

    if llm is None:
        base["answer"] = build_rule_answer(intent, facts)
        base["mode"] = "rules"
        return base

    user_prompt = (
        f"{_history_block(history)}"
        f"【体系事实表】\n{facts_markdown(facts)}\n\n"
        f"【用户问题】\n{str(question or '').strip()}\n\n"
        f"请按约定结构回答，只引用上面事实表中的数字。"
    )
    try:
        text = llm.complete(system=ADVISOR_SYSTEM_PROMPT, user=user_prompt)
        text = str(text or "").strip()
        if not text:
            raise RuntimeError("模型返回空内容")
        base["answer"] = text
        base["mode"] = "llm"
    except Exception as e:  # 模型不可用不能阻断咨询，退回规则答案并把原因带出去
        base["answer"] = build_rule_answer(intent, facts)
        base["mode"] = "rules"
        base["error"] = f"{type(e).__name__}: {e}"
    return base