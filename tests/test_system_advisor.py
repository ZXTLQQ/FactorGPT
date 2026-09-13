"""AI 体系咨询引擎（engine.system_advisor）与页面分发的契约测试。

覆盖三件事：

* **事实表抽取**：回测结果压成事实表后字段齐备，且不出现用 ``nan`` 冒充数字的格子；
* **两条回答路径**：离线规则答案必须引用事实表里的真实数字，LLM 路径受同一份事实约束，
  模型不可用时必须降级而不是抛错；
* **分发契约**：导航页面 key、``app.py`` 的 ``DISPATCH``、README 声明的页数三者一致。
"""

import copy
import math
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from engine.factor_system import (  # noqa: E402
    SystemMember,
    analyze_system,
    build_findings,
    build_synthetic_panel,
)
from engine.system_advisor import (  # noqa: E402
    DEFAULT_INTENT,
    INTENTS,
    advise,
    build_actions,
    build_rule_answer,
    distill,
    facts_markdown,
    match_intent,
    suggest_questions,
    suggest_questions_from_facts,
)
from engine.traditional_factors import get_all_factors  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# 只挑纯价量因子，保证合成面板上全部可算、不落进 errors 分支
FACTOR_NAMES = (
    "momentum_20d",
    "reversal_5d",
    "ma_cross_5_20",
    "close_to_high_20d",
    "realized_vol_20d",
    "rsi_14d",
)


def _read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fp:
        return fp.read()


def _members():
    by_name = {f.name: f for f in get_all_factors()}
    return [
        SystemMember(
            factor_name=by_name[nm].name,
            display_name=by_name[nm].display_name,
            dimension="价格行为与趋势" if by_name[nm].category == "price_trend" else "波动与不确定性",
            category=by_name[nm].category,
            source="static",
            direction=by_name[nm].direction,
            quality=by_name[nm].quality_score,
            code=by_name[nm].code,
        )
        for nm in FACTOR_NAMES
    ]


@pytest.fixture(scope="module")
def result():
    panel = build_synthetic_panel(n_symbols=8, days=300, seed=7)
    return analyze_system(
        panel, _members(), weight_mode="equal", n_quantiles=5, forward_periods=1,
        run_decay=True, run_diversification=True, run_spectral=True,
    )


@pytest.fixture(scope="module")
def facts(result):
    return distill(result)


class FakeLLM:
    """记录提示词、可控返回值的假客户端。"""

    def __init__(self, text="模型回答", raises=None):
        self.text = text
        self.raises = raises
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        if self.raises:
            raise self.raises
        return self.text


# ---------------------------------------------------------------------------
# 事实表
# ---------------------------------------------------------------------------
def test_distill_reports_missing_result():
    for bad in (None, {}, {"error": "数据源不可用"}):
        facts = distill(bad)
        assert facts["ok"] is False
        assert facts["reason"]


def test_distill_extracts_core_facts(facts):
    assert facts["ok"] is True
    assert facts["n_factors"] == len(FACTOR_NAMES)
    assert facts["level"] in {"strong", "weak", "none"}
    assert facts["noise_ratio"] is not None
    assert 0.0 <= facts["noise_ratio"] <= 1.0
    assert facts["n_signal"] + facts["n_inband"] <= facts["n_factors"]
    assert facts["findings"], "应带上 build_findings 的诊断结论"
    assert facts["decay"], "开启衰减分析后应产出衰减曲线"
    assert facts["n_evaluated"] >= 1


def test_facts_never_store_nan(facts):
    """仓库约定：不可测量处用 '—' 展示，不让 nan 冒充数字。"""
    offenders = [k for k, v in facts.items()
                 if isinstance(v, float) and not math.isfinite(v)]
    assert not offenders, f"事实表含 nan 字段：{offenders}"


def test_markdown_context_is_complete(facts):
    md = facts_markdown(facts)
    assert "【谱清洗" in md and "噪声带" in md
    assert "【IC 衰减】" in md
    assert "nan" not in md.lower()
    assert facts["spectrum_notes"], "谱清洗日志应被带进上下文"


def test_markdown_context_for_empty_result():
    md = facts_markdown(distill(None))
    assert "无可用回测结果" in md
    assert "nan" not in md.lower()


# ---------------------------------------------------------------------------
# 规则路径
# ---------------------------------------------------------------------------
def test_every_intent_has_an_answer(facts):
    for intent in INTENTS:
        text = build_rule_answer(intent, facts)
        assert isinstance(text, str) and len(text) > 80, f"{intent} 答案过短"
        assert "nan" not in text.lower(), f"{intent} 答案漏出 nan"
        assert "None" not in text, f"{intent} 答案漏出 None"


def test_overview_answer_cites_real_numbers(facts):
    text = build_rule_answer("overview", facts)
    assert f"{facts['ic']:.4f}" in text
    assert f"{facts['icir']:.2f}" in text
    assert facts["weight_label"] in text


def test_rule_answer_without_result_asks_for_backtest():
    text = build_rule_answer("overview", distill(None))
    assert "回测" in text


@pytest.mark.parametrize("question,intent", [
    ("这个体系整体怎么样？", "overview"),
    ("会不会过拟合？", "overfit"),
    ("样本相关矩阵里有多少是估计噪声？", "spectrum"),
    ("哪些因子在重复配置？", "redundancy"),
    ("换成最小方差能省多少风险？", "weight"),
    ("风险集中在哪个因子？", "risk"),
    ("适合多长的持有期？", "decay"),
    ("还值得加新因子吗？", "prune"),
    ("换手和交易成本吃掉多少收益？", "capacity"),
    ("下一步我该先做什么？", "next"),
])
def test_intent_routing(question, intent):
    assert match_intent(question) == intent


@pytest.mark.parametrize("question", ["", "   ", "今天天气不错", "随便聊聊"])
def test_intent_routing_falls_back(question):
    assert match_intent(question) == DEFAULT_INTENT
    assert DEFAULT_INTENT in INTENTS


def test_actions_are_prioritized_and_capped(facts):
    actions = build_actions(facts)
    assert actions, "完整回测结果应至少触发一条动作"
    assert len(actions) <= 6
    priorities = [a["priority"] for a in actions]
    assert priorities == sorted(priorities)
    for a in actions:
        assert a["title"] and a["detail"] and a["evidence"]
        assert "nan" not in a["detail"].lower()


def test_actions_without_result_tell_user_to_run_backtest():
    actions = build_actions(distill(None))
    assert len(actions) == 1
    assert actions[0]["priority"] == 0
    assert "回测" in actions[0]["title"]


def test_suggestions_are_unique_and_capped(facts, result):
    questions = suggest_questions(result)
    assert 1 <= len(questions) <= 6
    assert len(set(questions)) == len(questions)
    assert all(q.strip() for q in questions)
    same = suggest_questions(None)
    assert same and len(same) <= 6


def test_suggested_questions_route_to_a_specific_intent(result):
    """快捷问题点下去必须落在具体意图上，落到 general 等于白问一轮。"""
    for q in suggest_questions(result):
        intent = match_intent(q)
        assert intent != DEFAULT_INTENT, f"快捷问题「{q}」无法路由到具体意图"


def test_suggestions_react_to_diagnosis(facts):
    """建议清单要跟着诊断走：噪声偏重时必然出现噪声相关的追问。"""
    joined = " ".join(suggest_questions_from_facts(facts))
    assert "体系" in joined
    if facts["spectral_ok"] and facts["noise_ratio"] >= 0.5:
        assert "噪声" in joined
    if facts["redundant_pairs"]:
        assert "重复" in joined


# ---------------------------------------------------------------------------
# LLM 路径与降级
# ---------------------------------------------------------------------------
def test_advise_without_llm_uses_rules(result):
    out = advise("这个体系整体怎么样？", result)
    assert out["mode"] == "rules"
    assert out["error"] is None
    assert out["intent"] == "overview"
    assert out["intent_label"] == INTENTS["overview"]
    assert out["actions"]
    assert out["facts"]["ok"] is True


def test_advise_uses_injected_llm(result):
    llm = FakeLLM("这是模型给出的诊断")
    out = advise("整体怎么样？", result, llm=llm)
    assert out["mode"] == "llm"
    assert out["answer"] == "这是模型给出的诊断"
    system_prompt, user_prompt = llm.calls[0]
    assert "严禁使用事实表以外的数字" in system_prompt
    assert "【体系事实表】" in user_prompt
    assert "【用户问题】" in user_prompt


def test_advise_prompt_carries_facts_and_history(result):
    llm = FakeLLM()
    history = [{"role": "user", "content": "先看冗余"},
               {"role": "assistant", "content": "结论：平均相关 0.31"}]
    advise("那风险呢？", result, llm=llm, history=history)
    _, user_prompt = llm.calls[0]
    assert "先看冗余" in user_prompt and "平均相关 0.31" in user_prompt
    assert "不得作为数据来源" in user_prompt


def test_advise_degrades_when_llm_raises(result):
    out = advise("整体怎么样？", result, llm=FakeLLM(raises=RuntimeError("网络不可达")))
    assert out["mode"] == "rules"
    assert "RuntimeError" in out["error"] and "网络不可达" in out["error"]
    assert len(out["answer"]) > 200


def test_advise_degrades_on_empty_model_output(result):
    out = advise("整体怎么样？", result, llm=FakeLLM(text="   "))
    assert out["mode"] == "rules"
    assert out["error"] and "空内容" in out["error"]


# ---------------------------------------------------------------------------
# build_findings 的集中风险分支（曾引用未定义变量）
# ---------------------------------------------------------------------------
def test_build_findings_handles_concentrated_risk(result, facts):
    risk = result.get("spectral", {}).get("risk")
    if risk is None or not hasattr(risk, "top_risk_pct"):
        pytest.skip("谱清洗未产出风险分解")
    concentrated = copy.copy(risk)
    concentrated.top_risk_pct = 0.62
    clone = dict(result)
    clone["spectral"] = dict(result["spectral"], risk=concentrated)
    findings = build_findings(clone)
    assert any("风险分布比权重更集中" in f["text"] for f in findings)
    concentrated_facts = distill(clone)
    assert concentrated_facts["top_risk_pct"] == pytest.approx(0.62)


# ---------------------------------------------------------------------------
# 导航 / 分发 / README 契约
# ---------------------------------------------------------------------------
def _nav_keys():
    nav_src = _read("src", "ui", "nav.py")
    return re.findall(r'"key"\s*:\s*"([a-z_]+)"', nav_src)


def _dispatch_keys():
    app_src = _read("src", "ui", "app.py")
    block = re.search(r"^DISPATCH\s*=\s*\{(.*?)^\}", app_src, re.S | re.M)
    assert block, "app.py 中未找到 DISPATCH 字典"
    return re.findall(r'"([a-z_]+)"\s*:', block.group(1))


def _declared_page_count():
    """页数以 test_docs_contract.UI_PAGE_COUNT 为单一来源，避免两处各写一份。"""
    src = _read("tests", "test_docs_contract.py")
    m = re.search(r"UI_PAGE_COUNT\s*=\s*(\d+)", src)
    assert m, "test_docs_contract.py 未声明 UI_PAGE_COUNT"
    return int(m.group(1))


def test_nav_and_dispatch_stay_in_sync():
    nav_keys, dispatch_keys = _nav_keys(), _dispatch_keys()
    assert len(nav_keys) == len(set(nav_keys)), "导航 key 重复"
    assert sorted(nav_keys) == sorted(dispatch_keys), (
        f"导航与分发不一致：仅导航有 {sorted(set(nav_keys) - set(dispatch_keys))}，"
        f"仅分发现有 {sorted(set(dispatch_keys) - set(nav_keys))}"
    )
    assert len(nav_keys) == _declared_page_count()


def test_advisor_page_is_wired_up():
    assert "sys_advisor" in _nav_keys()
    nav_src = _read("src", "ui", "nav.py")
    assert "AI 体系咨询" in nav_src
    app_src = _read("src", "ui", "app.py")
    assert "render_system_advisor" in app_src
    ui_src = _read("src", "ui", "factor_system.py")
    assert "def render_system_advisor" in ui_src


def test_readme_declares_current_page_count():
    count = _declared_page_count()
    readme = _read("README.md")
    assert re.search(rf"{count}[- ]?page|{count}\s*页", readme, re.IGNORECASE), \
        f"README 未声明 {count} 页界面"
