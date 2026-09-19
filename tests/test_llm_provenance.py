"""「本轮到底有没有真的调用大模型」的可观测性回归测试。

背景：此前 Agent 节点把 LLM 异常 ``print`` 到启动 Streamlit 的那个终端后转入关键词
模板兜底，界面上依旧渲染出一份结构完整的因子报告；UI 侧又因为自己那套不做
``${VAR}`` 插值的 ``load_config`` 拿到了一串假密钥。两者叠加的结果是无法解释的
「连上了模型却一直离线运行」。这里把修复后的三条不变量钉住：

1. 未解析的 ``${VAR}`` 占位符在**调用时**被明确拒绝，而不是当作密钥发出去换一个
   必然的 401（构造不能失败，否则整个 UI 起不来）；
2. 兜底必须写进状态：``factor_source`` 区分 llm / template，``llm_error`` 留下原因，
   且反思阶段的失败要与生成阶段的失败合并而不是互相覆盖；
3. 报告正文首部必须标注来源，让模板兜底产物无法伪装成模型生成产物。
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from agent.nodes import FactorAgentNodes, _build_report, _provenance_note  # noqa: E402
from llm.client import LLMClient, unresolved_env_placeholder  # noqa: E402

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"

_FAKE_CODE = (
    "import pandas as pd\n"
    "def alpha_factor(df):\n"
    "    df['factor'] = df.groupby('symbol')['close'].pct_change().fillna(0)\n"
    "    return df[['date', 'symbol', 'factor']]\n"
)
_OK_REPLY = json.dumps({
    "name": "fake_mom",
    "description": "假的长动量因子",
    "code": _FAKE_CODE,
    "rationale": "测试用",
    "references": ["Jegadeesh & Titman (1993)"],
})


class _FakeLLM:
    """替身 LLM：要么返回一段合法回复，要么抛出指定异常。"""

    def __init__(self, reply: str = "", exc: Exception | None = None) -> None:
        self.reply = reply
        self.exc = exc
        self.model = "fake-model"

    def complete(self, system: str, user: str, temperature: float | None = None) -> str:
        if self.exc is not None:
            raise self.exc
        return self.reply


def _nodes(llm, tmp_path) -> FactorAgentNodes:
    return FactorAgentNodes(
        llm=llm,
        retriever=None,
        backtester=None,
        kline=pd.DataFrame(),
        config={"experiment_tracking": {"backend": "local", "dir": str(tmp_path)}},
    )


# ----------------------------------------------------------------------
# 1) 假密钥必须在调用时被拒绝
# ----------------------------------------------------------------------
def test_unresolved_placeholder_is_detected() -> None:
    assert unresolved_env_placeholder("${DEEPSEEK_API_KEY}") == "DEEPSEEK_API_KEY"
    assert unresolved_env_placeholder("  ${OPENAI_API_KEY}  ") == "OPENAI_API_KEY"
    assert unresolved_env_placeholder("sk-267abc") is None
    assert unresolved_env_placeholder("") is None
    assert unresolved_env_placeholder(None) is None
    # 形如 "前缀 ${VAR} 后缀" 不是占位符形态，不该被这里误伤
    assert unresolved_env_placeholder("prefix ${VAR} suffix") is None


def test_build_rejects_unresolved_key_but_construction_survives() -> None:
    client = LLMClient({
        "llm": {
            "provider": "deepseek",
            "api_key": "${DEEPSEEK_API_KEY}",
            "model": "deepseek-chat",
        }
    })
    # 构造阶段不得抛错：FactorAgent 在 UI 启动时就要建 LLMClient，
    # 抛错会让整个应用起不来。
    assert client.model == "deepseek-chat"
    with pytest.raises(ValueError) as excinfo:
        client._build()
    msg = str(excinfo.value)
    assert "DEEPSEEK_API_KEY" in msg and "占位符" in msg
    # 健康检查应当如实回答「不可用」
    assert client.available() is False


# ----------------------------------------------------------------------
# 2) 节点必须把来源写进状态
# ----------------------------------------------------------------------
def test_generate_factor_marks_llm_source(tmp_path) -> None:
    out = _nodes(_FakeLLM(_OK_REPLY), tmp_path).generate_factor({
        "factor_description": "一个动量因子",
        "knowledge_context": "",
        "iteration": 0,
    })
    assert out["factor_source"] == "llm"
    assert out["llm_error"] == ""
    assert "alpha_factor" in out["factor_code"]


def test_generate_factor_marks_template_fallback(tmp_path) -> None:
    out = _nodes(_FakeLLM(exc=RuntimeError("401 Unauthorized")), tmp_path).generate_factor({
        "factor_description": "一个动量因子",
        "knowledge_context": "",
        "iteration": 0,
    })
    # 兜底这件事本身允许，但不允许不留痕迹。
    assert out["factor_source"] == "template"
    assert "401" in out["llm_error"]
    # 且确实拿到了可用的模板代码（而不是空结果）
    assert out["factor_code"] and "rolling(20)" in out["factor_code"]


def test_generate_factor_reports_when_both_llm_and_template_fail(tmp_path) -> None:
    out = _nodes(_FakeLLM(exc=RuntimeError("boom")), tmp_path).generate_factor({
        "factor_description": "一段没有任何已知关键词的需求描述",
        "knowledge_context": "",
        "iteration": 0,
    })
    assert out["factor_source"] == "template"
    assert out["error"] == "因子生成失败"
    assert out["llm_error"]


def test_reflect_failure_is_merged_not_overwritten(tmp_path) -> None:
    out = _nodes(_FakeLLM(exc=RuntimeError("timeout")), tmp_path).reflect_and_refine({
        "factor_description": "d",
        "factor_code": "x",
        "metrics": {},
        "reflections": [],
        "iteration": 1,
        "llm_error": "生成阶段：401",
    })
    # 生成阶段的失败不能被反思阶段的失败覆盖掉
    assert "生成阶段：401" in out["llm_error"]
    assert "反思阶段" in out["llm_error"] and "timeout" in out["llm_error"]
    assert any("LLM 未参与本轮改进" in r for r in out["reflections"])


# ----------------------------------------------------------------------
# 3) 报告必须写明来源
# ----------------------------------------------------------------------
def _report(**kw) -> str:
    base = {
        "name": "f", "desc": "d", "code": "x", "metrics": {}, "knowledge": "", "reflections": [],
        "validation_ok": False, "validation_error": "", "error": None,
    }
    base.update(kw)
    return _build_report(**base)


def test_report_head_marks_template_provenance() -> None:
    rep = _report(factor_source="template", llm_error="生成阶段：401 Unauthorized",
                  llm_model="deepseek-chat")
    head = rep.split("**需求描述**")[0]
    assert "并非由大模型生成" in head
    assert "401" in head


def test_report_head_marks_llm_provenance() -> None:
    head = _report(factor_source="llm", llm_model="deepseek-chat").split("**需求描述**")[0]
    assert "因子来源：大模型生成" in head
    assert "deepseek-chat" in head


def test_report_stays_silent_for_legacy_callers() -> None:
    # 没有来源信息的旧调用路径不该凭空多出一行噪声
    assert "因子来源" not in _report()
    assert _provenance_note("", "", "") == ""


def test_provenance_note_collapses_multiline_errors() -> None:
    note = _provenance_note("template", "line1\nline2\nline3", "m")
    assert "\n" in note
    body = [ln for ln in note.splitlines() if ln.startswith("> 失败原因")]
    assert len(body) == 1 and "line1 line2 line3" in body[0]


# ----------------------------------------------------------------------
# 4) UI 侧的三处接线（源码契约，避免改动被静默回退）
# ----------------------------------------------------------------------
def test_ui_reads_interpolated_config_and_shows_provenance() -> None:
    src = (_SRC_ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    # UI 必须走带 ${VAR} 插值 + .env 注入的那一套 load_config
    assert "from llm.client import load_config as _load_interpolated" in src
    assert "_load_interpolated(str(CONFIG_PATH))" in src
    # 结果区必须渲染来源徽标，且挖掘前要校验未解析的密钥
    assert "_render_llm_provenance(d)" in src
    assert "unresolved_env_placeholder(" in src
    # 「测试连接」不等于「已应用」
    assert "_ui_llm_pending()" in src
