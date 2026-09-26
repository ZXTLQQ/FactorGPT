"""``src/agent/dialogue_model`` 对话能力层回归测试。

钉住四件事：

1. **合成语料的标签是确定的**（由生成过程决定，不存在"标错了还自认为对"），且
   跟进样本确实不含意图提示词——否则下面第 4 条的对照就失去意义；
2. **指代消解只在不该动的时候不动**：独立需求原样返回、上文无槽位时不编造；
3. **跟进句必须被补全**：这是离线多轮能不能接下去的关键，原样返回等于没做；
4. **本地模型相对正则的增量是可证的**：在"字面无提示词"的困难子集上，模型必须
   明显强于正则——否则这一层就只是把正则重写了一遍，没必要存在。

全部离线：只用 numpy，不联网、不下载模型。
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from agent import dialogue_model as DM  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_classifier_cache():
    """每条用例前后清掉分类器缓存。

    ``load_classifier`` 按目录缓存实例，不清的话前一条用例（乃至别的测试文件）
    留下的实例会串到后面的用例上，于是"换个目录重训"这类用例测的就不是自己了。
    """
    DM.reset_cache()
    yield
    DM.reset_cache()


# --------------------------------------------------------------------------
# 1. 语料
# --------------------------------------------------------------------------
def test_corpus_is_deterministic_and_labelled():
    a = DM.synthesize_corpus(per_template=3, seed=42)
    b = DM.synthesize_corpus(per_template=3, seed=42)
    assert [r["text"] for r in a] == [r["text"] for r in b]
    assert a and all(r["intent"] in DM.INTENT_LABELS for r in a)
    assert all(r["followup"] in DM.FOLLOWUP_LABELS for r in a)
    # 去重：同一句话重复计数会让"准确率"变成自欺欺人
    texts = [r["text"] for r in a]
    assert len(texts) == len(set(texts))


def test_followup_samples_carry_no_intent_keyword():
    """困难集的前提：跟进句字面没有因子/疑问关键词，正则才真的看不见。"""
    rows = [r for r in DM.synthesize_corpus(per_template=6, seed=7)
            if r["followup"] == "yes"]
    hard = {r["text"] for r in DM._hard_node_split(rows)}
    assert hard, "跟进样本若全被判为'有提示词'，困难集对照就无从谈起"


def test_hard_node_split_moves_easy_samples_out():
    rows = [{"text": "构建一个动量因子", "intent": "mining", "followup": "no"},
            {"text": "你好", "intent": "chitchat", "followup": "no"},
            {"text": "再来一个", "intent": "mining", "followup": "yes"}]
    assert [r["text"] for r in DM._hard_node_split(rows)] == ["再来一个"]


# --------------------------------------------------------------------------
# 2. 状态抽取
# --------------------------------------------------------------------------
def test_extract_state_reads_slots_from_history():
    history = [
        {"role": "user", "content": "构建一个20日动量因子，股票池沪深300"},
        {"role": "assistant", "agent": {"factor_name": "mom20",
                                        "metrics": {"ic": 0.05, "icir": 0.6}}},
    ]
    st = DM.extract_state(history)
    assert st.direction == "动量"
    assert st.window == "20日"
    assert st.universe == "沪深300"
    assert st.factor_name == "mom20"
    assert st.usable and st.has_mining


def test_extract_state_later_turn_overrides_earlier():
    history = [{"role": "user", "content": "构建一个20日动量因子"},
               {"role": "assistant", "agent": {"factor_name": "mom20"}},
               {"role": "user", "content": "改成60天"}]
    assert DM.extract_state(history).window == "60天"  # 最后一句为准


def test_extract_state_empty_history_is_unusable():
    st = DM.extract_state([])
    assert not st.usable and st.factor_name == ""


# --------------------------------------------------------------------------
# 3. 指代消解
# --------------------------------------------------------------------------
_HIST = [{"role": "user", "content": "构建一个20日动量因子，股票池沪深300"},
         {"role": "assistant", "agent": {"factor_name": "mom20",
                                         "metrics": {"ic": 0.05}}}]


def test_resolve_standalone_need_is_untouched():
    text = "构建一个低估值质量反转因子，股票池中证500"
    assert DM.resolve_reference(text, history=_HIST) == text


def test_resolve_followup_binds_slots_from_prior_turn():
    out = DM.resolve_reference("窗口改成60天", history=_HIST)
    assert "60天" in out and "动量" in out and "沪深300" in out
    assert out != "窗口改成60天"


def test_resolve_followup_overrides_only_the_slot_it_names():
    # 本句没提股票池 → 沿用上文的沪深300；本句提了窗口 → 覆盖上文的20日。
    out = DM.resolve_reference("把窗口调到120天", history=_HIST)
    assert "120天" in out and "20日" not in out and "沪深300" in out


def test_resolve_without_context_never_invents():
    # 上文无槽位可绑：宁可原样返回，也不许编出一个"动量因子"。
    assert DM.resolve_reference("再来一个") == "再来一个"
    assert DM.resolve_reference("换个方向", history=[]) == "换个方向"


def test_resolve_falls_back_to_previous_factor_name():
    hist = [{"role": "assistant", "agent": {"factor_name": "rev5"}}]
    out = DM.resolve_reference("再跑一次", history=hist)
    assert "rev5" in out


def test_is_followup_rule_agrees_on_pure_anaphora():
    st = DM.extract_state(_HIST)
    assert DM.is_followup("再来一个", state=st)[0] is True
    assert DM.is_followup("构建一个动量因子", state=st)[0] is False


def test_is_followup_false_when_no_bindable_state():
    # 没有可绑槽位时报"不是跟进"：判了也补不出来，还会误导调用方。
    assert DM.is_followup("再来一个", state=DM.DialogueState())[0] is False


# --------------------------------------------------------------------------
# 4. 训练 → 落盘 → 加载
# --------------------------------------------------------------------------
@pytest.fixture()
def _trained(tmp_path):
    # 种子与样本量都写死：下面第 5 节比的是"模型 vs 正则"的准确率差，
    # 语料一旦随环境漂移，这个差值就不再是测量而是运气。
    rep = DM.train_all({"dialogue": {"model_dir": str(tmp_path)}},
                       out_dir=str(tmp_path), per_template=10, seed=20260926)
    DM.reset_cache()
    return tmp_path, rep


def test_train_writes_all_artifacts(_trained):
    d, _ = _trained
    for name in ("vectorizer.json", "nb_intent.npz", "nb_followup.npz",
                 "training_report.json"):
        assert (d / name).exists(), name
    json.loads((d / "training_report.json").read_text(encoding="utf-8"))


def test_trained_classifier_round_trips(_trained):
    d, _ = _trained
    cli = DM.LocalDialogueClassifier(str(d))
    assert cli.available
    intent, conf = cli.predict_intent("构建一个动量因子")
    assert intent == "mining" and 0.0 <= conf <= 1.0
    assert cli.predict_followup("再来一个") >= 0.5
    assert cli.predict_followup("构建一个动量因子") < 0.5


def test_untrained_dir_is_silently_unavailable(tmp_path):
    # 产物缺失不能抛异常——判定链要继续往下走到正则。
    assert DM.LocalDialogueClassifier(str(tmp_path)).available is False
    assert DM.load_classifier({"dialogue": {"model_dir": str(tmp_path)}}) is None


def test_disabled_config_returns_none(_trained):
    d, _ = _trained
    assert DM.load_classifier({"dialogue": {"enabled": False,
                                            "model_dir": str(d)}}) is None


# --------------------------------------------------------------------------
# 5. 困难集：本地模型相对正则的增量（本层存在的理由）
# --------------------------------------------------------------------------
def test_local_model_beats_rule_on_hard_nodes(_trained):
    """可证伪的那一条：字面无提示词的输入上，模型必须明显强于正则。

    随机基线是 1/4 = 0.25。正则在这批样本上基本就是随机——它靠关键词活着，
    而困难集的定义就是"没有关键词"。若这条挂了，说明这一层没有带来任何信息，
    应当删掉而不是留着充数。
    """
    _, rep = _trained
    hard = rep["hard_node"]
    assert hard["n"] >= 10, hard
    assert hard["model"]["acc"] > hard["rule"]["acc"] + 0.2, hard
    assert hard["model"]["acc"] >= 0.8, hard
    assert hard["followup"]["recall"] >= 0.6, hard


def test_training_report_records_both_arms(_trained):
    _, rep = _trained
    # 报告必须同时给出模型与正则两臂，否则"模型更好"只是一句断言而非测量。
    assert set(rep["models"]) == {"intent_nb", "followup_nb"}
    assert set(rep["hard_node"]) >= {"n", "model", "rule", "followup"}
