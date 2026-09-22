"""图文识别模型训练（朴素贝叶斯 / Transformer / CNN / GNN）的契约测试。

全合成数据、小批量，CI 可跑。覆盖的都是"错了会静默出错"的点：

- 向量化→训练→落盘→加载→推理 全链路可用（缺一环就白训了）
- 朴素贝叶斯在无 torch 环境下也必须能训（这是降级路径的底线）
- 关系图必须稀疏：连成团会过平滑，GNN 直接退化成随机猜测
- 硬样本（无关键词、无标的）上 GNN 要优于纯文本模型，否则图没带来信息
- 本地模型缺失时 JEV 判定链要能继续走规则，不能抛异常
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from engine.jev import JEVClient  # noqa: E402
from engine.multimodal_train import (  # noqa: E402
    DATA_TYPE_LABELS,
    IMAGE_LABELS,
    TORCH_AVAILABLE,
    LocalMultimodalClassifier,
    MultinomialNB,
    TextVectorizer,
    build_material_graph,
    classification_metrics,
    synthesize_image_dataset,
    synthesize_text_corpus,
    train_all,
)


# ---------------------------------------------------------------- 向量化 / NB
def test_vectorizer_keeps_codes_and_cjk_chars():
    """中文按字、代码按词：切开代码会让"600519"退化成无意义的数字碎片。"""
    v = TextVectorizer().build(["600519 收盘价上涨", "预计 600519 需求回暖"])
    assert "600519" in v.vocab, "股票代码必须整词入表"
    assert any(len(t) == 1 and "一" <= t <= "鿿" for t in v.vocab), "中文需按字切"
    enc = v.encode(["600519 收盘价"])
    assert enc.shape[1] == v.max_len and enc.sum() > 0
    assert TextVectorizer.from_json(v.to_json()).vocab == v.vocab


def test_naive_bayes_learns_without_torch():
    """朴素贝叶斯是零依赖底线：torch 装没装都得能训能存能读。"""
    items = synthesize_text_corpus(n_per_class=25, seed=3)
    texts = [it["text"] for it in items]
    y = [it["data_type"] for it in items]
    v = TextVectorizer().build(texts)
    X = v.counts(texts)
    m = MultinomialNB().fit(X[:120], y[:120])
    pred = m.predict(X[120:])
    acc = classification_metrics(y[120:], pred, DATA_TYPE_LABELS)["acc"]
    assert acc > 1.0 / len(DATA_TYPE_LABELS) + 0.2, f"NB 没学到东西：acc={acc}"
    probs = m.predict_proba(X[120:])
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-8)
    p = Path("data/_tmp_nb.npz")
    m.save(p)
    assert MultinomialNB.load(p).classes == m.classes
    p.unlink(missing_ok=True)


def test_metrics_macro_f1_penalises_ignoring_rare_class():
    """macro-F1 而非 accuracy：只猜大类就能拿高 accuracy 的假象要被压住。"""
    y_true = ["a"] * 9 + ["b"]
    y_pred = ["a"] * 10
    met = classification_metrics(y_true, y_pred, ["a", "b"])
    assert met["acc"] == 0.9 and met["macro_f1"] < 0.6


# ---------------------------------------------------------------- 数据合成
def test_synthesized_corpus_has_hard_samples_and_labels():
    items = synthesize_text_corpus(n_per_class=20, seed=11)
    assert len(items) == 20 * len(DATA_TYPE_LABELS)
    hard = [it for it in items if it.get("hard")]
    assert 0.05 < len(hard) / len(items) < 0.45, "硬样本比例应在 25% 上下"
    for it in hard:
        assert it["data_type"] in DATA_TYPE_LABELS and it["sentiment"]
    assert all(it["forward_looking"] in (0, 1) for it in items)


def test_material_graph_is_sparse():
    """同标的材料可能上百份，连成团会让两层 GCN 过平滑（准确率≈随机）。"""
    g = build_material_graph(synthesize_text_corpus(n_per_class=30, seed=5), max_nodes=200)
    n = g["X"].shape[0]
    deg = (g["A"] > 0).sum(axis=1)
    # 未剪枝时同一标的会连成上百个节点的团，mean_deg 直接上百
    assert deg.mean() < 12 and deg.max() <= 16, f"度数未受控：mean={deg.mean()} max={deg.max()}"
    assert np.allclose(g["A"], g["A"].T, atol=1e-6), "邻接必须对称（无向图）"
    assert set(g["y"]) <= set(DATA_TYPE_LABELS)


def test_image_dataset_shapes_and_labels():
    X, y = synthesize_image_dataset(n_per_class=6, size=32, seed=2)
    assert X.shape == (6 * len(IMAGE_LABELS), 1, 32, 32)
    assert X.min() >= 0.0 and X.max() <= 1.0
    assert set(y) == set(IMAGE_LABELS)


# ---------------------------------------------------------------- 训练编排
def _mini_train(tmp_path, **kw):
    base = {"text_per_class": 20, "image_per_class": 6, "epochs": 1, "gcn_epochs": 4}
    base.update(kw)
    return train_all({"multimodal": {"model_dir": str(tmp_path)}}, **base)


def test_train_all_writes_artifacts_and_report(tmp_path):
    rep = _mini_train(tmp_path)
    assert (tmp_path / "vectorizer.json").is_file()
    assert (tmp_path / "nb_datatype.npz").is_file()
    assert (tmp_path / "nb_sentiment.npz").is_file()
    assert (tmp_path / "nb_forward.npz").is_file()
    assert (tmp_path / "training_report.json").is_file()
    assert rep["models"]["naive_bayes[data_type]"]["acc"] > 1 / len(DATA_TYPE_LABELS)
    for name in ("transformer[text]", "cnn[image]", "gcn[graph]"):
        if TORCH_AVAILABLE:
            assert (tmp_path / rep["models"][name]["artifact"]).is_file()
        else:
            assert rep["models"][name]["status"] == "skipped", \
                "无 torch 时必须明确标注跳过原因，而不是静默消失"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch 未安装")
def test_gcn_beats_text_model_on_hard_nodes(tmp_path):
    """硬样本（文本里既无类型关键词也无标的）只能靠图传播：GNN 必须更强。"""
    rep = _mini_train(tmp_path, text_per_class=60, epochs=2, gcn_epochs=60)
    chk = rep.get("hard_node_check")
    assert chk and chk["n"] >= 5, f"硬样本不足，无法对比：{chk}"
    assert chk["gcn"]["acc"] > chk["naive_bayes"]["acc"], \
        f"GNN 未体现关系传播价值：{chk}"


def test_classifier_roundtrip_on_trained_artifacts(tmp_path):
    """训完必须能加载回来做推理，且未训练时清清楚楚报不可用。"""
    _mini_train(tmp_path)
    cli = LocalMultimodalClassifier(tmp_path)
    assert cli.available
    out = cli.classify_text("公司发布公告，600519 营收增长，预计下半年回暖")
    assert out.get("data_type") in DATA_TYPE_LABELS
    assert out.get("sentiment") in {"negative", "neutral", "positive"}
    assert out.get("forward_looking") is None or 0.0 <= out["forward_looking"] <= 1.0

    empty = LocalMultimodalClassifier(tmp_path / "nope")
    assert not empty.available, "没有产物时必须报不可用，让调用方走规则"
    assert empty.classify_text("任意文本") == {"engine": "local-model"}


# ---------------------------------------------------------------- 判定链接入
def test_jev_falls_back_to_local_model_then_rules(tmp_path, monkeypatch):
    """三级链：无 Key 时用本地模型；本地模型也没有时用规则——都不许抛异常。"""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    _mini_train(tmp_path)
    cli = JEVClient({"jev": {"enabled": True, "fallback_to_heuristic": True},
                     "multimodal": {"model_dir": str(tmp_path)}})
    res = cli.analyze("公司发布公告：600519 2025-06-01 营收超预期", filename="a.txt")
    assert res["engine"] == "local-model", f"有本地产物时应优先用：{res}"
    assert res["data_type"] in DATA_TYPE_LABELS

    no_model = JEVClient({"jev": {"enabled": True, "fallback_to_heuristic": True},
                          "multimodal": {"model_dir": str(tmp_path / "nope")}})
    res2 = no_model.analyze("公司发布公告：600519 2025-06-01 营收超预期", filename="a.txt")
    assert res2["engine"] == "heuristic" and res2["data_type"] in DATA_TYPE_LABELS

    from engine.jev import override_data_type

    over = override_data_type(dict(res2), "quote_screenshot", source="cnn:candlestick")
    assert over["data_type"] == "quote_screenshot"
    assert "行情" in over["data_type_label"] or "数值" in over["data_type_label"]
    assert override_data_type(dict(res2), "not_a_label")["data_type"] == res2["data_type"]
