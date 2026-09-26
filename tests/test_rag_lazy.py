"""检索器 / 判定链的**惰性加载与缓存**回归测试。

这几条优化都是"把一笔固定开销从构造时挪到真正要用时"，共同的风险是：挪错了地方，
功能就静默消失或不生效——构造时不再建索引，若首次检索也没建起来，向量检索就永远
用不上，而且不会报错，只是检索质量悄悄变差。所以这里钉住的是**行为不变量**，
不是耗时数字（耗时在机器上漂移，行为不会）。

全部离线，不发网络请求、不下载任何模型。
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from rag import retriever as RAG  # noqa: E402


# --------------------------------------------------------------------------
# 向量索引：构造时不动，首次检索时才建
# --------------------------------------------------------------------------
class _FakeIndex:
    """替身索引：记录 build/query 次数，不碰 ChromaDB、不下载模型。"""

    def __init__(self) -> None:
        self.available = True
        self.built = 0
        self.queries = 0

    def build_from_seed(self) -> int:
        self.built += 1
        return 0

    def add_document(self, item) -> None:
        pass

    def query(self, query: str, top_k: int = 5):
        self.queries += 1
        return ["VECTOR-HIT"]


def test_vector_index_not_built_at_construction(monkeypatch):
    calls = []
    monkeypatch.setattr(RAG, "rag_deps_available", lambda: calls.append(1) or True)
    idx = _FakeIndex()
    RAG.FactorRetriever(index=idx, use_vector_store=None)
    # 构造的代价必须是 0：打开页面/建检索器不该替"检索一次"付这笔钱。
    assert calls == []
    assert idx.built == 0


def test_vector_index_built_once_on_first_retrieve(monkeypatch):
    monkeypatch.setattr(RAG, "rag_deps_available", lambda: True)
    idx = _FakeIndex()
    r = RAG.FactorRetriever(index=idx, use_vector_store=None)
    assert r.retrieve("动量因子") == ["VECTOR-HIT"]
    assert idx.built == 1  # 建了，且只建一次
    r.retrieve("动量因子")
    assert idx.built == 1


def test_vector_disabled_never_touches_index(monkeypatch):
    monkeypatch.setattr(RAG, "rag_deps_available", lambda: True)
    idx = _FakeIndex()
    r = RAG.FactorRetriever(index=idx, use_vector_store=False)
    out = r.retrieve("动量因子")
    assert "VECTOR-HIT" not in out
    assert idx.built == 0 and idx.queries == 0


def test_vector_failure_degrades_to_keyword_search():
    """向量库起不来时，检索本身仍要能用——增强项不是前提。"""
    idx = _FakeIndex()

    def _boom() -> int:
        raise RuntimeError("chromadb unavailable")

    idx.build_from_seed = _boom  # type: ignore[assignment]
    r = RAG.FactorRetriever(index=idx, use_vector_store=True)
    assert r.retrieve("动量因子")  # 不能是空结果，更不能抛异常
    assert r._use_vector is False
    assert "chromadb" in r._vector_error


# --------------------------------------------------------------------------
# 离线语料缓存：命中复用，内容变了必须失效
# --------------------------------------------------------------------------
@pytest.fixture()
def _knowledge_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(RAG, "_KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(RAG, "_OFFLINE_CACHE", None)
    monkeypatch.setattr(RAG, "_OFFLINE_FINGERPRINT", "")
    yield tmp_path
    monkeypatch.setattr(RAG, "_OFFLINE_CACHE", None)
    monkeypatch.setattr(RAG, "_OFFLINE_FINGERPRINT", "")


def _write_chunk(d: "os.PathLike", text: str, name: str = "chunks.jsonl") -> None:
    with open(os.path.join(str(d), name), "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": text, "title": "t", "source": "s"},
                           ensure_ascii=False) + "\n")


def test_offline_knowledge_cache_reuses_parsed_result(_knowledge_dir):
    _write_chunk(_knowledge_dir, "第一条知识")
    first = RAG._load_offline_knowledge()
    assert len(first) == 1
    second = RAG._load_offline_knowledge()
    assert second is first  # 命中缓存，不重新解析


def test_offline_knowledge_cache_invalidates_on_change(_knowledge_dir):
    """缓存必须随语料变化失效：OCR 导入是增量的，缓存成一次终身有效会让
    新导入的资料永远检索不到——这种 bug 不报错，只是安静地少给几条参考。"""
    _write_chunk(_knowledge_dir, "第一条知识")
    assert len(RAG._load_offline_knowledge()) == 1
    _write_chunk(_knowledge_dir, "第二条知识", name="chunks_b.jsonl")
    assert len(RAG._load_offline_knowledge()) == 2


def test_offline_knowledge_missing_dir_is_empty(_knowledge_dir):
    assert RAG._load_offline_knowledge() == []


# --------------------------------------------------------------------------
# JEV 本地模型：进程内共享，但按配置分片
# --------------------------------------------------------------------------
def test_jev_local_model_shared_across_clients(monkeypatch):
    """加载一次要 4 秒（torch 权重 + 词表），不该每建一个 JEVClient 就付一次。"""
    from engine import jev

    calls = []
    monkeypatch.setattr("engine.multimodal_train.load_local_classifier",
                        lambda cfg: calls.append(cfg) or object())
    jev.reset_local_cache()
    a = jev.JEVClient({"multimodal": {"model_dir": "m1"}})._ensure_local()
    b = jev.JEVClient({"multimodal": {"model_dir": "m1"}})._ensure_local()
    assert a is b and len(calls) == 1


def test_jev_local_model_cache_keyed_by_config(monkeypatch):
    """按配置分片：不带 key 的话，换个 model_dir 训出来的模型根本轮不到生效。"""
    from engine import jev

    calls = []
    monkeypatch.setattr("engine.multimodal_train.load_local_classifier",
                        lambda cfg: calls.append(cfg) or object())
    jev.reset_local_cache()
    jev.JEVClient({"multimodal": {"model_dir": "m1"}})._ensure_local()
    jev.JEVClient({"multimodal": {"model_dir": "m2"}})._ensure_local()
    assert len(calls) == 2
    jev.reset_local_cache()
