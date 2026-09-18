# -*- coding: utf-8 -*-
"""``src/llm/local_models`` 本机 Ollama 探测测试。

云端密钥会过期、会 401（本机 .env 里的 DeepSeek key 正是如此），而机器上往往
已经跑着一个 Ollama。让界面能列出**真实存在**的模型名，比让用户手敲
``qwen2.5-coder:7b`` 可靠得多。测试全部离线：monkeypatch 掉 urlopen。
"""
import io
import json
import os
import sys
import urllib.error

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from llm import local_models as LM  # noqa: E402


class _Resp:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def ollama_online(monkeypatch):
    """模拟 Ollama /api/tags 正常返回（含新旧两种字段写法）。"""
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        body = {"models": [{"name": "qwen2.5-coder:7b"},
                           {"model": "llama3.1:8b"},
                           {"name": "qwen2.5-coder:7b"}]}  # 重复项应被去重
        return _Resp(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(LM.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_probe_lists_available_models(ollama_online):
    res = LM.probe_ollama("http://localhost:11434/v1", use_cache=False)
    assert res["available"] is True
    assert res["models"] == ["llama3.1:8b", "qwen2.5-coder:7b"]  # 排序且去重
    assert res["base_url"] == "http://localhost:11434/v1"
    assert not res["error"]


def test_probe_hits_native_tags_endpoint_not_v1(ollama_online):
    LM.probe_ollama("http://localhost:11434/v1", use_cache=False)
    assert ollama_online[0].endswith("/api/tags")


def test_probe_offline_reports_reason(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(LM.urllib.request, "urlopen", boom)
    res = LM.probe_ollama("", use_cache=False)
    assert res["available"] is False
    assert res["models"] == []
    assert "Ollama" in res["error"] or "connection refused" in res["error"]


def test_probe_malformed_json_is_not_fatal(monkeypatch):
    monkeypatch.setattr(LM.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp(b"<html>not json</html>"))
    res = LM.probe_ollama("", use_cache=False)
    assert res["available"] is False and res["error"]


@pytest.mark.parametrize("raw,expected", [
    ("", "http://localhost:11434/v1"),
    ("http://localhost:11434", "http://localhost:11434/v1"),
    ("http://localhost:11434/", "http://localhost:11434/v1"),
    ("http://localhost:11434/v1", "http://localhost:11434/v1"),
    ("http://127.0.0.1:11434/api/tags", "http://127.0.0.1:11434/v1"),
    ("  http://localhost:11434/v1  ", "http://localhost:11434/v1"),
])
def test_normalize_base_url(raw, expected):
    assert LM.normalize_ollama_base_url(raw) == expected


def test_list_models_empty_when_offline(monkeypatch):
    monkeypatch.setattr(LM, "probe_ollama", lambda *a, **k: {"models": [], "available": False})
    assert LM.list_ollama_models() == []


def test_preferred_model_picks_coder_first():
    models = ["llama3.1:8b", "qwen2.5-coder:7b", "deepseek-r1:7b"]
    assert LM.preferred_model(models) == "qwen2.5-coder:7b"
    assert LM.preferred_model(["llama3.1:8b"]) == "llama3.1:8b"
    assert LM.preferred_model([]) == ""
