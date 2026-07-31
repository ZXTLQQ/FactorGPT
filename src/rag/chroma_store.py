"""
因子知识库检索入口（rag/chroma_store.py）

提供 `ensure_chroma()`：返回（并惰性构建）一个 FactorPaperIndex 单例，供
UI 知识库页面检索 / 上传使用。向量库（chromadb）不可用时自动降级为本地语料模式。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from rag.paper_index import FactorPaperIndex
from rag.retriever import rag_vector_enabled

_INDEX: Optional[FactorPaperIndex] = None


def _rag_config() -> dict:
    """读取 config.yaml 中的 rag 段配置（不触发任何下载）。"""
    try:
        cfg_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("rag", {}) or {}
    except Exception:
        return {}


def ensure_chroma(build: bool = False) -> FactorPaperIndex:
    """返回 FactorPaperIndex 单例；首次调用时仅创建客户端，不立即建集合。

    build=True 时才真正创建 Chroma 集合（会触发一次向量模型下载）；
    默认不构建，避免默认配置（use_vector_store=false）下启动即下载
    MiniLM ONNX 模型导致首屏极慢。是否构建由调用方按配置决定。
    """
    global _INDEX
    if _INDEX is None:
        _INDEX = FactorPaperIndex()
    if build and _INDEX.available:
        try:
            _INDEX.build_from_seed()
        except Exception:  # pragma: no cover
            pass
    return _INDEX
