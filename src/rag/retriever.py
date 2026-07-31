"""
因子知识检索器（src/rag/retriever.py）

对外提供统一的 `retrieve(query, top_k)` 接口，返回与查询最相关的因子知识文本列表。

优先使用 ChromaDB + BGE 向量检索（需 sentence-transformers + chromadb）。
在缺少上述依赖的环境下，自动降级为 `SimpleRetriever`：基于 jieba 分词与
TF-IDF 余弦相似度在本地语料上检索，保证系统在任何环境下都可运行。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from rag.paper_index import FactorPaperIndex, SEED_FACTORS
from rag.learned_library import LearnedFactorLibrary

# 项目根目录（src/rag/retriever.py -> factor-gpt/）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# 离线知识语料目录：由 scripts/ingest_knowledge.py 落地（如 PDF OCR 分块）
_KNOWLEDGE_DIR = _PROJECT_ROOT / "data" / "knowledge"


def _load_offline_knowledge() -> List[Dict]:
    """加载离线知识语料（data/knowledge/**/chunks*.jsonl）。

    这些条目通常由图片型 PDF 经 OCR 后分块落地（见 scripts/ingest_knowledge.py），
    用于在无向量库/向量模型不可用时，仍能被 SimpleRetriever（jieba）检索到。
    条目不含 ``code``，因此不会污染 retrieve_template 的可调用因子模板。
    """
    items: List[Dict] = []
    if not _KNOWLEDGE_DIR.exists():
        return items
    for f in sorted(_KNOWLEDGE_DIR.rglob("chunks*.jsonl")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    text = d.get("text") or d.get("description") or ""
                    if not text.strip():
                        continue
                    items.append(
                        {
                            "title": d.get("title")
                            or f"{(d.get('source') or '知识')}-第{d.get('page', '?')}页",
                            "category": d.get("category") or d.get("source") or "离线知识",
                            "formula": d.get("formula", "") or "",
                            "description": text,
                            "author": d.get("author", "") or "",
                            "url": "",
                            "reference": "",
                            "code": "",  # 无代码 -> 不进入可调用因子模板
                            "source": d.get("source", ""),
                        }
                    )
        except Exception:
            continue
    return items


def rag_deps_available() -> bool:
    """向量检索依赖（chromadb + sentence-transformers）是否就绪。"""
    try:
        import chromadb  # noqa: F401
        import sentence_transformers  # noqa: F401

        return True
    except Exception:
        return False


def rag_vector_enabled(cfg: Optional[dict]) -> bool:
    """根据配置与依赖自动决定是否启用向量检索。

    - 配置显式 ``use_vector_store: false`` 时强制关闭（离线 jieba 检索）；
    - 其余情况（含配置缺失/为 true）在依赖就绪时自动启用，并首次自动下载模型。
    """
    rag = (cfg or {}).get("rag", {}) if isinstance(cfg, dict) else {}
    flag = rag.get("use_vector_store", True)
    if flag is False:
        return False
    return rag_deps_available()


class SimpleRetriever:
    """轻量关键词检索器：jieba 分词 + TF-IDF 余弦相似度（无向量库依赖）。

    检索对象 = 内置 SEED_FACTORS + 已学习因子库（外部导入 + 自学习）。
    """

    def __init__(self, corpus: Optional[List[dict]] = None) -> None:
        self.corpus = corpus or SEED_FACTORS
        try:
            import jieba

            self._jieba = jieba
        except ImportError:
            self._jieba = None
        self._items = self.corpus
        self._docs = [self._fmt(c) for c in self.corpus]
        self._vecs = [self._vectorize(d) for d in self._docs]

    @staticmethod
    def _fmt(item: dict) -> str:
        lines = [
            f"【{item.get('title','')}】（{item.get('category','')}）",
            f"公式：{item.get('formula','')}",
            f"说明：{item.get('description','')}",
        ]
        if item.get("author"):
            lines.append(f"作者：{item['author']}")
        if item.get("url"):
            lines.append(f"链接：{item['url']}")
        if item.get("reference"):
            lines.append(f"来源：{item['reference']}")
        return "\n".join(lines)

    def _tokenize(self, text: str) -> List[str]:
        if self._jieba is not None:
            return [t for t in self._jieba.cut(text) if len(t.strip()) > 1]
        # 退化为按字符/空格切分
        return [t for t in text.replace("\n", " ").split() if len(t) > 1]

    def _vectorize(self, text: str):
        import numpy as np

        toks = self._tokenize(text)
        vec = {}
        for t in toks:
            vec[t] = vec.get(t, 0) + 1
        # TF 归一化
        norm = np.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}

    @staticmethod
    def _cosine(a: dict, b: dict) -> float:
        common = set(a) & set(b)
        return sum(a[k] * b[k] for k in common)

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        qv = self._vectorize(query)
        scored = [(self._cosine(qv, v), i) for i, v in enumerate(self._vecs)]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]
        if not top:  # 无任何命中时返回前 top_k 条，保证有上下文
            return self._docs[:top_k]
        return [self._docs[i] for _, i in top]

    def retrieve_items(self, query: str, top_k: int = 5) -> List[Dict]:
        """与 retrieve 类似，但返回原始因子对象列表（含 code 等字段）。"""
        qv = self._vectorize(query)
        scored = [(self._cosine(qv, v), i) for i, v in enumerate(self._vecs)]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]
        if not top:
            return [self._items[i] for i in range(min(top_k, len(self._items)))]
        return [self._items[i] for _, i in top]


class FactorRetriever:
    """因子知识检索器（自动选择向量库或本地检索）。

    同时融合「内置因子语料」与「已学习因子库」，使得外部导入或 Agent
    自学习得到的因子既能参与知识检索（学习），也能作为代码模板被复用（调用）。
    """

    def __init__(
        self,
        index: Optional[FactorPaperIndex] = None,
        top_k: int = 5,
        use_vector_store: Optional[bool] = None,
        learned: Optional[LearnedFactorLibrary] = None,
    ) -> None:
        self.top_k = top_k
        self.index = index or FactorPaperIndex()
        self.learned = learned or LearnedFactorLibrary()
        # 合并内置语料 + 已学习因子 + 离线知识语料（PDF OCR 等），作为统一检索语料
        offline_knowledge = _load_offline_knowledge()
        combined = SEED_FACTORS + self.learned.all() + offline_knowledge
        self.simple = SimpleRetriever(corpus=combined)
        # 默认使用离线关键词检索（jieba）；当配置允许且依赖就绪时自动启用向量检索
        # （首次会自动下载嵌入模型，无需手动编辑配置）。
        if use_vector_store is False:
            self._use_vector = False
        else:
            # use_vector_store 为 True 或 None（自动）：依赖就绪时启用向量检索
            self._use_vector = bool(use_vector_store or rag_deps_available()) and self.index.available
        if self._use_vector:
            try:
                self.index.build_from_seed()
                for it in self.learned.all():
                    try:
                        self.index.add_document(it)
                    except Exception:
                        pass
                self._use_vector = True
            except Exception:
                self._use_vector = False

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        k = top_k or self.top_k
        if self._use_vector:
            try:
                res = self.index.query(query, top_k=k)
                if res:
                    return res
            except Exception:
                pass
        return self.simple.retrieve(query, top_k=k)

    def retrieve_template(self, query: str, top_k: int = 1) -> List[Dict]:
        """检索最匹配且「含代码实现」的已学习因子，供 Agent 直接复用（调用）。

        返回因子对象列表，按相关性排序，每个对象含 'title' / 'code' 等字段。
        内部会拉取较大的候选池再按「含代码」筛选，提升调用召回率。
        """
        candidates = self.simple.retrieve_items(query, top_k=max(top_k * 4, 8))
        with_code = [it for it in candidates if it.get("code")]
        return with_code[:top_k]

    def render_context(self, query: str, top_k: Optional[int] = None) -> str:
        """检索并将结果拼为可供 LLM 使用的上下文文本。"""
        docs = self.retrieve(query, top_k)
        if not docs:
            return "（暂无相关因子知识）"
        return "\n\n".join(f"【参考 {i+1}】\n{d}" for i, d in enumerate(docs))
