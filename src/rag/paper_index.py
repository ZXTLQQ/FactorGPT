"""
因子知识库索引（src/rag/paper_index.py）

维护一份「因子知识语料」，包含经典量化因子的定义、公式、适用场景与参考实现，
既作为 RAG 检索的内容来源，也作为未接入向量库时的本地兜底知识。

提供：
- FactorPaperIndex: 基于 ChromaDB 构建/查询向量索引（可选）。
- SEED_FACTORS: 内置的经典因子语料（动量、反转、规模、价值、质量、波动率、
  流动性、成长、分析师情绪等）。
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# ----------------------------------------------------------------------
# HuggingFace 下载环境配置（国内镜像，解决 BGE 模型下载超时）
# ----------------------------------------------------------------------
def _rag_config() -> dict:
    """读取 config.yaml 的 rag 段配置（不触发任何下载）。"""
    try:
        cfg_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("rag", {}) or {}
    except Exception:
        return {}


def _apply_hf_env(cfg: dict) -> None:
    """在首次下载前应用 HuggingFace 下载配置：镜像端点 / 缓存目录 / 超时。

    sentence-transformers / chromadb 默认都从 huggingface.co 拉权重，国内网络
    常被阻断导致 WinError 10060。设置 HF_ENDPOINT 指向 hf-mirror.com 即可正常下载。
    仅当用户/系统环境变量未显式设置时才写入，避免覆盖用户意图。
    """
    endpoint = cfg.get("hf_endpoint", "https://hf-mirror.com")
    if endpoint and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = endpoint
    hf_home = cfg.get("hf_home", "")
    if hf_home and not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = hf_home
    timeout = cfg.get("hf_hub_timeout", 60)
    if not os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT"):
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(timeout)
    # huggingface_hub（如 1.22.0）在导入时把 HF_ENDPOINT 捕获为模块级常量 ENDPOINT，
    # 之后仅改 os.environ 不足以影响已捕获的常量，导致下载仍走官方源而超时。
    # 显式导入子模块并覆盖该常量；与启动脚本的环境变量注入互为冗余。
    try:
        from huggingface_hub import constants as _hf_const

        ep = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        if hasattr(_hf_const, "ENDPOINT"):
            _hf_const.ENDPOINT = ep
    except Exception:
        pass


# ----------------------------------------------------------------------
# 内置因子知识语料（作为 RAG 内容来源与兜底知识）
# ----------------------------------------------------------------------
SEED_FACTORS: List[Dict[str, str]] = [
    {
        "title": "动量因子 (Momentum)",
        "category": "动量/反转",
        "formula": "Mom = ∏(1+r_t) - 1 over [t-20, t-1]",
        "description": (
            "买入过去一段时间涨幅居前的股票、卖出跌幅居前的股票。Jegadeesh & Titman (1993) "
            "证明 3-12 月动量在美股显著。A 股中动量在中长期有效，但需注意短期反转与换手成本。"
        ),
    },
    {
        "title": "短期反转因子 (Short-term Reversal)",
        "category": "动量/反转",
        "formula": "Rev = -∑ r_t over [t-5, t-1]",
        "description": (
            "基于市场过度反应，过去一周表现差的股票在未来短期出现反弹。常与动量因子低相关，"
            "可作为组合分散化来源。"
        ),
    },
    {
        "title": "规模因子 (Size)",
        "category": "风格",
        "formula": "Size = -log(总市值)",
        "description": (
            "小市值股票长期跑赢大市值股票（Banz 1981）。A 股小市值效应历史上显著，"
            "但 2017 年后受大盘风格影响阶段性失效，需结合市值中性化处理。"
        ),
    },
    {
        "title": "价值因子 (Value)",
        "category": "基本面",
        "formula": "Value = -EP / BP / SP 等估值指标综合",
        "description": (
            "低估值股票相对高估值股票有超额收益（Graham, Fama-French HML）。常用 "
            "EP、BP、CFP、SP 等，建议做行业市值中性化后使用。"
        ),
    },
    {
        "title": "质量因子 (Quality)",
        "category": "基本面",
        "formula": "Quality = ROE / 毛利率 / 杠杆率 综合",
        "description": (
            "盈利能力强、经营质量高的公司长期跑赢（Novy-Marx 2013 的盈利能力因子）。"
            "可结合资产周转率与低财务杠杆构建。"
        ),
    },
    {
        "title": "波动率因子 (Volatility)",
        "category": "风险",
        "formula": "Vol = -std(r_t) over 20 日",
        "description": (
            "低波动股票风险调整后收益更高（低波动异象, Baker & Haugen）。做空高波动、"
            "做多低波动可获得稳定超额收益。"
        ),
    },
    {
        "title": "流动性因子 (Liquidity)",
        "category": "微观结构",
        "formula": "Liq = log(日均成交额) 或 Amihud 非流动性",
        "description": (
            "流动性差的股票要求更高风险补偿（Amihud 非流动性指标 = |收益|/成交额）。"
            "A 股中流动性因子与规模因子高度相关，中性化后更纯净。"
        ),
    },
    {
        "title": "成长因子 (Growth)",
        "category": "基本面",
        "formula": "Growth = 营收/利润同比增长率 或 价格趋势斜率",
        "description": (
            "高成长公司未来现金流预期更高。可用财务增速代理，也可用价格趋势斜率等"
            "量价指标近似捕捉市场对其成长的定价。"
        ),
    },
    {
        "title": "分析师情绪因子 (Analyst Sentiment)",
        "category": "情绪",
        "formula": "Sent = 一致预期净利润变化率 / 评级上调",
        "description": (
            "分析师盈利预测上调伴随 positive 异常收益（盈余漂移 PFE）。需结合新闻舆情数据，"
            "注意评级的滞后与乐观偏差。"
        ),
    },
    {
        "title": "财报质量因子 (Accruals)",
        "category": "基本面",
        "formula": "Accruals = (净利润 - 经营现金流) / 总资产",
        "description": (
            "应计利润高的公司未来收益较低（Sloan 1996）。高应计意味着盈利质量差，"
            "是质量因子的补充维度。"
        ),
    },
]


class FactorPaperIndex:
    """因子知识向量索引（ChromaDB，可选）。

    若环境中未安装 chromadb / sentence-transformers，则降级为「仅本地语料」模式，
    由 FactorRetriever 的 SimpleRetriever 提供基于 jieba 的轻量检索。
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        cfg = _rag_config()
        _apply_hf_env(cfg)
        self.persist_dir = persist_dir or cfg.get("chroma_persist_dir", "./chroma_db")
        self.embedding_model = embedding_model or cfg.get(
            "embedding_model", "BAAI/bge-small-zh-v1.5"
        )
        self._client = None
        self._collection = None
        self._available = False
        self._init_vector_store()

    def _init_vector_store(self) -> None:
        try:
            _apply_hf_env(_rag_config())
            import chromadb  # noqa: F401
            from chromadb.config import Settings  # type: ignore

            os.makedirs(self.persist_dir, exist_ok=True)
            client = chromadb.PersistentClient(
                path=self.persist_dir, settings=Settings(anonymized_telemetry=False)
            )
            self._client = client
            self._available = True
        except Exception as e:  # pragma: no cover
            print(f"[FactorPaperIndex] 向量库不可用，使用本地语料模式: {e}")
            self._available = False

    def _embedding_function(self):
        """构造嵌入函数：优先使用配置的 BGE 模型（sentence-transformers）。

        下载前会应用 HF 镜像/超时配置（见 _apply_hf_env）。若加载失败，返回 None，
        由 Chroma 使用其自带的 ONNX 默认嵌入模型（本地、无需联网），确保知识库
        始终能被填充与检索，而不会因向量模型下载失败留下空集合、导致用户上传的
        资料“消失”。
        """
        _apply_hf_env(_rag_config())
        try:
            from chromadb.utils import embedding_functions as ef  # type: ignore

            return ef.SentenceTransformerEmbeddingFunction(model_name=self.embedding_model)
        except Exception as e:  # pragma: no cover
            endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
            print(
                f"[FactorPaperIndex] 未加载 {self.embedding_model}（已尝试镜像 {endpoint}），"
                f"将回退到 Chroma 内置默认嵌入模型（离线可用）: {e}"
            )
            return None

    def _ensure_collection(self, rebuild_if_model_changed: bool = True):
        """获取（或创建）factor_knowledge 集合，并确保使用当前配置的嵌入函数。

        健壮性关键：Chroma 不允许在 get_or_create 时变更已持久化的嵌入函数
        （如集合曾用 Chroma 默认模型创建、现改用 BGE）。冲突时先读取旧文档文本，
        删除集合后用当前模型重建并重嵌，以保留用户已上传的资料，避免“消失”。
        """
        if self._collection is not None:
            return self._collection
        if not self._available:
            return None
        ef = self._embedding_function()
        try:
            return self._client.get_or_create_collection(
                name="factor_knowledge",
                metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_model},
                embedding_function=ef,
            )
        except ValueError:
            # 嵌入函数冲突：读取旧文档 → 删除 → 用当前模型重建 → 重嵌（保留上传）
            try:
                old = self._client.get_collection("factor_knowledge")
                data = old.get(include=["documents", "metadatas"])
                ids = data.get("ids") or []
                docs = data.get("documents") or []
                metas = data.get("metadatas")
            except Exception:
                ids, docs, metas = [], [], None
            try:
                self._client.delete_collection("factor_knowledge")
            except Exception:
                pass
            col = self._client.get_or_create_collection(
                name="factor_knowledge",
                metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_model},
                embedding_function=ef,
            )
            if ids:
                try:
                    col.add(documents=docs, ids=ids, metadatas=metas)
                except Exception as e:  # pragma: no cover
                    print(f"[FactorPaperIndex] 旧文档重嵌失败（已丢弃）: {e}")
            self._collection = col
            return col

    @property
    def available(self) -> bool:
        return self._available

    def _doc_text(self, item: Dict[str, str]) -> str:
        return (
            f"【{item['title']}】（{item['category']}）\n"
            f"公式：{item['formula']}\n"
            f"说明：{item['description']}"
        )

    def count(self) -> int:
        """返回当前集合中的文档数（不可用/空时为 0）。"""
        try:
            col = self._ensure_collection()
            return col.count() if col is not None else 0
        except Exception:  # pragma: no cover
            return 0

    def build_from_seed(self) -> int:
        """将内置语料写入向量库；不可用向量库时返回 -1。"""
        col = self._ensure_collection()
        if col is None:
            return -1
        docs = [self._doc_text(it) for it in SEED_FACTORS]
        ids = [f"seed_{i}" for i in range(len(SEED_FACTORS))]
        # 清空旧 seed 数据（幂等）
        try:
            col.delete(ids=ids)
        except Exception:
            pass
        col.add(documents=docs, ids=ids)
        return len(docs)

    def add_document(self, item: Dict[str, str]) -> None:
        col = self._ensure_collection()
        if col is None:
            return
        col.add(documents=[self._doc_text(item)], ids=[item.get("title", "doc")])

    def add_texts(self, texts, metadatas=None) -> int:
        """供 UI 知识库上传使用：将若干文本写入向量库并持久化（返回写入条数）。

        向量库不可用（collection 为 None）时返回 0，由调用方提示用户。
        使用 uuid 生成唯一 id，避免同一内容重复上传时的 id 冲突。
        """
        col = self._ensure_collection()
        if col is None:
            return 0
        if metadatas is None:
            metadatas = [{} for _ in texts]
        ids = [f"user_{uuid.uuid4().hex}" for _ in texts]
        col.add(documents=list(texts), ids=ids, metadatas=metadatas)
        return len(texts)

    def query(self, text: str, top_k: int = 5) -> List[str]:
        col = self._ensure_collection()
        if col is None:
            return []
        res = col.query(query_texts=[text], n_results=top_k)
        return res.get("documents", [[]])[0] if res else []

    def export_seed_json(self, path: str = "factor_corpus.json") -> str:
        """导出内置语料为 JSON 文件，便于人工扩充。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(SEED_FACTORS, f, ensure_ascii=False, indent=2)
        return path
