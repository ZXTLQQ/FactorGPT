"""图文识别模型的离线训练（朴素贝叶斯 / Transformer / CNN / GNN）。

为什么自己训这一层
------------------
上传材料进来时，第一件要做的事是**判定它是什么**：行情截图？公告？研报？新闻？
自己的交易流水？——判定错了，后面的「能不能做因子、怎么做」全是错的。此前这件事
只有两条路：JEV（要 Key，且每次调用都要联网）和本地正则（够快但糙）。

本模块补上第三条路：**在本机把小模型训出来，落盘，推理时优先用**。四类模型各有分工，
不是凑数：

- **朴素贝叶斯（文本）**：三个头——材料类型、情绪极性、是否含前瞻信息。
  词袋 + 加平滑的闭式解，无梯度、无迭代，几十毫秒训完；它的价值是当
  Transformer 不可用（无 torch）时仍有可用的统计判别。
- **Transformer（文本）**：自注意力建模词序与长程搭配（如"预计……将"这种跨距离的
  前瞻线索），是四个模型里唯一能捕捉语序的。
- **CNN（图像）**：上传的图片不做 OCR 也能分类——K 线截图 / 表格 / 文本页 / 其他，
  判定结果决定后续走"数值列抽取"还是"版面描述"。
- **GNN（关系图）**：把同批材料按**共享标的、同一日期**连成图做节点分类。
  直觉：一份只写了"公司发布公告"、没写代码的材料，如果它与另一份标了 600519
  的材料同处一张关系图，类型信息可以沿着边传过来——这是纯文本模型拿不到的信号。

训练数据从哪来
--------------
默认用**合成语料 / 合成图片**（可复现、零隐私、CI 能跑）：模板 + 关键词池生成，
标签由生成过程确定，不存在"标注错了还自认为对"的问题。想用真实数据就把文件放进
目录后用 ``--text-dir / --image-dir`` 传入（真实语料放在 ``data/`` 下，不入库）。

产物与降级
----------
训练产物写到 ``data/models/multimodal/``（``*.npz`` / ``*.pt`` + ``training_report.json``）。
**torch 缺失时 Transformer/CNN/GNN 跳过并在报告里写明原因，朴素贝叶斯照常训练**——
与本项目"重依赖缺失优雅降级"的约定一致，不会因为没装 torch 就让整条链路不可用。
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # torch 是重依赖：缺失时本模块仍需可用（NB 路径不依赖它）
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── 标签体系（与 engine/jev.py 的 QUESTIONS 保持一致）──────────────────
DATA_TYPE_LABELS = ["quote_screenshot", "announcement", "research", "news",
                    "trading_log", "macro", "other"]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
IMAGE_LABELS = ["candlestick", "table", "textpage", "other"]

DEFAULT_MODEL_DIR = "data/models/multimodal"

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[一-鿿]")
_PAD, _UNK = 0, 1


# ======================================================================
# 1. 文本向量化：NB 与 Transformer 共用同一套词表
# ======================================================================
class TextVectorizer:
    """词/字级词表 + 定长编码。

    中文按**字**切、英文数字按**词**切：金融文本里"预计""增持"这类二字词是
    判别主力，按字切对短文本更稳；而代码/日期/英文缩写必须整词保留，切开就废了。
    """

    def __init__(self, max_features: int = 4000, max_len: int = 64) -> None:
        self.max_features = int(max_features)
        self.max_len = int(max_len)
        self.vocab: Dict[str, int] = {}
        self.fitted = False

    # ------------------------------------------------------------------
    @staticmethod
    def tokenize(text: str) -> List[str]:
        return _TOKEN_RE.findall(str(text or "").lower())

    def build(self, texts: Sequence[str]) -> "TextVectorizer":
        counts: Dict[str, int] = {}
        for t in texts:
            for tk in self.tokenize(t):
                counts[tk] = counts.get(tk, 0) + 1
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: self.max_features]
        self.vocab = {tk: i + 2 for i, (tk, _) in enumerate(top)}  # 0=pad 1=unk
        self.fitted = True
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """编码成 ``[N, max_len]`` 的 int 数组（截断/补零）。"""
        out = np.zeros((len(texts), self.max_len), dtype=np.int64)
        for i, t in enumerate(texts):
            ids = [self.vocab.get(tk, _UNK) for tk in self.tokenize(t)]
            ids = ids[: self.max_len]
            out[i, : len(ids)] = ids
        return out

    def counts(self, texts: Sequence[str]) -> np.ndarray:
        """词袋计数矩阵 ``[N, V]``（朴素贝叶斯用；不截断、不补零）。"""
        v = len(self.vocab) + 2
        out = np.zeros((len(texts), v), dtype=np.float64)
        for i, t in enumerate(texts):
            for tk in self.tokenize(t):
                j = self.vocab.get(tk, _UNK)
                out[i, j] += 1.0
        return out

    def to_json(self) -> Dict[str, Any]:
        return {"max_features": self.max_features, "max_len": self.max_len,
                "vocab": self.vocab}

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "TextVectorizer":
        v = cls(int(d.get("max_features") or 4000), int(d.get("max_len") or 64))
        v.vocab = {str(k): int(i) for k, i in (d.get("vocab") or {}).items()}
        v.fitted = bool(v.vocab)
        return v


# ======================================================================
# 2. 朴素贝叶斯（闭式解，零依赖）
# ======================================================================
class MultinomialNB:
    """多项式朴素贝叶斯 + 加一平滑；对数域计算，避免长文本下浮点下溢。"""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.classes: List[str] = []
        self.log_prior: np.ndarray = np.zeros(0)
        self.log_likelihood: np.ndarray = np.zeros((0, 0))

    def fit(self, X: np.ndarray, y: Sequence[str]) -> "MultinomialNB":
        self.classes = sorted(set(y))
        k = len(self.classes)
        idx = {c: i for i, c in enumerate(self.classes)}
        v = X.shape[1]
        counts = np.zeros((k, v), dtype=np.float64)
        total = np.zeros(k, dtype=np.float64)
        for row, lab in zip(X, y):
            i = idx[lab]
            counts[i] += row
            total[i] += float(row.sum())
        self.log_likelihood = np.log(counts + self.alpha) - np.log(
            (total + self.alpha * v)[:, None])
        n = np.array([sum(1 for l in y if l == c) for c in self.classes], dtype=np.float64)
        self.log_prior = np.log(n / n.sum())
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.classes:
            raise RuntimeError("模型未训练")
        return X @ self.log_likelihood.T + self.log_prior

    def predict(self, X: np.ndarray) -> List[str]:
        lp = self.predict_log_proba(X)
        return [self.classes[int(i)] for i in lp.argmax(axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        lp = self.predict_log_proba(X)
        lp -= lp.max(axis=1, keepdims=True)
        p = np.exp(lp)
        return p / p.sum(axis=1, keepdims=True)

    def save(self, path: str | Path) -> None:
        np.savez(path, classes=np.array(self.classes),
                 log_likelihood=self.log_likelihood, log_prior=self.log_prior,
                 alpha=np.array([self.alpha]))

    @classmethod
    def load(cls, path: str | Path) -> "MultinomialNB":
        z = np.load(path, allow_pickle=True)
        m = cls(float(z["alpha"][0]))
        m.classes = [str(c) for c in z["classes"]]
        m.log_likelihood = z["log_likelihood"]
        m.log_prior = z["log_prior"]
        return m


# ======================================================================
# 3. torch 模型（Transformer / CNN / GNN）
# ======================================================================
if TORCH_AVAILABLE:  # pragma: no cover - 依赖 torch 是否安装

    class TextTransformer(nn.Module):
        """``Embedding + 位置编码 + N 层自注意力 + 掩码均值池化 + 分类头``。

        batch_first=True，padding 位置用 key_padding_mask 排除，避免 pad 参与
        均值池化把表征拉向"全 pad"的方向（这是自制 Transformer 最常见的静默退化）。
        """

        def __init__(self, vocab_size: int, n_classes: int, d_model: int = 48,
                     n_heads: int = 4, n_layers: int = 2, max_len: int = 64,
                     dim_ff: int = 96, dropout: float = 0.1) -> None:
            super().__init__()
            self.d_model = d_model
            self.tok = nn.Embedding(vocab_size, d_model, padding_idx=_PAD)
            self.pos = nn.Embedding(max_len, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
                dropout=dropout, batch_first=True, activation="relu")
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = nn.Linear(d_model, n_classes)

        def forward(self, idx: "torch.Tensor") -> "torch.Tensor":
            n = idx.size(1)
            pos = torch.arange(n, device=idx.device).unsqueeze(0)
            h = self.tok(idx) * (self.d_model ** 0.5) + self.pos(pos)
            mask = idx.eq(_PAD)
            h = self.encoder(h, src_key_padding_mask=mask)
            keep = (~mask).unsqueeze(-1).float()
            pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
            return self.head(pooled)

    class ImageCNN(nn.Module):
        """两层卷积 + 全局池化的小 CNN（32×32 灰度图足够，不追求 ImageNet 级）。"""

        def __init__(self, n_classes: int, in_ch: int = 1) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.fc = nn.Linear(32, n_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            h = torch.relu(self.bn1(self.conv1(x)))
            h = torch.max_pool2d(h, 2)
            h = torch.relu(self.bn2(self.conv2(h)))
            h = torch.max_pool2d(h, 2)
            h = h.mean(dim=(2, 3))          # 全局平均池化：不依赖输入尺寸
            return self.fc(h)

    class GCN(nn.Module):
        """两层 GCN：``H = Â ReLU(Â X W1) W2``（Â 为对称归一化邻接，含自环）。"""

        def __init__(self, in_dim: int, hidden: int, n_classes: int) -> None:
            super().__init__()
            self.w1 = nn.Linear(in_dim, hidden)
            self.w2 = nn.Linear(hidden, n_classes)

        def forward(self, x: "torch.Tensor", a: "torch.Tensor") -> "torch.Tensor":
            h = torch.relu(a @ self.w1(x))
            return a @ self.w2(h)


def _torch_device() -> str:
    return "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"


# ======================================================================
# 4. 指标与切分
# ======================================================================
def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str],
                           labels: Sequence[str]) -> Dict[str, float]:
    """准确率 + macro-F1（macro 而非 micro：材料类型天然不均衡，micro 会被大类掩盖）。"""
    n = len(y_true)
    if n == 0:
        return {"acc": 0.0, "macro_f1": 0.0, "n": 0}
    acc = float(np.mean([a == b for a, b in zip(y_true, y_pred)]))
    f1s = []
    for lab in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b == lab)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != lab and b == lab)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {"acc": round(acc, 4), "macro_f1": round(float(np.mean(f1s)), 4), "n": n}


def _split(n: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = max(1, round(n * val_ratio)) if n > 4 else 0
    if k >= n:
        k = max(1, n // 3)
    return perm[k:], perm[:k]


# ======================================================================
# 5. 合成训练数据（可复现、零隐私）
# ======================================================================
_TOPIC: Dict[str, Dict[str, List[str]]] = {
    "quote_screenshot": {
        "kw": ["收盘价", "开盘价", "成交量", "盘口", "K线", "涨跌幅", "分时", "五档"],
        "tpl": ["{code} 今日{kw}为{v}，成交额{a}元",
                "{date} {code} {kw}{v}，换手率{r}%",
                "行情截图：{code} {kw}{v}，最高{hx}"],
    },
    "announcement": {
        "kw": ["公告", "财报", "年报", "季报", "重大事项", "董事会", "决议"],
        "tpl": ["公司发布{kw}：{code} 本期营收{v}元",
                "{code} 关于{kw}的说明，披露日期{date}",
                "{kw}显示{code}净利润{v}元，同比{r}%"],
    },
    "research": {
        "kw": ["研报", "评级", "目标价", "投资建议", "行业研究", "深度报告"],
        "tpl": ["{kw}：维持{code}增持评级，目标价{v}元",
                "本次{kw}覆盖{code}，预计后续需求{v}",
                "{date} {kw}认为{code}估值{r}%"],
    },
    "news": {
        "kw": ["快讯", "记者获悉", "消息面上", "据报道", "市场传闻"],
        "tpl": ["{kw}：{code} 盘中异动，涨跌幅{r}%",
                "{date} {kw}{code}相关产业链变化",
                "{kw}，{code}回应称经营正常"],
    },
    "trading_log": {
        "kw": ["委托", "成交", "平仓", "开仓", "报单", "手数"],
        "tpl": ["{time} {kw}{code} {v}手，价格{v2}",
                "自身交易流水：{kw}{code}，{time} 成交{v}手",
                "{kw}记录 {time} {code} 方向{v}"],
    },
    "macro": {
        "kw": ["GDP", "CPI", "政策", "央行", "统计局", "PMI", "社融"],
        "tpl": ["{date} {kw}数据公布，同比{r}%",
                "{kw}口径调整，{date}起实施",
                "宏观：{kw}环比{v}，市场影响有限"],
    },
    "other": {
        "kw": ["会议纪要", "内部培训", "读书笔记", "系统日志", "待办"],
        "tpl": ["{kw}：请各部门于{date}前反馈",
                "这是一份{kw}，涉及流程与分工",
                "{kw}记录，无行情信息"],
    },
}
_POS_WORDS = ["利好", "增长", "超预期", "上涨", "回暖", "扩张", "买入", "增持"]
_NEG_WORDS = ["利空", "下滑", "不及预期", "下跌", "萎缩", "亏损", "减持", "承压"]
_FWD_WORDS = ["预计", "预测", "展望", "将", "forecast", "有望", "目标价", "预期"]
# 各类型共用的"套话"：它们不携带类型信息，掺进去是为了让任务**不平凡**——
# 否则模板之间零词汇重叠，朴素贝叶斯随手就是 100%，指标漂亮但证明不了任何事。
_FILLER = [
    "数据来源：公开信息与行情终端",
    "仅供参考，不构成投资建议",
    "本材料基于公开资料整理，未经第三方核实",
    "如需引用请联系研究员获取授权版本",
    "内部编号归档，供后续流程调阅",
    "相关结论以正式披露文件为准",
]
# 约 25% 的样本刻意**既不出现类型关键词、也不出现标的**（只有日期与归档编号）。
# 这批"硬样本"是 GNN 存在的理由：文本里没有任何线索，纯文本模型只能瞎猜，但它的
# 归档元数据（同标的批次）仍把它连进关系图，类型可以沿边传过来。报告单列这一组对比。
_HARD_TPL = [
    "材料归档：{date}，内容已整理",
    "{date} 相关资料入库，编号{v}",
    "记录：{time} 文件已归档，页码{v}",
    "{date} 附件已上传，共{v}页",
]


def _rand_value(rng: random.Random) -> str:
    return f"{rng.uniform(1, 9999):.2f}"


def synthesize_text_corpus(n_per_class: int = 120, seed: int = 7) -> List[Dict[str, Any]]:
    """合成带标注的材料语料：类型 / 情绪 / 是否前瞻 / 提到的标的与日期。

    标签由**生成过程**直接确定（不是事后用规则猜），因此不存在标签噪声；
    ``forward_looking`` 靠"是否含前瞻措辞"决定，模型学的正是这个语言线索。
    """
    rng = random.Random(seed)
    items: List[Dict[str, Any]] = []
    # 标的与类型**必须相关**，否则关系图里没有同质性（homophily），GNN 学不到任何
    # 东西（实测准确率≈随机）。真实世界里就是这样：期货合约的材料多是行情截图与
    # 交易流水，个股材料多是公告/研报/新闻——这条相关性正是图结构的价值来源。
    codes_by_type = {
        "quote_screenshot": ["AU2601", "IF2601", "CU2601"],
        "trading_log": ["AU2601", "IF2601", "RB2601"],
        "macro": ["IF2601", "AU2601", "600519"],
        "announcement": ["600519", "000001", "601318"],
        "research": ["600519", "300750", "000001"],
        "news": ["300750", "601318", "000001"],
        "other": ["600519", "000001", "300750"],
    }
    for lab in DATA_TYPE_LABELS:
        spec = _TOPIC[lab]
        for _ in range(int(n_per_class)):
            tpl = rng.choice(spec["tpl"])
            code = rng.choice(codes_by_type[lab])
            date = f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
            hard = rng.random() < 0.25           # 硬样本：不含类型关键词
            txt = (rng.choice(_HARD_TPL) if hard else tpl).format(
                kw=rng.choice(spec["kw"]), code=code, date=date,
                v=_rand_value(rng), v2=_rand_value(rng), a=f"{rng.randint(1, 99)}亿",
                r=f"{rng.uniform(-9, 9):.1f}", hx=_rand_value(rng),
                time=f"{rng.randint(9, 15)}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}",
            )
            # 情绪：随机掺入正/负向词，让标签与文本真正相关但不是一一映射
            sent = "neutral"
            roll = rng.random()
            if roll < 0.35:
                txt += "，" + rng.choice(_POS_WORDS) + "，机构关注度提升"
                sent = "positive"
            elif roll < 0.7:
                txt += "，" + rng.choice(_NEG_WORDS) + "，短期承压"
                sent = "negative"
            if rng.random() < 0.45:
                txt += "。" + rng.choice(_FILLER)
            fwd = 1 if any(w in txt for w in _FWD_WORDS) else 0
            items.append({
                "text": txt, "data_type": lab, "sentiment": sent,
                "forward_looking": fwd, "symbols": [code], "date": date, "hard": hard,
            })
    rng.shuffle(items)
    return items


def synthesize_image_dataset(n_per_class: int = 60, size: int = 32,
                             seed: int = 7) -> Tuple[np.ndarray, List[str]]:
    """合成四类灰度图：K 线截图 / 表格 / 文本页 / 其他（噪声块）。

    合成而非爬图：一是隐私与版权，二是判据可控——K 线必须有"竖影线+实体"、
    表格必须有正交网格线、文本页必须是横向短笔画，标签不会含糊。
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for lab in IMAGE_LABELS:
        for _ in range(int(n_per_class)):
            img = np.full((size, size), 245.0, dtype=np.float64)
            if lab == "candlestick":
                x = 2
                price = size / 2.0
                while x < size - 2:
                    wick = rng.integers(4, 10)
                    body = rng.integers(2, 6)
                    drift = rng.normal(0, 2.5)
                    hi = int(np.clip(price - wick / 2, 0, size - 1))
                    lo = int(np.clip(price + wick / 2, 0, size - 1))
                    img[hi:lo, x] = 40.0
                    top = int(np.clip(min(price, price + drift), 0, size - 1))
                    bot = int(np.clip(max(price, price + drift), 0, size - 1))
                    img[top:bot + 1, max(0, x - 1):x + 2] = 30.0
                    price = float(np.clip(price + drift, 3, size - 4))
                    x += 3
            elif lab == "table":
                for r in range(2, size - 1, 5):
                    img[r, 1:size - 1] = 60.0
                for c in range(2, size - 1, 7):
                    img[1:size - 1, c] = 60.0
                for _ in range(12):
                    r = int(rng.integers(3, size - 3))
                    c = int(rng.integers(3, size - 6))
                    img[r, c:c + int(rng.integers(2, 5))] = 90.0
            elif lab == "textpage":
                r = 3
                while r < size - 3:
                    c = 2
                    while c < size - 4:
                        w = int(rng.integers(2, 6))
                        img[r:r + 2, c:c + w] = 55.0
                        c += w + int(rng.integers(1, 3))
                    r += 5
            else:
                img = np.clip(rng.normal(200, 35, (size, size)), 0, 255)
                for _ in range(3):
                    r = int(rng.integers(2, size - 8))
                    c = int(rng.integers(2, size - 8))
                    img[r:r + int(rng.integers(3, 8)), c:c + int(rng.integers(3, 8))] = 120.0
            # 随机亮度/对比度 + 较强噪声：否则四类版式干净可分，CNN 随手 100%，
            # 指标没有参考价值（真实截图有压缩、光照、水印，脏得多）。
            img = img * float(rng.uniform(0.75, 1.15)) + float(rng.uniform(-18, 18))
            img += rng.normal(0, 12.0, img.shape)
            X.append(np.clip(img, 0, 255).astype(np.float32))
            y.append(lab)
    arr = np.stack(X)[:, None, :, :] / 255.0
    return arr.astype(np.float32), y


def build_material_graph(items: Sequence[Dict[str, Any]], max_nodes: int = 320,
                         max_features: int = 600, max_degree: int = 3) -> Dict[str, Any]:
    """把一批材料按「共享标的 / 同一日期」连成图，供 GNN 做半监督节点分类。

    建图理由：单份材料的文本可能信息稀薄（如"公司发布公告"没写代码），但它与
    同标的、同日的其他材料相连，类型可以沿边传播——这正是纯文本模型拿不到的信号。

    ``max_degree`` 是必需的：同一标的下可能有上百份材料，直接连成团（clique）会让
    两层 GCN 把所有节点的表征拉成同一个向量（过平滑），准确率直接掉到随机猜测。
    这里给每个节点**随机保留至多 max_degree 条同实体边**再对称化，图稀疏了，
    消息传递才真正携带"局部关系"信息。
    """
    nodes = list(items)[:max_nodes]
    texts = [str(it.get("text", "")) for it in nodes]
    vec = TextVectorizer(max_features=max_features, max_len=48).build(texts)
    X = vec.counts(texts)
    X = X / np.clip(X.sum(axis=1, keepdims=True), 1.0, None)
    n = len(nodes)
    A = np.eye(n, dtype=np.float32)
    by_sym: Dict[str, List[int]] = {}
    by_date: Dict[str, List[int]] = {}
    for i, it in enumerate(nodes):
        for s in (it.get("symbols") or []):
            by_sym.setdefault(str(s), []).append(i)
        if it.get("date"):
            by_date.setdefault(str(it["date"]), []).append(i)
    rng = np.random.default_rng(20240607)
    # 同标的边权重高于同日期边：日期本身几乎不携带类型信息，等权会在消息传递时
    # 把"同质性"信号稀释掉（实测等权时硬样本准确率只比随机高 4 个点）。
    for groups, w in ((by_sym.values(), 1.0), (by_date.values(), 0.4)):
        for g in groups:
            if len(g) < 2:
                continue
            for i in g:
                others = [j for j in g if j != i]
                if len(others) > max_degree:
                    others = list(rng.choice(others, size=max_degree, replace=False))
                for j in others:
                    A[i, j] = max(A[i, j], w)
    A = np.maximum(A, A.T)                                    # 对称化：无向图
    # 对称化会把别人"选中我"的边加回来，度数仍可能失控；再按权重每行只保留
    # max_degree 条邻边（kNN 式剪枝），保证消息传递停留在局部而不是整图。
    for _ in range(2):
        for i in range(n):
            row = A[i].copy()
            row[i] = -1.0
            keep = np.argsort(-row)[:max_degree]
            mask = np.zeros(n, dtype=bool)
            mask[keep] = True
            A[i, ~mask] = 0.0
        A = np.maximum(A, A.T)
        A[np.diag_indices(n)] = 1.0                           # 自环保留：两层 GCN 需要
    deg = A.sum(axis=1, keepdims=True)
    a_norm = A / np.sqrt(np.clip(deg * deg.T, 1e-8, None))   # 对称归一化 Â = D^-1/2 A D^-1/2
    return {"X": X.astype(np.float32), "A": a_norm.astype(np.float32),
            "y": [str(it.get("data_type", "other")) for it in nodes], "vectorizer": vec}


# ======================================================================
# 6. 训练编排
# ======================================================================
@dataclass
class TrainConfig:
    """训练超参（默认按 CPU 小数据调过：几分钟内跑完，不占满内存）。"""

    out_dir: str = DEFAULT_MODEL_DIR
    text_per_class: int = 150
    image_per_class: int = 80
    epochs: int = 12
    batch_size: int = 32
    lr: float = 3e-3
    # GCN 全批量训练，收敛比小批量慢得多，且对学习率更敏感：给它单独的轮数与步长
    gcn_epochs: int = 200
    gcn_lr: float = 2e-2
    val_ratio: float = 0.2
    seed: int = 7
    image_dir: str = ""
    text_dir: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _train_torch_classifier(model: "nn.Module", X: np.ndarray, y_idx: np.ndarray,
                            cfg: TrainConfig, labels: Sequence[str]) -> Dict[str, Any]:
    """通用 torch 小模型训练循环（Adam + 交叉熵，固定种子保证可复现）。"""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = _torch_device()
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    tr, va = _split(len(X), cfg.val_ratio, cfg.seed)
    xt = torch.from_numpy(np.ascontiguousarray(X[tr])).to(dev)
    yt = torch.from_numpy(y_idx[tr]).to(dev)
    history: List[float] = []
    model.train()
    for _ in range(int(cfg.epochs)):
        perm = torch.randperm(len(xt), generator=torch.Generator().manual_seed(cfg.seed))
        running = 0.0
        for s in range(0, len(xt), int(cfg.batch_size)):
            b = perm[s:s + int(cfg.batch_size)]
            opt.zero_grad()
            loss = loss_fn(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(b)
        history.append(round(running / max(1, len(xt)), 4))
    model.eval()
    with torch.no_grad():
        if len(va):
            logits = model(torch.from_numpy(np.ascontiguousarray(X[va])).to(dev))
            pred = [labels[int(i)] for i in logits.argmax(1).cpu().numpy()]
            truth = [labels[int(i)] for i in y_idx[va]]
        else:
            pred = truth = []
    met = classification_metrics(truth, pred, labels)
    met.update({"final_loss": history[-1] if history else None, "device": dev,
                "epochs": int(cfg.epochs), "params": int(sum(
                    p.numel() for p in model.parameters()))})
    return {"metrics": met, "history": history}


def train_all(config: Optional[dict] = None, **overrides: Any) -> Dict[str, Any]:
    """训练四个模型并落盘，返回训练报告（含每个模型的指标与跳过原因）。

    ``config`` 取 ``multimodal`` 段（见 config.yaml）；``overrides`` 覆盖超参，
    便于脚本传 ``epochs=20`` 之类。torch 缺失时只训朴素贝叶斯并在报告里写明。
    """
    cfg_d = dict((config or {}).get("multimodal") or {})
    if cfg_d.get("model_dir") and not cfg_d.get("out_dir"):
        # 配置文件里叫 model_dir（与推理侧一致），训练脚本里叫 out_dir
        cfg_d["out_dir"] = cfg_d.pop("model_dir")
    cfg_d.update({k: v for k, v in overrides.items() if v is not None})
    known = set(TrainConfig.__dataclass_fields__)
    cfg = TrainConfig(**{k: v for k, v in cfg_d.items() if k in known})

    out = Path(cfg.out_dir)
    if not out.is_absolute():
        out = Path.cwd() / out
    out.mkdir(parents=True, exist_ok=True)

    items = synthesize_text_corpus(cfg.text_per_class, cfg.seed)
    texts = [it["text"] for it in items]
    vec = TextVectorizer().build(texts)
    X_cnt = vec.counts(texts)
    X_idx = vec.encode(texts)
    (out / "vectorizer.json").write_text(
        json.dumps(vec.to_json(), ensure_ascii=False), encoding="utf-8")

    report: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg.as_dict(), "torch_available": TORCH_AVAILABLE,
        "dataset": {"texts": len(texts), "labels": DATA_TYPE_LABELS},
        "models": {},
    }
    tr, va = _split(len(texts), cfg.val_ratio, cfg.seed)
    labels = DATA_TYPE_LABELS

    # —— 朴素贝叶斯 × 3 个头（类型 / 情绪 / 是否前瞻）——
    nb_dt: Optional[MultinomialNB] = None
    for task, key, labs in (("data_type", "nb_datatype.npz", labels),
                            ("sentiment", "nb_sentiment.npz", SENTIMENT_LABELS),
                            ("forward_looking", "nb_forward.npz", ["0", "1"])):
        y = [str(it[task]) for it in items] if task != "forward_looking" \
            else [str(int(it["forward_looking"])) for it in items]
        m = MultinomialNB().fit(X_cnt[tr], [y[i] for i in tr])
        if task == "data_type":
            nb_dt = m
        pred = m.predict(X_cnt[va]) if len(va) else []
        truth = [y[i] for i in va] if len(va) else []
        m.save(out / key)
        report["models"][f"naive_bayes[{task}]"] = {
            "engine": "numpy", "artifact": key,
            **classification_metrics(truth, pred, labs)}

    # —— Transformer（文本类型）——
    if TORCH_AVAILABLE:
        y_idx = np.array([labels.index(it["data_type"]) for it in items], dtype=np.int64)
        model = TextTransformer(vocab_size=len(vec.vocab) + 2, n_classes=len(labels),
                               max_len=vec.max_len)
        res = _train_torch_classifier(model, X_idx, y_idx, cfg, labels)
        torch.save({"state": model.state_dict(),
                    "meta": {"vocab_size": len(vec.vocab) + 2, "max_len": vec.max_len,
                             "labels": labels, "d_model": model.d_model}},
                   out / "transformer_text.pt")
        report["models"]["transformer[text]"] = {"engine": "torch",
                                                 "artifact": "transformer_text.pt",
                                                 **res["metrics"]}

        # —— CNN（图片版式）——
        Xi, yi = synthesize_image_dataset(cfg.image_per_class, seed=cfg.seed)
        if cfg.image_dir:
            Xi, yi = _append_real_images(Xi, yi, cfg.image_dir)
        y2 = np.array([IMAGE_LABELS.index(v) for v in yi], dtype=np.int64)
        cnn = ImageCNN(n_classes=len(IMAGE_LABELS))
        res_c = _train_torch_classifier(cnn, Xi, y2, cfg, IMAGE_LABELS)
        torch.save({"state": cnn.state_dict(),
                    "meta": {"labels": IMAGE_LABELS, "size": int(Xi.shape[-1])}},
                   out / "cnn_image.pt")
        report["models"]["cnn[image]"] = {"engine": "torch", "artifact": "cnn_image.pt",
                                          **res_c["metrics"]}

        # —— GNN（材料关系图上的半监督节点分类）——
        g = build_material_graph(items, max_nodes=320)
        gy = np.array([labels.index(v) for v in g["y"]], dtype=np.int64)
        gcn = GCN(in_dim=g["X"].shape[1], hidden=32, n_classes=len(labels))
        torch.manual_seed(cfg.seed)
        opt = torch.optim.Adam(gcn.parameters(), lr=cfg.gcn_lr, weight_decay=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()
        dev = _torch_device()
        Xt = torch.from_numpy(g["X"]).to(dev)
        At = torch.from_numpy(g["A"]).to(dev)
        yt = torch.from_numpy(gy).to(dev)
        n_tr = max(1, int(len(gy) * 0.7))
        gcn.train()
        for _ in range(int(cfg.gcn_epochs)):
            opt.zero_grad()
            loss = loss_fn(gcn(Xt, At)[:n_tr], yt[:n_tr])
            loss.backward()
            opt.step()
        gcn.eval()
        with torch.no_grad():
            pred = [labels[int(i)] for i in gcn(Xt, At)[n_tr:].argmax(1).cpu().numpy()]
        truth = [labels[int(i)] for i in gy[n_tr:]]
        torch.save({"state": gcn.state_dict(),
                    "meta": {"labels": labels, "in_dim": int(g["X"].shape[1]),
                             "hidden": 32, "nodes": len(gy)}},
                   out / "gcn_material.pt")
        gcn_met = classification_metrics(truth, pred, labels)
        report["models"]["gcn[graph]"] = {"engine": "torch", "artifact": "gcn_material.pt",
                                          **gcn_met}
        # 硬样本（不含类型关键词）对照：GNN 若真学到了关系传播，这一组应明显
        # 高于纯文本模型；否则说明图的边没带来信息，这点必须如实暴露。
        hard_test = [i - n_tr for i, it in enumerate(items[:320])
                     if it.get("hard") and i >= n_tr]
        if hard_test and nb_dt is not None:
            gcn_hard = classification_metrics(
                [labels[int(i)] for i in gy[n_tr:][hard_test]],
                [pred[i] for i in hard_test], labels)
            nb_hard = classification_metrics(
                [labels[int(i)] for i in gy[n_tr:][hard_test]],
                nb_dt.predict(X_cnt[n_tr:][hard_test]), labels)
            report["hard_node_check"] = {
                "n": len(hard_test),
                "gcn": {"acc": gcn_hard["acc"], "macro_f1": gcn_hard["macro_f1"]},
                "naive_bayes": {"acc": nb_hard["acc"], "macro_f1": nb_hard["macro_f1"]},
                "note": "硬样本不含类型关键词，只能靠关系图传播标签",
            }
    else:
        for name in ("transformer[text]", "cnn[image]", "gcn[graph]"):
            report["models"][name] = {"engine": "torch", "status": "skipped",
                                      "reason": "torch 未安装，已降级为朴素贝叶斯"}

    (out / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _append_real_images(X: np.ndarray, y: List[str], image_dir: str) -> Tuple[np.ndarray, List[str]]:
    """把真实图片目录（子目录名=标签）并进训练集；读不到就原样返回，不打断训练。"""
    root = Path(image_dir)
    if not root.is_dir():
        return X, y
    frames, labels_added = [], []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or sub.name not in IMAGE_LABELS:
            continue
        for p in sorted(sub.glob("*"))[:200]:
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            try:
                from PIL import Image  # type: ignore

                with Image.open(p) as im:
                    arr = np.asarray(im.convert("L").resize(
                        (int(X.shape[-1]), int(X.shape[-1]))), dtype=np.float32)
            except Exception as e:  # noqa: BLE001
                logger.warning("[multimodal] 图片读取失败 %s: %s", p.name, e)
                continue
            frames.append(arr / 255.0)
            labels_added.append(sub.name)
    if not frames:
        return X, y
    add = np.stack(frames)[:, None, :, :]
    return np.concatenate([X, add], axis=0), list(y) + labels_added


# ======================================================================
# 7. 推理：本地模型优先，规则兜底
# ======================================================================
class LocalMultimodalClassifier:
    """加载训练产物做本地推理；任一模型缺失就报 ``available=False``，调用方降级。

    刻意不做"加载失败就现场训练"：训练是显式的、要耗时的动作，不该在用户上传
    一个文件的瞬间偷偷发生（也避免 Streamlit 多进程各训一份互相覆盖）。
    """

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR) -> None:
        self.dir = Path(model_dir)
        if not self.dir.is_absolute():
            self.dir = Path.cwd() / self.dir
        self.vectorizer: Optional[TextVectorizer] = None
        self.nb: Dict[str, MultinomialNB] = {}
        self.transformer: Any = None
        self.cnn: Any = None
        self.gcn: Any = None
        self.meta: Dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        vf = self.dir / "vectorizer.json"
        if vf.is_file():
            try:
                self.vectorizer = TextVectorizer.from_json(
                    json.loads(vf.read_text(encoding="utf-8")))
            except Exception as e:  # noqa: BLE001
                logger.warning("[multimodal] 词表加载失败: %s", e)
        for key, name in (("nb_datatype", "data_type"), ("nb_sentiment", "sentiment"),
                          ("nb_forward", "forward_looking")):
            p = self.dir / f"{key}.npz"
            if p.is_file():
                try:
                    self.nb[name] = MultinomialNB.load(p)
                except Exception as e:  # noqa: BLE001
                    logger.warning("[multimodal] %s 加载失败: %s", key, e)
        rf = self.dir / "training_report.json"
        if rf.is_file():
            try:
                self.meta = json.loads(rf.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self.meta = {}
        if not TORCH_AVAILABLE:
            return
        for attr, fn, builder in (
            ("transformer", "transformer_text.pt", self._build_transformer),
            ("cnn", "cnn_image.pt", self._build_cnn),
            ("gcn", "gcn_material.pt", self._build_gcn),
        ):
            p = self.dir / fn
            if p.is_file():
                try:
                    setattr(self, attr, builder(p))
                except Exception as e:  # noqa: BLE001
                    logger.warning("[multimodal] %s 加载失败: %s", fn, e)

    def _build_transformer(self, p: Path):
        ck = torch.load(p, map_location="cpu", weights_only=True)
        meta = ck.get("meta") or {}
        m = TextTransformer(vocab_size=int(meta.get("vocab_size", 4002)),
                            n_classes=len(meta.get("labels") or DATA_TYPE_LABELS),
                            max_len=int(meta.get("max_len", 64)))
        m.load_state_dict(ck["state"])
        m.eval()
        return {"model": m, "labels": meta.get("labels") or DATA_TYPE_LABELS}

    def _build_cnn(self, p: Path):
        ck = torch.load(p, map_location="cpu", weights_only=True)
        meta = ck.get("meta") or {}
        m = ImageCNN(n_classes=len(meta.get("labels") or IMAGE_LABELS))
        m.load_state_dict(ck["state"])
        m.eval()
        return {"model": m, "labels": meta.get("labels") or IMAGE_LABELS,
                "size": int(meta.get("size", 32))}

    def _build_gcn(self, p: Path):
        ck = torch.load(p, map_location="cpu", weights_only=True)
        meta = ck.get("meta") or {}
        m = GCN(in_dim=int(meta.get("in_dim", 600)), hidden=int(meta.get("hidden", 32)),
                n_classes=len(meta.get("labels") or DATA_TYPE_LABELS))
        m.load_state_dict(ck["state"])
        m.eval()
        return {"model": m, "labels": meta.get("labels") or DATA_TYPE_LABELS}

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return bool(self.nb) or self.transformer is not None or self.cnn is not None

    def summary(self) -> Dict[str, Any]:
        return {
            "dir": str(self.dir),
            "nb": sorted(self.nb),
            "transformer": self.transformer is not None,
            "cnn": self.cnn is not None,
            "gcn": self.gcn is not None,
            "torch": TORCH_AVAILABLE,
            "report": self.meta.get("models", {}),
        }

    def classify_text(self, text: str) -> Dict[str, Any]:
        """文本判定：类型（Transformer 优先，其次 NB）+ 情绪 + 是否前瞻。"""
        out: Dict[str, Any] = {"engine": "local-model"}
        if self.vectorizer is None:
            return out
        t = [str(text or "")]
        if self.transformer is not None:
            idx = torch.from_numpy(self.vectorizer.encode(t))
            with torch.no_grad():
                logits = self.transformer["model"](idx)
                probs = torch.softmax(logits, dim=-1)[0].numpy()
            labs = self.transformer["labels"]
            out["data_type"] = str(labs[int(probs.argmax())])
            out["data_type_probs"] = {str(l): round(float(p), 4)
                                      for l, p in zip(labs, probs)}
            out["data_type_model"] = "transformer"
        elif "data_type" in self.nb:
            m = self.nb["data_type"]
            probs = m.predict_proba(self.vectorizer.counts(t))[0]
            out["data_type"] = m.predict(self.vectorizer.counts(t))[0]
            out["data_type_probs"] = {str(c): round(float(p), 4)
                                      for c, p in zip(m.classes, probs)}
            out["data_type_model"] = "naive_bayes"
        if "sentiment" in self.nb:
            m = self.nb["sentiment"]
            probs = m.predict_proba(self.vectorizer.counts(t))[0]
            out["sentiment"] = m.predict(self.vectorizer.counts(t))[0]
            out["sentiment_probs"] = {str(c): round(float(p), 4)
                                      for c, p in zip(m.classes, probs)}
        if "forward_looking" in self.nb and "1" in self.nb["forward_looking"].classes:
            m = self.nb["forward_looking"]
            probs = m.predict_proba(self.vectorizer.counts(t))[0]
            out["forward_looking"] = round(float(probs[m.classes.index("1")]), 4)
        return out

    def classify_image(self, arr: np.ndarray) -> Dict[str, Any]:
        """图片版式判定（CNN）；输入为 ``[H, W]`` 或 ``[1, H, W]`` 灰度数组。"""
        out: Dict[str, Any] = {"engine": "local-model"}
        if self.cnn is None:
            return out
        size = int(self.cnn.get("size", 32))
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 3:
            a = a.mean(axis=0)
        try:
            from PIL import Image  # type: ignore

            im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((size, size))
            a = np.asarray(im, dtype=np.float32)
        except Exception:  # noqa: BLE001
            # 无 Pillow 时用最近邻缩放（版式分类对插值方式不敏感）
            h, w = a.shape
            ys = (np.arange(size) * h / size).astype(int)
            xs = (np.arange(size) * w / size).astype(int)
            a = a[np.ix_(ys, xs)]
        x = torch.from_numpy((a / 255.0)[None, None, :, :])
        with torch.no_grad():
            probs = torch.softmax(self.cnn["model"](x), dim=-1)[0].numpy()
        labs = self.cnn["labels"]
        out["image_type"] = str(labs[int(probs.argmax())])
        out["image_type_probs"] = {str(l): round(float(p), 4) for l, p in zip(labs, probs)}
        return out


def load_local_classifier(config: Optional[dict] = None) -> Optional[LocalMultimodalClassifier]:
    """按配置加载本地模型；未启用/无产物时返回 None（调用方走规则或 JEV）。"""
    cfg = ((config or {}).get("multimodal") or {})
    if not bool(cfg.get("enabled", True)):
        return None
    cli = LocalMultimodalClassifier(cfg.get("model_dir") or DEFAULT_MODEL_DIR)
    return cli if cli.available else None


__all__ = [
    "DATA_TYPE_LABELS", "IMAGE_LABELS", "SENTIMENT_LABELS", "TORCH_AVAILABLE",
    "LocalMultimodalClassifier", "MultinomialNB", "TextVectorizer", "TrainConfig",
    "build_material_graph", "classification_metrics", "load_local_classifier",
    "synthesize_image_dataset", "synthesize_text_corpus", "train_all",
]
