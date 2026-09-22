"""对话式挖掘的「文件上传 → 可挖因子」管道。

链路::

    上传(图片/文本/PDF/表格)
      → 落盘 data/uploads/
      → 解析成文本块 + （若有）结构化表
      → JEV 做结构化判定（类型/情绪/可对齐/是否前瞻/能否因子化）
      → ① 文本块 → 摘要注入挖掘 prompt（另类数据上下文）
      → ② 表格可对齐 → to_factor_time_series 直接派生外部因子列并入面板

设计取舍（都是被上一版的问题逼出来的）
--------------------------------------
1. **不把原文整篇塞进 prompt**。材料动辄几万字，塞进去既贵又会被 LLM 当成
   "什么都能用"。这里先经 JEV 压成可分支的判定，只把「结论 + 少量证据片段」
   注入，代码层面据此硬分流。
2. **前瞻信息必须显式拦截**。研报/公告常见"预计下半年…"，若直接做因子即前视
   偏差。JEV 的 ``forward_looking`` 高时只给警告并由 LLM 在 description 里说明，
   不做数值因子。
3. **图片不做假 OCR**。没有 tesseract 就只给尺寸/色彩/格式等结构化描述，
   绝不假装识别出了文字——假信号比没有信号更糟。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from engine.jev import JEVClient, override_data_type
from engine.unstructured_miner import DataUploadParser

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
TEXT_EXTS = {".txt", ".md", ".json", ".csv", ".xlsx", ".xls", ".pdf"}
SUPPORTED_EXTS = IMAGE_EXTS | TEXT_EXTS
_CNN_TYPE_LABELS = {"candlestick": "K线/行情截图", "table": "表格",
                    "textpage": "文本页/研报页", "other": "其他"}
# CNN 版式 → 材料类型：图片往往一个字都没有，文本线索必然判成"其他"，
# 这时候版式才是更硬的证据（K 线图 ≈ 行情截图，扫描文本页 ≈ 研报/公告）。
_CNN_TO_DATA_TYPE = {"candlestick": "quote_screenshot", "table": "quote_screenshot",
                     "textpage": "research"}


_NAME_LIMIT = 80


def _safe_name(name: str) -> str:
    """清洗成可落盘的文件名——**必须先保住扩展名，再截主干**。

    踩过的坑：旧实现是「整体清洗后统一截断到 80 字符」，论文类 PDF 的标题动辄
    90+ 字符，`.pdf` 正好落在截断线之外被切掉；落盘后 ``Path.suffix`` 为空，
    后面按扩展名分派时抛「不支持的文件类型：」——连扩展名都空着，完全看不懂。
    所以这里只截主干，扩展名永远原样保留。
    """
    raw = str(name or "upload")
    p = Path(raw)
    ext = p.suffix.lower()
    if not (len(ext) > 1 and all(c.isalnum() or c == "." for c in ext)):
        ext = ""
        stem = raw
    else:
        stem = p.stem
    stem = "".join(c for c in stem if c.isalnum() or c in "._-")
    keep = max(8, _NAME_LIMIT - len(ext))
    return (stem[:keep] or "upload") + ext


# ----------------------------------------------------------------------
# 长文上下文：预算 + 按页切片
# ----------------------------------------------------------------------
# 一篇论文正文常在 3~8 万字，早先固定 4000 字的摘录等于"只看开头"，
# 所以这里给出可调预算，并让「装不下」这件事**可见**（首页/末页保留，
# 中间省略多少页如实写明），而不是悄悄截断。
DEFAULT_CONTEXT_CHARS = 24000      # 单份材料的默认上下文预算（界面可调）
_MIN_PER_ITEM_CHARS = 2000         # 多份材料均分时的下限
_HEAD_FRAC = 0.65                  # 装不下时留给「开头」的比例，其余给结尾

# 章节标题启发式：PDF 里没有可靠的章节标记（outline 常缺失），只能从页首文本抓。
# 宁可漏，不可假——抓出来的目录会标注「自动识别，可能不全」。
_CHAPTER_RE = re.compile(
    r"^(?:"
    r"第\s*[0-9一二三四五六七八九十百]+\s*[章节篇部分讲]"
    r"|[0-9]+(?:\.[0-9]+){0,2}[\s、.]+[^\s0-9]"
    r"|[一二三四五六七八九十]+\s*[、.]\s*\S"
    r"|(?:abstract|introduction|related\s+work|background|method(?:ology)?|"
    r"experiment(?:s|al\s+setup)?|result(?:s)?|discussion|conclusion|"
    r"references|appendix)\b"
    r")", re.IGNORECASE)


def detect_chapters(pages: Sequence[str], max_titles: int = 40) \
        -> List[Tuple[str, int]]:
    """从页文本里抓章节标题 → ``[(标题, 页码1起), ...]``（启发式，可能不全）。

    只扫每页前若干行：真正的标题通常就在页首，扫全文会把正文里的
    "3.2 式" 之类误当成章节。
    """
    out: List[Tuple[str, int]] = []
    seen = set()
    for no, page in enumerate(pages, start=1):
        for line in str(page or "").splitlines()[:12]:
            s = line.strip()
            if not (2 <= len(s) <= 60):
                continue
            if not _CHAPTER_RE.match(s):
                continue
            key = s.lower()[:40]
            if key in seen:
                continue
            seen.add(key)
            out.append((s[:60], no))
            break
        if len(out) >= max_titles:
            break
    return out


def _render_pages(chunks: Sequence[Tuple[int, str]], n_pages: int) -> str:
    """给每页正文加页码标记：模型引用「第几页」时才有依据。"""
    return "\n".join(f"— 第 {no}/{n_pages} 页 —\n{txt}" for no, txt in chunks)


def _slice_pages(pages: Sequence[str], budget: int) \
        -> Tuple[List[Tuple[int, str]], str]:
    """预算装不下时按页保留「开头 + 结尾」，中间整段省略并如实标注。

    为什么是首尾而不是前 N 页：论文的贡献/结论/局限在结尾，只留开头等于
    把最该看的部分扔掉。省略的页数与字数必须写出来，否则模型会以为看到了全文。
    """
    n = len(pages)
    if sum(len(p) for p in pages) <= budget:
        return [(i, p) for i, p in enumerate(pages, 1)], ""

    head_budget = int(budget * _HEAD_FRAC)
    tail_budget = budget - head_budget
    hi, used = 0, 0
    while hi < n and used + len(pages[hi]) <= head_budget:
        used += len(pages[hi])
        hi += 1
    hi = max(1, hi)          # 至少给第一页，否则预算太小时一片空白
    ti, used_t = n, 0
    while ti > hi and used_t + len(pages[ti - 1]) <= tail_budget:
        ti -= 1
        used_t += len(pages[ti])
    ti = max(ti, hi + 1)     # 至少跳过一页，不然"省略了 0 页"等于没做切片
    if ti >= n and n - 1 > hi:
        # 单页就超过尾部预算时，宁可略微超一点也要留最后一页：结尾常有结论/贡献
        ti = n - 1

    head = [(i, p) for i, p in enumerate(pages[:hi], 1)]
    tail = [(i, p) for i, p in enumerate(pages[ti:], ti + 1)]
    skipped = ti - hi
    omitted = sum(len(p) for p in pages[hi:ti])
    note = (f"（上下文预算 {budget:,} 字装不下全文：按页保留开头 {hi} 页 + "
            f"结尾 {n - ti} 页，省略中间 {skipped} 页 / 约 {omitted:,} 字；"
            f"需要这部分请指定页码或章节再问一次）")
    return head + tail, note


def _fit_plain(text: str, budget: int) -> Tuple[str, str]:
    """无页结构（txt/md/json）时按字符首尾截断，同样标注省略量。"""
    if len(text) <= budget:
        return text, ""
    head = text[:int(budget * _HEAD_FRAC)]
    tail = text[-max(0, budget - len(head)):]
    omitted = len(text) - len(head) - len(tail)
    note = (f"（上下文预算 {budget:,} 字装不下全文：保留开头与结尾，"
            f"省略中间约 {omitted:,} 字）")
    return head + f"\n…（省略中间约 {omitted:,} 字）…\n" + tail, note


def _image_brief(path: Path, ocr_enabled: bool = False,
                 classifier: Any = None) -> Tuple[str, Dict[str, Any]]:
    """图片 → 结构化描述（+ 可选 OCR + 可选 CNN 版式判定）。

    没有装 Pillow / tesseract 时不硬啃，如实返回"无法解析"而不是编造文字；
    CNN 版式判定（K线/表格/文本页/其他）同理，模型没训过就不给，绝不猜。
    """
    meta: Dict[str, Any] = {"w": None, "h": None, "mode": None, "format": None}
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            meta.update({"w": im.width, "h": im.height, "mode": im.mode,
                         "format": im.format})
            if classifier is not None:
                try:
                    gray = np.asarray(im.convert("L"), dtype=np.float32)
                    pred = classifier.classify_image(gray)
                    if pred.get("image_type"):
                        meta["cnn_type"] = pred["image_type"]
                        meta["cnn_probs"] = pred.get("image_type_probs", {})
                except Exception as e:  # noqa: BLE001
                    meta["cnn_error"] = str(e)[:200]
            small = im.convert("RGB").resize((64, 64))
            palette = small.getcolors(64 * 64) or []
            top = sorted(palette, key=lambda c: -c[0])[:5]
            meta["dominant_colors"] = [
                f"#{r:02x}{g:02x}{b:02x}" for _, (r, g, b) in top]
    except Exception as e:  # noqa: BLE001
        meta["image_error"] = f"Pillow 不可用或图片损坏：{e}"
        return f"（图片无法解析：{e}）", meta

    text = (f"[图片] 尺寸 {meta['w']}x{meta['h']}，格式 {meta['format']}，"
            f"主色 {'/'.join(meta.get('dominant_colors', [])) or '未知'}")
    if meta.get("cnn_type"):
        text += f"\n[CNN 版式判定] {_CNN_TYPE_LABELS.get(meta['cnn_type'], meta['cnn_type'])}"
    if ocr_enabled:
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore

            with Image.open(path) as im:
                ocr = pytesseract.image_to_string(im, lang="chi_sim+eng")
            if ocr.strip():
                text += f"\n[OCR]\n{ocr.strip()[:3000]}"
                meta["ocr_chars"] = len(ocr.strip())
        except Exception as e:  # noqa: BLE001
            meta["ocr_error"] = str(e)[:200]
            text += f"\n（OCR 未启用：{meta['ocr_error']}）"
    else:
        text += "\n（未启用 OCR：仅结构描述，不臆测图中文字）"
    return text, meta


@dataclass
class UploadedItem:
    """一次上传的解析结果（可序列化，供 UI 展示与 prompt 注入）。"""

    name: str                       # 落盘文件名（含时间戳前缀）
    path: str
    kind: str                       # image / text / table
    size: int
    source_name: str = ""           # 用户看到的原始文件名（去重键：落盘名每次都不同）
    text: str = ""
    preview: str = ""
    # PDF 按页正文：分页切片 / 页码引用 / 章节目录都靠它；不落 index.json（太大）
    pages: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    jev: Dict[str, Any] = field(default_factory=dict)
    factor_name: Optional[str] = None
    factor: Optional[pd.Series] = None   # index=(date, symbol)

    def to_json(self) -> Dict[str, Any]:
        # pages 是全文副本，落盘会把 index.json 撑成几 MB，审计索引里不需要它
        d = {k: v for k, v in self.__dict__.items() if k not in ("factor", "pages")}
        d["has_factor"] = self.factor is not None
        d["n_pages"] = len(self.pages)
        return d


class UploadIngestor:
    """把上传文件变成挖掘可用的两样东西：上下文文本 + 外部因子列。"""

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self.cfg_root = cfg
        up = cfg.get("data", {}).get("uploads", {}) or {}
        self.cfg = up
        self.dir = Path(str(up.get("dir") or "data/uploads"))
        if not self.dir.is_absolute():
            self.dir = Path.cwd() / self.dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_file = Path(str(up.get("index_file") or "data/uploads/index.json"))
        if not self.index_file.is_absolute():
            self.index_file = Path.cwd() / self.index_file
        self.context_max_chars = int(
            up.get("context_max_chars") or DEFAULT_CONTEXT_CHARS)
        self.max_files = int(up.get("max_files") or 12)
        self.max_bytes = int(up.get("max_bytes") or 20 * 1024 * 1024)
        self.ocr_enabled = bool(up.get("ocr_enabled"))
        self.parse_images = bool(up.get("parse_images", True))
        self.jev = JEVClient(cfg)
        self._parser = DataUploadParser()
        self.items: List[UploadedItem] = []
        self._cnn: Any = None            # 本地 CNN 图片版式模型（惰性加载）
        self._cnn_loaded = False

    # ------------------------------------------------------------------
    @property
    def cnn(self) -> Any:
        """本地 CNN 版式分类器；未训练/未启用时为 None（调用方需容忍）。"""
        if not self._cnn_loaded:
            self._cnn_loaded = True
            try:
                from engine.multimodal_train import load_local_classifier

                self._cnn = load_local_classifier(self.cfg_root)
            except Exception as e:  # noqa: BLE001
                logger.debug("[upload] 本地图文模型不可用：%s", e)
                self._cnn = None
        return self._cnn

    # ------------------------------------------------------------------
    def save(self, name: str, data: bytes) -> Path:
        if len(data) > self.max_bytes:
            raise ValueError(
                f"{name} 超过单文件上限 {self.max_bytes / 1024 / 1024:.0f}MB")
        safe = _safe_name(name)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        digest = hashlib.md5(data[:4096]).hexdigest()[:6]
        path = self.dir / f"{stamp}_{digest}_{safe}"
        path.write_bytes(data)
        return path

    # ------------------------------------------------------------------
    def _parse_table(self, path: Path) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
        ext = path.suffix.lower()
        if ext in {".csv"}:
            df = pd.read_csv(path, encoding="utf-8", encoding_errors="ignore")
        elif ext in {".xlsx", ".xls"}:
            df = pd.read_excel(path, sheet_name=0)
        else:
            df = pd.DataFrame()
        if df.empty:
            return df, {}, ""
        mapping = self._parser._detect_column_mapping(list(df.columns))  # noqa: SLF001
        text = "；".join(f"{c}({mapping.get(c, '?')})" for c in list(df.columns)[:20])
        return df, mapping, f"[表格] {df.shape[0]} 行 × {df.shape[1]} 列，列映射：{text}"

    def _parse_text(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        ext = path.suffix.lower()
        if ext in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore"), {}
        if ext == ".json":
            try:
                raw = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
                return json.dumps(raw, ensure_ascii=False)[:20000], {}
            except Exception:  # noqa: BLE001
                return path.read_text(encoding="utf-8", errors="ignore"), {}
        # pdf / 其他：复用已有解析器（PyPDF2 可用时走正文抽取）
        # 逐页保留：全文概括要按页切片，压成一个字符串就没法再切了。
        try:
            df, meta = self._parser.parse_file(path)
            if "text" in df.columns:
                pages = [str(t) for t in df["text"].tolist()]
                meta = {**meta, "pages": pages}
                return "\n".join(pages), meta
        except Exception as e:  # noqa: BLE001
            logger.warning("[upload] %s 解析失败：%s", path.name, e)
        return "", {"parse_error": True}

    # ------------------------------------------------------------------
    def ingest_file(self, path: str | Path) -> UploadedItem:
        p = Path(path)
        ext = p.suffix.lower()
        if ext not in SUPPORTED_EXTS:
            raise ValueError(
                f"不支持的文件类型：{ext or '(无扩展名)'}（{p.name}），"
                f"支持 {sorted(SUPPORTED_EXTS)}")
        size = p.stat().st_size
        meta: Dict[str, Any] = {"ext": ext}
        df = pd.DataFrame()
        mapping: Dict[str, Any] = {}
        pages: List[str] = []

        if ext in IMAGE_EXTS:
            kind = "image"
            if not self.parse_images:
                text = "（已关闭图片解析）"
            else:
                text, imeta = _image_brief(p, self.ocr_enabled, classifier=self.cnn)
                meta.update(imeta)
        elif ext in {".csv", ".xlsx", ".xls"}:
            kind = "table"
            df, mapping, text = self._parse_table(p)
            meta.update({"shape": list(df.shape), "mapping": mapping})
        else:
            kind = "text"
            text, tmeta = self._parse_text(p)
            # pages 挂到 item.pages（不入 meta：meta 会随 index.json 落盘）
            pages = [str(x) for x in (tmeta.pop("pages", None) or []) if str(x).strip()]
            meta.update(tmeta)

        item = UploadedItem(
            name=p.name, path=str(p), kind=kind, size=size,
            text=str(text or ""), preview=str(text or "")[:400], meta=meta,
        )
        if pages:
            item.pages = pages
            meta["pages_n"] = len(pages)
            chapters = detect_chapters(pages)
            if chapters:
                meta["chapters"] = [[t, n] for t, n in chapters]
        # JEV 结构化判定（无 Key 时自动本地规则降级，不阻塞）
        try:
            item.jev = self.jev.analyze(item.text, filename=p.name)
        except Exception as e:  # noqa: BLE001
            item.jev = {"engine": "unavailable", "error": str(e)[:200], "summary": ""}

        # 图片：CNN 认出版式后覆盖类型判定（文本线索对无字图片必然判"其他"）
        if kind == "image" and item.meta.get("cnn_type"):
            dt = _CNN_TO_DATA_TYPE.get(str(item.meta["cnn_type"]))
            if dt:
                item.jev = override_data_type(item.jev, dt,
                                              source=f"cnn:{item.meta['cnn_type']}")

        # 表格且可对齐 → 直接派生外部因子
        if kind == "table" and "date" in mapping and "symbol" in mapping:
            try:
                ts = self._parser.to_factor_time_series(df, mapping)
                if not ts.empty:
                    item.factor_name = f"upload_{_safe_name(p.stem)}"
                    item.factor = ts.set_index(
                        [ts["date"].astype(str), ts["symbol"].astype(str)])["factor"]
                    item.meta["factor_rows"] = len(ts)
            except Exception as e:  # noqa: BLE001
                item.meta["factor_error"] = str(e)[:200]

        self.items.append(item)
        if len(self.items) > self.max_files:
            self.items = self.items[-self.max_files:]
        return item

    def ingest_bytes(self, name: str, data: bytes) -> UploadedItem:
        item = self.ingest_file(self.save(name, data))
        # 落盘名带时间戳+摘要，每次都不同；去重要用用户看到的原始名
        item.source_name = str(name or "") or item.name
        return item

    # ------------------------------------------------------------------
    def context_text(self, max_chars: Optional[int] = None,
                     per_item: Optional[int] = None) -> str:
        """给挖掘 / 问答 prompt 的材料上下文：JEV 判定 + 原文（超预算按页保留首尾）。

        ``max_chars`` 是**总预算**（默认 :data:`DEFAULT_CONTEXT_CHARS`，界面可调），
        多份材料均分；``per_item`` 显式给定单份预算时优先。

        早先固定每份 4000 字，长论文只能看到开头，"全文概括"必然答不全。现在：
        - 有页结构（PDF）走 :func:`_slice_pages`——开头 + 结尾，中间整段省略并写明页数；
        - 无页结构（txt/md/json）走 :func:`_fit_plain`，按字符首尾截断并写明字数；
        - 附页码标记与自动识别的目录，模型引用"第几页"才有依据。
        """
        if not self.items:
            return ""
        total = int(max_chars or self.context_max_chars)
        each = int(per_item or max(_MIN_PER_ITEM_CHARS, total // len(self.items)))
        blocks = [self.jev.render_many({it.name: it.jev for it in self.items})]
        for it in self.items:
            if not it.text:
                continue
            blocks.append(self.item_block(it, each))
        return "\n".join(b for b in blocks if b).strip()

    def item_block(self, it: UploadedItem, budget: int) -> str:
        """单份材料的上下文块（头部说明 + 正文），UI 与 prompt 共用。"""
        name = it.source_name or it.name
        size = f"{len(it.text):,} 字" + (f" / {len(it.pages)} 页" if it.pages else "")
        head = f"【{name} 材料正文（{size}）】"
        chapters = it.meta.get("chapters") or []
        if chapters:
            toc = "；".join(f"{t}(p{n})" for t, n in chapters[:24])
            head += f"\n目录（自动识别，可能不全）：{toc}"
        body, note = self.fit_text(it, budget)
        if note:
            head += f"\n{note}"
        return f"{head}\n{body}"

    @staticmethod
    def fit_text(it: UploadedItem, budget: int) -> Tuple[str, str]:
        """按预算取出正文，返回 ``(正文, 省略说明)``；没省略时说明为空串。"""
        if it.pages:
            chunks, note = _slice_pages(it.pages, budget)
            return _render_pages(chunks, len(it.pages)), note
        return _fit_plain(it.text, budget)

    def external_factors(self) -> Dict[str, pd.Series]:
        """可并入面板的外部因子（只有成功派生的表格才有）。"""
        out: Dict[str, pd.Series] = {}
        for it in self.items:
            if it.factor is not None and it.factor_name:
                out[it.factor_name] = it.factor
        return out

    # ------------------------------------------------------------------
    def persist_index(self) -> Path:
        """把本次会话的上传索引落盘（审计用：谁在什么时候传了什么、判定如何）。"""
        payload = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "items": [it.to_json() for it in self.items],
        }
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        self.index_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.index_file


__all__ = ["DEFAULT_CONTEXT_CHARS", "IMAGE_EXTS", "SUPPORTED_EXTS",
           "UploadIngestor", "UploadedItem", "detect_chapters"]
