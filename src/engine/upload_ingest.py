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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    meta: Dict[str, Any] = field(default_factory=dict)
    jev: Dict[str, Any] = field(default_factory=dict)
    factor_name: Optional[str] = None
    factor: Optional[pd.Series] = None   # index=(date, symbol)

    def to_json(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "factor"}
        d["has_factor"] = self.factor is not None
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
        try:
            df, meta = self._parser.parse_file(path)
            if "text" in df.columns:
                return "\n".join(str(t) for t in df["text"].tolist()), meta
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
            meta.update(tmeta)

        item = UploadedItem(
            name=p.name, path=str(p), kind=kind, size=size,
            text=str(text or ""), preview=str(text or "")[:400], meta=meta,
        )
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
    def context_text(self, max_chars: int = 4000) -> str:
        """给挖掘 prompt 的上下文：JEV 判定 + 少量原文证据。"""
        if not self.items:
            return ""
        blocks = [self.jev.render_many({it.name: it.jev for it in self.items})]
        for it in self.items:
            if it.text:
                blocks.append(f"【{it.name} 内容摘录】\n{it.text[:max_chars]}")
        return "\n".join(b for b in blocks if b).strip()

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


__all__ = ["IMAGE_EXTS", "SUPPORTED_EXTS", "UploadIngestor", "UploadedItem"]
