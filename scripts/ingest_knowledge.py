"""
知识库摄入脚本（scripts/ingest_knowledge.py）

把用户提供的两份资料接入 FactorGPT 的 Agent 知识库，使其可「自适应学习 / 调用」：

1) 因子字典 xlsx（含代码）
   -> 复用 scripts/import_factors.py 导入 data/learned_factors.jsonl
   -> Agent 既能检索学习因子定义/指标，又能直接复用其中的 Python 因子代码（调用）。

2) 因子日历 PDF（纯图片型，无内嵌文字）
   -> 用 EasyOCR（ch_sim+en）逐页增量 OCR（可断点续跑）
   -> 分块写入 Chroma 向量库 factor_knowledge（在线语义检索）
   -> 同时落地到 data/knowledge/<name>/chunks.jsonl（离线 jieba 兜底检索）
   -> FactorRetriever 会自动加载该目录，保证离线也能检索到 PDF 知识。

用法：
  python scripts/ingest_knowledge.py --xlsx "Desktop/因子字典_含代码.xlsx" \
      --pdf "Desktop/因子日历2024 (量化投资与机器学习) .pdf"
  # 仅跑 PDF（断点续跑）：
  python scripts/ingest_knowledge.py --skip-xlsx --pdf "..."
  # 测试少量页：
  python scripts/ingest_knowledge.py --skip-xlsx --pdf "..." --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # rag 包位于 src/ 下
sys.path.insert(0, str(ROOT))

CHUNK_SIZE = 320  # 每个知识块最大字符数（按句切分，不硬截断中文）
OCR_SCALE = 1.5   # 渲染缩放，平衡速度与识别率


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ----------------------------------------------------------------------
# 1) 因子字典 xlsx -> 已学习因子库（可调用）
# ----------------------------------------------------------------------
def ingest_xlsx(xlsx_path: str, source: str) -> None:
    xlsx = Path(xlsx_path)
    if not xlsx.exists():
        log(f"[xlsx] 文件不存在，跳过：{xlsx}")
        return
    log(f"[xlsx] 开始导入：{xlsx}  (source={source})")
    cmd = [sys.executable, str(ROOT / "scripts" / "import_factors.py"), str(xlsx), source]
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode == 0:
        log("[xlsx] 导入完成 -> data/learned_factors.jsonl")
    else:
        log(f"[xlsx] 导入脚本返回非零码 {r.returncode}，请检查上方输出")


# ----------------------------------------------------------------------
# 2) 因子日历 PDF -> OCR -> 向量库 + 离线语料
# ----------------------------------------------------------------------
def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    # 先按换行/句号等切句，再按长度聚合成块
    seps = ["。", "；", "；", ".", ";", "！", "？", "!", "?"]
    sentences: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in seps:
            sentences.append(buf)
            buf = ""
    if buf:
        sentences.append(buf)
    chunks: list[str] = []
    cur = ""
    for s in sentences:
        if len(cur) + len(s) <= size:
            cur += s
        else:
            if cur:
                chunks.append(cur)
            # 单句超长则硬切
            if len(s) > size:
                for i in range(0, len(s), size):
                    chunks.append(s[i : i + size])
                cur = ""
            else:
                cur = s
    if cur:
        chunks.append(cur)
    return [c.strip() for c in chunks if c.strip()]


def ingest_pdf(pdf_path: str, limit: int | None = None) -> None:
    pdf = Path(pdf_path)
    if not pdf.exists():
        log(f"[pdf] 文件不存在，跳过：{pdf}")
        return

    import fitz  # PyMuPDF

    try:
        import easyocr
    except Exception as e:
        log(f"[pdf] 缺少 easyocr（OCR 引擎），无法处理图片型 PDF：{e}")
        return

    name = "因子日历2024"
    out_dir = ROOT / "data" / "knowledge" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_jsonl = out_dir / "pages.jsonl"
    chunks_jsonl = out_dir / "chunks.jsonl"

    # 已完成的页（断点续跑）
    done: set[int] = set()
    if pages_jsonl.exists():
        for line in open(pages_jsonl, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line).get("page")))
            except Exception:
                pass
    # 已完成分块的页
    chunked_pages: set[int] = set()
    if chunks_jsonl.exists():
        for line in open(chunks_jsonl, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                chunked_pages.add(int(json.loads(line).get("page")))
            except Exception:
                pass

    log(f"[pdf] 打开：{pdf}")
    doc = fitz.open(str(pdf))
    n = len(doc)
    log(f"[pdf] 共 {n} 页；已完成 OCR={len(done)} 页，已分块={len(chunked_pages)} 页")

    reader = easyocr.Reader(["ch_sim", "en"], gpu=False, verbose=False)
    pages_added = 0
    chunks_added = 0

    # 直接把 fitz 像素转为 numpy 数组喂给 EasyOCR，避免中文路径导致 cv2.imread 失败
    import cv2
    import numpy as np

    def page_to_array(pix) -> "np.ndarray":
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        elif pix.n == 1:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return arr

    # 惰性加载向量库（首次才触发，失败也不影响离线语料落地）
    idx = None
    try:
        from rag.paper_index import FactorPaperIndex

        idx = FactorPaperIndex()
        if not idx.available:
            idx = None
            log("[pdf] 向量库不可用，仅落地离线语料（chunks.jsonl）")
    except Exception as e:
        log(f"[pdf] 向量库初始化失败，仅落地离线语料：{e}")
        idx = None

    pf_pages = open(pages_jsonl, "a", encoding="utf-8")
    pf_chunks = open(chunks_jsonl, "a", encoding="utf-8")

    try:
        for pno in range(n):
            if limit is not None and pno >= limit:
                log(f"[pdf] 已达 --limit={limit}，停止")
                break
            if pno in done and pno in chunked_pages:
                continue
            t0 = time.time()
            pix = doc[pno].get_pixmap(matrix=fitz.Matrix(OCR_SCALE, OCR_SCALE))
            img_arr = page_to_array(pix)
            lines = reader.readtext(img_arr, detail=0, paragraph=True)
            text = "\n".join(lines)
            dt = time.time() - t0

            if pno not in done:
                pf_pages.write(json.dumps({"page": pno, "text": text}, ensure_ascii=False) + "\n")
                pf_pages.flush()
                done.add(pno)
                pages_added += 1

            if pno not in chunked_pages:
                chunks = chunk_text(text)
                metas = []
                texts = []
                for ci, c in enumerate(chunks):
                    rec = {"page": pno, "chunk_id": ci, "text": c, "source": name}
                    pf_chunks.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    texts.append(c)
                    metas.append({"source": name, "page": pno, "type": "knowledge"})
                pf_chunks.flush()
                if texts and idx is not None:
                    try:
                        added = idx.add_texts(texts, metas)
                        if added:
                            chunks_added += added
                    except Exception as e:
                        log(f"[pdf] 第{pno}页写入向量库失败（离线语料仍保留）：{e}")
                chunked_pages.add(pno)

            if (pno + 1) % 10 == 0:
                log(f"[pdf] 进度 {pno+1}/{n}  本批OCR耗时 {dt:.1f}s/页")
    finally:
        pf_pages.close()
        pf_chunks.close()
        doc.close()

    log(f"[pdf] 完成：新增OCR页数={pages_added}，新增向量块={chunks_added}")
    log(f"[pdf] 离线语料：{chunks_jsonl}")


def main() -> None:
    ap = argparse.ArgumentParser(description="FactorGPT 知识库摄入")
    ap.add_argument("--xlsx", default=None, help="因子字典 xlsx 路径")
    ap.add_argument("--pdf", default=None, help="因子日历 PDF 路径")
    ap.add_argument("--xlsx-source", default="因子字典_含代码", help="xlsx 来源标记")
    ap.add_argument("--skip-xlsx", action="store_true", help="跳过 xlsx 导入")
    ap.add_argument("--limit", type=int, default=None, help="PDF 仅处理前 N 页（测试用）")
    args = ap.parse_args()

    os.chdir(str(ROOT))
    if not args.skip_xlsx and args.xlsx:
        ingest_xlsx(args.xlsx, args.xlsx_source)
    if args.pdf:
        ingest_pdf(args.pdf, limit=args.limit)
    log("全部摄入步骤结束。")


if __name__ == "__main__":
    main()
