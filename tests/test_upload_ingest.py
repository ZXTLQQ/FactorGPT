"""上传管道：文件名清洗与类型分派。

回归重点：长标题（论文 PDF 常见 90+ 字符）在清洗落盘时**不能把扩展名截掉**，
否则 Path.suffix 为空，后面按扩展名分派直接抛「不支持的文件类型：」（扩展名还是空的）。
"""
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.upload_ingest import (  # noqa: E402
    DEFAULT_CONTEXT_CHARS,
    SUPPORTED_EXTS,
    UploadedItem,
    UploadIngestor,
    _safe_name,
    _slice_pages,
    detect_chapters,
)

LONG_PDF = ("A KL Lens on Quantization Fast, Forward-Only Sensitivity "
            "for Mixed-Precision SSM-Transformer Models.pdf")


def _pdf_item(tmp_path, pages, chapters=None) -> UploadedItem:
    """造一份"已解析好的 PDF"：只测上下文切片，不真去解析 PDF。"""
    it = UploadedItem(name="p.pdf", path=str(tmp_path / "p.pdf"), kind="text",
                      size=1, text="\n".join(pages), pages=list(pages))
    if chapters is not None:
        it.meta["chapters"] = chapters
    return it


class TestSafeName:
    def test_long_title_keeps_extension(self):
        safe = _safe_name(LONG_PDF)
        assert Path(safe).suffix == ".pdf", f"扩展名被截掉了：{safe}"
        assert safe.endswith(".pdf")

    @pytest.mark.parametrize("name,ext", [
        ("research.pdf", ".pdf"),
        ("截屏 2024-01-02.png", ".png"),
        ("f.csv", ".csv"),
        ("a.b.xlsx", ".xlsx"),
        ("no_ext", ""),
        ("", ""),
    ])
    def test_extension_always_preserved(self, name, ext):
        assert Path(_safe_name(name)).suffix == ext
        assert ".." not in _safe_name(name).replace("...", "")  # 不产生怪异多点后缀

    def test_strips_path_and_unsafe_chars(self):
        safe = _safe_name("../../etc/passwd;rm -rf.pdf")
        assert "/" not in safe and "\\" not in safe
        assert Path(safe).suffix == ".pdf"

    def test_length_bounded(self):
        # 总长仍受 80 字符约束，但预算里给扩展名留了位置（主干让位给后缀）
        assert len(_safe_name("x" * 500 + ".pdf")) == 80


class TestIngestFile:
    def _ingestor(self, tmp_path) -> UploadIngestor:
        cfg = {"data": {"uploads": {
            "dir": str(tmp_path / "uploads"),
            "index_file": str(tmp_path / "uploads" / "index.json"),
            "max_files": 5,
            "parse_images": False,
        }}}
        return UploadIngestor(cfg)

    def test_long_pdf_title_ingests_as_text(self, tmp_path):
        """真机回归：这篇论文标题清洗后曾丢掉 .pdf，被当成未知类型拒绝。"""
        ing = self._ingestor(tmp_path)
        item = ing.ingest_bytes(LONG_PDF, b"%PDF-1.4\n% not a real pdf\n")
        assert item.kind == "text"
        assert Path(item.path).suffix == ".pdf"
        assert item.meta.get("ext") == ".pdf"

    def test_unsupported_ext_reports_it(self, tmp_path):
        ing = self._ingestor(tmp_path)
        with pytest.raises(ValueError, match=r"不支持的文件类型") as ei:
            ing.ingest_bytes("notes.docx", b"whatever")
        # 报错要能看出是哪个类型（旧版连扩展名都是空的）
        assert "docx" in str(ei.value) or "(无扩展名)" in str(ei.value)

    def test_source_name_is_the_dedupe_key(self, tmp_path):
        """落盘名带时间戳+摘要，每次都不同；去重要靠原始文件名。"""
        ing = self._ingestor(tmp_path)
        item = ing.ingest_bytes(LONG_PDF, b"%PDF-1.4\n")
        assert item.source_name == LONG_PDF
        assert item.source_name != Path(item.path).name

    def test_supported_exts_are_known(self):
        assert ".pdf" in SUPPORTED_EXTS and ".png" in SUPPORTED_EXTS

    def test_pdf_keeps_pages_and_toc(self, tmp_path, monkeypatch):
        """PDF 必须逐页保留：全文概括要按页切片，压成一个字符串就切不动了。"""
        ing = self._ingestor(tmp_path)
        pages = ["1 Introduction\n正文", "2 Method\n正文", "3 Conclusion\n结论"]
        monkeypatch.setattr(
            ing._parser, "parse_file",
            lambda _p: (pd.DataFrame({"text": pages, "page": range(1, len(pages) + 1)}), {}))
        item = ing.ingest_bytes("paper.pdf", b"%PDF-1.4")
        assert item.pages == pages and item.meta["pages_n"] == 3
        assert "pages" not in item.meta          # 全文副本不能进 meta（会落 index.json）
        assert item.meta["chapters"][0][0] == "1 Introduction"
        ctx = ing.context_text(per_item=10 ** 6)
        assert "— 第 1/3 页 —" in ctx and "— 第 3/3 页 —" in ctx


# ----------------------------------------------------------------------
# 长文上下文：预算 + 按页切片（"全文概括" 之前只有开头 4000 字）
# ----------------------------------------------------------------------
class TestPageSlicing:
    def _ing(self, tmp_path, items) -> UploadIngestor:
        ing = TestIngestFile()._ingestor(tmp_path)
        ing.items = list(items)
        return ing

    def test_budget_enough_keeps_every_page(self, tmp_path):
        pages = [f"第{i}页正文" * 10 for i in range(1, 11)]
        ing = self._ing(tmp_path, [_pdf_item(tmp_path, pages)])
        ctx = ing.context_text(per_item=10 ** 6)
        assert "第1页正文" in ctx and "第10页正文" in ctx
        assert "省略中间" not in ctx

    def test_over_budget_keeps_head_and_tail_and_says_so(self, tmp_path):
        pages = [f"P{i}-" + "字" * 200 for i in range(1, 13)]
        ing = self._ing(tmp_path, [_pdf_item(tmp_path, pages)])
        ctx = ing.context_text(per_item=1200)
        assert "P1-" in ctx                      # 开头保留
        assert "P12-" in ctx                     # 结尾保留（结论常在这）
        assert "P7-" not in ctx                  # 中段真的被省掉了
        assert "省略中间" in ctx and "/12 页" in ctx   # 页码标记与省略说明都在

    def test_slice_pages_reports_omitted_pages(self):
        pages = ["a" * 100 for _ in range(10)]
        chunks, note = _slice_pages(pages, 600)
        assert [no for no, _ in chunks] == [1, 2, 3, 9, 10]
        assert "省略中间 5 页" in note

    def test_slice_pages_single_huge_page_still_keeps_last(self):
        # 单页就超过尾部预算：宁可略微超一点也要留住最后一页
        pages = ["a" * 5000 for _ in range(3)]
        chunks, note = _slice_pages(pages, 1000)
        assert 3 in [no for no, _ in chunks]
        assert "省略中间" in note

    def test_plain_text_head_tail(self, tmp_path):
        it = UploadedItem(name="n.txt", path="n.txt", kind="text", size=1,
                          text="开头" + "中" * 5000 + "结尾")
        ing = self._ing(tmp_path, [it])
        body, note = ing.fit_text(it, 600)
        assert body.startswith("开头") and body.rstrip().endswith("结尾")
        assert "省略中间" in note

    def test_default_budget_is_raised(self, tmp_path):
        # 旧默认 4000 字 = 长论文只看开头，必须已经调大
        assert DEFAULT_CONTEXT_CHARS >= 20000
        ing = TestIngestFile()._ingestor(tmp_path)
        assert ing.context_max_chars == DEFAULT_CONTEXT_CHARS


class TestChapters:
    def test_detects_numbered_and_chinese_titles(self):
        pages = ["Abstract\nblah", "1 Introduction\nblah",
                 "第2章 方法\nblah", "正文里不该被当成标题的一句话"]
        got = detect_chapters(pages)
        assert ("Abstract", 1) in got
        assert ("1 Introduction", 2) in got
        assert any(t.startswith("第2章") and p == 3 for t, p in got)

    def test_toc_goes_into_context(self, tmp_path):
        pages = ["1 Introduction\n正文"] + ["内容"] * 8
        it = _pdf_item(tmp_path, pages, chapters=[["1 Introduction", 1]])
        ing = TestIngestFile()._ingestor(tmp_path)
        ing.items = [it]
        assert "1 Introduction(p1)" in ing.context_text(per_item=10 ** 6)

    def test_pages_not_written_to_index_json(self, tmp_path):
        it = _pdf_item(tmp_path, ["x" * 1000])
        d = it.to_json()
        assert "pages" not in d and d["n_pages"] == 1
