"""上传管道：文件名清洗与类型分派。

回归重点：长标题（论文 PDF 常见 90+ 字符）在清洗落盘时**不能把扩展名截掉**，
否则 Path.suffix 为空，后面按扩展名分派直接抛「不支持的文件类型：」（扩展名还是空的）。
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engine.upload_ingest import SUPPORTED_EXTS, UploadIngestor, _safe_name  # noqa: E402

LONG_PDF = ("A KL Lens on Quantization Fast, Forward-Only Sensitivity "
            "for Mixed-Precision SSM-Transformer Models.pdf")


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
