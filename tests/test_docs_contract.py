"""文档-代码契约测试：防止 README 声明与实现漂移回归。

每当实现变化（如新增 UI 页面、调整因子库、重命名数据列）而文档未同步时，
这些测试会立即失败，把「文档漂移」从人工巡检变成 CI 自动拦截。
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UI_PAGE_COUNT = 20
FACTOR_COUNT = 62


def _read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def test_ui_page_count_matches_readme():
    nav = _read(os.path.join("src", "ui", "nav.py"))
    keys = re.findall(r'"key":\s*"([a-z_]+)"', nav)
    assert len(keys) == UI_PAGE_COUNT, (
        f"src/ui/nav.py 应含 {UI_PAGE_COUNT} 个页面 key，当前 {len(keys)}")
    readme = _read("README.md")
    assert re.search(r"20[- ]?page|20\s*页", readme, re.IGNORECASE), \
        "README 未声明 20 页界面，需同步更新"


def test_factor_count_matches_docs():
    src = _read(os.path.join("src", "engine", "traditional_factors.py"))
    n = len(re.findall(r"category=CATEGORY_[A-Z_]+", src))
    assert n == FACTOR_COUNT, (
        f"traditional_factors.py 应含 {FACTOR_COUNT} 个因子，当前 {n}")
    for rel in ("README.md", os.path.join("hf_space", "README.md")):
        doc = _read(rel)
        assert re.search(r"62\s*(built-in|个|传统)?", doc, re.IGNORECASE), \
            f"{rel} 未声明 62 个传统因子，需同步更新"


def test_offline_schema_contract():
    """离线 parquet 物理列 instrument → 对外契约 symbol 的桥接必须保持。"""
    meta = json.loads(_read(os.path.join("data", "offline", "meta.json")))
    assert "instrument" in meta.get("columns", []), \
        "meta.json 的物理列名必须包含 instrument（供适配层重命名）"
    adapter = _read(os.path.join("src", "data", "offline_adapter.py"))
    assert "instrument" in adapter and "symbol" in adapter, \
        "offline_adapter.py 必须处理 instrument→symbol 重命名"
    assert "_de_norm_symbol" in adapter, \
        "offline_adapter.py 未执行 instrument→symbol 去规范化桥接，契约断裂"


def test_config_kronos_fallback_enabled():
    """kronos 模型不可用时必须优雅降级（fallback_to_stub=true）。"""
    cfg = _read("config.yaml")
    m = re.search(r"fallback_to_stub\s*:\s*(\w+)", cfg)
    assert m and m.group(1).lower() == "true", \
        "config.yaml 中 kronos.fallback_to_stub 应为 true（默认降级）"


ARCHIVED_ONESHOT = {
    "check_export_status.py", "check_garbled_dirs.py", "cleanup_entry.py",
    "cleanup_temp.py", "compile_all.py", "debug_ths_api.py",
    "export_offline_data.py", "final_check.py", "finalize_offline_data.py",
    "normalize_registry.py", "probe_library.py", "probe_ths_endpoint.py",
    "verify_miner_load.py",
}


def test_no_debug_residue_in_scripts():
    """一次性调试脚本必须归档到 scripts/_archive/，不得停留在顶层。"""
    script_dir = os.path.join(ROOT, "scripts")
    top_level = {f for f in os.listdir(script_dir) if f.endswith(".py")}
    residue = sorted(ARCHIVED_ONESHOT & top_level)
    assert not residue, f"scripts/ 顶层仍存在应归档的一次性脚本: {residue}"
    assert os.path.isdir(os.path.join(script_dir, "_archive")), \
        "scripts/_archive/ 归档目录不存在"


def test_no_debug_residue_in_root():
    """仓库根目录不得出现调试/临时脚本。"""
    root_files = {f for f in os.listdir(ROOT) if os.path.isfile(os.path.join(ROOT, f))}
    residue = [f for f in root_files
               if re.match(r"^(_smoke|norm_test|sina_probe|verify_mh|test_agent_quick|启动)", f)]
    assert not residue, f"仓库根目录存在调试残留: {residue}"
