"""打印原生热核的**为什么是这个后端**——跨平台，一行命令。

    python scripts/native_diag.py

存在的原因很实际：CI 红在"backend 不是 pybind11"时，真正想知道的是这台机器上
到底有什么编译器、动态库有没有编出来、扩展模块为什么 import 失败。这三件事散在
三处，靠肉眼判断一次要跑一轮 CI；这里是一次打全。

之所以单独做脚本而不是在 workflow 里写 heredoc：Windows runner 的默认 shell 是
pwsh，**不支持** ``<<PY`` 这种写法——写进去就是"只在 Linux 腿上能跑"的 CI。
"""
from __future__ import annotations

import os
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _probe_toolchain() -> None:
    for exe in ("cl", "clang++", "g++", "cc", "zig"):
        print(f"  {exe:8s} -> {shutil.which(exe) or '(not on PATH)'}")
    for key in ("FG_NATIVE_DLL", "FG_DISABLE_NATIVE", "FG_REQUIRE_NATIVE"):
        val = os.environ.get(key)
        print(f"  {key:18s} = {val if val else '(unset)'}")


def _probe_artifacts() -> None:
    hits = list(ROOT.glob("fg_native*"))
    build_dir = ROOT / "build" / "native"
    if build_dir.is_dir():
        hits += list(build_dir.glob("*"))
    for p in sorted(hits):
        print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size / 1024:.0f} KB)")
    if not hits:
        print("  (无构建产物)")


def main() -> int:
    print("== 工具链 / 环境变量 ==")
    _probe_toolchain()
    print("== 构建产物 ==")
    _probe_artifacts()
    print("== import fg_native ==")
    try:
        import fg_native  # type: ignore

        print(f"  ok: {getattr(fg_native, '__file__', '?')}")
    except Exception:
        print("  失败：")
        traceback.print_exc()
    print("== mining.native_kernels.backend() ==")
    try:
        from mining import native_kernels as NK

        print(f"  {NK.backend()}")
    except Exception:
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
