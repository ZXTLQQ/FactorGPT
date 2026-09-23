"""构建原生热核（``native/fg_kernels.cpp``）。

两种产物，同一份源码：

1. **动态库**（默认，``build/native/``）：任何 C++ 编译器都能编，之后由
   ``src/mining/native_kernels.py`` 用 ctypes 加载。**本机验证走这条路**——
   它不依赖 CPython ABI，所以即使机器上没有 MSVC（只有 zig/clang/gcc）也能
   把 C++ 侧的数值正确性验掉。
2. **pybind11 扩展模块**（``--python``）：需要能把扩展链到 CPython 的编译器
   （Windows 上即 MSVC），产物更快（少一次数组转换），CI 与打包 wheel 走这条。

编译器优先级：``MSVC (cl)`` → ``clang++`` → ``g++`` → ``zig c++``（pip 安装即可，
自带 clang，无需管理员权限）。选 zig 是因为"没有 VS 的 Windows 机器"是常态，
而 Python 扩展的编译验证不该被卡在这里。

    python scripts/build_native.py                 # 编动态库到 build/native/
    python scripts/build_native.py --python        # 再编 pybind11 扩展（需 MSVC/gcc）
    python scripts/build_native.py --print-cmd     # 只看将要执行的命令
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "native" / "fg_kernels.cpp"
BINDINGS = ROOT / "native" / "fg_bindings.cpp"
OUT_DIR = ROOT / "build" / "native"


def _detect() -> tuple[list[str], str]:
    """返回 ``(命令前缀, 名字)``；找不到就抛错。"""
    if shutil.which("cl"):
        return (["cl", "/O2", "/std:c++17", "/LD"], "msvc")
    for exe in ("clang++", "g++"):
        if shutil.which(exe):
            return ([exe, "-O2", "-std=c++17", "-shared", "-fPIC"], exe)
    if _zig_available():
        # zig 以 Python 包形式提供：``python -m ziglang c++``，无需管理员权限安装
        return ([sys.executable, "-m", "ziglang", "c++", "-O2", "-std=c++17",
                 "-shared"], "zig")
    raise SystemExit(
        "没有可用的 C++ 编译器。Windows 装 VS Build Tools，或 pip install ziglang "
        "（自带 clang，无需管理员权限）；Linux 装 g++/clang++ 即可。")


def _zig_available() -> bool:
    try:
        import ziglang  # type: ignore  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _run(cmd: list[str], dry: bool) -> int:
    printable = " ".join(cmd)
    print(f"[build] {printable}")
    if dry:
        return 0
    try:
        return subprocess.run(cmd, cwd=str(ROOT), check=False).returncode
    except FileNotFoundError:  # pragma: no cover - 环境缺编译器
        print(f"[build] 找不到命令：{cmd[0]}")
        return 127


def _lib_name() -> str:
    sysname = platform.system().lower()
    if sysname == "windows":
        return "fg_kernels.dll"
    if sysname == "darwin":
        return "libfg_kernels.dylib"
    return "libfg_kernels.so"


def build_dll(dry: bool = False) -> Path:
    prefix, who = _detect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / _lib_name()
    if who == "msvc":
        cmd = [*prefix, f"/Fe:{out}", str(SRC)]
        # MSVC 需要显式导出：源码里已用 __declspec(dllexport)，无需 .def
        # /LD 会连带生成 .lib/.exp，忽略即可
    else:
        target = ["-target", "x86_64-windows-gnu"] if (
            who == "zig" and platform.system().lower() == "windows") else []
        cmd = [*prefix, *target, "-o", str(out), str(SRC)]
    rc = _run(cmd, dry)
    if rc != 0:
        raise SystemExit(f"[build] 动态库编译失败（退出码 {rc}）")
    print(f"[build] 编译器 {who} → {out}")
    return out


def build_python_module(dry: bool = False) -> int:
    """编 pybind11 扩展：要求 setuptools + pybind11 + 与 CPython ABI 匹配的编译器。"""
    try:
        import pybind11  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"缺少 pybind11：{e}（pip install pybind11）")
    setup = ROOT / "native" / "setup_native.py"
    cmd = [sys.executable, str(setup), "build_ext", "--inplace",
           f"--build-lib={OUT_DIR}"]
    env = os.environ.copy()
    inc = str(Path(pybind11.get_include()))
    env.setdefault("FG_PYBIND_INCLUDE", inc)
    print(f"[build] pybind11 include: {inc}")
    return _run(cmd, dry)


def main() -> int:
    ap = argparse.ArgumentParser(description="构建 FactorGPT 原生热核")
    ap.add_argument("--python", action="store_true", help="额外构建 pybind11 扩展")
    ap.add_argument("--print-cmd", action="store_true", help="只打印命令不执行")
    args = ap.parse_args()
    if not SRC.exists():  # pragma: no cover
        raise SystemExit(f"源码不存在：{SRC}")
    out = build_dll(dry=args.print_cmd)
    if args.python:
        rc = build_python_module(dry=args.print_cmd)
        if rc != 0:
            return rc
    print(f"[done] {out}；运行期用 mining.native_kernels.backend() 查看是否生效")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
