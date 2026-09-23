"""构建 pybind11 扩展 ``fg_native``（可选加速，失败不影响主流程）。

    python native/setup_native.py build_ext --inplace

设计约束：**这一层必须是可选的**。``pip install`` 在没有编译器的机器上应当照常
成功（那时走 pandas 兜底），因此仓库的 requirements 里没有它，CI 里才强制验证。

ABI 提醒：Windows 上必须用与 CPython 相同的工具链（MSVC 14.x）；用 MinGW/zig 编
出来的动态库只能走 ``ctypes`` 路径（见 ``scripts/build_native.py``），不能直接作为
Python 扩展 import。这不是保守，而是 CPython 在 Windows 上的 ABI 事实。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    from setuptools import Extension, setup
except ImportError:  # pragma: no cover
    raise SystemExit("需要 setuptools")

try:
    import pybind11
except ImportError:  # pragma: no cover
    raise SystemExit("需要 pybind11：pip install pybind11")

includes = [str(Path(pybind11.get_include())),
            str(Path(pybind11.get_include(user=True)))]
extra = str(os.environ.get("FG_PYBIND_INCLUDE") or "").strip()
if extra and extra not in includes:
    includes.append(extra)

if sys.platform.startswith("win"):
    flags = ["/O2", "/std:c++17"]
else:
    flags = ["-O2", "-std=c++17", "-fvisibility=hidden"]

HERE = Path(__file__).resolve().parent


ext = Extension(
    name="fg_native",
    # 绝对路径：本脚本由 scripts/build_native.py 以仓库根为 cwd 调用，
    # 相对文件名会被解析到仓库根而不是 native/。
    sources=[str(HERE / "fg_kernels.cpp"), str(HERE / "fg_bindings.cpp")],
    include_dirs=includes,
    extra_compile_args=flags,
    language="c++",
)

setup(
    name="factorgpt-native",
    version="0.1.0",
    description="FactorGPT 原生横截面热核（可选加速）",
    ext_modules=[ext],
    zip_safe=False,
)
