# -*- coding: utf-8 -*-
"""编译 qlib_factorgpt 全部修改过的脚本，确认语法无误。"""
import py_compile
import os

BASE = r"E:/Qlib/qlib_factorgpt"
targets = [
    os.path.join(BASE, "data_bridge.py"),
    os.path.join(BASE, "daily_background.py"),
    os.path.join(BASE, "daily_mining.py"),
    os.path.join(BASE, "因子库", "tools", "miner.py"),
    os.path.join(BASE, "因子库", "tools", "library.py"),
    os.path.join(BASE, "因子库", "tools", "report_docx.py"),
]
ok, fail = [], []
for t in targets:
    try:
        py_compile.compile(t, doraise=True)
        ok.append(t)
    except Exception as e:  # noqa: BLE001
        fail.append((t, repr(e)[:160]))
print("COMPILE_OK", len(ok))
for t in ok:
    print("  OK", os.path.relpath(t, BASE))
print("COMPILE_FAIL", len(fail))
for t, e in fail:
    print("  FAIL", os.path.relpath(t, BASE), e)
