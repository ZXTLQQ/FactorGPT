# -*- coding: utf-8 -*-
"""用 Python 内部 API 探查因子库结构，规避 shell 中文编码问题。"""
import os
import json

BASE = r"E:/Qlib/qlib_factorgpt/因子库"

def tree(root, prefix="", maxdepth=2):
    if maxdepth < 0:
        return
    try:
        items = sorted(os.listdir(root))
    except Exception as e:  # noqa: BLE001
        print(prefix + "[ERR] " + repr(e))
        return
    for it in items:
        p = os.path.join(root, it)
        if os.path.isdir(p):
            print(prefix + "DIR  " + it + "/")
            tree(p, prefix + "     ", maxdepth - 1)
        else:
            sz = os.path.getsize(p)
            print(prefix + "FILE " + it + "  (" + str(sz) + " B)")

print("=== 因子库 顶层结构 ===")
tree(BASE, maxdepth=2)

print("\n=== registry.json ===")
reg = os.path.join(BASE, "registry.json")
if os.path.exists(reg):
    with open(reg, "r", encoding="utf-8") as f:
        print(json.dumps(json.load(f), ensure_ascii=False, indent=2))
else:
    print("NOT FOUND")

# 检查乱码目录是否已清理
print("\n=== qlib_factorgpt 顶层（清理检查） ===")
top = r"E:/Qlib/qlib_factorgpt"
for n in sorted(os.listdir(top)):
    p = os.path.join(top, n)
    print(("DIR " if os.path.isdir(p) else "FILE") + "  " + n)
