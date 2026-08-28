# -*- coding: utf-8 -*-
"""删除因 PowerShell 中文编码问题而冗余的中文入口脚本（ASCII 版已存在）。"""
import os

BASE = r"E:/Qlib/qlib_factorgpt"
targets = ["每日因子挖掘.py", "每日背景采集.py"]
for name in targets:
    p = os.path.join(BASE, name)
    if os.path.exists(p):
        os.remove(p)
        print("REMOVED:", name)
    else:
        print("NOT FOUND (skip):", name)

# 剩余中文命名的 .py 检查
print("\nRemaining .py at top:")
for n in sorted(os.listdir(BASE)):
    if n.endswith(".py"):
        print("  ", n)
