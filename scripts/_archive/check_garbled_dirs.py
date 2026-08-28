# -*- coding: utf-8 -*-
"""检查 qlib_factorgpt 下是否存在因 PowerShell 中文编码产生的乱码目录，并清理空目录。"""
import os

BASE = r"E:/Qlib/qlib_factorgpt"
garbled = []
for name in os.listdir(BASE):
    p = os.path.join(BASE, name)
    if not os.path.isdir(p):
        continue
    # 乱码目录的特征：包含 '鍥' 或 '搴' 等 GBK 误码字符
    if any(c in name for c in ["鍥", "搴", "鎸", "妗", "鏂", "鑳", "栤", "闈"]):
        items = os.listdir(p)
        garbled.append((name, items, len(items)))
        if not items:
            os.rmdir(p)
            print("REMOVED empty garbled dir:", repr(name))
        else:
            print("KEPT non-empty garbled dir:", repr(name), "->", items)

print("\nSummary:")
for n, it, sz in garbled:
    print("  ", repr(n), "size=", sz)
print("garbled dir count:", len(garbled))
