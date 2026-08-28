# -*- coding: utf-8 -*-
"""最终检查：历史数据目录结构 + README 内容 + 数据文件大小。"""
import os

OUT = r"E:/Qlib/qlib_factorgpt/因子库/历史数据/2023-2026"
print("=== 历史数据/2023-2026 目录 ===")
for fn in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, fn)
    if os.path.isfile(p):
        print(f"  FILE {fn}  {os.path.getsize(p)//1024} KB")
    else:
        n = len(os.listdir(p))
        print(f"  DIR  {fn}/  ({n} items)")

print("\n=== README.md ===")
rp = os.path.join(OUT, "README.md")
if os.path.exists(rp):
    print(open(rp, "r", encoding="utf-8").read())
