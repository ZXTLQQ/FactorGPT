# -*- coding: utf-8 -*-
"""检查离线数据导出进度。"""
import os
import json

OUT = r"E:/Qlib/qlib_factorgpt/因子库/历史数据/2023-2026"
print("目录存在:", os.path.isdir(OUT))
if os.path.isdir(OUT):
    for fn in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, fn)
        if os.path.isfile(p):
            print(f"  FILE {fn}  {os.path.getsize(p)} B")
        else:
            sub = os.listdir(p)
            print(f"  DIR  {fn}  ({len(sub)} items)")
            if fn.startswith("parts_") and sub:
                for f in sorted(sub)[:6]:
                    fp = os.path.join(p, f)
                    print(f"     {f}  {os.path.getsize(fp)} B")
    meta = os.path.join(OUT, "meta.json")
    if os.path.exists(meta):
        with open(meta, "r", encoding="utf-8") as f:
            print("meta.json:", json.load(f))
    else:
        print("meta.json: 未生成")
