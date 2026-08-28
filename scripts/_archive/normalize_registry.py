# -*- coding: utf-8 -*-
"""规范化 registry.json 中的路径字段（去掉 tools/.. 冗余段）。"""
import os
import json

REG = r"E:/Qlib/qlib_factorgpt/因子库/registry.json"
with open(REG, "r", encoding="utf-8") as f:
    reg = json.load(f)

changed = False
for x in reg["factors"]:
    d = x.get("dir")
    if d:
        nd = os.path.normpath(d)
        if nd != d:
            x["dir"] = nd
            changed = True
            print("normalized:", d, "->", nd)

if changed:
    with open(REG, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)
    print("registry updated.")
else:
    print("no change needed.")

# 校验归档目录确实存在
for x in reg["factors"]:
    print("check dir exists:", os.path.isdir(x["dir"]), x["dir"])
