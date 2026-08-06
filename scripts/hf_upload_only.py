"""把 hf_space/index.html 部署到 Hugging Face 静态 Space。

Token 从环境变量 HF_TOKEN 或项目根目录 .env 读取（.env 已 gitignore）。
注意静态 Space 的访问域名是 *.static.hf.space。
"""
import os
import sys
import time

import requests
from huggingface_hub import HfApi


def _hf_token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


TOKEN = _hf_token()
if not TOKEN:
    print("缺少 HF_TOKEN：请设置环境变量或在 .env 中写入 HF_TOKEN=hf_xxx")
    sys.exit(1)
SPACE = "ZxTLQQ/factorgpt-demo"
APP = "https://zxtlqq-factorgpt-demo.static.hf.space"  # 静态 Space 必须走 .static.hf.space
api = HfApi(token=TOKEN)

readme = """---
title: FactorGPT
emoji: 📊
colorFrom: indigo
colorTo: purple
sdk: static
pinned: true
license: mit
---

# FactorGPT

AI-Driven Quantitative Factor Discovery Platform for A-Share Markets.

FactorGPT leverages LLM agents to automatically discover, evaluate, and select alpha factors from multi-source financial data.

Features: AI-powered factor generation, multi-source data integration, rigorous evaluation (IC analysis + backtesting), real-time computation, interactive Plotly visualizations, extensible plugin architecture.

[GitHub Repository](https://github.com/ZXTLQQ/FactorGPT)
"""

with open("hf_space/index.html", "r", encoding="utf-8") as f:
    html = f.read()

# Also create a .nojekyll file so GitHub Pages doesn't interfere
nojekyll = ""

print("[1] Upload readme...")
api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=SPACE, repo_type="space", commit_message="Update README")
print("OK")

print("[2] Upload index.html...")
api.upload_file(path_or_fileobj=html.encode(), path_in_repo="index.html", repo_id=SPACE, repo_type="space", commit_message="Deploy page")
print("OK")

print("[3] Upload .nojekyll...")
api.upload_file(path_or_fileobj=nojekyll.encode(), path_in_repo=".nojekyll", repo_id=SPACE, repo_type="space", commit_message="Add .nojekyll")
print("OK")

print("[4] List files...")
for f in api.list_repo_files(SPACE, repo_type="space"):
    print(f"  {f}")

print("\n[5] Wait and verify...")
time.sleep(30)
for attempt in range(3):
    try:
        r = requests.get(APP, timeout=20)
        ok = r.status_code == 200 and "FactorGPT" in r.text
        print(f"  Check {attempt+1}: HTTP {r.status_code}, {'LIVE' if ok else 'waiting'} ({len(r.text)} bytes)")
        if ok:
            print("\n  *** SITE IS LIVE! ***")
            break
    except Exception as e:
        print(f"  Check {attempt+1}: Error - {e}")
    time.sleep(10)

print(f"\nSpace: https://huggingface.co/spaces/{SPACE}")
print(f"App:   {APP}")
print("\n若仍 404：静态 Space 无 Factory Rebuild 按钮，每次上传会自动重建，稍等 1~2 分钟再访问。")
