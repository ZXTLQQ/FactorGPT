#!/usr/bin/env python3
"""
FactorGPT → GitHub Topics 设置脚本
用法：
  set GITHUB_TOKEN=ghp_xxxxxxxxxxxx
  python scripts/set_github_topics.py
  
Token 创建：https://github.com/settings/tokens (勾选 'repo')
"""

import json
import os
import sys
import urllib.request

REPO = "ZXTLQQ/FactorGPT"

TOPICS = [
    "quantitative-finance",
    "alpha-factor",
    "factor-mining",
    "llm-agent",
    "langgraph",
    "a-share",
    "quant",
    "streamlit",
    "financial-ai",
    "backtesting",
    "genetic-programming",
    "multi-factor-model",
    "python",
    "deepseek",
    "ollama",
    "quantitative-trading",
    "natural-language-processing",
]


def main():
    token = os.environ.get("GITHUB_TOKEN", os.environ.get("GH_TOKEN", ""))
    if not token:
        print("[X] No GITHUB_TOKEN found.")
        print("    Set it:  set GITHUB_TOKEN=ghp_xxxxx")
        print("    Create:  https://github.com/settings/tokens (select 'repo' scope)")
        print(f"    Then:    python {__file__}")
        sys.exit(1)

    print(f"[*] Setting {len(TOPICS)} topics for {REPO}...")

    url = f"https://api.github.com/repos/{REPO}/topics"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.mercy-preview+json",
        "Content-Type": "application/json",
    }
    data = json.dumps({"names": TOPICS}).encode()

    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="PUT")
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
        names = result.get("names", [])
        print(f"[OK] Topics set successfully ({len(names)} topics):")
        for t in names:
            print(f"    - {t}")
    except urllib.error.HTTPError as e:
        print(f"[X] HTTP Error: {e.code} {e.reason}")
        body = e.read().decode()
        print(f"    {body}")
        sys.exit(1)
    except Exception as e:
        print(f"[X] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
