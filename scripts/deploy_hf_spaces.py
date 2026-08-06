#!/usr/bin/env python3
"""
FactorGPT → HuggingFace Spaces 部署脚本
用法：
  1. hf auth login   （先登录 HuggingFace）
  2. python scripts/deploy_hf_spaces.py
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "demo"
HF_SPACE_NAME = "factorgpt-demo"


def main():
    print("[*] Checking setup...")

    # Check hf CLI
    import subprocess
    r = subprocess.run(["hf", "auth", "whoami"], capture_output=True, text=True)
    if r.returncode != 0:
        print("[!] Not logged into HuggingFace. Run: hf auth login")
        sys.exit(1)
    username = r.stdout.strip()
    print(f"[✓] Logged in as: {username}")

    # Check demo dir
    if not DEMO_DIR.exists():
        print(f"[✗] Demo dir not found: {DEMO_DIR}")
        sys.exit(1)
    print(f"[✓] Demo dir: {DEMO_DIR}")

    # Create or update Space
    from huggingface_hub import HfApi, upload_folder

    api = HfApi()
    space_id = f"{username}/{HF_SPACE_NAME}"

    # Check if exists
    try:
        api.repo_info(repo_id=space_id, repo_type="space")
        print(f"[*] Space exists: https://huggingface.co/spaces/{space_id}")
        exists = True
    except Exception:
        print(f"[*] Creating new space: {space_id}")
        exists = False

    if not exists:
        try:
            api.create_repo(
                repo_id=space_id,
                repo_type="space",
                space_sdk="streamlit",
                private=False,
            )
            print(f"[+] Space created!")
        except Exception as e:
            print(f"[✗] Create error: {e}")
            sys.exit(1)

    # Upload files
    print(f"[*] Uploading files...")
    upload_folder(
        folder_path=DEMO_DIR,
        repo_id=space_id,
        repo_type="space",
        commit_message="Deploy FactorGPT demo v1.0",
    )
    print(f"[+] Files uploaded!")

    # Done
    print(f"\n{'='*50}")
    print(f"[✓] DEMO LIVE: https://huggingface.co/spaces/{space_id}")
    print(f"[*] Build takes ~3-5 minutes on first run")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
