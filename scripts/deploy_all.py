#!/usr/bin/env python3
"""
FactorGPT 一键部署脚本
-----------------------
完成三件事：
1. Git Push（SSH 密钥注册引导）
2. GitHub Topics 标签设置
3. HuggingFace Spaces 在线 Demo 部署

用法: python scripts/deploy_all.py
"""

import subprocess
import sys
import os
import json
import webbrowser
import urllib.request
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SSH_PUBKEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIkEGZLJOit4uist4MG2z+k1ppybs16JZzp/VrYY1HVg factorgpt@github"

GITHUB_REPO = "ZXTLQQ/FactorGPT"
GITHUB_OWNER = "ZXTLQQ"
GITHUB_REPO_NAME = "FactorGPT"

HF_SPACE_NAME = "factorgpt-demo"
HF_USERNAME = "ZXTLQQ"

GITHUB_TOPICS = [
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
    "natural-language-processing"
]


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def run(cmd, cwd=None, shell=True):
    """Run a shell command and return output."""
    result = subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, shell=shell,
                           capture_output=True, text=True)
    return result


# ================================================================
# STEP 1: Git Push via SSH
# ================================================================
def step1_git_push():
    section("STEP 1/3: Git Push to GitHub via SSH")
    
    # Check if remote is SSH
    result = run("git remote get-url origin")
    if "git@github.com" not in result.stdout:
        print("[*] Switching remote to SSH...")
        run(f"git remote set-url origin git@github.com:{GITHUB_REPO}.git")
        print(f"[✓] Remote set to: git@github.com:{GITHUB_REPO}.git")
    
    # Test SSH connection
    print("[*] Testing SSH connection to GitHub...")
    result = run("ssh -o StrictHostKeyChecking=accept-new -T git@github.com")
    
    if "successfully authenticated" in result.stderr:
        print("[✓] SSH to GitHub works! Pushing code...")
        result = run("git push origin main")
        if result.returncode == 0:
            print("[✓] Code pushed successfully!")
            return True
        else:
            print(f"[✗] Push failed:\n{result.stderr}")
            return False
    else:
        print(f"[!] SSH authentication failed. You need to add your public key to GitHub.")
        print(f"\n    PUBLIC KEY (already generated):\n    {SSH_PUBKEY}")
        print(f"\n    → Step 1: Copy the key above")
        print(f"    → Step 2: Go to https://github.com/settings/keys")
        print(f"    → Step 3: Click 'New SSH Key'")
        print(f"    → Step 4: Paste and save")
        print(f"\n    Or open directly:")
        try:
            webbrowser.open("https://github.com/settings/keys")
        except:
            pass
        
        input("\n    Press ENTER after you've added the SSH key to GitHub...")
        
        # Retry
        print("[*] Retrying SSH connection...")
        result = run("ssh -T git@github.com")
        if "successfully authenticated" in result.stderr:
            print("[✓] SSH works now! Pushing code...")
            result = run("git push origin main")
            if result.returncode == 0:
                print("[✓] Code pushed successfully!")
                return True
        
        print("[✗] Push still failed. Skipping this step - you can push later with `git push origin main`")
        return False


# ================================================================
# STEP 2: GitHub Topics
# ================================================================
def step2_github_topics():
    section("STEP 2/3: Set GitHub Repository Topics")
    
    print(f"[*] Topics to set ({len(GITHUB_TOPICS)} total):")
    for t in GITHUB_TOPICS:
        print(f"    • {t}")
    
    token = os.environ.get("GITHUB_TOKEN", os.environ.get("GH_TOKEN", ""))
    if not token:
        print(f"\n[!] No GITHUB_TOKEN environment variable found.")
        print(f"    To set topics automatically, create a token at:")
        print(f"    https://github.com/settings/tokens")
        print(f"    (needs 'repo' scope)")
        print(f"\n    Then set it: set GITHUB_TOKEN=ghp_xxxxxxxxxxxx")
        print(f"\n    Alternatively, set topics manually at:")
        print(f"    https://github.com/{GITHUB_REPO}")
        print(f"    (click the gear icon next to About → Topics)")
        
        use_manual = input("\n    Press 1 to open the repo page, or ENTER to skip: ").strip()
        if use_manual == "1":
            webbrowser.open(f"https://github.com/{GITHUB_REPO}")
        return False
    
    # Use GitHub API
    print(f"[*] Setting topics via GitHub API...")
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/topics"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.mercy-preview+json",
            "Content-Type": "application/json"
        }
        data = json.dumps({"names": GITHUB_TOPICS}).encode()
        
        req = urllib.request.Request(url, data=data, headers=headers, method="PUT")
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
        print(f"[✓] Topics set: {result.get('names', [])}")
        return True
    except Exception as e:
        print(f"[✗] Failed to set topics: {e}")
        print(f"    Set manually at: https://github.com/{GITHUB_REPO}")
        return False


# ================================================================
# STEP 3: HuggingFace Spaces
# ================================================================
def step3_hf_spaces():
    section("STEP 3/3: Deploy HuggingFace Spaces Demo")
    
    # Check hf CLI
    result = run("hf --version")
    if result.returncode != 0:
        print("[!] HuggingFace CLI not found. Installing...")
        run("pip install -U huggingface_hub")
    
    # Check login
    result = run("hf auth whoami")
    if "Not logged in" in result.stderr or result.returncode != 0:
        print("[!] Not logged into HuggingFace.")
        print("    → Step 1: Get your token at https://huggingface.co/settings/tokens")
        print("    → Step 2: Create a token with 'write' scope")
        print(f"    → Step 3: Run: hf auth login")
        print(f"\n    Opening token page...")
        try:
            webbrowser.open("https://huggingface.co/settings/tokens")
        except:
            pass
        
        token = input("\n    Paste your HuggingFace token (or ENTER to skip): ").strip()
        if token:
            result = run(f'hf auth login --token "{token}"')
            if result.returncode != 0:
                print(f"[✗] Login failed: {result.stderr}")
                return False
            print("[✓] Logged into HuggingFace!")
        else:
            print("[!] Skipping HF Spaces deployment.")
            return False
    
    # Confirm username
    result = run("hf auth whoami")
    username = result.stdout.strip()
    if not username:
        print("[!] Could not determine HF username.")
        return False
    print(f"[✓] Logged in as: {username}")
    
    # Check if Space already exists
    print(f"\n[*] Checking if space '{username}/{HF_SPACE_NAME}' exists...")
    
    # Build demo folder path
    demo_dir = PROJECT_ROOT / "demo"
    if not demo_dir.exists():
        print(f"[✗] Demo directory not found at: {demo_dir}")
        return False
    
    print(f"[*] Demo files in: {demo_dir}")
    for f in demo_dir.iterdir():
        print(f"    • {f.name}")
    
    # Create or update Space
    space_id = f"{username}/{HF_SPACE_NAME}"
    print(f"\n[*] Deploying to: https://huggingface.co/spaces/{space_id}")
    
    # Use huggingface_hub Python API for more control
    create_space_script = f"""
import sys
from huggingface_hub import HfApi, create_repo, upload_folder
from pathlib import Path

api = HfApi()
username = "{username}"
space_name = "{HF_SPACE_NAME}"
space_id = f"{username}/{space_name}"
demo_dir = Path(r"{demo_dir}")

# Check if space exists
try:
    space_info = api.repo_info(repo_id=space_id, repo_type="space")
    print(f"[*] Space exists: {{space_info.url}}")
    exists = True
except:
    print(f"[*] Space does not exist yet, creating...")
    exists = False

if not exists:
    try:
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="streamlit",
            private=False
        )
        print(f"[+] Space created: https://huggingface.co/spaces/{{space_id}}")
    except Exception as e:
        print(f"Error creating space: {{e}}")
        sys.exit(1)

# Upload files
print(f"[*] Uploading demo files...")
upload_folder(
    folder_path=demo_dir,
    repo_id=space_id,
    repo_type="space",
    commit_message="Deploy FactorGPT demo v1.0"
)
print(f"[+] All files uploaded!")
print(f"[+] Demo live at: https://huggingface.co/spaces/{{space_id}}")
print(f"[+] Give it ~2-5 minutes to build and start.")
"""
    
    with open(PROJECT_ROOT / "scripts" / "_hf_deploy.py", "w") as f:
        f.write(create_space_script)
    
    result = run(f'"{sys.executable}" "{PROJECT_ROOT / "scripts" / "_hf_deploy.py"}"')
    print(result.stdout)
    if result.returncode != 0:
        print(f"[✗] HF deploy error:\n{result.stderr}")
        return False
    
    # Cleanup temp script
    (PROJECT_ROOT / "scripts" / "_hf_deploy.py").unlink(missing_ok=True)
    
    print(f"\n[✓] Demo deployed: https://huggingface.co/spaces/{space_id}")
    print(f"    Note: First build may take 2-5 minutes.")
    try:
        webbrowser.open(f"https://huggingface.co/spaces/{space_id}")
    except:
        pass
    return True


# ================================================================
# Main
# ================================================================
def main():
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║     FactorGPT - 一键部署脚本                          ║
    ║     Git Push + GitHub Topics + HF Spaces Demo         ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    results = {}
    
    # Step 1: Git Push
    results["push"] = step1_git_push()
    
    # Step 2: GitHub Topics
    results["topics"] = step2_github_topics()
    
    # Step 3: HF Spaces
    results["spaces"] = step3_hf_spaces()
    
    # Summary
    section("SUMMARY")
    status_map = {True: "[✓] DONE", False: "[✗] SKIPPED/FAILED"}
    print(f"  Git Push:           {status_map.get(results['push'], '[!] Unknown')}")
    print(f"  GitHub Topics:      {status_map.get(results['topics'], '[!] Unknown')}")
    print(f"  HF Spaces Demo:     {status_map.get(results['spaces'], '[!] Unknown')}")
    
    if results["push"]:
        print(f"\n  Repository: https://github.com/{GITHUB_REPO}")
    if results["spaces"]:
        print(f"  Demo:       https://huggingface.co/spaces/{HF_USERNAME}/{HF_SPACE_NAME}")
    
    print(f"\n  All done! 🚀")


if __name__ == "__main__":
    main()
