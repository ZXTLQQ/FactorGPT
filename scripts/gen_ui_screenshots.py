"""截取 FactorGPT Streamlit 真实 UI 页面（Playwright + chromium，headless）。

后台启动 streamlit，按「分组展开 → 点击页面」逐级导航，全页截图到 文档归档/docs/assets/ui_*.png。
"""

from __future__ import annotations

import io
import os
import sys
import time
import subprocess
import urllib.request

# 修复 Windows 控制台 GBK 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="ignore")

from playwright.sync_api import sync_playwright

PORT = 8512
URL = f"http://localhost:{PORT}"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSET_DIR = os.path.join(ROOT, "docs", "assets")
os.makedirs(ASSET_DIR, exist_ok=True)

# label -> (所属分组标题文本, 输出文件名)
PAGES = [
    ("系统概览", "工作台", "ui_overview.png"),
    ("系统因子库", "因子体系", "ui_library.png"),
    ("体系搭建", "因子体系", "ui_sysbuild.png"),
    ("因子精炼厂", "因子挖掘", "ui_refinery.png"),
    ("操作记忆", "工作台", "ui_memory.png"),
]


def wait_server(timeout: int = 90) -> bool:
    for _ in range(timeout):
        try:
            urllib.request.urlopen(URL, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def main():
    env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")}
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "src/ui/app.py",
         "--server.headless", "true", "--server.port", str(PORT),
         "--server.enableCORS", "false", "--browser.gatherUsageStats", "false",
         "--theme.base", "light"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
    )
    current_group = "工作台"
    try:
        if not wait_server():
            print("ERROR: streamlit server not up")
            return
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1024})
            page.goto(URL, wait_until="load", timeout=30000)
            page.wait_for_selector("text=系统概览", timeout=30000)
            page.wait_for_timeout(2500)

            for label, group, fname in PAGES:
                if group != current_group:
                    try:
                        page.click(f"text={group}", timeout=6000)
                        current_group = group
                        page.wait_for_timeout(800)
                    except Exception as e:
                        print(f"WARN: 展开分组 {group} 失败: {e}")
                try:
                    page.click(f"text={label}", timeout=8000)
                    page.wait_for_timeout(3000)
                    page.screenshot(path=os.path.join(ASSET_DIR, fname), full_page=True)
                    print(f"shot {fname}")
                except Exception as e:
                    print(f"WARN: 截图 {label} 失败: {e}")
            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
