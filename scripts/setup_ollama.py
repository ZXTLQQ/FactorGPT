#!/usr/bin/env python
"""FactorGPT 本地部署 Ollama 的一键脚本(纯 Python, 跨平台)。

用法:
    python scripts/setup_ollama.py                安装 + 拉取默认模型(qwen2.5-coder:7b) + 切换 config
    python scripts/setup_ollama.py --skip-pull    仅安装并启动服务, 不拉取模型
    python scripts/setup_ollama.py --model llama3.1:8b  指定模型

说明:
    - Ollama Windows 安装需要管理员权限。本脚本会检测当前是否以管理员运行,
      若不是会弹出 UAC 提权窗口, 重新以管理员身份拉起自身(参数原样传递)。
    - 模型拉取较大(qwen2.5-coder:7b 约 4.7GB), 可用 --skip-pull 跳过, 稍后手动 `ollama pull`。
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import urllib.request

OLLAMA_INSTALLER = "https://ollama.com/download/OllamaSetup.exe"
DEFAULT_MODEL = "qwen2.5-coder:7b"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def self_elevate_and_exit() -> None:
    """以管理员身份重新拉起当前脚本(带相同参数), 然后退出当前进程。

    提权后的子进程会把全部输出写入 setup_ollama.log, 并在窗口末尾 pause 保持窗口,
    因此用户既能看窗口也能事后查看日志。
    """
    script = os.path.abspath(sys.argv[0])
    params = " ".join(f'"{a}"' for a in sys.argv[1:])
    log = os.path.join(ROOT, "setup_ollama.log")
    inner = f'"{sys.executable}" "{script}" {params} > "{log}" 2>&1'
    cmdline = f'/c "{inner} & echo. & echo [安装日志已写入 {log}, 按任意键关闭此窗口] & pause > nul"'
    print("[提权] 当前非管理员, 正在请求 UAC 提权并重跑脚本...")
    print(f"       请点击 UAC 的【是】。全部输出将写入日志: {log}")
    try:
        ctypes.windll.shell32.ShellExecuteW(
            None, "runas", "cmd.exe", cmdline, os.getcwd(), 1
        )
    except Exception as e:  # noqa: BLE001
        print(f"[提权] 失败: {e}")
        print("请手动以管理员身份打开 PowerShell, 再执行: "
              f'python "{script}" ' + " ".join(sys.argv[1:]))
    sys.exit(0)


def run(cmd, **kw):
    print(">>", " ".join(cmd) if isinstance(cmd, list) else cmd)
    return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, **kw)


def download_installer() -> str:
    tmp = os.environ.get("TEMP", ROOT)
    inst = os.path.join(tmp, "OllamaSetup.exe")
    if os.path.exists(inst) and os.path.getsize(inst) > 5_000_000:
        print(f"[1/4] 安装包已存在: {inst} ({os.path.getsize(inst)} bytes)")
        return inst
    if os.path.exists(inst):
        print(f"[1/4] 安装包疑似不完整({os.path.getsize(inst)} bytes), 重新下载...")
        os.remove(inst)
    print(f"[1/4] 下载 Ollama 安装包 -> {inst} ...")
    urllib.request.urlretrieve(OLLAMA_INSTALLER, inst)
    print(f"      下载完成 ({os.path.getsize(inst)} bytes)")
    return inst


def silent_install(inst: str) -> int:
    print("[2/4] 静默安装 Ollama(已提权)...")
    # NSIS 安装器静默参数为 /S; 捕获输出以便排查
    r = subprocess.run([inst, "/S"], capture_output=True, text=True)
    print(f"      安装退出码: {r.returncode}")
    if r.stdout.strip():
        print("      [stdout]", r.stdout.strip()[:500])
    if r.stderr.strip():
        print("      [stderr]", r.stderr.strip()[:500])
    if r.returncode != 0:
        print("      安装失败。若曾部分安装, 建议先卸载旧版本, 或手动双击安装包完成安装。")
    time.sleep(3)
    # 刷新 PATH(读取机器+用户环境变量)
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "[Environment]::GetEnvironmentVariable('Path','Machine')+';'+[Environment]::GetEnvironmentVariable('Path','User')"],
            capture_output=True, text=True,
        )
        if out.returncode == 0 and out.stdout.strip():
            os.environ["PATH"] = out.stdout.strip()
    except Exception:
        pass
    return r.returncode


def find_ollama() -> str | None:
    # 常见安装位置
    candidates = [
        r"C:\Program Files\Ollama\ollama.exe",
        r"C:\Users\%s\AppData\Local\Programs\Ollama\ollama.exe" % os.environ.get("USERNAME", ""),
        os.path.join(ROOT, "ollama.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # PATH 中查找
    try:
        r = subprocess.run(["where", "ollama"], capture_output=True, text=True, shell=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return None


def start_and_wait(ollama: str) -> bool:
    print("[3/4] 启动 Ollama 服务...")
    try:
        subprocess.Popen([ollama, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:  # noqa: BLE001
        print(f"      后台启动失败: {e}")
    for _ in range(30):
        try:
            with urllib.request.urlopen("http://localhost:11434/api/version", timeout=2) as resp:
                data = json.loads(resp.read().decode())
                print(f"      Ollama 已就绪, version={data.get('version')}")
                return True
        except Exception:
            time.sleep(2)
    print("      Ollama 服务未就绪(可稍后手动 `ollama serve`)")
    return False


def pull_model(ollama: str, model: str) -> None:
    print(f"[4/4] 拉取模型 {model} (可能较大, 请耐心等待)...")
    r = subprocess.run([ollama, "pull", model], capture_output=True, text=True)
    if r.returncode == 0:
        print(f"      模型 {model} 拉取完成")
    else:
        print(f"      模型拉取失败(退出码 {r.returncode})")
        if r.stderr.strip():
            print("      [stderr]", r.stderr.strip()[:500])
        print(f"      可稍后手动 `ollama pull {model}`")


def switch_config(model: str) -> None:
    cfg = os.path.join(ROOT, "config.yaml")
    if not os.path.exists(cfg):
        print("未找到 config.yaml, 跳过切换")
        return
    with open(cfg, encoding="utf-8") as f:
        txt = f.read()
    import re
    txt = re.sub(r"(?m)^\s*provider:\s*\S+", "provider: ollama", txt)
    txt = re.sub(r"(?m)^\s*model:\s*\S+", f"model: {model}", txt)
    txt = re.sub(r'(?m)^\s*base_url:\s*".*?"', 'base_url: "http://localhost:11434/v1"', txt)
    txt = re.sub(r'(?m)^\s*api_key:\s*".*?"', 'api_key: "ollama"', txt)
    with open(cfg, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"已更新 config.yaml -> provider=ollama, model={model}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pull", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    # 安装/启动服务需要管理员权限: 非管理员则自动提权重跑
    if os.name == "nt" and not is_admin():
        self_elevate_and_exit()

    inst = download_installer()
    rc = silent_install(inst)
    ollama = find_ollama()
    if ollama is None:
        print("未找到 ollama 可执行文件, 安装未成功。")
        print("请确认 UAC 已同意、或以管理员身份手动双击 "
              f"{inst} 完成安装后, 再执行 `ollama pull {args.model}`。")
        return 1
    print(f"ollama 路径: {ollama}")
    start_and_wait(ollama)
    if not args.skip_pull:
        pull_model(ollama, args.model)
    switch_config(args.model)
    print("\n完成。现在可运行: python run_agent.py \"请构建一个 20 日动量因子\"")
    # 管理员态运行时, 保持窗口不关闭, 便于查看结果
    if os.name == "nt" and is_admin():
        os.system("pause")
    return 0


if __name__ == "__main__":
    sys.exit(main())
