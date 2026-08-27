#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mx_query - 东方财富妙想 (MX) 数据接口统一入口。

将 factorgpt-skill/skills/mx-*/ 下的 6 个妙想技能脚本封装为一条命令，
自动完成跨平台输出目录适配（替代脚本默认的 /root/... 路径）与
API Key 注入（环境变量 MX_APIKEY 优先，回退到项目 .env 文件）。

用法:
  python scripts/mx_query.py data    "贵州茅台最新收盘价与PE"
  python scripts/mx_query.py search  "半导体行业最新研报"
  python scripts/mx_query.py xuangu  "最近5日涨幅超过20%的股票"
  python scripts/mx_query.py zixuan  "查询我的自选股行情"
  python scripts/mx_query.py moni    "查询我的模拟组合收益"
  python scripts/mx_query.py poster  "金融社区热门内容"
  python scripts/mx_query.py --list

输出目录默认为 <项目根>/output/mx_data/，可用最后一个参数覆盖。
"""

import os
import sys
import pathlib
import subprocess

# 技能简称 -> (目录名, 脚本文件名)
SKILLS = {
    "data": ("mx-data", "mx_data.py"),
    "search": ("mx-search", "mx_search.py"),
    "xuangu": ("mx-xuangu", "mx_xuangu.py"),
    "zixuan": ("mx-zixuan", "mx_zixuan.py"),
    "moni": ("mx-moni", "mx_moni.py"),
    "poster": ("mx-poster", "mx_poster.py"),
}

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_env_file(path: pathlib.Path) -> dict:
    """极简 .env 解析（仅 KEY=VALUE 行，忽略注释）。"""
    env = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key:
            env[key] = value
    return env


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "--list"):
        print(__doc__)
        return 0

    kind = args[0]
    if kind not in SKILLS:
        print(f"未知能力: {kind}，可用: {', '.join(SKILLS)}")
        return 2
    if len(args) < 2:
        print(f"用法: python scripts/mx_query.py {kind} \"查询问句\" [输出目录]")
        return 2

    query = args[1]
    out_dir = args[2] if len(args) > 2 else str(
        (PROJECT_ROOT / "output" / "mx_data").resolve()
    )

    dir_name, script_name = SKILLS[kind]
    script = PROJECT_ROOT / "factorgpt-skill" / "skills" / dir_name / script_name
    if not script.exists():
        print(f"技能包缺失: {script}\n请确认 factorgpt-skill/skills/ 已包含 mx-* 技能包。")
        return 2

    # 输出目录由入口统一创建（Windows 下不落 /root/... 路径）
    os.makedirs(out_dir, exist_ok=True)

    # 注入 MX_APIKEY：环境变量优先，回退项目 .env
    env = dict(os.environ)
    if not env.get("MX_APIKEY"):
        dotenv = load_env_file(PROJECT_ROOT / ".env")
        if dotenv.get("MX_APIKEY"):
            env["MX_APIKEY"] = dotenv["MX_APIKEY"]

    # 官方脚本含 emoji 打印，Windows GBK 控制台会抛 UnicodeEncodeError；
    # 强制子进程以 UTF-8 输出，避免查询成功后仍以非零码退出。
    env.setdefault("PYTHONIOENCODING", "utf-8")

    print(f"[mx_query] 能力={kind} 输出={out_dir}")
    return subprocess.call([sys.executable, str(script), query, out_dir], env=env)


if __name__ == "__main__":
    sys.exit(main())
