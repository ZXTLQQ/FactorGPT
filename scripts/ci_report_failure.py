"""把 CI 失败现场**发布成一个 issue**，让排障不用在 Actions 页面里翻日志。

    python scripts/ci_report_failure.py --log build_ext.log --title-suffix "windows/py3.11"

为什么要有它：Actions 的 job 日志需要鉴权才能下载，而非鉴权 REST API 只能看到
"哪一步红了"，看不到**为什么**。于是修 Windows 那两条腿就变成猜谜——每猜一次要
一轮 CI（约 4 分钟）。Issue 是唯一"我这边能读、CI 那边能写"的通道。

设计取舍：
- **只在失败时**才动 GitHub API（``if: failure()``），成功路径零副作用；
- 同标题的未关闭 issue 已存在时只追加评论，不刷屏；
- 没有 ``GITHUB_TOKEN``（本地跑）就只打印正文，不发请求——本机可以放心执行。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import urllib.error
import urllib.request
from pathlib import Path

LABEL = "ci-diagnostic"
MAX_LINES = 120


def _tail(path: Path, limit: int = MAX_LINES) -> str:
    if not path.exists():
        return f"(日志文件不存在: {path})"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    kept = lines[-limit:]
    dropped = len(lines) - len(kept)
    head = f"（共 {len(lines)} 行，已省略前 {dropped} 行）\n" if dropped else ""
    return head + "\n".join(kept)


def _api(method: str, url: str, token: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json",
                 "User-Agent": "factorgpt-ci-diag"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:400]
        raise SystemExit(f"GitHub API {e.code}: {body}")


def main() -> int:
    ap = argparse.ArgumentParser(description="把 CI 失败现场发成 issue")
    ap.add_argument("--log", required=True, help="失败步骤的输出文件")
    ap.add_argument("--title-suffix", default="", help="附加到标题里的环境信息")
    ap.add_argument("--step", default="Build extension in place")
    args = ap.parse_args()

    suffix = args.title_suffix or f"{platform.system()}/py{platform.python_version()}"
    title = f"[ci-diagnostic] {args.step} failed on {suffix}"
    body = (
        f"## 失败步骤\n\n`{args.step}`\n\n"
        f"## 环境\n\n- runner.os: `{os.environ.get('RUNNER_OS', 'unknown')}`\n"
        f"- python: `{platform.python_version()}`\n"
        f"- 工作流运行: {os.environ.get('RUN_URL', '(未提供)')}\n\n"
        f"## 构建输出末尾 {MAX_LINES} 行\n\n```text\n{_tail(Path(args.log))}\n```\n"
    )

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if not token or not repo:
        print("本地执行（无 GITHUB_TOKEN / GITHUB_REPOSITORY），仅打印正文：\n")
        print(body)
        return 0

    base = f"https://api.github.com/repos/{repo}"
    existing = _api("GET", f"{base}/issues?state=open&labels={LABEL}&per_page=30",
                    token)
    for issue in existing:
        if issue["title"] == title:
            _api("POST", f"{base}/issues/{issue['number']}/comments", token,
                 {"body": body})
            print(f"已追加到既有 issue #{issue['number']}")
            return 0
    created = _api("POST", f"{base}/issues", token,
                   {"title": title, "body": body, "labels": [LABEL]})
    print(f"已创建 issue #{created['number']}: {created['html_url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
