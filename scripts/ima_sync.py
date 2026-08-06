#!/usr/bin/env python3
"""FactorGPT <-> ima 知识库 研报同步脚本。

功能：
  1. 通过 ima 开放 API (openapi/wiki/v1) 列出你的知识库（含订阅/共享知识库）。
  2. 递归遍历知识库目录，下载其中「你自己拥有」的知识库里的研报/文档到本地 ima_research/。
  3. （可选）自动 git commit & push 到 GitHub 仓库。

重要平台限制（已实测确认）：
  ima 开放 API 的 get_media_info 对「订阅 / 共享知识库」返回
  code=220030「没有权限通过 skill 读取该知识库文件，请前往 ima 客户端查看」，
  即第三方/订阅知识库无法通过 API 导出原文，只能列目录、不能下载。
  因此本脚本可完整下载「你自己创建/拥有」的知识库，订阅库会被自动识别并跳过。

认证：需要 IMA_CLIENT_ID 与 IMA_API_KEY 两个值，优先级：
  环境变量 IMA_CLIENT_ID / IMA_API_KEY  >  .env 文件  >  脚本内默认值（请在 .env 中配置）。
  这两个值来自 https://ima.qq.com/agent-interface 页面（Client ID + API Key，有效期一个月）。

用法：
  python scripts/ima_sync.py                  # 同步所有「可下载」的知识库到 ima_research/
  python scripts/ima_sync.py --subscription   # 仅尝试订阅知识库（会被平台限制跳过，仅列目录）
  python scripts/ima_sync.py --kb-name 我的研报  # 按名称过滤
  python scripts/ima_sync.py --no-push        # 只下载不推送 GitHub
  python scripts/ima_sync.py --force            # 强制重下已存在的文件
  注：默认增量下载，已存在的文件会被跳过（= 标记），重跑只拉新增。
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------
BASE_URL = "https://ima.qq.com"
API_BASE = f"{BASE_URL}/openapi/wiki/v1"
DEFAULT_LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ima_research"
)


def _load_env_file(path):
    env = {}
    if not os.path.exists(path):
        return env
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def load_credentials():
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    file_env = _load_env_file(env_path)
    client_id = os.environ.get("IMA_CLIENT_ID") or file_env.get("IMA_CLIENT_ID") or ""
    api_key = os.environ.get("IMA_API_KEY") or file_env.get("IMA_API_KEY") or ""
    return client_id, api_key


# ---------------------------------------------------------------------------
# ima API 封装
# ---------------------------------------------------------------------------
def call_api(client_id, api_key, path, body, max_retry=5):
    url = f"{API_BASE}/{path}"
    data = json.dumps(body).encode("utf-8")
    last_err = None
    for attempt in range(max_retry):
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("ima-openapi-clientid", client_id)
        req.add_header("ima-openapi-apikey", api_key)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8", "replace")
                parsed = json.loads(raw) if raw.strip() else {}
                # 限流 / 频繁：等待后重试
                if parsed.get("code") in (220021,):
                    time.sleep(min(2 ** attempt + 2, 30))
                    last_err = "rate limited"
                    continue
                return parsed
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            if e.code in (429, 503):
                time.sleep(2 ** attempt + 1)
                last_err = f"HTTP {e.code}"
                continue
            raise RuntimeError(f"ima API {path} HTTP {e.code}: {raw[:400]}")
        except Exception as e:  # noqa
            time.sleep(2 ** attempt + 1)
            last_err = str(e)
            continue
    raise RuntimeError(f"ima API {path} 重试 {max_retry} 次仍失败（{last_err}）")


def search_knowledge_base(client_id, api_key, query="", limit=20):
    """搜索/列出知识库，自动翻页。返回 info_list 条目（kb_id, kb_name, ...）。"""
    results = []
    cursor = ""
    while True:
        body = {"query": query, "cursor": cursor, "limit": limit}
        resp = call_api(client_id, api_key, "search_knowledge_base", body)
        if resp.get("code") != 0:
            raise RuntimeError(f"search_knowledge_base failed: {resp}")
        data = resp.get("data", {})
        results.extend(data.get("info_list", []))
        if data.get("is_end") or not data.get("next_cursor"):
            break
        cursor = data.get("next_cursor", "")
        if not cursor:
            break
    return results


def get_knowledge_list(client_id, api_key, kb_id, folder_id=None, limit=50):
    """列出知识库（或某文件夹）下的条目，自动翻页。返回 knowledge_list 扁平列表。"""
    results = []
    cursor = ""
    while True:
        body = {
            "knowledge_base_id": kb_id,
            "cursor": cursor,
            "limit": limit,
        }
        if folder_id:
            body["folder_id"] = folder_id
        resp = call_api(client_id, api_key, "get_knowledge_list", body)
        if resp.get("code") != 0:
            raise RuntimeError(f"get_knowledge_list failed: {resp}")
        data = resp.get("data", {})
        results.extend(data.get("knowledge_list", []))
        if data.get("is_end") or not data.get("next_cursor"):
            break
        cursor = data.get("next_cursor", "")
        if not cursor:
            break
    return results


def get_media_info(client_id, api_key, media_id):
    """获取媒体下载信息。未知知识库返回 code=220030（订阅库禁止导出）或 code=0(含 url_info)。"""
    body = {"media_id": media_id}
    resp = call_api(client_id, api_key, "get_media_info", body)
    return resp


def download_file(url, dest_path, headers=None):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    req = urllib.request.Request(url)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=120) as resp:
        with open(dest_path, "wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)


# ---------------------------------------------------------------------------
# 同步主流程
# ---------------------------------------------------------------------------
def safe_name(name):
    keep = []
    for ch in name:
        if ch.isalnum() or ch in (" ", "-", "_", ".", "(", ")", "，", "：", "、", "的"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip().replace(" ", "_")[:120] or "untitled"


MEDIA_EXT = {
    1: ".pdf",
    3: ".docx",
    4: ".pptx",
    5: ".xlsx",
    7: ".md",
    13: ".txt",
    20: ".html",
    9: ".png",
    15: ".mp3",
}


def _collect_files(client_id, api_key, kb_id, folder_id, visited, acc):
    """递归收集某知识库/文件夹下的所有文件条目（media_type != 99）。"""
    items = get_knowledge_list(client_id, api_key, kb_id, folder_id=folder_id)
    for it in items:
        mtype = it.get("media_type")
        mid = it.get("media_id")
        if mtype == 99:  # 文件夹
            if mid in visited:
                continue
            visited.add(mid)
            _collect_files(client_id, api_key, kb_id, mid, visited, acc)
        elif mid:
            acc.append(it)
    return acc


def sync(client_id, api_key, local_dir, subscription_only=False, kb_name=None, do_push=True, force=False):
    print("[1/4] 列出知识库 ...")
    kbs = search_knowledge_base(client_id, api_key)
    if not kbs:
        print("  未找到任何知识库。请检查 IMA_CLIENT_ID / IMA_API_KEY 是否有效（有效期一个月）。")
        return

    targets = []
    for kb in kbs:
        name = kb.get("kb_name", "")
        kb_id = kb.get("kb_id", "")
        role = kb.get("role_type", "")
        btype = kb.get("base_type", "")
        content_count = kb.get("content_count", "0")
        is_owned = ("创建者" in role) or ("管理员" in role) or ("所有者" in role)
        is_sub = not is_owned
        tag = "[订阅/共享]" if is_sub else "[自有]"
        print(f"  - {name}  (id={kb_id}, 内容数={content_count}, 角色={role}, 类型={btype}) {tag}")
        if subscription_only and not is_sub:
            continue
        if kb_name and kb_name not in name:
            continue
        targets.append(kb)

    if not targets:
        print("  没有匹配的知识库，退出。")
        return

    print(f"\n[2/4] 尝试同步 {len(targets)} 个知识库到 {local_dir} ...")
    os.makedirs(local_dir, exist_ok=True)
    total_files = 0
    skipped_kbs = 0

    for kb in targets:
        kb_id = kb.get("kb_id")
        kb_name_safe = safe_name(kb.get("kb_name", "kb"))
        kb_dir = os.path.join(local_dir, kb_name_safe)

        # 先用根目录首个文件探测该知识库是否允许导出（订阅库会返回 220030），
        # 命中则整库跳过，避免对万级条目的订阅库做无谓遍历。
        try:
            root_items = get_knowledge_list(client_id, api_key, kb_id, limit=1)
        except Exception as e:  # noqa
            print(f"  [WARN] 知识库「{kb.get('kb_name')}」列目录失败（限流？）: {e}，跳过。")
            skipped_kbs += 1
            continue
        first_file = next((it for it in root_items if it.get("media_type") != 99), None)
        if first_file:
            probe = get_media_info(client_id, api_key, first_file["media_id"])
            if probe.get("code") == 220030:
                print(f"\n  [WARN] 知识库「{kb.get('kb_name')}」被平台限制：无法通过 API 导出文件"
                      f"（{probe.get('msg')}）。跳过该库。")
                skipped_kbs += 1
                continue

        # 收集全部文件条目（递归遍历文件夹）
        files = _collect_files(client_id, api_key, kb_id, None, set(), [])

        os.makedirs(kb_dir, exist_ok=True)
        kb_files = 0
        for it in files:
            mid = it["media_id"]
            title = it.get("title", mid)
            try:
                info = get_media_info(client_id, api_key, mid)
            except Exception as e:  # noqa
                print(f"    ! 获取媒体信息失败 {title}: {e}")
                continue
            if info.get("code") != 0:
                if info.get("code") == 220030:
                    print(f"    ! 跳过（无导出权限）: {title}")
                else:
                    print(f"    ! 获取失败({info.get('code')}): {title}")
                continue
            data = info.get("data", {})
            url_info = data.get("url_info")
            if not url_info or not url_info.get("url"):
                print(f"    ! 无下载链接: {title}")
                continue
            mtype = data.get("media_type", 1)
            ext = MEDIA_EXT.get(mtype, os.path.splitext(title)[1] or ".bin")
            base = safe_name(os.path.splitext(title)[0])
            dest = os.path.join(kb_dir, base + ext)
            # 增量下载：文件已存在则跳过（--force 可强制重下）
            if os.path.exists(dest) and not force:
                print(f"    = 已存在，跳过: {title}")
                continue
            try:
                download_file(url_info["url"], dest, url_info.get("headers"))
                kb_files += 1
                total_files += 1
                print(f"    + {title} -> {os.path.relpath(dest)}")
            except Exception as e:  # noqa
                print(f"    ! 下载失败 {title}: {e}")
            time.sleep(0.3)  # 礼貌限速

        print(f"  [OK] 知识库「{kb.get('kb_name')}」下载 {kb_files} 个文件。")

    print(f"\n[3/4] 共下载 {total_files} 个文件；跳过被平台限制的知识库 {skipped_kbs} 个。")

    if not do_push:
        print("[4/4] 跳过 GitHub 推送（--no-push）。")
        return

    print("[4/4] 提交并推送到 GitHub ...")
    _git_commit_push(local_dir, total_files)


def _git_commit_push(local_dir, n_files):
    root = os.path.dirname(local_dir)
    os.chdir(root)
    os.system("git add ima_research/")
    msg = f"chore: sync ima research reports ({n_files} files) via ima_sync.py"
    rc = os.system(f'git commit -m "{msg}"')
    if rc != 0:
        print("  (无新改动或 commit 失败，跳过)")
        return
    rc = os.system("git push origin HEAD")
    if rc != 0:
        print("  ! git push 失败，请检查 SSH/网络。")
    else:
        print("  已推送至 GitHub。")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subscription", action="store_true", help="仅尝试订阅知识库（会被平台限制跳过）")
    p.add_argument("--kb-name", default=None, help="按知识库名称过滤（子串匹配）")
    p.add_argument("--no-push", action="store_true", help="只下载，不推送 GitHub")
    p.add_argument("--force", action="store_true", help="强制重新下载已存在的文件")
    p.add_argument("--dir", default=DEFAULT_LOCAL_DIR, help="本地保存目录")
    args = p.parse_args()

    client_id, api_key = load_credentials()
    if not client_id or not api_key:
        print("缺少 IMA 凭证：请在 .env 中配置 IMA_CLIENT_ID 与 IMA_API_KEY。")
        sys.exit(1)

    sync(
        client_id,
        api_key,
        args.dir,
        subscription_only=args.subscription,
        kb_name=args.kb_name,
        do_push=not args.no_push,
        force=args.force,
    )


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa
        pass
    main()
