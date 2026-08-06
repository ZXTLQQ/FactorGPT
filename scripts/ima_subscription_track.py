#!/usr/bin/env python3
"""FactorGPT 订阅/共享知识库 自动清单 & 变更跟踪脚本（断点续爬版）。

背景（已实测确认的平台限制）：
  ima 开放 API 对「订阅 / 共享（第三方）知识库」禁止导出原文
  （get_media_info 返回 220030，search_knowledge 的 highlight_content 为空），
  但允许「列目录」：search_knowledge_base 列库 + get_knowledge_list 递归浏览文件树。
  本脚本利用这一能力，周期性把订阅库的**目录快照**落盘并与上一次快照 diff，
  自动生成「新增 / 移除」变更清单，帮你定位订阅库每天新增了哪些研报。

为什么需要「断点续爬」：
  get_knowledge_list 单页上限 50 条，一个 1.7 万篇的订阅库需约 350+ 次分页请求；
  ima 对开放 API 有账户级频控（code=220021 请求过于频繁），连续请求会被限流。
  因此脚本以「文件夹」为最小进度单元持久化 crawl_state.json：
    - 跑一半被限流 → 保存已完成的文件夹，下次运行从断点续爬，不重头再来；
    - 只有完整跑完一轮（所有文件夹都遍历完）才写 manifest.json 并计算 diff；
    - 配合每日定时任务，几天内即可建立完整基线，之后持续增量维护。

产物（默认写到 ima_subscription/ 目录）：
  - manifest.json               完整目录快照（含 media_id / 标题 / 层级），仅完成一轮后更新
  - subscription_changelog.md   追加式变更历史（每次完成一轮的增删明细）
  - subscription_index.csv      人类可读扁平目录（知识库, 层级, 标题, 类型, media_id）
  - crawl_state.json            （中间态，不推送）断点续爬进度

用法：
  python scripts/ima_subscription_track.py                  # 跟踪所有订阅库，完成一轮且有变更则推送 GitHub
  python scripts/ima_subscription_track.py --kb-name 智汇研  # 只跟踪名称包含「智汇研」的库
  python scripts/ima_subscription_track.py --no-push        # 只生成清单，不推送 GitHub
  python scripts/ima_subscription_track.py --delay 0.4      # 每次分页请求的礼貌间隔（秒，建议 >=0.3）
  python scripts/ima_subscription_track.py --max-pages 5    # 调试用：每库只爬前 N 页

注意 IMA_CLIENT_ID / IMA_API_KEY 有效期仅一个月，到期需在 ima.qq.com/agent-interface 重新生成。
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

from ima_sync import (
    load_credentials,
    search_knowledge_base,
    API_BASE,
)

DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ima_subscription"
)

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


def _now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _is_owned(kb):
    role = kb.get("role_type", "")
    return ("创建者" in role) or ("管理员" in role) or ("所有者" in role)


def _list_page(client_id, api_key, kb_id, folder_id, cursor, limit=50,
               delay=0.3, backoff=15, max_retry=10):
    """单页 get_knowledge_list，对频控 220021 做强退避重试。

    返回解析后的 data dict；若持续频控则返回 None（调用方据此判定中断、保存断点）。
    """
    body = {"knowledge_base_id": kb_id, "cursor": cursor, "limit": limit}
    if folder_id:
        body["folder_id"] = folder_id
    data = json.dumps(body).encode("utf-8")
    for attempt in range(max_retry):
        req = urllib.request.Request(f"{API_BASE}/get_knowledge_list", data=data, method="POST")
        req.add_header("ima-openapi-clientid", client_id)
        req.add_header("ima-openapi-apikey", api_key)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                parsed = json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(backoff + attempt * 5)
                continue
            return None
        except Exception:  # noqa
            time.sleep(backoff + attempt * 5)
            continue
        if parsed.get("code") == 220021:  # 频控
            print(f"      [频控] 退避 {backoff + attempt * 5}s 后重试 ...")
            time.sleep(backoff + attempt * 5)
            continue
        if parsed.get("code") != 0:
            print(f"      ! get_knowledge_list 错误 code={parsed.get('code')} msg={parsed.get('msg')}")
            return None
        if delay:
            time.sleep(delay)
        return parsed.get("data", {})
    return None  # 持续频控，放弃本页


def _crawl_folder(client_id, api_key, kb_id, folder_id, folder_path, state,
                  delay, limit, max_pages, page_budget):
    """递归爬一个文件夹。返回 (completed, page_budget_left)。频控中断时 completed=False。"""
    cursor = ""
    pages = 0
    while True:
        if max_pages is not None and pages >= max_pages:
            return True, page_budget  # 调试截断，视为该文件夹完成
        data = _list_page(client_id, api_key, kb_id, folder_id, cursor, limit, delay)
        if data is None:
            return False, page_budget  # 频控中断
        items = data.get("knowledge_list", [])
        for it in items:
            mtype = it.get("media_type")
            mid = it.get("media_id")
            title = it.get("title", mid)
            if mtype == 99:  # 文件夹
                if mid in state["done_folders"]:
                    continue
                state["done_folders"].append(mid)
                ok, page_budget = _crawl_folder(
                    client_id, api_key, kb_id, mid, folder_path + [title],
                    state, delay, limit, max_pages, page_budget)
                # 子文件夹处理完即落盘，保证断点可恢复
                _save_state(state)
                if not ok:
                    return False, page_budget
            elif mid:
                state["entries"][mid] = {
                    "media_id": mid, "title": title, "media_type": mtype,
                    "folder": "/".join(folder_path),
                }
        pages += 1
        if data.get("is_end") or not data.get("next_cursor"):
            break
        cursor = data.get("next_cursor", "")
        if not cursor:
            break
    return True, page_budget


def _save_state(state):
    path = state["_state_path"]
    data = {k: v for k, v in state.items() if k != "_state_path"}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)


def track(client_id, api_key, out_dir, do_push=True, kb_name=None,
          delay=0.3, max_pages=None, limit=50):
    os.makedirs(out_dir, exist_ok=True)
    state_path = os.path.join(out_dir, "crawl_state.json")

    print("[1/4] 列出知识库 ...")
    kbs = search_knowledge_base(client_id, api_key)
    if not kbs:
        print("  未找到任何知识库。请检查 IMA_CLIENT_ID / IMA_API_KEY 是否有效（有效期一个月）。")
        return

    sub_kbs = []
    for kb in kbs:
        owned = _is_owned(kb)
        tag = "[订阅/共享]" if not owned else "[自有]"
        print(f"  - {kb.get('kb_name')}  (id={kb.get('kb_id')}, 内容数={kb.get('content_count','0')}, "
              f"角色={kb.get('role_type','')}) {tag}")
        if owned:
            continue
        if kb_name and kb_name not in kb.get("kb_name", ""):
            continue
        sub_kbs.append(kb)
    if not sub_kbs:
        print("  没有匹配的订阅/共享知识库，退出。")
        return

    print(f"\n[2/4] 递归遍历 {len(sub_kbs)} 个订阅库的目录树（断点续爬）...")
    captured_at = _now_iso()
    all_entries = {}
    any_incomplete = False

    for kb in sub_kbs:
        kb_id = kb.get("kb_id")
        kb_name = kb.get("kb_name", "kb")
        # 读取该库的续爬进度（已完成文件夹 + 已收集条目）
        state = {"kb_id": kb_id, "kb_name": kb_name, "done_folders": [],
                 "entries": {}, "_state_path": state_path}
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as fh:
                    saved = json.load(fh)
                if saved.get("kb_id") == kb_id:
                    state["done_folders"] = saved.get("done_folders", [])
                    state["entries"] = saved.get("entries", {})
                    print(f"  > 「{kb_name}」从断点续爬：已完成 {len(state['done_folders'])} 个文件夹，"
                          f"已收集 {len(state['entries'])} 条")
            except Exception:  # noqa
                pass

        print(f"  > 遍历「{kb_name}」 ...")
        ok, _ = _crawl_folder(client_id, api_key, kb_id, None, [], state,
                              delay, limit, max_pages, None)
        if ok:
            print(f"    = 完成一轮：共 {len(state['entries'])} 个文件条目。")
            # 合并到总表
            for mid, e in state["entries"].items():
                e["kb_name"] = kb_name
                e["kb_id"] = kb_id
                all_entries[mid] = e
            # 清空该库断点（已完成）
            if os.path.exists(state_path):
                os.remove(state_path)
        else:
            any_incomplete = True
            print(f"    ! 被频控中断，已保存断点（{len(state['done_folders'])} 个文件夹 / "
                  f"{len(state['entries'])} 条）。下次运行自动续爬，无需重头开始。")
            # 保留 state 文件供续爬；本库条目暂不计入最终 manifest（避免不完整 diff）
            break

    if any_incomplete:
        print("\n[3/4] 本轮因频控未完成，已保存断点，暂不更新 manifest / 不计算 diff。")
        print("       建议：减小 --delay 无意义，反而应增大间隔或错峰运行；等待频控恢复后重跑本脚本即可续爬。")
        print("[4/4] 跳过 GitHub 推送。")
        return

    if not all_entries:
        print("\n[3/4] 未收集到任何文件条目（可能该库为空），退出。")
        return

    print(f"\n[3/4] 与上一轮快照 diff（共 {len(all_entries)} 条）...")
    prev_path = os.path.join(out_dir, "manifest.json")
    prev = {}
    if os.path.exists(prev_path):
        try:
            with open(prev_path, "r", encoding="utf-8") as fh:
                prev = {e["media_id"]: e for e in json.load(fh).get("files", [])}
        except Exception:  # noqa
            prev = {}

    cur_ids, prev_ids = set(all_entries), set(prev)
    added = [all_entries[m] for m in cur_ids - prev_ids]
    removed = [prev[m] for m in prev_ids - cur_ids]
    added_by_kb = {}
    for e in added:
        added_by_kb[e["kb_name"]] = added_by_kb.get(e["kb_name"], 0) + 1

    print(f"    当前: {len(cur_ids)}  |  上一轮: {len(prev_ids)}  |  新增: {len(added)}  |  移除: {len(removed)}")
    for kn, n in added_by_kb.items():
        print(f"      + 「{kn}」新增 {n} 篇")

    # manifest.json
    manifest = {
        "captured_at": captured_at,
        "total": len(cur_ids),
        "knowledge_bases": [kb.get("kb_name") for kb in sub_kbs],
        "files": list(all_entries.values()),
    }
    with open(prev_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print("    已写 manifest.json")

    # CSV
    csv_path = os.path.join(out_dir, "subscription_index.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["知识库", "文件夹层级", "标题", "类型", "media_id"])
        for e in sorted(all_entries.values(), key=lambda x: (x["kb_name"], x["folder"], x["title"])):
            w.writerow([e["kb_name"], e["folder"], e["title"],
                        MEDIA_EXT.get(e["media_type"], f"t{e['media_type']}"), e["media_id"]])
    print("    已写 subscription_index.csv")

    # changelog（仅当有变更）
    if added or removed:
        clog = os.path.join(out_dir, "subscription_changelog.md")
        with open(clog, "a", encoding="utf-8") as fh:
            fh.write(f"\n## {captured_at}\n\n")
            if added:
                fh.write(f"**新增 {len(added)} 篇：**\n\n")
                for e in added:
                    loc = f"「{e['kb_name']}」/ {e['folder']}" if e["folder"] else f"「{e['kb_name']}」"
                    fh.write(f"- [新增] {loc} / {e['title']}{MEDIA_EXT.get(e['media_type'],'')}  (id={e['media_id']})\n")
                fh.write("\n")
            if removed:
                fh.write(f"**移除 {len(removed)} 篇：**\n\n")
                for e in removed:
                    loc = f"「{e['kb_name']}」/ {e['folder']}" if e["folder"] else f"「{e['kb_name']}」"
                    fh.write(f"- [移除] {loc} / {e['title']}{e.get('ext','')}  (id={e['media_id']})\n")
                fh.write("\n")
        print(f"    已追加 subscription_changelog.md（新增 {len(added)} / 移除 {len(removed)}）")
    else:
        print("    无变化，未追加变更日志。")

    if not do_push:
        print("[4/4] 跳过 GitHub 推送（--no-push）。")
        return
    if not added and not removed:
        print("[4/4] 无变更，跳过 GitHub 推送。")
        return

    print("[4/4] 提交并推送到 GitHub ...")
    _git_commit_push(out_dir, len(added), len(removed))


def _git_commit_push(out_dir, n_added, n_removed):
    root = os.path.dirname(out_dir)
    os.chdir(root)
    os.system("git add ima_subscription/manifest.json "
              "ima_subscription/subscription_index.csv "
              "ima_subscription/subscription_changelog.md")
    msg = f"chore: update ima subscription manifest (+{n_added}/-{n_removed}) via ima_subscription_track.py"
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
    p.add_argument("--kb-name", default=None, help="只跟踪名称包含该子串的订阅知识库")
    p.add_argument("--no-push", action="store_true", help="只生成清单，不推送 GitHub")
    p.add_argument("--delay", type=float, default=0.3, help="每次分页请求的礼貌间隔（秒，建议>=0.3）")
    p.add_argument("--limit", type=int, default=50, help="每页条数（1-50）")
    p.add_argument("--max-pages", type=int, default=None, help="调试用：每库只爬前 N 页")
    p.add_argument("--dir", default=DEFAULT_OUT_DIR, help="清单输出目录")
    args = p.parse_args()

    client_id, api_key = load_credentials()
    if not client_id or not api_key:
        print("缺少 IMA 凭证：请在 .env 中配置 IMA_CLIENT_ID 与 IMA_API_KEY。")
        sys.exit(1)

    track(client_id, api_key, args.dir, do_push=not args.no_push,
          kb_name=args.kb_name, delay=args.delay, max_pages=args.max_pages, limit=args.limit)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa
        pass
    main()
