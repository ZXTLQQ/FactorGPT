#!/usr/bin/env python3
"""FactorGPT 订阅知识库「关键词定向监听」轻量模式。

与 ima_subscription_track.py 的关系：
  - ima_subscription_track.py  = 全量目录快照 + diff（准确但重：1.7 万篇需 350+ 次分页，
                                 极易触发账户级频控 220021，需断点续爬多天才能建立基线）。
  - 本脚本（轻量模式）          = 只用 search_knowledge 按关键词定向检索，每个关键词 1~3 次请求，
                                 十几个关键词也只需几十次请求，几乎不会触发频控，可高频运行。

原理：
  ima 的 search_knowledge 对订阅/共享知识库是放开的（只有 get_media_info 导出原文被 220030 拒绝、
  get_knowledge_list 全量翻页容易被 220021 限流）。因此用「你关心的研究主题词」定向命中订阅库，
  把命中的 media_id 存进 keyword_seen.json 作为基线；下次运行只要出现基线里没有的 media_id，
  就是订阅库**新增**的相关研报，直接产出「今日新增」清单，你照着清单去 ima 客户端复制到自有库即可。

相比全量爬取的取舍：
  只覆盖关键词命中的研报（非全库），但这正好是你真正要的——不关心的研报本来也不会去复制。

产物（默认写到 ima_subscription/）：
  - watch_keywords.json    监听关键词配置（首次运行自动生成默认量化研究词表，可自行编辑）
  - keyword_seen.json      已知命中基线（media_id -> 标题/知识库/首次发现时间）
  - keyword_hits.csv       全部命中的扁平清单（关键词, 知识库, 标题, 首次发现, media_id）
  - keyword_watch.md       追加式「新增研报」日报（只在有新增时追加）

用法：
  python scripts/ima_keyword_watch.py                       # 用配置词表监听所有订阅库
  python scripts/ima_keyword_watch.py --keywords 多因子,择时  # 本次只查这些词（不改配置）
  python scripts/ima_keyword_watch.py --add-keyword 高频交易   # 往配置词表追加一个词后再监听
  python scripts/ima_keyword_watch.py --include-owned       # 连自有库一起监听
  python scripts/ima_keyword_watch.py --no-push             # 只生成清单，不推送 GitHub
  python scripts/ima_keyword_watch.py --max-pages 1         # 每个关键词只取第一页（最省配额）
  python scripts/ima_keyword_watch.py --init                # 只建立基线，不报「新增」（首次运行推荐）

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

from ima_sync import load_credentials, search_knowledge_base, API_BASE

DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ima_subscription"
)

# 默认词表：围绕 FactorGPT 的因子研究场景，可在 watch_keywords.json 中自由增删。
# 选词原则——用「精确的研究主题词」而非宽泛品类词：实测「ETF」「期权」这类宽泛词单页即返回
# 100+ 条，噪音淹没信号；而「选股因子」「量化择时」这类词命中数为个位到二十几条，正是需要盯的增量。
DEFAULT_KEYWORDS = [
    "多因子", "选股因子", "因子择时", "量化择时", "风格轮动",
    "高频因子", "基本面量化", "行业轮动", "机器学习", "可转债量化",
    "量化选股", "红利低波", "小市值", "因子拥挤度",
]


def _now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _today():
    return datetime.now().strftime("%Y-%m-%d")


def _is_owned(kb):
    role = kb.get("role_type", "")
    return ("创建者" in role) or ("管理员" in role) or ("所有者" in role)


def _search_page(client_id, api_key, kb_id, query, cursor, delay=0.4,
                 backoff=10, max_retry=5):
    """单页 search_knowledge。频控 220021 时退避重试；持续失败返回 None。"""
    body = {"query": query, "cursor": cursor, "knowledge_base_id": kb_id}
    data = json.dumps(body).encode("utf-8")
    for attempt in range(max_retry):
        req = urllib.request.Request(f"{API_BASE}/search_knowledge", data=data, method="POST")
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
            print(f"      ! HTTP {e.code}")
            return None
        except Exception as e:  # noqa
            time.sleep(backoff + attempt * 5)
            continue
        code = parsed.get("code")
        if code == 220021:  # 账户级频控
            print(f"      [频控] 退避 {backoff + attempt * 5}s 后重试 ...")
            time.sleep(backoff + attempt * 5)
            continue
        if code != 0:
            print(f"      ! search_knowledge 错误 code={code} msg={parsed.get('msg')}")
            return None
        if delay:
            time.sleep(delay)
        return parsed.get("data", {})
    return None


def _search_all(client_id, api_key, kb_id, query, delay, max_pages):
    """一个关键词在一个库中的全部命中（受 max_pages 限制）。频控返回 None。"""
    hits, cursor, pages = [], "", 0
    while True:
        data = _search_page(client_id, api_key, kb_id, query, cursor, delay)
        if data is None:
            return None
        hits.extend(data.get("info_list", []))
        pages += 1
        if max_pages is not None and pages >= max_pages:
            break
        if data.get("is_end") or not data.get("next_cursor"):
            break
        cursor = data.get("next_cursor", "")
        if not cursor:
            break
    return hits


def load_keywords(out_dir, cli_keywords=None, add_keyword=None):
    """读取（必要时创建）关键词配置。CLI 指定时优先用 CLI，不写回配置。"""
    path = os.path.join(out_dir, "watch_keywords.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            words = cfg.get("keywords", [])
        except Exception:  # noqa
            words = list(DEFAULT_KEYWORDS)
    else:
        words = list(DEFAULT_KEYWORDS)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"keywords": words,
                       "_note": "监听关键词，可自由增删；每个词每轮约消耗 1~3 次 API 请求。"},
                      fh, ensure_ascii=False, indent=2)
        print(f"  已生成默认关键词配置：{os.path.relpath(path)}（{len(words)} 个词，可自行编辑）")

    if add_keyword:
        if add_keyword not in words:
            words.append(add_keyword)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"keywords": words,
                           "_note": "监听关键词，可自由增删；每个词每轮约消耗 1~3 次 API 请求。"},
                          fh, ensure_ascii=False, indent=2)
            print(f"  已追加关键词「{add_keyword}」到配置。")

    if cli_keywords:
        return [w.strip() for w in cli_keywords.split(",") if w.strip()]
    return words


def load_seen(out_dir):
    path = os.path.join(out_dir, "keyword_seen.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh).get("hits", {})
    except Exception:  # noqa
        return {}


def save_seen(out_dir, seen):
    path = os.path.join(out_dir, "keyword_seen.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"updated_at": _now_iso(), "total": sum(len(v) for v in seen.values()),
                   "hits": seen}, fh, ensure_ascii=False, indent=2)


def write_csv(out_dir, seen):
    path = os.path.join(out_dir, "keyword_hits.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["关键词", "知识库", "标题", "首次发现", "media_id"])
        for kw in sorted(seen):
            for mid, info in sorted(seen[kw].items(), key=lambda x: x[1].get("first_seen", "")):
                w.writerow([kw, info.get("kb_name", ""), info.get("title", ""),
                            info.get("first_seen", ""), mid])
    return path


def append_report(out_dir, new_by_kw, captured_at):
    path = os.path.join(out_dir, "keyword_watch.md")
    total = sum(len(v) for v in new_by_kw.values())
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"\n## {captured_at}  新增 {total} 篇\n\n")
        for kw in sorted(new_by_kw):
            items = new_by_kw[kw]
            if not items:
                continue
            fh.write(f"### 关键词：{kw}（{len(items)} 篇）\n\n")
            for it in items:
                fh.write(f"- {it['title']}\n")
                fh.write(f"  · 来源：{it['kb_name']}　· media_id: `{it['media_id']}`\n")
                snippet = (it.get("highlight") or "").replace("\n", " ").strip()
                if snippet:
                    fh.write(f"  > {snippet[:160]}\n")
            fh.write("\n")
    return path


def watch(client_id, api_key, out_dir, keywords, do_push=True, include_owned=False,
          kb_name=None, delay=0.4, max_pages=2, init_only=False):
    os.makedirs(out_dir, exist_ok=True)

    print("[1/4] 列出知识库 ...")
    kbs = search_knowledge_base(client_id, api_key)
    if not kbs:
        print("  未找到任何知识库。请检查 IMA_CLIENT_ID / IMA_API_KEY 是否有效（有效期一个月）。")
        return
    targets = []
    for kb in kbs:
        owned = _is_owned(kb)
        tag = "[自有]" if owned else "[订阅/共享]"
        print(f"  - {kb.get('kb_name')}  (内容数={kb.get('content_count','0')}) {tag}")
        if owned and not include_owned:
            continue
        if kb_name and kb_name not in kb.get("kb_name", ""):
            continue
        targets.append(kb)
    if not targets:
        print("  没有匹配的知识库，退出。")
        return

    print(f"\n[2/4] 关键词定向检索：{len(keywords)} 个词 × {len(targets)} 个库 "
          f"（每词最多 {max_pages} 页）...")
    seen = load_seen(out_dir)
    captured_at = _now_iso()
    new_by_kw = {}
    throttled = False
    n_req_kw = 0

    for kw in keywords:
        seen.setdefault(kw, {})
        found_new = []
        for kb in targets:
            kb_id, kbn = kb.get("kb_id"), kb.get("kb_name", "kb")
            hits = _search_all(client_id, api_key, kb_id, kw, delay, max_pages)
            if hits is None:
                print(f"  ! 「{kw}」@「{kbn}」被频控中断，本轮跳过该词（已完成的词已落盘）。")
                throttled = True
                break
            for h in hits:
                mid = h.get("media_id")
                if not mid:
                    continue
                if mid in seen[kw]:
                    continue
                rec = {"title": h.get("title", mid), "kb_name": kbn,
                       "folder_id": h.get("parent_folder_id", ""),
                       "first_seen": captured_at}
                seen[kw][mid] = rec
                found_new.append({**rec, "media_id": mid,
                                  "highlight": h.get("highlight_content", "")})
        n_req_kw += 1
        if found_new:
            new_by_kw[kw] = found_new
        print(f"  · {kw}: 累计命中 {len(seen[kw])} 篇" +
              (f"，本轮新增 {len(found_new)} 篇" if found_new else "，无新增"))
        if throttled:
            break

    save_seen(out_dir, seen)
    csv_path = write_csv(out_dir, seen)
    total_new = sum(len(v) for v in new_by_kw.values())
    print(f"\n[3/4] 已扫描 {n_req_kw}/{len(keywords)} 个关键词；"
          f"基线累计 {sum(len(v) for v in seen.values())} 篇；本轮新增 {total_new} 篇。")
    print(f"    已写 {os.path.relpath(csv_path)}、keyword_seen.json")

    if init_only:
        print("    --init 模式：仅建立基线，不生成新增日报。")
        total_new = 0
    elif total_new:
        rp = append_report(out_dir, new_by_kw, captured_at)
        print(f"    已追加 {os.path.relpath(rp)}")
        print("\n  === 本轮新增（去 ima 客户端复制到自有库即可） ===")
        for kw, items in new_by_kw.items():
            for it in items:
                print(f"    + [{kw}] {it['title']}")
    else:
        print("    无新增，未追加日报。")

    if throttled:
        print("    ! 本轮遭遇频控，部分关键词未扫描；下次运行会自动补上（基线已保存，不会漏报）。")

    if not do_push:
        print("[4/4] 跳过 GitHub 推送（--no-push）。")
        return
    if not total_new:
        print("[4/4] 无新增，跳过 GitHub 推送。")
        return
    print("[4/4] 提交并推送到 GitHub ...")
    _git_commit_push(out_dir, total_new)


def _git_commit_push(out_dir, n_new):
    root = os.path.dirname(out_dir)
    os.chdir(root)
    os.system("git add ima_subscription/keyword_seen.json "
              "ima_subscription/keyword_hits.csv "
              "ima_subscription/keyword_watch.md "
              "ima_subscription/watch_keywords.json")
    msg = f"chore: ima keyword watch (+{n_new} new reports) via ima_keyword_watch.py"
    if os.system(f'git commit -m "{msg}"') != 0:
        print("  (无新改动或 commit 失败，跳过)")
        return
    if os.system("git push origin HEAD") != 0:
        print("  ! git push 失败，请检查 SSH/网络。")
    else:
        print("  已推送至 GitHub。")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keywords", default=None, help="本次使用的关键词（英文逗号分隔），不写回配置")
    p.add_argument("--add-keyword", default=None, help="往配置词表追加一个关键词")
    p.add_argument("--kb-name", default=None, help="只监听名称包含该子串的知识库")
    p.add_argument("--include-owned", action="store_true", help="连自有库一起监听")
    p.add_argument("--no-push", action="store_true", help="只生成清单，不推送 GitHub")
    p.add_argument("--delay", type=float, default=0.4, help="每次请求的礼貌间隔（秒）")
    p.add_argument("--max-pages", type=int, default=2, help="每个关键词最多取几页（省配额）")
    p.add_argument("--init", action="store_true", help="仅建立基线，不报新增（首次运行推荐）")
    p.add_argument("--dir", default=DEFAULT_OUT_DIR, help="产物输出目录")
    args = p.parse_args()

    client_id, api_key = load_credentials()
    if not client_id or not api_key:
        print("缺少 IMA 凭证：请在 .env 中配置 IMA_CLIENT_ID 与 IMA_API_KEY。")
        sys.exit(1)

    os.makedirs(args.dir, exist_ok=True)
    keywords = load_keywords(args.dir, args.keywords, args.add_keyword)
    if not keywords:
        print("关键词为空，退出。")
        sys.exit(1)

    watch(client_id, api_key, args.dir, keywords,
          do_push=not args.no_push, include_owned=args.include_owned,
          kb_name=args.kb_name, delay=args.delay, max_pages=args.max_pages,
          init_only=args.init)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa
        pass
    main()
