"""知识库自动更新命令行工具。

用法示例：
  python scripts/update_knowledge.py --query "stock market factor model" --max 20
  python scripts/update_knowledge.py --dry-run            # 只打印，不写入
  python scripts/update_knowledge.py --source wechat        # 当前会提示需接入外部 API

说明：
  - 默认来源 arxiv（无需鉴权）；
  - wechat / research_report 为占位来源，需接入外部接口（见 src/rag/auto_update.py）。
"""
import argparse
import sys

sys.path.insert(0, "src")

from rag.auto_update import (  # noqa: E402
    fetch_arxiv,
    fetch_research_report,
    fetch_wechat,
    format_entries,
    save_knowledge,
)


def main():
    ap = argparse.ArgumentParser(description="FactorGPT 知识库自动更新")
    ap.add_argument("--query", default="factor investing quantitative factor stock selection alpha", help="检索词")
    ap.add_argument("--max", type=int, default=20, help="最大条数")
    ap.add_argument("--source", default="arxiv", choices=["arxiv", "wechat", "research_report"])
    ap.add_argument("--sort-by", default="submittedDate", help="arxiv 排序字段")
    ap.add_argument("--dry-run", action="store_true", help="只打印不写入")
    args = ap.parse_args()

    if args.source == "arxiv":
        papers = fetch_arxiv(query=args.query, max_results=args.max, sort_by=args.sort_by)
    elif args.source == "wechat":
        papers = fetch_wechat(args.query)
    else:
        papers = fetch_research_report(args.query)

    if not papers:
        print("[update_knowledge] 未获取到论文（可能是网络问题，或该来源尚未接入）。")
        return

    entries = format_entries(papers)
    print(f"[update_knowledge] 获取到 {len(entries)} 篇，示例标题：{papers[0]['title'][:60]}")
    if args.dry_run:
        print("[update_knowledge] --dry-run：未写入文件。")
        return
    n = save_knowledge(entries)
    print(f"[update_knowledge] 完成，写入 {n} 条。")


if __name__ == "__main__":
    main()
