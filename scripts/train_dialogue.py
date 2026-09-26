"""训练对话能力模型（意图头 + 跟进头）并落盘。

用法::

    python scripts/train_dialogue.py                       # 默认：合成语料
    python scripts/train_dialogue.py --per-template 80 --seed 1

产物在 ``data/models/dialogue/``（config.yaml → dialogue.model_dir）：
``vectorizer.json`` + ``nb_intent.npz`` + ``nb_followup.npz`` + ``training_report.json``。

训完之后：LLM 不可用时，``agent.intent`` 的判定链从
「LLM → 正则」变成「LLM → **本地模型** → 正则」——离线也能把「换个窗口」
这类指代补全成一句独立需求（``rewritten``），而不是把原话原样丢给流水线。
零依赖（只用 numpy），CI 里也能跑。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.dialogue_model import train_all  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练对话能力模型（意图 / 跟进）")
    p.add_argument("--out-dir", default="", help="产物目录（默认取 config.dialogue.model_dir）")
    p.add_argument("--per-template", type=int, default=0, help="每条模板生成的样本数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        from llm.client import load_config
    except Exception:  # noqa: BLE001
        load_config = lambda: {}  # noqa: E731
    cfg = load_config()

    report = train_all(
        cfg,
        out_dir=args.out_dir or None,
        per_template=args.per_template or 40,
        seed=args.seed or 20260926,
    )
    if args.quiet:
        return 0

    ds = report["dataset"]
    print(f"语料：{ds['rows']} 条（去重后），训练 {ds['train']} / 测试 {ds['test']}")
    print(f"产物目录：{report['config']['out_dir']}\n")
    head = f"{'模型':<20}{'准确率':>10}{'macro-F1':>12}{'样本':>8}"
    print(head)
    print("-" * len(head))
    for name, m in report["models"].items():
        print(f"{name:<20}{m.get('acc', 0):>10.4f}{m.get('macro_f1', 0):>12.4f}"
              f"{m.get('n', 0):>8}")

    hard = report["hard_node"]
    print(f"\n困难集（字面无任何意图提示词的样本，{hard.get('n', 0)} 条）：")
    if hard.get("n"):
        print(f"{'方法':<20}{'准确率':>10}{'macro-F1':>12}")
        print("-" * 42)
        for key, label in (("model", "本地模型"), ("rule", "正则兜底")):
            m = hard.get(key) or {}
            print(f"{label:<20}{m.get('acc', 0):>10.4f}{m.get('macro_f1', 0):>12.4f}")
        f = hard.get("followup") or {}
        print(f"{'跟进判定':<20}{f.get('precision', 0):>10.4f}{f.get('recall', 0):>12.4f}"
              "  ← 精确率 / 召回率")
    print("\n报告：", Path(report["config"]["out_dir"]) / "training_report.json")
    print(json.dumps(dict(report["config"]), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
