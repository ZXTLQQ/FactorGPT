"""训练图文识别模型（朴素贝叶斯 / Transformer / CNN / GNN）并落盘。

用法::

    python scripts/train_multimodal.py                     # 默认：合成语料 + 合成图片
    python scripts/train_multimodal.py --epochs 20 --text-per-class 300
    python scripts/train_multimodal.py --image-dir data/train_images  # 混入真实图片

产物在 ``data/models/multimodal/``（config.yaml → multimodal.model_dir）：
``vectorizer.json`` + ``nb_*.npz`` + ``transformer_text.pt`` / ``cnn_image.pt`` /
``gcn_material.pt`` + ``training_report.json``（每模型的 acc / macro-F1）。

训完之后：JEV 无 Key 时会自动先用这套本地模型判定（``engine="local-model"``），
本地模型也没有才回退正则；上传图片时 CNN 会给出版式类别（K线/表格/文本页/其他）。
torch 未安装时只训朴素贝叶斯，报告里写明跳过原因，不会报错退出。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from engine.multimodal_train import train_all  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练图文识别模型（NB/Transformer/CNN/GNN）")
    p.add_argument("--out-dir", default="", help="产物目录（默认取 config.multimodal.model_dir）")
    p.add_argument("--epochs", type=int, default=0, help="torch 模型训练轮数")
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.0)
    p.add_argument("--text-per-class", type=int, default=0, help="每类合成文本条数")
    p.add_argument("--image-per-class", type=int, default=0, help="每类合成图片张数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-dir", default="", help="真实图片目录（子目录名=标签）")
    p.add_argument("--text-dir", default="", help="真实文本目录（子目录名=标签，可选）")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        from llm.client import load_config
    except Exception:  # noqa: BLE001
        load_config = lambda: {}  # noqa: E731
    cfg = load_config()

    def _nz(v, default=None):
        return default if v in (0, 0.0, "", None) else v

    report = train_all(
        cfg,
        out_dir=_nz(args.out_dir),
        epochs=_nz(args.epochs),
        batch_size=_nz(args.batch_size),
        lr=_nz(args.lr),
        text_per_class=_nz(args.text_per_class),
        image_per_class=_nz(args.image_per_class),
        seed=_nz(args.seed),
        image_dir=args.image_dir or None,
        text_dir=args.text_dir or None,
    )
    if not args.quiet:
        print(f"torch 可用：{report['torch_available']}；文本样本 {report['dataset']['texts']} 条")
        print(f"产物目录：{report['config']['out_dir']}\n")
        head = f"{'模型':<28}{'准确率':>10}{'macro-F1':>12}{'样本':>8}"
        print(head)
        print("-" * len(head))
        for name, m in report["models"].items():
            if m.get("status") == "skipped":
                print(f"{name:<28}{'跳过':>10}  {m.get('reason', '')}")
                continue
            print(f"{name:<28}{m.get('acc', 0):>10.4f}{m.get('macro_f1', 0):>12.4f}"
                  f"{m.get('n', 0):>8}")
        print("\n报告：", Path(report["config"]["out_dir"]) / "training_report.json")
        print(json.dumps(dict(report["config"]), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
