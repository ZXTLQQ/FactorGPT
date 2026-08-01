"""小模型蒸馏脚本 (scripts/distill_factors.py)

方向三c：用已落地的 data/learned_factors.jsonl（已学习因子库）作为语料，
构造「自然语言需求 -> 因子代码」的指令微调数据集，并给出 LoRA 微调样例，
把云端大模型能力蒸馏到本地小模型（如 Qwen2.5-Coder），
摆脱每次依赖云端、利于答辩现场离线演示。

用法：
    # 仅构造数据集（零依赖，可立即跑）
    python scripts/distill_factors.py --src data/learned_factors.jsonl --out data/distill_train.jsonl

    # 在已装 peft/transformers 的环境执行 LoRA 微调
    python scripts/distill_factors.py --train --base Qwen/Qwen2.5-Coder-7B --out data/distill_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import os


def build_dataset(src: str, out: str, max_samples: int | None = None) -> int:
    """读 learned_factors.jsonl，构造 SFT 数据集（instruction / input / output）。"""
    rows = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if max_samples:
        rows = rows[:max_samples]

    instruction = (
        "你是 A股 量化因子工程师。根据需求生成可直接用于回测的因子代码："
        "函数名为 alpha_factor(df)，入参 df 为含 date/symbol/open/high/low/close/"
        "volume/amount/pct_chg 的长表，返回含 date/symbol/factor 三列的长表。"
        "禁止前视（不得引用未来信息），因子值应在当日收盘后即可计算。"
    )
    out_rows = []
    for r in rows:
        name = r.get("title") or r.get("name") or "因子"
        desc = r.get("description") or r.get("rationale") or ""
        code = r.get("code") or ""
        if not code:
            continue
        inp = f"因子名：{name}\n需求说明：{desc}"
        out_rows.append({"instruction": instruction, "input": inp, "output": code})

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[distill] 生成 {len(out_rows)} 条微调样本 -> {out}")
    return len(out_rows)


def train_lora(data_path: str, base_model: str, output_dir: str,
               epochs: int = 3, lora_r: int = 16, batch_size: int = 4) -> None:
    """LoRA 微调（可选依赖 peft + transformers）。未安装时给出提示并退出。"""
    try:
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        raise SystemExit(
            "蒸馏训练需要 peft/transformers/datasets，请先安装：\n"
            "  pip install peft transformers datasets accelerate\n"
            f"（导入失败：{e}）\n"
            "若只需生成数据集，直接运行不带 --train 的命令即可。"
        )
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto")
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    ds = load_dataset("json", data_files=data_path, split="train")

    def tok_fn(ex):
        prompt = f"### 指令\n{ex['instruction']}\n### 输入\n{ex['input']}\n### 输出\n"
        text = prompt + ex["output"] + tok.eos_token
        return tok(text, truncation=True, max_length=1024)

    ds = ds.map(tok_fn)
    args = TrainingArguments(
        output_dir=output_dir, per_device_train_batch_size=batch_size,
        num_train_epochs=epochs, logging_steps=5, save_strategy="epoch",
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, tokenizer=tok)
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[distill] LoRA 微调完成 -> {output_dir}")
    print("[distill] 部署：把 output_dir 作为 Ollama/vLLM 端点接入 config.yaml 的 llm.router.draft")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/learned_factors.jsonl")
    ap.add_argument("--out", default="data/distill_train.jsonl")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--train", action="store_true", help="执行 LoRA 微调（需 peft/transformers）")
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-7B")
    ap.add_argument("--output_dir", default="models/distilled_factor_coder")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    n = build_dataset(args.src, args.out, args.max_samples)
    if args.train:
        if n == 0:
            raise SystemExit("无可用样本，无法训练")
        train_lora(args.out, args.base, args.output_dir, epochs=args.epochs)


if __name__ == "__main__":
    main()
