"""JEV 判定链自检：回答「它到底接上了吗、现在到底在用哪一级」。

JEV 的降级链是三级的，而**配了代码不等于在用**——``jev.enabled: true`` 且链路
全通的情况下，若 ``TYPESAFE_API_KEY`` 没配，实际干活的仍是第二级本地模型；
界面上不会报错，用户只会觉得"判定好像不太准"。本脚本把这件事变成可查的：

::

    python scripts/check_jev.py            # 只看本机状态，不联网
    python scripts/check_jev.py --probe    # 真的发一次请求（需 Key）

退出码：0 = 至少有一级可用；1 = 三级全不可用（此时上传材料的判定会退化到正则，
且不会再有任何提示）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SAMPLE = ("【研报】贵州茅台2025Q3点评：预计下半年需求回暖，营收同比增长15%，"
          "超市场预期。维持买入评级，目标价1800元。600519 2025-09-30")


def _mask(key: str) -> str:
    return (key[:4] + "…" + key[-4:]) if len(key) > 10 else ("*" * len(key))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="JEV 判定链自检")
    p.add_argument("--probe", action="store_true",
                   help="向 JEV 端点发一次真实请求（需要 TYPESAFE_API_KEY）")
    p.add_argument("--text", default=SAMPLE, help="自检用的样例材料文本")
    args = p.parse_args(argv)

    from llm.client import load_config

    cfg = load_config() or {}
    from engine.jev import JEVClient

    cli = JEVClient(cfg)
    print("=== 第一级：JEV 在线判定 ===")
    print(f"  配置启用      : {cli.enabled}")
    print(f"  端点          : {cli.base_url}")
    print(f"  模型          : {cli.model}")
    print(f"  密钥环境变量  : {cli.api_key_env}")
    print(f"  密钥已设置    : {bool(cli.api_key)}"
          + (f"（{_mask(cli.api_key)}）" if cli.api_key else "  ← 未设置，本级不参与"))
    print(f"  可调用        : {cli.callable}")

    if args.probe and cli.callable:
        try:
            answers = cli.ask(f"【文件名】probe.txt\n【内容】\n{args.text}")
            print(f"  实测返回      : {len(answers)} 个问题的答案 → 在线级可用")
            for k, v in list(answers.items())[:3]:
                print(f"      {k}: {v}")
        except Exception as e:  # noqa: BLE001
            print(f"  实测失败      : {type(e).__name__}: {e}")

    print("\n=== 第二级：本地训练模型 ===")
    local = cli._ensure_local()
    ok = local is not None
    print(f"  模型目录      : {(cfg.get('multimodal') or {}).get('model_dir')}")
    print(f"  已训练并加载  : {ok}"
          + (f"（{type(local).__name__}）" if ok else "  ← 未训练，跑 scripts/train_multimodal.py"))

    print("\n=== 第三级：本地正则 ===")
    print(f"  允许降级      : {cli.fallback}（恒定可用）")

    print("\n=== 实测：这份材料现在由谁判定 ===")
    res = cli.analyze(args.text, filename="probe.txt")
    print(f"  生效引擎      : {res.get('engine')}   ← 这就是线上真正在用的那一级")
    print(f"  判定模型      : {res.get('model')}")
    print(f"  类型 / 情绪   : {res.get('data_type_label')} / {res.get('sentiment_label')}")
    print(f"  可对齐 / 可因子化 : {res.get('alignable')} / {res.get('factorizable')}")
    print(f"  前瞻信息      : {res.get('forward_looking')}")
    if cli.last_error:
        print(f"  降级原因      : {cli.last_error}")

    usable = cli.callable or ok or cli.fallback
    print("\n结论：" + ("判定链可用" if usable else "三级全不可用（上传材料将无任何结构化判定）"))
    return 0 if usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
