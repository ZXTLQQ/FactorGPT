"""因子精炼厂端到端验证脚本（离线 / 合成数据，无需 LLM 与网络）。

运行：python scripts/verify_refinery_pipeline.py
验证：六阶段流水线在合成数据上完整跑通，产出入选因子、复合因子与方法学报告。
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline.refinery import RefineryPipeline, RefineryConfig


def main() -> int:
    mp.freeze_support()

    # 小体量配置，保证 CI / 本地快速验证
    cfg = RefineryConfig(
        n_symbols=120,
        train_days=250,
        test_days=60,
        n_workers=2,
        seed=7,
        offline=True,
        output_dir="output/verify",
        rl_candidates=5,
        n_pool_seed=10,
    )

    print("[verify] 启动因子精炼厂六阶段流水线（离线合成）...")
    pipe = RefineryPipeline(cfg)
    result = pipe.run(requirement="混合反转与流动性，构建稳健 alpha")

    # 断言关键产物
    assert result.ore is not None, "数据底座缺失"
    assert len(result.candidates) > 0, "未生成候选因子"
    assert len(result.screened) > 0, "三级筛选后无入选因子"
    assert result.composite is not None, "AlphaPool 未合成复合因子"
    assert os.path.exists(result.report_path), f"方法学报告缺失: {result.report_path}"

    print("\n[verify] 阶段追踪：")
    for s in result.stage_trace:
        print(f"  - {s['stage']}: {s['note']}")

    print(f"\n[verify] 候选 {len(result.candidates)} → 入选 {len(result.screened)}")
    print(f"[verify] 复合 ICIR = {result.composite_metrics.get('icir', 0):+.3f}")
    print(f"[verify] 方法学报告：{result.report_path}")
    print("\n[verify] 全部断言通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
