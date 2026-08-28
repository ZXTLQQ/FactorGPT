"""精炼厂六阶段流水线冒烟测试（模拟 demo/app.py Tab4 的调用参数）。
运行：py -3 demo/_smoke_refinery.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline.refinery import RefineryPipeline, build_refinery_config

cfg = build_refinery_config({
    "offline": True,
    "use_real_data": False,
    "run_portfolio": False,
    "n_symbols": 50,
    "train_days": 250,
    "test_days": 60,
    "n_pool_seed": 8,
    "rl_candidates": 3,
    "rl_backend": "heuristic",
    "n_workers": 1,
    "screener": {"use_lasso": True, "use_human_collab": False,
                 "topk_ratio": 0.3, "min_keep": 2},
    "alpha_pool": {"ortho": True, "loo": False, "iterative": False},
    "rpn": {"n_quantiles": 5, "forward_periods": 1,
            "commission": 0.001, "risk_free_rate": 0.03,
            "parallel": False},
})

print("=== 精炼厂冒烟测试启动 ===")
pipe = RefineryPipeline(cfg)
result = pipe.run(requirement="混合日频与月频，结合动量与反转")

print("\n=== 流水线阶段追踪 ===")
for s in result.stage_trace:
    print(f"  [{s['stage']}] 耗时 {s['elapsed_s']:.1f}s - {s['note']}")

print(f"\n=== 入选因子：{len(result.screened)} 个 ===")
for c in result.screened:
    m = c.metrics
    print(f"  {c.name} 来源={c.source} ICIR={m.get('icir', 0):+.3f}")

cm = result.composite_metrics
print(f"\n=== 复合因子：ICIR={cm.get('icir', 0):+.3f} IC={cm.get('ic_mean', 0):+.3f} ===")
print(f"报告路径：{result.report_path}")
print("=== 精炼厂冒烟测试通过 ===")
