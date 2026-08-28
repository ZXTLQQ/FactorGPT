"""精炼厂最终验证：cache_only 模式读取预备整矿缓存，跑完六阶段全流程。
运行：py -3 demo/_smoke_refinery_final.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline.refinery import RefineryPipeline, build_refinery_config

cfg = build_refinery_config({
    "offline": True,
    "use_real_data": True,
    "cache_only": True,          # 完全离线，仅读取 data/cache/real_ore.pkl
    "run_portfolio": False,
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

print("=== 精炼厂最终验证（cache_only + 真实缓存）===")
pipe = RefineryPipeline(cfg)
result = pipe.run(requirement="混合日频与月频，结合动量与反转")

print("\n=== 流水线阶段追踪 ===")
for s in result.stage_trace:
    print(f"  [{s['stage']}] 耗时 {s['elapsed_s']:.1f}s - {s['note']}")

print(f"\n=== 入选因子：{len(result.screened)} ===")
for c in result.screened:
    m = c.metrics
    print(f"  {c.name} 来源={c.source} ICIR={m.get('icir', 0):+.3f}")

cm = result.composite_metrics
print(f"\n=== 复合因子：ICIR={cm.get('icir', 0):+.3f} IC={cm.get('ic_mean', 0):+.3f} ===")
print(f"报告路径：{result.report_path}")
assert result.report_path, "未生成方法学报告"
print("=== 精炼厂最终验证通过 ===")
