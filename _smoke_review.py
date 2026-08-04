"""冒烟测试：验证人机协同两段式调用真实作用于 screened。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from pipeline.refinery import RefineryPipeline, build_refinery_config


def main():
    cfg = build_refinery_config({
        "offline": True,
        "use_real_data": False,
        "run_portfolio": False,
        "n_symbols": 12,
        "train_days": 160,
        "test_days": 40,
        "n_pool_seed": 8,
        "rl_candidates": 3,
        "rl_backend": "heuristic",
        "n_workers": 1,
        "screener": {"use_lasso": True, "use_human_collab": True,
                     "topk_ratio": 0.5, "min_keep": 2},
    })
    pipe = RefineryPipeline(cfg)
    ctx = pipe.run_to_review("冒烟测试")
    names = [c.name for c in ctx.candidates]
    print("候选因子:", names)

    keep = names[: max(2, len(names) // 2)]
    print("模拟人工保留:", keep)
    res = pipe.resume_from_review(ctx, keep_names=keep)
    print("入选:", [c.name for c in res.screened])
    print("审计:", res.screen_audit)
    assert all(c.name in keep for c in res.screened), "人工评审未真实生效！"
    print("PASS：人机协同评审真实作用于 screened")


if __name__ == "__main__":
    main()
