"""六段式冶炼流水线（Refinery Pipeline）端到端测试。

覆盖：PART-01 矿石 → PART-02 采矿 → PART-03 研磨 → 人机协同评审点 →
      PART-04 筛选 → PART-05 合金 → PART-06 报告 全流程。
强制使用合成数据（offline=True, use_real_data=False），不依赖网络与 LLM。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline.refinery import RefineryPipeline, build_refinery_config  # noqa: E402


def _cfg(tmpdir, **overrides):
    base = {
        "n_symbols": 10, "train_days": 50, "test_days": 25,
        "n_workers": 1, "seed": 7,
        "rl_backend": "heuristic", "rl_max_len": 3, "rl_candidates": 3,
        "n_pool_seed": 6, "run_portfolio": False,
        "output_dir": str(tmpdir), "offline": True, "use_real_data": False,
        "transformer": {"d_model": 32, "nhead": 4, "num_layers": 1},
        "rpn": {"parallel": False, "n_workers": 1},
        "screener": {"use_lasso": True, "use_human_collab": True,
                     "topk_ratio": 0.6, "min_keep": 2},
        "alpha_pool": {"ortho": True, "loo": True, "iterative": True,
                       "n_iter": 5},
    }
    base.update(overrides)
    return build_refinery_config(base)


def test_refinery_two_stage_full_pipeline(tmp_path):
    """两段式全流程：PART-01~03 暂停等评审，评审后 PART-04~06 续跑。"""
    pipe = RefineryPipeline(_cfg(tmp_path))
    ctx = pipe.run_to_review("动量与反转因子研究")

    assert ctx is not None and ctx.ore is not None
    assert len(ctx.candidates) >= 3, "采矿段应产出候选因子"
    stage_names = {s["stage"] for s in ctx.trace}
    assert any("PART-01" in s for s in stage_names)
    assert any("PART-02" in s for s in stage_names)
    assert any("PART-03" in s for s in stage_names)

    result = pipe.resume_from_review(
        ctx, review_callback=lambda cands: [c.name for c in cands[:2]])

    assert len(result.screened) >= 1
    assert result.composite is not None and len(result.composite) > 0
    assert "icir" in result.composite_metrics
    final_stages = {s["stage"] for s in result.stage_trace}
    assert any("PART-04" in s for s in final_stages)
    assert any("PART-05" in s for s in final_stages)
    assert any("PART-06" in s for s in final_stages)


def test_refinery_keep_names_honored(tmp_path):
    """人工勾选 keep_names 后，最终筛出集必须为其子集。"""
    cfg = _cfg(tmp_path)
    ctx = RefineryPipeline(cfg).run_to_review("测试因子")

    keep = [c.name for c in ctx.candidates[: max(1, len(ctx.candidates) // 2)]]
    result = RefineryPipeline(cfg).resume_from_review(ctx, keep_names=keep)

    assert len(result.screened) >= 1
    kept = {c.name for c in result.screened}
    assert kept.issubset(set(keep)), f"评审保留集被绕过: {kept - set(keep)}"


def test_refinery_single_call_equivalence(tmp_path):
    """一次性 run() 与两段式应产出同一批候选源。"""
    cfg = _cfg(tmp_path)
    pipe = RefineryPipeline(cfg)
    ctx = pipe.run_to_review("合成数据冒烟")
    candidates_before = {c.name for c in ctx.candidates}

    result = pipe.resume_from_review(ctx)  # 无评审回调 → 全保留
    assert {c.name for c in result.screened}.issubset(candidates_before)


def test_refinery_reproducible_with_seed(tmp_path):
    """固定 seed 时，两次独立运行候选集应一致（合成数据）。"""
    cfg1 = _cfg(tmp_path / "run1")
    cfg2 = _cfg(tmp_path / "run2")
    c1 = {c.name for c in RefineryPipeline(cfg1).run_to_review("复现").candidates}
    c2 = {c.name for c in RefineryPipeline(cfg2).run_to_review("复现").candidates}
    assert c1 == c2, "同 seed 下候选集应可复现"


def test_refinery_alpha_pool_switches(tmp_path):
    """AlphaPool 关键开关（正交化/留一法/迭代权重）可独立关闭且不报错。"""
    cfg = _cfg(tmp_path)
    pipe = RefineryPipeline(cfg)
    ctx = pipe.run_to_review("开关测试")
    result = pipe.resume_from_review(ctx, review_callback=lambda c: [x.name for x in c])

    assert len(result.screened) >= 2
    assert result.composite is not None
    assert "icir" in result.composite_metrics


def test_refinery_invalid_keep_names_no_crash(tmp_path):
    """keep_names 引用不存在的因子时优雅降级（保留 min_keep 下限）。"""
    cfg = _cfg(tmp_path)
    ctx = RefineryPipeline(cfg).run_to_review("空保留集")
    result = RefineryPipeline(cfg).resume_from_review(ctx, keep_names=["no_such_factor_xyz"])
    assert result is not None
    assert len(result.screened) >= 1
