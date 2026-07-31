"""验证 sb3_contrib 的 MaskablePPO 已真实接入 RL 因子搜索（通过 refinery 入口，避免触发 agent 包顶层导入）。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline.refinery import RefineryPipeline, build_refinery_config


def main():
    rc = build_refinery_config({})
    rc.n_symbols = 80
    rc.train_days = 160
    rc.test_days = 40
    rc.n_workers = 2
    rc.rl_max_len = 4
    rc.rl_candidates = 4
    rc.rl_backend = "sb3"          # 强制走真 MaskablePPO 后端
    rc.offline = True
    rc.rpn.parallel = False

    pipe = RefineryPipeline(rc)
    result = pipe.run(requirement="混合日频与月频，结合短期反转与流动性")

    rl_cands = [c for c in result.candidates if c.source == "rl"]
    sb3_cands = [c for c in rl_cands if "SB3-MaskablePPO" in (c.description or "")]
    print(f"候选总数={len(result.candidates)}  RL候选={len(rl_cands)}  SB3真实MaskablePPO候选={len(sb3_cands)}")
    for c in rl_cands:
        print(f"  - {c.name} | src={c.source} | icir_proxy={c.metrics.get('icir_proxy'):.3f} | {c.description}")

    assert len(sb3_cands) >= 1, "未生成任何 SB3-MaskablePPO 组合因子"
    assert any("SB3-MaskablePPO" in (c.description or "") for c in result.candidates), "后端未使用真 MaskablePPO"
    print("SB3 接入验证通过")


if __name__ == "__main__":
    main()
