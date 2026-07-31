"""
FactorGPT 因子挖掘 Agent — 命令行演示入口

用法：
    # 单因子挖掘（LLM 矿场）
    python run_agent.py "请构建一个 20 日动量因子"

    # 六阶段因子精炼厂（离线演示，无需 API/网络）
    python run_agent.py --refinery "混合日频与月频，结合短期反转与流动性"
    python run_agent.py --refinery --offline "..."        # 显式离线
    python run_agent.py --refinery --no-offline "..."      # 接入 LLM 矿场

说明：
- 单因子模式优先通过 config.yaml 调用 DeepSeek 生成因子代码；
- 精炼厂模式串联「数据底座→三维生成→RPN 评估→三级筛选→AlphaPool→方法学总结」六道工序；
- 离线模式下用可复现合成数据演示全流程，便于 CI/演示；置 --no-offline 接入真实 LLM 矿场。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 将 src 目录加入 Python 路径，使 agent / engine / rag / llm 子包可被导入
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from llm.client import load_config
from agent.graph import FactorAgent


def run_single_factor(config, user_input: str) -> None:
    agent = FactorAgent(config)
    print("=" * 60)
    print("FactorGPT 因子挖掘 Agent 启动")
    print(f"需求：{user_input}")
    print("=" * 60)

    result = agent.run(user_input)

    print("\n" + "=" * 60)
    print("因子回测指标：")
    print("=" * 60)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("最终报告：")
    print("=" * 60)
    print(result["report"])

    chart_paths = result["state"].get("chart_paths") or []
    if chart_paths:
        print("\n" + "=" * 60)
        print("标准化回测图表：")
        print("=" * 60)
        for p in chart_paths:
            print(f"  - {p}")
    else:
        print("\n[图表] 本轮未生成图表（校验或回测未通过）")


def run_refinery(config, user_input: str, offline: bool) -> None:
    from pipeline.refinery import RefineryPipeline, build_refinery_config

    ref_cfg = build_refinery_config(config.get("refinery", {}))
    ref_cfg.offline = offline

    print("=" * 60)
    print("因子精炼厂 · 六阶段冶炼流水线启动")
    print(f"需求：{user_input or '(默认)'}")
    print(f"模式：{'离线演示' if offline else '接入 LLM 矿场'}")
    print("=" * 60)

    pipe = RefineryPipeline(ref_cfg)
    result = pipe.run(requirement=user_input or "")

    print("\n" + "=" * 60)
    print("流水线阶段追踪：")
    print("=" * 60)
    for s in result.stage_trace:
        print(f"  [{s['stage']}] 耗时 {s['elapsed_s']}s — {s['note']}")

    print("\n" + "=" * 60)
    print("入选因子（三级筛选后）：")
    print("=" * 60)
    for c in result.screened:
        m = c.metrics
        print(f"  {c.name:28s} 来源={c.source:11s} ICIR={m.get('icir', 0):+.3f} "
              f"稳定性={m.get('stability_score', 0):+.3f} 换手={m.get('turnover', 0):.3f}")

    print("\n" + "=" * 60)
    print("复合因子（AlphaPool 合成）：")
    print("=" * 60)
    cm = result.composite_metrics
    print(f"  ICIR={cm.get('icir', 0):+.3f}  IC={cm.get('ic_mean', 0):+.3f}  "
          f"稳定性={cm.get('stability_score', 0):+.3f}  换手={cm.get('turnover', 0):.3f}")
    if result.loo_result.get("enabled"):
        loo = result.loo_result
        print(f"  LOO 基础 ICIR={loo.get('base_icir', 0):+.3f}  "
              f"最强依赖因子={loo.get('most_dependent_factor')} "
              f"(剔除 ΔICIR={loo.get('most_dependent_drop', 0):+.3f})")

    print("\n" + "=" * 60)
    print(f"方法学总结报告：{result.report_path}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="FactorGPT 因子挖掘 / 精炼厂")
    parser.add_argument("requirement", nargs="*", help="因子挖掘需求描述")
    parser.add_argument("--refinery", action="store_true", help="运行六阶段因子精炼厂流水线")
    parser.add_argument("--offline", action="store_true", dest="offline", default=None,
                        help="精炼厂离线模式（默认）")
    parser.add_argument("--no-offline", action="store_false", dest="offline",
                        help="精炼厂接入 LLM 矿场（需 API/网络）")
    args = parser.parse_args()

    user_input = " ".join(args.requirement) or \
        "请构建一个 20 日动量反转因子，并做行业市值中性化处理"

    config = load_config()

    if args.refinery:
        offline = True if args.offline is None else args.offline
        run_refinery(config, " ".join(args.requirement), offline)
    else:
        run_single_factor(config, user_input)


if __name__ == "__main__":
    main()
