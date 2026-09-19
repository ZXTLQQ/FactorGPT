"""离线生成一份 ``src/mining`` 因子研究报告（合成面板，无网络、无 API Key）。

跑通这一条命令就能看到挖掘层的全貌：PIT 财务字段安装与前视自检、概念数量
字段、算子网格搜索的层级轨迹与剪枝统计、单因子四维评价、风险闸门、相关性
与增量信息，最后汇总成一份 Markdown + JSON。

第 5 步是**验收闸门**（``mining.triage``）：统计显著性（多重性校正按本次搜索
实际评估过的表达式数计）、选股域对照（按面板自身成交额分位分三档）、分层多尺度
挖掘（粗网格演化 → 失真区间诊断 → 局部加密）；三者都缺省开启，可用
``--no-acceptance`` 关闭。

    python scripts/mining_report_demo.py                       # 默认 120 × 1050
    python scripts/mining_report_demo.py --max-seconds 30      # 缩短搜索
    python scripts/mining_report_demo.py --no-acceptance       # 只跑挖掘，不跑验收
    python scripts/mining_report_demo.py --out output/r.md     # 指定输出
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mining import concept as CC  # noqa: E402
from mining import evaluator as EV  # noqa: E402
from mining import fundamental as FD  # noqa: E402
from mining import gridminer as GM  # noqa: E402
from mining import panel as PN  # noqa: E402
from mining import report as RP  # noqa: E402
from mining import risk as R  # noqa: E402
from mining import triage as TR  # noqa: E402

PIT_FIELDS = ("revenue", "net_profit", "gross_profit", "cfo", "total_assets",
              "total_equity", "total_liab", "shares")


def build_panel(n_symbols: int, n_days: int, seed: int):
    """合成面板 + 财务/概念字段；面板长度不足 TTM 需要时会把窗口撑到最小可用。"""
    n_days = max(n_days, FD.MIN_HISTORY_DAYS)
    pn = PN.PanelData.synthetic(n_symbols=n_symbols, n_days=n_days, seed=seed)
    R.install_risk_fields(pn)
    quarterly = FD.synthetic_quarterly(pn, seed=seed)
    FD.install_fundamentals(pn, quarterly, lag_days=1)
    members = CC.synthetic_concepts(pn, n_concepts=40, seed=seed)
    CC.install_concepts(pn, members)
    return pn, quarterly


def print_acceptance(acc: dict) -> None:
    """把三项验收的关键数字打到终端（完整表格在报告里）。"""
    sig = acc.get("significance") or {}
    if sig:
        summ = sig.get("summary") or {}
        print(f"      显著性：{sig.get('n_candidates')} 个候选 / 本次搜索评估 "
              f"{sig.get('n_trials')} 次 → 通过四道门槛 {summ.get('passed', 0)} 个")
        if sig.get("warning"):
            print(f"      {sig['warning']}")
    dom = acc.get("domains")
    if dom is not None and len(dom):
        for _, r in dom.iterrows():
            print(f"      选股域 {r['domain']}：覆盖 {r['coverage_pct']:.1f}%、日均 "
                  f"{r['avg_per_day']:.1f} 只、IC {r['ic']:.4f}、平均换手 "
                  f"{100 * r['avg_turnover']:.1f}%")
    msr = acc.get("multiscale") or {}
    if msr:
        res = msr.get("resource") or {}
        print(f"      分层多尺度：{res.get('selected_intervals')}/"
              f"{res.get('total_intervals')} 个区间加密、评估节省 "
              f"{100 * float(res.get('eval_saving') or 0):.1f}%、真实求值 "
              f"{msr.get('total_evals')} 次、候选 {len(msr.get('candidates') or [])} 个")
    for key, why in (acc.get("skipped") or {}).items():
        print(f"      跳过 {key}：{why}")


def acceptance_note(acc: dict) -> str:
    """写进报告备注的验收口径说明（n_trials 与样本外用法最容易被含糊过去）。"""
    sig = acc.get("significance") or {}
    if not sig:
        return "验收三项未执行"
    return (f"验收口径：显著性按本次搜索实际评估的 {sig.get('n_trials')} 个表达式"
            f"做多重性校正；选股域按面板自身成交额分位分档；分层多尺度挖掘的"
            f"评估次数节省不等于墙钟节省")


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 mining 层因子研究报告（离线）")
    ap.add_argument("--symbols", type=int, default=120)
    ap.add_argument("--days", type=int, default=1050)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--max-expr", type=int, default=200)
    ap.add_argument("--out", default=os.path.join("demo_output", "mining_report.md"))
    ap.add_argument("--no-acceptance", action="store_true",
                    help="跳过显著性 / 选股域 / 分层多尺度挖掘三项验收")
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap 重采样次数")
    ap.add_argument("--top-k", type=int, default=8, help="参与显著性检验的候选数")
    ap.add_argument("--n-intervals", type=int, default=6, help="多尺度挖掘的区间数")
    ap.add_argument("--n-select", type=int, default=2, help="被选中加密的区间数")
    args = ap.parse_args()

    t0 = time.time()
    print("[1/6] 构造合成面板与 PIT 字段 ...")
    panel, quarterly = build_panel(args.symbols, args.days, args.seed)
    pool = {k: panel.field(k) for k in
            ("size", "momentum", "liquidity", "reversal", "resid_vol")
            if k in panel.fields}

    print("[2/6] 数据前置条件自检 ...")
    history = FD.check_history(panel)
    pit = [FD.check_pit(panel, quarterly, f, lag_days=1)
           for f in PIT_FIELDS if f in panel.fields]

    print("[3/6] 网格搜索（离线，合成数据）...")
    cfg = GM.SearchConfig(horizon=5, max_layers=2, width=12,
                          max_expr=args.max_expr, max_seconds=args.max_seconds,
                          min_ic=0.02, top_k=8,
                          neutral_controls=("size",))
    search = GM.mine(panel, config=cfg, pool=pool)

    print("[4/6] 代表性因子的四维评价 ...")
    exprs = [
        ("量价自建", "zscore_cs(ts_pct(close, 20))"),
        ("基本面质量", FD.FUND_FACTOR_LIBRARY["ROE_TTM_z"]),
        ("概念数量", CC.CONCEPT_FACTOR_LIBRARY["CN_z"]),
        ("分组调整", "dgtw_cs(ts_corr(close, volume, 60), size, 5)"),
    ]
    reports = [EV.evaluate_expr(text, panel, pool=pool, with_risk=True)
               for _, text in exprs]
    for (label, _), rep in zip(exprs, reports):
        rep.name = label

    best = max(reports, key=lambda r: r.score)
    print("[5/6] 验收闸门：统计显著性 / 选股域对照 / 分层多尺度挖掘 ...")
    if args.no_acceptance:
        acc: dict = {"skipped": {"all": "--no-acceptance"}}
    else:
        t_acc = time.time()
        acc = TR.acceptance(panel, reports=reports, search=search,
                            factor=best.detail.get("factor"), horizon=5,
                            n_boot=args.n_boot, top_k=args.top_k,
                            n_intervals=args.n_intervals, n_select=args.n_select)
        print(f"      验收用时 {time.time() - t_acc:.1f}s")
    print_acceptance(acc)

    print("[6/6] 汇总报告 ...")
    ri = RP.ReportInput(
        panel=panel,
        title="FactorGPT 挖掘层研究报告（合成数据）",
        reports=reports,
        search=search,
        history=history,
        pit_checks=pit,
        pools={r.name: EV.pool_correlation(r.detail["factor"], pool)
               for r in reports if r.ok},
        incremental={r.name: EV.incremental_ic(r.detail["factor"], pool, panel.fwd(5))
                     for r in reports if r.ok},
        corr=EV.factor_corr_matrix({r.name: r.detail["factor"]
                                    for r in reports if r.ok}),
        significance=acc.get("significance") or {},
        domains=acc.get("domains"),
        universe=acc.get("universe") or {},
        multiscale=acc.get("multiscale") or {},
        notes=[f"合成面板：{args.symbols} 标的 × {len(panel.dates)} 交易日",
               f"随机种子 {args.seed}（同种子同配置可复现）",
               f"搜索预算 max_expr={args.max_expr}, max_seconds={args.max_seconds}",
               f"最高分因子：{best.name}（{best.score:.3f} / {best.grade}）",
               acceptance_note(acc)],
    )
    paths = RP.write_report(os.path.join(ROOT, args.out), ri)
    print(f"\n完成，用时 {time.time() - t0:.1f}s")
    for kind, path in paths.items():
        print(f"  {kind:8s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
