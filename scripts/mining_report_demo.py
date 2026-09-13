# -*- coding: utf-8 -*-
"""离线生成一份 ``src/mining`` 因子研究报告（合成面板，无网络、无 API Key）。

跑通这一条命令就能看到挖掘层的全貌：PIT 财务字段安装与前视自检、概念数量
字段、算子网格搜索的层级轨迹与剪枝统计、单因子四维评价、风险闸门、相关性
与增量信息，最后汇总成一份 Markdown + JSON。

    python scripts/mining_report_demo.py                       # 默认 120 × 1050
    python scripts/mining_report_demo.py --max-seconds 30      # 缩短搜索
    python scripts/mining_report_demo.py --out output/r.md     # 指定输出
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mining import concept as CC            # noqa: E402
from mining import evaluator as EV          # noqa: E402
from mining import fundamental as FD        # noqa: E402
from mining import gridminer as GM          # noqa: E402
from mining import panel as PN              # noqa: E402
from mining import report as RP             # noqa: E402
from mining import risk as R                # noqa: E402

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


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 mining 层因子研究报告（离线）")
    ap.add_argument("--symbols", type=int, default=120)
    ap.add_argument("--days", type=int, default=1050)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--max-expr", type=int, default=200)
    ap.add_argument("--out", default=os.path.join("demo_output", "mining_report.md"))
    args = ap.parse_args()

    t0 = time.time()
    print("[1/5] 构造合成面板与 PIT 字段 ...")
    panel, quarterly = build_panel(args.symbols, args.days, args.seed)
    pool = {k: panel.field(k) for k in
            ("size", "momentum", "liquidity", "reversal", "resid_vol")
            if k in panel.fields}

    print("[2/5] 数据前置条件自检 ...")
    history = FD.check_history(panel)
    pit = [FD.check_pit(panel, quarterly, f, lag_days=1)
           for f in PIT_FIELDS if f in panel.fields]

    print("[3/5] 网格搜索（离线，合成数据）...")
    cfg = GM.SearchConfig(horizon=5, max_layers=2, width=12,
                          max_expr=args.max_expr, max_seconds=args.max_seconds,
                          min_ic=0.02, top_k=8,
                          neutral_controls=("size",))
    search = GM.mine(panel, config=cfg, pool=pool)

    print("[4/5] 代表性因子的四维评价 ...")
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

    print("[5/5] 汇总报告 ...")
    best = max(reports, key=lambda r: r.score)
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
        notes=[f"合成面板：{args.symbols} 标的 × {len(panel.dates)} 交易日",
               f"随机种子 {args.seed}（同种子同配置可复现）",
               f"搜索预算 max_expr={args.max_expr}, max_seconds={args.max_seconds}",
               f"最高分因子：{best.name}（{best.score:.3f} / {best.grade}）"],
    )
    paths = RP.write_report(os.path.join(ROOT, args.out), ri)
    print(f"\n完成，用时 {time.time() - t0:.1f}s")
    for kind, path in paths.items():
        print(f"  {kind:8s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
