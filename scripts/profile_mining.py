"""挖掘层性能剖析：先分期计时，再对热点阶段做 cProfile 热点排行。

这是"要不要上 C++"的唯一依据——**先量，再决定**。分期计时的意思是：
 pytime / cProfile 分开跑，前者给出各阶段占比（哪些阶段值得优化），
 后者给出具体热点函数（优化要落到哪几行）。

    python scripts/profile_mining.py                     # 默认 120 × 1050、搜索 30s
    python scripts/profile_mining.py --max-seconds 10    # 快速冒烟
    python scripts/profile_mining.py --top 25            # 打印更多热点
    python scripts/profile_mining.py --no-acceptance     # 不跑验收闸门

输出两个阶段：
1. 阶段墙钟表（含占比）——如果占比最高的是面板构造或 LLM/IO，那 C++ 帮不上忙；
2. 热点函数排行（累计 tottime 降序，已过滤解释器等无关帧）。
"""
from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

try:
    from cli_utf8 import force_utf8_stdio
except ImportError:  # pragma: no cover
    def force_utf8_stdio() -> None:  # type: ignore[misc]
        return None

from mining import evaluator as EV  # noqa: E402
from mining import fundamental as FD  # noqa: E402
from mining import gridminer as GM  # noqa: E402
from mining import panel as PN  # noqa: E402
from mining import report as RP  # noqa: E402
from mining import risk as R  # noqa: E402
from mining import triage as TR  # noqa: E402

# 热点排行里不需要看到的帧：剖析自身的开销会淹没真正的热点
_NOISE_PREFIXES = (
    "profile_", "cProfile", "pstats", "argparse", "importlib", "runpy",
    "<frozen importlib",
)


def build_panel(n_symbols: int, n_days: int, seed: int):
    """与 mining_report_demo 同口径的合成面板（含 PIT 财务与概念字段）。"""
    from mining import concept as CC

    n_days = max(n_days, FD.MIN_HISTORY_DAYS)
    pn = PN.PanelData.synthetic(n_symbols=n_symbols, n_days=n_days, seed=seed)
    R.install_risk_fields(pn)
    quarterly = FD.synthetic_quarterly(pn, seed=seed)
    FD.install_fundamentals(pn, quarterly, lag_days=1)
    members = CC.synthetic_concepts(pn, n_concepts=40, seed=seed)
    CC.install_concepts(pn, members)
    return pn, quarterly


def _hotspots(fn: Callable[[], Any], top: int, min_rows: int = 12) -> str:
    """对 ``fn`` 做 cProfile，返回可读的热点排行（按 tottime 降序）。"""
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    buf = io.StringIO()
    st = pstats.Stats(pr, stream=buf).sort_stats("tottime")
    st.print_stats(max(min_rows, top * 2))
    # 只保留表头与函数行，丢掉 pstats 的调用树空行
    lines: List[str] = []
    for line in buf.getvalue().splitlines():
        s = line.strip()
        if not s or s.startswith("Ordered by") or s.startswith("ncalls"):
            continue
        if s.startswith("function calls") or "primitive calls" in s:
            continue
        if any(line.strip().startswith(p) for p in _NOISE_PREFIXES):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines[:top])


def _stage(name: str, fn: Callable[[], Any], acc: List[Tuple[str, float]]) -> Any:
    t = time.perf_counter()
    out = fn()
    acc.append((name, time.perf_counter() - t))
    return out


def main() -> int:
    force_utf8_stdio()   # Windows 控制台默认 cp1252，中文 print 会直接崩进程
    ap = argparse.ArgumentParser(description="挖掘层性能剖析（分期计时 + 热点排行）")
    ap.add_argument("--symbols", type=int, default=120)
    ap.add_argument("--days", type=int, default=1050)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--max-expr", type=int, default=200)
    ap.add_argument("--top", type=int, default=15, help="热点函数打印条数")
    ap.add_argument("--no-acceptance", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    stages: List[Tuple[str, float]] = []
    t_all = time.perf_counter()

    print("[1/5] 构造合成面板与 PIT 字段 ...")
    panel, _ = _stage("面板构造（含 PIT/概念安装）",
                      lambda: build_panel(args.symbols, args.days, args.seed), stages)
    pool = {k: panel.field(k) for k in
            ("size", "momentum", "liquidity", "reversal", "resid_vol")
            if k in panel.fields}

    print("[2/5] 网格搜索剖析 ...")
    cfg = GM.SearchConfig(horizon=5, max_layers=2, width=12,
                          max_expr=args.max_expr, max_seconds=args.max_seconds,
                          min_ic=0.02, top_k=8, neutral_controls=("size",))

    # 搜索本体要计时；热点排行单独再跑一次（剖析会放大开销，不能混在一起来量墙钟）
    search = _stage("网格搜索 mine()",
                    lambda: GM.mine(panel, config=cfg, pool=pool), stages)
    hot_search = _hotspots(
        lambda: GM.mine(panel, config=cfg, pool=pool), top=args.top)

    print("[3/5] 表达式求值剖析 ...")
    exprs = [
        "zscore_cs(ts_pct(close, 20))",
        "dgtw_cs(ts_corr(close, volume, 60), size, 5)",
        "zscore_cs(ts_stddev(ts_rank(close, 20), 10))",
        "rank_cs(ts_mean(liquidity, 15))",
    ]
    reports = _stage("四维评价 evaluate_expr ×4",
                     lambda: [EV.evaluate_expr(t, panel, pool=pool, with_risk=True)
                              for t in exprs], stages)
    hot_eval = _hotspots(
        lambda: [EV.evaluate_expr(t, panel, pool=pool, with_risk=False)
                 for t in exprs * 5], top=args.top)

    print("[4/5] 验收闸门剖析 ...")
    if args.no_acceptance:
        acc: Dict[str, Any] = {"skipped": {"all": "--no-acceptance"}}
        hot_triage = ""
    else:
        best = max(reports, key=lambda r: r.score)
        acc = _stage("验收闸门 acceptance()",
                     lambda: TR.acceptance(panel, reports=reports, search=search,
                                           factor=best.detail.get("factor"), horizon=5,
                                           n_boot=args.n_boot, top_k=8,
                                           n_intervals=6, n_select=2), stages)
        hot_triage = _hotspots(
            lambda: TR.significance_for_search(search, top_k=8, n_boot=args.n_boot),
            top=args.top)

    print("[5/5] 报告渲染 ...")
    _stage("报告渲染 render_markdown()",
           lambda: RP.render_markdown(RP.ReportInput(panel=panel, reports=reports)),
           stages)

    total = time.perf_counter() - t_all
    print("\n" + "=" * 72)
    print(f"阶段墙钟（总计 {total:.2f}s，另有剖析开销不计入）")
    print("=" * 72)
    for name, sec in sorted(stages, key=lambda kv: -kv[1]):
        print(f"  {sec:8.2f}s  {100 * sec / total:5.1f}%  {name}")
    print(f"  {'-' * 60}")
    print(f"  {sum(s for _, s in stages):8.2f}s  100.0%  合计（未含剖析放大部分）")

    for title, body in (("网格搜索 mine()", hot_search),
                        ("表达式求值 evaluate_expr", hot_eval),
                        ("显著性 bootstrap", hot_triage)):
        if not body:
            continue
        print("\n" + "=" * 72)
        print(f"热点排行 · {title}（tottime 降序）")
        print("=" * 72)
        print(body)

    print("\n读法：tottime 高且来自本仓库 src/mining/*、ops.py、expr.py 的才是该动手的；"
          "\n落在 numpy/pandas 内部（已经是 C 实现）或 {method 'xxx' of ...} 的，"
          "\n说明瓶颈在调用次数/中间对象，优化方向是「少算」而不是「换语言」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
