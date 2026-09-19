"""因子挖掘基线对照（scripts/mining_baseline.py）。

一次挖掘改完参数以后，"变好了没有"不能靠感觉。这个脚本用**同一个合成面板、
同一个随机种子**跑两遍搜索：

* ``legacy``——旧口径：全样本搜索、筛选分只看 |IC|×|ICIR|、预算均匀随机、
  剪枝逐日全量比对；
* ``new``——新口径：search/confirm 硬切分 + purged walk-forward 复核、
  筛选分加分段一致性与增量折扣、预算按算子族 UCB 分配、剪枝日期抽样。

两组都放到**同一个确认段**上复核（``mining.split`` 强制切分），这样能直接
读出"旧口径的 IC 里有多少是选择偏差"。

用法::

    python scripts/mining_baseline.py                  # 默认 100×600，1500 条预算
    python scripts/mining_baseline.py --expr 8000 --days 1050
    python scripts/mining_baseline.py --only new       # 只跑新口径（省时间）
    python scripts/mining_baseline.py --compare        # 与上次落盘的基线对比

产出落盘到 ``data/mining_baseline.json``，控制台打印对照表。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from mining import evaluator as EV  # noqa: E402
from mining import expr as ex  # noqa: E402
from mining import gridminer as GM  # noqa: E402
from mining import risk as R  # noqa: E402
from mining import split as SP  # noqa: E402
from mining.panel import PanelData  # noqa: E402

OUT_PATH = os.path.join(_HERE, "..", "data", "mining_baseline.json")

# 旧口径：把所有新开关关掉
LEGACY_OVERRIDES = {
    "split_mode": "off",
    "seg_consistency": False,
    "incremental": False,
    "adaptive_budget": False,
    "prune_sample_step": 1,
}


def make_panel(n_symbols: int, n_days: int, seed: int) -> PanelData:
    pn = PanelData.synthetic(n_symbols=n_symbols, n_days=n_days, seed=seed)
    R.install_risk_fields(pn)
    return pn


def confirm_oos(panel: PanelData, cands: Sequence[GM.Candidate], horizon: int,
                n_folds: int, embargo: Optional[int] = None
                ) -> Optional[Dict[str, Any]]:
    """把任意一批候选放到同一个确认段上复核（新/旧口径共用，保证可比）。"""
    plan = SP.make_split(panel, n_folds=n_folds, horizon=horizon,
                         embargo=embargo, mode="split")
    if not plan.has_confirm:
        return None
    ev = ex.Evaluator(panel, panel.registry)
    fwd = panel.fwd(horizon)
    rows: List[Dict[str, float]] = []
    for c in cands:
        try:
            full = ev.run(c.node)
        except Exception:
            continue
        per = [SP.fold_ic(full, fwd, f) for f in plan.folds]
        ics = [p["ic_mean"] for p in per if np.isfinite(p["ic_mean"])]
        if not ics:
            continue
        rows.append({"is": float(c.ic_mean), "oos": float(np.mean(ics))})
    if not rows:
        return None
    is_ic = np.array([r["is"] for r in rows], dtype=float)
    oos = np.array([r["oos"] for r in rows], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        decay = np.where(np.abs(is_ic) > 1e-9, 1.0 - oos / is_ic, np.nan)
    decay = decay[np.isfinite(decay)]
    return {
        "n": len(rows),
        "is_abs_ic": float(np.mean(np.abs(is_ic))),
        "oos_abs_ic": float(np.mean(np.abs(oos))),
        "oos_pos_rate": float(np.mean(np.sign(oos) == np.sign(is_ic))),
        "median_decay": float(np.median(decay)) if decay.size else None,
        "worst_decay": float(np.max(decay)) if decay.size else None,
        "sign_flip_rate": float(np.mean(np.sign(oos) * np.sign(is_ic) < 0)),
    }


def pool_corr(res: GM.SearchResult) -> Optional[float]:
    """入围因子池两两 |相关| 的平均值（越小说明挖出来的越"不是同一个因子"）。"""
    pool = {k: v for k, v in res.pool.items() if v is not None}
    if len(pool) < 2:
        return None
    mat = EV.factor_corr_matrix(pool)
    arr = mat.to_numpy(dtype=float)
    iu = np.triu_indices(arr.shape[0], k=1)
    vals = arr[iu][np.isfinite(arr[iu])]
    return float(np.mean(vals)) if vals.size else None


def run_once(panel: PanelData, cfg: GM.SearchConfig) -> Dict[str, Any]:
    res = GM.mine(panel, config=cfg, pool={"size": panel.field("size")})
    reps = [r for r in res.reports if r.ok]
    is_ic = [abs(r.metrics.get("rank_ic_mean", float("nan"))) for r in reps]
    is_icir = [abs(r.metrics.get("rank_icir", float("nan"))) for r in reps]
    top = res.candidates[:max(0, cfg.top_k)]
    oos = confirm_oos(panel, top, cfg.horizon, cfg.n_confirm_folds, cfg.embargo)
    return {
        "n_evaluated": int(res.n_evaluated),
        "n_candidates": len(res.candidates),
        "elapsed": round(float(res.elapsed), 3),
        "stop_reason": res.stop_reason,
        "split_mode": (res.split or {}).get("mode"),
        "is_abs_ic": float(np.mean(is_ic)) if is_ic else None,
        "is_abs_icir": float(np.mean(is_icir)) if is_icir else None,
        "pool_mean_abs_corr": pool_corr(res),
        # 搜索自己的确认段结果（新口径才有；旧口径这里为空）
        "self_oos": ((res.oos or {}).get("summary") or None),
        "confirm_oos": oos,
        "history": [h.to_dict() for h in res.history],
    }


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _delta(a: Any, b: Any) -> str:
    """new 相对 legacy 的变化（越低越好的指标自动带方向说明由列名给出）。"""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return "-"
    if abs(a) < 1e-12:
        return "-"
    return f"{(b - a) / abs(a):+.1%}"


def print_table(res: Dict[str, Dict[str, Any]], keys: Sequence[str]) -> None:
    print("\n=== baseline (legacy vs new) ===")
    print(f"{'metric':<24}{'legacy':>12}{'new':>12}{'delta':>12}")
    for k in keys:
        a = res.get("legacy", {}).get(k)
        b = res.get("new", {}).get(k)
        print(f"{k:<24}{_fmt(a):>12}{_fmt(b):>12}{_delta(a, b):>12}")
    lo = res.get("legacy", {}).get("confirm_oos") or {}
    nw = res.get("new", {}).get("confirm_oos") or {}
    print(f"{'confirm_is_abs_ic':<24}{_fmt(lo.get('is_abs_ic')):>12}"
          f"{_fmt(nw.get('is_abs_ic')):>12}"
          f"{_delta(lo.get('is_abs_ic'), nw.get('is_abs_ic')):>12}")
    print(f"{'confirm_oos_abs_ic':<24}{_fmt(lo.get('oos_abs_ic')):>12}"
          f"{_fmt(nw.get('oos_abs_ic')):>12}"
          f"{_delta(lo.get('oos_abs_ic'), nw.get('oos_abs_ic')):>12}")
    print(f"{'confirm_median_decay':<24}{_fmt(lo.get('median_decay')):>12}"
          f"{_fmt(nw.get('median_decay')):>12}"
          f"{_delta(lo.get('median_decay'), nw.get('median_decay')):>12}")
    print(f"{'confirm_sign_flip':<24}{_fmt(lo.get('sign_flip_rate')):>12}"
          f"{_fmt(nw.get('sign_flip_rate')):>12}"
          f"{_delta(lo.get('sign_flip_rate'), nw.get('sign_flip_rate')):>12}")
    print("\nNOTE: legacy 的确认段是它搜索时**看过**的（全样本搜索），所以它的"
          "OOS 数字天然虚高、不可信；\n      new 的 OOS 才是真样本外。两者之差"
          "就是选择偏差的量级——不要拿 confirm_oos_abs_ic 说 new 更差。\n"
          "      真正该看的是：new 的 |OOS IC| 与它自己的 |IS IC| 之比（self_oos"
          " 的 median_decay 越接近 0 越好）。")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="因子挖掘基线对照")
    ap.add_argument("--symbols", type=int, default=100)
    ap.add_argument("--days", type=int, default=600)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--expr", type=int, default=1500, help="求值预算（条）")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--only", choices=("both", "legacy", "new"), default="both")
    ap.add_argument("--compare", action="store_true",
                    help="与 data/mining_baseline.json 里上次的结果对比")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args(argv)

    panel = make_panel(args.symbols, args.days, args.seed)
    base_cfg = GM.SearchConfig(horizon=5, max_layers=args.layers, width=12,
                               max_expr=args.expr, max_seconds=args.seconds,
                               min_ic=0.015, top_k=args.topk, seed=42)
    out: Dict[str, Dict[str, Any]] = {}
    if args.only in ("both", "legacy"):
        cfg = GM.SearchConfig(**{**base_cfg.to_dict(), **LEGACY_OVERRIDES})
        out["legacy"] = run_once(panel, cfg)
    if args.only in ("both", "new"):
        out["new"] = run_once(panel, base_cfg)

    prev = None
    if args.compare and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            prev = json.load(fh)

    print_table(out, ("n_evaluated", "elapsed", "n_candidates",
                      "is_abs_ic", "is_abs_icir", "pool_mean_abs_corr"))
    if prev:
        print("\n=== vs last saved baseline ===")
        print(f"{'metric':<24}{'prev':>12}{'now':>12}{'delta':>12}")
        for tag in ("legacy", "new"):
            for k in ("is_abs_ic", "elapsed", "pool_mean_abs_corr"):
                a = (prev.get(tag) or {}).get(k)
                b = (out.get(tag) or {}).get(k)
                print(f"{tag + '.' + k:<24}{_fmt(a):>12}{_fmt(b):>12}"
                      f"{_delta(a, b):>12}")

    payload = {"args": vars(args), "result": out,
               "panel": {"n_symbols": args.symbols, "n_days": args.days,
                         "seed": args.seed}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
