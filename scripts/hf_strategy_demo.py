"""四类高频策略的成本化实测。**唯一判据是扣成本后还剩下什么**。

四类家族（用户口径）与本代码的对应关系：

============  ==================================================  ==========
家族          本仓库实现                                           判据
============  ==================================================  ==========
短周期趋势     :func:`ofi_trend_signal` + :func:`evaluate_signal`  每笔净 tick
事件驱动       :func:`event_signal`（波动/价差/持仓量突变）         事件后中价漂移
跨市场套利     :func:`calendar_spread_signal`（同品种跨期）         价差回复
被动做市       :func:`estimate_adverse_selection`                  逆向选择 tick
============  ==================================================  ==========

用法::

    python scripts/hf_strategy_demo.py --family all
    python scripts/hf_strategy_demo.py --family trend --contract au2602

关于「跨市场」：这份数据只有单一交易所的快照，没有跨交易所/跨资产的数据，
因此可验证的是**同期品种的期限结构套利**（同一条曲线上的相对错误定价），
而不是字面意义的跨市场。这是数据边界决定的，不是实现偷懒。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from data.hf_adapter import HFDataSource  # noqa: E402
from mining.hf import build_l2_features, make_forward_labels  # noqa: E402
from mining.hf_strategies import (  # noqa: E402
    calendar_spread_signal,
    estimate_adverse_selection,
    evaluate_signal,
    event_signal,
    ofi_trend_signal,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _source(file: str, orders: str) -> tuple:
    cfg = {}
    try:
        import yaml
        with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:  # noqa: BLE001
        cfg = {}
    return HFDataSource(config=cfg, file=file or ""), cfg.get("data", {}).get(
        "hf", {}).get("orders_file", orders)


def run_trend(src, contract: str, sessions) -> None:
    print(f"\n[A] 短周期趋势 OFI z-score —— {contract}")
    data = src.load_l2(contract, sessions=sessions)
    if contract not in data:
        print("  无该合约行情"); return
    d = data[contract]
    feat = build_l2_features(d)
    print(f"  {'h':<4}{'entry':<7}{'cost':<6}{'笔数':>6}{'净收益tick':>12}{'胜率':>9}{'sharpe/笔':>11}")
    for h in (10, 20):
        y = make_forward_labels(d, horizon_ticks=h, kind="ret")
        for entry in (1.5, 2.0, 2.5):
            sig = ofi_trend_signal(feat, d, entry=entry)
            for cost in (0.0, 1.0):
                r = evaluate_signal(sig, y, cost_ticks=cost, hold_ticks=h)
                if not r.get("ok"):
                    print(f"  h={h} entry={entry} cost={cost}: {r.get('reason')}")
                    continue
                print(f"  {h:<4}{entry:<7}{cost:<6}{r['n_trades']:>6}"
                      f"{r['mean_pnl_ticks']:>+12.4f}{r['win_rate']*100:>8.1f}%"
                      f"{r['sharpe_per_trade']:>+11.4f}")


def run_event(src, contract: str, sessions) -> None:
    print(f"\n[B] 事件驱动 —— {contract} 事件后 10 tick 的中价变动")
    data = src.load_l2(contract, sessions=sessions)
    if contract not in data:
        print("  无该合约行情"); return
    d = data[contract]
    raw = event_signal(d)
    y10 = make_forward_labels(d, horizon_ticks=10, kind="ret")
    cases = {k: v for k, v in raw.items() if k != "oi_dir"}
    # 持仓量突变必须带方向看，否则正负事件互相抵消
    if "oi_dir" in raw:
        cases["oi_shock_signed"] = raw["oi_dir"].where(raw.get("oi_shock", False))
    base = y10.mean()
    for name, mask in cases.items():
        hit = pd.Series(mask).reindex(d.index).fillna(0.0).astype(float).to_numpy()
        if isinstance(mask, pd.Series) and mask.dtype == bool:
            hits = pd.Series(mask).reindex(d.index).fillna(False).astype(bool).to_numpy()
            sign = np.ones(len(hit))
        else:
            hits = hit != 0
            sign = hit
        if hits.sum() == 0:
            print(f"  {name}: 0 次"); continue
        fut = pd.Series(y10.to_numpy()[hits] * sign[hits])
        print(f"  {name:<16} {int(hits.sum()):>5} 次 ({hits.mean()*100:.3f}%)  "
              f"事件后漂移={fut.mean():+.4f} tick (全样本基准 {base:+.4f})  "
              f"绝对值={fut.abs().mean():.4f}")


def run_calendar(src, symbol: str, sessions) -> None:
    print(f"\n[C] 跨期套利 —— {symbol} 期限结构均值回归")
    term = src.get_term_structure(symbol, sessions=sessions)
    print(f"  期限结构: {term.shape}, 缺失率 {term.isna().mean().mean()*100:.2f}%")
    cols = list(term.columns)
    pairs = [(cols[0], cols[1])] if len(cols) >= 2 else []
    if len(cols) >= 3:
        pairs += [(cols[0], cols[-1]), (cols[1], cols[-1])]
    for near, far in pairs:
        cs = calendar_spread_signal(term, near, far, w=300, entry=2.0, exit_=0.5)
        if not cs.get("ok"):
            print(f"  {near}-{far}: {cs.get('reason')}"); continue
        spread = cs["spread"]
        dret = spread.diff().shift(-10)          # 未来 10 tick 的价差变动
        pos = cs["position"].reindex(spread.index)
        for cost in (0.0, 1.0):
            r = evaluate_signal(pos, dret, cost_ticks=cost, hold_ticks=10)
            if r.get("ok"):
                print(f"  {near}-{far} cost={cost}: 持仓tick={int((pos != 0).sum())} "
                      f"笔数={r['n_trades']:<5} 净收益={r['mean_pnl_ticks']:+.4f} "
                      f"胜率={r['win_rate']*100:.1f}%")


def run_market_making(src, orders_file: str, contract: str, sessions) -> None:
    print(f"\n[D] 被动做市 —— {contract} 逆向选择成本（正 = 成交后价格往不利方向走）")
    try:
        orders = HFDataSource.load_orders(orders_file)
    except Exception as exc:  # noqa: BLE001
        print(f"  读取委托流水失败: {exc}"); return
    data = src.load_l2(contract, sessions=sessions)
    if contract not in data:
        print("  无该合约行情"); return
    snap = data[contract]
    sub = orders[orders["contract"].astype(str).str.lower() == contract.lower()].copy()
    if not len(sub):
        print("  无委托样本"); return
    grid = pd.DataFrame({"ts": snap["ts"].to_numpy(),
                         "bp1": snap["bp1"].astype("float64").to_numpy(),
                         "sp1": snap["sp1"].astype("float64").to_numpy()}).sort_values("ts")
    s2 = sub.sort_values("ts").reset_index(drop=True)
    q = pd.merge_asof(s2[["ts"]], grid, on="ts", direction="backward",
                      tolerance=pd.Timedelta(5, unit="s"))
    s2 = pd.concat([s2, q[["bp1", "sp1"]]], axis=1)
    s2 = s2[s2["bp1"].notna() & s2["sp1"].notna()]
    buy = s2["side"].astype(float) == 1
    crossed = np.where(buy, s2["price"] >= s2["sp1"] - 1e-9,
                       s2["price"] <= s2["bp1"] + 1e-9)
    s2["kind"] = np.where(crossed, "aggressive", "passive")
    print(f"  委托 {len(sub)} 笔，可对齐 {len(s2)} 笔 -> {s2['kind'].value_counts().to_dict()}")
    for kind in ("passive", "aggressive"):
        g = s2[(s2["kind"] == kind) & (s2["filled"] == 1)]
        if len(g) < 10:
            print(f"  {kind}: 成交样本 {len(g)} 不足"); continue
        row = {h: estimate_adverse_selection(g, snap, horizon_ticks=h) for h in (10, 20)}
        print(f"  {kind:<11} 成交 {len(g)} 笔 | "
              + " | ".join(f"h={h} 逆向选择={e['mean_adverse_ticks']:+.4f} tick "
                           f"(不利占比 {e['share_positive']*100:.1f}%)"
                           for h, e in row.items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="四类高频策略的成本化实测")
    ap.add_argument("--family", default="all",
                    choices=["all", "trend", "event", "calendar", "mm"])
    ap.add_argument("--contract", default="au2602")
    ap.add_argument("--symbol", default="au")
    ap.add_argument("--file", default="")
    ap.add_argument("--orders", default="")
    ap.add_argument("--sessions", default="day", choices=["day", "all"],
                    help="day=仅日盘 M/E（成交概率建模必须用日盘）")
    args = ap.parse_args(argv)

    src, orders_file = _source(args.file, args.orders)
    if not src.enabled:
        print("未找到高频 L2 数据文件：--file 或 config.yaml 的 data.hf.file")
        return 1
    sessions = ("M", "E") if args.sessions == "day" else None
    fam = args.family
    if fam in ("all", "trend"):
        for con in [args.contract, "au2608"]:
            if con == args.contract or fam == "all":
                run_trend(src, con, sessions)
    if fam in ("all", "event"):
        run_event(src, args.contract, sessions)
    if fam in ("all", "calendar"):
        run_calendar(src, args.symbol, sessions)
    if fam in ("all", "mm"):
        for con in [args.contract, "au2604", "ag2602"]:
            run_market_making(src, orders_file, con, sessions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
