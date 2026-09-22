"""端到端打通：L2 五档快照 → bar → 统一面板 → 表达式挖掘。

证明高频订单簿字段已经不是「旁路玩具」，而是挖进了同一套因子流水线：
它和日频量价共用 :class:`~mining.panel.FieldRegistry`、同一棵表达式树、
同一套 IC / 分组评价体系。和日频量价的区别只有两点：

- ``role`` 是 ``ROLE_ALT``（撮合前的另类信息），不是 ``ROLE_MARKET``（撮合后的结果）；
- 面板的时间栅格可以由调用方决定（30min / 5min / 1min），不再锁死在「日」。

用法::

    python scripts/hf_mining_demo.py --freq 30min --symbol au
    python scripts/hf_mining_demo.py --freq 1min  --symbol ag

注意频率衰减：bar 频率越低，订单簿信息被抹得越干净（见文档 4.3 节）。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from data.hf_adapter import HFDataSource  # noqa: E402
from mining import evaluator as EV  # noqa: E402
from mining import expr as ex  # noqa: E402
from mining.hf import build_l2_features, install_hf_features, register_hf_fields  # noqa: E402
from mining.panel import PanelData, default_registry  # noqa: E402

HF_FIELDS = ["obi_w5", "obi_l1", "spread_ticks", "micro_dev_ticks", "ofi", "rvol_20"]
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def build_long(pair_provider, contract: str, freq: str, fields):
    """把单合约快照按 bar 栅格聚成长表 ``(ts, symbol, 因子...)``。

    用 ``searchsorted`` 归桶而不是 ``resample``：后者会在非交易时段（午休、
    夜间休市）也造出空 bar，且与 K 线自身的时间戳容易对不齐。
    """
    raw = pair_provider.load_l2(contract)[contract]
    mk = pair_provider.get_minute_kline(contract, freq=freq)
    feats = build_l2_features(raw)
    cols = [c for c in fields if c in feats.columns]
    bar_ts = pd.DatetimeIndex(mk["datetime"])
    snap_ts = pd.to_datetime(raw["ts"]).to_numpy()
    pos = np.clip(np.searchsorted(bar_ts, snap_ts, side="right") - 1, 0, len(bar_ts) - 1)
    ok = snap_ts >= bar_ts[0]
    agg = (pd.DataFrame({c: feats[c].to_numpy() for c in cols})
             .assign(_bin=pos, _ok=ok).query("_ok").groupby("_bin")[cols].mean())
    agg.index = bar_ts[agg.index.to_numpy()]
    keep = agg.index                                  # 只保留真有快照支撑的 bar
    mk = mk[mk["datetime"].isin(keep)].reset_index(drop=True)
    out = agg.rename_axis("ts").reset_index()
    out["symbol"] = contract
    kline = mk.rename(columns={"datetime": "date"})
    kline["symbol"] = contract
    cols_keep = ["date", "symbol", "open", "high", "low", "close", "volume"]
    if "amount" in kline.columns:
        cols_keep.append("amount")
    return kline[cols_keep], out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="高频订单簿因子 → 统一挖掘面板")
    ap.add_argument("--file", default="", help="L2 快照 parquet，留空则查 config.yaml 的 data.hf")
    ap.add_argument("--symbol", default="au", help="品种代码（合约 < symbol + 月份）")
    ap.add_argument("--freq", default="30min", help="bar 频率，如 30min / 5min / 1min")
    ap.add_argument("--tolerance", default="600s",
                    help="陈旧观测耐受窗口：超过这个间隔就不允许前向填充")
    args = ap.parse_args(argv)

    cfg = {}
    try:
        import yaml
        with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"读取 config.yaml 失败（{exc}），退化为手工指定")
    src = HFDataSource(config=cfg, file=args.file or "")
    if not src.enabled:
        print("未找到高频 L2 数据文件：用 --file 指定绝对路径，"
              "或在 config.yaml 的 data.hf.file / search_dirs 里配置")
        return 1
    contracts = src.get_index_constituents(args.symbol)
    if not contracts:
        print(f"没有取到品种 {args.symbol} 的合约")
        return 1
    print(f"品种 {args.symbol}: {len(contracts)} 个合约 -> {contracts}")

    klines, hf_long = [], []
    for con in contracts:
        try:
            k, h = build_long(src, con, args.freq, HF_FIELDS)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {con}: {exc}")
            continue
        if len(k) and len(h):
            print(f"  {con}: {len(k)} 根有效 bar")
            klines.append(k)
            hf_long.append(h)

    kline = pd.concat(klines).sort_values(["date", "symbol"]).reset_index(drop=True)
    print(f"\nK 线 {args.freq}: {len(kline)} 行 / {kline['symbol'].nunique()} 合约 "
          f"/ {kline['date'].nunique()} 根")

    reg = default_registry()
    register_hf_fields(reg)
    panel = PanelData.from_kline(kline, forward_periods=(1, 3), registry=reg,
                                 name=f"hf-{args.symbol}")
    print("面板:", panel.n_dates, "根 bar,", len(panel.symbols), "合约")

    long_df = pd.concat(hf_long).dropna(subset=HF_FIELDS, how="all")
    got = install_hf_features(panel, long_df, fields=HF_FIELDS, tolerance=args.tolerance)
    print("装入面板的高频因子:", got)
    cov = panel.field("obi_w5").notna().mean(axis=1)
    print("截面覆盖率 均值=%.3f 最小=%.3f" % (cov.mean(), cov.min()))

    cfg = EV.EvalConfig(horizons=(1, 3), primary_horizon=1, min_stocks=3,
                        n_groups=3, rolling_ic_window=12, n_segments=2)
    print(f"\n{'表达式':<48} {'RankIC':>8} {'ICIR':>8} {'分数':>7} 评级")
    for text in ["zscore_cs(obi_w5)", "zscore_cs(micro_dev_ticks)",
                 "zscore_cs(spread_ticks)", "zscore_cs(ts_pct(close, 1))",
                 "mul(rank_cs(obi_w5), rank_cs(ts_pct(close, 1)))"]:
        errs = ex.validate(ex.parse(text), reg)
        if errs:
            print(f"{text:<48} 校验失败: {errs}")
            continue
        rep = EV.evaluate_expr(text, panel, config=cfg)
        m = rep.metrics
        flag = "" if rep.ok else "  无效: " + ";".join(rep.errors)
        print(f"{text:<48} {m.get('rank_ic_mean', float('nan')):>+8.4f} "
              f"{m.get('icir', float('nan')):>+8.3f} {rep.score:>7.3f} "
              f"[{rep.grade}]{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
