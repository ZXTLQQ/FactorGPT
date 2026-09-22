"""把高频 L2 快照压实为离线数据（data/offline/hf_*.parquet）。

用法::

    python scripts/hf_offline_build.py                 # 按 config.yaml 的 data.offline.hf 构建
    python scripts/hf_offline_build.py --symbols au,ag  # 只压某几个品种
    python scripts/hf_offline_build.py --max-contracts 40 --freq 1min

原始快照 382MB / 1580 万行不适合进挖掘面板；本脚本产出三张小表，合计几 MB，
可随仓库分发，之后 ``source=offline`` 也能直接取高频数据。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm.client import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="构建离线高频数据表")
    ap.add_argument("--symbols", default="", help="只压这些品种，逗号分隔，如 au,ag")
    ap.add_argument("--max-contracts", type=int, default=0, help="面板纳入的合约数上限")
    ap.add_argument("--freq", default="", help="重采样频率，如 1min / 5min / 500ms")
    args = ap.parse_args()

    cfg = load_config()
    hf_cfg = ((cfg.get("data", {}) or {}).get("offline", {}) or {}).get("hf", {}) or {}
    if args.max_contracts:
        hf_cfg["max_contracts"] = args.max_contracts
    if args.freq:
        hf_cfg["freq"] = args.freq
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] or None

    from data.hf_adapter import HFDataSource
    from data.hf_panel import build_daily_table, build_minute_panel, load_orders

    hf = HFDataSource(config=cfg, file=hf_cfg.get("file") or "")
    if not hf.enabled:
        print(f"[hf_build] 高频 L2 文件不可用：{hf.file or '<未配置>'}")
        return 1
    print(f"[hf_build] 源文件：{hf.file}")

    panel = build_minute_panel(
        hf, symbols=symbols,
        freq=str(hf_cfg.get("freq") or "1min"),
        max_contracts=int(hf_cfg.get("max_contracts") or 24),
    )
    print(f"[hf_build] 分钟面板：{len(panel)} 行 / {panel['symbol'].nunique()} 合约"
          f" / {panel['date'].nunique()} 分钟")

    daily = build_daily_table(hf, panel=panel)
    orders = load_orders(hf_cfg.get("orders_file") or "")

    out = ROOT / "data" / "offline"
    out.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out / "hf_panel_1min.parquet", index=False)
    daily.to_parquet(out / "hf_daily.parquet", index=False)
    orders.to_parquet(out / "hf_orders.parquet", index=False)

    meta = {
        "source_file": str(hf.file),
        "source_orders": str(hf_cfg.get("orders_file") or ""),
        "freq": str(hf_cfg.get("freq") or "1min"),
        "panel_rows": len(panel),
        "panel_contracts": int(panel["symbol"].nunique()) if len(panel) else 0,
        "panel_minutes": int(panel["date"].nunique()) if len(panel) else 0,
        "panel_columns": list(panel.columns),
        "daily_rows": len(daily),
        "orders_rows": len(orders),
    }
    (out / "hf_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    for f in ("hf_panel_1min.parquet", "hf_daily.parquet", "hf_orders.parquet"):
        p = out / f
        print(f"[hf_build] 写出 {p}（{p.stat().st_size / 1024 / 1024:.2f} MB）")
    print(f"[hf_build] 高频特征列：{len([c for c in panel.columns if c.startswith('hf_')])} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
