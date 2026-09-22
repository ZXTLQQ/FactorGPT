"""把高频 L2 快照压实成「可直接进挖掘流水线」的横截面面板。

为什么需要这一层
----------------
原始快照是 **15,801,xxx 行 × 795 合约** 的 500ms 数据（382MB），它有三个性质使其
无法直接喂给因子挖掘：

1. **行是事件不是截面**：挖掘框架的 ``alpha_factor(df)`` 契约要求
   ``df`` 有 ``date / symbol`` 两列做横截面，快照没有天然的「同一时刻多标的」结构；
2. **内存**：整份读进来会直接把进程打死；
3. **不可分发**：原始文件在本机桌面，仓库里没有。

本模块做三件事，把高频数据真正补进离线数据层：

- :func:`build_minute_panel`：把若干合约的快照重采样成**分钟级横截面面板**
  （``date`` = 分钟时间戳，``symbol`` = 合约），并附带订单簿衍生列，
  这样 Agent 生成的 ``alpha_factor(df)`` 可以用 ``ofi / obi_l1 / depth_ratio``
  这类真正的高频列，名实相符；
- :func:`build_daily_table`：合约日 K + 日内高频统计（供日频链路复用）；
- :func:`materialize`：把上面两张表 + 委托流水写成 ``data/offline/`` 下的 parquet，
  体积从 382MB 压到几 MB，可随仓库分发；之后 ``source=offline`` 也能取高频。

面板列契约（与 OfflineDataSource.get_daily_kline 保持一致，额外追加高频列）
--------------------------------------------------------------------------
``date, symbol, open, high, low, close, volume, amount, pct_chg`` 为必需列，
高频列一律以 ``hf_`` 前缀暴露，避免与日频字段混淆。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from data.hf_adapter import HFDataSource
from mining.hf import build_l2_features

logger = logging.getLogger(__name__)

# 重采样聚合口径：不同性质的量不能一律取均值——
# 订单流是**流量**（区间内累计才有意义），价位/失衡是**状态**（取均值才是这段的代表值）。
_FLOW_COLS = ("ofi", "dq_b", "dq_s", "d_volume", "d_turnover", "oi_change")
_STATE_COLS = (
    "spread", "spread_ticks", "rel_spread", "micro_dev_ticks", "depth_ratio",
    "obi_l1", "obi_all", "obi_w2", "obi_w3", "obi_w5", "book_slope",
    "best_share_bid", "best_share_ask", "rvol_20", "trend_20", "reversal_5",
    "trade_rate_20", "signed_flow_20", "volume_impulse_20",
)

BASE_COLUMNS = ["date", "symbol", "open", "high", "low", "close",
                "volume", "amount", "pct_chg"]


def _pick_contracts(hf: HFDataSource, symbols: Optional[Sequence[str]],
                    max_contracts: int) -> List[str]:
    """按活跃度（快照数）挑选合约；显式给了品种则取该品种全部合约。"""
    inv = hf.inventory()
    if symbols:
        keys = [str(s).strip().lower() for s in symbols if str(s).strip()]
        sub = inv[inv["symbol"].astype(str).str.lower().isin(keys)
                  | inv["contract"].astype(str).str.lower().isin(keys)]
        if len(sub):
            inv = sub
    return inv.sort_values("snapshots", ascending=False)["contract"].head(max_contracts).tolist()


def _minute_agg(d: pd.DataFrame, feat: pd.DataFrame, freq: str) -> pd.DataFrame:
    """单合约：把快照 + 特征重采样成分钟行。

    逐列重采样而不是一次 ``agg(...)``：后者会产生 MultiIndex 列，
    重命名一旦漏掉某个键就会静默留下多级列头，下游取列直接 KeyError。
    """
    idx = pd.DatetimeIndex(d["ts"])
    last = pd.to_numeric(d.get("last_price"), errors="coerce")
    last.index = idx
    rp = last.resample(freq)
    out = pd.DataFrame({"open": rp.first(), "high": rp.max(),
                        "low": rp.min(), "close": rp.last()})

    def _src(col: str) -> Optional[pd.Series]:
        s = feat[col] if col in feat.columns else (d[col] if col in d.columns else None)
        if s is None:
            return None
        s = pd.to_numeric(s, errors="coerce")
        s.index = idx
        return s

    dv = _src("d_volume")
    out["volume"] = dv.resample(freq).sum() if dv is not None else np.nan
    dt = _src("d_turnover")
    out["amount"] = dt.resample(freq).sum() if dt is not None else np.nan

    for c in _STATE_COLS:
        s = _src(c)
        if s is not None:
            out[f"hf_{c}"] = s.resample(freq).mean()
    for c in _FLOW_COLS:
        s = _src(c)
        if s is not None:
            out[f"hf_{c}"] = s.resample(freq).sum()
    # 该分钟没有快照 → close 为 NaN，不要留下假行
    return out[out["close"].notna()]


def build_minute_panel(hf: HFDataSource,
                       symbols: Optional[Sequence[str]] = None,
                       freq: str = "1min",
                       max_contracts: int = 24,
                       sessions: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """构建分钟级横截面面板：``date`` 为分钟时间戳，``symbol`` 为合约。

    这里的「截面」是同一分钟的所有合约——挖掘框架按 ``date`` 分组做截面 IC，
    于是它天然变成「日内横截面因子」，与日频选股链路同构、无需改评估器。
    """
    contracts = _pick_contracts(hf, symbols, max_contracts)
    frames = []
    for con in contracts:
        try:
            got = hf.load_l2(con, sessions=sessions)
        except Exception as e:  # noqa: BLE001
            logger.warning("[hf_panel] %s 读取失败，跳过: %s", con, e)
            continue
        d = got.get(con)
        if d is None or not len(d):
            continue
        try:
            feat = build_l2_features(d)
        except Exception as e:  # noqa: BLE001
            logger.warning("[hf_panel] %s 特征生成失败，跳过: %s", con, e)
            continue
        m = _minute_agg(d, feat, freq)
        if not len(m):
            continue
        m = m.reset_index().rename(columns={"index": "date", "ts": "date"})
        m["symbol"] = con
        frames.append(m)
    if not frames:
        return pd.DataFrame(columns=BASE_COLUMNS)
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    # 分钟收益（%）：同一合约内按分钟递推，跨会话不串味（panel 已按时间排序，
    # 会话之间的分钟是空档，shift 会拿到上一会话末，故用同一自然日的判断兜底）
    prev_close = panel.groupby("symbol")["close"].shift(1)
    same_day = (panel.groupby("symbol")["date"].shift(1).str.slice(0, 10)
                == panel["date"].str.slice(0, 10))
    panel["pct_chg"] = ((panel["close"] / prev_close - 1.0) * 100.0).where(same_day)
    for c in ("volume", "amount"):
        panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0.0)
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_daily_table(hf: HFDataSource,
                      symbols: Optional[Sequence[str]] = None,
                      max_contracts: int = 200,
                      panel: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """合约日 K（L2 聚合）+ 日内高频统计。

    日 K 部分复用 ``HFDataSource.get_daily_kline``（口径：close=末笔 LastPrice、
    pct_chg 对前**结算**价）；高频统计部分来自分钟面板的日内聚合。
    """
    contracts = _pick_contracts(hf, symbols, max_contracts)
    daily = hf.get_daily_kline(contracts)
    if daily.empty:
        return daily
    if panel is None or panel.empty:
        return daily
    p = panel.copy()
    p["day"] = p["date"].str.slice(0, 10)
    num = [c for c in p.columns if c.startswith("hf_")]
    agg = {c: "mean" for c in num if c not in ("hf_d_volume", "hf_d_turnover", "hf_ofi")}
    for c in ("hf_d_volume", "hf_d_turnover", "hf_ofi"):
        if c in p.columns:
            agg[c] = "sum"
    st = p.groupby(["symbol", "day"]).agg(agg).reset_index().rename(columns={"day": "date"})
    st = st.rename(columns={c: f"{c}_day" for c in agg})
    out = daily.merge(st, on=["symbol", "date"], how="left")
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def load_orders(path: str | Path) -> pd.DataFrame:
    """委托流水（xlsx → 规范化表），失败返回空表而不是抛错。"""
    try:
        return HFDataSource.load_orders(str(path))
    except Exception as e:  # noqa: BLE001
        logger.warning("[hf_panel] 委托流水读取失败: %s", e)
        return pd.DataFrame()


def _resolve(cfg: dict, key: str, default: str) -> Path:
    val = str(cfg.get(key) or default).strip()
    p = Path(val)
    return p if p.is_absolute() else Path.cwd() / p


def materialize(config: Optional[dict] = None, *,
                panel_out: Optional[str] = None,
                daily_out: Optional[str] = None,
                orders_out: Optional[str] = None,
                meta_out: Optional[str] = None) -> Dict[str, object]:
    """把高频数据压实并写入离线目录（``data/offline``）。

    返回写入摘要（行数 / 体积 / 合约数），供脚本与 UI 展示。
    """
    cfg = ((config or {}).get("data", {}) or {}).get("offline", {}) or {}
    hf_cfg = cfg.get("hf", {}) or {}
    hf = HFDataSource(config=config, file=hf_cfg.get("file") or "",
                      orders_file=hf_cfg.get("orders_file") or "")
    if not hf.enabled:
        raise FileNotFoundError(
            f"高频 L2 文件不可用：{hf.file or '<未配置>'}（config.yaml → data.offline.hf.file）")

    freq = str(hf_cfg.get("freq") or "1min")
    max_contracts = int(hf_cfg.get("max_contracts") or 24)
    panel = build_minute_panel(hf, freq=freq, max_contracts=max_contracts)
    daily = build_daily_table(hf, panel=panel)
    orders = load_orders(hf_cfg.get("orders_file") or "")

    p_panel = _resolve(hf_cfg, "panel_file", "data/offline/hf_panel_1min.parquet")
    p_daily = _resolve(hf_cfg, "daily_file", "data/offline/hf_daily.parquet")
    p_orders = _resolve(hf_cfg, "orders_parquet", "data/offline/hf_orders.parquet")
    p_meta = _resolve(hf_cfg, "meta_file", "data/offline/hf_meta.json")
    panel_out = Path(panel_out) if panel_out else p_panel
    daily_out = Path(daily_out) if daily_out else p_daily
    orders_out = Path(orders_out) if orders_out else p_orders
    meta_out = Path(meta_out) if meta_out else p_meta

    for p in (panel_out, daily_out, orders_out, meta_out):
        p.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(panel_out, index=False)
    daily.to_parquet(daily_out, index=False)
    orders.to_parquet(orders_out, index=False)

    meta = {
        "source_file": str(hf.file),
        "source_orders": str(hf_cfg.get("orders_file") or ""),
        "freq": freq,
        "panel_rows": int(len(panel)),
        "panel_contracts": int(panel["symbol"].nunique()) if len(panel) else 0,
        "panel_minutes": int(panel["date"].nunique()) if len(panel) else 0,
        "panel_columns": [c for c in panel.columns],
        "daily_rows": int(len(daily)),
        "orders_rows": int(len(orders)),
        "built_from": str(hf_cfg.get("file") or ""),
    }
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def _mb(p: Path) -> float:
        return round(p.stat().st_size / 1024 / 1024, 2) if p.exists() else 0.0

    meta.update({"panel_mb": _mb(panel_out), "daily_mb": _mb(daily_out),
                 "orders_mb": _mb(orders_out),
                 "panel_path": str(panel_out), "daily_path": str(daily_out),
                 "orders_path": str(orders_out), "meta_path": str(meta_out)})
    return meta


# ── 用户接入自有高频表（不关心它原本叫什么列名）──────────────────────
# 用户手上的分钟/高频数据列名千奇百怪（date/datetime/time/成交时间、symbol/code/
# contract/instrument/合约…）。这里只做「认列 + 归一」，不做任何补零/推断，
# 认不出来的列一律按数值列挂 hf_ 前缀，避免下游 KeyError。
USER_TABLE_EXTS = {".csv", ".txt", ".parquet", ".xlsx", ".xls", ".pqt"}

_COLUMN_ALIASES: Dict[str, tuple] = {
    "date": ("date", "datetime", "time", "ts", "timestamp", "bar_time", "minute",
             "trade_time", "日期", "时间", "成交时间", "交易日"),
    "symbol": ("symbol", "code", "contract", "instrument", "ticker", "sec_code",
               "wind_code", "合约", "标的代码", "证券代码", "代码"),
    "open": ("open", "open_price", "开盘", "开盘价"),
    "high": ("high", "high_price", "最高", "最高价"),
    "low": ("low", "low_price", "最低", "最低价"),
    "close": ("close", "close_price", "last", "last_price", "price", "收盘", "收盘价",
              "最新价", "价格"),
    "volume": ("volume", "vol", "qty", "成交量", "手数"),
    "amount": ("amount", "turnover", "amt", "成交额", "金额"),
}


def _match_column(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    """按别名认列：先全等（忽略大小写/空白），再包含匹配。"""
    norm = {str(c).strip().lower().replace(" ", ""): str(c) for c in columns}
    for a in aliases:
        if a in norm:
            return norm[a]
    for a in aliases:
        for k, orig in norm.items():
            if a in k:
                return orig
    return None


def normalize_user_panel(df: pd.DataFrame) -> pd.DataFrame:
    """把用户自有的分钟/高频表规范化成挖掘面板契约。

    契约（与 :func:`build_minute_panel` 输出一致）：
    ``date / symbol / open / high / low / close / volume / amount / pct_chg``
    + 若干 ``hf_*`` 数值列。

    ``symbol`` **不做 zfill(6)**：用户的标的可能是期货合约（``AU2601``）或
    港股/ETF 代码，补零会静默错位。认不出 date/close 时抛 ValueError——
    与其造出一个看似能跑的假面板，不如让用户知道缺哪一列。
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=BASE_COLUMNS)
    cols = list(df.columns)
    mapping: Dict[str, str] = {}
    for std, aliases in _COLUMN_ALIASES.items():
        got = _match_column(cols, aliases)
        if got is not None:
            mapping[std] = got
    if "date" not in mapping or "close" not in mapping:
        raise ValueError(
            f"用户高频表缺少可识别的 date/close 列（实到列：{cols[:12]}）；"
            "请保证至少有『时间』与『价格』两列")
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[mapping["date"]], errors="coerce")
    out = out[out["date"].notna()]
    if out.empty:
        raise ValueError("用户高频表的时间列无法解析为时间戳")
    out["symbol"] = df.loc[out.index, mapping["symbol"]].astype(str).str.strip()
    close = pd.to_numeric(df.loc[out.index, mapping["close"]], errors="coerce")
    out["close"] = close
    for std in ("open", "high", "low"):
        out[std] = (pd.to_numeric(df.loc[out.index, mapping[std]], errors="coerce")
                    if std in mapping else close)
    for std in ("volume", "amount"):
        out[std] = (pd.to_numeric(df.loc[out.index, mapping[std]], errors="coerce").fillna(0.0)
                    if std in mapping else 0.0)
    used = set(mapping.values())
    for c in cols:
        if c in used or str(c).startswith("hf_"):
            continue
        s = pd.to_numeric(df.loc[out.index, c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        name = str(c)
        out[f"hf_{name}" if not name.startswith("hf_") else name] = s
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out = (out.drop_duplicates(["date", "symbol"], keep="last")
              .sort_values(["symbol", "date"]).reset_index(drop=True))
    prev = out.groupby("symbol")["close"].shift(1)
    same_day = (out.groupby("symbol")["date"].shift(1).str.slice(0, 10)
                == out["date"].str.slice(0, 10))
    out["pct_chg"] = ((out["close"] / prev - 1.0) * 100.0).where(same_day)
    return out[BASE_COLUMNS + [c for c in out.columns if c.startswith("hf_")]]


def load_user_panel(path: str | Path) -> pd.DataFrame:
    """读取用户接入的高频表（csv/parquet/excel）并规范化；不存在则抛错。

    这条路径的意义：用户不必把数据压成我们的私有格式，只要表里有「时间+价格+
    标的」，Agent 的高频分支就能直接跑（列映射失败会明确报缺哪一列）。
    """
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.is_file():
        raise FileNotFoundError(f"用户高频表不存在：{p}")
    ext = p.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(p, encoding="utf-8", encoding_errors="ignore")
    elif ext in {".parquet", ".pqt"}:
        df = pd.read_parquet(p)
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(p, sheet_name=0)
    else:
        raise ValueError(f"不支持的用户高频表格式：{ext}（支持 {sorted(USER_TABLE_EXTS)}）")
    return normalize_user_panel(df)


def read_offline_table(path: str | Path) -> pd.DataFrame:
    """读取离线高频表；不存在时返回空表（调用方负责降级提示）。"""
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.is_file():
        return pd.DataFrame()
    return pd.read_parquet(p)


def read_offline_meta(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


__all__ = [
    "BASE_COLUMNS", "USER_TABLE_EXTS", "build_minute_panel", "build_daily_table",
    "load_orders", "load_user_panel", "normalize_user_panel", "materialize",
    "read_offline_table", "read_offline_meta",
]
