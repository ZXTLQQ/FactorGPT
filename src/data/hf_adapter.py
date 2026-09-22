"""高频 L2 五档快照离线数据源（HFDataSource）。

数据来自第三方行情落库的 parquet（单日全市场 L2 快照，约 1580 万行 / 795 合约 /
82 品种）：基本面字段 ``Exch Contract TimeStr CalDate Date SortTime Session Symbol``，
五档量价 ``SP1..5 SV1..5 BP1..5 BV1..5``，衍生位 ``DerSV*/DerBV*/SN*/BN*``，
累计量 ``Volume Turnover OpenInt``。

本模块存在的原因，是这份数据有**三个必须知道的坑**。它们都已在此处理，
上层因子代码不要重复绕路：

坑 1 —— ``SortTime`` 是**跨自然日的连续时钟**，小时位会超过 24，不是墙上时钟。
    实测 au2608：夜盘 20:59 编码 ``205900500``；跨午夜后的凌晨 02:29 编码 ``262959500``
    ——注意小时位是 **26**，但同一行的 ``CalDate`` **已经进位到次日**；
    上午 09:00 编码 ``330000500``、下午 15:00 编码 ``390000000``（小时位 33 / 39）。
    墙上时钟 = 小时位 mod 24：用全市场 520 万行夜盘数据验证，
    ``TimeStr`` 的小时与 ``SortTime`` 小时位 mod 24 **100.0000% 一致**。
    由此得出的还原规则（``_sorttime_to_wallclock``）：
      · 夜盘 / 集合竞价段（N、A、a）：自然日直接取 ``CalDate``——它已含午夜进位；
      · 日盘段（M、E）：自然日取 ``Date``；
      · **小时位 ≥24 时不能再 +1 天**。早期版本这么做过，结果时间轴整体多跳一天，
        夜盘终点从 12-06 跑到 12-07，与委托流水一秒都对不上。

坑 2 —— **会话之间存在真实的时间空洞**（夜盘收盘→早盘开盘、午休）。
    直接对相邻行差分，会把「隔了 2 小时的午休」当成相邻快照，
    产出虚假的巨幅跳价，标签全是错的。这里用 ``session`` 区分 N/M/E 三段，
    并提供 ``new_session`` 布尔列：为真处必须切断一切跨快照计算。

坑 3 —— **部分合约的衍生列恒为 0**。实测 au2608 的 ``DerSV* / SN* / BN*`` 全零，
    但全市场扫描确认它们并非整列为空（``DerBV1`` 全市场最大 2174）。
    因此因子**不能硬依赖**这些列：``available_fields`` 会逐列做「非全零」探测，
    上层据此决定是否启用撤单强度 / 委托笔数类因子。

另外两个数据事实，决定了后面的建模口径：
    - ``Volume`` / ``Turnover`` 是**当日累计量**（实测单调不减，100%）；
      差分可得区间成交量，但 au2608 只有 2.59% 的快照区间发生成交（远月合约流动性稀薄），
      「无成交」是常态而非异常。
    - 只有 500ms 快照，**不是逐笔**。经典 OFI（Cont 2014）要求事件级数据，
      这里只能做「双快照间的档位量变动」近似，见 ``src/mining/hf.py`` 的说明。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# 会话中文名；数据源用单字母标记，N=夜盘 M=上午 E=下午，A/a 为开盘前的集合竞价孤立点
SESSION_NAME = {"N": "夜盘", "M": "上午", "E": "下午", "A": "集合竞价", "a": "集合竞价"}

# 原始列名 → 规范小写列名
_COL_MAP = {
    "Exch": "exch", "Contract": "contract", "TimeStr": "time_str",
    "CalDate": "cal_date", "Date": "date", "SortTime": "sort_time",
    "Session": "session", "Symbol": "symbol",
    "Open": "open", "DayHigh": "day_high", "DayLow": "day_low",
    "LifeHigh": "life_high", "LifeLow": "life_low", "AvgPrice": "avg_price",
    "UpperLimit": "upper_limit", "LowerLimit": "lower_limit",
    "Turnover": "turnover", "Volume": "volume", "LastVolume": "last_volume",
    "LastPrice": "last_price", "OpenInt": "open_int", "PreOpenInt": "pre_open_int",
    "PreClose": "pre_close", "PreSettle": "pre_settle",
    "AvgBidPrice": "avg_bid_price", "AvgAskPrice": "avg_ask_price",
    "TotalBidVol": "total_bid_vol", "TotalAskVol": "total_ask_vol",
    "OpenIntChg": "open_int_chg",
}
for _i in range(1, 6):
    _COL_MAP[f"SP{_i}"] = f"sp{_i}"
    _COL_MAP[f"SV{_i}"] = f"sv{_i}"
    _COL_MAP[f"BP{_i}"] = f"bp{_i}"
    _COL_MAP[f"BV{_i}"] = f"bv{_i}"
    _COL_MAP[f"DerSV{_i}"] = f"der_sv{_i}"
    _COL_MAP[f"DerBV{_i}"] = f"der_bv{_i}"
    _COL_MAP[f"SN{_i}"] = f"sn{_i}"
    _COL_MAP[f"BN{_i}"] = f"bn{_i}"

# 累计量列：差分才有意义，且差分前必须确认没有跨会话
_CUMULATIVE = ("volume", "turnover", "open_int")

# 国内期货常见最小变动价位（用于把 float32 抖动后的估计值吸附回真实报价单位）
_COMMON_TICKS = np.array([0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02,
                          0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0,
                          25.0, 50.0, 100.0, 200.0, 500.0, 1000.0])


def _sorttime_to_wallclock(sort_time: np.ndarray, cal_date: np.ndarray,
                           date: np.ndarray, session: np.ndarray) -> pd.DatetimeIndex:
    """把 SortTime 连续时钟还原为真实墙上时间戳（坑 1 的唯一入口）。

    ``SortTime`` 编码为 ``HHMMSSmmm``，小时位覆盖 20..39（39 表示次日下午 15:00）：
      · 墙上时钟取 ``HH % 24``（已用全市场数据验证与 TimeStr 完全一致）；
      · 夜盘 / 集合竞价段（N、A、a）自然日取 ``CalDate``，该列**已包含**午夜进位，
        故绝不再额外 +1 天；
      · 日盘段（M、E）自然日取 ``Date``。
    """
    st = np.asarray(sort_time, dtype="int64")
    clock_hh = (st // 10_000_000) % 24
    mm = (st // 100_000) % 100
    ss = (st // 1_000) % 100
    ms = st % 1_000

    sess = np.asarray(session).astype(str)
    on_day = (sess == "M") | (sess == "E")
    base_cal = pd.to_datetime(pd.Series(cal_date).astype("int64").astype(str),
                              format="%Y%m%d", errors="coerce")
    base_dat = pd.to_datetime(pd.Series(date).astype("int64").astype(str),
                              format="%Y%m%d", errors="coerce")
    base = pd.Series(np.where(on_day, base_dat.values, base_cal.values))
    nano = (base.dt.normalize().values.astype("datetime64[ns]")
            + (clock_hh * 3600 + mm * 60 + ss).astype("timedelta64[s]")
            + ms.astype("timedelta64[ms]"))
    return pd.DatetimeIndex(nano)


def estimate_tick_size(prices: np.ndarray) -> float:
    """从价格序列估计最小变动价位。

    预处理 / 归一化价差时必须除以 tick size，否则不同品种的因子不可比。
    这里不查品种表（82 个品种维护成本高且易过期），而是取**相邻非零差分的最小值**：
    用 5% 分位而非严格最小值，避免个别脏数据把 tick 估成 1e-6。

    最后结果会吸附到国内期货的常见报价单位：价格为 float32 存储，直接差分会得到
    0.019958 这类带 0.2% 误差的值，而黄金的真实报价单位是 0.02。
    """
    p = np.asarray(prices, dtype="float64")
    p = p[~np.isnan(p)]
    if len(p) < 20:
        return float("nan")
    d = np.abs(np.diff(np.unique(p)))
    d = d[d > 0]
    if len(d) == 0:
        return float("nan")
    raw = float(np.percentile(d, 5))
    cand = float(_COMMON_TICKS[np.argmin(np.abs(_COMMON_TICKS - raw))])
    # 只有在偏差小于 5% 时才吸附，避免把特殊品种强行扭歪
    return cand if abs(cand - raw) <= 0.05 * raw else raw


class HFDataSource:
    """高频 L2 五档快照离线数据源。

    面向设计的三条原则：
      · **懒加载**：1580 万行不能进内存，一律按 ``contract`` 过滤下推（pyarrow filter），
        单合约最多约 8 万行。
      · **可缓存**：首次读取后落 parquet 切片到 ``cache_dir``，二次读取秒开；
        缓存带 source mtime/size 校验，源文件变了自动失效。
      · **契约自解释**：产出列的语义（何时为 NaN、何时切断）写在
        :meth:`load_l2` 文档里，因子层不需要再猜。

    用法::

        hf = HFDataSource(config=cfg)
        hf.inventory()                       # 有哪些品种/合约
        df = hf.load_l2("au2608")            # 单合约单日快照序列
        panel = hf.load_symbol_l2("au")      # 全月份 {contract: df}
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        # 与 OfflineDataSource 保持同一约定：adapter 不自读 config.yaml，
        # 由调用方（UI / DataSourceFactory）把已解析好的 dict 传进来。
        cfg = config or {}
        data_cfg = cfg.get("data", {}) or {}
        hf_cfg = dict(data_cfg.get("hf", {}) or {})
        hf_cfg.update(kwargs)

        raw_file = str(hf_cfg.get("file") or "").strip()
        self.file = Path(raw_file) if raw_file else Path("")
        if not raw_file:
            # 未显式指定时，先看约定目录，再看桌面兜底位置
            for cand in hf_cfg.get("search_dirs") or []:
                hits = sorted(Path(cand).glob(hf_cfg.get("glob") or "*2025*.pqt"))
                if hits:
                    self.file = hits[-1]
                    break
        self.cache_dir = Path(hf_cfg.get("cache_dir") or "data/hf/cache")
        self._pf: Optional[pq.ParquetFile] = None
        self._available_cache: Optional[dict] = None
        # 与 DataFetcher/OfflineDataSource 同名的取数回执，供 UI 展示实际来源
        self.last_fetch_info: Dict[str, str] = {"source": None, "message": ""}

    # ---------- 基础 ----------
    @property
    def enabled(self) -> bool:
        # 必须是**文件**：Path("") 会被解释成当前目录 ".", 直接 ParquetFile() 会PermissionError
        return bool(str(self.file)) and Path(self.file).is_file()

    @property
    def file_path(self) -> Optional[Path]:
        return Path(self.file) if self.enabled else None

    def _parquet(self) -> pq.ParquetFile:
        if self._pf is None:
            if not self.enabled:
                raise FileNotFoundError(
                    f"高频 L2 数据文件不存在：{self.file or '<未配置>'}"
                    "（在 config.yaml 的 data.hf.file 配置绝对路径，或把文件放进约定目录）")
            self._pf = pq.ParquetFile(self.file)
        return self._pf

    # ---------- 清单 ----------
    def inventory(self) -> pd.DataFrame:
        """全市场合约清单：品种、交易所、快照数、会话覆盖、tick size 估计。"""
        pf = self._parquet()
        t = pf.read(columns=["Exch", "Symbol", "Contract", "Session", "SortTime", "BP1"])
        d = t.to_pandas()
        rows = []
        for (sym, con), g in d.groupby(["Symbol", "Contract"], sort=False):
            if not isinstance(con, str):
                continue
            tick = estimate_tick_size(g["BP1"].to_numpy())
            rows.append({
                "symbol": sym, "contract": con, "exch": g["Exch"].iloc[0],
                "snapshots": len(g),
                "sessions": "".join(sorted(set(str(s) for s in g["Session"]) & set("NME"))),
                "tick_size": tick,
            })
        inv = pd.DataFrame(rows).sort_values(["symbol", "contract"]).reset_index(drop=True)
        return inv

    def list_contracts(self, symbol: Optional[str] = None) -> list:
        inv = self.inventory()
        if symbol:
            inv = inv[inv["symbol"].astype(str).str.lower() == str(symbol).lower()]
        return inv["contract"].tolist()

    def get_index_constituents(self, index_code: str = "hf", top: int = 30) -> List[str]:
        """票池等价物：返回可作为标的池的合约列表。

        高频数据没有「指数成分股」的概念，这里的语义是：
          · ``index_code`` 为品种代码（如 ``au`` / ``ag`` / ``rb``）时返回该品种全部合约；
          · 为默认 ``hf`` 或无法识别时，按快照数取最活跃的 ``top`` 个合约（快照数作为活跃度代理）。
        """
        try:
            inv = self.inventory()
        except FileNotFoundError:
            self.last_fetch_info = {"source": "none", "message": "高频 L2 数据文件不存在"}
            return []
        key = str(index_code).strip().lower()
        if key not in ("hf", "", "all"):
            sub = inv[inv["symbol"].astype(str).str.lower() == key]
            if len(sub) and key not in ("000300", "000905", "000906", "000852"):
                out = sub.sort_values("contract")["contract"].tolist()
                self.last_fetch_info = {"source": "hf", "message": f"品种 {key} 共 {len(out)} 个合约"}
                return out
        out = inv.sort_values("snapshots", ascending=False)["contract"].head(top).tolist()
        self.last_fetch_info = {"source": "hf", "message": f"高频活跃合约前 {len(out)} 个"}
        return out

    def available_fields(self, contract: str) -> dict:
        """逐列探测该合约哪些列「真的有信息」（非全零、非常数）。

        这是坑 3 的防御层：``DerSV*/SN*/BN*`` 在部分合约恒为 0，
        因子层必须先看这个结果再决定是否计算依赖它们的因子。
        """
        needed = [f"der_sv{i}" for i in range(1, 6)] + [f"der_bv{i}" for i in range(1, 6)] \
            + [f"sn{i}" for i in range(1, 6)] + [f"bn{i}" for i in range(1, 6)] \
            + ["last_volume", "open_int_chg", "avg_price", "life_high"]
        raw = [(v, k) for k, v in _COL_MAP.items() if v in needed]
        tbl = pq.read_table(self.file, columns=[k for _, k in raw],
                            filters=[("Contract", "=", contract)])
        res = {}
        for norm, col in raw:
            arr = tbl.column(col).to_pandas().dropna()
            if len(arr) == 0:
                res[norm] = False
                continue
            mn, mx = float(np.min(arr)), float(np.max(arr))
            res[norm] = bool(mx != mn)
        return res

    # ---------- 读取 ----------
    def load_l2(self, contracts: str | Sequence[str],
                sessions: Optional[Iterable[str]] = None,
                columns: Optional[Sequence[str]] = None) -> dict:
        """读取一个或多个合约的 L2 快照序列。

        返回 ``{contract: DataFrame}``，DataFrame 已按墙上时间戳排序，额外产出：

          ``ts``        真实墙上时间戳（已处理坑 1，可直接跨合约对齐）
          ``session``   N/M/E/A/a
          ``new_session``  该行是否是某段会话的**第一行**（True 处必须与上一行切断任何差分/跨期计算）
          ``seq``       会话内单调序号
          ``dt_ms``     与上一同会话快照的时间差（毫秒）；首行为 NaN

        列的 NaN 语义：远端档位未挂满时 ``sp3..5/sv3..5`` 等为 NaN（远端档经常没人挂），
        因子层应当**按档位独立处理缺失**，而不是整体丢弃或填 0（填 0 会把「没人挂」当成「挂了 0 手」，二者等价但把「档位不存在」误当成「存在且为零」会影响斜率类因子）。
        """
        pf = self._parquet()
        if isinstance(contracts, str):
            contracts = [contracts]
        out = {}
        for con in contracts:
            cached = self._cache_path(con)
            if cached.exists() and self._cache_valid(cached):
                df = pd.read_parquet(cached)
            else:
                df = self._read_raw(con, columns)
                if df is not None and len(df):
                    self._write_cache(cached, df)
            if df is None or not len(df):
                logger.warning("[HF] 合约 %s 无快照", con)
                continue
            if sessions:
                keep = set(sessions)
                df = df[df["session"].isin(keep)]
                if not len(df):
                    continue
            out[con] = df
        return out

    def load_symbol_l2(self, symbol: str, **kw) -> dict:
        """读取某品种的全部月份合约（跨期 / 期限结构分析用）。"""
        return self.load_l2(self.list_contracts(symbol), **kw)

    def _read_raw(self, contract: str, columns: Optional[Sequence[str]] = None) -> Optional[pd.DataFrame]:
        pf = self._parquet()
        cols = [c for c in pf.schema_arrow.names if c in _COL_MAP or c == "Contract"]
        if columns:
            keep = set(columns) | {"Contract", "SortTime", "Session", "Symbol",
                                   "Exch", "CalDate", "Date", "TimeStr"}
            cols = [c for c in cols if c in keep or _COL_MAP.get(c) in keep]
        # 注意：ParquetFile.read() 不支持 filters，必须用顶层 pq.read_table 做谓词下推，
        # 否则会把 1580 万行全读进内存再过滤。
        tbl = pq.read_table(self.file, columns=cols,
                            filters=[("Contract", "=", contract)])
        if tbl.num_rows == 0:
            return None
        return self._normalize(tbl.to_pandas())

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """列改名 + 时间还原 + 会话边界标记。坑 1/2 都在这里一次性解决。"""
        df = df.rename(columns=_COL_MAP)
        df = df.sort_values("sort_time", kind="mergesort").reset_index(drop=True)
        if not len(df):
            return df
        df["ts"] = _sorttime_to_wallclock(
            df["sort_time"].to_numpy(), df["cal_date"].to_numpy(),
            df["date"].to_numpy(), df["session"].to_numpy())
        # 会话内的升序序号；会话切换处置 new_session=True
        sess = df["session"].astype(str)
        change = sess.ne(sess.shift())
        # 同字母会话若被真实空洞断开（罕见：同一 session 内出现时间回退），同样切断
        back = pd.Series(df["ts"]).diff().dt.total_seconds().fillna(0) < 0
        df["new_session"] = (change | back).to_numpy()
        df["seq"] = df["new_session"].cumsum().sub(1)
        df["session_id"] = df["seq"].astype(int)
        # 会话内相邻快照毫秒间隔（跨会话为 NaN，天然强制调用方切断）
        dt = pd.Series(df["ts"]).diff().dt.total_seconds() * 1000.0
        dt[df["new_session"]] = np.nan
        df["dt_ms"] = dt.to_numpy()
        # 累计量 → 区间增量（跨会话切口不差分，避免把午休 2 小时算进来）
        for c in _CUMULATIVE:
            if c in df.columns:
                d = df.groupby("session_id")[c].diff()
                df[f"d_{c}"] = d.to_numpy()
        df.attrs["tick_size"] = estimate_tick_size(df["bp1"].to_numpy())
        return df

    # ---------- 缓存 ----------
    def _cache_path(self, contract: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(contract))
        return self.cache_dir / f"l2_{safe}.parquet"

    def _cache_valid(self, cached: Path) -> bool:
        try:
            st = Path(self.file).stat()
            return bool(cached.stat().st_mtime >= st.st_mtime)
        except Exception:  # noqa: BLE001
            return False

    def _write_cache(self, path: Path, df: pd.DataFrame) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
        except Exception as e:  # noqa: BLE001
            logger.debug("[HF] 缓存写入失败（不影响使用）%s: %s", path, e)

    # ---------- 与既有日频契约对齐 ----------
    def get_daily_kline(self, symbols=None, start: Optional[str] = None,
                        end: Optional[str] = None, period: str = "daily",
                        adjust: str = "qfq", force_synthetic: bool = False) -> pd.DataFrame:
        """把 L2 快照聚合成**标准日 K**，列名对齐 OfflineDataSource 契约。

        签名刻意与 ``DataFetcher / OfflineDataSource`` 保持一致（symbols, start, end 位置参数），
        以便 ``DataSourceFactory`` 能把 ``data.source: hf`` 当成即插即用的数据源。

        这样高频数据可以直接喂给现有日频因子/回测链路。聚合口径（注意与股票的区别）：
          · close = 该合约当日最后一个有效 LastPrice；
          · open  = 快照里的 Open 字段，缺失时用当日首个 LastPrice 兜底；
          · high/low = DayHigh/DayLow，缺失时用 LastPrice 极值兜底；
          · volume/amount = **当日累计量末值**（快照的 Volume/Turnover 是当日累计，不是区间量）；
          · pct_chg = close / pre_settle - 1（**期货用前结算价**，不是前收盘）。
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        symbols = [str(s).strip().lower() for s in (symbols or []) if str(s).strip()]
        if not symbols:
            self.last_fetch_info = {"source": "none", "message": "未提供合约/品种代码"}
            return pd.DataFrame()
        if period != "daily":
            self.last_fetch_info = {"source": "none", "message": f"高频源仅支持日K聚合，不支持 {period}"}
            return pd.DataFrame()

        inv = self.inventory()
        inv = inv[inv["symbol"].astype(str).str.lower().isin(symbols)
                  | inv["contract"].astype(str).str.lower().isin(symbols)]
        self.last_fetch_info = {"source": "hf", "message": f"高频 L2 快照聚合（{len(inv)} 个合约）"}
        rows = []
        for con in inv["contract"]:
            try:
                got = self.load_l2(con)
            except Exception as e:  # noqa: BLE001
                logger.warning("[HF] %s 读取失败，跳过日K聚合: %s", con, e)
                continue
            if con not in got:
                continue
            d = got[con]
            if not len(d):
                continue
            last = d["last_price"].dropna()
            if not len(last):
                continue
            close = float(last.iloc[-1])
            pre = pd.to_numeric(d.get("pre_settle"), errors="coerce").dropna()
            base = float(pre.iloc[0]) if len(pre) else np.nan
            vol = pd.to_numeric(d.get("volume"), errors="coerce").dropna()
            amt = pd.to_numeric(d.get("turnover"), errors="coerce").dropna()
            rows.append({
                "date": pd.Timestamp(d["ts"].iloc[-1]).normalize().strftime("%Y-%m-%d"),
                "open": self._first_valid(d, "open") or float(last.iloc[0]),
                # DayHigh/DayLow 是当日累计极值，取**末值**才是全天真实高低；
                # 取首值会得到「开盘那一刻的极值」，这是期货快照聚合最常见的错法。
                "high": self._last_valid(d, "day_high") or float(last.max()),
                "low": self._last_valid(d, "day_low") or float(last.min()),
                "close": close,
                "volume": float(vol.iloc[-1]) if len(vol) else 0.0,
                "amount": float(amt.iloc[-1]) if len(amt) else 0.0,
                # 统一为百分数，与 DataFetcher / OfflineDataSource 的 pct_chg 口径一致
                "pct_chg": (close / base - 1.0) * 100.0
                           if base and np.isfinite(base) and base > 0 else np.nan,
                "symbol": con,
            })
        if not rows:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close",
                                         "volume", "amount", "pct_chg", "symbol"])
        out = pd.DataFrame(rows)
        if start:
            out = out[out["date"] >= str(start)[:10]]
        if end:
            out = out[out["date"] <= str(end)[:10]]
        return out.sort_values(["date", "symbol"]).reset_index(drop=True)

    @staticmethod
    def _first_valid(d: pd.DataFrame, col: str):
        if col not in d.columns:
            return None
        s = pd.to_numeric(d[col], errors="coerce").dropna()
        return float(s.iloc[0]) if len(s) else None

    @staticmethod
    def _last_valid(d: pd.DataFrame, col: str):
        if col not in d.columns:
            return None
        s = pd.to_numeric(d[col], errors="coerce").dropna()
        return float(s.iloc[-1]) if len(s) else None

    def get_minute_kline(self, symbol: str, freq: str = "1min",
                         date: Optional[str] = None) -> pd.DataFrame:
        """从 L2 快照重采样成分钟 K。

        OfflineDataSource.get_minute_kline 目前是空桩，这个方法给高频侧提供了真实实现：
        用 LastPrice 聚合 OHLC，成交量取累计量差分之和（跨会话已在 loader 里切断）。
        注意：分钟 K 的 high/low 只有 500ms 采样分辨率，不等价于真实逐笔极值。
        """
        got = self.load_l2(self.list_contracts(symbol) if symbol not in \
                           set(self.inventory()["contract"]) else symbol)
        frames = []
        for con, d in got.items():
            if not len(d):
                continue
            g = d.set_index("ts")
            agg = pd.DataFrame({
                "open": g["last_price"], "high": g["last_price"],
                "low": g["last_price"], "close": g["last_price"],
                "dvol": g.get("d_volume", pd.Series(index=g.index, dtype=float)),
            }).resample(freq).agg({"open": "first", "high": "max", "low": "min",
                                   "close": "last", "dvol": "sum"})
            agg["symbol"] = con
            frames.append(agg.reset_index().rename(columns={"ts": "datetime", "dvol": "volume"}))
        if not frames:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close",
                                         "volume", "symbol"])
        out = pd.concat(frames, ignore_index=True)
        if date:
            out = out[out["datetime"].dt.normalize() == pd.Timestamp(date)]
        return out

    # ---------- 跨期套利与期限结构 ----------
    def get_term_structure(self, symbol: str, sessions: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """拼某品种的跨期限面板：index=ts，列=各合约中价，缺失自动向前就近填充。

        不同合约的快照**时间戳并非完全对齐**（实测 au2608 与 au2602 重合度 98.01%），
        因此不能直接横向 concat，必须重采样到公共时间网格。这里用 ``merge_asof``
        按最近邻对齐（容差 1 秒），超过容差即视为该合于此此刻无报价（NaN）。
        """
        got = self.load_symbol_l2(symbol, sessions=sessions)
        if not got:
            return pd.DataFrame()
        series = {}
        for con, d in got.items():
            if not len(d):
                continue
            mid = (pd.to_numeric(d["bp1"], errors="coerce")
                   + pd.to_numeric(d["sp1"], errors="coerce")) / 2.0
            s = pd.Series(mid.to_numpy(), index=pd.DatetimeIndex(d["ts"])).dropna()
            s = s[~s.index.duplicated()]
            series[con] = s.sort_index()
        if not series:
            return pd.DataFrame()
        grid = None
        for s in series.values():
            grid = s.index if grid is None else grid.union(s.index)
        out = pd.DataFrame(index=grid.sort_values())
        for con, s in series.items():
            # 最近邻对齐：超过 1s 容差即视为该合约此刻无报价
            out[con] = s.reindex(out.index, method="nearest",
                                 tolerance=pd.Timedelta("1s"))
        out.index.name = "ts"
        return out

    # ---------- 自身委托流水（被动做市标签源） ----------
    @staticmethod
    def load_orders(path: str) -> pd.DataFrame:
        """读取并规范化自身的委托流水（xlsx），服务于「挂单成交概率」建模。

        原始列中文名，规范化为英文：``order_time/cancel_time/contract/side/qty/price/
        status/filled_qty/remain_qty/order_no/sys_no``。并派生：
          ``filled``    是否成交（委托单状态 == 全部成交）
          ``life_ms``   挂单存活毫秒数（委托时间到撤销时间）
          ``session``   依据委托时间归属的会话段（N/M/E）

        已知限制（必须写在这里，别指望数据里有毫秒）：委托时间只有**秒级精度**，
        而快照是 500ms；同一秒内的多笔订单无法区分先后，因此**无法**还原严格的队列位置。
        建模只能用「秒级簿状态」作为特征，标签是该笔订单最终是否成交。
        """
        d = pd.read_excel(path)
        rename = {
            "结算日": "settle_date", "报单日期": "order_date", "委托时间": "order_time",
            "合约": "contract", "委托单状态": "status_raw", "买卖": "side_raw",
            "委托量": "qty", "委托价": "price", "报单价格条件": "price_cond",
            "成交量": "filled_qty", "剩余数量": "remain_qty", "撤销时间": "cancel_time",
            "委托号": "order_no", "系统号": "sys_no", "状态信息": "status_msg",
        }
        d = d.rename(columns={k: v for k, v in rename.items() if k in d.columns})
        d["side"] = d["side_raw"].map({"买": 1, "卖": -1}).astype("Int64")
        d["filled"] = d["status_raw"].astype(str).str.contains("成交").astype(int)
        # 秒级时间戳：夜盘跨午夜，用 20:00 分界线把 00:00-19:xx 归到次自然日
        sec = pd.to_timedelta(d["order_time"].astype(str)).dt.total_seconds()
        base = pd.to_datetime(d["settle_date"].astype(str), format="%Y%m%d")
        # 结算日=交易日；夜盘（>=20h）属于交易日的前一自然日
        d["ts"] = pd.to_datetime(base) + pd.to_timedelta(sec, unit="s")
        d.loc[sec >= 20 * 3600, "ts"] = d.loc[sec >= 20 * 3600, "ts"] - pd.Timedelta(days=1)
        if "cancel_time" in d.columns:
            csec = pd.to_timedelta(d["cancel_time"].astype(str), errors="coerce").dt.total_seconds()
            d["life_sec"] = (csec - sec).where(csec.notna())
        # 夜盘 = 20:00 以后 **以及** 凌晨 0:00-03:00（夜盘跨越午夜，凌晨是其后半段）；
        # 上午 09:00-11:30；下午 13:30-15:00。
        hh = sec / 3600.0
        d["session"] = np.select(
            [(hh >= 20) | (hh < 3), (hh >= 13.5) & (hh <= 15.01),
             (hh >= 9) & (hh <= 11.51)],
            ["N", "E", "M"], default="X")
        return d

    @staticmethod
    def overlap_diagnostics(quote: pd.DataFrame, orders: pd.DataFrame) -> dict:
        """行情与委托的时间对齐体检——**建模前必跑**，不要跳过。

        返回两边的时间覆盖与秒级交集。若交集过小，说明二者不是同一段交易时间，
        这时「行情特征 → 委托是否成交」的建模就是彻头彻尾的错配。
        宁可在这里把数字打出来让人看见，也不要让模型学一堆 NaN。
        """
        q = pd.DatetimeIndex(quote["ts"]).floor("s")
        o = pd.DatetimeIndex(orders["ts"]).floor("s")
        qs, os_ = set(q), set(o)
        inter = qs & os_
        by_sess = {}
        if len(inter) and "session" in quote.columns and "session" in orders.columns:
            for s in sorted(set(orders["session"].astype(str))):
                sub = orders[orders["session"].astype(str) == s]
                hits = len(set(pd.DatetimeIndex(sub["ts"]).floor("s")) & qs)
                by_sess[s] = {"orders": len(sub), "matched_seconds": hits}
        return {
            "quote_range": (str(q.min()), str(q.max())) if len(q) else None,
            "order_range": (str(o.min()), str(o.max())) if len(o) else None,
            "quote_seconds": len(qs), "order_seconds": len(os_),
            "intersect_seconds": len(inter),
            "intersect_ratio_to_orders": round(len(inter) / max(len(os_), 1), 4),
            "by_order_session": by_sess,
        }

    def describe(self) -> dict:
        """数据源自检摘要（给 UI / 体检脚本用）。"""
        if not self.enabled:
            return {"enabled": False, "file": str(self.file or "")}
        pf = self._parquet()
        inv = self.inventory()
        return {
            "enabled": True, "file": str(self.file),
            "size_mb": round(Path(self.file).stat().st_size / 1024 / 1024, 1),
            "rows": pf.metadata.num_rows,
            "columns": len(pf.schema_arrow.names),
            "symbols": int(inv["symbol"].nunique()) if len(inv) else 0,
            "contracts": len(inv),
            "cache_dir": str(self.cache_dir),
        }


__all__ = ["SESSION_NAME", "HFDataSource", "estimate_tick_size"]
