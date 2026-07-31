"""
数据获取器 (DataFetcher)

基于 akshare 的多源金融数据获取模块，提供日K线、分钟K线、财务数据、指数成分股、
新闻舆情、实时行情和行业分类等数据接口。

所有方法均包含完善的异常处理，失败时返回空 DataFrame 而非抛出异常。
"""

import os
import re
import pickle
import hashlib
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# 代理/直连策略统一由 netutil 管理：默认强制直连（规避 Windows 不可达系统代理
# 导致的 ProxyError）；当 config.yaml 的 proxy 段启用且给出地址，或已设置
# HTTP_PROXY 环境变量时，改走代理；localhost/127.0.0.1 始终直连。
from netutil import apply_proxy_settings, get_trust_env, patch_requests_session

apply_proxy_settings(None)   # 默认直连
patch_requests_session()     # 让 requests.Session 跟随当前策略


def _load_config_file() -> dict:
    """读取项目根目录 config.yaml（带模块级缓存）。

    仅用于 `DataFetcher()` 无参实例化时回退，使 prefer_sina / primary_source /
    proxy / tushare_token 等配置对所有调用点生效。无配置或解析失败时返回 {}。
    不在此处做 ${ENV} 变量替换：tushare_token 等已优先从 os.environ 读取。
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    cfg: dict = {}
    try:
        import yaml
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(root, "config.yaml")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
    except Exception:  # noqa: BLE001
        cfg = {}
    _CONFIG_CACHE = cfg
    return _CONFIG_CACHE


_CONFIG_CACHE: Optional[dict] = None


def _resolve_env_token(val) -> Optional[str]:
    """解析 tushare_token：支持 ${ENV} / $ENV 占位符；占位符未设置或为空时返回 None。

    避免把 config.yaml 里的字面量 '${TUSHARE_TOKEN}' 当成有效 token 传入 Tushare。
    """
    if not isinstance(val, str):
        return val or None
    s = val.strip()
    m = re.match(r"^\$\{(.+)\}$", s) or re.match(r"^\$([A-Za-z_][A-Za-z0-9_]*)$", s)
    if m:
        return os.environ.get(m.group(1)) or None
    return s or None


class DataFetcher:
    """金融数据获取器，封装 akshare 各数据接口。

    Attributes:
        tushare_token: tushare API token（可选），用于高级数据接口。
    """

    # A股日K线统一列名映射
    _KLINE_COLUMN_MAP: Dict[str, str] = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_chg",
        "涨跌额": "change",
        "换手率": "turnover",
    }

    def __init__(self, tushare_token: Optional[str] = None, config: Optional[dict] = None) -> None:
        """初始化 DataFetcher。

        Args:
            tushare_token: tushare API token，可选（也可经 config 注入）。
            config: 全局配置字典（config.yaml 解析结果），可选。会从
                config["data"] 读取 primary_source / synthetic_on_fail /
                ths_token / ths_api_base_url 等项。
        """
        # 未显式传入 config 时，默认加载项目根目录的 config.yaml，使 prefer_sina /
        # primary_source / proxy / tushare_token 等配置对无参 `DataFetcher()` 调用
        # （如 refinery、market_data、agent.graph 中的实例化）同样生效。
        if config is None or (isinstance(config, dict) and not config):
            config = _load_config_file()
        cfg = config or {}
        data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}

        # tushare token：显式参数优先，其次读取配置（支持 ${ENV} 占位符），再次读环境变量
        raw_token = tushare_token or data_cfg.get("tushare_token") or os.environ.get("TUSHARE_TOKEN")
        self.tushare_token = _resolve_env_token(raw_token)

        # 数据源与回退开关
        self.primary_source = data_cfg.get("primary_source", "akshare")
        # prefer_sina：部分网络环境下东方财富(push2his)的 HTTPS 连接会被对端重置
        # (RemoteDisconnected)，此时把新浪源提前为首选，避免大量无谓失败与日志噪音。
        # 在 config.yaml 将 data.primary_source 设为 "sina" 亦等价开启。
        self.prefer_sina = bool(data_cfg.get("prefer_sina", False)) or \
            str(data_cfg.get("primary_source", "")).lower() == "sina"
        # force_synthetic：全局离线模式（跳过网络，始终返回合成数据）
        self.force_synthetic = bool(data_cfg.get("force_synthetic", False))
        # synthetic_on_fail：实时源全部失败时回退到合成数据
        self.synthetic_on_fail = bool(data_cfg.get("synthetic_on_fail", False))
        # 本地缓存（预备数据源，抗网络波动）
        self.cache_dir = data_cfg.get("cache_dir", "data/cache")
        self.use_cache = bool(data_cfg.get("use_cache", True))   # 成功后写缓存；失败时优先读缓存
        self.cache_only = bool(data_cfg.get("cache_only", False))  # true 时完全离线，只读取预备缓存

        # 应用代理配置（config.yaml 的 proxy 段；localhost/127.0.0.1 始终直连）。
        # 未配置则保持强制直连，规避不可达系统代理导致的 ProxyError；若已设置
        # HTTP_PROXY 环境变量也会被尊重。proxy 段可位于配置顶层或 data 段。
        proxy_cfg = (cfg.get("proxy") if isinstance(cfg, dict) else None) or \
            data_cfg.get("proxy")
        apply_proxy_settings(proxy_cfg)

        # 每次行情查询后的实际数据源与提示（供 UI 展示）
        self.last_fetch_info: Dict[str, str] = {"source": None, "message": ""}

        # 同花顺 THS（可选，需配置 token 与 API 地址）
        self.ths_fetcher = None
        ths_token = (
            data_cfg.get("ths_api_token")
            or data_cfg.get("ths_token")
            or os.environ.get("THS_API_TOKEN")
            or os.environ.get("THS_TOKEN")
        )
        ths_base = data_cfg.get("ths_api_base_url") or os.environ.get("THS_API_BASE_URL")
        if ths_token and ths_base:
            try:
                from data.ths_fetcher import THSDataFetcher

                self.ths_fetcher = THSDataFetcher(ths_token, ths_base)
            except Exception as e:  # pragma: no cover
                print(f"[DataFetcher] 同花顺数据源初始化失败: {e}")
                self.ths_fetcher = None

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _standardize_date(date_str: str) -> str:
        """将各种日期字符串标准化为 YYYYMMDD 格式。

        Args:
            date_str: 输入日期字符串，支持 YYYY-MM-DD、YYYY/MM/DD 等格式。

        Returns:
            YYYYMMDD 格式的日期字符串。
        """
        date_str = str(date_str).strip().replace("/", "-").replace(".", "-")
        try:
            dt = pd.to_datetime(date_str)
            return dt.strftime("%Y%m%d")
        except (ValueError, TypeError):
            return date_str.replace("-", "")

    # ------------------------------------------------------------------
    # 本地预备缓存（抗网络波动）
    # ------------------------------------------------------------------
    def _ensure_cache_dir(self) -> None:
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception:  # noqa: BLE001
            pass

    def _save_cache(self, name: str, obj) -> None:
        """将任意可 pickle 对象写入缓存（行情/指数/行业市值/整矿）。"""
        try:
            self._ensure_cache_dir()
            with open(os.path.join(self.cache_dir, name + ".pkl"), "wb") as f:
                pickle.dump(obj, f)
        except Exception as e:  # noqa: BLE001
            logging.warning("[DataFetcher] 缓存写入失败 %s: %s", name, e)

    def _load_cache(self, name: str):
        p = os.path.join(self.cache_dir, name + ".pkl")
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception as e:  # noqa: BLE001
                logging.warning("[DataFetcher] 缓存读取失败 %s: %s", name, e)
        return None

    def _kline_cache_key(self, symbols, start, end, adjust) -> str:
        key = "|".join(sorted(str(s) for s in symbols)) + f"|{start}|{end}|{adjust}"
        return "kline_" + hashlib.md5(key.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # 日K线数据
    # ------------------------------------------------------------------

    def get_daily_kline(
        self,
        symbols: List[str],
        start: str,
        end: str,
        period: str = "daily",
        adjust: str = "qfq",
        force_synthetic: bool = False,
    ) -> pd.DataFrame:
        """获取多只股票的日K线数据（多数据源自动回退）。

        数据源优先级：
        1. akshare 东方财富（默认主用；但部分网络环境到 push2his 的连接会被对端重置）
        2. akshare 新浪（stock_zh_a_daily，作为东方财富的可靠回退；prefer_sina=true 时提前为首选）
        3. tushare（若配置了 token）
        4. 同花顺 THS（若配置了 token 与 API 地址）
        5. 合成数据（force_synthetic=True，或全部实时源失败且开启 synthetic_on_fail）

        每次调用后可通过 self.last_fetch_info 查看实际使用的源与提示信息，
        便于 UI 区分「股票代码错误」与「数据源网络不可用」。

        Returns:
            DataFrame（date/open/high/low/close/volume/amount/pct_chg/symbol），
            失败返回空 DataFrame。
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        symbols = [str(s).strip() for s in symbols if str(s).strip()]
        self.last_fetch_info = {"source": None, "message": ""}

        if not symbols:
            self.last_fetch_info = {"source": "none", "message": "未提供股票代码"}
            return pd.DataFrame()

        # cache_only：完全离线，仅读取已预备的行情缓存（现场防断网）
        if self.cache_only:
            cached = self._load_cache(self._kline_cache_key(symbols, start, end, adjust))
            if cached is not None and not cached.empty:
                self.last_fetch_info = {"source": "cache",
                                        "message": "cache_only 模式：已加载本地预备行情缓存"}
                return cached
            self.last_fetch_info = {"source": "none",
                                    "message": "cache_only 模式但无预备缓存，请先运行预备数据脚本"}
            return pd.DataFrame()

        if force_synthetic or self.force_synthetic:
            self.last_fetch_info = {
                "source": "synthetic",
                "message": "已使用合成数据（force_synthetic=True）",
            }
            return self._synthetic_daily_kline(symbols, start, end, adjust)

        start_fmt = self._standardize_date(start)
        end_fmt = self._standardize_date(end)

        # 依次尝试各数据源；首个返回非空者即采用。
        # 顺序：默认 东方财富 -> 新浪 -> Tushare -> 同花顺。
        # 若开启 prefer_sina（部分网络到东方财富的连接被对端重置）：
        #   直接把东方财富从候选链移除（已知该网络下 100% 失败），仅 新浪 -> Tushare -> 同花顺，
        #   彻底避免无谓的失败日志；新浪返回空时也不会再去撞东方财富。
        source_chain = []
        if self.prefer_sina:
            source_chain.append(("akshare_sina", self._fetch_single_sina, "新浪"))
        else:
            source_chain.append(("akshare_eastmoney", self._fetch_single_eastmoney, "东方财富"))
            source_chain.append(("akshare_sina", self._fetch_single_sina, "新浪"))
        if self.tushare_token:
            source_chain.append(("tushare", self._fetch_single_tushare, "Tushare"))
        if self.ths_fetcher is not None:
            source_chain.append(("ths", "_ths_", "同花顺"))

        frames = None
        source = None
        attempts: List[str] = []
        for src_name, fn, label in source_chain:
            if src_name == "_ths_":
                frames = self._fetch_batch_ths(symbols, start_fmt, end_fmt, period, adjust)
            else:
                frames = self._fetch_batch(fn, symbols, start_fmt, end_fmt, period, adjust)
            if frames:
                source = src_name
                if src_name == "akshare_sina" and self.prefer_sina:
                    self.last_fetch_info = {
                        "source": source,
                        "message": "已优先使用新浪数据源（东方财富在本网络被重置，已跳过）",
                    }
                elif src_name == "akshare_sina":
                    self.last_fetch_info = {
                        "source": source,
                        "message": "东方财富接口当前不可用，已自动切换至新浪数据源",
                    }
                break
            attempts.append(label)

        if frames:
            result = pd.concat(frames, ignore_index=True)
            result = result.sort_values(["symbol", "date"]).reset_index(drop=True)
            if self.use_cache:
                self._save_cache(self._kline_cache_key(symbols, start, end, adjust), result)
            if not self.last_fetch_info.get("source"):
                self.last_fetch_info = {"source": source, "message": ""}
            return result

        # 全部实时源失败：尝试本地预备缓存作为回退
        if self.use_cache:
            cached = self._load_cache(self._kline_cache_key(symbols, start, end, adjust))
            if cached is not None and not cached.empty:
                self.last_fetch_info = {"source": "cache",
                                        "message": "实时源不可用，已回退至本地预备行情缓存"}
                return cached

        # 全部实时源失败
        if self.synthetic_on_fail:
            self.last_fetch_info = {"source": "synthetic", "message": "实时数据源全部不可用，已回退至合成数据"}
            return self._synthetic_daily_kline(symbols, start, end, adjust)

        self.last_fetch_info = {
            "source": "none",
            "message": (
                f"所有实时数据源均不可用（已尝试：{', '.join(attempts) or '无'}）。"
                f"常见原因：当前网络无法访问东方财富/新浪服务器，或 akshare 接口已变更。"
                f"可在 config.yaml 将 data.synthetic_on_fail 设为 true 使用模拟数据。"
            ),
        }
        return pd.DataFrame()

    # ---- 多源回退辅助方法 ----
    def _fetch_batch(self, fn, symbols, start_fmt, end_fmt, period, adjust):
        frames = []
        for symbol in symbols:
            try:
                df = fn(symbol, start_fmt, end_fmt, period, adjust)
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception as e:
                print(f"[DataFetcher] {symbol} 经 {getattr(fn, '__name__', fn)} 获取失败: {e}")
        return frames

    def _fetch_batch_ths(self, symbols, start_fmt, end_fmt, period, adjust):
        try:
            df = self.ths_fetcher.get_daily_kline(symbols, start_fmt, end_fmt, period, adjust)
            if df is not None and not df.empty:
                return [df]
        except Exception as e:
            print(f"[DataFetcher] 同花顺行情获取失败: {e}")
        return []

    def _fetch_single_eastmoney(self, symbol, start_fmt, end_fmt, period, adjust):
        import akshare as ak

        df = ak.stock_zh_a_hist(
            symbol=symbol, period=period, start_date=start_fmt, end_date=end_fmt, adjust=adjust
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns=self._KLINE_COLUMN_MAP)
        return self._normalize_kline(df, symbol)

    def _fetch_single_sina(self, symbol, start_fmt, end_fmt, period, adjust):
        import akshare as ak

        sina_sym = self._to_sina_symbol(symbol)
        try:
            df = ak.stock_zh_a_daily(symbol=sina_sym, start_date=start_fmt, end_date=end_fmt, adjust=adjust)
        except TypeError:
            # 部分 akshare 版本要求 period 参数
            df = ak.stock_zh_a_daily(
                symbol=sina_sym, period="day", start_date=start_fmt, end_date=end_fmt, adjust=adjust
            )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume",
        })
        return self._normalize_kline(df, symbol)

    def _fetch_single_tushare(self, symbol, start_fmt, end_fmt, period, adjust):
        import tushare as ts

        pro = ts.pro_api(self.tushare_token)
        ts_code = f"{symbol}.SH" if symbol.startswith("6") else f"{symbol}.SZ"
        df = pro.daily(
            ts_code=ts_code, start_date=start_fmt, end_date=end_fmt,
            fields="ts_code,trade_date,open,high,low,close,vol,amount",
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"trade_date": "date", "vol": "volume"})
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.sort_values("date").reset_index(drop=True)
        return self._normalize_kline(df, symbol)

    @staticmethod
    def _to_sina_symbol(code: str) -> str:
        """将 A 股代码转换为新浪行情接口所需的 sh/sz/bj 前缀格式。"""
        code = str(code).strip()
        code = re.sub(r"^(sh|sz|bj)", "", code, flags=re.IGNORECASE)
        if code.startswith("6"):
            return "sh" + code
        if code.startswith(("0", "3")):
            return "sz" + code
        if code.startswith(("4", "8", "9")):
            return "bj" + code
        return "sh" + code

    @staticmethod
    def _normalize_kline(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """统一 K 线列名/类型，并补齐缺失列（amount/amplitude/pct_chg/...）。"""
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["symbol"] = symbol
        numeric_cols = ["open", "high", "low", "close", "volume", "amount", "amplitude", "pct_chg", "change", "turnover"]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in ["amount", "amplitude", "pct_chg", "change", "turnover"]:
            if c not in df.columns:
                df[c] = float("nan")
        if df["pct_chg"].isna().all() and "close" in df.columns:
            df["pct_chg"] = df["close"].pct_change() * 100
        return df

    def _synthetic_daily_kline(self, symbols, start, end, adjust) -> pd.DataFrame:
        """离线合成行情（随机游走 OHLC），仅用于无网络时的兜底/演示。"""
        import numpy as np

        start_fmt = self._standardize_date(start)
        end_fmt = self._standardize_date(end)
        dates = pd.bdate_range(start_fmt, end_fmt)
        rng = np.random.default_rng(42)
        frames = []
        for symbol in symbols:
            n = len(dates)
            if n == 0:
                continue
            ret = rng.normal(0.0005, 0.02, n)
            close = 10 * np.cumprod(1 + ret)
            open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.005, n))
            high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.01, n)))
            low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.01, n)))
            volume = rng.integers(int(1e5), int(1e6), n)
            frames.append(pd.DataFrame({
                "date": dates.strftime("%Y-%m-%d"),
                "open": open_.round(2), "high": high.round(2), "low": low.round(2),
                "close": close.round(2), "volume": volume.astype(float),
                "amount": (volume * close).round(2),
                "pct_chg": (pd.Series(close).pct_change() * 100).round(2).fillna(0).values,
                "symbol": symbol,
            }))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ------------------------------------------------------------------
    # 分钟K线数据
    # ------------------------------------------------------------------

    def get_minute_kline(
        self,
        symbols: List[str],
        period: str = "5",
        max_days: int = 30,
    ) -> pd.DataFrame:
        """获取多只股票的分钟K线数据。

        使用 akshare.stock_zh_a_hist_min_em（东方财富分钟K线接口），
        支持 1/5/15/30/60 分钟周期，覆盖最近约 1-2 个月数据。

        Args:
            symbols: 股票代码列表，如 ['000001', '600519']。
            period: K线周期，'1' / '5' / '15' / '30' / '60'（分钟）。
            max_days: 最大获取天数，默认 30 天。接口返回最近约 60 天数据，
                      设置此参数可限制数据量。

        Returns:
            DataFrame，列名：date / time / open / high / low / close /
            volume / amount / symbol。
            失败时返回空 DataFrame。

        Examples:
            >>> fetcher = DataFetcher()
            >>> df = fetcher.get_minute_kline(['000001', '600519'], period='5')
            >>> print(df.head())
        """
        try:
            import akshare as ak
        except ImportError:
            print("[DataFetcher] akshare 未安装，请执行 pip install akshare")
            return pd.DataFrame()

        # 分钟K线列名映射（东方财富接口的列名略有不同）
        minute_col_map: Dict[str, str] = {
            "时间": "time",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "均价": "avg_price",
        }

        frames: List[pd.DataFrame] = []
        for symbol in symbols:
            try:
                # stock_zh_a_hist_min_em 参数：
                #   symbol: 股票代码
                #   period: '1'/'5'/'15'/'30'/'60'
                #   adjust: ''不复权 / 'qfq'前复权 / 'hfq'后复权
                #   start_date: YYYY-MM-DD（可选，用于精确时间范围）
                df = ak.stock_zh_a_hist_min_em(
                    symbol=symbol,
                    period=period,
                    adjust="qfq",
                )
                if df is None or df.empty:
                    print(f"[DataFetcher] {symbol} 无分钟K线数据")
                    continue

                # 统一列名
                df = df.rename(columns=minute_col_map)

                # 保留需要的列
                needed_cols = [
                    "time", "open", "high", "low", "close",
                    "volume", "amount",
                ]
                available = [c for c in needed_cols if c in df.columns]
                df = df[available].copy()

                # 解析时间列，拆分为 date 和 time
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"], errors="coerce")
                    df["date"] = df["time"].dt.strftime("%Y-%m-%d")
                    df["time"] = df["time"].dt.strftime("%H:%M")
                else:
                    continue

                # 按 max_days 过滤（保留最近 N 个交易日的数据）
                if max_days and "date" in df.columns:
                    unique_dates = sorted(df["date"].unique(), reverse=True)
                    if len(unique_dates) > max_days:
                        cutoff_date = unique_dates[max_days - 1]
                        df = df[df["date"] >= cutoff_date].copy()

                # 添加 symbol 列
                df["symbol"] = symbol

                # 数值列类型转换
                numeric_cols = [
                    c for c in ["open", "high", "low", "close", "volume", "amount"]
                    if c in df.columns
                ]
                df[numeric_cols] = df[numeric_cols].apply(
                    pd.to_numeric, errors="coerce"
                )

                # 列顺序
                col_order = ["date", "time", "symbol"] + [
                    c for c in numeric_cols if c in df.columns
                ]
                df = df[[c for c in col_order if c in df.columns]]

                frames.append(df)
                print(
                    f"[DataFetcher] {symbol} 分钟K线（{period}min）"
                    f"获取 {len(df)} 条，覆盖 {df['date'].nunique()} 个交易日"
                )

            except Exception as e:
                print(f"[DataFetcher] {symbol} 分钟K线获取失败: {e}")
                continue

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result = result.sort_values(["symbol", "date", "time"]).reset_index(drop=True)
        return result

    def get_intraday_kline(
        self,
        symbol: str,
        trade_date: str,
    ) -> pd.DataFrame:
        """获取单只股票某一交易日的1分钟K线数据。

        使用 akshare.stock_zh_a_minute，适合做日内因子研究。

        Args:
            symbol: 股票代码，如 '000001'。
            trade_date: 交易日期，格式 YYYYMMDD 或 YYYY-MM-DD。

        Returns:
            DataFrame，包含该交易日所有1分钟K线数据。
            失败返回空 DataFrame。

        Examples:
            >>> fetcher = DataFetcher()
            >>> df = fetcher.get_intraday_kline('000001', '2024-01-05')
            >>> print(df.head())
        """
        try:
            import akshare as ak

            date_fmt = self._standardize_date(trade_date)
            df = ak.stock_zh_a_minute(
                symbol=symbol,
                period="1",
                adjust="qfq",
            )
            if df is None or df.empty:
                print(f"[DataFetcher] {symbol} 无 {trade_date} 分钟K线")
                return pd.DataFrame()

            # 列名映射
            col_map: Dict[str, str] = {
                "时间": "time",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount",
            }
            df = df.rename(columns=col_map)

            needed = ["time", "open", "high", "low", "close", "volume", "amount"]
            available = [c for c in needed if c in df.columns]
            df = df[available].copy()

            # 解析时间
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], errors="coerce")
                df["date"] = df["time"].dt.strftime("%Y-%m-%d")
                df["time"] = df["time"].dt.strftime("%H:%M")

            df["symbol"] = symbol
            numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            print(
                f"[DataFetcher] {symbol} 日内分钟K线获取 {len(df)} 条"
            )
            return df

        except ImportError:
            print("[DataFetcher] akshare 未安装")
            return pd.DataFrame()
        except Exception as e:
            print(f"[DataFetcher] {symbol} 日内分钟K线获取失败: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 财务数据
    # ------------------------------------------------------------------

    def get_financial_data(
        self,
        symbol: str,
        report_type: str = "年报",
    ) -> pd.DataFrame:
        """获取个股财务摘要数据（同花顺接口）。

        Args:
            symbol: 股票代码，如 '000001'。
            report_type: 报告类型，'年报' / '中报' / '一季报' / '三季报'。

        Returns:
            DataFrame，包含财务指标数据。失败返回空 DataFrame。
        """
        try:
            import akshare as ak

            df = ak.stock_financial_abstract_ths(
                symbol=symbol,
                indicator=report_type,
            )
            if df is None or df.empty:
                print(f"[DataFetcher] {symbol} 无财务数据（{report_type}）")
                return pd.DataFrame()

            # 列名统一小写
            df.columns = [c.lower().strip() for c in df.columns]
            df["symbol"] = symbol
            print(f"[DataFetcher] {symbol} 财务数据获取成功，{len(df)} 条记录")
            return df

        except ImportError:
            print("[DataFetcher] akshare 未安装")
            return pd.DataFrame()
        except Exception as e:
            print(f"[DataFetcher] {symbol} 财务数据获取失败: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 指数成分股
    # ------------------------------------------------------------------

    def get_index_constituents(
        self,
        index_code: str = "000906",
    ) -> List[str]:
        """获取指数成分股列表。

        Args:
            index_code: 指数代码，默认 '000906'（中证800）。

        Returns:
            股票代码字符串列表，如 ['000001', '000002']。失败返回空列表。
        """
        cache_name = f"index_{index_code}"
        if self.cache_only:
            cached = self._load_cache(cache_name)
            return cached if cached else []

        try:
            import akshare as ak

            df = ak.index_stock_cons(symbol=index_code)
            if df is None or df.empty:
                print(f"[DataFetcher] 指数 {index_code} 无成分股数据")
                return []

            # 尝试提取股票代码列
            code_col = None
            for col in ["品种代码", "stock_code", "constituent_code", "代码"]:
                if col in df.columns:
                    code_col = col
                    break

            if code_col is None:
                # 取第一列
                code_col = df.columns[0]

            symbols = df[code_col].astype(str).str.zfill(6).tolist()
            symbols = [s[:6] for s in symbols if s and s != "nan"]
            if self.use_cache and symbols:
                self._save_cache(cache_name, symbols)
            print(f"[DataFetcher] 指数 {index_code} 成分股 {len(symbols)} 只")
            return symbols

        except ImportError:
            cached = self._load_cache(cache_name)
            if cached:
                print("[DataFetcher] akshare 未安装，已回退指数成分股缓存")
                return cached
            print("[DataFetcher] akshare 未安装")
            return []
        except Exception as e:
            cached = self._load_cache(cache_name)
            if cached:
                print(f"[DataFetcher] 指数 {index_code} 成分股网络获取失败，已回退缓存（{len(cached)} 只）")
                return cached
            print(f"[DataFetcher] 指数 {index_code} 成分股获取失败: {e}")
            return []

    # ------------------------------------------------------------------
    # 新闻舆情
    # ------------------------------------------------------------------

    def get_news_sentiment(
        self,
        symbol: str,
        days: int = 30,
    ) -> pd.DataFrame:
        """获取个股新闻舆情数据。

        使用 akshare.stock_news_em 获取东方财富个股新闻，
        按发布时间过滤最近 N 天。

        Args:
            symbol: 股票代码，如 '000001'。
            days: 获取最近多少天的新闻，默认 30 天。

        Returns:
            DataFrame，至少包含 date / title / content 列。失败返回空 DataFrame。
        """
        try:
            import akshare as ak

            df = ak.stock_news_em(symbol=symbol)
            if df is None or df.empty:
                print(f"[DataFetcher] {symbol} 无新闻数据")
                return pd.DataFrame()

            # 列名统一小写
            df.columns = [c.lower().strip() for c in df.columns]

            # 时间列定位与过滤
            time_col = None
            for col in ["发布时间", "datetime", "date", "time"]:
                if col in df.columns:
                    time_col = col
                    break

            if time_col:
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
                cutoff = datetime.now() - timedelta(days=days)
                df = df[df[time_col] >= cutoff].copy()
                df["date"] = df[time_col].dt.strftime("%Y-%m-%d")

            df["symbol"] = symbol
            print(
                f"[DataFetcher] {symbol} 新闻舆情获取 {len(df)} 条（最近 {days} 天）"
            )
            return df

        except ImportError:
            print("[DataFetcher] akshare 未安装")
            return pd.DataFrame()
        except Exception as e:
            print(f"[DataFetcher] {symbol} 新闻获取失败: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 实时行情快照
    # ------------------------------------------------------------------

    def get_market_snapshot(
        self,
        symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """获取A股实时行情快照。

        Args:
            symbols: 股票代码列表，为 None 时获取全市场行情。

        Returns:
            DataFrame，行情快照数据。失败返回空 DataFrame。
        """
        try:
            import akshare as ak

            df = ak.stock_zh_a_spot_em()
            if df is None or df.empty:
                print("[DataFetcher] 实时行情获取为空")
                return pd.DataFrame()

            # 列名统一小写
            df.columns = [c.lower().strip().replace("（", "(").replace("）", ")") for c in df.columns]

            # 按 symbol 过滤
            if symbols:
                code_col = None
                for col in ["代码", "code", "symbol"]:
                    if col in df.columns:
                        code_col = col
                        break
                if code_col:
                    df[code_col] = df[code_col].astype(str).str.zfill(6)
                    symbol_set = {s.zfill(6) for s in symbols}
                    df = df[df[code_col].isin(symbol_set)].copy()

            print(f"[DataFetcher] 实时行情获取 {len(df)} 条")
            return df

        except ImportError:
            print("[DataFetcher] akshare 未安装")
            return pd.DataFrame()
        except Exception as e:
            print(f"[DataFetcher] 实时行情获取失败: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 行业分类
    # ------------------------------------------------------------------

    def get_industry_classification(self) -> pd.DataFrame:
        """获取东方财富行业板块分类。

        Returns:
            DataFrame，包含行业名称、代码等。失败返回空 DataFrame。
        """
        try:
            import akshare as ak

            df = ak.stock_board_industry_name_em()
            if df is None or df.empty:
                print("[DataFetcher] 行业分类数据为空")
                return pd.DataFrame()

            # 列名统一小写
            df.columns = [c.lower().strip() for c in df.columns]
            print(f"[DataFetcher] 行业分类获取 {len(df)} 个板块")
            return df

        except ImportError:
            print("[DataFetcher] akshare 未安装")
            return pd.DataFrame()
        except Exception as e:
            print(f"[DataFetcher] 行业分类获取失败: {e}")
            return pd.DataFrame()

    def _get_tushare_pro(self):
        """惰性初始化并缓存 Tushare pro 客户端；无 token 或初始化失败时返回 None。"""
        if not self.tushare_token:
            return None
        client = getattr(self, "_tushare_pro_client", None)
        if client is not None or getattr(self, "_tushare_pro_tried", False):
            return self._tushare_pro_client if client is not None else None
        self._tushare_pro_tried = True
        try:
            import tushare as ts
            self._tushare_pro_client = ts.pro_api(self.tushare_token)
        except Exception:  # noqa: BLE001
            self._tushare_pro_client = None
        return self._tushare_pro_client

    def _fetch_cap_baidu(self, symbols, mkt_cap):
        """市值兜底：经百度 `stock_zh_valuation_baidu` 逐股获取总市值（亿元→元）。

        仅在没有 Tushare token 且开启 prefer_sina 时使用（东方财富在本网络不可达）。
        首次较慢（约 100 只 × 2s），结果由上层 with_cache 落盘缓存。
        """
        try:
            import akshare as ak
            logging.warning(
                "[DataFetcher] 未配置 Tushare token，市值经百度逐股获取（共 %d 只，首次较慢，已缓存）",
                len(symbols),
            )
            for s in symbols:
                if pd.notna(mkt_cap.get(s)):
                    continue
                try:
                    d = ak.stock_zh_valuation_baidu(symbol=s, indicator="总市值", period="近一年")
                    if d is not None and not d.empty and "value" in d.columns:
                        val = pd.to_numeric(d["value"], errors="coerce").iloc[-1]
                        if pd.notna(val):
                            mkt_cap.loc[s] = float(val) * 1e8  # 亿元 -> 元
                except Exception:  # noqa: BLE001
                    continue
        except Exception as e:  # noqa: BLE001
            logging.warning("[DataFetcher] 百度市值获取失败，中性化市值维度将缺失: %s", e)
        return mkt_cap

    def get_industry_and_cap(self, symbols):
        """获取个股的行业分类与总市值（用于行业/市值中性化）。

        数据来源优先级：
          1. Tushare（若配置了 token）：daily_basic 取市值、stock_basic 取行业，
             一次全市场拉取，最快最全（推荐；本网络东方财富不可达时的首选）。
          2. 新浪/百度兜底（无 token 且 prefer_sina）：市值经百度逐股获取；
             行业经东方财富逐股获取（本网络将失败，优雅降级为仅市值中性化）。
          3. 东方财富默认路径（未开启 prefer_sina）：全市场快照 + 逐股行业。

        任一维度缺失都会精确告警，不会再把"行业缺失"误报成"全部缺失而跳过"。

        Returns:
            (industry, mkt_cap) 两个 pd.Series，索引为 6 位 symbol。
        """
        symbols = [str(s).zfill(6) for s in symbols]
        industry = pd.Series(index=pd.Index(symbols, dtype=str), dtype=object)
        mkt_cap = pd.Series(index=pd.Index(symbols, dtype=str), dtype=float)
        cache_name = "indcap_" + hashlib.md5(
            "|".join(sorted(set(symbols))).encode("utf-8")).hexdigest()
        if self.cache_only:
            cached = self._load_indcap_cache(cache_name)
            if cached is not None:
                return cached
            return industry, mkt_cap
        try:
            import akshare as ak

            # ---- 优先：Tushare（若配置了 token），一次拉全市场市值 + 行业 ----
            pro = self._get_tushare_pro()
            if pro is not None:
                try:
                    # 市值：回溯最近 7 个交易日取 daily_basic 总市值（万元 -> 元）
                    trade_date = datetime.now().strftime("%Y%m%d")
                    cap_df = None
                    for _ in range(7):
                        cap_df = pro.daily_basic(trade_date=trade_date, fields="ts_code,total_mv")
                        if cap_df is not None and not cap_df.empty:
                            break
                        trade_date = (datetime.strptime(trade_date, "%Y%m%d")
                                      - timedelta(days=1)).strftime("%Y%m%d")
                    if cap_df is not None and not cap_df.empty:
                        cap_df["symbol"] = cap_df["ts_code"].str[:6]
                        cap = pd.to_numeric(cap_df.set_index("symbol")["total_mv"], errors="coerce") * 1e4
                        mkt_cap.update(cap.reindex(symbols))
                except Exception as e:  # noqa: BLE001
                    logging.warning("[DataFetcher] Tushare 市值获取失败: %s", e)
                try:
                    # 行业：stock_basic 一次返回全 A 股行业分类（申万一级）
                    ind = pro.stock_basic(exchange="", list_status="L",
                                          fields="ts_code,symbol,industry")
                    if ind is not None and not ind.empty:
                        ind["symbol"] = ind["ts_code"].str[:6]
                        mapping = ind.dropna(subset=["industry"]).set_index("symbol")["industry"].astype(str)
                        industry.update(mapping.reindex(symbols))
                except Exception as e:  # noqa: BLE001
                    logging.warning("[DataFetcher] Tushare 行业获取失败: %s", e)

            # ---- 市值兜底：无 Tushare 或 Tushare 市值缺失时 ----
            if mkt_cap.notna().sum() == 0:
                if self.prefer_sina:
                    mkt_cap = self._fetch_cap_baidu(symbols, mkt_cap)
                else:
                    try:
                        spot = ak.stock_zh_a_spot_em()
                        spot.columns = [str(c).lower().strip() for c in spot.columns]
                        code_col = "代码" if "代码" in spot.columns else "code"
                        cap_col = "总市值" if "总市值" in spot.columns else None
                        if code_col in spot.columns and cap_col:
                            spot[code_col] = spot[code_col].astype(str).str.zfill(6)
                            cap = pd.to_numeric(spot.set_index(code_col)[cap_col], errors="coerce")
                            mkt_cap.update(cap.reindex(symbols))
                    except Exception as e:  # noqa: BLE001
                        logging.warning("[DataFetcher] 市值快照获取失败，中性化市值维度将缺失: %s", e)

            # ---- 行业兜底：无 Tushare 或 Tushare 行业缺失时（东方财富逐股） ----
            if industry.notna().sum() == 0:
                for s in symbols:
                    try:
                        info = ak.stock_individual_info_em(symbol=s)
                        if isinstance(info, pd.DataFrame) and "item" in info.columns:
                            row = info[info["item"].astype(str).str.contains("行业", na=False)]
                            if not row.empty:
                                industry.loc[s] = str(row["value"].iloc[0])
                    except Exception:  # noqa: BLE001
                        continue
        except ImportError:
            cached = self._load_indcap_cache(cache_name)
            if cached is not None:
                logging.warning("[DataFetcher] akshare 未安装，已回退行业/市值缓存")
                return cached
            logging.warning("[DataFetcher] akshare 未安装，行业/市值映射不可用；中性化将跳过")

        if self.use_cache:
            self._save_indcap_cache(cache_name, industry, mkt_cap)

        n_ind = int(industry.notna().sum())
        n_cap = int(mkt_cap.notna().sum())
        if n_ind == 0 and n_cap == 0:
            logging.warning(
                "[DataFetcher] 行业/市值映射全部缺失，中性化将跳过（检查网络/akshare，"
                "或配置 Tushare token 以启用行业+市值）")
        else:
            if n_cap == 0:
                logging.warning("[DataFetcher] 市值缺失，中性化仅做行业维度（配置 Tushare token 可补全市值）")
            if n_ind == 0:
                logging.warning(
                    "[DataFetcher] 行业缺失，中性化仅做市值维度（本网络东方财富不可达；"
                    "配置 Tushare token 可补全行业分类）")
            logging.info("[DataFetcher] 行业/市值映射：行业命中 %d、市值命中 %d / %d",
                         n_ind, n_cap, len(symbols))
        return industry, mkt_cap

    def _save_indcap_cache(self, name: str, industry: pd.Series, mkt_cap: pd.Series) -> None:
        df = pd.DataFrame({"industry": industry, "mkt_cap": mkt_cap})
        self._save_cache(name, df)

    def _load_indcap_cache(self, name: str):
        df = self._load_cache(name)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df["industry"], df["mkt_cap"]
        return None
