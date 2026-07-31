"""
市场行情数据获取（期货 / 期权 / 基金 / 债券 / 外汇 / 贵金属）
=============================================================

数据来源：AKShare（免费、无需注册）。所有接口统一封装为带 **缓存** 与
**异常处理** 的静态方法，任何网络失败或接口变更都不会让 UI 崩溃，而是返回
空的 DataFrame 与一段可读的错误信息，由上层 UI 给出友好提示。

方法约定：每个公开方法返回 ``(DataFrame, error_or_None)`` 二元组。
``error`` 为 ``None`` 时表示成功（但 DataFrame 仍可能为空，例如非交易时段）。
"""

from __future__ import annotations

import datetime as _dt
import functools
import time

import pandas as pd

try:  # akshare 为可选依赖，缺失时所有方法安全降级
    import akshare as ak
except Exception:  # pragma: no cover
    ak = None

from data.index_query import IndexQueryService  # noqa: F401
from data.cache_db import (  # noqa: F401
    get_cache_db, CacheDB,
    NS_QUOTE, NS_KLINE, NS_INTRADAY, NS_CONSTITUENT, NS_NEWS, NS_RESEARCH,
    NS_INDEX_SPOT,
)


# ----------------------------------------------------------------------
# 新浪 HTTP 实时行情（直连 hq.sinajs.cn）
# ----------------------------------------------------------------------
# 东方财富 / 部分 akshare 接口在本环境可能连接失败，但新浪实时接口在
# 非交易时段（盘后 / 午休 / 休市）仍然可用，并返回当日完整行情
# （开盘 / 最高 / 最低 / 现价 / 昨收 / 成交额 / 成交量 / 时间）。
# 因此这里提供直连新浪 HTTP 的可靠回退，保证非交易时段也能取到数据。
try:
    import requests as _requests
except Exception:  # pragma: no cover
    _requests = None

_SINA_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://finance.sina.com.cn",
}

# 新浪 hq.sinajs.cn 单次请求对代码数量有上限（过多会静默返回空），批量取数需分片。
_SINA_CHUNK = 150
# 每代码行情短时缓存，避免大批量（如指数全部成分股）在每次重渲染时重复抓取。
_QUOTE_CACHE: dict = {}
_QUOTE_TTL = 20
# 整批行情缓存（quotes_for 结果），避免成分股表每次重渲染都重新抓取上千只。
_QUOTES_CACHE: dict = {}
_QUOTES_TTL = 20


def _sina_http_text(codes: list) -> dict:
    """直连新浪 hq.sinajs.cn，批量取实时字符串。

    codes 形如 ``["sh600519", "sz000858", "s_sh000001"]``。
    返回 ``{原始代码: 逗号分隔值字符串}``；请求失败返回空 dict。
    """
    if _requests is None or not codes:
        return {}
    try:
        url = "https://hq.sinajs.cn/list=" + ",".join(codes)
        r = _requests.get(url, headers=_SINA_HEADERS, timeout=10)
        r.encoding = "gbk"
        out = {}
        for line in r.text.strip().split(";"):
            line = line.strip()
            if not line.startswith("var hq_str"):
                continue
            key, _, val = line.partition("=")
            sym = key.replace("var hq_str_", "").strip()
            out[sym] = val.strip().strip('"')
        return out
    except Exception:
        return {}


def _sina_symbol(code: str) -> str:
    """将 6 位 A 股代码转换为新浪股票代码（sh/sz 前缀）。"""
    code = str(code).strip().zfill(6)
    return ("sh" if code.startswith(("6", "9")) else "sz") + code


def _sina_index_prefix(code: str) -> str:
    """指数代码 -> 新浪前缀（上证 sh / 深证 sz / 北交 bj）。"""
    code = str(code).strip().zfill(6)
    if code.startswith(("0", "1")):
        return "sh" + code       # 上证 / 中证（000/001 开头）归上证通道
    if code.startswith(("3", "2")):
        return "sz" + code
    if code.startswith("8"):
        return "bj" + code
    return "sh" + code


def _sina_detail_df(codes: list) -> pd.DataFrame:
    """将新浪个股详情（hq_str_xxx=...）解析为规整 DataFrame。

    列与 akshare ``stock_zh_a_spot_em`` 对齐：代码 / 名称 / 最新价 /
    涨跌幅 / 涨跌额 / 成交量 / 成交额 / 今开 / 最高 / 最低 / 昨收。

    新浪 ``hq.sinajs.cn`` 单次请求对代码数量有上限（过多会静默返回空），
    故按 ``_SINA_CHUNK`` 分片请求并合并；同时复用 ``_QUOTE_CACHE`` 的
    每代码短时缓存，避免大批量（如指数全部成分股）在每次重渲染时重复抓取。
    """
    codes = [str(c).strip().zfill(6) for c in codes]
    if not codes:
        return pd.DataFrame()
    now = time.time()
    rows = []
    need = []
    for code in codes:
        hit = _QUOTE_CACHE.get(code)
        if hit is not None and now - hit[0] < _QUOTE_TTL:
            rows.append(hit[1])
        else:
            need.append(code)
    for i in range(0, len(need), _SINA_CHUNK):
        batch = need[i:i + _SINA_CHUNK]
        raw = _sina_http_text([_sina_symbol(c) for c in batch])
        for code in batch:
            sym = _sina_symbol(code)
            s = raw.get(sym, "")
            if not s:
                continue
            f = s.split(",")
            if len(f) < 32:
                continue
            try:
                price = float(f[3])
                prev_close = float(f[2])
            except Exception:
                continue
            chg = round(price - prev_close, 4)
            chg_pct = round(chg / prev_close * 100, 4) if prev_close else 0.0
            row = {
                "代码": code,
                "名称": f[0],
                "最新价": price,
                "涨跌幅": chg_pct,
                "涨跌额": chg,
                "成交量": float(f[8]) if f[8] else 0.0,
                "成交额": float(f[9]) if f[9] else 0.0,
                "今开": float(f[1]) if f[1] else None,
                "最高": float(f[4]) if f[4] else None,
                "最低": float(f[5]) if f[5] else None,
                "昨收": prev_close,
            }
            _QUOTE_CACHE[code] = (now, row)
            rows.append(row)
    return pd.DataFrame(rows)


def _sina_full_spot() -> pd.DataFrame:
    """用新浪直连分片批量构造沪深京 A 股全市场实时快照（akshare 全市场接口不可用时的兜底）。

    先经 ``ak.stock_info_a_code_name`` 取全部 A 股代码列表，再按 ``_SINA_CHUNK``
    分片调用 ``_sina_detail_df`` 合并。返回列与 ``stock_zh_a_spot_em`` 对齐。
    任何一步失败均返回空 DataFrame，交由上层继续回退。
    """
    if ak is None:
        return pd.DataFrame()
    try:
        info = ak.stock_info_a_code_name()
    except Exception:
        return pd.DataFrame()
    if info is None or info.empty:
        return pd.DataFrame()
    code_col = next((c for c in info.columns if "代码" in str(c)), info.columns[0])
    codes = [str(r[code_col]).strip().zfill(6) for _, r in info.iterrows()
             if str(r[code_col]).strip()]
    if not codes:
        return pd.DataFrame()
    return _sina_detail_df(codes)


# ----------------------------------------------------------------------
# 缓存（带 TTL）：避免 Streamlit 重渲染时反复请求网络
# ----------------------------------------------------------------------
_TTL_SECONDS = 300  # 5 分钟
_CACHE: dict = {}


def _cached(ttl: int = _TTL_SECONDS):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (fn.__qualname__, args, tuple(sorted(kwargs.items())))
            now = time.time()
            if key in _CACHE:
                ts, val = _CACHE[key]
                if now - ts < ttl:
                    return val
            val = fn(*args, **kwargs)
            _CACHE[key] = (now, val)
            return val
        return wrapper
    return deco


# K 线列名归一化：不同数据源（东财/同花顺=中文，新浪/Tushare=英文）列名不一致，
# 渲染层统一按中文列名取值，故在返回前统一规整，避免 KeyError（如 '收盘'）。
_KLINE_COL_MAP = {
    "日期": ("日期", "date", "Date", "trade_date", "时间", "datetime"),
    "开盘": ("开盘", "open", "Open"),
    "收盘": ("收盘", "close", "Close"),
    "最高": ("最高", "high", "High"),
    "最低": ("最低", "low", "Low"),
    "成交量": ("成交量", "volume", "vol", "Volume", "成交量(股)", "成交量(手)"),
    "成交额": ("成交额", "amount", "Amount"),
}


def _normalize_kline(df):
    """将 K 线 DataFrame 的列名统一为中文（日期/开盘/收盘/最高/最低/成交量/成交额）。

    兼容东财/同花顺（中文）与新浪/Tushare（英文）两种命名，缺失的列不补。
    """
    if df is None or df.empty:
        return df
    rename = {}
    for canon, aliases in _KLINE_COL_MAP.items():
        for a in aliases:
            if a in df.columns:
                if a != canon:
                    rename[a] = canon
                break
    if rename:
        df = df.rename(columns=rename)
    return df


def _safe(label: str):
    """装饰器：捕获异常，统一返回 ``(DataFrame, error_or_None)``。"""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if ak is None:
                return pd.DataFrame(), "未安装 akshare，请执行 pip install akshare"
            try:
                df = fn(*args, **kwargs)
                if df is None:
                    df = pd.DataFrame()
                elif not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                return df, None
            except Exception as e:  # 网络/接口异常
                return pd.DataFrame(), f"{label} 获取失败：{e}"
        return wrapper
    return deco


def _today_str() -> str:
    return _dt.date.today().strftime("%Y%m%d")


def _days_ago_str(days: int) -> str:
    return (_dt.date.today() - _dt.timedelta(days=days)).strftime("%Y%m%d")


class MarketDataFetcher:
    """封装期货 / 期权 / 基金 / 债券 / 外汇 / 贵金属等实时与历史行情。"""

    # ==================================================================
    # 期货
    # ==================================================================
    @staticmethod
    @_cached()
    @_safe("期货主力实时行情")
    def futures_main_spot():
        """全部期货主力合约实时行情（新浪）。"""
        return ak.futures_display_main_sina()

    @staticmethod
    @_cached()
    @_safe("期货 K 线")
    def futures_kline(symbol: str, days: int = 120):
        """期货历史 K 线（新浪）。symbol 如 ``V0``/``RB0``/``AU0``（主力连续）。

        东方财富期货历史接口在本环境不稳定，改用新浪日线（返回 date/open/
        high/low/close/volume），并截取最近 ``days`` 个交易日。
        """
        df = ak.futures_zh_daily_sina(symbol=symbol)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        return df.tail(int(days))

    # ==================================================================
    # 期权
    # ==================================================================
    @staticmethod
    @_cached()
    @_safe("上交所期权实时行情")
    def option_sse_spot():
        """上交所当日全部期权实时行情。"""
        return ak.option_current_day_sse()

    @staticmethod
    @_cached()
    @_safe("商品期权合约链")
    def option_commodity_chain(underlying: str):
        """商品期权合约链（新浪）。underlying 如 ``白糖期权``/``黄金期权``。"""
        return ak.option_commodity_contract_sina(symbol=underlying)

    @staticmethod
    @_cached()
    @_safe("期权 K 线")
    def option_kline(symbol: str):
        """单个期权合约历史 K 线（新浪）。symbol 为期权合约代码，如 ``au2012C392``。"""
        return ak.option_commodity_hist_sina(symbol=symbol)

    # ==================================================================
    # 基金
    # ==================================================================
    @staticmethod
    @_cached()
    @_safe("ETF 实时行情")
    def fund_etf_spot():
        """ETF 实时行情（东方财富）。"""
        return ak.fund_etf_spot_em()

    @staticmethod
    @_cached()
    @_safe("LOF 实时行情")
    def fund_lof_spot():
        """LOF 实时行情（东方财富）。"""
        return ak.fund_lof_spot_em()

    @staticmethod
    @_cached()
    @_safe("ETF K 线")
    def fund_etf_kline(symbol: str, period: str = "daily", days: int = 180):
        """ETF 历史 K 线（东方财富）。symbol 如 ``510050`` / ``159707``。"""
        return ak.fund_etf_hist_em(
            symbol=symbol,
            period=period,
            start_date=_days_ago_str(days * 2),
            end_date=_today_str(),
            adjust="",
        )

    @staticmethod
    @_cached()
    @_safe("开放基金排行")
    def fund_open_rank(category: str = "全部"):
        """开放基金排行（东方财富）。category 如 ``全部``/``股票型``/``混合型``/``债券型``。"""
        return ak.fund_open_fund_rank_em(symbol=category)

    @staticmethod
    @_cached()
    @_safe("基金净值走势")
    def fund_open_nav(symbol: str):
        """开放式基金单位净值走势（东方财富）。symbol 为 6 位基金代码。"""
        return ak.fund_open_fund_info_em(symbol=symbol, indicator="单位净值走势", period="近一年")

    # ==================================================================
    # 债券 / 外汇 / 贵金属（其他金融工具）
    # ==================================================================
    @staticmethod
    @_cached()
    @_safe("可转债实时行情")
    def bond_cov_spot():
        """沪深可转债实时行情。"""
        return ak.bond_zh_hs_cov_spot()

    @staticmethod
    @_cached()
    @_safe("外汇实时行情")
    def forex_spot():
        """外汇实时行情（东方财富）。"""
        return ak.forex_spot_em()

    @staticmethod
    @_cached()
    @_safe("上海金基准价")
    def gold_spot():
        """上海黄金交易所黄金基准价。"""
        return ak.spot_golden_benchmark_sge()

    @staticmethod
    @_cached()
    @_safe("中国银行外汇牌价")
    def currency_boc():
        """中国银行外汇牌价（现汇买入/卖出等）。"""
        return ak.currency_boc_safe()

    # ==================================================================
    # 股票（A 股）
    # ==================================================================
    @staticmethod
    def _sina_symbol(code: str) -> str:
        """6 位代码转新浪代码前缀：6->sh, 0/3->sz, 4/8->bj。"""
        code = code.strip()
        if code[:2].lower() in ("sh", "sz", "bj"):
            return code
        if code[0] == "6":
            return "sh" + code
        if code[0] in ("0", "3"):
            return "sz" + code
        return "bj" + code

    @staticmethod
    @_cached(ttl=10)
    @_safe("沪深A股实时行情")
    def stock_spot():
        """沪深京 A 股全部实时行情。多源回退以保证非交易时段也能取数：

        东方财富 ``stock_zh_a_spot_em`` → 新浪 ``stock_zh_a_spot`` →
        直连新浪 HTTP ``hq.sinajs.cn``（按代码分片批量取，构造全市场快照）。

        返回 ``(DataFrame, error)`` 元组（由 ``@_safe`` 包裹），列含
        代码 / 名称 / 最新价 / 涨跌幅 / 涨跌额 / 成交量 / 成交额 / 今开 /
        最高 / 最低 / 昨收。
        """
        if ak is not None:
            try:
                df = ak.stock_zh_a_spot_em()
                if df is not None and not df.empty:
                    return df
            except Exception:
                pass
            try:
                df = ak.stock_zh_a_spot()
                if df is not None and not df.empty:
                    return df
            except Exception:
                pass
            # 东财/新浪 akshare 全市场接口均失败（如本环境 numpy 兼容性报错），
            # 用新浪直连分片批量取全市场快照兜底，保证非交易时段也能取到数据。
            try:
                snap = _sina_full_spot()
                if snap is not None and not snap.empty:
                    return snap
            except Exception:
                pass
        return pd.DataFrame()

    @staticmethod
    @_cached(ttl=5)
    @_safe("个股实时行情")
    def stock_realtime(symbol: str):
        """单只 A 股实时行情（取全市场快照后按代码过滤，含最新价/涨跌幅/量额等）。

        复用 ``stock_spot`` 的短时缓存快照，避免每只股票单独拉取全市场数据。
        """
        df, err = MarketDataFetcher.stock_spot()
        code = str(symbol).strip().zfill(6)
        if df is not None and not df.empty:
            try:
                mask = df["代码"].astype(str).str.zfill(6) == code
            except Exception:
                mask = df["代码"].astype(str) == code
            row = df[mask]
            if not row.empty:
                return row.copy()
        # 全市场快照缺该只 → 直连新浪 HTTP 取单只当日行情（非交易时段可用）
        sina_df = _sina_detail_df([code])
        if not sina_df.empty:
            return sina_df
        return pd.DataFrame()

    @staticmethod
    @_cached(ttl=15)
    @_safe("股票 K 线")
    def stock_kline(symbol: str, period: str = "daily", days: int = 180, adjust: str = ""):
        """A 股历史 K 线。symbol 为 6 位代码，如 ``000001``/``600519``。

        period：``daily``/``weekly``/``monthly``；adjust：``""``（不复权）/
        ``qfq``（前复权）/``hfq``（后复权）。优先东方财富，失败时回退新浪日线。
        15 秒缓存，配合自动刷新可更新当日最后一根 K 线。
        """
        _min_rows = max(20, int(days * 0.5))
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period=period,
                start_date=_days_ago_str(days * 2),
                end_date=_today_str(),
                adjust=adjust,
            )
            if df is not None and not df.empty and len(df) >= _min_rows:
                return _normalize_kline(df)
        except Exception:
            pass
        sina_sym = MarketDataFetcher._sina_symbol(symbol)
        sina_adj = adjust if adjust in ("", "qfq", "hfq") else ""
        if ak is not None:
            try:
                d = ak.stock_zh_a_daily(symbol=sina_sym, adjust=sina_adj)
                if d is not None and isinstance(d, pd.DataFrame) and len(d) >= 5:
                    return _normalize_kline(d.tail(int(days)))
            except Exception:
                pass
        # 二级回退：Tushare（已在 config.yaml 配置 token，非交易时段可用）
        try:
            from data.fetcher import DataFetcher
            df = DataFetcher().get_daily_kline(
                symbols=[symbol.zfill(6)],
                start=_days_ago_str(days * 2), end=_today_str(),
                period="daily", adjust=sina_adj or "qfq",
            )
            if df is not None and not df.empty and len(df) >= 5:
                return _normalize_kline(df.tail(int(days)))
        except Exception:
            pass
        return pd.DataFrame()

    @staticmethod
    @_cached(ttl=15)
    @_safe("股票分时")
    def stock_intraday(symbol: str):
        """A 股当日分时数据（东方财富，失败时回退新浪分钟线）。symbol 为 6 位代码。"""
        try:
            return ak.stock_intraday_em(symbol=symbol)
        except Exception:
            sina_sym = MarketDataFetcher._sina_symbol(symbol)
            return ak.stock_zh_a_minute(symbol=sina_sym, period="1")

    # ------------------------------------------------------------------
    # 指数行情（五大指数 / 成分股 / 新闻研报）
    # ------------------------------------------------------------------
    @staticmethod
    @_safe("指数实时行情")
    def index_spot(code: str):
        """单只指数实时点（名称/最新/涨跌/涨跌幅），使用后台 SQLite 缓存。

        code 为指数代码，如 ``000001``（上证指数）/``399006``（创业板指）/
        ``899050``（北证50）/``000688``（科创50）。

        ``IndexQueryService.get_index_spot`` 可能返回 dict（同花顺）或历史末行
        dict（东财回退），此处统一规整为单行的 DataFrame。
        """
        cache = get_cache_db()
        hit = cache.get(NS_INDEX_SPOT, code, ttl=15)
        if hit is not None:
            return pd.DataFrame(hit)

        name, value, chg, chg_pct = code, None, None, None

        # 主源：IndexQueryService（THS / 东财 / 新浪历史末行），兼容多种字段命名。
        # 归一化历史末行用 close/pct_chg/change；同花顺实时用 收盘/最新/value 等。
        raw = IndexQueryService().get_index_spot(code)
        if isinstance(raw, dict) and raw:
            value = (raw.get("value") or raw.get("收盘") or raw.get("最新")
                     or raw.get("close") or raw.get("price"))
            chg = (raw.get("chg") or raw.get("涨跌") or raw.get("change") or raw.get("涨跌额"))
            chg_pct = (raw.get("chg_pct") or raw.get("涨跌幅") or raw.get("pct_chg"))
            name = raw.get("名称") or raw.get("name") or code
            if value is not None:
                try:
                    value = float(value)
                except Exception:
                    value = None
        elif isinstance(raw, pd.DataFrame) and not raw.empty:
            last = raw.iloc[-1]
            value = last.get("close") or last.get("value") or last.get("收盘")
            chg_pct = last.get("pct_chg") or last.get("chg_pct") or last.get("涨跌幅")
            chg = last.get("change") or last.get("chg") or last.get("涨跌")

        # 主源无效（None / 0 / NaN）→ 直连新浪 HTTP 取指数实时（非交易时段返回当日收盘）
        _invalid = value is None or (isinstance(value, float) and (value == 0 or value != value))
        if _invalid:
            s_sym = _sina_index_prefix(code)
            s_text = _sina_http_text(["s_" + s_sym])
            s = s_text.get("s_" + s_sym, "")
            if s:
                f = s.split(",")
                try:
                    name = f[0] if f else code
                    value = float(f[1]) if len(f) > 1 and f[1] not in ("", "-") else None
                    chg = float(f[2]) if len(f) > 2 and f[2] not in ("", "-") else None
                    chg_pct = float(f[3]) if len(f) > 3 and f[3] not in ("", "-") else None
                except Exception:
                    value = chg = chg_pct = None
        else:
            # 主源 value 有效，但归一化历史末行常缺涨跌额，用新浪补充 chg
            if chg is None or (isinstance(chg, float) and chg != chg):
                s_sym = _sina_index_prefix(code)
                s_text = _sina_http_text(["s_" + s_sym])
                s = s_text.get("s_" + s_sym, "")
                if s:
                    f = s.split(",")
                    try:
                        chg = float(f[2]) if len(f) > 2 and f[2] not in ("", "-") else chg
                    except Exception:
                        pass

        if value is None:
            return pd.DataFrame()
        df = pd.DataFrame([{
            "名称": name,
            "value": value,
            "chg": chg,
            "chg_pct": chg_pct,
        }])
        cache.set(NS_INDEX_SPOT, code, df.to_dict(orient="records"))
        return df

    @staticmethod
    @_safe("指数 K 线")
    def index_kline(code: str, period: str = "daily", days: int = 180, adjust: str = ""):
        """指数历史 K 线（含五大指数）。多源回退：同花顺/东财 → 新浪日线。

        带后台 SQLite 缓存（15 秒），配合自动刷新可更新当日最后一根 K 线。
        非交易时段仍可取到全部历史（含当日）的日 K 线。
        """
        cache = get_cache_db()
        ckey = f"{code}|{period}|{adjust}|{days}"
        hit = cache.get(NS_KLINE, "idx_" + ckey, ttl=15)
        if hit is not None:
            return _normalize_kline(pd.DataFrame(hit))
        _min_rows = max(20, int(days * 0.5))
        df = IndexQueryService().get_index_hist(
            code, start=_days_ago_str(days * 2), end=_today_str(),
            period=period, adjust=adjust,
        )
        if df is not None and not df.empty and len(df) >= _min_rows:
            df = _normalize_kline(df)
            cache.set(NS_KLINE, "idx_" + ckey, df.to_dict(orient="records"))
            return df
        # 主源行数不足/失败 → 新浪指数日线（akshare）回退
        if ak is not None:
            try:
                s_sym = _sina_index_prefix(code)
                d = ak.stock_zh_index_daily(symbol=s_sym)
                if d is not None and not d.empty and len(d) >= 5:
                    d = _normalize_kline(d.tail(int(days)))
                    cache.set(NS_KLINE, "idx_" + ckey, d.to_dict(orient="records"))
                    return d
            except Exception:
                pass
        return pd.DataFrame()

    @staticmethod
    def index_constituents(index_code: str):
        """返回 ``(list, error)``，list 形如 ``[{"code":.., "name":..}, ...]``（带长缓存）。

        多数据源回退：东方财富 ``index_stock_cons`` → 新浪 ``index_stock_cons_sina``。
        成分股变动缓慢，缓存 1 小时。注意：此方法返回 list 而非 DataFrame，
        故不使用 ``@_safe``（其会将非 DataFrame 返回值强制转为 DataFrame）。
        """
        cache = get_cache_db()
        hit = cache.get(NS_CONSTITUENT, index_code, ttl=3600)
        if hit is not None:
            return hit, None
        out = []
        df = None
        err = None
        try:
            df = ak.index_stock_cons(symbol=index_code)
        except Exception as e:  # noqa
            err = f"东方财富成分股接口失败：{e}"
            try:
                df = ak.index_stock_cons_sina(symbol=index_code)
                err = None
            except Exception as e2:  # noqa
                err = err or f"新浪成分股接口失败：{e2}"
        if df is not None and not df.empty:
            code_col = next((c for c in df.columns if "代码" in str(c)), df.columns[0])
            name_col = next((c for c in df.columns if "名称" in str(c)), df.columns[1])
            for _, r in df.iterrows():
                out.append({
                    "code": str(r[code_col]).strip().zfill(6),
                    "name": str(r[name_col]).strip(),
                })
        if out:
            cache.set(NS_CONSTITUENT, index_code, out)
            return out, None
        return [], err or "暂未获取到成分股（接口受限或非交易时段）"

    @staticmethod
    @_safe("个股新闻")
    def stock_news(symbol: str, days: int = 30):
        """个股近期新闻（东方财富 ``stock_news_em``），带后台 SQLite 缓存。"""
        cache = get_cache_db()
        ckey = f"{symbol}|{days}"
        hit = cache.get(NS_NEWS, ckey, ttl=600)
        if hit is not None:
            return pd.DataFrame(hit)
        from data.fetcher import DataFetcher

        df = DataFetcher.get_news_sentiment(symbol, date=None)
        if df is not None and not df.empty:
            df = df.head(int(days * 3))
            cache.set(NS_NEWS, ckey, df.to_dict(orient="records"))
        return df if df is not None else pd.DataFrame()

    @staticmethod
    @_safe("个股研报")
    def stock_research(symbol: str):
        """个股机构研报（东方财富 ``stock_research_report_em``），带后台 SQLite 缓存。"""
        cache = get_cache_db()
        hit = cache.get(NS_RESEARCH, symbol, ttl=1800)
        if hit is not None:
            return pd.DataFrame(hit)
        df = ak.stock_research_report_em(symbol=str(symbol).strip().zfill(6))
        if df is not None and not df.empty:
            cache.set(NS_RESEARCH, symbol, df.to_dict(orient="records"))
        return df if df is not None else pd.DataFrame()

    @staticmethod
    def quotes_for(codes):
        """返回 ``(dict, error)``，dict 形如 ``{code: {列:值}}``（带长缓存）。

        对一组 6 位代码一次性取实时行情（复用全市场快照，单次网络请求）。
        缺失代码不出现（接口不可用 / 非交易时段）。不使用 ``@_safe``，
        因其返回 dict 而非 DataFrame。
        """
        codes = [str(c).strip().zfill(6) for c in codes]
        # 整批缓存：成分股表每次重渲染都会调用本函数，命中缓存可避免重复抓取上千只。
        cache_key = tuple(sorted(codes))
        now = time.time()
        hit = _QUOTES_CACHE.get(cache_key)
        if hit is not None and now - hit[0] < _QUOTES_TTL:
            return hit[1], None
        df, err = MarketDataFetcher.stock_spot()
        out = {}
        if df is not None and not df.empty:
            snap = df.copy()
            snap["_code"] = snap["代码"].astype(str).str.zfill(6)
            sub = snap[snap["_code"].isin(codes)]
            for _, r in sub.iterrows():
                out[str(r["_code"])] = r.drop(labels=["_code"]).to_dict()
        # 快照缺失的部分（东财/新浪全市场接口失败或个股不在快照中），
        # 用直连新浪 HTTP 逐只补全，保证非交易时段也能取到当日行情。
        missing = [c for c in codes if c not in out]
        if missing:
            sina_df = _sina_detail_df(missing)
            if not sina_df.empty:
                for _, r in sina_df.iterrows():
                    out[str(r["代码"])] = r.to_dict()
        if out:
            _QUOTES_CACHE[cache_key] = (now, out)
            return out, None
        return {}, (err or "行情快照为空，且新浪直连取数失败（请检查网络）")


# 常用主力连续合约代码提示，供期货 K 线输入参考
# 每个元素为 (名称, 主力合约代码) 元组
FUTURES_MAIN_HINTS = [
    ("PVC", "V0"), ("螺纹钢", "RB0"), ("热卷", "HC0"), ("铁矿石", "I0"),
    ("焦炭", "J0"), ("焦煤", "JM0"), ("沪铜", "CU0"), ("沪铝", "AL0"),
    ("沪锌", "ZN0"), ("沪镍", "NI0"), ("沪金", "AU0"), ("沪银", "AG0"),
    ("沥青", "BU0"), ("燃油", "FU0"), ("PTA", "TA0"), ("甲醇", "MA0"),
    ("PP", "PP0"), ("乙二醇", "EG0"), ("白糖", "SR0"), ("棉花", "CF0"),
    ("豆一", "A0"), ("豆粕", "M0"), ("豆油", "Y0"), ("棕榈油", "P0"),
    ("玉米", "C0"), ("玉米淀粉", "CS0"), ("菜油", "OI0"), ("菜粕", "RM0"),
    ("纸浆", "SP0"), ("LPG", "PG0"),
]
# 仅代码列表，供文本拼接提示使用
FUTURES_SYMBOLS = [s for _, s in FUTURES_MAIN_HINTS]
