"""指数行情查询服务（优先同花顺 THS，AKShare 新浪回退）。

对外提供：
- COMMON_INDICES：常见 A 股指数代码字典（代码 -> 名称/交易所前缀）
- search(keyword)：按代码或名称模糊检索指数
- get_index_hist(index_code, start, end, period, adjust)：指数历史 K 线
- get_index_spot(index_code)：指数实时行情（优先 THS，否则取历史最新）
- get_index_intraday(index_code)：指数分时数据（优先 THS）

数据源策略：
1. 同花顺 THS（若已配置 THSDataFetcher）
2. AKShare 新浪接口（stock_zh_index_daily）—— 在东方财富被封锁的网络环境下可用
"""

from typing import Optional, List, Dict, Any

import re
import pandas as pd


class IndexQueryService:
    """指数行情查询服务。"""

    COMMON_INDICES: Dict[str, Dict[str, str]] = {
        "000001": {"name": "上证指数", "prefix": "sh"},
        "399001": {"name": "深证成指", "prefix": "sz"},
        "399006": {"name": "创业板指", "prefix": "sz"},
        "000300": {"name": "沪深300", "prefix": "sh"},
        "000905": {"name": "中证500", "prefix": "sh"},
        "000906": {"name": "中证800", "prefix": "sh"},
        "000016": {"name": "上证50", "prefix": "sh"},
        "000688": {"name": "科创50", "prefix": "sh"},
        "000852": {"name": "中证1000", "prefix": "sh"},
        "399005": {"name": "中小100", "prefix": "sz"},
        "000903": {"name": "中证100", "prefix": "sh"},
        "000010": {"name": "上证180", "prefix": "sh"},
        "931643": {"name": "科创100", "prefix": "sh"},
    }

    def __init__(self, ths_fetcher=None, config: Optional[dict] = None) -> None:
        """初始化。

        Args:
            ths_fetcher: 已初始化的 THSDataFetcher 实例（可选，优先数据源）。
            config: 全局配置字典（可选），预留扩展。
        """
        self.ths = ths_fetcher
        self.config = config or {}

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search(self, keyword: str) -> List[Dict[str, str]]:
        """按代码或名称模糊检索指数。"""
        kw = (keyword or "").strip().lower()
        if not kw:
            return [{"code": c, "name": m["name"]} for c, m in self.COMMON_INDICES.items()]
        return [
            {"code": code, "name": m["name"]}
            for code, m in self.COMMON_INDICES.items()
            if kw in code.lower() or kw in m["name"].lower()
        ]

    # ------------------------------------------------------------------
    # 历史 K 线
    # ------------------------------------------------------------------
    def get_index_hist(
        self,
        index_code: str,
        start: str,
        end: str,
        period: str = "daily",
        adjust: str = "",
    ) -> pd.DataFrame:
        """获取指数历史 K 线（多源回退）。"""
        code = self._normalize_code(index_code)
        start_fmt = self._fmt(start)
        end_fmt = self._fmt(end)

        # 1) 同花顺 THS 优先
        if self.ths is not None:
            try:
                df = self.ths.get_index_hist(code, start_fmt, end_fmt, period=period)
                if df is not None and not df.empty:
                    return self._normalize_index(df, code)
            except Exception as e:
                print(f"[IndexQuery] 同花顺指数历史获取失败: {e}")

        # 2) AKShare 新浪回退
        df = self._akshare_index_hist(code, start_fmt, end_fmt)
        if df is not None and not df.empty:
            return self._normalize_index(df, code)

        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 实时行情
    # ------------------------------------------------------------------
    def get_index_spot(self, index_code: str) -> Dict[str, Any]:
        """获取指数实时行情（优先 THS，否则取历史最新一日）。"""
        code = self._normalize_code(index_code)
        if self.ths is not None:
            try:
                spot = self.ths.get_index_spot(code)
                if spot:
                    return spot
            except Exception as e:
                print(f"[IndexQuery] 同花顺指数实时获取失败: {e}")

        df = self.get_index_hist(code, "20000101", self._fmt("today"))
        if df is not None and not df.empty:
            return df.iloc[-1].to_dict()
        return {}

    # ------------------------------------------------------------------
    # 分时
    # ------------------------------------------------------------------
    def get_index_intraday(self, index_code: str) -> pd.DataFrame:
        """获取指数分时数据（优先 THS）。"""
        code = self._normalize_code(index_code)
        if self.ths is not None:
            try:
                df = self.ths.get_index_intraday(code)
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                print(f"[IndexQuery] 同花顺指数分时获取失败: {e}")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _normalize_code(self, code: str) -> str:
        code = str(code).strip()
        code = re.sub(r"^(sh|sz|bj|ib)", "", code, flags=re.IGNORECASE)
        return code.zfill(6) if code.isdigit() else code

    def _fmt(self, d: str) -> str:
        if d in (None, "today", ""):
            return pd.Timestamp.now().strftime("%Y%m%d")
        return str(d).replace("-", "").replace("/", "")

    def _akshare_index_hist(self, code: str, start_fmt: str, end_fmt: str) -> pd.DataFrame:
        """AKShare 新浪指数历史（stock_zh_index_daily）。"""
        try:
            import akshare as ak

            prefix = self.COMMON_INDICES.get(code, {}).get("prefix", "sh")
            sym = f"{prefix}{code}"
            df = ak.stock_zh_index_daily(symbol=sym)
            if df is None or df.empty:
                return pd.DataFrame()

            # 兼容 date 作为列或作为索引的两种返回结构（不同 akshare 版本差异）
            if "date" not in df.columns:
                if str(df.index.name) in ("date", "日期"):
                    df = df.reset_index()
                elif "日期" in df.columns:
                    df = df.rename(columns={"日期": "date"})
                else:
                    # 兜底：把首个看起来像日期的索引重置为列
                    df = df.reset_index()
                # 重置后列名可能不是 'date'，统一改名
                for cand in ("date", "日期", "index"):
                    if cand in df.columns and cand != "date":
                        df = df.rename(columns={cand: "date"})
                        break
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            if df.empty:
                return pd.DataFrame()
            lo = pd.to_datetime(start_fmt)
            hi = pd.to_datetime(end_fmt)
            df = df.loc[(df["date"] >= lo) & (df["date"] <= hi)].copy()
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
            df = df.sort_values("date").reset_index(drop=True)
            df["symbol"] = code
            return df
        except Exception as e:
            print(f"[IndexQuery] AKShare 指数历史获取失败: {e}")
            return pd.DataFrame()

    @staticmethod
    def _normalize_index(df: pd.DataFrame, code: str) -> pd.DataFrame:
        df = df.copy()
        rename = {
            "日期": "date", "开盘": "open", "最高": "high", "最低": "low",
            "收盘": "close", "成交量": "volume", "成交额": "amount",
        }
        df = df.rename(columns=rename)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["symbol"] = code
        for c in ["open", "high", "low", "close", "volume", "amount"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in ["amount", "amplitude", "pct_chg", "change", "turnover"]:
            if c not in df.columns:
                df[c] = float("nan")
        if "pct_chg" in df.columns and df["pct_chg"].isna().all() and "close" in df.columns:
            df["pct_chg"] = df["close"].pct_change() * 100
        return df
