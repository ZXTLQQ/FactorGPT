"""
FactorGPT 稳定数据源适配器（NeoData）
====================================

本模块解决 FactorGPT 「数据源不稳定」的根因：原 ``DataFetcher`` 直接调用
akshare / sina / tushare / efinance 爬数据，上游网站改版即中断。

这里提供 ``NeoDataSource``，对外暴露与 ``DataFetcher`` **完全一致** 的公开方法签名，
但底层改为调用平台内置的 ``neodata-financial-search`` 技能（由平台维护可用性，
鉴权 token 持久化在 ``~/.workbuddy/.neodata_token`` 或 ``~/.codebuddy/.neodata_token``）。

设计原则
--------
1. 接口对齐：方法名 / 入参 / 返回（pandas DataFrame 列约定）与 ``DataFetcher`` 一致，
   上层 graph / refinery / factor_system / market_data 无需改动。
2. 安全回退：NeoData 未配置、网络失败或字段未覆盖时，按配置回退到 legacy ``DataFetcher``，
   保证切换过程不破坏现有功能（过渡期默认开启，可在 config 关闭以强制只用稳定源）。
3. 零新依赖：仅用标准库 ``urllib``，不改动 ``requirements.txt``。

启用方式
--------
在 ``config.yaml`` 中设置 ``data.source: neodata``，并把调用点 ``DataFetcher()``
替换为 ``get_data_source(config)``（见文件底部 ``DataSourceFactory``）。
或在 Skill 脚本中直接 ``from data.neo_adapter import get_data_source``。

注意
----
NeoData 网关的具体端点路径以平台 ``neodata-financial-search`` 技能 SKILL.md 为准；
本适配器的 ``NeoDataClient`` 已将端点设为可配置项（``data.neodata.base_url``），
只需填入正确路径即可，字段映射在 ``_map_*`` 方法中集中维护。
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import pandas as pd

# 路径引导：直接以脚本方式运行（python src/data/neo_adapter.py）时，
# 也能解析 ``data.fetcher`` 的绝对导入；作为模块导入时 src 已在路径上，无副作用。
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, ".."), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:  # 复用既有 DataFetcher 作为回退与字段约定参考
    from data.fetcher import DataFetcher
except Exception:  # pragma: no cover - 允许独立测试
    DataFetcher = None  # type: ignore


_TOKEN_CANDIDATES = (
    os.path.expanduser("~/.workbuddy/.neodata_token"),
    os.path.expanduser("~/.codebuddy/.neodata_token"),
)


def _load_neodata_token(env_name: str = "NEODATA_TOKEN") -> Optional[str]:
    """按 环境变量 -> 平台 token 文件 的优先级读取 NeoData 鉴权令牌。"""
    tok = os.environ.get(env_name)
    if tok:
        return tok.strip()
    for path in _TOKEN_CANDIDATES:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    return fh.read().strip()
            except Exception:  # noqa: BLE001
                continue
    return None


class NeoDataClient:
    """NeoData 网关的轻量 HTTP 客户端（标准库实现，无第三方依赖）。

    端点路径以平台 ``neodata-financial-search`` 技能 SKILL.md 为准；此处仅作占位，
    配置 ``base_url`` 后填入真实路径即可。所有方法返回解析后的 JSON（dict / list）。
    """

    def __init__(
        self,
        base_url: str = "",
        token: Optional[str] = None,
        token_env: str = "NEODATA_TOKEN",
        timeout: float = 15.0,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.token = token or _load_neodata_token(token_env)
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        """NeoData 是否已配置（有网关地址且有 token）。"""
        return bool(self.base_url) and bool(self.token)

    def _request(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if not self.configured:
            raise RuntimeError("NeoData 未配置：请在 config 设置 data.neodata.base_url 并确保 token 可用")
        url = f"{self.base_url}/{path.lstrip('/')}"
        if params:
            q = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
            url = f"{url}?{q}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.token or ''}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ---- 各取数端点的封装（路径以平台 SKILL.md 为准，此处为占位约定） ----
    def kline(self, symbol: str, start: str, end: str, period: str = "daily", adjust: str = "qfq") -> Any:
        return self._request(
            "v1/quote/kline",
            {"symbol": symbol, "start": start, "end": end, "period": period, "adjust": adjust},
        )

    def stock_list(self) -> Any:
        return self._request("v1/stock/list")

    def fundamentals(self, symbol: str) -> Any:
        return self._request("v1/stock/fundamentals", {"symbol": symbol})

    def industry_classification(self) -> Any:
        return self._request("v1/stock/industry")

    def index_constituents(self, index_code: str) -> Any:
        return self._request("v1/index/constituents", {"index_code": index_code})

    def news(self, symbol: str = "", limit: int = 20) -> Any:
        return self._request("v1/news", {"symbol": symbol, "limit": limit})


class NeoDataSource:
    """与 ``DataFetcher`` 接口对齐的稳定数据源。

    每个方法先用 NeoData 取数；未配置 / 失败 / 未覆盖时按 ``fallback_to_legacy``
    回退到 legacy ``DataFetcher``，并通过 ``last_fetch_info`` 暴露实际使用的源，
    便于上层区分「代码错误」与「数据源不可用」。
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        tushare_token: Optional[str] = None,
        fallback_to_legacy: bool = True,
    ) -> None:
        cfg = config or {}
        neo_cfg = cfg.get("neodata", {}) or {}
        self.client = NeoDataClient(
            base_url=neo_cfg.get("base_url", ""),
            token_env=neo_cfg.get("token_env", "NEODATA_TOKEN"),
        )
        self.fallback_to_legacy = neo_cfg.get("fallback_to_legacy", fallback_to_legacy)
        self.last_fetch_info: Dict[str, Any] = {"source": None, "message": ""}
        self._legacy: Optional["DataFetcher"] = None
        self._legacy_token = tushare_token

    # -- 内部工具 --
    def _legacy_fetcher(self) -> "DataFetcher":
        if self._legacy is None:
            if DataFetcher is None:
                raise RuntimeError("legacy DataFetcher 不可用，且 NeoData 未配置")
            self._legacy = DataFetcher(tushare_token=self._legacy_token)
        return self._legacy

    def _resolve(self, neodata_fn, legacy_fn, label: str):
        """优先 NeoData；失败回退 legacy；都不行返回空并标记。"""
        if self.client.configured:
            try:
                result = neodata_fn()
                if result is not None and not (isinstance(result, pd.DataFrame) and result.empty):
                    self.last_fetch_info = {"source": "neodata", "message": f"{label} 经 NeoData 取数成功"}
                    return result
            except Exception as e:  # noqa: BLE001
                self.last_fetch_info = {"source": "neodata", "message": f"{label} NeoData 失败: {e}"}
        if self.fallback_to_legacy:
            out = legacy_fn()
            self.last_fetch_info = {
                "source": "legacy",
                "message": f"{label} 回退 legacy DataFetcher（NeoData 不可用或未覆盖）",
            }
            return out
        return None

    # -- 字段映射：NeoData 原始 JSON -> FactorGPT 约定 DataFrame --
    @staticmethod
    def _map_kline(payload: Any, symbol: str) -> pd.DataFrame:
        rows = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # 兼容常见字段名 -> 统一约定：date/open/high/low/close/volume/amount/pct_chg/symbol
        rename = {
            "trade_date": "date", "datetime": "date", "vol": "volume",
            "circ_mv": "amount", "change_pct": "pct_chg", "pre_close": "pre_close",
        }
        df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)
        df["symbol"] = symbol
        return df

    # -- 公开方法（与 DataFetcher 对齐） --
    def get_daily_kline(
        self,
        symbols: List[str],
        start: str,
        end: str,
        period: str = "daily",
        adjust: str = "qfq",
        force_synthetic: bool = False,
    ) -> pd.DataFrame:
        if isinstance(symbols, str):
            symbols = [symbols]
        symbols = [str(s).strip() for s in symbols if str(s).strip()]
        if not symbols:
            self.last_fetch_info = {"source": "none", "message": "未提供股票代码"}
            return pd.DataFrame()

        def neo():
            frames = [self._map_kline(self.client.kline(s, start, end, period, adjust), s) for s in symbols]
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        def leg():
            return self._legacy_fetcher().get_daily_kline(symbols, start, end, period, adjust, force_synthetic)

        return self._resolve(neo, leg, "日K线")

    def get_financial_data(self, symbol: str, report_type: str = "年报") -> pd.DataFrame:
        def neo():
            payload = self.client.fundamentals(symbol)
            df = pd.DataFrame(payload.get("data", []) if isinstance(payload, dict) else payload)
            if not df.empty:
                df.columns = [c.lower().strip() for c in df.columns]
                df["symbol"] = symbol
            return df

        def leg():
            return self._legacy_fetcher().get_financial_data(symbol, report_type)

        return self._resolve(neo, leg, "财务数据")

    def get_industry_classification(self) -> pd.DataFrame:
        def neo():
            payload = self.client.industry_classification()
            return pd.DataFrame(payload.get("data", []) if isinstance(payload, dict) else payload)

        def leg():
            return self._legacy_fetcher().get_industry_classification()

        return self._resolve(neo, leg, "行业分类")

    def get_industry_and_cap(self, symbols):
        def neo():
            # NeoData 一次性行业+市值：若端点返回，则直接构造；否则回退 legacy
            raise NotImplementedError("NeoData 行业+市值聚合端点待按 SKILL.md 接入")

        def leg():
            return self._legacy_fetcher().get_industry_and_cap(symbols)

        return self._resolve(neo, leg, "行业与市值")

    def get_index_constituents(self, index_code: str = "000906") -> List[str]:
        def neo():
            payload = self.client.index_constituents(index_code)
            data = payload.get("data", payload) if isinstance(payload, dict) else payload
            return [str(x) for x in data] if data else []

        def leg():
            return self._legacy_fetcher().get_index_constituents(index_code)

        out = self._resolve(neo, leg, "指数成分股")
        return out if isinstance(out, list) else []

    def get_news_sentiment(self, symbol: str = "", limit: int = 20):
        def neo():
            payload = self.client.news(symbol, limit)
            return pd.DataFrame(payload.get("data", []) if isinstance(payload, dict) else payload)

        def leg():
            return self._legacy_fetcher().get_news_sentiment(symbol, limit)

        return self._resolve(neo, leg, "新闻情绪")

    def get_market_snapshot(self, *args, **kwargs):
        def leg():
            return self._legacy_fetcher().get_market_snapshot(*args, **kwargs)

        return self._resolve(lambda: None, leg, "市场快照")

    def get_minute_kline(self, *args, **kwargs):
        def leg():
            return self._legacy_fetcher().get_minute_kline(*args, **kwargs)

        return self._resolve(lambda: None, leg, "分钟K线")

    def get_intraday_kline(self, *args, **kwargs):
        def leg():
            return self._legacy_fetcher().get_intraday_kline(*args, **kwargs)

        return self._resolve(lambda: None, leg, "分时K线")


class DataSourceFactory:
    """数据源工厂：按 config ``data.source`` 决定使用 legacy 还是 NeoData 稳定源。"""

    @staticmethod
    def get_data_source(config: Optional[dict] = None, tushare_token: Optional[str] = None):
        cfg = config or {}
        data_cfg = cfg.get("data", {}) or {}
        source = (data_cfg.get("source") or "legacy").lower()
        if source == "neodata":
            return NeoDataSource(config=cfg, tushare_token=tushare_token)
        if DataFetcher is None:
            raise RuntimeError("legacy DataFetcher 不可用")
        return DataFetcher(tushare_token=tushare_token)


# 便捷函数：直接替换代码中的 ``DataFetcher()``
def get_data_source(config: Optional[dict] = None, tushare_token: Optional[str] = None):
    return DataSourceFactory.get_data_source(config, tushare_token)


if __name__ == "__main__":
    # 自测：验证适配器可实例化并能回退到 legacy 取数
    ds = get_data_source({"data": {"source": "neodata", "neodata": {"base_url": ""}}})
    print("数据源类型:", type(ds).__name__)
    snap = ds.get_daily_kline(["600519"], "2024-01-01", "2024-01-10")
    print("取数源:", ds.last_fetch_info.get("source"), "| 行数:", len(snap))
