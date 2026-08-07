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
2. 安全回退：NeoData 未配置、网络失败、鉴权失败或字段无法解析时，按配置回退到
   legacy ``DataFetcher``，保证切换过程不破坏现有功能。
3. 零新依赖：仅用标准库 ``urllib``，不改动 ``requirements.txt``。

⚠️ 重要现实约束（已对照平台 neodata-financial-search 技能 SKILL.md / reference.md 核实）：
   真实 NeoData 是「自然语言查询」单端点服务，请求体为
   ``{"query","channel":"neodata","sub_channel":"workbuddy"}``，成功响应里
   ``data.apiData.apiRecall[].content`` 是**自由文本块**（行情/财务/资金流描述），
   **并非结构化批量行情/财务 REST 接口**。因此它无法稳定提供因子引擎所需的：
   完整日 K 线时序（回测核心）、完整指数成分股列表、行业/市值映射、结构化财务报表。
   所以各 ``neo()`` 解析在多数场景下返回空，必须由 ``fallback_to_legacy`` 回退 legacy
   才能真正跑通回测——``data.neodata.fallback_to_legacy`` 当前**严禁设为 false**。
   本适配器目前仅把 NeoData 作为「研究问答」辅助接入，不替代 legacy 执行因子回测。

启用方式
--------
在 ``config.yaml`` 中设置 ``data.source: neodata``，并把调用点 ``DataFetcher()``
替换为 ``get_data_source(config)``（见文件底部 ``DataSourceFactory``）。
或在 Skill 脚本中直接 ``from data.neo_adapter import get_data_source``。

注意
----
NeoData 网关地址已从平台 ``neodata-financial-search`` 技能 SKILL.md 填入
``config.yaml`` 的 ``data.neodata.base_url``（真实端点
``https://copilot.tencent.com/agenttool/v1/neodata``）。字段映射在 ``_map_*`` 方法中
集中维护，且均为 best-effort：解析为空即回退 legacy。
"""

from __future__ import annotations

import json
import os
import re
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
    """NeoData 网关客户端（标准库实现，无第三方依赖）。

    真实服务为「自然语言查询」单端点（详见平台 neodata-financial-search 技能 SKILL.md / reference.md）：

        POST {base_url}
        body = {"query": <自然语言>, "channel": "neodata", "sub_channel": "workbuddy", "data_type": "api"}
        成功响应：data.apiData.apiRecall[].content 为自由文本块（行情/财务/资金流等描述），
                 并非可直接下载的结构化批量行情/财务数组。

    因此本客户端只负责「正确发请求 + 取回原始结果」；结构化解析在 NeoDataSource._map_* 中
    尽力而为，解析为空时由 ``fallback_to_legacy`` 回退 legacy——因子引擎所需的完整 K 线时序、
    指数成分股、行业/市值映射等结构化批量数据必须来自 legacy（NeoData 文本无法稳定提供）。
    """

    CHANNEL = "neodata"
    SUB_CHANNEL = "workbuddy"

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

    def _nl_query(self, query: str, data_type: str = "api") -> dict:
        """向真实 NeoData 端点发起自然语言查询，返回完整响应 JSON。

        鉴权失败（401/403）或业务错误（非 200）时抛 RuntimeError，由上层捕获并回退 legacy。
        """
        if not self.configured:
            raise RuntimeError("NeoData 未配置：请在 config 设置 data.neodata.base_url 并确保 token 可用")
        payload = {
            "query": query,
            "channel": self.CHANNEL,
            "sub_channel": self.SUB_CHANNEL,
        }
        if data_type and data_type != "all":
            payload["data_type"] = data_type
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token or ''}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # 鉴权失败
            if e.code in (401, 403):
                raise RuntimeError(
                    "NeoData 鉴权失败 HTTP %s：需平台专属 tempToken，非 IDE 会话 token" % e.code
                ) from e
            raise
        code = str(result.get("code", ""))
        if code not in ("200", "") or not result.get("suc", True):
            raise RuntimeError(f"NeoData 业务错误 code={code} msg={result.get('msg')}")
        return result

    # ---- 各取数场景 -> 自然语言查询（best-effort，供 _map_* 解析原始文本） ----
    def kline(self, symbol: str, start: str, end: str, period: str = "daily", adjust: str = "qfq") -> dict:
        return self._nl_query(f"{symbol} {start} 至 {end} 的每日开盘价 收盘价 最高价 最低价 成交量 涨跌幅（{adjust}）")

    def stock_list(self) -> dict:
        return self._nl_query("A股 全部股票代码与股票名称 列表")

    def fundamentals(self, symbol: str) -> dict:
        return self._nl_query(f"{symbol} 最新年报 营业收入 净利润 资产负债率 净资产收益率 毛利率")

    def industry_classification(self) -> dict:
        return self._nl_query("A股 申万一级行业分类 股票代码与行业名称 列表")

    def index_constituents(self, index_code: str) -> dict:
        return self._nl_query(f"{index_code} 指数 完整成分股 股票代码 列表")

    def news(self, symbol: str = "", limit: int = 20) -> dict:
        q = f"{symbol} 最近新闻与舆情" if symbol else "今日 市场 重大新闻"
        return self._nl_query(q)


def _extract_contents(result: Any, types: Optional[List[str]] = None) -> List[str]:
    """从 NeoData 响应中提取 apiRecall 的文本 content 列表（可过滤 type）。"""
    if not isinstance(result, dict):
        return []
    api = (result.get("data") or {}).get("apiData") or {}
    blocks = api.get("apiRecall") or []
    out: List[str] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if types and b.get("type") not in types:
            continue
        c = b.get("content")
        if c:
            out.append(str(c))
    return out


def _is_usable(result: Any) -> bool:
    """NeoData 结果是否可作为有效数据（非空）。空结果应触发 legacy 回退。"""
    if result is None:
        return False
    if isinstance(result, pd.DataFrame):
        return not result.empty
    if isinstance(result, (list, tuple)):
        return len(result) > 0
    return True


class NeoDataSource:
    """与 ``DataFetcher`` 接口对齐的数据源（见模块顶部「重要现实约束」）。

    ⚠️ 真实 NeoData 是「自然语言查询」服务，返回自由文本块，**无法稳定提供因子引擎所需的
    结构化批量数据**（完整 K 线时序、完整指数成分股、行业/市值映射、结构化财报）。因此各
    ``neo()`` 解析在多数场景下返回空，由 ``fallback_to_legacy`` 回退 legacy 才能真正跑通回测；
    ``data.neodata.fallback_to_legacy`` 当前必须保持 ``true``。

    每个方法先用 NeoData 取数；未配置 / 失败 / 鉴权失败 / 未覆盖 / 解析为空时按
    ``fallback_to_legacy`` 回退到 legacy ``DataFetcher``，并通过 ``last_fetch_info``
    暴露实际使用的源，便于上层区分「代码错误」与「数据源不可用」。
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
        """优先 NeoData；失败 / 空结果回退 legacy；都不行返回空并标记。"""
        if self.client.configured:
            try:
                result = neodata_fn()
                if _is_usable(result):
                    self.last_fetch_info = {"source": "neodata", "message": f"{label} 经 NeoData 取数成功"}
                    return result
                # 空结果视为未覆盖，继续走回退
            except Exception as e:  # noqa: BLE001
                self.last_fetch_info = {"source": "neodata", "message": f"{label} NeoData 失败: {e}"}
        if self.fallback_to_legacy:
            out = legacy_fn()
            self.last_fetch_info = {
                "source": "legacy",
                "message": f"{label} 回退 legacy DataFetcher（NeoData 不可用/未覆盖/解析为空）",
            }
            return out
        return pd.DataFrame()

    # -- 字段映射：NeoData 原始文本 -> FactorGPT 约定 DataFrame（best-effort） --
    # 真实 NeoData 返回自由文本块，无法可靠解析为结构化批量数据；无法解析时返回空（触发回退）。
    @staticmethod
    def _map_kline(result: Any, symbol: str) -> Optional[pd.DataFrame]:
        # NeoData 文本块无法稳定还原完整 OHLCV 时序，返回空（回退 legacy）。
        return None

    @staticmethod
    def _map_financials(result: Any, symbol: str) -> Optional[pd.DataFrame]:
        contents = _extract_contents(result, types=["basic_info"])
        if not contents:
            return None
        rows: Dict[str, str] = {}
        for line in contents[0].replace("；", ";").splitlines():
            if ":" in line or "：" in line:
                k, _, v = line.replace("：", ":").partition(":")
                k = k.strip().strip("【】")
                if k and v.strip():
                    rows[k] = v.strip()
        if not rows:
            return None
        df = pd.DataFrame([rows])
        df["symbol"] = symbol
        return df

    @staticmethod
    def _map_industry(result: Any) -> Optional[pd.DataFrame]:
        # 无法稳定还原全市场行业映射，返回空（回退 legacy）。
        return None

    @staticmethod
    def _map_index_constituents(result: Any) -> Optional[List[str]]:
        contents = _extract_contents(result)
        if not contents:
            return None
        codes = re.findall(r"\b\d{6}\b", " ".join(contents))
        return list(dict.fromkeys(codes)) or None

    @staticmethod
    def _map_news(result: Any) -> Optional[pd.DataFrame]:
        if not isinstance(result, dict):
            return None
        doc = (result.get("data") or {}).get("docData") or {}
        rows = []
        for grp in doc.get("docRecall") or []:
            for d in grp.get("docList") or []:
                rows.append({
                    "title": d.get("title", ""),
                    "content": d.get("content", ""),
                    "publish_time": d.get("publishTime"),
                    "source": d.get("source", ""),
                    "url": d.get("url", ""),
                })
        return pd.DataFrame(rows) if rows else None

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
            frames = []
            for s in symbols:
                df = self._map_kline(self.client.kline(s, start, end, period, adjust), s)
                if df is not None:
                    frames.append(df)
            return pd.concat(frames, ignore_index=True) if frames else None

        def leg():
            return self._legacy_fetcher().get_daily_kline(symbols, start, end, period, adjust, force_synthetic)

        return self._resolve(neo, leg, "日K线")

    def get_financial_data(self, symbol: str, report_type: str = "年报") -> pd.DataFrame:
        def neo():
            return self._map_financials(self.client.fundamentals(symbol), symbol)

        def leg():
            return self._legacy_fetcher().get_financial_data(symbol, report_type)

        return self._resolve(neo, leg, "财务数据")

    def get_industry_classification(self) -> pd.DataFrame:
        def neo():
            return self._map_industry(self.client.industry_classification())

        def leg():
            return self._legacy_fetcher().get_industry_classification()

        return self._resolve(neo, leg, "行业分类")

    def get_industry_and_cap(self, symbols):
        def neo():
            # NeoData 无稳定的「行业+市值」批量结构化端点，返回空（回退 legacy）。
            return None

        def leg():
            return self._legacy_fetcher().get_industry_and_cap(symbols)

        return self._resolve(neo, leg, "行业与市值")

    def get_index_constituents(self, index_code: str = "000906") -> List[str]:
        def neo():
            return self._map_index_constituents(self.client.index_constituents(index_code))

        def leg():
            return self._legacy_fetcher().get_index_constituents(index_code)

        out = self._resolve(neo, leg, "指数成分股")
        return out if isinstance(out, list) else []

    def get_news_sentiment(self, symbol: str = "", limit: int = 20):
        def neo():
            return self._map_news(self.client.news(symbol, limit))

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


def _project_config() -> dict:
    """读取项目全局 config.yaml。

    调用点若未显式传入全局配置（如 refinery / factor_system / market_data 的局部场景），
    工厂会回退到 ``config.yaml`` 的 ``data.source`` 开关，使全局切换在所有入口一致生效；
    默认仍为 legacy（本地自爬方案），不破坏任何现有行为。
    """
    try:
        from llm.client import load_config
        return load_config() or {}
    except Exception:  # pragma: no cover - 极端降级：直接读文件
        try:
            from data.fetcher import _load_config_file
            return _load_config_file() or {}
        except Exception:
            return {}


class DataSourceFactory:
    """数据源工厂：按 config ``data.source`` 决定使用 legacy 还是 NeoData 稳定源。

    - ``config=None`` 时自动读取项目全局 ``config.yaml``，保证 ``data.source`` 开关处处生效；
    - ``data.source`` 缺省为 ``legacy``，即保留原有的本地运行数据源（akshare/sina/tushare 自爬）；
    - 仅在显式设置 ``data.source: neodata`` 时才走平台稳定数据源（未配置时仍安全回退 legacy）。
    """

    @staticmethod
    def get_data_source(config: Optional[dict] = None, tushare_token: Optional[str] = None):
        cfg = config if isinstance(config, dict) else _project_config()
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
