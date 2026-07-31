"""
同花顺 iFinD MCP 网关数据源适配器 (THSDataFetcher)

背景
----
用户提供的同花顺 API 凭证是一段 JWE（JSON Web Encryption）令牌，头部形如：
    {"kid":"mcp-api","uid":"826525280","alg":"RSA-OAEP-256","enc":"A256GCM"}
其中 payload 由服务端 RSA 私钥加密，客户端无法解密；它仅作为「向同花顺 iFinD
MCP 网关发起鉴权」的 Bearer 令牌使用。

已实测确认的网关信息（2026-07）：
- 真实 MCP 服务域：https://api-mcp.51ifind.com:8643/ds-mcp-servers/
- 与官网控制台 https://mcp.51ifind.com 是两个不同入口（后者只是指引页）。
- 鉴权：HTTP Header `Authorization: Bearer <JWE令牌>`，已验证该令牌对下列端点有效：
    * hexin-ifind-ds-stock-mcp  (A股行情/财务/选股)  ← factor-gpt 主用
    * hexin-ifind-ds-fund-mcp   (公募基金)
    * hexin-ifind-ds-edb-mcp    (宏观经济)
    * hexin-ifind-ds-news-mcp   (公告资讯)
- 工具以自然语言 `query` 字符串驱动；行情历史数据用 `get_stock_performance`，
  返回 `data.answer` 内嵌的 Markdown 表格（中文单位 万/亿，非交易日列空值）。

本模块实现了：MCP initialize 握手 + tools/list 发现 + tools/call 取数，
并把 iFinD 返回的中文 Markdown 表格解析为标准 DataFrame（date/open/high/low/
close/volume/amount），与项目原有的 akshare DataFetcher 输出对齐。
"""

from __future__ import annotations

import json
import re
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

import pandas as pd


# ----------------------------------------------------------------------
# MCP-over-HTTP 客户端
# ----------------------------------------------------------------------
class THSDataFetcher:
    """同花顺 iFinD MCP 网关数据源。

    Args:
        token:    JWE 鉴权令牌（配置中读取，切勿打印）。
        base_url: 网关 MCP 端点，如 .../hexin-ifind-ds-stock-mcp 。
        timeout:  单次请求超时（秒）。
    """

    def __init__(
        self,
        token: str,
        base_url: str = "",
        timeout: float = 30.0,
    ) -> None:
        if not token:
            raise ValueError("THSDataFetcher 需要 token（同花顺 MCP 鉴权令牌）")
        self.token = token
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = timeout
        self._session_id: Optional[str] = None
        self._tools: List[Dict[str, Any]] = []
        self._server_info: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 底层 HTTP / JSON-RPC
    # ------------------------------------------------------------------
    def _post(self, payload: Dict[str, Any], expect_sse: bool = True) -> Any:
        if not self.base_url:
            raise ValueError("未配置 ths_api_base_url，无法连接同花顺网关")

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self.token}",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        req = urllib.request.Request(
            self.base_url, data=data, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                sid = resp.headers.get("Mcp-Session-Id") or resp.headers.get(
                    "mcp-session-id"
                )
                if sid:
                    self._session_id = sid
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(
                f"网关返回 HTTP {e.code}: {e.reason}。响应体: {body[:500]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"无法连接同花顺网关 ({self.base_url}): {e.reason}") from e

        return self._parse_response(raw, expect_sse)

    @staticmethod
    def _parse_response(raw: str, expect_sse: bool) -> Any:
        raw = (raw or "").strip()
        if not raw:
            return None
        if expect_sse and ("event:" in raw or "data:" in raw):
            last = None
            for line in raw.splitlines():
                line = line.strip()
                if line.startswith("data:"):
                    last = line[len("data:"):].strip()
            if last:
                try:
                    return json.loads(last)
                except json.JSONDecodeError:
                    return last
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    # ------------------------------------------------------------------
    # MCP 握手与工具发现
    # ------------------------------------------------------------------
    def initialize(self) -> Dict[str, Any]:
        resp = self._post(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "factor-gpt", "version": "0.1.0"},
                },
            },
            expect_sse=True,
        )
        if isinstance(resp, dict) and "result" in resp:
            self._server_info = resp["result"].get("serverInfo", {})
            try:
                self._post(
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized",
                        "params": {},
                    },
                    expect_sse=False,
                )
            except Exception:
                pass
            return self._server_info
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"initialize 失败: {resp['error']}")
        return self._server_info

    def list_tools(self) -> List[Dict[str, Any]]:
        resp = self._post(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            expect_sse=True,
        )
        tools = []
        if isinstance(resp, dict):
            if "result" in resp:
                tools = resp["result"].get("tools", []) or []
            elif "tools" in resp:
                tools = resp["tools"]
        self._tools = tools
        return tools

    def connect_and_discover(self) -> Dict[str, Any]:
        server = self.initialize()
        tools = self.list_tools()
        return {
            "server_info": server,
            "tool_count": len(tools),
            "tools": [
                {"name": t.get("name"), "description": t.get("description", "")}
                for t in tools
            ],
        }

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        resp = self._post(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
            expect_sse=True,
        )
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"tools/call[{name}] 失败: {resp['error']}")
        result = resp.get("result") if isinstance(resp, dict) else resp
        if isinstance(result, dict):
            content = result.get("content", [])
            texts = [
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            ]
            if texts:
                return "\n".join(texts)
        return result

    # ------------------------------------------------------------------
    # 工具名模糊匹配
    # ------------------------------------------------------------------
    def _find_tool(self, *keywords: str) -> Optional[str]:
        names = [t.get("name", "") for t in self._tools]
        for kw in keywords:
            for n in names:
                if kw.lower() in n.lower():
                    return n
        return None

    # ------------------------------------------------------------------
    # 对外数据接口（与 DataFetcher 对齐）
    # ------------------------------------------------------------------
    def get_universe(self, symbols: List[str]) -> List[str]:
        """直接使用配置提供的小宇宙（iFinD MCP 无指数成分股专用工具）。"""
        return list(symbols)

    def get_index_constituents(self, index_code: str = "000906") -> List[str]:
        """尽力而为：用 search_stocks 自然语言查询指数成分股。

        若解析失败返回空列表，由调用方回退到配置的 ths_symbols。
        """
        try:
            if not self._tools:
                self.list_tools()
            tool = self._find_tool("search_stocks") or "search_stocks"
            raw = self.call_tool(tool, {"query": f"{index_code} 指数成分股列表"})
            return self._extract_symbols(raw)
        except Exception as e:
            print(f"[THS] 指数成分股查询失败: {e}")
            return []

    def get_daily_kline(
        self,
        symbols: List[str],
        start: str,
        end: str,
        period: str = "daily",
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """获取日K线。使用 get_stock_performance（日频历史行情）。

        每只股票一次调用，覆盖 [start, end] 的开高低收/成交量/成交额。
        """
        try:
            if not self._tools:
                self.list_tools()
            tool = self._find_tool("performance", "kline", "hist", "quote")
            if not tool:
                print(f"[THS] 未发现行情工具，已发现: {[t.get('name') for t in self._tools]}")
                return pd.DataFrame()

            freq = "日线" if period in ("daily", "d") else "周线"
            freq = "日线"  # iFinD 历史行情默认日频；多周期由 query 指定
            out_frames: List[pd.DataFrame] = []
            for sym in symbols:
                try:
                    query = (
                        f"{sym} {start} 至 {end} {freq} "
                        f"开盘价 最高价 最低价 收盘价 成交量 成交额"
                    )
                    raw = self.call_tool(tool, {"query": query})
                    df = self._to_kline_df(raw, sym)
                    if df is not None and not df.empty:
                        out_frames.append(df)
                        print(f"[THS] {sym} 获取 {len(df)} 条交易日K线")
                except Exception as e:
                    print(f"[THS] {sym} K线获取失败: {e}")
                    continue
            if not out_frames:
                return pd.DataFrame()
            kline = pd.concat(out_frames, ignore_index=True)
            kline = kline.sort_values(["symbol", "date"]).reset_index(drop=True)
            return kline
        except Exception as e:
            print(f"[THS] 日K线获取失败: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 指数行情（基于同花顺 iFinD）
    # ------------------------------------------------------------------
    def get_index_hist(
        self, index_code: str, start: str, end: str, period: str = "daily"
    ) -> pd.DataFrame:
        """获取指数历史 K 线（同花顺 iFinD）。

        优先查找指数类 MCP 工具，回退到通用行情工具并以自然语言描述查询。
        失败时返回空 DataFrame。
        """
        try:
            if not self._tools:
                self.list_tools()
            tool = self._find_tool("index", "指数", "performance", "kline", "hist")
            if not tool:
                print(f"[THS] 未发现指数行情工具，已发现: {[t.get('name') for t in self._tools]}")
                return pd.DataFrame()
            query = (
                f"{index_code} 指数 {start} 至 {end} 日线 "
                f"开盘价 最高价 最低价 收盘价 成交量 成交额"
            )
            raw = self.call_tool(tool, {"query": query})
            df = self._to_kline_df(raw, str(index_code))
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            print(f"[THS] 指数历史获取失败({index_code}): {e}")
            return pd.DataFrame()

    def get_index_spot(self, index_code: str) -> Dict[str, Any]:
        """获取指数实时行情（同花顺 iFinD），失败返回空 dict。"""
        try:
            if not self._tools:
                self.list_tools()
            tool = self._find_tool("spot", "行情", "quote", "index")
            if not tool:
                return {}
            raw = self.call_tool(tool, {"query": f"{index_code} 指数 最新价 涨跌幅 涨跌额 成交量"})
            df = self._to_kline_df(raw, str(index_code))
            if df is not None and not df.empty:
                return df.iloc[-1].to_dict()
            answer = self._extract_answer(raw)
            return {"raw": answer[:500]} if answer else {}
        except Exception as e:
            print(f"[THS] 指数实时获取失败({index_code}): {e}")
            return {}

    def get_index_intraday(self, index_code: str) -> pd.DataFrame:
        """获取指数分时数据（同花顺 iFinD），失败返回空 DataFrame。"""
        try:
            if not self._tools:
                self.list_tools()
            tool = self._find_tool("intraday", "分时", "tick", "index")
            if not tool:
                return pd.DataFrame()
            raw = self.call_tool(tool, {"query": f"{index_code} 指数 分时 时间 价格 成交量"})
            df = self._to_kline_df(raw, str(index_code))
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            print(f"[THS] 指数分时获取失败({index_code}): {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # iFinD 返回解析
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_answer(raw: Any) -> str:
        """从 tools/call 结果中抽取 iFinD 的 Markdown 文本（data.answer）。"""
        if isinstance(raw, str):
            try:
                obj = json.loads(raw)
            except Exception:
                return raw
        else:
            obj = raw
        if isinstance(obj, dict):
            data = obj.get("data")
            if isinstance(data, dict) and "answer" in data:
                return str(data["answer"])
            if "answer" in obj:
                return str(obj["answer"])
        return str(raw) if raw is not None else ""

    @staticmethod
    def _cn_num(s: str) -> float:
        """解析中文数值：处理 万(1e4)/亿(1e8)/万亿(1e12)、逗号、百分号、空值。"""
        if s is None:
            return float("nan")
        s = str(s).strip()
        if s in ("", "\t", "-", "--", "None", "NaN", "nan"):
            return float("nan")
        mult = 1.0
        if "万亿" in s:
            mult = 1e12
            s = s.replace("万亿", "")
        elif "亿" in s:
            mult = 1e8
            s = s.replace("亿", "")
        elif "万" in s:
            mult = 1e4
            s = s.replace("万", "")
        s = s.replace(",", "").replace("%", "").replace("（", "").replace("）", "")
        try:
            return float(s) * mult
        except ValueError:
            return float("nan")

    _COL_MAP = [
        ("日期", "date"),
        ("开盘", "open"),
        ("最高", "high"),
        ("最低", "low"),
        ("收盘", "close"),
        ("成交量", "volume"),
        ("成交额", "amount"),
        ("证券代码", "symbol"),
    ]

    @classmethod
    def _to_kline_df(cls, raw: Any, symbol: str) -> Optional[pd.DataFrame]:
        answer = cls._extract_answer(raw)
        if not answer:
            return None

        # 仅取第一段 Markdown 表格：连续以 "|" 开头的行
        lines = answer.splitlines()
        table_lines: List[str] = []
        in_table = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|"):
                in_table = True
                table_lines.append(stripped)
            elif in_table:
                # 表格结束
                break
        if len(table_lines) < 3:
            return None

        # 首行表头，次行分隔，其余数据
        header_cells = [c.strip() for c in table_lines[0].strip("|").split("|")]
        # 建立 原始列名 -> 标准列名 映射
        rename = {}
        for h in header_cells:
            for key, std in cls._COL_MAP:
                if key in h and std not in rename.values():
                    rename[h] = std
                    break

        rows = []
        for row_line in table_lines[2:]:
            cells = [c.strip() for c in row_line.strip("|").split("|")]
            if len(cells) != len(header_cells):
                continue
            rows.append(dict(zip(header_cells, cells)))

        if not rows:
            return None
        df = pd.DataFrame(rows)
        df = df.rename(columns=rename)

        # 数值列转换
        for col in ("open", "high", "low", "close", "volume", "amount"):
            if col in df.columns:
                df[col] = df[col].map(cls._cn_num)
        # 日期
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime(
                "%Y-%m-%d"
            )
        df["symbol"] = symbol

        # 剔除非交易日（仅有收盘、无开/高/低）
        keep = ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]
        keep = [c for c in keep if c in df.columns]
        df = df[keep].dropna(subset=["open", "close"], how="all")
        if "open" in df.columns:
            df = df.dropna(subset=["open"])
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def _extract_symbols(raw: Any) -> List[str]:
        """从 search_stocks 的 Markdown 结果中抽取 6 位股票代码。"""
        answer = THSDataFetcher._extract_answer(raw)
        syms = re.findall(r"\b\d{6}\.[A-Z]{1,2}\b", answer)
        seen = set()
        out = []
        for s in syms:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out
