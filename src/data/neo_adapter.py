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

⚠️ 重要现实约束（2026-09-13 实测复核，详见 config.yaml 的 data.neodata 注释）：
   真实 NeoData 是「自然语言查询」单端点服务，请求体为
   ``{"query","channel":"neodata","sub_channel":"workbuddy"}``，成功响应里
   ``data.apiData.apiRecall[].content`` 是 markdown 文本块。
   **实测网关可用**（HTTP 200 / suc=true），行情类 content 就是完整的 OHLCV 表格，
   本适配器现已内置 markdown 表格解析（``_parse_md_tables`` + 各 ``_map_*``）。

   但服务端单次响应有**篇幅上限**（实测 ~1.5–1.7KB，表格恒定 6–11 行），由此决定能力边界：

   ==================  ==========================================================
   能力               实测结论
   ==================  ==========================================================
   日 K 线（≤5 交易日） 完整返回（5 日窗口 1154 字符、无省略）→ 可分块拼完整时序
   日 K 线（>5 交易日） 表格中间插「省略中间历史行情…禁止推断补全」占位行，行数不随
                       区间增长（8/6/6/8 行）→ 判定不可用，回退 legacy
   指数成分股          只返回权重最高的前 ~10 只（中证800 实测 11 行）→ 残缺，回退 legacy
   全市场行业映射       无此能力（问「申万一级行业列表」返回某个指数的行业分布且数值全
                       ``--``；问「行业板块列表」只返回涨幅前几名排行）→ 回退 legacy
   新闻舆情             查询返回空（apiRecall 0 块、docData 为空）→ 回退 legacy
   单股财务             可用（利润表 / 综合财务指标表格，含最新报告期）
   ==================  ==========================================================

   因此 ``data.neodata.fallback_to_legacy`` 仍**保持 true**：NeoData 可独立支撑
   「少量标的 + 短窗口行情」与「单股财务」，但批量/长时序/成分股/新闻必须由 legacy 提供。
   残缺数据一律返回 None 触发回退而非返回——把不完整时序或残缺票池喂给回测会**静默失真**，
   比直接报错更危险；确有研究性需要时可用 ``data.neodata.allow_partial: true`` 显式放开。

   鉴权：网关接受 IDE 会话令牌（``connect_cloud_service`` 现签发即可用），
   有效期约 12 小时，过期后返回 HTTP 401。

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
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

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
        成功响应：data.apiData.apiRecall[].content 为 markdown 文本块，其中行情/财务类块
                 内含标准 markdown 表格（见 ``_parse_md_tables``）。

    本客户端只负责「正确发请求 + 取回原始结果」；结构化解析在 ``NeoDataSource._map_*`` 中完成。
    注意响应有篇幅上限（实测 ~1.5–1.7KB）：长区间行情会被服务端省略中段并插入占位行，
    故完整批量时序、指数成分股、行业映射仍须由 legacy 提供（见模块顶部能力边界表）。
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
                    "NeoData 鉴权失败 HTTP %s：令牌缺失或已过期。网关接受当前 IDE 会话令牌"
                    "（connect_cloud_service 现签发即可用），但有效期约 12 小时；"
                    "缓存文件 ~/.workbuddy/.neodata_token 过期后需重新签发。" % e.code
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


# ---------------------------------------------------------------------------
# markdown 表格解析（NeoData 的 apiRecall[].content 是 markdown 文本块）
# ---------------------------------------------------------------------------

# 服务端单次响应有篇幅上限，长区间会在表格中间插入一行
# 「2024-01-02 ~ 2024-01-22 | 省略中间历史行情，不包含任何具体数值，禁止根据该标记推断、
#   补全或参与计算 | --」
# 解析器必须识别并**丢弃**该行（绝不插值），并置 truncated 标记供调用方判定数据不完整。
_OMIT_RE = re.compile(r"省略|禁止[^|]{0,8}推断")

# 空值/占位空值的常见写法（`--` 是 NeoData 的缺失值标记）
_NA_TOKENS = {"", "--", "-", "—", "n/a", "nan", "none", "null", "无"}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class _MdTable:
    """一段 markdown 表格。``truncated`` 为真表示含「省略…」占位行，数据不完整。"""

    columns: List[str]
    rows: List[List[str]]
    truncated: bool = False

    def to_frame(self) -> pd.DataFrame:
        """转 DataFrame（行宽与表头对齐，不足补空串）。"""
        width = len(self.columns)
        rows = [(r + [""] * width)[:width] for r in self.rows]
        return pd.DataFrame(rows, columns=self.columns)

    def column(self, *keywords: str) -> Optional[str]:
        """按关键词模糊匹配列名（同一问法下表头可能略有差异，如「收盘」/「收盘/最新」）。"""
        for col in self.columns:
            if any(k in col for k in keywords):
                return col
        return None


def _is_sep_row(cells: List[str]) -> bool:
    """判断 markdown 分隔行（``| :--- | ---: |``）。"""
    return bool(cells) and all((not c) or set(c) <= set(":- ") for c in cells)


def _parse_md_tables(text: Any) -> List[_MdTable]:
    """把一段文本中的所有 markdown 表格解析为 ``_MdTable`` 列表。

    规则：以 ``|`` 开头且上一行非表格的行为表头，其后紧跟的分隔行跳过，其余为数据行；
    「省略…」占位行被丢弃并置 ``truncated``。解析失败返回空列表（由调用方判定回退）。
    """
    tables: List[_MdTable] = []
    cols: Optional[List[str]] = None
    rows: List[List[str]] = []
    truncated = False

    def flush() -> None:
        nonlocal cols, rows, truncated
        if cols and rows:
            tables.append(_MdTable(cols, rows, truncated))
        cols, rows, truncated = None, [], False

    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            flush()
            continue
        if _OMIT_RE.search(line):
            truncated = True
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells or _is_sep_row(cells):
            continue
        if cols is None:
            cols = cells
            continue
        rows.append(cells)
    flush()
    return tables


def _num(value: Any) -> float:
    """单元格转数值：``'1,234.00'`` / ``'-0.55%'`` / ``'--'`` -> float 或 NaN。"""
    if value is None:
        return float("nan")
    s = str(value).strip().replace(",", "").replace("%", "")
    if s.startswith("+"):
        s = s[1:]
    if s.lower() in _NA_TOKENS:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


class NeoDataSource:
    """与 ``DataFetcher`` 接口对齐的数据源（见模块顶部「重要现实约束」）。

    ⚠️ 真实 NeoData 是「自然语言查询」服务，返回 markdown 表格文本，且**单次响应有篇幅
    上限**（实测 ~1.5–1.7KB）。本类已实现表格解析，能力边界（同模块顶部约束表）：

    - 可独立提供：单股/≤5 个交易日窗口的日 K（分块可拼完整）、单股财务（最新报告期）；
    - 必须回退 legacy：长区间日 K、指数成分股（服务端只给前 ~10 只）、全市场行业映射、
      新闻舆情、行业+市值批量。残缺数据一律返回 None 而非硬凑，避免回测静默失真。

    因此 ``data.neodata.fallback_to_legacy`` 当前**保持 true**。

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
        # 配置层级：优先 data.neodata（config.yaml 实际层级，与 DataSourceFactory 的
        # data.source 同级），回退顶层 neodata（旧写法兼容）。
        # 历史缺陷：只读 cfg["neodata"] 会漏掉 config.yaml 的 data.neodata，
        # 导致 base_url 恒为空、configured 恒为 False，data.source=neodata 静默退化为 legacy。
        neo_cfg = (cfg.get("data") or {}).get("neodata") or cfg.get("neodata") or {}
        self.client = NeoDataClient(
            base_url=neo_cfg.get("base_url", ""),
            token_env=neo_cfg.get("token_env", "NEODATA_TOKEN"),
        )
        self.fallback_to_legacy = neo_cfg.get("fallback_to_legacy", fallback_to_legacy)
        # allow_partial：是否接受「服务端截断」的残缺结果（默认 false，见模块顶部约束）。
        # 仅在研究性查看时开启；回测链路务必保持 false，否则票池/时序会静默缩水。
        self.allow_partial = bool(neo_cfg.get("allow_partial", False))
        # max_chunk_requests：短窗口分块抓取的总请求预算（1 请求 ≈ 5 个交易日 × 1 只标的）。
        # 超出预算即判定 NeoData 不适用并直接回退 legacy，避免对数年区间打出海量请求。
        try:
            self.max_chunk_requests = max(1, int(neo_cfg.get("max_chunk_requests", 8)))
        except (TypeError, ValueError):
            self.max_chunk_requests = 8
        self.last_fetch_info: Dict[str, Any] = {"source": None, "message": ""}
        # 各 neo() 分支放弃 NeoData 时可写入具体原因（如「区间超预算」），
        # 供 _resolve 生成更准确的回退说明，避免一律报成「解析为空」。
        self._skip_reason: Optional[str] = None
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
        reason = "未配置（缺网关地址或令牌）"
        self._skip_reason = None
        if self.client.configured:
            try:
                result = neodata_fn()
                if _is_usable(result):
                    self.last_fetch_info = {"source": "neodata", "message": f"{label} 经 NeoData 取数成功"}
                    return result
                # 空结果视为未覆盖，继续回退；若分支给出了更具体的原因则优先采用
                reason = self._skip_reason or "已连通但返回为空/未覆盖/解析为空"
            except Exception as e:  # noqa: BLE001
                reason = f"调用失败：{e}"
        if self.fallback_to_legacy:
            out = legacy_fn()
            # 保留 NeoData 侧的具体原因（未配置 / 401 令牌过期 / 解析为空），
            # 否则 UI 无法区分「未配置」与「令牌过期」两类问题。
            self.last_fetch_info = {
                "source": "legacy",
                "message": f"{label} 回退 legacy DataFetcher（NeoData {reason}）",
            }
            return out
        self.last_fetch_info = {
            "source": "none",
            "message": f"{label} NeoData 不可用（{reason}），且 fallback_to_legacy=false，返回空",
        }
        return pd.DataFrame()

    # -- 字段映射：NeoData markdown 表格 -> FactorGPT 约定 DataFrame --
    # 统一原则：只返回**完整**语义的数据；服务端截断（省略占位行 / 明显偏少的行数）
    # 一律返回 None 触发 legacy 回退，绝不用残缺数据静默污染回测。

    # 列名 -> 标准 K 线列名（与 DataFetcher._normalize_kline 的约定对齐）
    _KLINE_RENAME = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "pct_chg",
        "涨跌额": "change",
        "振幅": "amplitude",
        "换手率": "turnover",
        "前收": "pre_close",
    }
    _KLINE_NUMERIC = ("open", "high", "low", "close", "volume", "amount",
                      "pct_chg", "change", "amplitude", "turnover", "pre_close")

    # 少于该数量的成分股结果必为服务端截断样本（A 股宽基指数成分股最少也有 50 只）
    _MIN_PLAUSIBLE_CONSTITUENTS = 50

    def _map_kline(self, result: Any, symbol: str) -> Optional[pd.DataFrame]:
        """解析「统一行情查询」表格 -> 标准 K 线 DataFrame（date/open/…/symbol）。

        实测（2026-09-13）：单日与 ≤5 个交易日窗口**完整**返回；区间更长时表格中间会插
        「省略中间历史行情…禁止推断补全」占位行，且行数恒定不随区间增长。含该标记即说明
        本区间数据不完整 → 返回 None（回退 legacy），**绝不插值补全**；``allow_partial``
        为 true 时才接受残缺结果（研究性查看用）。
        """
        for text in _extract_contents(result):
            for tb in _parse_md_tables(text):
                if tb.column("日期") is None:
                    continue
                close_col = tb.column("收盘")
                if close_col is None:
                    continue
                if tb.truncated and not self.allow_partial:
                    return None
                df = tb.to_frame().rename(columns={**self._KLINE_RENAME, close_col: "close"})
                if "date" not in df.columns:
                    continue
                df["date"] = df["date"].astype(str).str.strip()
                df = df[df["date"].map(lambda d: bool(_DATE_RE.match(d)))]
                if df.empty:
                    continue
                for c in self._KLINE_NUMERIC:
                    if c in df.columns:
                        df[c] = df[c].map(_num)
                df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
                if df.empty:
                    continue
                # 复用 legacy 的归一化，保证列名/类型/缺失列补齐完全一致
                if DataFetcher is not None and hasattr(DataFetcher, "_normalize_kline"):
                    return DataFetcher._normalize_kline(df, symbol)
                df["symbol"] = symbol
                return df
        return None

    def _map_financials(self, result: Any, symbol: str) -> Optional[pd.DataFrame]:
        """解析财务块 -> 单行 DataFrame（best-effort）。

        两种形态：① 文本型「键: 值」块（``basic_info``）；② 表格型财报块
        （「利润表」/「综合财务指标」，服务端按报告期倒序，取首行即最新报告期）。
        """
        rows: Dict[str, str] = {}
        for text in _extract_contents(result, types=["basic_info"]):
            for line in str(text).replace("；", ";").splitlines():
                if ":" in line or "：" in line:
                    k, _, v = line.replace("：", ":").partition(":")
                    k = k.strip().strip("【】")
                    if k and v.strip():
                        rows[k] = v.strip()
        if not rows:
            for text in _extract_contents(result):
                for tb in _parse_md_tables(text):
                    if tb.column("报告期") is None:
                        continue
                    frame = tb.to_frame()
                    if frame.empty:
                        continue
                    latest = frame.iloc[0]
                    for col in frame.columns:
                        val = str(latest[col]).strip()
                        if val and val.lower() not in _NA_TOKENS:
                            rows[str(col)] = val
                    if rows:
                        break
                if rows:
                    break
        if not rows:
            return None
        df = pd.DataFrame([rows])
        df["symbol"] = symbol
        return df

    def _map_industry(self, result: Any) -> Optional[pd.DataFrame]:
        """固定返回 None：NeoData 无「全市场行业映射 / 完整板块清单」能力。

        实测：问「A股 申万一级行业分类 列表」返回的是某个指数的「指数行业分布」（数值全为
        ``--``）；问「行业板块列表」只返回当日涨幅前几名的「板块涨跌排行」。二者都不是完整
        板块清单，返回残缺列表会让上层误以为全市场只有几个板块，故一律回退 legacy/offline。
        """
        return None

    def _map_index_constituents(self, result: Any) -> Optional[List[str]]:
        """解析「指数成分及权重」表格的成份证券代码列 -> 6 位裸代码列表。

        旧实现用 ``re.findall(r"\\b\\d{6}\\b", 全文)`` 抓数字，会把**指数自身代码**
        （如 000906）和权重/日期数字混入成分股，产生脏票池；现改为按表格列解析并剥离
        ``.SH``/``.SZ`` 后缀。

        ⚠️ 服务端单次只返回权重最高的前 ~10 只（实测中证800 仅 11 行 / 1.6KB），并非完整
        成分股。残缺票池会静默缩小回测股票池，故默认返回 None 触发 legacy 回退；
        仅当 ``data.neodata.allow_partial: true`` 时接受。
        """
        for text in _extract_contents(result):
            for tb in _parse_md_tables(text):
                col = tb.column("成份证券代码", "成分证券代码")
                if col is None:
                    continue
                codes: List[str] = []
                for value in tb.to_frame()[col].astype(str):
                    m = re.match(r"^\s*(\d{6})(?:\.\w+)?\s*$", value)
                    if m:
                        codes.append(m.group(1))
                codes = list(dict.fromkeys(codes))
                if not codes:
                    continue
                if len(codes) < self._MIN_PLAUSIBLE_CONSTITUENTS and not self.allow_partial:
                    return None
                return codes
        return None

    # ⚠️ 实测（2026-09-13）：新闻类查询返回空（apiRecall 0 块、docData 为空），
    # 故本解析实际不会命中，新闻仍由 legacy 提供；保留解析以便服务端后续开放该能力。
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
            if period != "daily":
                self._skip_reason = f"NeoData 仅提供日线，period={period} 不适用"
                return None        # NeoData 只有日线表格，分钟/周/月线不适用
            windows = self._chunk_windows(start, end, len(symbols))
            if windows is None:
                # 区间过长或请求量超预算：判定不适用，回退 legacy（服务端单次有篇幅上限）
                self._skip_reason = (
                    f"区间 {start}~{end} × {len(symbols)} 只超出分块请求预算"
                    f"（max_chunk_requests={self.max_chunk_requests}），NeoData 仅支持短窗口"
                )
                return None
            frames = []
            for s in symbols:      # 按标的分组抓取，保持与 legacy 一致的行顺序
                for w_start, w_end in windows:
                    df = self._map_kline(self.client.kline(s, w_start, w_end, period, adjust), s)
                    if df is None:
                        return None    # 任一段拿不到完整数据即整体放弃，不拼残缺时序
                    frames.append(df)
            if not frames:
                return None
            out = pd.concat(frames, ignore_index=True)
            # 分块边界可能带回区间外的交易日（实测会向前多给 1~2 根），按请求区间裁剪
            if "date" in out.columns:
                out = out[(out["date"] >= str(start)) & (out["date"] <= str(end))]
            out = out.drop_duplicates(subset=["symbol", "date"]).reset_index(drop=True)
            return out if not out.empty else None

        def leg():
            return self._legacy_fetcher().get_daily_kline(symbols, start, end, period, adjust, force_synthetic)

        return self._resolve(neo, leg, "日K线")

    # NeoData 单次响应篇幅上限：≤5 个交易日可完整返回（实测 5 日窗口 1154 字符无省略），
    # 10 个交易日即出现「省略…」占位行，故分块粒度取 5 个交易日。
    _CHUNK_TRADING_DAYS = 5

    def _chunk_windows(self, start: str, end: str, n_symbols: int = 1) -> Optional[List[Tuple[str, str]]]:
        """把 ``[start, end]`` 切成每段约 5 个交易日的窗口，供分块拼完整时序。

        预估总请求数（窗口数 × 标的数）超出 ``max_chunk_requests`` 预算时返回 None，
        表示 NeoData 不适用，调用方应直接回退 legacy——否则对「数年 × 数百只」的回测
        区间会打出成千上万次请求。实测：单标的 2 个月 ≈ 8~9 次请求，尚可接受；
        5 年区间 ≈ 250 次/标的，必须回退。
        """
        try:
            s, e = pd.Timestamp(start), pd.Timestamp(end)
        except Exception:  # noqa: BLE001
            return None
        if pd.isna(s) or pd.isna(e) or e < s:
            return None
        cal_days = (e - s).days + 1
        # 自然日 -> 交易日近似（一周约 5 个交易日），宁可高估窗口数以保守控制预算
        est_trading_days = max(1, int(cal_days * 5 / 7) + 1)
        n_chunks = -(-est_trading_days // self._CHUNK_TRADING_DAYS)      # 向上取整
        if n_chunks * max(1, n_symbols) > self.max_chunk_requests:
            return None
        span = max(1, cal_days // n_chunks)
        windows: List[Tuple[str, str]] = []
        cur = s
        # 用标准库 timedelta：numpy 2.x + pandas 2.x 下 pd.Timedelta(days=int) 会触发
        # DeprecationWarning（generic unit），在 filterwarnings=error 的测试环境会直接失败。
        while cur <= e:
            nxt = min(cur + timedelta(days=span - 1), e)
            windows.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
            cur = nxt + timedelta(days=1)
        return windows or None

    def get_financial_data(self, symbol: str, report_type: str = "年报") -> pd.DataFrame:
        def neo():
            return self._map_financials(self.client.fundamentals(symbol), symbol)

        def leg():
            return self._legacy_fetcher().get_financial_data(symbol, report_type)

        return self._resolve(neo, leg, "财务数据")

    def get_industry_classification(self) -> pd.DataFrame:
        def neo():
            self._skip_reason = "无全市场行业/板块清单能力（只返回涨幅前几名排行）"
            return self._map_industry(self.client.industry_classification())

        def leg():
            return self._legacy_fetcher().get_industry_classification()

        return self._resolve(neo, leg, "行业分类")

    def get_industry_and_cap(self, symbols):
        def neo():
            # NeoData 只能逐股查行业、无批量行业+市值端点，票池规模下请求数不可接受
            self._skip_reason = "仅支持逐股行业查询，无批量行业+市值端点"
            return None

        def leg():
            return self._legacy_fetcher().get_industry_and_cap(symbols)

        return self._resolve(neo, leg, "行业与市值")

    def get_index_constituents(self, index_code: str = "000906") -> List[str]:
        def neo():
            codes = self._map_index_constituents(self.client.index_constituents(index_code))
            if codes is None:
                self._skip_reason = (
                    "服务端单次只返回权重前 ~10 只成分股（篇幅上限），残缺票池不可用于回测"
                )
            return codes

        def leg():
            return self._legacy_fetcher().get_index_constituents(index_code)

        out = self._resolve(neo, leg, "指数成分股")
        return out if isinstance(out, list) else []

    def get_news_sentiment(self, symbol: str = "", limit: int = 20):
        def neo():
            # 实测：新闻类查询在 NeoData 侧返回空（apiRecall 0 块），新闻仍由 legacy 提供
            self._skip_reason = "新闻查询返回空，该能力仍由 legacy 提供"
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
    """数据源工厂：按 config ``data.source`` 决定使用 legacy / neodata / offline 数据源。

    - ``config=None`` 时自动读取项目全局 ``config.yaml``，保证 ``data.source`` 开关处处生效；
    - ``data.source`` 缺省为 ``legacy``，即保留原有的本地运行数据源（akshare/sina/tushare 自爬）；
    - ``data.source: neodata`` 时走平台稳定数据源（未配置时仍安全回退 legacy）；
    - ``data.source: offline`` 时使用 ``OfflineDataSource``（仓库内置的本地 parquet，
      完全离线、不触网；数据文件随仓库分发在 ``data/offline/``，克隆即用）；
    - ``data.source: hf`` 时使用 ``HFDataSource``（离线的**期货 L2 五档快照**，
      500ms 快照级数据；除日 K 聚合外还提供 load_l2/get_term_structure 等高频接口，
      见 ``config.yaml`` 的 ``data.hf`` 段与 ``docs/高频数据接入与因子挖掘.md``）。
    """

    @staticmethod
    def get_data_source(config: Optional[dict] = None, tushare_token: Optional[str] = None):
        cfg = config if isinstance(config, dict) else _project_config()
        data_cfg = cfg.get("data", {}) or {}
        source = (data_cfg.get("source") or "legacy").lower()
        if source == "offline":
            from data.offline_adapter import OfflineDataSource

            return OfflineDataSource(config=cfg)
        if source == "hf":
            from data.hf_adapter import HFDataSource

            return HFDataSource(config=cfg)
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
