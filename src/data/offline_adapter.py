# -*- coding: utf-8 -*-
"""离线数据源适配器（OfflineDataSource）。

读取仓库内置的本地 parquet（``data/offline/``，随项目分发、克隆即用），提供与
``DataFetcher`` / ``NeoDataSource`` 接口对齐的数据，实现**完全离线**取数（不触网）。

启用方式
--------
在 ``config.yaml`` 设置 ``data.source: offline``，并把调用点 ``DataFetcher()``
替换为 ``get_data_source(config)``（见 ``data.neo_adapter.DataSourceFactory``）。
默认即读取随仓库分发的 ``data/offline/``，无需任何联网或额外准备。

数据文件
--------
``data/offline/`` 下按指数池存放：

    data/offline/bars_<index>_part*.parquet   # 日K分片（含复权因子列）
    data/offline/constituents_<index>.json    # 指数成分股列表
    data/offline/meta.json                    # 导出元信息（时间范围、股票数、交易日数）

离线数据源各方法
----------------
- ``get_daily_kline``: 从 parquet 过滤 symbol/日期区间，返回
  ``date/open/high/low/close/volume/amount/pct_chg/symbol``（前复权 qfq 对齐 DataFetcher）。
- ``get_index_constituents``: 读取成分股 JSON（默认 csi800）。
- ``get_industry_and_cap`` / ``get_industry_classification`` / ``get_financial_data``:
  离线数据无行业/市值/财务字段，返回空（由上层中性化/多模态能力降级，不影响日K回测）。
- 其余方法（新闻情绪/快照/分钟K）无离线数据，返回空，不尝试联网。

注意：parquet 中的复权因子列按区间末因子归一化折算为前复权价（qfq），
以对齐 legacy DataFetcher 的默认复权语义。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# 允许从项目根目录 import
ROOT = Path(__file__).resolve().parents[2]


def _default_offline_dir() -> Path:
    """默认离线数据目录（config.data.offline.dir 未指定时）。"""
    env = os.environ.get("FACTORGPT_OFFLINE_DIR", "")
    if env:
        return Path(env)
    return ROOT / "data" / "offline"


class OfflineDataSource:
    """与 DataFetcher 接口对齐的完全离线数据源。"""

    def __init__(self, config: Optional[dict] = None, **kwargs: Any) -> None:
        cfg = config or {}
        data_cfg = cfg.get("data", {}) or {}
        offline_cfg = data_cfg.get("offline", {}) or {}
        index = offline_cfg.get("index") or data_cfg.get("offline_index") or "csi800"
        base = offline_cfg.get("dir") or str(_default_offline_dir())
        self.base = Path(base)
        self.index = str(index).lower()
        self.last_fetch_info: Dict[str, Any] = {"source": None, "message": ""}
        self._bars: Optional[pd.DataFrame] = None
        self._constituents: Optional[List[str]] = None
        self._meta: Dict[str, Any] = {}

        # 启动时预检查数据文件，缺失时给出明确指引
        if not self._bars_paths:
            self.last_fetch_info = {
                "source": "none",
                "message": (
                    f"离线数据缺失：{self.base} 下未找到 bars_{self.index}_*.parquet。"
                    f"请将随仓库分发的 data/offline/ 完整拷贝到 {self.base}。"
                ),
            }

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @property
    def _bars_paths(self) -> List[Path]:
        """日K parquet 分片路径（支持 bars_<index>_part*.parquet 多分片）。"""
        globs = sorted(self.base.glob(f"bars_{self.index}_part*.parquet"))
        # 兼容旧式单文件命名
        single = self.base / f"bars_{self.index}.parquet"
        if not globs and single.exists():
            globs = [single]
        return globs

    @property
    def _constituents_path(self) -> Path:
        return self.base / f"constituents_{self.index}.json"

    @property
    def _meta_path(self) -> Path:
        return self.base / "meta.json"

    def _load_bars(self) -> pd.DataFrame:
        """惰性加载全量日K parquet 分片（约 340 万行，内存 ~200MB，可接受）。"""
        if self._bars is None:
            paths = self._bars_paths
            if not paths:
                self._bars = pd.DataFrame()
            else:
                frames = [pd.read_parquet(p) for p in paths]
                self._bars = pd.concat(frames, ignore_index=True)
        return self._bars

    def _load_constituents(self) -> List[str]:
        if self._constituents is None:
            if self._constituents_path.exists():
                with open(self._constituents_path, "r", encoding="utf-8") as f:
                    self._constituents = json.load(f)
            else:
                self._constituents = []
        return self._constituents

    def _load_meta(self) -> Dict[str, Any]:
        if not self._meta and self._meta_path.exists():
            with open(self._meta_path, "r", encoding="utf-8") as f:
                self._meta = json.load(f)
        return self._meta

    @staticmethod
    def _norm_symbol(symbol: str) -> str:
        """归一化股票代码：600519 -> SH600519（对齐数据文件中的 instrument 命名）。"""
        s = str(symbol).strip().upper().replace(".", "").replace("_", "")
        if s.startswith(("SH", "SZ", "BJ")):
            return s
        # 裸 6 位代码：6/9 开头 -> SH，其余 -> SZ
        if len(s) == 6:
            return ("SH" if s[0] in "69" else "SZ") + s
        return s

    @staticmethod
    def _de_norm_symbol(inst: str) -> str:
        """instrument 代码 -> 6 位裸代码：SH600519 -> 600519。"""
        s = str(inst).upper()
        if s.startswith(("SH", "SZ", "BJ")) and len(s) == 8:
            return s[2:]
        return s

    # ------------------------------------------------------------------
    # 公开方法（与 DataFetcher 对齐）
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
        """从离线 parquet 返回日K（前复权，列与 DataFetcher 对齐）。"""
        if isinstance(symbols, str):
            symbols = [symbols]
        symbols = [str(s).strip() for s in symbols if str(s).strip()]
        if not symbols:
            self.last_fetch_info = {"source": "none", "message": "未提供股票代码"}
            return pd.DataFrame()

        if period != "daily":
            self.last_fetch_info = {"source": "none", "message": f"离线数据源仅支持日K，不支持 {period}"}
            return pd.DataFrame()

        bars = self._load_bars()
        if bars is None or bars.empty:
            self.last_fetch_info = {"source": "none", "message": "离线数据缺失，请检查 data/offline/ 目录完整性"}
            return pd.DataFrame()

        # 过滤：instrument 精确匹配 + 兼容裸代码
        norm = {self._norm_symbol(s): s for s in symbols}
        insts = [self._norm_symbol(s) for s in symbols]
        sub = bars[bars["instrument"].isin(insts)].copy()
        if sub.empty:
            # 尝试裸代码匹配（instrument 本身就是 6 位）
            sub = bars[bars["instrument"].isin([s for s in symbols])].copy()
        if sub.empty:
            self.last_fetch_info = {
                "source": "offline",
                "message": f"离线数据中未找到股票 {symbols}（index={self.index}）",
            }
            return pd.DataFrame()

        # 日期过滤
        start_d, end_d = str(start)[:10], str(end)[:10]
        sub = sub[(sub["date"] >= start_d) & (sub["date"] <= end_d)].copy()
        if sub.empty:
            self.last_fetch_info = {
                "source": "offline",
                "message": f"离线数据在 {start_d}~{end_d} 区间无行情（index={self.index}）",
            }
            return pd.DataFrame()

        # 前复权：qfq = raw * factor / factor_last（每只股票独立归一化）
        if adjust in ("qfq", "hfq"):
            for inst, grp in sub.groupby("instrument"):
                fac = grp["factor"]
                if fac.iloc[-1] and fac.iloc[-1] == fac.iloc[-1]:  # 非零且非 NaN
                    sub.loc[grp.index, ["open", "high", "low", "close"]] = (
                        grp[["open", "high", "low", "close"]] * fac.iloc[-1] / fac.values[:, None]
                    )

        # 规范化列：instrument -> symbol（6位裸代码），并计算 pct_chg
        sub["symbol"] = sub["instrument"].map(self._de_norm_symbol)
        sub = sub.sort_values(["symbol", "date"]).reset_index(drop=True)
        out = pd.DataFrame(
            {
                "date": sub["date"],
                "symbol": sub["symbol"],
                "open": sub["open"],
                "high": sub["high"],
                "low": sub["low"],
                "close": sub["close"],
                "volume": sub["volume"],
                "amount": sub["amount"],
            }
        )
        # 涨跌幅（与 DataFetcher 对齐的 pct_chg 列）
        out["pct_chg"] = out.groupby("symbol")["close"].pct_change().fillna(0.0) * 100.0
        self.last_fetch_info = {
            "source": "offline",
            "message": f"离线数据（index={self.index}，{len(out)} 行）",
        }
        return out

    def get_index_constituents(self, index_code: str = "000906") -> List[str]:
        """返回离线成分股列表（6 位裸代码）。"""
        codes = self._load_constituents()
        out = [self._de_norm_symbol(c) for c in codes]
        if not out:
            self.last_fetch_info = {
                "source": "none",
                "message": f"离线成分股缺失：{self._constituents_path} 不存在",
            }
        else:
            self.last_fetch_info = {
                "source": "offline",
                "message": f"离线成分股（index={self.index}，{len(out)} 只）",
            }
        return out

    def get_industry_and_cap(self, symbols: List[str]) -> pd.DataFrame:
        """离线数据无行业/市值字段，返回空（中性化维度缺失，不中断流水线）。"""
        self.last_fetch_info = {"source": "offline", "message": "离线数据无行业/市值字段，返回空"}
        return pd.DataFrame()

    def get_industry_classification(self) -> pd.DataFrame:
        self.last_fetch_info = {"source": "offline", "message": "离线数据无行业分类，返回空"}
        return pd.DataFrame()

    def get_financial_data(self, symbol: str, report_type: str = "年报") -> pd.DataFrame:
        self.last_fetch_info = {"source": "offline", "message": "离线数据无财务字段，返回空"}
        return pd.DataFrame()

    def get_news_sentiment(self, symbol: str = "", limit: int = 20) -> pd.DataFrame:
        self.last_fetch_info = {"source": "offline", "message": "离线数据无新闻情绪，返回空"}
        return pd.DataFrame()

    def get_market_snapshot(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        self.last_fetch_info = {"source": "offline", "message": "离线数据无市场快照，返回空"}
        return pd.DataFrame()

    def get_minute_kline(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        self.last_fetch_info = {"source": "offline", "message": "离线数据无分钟K，返回空"}
        return pd.DataFrame()


if __name__ == "__main__":
    ds = OfflineDataSource({"data": {"offline": {"index": "csi800"}}})
    print("数据源:", type(ds).__name__)
    print("meta:", ds._load_meta())
    cons = ds.get_index_constituents()
    print("成分股数量:", len(cons), "| 前3:", cons[:3])
    kl = ds.get_daily_kline(["600519"], "2024-01-01", "2024-01-10")
    print("茅台日K:", kl.shape)
    if not kl.empty:
        print(kl.head(3))
