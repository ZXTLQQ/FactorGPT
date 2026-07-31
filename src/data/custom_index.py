"""自主构建指数服务模块（Custom Index Builder）。

给定成分股列表与加权方案，基于各成分股日收盘价合成自定义指数时间序列：

- equal      等权：成分股日收益率等权平均后累乘
- price      价格加权：类道琼斯，按收盘价求和后相对基期归一（需基期价格）
- market_cap 市值加权：因缺乏实时股本，自动回退到价格加权
- custom     用户自定义权重：对各成分股收益率按权重加权

合成指数被归一化到 base_value（默认 1000），并可计算年化收益、波动率、
最大回撤、夏普等统计指标，便于与沪深300等基准对比。
"""

from typing import List, Optional, Dict, Any

import re
import pandas as pd
import numpy as np


class CustomIndexBuilder:
    """自主构建指数。"""

    VALID_WEIGHTINGS = ("equal", "price", "market_cap", "custom")

    def __init__(self, fetcher) -> None:
        """初始化。

        Args:
            fetcher: DataFetcher 实例，用于获取成分股日收盘价（已含多源回退）。
        """
        self.fetcher = fetcher

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------
    def build(
        self,
        constituents: List[str],
        start: str,
        end: str,
        base_date: Optional[str] = None,
        base_value: float = 1000.0,
        weighting: str = "equal",
        weights: Any = None,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """构建自定义指数时间序列。

        Args:
            constituents: 成分股代码列表，如 ["000001", "600519"]。
            start / end: 起止日期（支持多种格式）。
            base_date: 基期日期（可选）；价格加权方式下作为除数基准日，
                未提供则取数据首个交易日。
            base_value: 基期指数点位，默认 1000。
            weighting: 加权方式 equal / price / market_cap / custom。
            weights: custom 方式下的权重，支持 dict 或 "代码:权重,代码:权重" 字符串。
            adjust: 复权方式 qfq / hfq / ""。

        Returns:
            DataFrame（date / index_value / ret），失败或为空返回空 DataFrame。
        """
        constituents = [str(c).strip().zfill(6) for c in constituents if str(c).strip()]
        if not constituents:
            raise ValueError("成分股列表不能为空")
        if weighting not in self.VALID_WEIGHTINGS:
            raise ValueError(f"未知加权方式: {weighting}")

        w_vec = None
        if weighting == "custom":
            w_map = self._parse_weights(weights, constituents)
            w_vec = np.array([w_map.get(c, 0.0) for c in constituents], dtype=float)
            if w_vec.sum() == 0:
                weighting = "equal"
            else:
                w_vec = w_vec / w_vec.sum()

        panel = self._collect_closes(constituents, start, end, adjust)
        if panel.empty:
            return pd.DataFrame()

        # 对齐交易日（取并集并前向填充）
        panel = panel.sort_index().ffill().dropna(how="all")
        if panel.empty:
            return pd.DataFrame()

        if weighting in ("equal", "custom"):
            rets = panel.pct_change().fillna(0.0)
            vec = w_vec if weighting == "custom" else np.ones(len(constituents)) / len(constituents)
            port_ret = rets.mul(vec, axis=1).sum(axis=1)
            idx = (1.0 + port_ret).cumprod()
            idx = idx / idx.iloc[0] * base_value
        else:
            # price-weighted（道琼斯式）：基期总和作除数
            if base_date:
                base_row = panel.loc[: self._as_date(base_date)]
                base = base_row.iloc[-1] if not base_row.empty else panel.iloc[0]
            else:
                base = panel.iloc[0]
            divisor = base.sum()
            idx = panel.sum(axis=1) / divisor * base_value

        result = pd.DataFrame({"date": idx.index, "index_value": idx.values})
        result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
        result = result.reset_index(drop=True)
        result["ret"] = result["index_value"].pct_change() * 100
        return result

    # ------------------------------------------------------------------
    # 统计指标
    # ------------------------------------------------------------------
    def stats(self, idx_df: pd.DataFrame) -> Dict[str, float]:
        """计算指数绩效指标：累计/年化收益、波动率、最大回撤、夏普。"""
        if idx_df is None or idx_df.empty or "index_value" not in idx_df.columns:
            return {}
        vals = pd.to_numeric(idx_df["index_value"], errors="coerce").dropna().values
        if len(vals) < 2 or vals[0] == 0:
            return {}
        rets = pd.Series(vals).pct_change().dropna().values
        if len(rets) < 2:
            return {"total_return": 0.0, "annualized_return": 0.0,
                    "volatility": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}

        total = vals[-1] / vals[0] - 1.0
        years = len(rets) / 252.0
        ann = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 and (1.0 + total) > 0 else 0.0
        vol = rets.std() * np.sqrt(252)
        cummax = np.maximum.accumulate(vals)
        mdd = ((vals - cummax) / cummax).min()
        sharpe = (ann - 0.0) / vol if vol > 0 else 0.0
        return {
            "total_return": round(float(total * 100), 2),
            "annualized_return": round(float(ann * 100), 2),
            "volatility": round(float(vol * 100), 2),
            "max_drawdown": round(float(mdd * 100), 2),
            "sharpe": round(float(sharpe), 2),
        }

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _collect_closes(self, constituents, start, end, adjust) -> pd.DataFrame:
        start_fmt = self.fetcher._standardize_date(start)
        end_fmt = self.fetcher._standardize_date(end)
        closes: Dict[str, pd.Series] = {}
        for code in constituents:
            try:
                df = self.fetcher.get_daily_kline([code], start_fmt, end_fmt, "daily", adjust)
                if df is None or df.empty or "close" not in df.columns:
                    print(f"[CustomIndex] {code} 无行情数据，已跳过")
                    continue
                s = df.set_index("date")["close"]
                s.index = pd.to_datetime(s.index)
                closes[code] = s
            except Exception as e:
                print(f"[CustomIndex] {code} 行情获取失败: {e}")
        if not closes:
            return pd.DataFrame()
        return pd.DataFrame(closes)

    @staticmethod
    def _as_date(d: str):
        return pd.to_datetime(str(d).replace("/", "-"))

    @staticmethod
    def _parse_weights(weights: Any, constituents: List[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if isinstance(weights, dict):
            for k, v in weights.items():
                try:
                    out[str(k).strip().zfill(6)] = float(v)
                except (TypeError, ValueError):
                    continue
            return CustomIndexBuilder._normalize_wmap(out)
        if isinstance(weights, str):
            for part in re.split(r"[,\s;]+", weights.strip()):
                if not part:
                    continue
                if ":" in part:
                    k, v = part.split(":", 1)
                    try:
                        out[str(k).strip().zfill(6)] = float(v)
                    except (TypeError, ValueError):
                        continue
                else:
                    out[str(part).strip().zfill(6)] = 1.0
            return CustomIndexBuilder._normalize_wmap(out)
        # 无有效权重则等权
        return {c: 1.0 / len(constituents) for c in constituents}

    @staticmethod
    def _normalize_wmap(wmap: Dict[str, float]) -> Dict[str, float]:
        tot = sum(wmap.values())
        if tot == 0:
            return wmap
        return {k: v / tot for k, v in wmap.items()}
