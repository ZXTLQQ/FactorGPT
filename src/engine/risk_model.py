"""Barra 风格风险模型归因（轻量实现）。

用于对「因子值」做风格与行业暴露归因，回答两个问题：
  1) 这个因子在「偷偷」押注哪些风格（规模 / 动量 / 波动率）？
  2) 这个因子对行业是否中性？是否存在某行业重度偏配？

说明：完整 Barra 模型需要估值、杠杆、成长等基本面因子与协方差矩阵，
本模块仅基于行情可得字段（市值、收益、波动率、行业）做**代理风格暴露**估计，
用于快速诊断因子暴露结构与行业中性程度，足够支撑竞赛答辩的「风险归因」展示。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


class RiskModel:
    """基于行情代理变量的风格 / 行业暴露归因。"""

    def style_exposure(
        self,
        factor_series: pd.Series,
        kline: pd.DataFrame,
        mkt_cap: Optional[pd.Series] = None,
        lookback: int = 20,
    ) -> Dict[str, float]:
        """因子对常见风格的暴露（时序平均截面秩相关）。

        返回每个风格的暴露值，正值表示因子在该风格上有正向载荷。
        风格代理变量：
          - size（规模）：截面 log(市值) 排名
          - momentum（动量）：过去 lookback 日累计收益
          - volatility（波动率）：过去 lookback 日收益标准差
        """
        if factor_series is None or kline is None or kline.empty:
            return {}
        try:
            f = factor_series.rename("factor").reset_index()
            panel = kline[["date", "symbol", "pct_chg"]].merge(f, on=["date", "symbol"], how="inner")
            if panel.empty:
                return {}

            # 风格代理变量（按 symbol 截面构造）
            if mkt_cap is not None and not mkt_cap.empty:
                mc = mkt_cap.rename("size").reset_index()
                mc = mc.rename(columns={mc.columns[0]: "symbol"})
                panel = panel.merge(mc, on="symbol", how="left")
                panel["size"] = panel.groupby("date")["size"].rank(pct=True)
            else:
                panel["size"] = np.nan

            panel = panel.sort_values(["symbol", "date"])
            panel["momentum"] = panel.groupby("symbol")["pct_chg"].transform(
                lambda x: (1 + x).rolling(lookback).apply(lambda w: w.prod() - 1, raw=True)
            )
            panel["volatility"] = panel.groupby("symbol")["pct_chg"].transform(
                lambda x: x.rolling(lookback).std()
            )
            panel["factor"] = panel.groupby("date")["factor"].rank(pct=True)

            exposures: Dict[str, float] = {}
            for style in ["size", "momentum", "volatility"]:
                if style not in panel.columns:
                    continue
                corrs = []
                for _, g in panel.groupby("date"):
                    sub = g[[style, "factor"]].dropna()
                    if len(sub) >= 10:
                        corrs.append(sub[style].corr(sub["factor"]))
                exposures[style] = float(np.nanmean(corrs)) if corrs else float("nan")
            return exposures
        except Exception as e:  # noqa: BLE001
            return {"error": f"风格暴露计算失败: {type(e).__name__}: {e}"}

    def industry_exposure(
        self,
        factor_series: pd.Series,
        industry: Optional[pd.Series] = None,
        neutral_threshold: float = 0.1,
    ) -> Dict:
        """行业暴露与中性度诊断。

        返回：各行业因子均值的秩（by_industry）、跨行业最大偏配幅度（max_bias，
        秩单位 0~1）、以及是否中性（neutral）。
        """
        if factor_series is None or industry is None or industry.empty:
            return {"neutral": True, "max_bias": 0.0, "by_industry": {}}
        try:
            ind = industry.rename("industry").reset_index()
            ind = ind.rename(columns={ind.columns[0]: "symbol"})
            f = factor_series.rename("factor").reset_index()
            panel = f.merge(ind, on="symbol", how="inner")
            if panel.empty:
                return {"neutral": True, "max_bias": 0.0, "by_industry": {}}
            panel["factor_rank"] = panel.groupby("date")["factor"].rank(pct=True)
            by_ind = panel.groupby("industry")["factor_rank"].mean().to_dict()
            by_ind = {str(k): float(v) for k, v in by_ind.items()}
            vals = list(by_ind.values())
            max_bias = float(max(vals) - min(vals)) if vals else 0.0
            return {
                "by_industry": by_ind,
                "max_bias": max_bias,
                "neutral": max_bias <= neutral_threshold,
            }
        except Exception as e:  # noqa: BLE001
            return {"neutral": True, "max_bias": 0.0, "by_industry": {}, "error": str(e)}

    def attribution_report(self, style: Dict[str, float], industry: Dict) -> str:
        """生成一段人类可读的风险归因文本。"""
        lines = ["**风险暴露归因**"]
        if style and "error" not in style:
            parts = []
            label = {"size": "规模", "momentum": "动量", "volatility": "波动率"}
            for k, v in style.items():
                if isinstance(v, float) and not np.isnan(v):
                    direction = "正向" if v > 0.02 else ("负向" if v < -0.02 else "中性")
                    parts.append(f"{label.get(k, k)}（{direction}，暴露={v:+.3f}）")
            if parts:
                lines.append("- 风格暴露：" + "；".join(parts))
        if industry:
            if industry.get("neutral"):
                lines.append(f"- 行业暴露：中性（跨行业最大偏配 {industry.get('max_bias', 0):.3f}）")
            else:
                top = max(industry.get("by_industry", {}).items(), key=lambda kv: kv[1], default=("", 0))
                lines.append(
                    f"- 行业暴露：**非中性**（跨行业最大偏配 {industry.get('max_bias', 0):.3f}，"
                    f"偏高行业：{top[0]}）"
                )
        return "\n".join(lines)


def factor_risk_attribution(
    factor_series: pd.Series,
    kline: pd.DataFrame,
    industry: Optional[pd.Series] = None,
    mkt_cap: Optional[pd.Series] = None,
) -> Dict:
    """一站式风险归因：返回 {style, industry, report}。"""
    rm = RiskModel()
    style = rm.style_exposure(factor_series, kline, mkt_cap=mkt_cap)
    ind = rm.industry_exposure(factor_series, industry=industry)
    return {"style": style, "industry": ind, "report": rm.attribution_report(style, ind)}
