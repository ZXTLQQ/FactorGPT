"""数据广度与多模态（P1）：把基本面 / 估值 / 资金流 / 新闻情绪纳入因子池。

精炼厂原本只消费量价因子（分钟/日频价量）。本模块将四类非量价数据模态接入因子池，
形成「量价 + 基本面 + 文本」多模态信号，是「金融创新」的强叙事点：

  * 资金流（主力净流入）        —— 捕捉机构博弈信息
  * 估值（PE / PB）            —— 价值维度
  * 基本面（ROE / 营收增长）   —— 质量维度
  * 新闻情绪（正负面打分）      —— 文本/另类数据维度

所有外部接口均做容错：某一模态不可用（离线 / 无权限 / API 变更）时自动跳过，
返回已成功获取的因子子集，保证精炼厂流水线不中断。
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger("factor_gpt.multimodal")

# 极简中文财经情绪词典（用于新闻文本打分，无需外部模型）
_POS = ["上涨", "增长", "盈利", "利好", "超预期", "突破", "增持", "买入", "回购", "中标", "签约", "获批"]
_NEG = ["下跌", "亏损", "利空", "低于预期", "下滑", "减持", "卖出", "处罚", "诉讼", "违约", "退市", "风险"]


def _market_prefix(symbol: str) -> str:
    return "sh" if symbol.startswith("6") else "sz"


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        logger.debug("multimodal 数据源调用失败: %s", e)
        return None


def build_multimodal_factors(kline: pd.DataFrame, symbols: list) -> dict:
    """构建多模态因子，返回 name -> (date,symbol) 索引 Series。

    各模态独立尝试；任一部分失败即跳过该模态因子，不影响其余。
    """
    factors: dict = {}
    kline = kline.copy()
    kline["date"] = kline["date"].astype(str)

    dates = sorted(kline["date"].unique())

    # 1) 资金流：主力净流入额（标准化为近 20 日 z-score 截面因子）
    fund_flow = _collect_fund_flow(symbols)
    if fund_flow is not None and not fund_flow.empty:
        factors["mm_main_net_inflow"] = _panel_to_factor(fund_flow, kline, "main_net_inflow", dates)

    # 2) 估值：市盈率（截面对数化，低估值得分高）
    valuation = _collect_valuation(symbols)
    if valuation is not None and not valuation.empty:
        factors["mm_pe_ttm"] = _panel_to_factor(valuation, kline, "pe", dates, log=True)

    # 3) 基本面：净资产收益率 ROE
    financial = _collect_financial(symbols)
    if financial is not None and not financial.empty:
        factors["mm_roe"] = _panel_to_factor(financial, kline, "roe", dates)

    # 4) 新闻情绪：每日正负面净打分
    sentiment = _collect_news_sentiment(symbols)
    if sentiment is not None and not sentiment.empty:
        factors["mm_news_sentiment"] = _panel_to_factor(sentiment, kline, "sentiment", dates)

    logger.info("多模态因子接入 %d 个：%s", len(factors), list(factors.keys()))
    return factors


def _panel_to_factor(panel: pd.DataFrame, kline: pd.DataFrame, col: str, dates,
                     log: bool = False) -> pd.Series:
    """将「symbol x date 面板」对齐到 kline 的 (date,symbol) 索引，做截面 z-score。"""
    panel = panel.copy()
    panel["date"] = panel["date"].astype(str)
    if col not in panel.columns:
        return pd.Series(dtype=float)
    s = panel.dropna(subset=[col]).set_index(["date", "symbol"])[col]
    if log:
        s = np.log1p(s.clip(lower=0))
    # 仅保留 kline 覆盖的 (date,symbol)
    idx = kline.set_index(["date", "symbol"]).index
    s = s.reindex(idx)
    # 逐日截面 z-score（更稳健的可比性）
    def _z(x):
        return (x - x.mean()) / (x.std() + 1e-12)
    s = s.groupby(level=0, group_keys=False).transform(_z)
    return s.dropna()


def _collect_fund_flow(symbols: list) -> pd.DataFrame:
    ak = _import_akshare()
    if ak is None:
        return pd.DataFrame()
    rows = []
    for sym in symbols:
        df = _safe(ak.stock_individual_fund_flow, stock=sym, market=_market_prefix(sym))
        if df is None or df.empty:
            continue
        df = df.rename(columns={"日期": "date", "主力净流入额": "main_net_inflow"})
        df["symbol"] = sym
        df = df[["date", "symbol", "main_net_inflow"]]
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _collect_valuation(symbols: list) -> pd.DataFrame:
    ak = _import_akshare()
    if ak is None:
        return pd.DataFrame()
    rows = []
    for sym in symbols:
        df = _safe(ak.stock_a_indicator, symbol=sym)
        if df is None or df.empty:
            continue
        df = df.rename(columns={"日期": "date", "市盈率-动态": "pe", "市净率": "pb"})
        df["symbol"] = sym
        for c in ("pe", "pb"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[["date", "symbol", "pe", "pb"]]
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _collect_financial(symbols: list) -> pd.DataFrame:
    ak = _import_akshare()
    if ak is None:
        return pd.DataFrame()
    rows = []
    for sym in symbols:
        df = _safe(ak.stock_financial_analysis_indicator, symbol=sym)
        if df is None or df.empty:
            continue
        df = df.rename(columns={"日期": "date", "净资产收益率": "roe",
                                "营业收入增长率": "revenue_growth"})
        df["symbol"] = sym
        for c in ("roe", "revenue_growth"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[["date", "symbol", "roe", "revenue_growth"]]
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _collect_news_sentiment(symbols: list, max_news: int = 60) -> pd.DataFrame:
    ak = _import_akshare()
    if ak is None:
        return pd.DataFrame()
    rows = []
    for sym in symbols:
        df = _safe(ak.stock_news_em, symbol=sym)
        if df is None or df.empty:
            continue
        df = df.rename(columns={"发布时间": "date", "新闻标题": "title", "新闻内容": "content"})
        df["symbol"] = sym
        # 解析日期（如 2024-03-01 09:30）
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["text"] = (df.get("title", "").fillna("") + " " + df.get("content", "").fillna(""))
        df["sentiment"] = df["text"].apply(_sentiment_score)
        daily = df.dropna(subset=["date"]).groupby(["date", "symbol"])["sentiment"].mean().reset_index()
        rows.append(daily)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _sentiment_score(text: str) -> float:
    if not isinstance(text, str):
        return 0.0
    pos = sum(len(re.findall(re.escape(w), text)) for w in _POS)
    neg = sum(len(re.findall(re.escape(w), text)) for w in _NEG)
    total = pos + neg
    return (pos - neg) / total if total else 0.0


def _import_akshare():
    try:
        import akshare as ak  # noqa: F401
        return ak
    except ImportError:
        logger.debug("akshare 未安装，多模态因子降级为空")
        return None
