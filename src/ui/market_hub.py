"""
行情中心（Market Hub）
====================

仿同花顺 / 东方财富的交易终端式行情架构页面，作为 FactorGPT 的「市场数据」
统一入口，并与因子挖掘模块互联互通。

功能结构：
1. 五大指数总览：上证指数 / 深证成指 / 创业板指 / 北证50 / 科创50，
   每张卡片含实时点、涨跌幅与迷你 K 线；点选指数下钻查看成分股行情。
2. 成分股行情表：最新价 / 涨跌幅 / 涨跌额 / 成交额等，行可点击弹出个股详情。
3. 个股详情弹窗：大字最新价 + K 线 + 当日分时 + 指标 + 新闻 / 研报，
   并可一键「用该股票运行因子挖掘」「查询相关因子知识」。
4. 走势对比：可选股票 / 指数 / 期货，归一化叠加对比（交互式）。
5. 后端 SQLite 缓存状态：展示短时数据存储调用情况。
所有行情数据经 ``src/data/cache_db`` 短时落库，支撑自动刷新下的低延迟复用。
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.market_data import MarketDataFetcher, FUTURES_MAIN_HINTS
from data.cache_db import get_cache_db, NS_QUOTE, NS_KLINE, NS_NEWS, NS_RESEARCH, NS_CONSTITUENT, NS_INDEX_SPOT

# 五大核心指数
MAJOR_INDICES = [
    {"name": "上证指数", "code": "000001", "market": "sh"},
    {"name": "深证成指", "code": "399001", "market": "sz"},
    {"name": "创业板指", "code": "399006", "market": "sz"},
    {"name": "北证50", "code": "899050", "market": "bj"},
    {"name": "科创50", "code": "000688", "market": "sh"},
]
INDEX_BY_CODE = {x["code"]: x for x in MAJOR_INDICES}

# A 股配色
C_UP = "#ef232a"    # 涨：红
C_DOWN = "#14b143"  # 跌：绿
C_FLAT = "#888888"
C_BG = "#0e1117"


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def _f(x, nd=2):
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:
        return "--"


def _color(v):
    try:
        v = float(v)
    except Exception:
        return C_FLAT
    return C_UP if v > 0 else (C_DOWN if v < 0 else C_FLAT)


def _fmt_amount(v):
    try:
        v = float(v)
    except Exception:
        return "--"
    if abs(v) >= 1e8:
        return f"{v/1e8:.2f}亿"
    if abs(v) >= 1e4:
        return f"{v/1e4:.2f}万"
    return f"{v:.0f}"


def _sina_symbol(code: str) -> str:
    code = str(code).strip().zfill(6)
    if code.startswith("6"):
        return "sh" + code
    if code.startswith(("0", "3")):
        return "sz" + code
    if code.startswith(("4", "8")):
        return "bj" + code
    return "sh" + code


# ----------------------------------------------------------------------
# 图表
# ----------------------------------------------------------------------
def _candlestick(df, height=380, with_volume=True, title=""):
    """通用 K 线图（含成交量副图）。"""
    fig = go.Figure()
    if df is None or df.empty:
        fig.add_annotation(text="暂无行情数据", showarrow=False)
        fig.update_layout(height=height, template="plotly_dark",
                          paper_bgcolor=C_BG, plot_bgcolor=C_BG)
        return fig
    date_col = next((c for c in df.columns if "日期" in str(c)), df.columns[0])
    req = ["开盘", "收盘", "最高", "最低"]
    if all(c in df.columns for c in req):
        fig.add_trace(go.Candlestick(
            x=df[date_col], open=df["开盘"], close=df["收盘"],
            high=df["最高"], low=df["最低"],
            increasing_line_color=C_UP, decreasing_line_color=C_DOWN,
            name="K线",
        ))
    else:
        fig.add_trace(go.Scatter(x=df[date_col], y=df["收盘"],
                                  line=dict(color="#4ea1ff"), name="收盘"))
    if with_volume and "成交量" in df.columns:
        fig.add_trace(go.Bar(x=df[date_col], y=df["成交量"],
                             name="成交量", marker_color="#3a4a5a",
                             yaxis="y2", opacity=0.5))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right",
                                      showgrid=False, visible=False))
    fig.update_layout(
        height=height, template="plotly_dark", paper_bgcolor=C_BG,
        plot_bgcolor=C_BG, title=title, xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.02, x=0),
        hovermode="x unified",
    )
    return fig


def _mini_line(series, color, height=46):
    """迷你走势线（卡片用）。"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=series, line=dict(color=color, width=1.6),
                             showlegend=False))
    fig.update_layout(height=height, template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(visible=False),
                      yaxis=dict(visible=False))
    return fig


def _compare_figure(items):
    """归一化为起始=0% 的叠加折线。items: [(label, df, kind)]"""
    fig = go.Figure()
    palette = ["#4ea1ff", "#ef232a", "#14b143", "#f5a623", "#b06bff", "#22d3ee"]
    for i, (label, df, kind) in enumerate(items):
        if df is None or df.empty:
            continue
        date_col = next((c for c in df.columns if "日期" in str(c)), df.columns[0])
        close = pd.to_numeric(df["收盘"], errors="coerce").dropna().reset_index(drop=True)
        if close.empty:
            continue
        base = close.iloc[0]
        norm = (close / base - 1) * 100
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(x=df[date_col].iloc[: len(norm)], y=norm,
                                 mode="lines", name=label,
                                 line=dict(color=color, width=2)))
    fig.update_layout(height=460, template="plotly_dark", paper_bgcolor=C_BG,
                      plot_bgcolor=C_BG, margin=dict(l=40, r=20, t=30, b=30),
                      yaxis_title="区间涨跌幅 (%)", hovermode="x unified",
                      legend=dict(orientation="h", y=1.04, x=0))
    return fig


# ----------------------------------------------------------------------
# 数据获取封装（带短时缓存已内置于 MarketDataFetcher）
# ----------------------------------------------------------------------
def _index_spot_dict(code: str) -> dict:
    df, err = MarketDataFetcher.index_spot(code)
    if err or df is None or df.empty:
        return {}
    r = df.iloc[0].to_dict()
    # 兼容多种返回列名（同花顺 dict / 东财历史末行 / 规整后 DataFrame）
    name = r.get("名称") or r.get("name") or code
    val = r.get("value") or r.get("最新") or r.get("收盘")
    chg = r.get("chg") or r.get("涨跌")
    pct = r.get("chg_pct") or r.get("涨跌幅")
    return {"name": name, "value": val, "chg": chg, "pct": pct}


def _constituents_table(index_code: str):
    """返回 ``(DataFrame, cons, has_quote)``。

    has_quote 表示是否成功取到实时行情（含非交易时段当日收盘数据）。
    行情缺失时相关列以 ``"--"`` 占位，由上层 UI 给出友好提示。
    """
    cons, _ = MarketDataFetcher.index_constituents(index_code)
    if not cons:
        return pd.DataFrame(), cons, False
    quotes, _ = MarketDataFetcher.quotes_for([c["code"] for c in cons])
    has_quote = bool(quotes)
    rows = []
    for c in cons:
        q = quotes.get(c["code"], {})
        rows.append({
            "代码": c["code"],
            "名称": c["name"],
            "最新价": q.get("最新价") if has_quote else "--",
            "涨跌幅(%)": q.get("涨跌幅") if has_quote else "--",
            "涨跌额": q.get("涨跌额") if has_quote else "--",
            "成交额": q.get("成交额") if has_quote else "--",
            "总量": q.get("成交量") if has_quote else "--",
        })
    df = pd.DataFrame(rows)
    return df, cons, has_quote


# ----------------------------------------------------------------------
# 股票详情弹窗（与因子挖掘模块联动）
# ----------------------------------------------------------------------
@st.dialog("个股行情 · 因子联动")
def _stock_dialog(code: str, name: str):
    code6 = str(code).strip().zfill(6)
    st.markdown(f"### {name} &nbsp; <span style='color:#888'>{code6}</span>",
                unsafe_allow_html=True)

    # 实时报价（多源回退：东财/新浪快照 → 新浪 HTTP 直连，非交易时段可用）
    spot = MarketDataFetcher.stock_realtime(code6)
    row = None
    latest = None
    pct = None
    if spot is not None and not spot.empty:
        row = spot.iloc[0]
        latest = row.get("最新价")
        pct = row.get("涨跌幅")
    if row is None:
        st.info("📴 实时报价暂未取到（非交易时段或网络受限）。"
                "已自动回退新浪直连取当日收盘数据；若仍为空，请检查网络或稍后重试。")
    col1, col2, col3 = st.columns(3)
    col1.metric("最新价", _f(latest), delta=_f(pct) + "%" if pct is not None else None)
    if row is not None:
        col2.metric("涨跌额", _f(row.get("涨跌额")))
        col3.metric("成交额", _fmt_amount(row.get("成交额")))

    # K 线（日线，近 120 交易日）
    with st.spinner("加载 K 线…"):
        kdf, kerr = MarketDataFetcher.stock_kline(symbol=code6, period="daily",
                                                  days=120, adjust="qfq")
    st.plotly_chart(_candlestick(kdf, height=320, title="日 K（前复权）"),
                    width='stretch')

    # 当日分时
    with st.spinner("加载分时…"):
        idf, ierr = MarketDataFetcher.stock_intraday(symbol=code6)
    if idf is not None and not idf.empty:
        st.plotly_chart(_candlestick(idf, height=200, with_volume=False,
                                     title="当日分时"),
                        width='stretch')
    else:
        st.caption("分时数据暂不可用")

    # 资讯 / 研报
    tab_news, tab_research = st.tabs(["新闻资讯", "机构研报"])
    with tab_news:
        ndf, _ = MarketDataFetcher.stock_news(code6, days=30)
        if ndf is not None and not ndf.empty:
            ncol = next((c for c in ndf.columns if "新闻" in str(c) or "标题" in str(c)),
                        ndf.columns[0])
            tcol = next((c for c in ndf.columns if "时间" in str(c) or "日期" in str(c)), None)
            for _, r in ndf.head(12).iterrows():
                t = f" [{r[tcol]}]" if tcol and tcol in r else ""
                st.markdown(f"- {r[ncol]}{t}")
        else:
            st.caption("暂无新闻")
    with tab_research:
        rdf, _ = MarketDataFetcher.stock_research(code6)
        if rdf is not None and not rdf.empty:
            for _, r in rdf.head(12).iterrows():
                cols = " · ".join(str(r[c]) for c in rdf.columns[:3])
                st.markdown(f"- {cols}")
        else:
            st.caption("暂无研报")

    # 与因子挖掘模块联动
    st.divider()
    st.markdown("#### 🔗 因子挖掘联动")
    c1, c2, c3 = st.columns(3)
    if c1.button("🤖 运行因子挖掘", key="mh_run_factor"):
        with st.spinner("正在调用因子挖掘 Agent…"):
            report = _run_factor_for_stock(code6, name)
        st.markdown(report)
    if c2.button("🔍 相关因子知识", key="mh_rag"):
        with st.spinner("检索因子知识库…"):
            kbase = _related_factor_knowledge(code6, name)
        if kbase:
            for k in kbase:
                st.markdown(f"- {k}")
        else:
            st.caption("知识库暂无相关内容")
    if c3.button("⭐ 加入自选", key="mh_add_wl"):
        wl = st.session_state.get("stock_watchlist", [])
        if code6 not in wl:
            wl.append(code6)
            st.session_state.stock_watchlist = wl
            st.success("已加入自选股")
        else:
            st.info("已在自选股中")

    if st.button("关闭", key="mh_close"):
        st.session_state.mh_stock = None
        st.rerun()


@st.cache_resource
def _factor_agent():
    """构建并缓存因子挖掘 Agent（与主页共享同一 FactorAgent 类）。"""
    from agent.graph import FactorAgent
    import yaml
    cfg_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    return FactorAgent(cfg)


def _run_factor_for_stock(code: str, name: str) -> str:
    """调用因子挖掘 Agent，围绕该股票生成因子思路。"""
    try:
        agent = _factor_agent()
    except Exception as e:  # noqa
        return f"因子挖掘模块未就绪：{e}"
    prompt = (
        f"请围绕 A 股股票 {name}（代码 {code}）设计 2-3 个可量化的选股/择时因子，"
        f"给出每个因子的名称、计算逻辑、所需字段与回测思路，并说明其经济学含义。"
    )
    try:
        res = agent.run(prompt)
        if isinstance(res, dict):
            return res.get("report") or res.get("answer") or str(res)
        return str(res)
    except Exception as e:  # noqa
        return f"因子挖掘调用失败：{e}"


def _related_factor_knowledge(code: str, name: str) -> list:
    """基于股票名称/代码检索因子知识库（离线 + 向量）。"""
    try:
        from rag.paper_index import FactorPaperIndex
        from rag.retriever import FactorRetriever
    except Exception:
        return []
    idx = FactorPaperIndex()
    if not idx.available:
        return []
    retr = FactorRetriever(index=idx, top_k=5)
    res = retr.retrieve(f"{name} {code} 选股因子 研报")
    return [r[:200] for r in res]


# ----------------------------------------------------------------------
# 指数总览 + 成分股下钻
# ----------------------------------------------------------------------
def _render_index_overview():
    st.markdown("#### 五大核心指数 · 实时")
    cols = st.columns(5)
    for i, idx in enumerate(MAJOR_INDICES):
        with cols[i]:
            spot = _index_spot_dict(idx["code"])
            val = spot.get("value")
            pct = spot.get("pct")
            chg = spot.get("chg")
            col = _color(pct)
            # 迷你 K 线
            with st.spinner(""):
                k, _ = MarketDataFetcher.index_kline(idx["code"], days=40)
            if k is not None and not k.empty:
                close = pd.to_numeric(k["收盘"], errors="coerce").dropna()
                st.plotly_chart(_mini_line(close, col), width='stretch')
            else:
                st.caption("—")
            st.markdown(
                f"<div style='text-align:center'>"
                f"<b>{idx['name']}</b><br>"
                f"<span style='font-size:18px;color:{col}'>{_f(val)}</span><br>"
                f"<span style='color:{col}'>{_f(chg)} ({_f(pct)}%)</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button("查看成分股 →", key=f"idx_{idx['code']}"):
                st.session_state.mh_index = idx["code"]
                st.session_state.mh_cons_df = None
                st.rerun()

    st.divider()

    # 已选指数详情
    sel = st.session_state.get("mh_index")
    if not sel:
        st.info("👆 点击上方指数卡片，查看其成分股行情与 K 线。")
        return
    meta = INDEX_BY_CODE[sel]
    st.markdown(f"#### {meta['name']}（{sel}）成分股行情")

    # 指数 K 线
    with st.spinner("加载指数 K 线…"):
        kdf, _ = MarketDataFetcher.index_kline(sel, days=180)
    st.plotly_chart(_candlestick(kdf, height=320, title=f"{meta['name']} 日 K"),
                    width='stretch')

    # 先读取上一轮渲染的成分股表（保证点击行号映射到正确的股票）
    stored = st.session_state.get("mh_cons_df")
    if not st.session_state.get("mh_stock"):
        sel_rows = st.session_state.get("mh_cons_tbl", {}).get("selection", {}).get("rows", [])
        if sel_rows and stored is not None:
            row = stored.iloc[sel_rows[0]]
            st.session_state.mh_stock = (str(row["代码"]).zfill(6), row["名称"])
            st.session_state["mh_cons_tbl"] = {"selection": {"rows": []}}
            st.rerun()
            return

    # 成分股表
    with st.spinner("加载成分股行情…"):
        df, cons, has_quote = _constituents_table(sel)
    if df.empty:
        st.warning("成分股列表暂不可用（指数成分接口受限）。可在「走势对比」中手动输入代码查看。")
        return
    if not has_quote:
        st.info("📴 实时行情接口暂未取得数据（非交易时段或网络受限）。"
                "已自动回退新浪直连取当日收盘数据；若仍为空，请检查网络或稍后重试。")
    df_disp = df.copy()
    for c in ("最新价", "涨跌幅(%)", "涨跌额"):
        df_disp[c] = pd.to_numeric(df_disp[c], errors="coerce")
    df_disp = df_disp.sort_values("涨跌幅(%)", ascending=False).reset_index(drop=True)
    st.session_state.mh_cons_df = df_disp

    st.dataframe(
        df_disp,
        width='stretch',
        height=460,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "涨跌幅(%)": st.column_config.NumberColumn(format="%.2f%%",
                help="当日涨跌幅"),
            "最新价": st.column_config.NumberColumn(format="%.2f"),
            "成交额": st.column_config.TextColumn(),
        },
        key="mh_cons_tbl",
    )


# ----------------------------------------------------------------------
# 走势对比
# ----------------------------------------------------------------------
def _render_compare():
    st.markdown("#### 走势对比（归一化：起点 = 0%）")
    st.caption("选择股票 / 指数 / 期货，叠加对比区间涨跌幅。支持手动输入代码。")

    # 预设选择
    idx_opt = {f"{x['name']}({x['code']})": x["code"] for x in MAJOR_INDICES}
    fut_opt = {f"{n}({s})": s for n, s in FUTURES_MAIN_HINTS}
    wl = st.session_state.get("stock_watchlist", [])
    wl_opt = {f"自选 {c}": c for c in wl}

    all_presets = {}
    all_presets.update(idx_opt)
    all_presets.update(fut_opt)
    all_presets.update(wl_opt)

    picks = st.multiselect(
        "选择对比标的（≤6）",
        options=list(all_presets.keys()),
        default=list(idx_opt.keys())[:3],
        max_selections=6,
        key="mh_cmp_picks",
    )
    custom = st.text_input("或输入代码/期货符号（回车添加，如 600519 / au9999）",
                           key="mh_cmp_custom")
    period = st.selectbox("周期", ["daily", "weekly", "monthly"], index=0)
    days = st.slider("回看天数", 30, 365, 180, step=5)

    targets = list(picks)
    if custom.strip():
        targets.append(custom.strip())

    items = []
    for t in targets:
        code = all_presets.get(t, t)
        if code in idx_opt.values():
            df, err = MarketDataFetcher.index_kline(code, period=period, days=days)
            label = next((k for k, v in idx_opt.items() if v == code), code)
        elif code in [s for _, s in FUTURES_MAIN_HINTS] or ("9999" in code or code.isalpha()):
            df, err = MarketDataFetcher.futures_kline(symbol=code, period=period, days=days)
            label = t
        else:
            df, err = MarketDataFetcher.stock_kline(symbol=code, period=period,
                                                    days=days, adjust="qfq")
            label = t
        items.append((label, df, "stock" if df is not None else "na"))

    if items:
        st.plotly_chart(_compare_figure(items), width='stretch')
    else:
        st.info("请选择至少一个对比标的。")


# ----------------------------------------------------------------------
# 自选股
# ----------------------------------------------------------------------
def _render_watchlist():
    st.markdown("#### 我的自选股")
    wl = st.session_state.get("stock_watchlist", [])
    add = st.text_input("添加自选（6 位代码）", key="mh_wl_add")
    if st.button("➕ 添加", key="mh_wl_btn") and add.strip():
        c = add.strip().zfill(6)
        if c not in wl:
            wl.append(c)
            st.session_state.stock_watchlist = wl
            st.rerun()
    if not wl:
        st.info("自选股为空，可在个股弹窗中「加入自选」。")
        return
    # 先读取上一轮渲染的表，保证点击行号映射到正确的股票
    stored = st.session_state.get("mh_wl_df")
    if not st.session_state.get("mh_stock"):
        sel = st.session_state.get("mh_wl_tbl", {}).get("selection", {}).get("rows", [])
        if sel and stored is not None:
            r = stored.iloc[sel[0]]
            st.session_state.mh_stock = (str(r["代码"]).zfill(6), r["名称"])
            st.session_state["mh_wl_tbl"] = {"selection": {"rows": []}}
            st.rerun()
            return
    quotes, _ = MarketDataFetcher.quotes_for(wl)
    rows = []
    for c in wl:
        q = quotes.get(c, {})
        name = q.get("名称", c)
        rows.append({
            "代码": c, "名称": name,
            "最新价": q.get("最新价"), "涨跌幅(%)": q.get("涨跌幅"),
            "涨跌额": q.get("涨跌额"), "成交额": q.get("成交额"),
        })
    df = pd.DataFrame(rows)
    df["涨跌幅(%)"] = pd.to_numeric(df["涨跌幅(%)"], errors="coerce")
    df = df.sort_values("涨跌幅(%)", ascending=False).reset_index(drop=True)
    st.session_state.mh_wl_df = df
    st.dataframe(df, width='stretch', height=400,
                 selection_mode="single-row", on_select="rerun", key="mh_wl_tbl")


# ----------------------------------------------------------------------
# 后端缓存状态
# ----------------------------------------------------------------------
def _render_db_status():
    st.markdown("#### 🗄️ 后端短时缓存（SQLite）")
    try:
        stats = get_cache_db().stats()
    except Exception as e:  # noqa
        st.caption(f"缓存读取失败：{e}")
        return
    if not stats:
        st.caption("暂无缓存数据。")
        return
    ns_name = {
        NS_QUOTE: "实时报价", NS_KLINE: "K线", NS_INTRADAY: "分时",
        NS_CONSTITUENT: "指数成分", NS_NEWS: "新闻", NS_RESEARCH: "研报",
        NS_INDEX_SPOT: "指数点",
    }
    rows = [{"命名空间": ns_name.get(k, k), "条数": v["count"],
             "最近更新": datetime.fromtimestamp(v["last"]).strftime("%H:%M:%S")
             if v["last"] else "--"} for k, v in stats.items()]
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    if st.button("清空缓存", key="mh_clear_db"):
        get_cache_db().clear()
        st.success("已清空后端缓存")
        st.rerun()


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def render_market_hub():
    st.markdown("## 📈 行情中心 · Market Hub")

    # 自动刷新控制
    auto = st.checkbox("自动刷新", value=True, key="mh_auto")
    interval = st.slider("刷新间隔（秒）", 10, 120, 20, step=5, key="mh_interval")

    # 自定义 Tab（session_state 持久化，避免自动刷新时跳回首个标签）
    TABS = ["指数行情", "走势对比", "自选股", "后端缓存"]
    if "mh_tab" not in st.session_state:
        st.session_state.mh_tab = TABS[0]
    sel_cols = st.columns(len(TABS))
    for i, t in enumerate(TABS):
        active = st.session_state.mh_tab == t
        if sel_cols[i].button(t, width='stretch',
                              type="primary" if active else "secondary",
                              key=f"mh_tabbtn_{i}"):
            st.session_state.mh_tab = t
    st.divider()

    tab = st.session_state.mh_tab
    if tab == "指数行情":
        _render_index_overview()
    elif tab == "走势对比":
        _render_compare()
    elif tab == "自选股":
        _render_watchlist()
    else:
        _render_db_status()

    # 股票详情弹窗
    pick = st.session_state.get("mh_stock")
    if pick:
        _stock_dialog(pick[0], pick[1])

    # 自动刷新（主页面，非弹窗内）
    if auto and not pick:
        time.sleep(interval)
        st.rerun()
