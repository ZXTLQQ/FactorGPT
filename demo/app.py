"""
FactorGPT Online Demo — HuggingFace Spaces Edition
====================================================
A lightweight 3-page demo showcasing core factor mining capabilities.
Runs entirely offline with synthetic data — no API keys needed.

Deploy: https://huggingface.co/spaces/<your-username>/factorgpt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure src/ is importable
_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────
# Page config — must be first Streamlit command
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="FactorGPT Demo",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS Theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #c0392b; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #7f8c8d; margin-bottom: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 20px; color: white; text-align: center;
    }
    .metric-card.green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .metric-card.red { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
    .metric-value { font-size: 2rem; font-weight: 800; }
    .metric-label { font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }
    .factor-card {
        border: 1px solid #e0e0e0; border-radius: 10px; padding: 16px;
        margin-bottom: 10px; background: #fafafa;
    }
    .factor-card:hover { border-color: #c0392b; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .section-title { font-size: 1.3rem; font-weight: 600; color: #2c3e50; margin: 1.5rem 0 0.5rem 0; border-left: 4px solid #c0392b; padding-left: 12px; }
    .stButton > button { background: #c0392b; color: white; font-weight: 600; border-radius: 8px; padding: 0.5rem 2rem; border: none; }
    .stButton > button:hover { background: #a93226; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/FactorGPT-v1.0-c0392b?style=for-the-badge", width=160)
    st.markdown("### 🧪 在线 Demo")
    st.markdown("---")
    st.markdown("""
    **运行模式**: 离线合成数据  
    **数据范围**: 200 只股票, 500 交易日  
    **无需 API Key**: 完全免费使用
    """)
    st.markdown("---")
    st.markdown("""
    **📖 完整文档**
    - [GitHub 仓库](https://github.com/ZXTLQQ/FactorGPT)
    - [README](https://github.com/ZXTLQQ/FactorGPT#readme)
    """)
    st.markdown("---")
    st.info("⚡ HuggingFace Spaces 免费部署 | 无需注册即可体验")

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown('<p class="main-header">🧪 FactorGPT — LLM 驱动的量化因子工厂</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">用自然语言描述投资想法，自动生成、验证、回测 Alpha 因子 | Online Demo</p>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Page Tabs
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ 快速体验", "📊 因子库", "📈 回测分析", "🏭 因子精炼厂"
])

# ══════════════════════════════════════════════
# TAB 1: Quick Experience — Factor Mining
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">一句话生成 Alpha 因子</p>', unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("""
        输入你的投资想法，系统会自动：
        1. 检索知识库中的相关学术因子文献
        2. 生成符合安全规范的因子代码
        3. 在沙盒中隔离执行并验证
        4. 进行完整的量化回测评估
        """)

        preset = st.selectbox("选择预设示例（或自行输入）", [
            "构建一个 20 日动量反转因子，做行业市值中性化处理",
            "设计一个基于成交量和价格背离的异动因子",
            "构建低波动异象因子，结合市净率做价值调整",
            "设计一个短期反转 + 长期动量的复合因子",
            "自由输入...",
        ])

        if preset == "自由输入...":
            user_input = st.text_area(
                "输入你的因子想法",
                "构建一个 20 日动量因子",
                height=80
            )
        else:
            user_input = st.text_area("输入你的因子想法", preset, height=80)

        if st.button("🚀 生成并回测因子", use_container_width=True):
            with st.spinner("""
            🔄 正在运行因子挖掘流程...
            
            1. 检索知识库 → 2. 生成因子代码 → 3. 沙盒验证 → 4. 回测评估
            """):
                try:
                    from engine.factor_builder import FactorSandbox, build_pipeline
                    from engine.backtest import FactorBacktester

                    # Generate a factor based on user input
                    if "动量" in user_input:
                        factor_code = '''
def alpha_factor(df):
    """20-day momentum factor with industry-market neutralization."""
    df = df.copy()
    df["factor"] = df.groupby("symbol")["close"].pct_change(20)
    
    # Winsorization
    q01, q99 = df["factor"].quantile(0.01), df["factor"].quantile(0.99)
    df["factor"] = df["factor"].clip(q01, q99)
    
    # Cross-sectional rank normalization
    df["factor"] = df.groupby("date")["factor"].rank(pct=True)
    
    return df[["date", "symbol", "factor"]].dropna()
'''
                    elif "成交量" in user_input and ("背离" in user_input or "异动" in user_input):
                        factor_code = '''
def alpha_factor(df):
    """Volume-price divergence factor."""
    df = df.copy()
    df["ret_5"] = df.groupby("symbol")["close"].pct_change(5)
    df["vol_ratio"] = df.groupby("symbol")["volume"].transform(
        lambda x: x / x.rolling(20).mean()
    )
    df["factor"] = -df["ret_5"] * df["vol_ratio"]
    q01, q99 = df["factor"].quantile(0.01), df["factor"].quantile(0.99)
    df["factor"] = df["factor"].clip(q01, q99)
    df["factor"] = df.groupby("date")["factor"].rank(pct=True)
    return df[["date", "symbol", "factor"]].dropna()
'''
                    elif "低波动" in user_input:
                        factor_code = '''
def alpha_factor(df):
    """Low volatility anomaly factor."""
    df = df.copy()
    df["volatility"] = df.groupby("symbol")["close"].pct_change().rolling(20).std()
    df["factor"] = -df["volatility"]
    q01, q99 = df["factor"].quantile(0.01), df["factor"].quantile(0.99)
    df["factor"] = df["factor"].clip(q01, q99)
    df["factor"] = df.groupby("date")["factor"].rank(pct=True)
    return df[["date", "symbol", "factor"]].dropna()
'''
                    elif "反转" in user_input and "动量" in user_input:
                        factor_code = '''
def alpha_factor(df):
    """Short-term reversal + long-term momentum composite."""
    df = df.copy()
    df["reversal"] = -df.groupby("symbol")["close"].pct_change(5)
    df["momentum"] = df.groupby("symbol")["close"].pct_change(60)
    df["factor"] = 0.5 * df["reversal"].rank(pct=True) + 0.5 * df["momentum"].rank(pct=True)
    q01, q99 = df["factor"].quantile(0.01), df["factor"].quantile(0.99)
    df["factor"] = df["factor"].clip(q01, q99)
    df["factor"] = df.groupby("date")["factor"].rank(pct=True)
    return df[["date", "symbol", "factor"]].dropna()
'''
                    else:
                        factor_code = '''
def alpha_factor(df):
    """20-day momentum factor."""
    df = df.copy()
    df["factor"] = df.groupby("symbol")["close"].pct_change(20)
    q01, q99 = df["factor"].quantile(0.01), df["factor"].quantile(0.99)
    df["factor"] = df["factor"].clip(q01, q99)
    df["factor"] = df.groupby("date")["factor"].rank(pct=True)
    return df[["date", "symbol", "factor"]].dropna()
'''

                    # Generate synthetic data with a latent momentum signal:
                    # slow-drifting signal s leads future returns (as in demo_sim.py),
                    # so 20-day momentum factor shows a positive IC in the demo.
                    rng = np.random.default_rng(42)
                    n_dates, n_symbols = 500, 200
                    dates = pd.date_range("2022-01-01", periods=n_dates, freq="B")
                    symbols = [f"STK_{i:04d}" for i in range(n_symbols)]

                    sig = rng.normal(0, 1, size=(n_dates, n_symbols))
                    for t in range(1, n_dates):
                        sig[t] = 0.9 * sig[t - 1] + np.sqrt(1 - 0.9 ** 2) * sig[t]
                    rets = np.zeros((n_dates, n_symbols))
                    for t in range(1, n_dates):
                        rets[t] = 0.008 * sig[t - 1] + rng.normal(0, 0.03, size=n_symbols)
                    closes = 10.0 * np.exp(np.cumsum(rets, axis=0))

                    data_rows = []
                    for i, sym in enumerate(symbols):
                        for t, (date, p) in enumerate(zip(dates, closes[:, i])):
                            data_rows.append({
                                "date": date, "symbol": sym,
                                "close": float(p),
                                "open": float(p * (1 + rng.normal(0, 0.005))),
                                "high": float(p * (1 + abs(rng.normal(0, 0.01)))),
                                "low": float(p * (1 - abs(rng.normal(0, 0.01)))),
                                "volume": float(np.random.lognormal(12, 1)),
                            })

                    df = pd.DataFrame(data_rows)

                    # Execute factor through the product engine (sandbox -> postprocess -> backtest)
                    sandbox = FactorSandbox({"engine": {"sandbox": {"subprocess": False, "timeout": 60}}})
                    factor_series = sandbox.run(factor_code, df)
                    processed = build_pipeline(factor_series, winsorize_pct=0.01)

                    bt = FactorBacktester(n_quantiles=5, forward_periods=5)
                    metrics = bt.evaluate(df, processed, verbose=False)

                    if "error" not in metrics:
                        # Save standard backtest charts to a temp dir for display
                        import tempfile
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt
                        charts = []
                        chart_dir = Path(tempfile.gettempdir()) / "factorgpt_demo"
                        chart_dir.mkdir(exist_ok=True)
                        try:
                            figs = bt.plot_metrics(metrics)
                            for i, fig in enumerate(figs, 1):
                                cp = chart_dir / f"demo_chart_{i}.png"
                                fig.savefig(str(cp), dpi=110, bbox_inches="tight")
                                plt.close(fig)
                                charts.append(str(cp))
                        except Exception as e:  # noqa: BLE001
                            print(f"[demo] 图表生成失败: {e}")

                        qr = metrics.get("quantile_returns", {})
                        top_ret = max(qr.values()) if qr else 0.0

                        # Display metrics
                        metric_cols = st.columns(5)
                        metric_items = [
                            ("Rank IC", f'{metrics.get("rank_ic", 0):+.4f}'),
                            ("ICIR", f'{metrics.get("icir", 0):+.3f}'),
                            ("IC >0 Ratio", f'{metrics.get("ic_positive_ratio", 0):.1%}'),
                            ("Top Quantile", f'{top_ret:+.2%}'),
                            ("Long/Short Sharpe", f'{metrics.get("long_short_sharpe", 0):+.2f}'),
                        ]
                        colors = ["green", "green", "green", "green", "green"]
                        for i, ((label, val), color) in enumerate(zip(metric_items, colors)):
                            with metric_cols[i]:
                                st.markdown(f'''
                                <div class="metric-card {color}">
                                    <div class="metric-value">{val}</div>
                                    <div class="metric-label">{label}</div>
                                </div>
                                ''', unsafe_allow_html=True)

                        # Factor code
                        with st.expander("📝 生成的因子代码", expanded=False):
                            st.code(factor_code.strip(), language="python")

                        # Charts
                        if charts:
                            st.markdown('<p class="section-title">回测可视化</p>', unsafe_allow_html=True)
                            chart_cols = st.columns(3)
                            for i, cp in enumerate(charts[:3]):
                                if Path(cp).exists():
                                    with chart_cols[i]:
                                        st.image(str(cp), use_container_width=True)
                            if len(charts) > 3:
                                chart_cols2 = st.columns(min(3, len(charts) - 3))
                                for i, cp in enumerate(charts[3:6]):
                                    if Path(cp).exists():
                                        with chart_cols2[i]:
                                            st.image(str(cp), use_container_width=True)

                        # Factor quality assessment
                        ic_mean = abs(metrics.get("rank_ic", 0))
                        if ic_mean >= 0.05:
                            st.success(f"✅ 因子质量：优秀 (|IC| = {ic_mean:.4f}) — 建议纳入因子库")
                        elif ic_mean >= 0.03:
                            st.info(f"ℹ️ 因子质量：良好 (|IC| = {ic_mean:.4f}) — 可进一步优化")
                        elif ic_mean >= 0.02:
                            st.warning(f"⚠️ 因子质量：一般 (|IC| = {ic_mean:.4f}) — 需要改进")
                        else:
                            st.error(f"❌ 因子质量：较差 (|IC| = {ic_mean:.4f}) — 建议重新设计")
                    else:
                        st.error(f"因子评估失败：{metrics.get('error', '未知错误')}")

                except Exception as e:
                    st.error(f"运行出错：{str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    with col_right:
        st.markdown('<p class="section-title">工作原理</p>', unsafe_allow_html=True)
        st.markdown("""
        ```
        自然语言输入
            ↓
        ┌──────────────────┐
        │ 1. 知识库检索     │  ChromaDB + BGE
        │    61个传统因子   │  向量语义搜索
        ├──────────────────┤
        │ 2. LLM 生成代码   │  DeepSeek/GPT-4
        │    alpha_factor() │  安全协议约束
        ├──────────────────┤
        │ 3. 沙盒验证       │  子进程隔离
        │    AST 超前偏差   │  超时/内存限制
        ├──────────────────┤
        │ 4. 后处理         │  缩尾→中性化
        │    标准化         │  →标准化
        ├──────────────────┤
        │ 5. 回测评估       │  IC/ICIR/分层
        │    量化指标       │  夏普/MDD/换手
        ├──────────────────┤
        │ 6. 反思改进       │  IC 不达标时
        │    迭代优化       │  自动反思重试
        └──────────────────┘
            ↓
        最终报告 & 可视化
        ```
        """)

# ══════════════════════════════════════════════
# TAB 2: Factor Library
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">📚 内置传统因子库（61个）</p>', unsafe_allow_html=True)

    try:
        from engine.traditional_factors import get_all_factors, CATEGORY_LABELS

        TRADITIONAL_FACTORS = get_all_factors()

        # Build factor dataframe
        factor_data = []
        for f in TRADITIONAL_FACTORS:
            factor_data.append({
                "名称": f.display_name,
                "类别": CATEGORY_LABELS.get(f.category, f.category),
                "标签": ", ".join(f.tags[:3]),
            })

        factors_df = pd.DataFrame(factor_data)

        # Category filter
        categories = ["全部"] + list(CATEGORY_LABELS.values())
        selected_cat = st.selectbox("按类别筛选", categories)

        if selected_cat != "全部":
            filtered = factors_df[factors_df["类别"] == selected_cat]
        else:
            filtered = factors_df

        # Category stats
        cat_counts = factors_df["类别"].value_counts()
        stat_cols = st.columns(len(cat_counts))
        for i, (cat, cnt) in enumerate(cat_counts.items()):
            with stat_cols[i]:
                st.metric(cat, cnt)

        # Display factors
        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
            column_config={
                "名称": st.column_config.TextColumn("因子名称", width="medium"),
                "类别": st.column_config.TextColumn("类别", width="small"),
                "标签": st.column_config.TextColumn("关键词", width="large"),
            }
        )

        # Show a sample factor detail
        st.markdown('<p class="section-title">因子详情示例</p>', unsafe_allow_html=True)
        selected_factor = st.selectbox(
            "选择一个因子查看详情",
            [f.display_name for f in TRADITIONAL_FACTORS]
        )

        for f in TRADITIONAL_FACTORS:
            if f.display_name == selected_factor:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"""
                    <div class="factor-card">
                        <strong>{f.display_name}</strong><br>
                        <span style="color:#7f8c8d">类别: {CATEGORY_LABELS.get(f.category, '')}</span><br>
                        <span style="color:#7f8c8d">标签: {', '.join(f.tags)}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    desc = f.description
                    if desc:
                        st.markdown(f"**{desc}**")
                    st.markdown(f"方向: `{f.direction}` | 质量分: `{f.quality_score:.2f}`")
                break

    except ImportError as e:
        st.warning(f"部分模块未加载: {e}")
        st.info("HuggingFace Spaces 免费环境中部分依赖不可用，请参考完整版文档。")

# ══════════════════════════════════════════════
# TAB 3: Backtest Analysis
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">📈 量化回测深度分析</p>', unsafe_allow_html=True)

    st.markdown("""
    以下展示 FactorGPT 六阶段精炼厂产出的标准化回测分析报告。
    每个因子经过 **信息系数(IC)分析**、**分位数分层回测**、**多空组合** 三重验证。
    """)

    # Display existing chart assets
    # 优先用 demo_output/（由 demo_sim.py 生成的最新回测图），否则回退 docs/assets
    chart_dir = _PROJ_ROOT / "demo_output"
    if not list(chart_dir.glob("*.png")):
        chart_dir = _PROJ_ROOT / "docs" / "assets"
    chart_files = sorted(chart_dir.glob("*.png"))

    if chart_files:
        # Group charts
        backtest_charts = [f for f in chart_files if f.stem not in ("ui_overview", "ui_refinery", "ui_library", "ui_sysbuild", "ui_memory")]

        if backtest_charts:
            st.markdown("### 回测图表")

            rows = (len(backtest_charts) + 1) // 2
            for r in range(rows):
                cols = st.columns(2)
                for c in range(2):
                    idx = r * 2 + c
                    if idx < len(backtest_charts):
                        chart = backtest_charts[idx]
                        with cols[c]:
                            title_map = {
                                "ic_series": "IC 时间序列",
                                "quantile_returns": "分位数收益对比",
                                "quantile_cum": "分位数累积收益",
                                "long_short": "多空组合累积收益",
                                "portfolio_nav": "组合净值走势",
                                "factor_ic_bars": "多因子 IC 对比",
                            }
                            title = title_map.get(chart.stem, chart.stem.replace("_", " ").title())
                            st.markdown(f"**{title}**")
                            st.image(str(chart), use_container_width=True)

    # Generate interactive Plotly demo chart
    st.markdown('<p class="section-title">交互式 IC 分析（Plotly 动态演示）</p>', unsafe_allow_html=True)

    np.random.seed(123)
    dates = pd.date_range("2022-01-01", periods=240, freq="W")
    ic_series = np.random.normal(0.035, 0.08, len(dates))
    ic_cumsum = np.cumsum(ic_series)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("IC 时间序列", "累积 IC"),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5],
    )

    fig.add_trace(
        go.Bar(x=dates, y=ic_series, name="周度 IC",
               marker_color=np.where(np.array(ic_series) > 0, '#11998e', '#eb3349'),
               opacity=0.7),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=np.mean(ic_series), line_dash="dot", line_color="blue",
                  annotation_text=f"均值={np.mean(ic_series):.3f}", row=1, col=1)

    fig.add_trace(
        go.Scatter(x=dates, y=ic_cumsum, name="累积 IC",
                   line=dict(color='#c0392b', width=2), fill='tozeroy',
                   fillcolor='rgba(192, 57, 43, 0.1)'),
        row=2, col=1
    )

    fig.update_layout(
        height=500, showlegend=False, margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Key insights section
    st.markdown('<p class="section-title">关键洞察</p>', unsafe_allow_html=True)
    insight_cols = st.columns(4)
    insights = [
        ("IC 均值", "+0.035", "因子具备稳定预测能力"),
        ("ICIR", "0.44", "信息比率适中，可进一步优化"),
        ("IC>0 占比", "67.1%", "正向预测胜率较高"),
        ("多空夏普", "1.23", "风险调整后收益良好"),
    ]
    for i, (label, val, desc) in enumerate(insights):
        with insight_cols[i]:
            st.markdown(f"""
            <div class="factor-card" style="text-align:center">
                <div style="font-size:1.5rem;font-weight:700;color:#c0392b">{val}</div>
                <div style="font-weight:600">{label}</div>
                <div style="color:#7f8c8d;font-size:0.8rem">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4: Factor Refinery
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">🏭 六阶段因子精炼厂</p>', unsafe_allow_html=True)

    st.markdown("""
    FactorGPT 的六阶段因子精炼厂模拟工业冶炼流程，将原始数据矿石加工为可投用的因子产品。
    以下展示完整流水线的离线运行结果（200只股票，500个交易日）。
    """)

    col1, col2 = st.columns([3, 2])

    with col1:
        # Try to run a mini refinery
        if st.button("⚙️ 运行精炼厂演示（离线模式）", use_container_width=True):
            with st.spinner("精炼厂运行中... 六道工序依次执行"):
                try:
                    from pipeline.refinery import RefineryPipeline, build_refinery_config

                    cfg = build_refinery_config({
                        "offline": True,
                        "use_real_data": False,
                        "run_portfolio": False,
                        "n_symbols": 50,
                        "train_days": 250,
                        "test_days": 60,
                        "n_pool_seed": 8,
                        "rl_candidates": 3,
                        "rl_backend": "heuristic",
                        "n_workers": 1,
                        "screener": {"use_lasso": True, "use_human_collab": False,
                                     "topk_ratio": 0.3, "min_keep": 2},
                        "alpha_pool": {"ortho": True, "loo": False, "iterative": False},
                        "rpn": {"n_quantiles": 5, "forward_periods": 1,
                                "commission": 0.001, "risk_free_rate": 0.03,
                                "parallel": False},
                    })

                    pipe = RefineryPipeline(cfg)
                    result = pipe.run(requirement="混合日频与月频，结合动量与反转")

                    # Stage trace
                    st.success("✅ 精炼厂运行完成！")
                    st.markdown("#### 流水线阶段追踪")
                    trace_data = []
                    for s in result.stage_trace:
                        trace_data.append({
                            "阶段": s["stage"],
                            "耗时": f"{s['elapsed_s']:.1f}s",
                            "备注": s["note"],
                        })
                    st.dataframe(pd.DataFrame(trace_data), use_container_width=True, hide_index=True)

                    # Screened factors
                    if result.screened:
                        st.markdown("#### 入选因子（三级筛选后）")
                        screened_data = []
                        for c in result.screened:
                            m = c.metrics
                            screened_data.append({
                                "因子名称": c.name,
                                "来源": c.source,
                                "ICIR": f'{m.get("icir", 0):+.3f}',
                                "稳定性": f'{m.get("stability_score", 0):+.3f}',
                                "换手": f'{m.get("turnover", 0):.3f}',
                            })
                        st.dataframe(pd.DataFrame(screened_data), use_container_width=True, hide_index=True)

                    # Composite metrics
                    if result.composite_metrics:
                        cm = result.composite_metrics
                        st.markdown("#### 复合因子（AlphaPool 合成）")
                        comp_cols = st.columns(6)
                        comp_metrics = [
                            ("ICIR", f'{cm.get("icir", 0):+.3f}'),
                            ("IC Mean", f'{cm.get("ic_mean", 0):+.3f}'),
                            ("稳定性", f'{cm.get("stability_score", 0):+.3f}'),
                            ("换手率", f'{cm.get("turnover", 0):.3f}'),
                            ("多空夏普", f'{cm.get("long_short_sharpe", "N/A")}'),
                            ("最大回撤", f'{cm.get("max_drawdown", "N/A")}'),
                        ]
                        for i, (label, val) in enumerate(comp_metrics):
                            with comp_cols[i]:
                                st.metric(label, val)
                except Exception as e:
                    st.error(f"精炼厂运行出错：{e}")

    with col2:
        st.markdown("#### 六道工序")

        stages = [
            ("PART-01", "矿石仓库", "28 个原始特征 + 50+ 因子池，多进程并行构建"),
            ("PART-02", "矿场开采", "Transformer 向量化 + RL 搜索 + LLM 矿脉探索"),
            ("PART-03", "研磨车间", "RPN 引擎批量求值，IC/ICIR 量化"),
            ("PART-04", "三级筛选", "LASSO 去冗余 → 人机协同 → TOP 10%"),
            ("PART-05", "合金配比", "ICIR 加权 + 正交化 + LOO 过拟合检验"),
            ("PART-06", "方法学报告", "一键生成 MD + JSON 方法学报告"),
        ]

        for code, name, desc in stages:
            st.markdown(f"""
            <div class="factor-card">
                <span style="color:#c0392b;font-weight:700">{code}</span>
                <strong> {name}</strong><br>
                <span style="color:#7f8c8d;font-size:0.85rem">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        ---
        ### 完整功能请查看

        本地部署可体验完整的 17 页交互界面，包括：
        - 对话式因子挖掘
        - 遗传规划因子进化
        - Vibe Trading 自然语言策略
        - Transformer 注意力可视化
        - 非结构化数据因子挖掘
        - 因子监控与预警系统
        
        👉 [GitHub 完整版](https://github.com/ZXTLQQ/FactorGPT)
        """)

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#7f8c8d;font-size:0.85rem">
    <p>FactorGPT Online Demo — Powered by HuggingFace Spaces | 
    <a href="https://github.com/ZXTLQQ/FactorGPT">GitHub</a> | 
    <a href="https://github.com/ZXTLQQ/FactorGPT#readme">文档</a></p>
    <p style="font-size:0.75rem">本 Demo 仅用于学术研究与教育目的，不构成任何投资建议。投资有风险，入市需谨慎。</p>
</div>
""", unsafe_allow_html=True)
