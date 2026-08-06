"""
FactorGPT - AI-Driven Factor Mining & Quantitative Analysis Platform
HuggingFace Spaces Demo
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(
    page_title="FactorGPT - AI Factor Mining",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4f46e5;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1f2937;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    .feature-tag {
        display: inline-block;
        background: #ede9fe;
        color: #6d28d9;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# === Sidebar ===
with st.sidebar:
    st.markdown("### FactorGPT - AI Factor Mining")
    st.markdown("---")

    st.markdown("**Configuration**")
    model_choice = st.selectbox(
        "LLM Backend",
        ["DeepSeek-V3", "GPT-4o", "Claude-3.5-Sonnet", "Qwen-72B", "Ollama (Local)"],
        index=0
    )
    market = st.selectbox("Market", ["A-Share (China)", "US Stocks", "HK Stocks"], index=0)

    st.markdown("**Date Range**")
    start_date = st.date_input("Start", value=datetime.now() - timedelta(days=365))
    end_date = st.date_input("End", value=datetime.now())

    st.markdown("**Strategy**")
    strategy = st.selectbox(
        "Factor Type",
        ["Multi-Factor Alpha", "Momentum", "Mean Reversion", "Quality Value", "Smart Beta"]
    )

    st.markdown("---")
    st.markdown("**Advanced**")
    max_factors = st.slider("Max Factors", 5, 50, 20)
    ic_threshold = st.slider("IC Threshold", 0.01, 0.10, 0.03, step=0.01)

    st.markdown("---")
    st.button("Run Factor Mining", type="primary", use_container_width=True)
    st.caption("Demo mode — results are simulated")

# === Main Content ===
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<p class="main-header">FactorGPT</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Driven Quantitative Factor Discovery &amp; Analysis Platform</p>', unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: right; padding-top: 1rem;">
        <a href="https://github.com/ZXTLQQ/FactorGPT" target="_blank">
            <img src="https://img.shields.io/github/stars/ZXTLQQ/FactorGPT?style=social" alt="GitHub stars">
        </a>
    </div>
    """, unsafe_allow_html=True)

# === Key Metrics ===
st.markdown('<p class="section-title">Platform Overview</p>', unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
metrics_data = [
    ("69", "Python Modules", "#6366f1"),
    ("1,200+", "Built-in Factors", "#8b5cf6"),
    ("7", "Data Sources", "#a855f7"),
    ("3", "LLM Backends", "#c084fc"),
    ("17", "Topics", "#e879f9"),
]

for col, (value, label, color) in zip([m1, m2, m3, m4, m5], metrics_data):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {color}">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

# === Sample Factor Performance Chart ===
st.markdown('<p class="section-title">Factor Performance (Demo)</p>', unsafe_allow_html=True)

np.random.seed(42)
dates = pd.date_range(end=datetime.now(), periods=60, freq='B')
factors = ['Alpha_001', 'Alpha_002', 'Alpha_003', 'Alpha_004']
data = {}
for f in factors:
    base = np.random.randn() * 0.3
    data[f] = np.cumsum(np.random.randn(60) * 0.02 + base * 0.01)

fig = go.Figure()
colors = ['#6366f1', '#8b5cf6', '#a855f7', '#c084fc']
for f, c in zip(factors, colors):
    fig.add_trace(go.Scatter(
        x=dates, y=data[f], mode='lines', name=f,
        line=dict(color=c, width=2), hovertemplate='%{y:.3f}'
    ))

fig.update_layout(
    template='plotly_white',
    height=400,
    margin=dict(l=20, r=20, t=10, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    yaxis_title="Cumulative Return",
    hovermode='x unified'
)
st.plotly_chart(fig, use_container_width=True)

# === Architecture & Features ===
st.markdown('<p class="section-title">System Architecture</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Pipeline", "Components", "Tech Stack"])

with tab1:
    st.markdown("""
    ### Factor Mining Pipeline
    
    1. **Data Collection** — Multi-source financial data gathering (mootdx, EastMoney, Sina, Tencent)
    2. **Factor Extraction** — LLM-powered factor idea generation and expression construction
    3. **Factor Evaluation** — IC analysis, IR computation, group backtesting
    4. **Factor Selection** — Multi-period stability check, correlation filtering
    5. **Portfolio Construction** — Weight optimization and live deployment
    """)
    
    stages = ['Data', 'Extract', 'Evaluate', 'Select', 'Deploy']
    values = [100, 85, 70, 50, 35]
    fig2 = go.Figure(go.Funnel(
        y=stages, x=values,
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(color=["#6366f1", "#7c3aed", "#8b5cf6", "#a78bfa", "#c4b5fd"])
    ))
    fig2.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=20))
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.markdown("""
    ### Core Components
    
    | Component | Description | Implementation |
    |-----------|-------------|----------------|
    | Factor Engine | Expression parsing & computation | Python + Numexpr |
    | LLM Agent | Factor idea generation | LangGraph + DeepSeek |
    | Backtest Engine | Multi-period factor evaluation | Custom + Alphalens |
    | Data Layer | 7-source data integration | Async + Caching |
    | Web UI | Interactive analysis dashboards | Streamlit + Plotly |
    """)

with tab3:
    tags = [
        "Python", "LangGraph", "DeepSeek", "Ollama", "Streamlit",
        "mootdx", "Numexpr", "Pandas", "Plotly", "Alphalens",
        "Quantitative Finance", "Alpha Mining", "Multi-Factor Model",
        "A-Share", "Backtesting", "Genetic Programming", "NLP"
    ]
    tag_html = " ".join([f'<span class="feature-tag">{t}</span>' for t in tags])
    st.markdown(f'<div style="line-height: 2.5;">{tag_html}</div>', unsafe_allow_html=True)

# === Factor Analysis Demo ===
st.markdown('<p class="section-title">IC Analysis Heatmap (Demo)</p>', unsafe_allow_html=True)

periods = ['1D', '5D', '10D', '20D']
factors_list = ['Alpha_001', 'Alpha_002', 'Alpha_003', 'Alpha_004', 'Alpha_005',
                'Alpha_006', 'Alpha_007', 'Alpha_008']

ic_data = np.random.randn(len(factors_list), len(periods)) * 0.1 + 0.03
ic_df = pd.DataFrame(ic_data, index=factors_list, columns=periods)

fig3 = px.imshow(
    ic_df, text_auto='.3f', aspect='auto',
    color_continuous_scale='RdBu_r', zmin=-0.1, zmax=0.1,
    labels=dict(x="Holding Period", y="Factor", color="IC")
)
fig3.update_layout(height=400, margin=dict(l=20, r=20, t=10, b=20))
st.plotly_chart(fig3, use_container_width=True)

# === Footer ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9ca3af; font-size: 0.85rem; padding: 1rem 0;">
    <p><strong>FactorGPT</strong> — AI-Driven Quantitative Factor Mining Platform</p>
    <p>
        <a href="https://github.com/ZXTLQQ/FactorGPT" target="_blank">GitHub</a> · 
        <a href="https://huggingface.co/ZxTLQQ/FactorGPT" target="_blank">HuggingFace</a> · 
        <a href="https://github.com/ZXTLQQ/FactorGPT/blob/main/README.md" target="_blank">Documentation</a>
    </p>
    <p style="margin-top: 0.5rem;">This is a demo version. For full functionality, please deploy locally.</p>
</div>
""", unsafe_allow_html=True)
