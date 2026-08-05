"""FactorGPT 红白主题设计系统（src/ui/theme.py）。

统一整个 Streamlit 应用的视觉语言：中国红 + 纯白 + 克制的灰阶。
对外暴露三类能力：

1. ``inject_theme()``  —— 一次性注入全局 CSS（含侧边栏、按钮、表格、指标卡等）。
2. 结构化组件      —— ``page_header`` / ``section`` / ``kpi_row`` / ``badge``
   ``insight_box`` / ``factor_chip`` / ``divider`` 等，保证各页面风格一致。
3. 图表主题        —— ``PALETTE`` 与 ``style_fig()``，让 Plotly 图与界面同色系。
"""

from __future__ import annotations

import html
from typing import Any, Dict, Iterable, List, Optional, Sequence

import streamlit as st

# ---------------------------------------------------------------------------
# 设计令牌
# ---------------------------------------------------------------------------
RED = "#C8102E"          # 主色：中国红
RED_DEEP = "#8E0F22"     # 深红：标题/强调
RED_BRIGHT = "#E8384F"   # 亮红：高亮/悬停
RED_SOFT = "#FDECEE"     # 淡红：卡片底色
RED_LINE = "#F3D2D7"     # 淡红：分隔线/边框

INK = "#1B1F24"          # 主文字
INK_SUB = "#5A6472"      # 次级文字
INK_MUTED = "#8A94A6"    # 弱化文字
LINE = "#EBEEF3"         # 中性边框
BG = "#FFFFFF"
BG_SOFT = "#FAFBFC"

GREEN = "#1E9E6A"        # 正向
AMBER = "#E8A33D"        # 提示
BLUE = "#2C6FBB"         # 中性对照
PURPLE = "#7A4FBF"

# Plotly 分类色板：以红为主，辅以低饱和对照色
PALETTE: List[str] = [
    RED, "#E8384F", "#F2707F", "#8E0F22", "#D46A6A",
    "#2C6FBB", "#1E9E6A", "#E8A33D", "#7A4FBF", "#5A6472",
]

# 连续色标：白 -> 红
COLORSCALE_RED = [
    [0.0, "#FFFFFF"], [0.25, "#FBDDE1"], [0.5, "#F09AA6"],
    [0.75, "#DC4459"], [1.0, RED_DEEP],
]
# 发散色标：蓝 -> 白 -> 红（相关性矩阵用）
COLORSCALE_DIVERGING = [
    [0.0, "#2C6FBB"], [0.25, "#9BBEDE"], [0.5, "#FFFFFF"],
    [0.75, "#EE9AA6"], [1.0, RED_DEEP],
]

FONT_STACK = (
    '"PingFang SC","Microsoft YaHei","Hiragino Sans GB",'
    '"Helvetica Neue",Arial,sans-serif'
)


# ---------------------------------------------------------------------------
# 全局 CSS
# ---------------------------------------------------------------------------
_CSS = f"""
<style>
:root {{
  --fg-red: {RED};
  --fg-red-deep: {RED_DEEP};
  --fg-red-bright: {RED_BRIGHT};
  --fg-red-soft: {RED_SOFT};
  --fg-red-line: {RED_LINE};
  --fg-ink: {INK};
  --fg-ink-sub: {INK_SUB};
  --fg-ink-muted: {INK_MUTED};
  --fg-line: {LINE};
  --fg-bg-soft: {BG_SOFT};
  --fg-radius: 10px;
  --fg-shadow: 0 1px 3px rgba(27,31,36,.06), 0 4px 16px rgba(200,16,46,.05);
}}

html, body, [class*="css"], .stApp {{
  font-family: {FONT_STACK};
  color: var(--fg-ink);
}}
.stApp {{ background: {BG}; }}

/* 收紧主区留白，让内容更饱满 */
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }}

/* ---------- 标题 ---------- */
h1, h2, h3, h4 {{ color: var(--fg-ink); font-weight: 700; letter-spacing: -.2px; }}
h1 {{ font-size: 1.6rem; }}
h2 {{ font-size: 1.25rem; }}
h3 {{ font-size: 1.05rem; }}

/* ---------- 侧边栏（复合式目录） ---------- */
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #FFFFFF 0%, {RED_SOFT} 100%);
  border-right: 1px solid var(--fg-red-line);
}}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}

.fg-brand {{
  display: flex; align-items: center; gap: 10px;
  padding: 4px 2px 14px 2px; margin-bottom: 6px;
  border-bottom: 1px solid var(--fg-red-line);
}}
.fg-brand-mark {{
  width: 34px; height: 34px; border-radius: 9px; flex: 0 0 34px;
  background: linear-gradient(135deg, {RED} 0%, {RED_DEEP} 100%);
  color: #fff; font-weight: 800; font-size: 15px;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 2px 8px rgba(200,16,46,.28);
}}
.fg-brand-text b {{ display: block; font-size: 15px; line-height: 1.2; color: var(--fg-red-deep); }}
.fg-brand-text span {{ font-size: 11px; color: var(--fg-ink-muted); letter-spacing: .4px; }}

/* 目录分组标题 */
.fg-navgroup {{
  font-size: 11px; font-weight: 700; letter-spacing: 1.2px;
  color: var(--fg-ink-muted); text-transform: uppercase;
  margin: 14px 0 6px 4px;
}}

/* 侧边栏按钮 = 二级目录项 */
section[data-testid="stSidebar"] .stButton > button {{
  width: 100%; text-align: left; justify-content: flex-start;
  background: transparent; color: var(--fg-ink-sub);
  border: 1px solid transparent; border-radius: 8px;
  padding: .38rem .6rem; font-size: 13.5px; font-weight: 500;
  margin-bottom: 2px; transition: all .15s ease;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
  background: #fff; color: var(--fg-red); border-color: var(--fg-red-line);
  transform: translateX(2px);
}}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
  background: linear-gradient(135deg, {RED} 0%, {RED_DEEP} 100%);
  color: #fff; font-weight: 600; border-color: transparent;
  box-shadow: 0 2px 8px rgba(200,16,46,.25);
}}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
  color: #fff; transform: translateX(2px);
}}

/* 侧边栏折叠面板 */
section[data-testid="stSidebar"] div[data-testid="stExpander"] {{
  border: 1px solid var(--fg-red-line); border-radius: 9px;
  background: rgba(255,255,255,.72); margin-bottom: 6px;
}}
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary {{
  font-size: 13px; font-weight: 650; color: var(--fg-red-deep); padding: .35rem .6rem;
}}
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary:hover {{ color: var(--fg-red); }}

/* ---------- 主区按钮 ---------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
  border-radius: 8px; font-weight: 600; font-size: 13.5px;
  border: 1px solid var(--fg-red-line); color: var(--fg-red);
  background: #fff; transition: all .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {{
  border-color: var(--fg-red); background: var(--fg-red-soft); color: var(--fg-red-deep);
}}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
  background: linear-gradient(135deg, {RED} 0%, {RED_DEEP} 100%);
  color: #fff; border-color: transparent;
  box-shadow: 0 2px 10px rgba(200,16,46,.25);
}}
.stButton > button[kind="primary"]:hover {{ filter: brightness(1.06); color: #fff; }}

/* ---------- 页头 ---------- */
.fg-page-head {{
  border-left: 4px solid var(--fg-red);
  padding: 2px 0 2px 14px; margin: 0 0 18px 0;
}}
.fg-page-head h1 {{ margin: 0; font-size: 1.45rem; color: var(--fg-ink); }}
.fg-page-head p {{ margin: 4px 0 0 0; font-size: 13px; color: var(--fg-ink-sub); }}

/* ---------- 区块标题 ---------- */
.fg-section {{ margin: 22px 0 10px 0; }}
.fg-section .t {{
  font-size: 15px; font-weight: 700; color: var(--fg-ink);
  display: flex; align-items: center; gap: 8px;
}}
.fg-section .t::before {{
  content: ""; width: 3px; height: 14px; border-radius: 2px;
  background: linear-gradient(180deg, {RED} 0%, {RED_DEEP} 100%);
}}
.fg-section .d {{ font-size: 12.5px; color: var(--fg-ink-muted); margin: 4px 0 0 11px; }}

/* ---------- KPI 卡 ---------- */
.fg-kpis {{ display: grid; gap: 12px; margin: 6px 0 4px 0; }}
.fg-kpi {{
  background: #fff; border: 1px solid var(--fg-line);
  border-top: 3px solid var(--fg-red);
  border-radius: var(--fg-radius); padding: 14px 16px;
  box-shadow: var(--fg-shadow); transition: all .18s ease;
}}
.fg-kpi:hover {{ transform: translateY(-2px); border-color: var(--fg-red-line); }}
.fg-kpi .lab {{ font-size: 12px; color: var(--fg-ink-muted); font-weight: 500; }}
.fg-kpi .val {{
  font-size: 25px; font-weight: 800; color: var(--fg-red-deep);
  line-height: 1.25; margin-top: 3px;
  font-variant-numeric: tabular-nums;
}}
.fg-kpi .sub {{ font-size: 11.5px; color: var(--fg-ink-muted); margin-top: 3px; }}
.fg-kpi.pos {{ border-top-color: {GREEN}; }} .fg-kpi.pos .val {{ color: {GREEN}; }}
.fg-kpi.warn {{ border-top-color: {AMBER}; }} .fg-kpi.warn .val {{ color: #B87A17; }}
.fg-kpi.neutral {{ border-top-color: {BLUE}; }} .fg-kpi.neutral .val {{ color: {BLUE}; }}
.fg-kpi.ink {{ border-top-color: {INK_SUB}; }} .fg-kpi.ink .val {{ color: var(--fg-ink); }}

/* ---------- 通用卡片 ---------- */
.fg-card {{
  background: #fff; border: 1px solid var(--fg-line); border-radius: var(--fg-radius);
  padding: 16px 18px; box-shadow: var(--fg-shadow); margin-bottom: 12px;
}}
.fg-card h4 {{ margin: 0 0 8px 0; font-size: 14px; color: var(--fg-red-deep); }}

/* ---------- 洞察 / 提示条 ---------- */
.fg-insight {{
  background: var(--fg-red-soft); border-left: 3px solid var(--fg-red);
  border-radius: 0 8px 8px 0; padding: 11px 14px; margin: 8px 0;
  font-size: 13px; color: #6B2029; line-height: 1.7;
}}
.fg-insight b {{ color: var(--fg-red-deep); }}
.fg-insight.info {{ background: #EEF4FB; border-left-color: {BLUE}; color: #1F4D7A; }}
.fg-insight.ok   {{ background: #EAF7F1; border-left-color: {GREEN}; color: #146645; }}
.fg-insight.warn {{ background: #FDF5E7; border-left-color: {AMBER}; color: #7A5410; }}

/* ---------- 徽章 ---------- */
.fg-badge {{
  display: inline-block; padding: 2px 9px; border-radius: 999px;
  font-size: 11px; font-weight: 650; line-height: 1.7;
  background: var(--fg-red-soft); color: var(--fg-red-deep);
  border: 1px solid var(--fg-red-line); margin-right: 5px;
}}
.fg-badge.gray {{ background: #F3F5F8; color: {INK_SUB}; border-color: {LINE}; }}
.fg-badge.green {{ background: #EAF7F1; color: #146645; border-color: #BFE6D5; }}
.fg-badge.blue {{ background: #EEF4FB; color: #1F4D7A; border-color: #C6DCF0; }}
.fg-badge.amber {{ background: #FDF5E7; color: #7A5410; border-color: #F2DFB8; }}
.fg-badge.solid {{ background: {RED}; color: #fff; border-color: {RED}; }}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{
  gap: 2px; border-bottom: 1px solid var(--fg-line); background: transparent;
}}
.stTabs [data-baseweb="tab"] {{
  height: 38px; padding: 0 15px; background: transparent;
  border-radius: 8px 8px 0 0; font-size: 13.5px; font-weight: 600; color: var(--fg-ink-sub);
}}
.stTabs [aria-selected="true"] {{
  background: var(--fg-red-soft); color: var(--fg-red-deep);
  border-bottom: 2px solid var(--fg-red);
}}

/* ---------- 表格 ---------- */
div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {{
  border: 1px solid var(--fg-line); border-radius: var(--fg-radius); overflow: hidden;
}}

/* ---------- 输入控件 ---------- */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
div[data-baseweb="select"] > div {{ border-radius: 8px !important; }}
.stTextInput input:focus, .stTextArea textarea:focus {{
  border-color: var(--fg-red) !important; box-shadow: 0 0 0 2px rgba(200,16,46,.10) !important;
}}
div[data-baseweb="tag"] {{ background: {RED} !important; }}

/* 滑块 / 进度条 */
div[data-testid="stSlider"] div[role="slider"] {{ background: {RED} !important; }}
.stProgress > div > div > div > div {{ background: linear-gradient(90deg, {RED}, {RED_BRIGHT}); }}

/* 主区折叠面板 */
div[data-testid="stExpander"] {{
  border: 1px solid var(--fg-line); border-radius: var(--fg-radius); background: #fff;
}}
div[data-testid="stExpander"] summary {{ font-size: 13.5px; font-weight: 600; color: var(--fg-ink); }}
div[data-testid="stExpander"] summary:hover {{ color: var(--fg-red); }}

/* 指标组件 */
div[data-testid="stMetricValue"] {{ color: var(--fg-red-deep); font-weight: 700; }}

/* 分隔线 */
hr {{ border-color: var(--fg-line); }}

/* 页脚 */
.fg-foot {{
  margin-top: 26px; padding-top: 14px; border-top: 1px solid var(--fg-line);
  font-size: 11.5px; color: var(--fg-ink-muted); line-height: 1.8;
}}
.fg-foot b {{ color: var(--fg-red-deep); }}

/* 空态 */
.fg-empty {{
  border: 1px dashed var(--fg-red-line); border-radius: var(--fg-radius);
  background: var(--fg-bg-soft); padding: 30px 18px; text-align: center;
}}
.fg-empty .i {{ font-size: 26px; }}
.fg-empty .t {{ font-size: 14px; font-weight: 650; color: var(--fg-ink); margin-top: 6px; }}
.fg-empty .d {{ font-size: 12.5px; color: var(--fg-ink-muted); margin-top: 4px; }}

/* 因子标签 */
.fg-chip {{
  display: inline-flex; align-items: center; gap: 6px;
  background: #fff; border: 1px solid var(--fg-red-line); border-radius: 999px;
  padding: 3px 11px; font-size: 12px; color: var(--fg-ink); margin: 0 5px 5px 0;
}}
.fg-chip i {{ font-style: normal; color: var(--fg-red); font-weight: 700; }}

/* 数据流水线步骤条 */
.fg-steps {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 10px 0; }}
.fg-step {{
  flex: 1 1 150px; background: #fff; border: 1px solid var(--fg-line);
  border-radius: 8px; padding: 9px 12px;
}}
.fg-step .n {{ font-size: 10.5px; color: var(--fg-red); font-weight: 800; letter-spacing: .8px; }}
.fg-step .t {{ font-size: 13px; font-weight: 650; margin-top: 2px; }}
.fg-step .d {{ font-size: 11.5px; color: var(--fg-ink-muted); margin-top: 2px; }}
.fg-step.done {{ background: var(--fg-red-soft); border-color: var(--fg-red-line); }}
</style>
"""


def inject_theme() -> None:
    """注入全局样式（每次脚本重跑调用一次即可）。"""
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 结构化组件
# ---------------------------------------------------------------------------
def _esc(text: Any) -> str:
    return html.escape(str(text))


def brand(title: str = "FactorGPT", subtitle: str = "QUANT FACTOR STUDIO", mark: str = "F") -> None:
    """侧边栏品牌区。"""
    st.markdown(
        f"""<div class="fg-brand">
              <div class="fg-brand-mark">{_esc(mark)}</div>
              <div class="fg-brand-text"><b>{_esc(title)}</b><span>{_esc(subtitle)}</span></div>
            </div>""",
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "", icon: str = "") -> None:
    """页面主标题区。"""
    head = f"{icon} {title}".strip()
    sub = f"<p>{_esc(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f'<div class="fg-page-head"><h1>{_esc(head)}</h1>{sub}</div>',
        unsafe_allow_html=True,
    )


def section(title: str, desc: str = "") -> None:
    """区块小标题。"""
    d = f'<div class="d">{_esc(desc)}</div>' if desc else ""
    st.markdown(
        f'<div class="fg-section"><div class="t">{_esc(title)}</div>{d}</div>',
        unsafe_allow_html=True,
    )


def kpi_row(items: Sequence[Dict[str, Any]], columns: Optional[int] = None) -> None:
    """一行 KPI 卡片。

    Args:
        items: 每项支持 ``label`` / ``value`` / ``sub`` / ``tone``；
               tone 取 ``red``（默认）/ ``pos`` / ``warn`` / ``neutral`` / ``ink``。
        columns: 每行列数，默认按条目数自适应。
    """
    if not items:
        return
    n = columns or len(items)
    cards = []
    for it in items:
        tone = it.get("tone", "red")
        tone_cls = "" if tone == "red" else f" {tone}"
        sub = f'<div class="sub">{_esc(it.get("sub", ""))}</div>' if it.get("sub") else ""
        cards.append(
            f'<div class="fg-kpi{tone_cls}">'
            f'<div class="lab">{_esc(it.get("label", ""))}</div>'
            f'<div class="val">{_esc(it.get("value", "-"))}</div>{sub}</div>'
        )
    st.markdown(
        f'<div class="fg-kpis" style="grid-template-columns:repeat({n},minmax(0,1fr))">'
        + "".join(cards)
        + "</div>",
        unsafe_allow_html=True,
    )


def insight(text: str, tone: str = "red") -> None:
    """洞察 / 提示条。``tone``: red | info | ok | warn。允许内嵌 <b> 标签。"""
    cls = "" if tone == "red" else f" {tone}"
    st.markdown(f'<div class="fg-insight{cls}">{text}</div>', unsafe_allow_html=True)


def badge(text: str, tone: str = "red") -> str:
    """返回徽章 HTML 片段（供拼接使用）。"""
    cls = "" if tone == "red" else f" {tone}"
    return f'<span class="fg-badge{cls}">{_esc(text)}</span>'


def badges(items: Iterable[str], tone: str = "red") -> None:
    """直接渲染一组徽章。"""
    st.markdown("".join(badge(i, tone) for i in items), unsafe_allow_html=True)


def chips(items: Iterable[Any]) -> None:
    """渲染因子标签组。每项可为字符串或 ``(名称, 说明)`` 二元组。"""
    parts = []
    for it in items:
        if isinstance(it, (tuple, list)) and len(it) >= 2:
            parts.append(f'<span class="fg-chip"><i>{_esc(it[0])}</i>{_esc(it[1])}</span>')
        else:
            parts.append(f'<span class="fg-chip">{_esc(it)}</span>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def empty_state(title: str, desc: str = "", icon: str = "○") -> None:
    """空状态占位。"""
    st.markdown(
        f'<div class="fg-empty"><div class="i">{_esc(icon)}</div>'
        f'<div class="t">{_esc(title)}</div><div class="d">{_esc(desc)}</div></div>',
        unsafe_allow_html=True,
    )


def steps(items: Sequence[Dict[str, Any]]) -> None:
    """流程步骤条。每项 ``{"n": "01", "title": ..., "desc": ..., "done": bool}``。"""
    parts = []
    for it in items:
        cls = " done" if it.get("done") else ""
        parts.append(
            f'<div class="fg-step{cls}"><div class="n">{_esc(it.get("n", ""))}</div>'
            f'<div class="t">{_esc(it.get("title", ""))}</div>'
            f'<div class="d">{_esc(it.get("desc", ""))}</div></div>'
        )
    st.markdown(f'<div class="fg-steps">{"".join(parts)}</div>', unsafe_allow_html=True)


def footer(text: str) -> None:
    """页脚说明。"""
    st.markdown(f'<div class="fg-foot">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 图表主题
# ---------------------------------------------------------------------------
def style_fig(fig, height: Optional[int] = None, legend: bool = True,
              title: str = "", margin: Optional[Dict[str, int]] = None):
    """把 Plotly 图统一到红白主题上。

    Args:
        fig: plotly Figure。
        height: 图高（像素）。
        legend: 是否显示图例。
        title: 图标题（留空则不显示）。
        margin: 自定义边距。
    """
    fig.update_layout(
        template="plotly_white",
        colorway=PALETTE,
        font=dict(family=FONT_STACK, size=12, color=INK),
        title=dict(text=title, font=dict(size=14, color=RED_DEEP), x=0.01, xanchor="left")
        if title
        else None,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=margin or dict(l=48, r=24, t=44 if title else 22, b=40),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=RED_LINE,
                        font=dict(family=FONT_STACK, size=12, color=INK)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    font=dict(size=11)),
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(showgrid=False, linecolor=LINE, zeroline=False,
                     tickfont=dict(size=11, color=INK_SUB))
    fig.update_yaxes(showgrid=True, gridcolor="#F2F4F7", linecolor=LINE, zeroline=False,
                     tickfont=dict(size=11, color=INK_SUB))
    return fig
