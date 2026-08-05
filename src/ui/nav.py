"""复合式二级目录导航（src/ui/nav.py）。

把原先扁平的 17 个页面按业务域整合为 6 个一级分组、二级页面挂在分组下，
形成「分组 → 页面」的树形目录。当前页面会持久化到 SQLite，
下次打开应用自动回到上次所在的位置。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from . import theme

# ---------------------------------------------------------------------------
# 目录结构
# ---------------------------------------------------------------------------
# key      : 页面唯一标识，同时作为 render 函数注册表的键
# label    : 目录中显示的名称
# icon     : 前缀图标
# title    : 页面大标题（缺省用 label）
# desc     : 页面副标题
NAV_GROUPS: List[Dict[str, Any]] = [
    {
        "group": "工作台",
        "icon": "◈",
        "items": [
            {"key": "overview", "label": "系统概览", "icon": "🏠",
             "title": "系统概览", "desc": "研究资产总览与快捷入口"},
            {"key": "memory", "label": "操作记忆", "icon": "🗂",
             "title": "操作记忆中心", "desc": "本地数据库中的操作时间线、挖掘沉淀与现场恢复"},
        ],
    },
    {
        "group": "因子挖掘",
        "icon": "⚗",
        "items": [
            {"key": "agent", "label": "智能挖掘 Agent", "icon": "🤖",
             "title": "智能挖掘 Agent", "desc": "单次需求驱动的检索 / 生成 / 回测 / 反思闭环"},
            {"key": "chat", "label": "对话式挖掘", "icon": "💬",
             "title": "对话式因子挖掘", "desc": "多轮对话逐步澄清需求并产出因子，历史自动落盘"},
            {"key": "refinery", "label": "因子精炼厂", "icon": "🏭",
             "title": "因子精炼厂", "desc": "RL + RAG + Transformer 复合因子流水线"},
            {"key": "gp", "label": "遗传规划挖掘", "icon": "🧬",
             "title": "遗传规划因子挖掘", "desc": "基于因子库种子的批量变体生成与筛选"},
            {"key": "vibe", "label": "Vibe Trading", "icon": "🚀",
             "title": "Vibe Trading", "desc": "自然语言驱动的策略构思与快速验证"},
        ],
    },
    {
        "group": "因子体系",
        "icon": "🧱",
        "items": [
            {"key": "sys_build", "label": "体系搭建", "icon": "🧱",
             "title": "因子体系搭建", "desc": "从挖掘产出与系统因子库中挑选因子，配置维度与权重，组装成可回测的因子体系"},
            {"key": "sys_analysis", "label": "体系回测分析", "icon": "📊",
             "title": "因子体系分析仪表盘", "desc": "对已保存体系执行合成回测，输出 IC / ICIR / 分层 / 相关性 / 主成分全景诊断"},
            {"key": "library", "label": "系统因子库", "icon": "📚",
             "title": "系统因子库", "desc": "五大类预置传统因子的检索与代码查看"},
            {"key": "monitor", "label": "因子监控", "icon": "📡",
             "title": "因子监控", "desc": "线上因子的表现跟踪与衰减预警"},
        ],
    },
    {
        "group": "数据中心",
        "icon": "📈",
        "items": [
            {"key": "market", "label": "行情中心", "icon": "📈",
             "title": "行情中心", "desc": "指数 / 成分股行情、个股 K 线与资讯研报"},
            {"key": "futures", "label": "期货 & 期权", "icon": "🛢",
             "title": "期货 & 期权", "desc": "期货主力实时、期货/期权 K 线与合约链"},
            {"key": "funds", "label": "基金行情", "icon": "💰",
             "title": "基金行情", "desc": "ETF / LOF 实时、净值走势与开放基金排行"},
            {"key": "bonds", "label": "债券 / 外汇", "icon": "🪙",
             "title": "债券 / 外汇", "desc": "可转债、外汇牌价与上海金基准价"},
        ],
    },
    {
        "group": "智能分析",
        "icon": "🧠",
        "items": [
            {"key": "unstructured", "label": "非结构化数据", "icon": "📄",
             "title": "非结构化数据分析", "desc": "研报 / 公告 / 新闻的解析与因子线索抽取"},
            {"key": "transformer", "label": "Transformer 分析", "icon": "🧠",
             "title": "Transformer 耦合分析", "desc": "深度模型对因子表达的耦合与增强"},
            {"key": "kb", "label": "知识库", "icon": "📕",
             "title": "研究知识库", "desc": "RAG 语料检索与文档上传"},
        ],
    },
    {
        "group": "系统",
        "icon": "⚙",
        "items": [
            {"key": "delivery", "label": "产品交付", "icon": "📦",
             "title": "产品交付", "desc": "策略成果打包与交付物生成"},
            {"key": "config", "label": "运行配置", "icon": "⚙️",
             "title": "运行配置", "desc": "模型、向量库与数据源等运行时参数"},
        ],
    },
]

DEFAULT_PAGE = "overview"
_STATE_KEY = "fg_active_page"


# ---------------------------------------------------------------------------
# 查询helper
# ---------------------------------------------------------------------------
def all_pages() -> Dict[str, Dict[str, Any]]:
    """key -> 页面元信息。"""
    out: Dict[str, Dict[str, Any]] = {}
    for g in NAV_GROUPS:
        for it in g["items"]:
            out[it["key"]] = {**it, "group": g["group"]}
    return out


def page_meta(key: str) -> Dict[str, Any]:
    return all_pages().get(key, {"key": key, "label": key, "icon": "", "title": key, "desc": ""})


def group_of(key: str) -> str:
    return page_meta(key).get("group", "")


def current_page() -> str:
    return st.session_state.get(_STATE_KEY, DEFAULT_PAGE)


def goto(key: str) -> None:
    """跳转到指定页面（同时写入持久化状态）。"""
    st.session_state[_STATE_KEY] = key
    _persist(key)


def _persist(key: str) -> None:
    try:
        from store import state as state_repo
        state_repo.set("active_page", key)
    except Exception:
        pass


def _restore() -> str:
    try:
        from store import state as state_repo
        saved = state_repo.get("active_page")
    except Exception:
        saved = None
    return saved if saved in all_pages() else DEFAULT_PAGE


def init_page_state() -> None:
    """首帧从数据库恢复上次所在页面。"""
    if _STATE_KEY not in st.session_state:
        st.session_state[_STATE_KEY] = _restore()


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def render_sidebar(badge_counts: Optional[Dict[str, Any]] = None) -> str:
    """渲染侧边栏复合式目录，返回当前选中的页面 key。

    Args:
        badge_counts: 可选的 ``页面key -> 角标文本``，用于在目录项后展示数量。
    """
    init_page_state()
    active = current_page()
    active_group = group_of(active)
    badge_counts = badge_counts or {}

    with st.sidebar:
        theme.brand()

        for g in NAV_GROUPS:
            gname = g["group"]
            expanded = gname == active_group
            with st.expander(f'{g["icon"]}  {gname}', expanded=expanded):
                for it in g["items"]:
                    key = it["key"]
                    suffix = badge_counts.get(key)
                    label = f'{it["icon"]}  {it["label"]}'
                    if suffix:
                        label += f"  · {suffix}"
                    if st.button(
                        label,
                        key=f"nav_btn_{key}",
                        use_container_width=True,
                        type="primary" if key == active else "secondary",
                    ):
                        if key != active:
                            goto(key)
                            st.rerun()

    return active


def render_page_header(key: str) -> None:
    """按目录元信息渲染页面标题区。"""
    meta = page_meta(key)
    theme.page_header(meta.get("title") or meta.get("label", ""),
                      meta.get("desc", ""), meta.get("icon", ""))


def quick_links(keys: List[str], columns: int = 4) -> None:
    """在页面内渲染一排快捷跳转按钮（用于概览页）。"""
    metas = [page_meta(k) for k in keys]
    cols = st.columns(columns)
    for i, m in enumerate(metas):
        with cols[i % columns]:
            if st.button(f'{m["icon"]} {m["label"]}', key=f"quick_{m['key']}",
                         use_container_width=True):
                goto(m["key"])
                st.rerun()
