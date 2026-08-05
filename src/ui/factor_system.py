"""因子体系搭建与分析仪表盘（src/ui/factor_system.py）。

两个页面：

* :func:`render_system_builder`  —— 从「挖掘产出 + 系统因子库」中挑选因子，
  配置维度归类与权重方案，保存为可复用的因子体系。
* :func:`render_system_analysis` —— 对已保存体系执行合成回测，输出
  IC / ICIR / 分层 / 相关性 / 主成分 / 衰减 / 分散化极限的全景诊断仪表盘。

所有用户选择（筛选条件、勾选的因子、权重方案、回测参数）都会写入本地 SQLite，
关闭应用后重新打开可无缝恢复现场。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.factor_system import (
    WEIGHT_DIMENSION,
    WEIGHT_EQUAL,
    WEIGHT_IC,
    WEIGHT_ICIR,
    WEIGHT_LABELS,
    WEIGHT_MANUAL,
    WEIGHT_QUALITY,
    SystemMember,
    analyze_system,
    build_findings,
    load_market_panel,
    resolve_weights,
)
from engine.traditional_factors import CATEGORY_LABELS
from store import mining as mining_repo
from store import ops as ops_repo
from store import runs as runs_repo
from store import state as state_repo
from store import systems as systems_repo

from . import theme

# 体系维度：比原始 category 更贴近投研语言，可自由改写
DIMENSIONS: List[str] = [
    "趋势动量", "波动风险", "流动性", "量价背离", "量价形态",
    "基本面质量", "情绪资金", "事件驱动", "另类数据", "未分类",
]

# 传统因子五大类 -> 体系维度的默认映射
_CATEGORY_TO_DIM: Dict[str, str] = {
    "price_trend": "趋势动量",
    "volatility_uncertainty": "波动风险",
    "trading_difficulty": "流动性",
    "price_volume_divergence": "量价背离",
    "volume_price_formula": "量价形态",
}

_SOURCE_LABELS: Dict[str, str] = {
    "static": "系统因子库",
    "generated": "挖掘产出",
    "user": "自定义",
    "mining": "挖掘记录",
}

_SEL_KEY = "fs_builder_members"      # 持久化：当前搭建中的成分因子
_FILTER_KEY = "fs_builder_filters"   # 持久化：因子池筛选条件
_PARAM_KEY = "fs_analysis_params"    # 持久化：回测参数


# ===========================================================================
# 工具
# ===========================================================================
def _fmt(v: Any, digits: int = 4, pct: bool = False, default: str = "—") -> str:
    """安全格式化数字。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f"{f * 100:.2f}%" if pct else f"{f:.{digits}f}"


def _get_library():
    """拿到全局因子库实例（失败时返回 None，不阻塞页面）。"""
    try:
        from agent.integration import get_library
        return get_library()
    except Exception:
        try:
            from engine.factor_library import FactorLibrary
            return FactorLibrary()
        except Exception:
            return None


@st.cache_data(show_spinner=False, ttl=120)
def _load_factor_pool() -> pd.DataFrame:
    """汇总全部可选因子：系统因子库 + 挖掘产出 + 数据库挖掘记录。"""
    rows: List[Dict[str, Any]] = []
    seen: set = set()

    lib = _get_library()
    if lib is not None:
        try:
            for fd in lib.list_all():
                if fd.name in seen:
                    continue
                seen.add(fd.name)
                src = "static"
                if getattr(fd, "source", "") not in ("traditional_library", "static"):
                    src = "user" if getattr(fd, "source", "") == "user" else "generated"
                rows.append({
                    "选择": False,
                    "factor_name": fd.name,
                    "display_name": fd.display_name or fd.name,
                    "category": fd.category,
                    "category_label": CATEGORY_LABELS.get(fd.category, fd.category or "-"),
                    "dimension": _CATEGORY_TO_DIM.get(fd.category, "未分类"),
                    "source": src,
                    "source_label": _SOURCE_LABELS.get(src, src),
                    "direction": fd.direction or "positive",
                    "quality": float(getattr(fd, "quality_score", 0.5) or 0.5),
                    "description": (getattr(fd, "description", "") or "")[:120],
                    "tags": ", ".join(getattr(fd, "tags", []) or [])[:60],
                    "code": fd.code or "",
                })
        except Exception:
            pass

    # 数据库中的挖掘记录（Agent / GP / 精炼厂沉淀）
    try:
        for rec in mining_repo.list(limit=300):
            name = rec.get("factor_name") or ""
            code = rec.get("code") or ""
            if not name or not code or name in seen:
                continue
            seen.add(name)
            metrics = rec.get("metrics") or {}
            q = metrics.get("quality_score")
            if q is None:
                ic = abs(float(metrics.get("ic", 0) or 0))
                q = min(0.95, 0.4 + ic * 8)
            rows.append({
                "选择": False,
                "factor_name": name,
                "display_name": name,
                "category": rec.get("module", "mining"),
                "category_label": f"挖掘·{rec.get('module', '')}",
                "dimension": "未分类",
                "source": "mining",
                "source_label": _SOURCE_LABELS["mining"],
                "direction": "positive",
                "quality": float(q),
                "description": (rec.get("query") or rec.get("expression") or "")[:120],
                "tags": rec.get("module", ""),
                "code": code,
            })
    except Exception:
        pass

    if not rows:
        return pd.DataFrame(
            columns=["选择", "factor_name", "display_name", "category", "category_label",
                     "dimension", "source", "source_label", "direction", "quality",
                     "description", "tags", "code"]
        )
    return pd.DataFrame(rows).sort_values(
        ["source", "quality"], ascending=[True, False]
    ).reset_index(drop=True)


def _restore(key: str, default: Any) -> Any:
    """从 session_state 或数据库恢复状态。"""
    if key in st.session_state:
        return st.session_state[key]
    val = state_repo.get(key, default)
    st.session_state[key] = val
    return val


def _persist(key: str, value: Any) -> None:
    st.session_state[key] = value
    try:
        state_repo.set(key, value)
    except Exception:
        pass


# ===========================================================================
# 页面一：因子体系搭建
# ===========================================================================
def render_system_builder() -> None:
    """因子体系搭建页。"""
    theme.steps([
        {"n": "STEP 01", "title": "挑选因子", "desc": "从挖掘产出与系统库中筛选候选"},
        {"n": "STEP 02", "title": "维度归类", "desc": "为每个因子指定体系维度与方向"},
        {"n": "STEP 03", "title": "配置权重", "desc": "等权 / 质量 / IC / 维度均衡 / 手动"},
        {"n": "STEP 04", "title": "保存体系", "desc": "落库后可在分析仪表盘反复回测"},
    ])

    members: List[Dict[str, Any]] = _restore(_SEL_KEY, [])

    tab_pool, tab_config, tab_save, tab_manage = st.tabs(
        ["① 因子池", "② 维度与权重", "③ 保存体系", "④ 体系管理"]
    )

    with tab_pool:
        members = _render_pool(members)
    with tab_config:
        members = _render_config(members)
    with tab_save:
        _render_save(members)
    with tab_manage:
        _render_manage()


# --------------------------------------------------------------------- 因子池
def _render_pool(members: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pool = _load_factor_pool()
    if pool.empty:
        theme.empty_state(
            "因子池为空",
            "请先在「因子挖掘」分组下运行 Agent / 遗传规划 / 精炼厂产出因子，或确认系统因子库已加载。",
            "◌",
        )
        return members

    filters = _restore(_FILTER_KEY, {})
    theme.section("筛选候选因子", f"共 {len(pool)} 个可选因子，按来源 / 类别 / 质量分快速缩小范围")

    c1, c2, c3, c4 = st.columns([1.1, 1.4, 1, 1.3])
    with c1:
        src_opts = sorted(pool["source_label"].unique().tolist())
        sel_src = st.multiselect("来源", src_opts,
                                 default=filters.get("sources") or src_opts, key="fs_f_src")
    with c2:
        cat_opts = sorted(pool["category_label"].unique().tolist())
        sel_cat = st.multiselect("类别", cat_opts,
                                 default=filters.get("cats") or [], key="fs_f_cat")
    with c3:
        min_q = st.slider("最低质量分", 0.0, 1.0,
                          float(filters.get("min_q", 0.0)), 0.05, key="fs_f_q")
    with c4:
        kw = st.text_input("关键词", value=filters.get("kw", ""),
                           placeholder="名称 / 描述 / 标签", key="fs_f_kw")

    _persist(_FILTER_KEY, {"sources": sel_src, "cats": sel_cat, "min_q": min_q, "kw": kw})

    view = pool.copy()
    if sel_src:
        view = view[view["source_label"].isin(sel_src)]
    if sel_cat:
        view = view[view["category_label"].isin(sel_cat)]
    view = view[view["quality"] >= min_q]
    if kw:
        k = kw.lower()
        mask = (
            view["factor_name"].str.lower().str.contains(k, na=False)
            | view["display_name"].str.lower().str.contains(k, na=False)
            | view["description"].str.lower().str.contains(k, na=False)
            | view["tags"].str.lower().str.contains(k, na=False)
        )
        view = view[mask]

    chosen = {m["factor_name"] for m in members}
    view = view.reset_index(drop=True)
    view["选择"] = view["factor_name"].isin(chosen)

    theme.kpi_row([
        {"label": "筛选结果", "value": len(view), "sub": f"总池 {len(pool)}"},
        {"label": "已入选", "value": len(members), "sub": "进入体系的因子数", "tone": "pos"},
        {"label": "系统因子", "value": int((pool["source"] == "static").sum()), "tone": "neutral"},
        {"label": "挖掘因子", "value": int((pool["source"] != "static").sum()), "tone": "ink"},
    ])

    st.caption("勾选左侧「选择」列即可加入体系；表格支持点击列头排序。")
    edited = st.data_editor(
        view[["选择", "display_name", "factor_name", "source_label",
              "category_label", "direction", "quality", "description"]],
        hide_index=True,
        use_container_width=True,
        height=430,
        column_config={
            "选择": st.column_config.CheckboxColumn("选择", width="small"),
            "display_name": st.column_config.TextColumn("因子名称", width="medium", disabled=True),
            "factor_name": st.column_config.TextColumn("标识", width="medium", disabled=True),
            "source_label": st.column_config.TextColumn("来源", width="small", disabled=True),
            "category_label": st.column_config.TextColumn("类别", width="medium", disabled=True),
            "direction": st.column_config.TextColumn("方向", width="small", disabled=True),
            "quality": st.column_config.ProgressColumn(
                "质量分", min_value=0.0, max_value=1.0, format="%.2f", width="small"),
            "description": st.column_config.TextColumn("说明", width="large", disabled=True),
        },
        key="fs_pool_editor",
    )

    b1, b2, b3, _ = st.columns([1.2, 1.2, 1.2, 3])
    with b1:
        apply_sel = st.button("应用勾选", type="primary", use_container_width=True)
    with b2:
        add_top = st.button("按质量分补入 Top10", use_container_width=True)
    with b3:
        clear = st.button("清空已选", use_container_width=True)

    lookup = pool.set_index("factor_name")

    if apply_sel:
        names = edited.loc[edited["选择"] == True, "factor_name"].tolist()  # noqa: E712
        members = _merge_members(members, names, lookup)
        _persist(_SEL_KEY, members)
        ops_repo.log("factor_system", "select", f"更新体系候选因子，共 {len(members)} 个")
        st.success(f"已更新体系候选因子：{len(members)} 个")
        st.rerun()

    if add_top:
        names = view.sort_values("quality", ascending=False)["factor_name"].head(10).tolist()
        members = _merge_members(members, list({*[m["factor_name"] for m in members], *names}), lookup)
        _persist(_SEL_KEY, members)
        st.success(f"已补入质量分 Top10，共 {len(members)} 个因子")
        st.rerun()

    if clear:
        _persist(_SEL_KEY, [])
        ops_repo.log("factor_system", "clear", "清空体系候选因子")
        st.rerun()

    return members


def _merge_members(existing: List[Dict[str, Any]], names: Sequence[str],
                   lookup: pd.DataFrame) -> List[Dict[str, Any]]:
    """按最新勾选结果重建成分列表，保留已有的维度 / 权重设置。"""
    old = {m["factor_name"]: m for m in existing}
    out: List[Dict[str, Any]] = []
    for n in names:
        if n in old:
            out.append(old[n])
            continue
        if n not in lookup.index:
            continue
        r = lookup.loc[n]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        out.append({
            "factor_name": n,
            "display_name": str(r["display_name"]),
            "dimension": str(r["dimension"]),
            "category": str(r["category"]),
            "source": str(r["source"]),
            "direction": str(r["direction"]),
            "weight": 0.0,
            "quality": float(r["quality"]),
            "code": str(r["code"]),
            "meta": {"description": str(r["description"])},
        })
    return out


# ------------------------------------------------------------- 维度与权重
def _render_config(members: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not members:
        theme.empty_state("尚未选择因子", "请先到「① 因子池」勾选候选因子。", "◌")
        return members

    theme.section("维度归类与方向", "维度决定体系的风格结构；方向为 negative 的因子在合成时会自动取负对齐")

    df = pd.DataFrame(members)
    df["_manual_w"] = df["weight"].astype(float)

    edited = st.data_editor(
        df[["display_name", "factor_name", "dimension", "direction", "quality", "_manual_w"]],
        hide_index=True,
        use_container_width=True,
        height=min(430, 60 + 36 * len(df)),
        column_config={
            "display_name": st.column_config.TextColumn("因子名称", disabled=True, width="medium"),
            "factor_name": st.column_config.TextColumn("标识", disabled=True, width="medium"),
            "dimension": st.column_config.SelectboxColumn("体系维度", options=DIMENSIONS, width="medium"),
            "direction": st.column_config.SelectboxColumn(
                "方向", options=["positive", "negative", "none"], width="small"),
            "quality": st.column_config.NumberColumn(
                "质量分", min_value=0.0, max_value=1.0, step=0.05, format="%.2f", width="small"),
            "_manual_w": st.column_config.NumberColumn(
                "手动权重", min_value=0.0, max_value=100.0, step=0.5, format="%.2f", width="small",
                help="仅在权重方案选择「手动指定」时生效，系统会自动归一化"),
        },
        key="fs_cfg_editor",
    )

    theme.section("权重方案", "决定各因子在合成时的相对话语权")
    modes = [WEIGHT_EQUAL, WEIGHT_QUALITY, WEIGHT_DIMENSION, WEIGHT_IC, WEIGHT_ICIR, WEIGHT_MANUAL]
    saved_mode = _restore("fs_weight_mode", WEIGHT_EQUAL)
    c1, c2 = st.columns([1.6, 2.4])
    with c1:
        mode = st.selectbox(
            "权重方案", modes,
            index=modes.index(saved_mode) if saved_mode in modes else 0,
            format_func=lambda m: WEIGHT_LABELS.get(m, m),
            key="fs_wmode",
        )
    with c2:
        if mode in (WEIGHT_IC, WEIGHT_ICIR):
            theme.insight(
                "IC / ICIR 加权需要先跑一次单因子回测，权重会在「体系回测分析」页运行时自动确定。",
                "info",
            )
        elif mode == WEIGHT_MANUAL:
            theme.insight("手动权重取自上表「手动权重」列，保存前会自动归一化到 100%。", "info")
        else:
            theme.insight(f"当前方案：<b>{WEIGHT_LABELS[mode]}</b>。可在保存后于分析页临时切换对比。", "info")

    if mode != saved_mode:
        _persist("fs_weight_mode", mode)

    # 回写编辑结果
    updated: List[Dict[str, Any]] = []
    for i, m in enumerate(members):
        row = edited.iloc[i]
        nm = dict(m)
        nm["dimension"] = str(row["dimension"])
        nm["direction"] = str(row["direction"])
        nm["quality"] = float(row["quality"])
        nm["weight"] = float(row["_manual_w"])
        updated.append(nm)
    if updated != members:
        _persist(_SEL_KEY, updated)
        members = updated

    # 权重预览
    sm = [SystemMember.from_dict(m) for m in members]
    preview_mode = mode if mode not in (WEIGHT_IC, WEIGHT_ICIR) else WEIGHT_EQUAL
    weights = resolve_weights(sm, preview_mode)

    theme.section("结构预览", "左：各维度权重占比；右：各因子权重明细")
    c1, c2 = st.columns([1, 1.35])
    with c1:
        dim_w: Dict[str, float] = {}
        for m in members:
            dim_w[m["dimension"]] = dim_w.get(m["dimension"], 0.0) + weights.get(m["factor_name"], 0.0)
        if dim_w:
            fig = go.Figure(go.Pie(
                labels=list(dim_w.keys()),
                values=[round(v * 100, 2) for v in dim_w.values()],
                hole=0.58,
                marker=dict(colors=theme.PALETTE, line=dict(color="#fff", width=2)),
                textinfo="label+percent", textfont=dict(size=11),
            ))
            st.plotly_chart(theme.style_fig(fig, height=310, legend=False, title="维度权重分布"),
                            use_container_width=True)
    with c2:
        order = sorted(members, key=lambda m: -weights.get(m["factor_name"], 0.0))
        fig = go.Figure(go.Bar(
            x=[weights.get(m["factor_name"], 0) * 100 for m in order][::-1],
            y=[m["display_name"][:22] for m in order][::-1],
            orientation="h",
            marker=dict(color=theme.RED, line=dict(color=theme.RED_DEEP, width=0.5)),
            text=[f'{weights.get(m["factor_name"], 0) * 100:.1f}%' for m in order][::-1],
            textposition="outside", textfont=dict(size=10),
        ))
        st.plotly_chart(
            theme.style_fig(fig, height=max(310, 24 * len(order) + 70), legend=False,
                            title="因子权重明细（%）"),
            use_container_width=True,
        )

    return members


# ------------------------------------------------------------------ 保存
def _render_save(members: List[Dict[str, Any]]) -> None:
    if not members:
        theme.empty_state("尚未选择因子", "请先到「① 因子池」勾选候选因子。", "◌")
        return

    mode = st.session_state.get("fs_weight_mode", WEIGHT_EQUAL)
    sm = [SystemMember.from_dict(m) for m in members]
    weights = resolve_weights(sm, mode if mode not in (WEIGHT_IC, WEIGHT_ICIR) else WEIGHT_EQUAL)
    dims = sorted({m["dimension"] for m in members})

    theme.section("体系概要", "确认无误后保存到本地数据库")
    theme.kpi_row([
        {"label": "成分因子", "value": len(members)},
        {"label": "覆盖维度", "value": len(dims), "tone": "neutral"},
        {"label": "权重方案", "value": WEIGHT_LABELS.get(mode, mode)[:6], "tone": "ink"},
        {"label": "平均质量分", "value": _fmt(np.mean([m["quality"] for m in members]), 2),
         "tone": "pos"},
    ])
    theme.chips([(m["display_name"][:16], f'{weights.get(m["factor_name"], 0) * 100:.0f}%')
                 for m in members])

    with st.form("fs_save_form"):
        c1, c2 = st.columns([1.4, 2.6])
        with c1:
            name = st.text_input("体系名称 *", placeholder="例如：多维Alpha体系V1")
        with c2:
            tags = st.text_input("标签", placeholder="逗号分隔，例如：中频, 全市场, 量价")
        desc = st.text_area("体系说明", placeholder="记录构建思路、适用市场与调仓频率等",
                            height=80)
        submitted = st.form_submit_button("保存因子体系", type="primary", use_container_width=True)

    if submitted:
        if not name.strip():
            st.error("请填写体系名称")
            return
        payload = []
        for m in members:
            d = dict(m)
            d["weight"] = float(weights.get(m["factor_name"], 0.0))
            payload.append(d)
        try:
            sid = systems_repo.save(
                name=name.strip(),
                members=payload,
                description=desc.strip(),
                weight_mode=mode,
                config={"dimensions": dims},
                tags=[t.strip() for t in tags.split(",") if t.strip()],
            )
        except Exception as e:
            st.error(f"保存失败：{e}")
            return
        ops_repo.log("factor_system", "save",
                     f"保存因子体系「{name.strip()}」（{len(payload)} 因子）",
                     payload={"system_id": sid})
        _persist("fs_active_system", sid)
        st.success(f"已保存因子体系「{name.strip()}」，可前往「体系回测分析」执行回测。")
        st.balloons()


# ------------------------------------------------------------------ 管理
def _render_manage() -> None:
    items = systems_repo.list()
    if not items:
        theme.empty_state("暂无已保存体系", "在「③ 保存体系」中创建第一个因子体系。", "◌")
        return

    theme.section("已保存的因子体系", f"共 {len(items)} 个，数据存放于本地 SQLite，可随时载入继续编辑")
    df = pd.DataFrame([{
        "名称": it["name"],
        "因子数": it["n_factors"],
        "权重方案": WEIGHT_LABELS.get(it["weight_mode"], it["weight_mode"]),
        "标签": ", ".join(it.get("tags") or []),
        "说明": (it.get("description") or "")[:60],
        "更新时间": it["updated_at"],
    } for it in items])
    st.dataframe(df, hide_index=True, use_container_width=True, height=min(360, 60 + 36 * len(df)))

    c1, c2, c3 = st.columns([2.4, 1.1, 1.1])
    with c1:
        names = [it["name"] for it in items]
        pick = st.selectbox("选择体系", names, key="fs_manage_pick")
    sid = next((it["id"] for it in items if it["name"] == pick), None)
    with c2:
        if st.button("载入编辑", use_container_width=True) and sid:
            sysobj = systems_repo.get(sid)
            if sysobj:
                _persist(_SEL_KEY, sysobj["members"])
                _persist("fs_weight_mode", sysobj["weight_mode"])
                ops_repo.log("factor_system", "load", f"载入体系「{pick}」")
                st.success(f"已载入「{pick}」，切换到「② 维度与权重」继续编辑。")
                st.rerun()
    with c3:
        if st.button("删除体系", use_container_width=True) and sid:
            systems_repo.delete(sid)
            ops_repo.log("factor_system", "delete", f"删除体系「{pick}」")
            st.warning(f"已删除「{pick}」")
            st.rerun()


# ===========================================================================
# 页面二：因子体系回测分析
# ===========================================================================
def render_system_analysis() -> None:
    """因子体系分析仪表盘。"""
    items = systems_repo.list()
    if not items:
        theme.empty_state(
            "还没有可分析的因子体系",
            "请先到「因子体系 → 体系搭建」创建并保存一个体系。",
            "◌",
        )
        return

    params: Dict[str, Any] = _restore(_PARAM_KEY, {})
    active_sid = _restore("fs_active_system", items[0]["id"])
    names = [it["name"] for it in items]
    idx = next((i for i, it in enumerate(items) if it["id"] == active_sid), 0)

    theme.section("回测配置", "样本规模越大越贴近真实，但耗时也更长；参数会自动记忆")
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
    with c1:
        pick = st.selectbox("因子体系", names, index=idx, key="fs_an_sys")
    with c2:
        n_symbols = st.number_input("股票数", 20, 300, int(params.get("n_symbols", 60)), 10)
    with c3:
        days = st.number_input("交易日数", 120, 1200, int(params.get("days", 400)), 20)
    with c4:
        n_q = st.number_input("分层数", 3, 10, int(params.get("n_quantiles", 5)), 1)
    with c5:
        fwd = st.number_input("持有期", 1, 20, int(params.get("forward_periods", 1)), 1)

    c6, c7, c8 = st.columns([2, 2, 2])
    modes = [WEIGHT_EQUAL, WEIGHT_QUALITY, WEIGHT_DIMENSION, WEIGHT_IC, WEIGHT_ICIR, WEIGHT_MANUAL]
    sysobj = next((it for it in items if it["name"] == pick), items[0])
    with c6:
        default_mode = sysobj["weight_mode"] if sysobj["weight_mode"] in modes else WEIGHT_EQUAL
        wmode = st.selectbox("权重方案（可临时覆盖）", modes,
                             index=modes.index(default_mode),
                             format_func=lambda m: WEIGHT_LABELS.get(m, m), key="fs_an_wmode")
    with c7:
        deep = st.checkbox("深度诊断（IC 衰减 + 分散化极限）",
                           value=bool(params.get("deep", True)),
                           help="额外执行多持有期与逐步纳入因子的回测，耗时约增加 2-3 倍")
    with c8:
        force_refresh = st.checkbox("强制重新拉取行情", value=False,
                                    help="不勾选时优先使用本地缓存，可离线运行")

    run = st.button("运行体系回测", type="primary", use_container_width=True)

    _persist(_PARAM_KEY, {
        "n_symbols": int(n_symbols), "days": int(days),
        "n_quantiles": int(n_q), "forward_periods": int(fwd), "deep": bool(deep),
    })
    _persist("fs_active_system", sysobj["id"])

    result_key = f"fs_result_{sysobj['id']}"

    if run:
        full = systems_repo.get(sysobj["id"])
        if not full or not full["members"]:
            st.error("该体系没有成分因子")
            return
        with st.status("正在执行因子体系回测…", expanded=True) as status:
            st.write("① 准备行情面板…")
            kline, meta = load_market_panel(
                n_symbols=int(n_symbols), days=int(days), force_refresh=force_refresh
            )
            st.write(f"　　数据来源：{meta['source']}｜{meta['n_symbols']} 只 × {meta['n_dates']} 日")

            bar = st.progress(0.0, text="② 计算因子矩阵…")

            def _prog(done: int, total: int, name: str) -> None:
                bar.progress(min(done / max(total, 1), 1.0), text=f"② 计算因子：{name} ({done}/{total})")

            result = analyze_system(
                kline,
                [SystemMember.from_dict(m) for m in full["members"]],
                weight_mode=wmode,
                n_quantiles=int(n_q),
                forward_periods=int(fwd),
                run_decay=deep,
                run_diversification=deep,
                progress=_prog,
            )
            bar.progress(1.0, text="③ 汇总诊断结果…")

            if result.get("error"):
                status.update(label="回测失败", state="error")
                st.error(result["error"])
                if result.get("errors"):
                    st.json(result["errors"])
                return

            result["_meta"] = meta
            result["_system"] = {"id": sysobj["id"], "name": sysobj["name"]}
            st.session_state[result_key] = result
            status.update(label="回测完成", state="complete")

        # 落库
        cm = result["composite_metrics"]
        try:
            runs_repo.save(
                system_id=sysobj["id"], system_name=sysobj["name"],
                metrics={k: v for k, v in cm.items() if isinstance(v, (int, float, str))},
                params={"n_symbols": int(n_symbols), "days": int(days),
                        "n_quantiles": int(n_q), "forward_periods": int(fwd),
                        "weight_mode": wmode},
                details={"weights": result["weights"], "dimensions": result["dimensions"],
                         "findings": build_findings(result)},
                universe=f"{meta['n_symbols']}只", period=meta["period"],
            )
            ops_repo.log("factor_system", "backtest",
                         f"体系「{sysobj['name']}」回测完成：IC={_fmt(cm.get('ic'))}")
        except Exception:
            pass

    result = st.session_state.get(result_key)
    if not result:
        theme.insight(
            f"体系 <b>{sysobj['name']}</b> 共 {sysobj['n_factors']} 个因子，"
            f"点击上方「运行体系回测」生成分析仪表盘。", "info"
        )
        _render_run_history(sysobj["id"])
        return

    _render_dashboard(result)
    _render_run_history(sysobj["id"])


# --------------------------------------------------------------- 仪表盘
def _render_dashboard(result: Dict[str, Any]) -> None:
    cm = result["composite_metrics"]
    corr = result["correlation"]
    meta = result.get("_meta", {})

    st.markdown("---")
    theme.section(
        "体系核心指标",
        f'数据来源：{meta.get("source", "-")}｜{meta.get("period", "-")}｜'
        f'{meta.get("n_symbols", 0)} 只股票 × {meta.get("n_dates", 0)} 个交易日',
    )

    ic = cm.get("ic", float("nan"))
    icir = cm.get("icir", float("nan"))
    theme.kpi_row([
        {"label": "体系 IC", "value": _fmt(ic), "sub": "截面相关系数均值",
         "tone": "pos" if _num(ic) > 0.02 else "red"},
        {"label": "Rank IC", "value": _fmt(cm.get("rank_ic")), "sub": "秩相关，抗异常值"},
        {"label": "ICIR", "value": _fmt(icir, 2), "sub": "IC 稳定性",
         "tone": "pos" if _num(icir) > 0.4 else "warn"},
        {"label": "IC 胜率", "value": _fmt(cm.get("ic_positive_ratio"), pct=True), "sub": "IC>0 占比"},
        {"label": "多空夏普", "value": _fmt(cm.get("long_short_sharpe"), 2), "sub": "年化",
         "tone": "neutral"},
        {"label": "最大回撤", "value": _fmt(cm.get("max_drawdown"), pct=True), "sub": "多空组合",
         "tone": "warn"},
    ], columns=6)

    theme.kpi_row([
        {"label": "成分因子", "value": result["n_factors"], "tone": "ink"},
        {"label": "有效维度", "value": _fmt(corr.get("effective_factors"), 1),
         "sub": "主成分折算", "tone": "neutral"},
        {"label": "平均相关", "value": _fmt(corr.get("mean_abs_corr"), 2),
         "sub": "越低越分散", "tone": "pos" if _num(corr.get("mean_abs_corr"), 1) < 0.4 else "warn"},
        {"label": "多空年化", "value": _fmt(_num(cm.get("long_short_return")) * 252, pct=True),
         "sub": "日均 × 252"},
        {"label": "换手率", "value": _fmt(cm.get("turnover"), pct=True), "sub": "日均"},
        {"label": "覆盖度", "value": _fmt(cm.get("coverage"), pct=True), "sub": "非空占比",
         "tone": "ink"},
    ], columns=6)

    # 核心发现
    findings = build_findings(result)
    if findings:
        theme.section("核心发现", "系统根据诊断结果自动生成的结论与改进建议")
        for f in findings:
            theme.insight(f["text"], f.get("tone", "red"))

    tabs = st.tabs(["结构透视", "收益表现", "稳定性诊断", "相关性与主成分", "因子明细"])

    with tabs[0]:
        _tab_structure(result)
    with tabs[1]:
        _tab_performance(result)
    with tabs[2]:
        _tab_stability(result)
    with tabs[3]:
        _tab_correlation(result)
    with tabs[4]:
        _tab_detail(result)


def _num(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------- Tab: 结构透视
def _tab_structure(result: Dict[str, Any]) -> None:
    dims = result.get("dimensions") or []
    weights = result.get("weights") or {}
    members = result.get("members") or []

    c1, c2 = st.columns([1, 1.3])
    with c1:
        if dims:
            fig = go.Figure(go.Pie(
                labels=[d["dimension"] for d in dims],
                values=[round(d["weight"] * 100, 2) for d in dims],
                hole=0.58,
                marker=dict(colors=theme.PALETTE, line=dict(color="#fff", width=2)),
                textinfo="label+percent", textfont=dict(size=11),
            ))
            st.plotly_chart(theme.style_fig(fig, height=340, legend=False, title="维度权重分布"),
                            use_container_width=True)
    with c2:
        if dims:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[d["dimension"] for d in dims],
                y=[round(d["weight"] * 100, 2) for d in dims],
                name="权重(%)", marker=dict(color=theme.RED), yaxis="y",
                text=[f'{d["weight"] * 100:.0f}%' for d in dims], textposition="outside",
                textfont=dict(size=10),
            ))
            fig.add_trace(go.Scatter(
                x=[d["dimension"] for d in dims],
                y=[_num(d.get("mean_icir")) for d in dims],
                name="维度均值 ICIR", mode="lines+markers", yaxis="y2",
                line=dict(color=theme.BLUE, width=2), marker=dict(size=7),
            ))
            fig.update_layout(
                yaxis=dict(title="权重(%)"),
                yaxis2=dict(title="ICIR", overlaying="y", side="right", showgrid=False),
            )
            st.plotly_chart(theme.style_fig(fig, height=340, title="维度权重 vs 维度表现"),
                            use_container_width=True)

    # 维度雷达
    if len(dims) >= 3:
        theme.section("维度质量雷达", "各维度的平均 IC / ICIR / 因子数量归一化后对比")
        labels = [d["dimension"] for d in dims]
        max_ic = max([abs(_num(d.get("mean_ic"))) for d in dims] + [1e-9])
        max_icir = max([abs(_num(d.get("mean_icir"))) for d in dims] + [1e-9])
        max_n = max([d["n_factors"] for d in dims] + [1])
        fig = go.Figure()
        for key, name, mx, color in (
            ("mean_ic", "平均 IC", max_ic, theme.RED),
            ("mean_icir", "平均 ICIR", max_icir, theme.BLUE),
        ):
            vals = [abs(_num(d.get(key))) / mx * 100 for d in dims]
            fig.add_trace(go.Scatterpolar(
                r=vals + vals[:1], theta=labels + labels[:1], fill="toself", name=name,
                line=dict(color=color, width=2), opacity=0.55,
            ))
        vals = [d["n_factors"] / max_n * 100 for d in dims]
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=labels + labels[:1], name="因子数量",
            line=dict(color=theme.INK_MUTED, width=1.5, dash="dot"), fill=None,
        ))
        fig.update_layout(polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#F0F2F5", tickfont=dict(size=9)),
            angularaxis=dict(gridcolor="#F0F2F5", tickfont=dict(size=11)),
            bgcolor="#fff",
        ))
        st.plotly_chart(theme.style_fig(fig, height=420), use_container_width=True)

    # 因子权重条形图
    if members:
        order = sorted(members, key=lambda m: -weights.get(m["factor_name"], 0))
        dim_color = {d["dimension"]: theme.PALETTE[i % len(theme.PALETTE)]
                     for i, d in enumerate(dims)}
        fig = go.Figure(go.Bar(
            x=[weights.get(m["factor_name"], 0) * 100 for m in order][::-1],
            y=[m["display_name"][:24] for m in order][::-1],
            orientation="h",
            marker=dict(color=[dim_color.get(m["dimension"], theme.RED) for m in order][::-1]),
            text=[f'{weights.get(m["factor_name"], 0) * 100:.1f}%' for m in order][::-1],
            textposition="outside", textfont=dict(size=10),
            customdata=[[m["dimension"]] for m in order][::-1],
            hovertemplate="%{y}<br>维度：%{customdata[0]}<br>权重：%{x:.2f}%<extra></extra>",
        ))
        st.plotly_chart(
            theme.style_fig(fig, height=max(300, 24 * len(order) + 80), legend=False,
                            title="因子权重明细（按维度着色）"),
            use_container_width=True,
        )


# --------------------------------------------------------- Tab: 收益表现
def _tab_performance(result: Dict[str, Any]) -> None:
    cm = result["composite_metrics"]
    qcum = result.get("quantile_cum") or {}
    ls = result.get("ls_series")

    c1, c2 = st.columns(2)
    with c1:
        if qcum:
            fig = go.Figure()
            keys = sorted(qcum.keys(), key=lambda k: int(k))
            n = len(keys)
            for i, g in enumerate(keys):
                s = qcum[g]
                if isinstance(s, dict):
                    s = pd.Series(s)
                if s is None or len(s) == 0:
                    continue
                shade = i / max(n - 1, 1)
                color = f"rgba({int(200 - 140 * (1 - shade))},{int(16 + 90 * (1 - shade))},{int(46 + 90 * (1 - shade))},1)"
                fig.add_trace(go.Scatter(
                    x=list(s.index), y=[v * 100 for v in s.values], mode="lines",
                    name=f"Q{int(g) + 1}",
                    line=dict(width=2 if i in (0, n - 1) else 1.2, color=color),
                ))
            fig.add_hline(y=0, line=dict(color=theme.INK_MUTED, width=1, dash="dash"))
            st.plotly_chart(theme.style_fig(fig, height=330, title="分层累计收益（%）"),
                            use_container_width=True)
    with c2:
        if ls is not None and len(ls) > 0:
            s = pd.Series(ls).dropna()
            cum = ((1 + s).cumprod() - 1) * 100
            dd = (cum / 100 + 1) / (cum / 100 + 1).cummax() - 1
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(cum.index), y=cum.values, mode="lines", name="多空累计收益",
                line=dict(color=theme.RED, width=2.2), fill="tozeroy",
                fillcolor="rgba(200,16,46,0.08)",
            ))
            fig.add_trace(go.Scatter(
                x=list(dd.index), y=dd.values * 100, mode="lines", name="回撤",
                line=dict(color=theme.INK_MUTED, width=1, dash="dot"), yaxis="y2",
            ))
            fig.update_layout(yaxis=dict(title="累计收益(%)"),
                              yaxis2=dict(title="回撤(%)", overlaying="y", side="right",
                                          showgrid=False))
            st.plotly_chart(theme.style_fig(fig, height=330, title="多空组合净值与回撤"),
                            use_container_width=True)

    # 分层统计表
    qstats = cm.get("quantile_stats") or {}
    if qstats:
        theme.section("分层组合统计", "单调性是因子有效性的重要证据：从 Q1 到 Qn 应呈稳定递增或递减")
        rows = []
        for g in sorted(qstats.keys(), key=lambda k: int(k)):
            v = qstats[g]
            rows.append({
                "分层": f"Q{int(g) + 1}",
                "日均收益": _fmt(v.get("mean_ret"), pct=True),
                "年化收益": _fmt(v.get("ann_ret"), pct=True),
                "累计收益": _fmt(v.get("cum_ret"), pct=True),
                "夏普": _fmt(v.get("sharpe"), 2),
                "样本天数": int(v.get("n", 0)),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ------------------------------------------------------- Tab: 稳定性诊断
def _tab_stability(result: Dict[str, Any]) -> None:
    ic_series = result.get("ic_series")
    decay = result.get("decay") or []
    divers = result.get("diversification") or []

    if ic_series is not None and len(ic_series) > 0:
        s = pd.Series(ic_series).dropna()
        cum = s.cumsum()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(s.index), y=s.values, name="日度 IC",
            marker=dict(color=[theme.RED if v >= 0 else theme.BLUE for v in s.values]),
            opacity=0.55,
        ))
        fig.add_trace(go.Scatter(
            x=list(cum.index), y=cum.values, name="累计 IC", yaxis="y2",
            line=dict(color=theme.RED_DEEP, width=2.2),
        ))
        fig.update_layout(yaxis=dict(title="日度 IC"),
                          yaxis2=dict(title="累计 IC", overlaying="y", side="right", showgrid=False))
        st.plotly_chart(theme.style_fig(fig, height=330, title="IC 时间序列与累计 IC"),
                        use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if decay:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[d["period"] for d in decay], y=[d["ic"] for d in decay],
                mode="lines+markers", name="IC",
                line=dict(color=theme.RED, width=2.4), marker=dict(size=8),
                fill="tozeroy", fillcolor="rgba(200,16,46,0.08)",
            ))
            fig.add_trace(go.Scatter(
                x=[d["period"] for d in decay], y=[d["rank_ic"] for d in decay],
                mode="lines+markers", name="Rank IC",
                line=dict(color=theme.BLUE, width=2, dash="dash"), marker=dict(size=7),
            ))
            fig.update_xaxes(title="持有期（交易日）")
            st.plotly_chart(theme.style_fig(fig, height=320, title="IC 衰减曲线"),
                            use_container_width=True)
        else:
            theme.empty_state("未运行衰减诊断", "勾选「深度诊断」后重新回测即可查看。", "◌")
    with c2:
        if divers:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[d["n_factors"] for d in divers], y=[d["icir"] for d in divers],
                name="体系 ICIR", marker=dict(color=theme.RED), opacity=0.85,
            ))
            fig.add_trace(go.Scatter(
                x=[d["n_factors"] for d in divers], y=[d["sharpe"] for d in divers],
                mode="lines+markers", name="多空夏普", yaxis="y2",
                line=dict(color=theme.BLUE, width=2), marker=dict(size=7),
            ))
            fig.update_xaxes(title="纳入因子数（按 |ICIR| 降序）")
            fig.update_layout(yaxis=dict(title="ICIR"),
                              yaxis2=dict(title="夏普", overlaying="y", side="right",
                                          showgrid=False))
            st.plotly_chart(theme.style_fig(fig, height=320, title="分散化收益递减曲线"),
                            use_container_width=True)
            best = max(divers, key=lambda d: abs(d["icir"]))
            theme.insight(
                f"纳入 <b>{best['n_factors']}</b> 个因子时体系 ICIR 达到峰值 "
                f"{best['icir']:.2f}；继续加因子的边际收益已趋于平坦。",
                "info" if best["n_factors"] < len(divers) else "ok",
            )
        else:
            theme.empty_state("未运行分散化诊断", "勾选「深度诊断」并保证体系有 2 个以上因子。", "◌")

    # Alpha 强度 vs 稳定性
    stats = result.get("factor_stats") or {}
    members = {m["factor_name"]: m for m in (result.get("members") or [])}
    pts = [(n, v) for n, v in stats.items() if "error" not in v]
    if len(pts) >= 3:
        theme.section("Alpha 强度 vs 稳定性",
                      "横轴 |IC| 代表信号强度，纵轴 |ICIR| 代表稳定性；右上角为优质因子")
        fig = go.Figure()
        dims = sorted({members.get(n, {}).get("dimension", "未分类") for n, _ in pts})
        for i, d in enumerate(dims):
            sub = [(n, v) for n, v in pts if members.get(n, {}).get("dimension", "未分类") == d]
            if not sub:
                continue
            fig.add_trace(go.Scatter(
                x=[abs(_num(v.get("ic"))) for _, v in sub],
                y=[abs(_num(v.get("icir"))) for _, v in sub],
                mode="markers+text", name=d,
                text=[members.get(n, {}).get("display_name", n)[:10] for n, _ in sub],
                textposition="top center", textfont=dict(size=9, color=theme.INK_SUB),
                marker=dict(size=[8 + 26 * abs(_num(v.get("long_short_sharpe"))) / 3
                                  for _, v in sub],
                            color=theme.PALETTE[i % len(theme.PALETTE)],
                            line=dict(color="#fff", width=1), opacity=0.8),
                hovertemplate="%{text}<br>|IC|=%{x:.4f}<br>|ICIR|=%{y:.2f}<extra></extra>",
            ))
        fig.add_hline(y=0.4, line=dict(color=theme.RED_LINE, width=1, dash="dash"))
        fig.add_vline(x=0.02, line=dict(color=theme.RED_LINE, width=1, dash="dash"))
        fig.update_xaxes(title="|IC|")
        fig.update_yaxes(title="|ICIR|")
        st.plotly_chart(theme.style_fig(fig, height=430), use_container_width=True)


# --------------------------------------------------- Tab: 相关性与主成分
def _tab_correlation(result: Dict[str, Any]) -> None:
    corr_info = result.get("correlation") or {}
    corr = corr_info.get("corr")
    if corr is None or (isinstance(corr, pd.DataFrame) and corr.empty):
        theme.empty_state("因子数量不足", "至少需要 2 个成功计算的因子才能做相关性分析。", "◌")
        return
    if isinstance(corr, dict):
        corr = pd.DataFrame(corr)

    members = {m["factor_name"]: m for m in (result.get("members") or [])}
    labels = [members.get(c, {}).get("display_name", c)[:14] for c in corr.columns]

    c1, c2 = st.columns([1.35, 1])
    with c1:
        fig = go.Figure(go.Heatmap(
            z=corr.to_numpy(), x=labels, y=labels,
            colorscale=theme.COLORSCALE_DIVERGING, zmid=0, zmin=-1, zmax=1,
            colorbar=dict(thickness=12, len=0.85, tickfont=dict(size=10)),
            hovertemplate="%{y} × %{x}<br>ρ = %{z:.3f}<extra></extra>",
        ))
        fig.update_xaxes(tickangle=-40, tickfont=dict(size=9))
        fig.update_yaxes(tickfont=dict(size=9))
        st.plotly_chart(
            theme.style_fig(fig, height=max(360, 22 * len(labels) + 120), legend=False,
                            title="因子相关性矩阵"),
            use_container_width=True,
        )
    with c2:
        ev = corr_info.get("explained_variance") or []
        if ev:
            k = min(len(ev), 10)
            cum = np.cumsum(ev[:k]) * 100
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f"PC{i + 1}" for i in range(k)], y=[v * 100 for v in ev[:k]],
                name="方差解释比(%)", marker=dict(color=theme.RED),
                text=[f"{v * 100:.0f}%" for v in ev[:k]], textposition="outside",
                textfont=dict(size=9),
            ))
            fig.add_trace(go.Scatter(
                x=[f"PC{i + 1}" for i in range(k)], y=cum, mode="lines+markers",
                name="累计(%)", yaxis="y2", line=dict(color=theme.BLUE, width=2),
                marker=dict(size=6),
            ))
            fig.update_layout(yaxis=dict(title="解释比(%)"),
                              yaxis2=dict(title="累计(%)", overlaying="y", side="right",
                                          range=[0, 105], showgrid=False))
            st.plotly_chart(theme.style_fig(fig, height=340, title="主成分方差解释"),
                            use_container_width=True)

        pc1 = corr_info.get("pc1_loadings") or {}
        if pc1:
            order = sorted(pc1.items(), key=lambda kv: -abs(kv[1]))[:12]
            fig = go.Figure(go.Bar(
                x=[v for _, v in order][::-1],
                y=[members.get(k, {}).get("display_name", k)[:14] for k, _ in order][::-1],
                orientation="h",
                marker=dict(color=[theme.RED if v >= 0 else theme.BLUE for _, v in order][::-1]),
            ))
            st.plotly_chart(
                theme.style_fig(fig, height=max(260, 22 * len(order) + 70), legend=False,
                                title="第一主成分载荷（共同风险暴露）"),
                use_container_width=True,
            )

    pairs = corr_info.get("redundant_pairs") or []
    if pairs:
        theme.section("高相关因子对", "|ρ| ≥ 0.8 的因子对，建议二选一或做正交化处理")
        st.dataframe(
            pd.DataFrame([{
                "因子 A": members.get(p["a"], {}).get("display_name", p["a"]),
                "因子 B": members.get(p["b"], {}).get("display_name", p["b"]),
                "相关系数": round(p["corr"], 3),
            } for p in pairs]),
            hide_index=True, use_container_width=True,
        )
    else:
        theme.insight("未发现 |ρ| ≥ 0.8 的高相关因子对，体系冗余度可控。", "ok")


# --------------------------------------------------------- Tab: 因子明细
def _tab_detail(result: Dict[str, Any]) -> None:
    stats = result.get("factor_stats") or {}
    weights = result.get("weights") or {}
    members = result.get("members") or []

    rows = []
    for m in members:
        n = m["factor_name"]
        s = stats.get(n, {})
        rows.append({
            "因子名称": m["display_name"],
            "维度": m["dimension"],
            "来源": _SOURCE_LABELS.get(m["source"], m["source"]),
            "方向": m["direction"],
            "权重": round(weights.get(n, 0) * 100, 2),
            "IC": round(_num(s.get("ic")), 4),
            "RankIC": round(_num(s.get("rank_ic")), 4),
            "ICIR": round(_num(s.get("icir")), 3),
            "IC胜率": round(_num(s.get("ic_positive_ratio")) * 100, 1),
            "多空夏普": round(_num(s.get("long_short_sharpe")), 2),
            "覆盖度": round(_num(s.get("coverage")) * 100, 1),
            "状态": s.get("error", "正常")[:30],
        })
    if not rows:
        theme.empty_state("无因子明细", "", "◌")
        return

    df = pd.DataFrame(rows).sort_values("权重", ascending=False)
    st.dataframe(
        df, hide_index=True, use_container_width=True,
        height=min(520, 60 + 36 * len(df)),
        column_config={
            "权重": st.column_config.ProgressColumn(
                "权重(%)", min_value=0.0,
                max_value=float(max(df["权重"].max(), 1)), format="%.2f"),
            "IC胜率": st.column_config.NumberColumn("IC胜率(%)", format="%.1f"),
            "覆盖度": st.column_config.NumberColumn("覆盖度(%)", format="%.1f"),
        },
    )

    st.download_button(
        "导出因子明细 CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f'factor_system_{result.get("_system", {}).get("name", "detail")}.csv',
        mime="text/csv",
    )

    errors = result.get("errors") or {}
    if errors:
        with st.expander(f"计算失败的因子（{len(errors)} 个）"):
            st.dataframe(
                pd.DataFrame([{"因子": k, "失败原因": v} for k, v in errors.items()]),
                hide_index=True, use_container_width=True,
            )


# ------------------------------------------------------------- 历史记录
def _render_run_history(system_id: int) -> None:
    history = runs_repo.list(system_id=system_id, limit=20)
    if not history:
        return
    st.markdown("---")
    theme.section("历史回测记录", "同一体系在不同参数下的表现对比，全部保存在本地数据库")
    rows = []
    for h in history:
        m = h["metrics"]
        p = h["params"]
        rows.append({
            "时间": h["created_at"],
            "权重方案": WEIGHT_LABELS.get(p.get("weight_mode", ""), p.get("weight_mode", "-")),
            "股票池": h["universe"],
            "持有期": p.get("forward_periods", "-"),
            "IC": _fmt(m.get("ic")),
            "ICIR": _fmt(m.get("icir"), 2),
            "多空夏普": _fmt(m.get("long_short_sharpe"), 2),
            "最大回撤": _fmt(m.get("max_drawdown"), pct=True),
            "区间": h["period"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 height=min(320, 60 + 36 * len(rows)))

    if len(history) >= 2:
        fig = go.Figure()
        xs = [h["created_at"][5:16] for h in reversed(history)]
        fig.add_trace(go.Scatter(
            x=xs, y=[_num(h["metrics"].get("ic")) for h in reversed(history)],
            mode="lines+markers", name="IC", line=dict(color=theme.RED, width=2)))
        fig.add_trace(go.Scatter(
            x=xs, y=[_num(h["metrics"].get("icir")) for h in reversed(history)],
            mode="lines+markers", name="ICIR", yaxis="y2",
            line=dict(color=theme.BLUE, width=2, dash="dash")))
        fig.update_layout(yaxis=dict(title="IC"),
                          yaxis2=dict(title="ICIR", overlaying="y", side="right", showgrid=False))
        st.plotly_chart(theme.style_fig(fig, height=280, title="历次回测走势"),
                        use_container_width=True)
