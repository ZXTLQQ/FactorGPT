"""
FactorGPT — Streamlit 交互入口
==============================

界面采用「红白配色 + 复合式二级目录」，17 个功能页按业务域整合为 6 个分组：

- ◈ 工作台：系统概览、操作记忆
- ⚗ 因子挖掘：智能挖掘 Agent、对话式挖掘、因子精炼厂、遗传规划挖掘、Vibe Trading
- 🧱 因子体系：体系搭建、体系回测分析、系统因子库、因子监控
- 📈 数据中心：行情中心、期货 & 期权、基金行情、债券 / 外汇
- 🧠 智能分析：非结构化数据、Transformer 分析、知识库
- ⚙ 系统：产品交付、运行配置

核心能力：
- 因子体系搭建：把挖掘产出与系统因子库中的因子组装成带维度与权重的因子体系，
  一键执行合成回测并输出 IC / ICIR / 分层 / 相关性 / 主成分 / 衰减全景诊断。
- 操作记忆：界面选择、操作日志、挖掘产出、因子体系与回测结果全部落盘到本地
  SQLite（``data/factorgpt.db``），关闭应用后重新打开自动恢复现场。
- 模型可切换：侧边栏支持 DeepSeek / OpenAI / Qwen / 任意 OpenAI 兼容端点
  （含本地 Ollama、vLLM、OpenRouter 等）。
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd
import yaml

# 允许以 `python -m ui.app` 或 `streamlit run src/ui/app.py` 两种方式运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.graph import FactorAgent   # noqa: E402
from agent.vibe_trading import VibeTradingSession  # noqa: E402
from agent.integration import (
    get_library, get_coupling, get_unstructured_manager, get_text_analyzer,
    query_to_factor_suggestions, build_enriched_knowledge,
    mass_produce_from_library, analyze_unstructured_file,
)
from ui.methodologist import run_methodologist, get_factor_name_from_report  # noqa: E402
from ui.market_hub import render_market_hub  # noqa: E402
from ui import nav, theme  # noqa: E402
from ui.factor_system import render_system_analysis, render_system_builder  # noqa: E402
from rag.chroma_store import ensure_chroma  # noqa: E402
from rag.retriever import rag_vector_enabled  # noqa: E402
from store import chats as chat_repo  # noqa: E402
from store import database as db  # noqa: E402
from store import mining as mining_repo  # noqa: E402
from store import ops as ops_repo  # noqa: E402
from store import runs as runs_repo  # noqa: E402
from store import state as state_repo  # noqa: E402
from store import systems as systems_repo  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"


# ----------------------------------------------------------------------
# 供应商预设（OpenAI 兼容接口）
# ----------------------------------------------------------------------
PROVIDER_PRESETS = {
    # 官方文档推荐 base_url（OpenAI 兼容，不带 /v1）；模型名以官方当前 V4 系列为准。
    "DeepSeek": {"provider": "deepseek", "base_url": "https://api.deepseek.com", "model": "deepseek-v4-flash"},
    "OpenAI": {"provider": "openai", "base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "通义千问 (Qwen)": {"provider": "qwen", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus"},
    "OpenAI 兼容 (自定义)": {"provider": "custom", "base_url": "", "model": ""},
}
PROVIDER_NAME_MAP = {
    "deepseek": "DeepSeek",
    "openai": "OpenAI",
    "qwen": "通义千问 (Qwen)",
    "custom": "OpenAI 兼容 (自定义)",
}


# ----------------------------------------------------------------------
# 配置加载 & 资源缓存
# ----------------------------------------------------------------------
@st.cache_resource
def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


@st.cache_resource
def get_agent():
    """缓存单例 FactorAgent（数据加载较重，仅构建一次）。模型切换通过
    `agent.update_llm()` 在运行时完成，不会触发重加载。"""
    cfg = load_config()
    agent = FactorAgent(cfg)
    return agent


# ----------------------------------------------------------------------
# 模型 / API 会话状态
# ----------------------------------------------------------------------
def _init_llm_session():
    if "llm_cfg" not in st.session_state:
        llm = load_config().get("llm", {})
        st.session_state.llm_cfg = {
            "provider": llm.get("provider", "deepseek"),
            "model": llm.get("model", "deepseek-chat"),
            "api_key": llm.get("api_key", ""),
            "base_url": llm.get("base_url", "https://api.deepseek.com/v1"),
            "temperature": float(llm.get("temperature", 0.3)),
        }
    s = st.session_state.llm_cfg
    if "ui_api_key" not in st.session_state:
        st.session_state.ui_api_key = s.get("api_key", "")
    if "ui_base_url" not in st.session_state:
        st.session_state.ui_base_url = s.get("base_url", "")
    if "ui_model" not in st.session_state:
        st.session_state.ui_model = s.get("model", "")
    if "ui_temp" not in st.session_state:
        st.session_state.ui_temp = s.get("temperature", 0.3)
    if "ui_provider_value" not in st.session_state:
        st.session_state.ui_provider_value = s.get("provider", "deepseek")
    if "ui_provider" not in st.session_state:
        st.session_state.ui_provider = PROVIDER_NAME_MAP.get(
            st.session_state.ui_provider_value, "OpenAI 兼容 (自定义)"
        )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def _on_provider_change():
    sel = st.session_state.get("ui_provider")
    preset = PROVIDER_PRESETS.get(sel)
    if preset:
        st.session_state.ui_provider_value = preset["provider"]
        st.session_state.ui_base_url = preset["base_url"]
        st.session_state.ui_model = preset["model"]


def _render_model_panel():
    """侧边栏「模型 / API 设置」面板：用户通过 API Key 切换模型 / 供应商。"""
    active = st.session_state.llm_cfg
    with st.sidebar.expander("⚙️ 模型 / API 设置", expanded=not bool(active.get("api_key"))):
        st.caption("切换到其他模型或 OpenAI 兼容端点（如本地 Ollama、vLLM、OpenRouter）。")
        names = list(PROVIDER_PRESETS.keys())
        idx = names.index(st.session_state.get("ui_provider", "DeepSeek")) if st.session_state.get("ui_provider") in names else 0
        st.selectbox("供应商", names, index=idx, key="ui_provider", on_change=_on_provider_change)
        st.text_input("API Key", type="password", key="ui_api_key",
                      help="留空则使用自定义端点（如本地无鉴权服务）。")
        st.text_input("Base URL", key="ui_base_url",
                      help="OpenAI 兼容接口地址；自定义端点留空使用默认。")
        st.text_input("模型名称", key="ui_model", help="如 deepseek-chat / gpt-4o / qwen-plus。")
        st.slider("温度", 0.0, 1.0, step=0.05, key="ui_temp")

        c1, c2, c3 = st.columns(3)
        with c1:
            apply = st.button("应用配置", key="btn_apply", width='stretch')
        with c2:
            test = st.button("测试连接", key="btn_test", width='stretch')
        with c3:
            save = st.button("保存配置", key="btn_save", width='stretch')

        if apply:
            st.session_state.llm_cfg = {
                "provider": st.session_state.ui_provider_value,
                "model": st.session_state.ui_model,
                "api_key": st.session_state.ui_api_key,
                "base_url": st.session_state.ui_base_url,
                "temperature": st.session_state.ui_temp,
            }
            st.success("已应用：下一轮对话 / 挖掘将使用新模型。")
        if test:
            _test_connection()
        if save:
            _save_llm_to_config()

    # 当前生效模型提示
    st.sidebar.caption(
        f"当前模型：**{active.get('model')}**  ({PROVIDER_NAME_MAP.get(active.get('provider'), active.get('provider'))})"
    )


def _test_connection():
    from llm.client import LLMClient

    try:
        c = LLMClient()
        # 防御：清除 Base URL 头尾混入的制表符/空格（常因从文档复制粘贴带入，
        # 会导致 urllib 报 "Invalid non-printable ASCII character in URL"）。
        base_url = (st.session_state.ui_base_url or "").strip()
        c.set_model(
            provider=st.session_state.ui_provider_value,
            model=st.session_state.ui_model,
            api_key=st.session_state.ui_api_key,
            base_url=base_url,
            temperature=st.session_state.ui_temp,
        )
        resp = c.chat([{"role": "user", "content": "ping，只回复 ok"}])
        st.success(f"连接成功 ✅ 模型回复：{str(resp)[:80]}")
    except Exception as e:
        msg = str(e)
        if "404" in msg:
            st.error(
                f"连接失败 ❌ 404：请确认 Base URL 为 OpenAI 兼容端点 "
                f"(DeepSeek 官方为 https://api.deepseek.com，不要填 /anthropic)，"
                f"且模型名存在。原始错误：{msg}"
            )
        else:
            st.error(f"连接失败 ❌ {msg}")


def _save_llm_to_config():
    try:
        data = yaml.safe_load(open(CONFIG_PATH, "r", encoding="utf-8")) or {}
        data.setdefault("llm", {})
        data["llm"].update({
            "provider": st.session_state.ui_provider_value,
            "model": st.session_state.ui_model,
            "api_key": st.session_state.ui_api_key,
            "base_url": st.session_state.ui_base_url,
            "temperature": st.session_state.ui_temp,
        })
        yaml.safe_dump(data, open(CONFIG_PATH, "w", encoding="utf-8"),
                       allow_unicode=True, sort_keys=False)
        st.success("已写入 config.yaml（含 API Key，请注意本地保密）。")
    except Exception as e:
        st.error(f"保存失败：{e}")


def _init_data_source_session():
    """初始化数据源会话状态：初值全部取自 config.yaml 的 data 段（保留默认）。

    仅当用户在面板中主动「应用」或「保存」时才会覆写对应值，未调整则维持原默认。
    """
    data = load_config().get("data", {}) or {}
    defaults = {
        "ui_ds_source": "legacy",
        "ui_ds_primary": "akshare",
        "ui_ds_prefer_sina": True,
        "ui_ds_tushare_token": "",
        "ui_ds_neodata_base_url": "",
        "ui_ds_ths_base_url": "",
        "ui_ds_ths_token": "",
        "ui_ds_synthetic_on_fail": False,
        "ui_ds_force_synthetic": False,
        "ui_ds_proxy_enabled": False,
        "ui_ds_proxy_http": "",
        "ui_ds_proxy_https": "",
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = data.get(key.replace("ui_ds_", ""), default)


def _render_data_source_panel():
    """侧边栏「数据源设置」面板：允许用户自行调整数据源接口。

    未调整任何字段时，所有控件初值来自 config.yaml，等价于保留原默认数据源路径；
    点击「应用配置」即时生效（热更新到运行的 Agent），「保存配置」持久化到 config.yaml。
    """
    _init_data_source_session()
    with st.sidebar.expander("🗄️ 数据源设置", expanded=False):
        st.caption("调整行情/数据接口；不改动则沿用 config.yaml 默认数据源。")
        st.selectbox(
            "数据源开关",
            ["legacy", "neodata"],
            index=0 if st.session_state.ui_ds_source != "neodata" else 1,
            key="ui_ds_source",
            help="legacy=沿用 akshare/sina/tushare 本地自爬；neodata=走平台稳定源（需可用网关）。",
        )
        st.selectbox(
            "主力数据源（legacy）",
            ["akshare", "tushare", "ths", "sina"],
            index=["akshare", "tushare", "ths", "sina"].index(
                st.session_state.ui_ds_primary
            ) if st.session_state.ui_ds_primary in ("akshare", "tushare", "ths", "sina") else 0,
            key="ui_ds_primary",
            help="legacy 模式下的子优先级：akshare / tushare / ths / sina。",
        )
        st.checkbox("优先新浪源 (prefer_sina)", key="ui_ds_prefer_sina",
                    help="部分网络下东方财富连接会被重置，置 true 可让新浪成为首选。")
        st.text_input("Tushare Token", type="password", key="ui_ds_tushare_token",
                      help="legacy=tushare 时填写；留空则忽略。")
        st.text_input("NeoData 网关地址", key="ui_ds_neodata_base_url",
                      help="仅 source=neodata 时生效。")
        st.text_input("同花顺网关端点", key="ui_ds_ths_base_url",
                      help="primary_source=ths 时填写 MCP 端点。")
        st.text_input("同花顺 Token", type="password", key="ui_ds_ths_token",
                      help="primary_source=ths 时填写 JWE 鉴权令牌。")
        st.checkbox("实时源失败自动回退合成数据", key="ui_ds_synthetic_on_fail")
        st.checkbox("强制合成数据（离线）", key="ui_ds_force_synthetic")
        st.divider()
        st.caption("网络代理（访问行情源 / 远程端点）")
        st.checkbox("启用代理", key="ui_ds_proxy_enabled")
        st.text_input("HTTP 代理", key="ui_ds_proxy_http", placeholder="http://127.0.0.1:7890")
        st.text_input("HTTPS 代理", key="ui_ds_proxy_https", placeholder="留空复用 HTTP")

        c1, c2 = st.columns(2)
        with c1:
            apply = st.button("应用配置", key="btn_ds_apply", width="stretch")
        with c2:
            save = st.button("保存配置", key="btn_ds_save", width="stretch")

        if apply:
            _apply_data_source_to_session()
            agent = get_agent()
            agent.reload_data_config()
            st.success("已应用：下一轮取数将使用新数据源设置。")
        if save:
            _save_data_source_to_config()
            agent = get_agent()
            agent.reload_data_config()
            st.success("已保存配置到 config.yaml，并即时生效。")

    st.sidebar.caption(
        f"当前数据源：**{st.session_state.ui_ds_source}**"
        f"（主力：{st.session_state.ui_ds_primary}）"
    )


def _collect_data_source_cfg():
    """从会话状态收集 data 段字段（仅覆盖用户可调项，保留其余默认项）。"""
    return {
        "source": st.session_state.ui_ds_source,
        "primary_source": st.session_state.ui_ds_primary,
        "prefer_sina": bool(st.session_state.ui_ds_prefer_sina),
        "tushare_token": st.session_state.ui_ds_tushare_token or "",
        "neodata": {"base_url": st.session_state.ui_ds_neodata_base_url or ""},
        "ths_api_base_url": st.session_state.ui_ds_ths_base_url or "",
        "ths_api_token": st.session_state.ui_ds_ths_token or "",
        "synthetic_on_fail": bool(st.session_state.ui_ds_synthetic_on_fail),
        "force_synthetic": bool(st.session_state.ui_ds_force_synthetic),
        "proxy": {
            "enabled": bool(st.session_state.ui_ds_proxy_enabled),
            "http": st.session_state.ui_ds_proxy_http or "",
            "https": st.session_state.ui_ds_proxy_https or "",
        },
    }


def _apply_data_source_to_session():
    """把会话状态中的 data 段覆盖到内存 config（不落盘）。"""
    cfg = load_config()
    cfg["data"] = {**(cfg.get("data") or {}), **_collect_data_source_cfg()}
    st.session_state._runtime_data_cfg = _collect_data_source_cfg()


def _save_data_source_to_config():
    """把会话状态中的数据源设置写入 config.yaml 的 data 段（保留其他段与未改字段）。"""
    try:
        data = yaml.safe_load(open(CONFIG_PATH, "r", encoding="utf-8")) or {}
        merged = {**(data.get("data") or {}), **_collect_data_source_cfg()}
        data["data"] = merged
        yaml.safe_dump(data, open(CONFIG_PATH, "w", encoding="utf-8"),
                       allow_unicode=True, sort_keys=False)
    except Exception as e:
        st.error(f"数据源配置保存失败：{e}")


def _apply_model(agent):
    """运行前将 session 中的模型配置同步到 Agent 的 LLM 客户端。"""
    cfg = st.session_state.llm_cfg
    agent.update_llm(
        provider=cfg.get("provider"),
        model=cfg.get("model"),
        api_key=cfg.get("api_key", ""),
        base_url=cfg.get("base_url", ""),
        temperature=cfg.get("temperature", 0.3),
    )


# ----------------------------------------------------------------------
# 结果渲染工具
# ----------------------------------------------------------------------
def _render_metrics_table(metrics: dict):
    rows = []
    for k, v in metrics.items():
        if isinstance(v, bool):
            rows.append((k, str(v)))
        elif isinstance(v, (int, float)):
            rows.append((k, f"{v:.4f}"))
        else:
            rows.append((k, str(v)))
    st.table(pd.DataFrame(rows, columns=["指标", "值"]))


def _render_agent_dict(d: dict, with_method: bool = False):
    st.markdown(d.get("report", ""))
    if d.get("metrics"):
        with st.expander("📊 关键指标", expanded=False):
            _render_metrics_table(d["metrics"])
    if d.get("code"):
        with st.expander("🧮 因子公式代码", expanded=False):
            st.code(d["code"], language="python")
    for fig in d.get("charts") or []:
        try:
            st.pyplot(fig)
        except Exception:
            pass
    if with_method and d.get("method"):
        with st.expander("📋 方法学解读", expanded=False):
            st.markdown(d["method"])


def _build_agent_dict(result: dict, with_method: bool = False) -> dict:
    state = result.get("state", {})
    d = {
        "report": result.get("report", ""),
        "code": state.get("code"),
        "metrics": result.get("metrics", state.get("metrics", {})),
        "charts": state.get("charts") or [],
        "method": None,
    }
    if with_method:
        try:
            name = get_factor_name_from_report(d["report"]) or "复合因子"
            d["method"] = run_methodologist(name, d["report"])
        except Exception:
            d["method"] = None
    return d


# ----------------------------------------------------------------------
# 行情组件：通用表格 / 图表渲染
# ----------------------------------------------------------------------
def _display_table(df, err, key, height=430):
    """渲染行情 DataFrame：带筛选框、空数据提示、CSV 下载。

    返回 ``(display_df, has_error)``，便于调用方在需要时继续基于数据绘图。
    """
    if err:
        st.error(err)
        return df, True
    if df is None or df.empty:
        st.info("暂无数据（可能处于非交易时段，或接口返回为空）。可稍后点击刷新重试。")
        return df, False
    q = st.text_input("筛选（匹配任意列文本）", key=f"{key}_q",
                      placeholder="输入关键字过滤，如 螺纹 / 510050 / 黄金")
    show = df
    if q:
        mask = df.astype(str).apply(
            lambda r: r.str.contains(q, case=False, na=False).any(), axis=1
        )
        show = df[mask]
    st.dataframe(show, width='stretch', height=height)
    csv = show.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下载 CSV", csv, file_name=f"{key}.csv", key=f"{key}_dl")
    return show, False


def _line_chart(df, key, value_hints=("收盘", "close", "净值"), date_hints=("日期", "date")):
    """对含日期/数值列的行情 DataFrame 画交互式折线图（plotly）。"""
    if df is None or df.empty:
        return
    date_col = next(
        (c for c in df.columns if any(h.lower() in str(c).lower() for h in date_hints)), None
    )
    val_col = next(
        (c for c in df.columns if any(h.lower() in str(c).lower() for h in value_hints)), None
    )
    if date_col is None or val_col is None:
        return
    import plotly.express as px

    d = df[[date_col, val_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col, val_col])
    if d.empty:
        return
    fig = px.line(d, x=date_col, y=val_col, title=f"{key}：{val_col}")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=360)
    st.plotly_chart(fig, width='stretch')


def _candlestick_chart(df, key, date_hints=("日期", "date"),
                       o_hints=("开盘", "open"), h_hints=("最高", "high"),
                       l_hints=("最低", "low"), c_hints=("收盘", "close")):
    """对含 OHLC 列的行情 DataFrame 画交互式 K 线（蜡烛图，plotly）。"""
    if df is None or df.empty:
        return
    def _col(hints):
        return next((c for c in df.columns
                     if any(h.lower() in str(c).lower() for h in hints)), None)
    date_col = _col(date_hints)
    o, h, l, c = _col(o_hints), _col(h_hints), _col(l_hints), _col(c_hints)
    if None in (date_col, o, h, l, c):
        return
    import plotly.graph_objects as go

    d = df[[date_col, o, h, l, c]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    for col in (o, h, l, c):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=[o, h, l, c])
    if d.empty:
        return
    fig = go.Figure(data=[go.Candlestick(
        x=d[date_col], open=d[o], high=d[h], low=d[l], close=d[c],
        increasing_line_color="#ef232a", decreasing_line_color="#14b143",
        name=key,
    )])
    fig.update_layout(
        title=f"{key} K 线",
        margin=dict(l=20, r=20, t=40, b=20), height=420,
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, width='stretch')


# ----------------------------------------------------------------------
# 页面：概览
# ----------------------------------------------------------------------
def render_overview():
    """工作台首页：研究资产总览 + 研究闭环 + 快捷入口。"""
    # —— 研究资产统计（全部来自本地数据库与因子库）——
    try:
        lib_stats = get_library().statistics()
        n_lib = int(lib_stats.get("total", 0))
    except Exception:
        lib_stats, n_lib = {}, 0
    try:
        sys_list = systems_repo.list()
        n_mining = len(mining_repo.list(limit=1000))
        recent_ops = ops_repo.recent(limit=8)
        run_list = runs_repo.list(limit=200)
    except Exception:
        sys_list, n_mining, recent_ops, run_list = [], 0, [], []

    theme.kpi_row([
        {"label": "因子库总量", "value": n_lib, "sub": "系统 + 挖掘 + 自定义"},
        {"label": "挖掘沉淀", "value": n_mining, "sub": "已落盘的挖掘产出", "tone": "ink"},
        {"label": "因子体系", "value": len(sys_list), "sub": "已保存可回测体系", "tone": "pos"},
        {"label": "体系回测", "value": len(run_list), "sub": "历史回测记录", "tone": "neutral"},
        {"label": "操作记录", "value": sum(ops_repo.module_counts().values()) if recent_ops else 0,
         "sub": "本地操作时间线", "tone": "warn"},
    ], columns=5)

    theme.section("研究闭环", "从想法到可交付策略的标准路径，每一步的产出都会自动落盘")
    theme.steps([
        {"n": "01", "title": "挖掘因子", "desc": "Agent / 遗传规划 / 精炼厂产出候选因子"},
        {"n": "02", "title": "搭建体系", "desc": "挑选因子、归类维度、配置权重"},
        {"n": "03", "title": "回测诊断", "desc": "IC / 分层 / 相关性 / 主成分全景分析"},
        {"n": "04", "title": "监控交付", "desc": "上线跟踪衰减并打包交付物"},
    ])

    c1, c2 = st.columns([1.45, 1])
    with c1:
        theme.section("快捷入口", "点击直达高频功能")
        nav.quick_links(["agent", "sys_build", "sys_analysis", "market"], columns=4)
        nav.quick_links(["refinery", "gp", "library", "memory"], columns=4)

        if sys_list:
            theme.section("最近的因子体系", "按更新时间排序")
            st.dataframe(
                pd.DataFrame([{
                    "体系名称": s["name"],
                    "因子数": s["n_factors"],
                    "更新时间": s["updated_at"],
                } for s in sys_list[:6]]),
                hide_index=True, use_container_width=True,
            )
        else:
            theme.insight(
                "还没有因子体系。先到「因子挖掘」产出候选因子，再到"
                "「因子体系 → 体系搭建」组装第一个体系。", "info",
            )
    with c2:
        theme.section("最近操作", "本地数据库中的操作时间线")
        if recent_ops:
            for op in recent_ops:
                st.markdown(
                    f'<div style="padding:7px 0;border-bottom:1px solid {theme.LINE};font-size:12.5px">'
                    f'<span style="color:{theme.INK_MUTED}">{op["ts"][5:16]}</span>　'
                    f'{theme.badge(op["module"], "gray")}'
                    f'<span style="color:{theme.INK}">{op["summary"] or op["action"]}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            theme.empty_state("暂无操作记录", "开始使用后这里会记录你的每一步研究动作。", "◌")

    theme.footer(
        f'<b>数据落盘</b>：{db.get_db_path()}　·　体积 {db.db_size_kb()} KB<br>'
        "所有界面状态与研究产出均保存在本地，不依赖任何云端服务，便于离线开发与版本管理。"
    )


# ----------------------------------------------------------------------
# 页面：操作记忆中心
# ----------------------------------------------------------------------
def render_memory():
    """展示并管理本地数据库中的「操作记忆」。"""
    stats = db.db_stats()
    theme.kpi_row([
        {"label": "操作日志", "value": stats.get("operation_log", 0)},
        {"label": "界面状态", "value": stats.get("app_state", 0), "tone": "ink"},
        {"label": "挖掘记录", "value": stats.get("mining_records", 0), "tone": "neutral"},
        {"label": "因子体系", "value": stats.get("factor_systems", 0), "tone": "pos"},
        {"label": "回测记录", "value": stats.get("backtest_runs", 0), "tone": "warn"},
        {"label": "对话消息", "value": stats.get("chat_messages", 0), "tone": "ink"},
    ], columns=6)
    theme.insight(
        f'数据库文件：<b>{db.get_db_path()}</b>（{db.db_size_kb()} KB）。'
        "该文件可直接备份、随项目迁移，或用任意 SQLite 客户端打开做二次分析。", "info",
    )

    t1, t2, t3, t4 = st.tabs(["操作时间线", "挖掘沉淀", "回测记录", "数据维护"])

    with t1:
        counts = ops_repo.module_counts()
        if counts:
            theme.badges([f"{k} · {v}" for k, v in counts.items()], "gray")
        modules = ["全部"] + list(counts.keys())
        c1, c2 = st.columns([1.2, 3])
        with c1:
            pick = st.selectbox("模块筛选", modules, key="mem_mod")
        rows = ops_repo.recent(limit=300, module=None if pick == "全部" else pick)
        if rows:
            st.dataframe(
                pd.DataFrame([{
                    "时间": r["ts"], "模块": r["module"], "动作": r["action"],
                    "说明": r["summary"], "状态": r["status"],
                } for r in rows]),
                hide_index=True, use_container_width=True, height=440,
            )
        else:
            theme.empty_state("暂无操作日志", "", "◌")

    with t2:
        recs = mining_repo.list(limit=300)
        if recs:
            st.dataframe(
                pd.DataFrame([{
                    "时间": r["ts"], "来源": r["module"], "因子名": r["factor_name"],
                    "IC": round(float(r["metrics"].get("ic", 0) or 0), 4),
                    "ICIR": round(float(r["metrics"].get("icir", 0) or 0), 3),
                    "需求/表达式": (r["query"] or r["expression"])[:70],
                } for r in recs]),
                hide_index=True, use_container_width=True, height=440,
            )
            theme.insight("这些因子会自动出现在「因子体系 → 体系搭建」的候选因子池中。", "ok")
        else:
            theme.empty_state("暂无挖掘沉淀", "运行 Agent / 遗传规划 / 精炼厂后自动记录。", "◌")

    with t3:
        rl = runs_repo.list(limit=200)
        if rl:
            st.dataframe(
                pd.DataFrame([{
                    "时间": r["created_at"], "体系": r["system_name"],
                    "股票池": r["universe"], "区间": r["period"],
                    "IC": round(float(r["metrics"].get("ic", 0) or 0), 4),
                    "ICIR": round(float(r["metrics"].get("icir", 0) or 0), 3),
                    "多空夏普": round(float(r["metrics"].get("long_short_sharpe", 0) or 0), 2),
                } for r in rl]),
                hide_index=True, use_container_width=True, height=440,
            )
        else:
            theme.empty_state("暂无回测记录", "在「体系回测分析」运行后自动记录。", "◌")

    with t4:
        theme.section("清理与导出", "谨慎操作：清理后无法恢复，建议先备份数据库文件")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("清空操作日志", use_container_width=True):
                ops_repo.clear()
                st.success("已清空操作日志")
                st.rerun()
        with c2:
            if st.button("清空挖掘沉淀", use_container_width=True):
                mining_repo.clear()
                st.success("已清空挖掘沉淀")
                st.rerun()
        with c3:
            if st.button("重置界面记忆", use_container_width=True,
                         help="清除记住的筛选条件、页面位置等，不影响因子体系与回测结果"):
                state_repo.clear()
                st.success("已重置界面记忆")
                st.rerun()

        saved = state_repo.all()
        if saved:
            with st.expander(f"当前记住的界面状态（{len(saved)} 项）"):
                st.json({k: v for k, v in list(saved.items())[:40]})


# ----------------------------------------------------------------------
# 页面：单次因子挖掘（Agent）
# ----------------------------------------------------------------------
def render_factor_agent():
    active = st.session_state.llm_cfg
    st.caption(
        f"模型：**{active.get('model')}** · 输入自然语言需求，Agent 自动检索/生成/回测/反思。"
    )

    user_input = st.text_area(
        "因子需求描述",
        value="混合日频与月频，结合短期反转与流动性，构建低估值质量因子",
        height=90,
    )
    auto_method = st.checkbox("自动生成方法学解读", value=True)
    run_btn = st.button("🚀 运行因子挖掘", type="primary")

    if run_btn and user_input.strip():
        agent = get_agent()
        _apply_model(agent)
        with st.spinner("Agent 正在挖掘因子（可能多轮反思）..."):
            result = agent.run(user_input, max_iterations=None)
        d = _build_agent_dict(result, with_method=auto_method)
        _render_agent_dict(d, with_method=auto_method)


# ----------------------------------------------------------------------
# 页面：Agent 对话（多轮）
# ----------------------------------------------------------------------
def render_agent_chat():
    st.caption("多轮对话式因子挖掘：输入需求，Agent 自动检索知识、生成/回测因子并反思迭代。")
    active = st.session_state.llm_cfg
    st.caption(f"当前模型：**{active.get('model')}**")

    if st.button("🧹 清空对话"):
        st.session_state.chat_history = []
        st.rerun()

    auto_method = st.checkbox("每条回复自动附方法学解读", value=True, key="chat_method")

    # 渲染历史
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                _render_agent_dict(msg.get("agent", {}), with_method=bool(msg.get("agent", {}).get("method")))

    if prompt := st.chat_input("描述你想要的因子，例如：低估值且现金流稳健的反转因子"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        agent = get_agent()
        _apply_model(agent)
        with st.spinner("Agent 正在挖掘因子..."):
            result = agent.run(prompt, max_iterations=None)
        d = _build_agent_dict(result, with_method=auto_method)
        st.session_state.chat_history.append({"role": "assistant", "agent": d})
        st.rerun()


# ----------------------------------------------------------------------
# 页面：因子精炼厂
# ----------------------------------------------------------------------
def render_refinery():
    st.caption("RL(MaskablePPO) 因子组合搜索 + RAG 知识 + Transformer 编码 的复合因子管线。")
    try:
        from pipeline.refinery import RefineryPipeline
    except Exception as e:
        st.error(f"精炼厂模块加载失败：{e}")
        return

    cfg = load_config().get("refinery", {})
    mode = st.radio(
        "运行模式",
        ["无人值守（一键跑完六道工序）", "人机协同评审（PART-04 第二级人工介入）"],
        horizontal=True,
        help="人机协同模式会在 PART-03 评估完成后暂停，由研究员在下方表格勾选保留/剔除，"
             "评审结果真实作用于 PART-04 第二级筛选，再续跑 PART-05/06。",
    )
    human_mode = mode.startswith("人机协同")

    with st.form("refinery_form"):
        desc = st.text_input("复合因子目标描述", value="混合日频与月频，结合短期反转与流动性")
        seed = st.number_input("基础因子候选取样数", min_value=4, max_value=40,
                               value=int(cfg.get("n_pool_seed", 12)))
        cand = st.number_input("RL 候选因子数", min_value=1, max_value=30,
                               value=int(cfg.get("rl_candidates", 6)))
        backend = st.selectbox("RL 后端", ["auto", "sb3", "heuristic"],
                               index=["auto", "sb3", "heuristic"].index(cfg.get("rl_backend", "auto")))
        submitted = st.form_submit_button(
            "🔎 生成并评估候选（进入评审）" if human_mode else "🚀 运行精炼厂", type="primary")

    def _build_pipe():
        from pipeline.refinery import RefineryPipeline, build_refinery_config

        merged = dict(cfg)
        merged.update({"n_pool_seed": seed, "rl_candidates": cand, "rl_backend": backend})
        return RefineryPipeline(build_refinery_config(merged))

    if submitted and desc.strip():
        # 清空上一轮评审态，避免参数变更后仍沿用旧候选
        st.session_state.pop("refinery_ctx", None)
        st.session_state.pop("refinery_pipe", None)
        st.session_state.pop("refinery_result", None)
        if human_mode:
            with st.spinner("PART-01~03 运行中（矿石 → 三维生成 → RPN 评估）..."):
                try:
                    pipe = _build_pipe()
                    ctx = pipe.run_to_review(desc)
                except Exception as e:
                    st.exception(e)
                    return
            st.session_state["refinery_pipe"] = pipe
            st.session_state["refinery_ctx"] = ctx
        else:
            with st.spinner("精炼厂运行中（RL 训练 + RAG + Transformer）..."):
                try:
                    st.session_state["refinery_result"] = _build_pipe().run(desc)
                except Exception as e:
                    st.exception(e)
                    return

    # ── 人机协同评审面板（PART-04 第二级）────────────────────────────────
    ctx = st.session_state.get("refinery_ctx")
    if ctx is not None and st.session_state.get("refinery_result") is None:
        st.subheader("🧑‍🔬 PART-04 第二级 · 人机协同评审")
        st.caption("勾选「保留」的因子将进入 TOP-K 截断与 AlphaPool 合成；取消勾选即被人工剔除。"
                   "评审结果会写入方法学报告的审计留痕。")
        rows = []
        for c in ctx.candidates:
            m = c.metrics or {}
            rows.append({
                "保留": True,
                "因子": c.name,
                "来源": c.source,
                "ICIR": round(float(m.get("icir", 0) or 0), 4),
                "RankIC均值": round(float(m.get("rank_ic_mean", m.get("ic_mean", 0)) or 0), 4),
                "稳定性": round(float(m.get("stability_score", 0) or 0), 4),
                "换手": round(float(m.get("turnover", 0) or 0), 4),
                "说明": (c.description or c.rationale or "")[:60],
            })
        df_review = pd.DataFrame(rows).sort_values("ICIR", ascending=False, ignore_index=True)
        edited = st.data_editor(
            df_review,
            use_container_width=True,
            hide_index=True,
            disabled=["因子", "来源", "ICIR", "RankIC均值", "稳定性", "换手", "说明"],
            column_config={"保留": st.column_config.CheckboxColumn("保留", help="取消勾选表示人工剔除")},
            key="refinery_review_editor",
        )
        keep_names = [str(r["因子"]) for _, r in edited.iterrows() if bool(r["保留"])]
        st.write(f"当前保留 **{len(keep_names)}** / {len(edited)} 个候选因子")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            go = st.button("✅ 按我的评审继续冶炼（PART-04~06）", type="primary")
        with col_b:
            if st.button("↩️ 放弃本次评审"):
                st.session_state.pop("refinery_ctx", None)
                st.session_state.pop("refinery_pipe", None)
                st.rerun()
        if go:
            if not keep_names:
                st.warning("至少需保留 1 个因子，否则无法进行 AlphaPool 合成。")
            else:
                with st.spinner("PART-04~06 运行中（三级筛选 → 合金配比 → 方法学报告）..."):
                    try:
                        pipe = st.session_state["refinery_pipe"]
                        st.session_state["refinery_result"] = pipe.resume_from_review(
                            ctx, keep_names=keep_names)
                    except Exception as e:
                        st.exception(e)
                        return
                st.session_state.pop("refinery_ctx", None)
                st.rerun()

    # ── 结果展示 ──────────────────────────────────────────────────────
    result = st.session_state.get("refinery_result")
    if result is not None:
        st.success(f"精炼厂完成：候选 {len(result.candidates)} → 入选 {len(result.screened)}")
        ic = result.composite_metrics.get("icir")
        st.metric("复合 ICIR", f"{ic:.4f}" if isinstance(ic, (int, float)) else str(ic))
        audit = getattr(result, "screen_audit", None)
        if audit:
            with st.expander("三级筛选审计留痕", expanded=False):
                st.json(audit)
        if result.composite is not None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(range(len(result.composite)), result.composite.values)
            ax.set_title("Composite Factor Series")
            ax.set_xlabel("sample")
            ax.set_ylabel("value")
            st.pyplot(fig)
        if result.report_path:
            st.info(f"方法学报告已生成：{result.report_path}")
        with st.expander("阶段耗时明细", expanded=False):
            st.table(pd.DataFrame(result.stage_trace))


# ----------------------------------------------------------------------
# 页面：知识库
# ----------------------------------------------------------------------
def render_kb():
    cfg = load_config()
    use_vs = rag_vector_enabled(cfg)
    if use_vs:
        with st.spinner("首次使用正在自动下载向量模型（BGE），请稍候…"):
            try:
                store = ensure_chroma(build=True)
            except Exception as e:
                st.error(f"Chroma 初始化失败：{e}")
                return
    else:
        try:
            store = ensure_chroma(build=False)
        except Exception as e:
            st.error(f"Chroma 初始化失败：{e}")
            return

    tab1, tab2 = st.tabs(["检索", "上传"])
    if use_vs and not store.available:
        st.warning("向量检索依赖不可用，已降级为本地关键词检索模式（编辑 config.yaml 关掉 rag.use_vector_store 可彻底关闭）。")
    elif not use_vs:
        st.info(
            "当前为离线关键词检索模式（rag.use_vector_store=false 或依赖未安装）："
            "基于 jieba + TF-IDF，秒级响应、无需下载模型。"
        )
    # 知识库状态提示：空库时明确告知用户需要先建库/上传，避免“检索不出内容”却无提示
    if use_vs and store.available:
        try:
            n_docs = store.count()
        except Exception:
            n_docs = 0
        if n_docs == 0:
            st.warning(
                "知识库当前为空：内置因子语料尚未写入或上传的文档未持久化。"
                "请点击「上传」标签添加文档，或确认 config.yaml 的 rag.use_vector_store=true "
                "且 BGE 向量模型可下载（国内已自动走 hf-mirror.com 镜像）。"
            )
        else:
            st.caption(f"向量知识库当前共有 {n_docs} 条文档。")
    with tab1:
        q = st.text_input("检索查询")
        if q:
            try:
                if use_vs and store.available:
                    hits = store.query(q, top_k=5)
                else:
                    from rag.retriever import SimpleRetriever

                    hits = SimpleRetriever().retrieve(q, top_k=5)
                if not hits:
                    st.info("未检索到相关内容。")
                for h in hits:
                    st.markdown(f"- {h}")
            except Exception as e:
                st.error(str(e))
    with tab2:
        uploaded = st.file_uploader("上传因子研究文档 (.txt/.md)", type=["txt", "md"])
        if uploaded is not None:
            text = uploaded.read().decode("utf-8", errors="ignore")
            try:
                if use_vs and store.available:
                    n = store.add_texts([text], [{"source": uploaded.name}])
                    if n > 0:
                        st.success("已加入向量知识库。")
                    else:
                        st.warning("向量库不可用，内容未持久化（请安装 chromadb）。")
                else:
                    st.warning(
                        "离线模式下上传不持久化；如需持久化请开启 config.yaml 的 "
                        "rag.use_vector_store=true 并安装 sentence-transformers。"
                    )
            except Exception as e:
                st.error(str(e))


# ----------------------------------------------------------------------
# 页面：配置
# ----------------------------------------------------------------------
def render_config():
    st.markdown("运行时配置（也可直接编辑 `config.yaml`）。")
    cfg = load_config()
    st.json({k: (v if k != "llm" else {**v, "api_key": "***" if v.get("api_key") else ""})
             for k, v in cfg.items()}, expanded=True)


# ----------------------------------------------------------------------
# 页面：产品交付
# ----------------------------------------------------------------------
def render_delivery():
    st.caption("一键导出因子表达式 + 调仓清单 CSV + 可解释 HTML/PDF 报告，并与中证800等权基准对比。")

    cfg = load_config().get("refinery", {})
    force_synthetic = st.checkbox(
        "离线演示（合成数据，跳过网络）", value=not bool(cfg.get("use_real_data")),
        help="勾选后使用可复现合成数据，避免联网与 akshare 依赖；取消则按 config 跑真实数据。",
    )

    if st.button("🚀 运行精炼厂并生成交付物", type="primary"):
        with st.spinner("精炼厂运行中（生成因子 + 组合回测 + 过拟合检验 + 导出）..."):
            try:
                from pipeline.refinery import RefineryPipeline, build_refinery_config

                merged = dict(cfg)
                if force_synthetic:
                    merged.update({"use_real_data": False, "cache_only": False,
                                   "multimodal": False})
                rcfg = build_refinery_config(merged)
                pipe = RefineryPipeline(rcfg)
                result = pipe.run("混合日频与月频，结合短期反转与流动性")
            except Exception as e:
                st.exception(e)
                return

        cm = result.composite_metrics or {}
        pm = (result.portfolio or {}).get("metrics", {}) or {}
        bench = result.benchmark_comparison or {}
        rb = result.robustness or {}
        zoo = result.factor_zoo or {}
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("复合 ICIR", f"{cm.get('icir', 0):.3f}")
        c2.metric("组合年化", f"{pm.get('ann_return', 0):.2%}")
        c3.metric("组合夏普", f"{pm.get('sharpe', 0):.2f}")
        c4.metric("最大回撤", f"{pm.get('max_drawdown', 0):.2%}")
        c5.metric("基准信息比率", f"{bench.get('benchmark_info_ratio', 0):.2f}")
        with st.expander("过拟合检验 / 因子动物园", expanded=False):
            st.markdown(
                f"- 过拟合 verdict：**{rb.get('verdict', '—')}** ｜ "
                f"Deflated Sharpe Ratio：{rb.get('deflated_sharpe_ratio', '—')}"
            )
            st.markdown(
                f"- 因子动物园增量 ICIR：**{zoo.get('incremental_icir', '—')}** ｜ "
                f"与最强基准最大相关性：{zoo.get('max_abs_corr', '—')}"
            )
            if result.portfolio:
                st.markdown("组合约束：" + _assumptions_summary(result))
        st.success(f"交付物已生成到 `{rcfg.output_dir}/`")

    # ---- 历史交付物下载（取 output 目录最新文件） ----
    st.divider()
    st.subheader("📁 交付物下载（最新一次导出）")
    out_dir = Path(cfg.get("output_dir", "output"))
    if not out_dir.exists():
        st.info("尚未生成交付物，请先点击上方按钮运行精炼厂。")
        return
    patterns = {
        "可解释 HTML 报告": "report_*.html",
        "PDF 报告": "report_*.pdf",
        "因子表达式 CSV": "factors_*.csv",
        "调仓清单 CSV": "rebalance_*.csv",
        "结构化 JSON": "meta_*.json",
    }
    for label, pat in patterns.items():
        matches = sorted(out_dir.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            f = matches[0]
            data = f.read_bytes()
            st.download_button(label, data, file_name=f.name, key=f"dl_{f.name}",
                               mime="application/octet-stream")

    # ---- HTML 报告预览（内联渲染） ----
    html_matches = sorted(out_dir.glob("report_*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    if html_matches:
        with st.expander("🖥 报告预览", expanded=False):
            html = html_matches[0].read_text(encoding="utf-8")
            try:
                st.html(html, height=900, scrolling=True)
            except (AttributeError, TypeError):
                st.components.v1.html(html, height=900, scrolling=True)


def _assumptions_summary(result) -> str:
    a = (result.portfolio or {}).get("assumptions")
    if not a:
        return "（组合回测未启用）"
    return (f"T+1={a.get('t_plus_one')}；allow_short={a.get('allow_short')}；"
            f"涨跌停阈值={a.get('limit_up_pct')}；佣金={a.get('commission')}；"
            f"成本模式={a.get('cost_mode')}；流动性门槛={a.get('min_daily_amount')}；"
            f"选股比例={a.get('top_frac')}")


# ----------------------------------------------------------------------
# 页面：期货 & 期权
# ----------------------------------------------------------------------
def render_futures_options():
    st.caption("期货主力实时行情、期货/期权 K 线、上交所期权与商品期权合约链（数据：AKShare）。")
    from data.market_data import MarketDataFetcher, FUTURES_MAIN_HINTS, FUTURES_SYMBOLS

    tab1, tab2, tab3, tab4 = st.tabs(
        ["期货主力实时", "期货 K 线", "上交所期权实时", "商品期权合约链"]
    )
    with tab1:
        if st.button("刷新期货主力行情", key="fut_refresh", width='stretch'):
            df, err = MarketDataFetcher.futures_main_spot()
            _display_table(df, err, "fut_main")
    with tab2:
        col1, col2 = st.columns([2, 1])
        sym = col1.text_input("主力合约代码", value="RB0", key="fut_sym",
                              help="常见代码：" + " ".join(FUTURES_SYMBOLS[:14]) + " …")
        days = col2.number_input("回溯天数", min_value=30, max_value=720,
                                 value=180, key="fut_days")
        if st.button("绘制期货 K 线", key="fut_k", width='stretch'):
            df, err = MarketDataFetcher.futures_kline(sym.strip().upper(), days=int(days))
            if err:
                st.error(err)
            elif df.empty:
                st.info("暂无数据（非交易时段或代码无效）。")
            else:
                _line_chart(df, f"期货 {sym}")
                _display_table(df, None, "fut_kline")
    with tab3:
        if st.button("刷新上交所期权行情", key="opt_sse", width='stretch'):
            df, err = MarketDataFetcher.option_sse_spot()
            _display_table(df, err, "opt_sse")
    with tab4:
        underlying = st.text_input("商品期权标的", value="黄金期权", key="opt_und",
                                   help="如 黄金期权 / 白糖期权 / 豆粕期权 / 铜期权")
        if st.button("查询商品期权合约链", key="opt_chain", width='stretch'):
            df, err = MarketDataFetcher.option_commodity_chain(underlying.strip())
            if err:
                st.error(err)
            elif df.empty:
                st.info("暂无数据（非交易时段或标的名称无效，可参考交易所命名）。")
            else:
                _display_table(df, None, "opt_chain")


# ----------------------------------------------------------------------
# 页面：基金行情
# ----------------------------------------------------------------------
def render_funds():
    st.caption("ETF/LOF 实时行情、基金净值走势、开放基金排行（数据：AKShare）。")
    from data.market_data import MarketDataFetcher

    tab1, tab2, tab3, tab4 = st.tabs(
        ["ETF 实时", "LOF 实时", "ETF K 线 / 净值", "开放基金排行"]
    )
    with tab1:
        if st.button("刷新 ETF 实时行情", key="etf_refresh", width='stretch'):
            df, err = MarketDataFetcher.fund_etf_spot()
            _display_table(df, err, "etf_spot")
    with tab2:
        if st.button("刷新 LOF 实时行情", key="lof_refresh", width='stretch'):
            df, err = MarketDataFetcher.fund_lof_spot()
            _display_table(df, err, "lof_spot")
    with tab3:
        code = st.text_input("ETF 代码", value="510050", key="etf_code",
                             help="如 510050（华夏上证50ETF）/ 159915（易方达创业板ETF）")
        days = st.number_input("回溯天数", min_value=30, max_value=720,
                               value=180, key="etf_days")
        if st.button("绘制 ETF K 线", key="etf_k", width='stretch'):
            df, err = MarketDataFetcher.fund_etf_kline(code.strip(), days=int(days))
            if err:
                st.error(err)
            elif df.empty:
                st.info("暂无数据（非交易时段或代码无效）。")
            else:
                _line_chart(df, f"ETF {code}")
                _display_table(df, None, "etf_kline")
        st.divider()
        nav_code = st.text_input("开放式基金代码（净值）", value="110011", key="nav_code",
                                 help="如 110011（易方达中小盘）/ 161725（招商中证白酒）")
        if st.button("查询基金净值走势", key="nav_btn", width='stretch'):
            df, err = MarketDataFetcher.fund_open_nav(nav_code.strip())
            if err:
                st.error(err)
            elif df.empty:
                st.info("暂无数据（代码无效或接口返回空）。")
            else:
                _line_chart(df, f"基金 {nav_code} 净值", value_hints=("净值", "nav", "单位净值"))
                _display_table(df, None, "nav_hist")
    with tab4:
        cat = st.selectbox("基金类别", ["全部", "股票型", "混合型", "债券型",
                                   "指数型", "QDII", "货币型"], key="fund_cat")
        if st.button("查询开放基金排行", key="fund_rank", width='stretch'):
            df, err = MarketDataFetcher.fund_open_rank(category=cat)
            _display_table(df, err, "fund_rank")


# ----------------------------------------------------------------------
# 页面：债券 / 外汇 / 贵金属
# ----------------------------------------------------------------------
def render_bonds_forex():
    st.caption("可转债实时、外汇实时、上海金基准价、中行外汇牌价（数据：AKShare）。")
    from data.market_data import MarketDataFetcher

    tab1, tab2, tab3, tab4 = st.tabs(
        ["可转债实时", "外汇实时", "上海金基准价", "中行外汇牌价"]
    )
    with tab1:
        if st.button("刷新可转债行情", key="cov_refresh", width='stretch'):
            df, err = MarketDataFetcher.bond_cov_spot()
            _display_table(df, err, "cov_spot")
    with tab2:
        if st.button("刷新外汇行情", key="fx_refresh", width='stretch'):
            df, err = MarketDataFetcher.forex_spot()
            _display_table(df, err, "fx_spot")
    with tab3:
        if st.button("刷新上海金基准价", key="gold_refresh", width='stretch'):
            df, err = MarketDataFetcher.gold_spot()
            _display_table(df, err, "gold_spot")
    with tab4:
        if st.button("刷新中行外汇牌价", key="boc_refresh", width='stretch'):
            df, err = MarketDataFetcher.currency_boc()
            _display_table(df, err, "boc_spot")


# ----------------------------------------------------------------------
# 页面：股票行情（A 股实时 + K 线）
# ----------------------------------------------------------------------
def render_stocks():
    import time

    from data.market_data import MarketDataFetcher

    st.title("📊 股票行情 (A 股)")
    st.caption("实时行情 + K 线（蜡烛图）+ 当日分时，仿同花顺操作台（数据：AKShare）。")

    # ---- 自选股管理（会话级） ----
    if "stock_watchlist" not in st.session_state:
        st.session_state.stock_watchlist = ["600519", "000001", "300750",
                                            "600036", "002594"]

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    new_code = ctrl_col1.text_input("添加自选股（6 位代码）", key="stock_add",
                                     placeholder="如 600519 / 000001")
    if ctrl_col1.button("➕ 添加", key="stock_add_btn") and new_code.strip():
        code = new_code.strip().zfill(6)
        if code not in st.session_state.stock_watchlist:
            st.session_state.stock_watchlist.append(code)
    if st.session_state.stock_watchlist:
        rm = ctrl_col2.selectbox("移除", st.session_state.stock_watchlist, key="stock_rm")
        if ctrl_col3.button("➖ 移除", key="stock_rm_btn"):
            st.session_state.stock_watchlist.remove(rm)

    # ---- 自动刷新控制 ----
    auto = st.checkbox("自动刷新", value=True, key="stock_auto")
    interval = st.slider("刷新间隔（秒）", 10, 120, 20, key="stock_interval") if auto else 20

    def _fmt_amount(v):
        try:
            v = float(v)
        except Exception:
            return "-"
        if v >= 1e8:
            return f"{v / 1e8:.2f}亿"
        if v >= 1e4:
            return f"{v / 1e4:.2f}万"
        return f"{v:.0f}"

    def _rows_for(codes):
        out = []
        for c in codes:
            df, err = MarketDataFetcher.stock_realtime(c)
            if err or df is None or df.empty:
                out.append({"代码": c, "名称": "—", "最新价": float("nan"),
                            "涨跌幅": float("nan"), "涨跌额": float("nan"),
                            "成交额": float("nan"), "_err": bool(err)})
            else:
                r = df.iloc[0].to_dict()
                out.append({**r, "_err": False})
        return out

    def _style_watchlist(df):
        def _row_color(row):
            try:
                pct = float(row["涨跌幅"])
            except Exception:
                pct = 0.0
            color = "#ef232a" if pct > 0 else ("#14b143" if pct < 0 else "#888888")
            style = []
            for col in df.columns:
                style.append(f"color:{color};font-weight:600"
                             if col in ("最新价", "涨跌幅", "涨跌额") else "")
            return style
        return (df.style.apply(_row_color, axis=1)
                .format({"涨跌幅": "{:.2f}%", "涨跌额": "{:.2f}",
                         "最新价": "{:.2f}", "成交额": _fmt_amount}))

    wl = st.session_state.stock_watchlist
    rows = _rows_for(wl)
    wdf = pd.DataFrame(rows, columns=["代码", "名称", "最新价", "涨跌幅",
                                      "涨跌额", "成交额"])
    wdf_disp = wdf.dropna(subset=["最新价"]) if wdf["最新价"].isna().any() else wdf
    st.subheader("自选股实时报价")
    if wdf_disp.empty:
        st.info("暂无实时数据（接口暂不可用或非交易时段），将随自动刷新重试。")
    else:
        st.dataframe(_style_watchlist(wdf_disp), width='stretch', height=300)

    # ---- 选中个股详情 ----
    st.divider()
    name_map = {r["代码"]: r.get("名称", r["代码"]) for r in rows}
    opts = [f"{name_map.get(c, c)} ({c})" for c in wl]
    sel_label = st.selectbox("查看个股", opts, key="stock_sel")
    sel_code = sel_label.rsplit("(", 1)[-1].rstrip(")")
    sel_row = next((r for r in rows if str(r["代码"]) == sel_code), None)

    if sel_row is None or (isinstance(sel_row.get("最新价"), float) and pd.isna(sel_row.get("最新价"))):
        st.info(f"{sel_code} 暂无实时行情（接口暂不可用或非交易时段）。")
    else:
        try:
            price = float(sel_row.get("最新价"))
            pct = float(sel_row.get("涨跌幅"))
            chg = float(sel_row.get("涨跌额"))
        except Exception:
            price = pct = chg = 0.0
        color = "#ef232a" if pct > 0 else ("#14b143" if pct < 0 else "#888888")
        head_l, head_r = st.columns([1, 2])
        head_l.markdown(
            f"<div style='font-size:40px;font-weight:700;color:{color}'>"
            f"{price:.2f}</div>", unsafe_allow_html=True)
        head_r.markdown(
            f"<div style='font-size:20px;color:{color};margin-top:12px'>"
            f"{pct:+.2f}%　{chg:+.2f}</div>", unsafe_allow_html=True)
        st.markdown(
            f"**{sel_row.get('名称', sel_code)}**（{sel_code}）")

        mcols = st.columns(5)
        metrics = [
            ("今开", sel_row.get("今开")),
            ("昨收", sel_row.get("昨收")),
            ("最高", sel_row.get("最高")),
            ("最低", sel_row.get("最低")),
            ("成交量", _fmt_amount(sel_row.get("成交量"))),
            ("成交额", _fmt_amount(sel_row.get("成交额"))),
            ("换手率", sel_row.get("换手率")),
            ("振幅", sel_row.get("振幅")),
            ("市盈率", sel_row.get("市盈率-动态") or sel_row.get("市盈率")),
            ("量比", sel_row.get("量比")),
        ]
        for i, (k, v) in enumerate(metrics):
            try:
                v = f"{float(v):.2f}" if v is not None else "-"
            except Exception:
                v = str(v) if v is not None else "-"
            mcols[i % 5].metric(k, v)

        # ---- K 线 ----
        kc1, kc2, kc3 = st.columns([2, 1, 1])
        period = kc1.selectbox("周期", ["daily", "weekly", "monthly"], key="stock_period")
        days = kc2.number_input("回溯天数", 30, 720, 180, key="stock_days")
        adjust = kc3.selectbox("复权", ["", "qfq", "hfq"],
                               format_func=lambda x: {"": "不复权", "qfq": "前复权",
                                                      "hfq": "后复权"}[x], key="stock_adjust")
        kdf, kerr = MarketDataFetcher.stock_kline(
            sel_code, period=period, days=int(days), adjust=adjust)
        if kerr:
            st.error(kerr)
        elif kdf is None or kdf.empty:
            st.info("暂无 K 线数据（非交易时段或代码无效）。")
        else:
            _candlestick_chart(kdf, f"{sel_code} K 线")

        # ---- 当日分时 ----
        idf, ierr = MarketDataFetcher.stock_intraday(sel_code)
        if ierr:
            st.error(ierr)
        elif idf is None or idf.empty:
            st.info("暂无分时数据（非交易时段或代码无效）。")
        else:
            _line_chart(idf, f"{sel_code} 当日分时",
                        value_hints=("成交价", "价格", "price", "close"),
                        date_hints=("时间", "date", "time"))

    if auto:
        st.caption(f"⏱ 每 {interval}s 自动刷新 · 最近更新 "
                   f"{pd.Timestamp.now():%H:%M:%S}")
        time.sleep(interval)
        st.rerun()


# ----------------------------------------------------------------------
# 因子实时监控（新增）
# ----------------------------------------------------------------------
def render_monitor():
    """📡 因子实时监控：展示已学习因子的 IC 水平、类别分布与衰减趋势。"""
    import plotly.express as px
    from rag.learned_library import LearnedFactorLibrary

    if st.button("🔄 刷新"):
        st.rerun()

    lib = LearnedFactorLibrary()
    factors = lib.all()
    if not factors:
        st.info("暂无已学习因子。请先通过「🤖 因子挖掘 (Agent)」或「🏭 因子精炼厂」生成因子，"
                "系统会自动将其写入学习库并出现在此看板。")
        return

    rows = []
    for f in factors:
        m = f.get("metrics") or {}
        rows.append({
            "因子": f.get("title") or f.get("name") or "未命名",
            "类别": f.get("category") or "未知",
            "IC": m.get("ic"),
            "RankIC": m.get("rank_ic"),
            "多空Sharpe": m.get("long_short_sharpe"),
            "覆盖率": m.get("coverage"),
        })
    df = pd.DataFrame(rows)

    c1, c2, c3 = st.columns(3)
    c1.metric("因子总数", len(df))
    valid_ic = df["IC"].dropna()
    c2.metric("平均 |IC|", f"{valid_ic.abs().mean():.4f}" if len(valid_ic) else "—")
    c3.metric("弱因子(IC<0.02)", int((valid_ic.abs() < 0.02).sum()))

    if df["IC"].notna().any():
        fig = px.bar(df.dropna(subset=["IC"]), x="因子", y="IC", color="类别",
                     title="各因子 IC 水平（样本内）", text="IC")
        fig.update_layout(height=360)
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if df["类别"].notna().any():
            cnt = df["类别"].value_counts().reset_index()
            cnt.columns = ["类别", "数量"]
            fig2 = px.pie(cnt, names="类别", values="数量", title="因子类别分布")
            st.plotly_chart(fig2, use_container_width=True)
    with col_b:
        st.subheader("因子健康度")
        for _, r in df.iterrows():
            ic = r["IC"]
            if ic is None or (isinstance(ic, float) and pd.isna(ic)):
                st.warning(f"{r['因子']}：缺少 IC 指标")
            elif abs(ic) < 0.02:
                st.warning(f"{r['因子']}：IC={ic:.4f} 偏弱，建议关注过拟合/衰减")
            else:
                st.success(f"{r['因子']}：IC={ic:.4f} 健康")

    decay_rows = []
    for f in factors:
        iby = (f.get("metrics") or {}).get("ic_by_year")
        if isinstance(iby, dict):
            for yr, v in iby.items():
                decay_rows.append({"因子": f.get("title") or f.get("name"), "年份": str(yr), "IC": v})
    if decay_rows:
        ddf = pd.DataFrame(decay_rows)
        fig3 = px.line(ddf, x="年份", y="IC", color="因子",
                       title="IC 随年份变化（衰减监测）", markers=True)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.caption("提示：因子若记录 `ic_by_year`，此处会展示 IC 随年份的衰减曲线。")


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def render_vibe_trading():
    st.caption("用自然语言描述交易想法，Agent 借助 Vibe-Trading 量化 Alpha 库生成并回测因子。"
               "可选调用原生 vibetrading 引擎（加密货币，需安装并联网）。")
    user_input = st.text_area(
        "交易策略描述（自然语言）",
        value="低估值、高 ROE 且现金流稳健的质量因子，做行业中性，持有约 20 个交易日",
        height=90,
    )
    col1, col2 = st.columns(2)
    with col1:
        seed_lib = st.checkbox("将 Vibe-Trading Alpha 库注入 RAG", value=True)
    with col2:
        use_native = st.checkbox("优先调用原生 vibetrading 引擎", value=False)
    auto_method = st.checkbox("自动生成方法学解读", value=True)
    run_btn = st.button("🚀 运行 Vibe Trading", type="primary")

    if run_btn and user_input.strip():
        agent = get_agent()
        _apply_model(agent)
        session = VibeTradingSession(agent)
        with st.spinner("Vibe Trading 正在将想法转化为可回测因子..."):
            result = session.run(user_input, use_native=use_native, seed_library=seed_lib)
        d = _build_agent_dict(result, with_method=auto_method)
        _render_agent_dict(d, with_method=auto_method)


# ----------------------------------------------------------------------
# 页面：传统因子库浏览器
# ----------------------------------------------------------------------
def render_traditional_factors():
    st.caption("五大方向 · 55+ 预置因子 · 因子簇参数扩增 · 与遗传规划/LLM 联动")
    from src.engine.traditional_factors import CATEGORY_LABELS, ALL_CATEGORIES, get_factors_by_category, get_all_factors, export_all_to_dict

    # 统计卡片
    library = get_library()
    stats = library.statistics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("因子总数", stats["total"])
    c2.metric("预置因子", stats["by_source"]["static"])
    c3.metric("GP 挖掘", stats["by_source"]["generated"])
    c4.metric("用户定义", stats["by_source"]["user"])

    st.divider()

    # 分类筛选
    tabs = st.tabs([CATEGORY_LABELS[c] for c in ALL_CATEGORIES] + ["全量因子"])
    for idx, tab in enumerate(tabs):
        with tab:
            if idx < len(ALL_CATEGORIES):
                cat = ALL_CATEGORIES[idx]
                factors = get_factors_by_category(cat)
                st.subheader(f"{CATEGORY_LABELS[cat]}（{len(factors)} 个）")
                for f in factors[:12]:
                    with st.expander(f"{f.display_name} — {f.name}"):
                        st.caption(f"[{f.direction}] 质量分: {f.quality_score:.2f}")
                        st.text(f.description)
                        st.code(f.code[:500] + ("..." if len(f.code) > 500 else ""), language="python")
                        st.caption(f"标签: {' · '.join(f.tags)}")
            else:
                all_f = [f.to_dict() for f in get_all_factors()]
                query = st.text_input("搜索因子", key="all_factor_search")
                filtered = [f for f in all_f if query.lower() in f["name"].lower() or query.lower() in f["display_name"].lower()] if query else all_f
                st.subheader(f"共 {len(filtered)} 个因子")
                df = pd.DataFrame(filtered)[["name", "display_name", "category_label", "direction", "quality_score", "tags"]]
                st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # 因子簇扩增
    st.subheader("📈 批量参数扩增")
    col_a, col_b = st.columns([3, 1])
    with col_a:
        windows = st.multiselect("扩增窗口期", [3, 5, 10, 20, 30, 60, 120], default=[5, 10, 20, 30, 60])
    with col_b:
        expand_btn = st.button("执行扩增", type="primary")
    if expand_btn and windows:
        with st.spinner(f"正在按 {len(windows)} 个窗口扩增..."):
            expanded = library.cluster_expand_all(windows=windows)
        st.success(f"新增 {len(expanded)} 个因子变体")
        new_stats = library.statistics()
        st.metric("因子总数", new_stats["total"])


# ----------------------------------------------------------------------
# 页面：遗传规划因子挖掘
# ----------------------------------------------------------------------
def render_gp_mining():
    st.caption("因子簇驱动演化 · 岛屿模型 · 事件窗口感知 · 批量海量生产")

    col1, col2 = st.columns([1, 1])
    with col1:
        generations = st.slider("演化代数", 3, 20, 8)
        pop_size = st.slider("每簇种群大小", 10, 80, 30)
    with col2:
        top_k = st.slider("保留顶级因子数", 5, 50, 15)
        auto_save = st.checkbox("自动入库", value=True)

    run_btn = st.button("🧬 启动 GP 演化", type="primary")

    if run_btn:
        library = get_library()
        from src.engine.genetic_enhanced import EnhancedFactorEvolver

        st.info("GP 挖掘需要行情数据。请在下方输入股票代码或选择缓存数据。")
        st.caption("提示：若当前环境有 real_ore.pkl 缓存，将自动使用。")

    st.divider()

    # 批量生产一键盘
    st.subheader("🏭 一键批量生产")
    st.caption("组合参数扩增 + GP演化 + 质量筛选 + 去重融合，全流程自动化")
    mass_btn = st.button("🚀 开始批量生产", type="primary", use_container_width=True)
    if mass_btn:
        with st.spinner("正在批量生产因子（参数扩增 → GP 演化 → 质量筛选 → 去重入库）..."):
            result = mass_produce_from_library(kline=None, generations=6, windows=[5, 10, 20, 30, 60])
            st.success(f"生产完成：{result['total_in_library']} 个因子在库中")
            st.json(result["stats"])


# ----------------------------------------------------------------------
# 页面：非结构化数据挖掘
# ----------------------------------------------------------------------
def render_unstructured():
    st.caption("上传新闻/研报/公告/财务数据 · 自动解析 · 情感因子 · 另类数据源")

    tab1, tab2, tab3 = st.tabs(["文件上传", "另类数据源", "文本分析"])

    with tab1:
        st.subheader("上传数据文件")
        st.caption("支持格式：CSV (.csv)、Excel (.xlsx/.xls)、JSON (.json)、TXT (.txt)、PDF (.pdf)")

        uploaded = st.file_uploader("选择文件", type=["csv", "xlsx", "xls", "json", "txt", "pdf"])
        if uploaded is not None:
            # 保存到临时目录
            tmp_dir = Path("data/uploads")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / uploaded.name
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())

            with st.spinner("正在解析文件..."):
                result = analyze_unstructured_file(str(tmp_path))

            st.success(f"解析完成：{result['meta']['shape'][0]} 行 × {result['meta']['shape'][1]} 列")
            st.json(result["meta"]["column_mapping"])

            if result.get("sentiment"):
                s = result["sentiment"]
                c1, c2, c3 = st.columns(3)
                c1.metric("情感均值", f"{s['mean']:.3f}")
                c2.metric("正面词命中", s["pos_hits"])
                c3.metric("负面词命中", s["neg_hits"])
                if s.get("industries"):
                    st.caption(f"涉及行业：{' · '.join(set(s['industries']))}")

            if result.get("factor_ready"):
                st.success("因子时序就绪，可投入回测")

            st.subheader("数据预览")
            st.dataframe(pd.DataFrame(result["preview"]), use_container_width=True)

    with tab2:
        st.subheader("另类数据源管理")
        mgr = get_unstructured_manager()
        sources = mgr.list_sources()

        if not sources:
            st.info("暂无注册的另类数据源。请先在「文件上传」Tab 中上传数据。")

        # 模拟数据源注册
        mock_sources = [
            {"id": "social_media", "type": "social_media", "description": "社交媒体热词讨论热度（需接入 API）"},
            {"id": "supply_chain", "type": "supply_chain", "description": "供应链关系图谱数据"},
            {"id": "search_trends", "type": "search_trends", "description": "搜索趋势指数（需接入 API）"},
            {"id": "satellite", "type": "satellite", "description": "卫星图像经济活跃度特征"},
        ]
        for src in mock_sources:
            with st.expander(f"{src['id']} — {src['type']}"):
                st.text(src["description"])
                st.button(f"注册 {src['id']}", key=f"reg_{src['id']}", disabled=True if src['id'] in sources else False)

    with tab3:
        st.subheader("中文文本情感分析")
        st.caption("基于关键词 + 规则的中文金融文本分析引擎（零 NLP 模型依赖）")
        sample_text = st.text_area(
            "输入文本",
            value="宁德时代2024年净利润同比增长30%，超出市场预期。分析师普遍看好其储能业务增长前景，"
                  "但需关注上游锂价波动风险。公司近期宣布回购10亿元股份，提振市场信心。",
            height=120,
        )
        if st.button("分析情感"):
            analyzer = get_text_analyzer()
            result = analyzer.analyze(sample_text)
            sentiment_color = "🟢" if result["sentiment"] > 0.1 else ("🔴" if result["sentiment"] < -0.1 else "🟡")
            st.subheader(f"{sentiment_color} 情感分数: {result['sentiment']:.4f}")
            c1, c2, c3 = st.columns(3)
            c1.metric("正面词", result["pos_hits"])
            c2.metric("负面词", result["neg_hits"])
            c3.metric("不确定词", result["uncertainty_hits"])
            if result["industries"]:
                st.caption(f"行业：{' · '.join(result['industries'])}")
            if result["entities"]:
                st.caption(f"实体：{' · '.join(result['entities'])}")


# ----------------------------------------------------------------------
# 页面：Transformer 分析
# ----------------------------------------------------------------------
def render_transformer():
    st.caption("因子编码 · 自注意力融合 · 模式记忆 · 多样性指导")

    coupling = get_coupling()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("因子模式记忆库")
        diversity = coupling.memory.get_diversity_guidance([])
        st.info(f"已记录 {len(coupling.memory._winners)} 个成功模式, {len(coupling.memory._losers)} 个失败模式")

        if diversity.get("explored_regions"):
            st.caption("已探索区域")
            for r in diversity["explored_regions"]:
                st.text(r)

        if diversity.get("underexplored_hints"):
            st.caption("建议探索方向")
            for h in diversity["underexplored_hints"]:
                st.text(h)

    with col2:
        st.subheader("Agent 智能推荐")
        query = st.text_input("输入研究需求", value="低波动动量因子，20天窗口")
        if st.button("获取推荐"):
            from src.agent.integration import build_enhanced_context as build_ctx
            ctx = build_ctx({"requirement": query}, library=get_library(), coupling=coupling)

            recs = ctx.get("factor_suggestions", [])
            st.caption(f"找到 {len(recs)} 个相关因子模板")
            for r in recs[:8]:
                st.text(f"- {r['display_name']} ({r['name']}) [{r['direction']}]")

            if ctx.get("knowledge_text"):
                with st.expander("先验知识文本"):
                    st.text(ctx["knowledge_text"][:1000])


# 页面 key -> render 函数（与 src/ui/nav.py 的 NAV_GROUPS 保持一致）
DISPATCH = {
    "overview": render_overview,
    "memory": render_memory,
    "agent": render_factor_agent,
    "chat": render_agent_chat,
    "refinery": render_refinery,
    "gp": render_gp_mining,
    "vibe": render_vibe_trading,
    "sys_build": render_system_builder,
    "sys_analysis": render_system_analysis,
    "library": render_traditional_factors,
    "monitor": render_monitor,
    "market": render_market_hub,
    "futures": render_futures_options,
    "funds": render_funds,
    "bonds": render_bonds_forex,
    "unstructured": render_unstructured,
    "transformer": render_transformer,
    "kb": render_kb,
    "delivery": render_delivery,
    "config": render_config,
}


def _badge_counts() -> Dict[str, str]:
    """侧边栏目录角标（可选）。任一统计失败都不影响导航。"""
    try:
        bc: Dict[str, str] = {}
        try:
            n = len(mining_repo.list(limit=1))
            if n:
                bc["memory"] = f"{n} 条"
        except Exception:
            pass
        try:
            n = len(systems_repo.list())
            if n:
                bc["sys_build"] = f"{n} 个"
        except Exception:
            pass
        try:
            n = len(runs_repo.list(limit=1000))
            if n:
                bc["sys_analysis"] = f"{n} 次"
        except Exception:
            pass
        return bc
    except Exception:
        return {}


def main():
    st.set_page_config(page_title="FactorGPT", layout="wide", page_icon="🧱")
    theme.inject_theme()
    try:
        db.init_db()
    except Exception:
        pass
    _init_llm_session()

    page = nav.render_sidebar(_badge_counts())
    _render_model_panel()
    _render_data_source_panel()
    nav.render_page_header(page)

    DISPATCH.get(page, render_overview)()
    st.divider()
    st.caption("FactorGPT · 基于 LLM 的量化因子研究 Agent · 图中 IC / Sharpe 等为样本内/样本外回测结果，"
               "仅供研究演示，不构成投资建议。")


if __name__ == "__main__":
    main()
