"""
因子挖掘智能体 — Streamlit 交互入口
====================================

页面：
- 🏠 概览
- 🤖 因子挖掘 (Agent)：单次需求 -> 检索/生成/回测/反思闭环 + 方法学解读
- 💬 Agent 对话：多轮对话式因子挖掘（新增）
- 🏭 因子精炼厂：RL + RAG + Transformer 复合因子管线
- 📈 行情中心 (Market Hub)：五大指数 / 成分股行情、个股弹窗（K线+资讯研报）、
  走势对比、与因子挖掘模块联动，数据经后端 SQLite 短时缓存
- 📈 期货 & 期权：期货主力实时、期货/期权 K 线、上交所期权、商品期权合约链
- 💰 基金行情：ETF/LOF 实时、ETF K 线、基金净值走势、开放基金排行
- 🪙 债券 / 外汇：可转债、外汇、上海金基准价、中行外汇牌价
- 📚 知识库：RAG 检索 / 上传
- ⚙️ 配置：chroma / tushare 等运行时配置

新增能力：
- ⚙️ 侧边栏「模型 / API 设置」面板：用户可切换供应商（DeepSeek / OpenAI /
  Qwen / 任意 OpenAI 兼容端点），填写 API Key / Base URL / 模型，支持「应用
  配置」「测试连接」「保存到 config.yaml」。切换后 Agent 即可接入其他模型
  （含本地 Ollama、vLLM、OpenRouter 等）。
"""

import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import yaml

# 允许以 `python -m ui.app` 或 `streamlit run src/ui/app.py` 两种方式运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.graph import FactorAgent   # noqa: E402
from ui.methodologist import run_methodologist, get_factor_name_from_report  # noqa: E402
from ui.market_hub import render_market_hub  # noqa: E402
from rag.chroma_store import ensure_chroma  # noqa: E402
from rag.retriever import rag_vector_enabled  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"


# ----------------------------------------------------------------------
# 供应商预设（OpenAI 兼容接口）
# ----------------------------------------------------------------------
PROVIDER_PRESETS = {
    "DeepSeek": {"provider": "deepseek", "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
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
        c.set_model(
            provider=st.session_state.ui_provider_value,
            model=st.session_state.ui_model,
            api_key=st.session_state.ui_api_key,
            base_url=st.session_state.ui_base_url,
            temperature=st.session_state.ui_temp,
        )
        resp = c.chat([{"role": "user", "content": "ping，只回复 ok"}], max_tokens=16)
        st.success(f"连接成功 ✅ 模型回复：{str(resp)[:80]}")
    except Exception as e:
        st.error(f"连接失败 ❌ {e}")


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
    st.title("因子挖掘智能体 · FactorGPT")
    st.caption("检索增强 + 反思式因子工程：用 LLM 把研究想法变成可回测的选股因子。")
    st.markdown(
        """
        **工作流**：检索知识 → 生成因子 → 校验计算 → 回测评价 → 反思迭代。

        - 🤖 **因子挖掘 (Agent)**：单次输入需求，跑完整闭环并给出方法学解读；
        - 💬 **Agent 对话**：多轮对话式挖掘，连续追问、迭代改进；
        - 🏭 **因子精炼厂**：RL（MaskablePPO）+ RAG + Transformer 复合因子；
        - 📊 **股票行情 (A 股)**：沪深京 A 股实时行情、个股 K 线（蜡烛图）、当日分时；
        - 📈 **期货 & 期权** / 💰 **基金行情** / 🪙 **债券/外汇**：期货期权、ETF/LOF、
          基金净值、可转债、外汇、贵金属等实时行情与 K 线（数据：AKShare）；
        - ⚙️ **侧边栏模型设置**：随时切换模型 / API Key，接入其他大模型。
        """
    )


# ----------------------------------------------------------------------
# 页面：单次因子挖掘（Agent）
# ----------------------------------------------------------------------
def render_factor_agent():
    st.title("🤖 因子挖掘 (Agent)")
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
    st.title("💬 Agent 对话")
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
    st.title("🏭 因子精炼厂")
    st.caption("RL(MaskablePPO) 因子组合搜索 + RAG 知识 + Transformer 编码 的复合因子管线。")
    try:
        from pipeline.refinery import RefineryPipeline
    except Exception as e:
        st.error(f"精炼厂模块加载失败：{e}")
        return

    cfg = load_config().get("refinery", {})
    with st.form("refinery_form"):
        desc = st.text_input("复合因子目标描述", value="混合日频与月频，结合短期反转与流动性")
        seed = st.number_input("基础因子候选取样数", min_value=4, max_value=40,
                               value=int(cfg.get("n_pool_seed", 12)))
        cand = st.number_input("RL 候选因子数", min_value=1, max_value=30,
                               value=int(cfg.get("rl_candidates", 6)))
        backend = st.selectbox("RL 后端", ["auto", "sb3", "heuristic"],
                               index=["auto", "sb3", "heuristic"].index(cfg.get("rl_backend", "auto")))
        submitted = st.form_submit_button("🚀 运行精炼厂", type="primary")

    if submitted and desc.strip():
        with st.spinner("精炼厂运行中（RL 训练 + RAG + Transformer）..."):
            try:
                from pipeline.refinery import RefineryPipeline, build_refinery_config

                merged = dict(cfg)
                merged.update(
                    {"n_pool_seed": seed, "rl_candidates": cand, "rl_backend": backend}
                )
                rcfg = build_refinery_config(merged)
                pipe = RefineryPipeline(rcfg)
                result = pipe.run(desc)
            except Exception as e:
                st.exception(e)
                return
        st.success(f"精炼厂完成：候选 {len(result.candidates)} → 入选 {len(result.screened)}")
        ic = result.composite_metrics.get("icir")
        st.metric("复合 ICIR", f"{ic:.4f}" if isinstance(ic, (int, float)) else str(ic))
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
    st.title("📚 知识库 (RAG)")
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
    st.title("⚙️ 配置")
    st.markdown("运行时配置（也可直接编辑 `config.yaml`）。")
    cfg = load_config()
    st.json({k: (v if k != "llm" else {**v, "api_key": "***" if v.get("api_key") else ""})
             for k, v in cfg.items()}, expanded=True)


# ----------------------------------------------------------------------
# 页面：产品交付
# ----------------------------------------------------------------------
def render_delivery():
    st.title("📦 产品交付")
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
    st.title("📈 期货 & 期权")
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
    st.title("💰 基金行情")
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
    st.title("🪙 债券 / 外汇 / 贵金属")
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
    from rag.learned_library import LearnedFactorLibrary

    st.title("📡 因子实时监控")
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
PAGES = {
    "🏠 概览": render_overview,
    "🤖 因子挖掘 (Agent)": render_factor_agent,
    "💬 Agent 对话": render_agent_chat,
    "🏭 因子精炼厂": render_refinery,
    "📦 产品交付": render_delivery,
    "📈 行情中心": render_market_hub,
    "📈 期货 & 期权": render_futures_options,
    "💰 基金行情": render_funds,
    "🪙 债券 / 外汇": render_bonds_forex,
    "📡 因子监控": render_monitor,
    "📚 知识库": render_kb,
    "⚙️ 配置": render_config,
}


def main():
    st.set_page_config(page_title="FactorGPT", layout="wide")
    _init_llm_session()

    st.sidebar.title("FactorGPT")
    choice = st.sidebar.radio("导航", list(PAGES.keys()))
    _render_model_panel()
    PAGES[choice]()
    st.divider()
    st.caption("FactorGPT · 基于 LLM 的量化因子研究 Agent · 图中 IC / Sharpe 等为样本内/样本外回测结果，"
               "仅供研究演示，不构成投资建议。")


if __name__ == "__main__":
    main()
