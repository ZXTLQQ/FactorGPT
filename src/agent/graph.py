"""
Agent 编排图（src/agent/graph.py）

FactorAgent 负责：
- 加载/构造回测所需行情数据（优先 AKShare，失败则合成数据以保证可演示）；
- 组装 LangGraph 状态图，串联「检索 -> 生成 -> 校验 -> 评价 -> 反思」闭环；
- 暴露 `run(user_input)` 接口，返回最终报告与完整状态。

工作流图：

    START -> retrieve_knowledge -> generate_factor -> validate_and_compute
                                    |                      |
                          (校验失败/轮次未达上限)        (校验成功)
                                    |                      |
                            reflect_and_refine <----  evaluate_factor
                                    |                      |
                                    +---- (轮次达上限/达标) -> finalize -> END
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from agent.nodes import FactorAgentNodes
from agent.state import AgentState
from engine.backtest import FactorBacktester
from llm.client import LLMClient, load_config
from rag.retriever import FactorRetriever, rag_vector_enabled
from rag.learned_library import LearnedFactorLibrary, DEFAULT_LEARNED_PATH


class FactorAgent:
    """金融量化因子开发 Agent（LangGraph 编排）。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or load_config()
        self.llm = LLMClient(self.config)
        # 已学习因子库（外部导入 + 自学习），供检索与代码复用
        learned_path = self.config.get("rag", {}).get(
            "learned_library_path", DEFAULT_LEARNED_PATH
        )
        if not os.path.isabs(learned_path):
            learned_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), learned_path
            )
        self.learned = LearnedFactorLibrary(learned_path)
        self.retriever = FactorRetriever(
            use_vector_store=rag_vector_enabled(self.config),
            learned=self.learned,
        )
        self.backtester = FactorBacktester(
            n_quantiles=self.config.get("backtest", {}).get("n_quantiles", 5),
            commission=self.config.get("backtest", {}).get("commission", 0.001),
            risk_free_rate=self.config.get("backtest", {}).get("risk_free_rate", 0.03),
        )
        # 数据准备
        self.kline, self.industry, self.mkt_cap = self._load_data()
        # 样本外（OOS）切分：训练集用于生成-反思闭环，测试集仅做独立验证，
        # 杜绝「看着答案改作业」式过拟合（配置见 config.agent.oos）。
        oos_cfg = self.config.get("agent", {}).get("oos", {})
        oos_enabled = bool(oos_cfg.get("enabled", True))
        if oos_enabled:
            self.train_kline, self.test_kline = self._split_oos(
                self.kline,
                test_frac=float(oos_cfg.get("test_frac", 0.2)),
                min_test_days=int(oos_cfg.get("min_test_days", 60)),
            )
        else:
            self.train_kline, self.test_kline = self.kline, None
        # 节点集合
        self.nodes = FactorAgentNodes(
            llm=self.llm,
            retriever=self.retriever,
            backtester=self.backtester,
            kline=self.kline,
            industry=self.industry,
            mkt_cap=self.mkt_cap,
            config=self.config,
            learned=self.learned,
            train_kline=self.train_kline,
            test_kline=self.test_kline,
        )
        self.max_iterations = int(self.config.get("agent", {}).get("max_iterations", 3))
        self.metrics_threshold = float(
            self.config.get("agent", {}).get("metrics_threshold", 0.02)
        )
        self._graph = None

    # ------------------------------------------------------------------
    # 运行时模型切换（用户在 UI 中通过 API Key / 模型切换时调用）
    # ------------------------------------------------------------------
    def update_llm(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """替换底层 LLM 客户端并同步到所有节点引用。

        在对话/挖掘前调用，即可让 Agent 切换到用户选择的其他模型或
        OpenAI 兼容端点（如本地 Ollama、vLLM、OpenRouter 等）。
        """
        self.llm.set_model(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )
        # 节点持有 llm 引用，需同步，否则仍调用旧客户端。
        if self.nodes is not None:
            self.nodes.llm = self.llm
        logger.info(
            "Agent LLM 已切换：provider=%s model=%s",
            self.llm.provider,
            self.llm.model,
        )

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------
    def _load_data(self):
        cfg = self.config.get("data", {})
        universe_size = int(cfg.get("universe_size", 80))
        start = cfg.get("default_start_date", "2020-01-01")
        end = cfg.get("default_end_date", "2024-12-31")
        index_code = cfg.get("default_index", "000906")
        tushare_token = cfg.get("tushare_token")
        primary_source = cfg.get("primary_source", "akshare")

        kline = None
        force_synthetic = bool(
            cfg.get("force_synthetic")
            or os.environ.get("FACTORGPT_FORCE_SYNTHETIC")
        )
        try:
            if force_synthetic:
                raise RuntimeError("已配置强制使用合成数据")

            # —— 同花顺 MCP 网关数据源 ——
            if primary_source == "ths":
                kline = self._load_data_ths(cfg, universe_size, start, end, index_code)
            else:
                from data.neo_adapter import get_data_source

                # 数据源走工厂：默认 legacy（本地自爬），data.source=neodata 时切稳定源
                fetcher = get_data_source(self.config, tushare_token=tushare_token)
                symbols = fetcher.get_index_constituents(index_code)[:universe_size]
                if symbols:
                    kline = fetcher.get_daily_kline(
                        symbols, start=start, end=end, period="daily", adjust="qfq"
                    )
        except Exception as e:
            print(f"[FactorAgent] 行情获取失败，使用合成数据: {e}")

        if kline is None or kline.empty:
            kline, industry, mkt_cap = self._synthetic_data(universe_size, start, end)
            return kline, industry, mkt_cap

        # 真实数据：补充行业 / 市值映射，使行业/市值中性化真正可运行（不再静默跳过）
        from data.neo_adapter import get_data_source

        symbols = sorted(kline["symbol"].astype(str).str.zfill(6).unique().tolist())
        # 数据源走工厂：默认 legacy（本地自爬），data.source=neodata 时切稳定源
        industry, mkt_cap = get_data_source(
            self.config, tushare_token=cfg.get("tushare_token")
        ).get_industry_and_cap(symbols)
        n_missing = int(industry.isna().sum() + mkt_cap.isna().sum())
        if industry.notna().any() and mkt_cap.notna().any():
            print(f"[FactorAgent] 使用真实行情：{kline['symbol'].nunique()} 只标的，"
                  f"{kline['date'].nunique()} 个交易日；行业/市值映射已加载，中性化将启用"
                  f"（缺失 {n_missing} 项）")
        else:
            print(f"[FactorAgent][警告] 行业/市值映射获取失败，中性化将跳过"
                  f"（请检查网络/akshare 可用性；缺失 {n_missing} 项）")
        return kline, industry, mkt_cap

    def _load_data_ths(self, cfg, universe_size, start, end, index_code):
        """从同花顺 MCP 网关加载行情；失败抛异常由调用方回退合成数据。"""
        token = cfg.get("ths_api_token") or os.environ.get("THS_API_TOKEN")
        base_url = cfg.get("ths_api_base_url") or os.environ.get("THS_API_BASE_URL")
        if not base_url:
            raise RuntimeError("未配置 ths_api_base_url（同花顺 MCP 网关端点）")
        if not token:
            raise RuntimeError("未配置 ths_api_token（同花顺 MCP 鉴权令牌）")
        from data.ths_fetcher import THSDataFetcher

        ths = THSDataFetcher(token=token, base_url=base_url)
        diag = ths.connect_and_discover()
        print(f"[FactorAgent] 同花顺网关握手成功：serverInfo={diag['server_info']}，"
              f"可用工具 {diag['tool_count']} 个")

        # iFinD MCP 无「指数成分股」专用工具：优先用配置的小宇宙，否则尽力查询。
        symbols = list(cfg.get("ths_symbols") or [])
        if not symbols:
            symbols = ths.get_index_constituents(index_code)
        symbols = symbols[:universe_size]
        if not symbols:
            raise RuntimeError("未配置 ths_symbols 且指数成分股查询为空")

        kline = ths.get_daily_kline(symbols, start=start, end=end)
        if kline is None or kline.empty:
            raise RuntimeError("同花顺网关未返回日K线数据")
        return kline

    @staticmethod
    def _split_oos(kline, test_frac: float = 0.2, min_test_days: int = 60):
        """将行情按时间切分为训练集与样本外（OOS）测试集。

        训练集用于因子生成-反思闭环，测试集仅用于最终独立验证，
        以避免 Agent 在回测窗口内反复调参导致过拟合。
        """
        if kline is None or kline.empty or "date" not in kline.columns:
            return kline, None
        dates = np.sort(pd.to_datetime(kline["date"]).unique())
        n_test = max(min_test_days, int(round(len(dates) * test_frac)))
        if n_test >= len(dates) - 1:
            # 数据量不足以切分，退回全量（不报 OOS）
            return kline, None
        split = dates[-n_test]
        mask = pd.to_datetime(kline["date"]) >= split
        test = kline[mask].copy()
        train = kline[~mask].copy()
        print(f"[FactorAgent] 样本外切分：训练 {train['date'].nunique()} 日 / "
              f"样本外 {test['date'].nunique()} 日（切分日 {pd.Timestamp(split).date()}）")
        return train, test

    @staticmethod
    def _synthetic_data(n_symbols: int, start: str, end: str):
        """构造可复现的合成 OHLCV 数据，保证离线可演示。"""
        rng = np.random.default_rng(2024)
        dates = pd.bdate_range(start, end)
        symbols = [f"{i:06d}" for i in range(1, n_symbols + 1)]
        industries = ["金融", "科技", "消费", "医药", "能源", "制造"]
        records = []
        for sym in symbols:
            price = rng.uniform(5, 50)
            ind = rng.choice(industries)
            cap = rng.lognormal(10, 1)
            for d in dates:
                ret = rng.normal(0.0005, 0.02)
                price *= (1 + ret)
                open_p = price * (1 + rng.normal(0, 0.005))
                high = max(open_p, price) * (1 + abs(rng.normal(0, 0.008)))
                low = min(open_p, price) * (1 - abs(rng.normal(0, 0.008)))
                vol = int(rng.integers(1e5, 1e6))
                amount = vol * price
                records.append([d.strftime("%Y-%m-%d"), sym, open_p, high, low, price, vol, amount, ind, cap])
        df = pd.DataFrame(
            records,
            columns=["date", "symbol", "open", "high", "low", "close", "volume", "amount", "industry", "mkt_cap"],
        )
        industry = df.drop_duplicates("symbol").set_index("symbol")["industry"]
        mkt_cap = df.drop_duplicates("symbol").set_index("symbol")["mkt_cap"]
        df = df[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]]
        print(f"[FactorAgent] 使用合成行情：{n_symbols} 只标的，{len(dates)} 个交易日")
        return df, industry, mkt_cap

    # ------------------------------------------------------------------
    # 图构建
    # ------------------------------------------------------------------
    def _build_graph(self):
        from langgraph.graph import END, StateGraph

        g = StateGraph(AgentState)
        g.add_node("retrieve_knowledge", self.nodes.retrieve_knowledge)
        g.add_node("generate_factor", self.nodes.generate_factor)
        g.add_node("validate_and_compute", self.nodes.validate_and_compute)
        g.add_node("evaluate_factor", self.nodes.evaluate_factor)
        g.add_node("reflect_and_refine", self.nodes.reflect_and_refine)
        g.add_node("learn_factor", self.nodes.learn_factor)
        g.add_node("finalize", self.nodes.finalize)

        g.set_entry_point("retrieve_knowledge")
        g.add_edge("retrieve_knowledge", "generate_factor")
        g.add_edge("generate_factor", "validate_and_compute")

        g.add_conditional_edges(
            "validate_and_compute",
            self._route_after_validate,
            {
                "evaluate": "evaluate_factor",
                "reflect": "reflect_and_refine",
                "finalize": "finalize",
            },
        )
        g.add_edge("reflect_and_refine", "validate_and_compute")

        g.add_conditional_edges(
            "evaluate_factor",
            self._route_after_evaluate,
            {
                "reflect": "reflect_and_refine",
                "learn": "learn_factor",
            },
        )
        g.add_edge("learn_factor", "finalize")
        g.add_edge("finalize", END)
        return g.compile()

    def _route_after_validate(self, state: AgentState) -> str:
        if state.get("validation_ok"):
            return "evaluate"
        if int(state.get("iteration", 0)) >= self.max_iterations:
            return "finalize"
        return "reflect"

    def _route_after_evaluate(self, state: AgentState) -> str:
        metrics = state.get("metrics", {})
        if "error" in metrics:
            return "finalize" if int(state.get("iteration", 0)) >= self.max_iterations else "reflect"
        ic = abs(metrics.get("ic", 0.0) or 0.0)
        good = ic >= self.metrics_threshold
        if good or int(state.get("iteration", 0)) >= self.max_iterations:
            return "learn"
        return "reflect"

    # ------------------------------------------------------------------
    # 运行接口
    # ------------------------------------------------------------------
    def run(self, user_input: str, max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """运行因子挖掘工作流。

        Args:
            user_input: 用户的因子需求描述（自然语言）。
            max_iterations: 覆盖最大生成-反思轮数。

        Returns:
            {"report": str, "state": dict, "metrics": dict}
        """
        if self._graph is None:
            self._graph = self._build_graph()
        if max_iterations is not None:
            self.max_iterations = max_iterations

        init_state: AgentState = {
            "user_input": user_input,
            "factor_description": user_input,
            "max_iterations": self.max_iterations,
            "iteration": 0,
            "reflections": [],
        }
        result = self._graph.invoke(init_state)
        return {
            "report": result.get("report", ""),
            "state": result,
            "metrics": {k: v for k, v in result.get("metrics", {}).items() if not k.startswith("_")},
        }
