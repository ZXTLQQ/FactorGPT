"""
Vibe-Trading 集成模块（src/agent/vibe_trading.py）

将「Vibe-Trading」的自然语言策略范式接入 FactorGPT 的因子挖掘 Agent：

1. 知识层：内置一份 Vibe-Trading / HKUDS 风格的量化 Alpha 信号目录
   （data/vibe_trading_alpha_catalog.json），可注入到「已学习因子库」，
   从而自动参与 RAG 检索与因子模板复用。
2. 工作流层：``VibeTradingSession`` 把用户的自然语言交易想法 enrichment
   后交给现有 FactorAgent 的 generate→validate→evaluate→reflect 闭环，
   产出可回测因子与报告（即「describe → factor → backtest」）。
3. 原生引擎（可选）：若用户已 ``pip install vibetrading``（VibeTradingLabs
   的 PyPI 包），可优先调用其原生 ``generate/validate/backtest`` 流程
   （面向加密货币，需网络行情），否则自动降级到 FactorGPT 引擎。

所有外部依赖均为 try/except 保护，缺失或不可用时自动降级，保证离线可用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

# factor-gpt/ 项目根：src/agent/vibe_trading.py -> factor-gpt/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CATALOG_PATH = _PROJECT_ROOT / "data" / "vibe_trading_alpha_catalog.json"

VIBE_SOURCE = "vibe_trading"


def load_vibe_alpha_catalog(path: Optional[str] = None) -> List[Dict]:
    """加载 Vibe-Trading 量化 Alpha 信号目录。"""
    p = Path(path) if path else DEFAULT_CATALOG_PATH
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            items = json.load(f)
        for it in items:
            it.setdefault("source", VIBE_SOURCE)
        return items
    except Exception:
        return []


def seed_vibe_to_library(lib=None, catalog: Optional[List[Dict]] = None) -> int:
    """把 Vibe-Trading Alpha 目录写入「已学习因子库」（按 title 幂等去重）。

    返回新写入的条目数（已存在则合并更新，不计入）。
    """
    from rag.learned_library import LearnedFactorLibrary

    if lib is None:
        lib = LearnedFactorLibrary()
    if catalog is None:
        catalog = load_vibe_alpha_catalog()
    return lib.add_many(catalog)


def build_vibe_context(strategy: str, catalog: Optional[List[Dict]] = None,
                       max_items: int = 16) -> str:
    """把交易想法与 Vibe-Trading Alpha 参考库拼接为发给 Agent 的需求文本。"""
    if catalog is None:
        catalog = load_vibe_alpha_catalog()
    lines: List[str] = [
        "[Vibe-Trading 量化 Alpha 参考库]（Vibe-Trading / HKUDS 风格量化信号，仅作思路启发）",
    ]
    for it in catalog[:max_items]:
        title = it.get("title", "")
        cat = it.get("category", "")
        formula = it.get("formula", "")
        desc = it.get("description", "")
        lines.append(f"- 【{cat}】{title}：{formula}；{desc}")
    lines.append("")
    lines.append("请结合上述 Alpha 思路，将下面的自然语言交易想法转化为「可回测、无前视偏差」"
                 "的选股因子（需含必要的行业/市值中性化处理，并在代码中显式使用 shift(1) 取前值）：")
    lines.append(strategy)
    return "\n".join(lines)


def generate_vibe_strategy_native(prompt: str) -> str:
    """可选：调用真实的 ``vibetrading`` PyPI 包生成并回测策略（加密货币，需网络）。

    任何依赖缺失/不可用时抛出 RuntimeError，由调用方降级到 FactorGPT 引擎。
    """
    try:
        from vibetrading.strategy import generate, validate  # type: ignore
        from vibetrading.backtest import run as backtest_run  # type: ignore
        from vibetrading.analyze import analyze  # type: ignore
    except Exception as e:  # 包未安装
        raise RuntimeError(f"vibetrading 包不可用：{e}") from e

    strategy = generate(prompt)
    issues = validate(strategy)
    if issues:
        return f"[Vibe-Trading 原生引擎] 策略校验未通过：{issues}"
    result = backtest_run(strategy)
    analysis = analyze(result)
    return (f"[Vibe-Trading 原生引擎] 策略已生成并回测。\n"
            f"校验问题：{issues or '无'}\n回测摘要：{analysis}")


class VibeTradingSession:
    """把 Vibe-Trading 工作流包装到现有 FactorAgent 上。"""

    def __init__(self, agent, catalog: Optional[List[Dict]] = None) -> None:
        self.agent = agent
        self.catalog = catalog if catalog is not None else load_vibe_alpha_catalog()

    def run(self, strategy: str, max_iterations=None, use_native: Optional[bool] = None,
            seed_library: bool = True) -> Dict:
        """运行一次 Vibe-Trading 风格的因子挖掘。

        - use_native=True 时优先尝试原生 vibetrading 引擎（失败则降级）；
        - use_native=False 时仅用 FactorGPT 引擎；
        - use_native=None（默认）时若包可用则自动尝试原生引擎。
        - seed_library=True 时把 Alpha 目录注入已学习因子库（RAG 永久增益）。
        """
        if seed_library:
            try:
                seed_vibe_to_library(catalog=self.catalog)
            except Exception:
                pass

        want_native = bool(use_native) if use_native is not None else True
        if want_native:
            try:
                native_report = generate_vibe_strategy_native(strategy)
                return {
                    "metrics": {},
                    "report": native_report,
                    "state": {},
                    "vibe_engine": "native",
                }
            except Exception as e:
                # 原生引擎不可用（未安装/无网络）→ 降级到 FactorGPT 引擎
                native_fallback = f"（原生 Vibe-Trading 引擎不可用，已降级至 FactorGPT 引擎：{e}）"
            else:
                return {
                    "metrics": {},
                    "report": native_report,
                    "state": {},
                    "vibe_engine": "native",
                }
        else:
            native_fallback = ""

        enriched = build_vibe_context(strategy, catalog=self.catalog)
        result = self.agent.run(enriched, max_iterations=max_iterations)
        # 标注本次由 Vibe-Trading 工作流驱动
        report = result.get("report", "")
        if native_fallback:
            report = native_fallback + "\n\n" + report
        result["report"] = "[Vibe-Trading 工作流 · FactorGPT 引擎]\n" + report
        result["vibe_engine"] = "factor_gpt"
        return result


def vibe_run(strategy: str, config=None, use_native: Optional[bool] = None,
             seed_library: bool = True, max_iterations=None) -> Dict:
    """便捷入口：构造 Agent 并运行一次 Vibe-Trading 因子挖掘。"""
    from agent.graph import FactorAgent

    if config is None:
        from llm.client import load_config
        config = load_config()
    agent = FactorAgent(config)
    session = VibeTradingSession(agent)
    return session.run(strategy, max_iterations=max_iterations,
                       use_native=use_native, seed_library=seed_library)
