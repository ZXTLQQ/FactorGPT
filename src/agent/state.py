"""
Agent 状态定义（src/agent/state.py）

定义 LangGraph 工作流在节点间传递的状态结构。采用 TypedDict + total=False，
允许各节点只更新自己关心的字段，并由 LangGraph 做浅合并。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd


class AgentState(TypedDict, total=False):
    # —— 输入 ——
    user_input: str                      # 用户原始需求
    factor_description: str              # 经澄清/标准化的因子描述
    max_iterations: int                  # 最大生成-反思轮数
    # 最近若干轮对话压成的上下文（src/agent/context.py）。多轮工作里第二句往往
    # 是「换个窗口」「再激进一点」这类指代，没有它每轮都只能从零重挖。
    dialogue_context: str

    # —— RAG ——
    knowledge_context: str               # 检索到的因子知识上下文
    reuse_template: Optional[dict]        # 命中的「可复用因子模板」（来自学习库，含 code）

    # —— 生成与校验 ——
    factor_name: str                     # 因子命名
    factor_code: str                     # 当前因子代码
    validation_ok: bool                  # 代码是否通过沙箱校验
    validation_error: str                # 校验失败原因
    # 当前代码的真实来源：'llm' = 大模型生成，'template' = LLM 失败后的关键词
    # 模板兜底。用于让"这次到底有没有真的调模型"可见、可审计，而不是靠猜。
    factor_source: str
    # 历次 LLM 调用失败的原因（生成 / 反思节点追加）。空字符串表示未失败过。
    llm_error: str

    # —— 计算与评价 ——
    factor_long: Optional[pd.DataFrame]  # 计算得到的因子长表（date,symbol,factor）
    metrics: Dict[str, Any]              # 回测指标

    # —— 反思与循环控制 ——
    iteration: int                       # 已进行的生成次数
    reflections: List[str]               # 历史反思记录

    # —— 输出 ——
    report: str                          # 最终报告
    error: str                           # 致命错误
    learned_saved: Optional[dict]        # 本轮是否已写入学习库 {'is_new': bool, 'title': str}
    chart_paths: List[str]               # 标准化回测图（PNG）路径列表
    forward_test: Optional[dict]         # Headline Arena 前瞻检验影子摘要（dry_run 账本）
