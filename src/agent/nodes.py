"""
Agent 节点实现（src/agent/nodes.py）

FactorAgentNodes 持有 LLM、检索器、回测器等依赖，并以「节点方法」形式实现
因子挖掘工作流的每一步。每个节点接收 AgentState，返回需要更新的字段字典，
由 LangGraph 自动合并。

工作流节点：
  retrieve_knowledge -> generate_factor -> validate_and_compute
  validate_and_compute 失败 -> reflect_and_refine -> validate_and_compute（循环）
  validate_and_compute 成功 -> evaluate_factor
  evaluate_factor 不达标 -> reflect_and_refine -> validate_and_compute（循环）
  达标或达到最大轮数 -> finalize
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# 抑制因子后处理过程中的 pandas FutureWarning 噪声（不影响计算结果）
warnings.filterwarnings("ignore", category=FutureWarning)

from engine.factor_builder import FactorSandbox, build_pipeline, generate_from_keywords
from llm.client import extract_code_block, extract_json


class FactorAgentNodes:
    """因子挖掘 Agent 的节点集合。"""

    def __init__(
        self,
        llm,
        retriever,
        backtester,
        kline: pd.DataFrame,
        industry: Optional[pd.Series] = None,
        mkt_cap: Optional[pd.Series] = None,
        config: Optional[dict] = None,
        learned=None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.backtester = backtester
        self.kline = kline
        self.industry = industry
        self.mkt_cap = mkt_cap
        self.config = config or {}
        self.learned = learned  # 已学习因子库（LearnedFactorLibrary 实例）

    # ------------------------------------------------------------------
    # 1) 知识检索
    # ------------------------------------------------------------------
    def retrieve_knowledge(self, state: dict) -> dict:
        query = state.get("user_input") or state.get("factor_description", "")
        top_k = self.config.get("rag", {}).get("top_k", 5)
        context = self.retriever.render_context(query, top_k=top_k)
        # 命中与需求最匹配、且含代码实现的「已学习因子」，供后续直接复用（调用）
        templates = []
        if self.learned is not None:
            templates = self.retriever.retrieve_template(query, top_k=1)
        return {"knowledge_context": context, "reuse_template": templates[0] if templates else None}

    # ------------------------------------------------------------------
    # 2) 因子代码生成
    # ------------------------------------------------------------------
    def _build_generate_prompt(self, description: str, knowledge: str) -> str:
        return (
            "你是一名资深的量化金融因子工程师。请根据用户需求与检索到的因子知识，"
            "编写一个用于计算选股因子的 Python 函数。\n\n"
            "【严格契约】\n"
            "1. 定义函数 `def alpha_factor(df: pd.DataFrame) -> pd.DataFrame:`\n"
            "2. 输入 df 至少包含列：date, symbol, open, high, low, close, volume, amount, pct_chg\n"
            "3. 在函数内新增一列 'factor' 计算因子值；最后 `return df[['date','symbol','factor']]`\n"
            "   （返回 DataFrame 即可，行顺序不限，框架会按 date,symbol 重新对齐）\n"
            "4. 严禁前视偏差：对收益率/价格等必须使用分组后的 .shift(1) 或滚动窗口后再 shift\n"
            "5. 分组运算请用 `df.groupby('symbol')[...]`（df 自带 symbol 列），不要对裸 Series 再 groupby\n"
            "6. 仅可使用 pandas / numpy（不要 import 其它库）\n\n"
            "【数值健壮性要求（务必遵守，否则沙箱校验必失败）】\n"
            "7. 函数体顶部必须 `import pandas as pd` 与 `import numpy as np`\n"
            "8. 任何 `rolling().mean()/std()`、`rank()` 后再标准化时先做 `.fillna(0)`；"
            "对 `std()` 可能为 0 的情况用 `replace(0, np.nan).fillna(0)` 保护，避免除以 0 产生 inf\n"
            "9. 如需截面回归/正交化（剔除某因子线性影响），严禁对含 NaN/Inf 的截面直接做 `np.linalg.lstsq`："
            "必须先用 `mask = np.isfinite(x) & np.isfinite(y)` 过滤缺失行，且当自变量标准差≈0（近常数）时直接返回 0 残差；"
            "更推荐使用 `scipy.stats.linregress`（自带 NaN 处理）或对 rank 做简单相减式中性化，更稳定\n"
            "10. 最终 'factor' 尽量输出为截面 rank（0~1）或 z-score；整列不得为 NaN，"
            "缺失值用 `.fillna(0)` 兜底\n\n"
            "【输出格式】仅返回一个 JSON 对象：\n"
            '{"name": "因子英文命名", "description": "一句话中文说明", '
            '"code": "完整Python代码", '
            '"rationale": "因子的经济学/统计逻辑（自然语言 2-4 句，可从行为金融、风险补偿、'
            '微观结构或市场无效性等角度解释为何有效）", '
            '"references": ["1-3 条可引用的学术或实务依据，如 Fama-French (1993)、'
            'Jegadeesh & Titman (1993) 动量、Carhart (1997) 四因子、或行业实践来源"]}\n\n'
            "【重要】rationale 与 references 是评审最看重的「方法学可信度」证据，"
            "务必结合上述因子知识与金融学常识给出，避免空话。\n\n"
            f"【用户需求】\n{description}\n\n"
            f"【相关因子知识】\n{knowledge}\n"
        )

    def generate_factor(self, state: dict) -> dict:
        description = state.get("factor_description") or state.get("user_input", "")
        knowledge = state.get("knowledge_context", "")
        iteration = int(state.get("iteration", 0)) + 1

        # 复用命中学习库中的因子模板（直接「调用」已验证代码，加速收敛、保证可运行）
        reuse = state.get("reuse_template")
        if isinstance(reuse, dict) and reuse.get("code"):
            knowledge += (
                "\n\n【可复用因子模板（来自学习库，已验证可运行）】\n"
                f"名称：{reuse.get('title', '')}\n"
                f"类别：{reuse.get('category', '')}\n"
                f"代码：\n{reuse['code']}\n"
                "若需求与该模板一致或相近，可直接采用/改编此代码，只需确保返回 "
                "df[['date','symbol','factor']] 且满足无前视。"
            )

        name, desc, code = None, description, None
        rationale, references = "", []
        # 优先调用 LLM
        try:
            raw = self.llm.complete(
                system="你是量化因子工程专家，输出严格符合约定的 JSON。",
                user=self._build_generate_prompt(description, knowledge),
                temperature=0.4,
            )
            parsed = extract_json(raw)
            if isinstance(parsed, dict):
                code = parsed.get("code") or extract_code_block(raw)
                name = parsed.get("name")
                desc = parsed.get("description") or description
                rationale = parsed.get("rationale", "") or ""
                refs = parsed.get("references", []) or []
                references = refs if isinstance(refs, list) else [str(refs)]
        except Exception as e:
            print(f"[generate_factor] LLM 调用失败，启用关键词模板兜底: {e}")

        # 兜底：关键词模板
        if not code:
            tpl = generate_from_keywords(description)
            if tpl:
                code = tpl["code"]
                name = tpl["name"]
                desc = tpl["desc"]

        if not code:
            return {
                "iteration": iteration,
                "factor_code": "",
                "validation_ok": False,
                "validation_error": "无法生成因子代码（LLM 与模板均失败）",
                "error": "因子生成失败",
            }

        return {
            "iteration": iteration,
            "factor_name": name or "custom_factor",
            "factor_description": desc,
            "factor_code": code,
            "factor_rationale": rationale,
            "factor_references": references,
            "validation_ok": False,  # 待校验节点确认
        }

    # ------------------------------------------------------------------
    # 3) 沙箱校验 + 因子计算
    # ------------------------------------------------------------------
    def validate_and_compute(self, state: dict) -> dict:
        code = state.get("factor_code", "")
        if not code:
            return {"validation_ok": False, "validation_error": "无因子代码"}

        sandbox = FactorSandbox()
        try:
            raw_series = sandbox.run(code, self.kline)
            processed = build_pipeline(
                raw_series,
                winsorize_pct=self.config.get("backtest", {}).get("winsorize_pct", 0.01),
                industry=self.industry,
                mkt_cap=self.mkt_cap,
            )
            factor_long = processed.reset_index()
            if isinstance(factor_long.columns, pd.MultiIndex):
                factor_long.columns = ["date", "symbol", "factor"]
            else:
                cols = list(factor_long.columns)
                rename = {cols[0]: "date", cols[1]: "symbol"}
                value_col = [c for c in cols if c not in ("date", "symbol")][0]
                rename[value_col] = "factor"
                factor_long = factor_long.rename(columns=rename)
            factor_long = factor_long[["date", "symbol", "factor"]]
            coverage = float(processed.notna().mean())
            return {
                "validation_ok": True,
                "validation_error": "",
                "factor_long": factor_long,
                "metrics": {"_coverage": coverage},
            }
        except Exception as e:
            err = str(e)
            hint = ""
            if any(k in err for k in ("SVD", "LinAlg", "did not converge", "singular",
                                      "NaN", "inf", "divide", "zero", "ValueError")):
                hint = (
                    "\n[修复建议] 该错误通常由数值不稳定引起：请在使用 np.linalg.lstsq / 回归 / 标准化前，"
                    "用 dropna() 或 np.isfinite 过滤 NaN/Inf 行；对 std() 可能为 0 的情况做保护"
                    "（replace(0, np.nan).fillna(0)）；自变量近常数时直接返回 0。"
                    "优先改用 scipy.stats.linregress 或对 rank 做相减式中性化。"
                )
            return {"validation_ok": False, "validation_error": err + hint}

    # ------------------------------------------------------------------
    # 4) 因子回测评价
    # ------------------------------------------------------------------
    def evaluate_factor(self, state: dict) -> dict:
        factor_long = state.get("factor_long")
        if factor_long is None or factor_long.empty:
            return {"metrics": {"error": "因子为空，无法评价"}, "error": "因子计算为空"}

        # 防御：去除重复 (date,symbol) 行（LLM 生成代码偶发重复，会导致
        # 非唯一多索引，使回测 join 异常）。保留最后一条，保证索引唯一。
        if factor_long.duplicated(["date", "symbol"]).any():
            factor_long = factor_long.drop_duplicates(["date", "symbol"], keep="last")
        factor_series = pd.Series(
            factor_long["factor"].values,
            index=pd.MultiIndex.from_arrays([factor_long["date"], factor_long["symbol"]]),
            name="factor",
        )
        if factor_series.index.duplicated().any():
            factor_series = factor_series[~factor_series.index.duplicated(keep="last")]
        try:
            metrics = self.backtester.evaluate(self.kline, factor_series)
        except Exception as e:  # noqa: BLE001
            # 回测异常不应中断整个 Agent 流程，转为可展示的错误指标
            return {"metrics": {"error": f"回测执行异常: {type(e).__name__}: {e}"},
                    "error": "回测执行异常"}
        # 合并覆盖率
        if "_coverage" in state.get("metrics", {}):
            metrics["coverage"] = state["metrics"]["_coverage"]
        if "error" in metrics:
            return {"metrics": metrics, "error": metrics["error"]}
        # 生成并保存标准化回测图表（IC 序列 / 分层收益 / 多空权益 / 分层累积收益）
        chart_paths = self._save_charts(metrics, state.get("factor_name", "factor"))
        return {"metrics": metrics, "chart_paths": chart_paths}

    # ------------------------------------------------------------------
    # 4.1) 回测图表落盘
    # ------------------------------------------------------------------
    def _save_charts(self, metrics: dict, factor_name: str) -> List[str]:
        """调用回测引擎生成图表并保存为 PNG，返回路径列表（供报告引用）。"""
        try:
            import os
            import re

            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            figs = self.backtester.plot_metrics(metrics)
            if not figs:
                return []
            out_dir = self.config.get("backtest", {}).get("output_dir", "output")
            os.makedirs(out_dir, exist_ok=True)
            safe = re.sub(r"[^\w一-龥]+", "_", str(factor_name))[:40] or "factor"
            names = ["ic_ts", "quantile_bar", "ls_equity", "quantile_cum"]
            paths: List[str] = []
            for i, fig in enumerate(figs):
                nm = names[i] if i < len(names) else f"chart_{i}"
                p = os.path.join(out_dir, f"backtest_{safe}_{nm}.png")
                fig.savefig(p, dpi=120, bbox_inches="tight")
                plt.close(fig)
                paths.append(p)
            print(f"[图表] 已保存 {len(paths)} 张回测图到 {out_dir}/")
            return paths
        except Exception as e:
            print(f"[图表] 生成失败: {e}")
            return []

    # ------------------------------------------------------------------
    # 5) 反思与改进
    # ------------------------------------------------------------------
    def reflect_and_refine(self, state: dict) -> dict:
        description = state.get("factor_description") or state.get("user_input", "")
        code = state.get("factor_code", "")
        metrics = state.get("metrics", {})
        history = state.get("reflections", [])
        iteration = int(state.get("iteration", 0)) + 1

        # 仅保留可展示的指标（去掉内部序列）
        show_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}

        new_code = None
        try:
            raw = self.llm.complete(
                system=(
                "你是严谨的量化因子研究员。请反思上一版因子效果差的原因，"
                "并输出改进后的因子代码。仍须遵守：定义 alpha_factor(df)->DataFrame、"
                "在函数内新增 'factor' 列并返回 df[['date','symbol','factor']]、"
                "对分组结果使用 df.groupby('symbol')[...].shift(1) 避免前视、仅用 pandas/numpy。"
                "若上一版在校验/计算阶段失败，请务必先读懂失败原因（例如 SVD 未收敛 = 回归/正交化前"
                "未剔除 NaN/Inf；除以 0 = 未保护 std() 为 0），针对性修复，不要泛泛重写整体代码。"
                "只返回 JSON：{\"reflection\": \"反思说明\", \"code\": \"完整代码\"}"
                ),
                user=(
                    f"【需求】{description}\n"
                    f"【上一版代码】\n{code}\n"
                    f"【上一版指标】\n{json.dumps(show_metrics, ensure_ascii=False, default=str)}\n"
                    + (f"【上一版校验/计算失败原因（必须精准定位并修复此错误，不要泛泛重写）】\n{state.get('validation_error')}\n"
                       if state.get("validation_error") else "")
                    + (f"【上一版回测错误】\n{metrics.get('error')}\n"
                       if isinstance(metrics, dict) and metrics.get("error") else "")
                    + f"【历史反思】\n{chr(10).join(history) if history else '（无）'}\n"
                    + "请聚焦上述失败原因进行修复，保持 alpha_factor 契约不变。"
                ),
                temperature=0.5,
            )
            parsed = extract_json(raw)
            if isinstance(parsed, dict) and parsed.get("code"):
                new_code = parsed["code"]
                reflection = parsed.get("reflection", "")
            else:
                new_code = extract_code_block(raw)
                reflection = ""
        except Exception as e:
            print(f"[reflect_and_refine] LLM 反思失败，沿用原代码: {e}")
            reflection = f"LLM 反思不可用: {e}"

        reflections = list(history) + [f"第{iteration}轮反思: {reflection or '（无）'}"]

        if not new_code:
            # 无法改进，直接终局
            return {
                "iteration": iteration,
                "reflections": reflections,
                "error": "反思阶段无法生成改进代码",
            }

        return {
            "iteration": iteration,
            "factor_code": new_code,
            "reflections": reflections,
            "validation_ok": False,
        }

    # ------------------------------------------------------------------
    # 6) 终局报告
    # ------------------------------------------------------------------
    @staticmethod
    def _jsonable_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """将回测指标中的 numpy / pandas 类型转为可 JSON 序列化的基础类型。"""
        out: Dict[str, Any] = {}
        for k, v in metrics.items():
            if k == "quantile_returns":
                continue  # 内部序列，不持久化
            try:
                if isinstance(v, (np.floating,)):
                    out[k] = float(v)
                elif isinstance(v, (np.integer,)):
                    out[k] = int(v)
                elif isinstance(v, (float, int, str, bool)) or v is None:
                    out[k] = v
                else:
                    out[k] = str(v)
            except Exception:
                continue
        return out

    def learn_factor(self, state: dict) -> dict:
        """自主学习的写入环节：将本轮「已通过校验且回测无错误」的因子存入学习库。

        这样 Agent 在后续任务中可通过检索复用该因子（调用），形成持续积累。
        失败或未达标的因子不入库，避免污染学习库。
        """
        if self.learned is None:
            return {}
        if not state.get("validation_ok"):
            return {}
        metrics = state.get("metrics", {})
        if "error" in metrics:
            return {}
        code = state.get("factor_code", "")
        name = state.get("factor_name") or "custom_factor"
        desc = state.get("factor_description") or state.get("user_input", "")
        if not code:
            return {}

        record = {
            "title": name,
            "category": "自学习/agent生成",
            "formula": "",
            "description": desc,
            "code": code,
            "source": "self_learned",
            "metrics": self._jsonable_metrics(metrics),
        }
        try:
            is_new = self.learned.add(record)
            return {"learned_saved": {"is_new": is_new, "title": name}}
        except Exception as e:  # pragma: no cover
            print(f"[learn_factor] 写入学习库失败: {e}")
            return {}

    def finalize(self, state: dict) -> dict:
        name = state.get("factor_name", "custom_factor")
        desc = state.get("factor_description", state.get("user_input", ""))
        code = state.get("factor_code", "")
        metrics = {k: v for k, v in state.get("metrics", {}).items() if not k.startswith("_")}
        knowledge = state.get("knowledge_context", "")
        reflections = state.get("reflections", [])

        report = _build_report(
            name=name, desc=desc, code=code, metrics=metrics,
            knowledge=knowledge, reflections=reflections,
            validation_ok=state.get("validation_ok", False),
            validation_error=state.get("validation_error", ""),
            error=state.get("error"),
            chart_paths=state.get("chart_paths"),
        )
        saved = state.get("learned_saved")
        if saved:
            verb = "新增至" if saved.get("is_new") else "更新到"
            report += (
                f"\n\n## 七、自主学习\n"
                f"> 本轮因子已{verb}学习库：`{saved.get('title')}`。"
                f"后续任务可在检索中复用该因子（调用）。\n"
            )
        return {"report": report}


def _fmt_val(v):
    if isinstance(v, float):
        if abs(v) < 1e-4 and v != 0:
            return f"{v:.2e}"
        return f"{v:.4f}"
    return str(v)


def _build_report(
    name, desc, code, metrics, knowledge, reflections,
    validation_ok, validation_error, error, chart_paths=None,
) -> str:
    lines = []
    lines.append(f"# 因子挖掘报告：{name}\n")
    lines.append(f"**需求描述**：{desc}\n")

    if error:
        lines.append(f"\n> ⚠️ 流程异常：{error}\n")
    if not validation_ok:
        lines.append(f"\n> 代码校验未通过：{validation_error}\n")

    lines.append("\n## 一、因子代码\n")
    lines.append("```python\n" + code + "\n```\n")

    lines.append("\n## 二、回测指标\n")
    if metrics and "error" not in metrics:
        lines.append("| 指标 | 值 |")
        lines.append("|------|------|")
        for k in ["ic", "rank_ic", "icir", "ic_positive_ratio", "long_short_return",
                  "long_short_sharpe", "long_short_cum_return", "max_drawdown",
                  "turnover", "coverage", "n_stocks", "n_dates"]:
            if k in metrics:
                lines.append(f"| {k} | {_fmt_val(metrics[k])} |")
        # 分位数收益
        qr = metrics.get("quantile_returns")
        if isinstance(qr, dict):
            parts = ", ".join(f"Q{i+1}={_fmt_val(v)}" for i, v in sorted(qr.items()))
            lines.append(f"| 分位数收益 | {parts} |")
    else:
        lines.append("（无有效回测结果）\n")

    # —— 三、分层（分组）回测结果 ——
    lines.append("\n## 三、分层回测结果\n")
    qs = metrics.get("quantile_stats") if isinstance(metrics, dict) else None
    if isinstance(qs, dict) and qs:
        lines.append("| 分位组 | 平均日收益 | 年化收益 | 累积收益 | Sharpe | 样本数 |")
        lines.append("|--------|----------|----------|----------|---------|--------|")
        for g in sorted(qs.keys()):
            s = qs[g]
            lines.append(
                f"| Q{int(g) + 1} | {_fmt_val(s.get('mean_ret', float('nan')))} "
                f"| {_fmt_val(s.get('ann_ret', float('nan')))} "
                f"| {_fmt_val(s.get('cum_ret', float('nan')))} "
                f"| {_fmt_val(s.get('sharpe', float('nan')))} "
                f"| {s.get('n', 0)} |"
            )
        # 多空组合行
        ls_ret = metrics.get("long_short_return")
        ls_sharpe = metrics.get("long_short_sharpe")
        ls_cum = metrics.get("long_short_cum_return")
        if ls_ret is not None:
            lines.append(
                f"| **多空(Qn-Q1)** | {_fmt_val(ls_ret)} | — "
                f"| {_fmt_val(ls_cum) if ls_cum is not None else '—'} "
                f"| {_fmt_val(ls_sharpe) if ls_sharpe is not None else '—'} | — |"
            )
        lines.append("\n> 单调性检验：若累积收益随分位组单调递增/递减，说明因子具备稳定选股能力。\n")
    else:
        lines.append("（无有效分层结果）\n")

    # —— 四、回测图表 ——
    lines.append("\n## 四、回测图表\n")
    if chart_paths:
        titles = ["IC 时间序列", "分位数分组平均收益", "多空对冲累计收益", "分层累积收益曲线"]
        for i, p in enumerate(chart_paths):
            t = titles[i] if i < len(titles) else f"图表{i+1}"
            lines.append(f"![{t}]({p.replace(chr(92), '/')})\n")
        lines.append("")
    else:
        lines.append("（未生成图表；通常由校验/回测未通过导致）\n")

    if knowledge:
        lines.append("\n## 五、参考因子知识\n")
        lines.append(knowledge + "\n")

    if reflections:
        lines.append("\n## 六、反思记录\n")
        for r in reflections:
            lines.append(f"- {r}\n")

    return "\n".join(lines)
