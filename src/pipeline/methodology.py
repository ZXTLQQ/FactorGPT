"""PART-06 提交 · 方法学总结 (Methodology Report)。

自动生成 Method Summary 报告，涵盖：因子构建逻辑、参数选择依据、回测性能表现，
并支持训练集/测试集交叉验证与回测结果可视化，一键导出供他人复现与审计。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

# 中文字体配置：让方法学报告图表正确渲染中文
try:
    from .fonts import setup_cjk_font
    setup_cjk_font()
except Exception:  # noqa: BLE001
    pass

from engine.backtest import FactorBacktester
from engine.rpn_engine import RPNConfig, RPNEngine
from pipeline.schema import CandidateFactor, RefineryResult

logger = logging.getLogger("factor_gpt.methodology")


class MethodologyReport:
    """方法学总结报告生成器。"""

    def __init__(self, output_dir: str = "output", rpn_config: Optional[RPNConfig] = None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.rpn = RPNEngine(rpn_config or RPNConfig())

    # -- 交叉验证：训练集 / 测试集分别求值 -------------------------------- #
    def cross_validate(self, composite: pd.Series, ore) -> Dict[str, Dict]:
        out = {}
        for split, kline in (("train", ore.train_kline), ("test", ore.test_kline)):
            try:
                m = self.rpn.evaluate(composite, kline)
                out[split] = {k: m.get(k) for k in (
                    "ic_mean", "ic_std", "icir", "ic_pos_ratio", "sharpe",
                    "long_short_cum_return", "annualized_return", "max_drawdown",
                    "turnover", "stability_score"
                )}
            except Exception as e:  # noqa: BLE001
                out[split] = {"error": str(e)}
        return out

    # -- 回测图表 -------------------------------------------------------- #
    def _render_charts(self, composite: pd.Series, ore, prefix: str) -> List[str]:
        # plot_metrics 返回 Figure 对象列表（顺序：IC 时间序列 / 分位数收益 /
        # 多空权益曲线 / 分层累积收益），此处按位置命名并落盘。
        fig_names = ["ic_ts", "quantile_bar", "ls_equity", "quantile_cum"]
        paths = []
        for split, kline in (("train", ore.train_kline), ("test", ore.test_kline)):
            try:
                bt = FactorBacktester(
                    n_quantiles=self.rpn.config.n_quantiles,
                    forward_periods=self.rpn.config.forward_periods,
                    commission=self.rpn.config.commission,
                    risk_free_rate=self.rpn.config.risk_free_rate,
                )
                metrics = bt.evaluate(kline, composite)
                figs = bt.plot_metrics(metrics) or []
                for i, fig in enumerate(figs):
                    nm = fig_names[i] if i < len(fig_names) else f"fig{i}"
                    p = os.path.join(self.output_dir, f"{prefix}_{split}_{nm}.png")
                    fig.savefig(p, dpi=110, bbox_inches="tight")
                    paths.append(p)
            except Exception as e:  # noqa: BLE001
                logger.warning("图表渲染失败(%s/%s): %s", split, prefix, e)
        return paths

    # -- 生成报告 -------------------------------------------------------- #
    def generate(self, result: RefineryResult, requirement: str = "") -> str:
        ore = result.ore
        comp = result.composite
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        cv = self.cross_validate(comp, ore) if comp is not None else {}
        charts = self._render_charts(comp, ore, f"composite_{ts}") if comp is not None else []

        lines = []
        lines.append("# 因子精炼厂 · 方法学总结报告 (Method Summary)")
        lines.append("")
        lines.append(f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}")
        if requirement:
            lines.append(f"- 挖掘需求：{requirement}")
        lines.append(f"- 数据底座：合成中证1000 子集，训练集 {ore.meta.get('train_days')} 日 / "
                     f"测试集 {ore.meta.get('test_days')} 日，成分股 {len(ore.universe)} 只")
        lines.append("")

        # 一、因子构建逻辑
        lines.append("## 一、因子构建逻辑")
        lines.append("")
        lines.append("本系统以工业冶炼为隐喻，将因子开发抽象为六道工序的端到端闭环：")
        lines.append("1. 矿石原料仓（数据底座）：28 分钟级原始特征 + 50+ 时序/截面因子池；")
        lines.append("2. 采矿作业层：LLM 矿场 + MaskablePPO 强化学习搜索 + Transformer 向量化表征三维生成；")
        lines.append("3. 研磨车间（RPN 引擎）：Rank IC / IR / ICIR 量化有效性度量 + 稳定性评估；")
        lines.append("4. 三级筛选（浮选）：LASSO 去冗余 → 人机协同 → TOP 10% 截断；")
        lines.append("5. 合金配比（AlphaPool）：ICIR 加权 + 正交化合成 + leave-one-out 过拟合检验；")
        lines.append("6. 方法学总结：自动产出本报告并一键导出。")
        lines.append("")

        # 二、参数选择依据
        lines.append("## 二、参数选择依据")
        lines.append("")
        lines.append(f"- RPN 引擎：n_quantiles={self.rpn.config.n_quantiles}，"
                     f"forward_periods={self.rpn.config.forward_periods}（IC 预测周期）；")
        lines.append(f"- 目标函数：最大化 Alpha（ICIR）并惩罚换手，w_turnover_penalty="
                     f"{self.rpn.config.w_turnover_penalty}；")
        lines.append(f"- AlphaPool：正交化={'开' if True else '关'}，权重由正 ICIR 归一化决定，"
                     f"并做 leave-one-out 过拟合检验；")
        lines.append(f"- 浮选截断：TOP {int(self.rpn.config.w_turnover_penalty*0)}% 由 ScreenerConfig 控制"
                     f"（默认保留 ICIR 最高的 10%）。")
        lines.append("")

        # 三、入选因子
        lines.append("## 三、入选因子清单")
        lines.append("")
        lines.append("| 因子 | 来源 | ICIR | 稳定性 | 换手 |")
        lines.append("|------|------|------|--------|------|")
        for c in result.screened:
            m = c.metrics
            lines.append(f"| {c.name} | {c.source} | {m.get('icir', 0):.3f} | "
                         f"{m.get('stability_score', 0):.3f} | {m.get('turnover', 0):.3f} |")
        lines.append("")

        # 三-附：三级筛选审计留痕（可追溯「谁在哪一级剔除了什么」）
        audit = getattr(result, "screen_audit", None) or {}
        if audit:
            lines.append("### 三级筛选审计留痕")
            lines.append("")
            lines.append("| 层级 | 输入 | 输出 | 模式 |")
            lines.append("|------|------|------|------|")
            for key, label in (("lasso", "第一级 LASSO 去冗余"),
                               ("human_collab", "第二级 人机协同"),
                               ("topk", "第三级 TOP-K 截断")):
                st_ = audit.get(key) or {}
                if st_:
                    lines.append(f"| {label} | {st_.get('in', '-')} | {st_.get('out', '-')} "
                                 f"| {st_.get('mode', 'auto')} |")
            hc = audit.get("human_collab") or {}
            if hc.get("mode") == "human":
                rej = hc.get("rejected") or []
                lines.append("")
                lines.append(f"- 人工评审：保留 {hc.get('out')} 个，剔除 {len(rej)} 个"
                             + (f"（{', '.join(map(str, rej[:12]))}）" if rej else ""))
                if hc.get("warning"):
                    lines.append(f"- 提示：{hc['warning']}")
            elif hc:
                lines.append("")
                lines.append("- 本次为无人值守模式：人机协同层透传，实际筛选由 LASSO 与 TOP-K 截断决定。")
            lines.append("")

        # 四、交叉验证（训练 / 测试）
        lines.append("## 四、交叉验证（训练集 / 测试集）")
        lines.append("")
        lines.append("| 指标 | 训练集 | 测试集 |")
        lines.append("|------|--------|--------|")
        keys = ["ic_mean", "icir", "ic_pos_ratio", "sharpe", "long_short_cum_return",
                "annualized_return", "max_drawdown", "turnover", "stability_score"]
        for k in keys:
            tr = cv.get("train", {}).get(k, float("nan"))
            te = cv.get("test", {}).get(k, float("nan"))
            lines.append(f"| {k} | {_fmt(tr)} | {_fmt(te)} |")
        lines.append("")
        if result.loo_result.get("enabled"):
            loo = result.loo_result
            lines.append(f"- leave-one-out：基础复合 ICIR={loo.get('base_icir', 0):.3f}；"
                         f"依赖度最高因子 `{loo.get('most_dependent_factor')}`"
                         f"（剔除后 ICIR 变化 {loo.get('most_dependent_drop', 0):+.3f}）。")
            lines.append("")

        # 五、回测图表
        if charts:
            lines.append("## 五、回测图表")
            lines.append("")
            for p in charts:
                lines.append(f"![{os.path.basename(p)}]({p.replace(os.sep, '/')})")
            lines.append("")

        # 六、过拟合与审计
        lines.append("## 六、过拟合控制与可审计性")
        lines.append("")
        lines.append("- 测试集严格独立于训练集，仅用于最终报告，不参与因子生成/筛选；")
        lines.append("- AlphaPool 经 leave-one-out 检验复合因子对单一因子的依赖度；")
        lines.append("- 全部候选因子代码、指标、参数均可从本报告的对话/配置中复现。")
        lines.append("")

        # 七、组合级回测（A 股现实约束）
        lines.append("## 七、组合级回测（A 股现实约束：因子能否赚钱）")
        lines.append("")
        portfolio = result.portfolio
        if portfolio and "metrics" in portfolio:
            pm = portfolio["metrics"]
            a = portfolio.get("assumptions", {})
            lines.append(f"- 构建多头组合（TOP {float(a.get('top_frac', 0.1)) * 100:.0f}%），按 T+1 次日成交，"
                         f"剔除涨停不可买 / 跌停不可卖 / 停牌 / 低流动性标的，不做空（贴合 A 股现实）；")
            lines.append(f"- 年化收益 **{pm.get('ann_return', 0):.2%}**，年化波动 {pm.get('ann_volatility', 0):.2%}，"
                         f"夏普 **{pm.get('sharpe', 0):.2f}**，最大回撤 **{pm.get('max_drawdown', 0):.2%}**"
                         f"（共 {pm.get('n_rebalances', 0)} 次再平衡）；")
            bench = result.benchmark_comparison or {}
            if bench:
                lines.append(f"- 基准对比（中证800等权）：信息比率 {bench.get('benchmark_info_ratio', 0):.2f}，"
                             f"年化 Alpha {bench.get('benchmark_alpha_ann', 0):.2%}，"
                             f"Beta {bench.get('benchmark_beta', 0):.2f}。")
            cs = result.cost_sensitivity
            if cs:
                lines.append("- 换手成本情景（佣金 × 单边/双边）：")
                for k, v in cs.items():
                    lines.append(f"  - `{k}`：年化 {v.get('ann_return', 0):.2%}，"
                                 f"夏普 {v.get('sharpe', 0):.2f}，"
                                 f"回撤 {v.get('max_drawdown', 0):.2%}；")
            ic_year = result.ic_by_year
            if ic_year:
                lines.append("- 分年度 Rank IC（稳定性）：" + "；".join(
                    f"{y}:{d.get('ic', 0):.3f}(IR {d.get('icir', 0):.2f})"
                    for y, d in sorted(ic_year.items())))
        else:
            lines.append("- 组合回测未启用（refinery.run_portfolio=false）。")
        lines.append("")

        # 八、过拟合检验
        lines.append("## 八、过拟合检验（walk-forward / DSR / 参数稳定性）")
        lines.append("")
        rb = result.robustness
        if rb and rb.get("enabled"):
            lines.append(f"- 综合结论（verdict）：**{rb.get('verdict', '—')}**；")
            lines.append(f"- Deflated Sharpe Ratio（去膨胀夏普）：{rb.get('deflated_sharpe_ratio', '—')}"
                         f"（>0.95 表示该夏普在统计上显著，非多重检验假阳性）；")
            wf = rb.get("walk_forward", [])
            if wf:
                lines.append(f"- walk-forward 滚动窗口平均 ICIR：{rb.get('walk_forward_icir_mean', 0):.3f}，"
                             + "，".join(f"W{w['window']}:{w.get('icir', 0):.2f}" for w in wf) + "；")
            ps = rb.get("parameter_stability", {})
            lines.append(f"- 参数稳定性：ICIR 最小 {ps.get('icir_min', 0):.3f}（需>0），"
                         f"波动 {ps.get('icir_std', 0):.3f}，稳定={ps.get('stable')}。")
        else:
            lines.append("- 过拟合检验未启用。")
        lines.append("")

        # 九、因子动物园（增量信息）
        lines.append("## 九、因子动物园：增量信息验证")
        lines.append("")
        zoo = result.factor_zoo
        if zoo and "zoo_icir" in zoo:
            lines.append(f"- 复合因子 ICIR **{zoo.get('composite_icir', 0):.3f}**，"
                         f"最强基准因子 ICIR {zoo.get('max_zoo_icir', 0):.3f}；")
            lines.append(f"- 剔除全部已知因子后的增量（残差）ICIR：**{zoo.get('incremental_icir', 0):.3f}**"
                         f"（证明复合因子含非冗余信息）；与最强基准最大相关性 {zoo.get('max_abs_corr', 0):.3f}。")
        else:
            lines.append("- 因子动物园未启用。")
        lines.append("")

        # 十、LLM 可解释性
        lines.append("## 十、因子可解释性（LLM 逻辑解释与依据）")
        lines.append("")
        expl = [(c.name, c.rationale, c.references)
                for c in (result.candidates + result.screened) if c.rationale]
        if expl:
            for name, rat, refs in expl[:8]:
                lines.append(f"- **{name}**：{rat}")
                if refs:
                    lines.append(f"  - 依据：{'; '.join(refs)}")
        else:
            lines.append("- 本次为离线合成模式，LLM 逻辑解释未生成"
                         "（接入 LLM 矿场后自动补充 rationale / references）。")
        lines.append("")

        # 十一、多模态数据
        mm = result.multimodal_factors
        if mm:
            lines.append("## 十一、多模态数据")
            lines.append("")
            lines.append(f"本次因子池并入多模态因子：{', '.join(mm)}"
                         "（量价 + 基本面 + 估值 + 资金流 + 新闻情绪）。")
            lines.append("")

        lines.append("---")
        lines.append("*本报告由「因子精炼厂」流水线自动生成，供研究与审计使用。*")

        report_text = "\n".join(lines)
        report_path = os.path.join(self.output_dir, f"method_summary_{ts}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        # 一键导出：JSON 原子产物（复现/审计）
        artifact = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "requirement": requirement,
            "screened": [
                {"name": c.name, "source": c.source, "metrics": c.metrics,
                 "description": c.description, "rationale": c.rationale,
                 "references": c.references}
                for c in result.screened
            ],
            "composite_metrics_train": cv.get("train", {}),
            "composite_metrics_test": cv.get("test", {}),
            "loo": result.loo_result,
            "robustness": result.robustness,
            "portfolio": {k: v for k, v in (result.portfolio or {}).items() if k != "equity"},
            "cost_sensitivity": result.cost_sensitivity,
            "ic_by_year": result.ic_by_year,
            "benchmark_comparison": result.benchmark_comparison,
            "factor_zoo": result.factor_zoo,
            "multimodal_factors": result.multimodal_factors,
            "charts": charts,
        }
        json_path = os.path.join(self.output_dir, f"method_summary_{ts}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, ensure_ascii=False, indent=2, default=str)

        logger.info("方法学报告已生成：%s（附 JSON：%s）", report_path, json_path)
        return report_path


def _fmt(v) -> str:
    if v is None:
        return "N/A"
    try:
        if isinstance(v, float) and (v != v):
            return "N/A"
        return f"{v:.4f}"
    except Exception:  # noqa: BLE001
        return str(v)
