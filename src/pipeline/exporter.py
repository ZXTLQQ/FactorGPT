"""产品化交付（P1）：一键导出因子表达式 + 调仓清单 CSV + 可解释 PDF/HTML 报告。

把「研究」变成「可交付的金融创新产品」：评审/客户可直接拿到
  * 因子表达式（各候选因子 IC/ICIR 与 LLM 逻辑解释）；
  * 调仓清单 CSV（每个再平衡日的多头持仓与权重，含涨跌停/停牌/流动性约束）；
  * 可解释 HTML 报告（净值曲线、分年度 IC、因子动物园增量信息、基准对比）；
  * 自包含 PDF 报告（同一套图表，便于打印/归档）；
  * 结构化 JSON（复合指标、过拟合检验、组合表现，便于二次开发）。

基准对比采用中证800等权代理（equal_weight_benchmark），给出信息比率 / 年化 alpha。
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from engine.backtest import FactorBacktester

# 中文字体配置：让 PDF/HTML 图表正确渲染中文（注册字体文件并修复 fallback 链）
try:
    from .fonts import setup_cjk_font
    setup_cjk_font()
except Exception:  # noqa: BLE001
    pass

logger = logging.getLogger("factor_gpt.exporter")


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt_close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def plt_close(fig):
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:  # noqa: BLE001
        pass


class Exporter:
    """精炼厂交付物导出器。"""

    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 因子表达式 CSV
    # ------------------------------------------------------------------
    def export_factor_expressions(self, result, ts: str) -> str:
        rows: List[Dict[str, Any]] = []
        for c in result.candidates + result.screened:
            rows.append({
                "name": c.name,
                "source": c.source,
                "ic": c.metrics.get("ic"),
                "icir": c.metrics.get("icir"),
                "ic_positive_ratio": c.metrics.get("ic_positive_ratio"),
                "description": c.description,
                "rationale": c.rationale,
                "references": "; ".join(c.references) if c.references else "",
                "code": (c.code or "")[:2000],
            })
        df = pd.DataFrame(rows)
        path = os.path.join(self.output_dir, f"factors_{ts}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    # ------------------------------------------------------------------
    # 调仓清单 CSV
    # ------------------------------------------------------------------
    def export_rebalance(self, result, ts: str) -> Optional[str]:
        portfolio = result.portfolio
        if not portfolio or "rebalance_list" not in portfolio:
            return None
        rows = []
        for rb in portfolio["rebalance_list"]:
            date = rb["date"]
            for sym, w in rb["weights"].items():
                rows.append({"date": date, "symbol": sym, "weight": w})
        if not rows:
            return None
        df = pd.DataFrame(rows)
        path = os.path.join(self.output_dir, f"rebalance_{ts}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    # ------------------------------------------------------------------
    # 图表
    # ------------------------------------------------------------------
    def _fig_equity(self, result) -> Optional[object]:
        portfolio = result.portfolio
        if not portfolio or "equity" not in portfolio:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:  # noqa: BLE001
            return None
        nav = portfolio["equity"]
        bench_ret = FactorBacktester().equal_weight_benchmark(result.ore.train_kline)
        bnav = (1 + bench_ret.reindex(nav.index).fillna(0)).cumprod()
        bnav = bnav * (nav.iloc[0] if len(nav) else 1)
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.plot(nav.index, nav.values, label="复合因子多头组合", color="#1f77b4", lw=1.6)
        ax.plot(bnav.index, bnav.values, label="中证800等权基准", color="#888", lw=1.2, ls="--")
        ax.set_title("组合净值曲线（A 股现实约束）")
        ax.set_ylabel("净值")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        return fig

    def _fig_ic_year(self, result) -> Optional[object]:
        ic_year = result.ic_by_year
        if not ic_year:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:  # noqa: BLE001
            return None
        years = sorted(ic_year.keys())
        ic = [ic_year[y].get("ic", np.nan) for y in years]
        fig, ax = plt.subplots(figsize=(8, 3.0))
        ax.bar(years, ic, color=["#2ca02c" if v and v > 0 else "#d62728" for v in ic])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title("分年度 Rank IC（稳定性检验）")
        ax.set_ylabel("IC")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.grid(alpha=0.3, axis="y")
        return fig

    def _fig_zoo(self, result) -> Optional[object]:
        zoo = result.factor_zoo
        if not zoo or "zoo_icir" not in zoo:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:  # noqa: BLE001
            return None
        z = zoo["zoo_icir"]
        comp = zoo.get("composite_icir", np.nan)
        incr = zoo.get("incremental_icir", np.nan)
        names = list(z.keys()) + ["复合因子", "增量(残差)"]
        vals = [z[n] for n in z] + [comp, incr]
        fig, ax = plt.subplots(figsize=(8, 3.2))
        colors = ["#1f77b4"] * len(z) + ["#ff7f0e", "#9467bd"]
        ax.bar(names, vals, color=colors)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title("因子动物园：各因子 ICIR 对比（验证增量信息）")
        ax.set_ylabel("ICIR")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.grid(alpha=0.3, axis="y")
        return fig

    # ------------------------------------------------------------------
    # HTML 报告
    # ------------------------------------------------------------------
    def export_html(self, result, ts: str) -> str:
        figs = {
            "equity": self._fig_equity(result),
            "ic_year": self._fig_ic_year(result),
            "zoo": self._fig_zoo(result),
        }
        img = {k: (_b64(v) if v is not None else "") for k, v in figs.items()}

        m = result.composite_metrics or {}
        pm = (result.portfolio or {}).get("metrics", {}) or {}
        rb = result.robustness or {}
        zoo = result.factor_zoo or {}
        bench = result.benchmark_comparison or {}

        factors_tbl = self._factors_table(result)
        robustness_tbl = self._robustness_table(result)
        zoo_tbl = self._zoo_table(result)

        html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>精炼厂因子交付报告 {ts}</title>
<style>
 body{{font-family:-apple-system,"Segoe UI",Roboto,"Microsoft YaHei",sans-serif;margin:0;padding:24px;color:#222;background:#fafafa}}
 h1{{font-size:22px;border-left:5px solid #1f77b4;padding-left:10px}}
 h2{{font-size:17px;margin-top:28px;color:#1f77b4}}
 .card{{background:#fff;border:1px solid #eee;border-radius:8px;padding:16px;margin:12px 0;box-shadow:0 1px 3px rgba(0,0,0,.05)}}
 .kpis{{display:flex;flex-wrap:wrap;gap:12px}}
 .kpi{{flex:1;min-width:140px;background:#fff;border:1px solid #eee;border-radius:8px;padding:12px}}
 .kpi .v{{font-size:20px;font-weight:700}}
 .kpi .l{{font-size:12px;color:#666}}
 table{{border-collapse:collapse;width:100%;font-size:13px}}
 th,td{{border:1px solid #e3e3e3;padding:6px 8px;text-align:left}}
 th{{background:#f0f4f8}}
 img{{max-width:100%;border:1px solid #eee;border-radius:6px;margin-top:6px}}
 .meta{{font-size:12px;color:#888}}
</style></head><body>
<h1>精炼厂因子交付报告</h1>
<p class="meta">生成时间 {datetime.now():%Y-%m-%d %H:%M} ｜ 数据源：{'真实行情' if result.ore.meta.get('source')=='real' else '合成数据'}
｜ 多模态因子：{', '.join(result.multimodal_factors) if result.multimodal_factors else '未启用'}</p>

<div class="card"><h2>核心指标</h2><div class="kpis">
 <div class="kpi"><div class="v">{m.get('icir',0):.3f}</div><div class="l">复合 Rank ICIR</div></div>
 <div class="kpi"><div class="v">{m.get('ic',0):.3f}</div><div class="l">复合 Rank IC</div></div>
 <div class="kpi"><div class="v">{pm.get('ann_return',0):.2%}</div><div class="l">组合年化收益</div></div>
 <div class="kpi"><div class="v">{pm.get('sharpe',0):.2f}</div><div class="l">组合夏普</div></div>
 <div class="kpi"><div class="v">{pm.get('max_drawdown',0):.2%}</div><div class="l">最大回撤</div></div>
 <div class="kpi"><div class="v">{bench.get('benchmark_info_ratio',0):.2f}</div><div class="l">基准信息比率</div></div>
</div></div>

<div class="card"><h2>组合净值曲线（含中证800等权基准）</h2>
{'<img src="data:image/png;base64,'+img["equity"]+'"/>' if img["equity"] else '<p>无组合回测结果</p>'}</div>

<div class="card"><h2>分年度 Rank IC（过拟合防线之一）</h2>
{'<img src="data:image/png;base64,'+img["ic_year"]+'"/>' if img["ic_year"] else '<p>无分年度数据</p>'}</div>

<div class="card"><h2>因子动物园：增量信息验证</h2>
{'<img src="data:image/png;base64,'+img["zoo"]+'"/>' if img["zoo"] else '<p>无动物园对比</p>'}
{zoo_tbl}</div>

<div class="card"><h2>过拟合检验（walk-forward / DSR / 参数稳定性）</h2>
{robustness_tbl}
<p class="meta">结论（verdict）：<b>{rb.get('verdict','—')}</b> ｜ DSR={rb.get('deflated_sharpe_ratio','—')}（>0.95 表示夏普统计显著）</p></div>

<div class="card"><h2>候选因子与可解释性</h2>
{factors_tbl}</div>

<div class="card"><h2>组合构建现实约束</h2>
<p class="meta">{self._assumptions_text(result)}</p></div>
</body></html>"""
        path = os.path.join(self.output_dir, f"report_{ts}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path

    # ------------------------------------------------------------------
    # PDF 报告
    # ------------------------------------------------------------------
    def export_pdf(self, result, ts: str) -> str:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
        except Exception as e:  # noqa: BLE001
            logger.warning("PDF 导出依赖缺失: %s", e)
            return ""

        figs = [self._fig_equity(result), self._fig_ic_year(result), self._fig_zoo(result)]
        path = os.path.join(self.output_dir, f"report_{ts}.pdf")
        with PdfPages(path) as pdf:
            # 摘要页
            m = result.composite_metrics or {}
            pm = (result.portfolio or {}).get("metrics", {}) or {}
            bench = result.benchmark_comparison or {}
            rb = result.robustness or {}
            zoo = result.factor_zoo or {}
            lines = [
                "精炼厂因子交付报告", "",
                f"生成时间: {datetime.now():%Y-%m-%d %H:%M}",
                f"数据源: {'真实行情' if result.ore.meta.get('source')=='real' else '合成数据'}",
                f"多模态因子: {', '.join(result.multimodal_factors) if result.multimodal_factors else '未启用'}", "",
                f"复合 Rank ICIR : {m.get('icir',0):.3f}",
                f"复合 Rank IC   : {m.get('ic',0):.3f}",
                f"组合年化收益   : {pm.get('ann_return',0):.2%}",
                f"组合夏普       : {pm.get('sharpe',0):.2f}",
                f"最大回撤       : {pm.get('max_drawdown',0):.2%}",
                f"基准信息比率   : {bench.get('benchmark_info_ratio',0):.2f}",
                f"基准年化 Alpha : {bench.get('benchmark_alpha_ann',0):.2%}", "",
                f"过拟合检验 verdict : {rb.get('verdict','—')}",
                f"Deflated Sharpe Ratio : {rb.get('deflated_sharpe_ratio','—')}",
                f"因子动物园增量 ICIR : {zoo.get('incremental_icir','—')}",
                f"与最强基准最大相关性 : {zoo.get('max_abs_corr','—')}",
            ]
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.08, 0.95, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
            pdf.savefig(fig)
            plt.close(fig)
            for fig in figs:
                if fig is not None:
                    pdf.savefig(fig)
                    plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # JSON 摘要
    # ------------------------------------------------------------------
    def export_json(self, result, ts: str) -> str:
        def _clean(o):
            if isinstance(o, dict):
                return {k: _clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_clean(v) for v in o]
            if isinstance(o, (np.floating,)):
                o = float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, float) and np.isnan(o):
                return None
            return o

        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source": result.ore.meta.get("source"),
            "multimodal_factors": result.multimodal_factors,
            "composite_metrics": result.composite_metrics,
            "loo_result": result.loo_result,
            "robustness": result.robustness,
            "portfolio": {k: v for k, v in (result.portfolio or {}).items()
                          if k != "equity"},
            "cost_sensitivity": result.cost_sensitivity,
            "ic_by_year": result.ic_by_year,
            "benchmark_comparison": result.benchmark_comparison,
            "factor_zoo": result.factor_zoo,
            "candidates": [
                {"name": c.name, "source": c.source, "ic": c.metrics.get("ic"),
                 "icir": c.metrics.get("icir"), "rationale": c.rationale,
                 "references": c.references}
                for c in (result.candidates + result.screened)
            ],
        }
        path = os.path.join(self.output_dir, f"meta_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_clean(payload), f, ensure_ascii=False, indent=2, default=str)
        return path

    # ------------------------------------------------------------------
    # 表格 HTML
    # ------------------------------------------------------------------
    def _factors_table(self, result) -> str:
        rows = ""
        for c in result.candidates + result.screened:
            rationale = (c.rationale or "").replace("\n", " ")
            refs = "; ".join(c.references) if c.references else "—"
            rows += (f"<tr><td>{c.name}</td><td>{c.source}</td>"
                     f"<td>{c.metrics.get('icir',0):.3f}</td>"
                     f"<td>{rationale or '—'}</td><td>{refs}</td></tr>")
        return (f"<table><thead><tr><th>因子</th><th>来源</th><th>ICIR</th>"
                f"<th>逻辑解释</th><th>依据</th></tr></thead><tbody>{rows}</tbody></table>")

    def _robustness_table(self, result) -> str:
        rb = result.robustness
        if not rb or not rb.get("enabled"):
            return "<p>过拟合检验未启用。</p>"
        wf = rb.get("walk_forward", [])
        rows = "".join(
            f"<tr><td>窗口 {w['window']}</td><td>{w.get('ic',0):.3f}</td>"
            f"<td>{w.get('icir',0):.3f}</td><td>{w.get('ic_positive_ratio',0):.2%}</td></tr>"
            for w in wf)
        ps = rb.get("parameter_stability", {})
        return (f"<p>walk-forward 平均 ICIR: <b>{rb.get('walk_forward_icir_mean',0):.3f}</b> ｜ "
                f"DSR: <b>{rb.get('deflated_sharpe_ratio','—')}</b></p>"
                f"<table><thead><tr><th>滚动窗口</th><th>IC</th><th>ICIR</th><th>IC 正率</th></tr>"
                f"</thead><tbody>{rows}</tbody></table>"
                f"<p class='meta'>参数稳定性：ICIR 最小值 {ps.get('icir_min',0):.3f}（需>0），"
                f"波动 {ps.get('icir_std',0):.3f}，稳定={ps.get('stable')}</p>")

    def _zoo_table(self, result) -> str:
        zoo = result.factor_zoo
        if not zoo or "zoo_icir" not in zoo:
            return ""
        rows = "".join(f"<tr><td>{n}</td><td>{v:.3f}</td></tr>"
                       for n, v in zoo["zoo_icir"].items())
        return (f"<p>复合 ICIR: <b>{zoo.get('composite_icir',0):.3f}</b> ｜ "
                f"最强基准 ICIR: {zoo.get('max_zoo_icir',0):.3f} ｜ "
                f"增量(残差) ICIR: <b>{zoo.get('incremental_icir',0):.3f}</b> ｜ "
                f"与最强基准最大相关性: {zoo.get('max_abs_corr',0):.3f}</p>"
                f"<table><thead><tr><th>动物园因子</th><th>ICIR</th></tr></thead>"
                f"<tbody>{rows}</tbody></table>")

    def _assumptions_text(self, result) -> str:
        a = (result.portfolio or {}).get("assumptions")
        if not a:
            return "（组合回测未启用）"
        return (f"T+1 次日成交={a.get('t_plus_one')}；做空限制 allow_short={a.get('allow_short')}；"
                f"涨跌停阈值={a.get('limit_up_pct')}；佣金={a.get('commission')}；"
                f"印花税={a.get('stamp_tax')}；成本模式={a.get('cost_mode')}；"
                f"流动性门槛(元/日)={a.get('min_daily_amount')}；选股比例={a.get('top_frac')}；"
                f"持有期={a.get('forward_periods')} 日。")

    # ------------------------------------------------------------------
    # 一键导出全部
    # ------------------------------------------------------------------
    def export_all(self, result) -> Dict[str, str]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = {}
        out["factors_csv"] = self.export_factor_expressions(result, ts)
        rb = self.export_rebalance(result, ts)
        if rb:
            out["rebalance_csv"] = rb
        out["html"] = self.export_html(result, ts)
        out["pdf"] = self.export_pdf(result, ts)
        out["json"] = self.export_json(result, ts)
        return out
