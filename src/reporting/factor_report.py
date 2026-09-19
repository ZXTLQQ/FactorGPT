"""因子回测的深度分析报告（src/reporting/factor_report.py）。

回答三个此前只给数字、不给结论的问题：
  1. 这个因子**统计上站得住吗**（显著性、稳定性、分布形态、衰减速度）；
  2. 钱是**怎么赚的**（分位单调性、多空结构、回撤与换手代价）；
  3. 接下来**该改什么**（由大模型基于完整统计画像给出可执行建议）。

由三块组成，可单独调用也可一次性生成：
  - ``statistical_diagnostics``：统计性质体检（Newey-West t、bootstrap 置信区间、
    Jarque-Bera 正态性、Ljung-Box 自相关、IC 衰减、分位单调性、回撤与换手等）；
  - ``render_charts``：12 张图表（IC 序列/分布/衰减、分位收益、多空净值与回撤、
    月度热力图、滚动 ICIR、换手、分位累计、年度对比、因子相关、收益分布）；
  - ``llm_interpret``：把上述统计画像**完整**喂给大模型（图表用关键数值摘要表示，
    因为模型看不到图），产出结构化解读；LLM 不可用时降级为规则化结论，绝不静默跳过。

零新增依赖：matplotlib 必需，scipy 可选（缺失时部分检验降级）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # 服务端/无显示环境必须
import matplotlib.pyplot as plt  # noqa: E402

try:
    from scipy import stats as _st
    _HAS_SCIPY = True
except Exception:  # noqa: BLE001
    _HAS_SCIPY = False

# 中文字体：Windows / macOS / Linux 常见字体依次尝试，缺字体时不报错
_FONT_CANDIDATES = ["SimHei", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC",
                    "WenQuanYi Micro Hei", "Arial Unicode MS"]
_PALETTE = ["#506AAA", "#728DC1", "#91BBDF", "#B6DBF1", "#F6D3D9", "#E79FBF", "#E17692"]


def _setup_style() -> None:
    plt.rcParams["font.sans-serif"] = _FONT_CANDIDATES + plt.rcParams.get("font.sans-serif", [])
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25


def _safe_name(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(s))[:60]


# ======================================================================
# 一、统计性质诊断
# ======================================================================

def newey_west_t(ic: pd.Series, lags: Optional[int] = None) -> float:
    """IC 均值的 Newey-West 稳健 t 统计量（处理 IC 序列的自相关与异方差）。

    直接用 t = mean/std·√n 会因 IC 自相关而**高估**显著性，这里按 Newey-West
    调整长期方差： Var̂ = γ0 + 2Σ_{l=1..L}(1 - l/(L+1))γ_l。
    """
    s = pd.Series(ic, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(s)
    if n < 10:
        return float("nan")
    x = s.to_numpy(dtype=float) - s.mean()
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    lags = int(max(0, min(lags, n - 1)))
    g0 = float(np.dot(x, x) / n)
    long_var = g0
    for l in range(1, lags + 1):
        gl = float(np.dot(x[l:], x[:-l]) / n)
        long_var += 2.0 * (1.0 - l / (lags + 1.0)) * gl
    long_var = max(long_var, 1e-18)
    se = np.sqrt(long_var / n)
    return float(s.mean() / se) if se > 0 else float("nan")


def bootstrap_ic_ci(ic: pd.Series, n_boot: int = 2000, seed: int = 42,
                    alpha: float = 0.05) -> Dict[str, float]:
    """IC 均值的 bootstrap 百分位置信区间（不依赖正态假设）。"""
    s = pd.Series(ic, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(s) < 10:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    means = rng.choice(s, size=(int(n_boot), len(s)), replace=True).mean(axis=1)
    return {
        "mean": float(s.mean()),
        "lo": float(np.percentile(means, 100 * alpha / 2)),
        "hi": float(np.percentile(means, 100 * (1 - alpha / 2))),
    }


def ic_decay(ic: pd.Series, max_lag: int = 10) -> pd.Series:
    """IC 衰减：IC 序列的滞后自相关，反映信号有效期的长短。

    lag k 的相关越接近 1，说明信号越「慢」、可容纳的换手越低；
    快速衰减到 0 则意味着必须高频换手才能实现，成本压力大。
    """
    s = pd.Series(ic, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    out = {}
    for k in range(1, int(max_lag) + 1):
        if len(s) <= k + 2:
            break
        out[f"lag{k}"] = float(s.autocorr(lag=k))
    return pd.Series(out, dtype=float)


def quantile_monotonicity(quantile_returns: Optional[Dict]) -> Dict[str, float]:
    """分位数单调性：分位序号与收益的 Spearman 相关 + 首尾价差。

    单调性好（接近 ±1）说明因子排序逻辑稳定，不是靠个别极端分位撑起来的。
    """
    if not quantile_returns:
        return {"spearman": float("nan"), "spread": float("nan")}
    items = sorted((int(k), float(v)) for k, v in quantile_returns.items())
    if len(items) < 3:
        return {"spearman": float("nan"), "spread": float("nan")}
    ranks = np.array([k for k, _ in items], dtype=float)
    vals = np.array([v for _, v in items], dtype=float)
    rho = float(pd.Series(ranks).corr(pd.Series(vals), method="spearman"))
    return {"spearman": rho, "spread": float(vals[-1] - vals[0])}


def _infer_periods_per_year(s: pd.Series) -> float:
    """推断净值序列的年化频次：日频取 252；跨度明显大于 3 天时按实际间隔折算（周/月频）。"""
    idx = s.index
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 3:
        step_days = float(pd.Series(idx).diff().dt.total_seconds().median() / 86400.0)
        if np.isfinite(step_days) and step_days > 3.0:
            return float(365.25 / step_days)
    return 252.0


def _drawdown_stats(nav: pd.Series) -> Dict[str, float]:
    """净值曲线的回撤与收益类指标。

    样本不足半年（或 20 期）时不做年化：短样本外推会把 40 天的 2.8 倍收益放大成
    6 倍的"年化"，Calmar 随之失真到三位数。此时以区间口径（累计收益/最大回撤）
    给出 calmar，并在 `annualized` 标记口径，供报告与解读如实标注。
    """
    s = pd.Series(nav, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 5:
        return {}
    r = s.pct_change().dropna()
    peak = s.cummax()
    dd = s / peak - 1.0
    max_dd = float(dd.min())
    ppy = _infer_periods_per_year(s)
    n = len(s)
    total_ret = float(s.iloc[-1] / s.iloc[0]) - 1.0 if s.iloc[0] > 0 else float("nan")
    annualized = bool(np.isfinite(total_ret) and n >= max(20, int(ppy // 2)))
    ann = float((1.0 + total_ret) ** (ppy / n) - 1.0) if annualized else float("nan")
    vol = float(r.std() * np.sqrt(ppy)) if len(r) > 1 else float("nan")
    sharpe = float(r.mean() / r.std() * np.sqrt(ppy)) if len(r) > 1 and r.std() > 0 else float("nan")
    downside = r[r < 0]
    sortino = float(r.mean() / downside.std() * np.sqrt(ppy)) if len(downside) > 1 and downside.std() > 0 else float("nan")
    base = ann if annualized else total_ret
    calmar = float(base / abs(max_dd)) if max_dd < 0 and np.isfinite(base) else float("nan")
    # 月度胜率
    m = s.resample("ME").last().pct_change().dropna() if isinstance(s.index, pd.DatetimeIndex) else pd.Series(dtype=float)
    win = float((m > 0).mean()) if len(m) else float("nan")
    return {"ann_return": ann, "ann_vol": vol, "sharpe": sharpe, "sortino": sortino,
            "max_drawdown": max_dd, "calmar": calmar, "monthly_win_rate": win,
            "total_return": total_ret, "periods_per_year": ppy, "annualized": annualized,
            "n_days": n}


def statistical_diagnostics(
    eval_result: Dict,
    portfolio: Optional[Dict] = None,
    ic_series: Optional[pd.Series] = None,
) -> Dict:
    """把回测 dict 体检成一份结构化统计画像（供图表与大模型共同消费）。

    Args:
        eval_result: :meth:`engine.backtest.FactorBacktester.evaluate` 的返回值。
        portfolio: :meth:`~engine.backtest.FactorBacktester.realistic_portfolio` 的返回值（可选）。
        ic_series: 显式提供 IC 序列（缺省取 eval_result['_ic_series']）。

    Returns:
        分段字典：{"ic": {...}, "distribution": {...}, "decay": {...},
                  "quantile": {...}, "nav": {...}, "trade": {...}}
    """
    if not eval_result:
        return {}
    ic = ic_series if ic_series is not None else eval_result.get("_ic_series")
    ic = pd.Series(ic, dtype=float) if ic is not None else pd.Series(dtype=float)
    ic = ic.replace([np.inf, -np.inf], np.nan).dropna()

    out: Dict = {}
    # --- IC 显著性 ---
    n = len(ic)
    mean = float(ic.mean()) if n else float("nan")
    std = float(ic.std(ddof=1)) if n > 1 else float("nan")
    out["ic"] = {
        "n": n,
        "mean": mean,
        "std": std,
        "icir": (mean / std) if std and np.isfinite(std) and std > 0 else float("nan"),
        "positive_ratio": float((ic > 0).mean()) if n else float("nan"),
        "t_naive": float(mean / std * np.sqrt(n)) if std and std > 0 and n else float("nan"),
        "t_newey_west": newey_west_t(ic),
        "bootstrap_ci": bootstrap_ic_ci(ic),
    }

    # --- 分布形态 ---
    if n >= 8:
        skew = float(ic.skew())
        kurt = float(ic.kurt())
        jb_p = float("nan")
        if _HAS_SCIPY:
            try:
                jb_p = float(_st.jarque_bera(ic.to_numpy(dtype=float))[1])
            except Exception:  # noqa: BLE001
                jb_p = float("nan")
        # Ljung-Box 风格自相关检验（前 5 阶）
        acf = [float(ic.autocorr(lag=k)) for k in range(1, 6)]
        q = float(len(ic) * (len(ic) + 2) * sum(a ** 2 / max(1, len(ic) - i - 1)
                                                for i, a in enumerate(acf)))
        lb_p = float("nan")
        if _HAS_SCIPY:
            try:
                lb_p = float(1 - _st.chi2.cdf(q, df=5))
            except Exception:  # noqa: BLE001
                lb_p = float("nan")
        out["distribution"] = {"skew": skew, "kurtosis": kurt, "jarque_bera_p": jb_p,
                               "acf_1_5": acf, "ljung_box_stat": q, "ljung_box_p": lb_p}
    else:
        out["distribution"] = {}

    out["decay"] = ic_decay(ic).to_dict() if len(ic) > 12 else {}

    # --- 分位结构 ---
    qr = eval_result.get("quantile_returns")
    out["quantile"] = {
        **(quantile_monotonicity(qr) if isinstance(qr, dict) else {}),
        "quantile_returns": qr if isinstance(qr, dict) else {},
        "long_short_return": eval_result.get("long_short_return"),
        "long_short_sharpe": eval_result.get("long_short_sharpe"),
        "max_drawdown": eval_result.get("max_drawdown"),
        "annualized_volatility": eval_result.get("annualized_volatility"),
    }

    # --- 多空 / 组合净值 ---
    ls = eval_result.get("_ls_series")
    if ls is not None and len(pd.Series(ls).dropna()) > 5:
        out["nav"] = _drawdown_stats(pd.Series(ls, dtype=float).dropna())
    elif portfolio and isinstance(portfolio.get("equity"), pd.Series):
        out["nav"] = _drawdown_stats(portfolio["equity"])
    else:
        out["nav"] = {}

    # --- 交易与覆盖 ---
    out["trade"] = {
        "turnover": eval_result.get("turnover"),
        "avg_turnover": (portfolio or {}).get("metrics", {}).get("avg_turnover"),
        "coverage": eval_result.get("coverage"),
        "n_stocks": eval_result.get("n_stocks"),
        "n_dates": eval_result.get("n_dates"),
        "assumptions": (portfolio or {}).get("assumptions", {}),
        "portfolio_metrics": (portfolio or {}).get("metrics", {}),
    }
    return out


# ======================================================================
# 二、图表
# ======================================================================

def render_charts(
    eval_result: Dict,
    portfolio: Optional[Dict] = None,
    diag: Optional[Dict] = None,
    output_dir: str = "output/factor_report",
    factor_name: str = "factor",
    factors: Optional[pd.DataFrame] = None,
) -> Dict[str, str]:
    """生成 12 张分析图表，返回 {图表名: 文件路径}。"""
    _setup_style()
    os.makedirs(output_dir, exist_ok=True)
    diag = diag or statistical_diagnostics(eval_result, portfolio)
    paths: Dict[str, str] = {}
    tag = _safe_name(factor_name)

    def _save(fig, key: str) -> None:
        p = os.path.join(output_dir, f"{tag}_{key}.png")
        try:
            fig.tight_layout()
            fig.savefig(p, bbox_inches="tight")
            paths[key] = p
        except Exception:  # noqa: BLE001
            pass
        finally:
            plt.close(fig)

    ic = eval_result.get("_ic_series")
    ic = pd.Series(ic, dtype=float).dropna() if ic is not None else pd.Series(dtype=float)

    # 1) IC 序列 + 累计 IC
    if len(ic) > 3:
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.bar(range(len(ic)), ic.values, color=_PALETTE[0], alpha=0.65, width=1.0)
        ax.axhline(0, color="#333", lw=0.8)
        ax.axhline(float(ic.mean()), color=_PALETTE[6], lw=1.2, ls="--")
        ax2 = ax.twinx()
        ax2.plot(range(len(ic)), ic.cumsum().values, color=_PALETTE[4], lw=1.8)
        ax2.set_ylabel("累计 IC")
        ax.set_ylabel("IC")
        ax.set_xlabel("期数")
        _save(fig, "ic_series")

        # 2) IC 分布 + 正态拟合
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.hist(ic.values, bins=30, color=_PALETTE[1], alpha=0.8, density=True)
        if _HAS_SCIPY and ic.std() > 0:
            xs = np.linspace(ic.min(), ic.max(), 120)
            ax.plot(xs, _st.norm.pdf(xs, ic.mean(), ic.std()), color=_PALETTE[6], lw=1.6)
        ax.axvline(float(ic.mean()), color="#333", ls="--", lw=1.0)
        ax.set_xlabel("IC"); ax.set_ylabel("密度")
        _save(fig, "ic_distribution")

        # 3) 滚动 IC 与滚动 ICIR
        if len(ic) > 60:
            w = min(60, max(20, len(ic) // 5))
            roll_mean = ic.rolling(w).mean()
            roll_icir = ic.rolling(w).mean() / ic.rolling(w).std()
            fig, ax = plt.subplots(figsize=(9, 3.6))
            ax.plot(range(len(ic)), roll_mean.values, color=_PALETTE[0], label=f"滚动 IC({w})")
            ax2 = ax.twinx()
            ax2.plot(range(len(ic)), roll_icir.values, color=_PALETTE[6], label="滚动 ICIR")
            ax2.axhline(0, color="#333", lw=0.6)
            ax.set_xlabel("期数"); ax.set_ylabel("滚动 IC")
            ax2.set_ylabel("滚动 ICIR")
            _save(fig, "rolling_ic")

        # 4) IC 衰减
        dec = diag.get("decay") or {}
        if dec:
            fig, ax = plt.subplots(figsize=(6.4, 3.2))
            ks = list(dec.keys())
            ax.bar(ks, [dec[k] for k in ks], color=_PALETTE[2])
            ax.axhline(0, color="#333", lw=0.8)
            ax.set_ylabel("滞后自相关")
            _save(fig, "ic_decay")

    # 5) 分位数收益
    qr = eval_result.get("quantile_returns")
    if isinstance(qr, dict) and qr:
        items = sorted((int(k), float(v)) for k, v in qr.items())
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.bar([f"Q{k}" for k, _ in items], [v for _, v in items], color=_PALETTE[1])
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_ylabel("平均收益")
        _save(fig, "quantile_returns")

    # 6) 分位累计净值
    qc = eval_result.get("quantile_cum")
    if isinstance(qc, dict) and qc:
        fig, ax = plt.subplots(figsize=(9, 3.8))
        for i, (k, v) in enumerate(sorted(qc.items(), key=lambda x: int(x[0]))):
            s = pd.Series(v, dtype=float).dropna()
            ax.plot(range(len(s)), s.values, label=f"Q{k}", color=_PALETTE[i % len(_PALETTE)], lw=1.3)
        ax.legend(ncol=5, fontsize=8)
        ax.set_xlabel("期数"); ax.set_ylabel("累计净值")
        _save(fig, "quantile_cum")

    # 7) 多空净值 + 回撤
    ls = eval_result.get("_ls_series")
    nav = pd.Series(ls, dtype=float).dropna() if ls is not None else pd.Series(dtype=float)
    if nav.empty and portfolio and isinstance(portfolio.get("equity"), pd.Series):
        nav = portfolio["equity"].dropna()
    if len(nav) > 5:
        cum = nav.cumsum() if nav.abs().median() < 0.5 else nav / nav.iloc[0]
        dd = cum / cum.cummax() - 1.0
        fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
        axes[0].plot(range(len(cum)), cum.values, color=_PALETTE[0], lw=1.6)
        axes[0].set_ylabel("累计净值")
        axes[1].fill_between(range(len(dd)), dd.values, 0, color=_PALETTE[6], alpha=0.55)
        axes[1].set_ylabel("回撤")
        axes[1].set_xlabel("期数")
        _save(fig, "nav_drawdown")

        # 8) 月度收益热力图
        if isinstance(nav.index, pd.DatetimeIndex) and len(nav) > 60:
            try:
                m = nav.resample("ME").sum()
                tbl = pd.DataFrame({"y": m.index.year, "m": m.index.month, "v": m.values})
                piv = tbl.pivot_table(index="y", columns="m", values="v", aggfunc="sum")
                fig, ax = plt.subplots(figsize=(8.4, max(2.2, 0.55 * len(piv))))
                im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto")
                ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
                ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
                fig.colorbar(im, ax=ax, shrink=0.8)
                _save(fig, "monthly_heatmap")
            except Exception:  # noqa: BLE001
                pass

        # 9) 收益分布
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        r = nav.diff().dropna() if nav.abs().median() > 0.5 else nav
        ax.hist(r.values, bins=40, color=_PALETTE[2], alpha=0.85)
        ax.axvline(float(np.mean(r)), color="#333", ls="--", lw=1.0)
        ax.set_xlabel("收益"); ax.set_ylabel("频数")
        _save(fig, "return_distribution")

    # 10) 换手率序列
    if portfolio and portfolio.get("rebalance_list"):
        tos = []
        pw: Dict[str, float] = {}
        for rb in portfolio["rebalance_list"]:
            w = rb.get("weights", {})
            tos.append(0.5 * sum(abs(w.get(s, 0) - pw.get(s, 0)) for s in set(pw) | set(w)))
            pw = w
        if tos:
            fig, ax = plt.subplots(figsize=(9, 3.2))
            ax.plot(range(len(tos)), tos, color=_PALETTE[3], lw=1.2)
            ax.axhline(float(np.mean(tos)), color=_PALETTE[6], ls="--", lw=1.2)
            ax.set_xlabel("调仓次数"); ax.set_ylabel("单边换手率")
            _save(fig, "turnover")

    # 11) 因子相关性热力图（多因子时）
    if factors is not None and factors.shape[1] > 1:
        try:
            c = factors.apply(pd.to_numeric, errors="coerce").corr()
            fig, ax = plt.subplots(figsize=(max(4.0, 0.7 * len(c)), max(3.4, 0.7 * len(c))))
            im = ax.imshow(c.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(c))); ax.set_yticklabels(c.columns)
            fig.colorbar(im, ax=ax, shrink=0.8)
            _save(fig, "factor_correlation")
        except Exception:  # noqa: BLE001
            pass

    # 12) 关键指标汇总条形图
    iconf = diag.get("ic", {})
    keys = ["mean", "icir", "positive_ratio", "t_newey_west"]
    vals = [iconf.get(k, float("nan")) for k in keys]
    if any(np.isfinite(v) for v in vals):
        fig, ax = plt.subplots(figsize=(6.8, 3.2))
        ax.bar(["IC均值", "ICIR", "IC为正比例", "NW-t"], [0 if not np.isfinite(v) else v for v in vals],
               color=[_PALETTE[i] for i in range(4)])
        ax.set_ylabel("数值")
        _save(fig, "metrics_summary")

    return paths


# ======================================================================
# 三、大模型解读
# ======================================================================

_INTERPRET_SYSTEM = (
    "你是一名量化研究员，负责审阅因子回测结果并给出可执行结论。"
    "要求：1) 只依据提供的统计数字，不得编造任何未给出的指标；"
    "2) 先判断信号是否统计显著（看 Newey-West t 与 bootstrap 置信区间是否跨越 0），"
    "再判断经济显著性（扣掉换手成本后是否还有超额）；"
    "3) 明确指出最大的风险点（过拟合/衰减过快/分位不单调/回撤/容量）；"
    "4) 给出 3 条以内的下一步改进建议，必须具体可执行；"
    "5) 用中文、条目化输出，不要客套话。"
)


def build_llm_payload(diag: Dict, chart_paths: Optional[Dict[str, str]] = None,
                      factor_name: str = "", factor_expr: str = "") -> str:
    """把统计画像压缩成大模型可读的紧凑文本（图表以清单 + 关键数值表示）。"""
    payload = {
        "因子名称": factor_name,
        "因子表达式": factor_expr,
        "IC统计": diag.get("ic", {}),
        "分布与自相关": diag.get("distribution", {}),
        "IC衰减": diag.get("decay", {}),
        "分位与多空": diag.get("quantile", {}),
        "净值与回撤": diag.get("nav", {}),
        "交易与成本": diag.get("trade", {}),
        "已生成图表": sorted((chart_paths or {}).keys()),
    }

    def _round(o):
        if isinstance(o, float):
            return round(o, 6) if np.isfinite(o) else None
        if isinstance(o, dict):
            return {k: _round(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_round(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            v = float(o)
            return round(v, 6) if np.isfinite(v) else None
        return o

    return json.dumps(_round(payload), ensure_ascii=False, indent=1)


def _rule_based_interpretation(diag: Dict) -> str:
    """LLM 不可用时的规则化结论（保证离线也能拿到解读，而非空白）。"""
    ic = diag.get("ic", {})
    mean = ic.get("mean")
    t_nw = ic.get("t_newey_west")
    lo, hi = ic.get("bootstrap_ci", {}).get("lo"), ic.get("bootstrap_ci", {}).get("hi")
    lines: List[str] = ["【离线规则化解读（LLM 未启用）】"]
    if not np.isfinite(mean) if isinstance(mean, float) else mean is None:
        lines.append("- IC 序列缺失或样本不足，无法给出结论。")
        return "\n".join(lines)
    lines.append(f"- IC 均值 {mean:.4f}，ICIR {ic.get('icir', float('nan')):.4f}，"
                 f"IC 为正比例 {ic.get('positive_ratio', float('nan')):.1%}。")
    if np.isfinite(t_nw) if isinstance(t_nw, float) else t_nw is not None:
        sig = abs(t_nw) >= 2.0
        lines.append(f"- Newey-West t = {t_nw:.2f}（|t|≥2 视为显著）：{'显著' if sig else '不显著'}。")
    if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
        lines.append(f"- IC 均值 95% 置信区间 [{lo:.4f}, {hi:.4f}]："
                     f"{'不含 0，方向可信' if lo * hi > 0 else '跨越 0，方向不可靠'}。")
    q = diag.get("quantile", {})
    sp = q.get("spearman")
    if sp is not None and np.isfinite(sp):
        lines.append(f"- 分位单调性 Spearman = {sp:.3f}："
                     f"{'单调性良好' if abs(sp) > 0.8 else '单调性偏弱，可能存在个别分位主导'}。")
    dec = diag.get("decay", {})
    if dec:
        l1 = dec.get("lag1")
        if l1 is not None and np.isfinite(l1):
            lines.append(f"- IC 一阶自相关 {l1:.3f}："
                         f"{'信号偏慢，可用低频换手' if l1 > 0.3 else '衰减快，需高频换手，注意成本'}。")
    nav = diag.get("nav", {})
    if nav.get("max_drawdown") is not None and np.isfinite(nav["max_drawdown"]):
        tag = "年化 Calmar" if nav.get("annualized") else "区间 Calmar"
        lines.append(f"- 最大回撤 {nav['max_drawdown']:.2%}，{tag} "
                     f"{nav.get('calmar', float('nan')):.3f}"
                     f"（区间累计收益 {nav.get('total_return', float('nan')):.2%}）。")
    lines.append("- 建议：补充样本外/跨市场验证，并检查换手成本敏感性后再上线。")
    return "\n".join(lines)


def llm_interpret(diag: Dict, chart_paths: Optional[Dict[str, str]] = None,
                  factor_name: str = "", factor_expr: str = "",
                  llm_client=None, temperature: float = 0.2) -> Dict[str, str]:
    """调用大模型给出深度解读；不可用时降级为规则化结论。

    Returns:
        {mode: 'llm'|'rule', text: 解读文本, payload: 喂给模型的统计画像}
    """
    payload = build_llm_payload(diag, chart_paths, factor_name, factor_expr)
    client = llm_client
    if client is None:
        try:
            from llm.client import LLMClient
            client = LLMClient()
        except Exception:  # noqa: BLE001
            client = None
    if client is not None:
        try:
            text = client.complete(_INTERPRET_SYSTEM,
                                   f"以下是因子回测的完整统计画像：\n{payload}\n\n请给出审阅结论。",
                                   temperature=temperature)
            if text and text.strip():
                return {"mode": "llm", "text": text.strip(), "payload": payload}
        except Exception as e:  # noqa: BLE001
            return {"mode": "rule",
                    "text": _rule_based_interpretation(diag) + f"\n\n（LLM 调用失败：{type(e).__name__}：{e}）",
                    "payload": payload}
    return {"mode": "rule", "text": _rule_based_interpretation(diag), "payload": payload}


# ======================================================================
# 四、总入口
# ======================================================================

@dataclass
class FactorReport:
    """一次完整分析的输出。"""

    factor_name: str = ""
    diagnostics: Dict = field(default_factory=dict)
    charts: Dict[str, str] = field(default_factory=dict)
    interpretation: Dict[str, str] = field(default_factory=dict)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """把统计体检 + 图表清单 + 模型解读拼成一份 Markdown 报告。"""
        d = self.diagnostics
        lines = [f"# 因子分析报告：{self.factor_name or '(未命名)'}", ""]
        ic = d.get("ic", {})
        if ic:
            lines += ["## 1. IC 统计", "",
                      f"- 样本期数：{ic.get('n')}",
                      f"- IC 均值：{ic.get('mean'):.4f}，标准差：{ic.get('std'):.4f}，ICIR：{ic.get('icir'):.4f}",
                      f"- IC 为正比例：{ic.get('positive_ratio'):.1%}",
                      f"- t 检验：朴素 {ic.get('t_naive'):.2f}，Newey-West {ic.get('t_newey_west'):.2f}"]
            ci = ic.get("bootstrap_ci", {})
            if ci:
                lines.append(f"- Bootstrap 95% CI：[{ci.get('lo'):.4f}, {ci.get('hi'):.4f}]")
            lines.append("")
        dist = d.get("distribution", {})
        if dist:
            lines += ["## 2. 分布与自相关", "",
                      f"- 偏度 {dist.get('skew'):.3f}，峰度 {dist.get('kurtosis'):.3f}，"
                      f"Jarque-Bera p={dist.get('jarque_bera_p')}",
                      f"- Ljung-Box(5) 统计量 {dist.get('ljung_box_stat'):.2f}，p={dist.get('ljung_box_p')}",
                      ""]
        if d.get("decay"):
            lines += ["## 3. IC 衰减", "",
                      "  ".join(f"{k}={v:.3f}" for k, v in d["decay"].items()), ""]
        nav = d.get("nav", {})
        if nav:
            basis = "年化口径" if nav.get("annualized") else \
                f"区间口径（样本 {nav.get('n_days')} 期 < 半年，不做年化外推）"
            ann_line = (f"- 年化收益 {nav['ann_return']:.2%}，年化波动 {nav['ann_vol']:.2%}"
                        if nav.get("annualized") and np.isfinite(nav.get("ann_return", float("nan")))
                        else "- 年化收益 —（样本不足半年，不做年化外推）")
            lines += ["## 4. 净值表现", "",
                      f"- 区间累计收益 {nav.get('total_return'):.2%}；统计口径：{basis}",
                      ann_line,
                      f"- Sharpe {nav.get('sharpe'):.3f}，Sortino {nav.get('sortino'):.3f}",
                      f"- 最大回撤 {nav.get('max_drawdown'):.2%}，Calmar {nav.get('calmar'):.3f}",
                      f"- 月度胜率 {nav.get('monthly_win_rate'):.1%}", ""]
        if self.charts:
            lines += ["## 5. 图表", ""] + [f"- {k}：`{v}`" for k, v in sorted(self.charts.items())] + [""]
        if self.interpretation.get("text"):
            lines += [f"## 6. 分析结论（{self.interpretation.get('mode')}）", "",
                      self.interpretation["text"], ""]
        return "\n".join(lines)


def generate_factor_report(
    eval_result: Dict,
    portfolio: Optional[Dict] = None,
    factor_name: str = "",
    factor_expr: str = "",
    factors: Optional[pd.DataFrame] = None,
    output_dir: str = "output/factor_report",
    llm_client=None,
    make_charts: bool = True,
) -> FactorReport:
    """一次性生成「统计体检 + 图表 + 大模型解读」的完整报告。

    Args:
        eval_result: FactorBacktester.evaluate 的返回值。
        portfolio: FactorBacktester.realistic_portfolio 的返回值（可选）。
        factor_name / factor_expr: 因子名称与表达式（进入模型上下文）。
        factors: 多因子截面矩阵（可选，用于相关性热力图）。
        output_dir: 图表输出目录。
        llm_client: 已构造的 LLMClient；缺省自动构建，失败则降级。
    """
    rep = FactorReport(factor_name=factor_name)
    rep.diagnostics = statistical_diagnostics(eval_result, portfolio)
    if make_charts:
        try:
            rep.charts = render_charts(eval_result, portfolio, rep.diagnostics,
                                       output_dir=output_dir, factor_name=factor_name,
                                       factors=factors)
        except Exception as e:  # noqa: BLE001
            rep.charts = {"error": f"图表生成失败：{type(e).__name__}: {e}"}
    rep.interpretation = llm_interpret(rep.diagnostics, rep.charts, factor_name,
                                       factor_expr, llm_client=llm_client)

    # 关键指标表（便于 UI 直接展示）
    rows = []
    for section, items in rep.diagnostics.items():
        if not isinstance(items, dict):
            continue
        for k, v in items.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                rows.append({"类别": section, "指标": k, "数值": float(v)})
    if rows:
        rep.tables["metrics"] = pd.DataFrame(rows)

    # 落盘 Markdown
    try:
        os.makedirs(output_dir, exist_ok=True)
        p = os.path.join(output_dir, f"{_safe_name(factor_name) or 'factor'}_report.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write(rep.to_markdown())
        rep.charts.setdefault("_report_md", p)
    except Exception:  # noqa: BLE001
        pass
    return rep
