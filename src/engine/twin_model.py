"""风险模型 + 预测模型双引擎（src/engine/twin_model.py）。

把「预测收益」和「控制风险」拆成两个各司其职的模型，而不是用一个模型干两件事：

- 预测模型（``fit_robust_predictor``）：追求**鲁棒性**。
  提供 OLS / Huber / LAD(L1) 三种估计，用 IRLS 自实现（零新增依赖），
  并用扰动压力测试（加噪、剔除极端样本）比较它们谁在脏数据下不崩。
  默认推荐 Huber：对异常收益不敏感，又保留大部分效率。

- 风险模型（``InterpretableRiskModel``）：追求**解释力**。
  结构化风险分解：把组合风险拆到「风格因子暴露」与「个股特异」两处，
  给出边际风险贡献 MCTR、成分风险贡献 CCTR、风险集中度与有效下注数，
  回答「风险到底来自哪个因子、哪几只票」，而不是只给一个波动率数字。

二者通过 ``combine_signal_and_risk`` 衔接：预测模型给信号，风险模型给约束
（限制单因子暴露与集中度），形成「预测 + 风控」的闭环。

零新增依赖：numpy/pandas 为必需，scipy 为可选（缺失时统计检验降级）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    _HAS_SCIPY = True
except Exception:  # noqa: BLE001
    _HAS_SCIPY = False


# ======================================================================
# 一、鲁棒预测模型
# ======================================================================

def _scale_from_resid(r: np.ndarray) -> float:
    """残差尺度估计：1.4826 × MAD（对异常值稳健），退化为 0 时用标准差。"""
    mad = float(np.median(np.abs(r - np.median(r))))
    s = 1.4826 * mad
    if s <= 1e-12:
        s = float(np.std(r))
    return max(s, 1e-12)


def fit_robust_predictor(
    x: pd.DataFrame,
    y: pd.Series,
    method: str = "huber",
    fit_intercept: bool = True,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> Dict:
    """鲁棒回归拟合（IRLS 自实现）。

    Args:
        method: 'ols'（最小二乘，效率最高但对异常值敏感）
                'huber'（默认，折中：小残差平方、大残差线性）
                'lad'（最小绝对偏差，最抗异常值但效率较低）
        fit_intercept: 是否拟合截距。

    Returns:
        {coef: pd.Series, intercept: float, method, n_iter, scale, fitted, resid}
    """
    if x is None or x.empty or y is None:
        return {}
    xm = x.apply(pd.to_numeric, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    ok = xm.notna().all(axis=1) & yv.notna()
    xm, yv = xm[ok], yv[ok]
    if len(xm) < 5:
        return {}

    cols = list(xm.columns)
    a = xm.to_numpy(dtype=float)
    b = yv.to_numpy(dtype=float)
    if fit_intercept:
        a = np.column_stack([np.ones(len(a)), a])
        names = ["(intercept)"] + cols  # noqa: RUF005  热路径，保持可读
    else:
        names = list(cols)

    beta = np.linalg.lstsq(a, b, rcond=None)[0]
    if method in ("ols", "ridge"):
        resid = b - a @ beta
        return {
            "coef": pd.Series(beta[1:] if fit_intercept else beta, index=cols),
            "intercept": float(beta[0]) if fit_intercept else 0.0,
            "method": "ols", "n_iter": 1,
            "scale": _scale_from_resid(resid),
            "fitted": pd.Series(a @ beta, index=xm.index),
            "resid": pd.Series(resid, index=xm.index),
        }

    # IRLS：Huber / LAD 统一为「残差加权最小二乘」
    n_iter = 0
    for it in range(int(max_iter)):
        resid = b - a @ beta
        scale = _scale_from_resid(resid)
        u = resid / scale
        if method == "huber":
            delta = 1.345
            w = np.where(np.abs(u) <= delta, 1.0, delta / np.maximum(np.abs(u), 1e-12))
        else:  # lad / l1
            w = 1.0 / np.maximum(np.abs(u), 1e-6)
        aw = a * np.sqrt(w)[:, None]
        bw = b * np.sqrt(w)
        new_beta = np.linalg.lstsq(aw, bw, rcond=None)[0]
        n_iter = it + 1
        if np.max(np.abs(new_beta - beta)) < tol:
            beta = new_beta
            break
        beta = new_beta

    resid = b - a @ beta
    return {
        "coef": pd.Series(beta[1:] if fit_intercept else beta, index=cols),
        "intercept": float(beta[0]) if fit_intercept else 0.0,
        "method": method, "n_iter": n_iter,
        "scale": _scale_from_resid(resid),
        "fitted": pd.Series(a @ beta, index=xm.index),
        "resid": pd.Series(resid, index=xm.index),
    }


def robustness_curve(
    x: pd.DataFrame,
    y: pd.Series,
    methods: Sequence[str] = ("ols", "huber", "lad"),
    noise_levels: Sequence[float] = (0.0, 0.10, 0.25, 0.50),
    oos_ratio: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """扰动压力测试：比较各估计量在脏数据下的性能衰减。

    两类扰动：① 对目标 y 叠加高斯噪声（σ = level × std(y)）；
    ② 剔除 |y| 最极端的 1% 样本。评价用**样本外** RankIC 与 R²（时序末段切分）。

    Returns:
        DataFrame，行= (method, 扰动场景)，列= oos_ic / oos_r2 / decay（相对无噪声的衰减）。
    """
    if x is None or x.empty or y is None or len(x) < 20:
        return pd.DataFrame()
    xm = x.apply(pd.to_numeric, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    ok = xm.notna().all(axis=1) & yv.notna()
    xm, yv = xm[ok], yv[ok]
    n = len(xm)
    cut = max(1, int(n * (1 - oos_ratio)))
    x_tr, x_te = xm.iloc[:cut], xm.iloc[cut:]
    y_tr, y_te = yv.iloc[:cut], yv.iloc[cut:]
    if len(x_te) < 5:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    sd = float(y_tr.std()) or 1.0
    rows: List[Dict] = []
    for m in methods:
        base_ic = None
        for level in noise_levels:
            y_noise = y_tr.copy()
            if level > 0:
                y_noise = y_noise + rng.normal(scale=sd * level, size=len(y_tr))
            res = fit_robust_predictor(x_tr, y_noise, method=m)
            if not res:
                continue
            a_te = x_te.to_numpy(dtype=float)
            pred = res["intercept"] + a_te @ res["coef"].reindex(x_te.columns).fillna(0.0).to_numpy()
            ic = _rank_ic(pred, y_te.to_numpy(dtype=float))
            ss_res = float(((y_te.to_numpy(dtype=float) - pred) ** 2).sum())
            ss_tot = float(((y_te.to_numpy(dtype=float) - y_te.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            if base_ic is None:
                base_ic = ic
            decay = (ic / base_ic - 1.0) if (base_ic and abs(base_ic) > 1e-9) else 0.0
            rows.append({"method": m, "scenario": f"noise={level:.0%}",
                         "oos_ic": ic, "oos_r2": r2, "decay": decay})

        # 剔除极端样本场景
        thr = float(y_tr.abs().quantile(0.99))
        keep = y_tr.abs() <= thr
        if keep.sum() > 10:
            res = fit_robust_predictor(x_tr[keep], y_tr[keep], method=m)
            if res:
                a_te = x_te.to_numpy(dtype=float)
                pred = res["intercept"] + a_te @ res["coef"].reindex(x_te.columns).fillna(0.0).to_numpy()
                ic = _rank_ic(pred, y_te.to_numpy(dtype=float))
                rows.append({"method": m, "scenario": "drop_top1%",
                             "oos_ic": ic, "oos_r2": float("nan"),
                             "decay": (ic / base_ic - 1.0) if (base_ic and abs(base_ic) > 1e-9) else 0.0})
    df = pd.DataFrame(rows)
    return df


def recommend_predictor(curve: pd.DataFrame) -> Dict:
    """依据压力测试结果推荐估计量：扰动下平均衰减最小者胜出（鲁棒性优先）。

    Returns: {method, mean_decay, mean_ic, by_method}
    """
    if curve is None or curve.empty:
        return {}
    g = curve.groupby("method", sort=False).agg(
        mean_ic=("oos_ic", "mean"),
        worst_ic=("oos_ic", "min"),
        mean_decay=("decay", "mean"),
    ).sort_values("mean_decay", ascending=False)  # decay 为负，越大（越接近 0）越好
    best = g.index[0]
    return {
        "method": best,
        "mean_decay": float(g.loc[best, "mean_decay"]),
        "mean_ic": float(g.loc[best, "mean_ic"]),
        "by_method": g,
    }


def _rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size != a.size:
        return 0.0
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


# ======================================================================
# 二、可解释风险模型
# ======================================================================

def ledoit_wolf_shrink(returns: pd.DataFrame, target: str = "diag") -> np.ndarray:
    """协方差收缩估计（对抗「样本不足 → 协方差估计失真」）。

    target='diag'：收缩到对角阵，强度用 Ledoit-Wolf(2003) 闭式估计；
    target='cc'：收缩到恒定相关矩阵，强度用保守的 min(N/T, 1)。
    """
    m = returns.apply(pd.to_numeric, errors="coerce").dropna() if returns is not None else None
    if m is None or len(m) < 3:
        return np.eye(returns.shape[1]) if returns is not None and returns.shape[1] else np.eye(1)
    y = m.to_numpy(dtype=float)
    t, n = y.shape
    s = np.cov(y, rowvar=False, ddof=1)
    if target == "cc" and n > 1:
        sd = np.sqrt(np.diag(s))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = s / np.outer(sd, sd)
        rbar = float(np.nansum(r) - n) / max(1, n * (n - 1))
        f = np.clip(rbar, -0.99, 0.99) * np.outer(sd, sd)
        np.fill_diagonal(f, np.diag(s))
        delta = float(min(1.0, n / max(1, t)))
        return (1 - delta) * s + delta * f
    # target='diag'：LW2003
    mu = float(np.trace(s) / n)
    f = mu * np.eye(n)
    d2 = float(((s - f) ** 2).sum())
    if d2 <= 1e-18:
        return s
    yc = y - y.mean(axis=0)
    pi_ii = float((((yc ** 2).T @ (yc ** 2)) / t - s ** 2).diagonal().sum())
    delta = float(np.clip(pi_ii / d2 / t, 0.0, 1.0))
    return (1 - delta) * s + delta * f


@dataclass
class RiskDecomposition:
    """组合风险分解结果（全部为可解释口径）。"""

    total_vol: float = 0.0                       # 年化波动率
    factor_var: float = 0.0                      # 因子风险（方差）
    specific_var: float = 0.0                    # 特异风险（方差）
    factor_contrib: pd.Series = field(default_factory=pd.Series)   # 各风格因子的风险贡献
    mctr: pd.Series = field(default_factory=pd.Series)             # 边际风险贡献（按标的）
    cctr: pd.Series = field(default_factory=pd.Series)             # 成分风险贡献
    concentration_hhi: float = 0.0               # 风险集中度（HHI）
    n_effective_bets: float = 0.0                # 有效下注数 = 1/HHI
    top_risk_names: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "total_vol": self.total_vol,
            "factor_var": self.factor_var,
            "specific_var": self.specific_var,
            "factor_share": self.factor_var / (self.factor_var + self.specific_var)
            if (self.factor_var + self.specific_var) > 0 else 0.0,
            "concentration_hhi": self.concentration_hhi,
            "n_effective_bets": self.n_effective_bets,
            "top_risk_names": self.top_risk_names,
        }


class InterpretableRiskModel:
    """结构化风险模型：风格因子 + 特异风险，输出可归因的风险分解。

    用法：
        rm = InterpretableRiskModel()
        rm.fit(exposures=df[['size','momentum','volatility']], returns=df['fwd_ret'])
        dec = rm.risk_decomposition(weights)   # weights: Series, index=symbol
        print(rm.explain(dec))

    Args:
        annualize: 年化系数（日频默认 252）。
    """

    def __init__(self, annualize: int = 252, shrink: bool = True) -> None:
        self.annualize = annualize
        self.shrink = shrink
        self.factor_cov: Optional[pd.DataFrame] = None
        self.specific_var: Optional[pd.Series] = None
        self.factor_names: List[str] = []
        self.resid_var: float = 0.0

    def fit(self, exposures: pd.DataFrame, returns: pd.Series,
            dates: Optional[pd.Series] = None) -> "InterpretableRiskModel":
        """用「截面回归 + 时序协方差」估计因子协方差 F 与特异方差 Δ。

        对每期做一次截面回归得到因子收益 f_t，再对 f_t 序列估协方差（可选收缩）；
        特异方差取各期残差方差的均值。

        Args:
            dates: 显式给出每行的日期（推荐）。缺省时按 returns 索引的首层分组，
                   索引非 MultiIndex 则退化为「全样本单期」，此时无法估计协方差。
        """
        if exposures is None or exposures.empty or returns is None:
            return self
        x = exposures.apply(pd.to_numeric, errors="coerce")
        y = pd.to_numeric(returns, errors="coerce")
        self.factor_names = list(x.columns)
        fac_ret: Dict[str, List[float]] = {c: [] for c in self.factor_names}
        resid_vars: List[float] = []

        if dates is not None:
            key = pd.Series(np.asarray(dates).reshape(-1), index=x.index)
            groups = [(d, y.loc[key.index[key == d]]) for d in pd.unique(key)]
        elif y.index.nlevels > 1:
            groups = list(y.groupby(level=0))
        else:
            groups = [("all", y)]

        for _, yy in groups:
            xx = x.loc[yy.index]
            ok = xx.notna().all(axis=1) & yy.notna()
            xx, yy2 = xx[ok], yy[ok]
            if len(xx) < max(3, len(self.factor_names) + 1):
                continue
            a = np.column_stack([np.ones(len(xx)), xx.to_numpy(dtype=float)])
            b = yy2.to_numpy(dtype=float)
            beta, *_ = np.linalg.lstsq(a, b, rcond=None)
            for i, c in enumerate(self.factor_names):
                fac_ret[c].append(beta[i + 1])
            resid_vars.append(float(np.var(b - a @ beta, ddof=1)))

        f_df = pd.DataFrame(fac_ret)
        if len(f_df) < 3:
            return self
        cov = ledoit_wolf_shrink(f_df) if self.shrink else np.cov(f_df.to_numpy(dtype=float), rowvar=False)
        self.factor_cov = pd.DataFrame(cov, index=self.factor_names, columns=self.factor_names)
        self.resid_var = float(np.mean(resid_vars)) if resid_vars else 0.0
        return self

    def risk_decomposition(self, weights: pd.Series,
                           exposures: Optional[pd.DataFrame] = None) -> RiskDecomposition:
        """对给定持仓做风险分解；exposures 缺省时用「因子暴露=1」的退化口径。

        不显式构造 N×N 协方差矩阵（N 大时既慢又占内存），而是用
            (Σw)_i = B_i' F (B'w) + Δ_i w_i
        直接计算，复杂度 O(N·K)。
        """
        if weights is None or weights.empty or self.factor_cov is None:
            return RiskDecomposition()
        w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
        if exposures is not None and not exposures.empty:
            b = exposures.reindex(w.index).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        else:
            b = pd.DataFrame(dict.fromkeys(self.factor_names, 1.0), index=w.index)
        f = self.factor_cov.to_numpy(dtype=float)
        bmat = b[self.factor_names].to_numpy(dtype=float)
        wv = w.to_numpy(dtype=float)

        fp = bmat.T @ wv                       # K 维组合因子暴露
        factor_var = float(fp @ f @ fp)
        specific_var = float(self.resid_var * np.sum(wv ** 2))
        total_var = factor_var + specific_var
        if total_var <= 0:
            return RiskDecomposition()

        # 因子层面的边际贡献（按 f_k × (Ff)_k 分解，和恰为因子方差）
        f_contrib = pd.Series(fp * (f @ fp), index=self.factor_names)

        # 标的层面：MCTR_i = (Σw)_i / σ_p，CCTR_i = w_i × MCTR_i
        sigma_w = bmat @ (f @ fp) + self.resid_var * wv
        vol = float(np.sqrt(total_var))
        mctr = pd.Series(sigma_w / vol, index=w.index)
        cctr = w * mctr
        # 欧拉分解：Σ_i CCTR_i = w'Σw / σ_p = σ_p，故占比须除以 σ_p（不是方差）才和为 1
        share = cctr / vol
        hhi = float((share ** 2).sum())
        top = [str(i) for i in cctr.abs().sort_values(ascending=False).head(5).index.tolist()]

        return RiskDecomposition(
            total_vol=vol * np.sqrt(self.annualize),
            factor_var=factor_var,
            specific_var=specific_var,
            factor_contrib=f_contrib,
            mctr=mctr,
            cctr=cctr,
            concentration_hhi=hhi,
            n_effective_bets=(1.0 / hhi) if hhi > 0 else float("nan"),
            top_risk_names=top,
        )

    def explain(self, dec: RiskDecomposition) -> str:
        """把风险分解翻译成人话（供 UI / LLM 报告直接展示）。"""
        if not dec.factor_contrib.size:
            return "风险模型未拟合或持仓为空，无法分解。"
        total = dec.factor_var + dec.specific_var
        lines = [
            f"年化波动率 {dec.total_vol:.2%}（因子风险占比 {dec.factor_var / total:.1%}，"
            f"特异风险占比 {dec.specific_var / total:.1%}）",
            f"风险集中度 HHI={dec.concentration_hhi:.4f}，"
            f"有效下注数≈{dec.n_effective_bets:.1f}（越接近持仓数越分散）",
        ]
        top_f = dec.factor_contrib.reindex(
            dec.factor_contrib.abs().sort_values(ascending=False).index)
        lines.append("风险贡献 Top 因子：" + "、".join(
            f"{k}={v / total:.1%}" for k, v in top_f.head(3).items()))
        if dec.top_risk_names:
            lines.append("风险贡献 Top 标的：" + "、".join(dec.top_risk_names[:5]))
        if dec.n_effective_bets < 5:
            lines.append("提示：有效下注数偏低，组合风险集中于少数标的，建议放宽持仓数或做行业中性。")
        return "\n".join(lines)


def combine_signal_and_risk(
    signal: pd.Series,
    exposures: Optional[pd.DataFrame],
    risk_model: InterpretableRiskModel,
    max_factor_exposure: float = 0.5,
    max_weight: float = 0.1,
) -> pd.Series:
    """预测信号 → 经风险约束的权重（预测模型与风险模型的衔接点）。

    先按信号排序取多头，再对权重做两步约束：
      ① 单标的上限（控制集中度）；
      ② 单风格因子暴露上限（控制因子押注），超出时按等比例缩放至阈值。

    Returns:
        归一化后的权重（和为 1）。
    """
    if signal is None or signal.empty:
        return signal if signal is not None else pd.Series(dtype=float)
    s = pd.to_numeric(signal, errors="coerce").dropna().sort_values(ascending=False)
    if s.empty:
        return pd.Series(dtype=float)
    w = pd.Series(1.0, index=s.index)
    w = w / w.sum()
    if max_weight > 0:
        w = w.clip(upper=max_weight)
        w = w / w.sum()
    if exposures is not None and not exposures.empty and risk_model.factor_cov is not None:
        b = exposures.reindex(w.index).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        cols = [c for c in risk_model.factor_names if c in b.columns]
        if cols:
            fp = b[cols].to_numpy(dtype=float).T @ w.to_numpy(dtype=float)
            scale = 1.0
            for v in np.abs(fp):
                if v > max_factor_exposure and v > 0:
                    scale = min(scale, max_factor_exposure / v)
            if scale < 1.0:
                # 因子暴露超限：向等权方向收缩（保留排序，降低押注强度）
                equal = pd.Series(1.0, index=w.index) / len(w)
                w = equal + scale * (w - equal)
                w = w.clip(lower=0.0)
                w = w / w.sum() if w.sum() > 0 else equal
    return w
