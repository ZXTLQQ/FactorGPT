# -*- coding: utf-8 -*-
"""风险因子视角的因子评估（天风证券《BetaAgent：交易风险因子挖掘和评估系统》落地）。

与常规 alpha 评价（IC 越大越好）相反，交易风险因子的验收标准在报告里被明确
写成六道**硬闸门**，本模块逐条实现：

1. **增量解释力**：在既有风险模型暴露（规模/β/动量/残差波动/流动性/反转）之上
   的 ΔR² 与调整 ΔR² 必须为正且达阈值 —— 因子要"解释别人没解释的方差"。
2. **回归系数显著性**：日截面回归系数 |t| 均值、|t|>2 的天数占比 —— 拒绝
   "靠少数极端日撑起来的"因子。
3. **低共线性**：对风险模型暴露的 VIF 上限（岭回归稳健估计，避免完全共线爆表）。
4. **时序稳定性**：因子自相关 L1~L5 必须高、衰减必须小 —— 风险因子要"耐扛"。
5. **风险型 IC 判据（反直觉但关键）**：|RankIC| 均值要高，但**带方向的
   mean(RankIC) 要接近 0** —— 即风险因子应解释横截面差异，却不该提供方向性
   alpha（否则它是 alpha 因子而非风险因子）。
6. **拥挤度**：与既有因子池的相关性、对风险暴露的解释集中度 —— 拥挤的风险
   因子在极端行情会失效。

另提供因子收益分解与组合风险贡献（Newey-West 协方差），用于回答
"组合波动里有多少来自这个因子"。
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import ops
from .panel import PanelData

STYLE_SPECS: Dict[str, str] = {
    "size": "规模：log(20 日均成交额) 横截面标准化",
    "beta": "市场 β：60 日收益对等权市场收益的回归斜率",
    "momentum": "动量：20 日收益",
    "resid_vol": "残差波动：对市场回归后残差收益的 20 日标准差",
    "liquidity": "流动性：Amihud 非流动性 log(1+|ret|/成交额) 的 20 日均值",
    "reversal": "反转：近 5 日收益取负",
}


def broadcast(series: pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    """把单列序列（如市场收益）广播成与 ``like`` 同形状的宽表。

    时间轴按索引对齐，横截面方向复制；这是把「时序序列」喂给逐列滚动
    算子的标准做法（也用于概念因子的市场参照）。
    """
    v = series.reindex(like.index).to_numpy(dtype=np.float64)[:, None]
    return pd.DataFrame(np.repeat(v, like.shape[1], axis=1),
                        index=like.index, columns=like.columns)


def default_risk_exposures(panel: PanelData, beta_window: int = 60,
                           vol_window: int = 20, mom_window: int = 20,
                           liq_window: int = 20,
                           reversal_window: int = 5) -> Dict[str, pd.DataFrame]:
    """构造默认风格风险暴露（离线数据即可算出，不依赖外部风险模型文件）。"""
    ret = panel.field("ret")
    amount = panel.field("amount")
    close = panel.field("close")
    # 市场收益广播成与个股同形状的宽表，供滚动 beta / 残差波动使用
    mkt_wide = broadcast(panel.market_ret(), ret)

    beta = ops.rolling_binary(mkt_wide, ret, beta_window, "beta")
    resid = ret - beta * mkt_wide
    amihud = np.log1p(ret.abs() / amount.where(amount > 0))
    exposures = {
        "size": ops.cs_zscore(np.log(ops.rolling_unary(amount, 20, "mean"))),
        "beta": ops.cs_zscore(ops.rolling_unary(beta, 20, "mean")),
        "momentum": ops.cs_zscore(ops.ts_pct(close, mom_window)),
        "resid_vol": ops.cs_zscore(ops.rolling_unary(resid, vol_window, "std")),
        "liquidity": ops.cs_zscore(ops.rolling_unary(amihud, liq_window, "mean")),
        "reversal": ops.cs_zscore(-ops.ts_pct(close, reversal_window)),
    }
    return exposures


def align_exposures(exposures: Dict[str, pd.DataFrame],
                    index: pd.Index) -> Dict[str, pd.DataFrame]:
    return {k: v.reindex(index=index) for k, v in exposures.items()}


def install_risk_fields(panel: PanelData, **kwargs) -> Dict[str, pd.DataFrame]:
    """把六个风格暴露写回面板，使其在表达式里可按名引用。

    例：``risk.install_risk_fields(panel)`` 之后即可写
    ``neutral(zscore_cs(ts_pct(close, 20)), size, beta, resid_vol)``。
    """
    exps = default_risk_exposures(panel, **kwargs)
    for name, frame in exps.items():
        if name not in panel.fields:
            panel.fields[name] = frame.reindex(
                index=panel.dates, columns=panel.symbols).astype(float)
    return exps


# --------------------------------------------------------------------------
# 截面回归基元
# --------------------------------------------------------------------------
def _design(y: np.ndarray, xs: Sequence[np.ndarray],
            add_const: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    cols = ([np.ones_like(y)] if add_const else []) + list(xs)
    X = np.column_stack(cols)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[mask], y[mask]


def cross_sectional_ols(y: pd.DataFrame, xs: Sequence[pd.DataFrame],
                        add_const: bool = True, min_stocks: int = 20
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series,
                                   pd.DataFrame]:
    """逐日横截面 OLS。

    返回 ``(params, resid, r2, tstat)``：系数、残差面板、每日 R²、系数 t 值。
    样本不足或退化的交易日返回 NaN（不参与后续均值），绝不外推。
    """
    names = [f"x{i}" for i in range(len(xs))]
    params = pd.DataFrame(index=y.index, columns=([] if not add_const else ["const"]) + names,
                          dtype=float)
    tstats = pd.DataFrame(index=y.index, columns=params.columns, dtype=float)
    resid = y.copy().astype(float)
    r2 = pd.Series(np.nan, index=y.index, dtype=float)
    y_np = y.to_numpy(dtype=np.float64)
    x_np = [x.reindex_like(y).to_numpy(dtype=np.float64) for x in xs]
    for i, dt in enumerate(y.index):
        yy = y_np[i]
        cols = [x[i] for x in x_np]
        X, yv = _design(yy, cols, add_const)
        n, k = X.shape
        if n < max(min_stocks, k + 2):
            continue
        try:
            xtx_inv = np.linalg.pinv(X.T @ X)
            beta = xtx_inv @ (X.T @ yv)
        except np.linalg.LinAlgError:  # pragma: no cover - 极端退化
            continue
        fitted = X @ beta
        err = yv - fitted
        ss_res = float(err @ err)
        ss_tot = float(((yv - yv.mean()) ** 2).sum())
        dof = max(n - k, 1)
        sigma2 = ss_res / dof
        se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 1e-300))
        params.loc[dt] = beta
        tstats.loc[dt] = beta / se
        if ss_tot > 0:
            r2.loc[dt] = 1.0 - ss_res / ss_tot
        # 残差写回（未参与回归的样本保持 NaN）
        mask = np.isfinite(yy) & np.all(np.isfinite(np.column_stack(cols)), axis=1)
        row = np.full(len(yy), np.nan)
        row[mask] = err
        resid.iloc[i] = row
    return params, resid, r2, tstats


def _adj_r2(r2: pd.Series, n: pd.Series, k: int) -> pd.Series:
    return 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1).clip(lower=1)


def delta_r2(candidate: pd.DataFrame, fwd: pd.DataFrame,
             exposures: Dict[str, pd.DataFrame], min_stocks: int = 20,
             standardize: bool = True) -> Dict[str, float]:
    """在风险暴露之上加入候选因子后的增量解释力（天风闸门 1、2）。

    返回 ``delta_r2`` / ``delta_r2_adj`` / ``r2_base`` / ``r2_full`` /
    ``t_mean`` / ``t_gt2_ratio`` / ``beta_mean`` / ``n_dates``。
    """
    cand = ops.cs_zscore(candidate) if standardize else candidate
    names = list(exposures)
    exp = [exposures[k].reindex_like(fwd) for k in names]
    valid = fwd.notna() & cand.notna()
    for x in exp:
        valid &= x.notna()
    y = fwd.where(valid)
    cand_v = cand.where(valid)
    exp_v = [x.where(valid) for x in exp]

    _, _, r2_base, _ = cross_sectional_ols(y, exp_v, min_stocks=min_stocks)
    params, _, r2_full, tstats = cross_sectional_ols(
        y, exp_v + [cand_v], min_stocks=min_stocks)
    cand_col = params.columns[-1]
    n_obs = valid.sum(axis=1).astype(float).replace(0.0, np.nan)

    db = r2_base.dropna()
    df_ = r2_full.dropna()
    both = db.index.intersection(df_.index)
    if len(both) == 0:
        return {"delta_r2": 0.0, "delta_r2_adj": 0.0, "r2_base": 0.0,
                "r2_full": 0.0, "t_mean": 0.0, "t_gt2_ratio": 0.0,
                "beta_mean": 0.0, "n_dates": 0}
    k_base = len(exp) + 1
    k_full = k_base + 1
    adj_base = _adj_r2(db.loc[both], n_obs.loc[both], k_base)
    adj_full = _adj_r2(df_.loc[both], n_obs.loc[both], k_full)
    t_series = tstats.loc[both, cand_col].abs().dropna()
    return {
        "delta_r2": float((df_.loc[both] - db.loc[both]).mean()),
        "delta_r2_adj": float((adj_full - adj_base).mean()),
        "r2_base": float(db.loc[both].mean()),
        "r2_full": float(df_.loc[both].mean()),
        "t_mean": float(t_series.mean()) if len(t_series) else 0.0,
        "t_gt2_ratio": float((t_series > 2.0).mean()) if len(t_series) else 0.0,
        "beta_mean": float(params.loc[both, cand_col].mean()),
        "n_dates": int(len(both)),
    }


def vif(candidate: pd.DataFrame, others: Dict[str, pd.DataFrame],
        ridge: float = 1e-4, min_stocks: int = 20) -> float:
    """方差膨胀因子（岭回归稳健版）：对风险暴露的共线性强度。

    ``VIF = 1/(1-R²)``，R² 取"候选因子对全部暴露回归"的日均 R²。
    用小岭参数避免暴露之间完全共线导致的数值爆表。
    """
    if not others:
        return 1.0
    cand = ops.cs_zscore(candidate)
    xs = [ops.cs_zscore(v.reindex_like(cand)) for v in others.values()]
    y_np = cand.to_numpy(dtype=np.float64)
    x_np = [x.to_numpy(dtype=np.float64) for x in xs]
    r2_list: List[float] = []
    for i in range(len(cand.index)):
        yy = y_np[i]
        cols = [x[i] for x in x_np]
        X, yv = _design(yy, cols)
        n, k = X.shape
        if n < max(min_stocks, k + 2):
            continue
        gram = X.T @ X + ridge * np.eye(k) * max(1.0, float(np.trace(X.T @ X)) / k)
        try:
            beta = np.linalg.solve(gram, X.T @ yv)
        except np.linalg.LinAlgError:  # pragma: no cover
            continue
        pred = X @ beta
        ss_res = float(((yv - pred) ** 2).sum())
        ss_tot = float(((yv - yv.mean()) ** 2).sum())
        if ss_tot > 0:
            r2_list.append(1.0 - ss_res / ss_tot)
    if not r2_list:
        return float("nan")
    r2_mean = float(min(max(np.mean(r2_list), 0.0), 1.0 - 1e-9))
    return float(1.0 / (1.0 - r2_mean))


def autocorr(factor: pd.DataFrame, lags: Sequence[int] = (1, 2, 3, 4, 5),
             use_rank: bool = True) -> Dict[str, float]:
    """因子横截面自相关（时序稳定性；天风闸门 4）。"""
    x = ops.cs_rank(factor) if use_rank else factor
    out: Dict[str, float] = {}
    for k in lags:
        # 向量化的逐日相关：一次算完整列，退化行由 ops 层统一给 NaN
        s = ops.cs_corr(x, x.shift(k), rank=False, min_stocks=10)
        vals = s.dropna()
        out[f"ac_l{k}"] = float(vals.mean()) if len(vals) else float("nan")
    l1 = out.get("ac_l1", float("nan"))
    l5 = out.get(f"ac_l{lags[-1]}", float("nan"))
    out["ac_decay"] = (float(abs(l1 - l5)) if np.isfinite(l1) and np.isfinite(l5)
                       else float("nan"))
    return out


def crowding(factor: pd.DataFrame,
             pool: Optional[Dict[str, pd.DataFrame]] = None,
             exposures: Optional[Dict[str, pd.DataFrame]] = None,
             turnover: Optional[float] = None) -> Dict[str, float]:
    """拥挤度（天风闸门 6，本项目的可计算定义）。

    - ``pool_max_abs_corr`` / ``pool_mean_abs_corr``：与既有因子池的横截面
      相关（Spearman，日频取均值）。越高说明与存量因子重复、越拥挤。
    - ``exposure_hhi``：候选因子对风险暴露解释力的**集中度**（HHI）。
      集中度高说明它只是某个已知风险的代理。
    - ``crowding_score``：``0.5×pool_mean_abs_corr + 0.5×exposure_hhi``，
      范围 [0,1]，越大越拥挤（本项目自定义口径，便于横向排序）。
    """
    fr = ops.cs_rank(factor)
    out: Dict[str, float] = {"pool_max_abs_corr": 0.0, "pool_mean_abs_corr": 0.0,
                             "exposure_hhi": 0.0, "crowding_score": 0.0}
    if pool:
        per_pool: List[float] = []
        for name, f in pool.items():
            if name == "factor" or f is None:
                continue
            other = ops.cs_rank(f.reindex_like(factor))
            s = ops.cs_corr(fr, other, rank=False, min_stocks=10).abs().dropna()
            if len(s):
                per_pool.append(float(s.mean()))
        if per_pool:
            out["pool_max_abs_corr"] = float(np.max(per_pool))
            out["pool_mean_abs_corr"] = float(np.mean(per_pool))
    if exposures:
        strengths: List[float] = []
        for name, ex in exposures.items():
            other = ops.cs_rank(ex.reindex_like(factor))
            s = ops.cs_corr(fr, other, rank=False, min_stocks=10).abs().dropna()
            strengths.append(float(s.mean()) if len(s) else 0.0)
        tot = float(sum(strengths))
        if tot > 0:
            p = np.array(strengths) / tot
            out["exposure_hhi"] = float((p ** 2).sum())
    out["crowding_score"] = float(0.5 * out["pool_mean_abs_corr"]
                                  + 0.5 * out["exposure_hhi"])
    if turnover is not None:
        out["turnover"] = float(turnover)
    return out


# --------------------------------------------------------------------------
# 因子收益分解 / 组合风险贡献
# --------------------------------------------------------------------------
def factor_returns(fwd: pd.DataFrame, factors: Dict[str, pd.DataFrame],
                   min_stocks: int = 20) -> pd.DataFrame:
    """逐日横截面回归得到的因子收益序列（多因子模型的 f_t）。"""
    names = list(factors)
    xs = [ops.cs_zscore(factors[k]).reindex_like(fwd) for k in names]
    params, _, _, tstats = cross_sectional_ols(fwd, xs, add_const=True,
                                               min_stocks=min_stocks)
    out = params[[c for c in params.columns if c != "const"]].copy()
    out.columns = names
    out.attrs["tstats"] = tstats
    return out


def factor_cov(rets: pd.DataFrame, nw_lag: int = 1) -> pd.DataFrame:
    """Newey-West 调整的因子收益协方差（重叠持仓下自相关不可忽略）。"""
    x = rets.dropna().to_numpy(dtype=np.float64)
    if x.shape[0] < 3:
        return pd.DataFrame(np.eye(x.shape[1]) * np.nan, index=rets.columns,
                            columns=rets.columns)
    x = x - x.mean(axis=0, keepdims=True)
    t = x.shape[0]
    gamma0 = (x.T @ x) / t
    cov = gamma0.copy()
    for lag in range(1, int(nw_lag) + 1):
        w = 1.0 - lag / (nw_lag + 1.0)
        g = (x[lag:].T @ x[:-lag]) / t
        cov += w * (g + g.T)
    return pd.DataFrame(cov, index=rets.columns, columns=rets.columns)


def portfolio_risk_contribution(weights: Dict[str, float],
                                factor_rets: pd.DataFrame,
                                cov: Optional[pd.DataFrame] = None
                                ) -> pd.DataFrame:
    """组合层面风险贡献：各因子的边际/成分贡献，成分波动贡献之和 = 组合波动。"""
    names = [c for c in factor_rets.columns if c in weights]
    if not names:
        raise ValueError("权重与因子收益列名无交集")
    w = np.array([float(weights[n]) for n in names])
    sigma = (cov.loc[names, names].to_numpy(dtype=np.float64)
             if cov is not None else factor_cov(factor_rets[names]).to_numpy())
    port_var = float(w @ sigma @ w)
    port_vol = math.sqrt(max(port_var, 0.0))
    marginal = (sigma @ w) / port_vol if port_vol > 0 else np.zeros_like(w)
    comp_vol = w * marginal
    total_var = port_var if port_var > 0 else np.nan
    df = pd.DataFrame({
        "weight": w,
        "marginal_contribution": marginal,
        "component_vol": comp_vol,
        "component_var": comp_vol * port_vol,
        "pct_of_total_vol": comp_vol / port_vol if port_vol > 0 else np.nan,
    }, index=names)
    df.attrs["portfolio_vol"] = port_vol
    df.attrs["portfolio_var"] = total_var
    return df


def risk_report(candidate: pd.DataFrame, fwd: pd.DataFrame,
                exposures: Dict[str, pd.DataFrame],
                pool: Optional[Dict[str, pd.DataFrame]] = None,
                turnover: Optional[float] = None,
                min_stocks: int = 20) -> Dict[str, object]:
    """一站式风险体检（供 evaluator 的 risk profile 调用）。"""
    d = delta_r2(candidate, fwd, exposures, min_stocks=min_stocks)
    ac = autocorr(candidate)
    cr = crowding(candidate, pool=pool, exposures=exposures, turnover=turnover)
    out: Dict[str, object] = dict(d)
    out.update(ac)
    out.update(cr)
    out["vif"] = vif(candidate, exposures, min_stocks=min_stocks)
    return out
