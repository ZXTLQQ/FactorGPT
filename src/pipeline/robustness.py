"""过拟合检验（P0）：walk-forward 滚动窗口、Deflated Sharpe Ratio、参数稳定性。

学生量化项目最易被评审质疑「历史拟合 / 过拟合」。本模块提供三条独立防线，
用于在交付前量化因子是否真正具备样本外预测力，而不是在训练集上偶然拟合：

1. walk_forward：把样本切成若干滚动窗口，只在窗口内计算 Rank IC / ICIR，
   检验因子在不同时间段是否稳定有效（避免「只在某一段有效」）。
2. deflated_sharpe_ratio（DSR）：Bailey & López de Prado (2014) 提出的
   「去膨胀夏普比率」，对多重检验与非正态性进行修正，给出
   「该夏普比率在统计上不为零」的概率（而非 naive t 检验）。
3. parameter_stability：扰动持有期 / 分组数等参数，观察 ICIR 是否剧烈漂移，
   验证结论对参数选择不敏感。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.backtest import FactorBacktester


class RobustnessValidator:
    """因子稳健性 / 过拟合检验器。"""

    def __init__(self, forward_periods: int = 1, n_quantiles: int = 5) -> None:
        self.bt = FactorBacktester(forward_periods=forward_periods, n_quantiles=n_quantiles)

    # ------------------------------------------------------------------
    # 1) walk-forward 滚动窗口
    # ------------------------------------------------------------------
    def walk_forward(
        self,
        kline: pd.DataFrame,
        factor: pd.Series,
        n_splits: int = 5,
    ) -> list:
        """滚动窗口样本外检验：将交易日等分 n_splits 段，逐段计算 IC / ICIR。

        Returns:
            每个窗口的 {window, ic, icir, ic_pos_ratio, n_dates} 列表。
        """
        dates = sorted(kline["date"].unique())
        n = len(dates)
        if n < n_splits * 2:
            return []
        results = []
        for k in range(n_splits):
            start = int(n * k / n_splits)
            end = int(n * (k + 1) / n_splits)
            test_dates = set(dates[start:end])
            mask = factor.index.get_level_values("date").isin(test_dates)
            f_sub = factor[mask]
            kl_sub = kline[kline["date"].isin(test_dates)]
            if f_sub.empty:
                continue
            m = self.bt.evaluate(kl_sub, f_sub)
            if "icir" in m and not np.isnan(m.get("icir", np.nan)):
                results.append({
                    "window": k + 1,
                    "ic": m.get("ic"),
                    "icir": m.get("icir"),
                    "ic_pos_ratio": m.get("ic_positive_ratio"),
                    "n_dates": m.get("n_dates"),
                })
        return results

    # ------------------------------------------------------------------
    # 2) Deflated Sharpe Ratio
    # ------------------------------------------------------------------
    @staticmethod
    def deflated_sharpe_ratio(
        sharpe: float,
        n_obs: int,
        n_trials: int = 1,
        skew: float = 0.0,
        kurtosis: float = 3.0,
        alpha: float = 0.05,
    ) -> float:
        """去膨胀夏普比率（DSR）。

        返回 P(SR_true > SR_critical)，即「该夏普比率在统计上显著不为零」的概率。
        - n_trials：隐含搜寻的因子/参数组合数（用于多重检验修正），
          数值越大，临界夏普越高，越难通过 —— 这正是防止「试出显著」的关键。
        - skew / kurtosis：样本收益（或逐日 IC）的偏度 / 峰度，修正非正态性。

        References:
            Bailey, D. H., & López de Prado, M. (2014).
            "The Deflated Sharpe Ratio: Correcting for Selection Bias,
            Backtest Overfitting and Non-Normality." Journal of Portfolio Management.
        """
        try:
            from scipy.stats import norm
        except ImportError:  # pragma: no cover
            return float("nan")
        if n_obs <= 1 or np.isnan(sharpe):
            return float("nan")
        z = norm.ppf(1 - alpha)
        if n_trials <= 1:
            e_max = z  # 单 trial 时退化为普通单边临界值
        else:
            # n 个标准正态独立样本最大值的期望近似
            e_max = norm.ppf(1.0 - 1.0 / n_trials)
        # 非正态性调整（小 SR 近似，Bailey & de Prado）
        adj = (1.0 + (skew / 6.0) * sharpe
               - (kurtosis / 24.0) * sharpe ** 2
               + (skew ** 2 / 36.0) * sharpe ** 3)
        sr_critical = e_max * adj / np.sqrt(n_obs)
        se_sr = 1.0 / np.sqrt(n_obs)
        arg = (sharpe - sr_critical) / se_sr
        return float(norm.cdf(arg))

    # ------------------------------------------------------------------
    # 3) 参数稳定性
    # ------------------------------------------------------------------
    def parameter_stability(
        self,
        kline: pd.DataFrame,
        factor: pd.Series,
        forward_grid: tuple = (1, 2, 3, 5, 10),
        quantile_grid: tuple = (3, 5, 10),
    ) -> dict:
        """扰动持有期 / 分组数，检验 ICIR 是否稳定。

        若在不同持有期、不同分组数下 ICIR 始终为正且波动小，说明因子结论
        对参数选择不敏感，过拟合风险低。
        """
        records = []
        for fp in forward_grid:
            bt = FactorBacktester(forward_periods=fp, n_quantiles=5)
            m = bt.evaluate(kline, factor)
            records.append({
                "param": "forward_periods", "value": fp,
                "ic": m.get("ic"), "icir": m.get("icir"),
            })
        for nq in quantile_grid:
            bt = FactorBacktester(forward_periods=1, n_quantiles=nq)
            m = bt.evaluate(kline, factor)
            records.append({
                "param": "n_quantiles", "value": nq,
                "ic": m.get("ic"), "icir": m.get("icir"),
            })
        icirs = [r["icir"] for r in records if r["icir"] is not None and not np.isnan(r["icir"])]
        summary = {
            "records": records,
            "icir_std": float(np.std(icirs)) if icirs else float("nan"),
            "icir_min": float(np.min(icirs)) if icirs else float("nan"),
            "icir_mean": float(np.mean(icirs)) if icirs else float("nan"),
            "stable": bool(np.min(icirs) > 0) if icirs else False,
        }
        return summary

    # ------------------------------------------------------------------
    # 综合校验
    # ------------------------------------------------------------------
    def validate(
        self,
        kline: pd.DataFrame,
        factor: pd.Series,
        n_trials: int = 10,
        label: str = "factor",
    ) -> dict:
        """执行完整过拟合检验并给出 PASS / REVIEW 结论。

        Args:
            kline: 行情长表。
            factor: 因子 Series（索引 date,symbol）。
            n_trials: 估计搜寻的因子/参数组合数（多重检验修正强度）。
            label: 因子标签（用于报告）。

        Returns:
            含 walk_forward / deflated_sharpe_ratio / parameter_stability / verdict 的字典。
        """
        wf = self.walk_forward(kline, factor)
        wf_icirs = [w["icir"] for w in wf if not np.isnan(w["icir"])]
        m = self.bt.evaluate(kline, factor)
        sharpe = m.get("long_short_sharpe")
        n_obs = m.get("n_dates", 0)
        ic_s = m.get("_ic_series")
        skew = float(ic_s.skew()) if ic_s is not None and len(ic_s) else 0.0
        kurt = float(ic_s.kurt()) if ic_s is not None and len(ic_s) else 3.0
        dsr = self.deflated_sharpe_ratio(sharpe, n_obs, n_trials, skew, kurt)
        ps = self.parameter_stability(kline, factor)

        wf_ok = bool(np.mean(wf_icirs) > 0) if wf_icirs else False
        dsr_ok = bool(dsr > 0.95) if not np.isnan(dsr) else False
        verdict = "PASS" if (wf_ok and dsr_ok and ps["stable"]) else "REVIEW"

        return {
            "label": label,
            "walk_forward": wf,
            "walk_forward_icir_mean": float(np.mean(wf_icirs)) if wf_icirs else float("nan"),
            "deflated_sharpe_ratio": dsr,
            "dsr_skew": skew,
            "dsr_kurtosis": kurt,
            "parameter_stability": ps,
            "verdict": verdict,
        }
