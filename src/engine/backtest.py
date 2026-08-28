"""
因子回测与评价引擎（src/engine/backtest.py）

基于因子值与前向收益，计算量化选股中常用的因子评价指标：
- IC（Pearson 相关系数）与 RankIC（Spearman 秩相关）
- ICIR（信息系数稳定度 = mean(IC)/std(IC)）
- IC 为正的比例
- 分位数分组收益（多头/空头/多空对冲）
- 多空组合累计收益、夏普、最大回撤、换手率
- 因子覆盖率（非空比例）

同时提供基于 matplotlib / seaborn 的可视化函数，输出 IC 时间序列、
分位数收益柱状图与多空权益曲线。若安装 alphalens-reloaded 也可调用其接口。

注意：所有收益计算均使用 shift(-forward_periods) 构造「未来收益」，
严格避免前视偏差（look-ahead bias）。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# 抑制 groupby.apply 等产生的 pandas FutureWarning 噪声（不影响结果）
warnings.filterwarnings("ignore", category=FutureWarning)

try:  # 可选依赖
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 尝试启用中文字体（Windows 常见为 SimHei / Microsoft YaHei）
    for _cjk in ("SimHei", "Microsoft YaHei", "Arial Unicode MS", "WenQuanYi Micro Hei"):
        try:
            plt.rcParams["font.sans-serif"] = [_cjk]
            plt.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue

    _HAS_PLOT = True
except ImportError:  # pragma: no cover
    _HAS_PLOT = False


def portfolio_turnover(factor_series: pd.Series, kline: pd.DataFrame, top_frac: float = 0.1) -> Optional[float]:
    """多头组合实际换手率，口径与 FactorBacktester.realistic_portfolio 完全一致。

    每日截面取因子值最高的 top_frac 等权做多，按「日内权重变动绝对值之和」求日度换手，
    再跨日平均。替代 evaluate() 中基于排名差的近似换手，使两个口径可比。
    """
    if factor_series is None or kline is None or kline.empty:
        return None
    try:
        f = factor_series.rename("factor").reset_index()
        merged = kline[["date", "symbol", "pct_chg"]].merge(f, on=["date", "symbol"], how="inner")
        if merged.empty:
            return None
        merged = merged.sort_values(["date", "factor"], ascending=[True, False])
        merged["rank_in_date"] = merged.groupby("date").cumcount()
        n_per = merged.groupby("date")["symbol"].transform("count")
        k = (n_per * top_frac).clip(lower=1).astype(int)
        merged["weight"] = np.where(merged["rank_in_date"] < k, 1.0 / k, 0.0)
        merged = merged.sort_values(["symbol", "date"])
        merged["w_prev"] = merged.groupby("symbol")["weight"].shift(1).fillna(0.0)
        day_turn = merged.groupby("date").apply(lambda g: float(np.abs(g["weight"] - g["w_prev"]).sum()))
        return float(day_turn.mean()) if len(day_turn) else None
    except Exception:
        return None


def _ic_series(panel: pd.DataFrame, method: str = "pearson") -> pd.Series:
    """向量化计算逐日 IC 序列（比 groupby.apply 更快，适合大规模数据）。

    返回以 date 为索引的 Series。pearson 用 numpy 批量计算；spearman 退化为
    逐日秩相关。截面样本不足或因子/收益为常数时返回 nan。
    """
    dates = panel["date"].to_numpy()
    f = panel["factor"].to_numpy(dtype=float)
    y = panel["fwd_ret"].to_numpy(dtype=float)
    uniq = np.unique(dates)
    out = np.empty(len(uniq), dtype=float)
    out[:] = np.nan
    for i, d in enumerate(uniq):
        m = dates == d
        xv, yv = f[m].copy(), y[m].copy()
        if len(xv) < 2 or np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
            continue
        if method == "spearman":
            xv = pd.Series(xv).rank().to_numpy()
            yv = pd.Series(yv).rank().to_numpy()
        c = np.corrcoef(xv, yv)[0, 1]
        out[i] = c if np.isfinite(c) else np.nan
    return pd.Series(out, index=pd.Index(uniq, name="date")).dropna()


def batch_evaluate(
    kline: pd.DataFrame,
    factors: Dict[str, pd.Series],
    n_quantiles: int = 5,
    forward_periods: int = 1,
    n_jobs: int = -1,
    verbose: bool = True,
) -> Dict[str, dict]:
    """并行回测多个因子（factors: name -> 因子 Series）。

    用 joblib 并行（可选依赖），未安装时退化为串行。返回 name -> 指标字典。
    """
    names = list(factors.keys())
    bt = FactorBacktester(n_quantiles=n_quantiles, forward_periods=forward_periods)

    def _one(name):
        try:
            return name, bt.evaluate(kline, factors[name], verbose=False)
        except Exception as e:  # 单因子失败不影响其余
            return name, {"error": str(e)}

    try:
        from joblib import Parallel, delayed
        use_joblib = True
    except Exception:
        use_joblib = False
    if use_joblib:
        res = Parallel(n_jobs=n_jobs)(delayed(_one)(n) for n in names)
    else:
        res = [_one(n) for n in names]
    out = {n: m for n, m in res}
    if verbose:
        print(f"[batch_evaluate] 完成 {len(out)} 个因子回测（{'joblib' if use_joblib else 'serial'} 后端）")
    return out


class FactorBacktester:
    """因子回测器：输入行情长表与因子长表，输出评价指标与图表。"""

    def __init__(
        self,
        n_quantiles: int = 5,
        forward_periods: int = 1,
        commission: float = 0.001,
        risk_free_rate: float = 0.03,
    ) -> None:
        self.n_quantiles = n_quantiles
        self.forward_periods = forward_periods
        self.commission = commission
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # 数据准备
    # ------------------------------------------------------------------
    def _prepare(self, kline: pd.DataFrame, factor: pd.Series):
        """合并行情与因子，构造前向收益，返回面板 DataFrame。"""
        df = kline[["date", "symbol", "close"]].copy()
        # 统一 date 为字符串，规避 datetime64 与 object 混用导致的 merge 失败
        df["date"] = df["date"].astype(str)
        df = df.sort_values(["symbol", "date"])

        # 个股收益率
        df["ret"] = df.groupby("symbol")["close"].pct_change()
        # 未来收益（关键：避免前视）
        df["fwd_ret"] = df.groupby("symbol")["ret"].shift(-self.forward_periods)

        # 因子对齐（按 date,symbol 索引）：同步将因子索引 date 层转为字符串
        fac = factor.rename("factor")
        # 防御：因子索引非唯一会导致 join 异常，去重保留最后一条
        if fac.index.duplicated().any():
            fac = fac[~fac.index.duplicated(keep="last")]
        if isinstance(fac.index, pd.MultiIndex):
            fac.index = fac.index.set_levels(fac.index.levels[0].astype(str), level=0)
        else:
            fac.index = pd.Index(fac.index.astype(str))
        panel = df.join(fac, on=["date", "symbol"])
        panel = panel.dropna(subset=["factor", "fwd_ret"])
        return panel

    # ------------------------------------------------------------------
    # 评价指标
    # ------------------------------------------------------------------
    def evaluate(self, kline: pd.DataFrame, factor: pd.Series, verbose: bool = True) -> dict:
        """执行回测并返回指标字典。

        Args:
            kline: 行情长表（date, symbol, close ...）。
            factor: 因子 Series，索引 (date, symbol)。

        Returns:
            指标字典，键包含 ic / rank_ic / icir / ic_positive_ratio /
            quantile_returns / long_short_return / long_short_sharpe /
            long_short_cum_return / max_drawdown / turnover / coverage /
            n_stocks / n_dates。
        """
        panel = self._prepare(kline, factor)
        if panel.empty:
            return {"error": "因子与行情无有效交集，无法回测"}

        # 逐日截面相关（对子 DataFrame 计算 factor 与 fwd_ret 的相关）
        # 防御：截面样本不足（<2）或因子/收益为常量时相关系数无定义，直接返回
        # nan 并抑制 numpy/scipy 的 RuntimeWarning 与 ConstantInputWarning，
        # 避免刷屏，并保证无有效数据时回测仍可产出指标字典。
        ic_series = _ic_series(panel, "pearson")
        rankic_series = _ic_series(panel, "spearman")
        ic_series = ic_series.dropna()
        rankic_series = rankic_series.dropna()

        ic = float(ic_series.mean()) if len(ic_series) else float("nan")
        rank_ic = float(rankic_series.mean()) if len(rankic_series) else float("nan")
        ic_std = float(ic_series.std()) if len(ic_series) > 1 else float("nan")
        icir = float(ic / ic_std) if ic_std and not np.isnan(ic_std) and ic_std > 0 else float("nan")
        ic_pos_ratio = float((ic_series > 0).mean()) if len(ic_series) else float("nan")

        # 分位数分组收益（用 transform 保持与 panel 同索引）
        panel = panel.copy()
        panel["group"] = panel.groupby("date")["factor"].transform(
            lambda x: pd.qcut(x, self.n_quantiles, labels=False, duplicates="drop")
        )
        grp_ret = panel.groupby(["date", "group"])["fwd_ret"].mean().unstack()
        quantile_returns = {
            int(g): float(grp_ret[g].mean()) for g in grp_ret.columns
        }

        # 多空对冲（最高组 - 最低组）
        if self.n_quantiles in grp_ret.columns and 0 in grp_ret.columns:
            ls = grp_ret[self.n_quantiles - 1] - grp_ret[0]
        else:
            ls = grp_ret.max(axis=1) - grp_ret.min(axis=1)
        ls = ls.dropna()
        long_short_return = float(ls.mean()) if len(ls) else float("nan")
        ls_sharpe = (
            float(ls.mean() / ls.std() * np.sqrt(252))
            if len(ls) > 1 and ls.std() > 0
            else float("nan")
        )
        cum_ls = float((1 + ls).cumprod().iloc[-1] - 1) if len(ls) else float("nan")
        # 最大回撤
        if len(ls):
            equity = (1 + ls).cumprod()
            max_dd = float((equity / equity.cummax() - 1).min())
        else:
            max_dd = float("nan")

        # 换手率：与 realistic_portfolio 口径一致的「多头组合实际换手」
        turnover = portfolio_turnover(factor, kline, top_frac=0.1)
        if turnover is None:
            # 极端情况下（数据不足）回退到因子排名变化近似
            panel_sorted = panel.sort_values(["symbol", "date"]).copy()
            panel_sorted["_rank"] = panel_sorted.groupby("symbol")["factor"].rank(pct=True)
            panel_sorted["_prev_rank"] = panel_sorted.groupby("symbol")["_rank"].shift(1)
            turnover = float((panel_sorted["_rank"] - panel_sorted["_prev_rank"]).abs().mean())

        coverage = float(factor.notna().mean())

        # —— 分层（分组）回测明细：各分位组合的累积收益曲线、年化收益与 Sharpe ——
        quantile_cum: Dict[int, pd.Series] = {}
        quantile_stats: Dict[int, Dict[str, float]] = {}
        for g in grp_ret.columns:
            gret = grp_ret[g].dropna()
            if gret.empty:
                continue
            cum = (1.0 + gret).cumprod() - 1.0
            ann = gret.mean() * 252.0
            vol = gret.std() * np.sqrt(252.0)
            sharpe_g = float(ann / vol) if vol and vol > 0 else 0.0
            quantile_cum[int(g)] = cum
            quantile_stats[int(g)] = {
                "mean_ret": float(gret.mean()),
                "ann_ret": float(ann),
                "cum_ret": float(cum.iloc[-1]) if len(cum) else 0.0,
                "sharpe": sharpe_g,
                "n": int(len(gret)),
            }

        return {
            "ic": ic,
            "rank_ic": rank_ic,
            "icir": icir,
            "ic_positive_ratio": ic_pos_ratio,
            "quantile_returns": quantile_returns,
            "long_short_return": long_short_return,
            "long_short_sharpe": ls_sharpe,
            "long_short_cum_return": cum_ls,
            "max_drawdown": max_dd,
            "annualized_volatility": float(np.nanstd(ls) * np.sqrt(252)) if len(ls) else float("nan"),
            "turnover": turnover,
            "coverage": coverage,
            "n_stocks": int(panel["symbol"].nunique()),
            "n_dates": int(panel["date"].nunique()),
            "quantile_cum": quantile_cum,
            "quantile_stats": quantile_stats,
            "_ic_series": ic_series,  # 供绘图使用，不进入报告表格
            "_ls_series": ls,
        }

    # ------------------------------------------------------------------
    # A 股现实约束下的组合级回测（P0：因子能否赚钱 + 风控）
    # ------------------------------------------------------------------
    def ic_by_year(self, kline: pd.DataFrame, factor: pd.Series) -> Dict[str, Dict[str, float]]:
        """分年度 IC：逐年分解因子稳定性，识别「某些年份失效」的风险。

        Returns:
            年 -> {ic, rank_ic, icir, ic_positive_ratio, n_dates}。
        """
        panel = self._prepare(kline, factor)
        if panel.empty:
            return {}
        panel = panel.copy()
        if panel["date"].dtype == object:
            panel["year"] = panel["date"].str[:4]
        else:
            panel["year"] = pd.to_datetime(panel["date"]).dt.year.astype(str)
        out: Dict[str, Dict[str, float]] = {}
        for y, g in panel.groupby("year"):
            ic_s = g.groupby("date").apply(lambda x: x["factor"].corr(x["fwd_ret"])).dropna()
            ric_s = g.groupby("date").apply(
                lambda x: x["factor"].corr(x["fwd_ret"], method="spearman")).dropna()
            if ic_s.empty:
                continue
            ic_std = ic_s.std() if len(ic_s) > 1 else np.nan
            out[str(y)] = {
                "ic": float(ic_s.mean()),
                "rank_ic": float(ric_s.mean()) if len(ric_s) else float("nan"),
                "icir": float(ic_s.mean() / ic_std) if ic_std and ic_std > 0 else float("nan"),
                "ic_positive_ratio": float((ic_s > 0).mean()),
                "n_dates": int(len(ic_s)),
            }
        return out

    def realistic_portfolio(
        self,
        kline: pd.DataFrame,
        factor: pd.Series,
        top_frac: float = 0.1,
        commission: float = 0.0003,
        stamp_tax: float = 0.0005,
        cost_mode: str = "two_side",
        min_daily_amount: float = 0.0,
        limit_up_pct: float = 0.095,
        t_plus_one: bool = True,
        allow_short: bool = False,
        benchmark_ret: Optional[pd.Series] = None,
        start_equity: float = 1.0,
    ) -> Dict[str, Any]:
        """A 股现实约束下的组合级回测：给出「因子能否赚钱」的直接证据。

        与学术多空对冲不同，本方法引入 A 股真实落地约束：
        - t+1：当日信号于次交易日收盘成交，天然规避当日回转（T+1）限制；
        - 涨跌停：涨停日无法买入、跌停日无法卖出（按 fwd_ret 判定）；
        - 停牌：成交量/成交额为零视为停牌，跳过该标的；
        - 流动性门槛：日成交额低于 min_daily_amount 的标的剔除；
        - 做空限制：默认 allow_short=False，构建多头组合（long-only），
          更贴近 A 股无法自由做空的现实（学术多空对冲仅作参考）。

        Args:
            kline: 行情长表（date, symbol, close, volume, amount ...）。
            factor: 因子 Series，索引 (date, symbol)。
            top_frac: 选入多头组合的分位比例（默认前 10%）。
            commission: 单边佣金率（默认万三）。
            stamp_tax: 印花税（卖出征收，默认万五）。
            cost_mode: 'one_side' 仅计买入佣金，'two_side' 双边计佣金。
            min_daily_amount: 流动性门槛（元/日），0 表示不限制。
            limit_up_pct: 涨跌停阈值（主板约 10%，取保守 9.5%）。
            t_plus_one: 是否次日成交（默认 True，符合 T+1）。
            allow_short: 是否允许做空（默认 False，贴近 A 股现实）。
            benchmark_ret: 基准日收益序列（索引为日期），用于计算超额/信息比率。
            start_equity: 初始净值。

        Returns:
            含 equity（净值序列）、rebalance_list（调仓清单）、
            metrics（年化收益/波动/夏普/回撤/信息比率等）与 assumptions 的字典。
        """
        # 兼容合成行情（可能无 volume/amount）：缺失时填充正值，避免误判停牌/流动性
        df = kline[["date", "symbol", "close"]].copy()
        if "volume" in kline.columns and kline["volume"].notna().any():
            df["volume"] = kline["volume"].fillna(1.0)
        else:
            df["volume"] = 1.0
        if "amount" in kline.columns and kline["amount"].notna().any():
            df["amount"] = kline["amount"].fillna(df["close"] * df["volume"])
        else:
            df["amount"] = df["close"] * df["volume"]
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        df["ret"] = df.groupby("symbol")["close"].pct_change()
        df["fwd_ret"] = df.groupby("symbol")["ret"].shift(-self.forward_periods)
        # 次交易日成交量/成交额，用于停牌与流动性判定
        df["vol_next"] = df.groupby("symbol")["volume"].shift(-self.forward_periods)
        df["amt_next"] = df.groupby("symbol")["amount"].shift(-self.forward_periods)
        fac = factor.rename("factor")
        panel = df.join(fac, on=["date", "symbol"]).dropna(subset=["factor", "fwd_ret"])

        dates = sorted(panel["date"].unique())
        if len(dates) < 3:
            return {"error": "有效交易日不足，无法回测"}

        equity = start_equity
        equity_curve: Dict[str, float] = {}
        net_rets: List[float] = []
        rebalance_list: List[Dict[str, Any]] = []
        prev_w: Dict[str, float] = {}

        for i, t in enumerate(dates[:-1]):
            trade_date = dates[i + 1]
            sub = panel[panel["date"] == t].set_index("symbol")
            held = [s for s in prev_w if s in sub.index]
            gross = float(sum(
                prev_w[s] * sub.loc[s, "fwd_ret"]
                for s in held
                if not np.isnan(sub.loc[s, "fwd_ret"])
            ))

            tradable = sub[(sub["vol_next"] > 0) & (sub["amt_next"] >= min_daily_amount)]

            # 1) 初始化新权重：不可交易（停牌/缺流动性）的持仓无法买卖，必须冻结沿用上期权重。
            #    否则停牌股会被静默清零，导致权益断裂（修复 1.2）。
            new_w: Dict[str, float] = {}
            for s in sub.index:
                if s in prev_w and prev_w[s] > 0.0 and s not in tradable.index:
                    new_w[s] = prev_w[s]

            # 2) 可交易标的的目标权重（等权）
            if not tradable.empty:
                ranked = tradable["factor"].sort_values(ascending=False)
                n_long = max(1, int(np.ceil(top_frac * len(ranked))))
                longs = set(ranked.head(n_long).index)
                for s in longs:
                    chg = sub.loc[s, "fwd_ret"]  # 次交易日涨跌幅，用于涨跌停判定
                    # 涨停无法买入（A 股现实约束，无论做空与否均适用，修复 1.3）
                    if chg >= limit_up_pct - 1e-6:
                        continue
                    new_w[s] = 1.0
                # 跌停无法卖出：持有中的可交易标的若跌停且未被选中，强制保留原权重
                for s in tradable.index:
                    if s in prev_w and prev_w[s] > 0.0 and s not in longs:
                        chg = sub.loc[s, "fwd_ret"]
                        if chg <= -limit_up_pct + 1e-6:
                            new_w[s] = prev_w[s]

            # 3) 归一化；全空则回退到上期权重
            tot = sum(new_w.values())
            if tot > 0:
                new_w = {s: w / tot for s, w in new_w.items()}
            else:
                new_w = {s: prev_w.get(s, 0.0) for s in sub.index if prev_w.get(s, 0.0) > 0.0}
                tot = sum(new_w.values())
                if tot > 0:
                    new_w = {s: w / tot for s, w in new_w.items()}
                else:
                    new_w = dict(prev_w)

            all_sym = set(prev_w) | set(new_w)
            turnover = 0.5 * sum(abs(new_w.get(s, 0) - prev_w.get(s, 0)) for s in all_sym)
            sells = sum(max(0.0, prev_w.get(s, 0) - new_w.get(s, 0)) for s in all_sym)
            cost = turnover * commission * (2 if cost_mode == "two_side" else 1) + stamp_tax * sells
            net = gross - cost
            equity *= (1.0 + net)
            equity_curve[trade_date] = float(equity)
            net_rets.append(float(net))
            rebalance_list.append({
                "date": trade_date,
                "weights": {s: round(new_w[s], 6) for s in new_w if new_w[s] > 0},
            })
            prev_w = new_w

        nav = pd.Series(equity_curve)
        rets = pd.Series(net_rets, index=list(equity_curve.keys()))
        ann_ret = float(rets.mean() * 252.0)
        ann_vol = float(rets.std() * np.sqrt(252.0)) if len(rets) > 1 else float("nan")
        max_dd = float(((nav / nav.cummax()) - 1).min()) if len(nav) else float("nan")
        sharpe = float((ann_ret - self.risk_free_rate) / ann_vol) if ann_vol and ann_vol > 0 else float("nan")
        cum_ret = float(nav.iloc[-1] / nav.iloc[0] - 1.0) if len(nav) else float("nan")

        metrics: Dict[str, Any] = {
            "ann_return": ann_ret,
            "ann_volatility": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "cum_return": cum_ret,
            "n_rebalances": len(rebalance_list),
            "final_equity": float(equity),
        }
        # 平均单边换手（用于成本情景对比）
        if len(rebalance_list) > 1:
            tos = []
            pw = {}
            for rb in rebalance_list:
                w = rb["weights"]
                alls = set(pw) | set(w)
                to = 0.5 * sum(abs(w.get(s, 0) - pw.get(s, 0)) for s in alls)
                tos.append(to)
                pw = w
            metrics["avg_turnover"] = float(np.mean(tos))

        if benchmark_ret is not None and len(rets):
            bench = benchmark_ret.reindex(rets.index).fillna(0.0)
            excess = rets - bench
            info_ratio = float(excess.mean() / excess.std() * np.sqrt(252.0)) if excess.std() > 0 else float("nan")
            cov = np.cov(rets.values, bench.values)
            beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else float("nan")
            alpha_ann = float((rets.mean() - beta * bench.mean()) * 252.0)
            metrics["benchmark_info_ratio"] = info_ratio
            metrics["benchmark_alpha_ann"] = alpha_ann
            metrics["benchmark_beta"] = beta

        assumptions = {
            "t_plus_one": t_plus_one,
            "allow_short": allow_short,
            "limit_up_pct": limit_up_pct,
            "commission": commission,
            "stamp_tax": stamp_tax,
            "cost_mode": cost_mode,
            "min_daily_amount": min_daily_amount,
            "top_frac": top_frac,
            "forward_periods": self.forward_periods,
        }
        return {
            "equity": nav,
            "rebalance_list": rebalance_list,
            "metrics": metrics,
            "assumptions": assumptions,
        }

    def cost_sensitivity(
        self,
        kline: pd.DataFrame,
        factor: pd.Series,
        top_frac: float = 0.1,
        commission_grid: tuple = (0.0001, 0.0003, 0.0005),
        cost_modes: tuple = ("one_side", "two_side"),
        **opts,
    ) -> Dict[str, Dict[str, float]]:
        """换手成本情景分析：不同佣金率 × 单边/双边下的组合表现，量化「摩擦成本吃掉多少收益」。"""
        grid: Dict[str, Dict[str, float]] = {}
        for mode in cost_modes:
            for comm in commission_grid:
                res = self.realistic_portfolio(
                    kline, factor, top_frac=top_frac,
                    commission=comm, cost_mode=mode, **opts)
                if "error" in res:
                    continue
                m = res["metrics"]
                key = f"{mode}@{(comm * 10000):.1f}bp"
                grid[key] = {
                    "ann_return": m["ann_return"],
                    "sharpe": m["sharpe"],
                    "max_drawdown": m["max_drawdown"],
                    "cum_return": m["cum_return"],
                    "avg_turnover": m.get("avg_turnover", float("nan")),
                }
        return grid

    # ------------------------------------------------------------------
    # 基准
    # ------------------------------------------------------------------
    def equal_weight_benchmark(self, kline: pd.DataFrame) -> pd.Series:
        """等权基准日收益：成分股每日收益的简单平均（如中证800等权代理）。"""
        df = kline[["date", "symbol", "close"]].copy().sort_values(["symbol", "date"])
        df["ret"] = df.groupby("symbol")["close"].pct_change()
        bench = df.dropna(subset=["ret"]).groupby("date")["ret"].mean()
        return bench


# 模块级便捷函数
def equal_weight_benchmark(kline: pd.DataFrame) -> pd.Series:
    """等权基准日收益（模块级便捷入口）。"""
    return FactorBacktester().equal_weight_benchmark(kline)


    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------
def plot_metrics(self, metrics: dict) -> list:
        """根据回测指标生成图表列表（matplotlib Figure）。

        Returns:
            Figure 对象列表（IC 时间序列、分位数收益、多空权益曲线）。
            若缺少绘图依赖则返回空列表。
        """
        if not _HAS_PLOT or "_ic_series" not in metrics:
            return []
        sns.set_style("whitegrid")
        # 配置中文字体，避免标题/坐标轴中文显示为方块（Windows 优先微软雅黑/黑体）
        try:
            import matplotlib.pyplot as plt

            plt.rcParams["font.sans-serif"] = [
                "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
            ]
            plt.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass
        figs = []

        def _dt(ix):
            """字符串日期索引转 datetime，确保 DateFormatter 正确显示。"""
            try:
                return pd.to_datetime(ix)
            except Exception:
                return ix

        # 1) IC 时间序列
        ic = metrics["_ic_series"]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(_dt(ic.index), ic.values, lw=0.8, color="#2c7fb8")
        ax.axhline(ic.mean(), color="red", ls="--", lw=1, label=f"均值 {ic.mean():.4f}")
        ax.set_title("IC 时间序列")
        ax.set_xlabel("日期")
        ax.set_ylabel("IC")
        ax.legend(loc="lower right")
        # X 轴 tick 过多会压扁成黑条 → 限制最大刻度数并自动格式化日期
        try:
            from matplotlib.dates import DateFormatter
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
            ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        except Exception:
            pass
        fig.autofmt_xdate(rotation=0, ha="center")
        fig.tight_layout()
        figs.append(fig)

        # 2) 分位数收益
        qr = metrics.get("quantile_returns", {})
        if qr:
            fig, ax = plt.subplots(figsize=(8, 3))
            groups = sorted(qr.keys())
            vals = [qr[g] for g in groups]
            ax.bar([f"Q{g+1}" for g in groups], vals, color="#41ab5d")
            ax.axhline(0, color="black", lw=0.8)
            ax.set_title("分位数分组平均收益")
            ax.set_xlabel("分位组")
            ax.set_ylabel("未来收益均值")
            fig.tight_layout()
            figs.append(fig)

        # 3) 多空权益曲线
        ls = metrics.get("_ls_series")
        if ls is not None and len(ls):
            fig, ax = plt.subplots(figsize=(8, 3))
            equity = (1 + ls).cumprod()
            ax.plot(_dt(equity.index), equity.values, color="#c51b8a", lw=1)
            ax.set_title("多空对冲累计收益")
            ax.set_xlabel("日期")
            ax.set_ylabel("累计净值")
            try:
                from matplotlib.dates import DateFormatter
                from matplotlib.ticker import MaxNLocator
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
                ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
            except Exception:
                pass
            fig.autofmt_xdate(rotation=0, ha="center")
            fig.tight_layout()
            figs.append(fig)

        # 4) 分层累积收益曲线
        qc = metrics.get("quantile_cum")
        if qc:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            for g, s in sorted(qc.items()):
                ax.plot(_dt(s.index), s.values, lw=0.9, label=f"Q{int(g) + 1}")
            ax.axhline(0, color="gray", lw=0.8, ls="--")
            ax.set_title("分层累积收益曲线（Q1 最低分位 -> Qn 最高分位）")
            ax.set_xlabel("日期")
            ax.set_ylabel("累积收益")
            ax.legend(ncol=5, fontsize=8)
            try:
                from matplotlib.dates import DateFormatter
                from matplotlib.ticker import MaxNLocator
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
                ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
            except Exception:
                pass
            fig.autofmt_xdate(rotation=0, ha="center")
            fig.tight_layout()
            figs.append(fig)

        return figs


# 修复：将模块级 plot_metrics 绑定为 FactorBacktester 实例方法。
# 原定义因缩进错误位于模块函数 equal_weight_benchmark 体内（嵌套局部函数），
# 对外不可见，导致调用 bt.plot_metrics(...) 报
# 'FactorBacktester' object has no attribute 'plot_metrics'。
FactorBacktester.plot_metrics = lambda self, metrics: plot_metrics(self, metrics)
