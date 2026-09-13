# -*- coding: utf-8 -*-
"""``src/mining`` 因子挖掘层回归测试。

把四篇卖方研报的落地从"一次性冒烟脚本"固化为可回归的测试：
算子库（NumPy/numba 双实现与 pandas 对拍）、强类型表达式树、PIT 面板、
多维度评价、天风风险闸门、山西算子网格搜索，以及西部《概念数量因子》与
中信建投《量价 × 基本面统一框架》。

测试组织原则：小面板（100 × 360）跑算子 / 类型 / 评价类断言；大面板
（120 × 1050）只给需要四个季度 TTM 历史的基本面与概念用例——面板构造
与财报安装的代价高，用 module 级 fixture 共享。安装器都是幂等的，因此
用例之间的执行顺序不影响结果。
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mining import concept as CC          # noqa: E402
from mining import evaluator as EV        # noqa: E402
from mining import expr as ex             # noqa: E402
from mining import fundamental as FD      # noqa: E402
from mining import gridminer as GM        # noqa: E402
from mining import ops                    # noqa: E402
from mining import risk as R              # noqa: E402
from mining.panel import (DIM_FLAG, FieldMeta, PanelData,  # noqa: E402
                          asof_align, residualize_cs)

POOL_FIELDS = ("size", "beta", "momentum", "liquidity", "reversal")


# --------------------------------------------------------------------------
# fixture
# --------------------------------------------------------------------------
def _panel(n_symbols: int, n_days: int, seed: int = 5) -> PanelData:
    pn = PanelData.synthetic(n_symbols=n_symbols, n_days=n_days, seed=seed)
    R.install_risk_fields(pn)
    return pn


@pytest.fixture(scope="module")
def small_panel() -> PanelData:
    return _panel(100, 360)


@pytest.fixture(scope="module")
def deep_panel() -> PanelData:
    """TTM 因子要 ``ts_delay(x, 250/500/750)``，360 天面板会让它们天然全 NaN。"""
    return _panel(120, 1050)


@pytest.fixture(scope="module")
def pool(small_panel: PanelData):
    return {k: small_panel.field(k) for k in POOL_FIELDS}


@pytest.fixture(scope="module")
def search_result(small_panel: PanelData):
    """一次小规模网格搜索，供预算 / 报告两个用例共用（省一次 10 秒级跑批）。"""
    cfg = GM.SearchConfig(horizon=5, max_layers=2, width=10, max_expr=200,
                          max_seconds=90, min_ic=0.02, top_k=5)
    return GM.mine(small_panel, config=cfg,
                   pool={"size": small_panel.field("size")})


def _rand_frame(seed: int = 7, n_t: int = 120, n_s: int = 25
                ) -> "tuple[pd.DataFrame, np.random.Generator]":
    """带 5% 缺失的随机面板（固定 seed，保证对拍可复现）。"""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n_t, n_s)) * 2 + 1
    raw[rng.random((n_t, n_s)) < 0.05] = np.nan
    df = pd.DataFrame(raw, index=pd.bdate_range("2023-01-02", periods=n_t),
                      columns=[f"s{i:02d}" for i in range(n_s)])
    return df, rng


# --------------------------------------------------------------------------
# 1. 算子库：与 pandas / 手算对拍
# --------------------------------------------------------------------------
def test_rolling_unary_matches_pandas():
    """滚动核（含 nan 跳过与 min_periods 口径）必须与 pandas 逐点一致。"""
    df, _ = _rand_frame()
    w = 10
    ref = {
        "mean": df.rolling(w, min_periods=5).mean(),
        "std": df.rolling(w, min_periods=5).std(),
        "sum": df.rolling(w, min_periods=5).sum(),
        "min": df.rolling(w, min_periods=5).min(),
        "max": df.rolling(w, min_periods=5).max(),
        "median": df.rolling(w, min_periods=5).median(),
        "skew": df.rolling(w, min_periods=5).skew(),
        "count": df.rolling(w, min_periods=5).count(),
    }
    for kind, exp in ref.items():
        got = ops.rolling_unary(df, w, kind).to_numpy()
        diff = float(np.nanmax(np.abs(got - exp.to_numpy())))
        assert diff < 1e-9, f"{kind} 与 pandas 偏差 {diff:.3e}"


def test_rolling_binary_matches_pandas():
    df, rng = _rand_frame()
    other = pd.DataFrame(rng.standard_normal(df.shape), index=df.index,
                         columns=df.columns)
    got = ops.rolling_binary(df, other, 20, "corr").to_numpy()
    exp = df.rolling(20, min_periods=10).corr(other).to_numpy()
    assert float(np.nanmax(np.abs(got - exp))) < 1e-9


def test_ts_hand_checks():
    """ts_rank / ts_decay_linear / ts_slope / ts_beta 用独立手算复核。

    这四个算子的口径（分位定义、线性衰减权重、最小二乘斜率）在研报里各写各的，
    必须钉死在测试里，否则以后"优化"实现时很容易悄悄改掉语义。
    """
    df, rng = _rand_frame()
    w, i, col = 10, 60, "s03"
    win = df[col].iloc[i - w + 1:i + 1]

    # ts_rank：当前值在窗口内的分位（"<=" 口径）
    got = float(ops.rolling_unary(df, w, "rank").loc[df.index[i], col])
    valid = win.dropna()
    exp = float((valid <= win.iloc[-1]).sum() / len(valid))
    assert got == pytest.approx(exp, abs=1e-12)

    # ts_decay_linear：权重 ∝ (w - age)，age=0 为今日
    got = float(ops.rolling_unary(df, w, "decay").loc[df.index[i], col])
    num = den = 0.0
    for k in range(w):
        v = df[col].iloc[i - k]
        if np.isfinite(v):
            num += float(v) * (w - k)
            den += (w - k)
    assert got == pytest.approx(num / den, rel=1e-12)

    # ts_slope：对时间轴做最小二乘
    y = win.to_numpy()
    m = np.isfinite(y)
    exp_slope = float(np.polyfit(np.arange(w)[m], y[m], 1)[0])
    assert float(ops._ts_slope(df, w).loc[df.index[i], col]) == \
        pytest.approx(exp_slope, rel=1e-9)

    # ts_beta：对自变量（另一条序列）做最小二乘
    other = pd.DataFrame(rng.standard_normal(df.shape), index=df.index,
                         columns=df.columns)
    got = float(ops.rolling_binary(other, df, 20, "beta").iloc[i, 3])
    x = other.iloc[i - 19:i + 1, 3].to_numpy()
    yb = df.iloc[i - 19:i + 1, 3].to_numpy()
    mm = np.isfinite(x) & np.isfinite(yb)
    assert got == pytest.approx(float(np.polyfit(x[mm], yb[mm], 1)[0]),
                                rel=1e-9)


def test_cs_ops_invariants():
    df, _ = _rand_frame()
    z = ops.cs_zscore(df)
    assert float(z.mean(axis=1).abs().max()) < 1e-9
    assert float((z.std(axis=1, ddof=1) - 1).abs().max()) < 1e-9
    r = ops.cs_rank(df)                # pct 秩：最小值是 1/n，不是 0
    assert float(r.max().max()) == 1.0
    assert float(r.min().min()) >= 1.0 / df.shape[1] - 1e-12
    q = ops.cs_quantile(df, 5)
    left = set(np.unique(q.to_numpy()[np.isfinite(q.to_numpy())]))
    assert left <= {0.0, 0.25, 0.5, 0.75, 1.0}, left


def test_cs_corr_pearson_matches_daily_corrcoef():
    """向量化横截面 Pearson 相关必须与逐日 ``np.corrcoef`` 逐点等价。"""
    a, _ = _rand_frame(1)
    b, _ = _rand_frame(2)
    got = ops.cs_corr(a, b, rank=False, min_stocks=10).to_numpy()
    exp = np.full(len(a), np.nan)
    for i in range(len(a)):
        x, y = a.iloc[i], b.iloc[i]
        m = x.notna() & y.notna()
        if int(m.sum()) < 10:            # 样本不足的截面必须给 NaN，不能给 0
            continue
        xv, yv = x[m].to_numpy(), y[m].to_numpy()
        if np.std(xv) == 0 or np.std(yv) == 0:
            continue
        exp[i] = float(np.corrcoef(xv, yv)[0, 1])
    np.testing.assert_allclose(got, exp, rtol=1e-9, atol=1e-12)


def test_cs_corr_rank_convention():
    """RankIC 口径 = **先各自横截面排名，再在共同有效样本上求相关**。

    这与 pandas ``rank()`` 后 ``corrwith``（pairwise 删缺失）一致，但当两个
    变量的缺失分布不同时，它并不等于"先取共同样本再排名"的严格 Spearman；
    两种算法在缺失错位时会有可见差异，因此这里把口径钉死，避免被当成 bug 改掉。
    """
    a, _ = _rand_frame(1)
    b, _ = _rand_frame(2)
    got = ops.cs_corr(a, b, rank=True, min_stocks=10).to_numpy()
    ra = ops.cs_rank(a).to_numpy()
    rb = ops.cs_rank(b).to_numpy()
    exp = np.full(len(a), np.nan)
    for i in range(len(a)):
        m = np.isfinite(ra[i]) & np.isfinite(rb[i])
        if int(m.sum()) < 10:
            continue
        if np.std(ra[i][m]) == 0 or np.std(rb[i][m]) == 0:
            continue
        exp[i] = float(np.corrcoef(ra[i][m], rb[i][m])[0, 1])
    np.testing.assert_allclose(got, exp, rtol=1e-9, atol=1e-12)
    assert np.isfinite(got).any()
    assert a.isna().to_numpy().any() and b.isna().to_numpy().any(), \
        "本用例刻意让两个变量的缺失错位，否则退化成严格 Spearman"


def test_cs_dgtw_removes_group_levels():
    """DGTW 的核心性质：**组内**去均值/标准化，组间差异被整体剥离。"""
    rng = np.random.default_rng(3)
    n_t, n_s = 40, 50
    dates = pd.bdate_range("2024-01-01", periods=n_t)
    cols = [f"s{i:02d}" for i in range(n_s)]
    group = pd.DataFrame(rng.standard_normal((n_t, n_s)),
                         index=dates, columns=cols)
    # 构造"强组间差异"的因子：分位组均值依次抬升，组内只有小噪声
    g = group.rank(axis=1, pct=True).to_numpy()
    bucket = np.clip(np.floor(g * 5).astype(int), 0, 4)
    x = bucket * 10.0 + rng.standard_normal((n_t, n_s)) * 0.1
    fac = pd.DataFrame(x, index=dates, columns=cols)

    out = ops.cs_dgtw(fac, group, 5).to_numpy()
    for t in range(0, n_t, 7):          # 逐日逐组核查（抽 6 天即可）
        for b in range(5):
            vals = out[t][bucket[t] == b]
            vals = vals[np.isfinite(vals)]
            assert abs(float(vals.mean())) < 1e-9, f"第 {t} 天第 {b} 组仍有非零均值"
            # 组内标准化后标准差为 1（ddof=1 口径），量纲被统一
            assert float(vals.std(ddof=1)) == pytest.approx(1.0, rel=1e-6)
    # 原始因子的组间差异必须很大，说明上面的断言不是"本来就成立"
    raw_means = [x[bucket == b].mean() for b in range(5)]
    assert max(raw_means) - min(raw_means) > 30.0
    # 组内去均值 ⇒ 组间差异被整体剥离（DGTW 相比直线中性化的关键差别）
    out_means = [out[bucket == b].mean() for b in range(5)]
    assert max(out_means) - min(out_means) < 1e-9


# --------------------------------------------------------------------------
# 2. 表达式树：解析 / 类型闸门 / 缓存
# --------------------------------------------------------------------------
def test_expr_roundtrip_and_dedup(small_panel):
    reg = small_panel.registry
    for text in ("zscore_cs(ts_pct(close, 20))",
                 "ts_corr(close, volume, 60)",
                 "mul(rank_cs(ts_std(ret, 20)), rank_cs(amount))",
                 "-zscore_cs(ts_pct(close, 20)) + zscore_cs(ts_pct(volume, 20))",
                 "spread(ts_mean(vwap, 5), ts_mean(close, 20))",
                 "neutral(zscore_cs(ts_pct(close, 20)), size)"):
        node = ex.parse(text)
        assert ex.parse(node.render()).key() == node.key(), text
        assert ex.validate(node, reg) == [], text

    a = ex.parse("zscore_cs(ts_pct(close, 20))")
    b = ex.parse("zscore_cs(ts_pct(volume, 20))")
    assert ex.Call("add", (a, b)).key() == ex.Call("add", (b, a)).key()
    assert ex.Call("sub", (a, b)).key() != ex.Call("sub", (b, a)).key()


def test_expr_type_gates(small_panel):
    """量纲闸门：不同维度不可加减，flag 不可进算术，乘法是合法跨量纲通道。"""
    reg = small_panel.registry.copy()
    reg.register(FieldMeta("is_st", DIM_FLAG, "risk", "X", "test", "ST 标志"))

    for text in ("add(close, amount)", "sub(close, volume)",
                 "log(is_st)", "add(is_st, is_st)"):
        assert ex.validate(ex.parse(text), reg), f"{text} 应被类型闸门拒绝"
    assert ex.validate(ex.parse("mul(close, amount)"), reg) == []


def test_expr_neutral_only_outermost(small_panel):
    reg = small_panel.registry
    ok = ex.parse("neutral(zscore_cs(ts_pct(close, 20)), size)")
    assert ex.validate(ok, reg) == []
    for text in ("ts_mean(neutral(close, size), 20)",
                 "neutral(zscore_cs(close), neutral(amount, size))"):
        errs = ex.validate(ex.parse(text), reg)
        assert errs and "最外层" in errs[0], errs


def test_expr_type_propagation(small_panel):
    reg = small_panel.registry
    assert ex.infer_type(ex.parse("mul(close, volume)"), reg).dimension == "amount"
    assert ex.infer_type(ex.parse("div(close, amount)"), reg).dimension == "ratio"
    t = ex.infer_type(ex.parse("add(zscore_cs(close), zscore_cs(amount))"), reg)
    assert t.dimension == "score" and t.semantics == "mix"


def test_expr_serialization_and_cache(small_panel):
    node = ex.parse("neutral(zscore_cs(ts_pct(close, 20)), size)")
    assert ex.from_dict(node.to_dict()).key() == node.key()

    ev = ex.Evaluator(small_panel, small_panel.registry)
    first = ev.run(node)
    assert first.shape == (small_panel.n_dates, small_panel.n_symbols)
    ev.run(node)
    ev.run(ex.parse("add(zscore_cs(ts_pct(close, 20)),"
                    " zscore_cs(ts_pct(volume, 20)))"))
    assert ev.hits >= 1, "同一子表达式重复求值必须命中缓存"


def test_expr_lookback_warmup():
    """预热窗口：delay 类多算一档窗口，ts2 类按整窗计，cs2 的分组参数不计时间。"""
    cases = {
        "close": 0,
        "ts_mean(close, 20)": 19,
        "ts_corr(close, volume, 60)": 60,
        "ts_delay(close, 250)": 250,
        "ts_mean(ts_delay(close, 250), 20)": 269,
        "neutral(ts_pct(close, 60), size)": 60,
        "dgtw_cs(ts_mean(close, 20), size, 5)": 19,
    }
    for text, exp in cases.items():
        assert ex.lookback(ex.parse(text)) == exp, text


# --------------------------------------------------------------------------
# 3. 面板：PIT 对齐与切片
# --------------------------------------------------------------------------
def test_asof_align_no_lookahead():
    dates = pd.bdate_range("2023-01-02", periods=10)
    long_df = pd.DataFrame({
        "symbol": ["A", "A", "B"],
        "ann_date": ["2023-01-03", "2023-01-06", "2023-01-04"],
        "value": [1.0, 2.0, 5.0],
    })
    pit = asof_align(long_df, dates, lag_days=1)
    assert np.isnan(pit.loc["2023-01-03", "A"]), "公告+滞后之前必须为 NaN"
    assert pit.loc["2023-01-04", "A"] == 1.0
    assert pit.loc["2023-01-06", "A"] == 1.0, "01-06 的公告当天不可用"
    assert pit.loc["2023-01-09", "A"] == 2.0, "01-06+1 天落到下一交易日"
    assert np.isnan(pit.loc["2023-01-04", "B"])
    assert pit.loc["2023-01-05", "B"] == 5.0


def test_residualize_cs_orthogonalizes(small_panel):
    mom = ops.cs_zscore(ops.ts_pct(small_panel.field("close"), 60))
    size = ops.cs_zscore(small_panel.field("size"))
    res = residualize_cs(mom, [small_panel.field("size")])
    left = float(ops.cs_corr(res, size, min_stocks=20).mean())
    assert abs(left) < 1e-9, f"中性化后与控件仍残留相关 {left:.2e}"


def test_coverage_adj_ignores_warmup(small_panel):
    """预热期造成的低覆盖率不应被判成"因子无效"。"""
    fac = ops.ts_mean(ops.ts_delay(small_panel.field("close"), 250), 20)
    st = EV.data_quality(fac, 269)
    assert st["coverage"] < 0.6, "原始覆盖率应包含必然为空的预热期"
    assert st["coverage_adj"] > 0.99, "扣除预热后应接近满覆盖"
    assert st["warmup"] == 269


# --------------------------------------------------------------------------
# 4. 多维度评价
# --------------------------------------------------------------------------
def test_evaluate_structure_directions(small_panel, pool):
    """合成面板埋入的结构必须被评价体系检出，方向不能反。"""
    cases = [
        ("zscore_cs(ts_pct(close, 60))", "+", 5, "长窗口动量"),
        ("zscore_cs(ts_pct(close, 20))", "+", 5, "中窗口动量"),
        ("zscore_cs(ts_std(ret, 20))", "-", 5, "低波动异象"),
        ("rank_cs(amount)", "-", 5, "流动性"),
        ("zscore_cs(ts_decay_linear(ret, 10))", "-", 1, "短期反转"),
    ]
    for text, sign, h, why in cases:
        r = EV.evaluate_expr(text, small_panel, pool=pool,
                             config=EV.EvalConfig(primary_horizon=h))
        assert r.ok, (text, r.errors)
        ic = r.metrics["rank_ic_mean"]
        if sign == "+":
            assert ic > 0.01, f"{why} 应正 IC，实际 {ic:+.4f}"
        else:
            assert ic < -0.01, f"{why} 应负 IC，实际 {ic:+.4f}"


def test_evaluate_metrics_types_and_group_detail(small_panel, pool):
    r = EV.evaluate_expr("zscore_cs(ts_pct(close, 60))", small_panel, pool=pool)
    assert all(isinstance(v, float) for v in r.metrics.values()), \
        [k for k, v in r.metrics.items() if not isinstance(v, float)]
    assert isinstance(r.detail["group_returns"], list)
    assert len(r.detail["group_returns"]) == EV.EvalConfig().n_groups
    assert r.metrics["group_ls_t"] > 0, "多空方向应与 IC 同号"


def test_quantile_stats_survives_empty_groups():
    """窄截面 / 空组是正常数据状态：不得抛 RuntimeWarning，也不得编造 0 收益。"""
    idx = pd.bdate_range("2024-01-01", periods=6)
    cols = [f"s{i}" for i in range(12)]
    factor = pd.DataFrame(np.arange(72, dtype=float).reshape(6, 12),
                          index=idx, columns=cols)
    fwd = pd.DataFrame(np.arange(72, dtype=float).reshape(6, 12) * 1e-3,
                       index=idx, columns=cols)
    fwd.iloc[0] = np.nan            # 首个截面全部缺失 → 空组
    factor.iloc[1, :3] = np.nan
    with np.errstate(all="raise"):
        st = EV.quantile_stats(factor, fwd, n_groups=5, min_stocks=5)
    assert np.isfinite(st["group_ls_mean"])
    assert len(st["group_returns"]) == 5


def test_evaluate_rejects_invalid(small_panel):
    """类型闸门 / 未注册字段 / 算子位置错误都必须让评价器显式失败，而不是静默出数。"""
    for text in ("add(close, amount)", "unknown_field",
                 "ts_mean(neutral(zscore_cs(close), size), 20)"):
        r = EV.evaluate_expr(text, small_panel)
        assert not r.ok and r.errors, f"{text} 应被判为无效"
    with pytest.raises(ex.ExprParseError):
        ex.parse("ts_std(close, 0)")


# --------------------------------------------------------------------------
# 5. 风险闸门
# --------------------------------------------------------------------------
def test_risk_report_and_risk_contribution(small_panel):
    exp = R.default_risk_exposures(small_panel)
    assert "size" in exp and "reversal" in exp
    mom = ops.cs_zscore(ops.ts_pct(small_panel.field("close"), 60))
    fwd5 = small_panel.fwd(5)
    rep = R.risk_report(mom, fwd5, exp, pool={"reversal": exp["reversal"]})
    assert rep["delta_r2"] > 0
    for key in ("delta_r2_adj", "t_mean", "t_gt2_ratio", "vif", "ac_l1",
                "crowding_score", "exposure_hhi"):
        assert key in rep, key

    fr = R.factor_returns(fwd5, exp)
    cov = R.factor_cov(fr, nw_lag=1)
    w = {k: 1.0 / len(exp) for k in exp}
    contrib = R.portfolio_risk_contribution(w, fr, cov)
    gap = float(contrib["component_vol"].sum()) - float(contrib.attrs["portfolio_vol"])
    assert abs(gap) < 1e-12, "成分波动贡献之和必须等于组合波动"


def test_risk_metrics_survive_degenerate_input():
    """全常数截面不得触发除零告警，也不得编造出 0 自相关。

    历史 bug：``np.corrcoef`` 逐日循环在标准差为 0 的行上抛
    ``invalid value encountered in divide``，且被静默吞成 0。现在统一走
    ``ops.cs_corr``，退化截面给 NaN，调用方按 NaN 处理。
    """
    idx = pd.bdate_range("2024-01-01", periods=30)
    cols = [f"s{i}" for i in range(15)]
    const = pd.DataFrame(1.0, index=idx, columns=cols)
    with np.errstate(all="raise"):
        ac = R.autocorr(const, lags=(1, 2))
        cr = R.crowding(const, pool={"dup": const})
    assert set(ac) == {"ac_l1", "ac_l2", "ac_decay"}
    assert all(not np.isfinite(v) for v in ac.values()), ac
    assert 0.0 <= cr["crowding_score"] <= 1.0


# --------------------------------------------------------------------------
# 6. 算子网格搜索
# --------------------------------------------------------------------------
def test_gridminer_budget_and_reproducibility(small_panel, search_result):
    cfg = GM.SearchConfig(**search_result.config)      # SearchResult.config 是 dict
    res = search_result
    assert res.history, "应有逐层审计记录"
    assert res.n_evaluated <= cfg.max_expr, "求值预算是硬约束"
    assert res.candidates, "应至少产出候选"
    res2 = GM.mine(small_panel, config=cfg, pool={"size": small_panel.field("size")})
    assert [c.expression for c in res.candidates] == \
           [c.expression for c in res2.candidates], "同 seed 同配置必须可复现"


# --------------------------------------------------------------------------
# 7. 中信建投：PIT 财务 → 量价 × 基本面统一框架
# --------------------------------------------------------------------------
def test_fundamental_history_requirement(deep_panel):
    hist = FD.check_history(deep_panel)
    assert hist["ok"], hist["advice"]
    assert hist["n_days"] >= FD.MIN_HISTORY_DAYS


def test_fundamental_pit_install_and_selfcheck(deep_panel):
    q = FD.synthetic_quarterly(deep_panel, seed=7)
    inst = FD.install_fundamentals(deep_panel, q, lag_days=1)
    assert set(inst) >= {"revenue", "net_profit", "gross_profit", "cfo",
                         "total_assets", "total_equity", "total_liab",
                         "shares", "market_cap"}
    bad = {f: FD.check_pit(deep_panel, q, f, lag_days=1)
           for f in ("revenue", "net_profit", "gross_profit", "cfo",
                     "total_assets", "total_equity", "total_liab", "shares")}
    assert all(r["ok"] for r in bad.values()), \
        {f: r for f, r in bad.items() if not r["ok"]}


def test_fundamental_factor_library_types_and_directions(deep_panel):
    assert FD.library_errors(deep_panel.registry) == {}
    pool = {k: deep_panel.field(k) for k in
            ("size", "momentum", "liquidity", "resid_vol")}
    expect = {"ROE_TTM_z": "+", "净利润同比_z": "+", "PB": "+",
              "应计利润_z": "-", "资产负债率": "-"}
    for name, sign in expect.items():
        r = EV.evaluate_expr(FD.FUND_FACTOR_LIBRARY[name], deep_panel, pool=pool)
        assert r.ok, (name, r.errors)
        ic = r.metrics["rank_ic_mean"]
        ok = ic > 0.005 if sign == "+" else ic < -0.005
        assert ok, f"{name} 方向不符：{ic:+.4f}（预期 {sign}）"


def test_fundamental_mixed_templates_type_safe(deep_panel):
    mix = FD.mixed_templates(FD.market_factor_library()["动量"],
                             FD.FUND_FACTOR_LIBRARY["ROE_TTM_z"])
    assert mix
    bad = {k: ex.validate(ex.parse(t), deep_panel.registry)
           for k, t in mix.items()}
    assert not any(bad.values()), {k: v for k, v in bad.items() if v}


# --------------------------------------------------------------------------
# 8. 西部证券：概念数量因子
# --------------------------------------------------------------------------
def test_concept_membership_respects_intervals():
    """成分关系必须按区间生效：到期即归零，禁止 ffill 到面板末尾。"""
    pn = PanelData.synthetic(n_symbols=3, n_days=30, seed=1)
    syms = list(pn.symbols)
    dates = pn.dates
    mem = pd.DataFrame([
        {"symbol": syms[0], "concept": "C1",
         "start_date": dates[5], "end_date": dates[10]},
        {"symbol": syms[1], "concept": "C1",
         "start_date": dates[5], "end_date": pd.NaT},
        {"symbol": syms[2], "concept": "C1",
         "start_date": dates[8], "end_date": pd.NaT},
    ])
    cnt, avg_size = CC.concept_membership_counts(mem, pn)
    total = cnt.sum(axis=1)
    assert total[:5].sum() == 0.0, "生效日之前不应计入"
    assert total[5] == 2.0 and total[7] == 2.0
    assert total[8] == 3.0 and total[10] == 3.0
    assert total[11] == 2.0 and cnt[11, 0] == 0.0, "区间结束后必须归零"
    assert avg_size[5, 0] == pytest.approx(2.0)
    assert avg_size[8, 0] == pytest.approx(3.0)


def test_concept_library_and_directions(deep_panel):
    mem = CC.synthetic_concepts(deep_panel, n_concepts=40, seed=13)
    cf = CC.install_concepts(deep_panel, mem)
    assert cf["concept_count"].to_numpy().max() >= 3
    assert CC.concept_library_errors(deep_panel.registry) == {}

    pool = {k: deep_panel.field(k) for k in
            ("size", "momentum", "liquidity", "resid_vol")}
    for name in ("CN", "CN_z", "ACN_60", "稀缺度", "热度"):
        r = EV.evaluate_expr(CC.CONCEPT_FACTOR_LIBRARY[name], deep_panel,
                             pool=pool)
        assert r.ok, (name, r.errors)
        ic = r.metrics["rank_ic_mean"]
        assert ic > 0.005, f"{name} 方向应为正，实际 {ic:+.4f}"


def test_dgtw_beats_linear_neutralization(deep_panel):
    """概念数量与市值高度相关；DGTW 分组调整要比线性中性化更彻底。

    合成数据的成员概率是规模的 sigmoid（饱和型非线性），线性正交只能剥掉
    线性部分，这正是西部证券推荐分组调整的原因，断言把这个差距钉住。
    """
    cn = deep_panel.field("concept_count")
    size = deep_panel.field("size")
    raw = abs(float(EV.rank_ic(cn, size).mean()))
    neu = abs(float(EV.rank_ic(
        EV.evaluate_expr("neutral(concept_count, size)", deep_panel)
        .detail["factor"], size).mean()))
    dgtw = abs(float(EV.rank_ic(
        EV.evaluate_expr("dgtw_cs(concept_count, size, 5)", deep_panel)
        .detail["factor"], size).mean()))
    assert raw > 0.3, f"合成数据应保留概念数×市值的机械关系，实际 {raw:.3f}"
    assert neu < raw * 0.75, f"线性中性化削弱不足：{raw:.3f} → {neu:.3f}"
    assert dgtw < raw * 0.35, f"DGTW 削弱不足：{raw:.3f} → {dgtw:.3f}"
    assert dgtw < neu, f"DGTW({dgtw:.3f}) 应比线性中性化({neu:.3f}) 更彻底"


# --------------------------------------------------------------------------
# 9. 研究报告：把上面所有产物落成可复核的文档
# --------------------------------------------------------------------------
def test_report_markdown_and_json_payload(small_panel, pool, search_result, tmp_path):
    """报告必须自包含、可复现、且**不含伪造的 nan**（缺失一律渲染成 ``—``）。"""
    from mining import report as RP

    exprs = ("zscore_cs(ts_pct(close, 20))", "ts_corr(close, volume, 60)")
    reports = [EV.evaluate_expr(t, small_panel, pool=pool, with_risk=True)
               for t in exprs]
    mom = EV.evaluate_expr(exprs[0], small_panel, pool=pool).detail["factor"]
    ri = RP.ReportInput(
        panel=small_panel,
        title="回归测试报告",
        reports=reports,
        search=search_result,
        history={"n_days": 360, "ok": True, "advice": ""},
        pit_checks=[{"field": "revenue", "symbols_checked": 10, "mismatch_points": 0,
                     "lookahead_points": 0, "ok": True}],
        pools={"mom20": EV.pool_correlation(mom, pool)},
        incremental={"mom20": EV.incremental_ic(mom, pool, small_panel.fwd(5))},
        corr=EV.factor_corr_matrix({r.name: r.detail["factor"] for r in reports}),
        notes=["synthetic panel"],
        generated_at="2026-01-01T00:00:00Z",
    )

    md = RP.render_markdown(ri)
    for head in ("## 摘要", "## 数据前置条件", "## 网格搜索", "## 因子 1",
                 "## 风险闸门", "## 因子相关性", "## 池内相关性",
                 "## 增量信息（相对既有因子池）", "## 复现信息"):
        assert head in md, f"缺少章节 {head}"
    assert md.startswith("# 回归测试报告")
    assert "2026-01-01T00:00:00Z" in md, "显式生成时刻必须被原样保留（可复现）"
    assert "nan" not in md.lower(), "文档里不得出现 nan，缺失值应渲染为 —"
    assert "`dgtw_groups`" in md, "复现信息应列出搜索配置"
    assert "| `max_expr` |" in md and ".000" not in md.split("## 复现信息")[1], \
        "整数配置项不得被渲染成 120.000 这种伪小数"
    assert r"暴露平均 \|t\|" in md, "表格单元里的竖线必须转义，否则列会被切碎"
    assert search_result.stop_reason in md

    payload = RP.report_payload(ri)
    # allow_nan=False：载荷里一旦混入 NaN 就直接抛错，这是"没有伪造缺失值"的硬约束
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    assert payload["n_reports"] == len(reports)
    assert payload["search"]["n_evaluated"] == search_result.n_evaluated

    paths = RP.write_report(str(tmp_path / "sub" / "report.md"), ri)
    assert os.path.isfile(paths["markdown"]) and os.path.isfile(paths["json"])
    with open(paths["markdown"], encoding="utf-8") as fh:
        assert fh.read() == md, "落盘内容必须与渲染结果逐字一致"
    with open(paths["json"], encoding="utf-8") as fh:
        assert json.load(fh)["title"] == "回归测试报告"
    assert text


def test_report_survives_missing_inputs_and_failed_factors(small_panel):
    """缺项少写、坏因子记因——报告生成器不许在流水线末尾抛异常。"""
    from mining import report as RP

    ri = RP.build_from_expressions(
        small_panel, ["unknown_field", "zscore_cs(ts_pct(close, 20))"],
        title="容错测试", generated_at="2026-01-01T00:00:00Z")
    assert len(ri.reports) == 2
    md = RP.render_markdown(ri)
    assert "## 因子 1" in md and "## 因子 2" in md
    assert "未注册字段" in md, "失败原因必须留在报告里"
    assert "## 网格搜索" not in md, "没有搜索输入就不该凭空渲染搜索章节"
    payload = RP.report_payload(ri)
    assert payload["reports"][0]["errors"], "失败因子的错误要进载荷"
    json.dumps(payload, ensure_ascii=False, allow_nan=False)

    empty = RP.render_markdown(RP.ReportInput(title="空报告"))
    assert empty.startswith("# 空报告") and "## 摘要" not in empty
