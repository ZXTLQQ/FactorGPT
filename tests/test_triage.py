# -*- coding: utf-8 -*-
"""``src/mining/triage`` 验收闸门回归测试。

``engine.significance`` / ``engine.universe`` / ``engine.multiscale_gp`` 三个模块
此前只有库级实现，没有任何真实调用链路。``mining.triage`` 是它们与挖掘层之间的
薄适配，本文件钉住三件事：

1. **桥接口径**：宽表面板 → 长表时 ``pct_chg`` 的百分数量纲、前瞻收益的组内
   ``shift(−h)`` 与"末尾置 NaN 而非 0"，都必须逐点可核对 —— 前瞻标签算错一次，
   后面所有 IC 都是前视污染的产物，而且表现是"异常漂亮"，最难自查；
2. **口径不许被悄悄放宽**：多重性校正的分母是**本次搜索实际评估过的表达式数**、
   选股域三档按面板自身成交额分位（不是引擎的绝对门槛）、多尺度挖掘的评估次数
   节省与墙钟节省分开报；
3. **报告层**：三项验收各自成章、缺项整段省略、表格列数自洽、载荷可 JSON 序列化。

测试成本：多尺度挖掘在两个用例里各跑一次（module 级 fixture + 复现性用例），
面板取 30 × 320，并把两个尺度代的种群压到 10 / 8，整体在秒级；§8 另跑两次
小规模 GP 演化（2 代 × 每簇 8 个体）验证复现性，同样是秒级。
"""
import ast
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mining import evaluator as EV            # noqa: E402
from mining import report as RP               # noqa: E402
from mining import risk as R                  # noqa: E402
from mining import triage as TR               # noqa: E402
from mining.panel import PanelData            # noqa: E402

N_SYMBOLS, N_DAYS, SEED = 30, 320, 5
HORIZON = 5


# --------------------------------------------------------------------------
# fixture
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pn() -> PanelData:
    panel = PanelData.synthetic(n_symbols=N_SYMBOLS, n_days=N_DAYS, seed=SEED)
    R.install_risk_fields(panel)
    return panel


@pytest.fixture(scope="module")
def pool(pn: PanelData):
    return {"size": pn.field("size")}


@pytest.fixture(scope="module")
def reports(pn: PanelData, pool):
    """两个真实的 ``FactorReport``（含 ``detail['ic_series']``），名字显式指定。"""
    out = []
    for name, text in (("手工动量", "zscore_cs(ts_pct(close, 60))"),
                       ("手工流动性", "rank_cs(amount)")):
        rep = EV.evaluate_expr(text, pn, pool=pool)
        assert rep.ok, rep.errors
        rep.name = name
        out.append(rep)
    return out


@pytest.fixture(scope="module")
def specs(pn: PanelData):
    return TR.quantile_domains(pn, top_n=20)


@pytest.fixture(scope="module")
def domains(pn: PanelData, specs):
    return TR.domain_check(pn, pn.field("momentum"), specs=specs,
                           horizon=HORIZON, min_count=5, top_n=20)


@pytest.fixture(scope="module")
def universe(pn: PanelData, specs):
    return TR.universe_check(pn, spec=specs[1], horizon=HORIZON, keep_flags=False)


@pytest.fixture(scope="module")
def ic_map(reports):
    """一强一噪声：强信号用固定种子的高均值低波动序列，噪声均值精确为 0。

    噪声序列**显式去中心**，因此 ``observed ≈ 0``，bootstrap 的 p 值必然为 1，
    不需要靠"这个种子恰好不显著"来撑着断言。
    """
    rng = np.random.default_rng(20260913)
    strong = pd.Series(rng.normal(0.06, 0.08, size=250))
    noise = pd.Series(rng.normal(0.0, 0.08, size=250))
    noise = noise - noise.mean()
    assert abs(float(noise.mean())) < 1e-15, "去中心只剩浮点残差"
    return {"强信号": strong, "纯噪声": noise}


@pytest.fixture(scope="module")
def stub_search(reports):
    """带 ``best_reports`` / ``n_evaluated`` 的搜索替身（不跑真实网格搜索）。"""
    return _StubSearch(reports, n_evaluated=137, stop_reason="stub-budget")


class _StubSearch:
    def __init__(self, reports, n_evaluated: int, stop_reason: str = "") -> None:
        self._reports = list(reports)
        self.n_evaluated = int(n_evaluated)
        self.stop_reason = stop_reason

    def best_reports(self, k: int = 10):
        return self._reports[:int(k)]


@pytest.fixture(scope="module")
def multiscale(pn: PanelData):
    """多尺度挖掘结果（压小种群与代数，保持口径完整）。"""
    from engine.multiscale_gp import ScaleSpec

    levels = (ScaleSpec(name="coarse", freq="M", generations=2, pop_size=10, elite=3),
              ScaleSpec(name="fine", freq="D", generations=2, pop_size=8, elite=3))
    return TR.multiscale_mine(pn, horizon=HORIZON, n_intervals=4, n_select=1,
                              seed=7, terminal_weight=0.35, levels=levels)


@pytest.fixture(scope="module")
def acceptance(pn: PanelData, reports, stub_search):
    """一次跑齐（多尺度已在别处覆盖，这里关掉以省一次演化）。"""
    return TR.acceptance(pn, reports=reports, search=stub_search,
                         factor=pn.field("momentum"), horizon=HORIZON,
                         n_boot=300, top_k=2, top_n=20, do_multiscale=False)


# --------------------------------------------------------------------------
# 1. 宽表 → 长表：桥接口径
# --------------------------------------------------------------------------
def test_panel_long_adds_percent_pct_chg(pn: PanelData):
    """``pct_chg`` 必须是**百分数**量纲，且与 ``close`` 同一数据源。"""
    long = TR.panel_long(pn)
    assert {"date", "symbol", "close", "volume", "amount",
            "pct_chg"} <= set(long.columns)
    assert float(long["pct_chg"].abs().max()) > 1.0, "百分数量纲下日涨跌幅必然 > 1"

    wide = pn.field("close")
    ref = pd.DataFrame((wide / wide.shift(1) - 1.0) * 100.0)
    ref_long = PanelData._to_long({"ref_pct": ref})
    merged = long[["date", "symbol", "pct_chg"]].merge(
        ref_long, on=["date", "symbol"], how="inner", validate="one_to_one")
    assert len(merged) == pn.n_dates * pn.n_symbols, "长表必须逐 (日期, 标的) 唯一"
    np.testing.assert_allclose(merged["pct_chg"].to_numpy(dtype=float),
                               merged["ref_pct"].to_numpy(dtype=float),
                               rtol=1e-12, atol=1e-14, equal_nan=True)


def test_panel_long_skips_unknown_fields_but_needs_close(pn: PanelData):
    """顺手多写一个字段名不该让整段失败；但缺 ``close`` 必须显式报错。"""
    sub = TR.panel_long(pn, fields=("close", "not_a_field"))
    assert set(sub.columns) == {"date", "symbol", "close", "pct_chg"}

    bare = PanelData({"volume": pn.field("volume")})
    with pytest.raises(ValueError):
        TR.panel_long(bare, fields=("volume",))


def test_kline_long_forward_return_matches_manual(pn: PanelData):
    """前瞻收益 = 组内 ``close[t+h]/close[t] − 1``，末尾 h 期 NaN 而非 0。"""
    long = TR.panel_long(pn, fields=TR.KLINE_FIELDS).sort_values(["symbol", "date"])
    ref = long.copy()
    ref["fwd"] = (ref.groupby("symbol")["close"].shift(-HORIZON) / ref["close"] - 1.0)

    kline = TR.kline_long(pn, horizon=HORIZON)
    assert len(kline) == pn.n_dates * pn.n_symbols
    merged = kline[["date", "symbol", "fwd_ret"]].merge(
        ref[["date", "symbol", "fwd"]], on=["date", "symbol"], how="inner",
        validate="one_to_one")
    assert len(merged) == len(kline)
    np.testing.assert_allclose(merged["fwd_ret"].to_numpy(dtype=float),
                               merged["fwd"].to_numpy(dtype=float),
                               rtol=1e-12, atol=1e-15, equal_nan=True)

    assert bool(kline.groupby("symbol")["fwd_ret"].tail(HORIZON).isna().all())
    finite = int(np.isfinite(kline["fwd_ret"].to_numpy(dtype=float)).sum())
    assert finite == len(kline) - HORIZON * pn.n_symbols


def test_kline_long_forward_return_has_no_cross_symbol_bleed(pn: PanelData):
    """改动某标的某一天的收盘价，只允许影响该标的的**两行**前瞻收益。

    这是前瞻收益唯一可靠的独立复核方式：跨标的 ``shift``、多算一天、
    或把末尾填成 0，都会让"受影响行集合"大于这两行。
    """
    base = TR.kline_long(pn, horizon=HORIZON)
    sym = pn.symbols[3]
    move_at = pn.dates[200]

    fields = {k: pn.field(k).copy() for k in TR.KLINE_FIELDS}
    fields["close"].loc[move_at, sym] *= 2.0
    perturbed = PanelData(fields, forward_periods=(HORIZON,), name="perturbed")
    got = TR.kline_long(perturbed, horizon=HORIZON)

    changed = (base["fwd_ret"].fillna(-999.0) != got["fwd_ret"].fillna(-999.0))
    rows = base.loc[changed.to_numpy(), ["symbol", "date"]]
    i = list(pn.dates).index(move_at)
    expected = {(sym, move_at), (sym, pn.dates[i - HORIZON])}
    assert {(s, d) for s, d in zip(rows["symbol"], rows["date"])} == expected


# --------------------------------------------------------------------------
# 2. 统计显著性
# --------------------------------------------------------------------------
def test_ic_series_of_extracts_reports_and_skips_empties(reports):
    items = TR.ic_series_of(reports)
    assert [name for name, _ in items] == [r.name for r in reports]
    assert all(isinstance(ic, pd.Series) and len(ic) > 0 for _, ic in items)

    class _Rep:
        def __init__(self, name=None, detail=None):
            if name is not None:
                self.name = name
            self.detail = {} if detail is None else detail

    mixed = [_Rep(detail={"ic_series": pd.Series(dtype=float)}),   # 空序列
             _Rep(detail={"other": 1}),                            # 没有 IC 序列
             _Rep(detail={}),                                      # 没有 detail 内容
             _Rep(detail={"ic_series": [0.1, -0.2, 0.3]})]         # 无名字 → 自动命名
    got = TR.ic_series_of(mixed)
    assert [name for name, _ in got] == ["factor_1"]
    assert TR.ic_series_of([]) == []


def test_significance_check_multiplicity_scales_with_trials(ic_map):
    """选择校正必须随"试了多少次"单调收紧，且两道 p 值口径都要留痕。"""
    few = TR.significance_check(ic_map, n_trials=1, n_boot=400, seed=1)
    many = TR.significance_check(ic_map, n_trials=2000, n_boot=400, seed=1)

    assert few["n_trials"] == 1 and many["n_trials"] == 2000
    assert few["n_candidates"] == many["n_candidates"] == 2
    assert few["min_dates"] == 20, "报告层要按实际口径渲染样本量门槛"

    t_few, t_many = few["table"], many["table"]
    np.testing.assert_allclose(t_few["p_selection"], t_few["p_boot"])
    assert (t_many["p_selection"] >= t_many["p_boot"] - 1e-12).all(), \
        "Šidák 校正只会让 p 变大"
    assert (t_many["p_selection"] >= t_few["p_selection"] - 1e-12).all()

    crit_few = few["summary"]["selection"]["ic_crit"]
    crit_many = many["summary"]["selection"]["ic_crit"]
    assert crit_many > crit_few, f"门槛应随尝试次数上升：{crit_few:.4f} → {crit_many:.4f}"
    assert (t_many["ic_threshold"] == crit_many).all()

    assert len(few["table"]) == 2 and set(few["table"].columns) >= {
        "factor", "n", "ic", "icir", "p_boot", "p_neff", "p_selection",
        "q_value", "passed", "enough_samples", "ic_threshold"}


def test_significance_check_separates_signal_from_search_noise(ic_map):
    """只试 1 次时强信号通过四道门槛、均值为 0 的噪声不通过；
    试 2000 次后连强信号也被门槛挡下 —— 这正是多重性校正存在的意义。"""
    once = TR.significance_check(ic_map, n_trials=1, n_boot=400, seed=11)
    table = once["table"]
    assert list(table["factor"]) == ["强信号", "纯噪声"], "通过的候选应排在前面"
    assert bool(table["passed"].iloc[0]) and not bool(table["passed"].iloc[1])
    assert float(table["p_boot"].iloc[1]) == pytest.approx(1.0)
    assert once["summary"]["passed"] == 1
    assert once["warning"] is None, "有候选通过时不应给出过度拟合警告"

    swept = TR.significance_check(ic_map, n_trials=2000, n_boot=400, seed=11)
    assert swept["summary"]["passed"] == 0
    assert "结论" in str(swept["warning"])


def test_significance_check_survives_empty_and_shrunken_input():
    """没有候选、候选中途缺序列：都不许抛异常，空结果要能被界面直接读。"""
    empty = TR.significance_check([])
    assert empty["table"].empty and empty["n_candidates"] == 0
    assert empty["summary"]["n_tested"] == 0
    assert empty["warning"] == "没有任何候选因子可用于显著性检验。"
    assert empty["markdown"] == "（无候选）"

    tiny = TR.significance_check({"短序列": [0.1, 0.2]}, n_trials=3, n_boot=100)
    assert len(tiny["table"]) == 1
    row = tiny["table"].iloc[0]
    assert row["n"] == 2 and not bool(row["enough_samples"]) and not bool(row["passed"])
    assert not bool(row["fdr_rejected"]), "样本不足的候选永远不许被判显著"


def test_significance_warning_explains_undefined_threshold(pn: PanelData, reports):
    """只有一个候选时"候选间的离散度"没有定义：说明原因，不许把 nan 打给用户看。"""
    rng = np.random.default_rng(7)
    alone = {"独苗": pd.Series(rng.normal(0.01, 0.08, size=250))}
    sig = TR.significance_check(alone, n_trials=50, n_boot=200, seed=5)
    assert sig["summary"]["passed"] == 0
    assert not np.isfinite(sig["summary"]["selection"]["ic_crit"])
    assert "nan" not in str(sig["warning"]).lower()
    assert "门槛未定义" in str(sig["warning"])

    md = RP.render_markdown(_report_input(pn, reports, significance=sig))
    assert "门槛未定义" in md and "nan" not in md.lower()


def test_significance_for_search_uses_evaluated_count(stub_search, reports, ic_map):
    """尝试次数取 ``search.n_evaluated``，而不是候选个数。"""
    sig = TR.significance_for_search(stub_search, top_k=2, n_boot=300, seed=3)
    assert sig["n_trials"] == stub_search.n_evaluated == 137
    assert sig["top_k"] == 2 and sig["n_candidates"] == 2
    assert [str(x) for x in sig["table"]["factor"]] == [r.name for r in reports]
    assert sig["stop_reason"] == "stub-budget"

    blind = TR.significance_for_search(_StubSearch([], n_evaluated=99), top_k=5)
    assert blind["n_candidates"] == 0 and blind["table"].empty
    assert blind["n_trials"] == 99
    assert len(ic_map) == 2


# --------------------------------------------------------------------------
# 3. 选股域
# --------------------------------------------------------------------------
def test_quantile_domains_use_panel_own_quantiles(pn: PanelData, specs):
    """三档门槛必须等于**面板自身**成交额分位，而不是引擎的绝对门槛。"""
    assert len(specs) == 3
    amount = TR.panel_long(pn, fields=("close", "volume", "amount"))["amount"]
    for spec, q in zip(specs, (0.20, 0.50, 0.80)):
        assert spec.min_amount == pytest.approx(float(amount.quantile(q)))
    assert [s.min_history for s in specs] == [20, 60, 120], "越严的域要求越长的历史"
    assert [s.min_amount for s in specs] == sorted(s.min_amount for s in specs)
    assert all(s.top_n == 20 for s in specs)
    assert len({s.label for s in specs}) == 3, "三档标签必须唯一（界面按标签对照）"

    preset = TR._engine("universe").preset_domains(20)
    assert preset[2].min_amount == pytest.approx(1.0e8), \
        "引擎绝对门槛未变；两者量纲不同是 quantile_domains 存在的理由"


def test_universe_check_defaults_to_middle_domain(pn: PanelData, specs):
    uni = TR.universe_check(pn, horizon=HORIZON, keep_flags=False)
    assert uni["ok"] and "flags" not in uni
    assert uni["spec"]["label"] == specs[1].label
    assert uni["spec"]["min_amount"] == pytest.approx(specs[1].min_amount)
    assert uni["n_rows"] == pn.n_dates * pn.n_symbols
    assert uni["n_dates"] == pn.n_dates
    assert uni["n_tradable"] + sum(uni["reasons"].values()) == uni["n_rows"], \
        "每条记录要么可交易，要么恰好有一个剔除原因"
    assert set(uni["reasons"]) == set(TR._engine("universe").REASONS)
    assert 0.0 < uni["coverage_pct"] < 100.0
    assert uni["min_per_day"] <= uni["avg_per_day"] <= uni["max_per_day"]


def test_universe_check_keeps_row_level_flags_on_demand(pn: PanelData, specs):
    out = TR.universe_check(pn, spec=specs[0], horizon=HORIZON, keep_flags=True)
    flags = out["flags"]
    assert len(flags) == pn.n_dates * pn.n_symbols
    assert {"date", "symbol", "tradable", "reason", "amount", "hist_days",
            "close", "pct_chg"} <= set(flags.columns)
    assert flags["tradable"].dtype == bool
    assert (flags.loc[flags["tradable"], "reason"] == "").all()
    assert flags["hist_days"].max() == pn.n_dates, "历史天数是逐标的滚动计数"
    assert flags["amount"].notna().all()


def test_domain_check_strictness_and_factor_input_forms(pn: PanelData, specs, domains):
    assert list(domains["domain"]) == [s.label for s in specs]
    assert len(domains) == 3
    assert domains["coverage_pct"].is_monotonic_decreasing
    assert domains["avg_per_day"].is_monotonic_decreasing
    assert domains["ic"].notna().all() and (domains["n_ic"] > 0).all()
    assert domains["t_stat"].notna().all()
    assert domains["avg_turnover"].between(0.0, 1.0).all(), \
        "单边换手率 = Σ|Δw|/2，全部换一遍才是 1.0"

    wide = pn.field("momentum")
    series = (PanelData._to_long({"factor": wide})
              .set_index(["date", "symbol"])["factor"])
    again = TR.domain_check(pn, series, specs=specs, horizon=HORIZON,
                            min_count=5, top_n=20)
    np.testing.assert_allclose(again["ic"].to_numpy(dtype=float),
                               domains["ic"].to_numpy(dtype=float),
                               rtol=1e-9, atol=1e-12)
    with pytest.raises(TypeError):
        TR.domain_check(pn, pn.field("momentum").to_numpy())


# --------------------------------------------------------------------------
# 4. 分层多尺度挖掘
# --------------------------------------------------------------------------
def test_gp_ts_corr_eval_is_groupwise_and_warning_free(pn: PanelData):
    """GP 求值路径的 ``ts_corr`` 必须逐标的滚动相关，且不许丢出弃用告警。

    ``pytest.ini`` 里 ``filterwarnings = error`` 会把 pandas 的 GroupBy.apply
    弃用告警直接升级成失败 —— 这条链路（GP 求值）此前只被 ``triage`` 打通，
    正是它把 ``groupby(...).apply`` 的隐患暴露出来的，所以回归钉在这里。
    """
    from engine import genetic_enhanced as GE

    long = (TR.panel_long(pn, fields=TR.KLINE_FIELDS)
            .sort_values(["symbol", "date"]).reset_index(drop=True))
    expr = ("ts_corr", ("col", "close"), ("col", "volume"), ("const", 20))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = GE.eval_expr(expr, long)
    assert got.index.equals(long.index)

    # 恒等式自查：同一个序列自相关恒为 1、反号恒为 −1（与实现无关的独立口径）
    same = GE.eval_expr(("ts_corr", ("col", "close"), ("col", "close"),
                         ("const", 20)), long)
    flip = GE.eval_expr(("ts_corr", ("col", "close"),
                         ("neg", ("col", "close")), ("const", 20)), long)
    ok = np.isfinite(same.to_numpy(dtype=float)) & np.isfinite(flip.to_numpy(dtype=float))
    assert int(ok.sum()) == len(long) - (GE._min_periods(20, 10) - 1) * pn.n_symbols, \
        "预热期只该按 min_periods 逐标的扣，不该整段作废"
    np.testing.assert_allclose(same.to_numpy(dtype=float)[ok], 1.0, atol=1e-8)
    np.testing.assert_allclose(flip.to_numpy(dtype=float)[ok], -1.0, atol=1e-8)

    # 任取一只标的的最后一个窗口手算皮尔逊相关，逐位对齐
    sym = pn.symbols[0]
    piece = long.loc[long["symbol"] == sym]
    c = piece["close"].to_numpy(dtype=float)[-20:]
    v = piece["volume"].to_numpy(dtype=float)[-20:]
    assert float(got.loc[piece.index[-1]]) == pytest.approx(
        float(np.corrcoef(c, v)[0, 1]), abs=1e-10)
    assert float(got.loc[piece.index[0]]) != float(got.loc[piece.index[0]]), \
        "每个标的的最前面几期没有足够样本，应为 NaN 而不是别的标的的值"


def test_multiscale_mine_reports_budget_honestly(pn: PanelData, multiscale):
    ms = multiscale
    res = ms["resource"]
    assert ms["horizon"] == HORIZON
    assert ms["n_symbols"] == pn.n_symbols and ms["n_dates_panel"] == pn.n_dates
    assert ms["n_intervals"] == 4 and ms["n_select"] == 1

    # 资源口径：粗尺度 = 种群 × 代数；细尺度 = 种群 × 代数 × 被选中区间；
    # 暴力基准 = 细尺度种群 × 代数 × 全部区间。
    assert res["coarse_evals"] == 10 * 2
    assert res["total_intervals"] == 4
    assert res["selected_intervals"] == sum(1 for s in ms["intervals"] if s["selected"])
    assert res["selected_intervals"] <= ms["n_select"]
    assert res["brute_force_evals"] == 8 * 2 * 4
    assert res["fine_evals"] == 8 * 2 * res["selected_intervals"]
    used = res["coarse_evals"] + res["fine_evals"]
    assert res["brute_force_evals"] > used, "只有少数区间加密才会省预算"
    assert res["speedup"] == pytest.approx(res["brute_force_evals"] / used)
    assert res["eval_saving"] == pytest.approx(1.0 - 1.0 / res["speedup"])
    assert ms["total_evals"] > 0, "真实去重求值次数要单独给出"

    last = ms["intervals"][-1]
    assert not last["selected"] and not last["eligible"], \
        "末段没有下一段可比，d_H/MSD 为 NaN → 不可选（不许当 0 分参与竞争）"
    assert sum(s["n_dates"] for s in ms["intervals"]) == ms["train_dates"], \
        "区间是对训练段的无重叠划分"
    assert ms["train_dates"] + ms["test_dates"] == pn.n_dates
    assert ms["test_dates"] > 0, "样本外段必须留出来，否则区间选择会吃掉测试信息"


def test_multiscale_candidates_are_executable_and_json_safe(multiscale):
    ms = multiscale
    cands = ms["candidates"]
    assert cands, "被选中的区间应至少产出候选"
    for c in cands:
        assert {"expr", "code", "interval", "ic_interval", "ic_full", "ic_train",
                "ic_test", "fitness", "features", "family"} <= set(c)
        assert "def alpha_factor(df)" in c["code"]
        assert isinstance(c["features"], dict) and "depth" in c["features"]
        assert c["ic_full"] is not None
    # 界面与载荷直接消费这份结果：非有限值必须已折成 None，不许留裸 NaN
    # （裸 NaN 不是合法 JSON，前端反序列化会直接失败）
    text = json.dumps(ms, ensure_ascii=False, allow_nan=False)
    assert "def alpha_factor(df)" in text
    assert float(ms["terminal_weight"]) == pytest.approx(0.35)
    assert ms["intervals"][-1]["d_h"] is None and ms["intervals"][-1]["msd"] is None


def test_multiscale_mine_is_reproducible(pn: PanelData, multiscale):
    from engine.multiscale_gp import ScaleSpec

    levels = (ScaleSpec(name="coarse", freq="M", generations=2, pop_size=10, elite=3),
              ScaleSpec(name="fine", freq="D", generations=2, pop_size=8, elite=3))
    again = TR.multiscale_mine(pn, horizon=HORIZON, n_intervals=4, n_select=1,
                               seed=7, terminal_weight=0.35, levels=levels)
    assert [c["expr"] for c in again["candidates"]] == \
           [c["expr"] for c in multiscale["candidates"]], "同 seed 同配置必须可复现"
    assert [s["selected"] for s in again["intervals"]] == \
           [s["selected"] for s in multiscale["intervals"]]
    assert again["resource"]["fine_evals"] == multiscale["resource"]["fine_evals"]


def test_multiscale_fold2_refines_within_selected_intervals(pn: PanelData):
    """多折细化（文献 §4.1.2）：第二折只在已选父区间内部挑最差子区间加密。

    预算口径要如实入账（fold2_evals 单列，暴力基准同步上抬）；替换只发生在
    子区间 IC 确有提升时（ic_before 记录被替换前的折叠 1 水平）；载荷仍可
    严格 JSON 化（无裸 NaN）。
    """
    from engine.multiscale_gp import ScaleSpec

    levels = (ScaleSpec(name="coarse", freq="M", generations=2, pop_size=10, elite=3),
              ScaleSpec(name="fine", freq="D", generations=2, pop_size=8, elite=3),
              ScaleSpec(name="fine2", freq="D", generations=2, pop_size=4, elite=2))
    res = TR.multiscale_mine(pn, horizon=HORIZON, n_intervals=4, n_select=2,
                             n_folds=2, seed=7, terminal_weight=0.35, levels=levels)
    assert res["n_folds"] == 2

    r = res["resource"]
    assert r["fold2_evals"] == 4 * 2 * len(res["candidates"])
    assert r["brute_force_evals"] == 8 * 2 * 4 + 4 * 2 * 2 * len(res["candidates"])
    used = r["coarse_evals"] + r["fine_evals"] + r["fold2_evals"]
    assert r["speedup"] == pytest.approx(r["brute_force_evals"] / used)

    for c in res["candidates"]:
        if "refined2" in c:
            assert c["ic_interval"] > c["refined2"]["ic_before"], \
                "只有子区间 IC 真的提升了才允许替换"
            assert c["refined2"]["sub_index"] in (0, 1)
    assert json.dumps(res, ensure_ascii=False, allow_nan=False), "载荷不能含裸 NaN"


# --------------------------------------------------------------------------
# 5. 一次跑齐：三类缺项各自跳过
# --------------------------------------------------------------------------
def test_acceptance_runs_all_three_sections(acceptance, stub_search):
    acc = acceptance
    assert acc["skipped"] == {"multiscale": "已关闭分层多尺度挖掘"}
    assert acc["significance"]["n_trials"] == stub_search.n_evaluated
    assert len(acc["domains"]) == 3
    assert acc["universe"]["ok"] and "flags" not in acc["universe"], \
        "总入口不回传逐行标签（面板量级，会撑爆报告 JSON）"


def test_acceptance_skips_what_it_cannot_do(pn: PanelData, reports, stub_search):
    """缺什么少写什么，而不是给一个"看起来跑过了"的空结果。"""
    bare = TR.acceptance(pn, do_multiscale=False)
    assert set(bare["skipped"]) == {"significance", "domains", "multiscale"}
    assert "significance" not in bare and "domains" not in bare
    assert bare["skipped"]["significance"] == "没有带 IC 序列的候选因子"
    assert bare["skipped"]["domains"] == "未提供因子值"

    blank_search = _StubSearch([], n_evaluated=42)
    no_cand = TR.acceptance(pn, search=blank_search, factor=pn.field("momentum"),
                            n_boot=200, top_n=20, do_multiscale=False)
    assert no_cand["skipped"]["significance"] == "搜索没有产生可复核的候选报告"
    assert "domains" in no_cand, "显著性缺失不该连带把选股域也关掉"

    off = TR.acceptance(pn, reports=reports, factor=pn.field("momentum"),
                        n_boot=200, top_n=20, do_domain=False, do_multiscale=False)
    assert off["skipped"]["domains"] == "已关闭选股域对照"


def test_acceptance_prefers_search_candidates_over_handpicked(pn: PanelData,
                                                             reports, stub_search):
    """给了搜索就按搜索自己的候选做检验（多重性校正是对这次搜索收的税），
    手工挑的因子只作为回退路径。"""
    acc = TR.acceptance(pn, reports=reports, search=stub_search,
                        factor=pn.field("momentum"), n_boot=200, top_k=1,
                        top_n=20, do_domain=False, do_multiscale=False)
    assert list(acc["significance"]["table"]["factor"]) == [reports[0].name]

    fallback = TR.acceptance(pn, reports=reports, search=None,
                             factor=pn.field("momentum"), n_boot=200,
                             top_n=20, do_domain=False, do_multiscale=False)
    assert sorted(fallback["significance"]["table"]["factor"]) == \
        sorted(r.name for r in reports)


# --------------------------------------------------------------------------
# 6. 报告层：三项验收各自成章
# --------------------------------------------------------------------------
def _table_blocks(md: str):
    """把 Markdown 表格块切成逐行单元格（跳过 ``---`` 分隔行）。"""
    blocks, cur = [], []
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").replace(r"\|", "").split("|")]
            if not "".join(cells).replace("-", "").replace(":", ""):
                continue
            cur.append(cells)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    return blocks


def _report_input(pn, reports, **kw):
    return RP.ReportInput(panel=pn, title="验收闸门测试报告", reports=reports,
                          generated_at="2026-01-01T00:00:00Z", **kw)


def test_report_renders_acceptance_sections(pn: PanelData, reports, specs,
                                            domains, universe, multiscale,
                                            acceptance, tmp_path):
    ri = _report_input(pn, reports,
                       significance=acceptance["significance"],
                       domains=domains, universe=universe, multiscale=multiscale)
    md = RP.render_markdown(ri)
    sig_at = md.index("## 统计显著性检验")
    dom_at = md.index("## 选股域对照")
    ms_at = md.index("## 分层多尺度挖掘")
    assert sig_at < md.index("## 因子 1") < dom_at < ms_at, \
        "显著性在因子清单之前（它决定这批候选值不值得看），域与多尺度在其后"

    assert "137" in md, "多重性校正的分母必须出现在报告里"
    assert "结论：**结论**" not in md, "引擎自带前缀与报告标签不许叠加"
    assert "**结论**：" in md
    assert f"{specs[1].min_amount / 1e4:.0f} 万" in md, "成交额按万元渲染"
    assert f"{int(specs[2].min_amount)}" not in md, "绝对额（元）不许裸奔进文档"
    assert RP._expr_brief(multiscale["candidates"][0]["expr"]) in md
    assert "nan" not in md.lower(), "缺失值一律渲染为 —"

    for cells in _table_blocks(md):
        assert all(len(r) == len(cells[0]) for r in cells), \
            f"表格列数与表头不一致：{cells}"

    paths = RP.write_report(os.path.join(str(tmp_path), "r.md"), ri)
    with open(paths["markdown"], encoding="utf-8") as fh:
        assert fh.read() == md


def test_report_omits_missing_sections(pn: PanelData, reports, domains):
    """缺项整段省略：不许渲染空表，也不许凭空出现章节标题。"""
    empty = RP.render_markdown(_report_input(pn, reports))
    for head in ("## 统计显著性检验", "## 选股域对照", "## 分层多尺度挖掘"):
        assert head not in empty, f"无输入时不该有 {head}"

    only_domain = RP.render_markdown(_report_input(pn, reports, domains=domains))
    assert "## 选股域对照" in only_domain
    assert "## 统计显著性检验" not in only_domain
    assert "## 分层多尺度挖掘" not in only_domain


def test_report_payload_carries_acceptance_without_row_flags(pn: PanelData, reports,
                                                             domains, universe,
                                                             multiscale, acceptance):
    ri = _report_input(pn, reports, significance=acceptance["significance"],
                       domains=domains, universe=universe, multiscale=multiscale)
    payload = RP.report_payload(ri)
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    assert payload["significance"]["n_trials"] == 137
    assert payload["significance"]["n_candidates"] == 2
    assert len(payload["domains"]) == 3
    assert payload["universe"]["ok"] and "spec" in payload["universe"]
    assert "flags" not in payload["universe"], "逐行标签必须被挡在载荷之外"
    assert payload["multiscale"]["resource"]["total_intervals"] == 4
    assert payload["multiscale"]["candidates"][0]["code"], "代码要能落到载荷里"
    assert "def alpha_factor(df)" in text


def test_expr_brief_only_truncates_display(multiscale):
    """表格里的表达式只展示结构；完整精度留在 JSON 载荷中。"""
    expr = multiscale["candidates"][0]["expr"]
    brief = RP._expr_brief(expr)
    assert not re.search(r"\d+\.\d{4,}", brief), brief
    assert len(brief) <= max(len(expr), 1)
    assert RP._expr_brief(None) == "—"
    raw = "(params (0.49964224371608046 0.10000000000000001))"
    assert RP._expr_brief(raw) == "(params (0.499 0.100))"


# --------------------------------------------------------------------------
# 7. 界面接入（静态契约：不导入 streamlit，也不渲染页面）
# --------------------------------------------------------------------------
def _ui_funcs():
    """解析 ``src/ui/app.py``，按函数名取回 AST 节点。"""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    with open(os.path.join(root, "src", "ui", "app.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename="src/ui/app.py")
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def test_ui_mining_page_is_wired_to_triage():
    """挖掘页必须真的把四个标签接上去，而且引用的 triage 接口都得存在。

    这条是**静态**契约（不导入 streamlit、不开页面）：模块里改个名、删个函数，
    界面要等用户点开标签那一刻才炸，静态检查能把这次爆炸提前到 `pytest`。
    """
    funcs = _ui_funcs()
    gp = funcs["render_gp_mining"]

    labels, calls = None, set()
    for node in ast.walk(gp):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "tabs"):
            labels = [e.value for e in node.args[0].elts]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.add(node.func.id)
    assert labels is not None and len(labels) == 4, "挖掘页是四标签结构"
    assert any("多尺度" in s for s in labels) and any("显著性" in s for s in labels) \
        and any("选股域" in s for s in labels), labels
    assert {"_gp_evolve_tab", "_gp_multiscale_tab", "_gp_significance_tab",
            "_gp_domain_tab"} <= calls, calls
    for name in ("_gp_multiscale_tab", "_gp_significance_tab", "_gp_domain_tab"):
        assert name in funcs, f"{name} 必须真的存在（不是只有个名字）"

    used = {n.attr for g in funcs.values() for n in ast.walk(g)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "TR"}
    assert used, "挖掘页应当通过 mining.triage 调用三个引擎模块"
    missing = sorted(name for name in used if not hasattr(TR, name))
    assert not missing, f"界面引用了 triage 里不存在的接口：{missing}"

    nav_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "src", "ui", "nav.py")
    with open(nav_path, encoding="utf-8") as fh:
        nav_text = fh.read()
    assert all(k in nav_text for k in ("多尺度", "显著性", "选股域")), \
        "目录里的条目描述要跟着一起更新，否则用户根本不知道这里有这三个入口"


# --------------------------------------------------------------------------
# 8. 批量演化页：数据源输入口 + GP 链路
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gp_kline(pn: PanelData) -> pd.DataFrame:
    """GP 用的行情长表（合成面板 → 长表，含 ``pct_chg``，全程不触网）。"""
    return TR.panel_long(pn, fields=TR.KLINE_FIELDS)


def _ui_names_in(funcs, names):
    """收集若干界面函数里出现过的属性名 / 变量名 / 字符串常量。"""
    out = set()
    for name in names:
        assert name in funcs, f"{name} 必须真的存在（不是只有个名字）"
        for node in ast.walk(funcs[name]):
            if isinstance(node, ast.Attribute):
                out.add(node.attr)
            elif isinstance(node, ast.Name):
                out.add(node.id)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                out.add(node.value)
    return out


def test_ui_gp_evolve_tab_has_the_input_it_promises():
    """演化页必须真的提供它承诺的那个输入口。

    踩过的坑：点「启动 GP 演化」只弹一句「请在下方输入股票代码或选择缓存数据」，
    可页面下方既没有股票代码输入框、也没有任何数据源选择控件 —— 提示指向一个并
    不存在的地方，用户唯一的出路是关掉页面。这里钉成静态契约：
    **能填代码 + 能选数据源 + 真的把这两样喂给演化**，缺一不可。
    """
    funcs = _ui_funcs()
    names = _ui_names_in(funcs, ("_gp_evolve_tab", "_gp_evolve_data_controls",
                                 "_gp_resolve_kline", "_gp_evolve_results",
                                 "_gp_mass_produce"))
    for widget in ("text_input", "selectbox", "number_input"):
        assert widget in names, f"演化页缺 {widget}：数据源与股票代码必须能选、能填"
    assert "gp_evo_codes" in names, "股票代码输入框要带 st.key，测试与回放都靠它定位"
    assert "resolve_market_panel" in names, "填进来的数据源/代码必须真的喂给取数入口"
    assert "EnhancedFactorEvolver" in names, "演化本体要真的被实例化"
    assert "evolve_clusters" in names, "簇驱动演化要真的被调用"
    assert "mass_produce_from_library" in names, "一键批量生产也要走同一套取数"


def test_ui_gp_resolve_wires_code_input_into_data_layer():
    """代码框的值必须真的走到取数入口。

    只放一个装饰性输入框比不放更糟：用户以为它生效了，实际取数与它无关。
    """
    funcs = _ui_funcs()
    calls = [n for n in ast.walk(funcs["_gp_resolve_kline"]) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "resolve_market_panel"]
    assert calls, "_gp_resolve_kline 必须调用 engine.factor_system.resolve_market_panel"
    kwargs = {k.arg for c in calls for k in c.keywords}
    assert {"source", "symbols", "n_symbols", "days"} <= kwargs, kwargs


def test_parse_symbols_normalises_user_input():
    """用户写法的归一：前后缀、全角标点、去重保序。"""
    from engine import factor_system as FS

    got = FS.parse_symbols("600519, 000001 600036.SH、sh600519；000002/600000|000858")
    assert got == ["600519", "000001", "600036", "000002", "600000", "000858"], got
    assert FS.parse_symbols(["SH600519", "600519", "000001.SZ"]) == ["600519", "000001"]
    assert FS.parse_symbols("  ") == [] and FS.parse_symbols(None) == []


def test_resolve_market_panel_passes_symbols_and_reports_missing(monkeypatch):
    """指定代码要原样（归一后）传到取数层，缺谁报谁。"""
    from engine import factor_system as FS

    stub = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-02", "2024-01-02"],
        "symbol": ["600519.SH", "SH000001", "000002"],
        "close": [1700.0, 12.0, 8.0],
    })
    seen = {}

    def fake_panel(**kw):
        seen.clear()
        seen.update(kw)
        return stub, {"source": "stub"}

    monkeypatch.setattr(FS, "load_market_panel", fake_panel)

    kline, meta = FS.resolve_market_panel(source="cache", symbols=[" sh600519 ", "000001"],
                                          n_symbols=7, days=100)
    assert seen["symbols"] == ["600519", "000001"], seen        # 归一后原样下传
    assert seen["prefer_cache"] is True and seen["days"] == 100, seen
    assert "missing" not in meta and "message" not in meta, meta

    kline, meta = FS.resolve_market_panel(source="cache", symbols=["600519", "999999"])
    assert meta["missing"] == ["999999"] and "999999" in meta["message"], meta


def test_cached_panel_filters_by_requested_codes(monkeypatch):
    """整矿缓存按代码筛：写法不同也算命中，且不再被"取前 N 只"裁掉。"""
    from engine import factor_system as FS

    cached = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03"] * 3,
        "symbol": ["600519.SH", "600519.SH", "SH000001", "SH000001", "000002", "000002"],
        "close": [1700.0, 1710.0, 12.0, 12.1, 8.0, 8.2],
    })
    monkeypatch.setattr(FS, "_load_from_ore_cache", lambda cache_dir: cached)

    kline, meta = FS.load_market_panel(n_symbols=1, days=100, symbols=["600519", "000001"])
    assert sorted(kline["symbol"].unique()) == ["600519.SH", "SH000001"], kline["symbol"]
    assert {"date", "symbol", "close", "pct_chg"} <= set(kline.columns)

    kline, _ = FS.load_market_panel(n_symbols=2, days=100)
    assert kline["symbol"].nunique() == 2, "没点名时才按上限裁剪"


def test_resolve_market_panel_refuses_placeholder_panel_for_requested_codes(monkeypatch):
    """点名代码一个都没取到时必须交白卷，不许拿兜底合成数据冒充。

    这是最不能接受的失败方式：用户要的是 600519，界面却拿 S000001 这种占位标的
    跑完一轮并给出漂亮的 IC —— 结果查不出原因。
    """
    from engine import factor_system as FS

    placeholder = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-02"],
        "symbol": ["S000001", "S000002"],
        "close": [10.0, 20.0],
    })
    monkeypatch.setattr(FS, "load_market_panel",
                        lambda **kw: (placeholder, {"source": "合成数据（离线兜底）"}))

    kline, meta = FS.resolve_market_panel(source="cache", symbols=["600519"])
    assert kline.empty, "取不到点名代码时不许回落合成占位数据"
    assert meta["requested"] == ["600519"] and meta["missing"] == ["600519"], meta
    assert "600519" in meta["message"], meta

    kline, meta = FS.resolve_market_panel(source="cache")   # 没点名 → 兜底仍然可用
    assert not kline.empty and meta["source"] == "合成数据（离线兜底）"


def test_resolve_market_panel_synthetic_and_unknown_source():
    """合成面板（离线流程验证）要明说忽略了真实代码；未知来源直接报错。"""
    from engine import factor_system as FS

    kline, meta = FS.resolve_market_panel(source="synthetic", symbols=["600519"],
                                          n_symbols=6, days=90)
    assert not kline.empty and meta["n_symbols"] >= 6
    assert "600519" in meta["message"] and "占位" in meta["message"], meta

    with pytest.raises(ValueError):
        FS.resolve_market_panel(source="shenzhen")


def test_resolve_market_panel_offline_reads_shipped_dataset():
    """离线 Parquet 真的能出数据（不联网）——这是"无网也能挖掘"的兜底承诺。"""
    from engine import factor_system as FS

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "offline")
    if not os.path.isdir(root) or not any(f.endswith(".parquet") for f in os.listdir(root)):
        pytest.skip("离线数据集未随仓库分发")

    kline, meta = FS.resolve_market_panel(source="offline", n_symbols=5, days=120)
    assert not kline.empty and meta["source"].startswith("离线 Parquet"), meta
    assert set(TR.KLINE_FIELDS) | {"date", "symbol", "pct_chg"} <= set(kline.columns)

    code = str(sorted(kline["symbol"].unique())[0])
    got, meta2 = FS.resolve_market_panel(source="offline", symbols=[code, "999999"],
                                         n_symbols=5, days=120)
    assert code in set(got["symbol"]) and meta2["missing"] == ["999999"], meta2


def test_gp_fitness_matches_groupwise_apply_reference(gp_kline: pd.DataFrame):
    """``_fitness`` 必须与逐日 ``groupby.apply(corr)`` 逐点一致。

    这里换掉了原先的 ``groupby("date").apply(lambda g: ...)``：pandas ≥ 2.2 会把
    分组列一并交给回调并抛 ``FutureWarning``，而本仓库 pytest 把告警升级成错误 ——
    也就是说这条链路此前根本跑不亮测试。等价性必须钉住：逐日 IC 的分子分母一旦被
    "顺手优化" 错一点，适应度全变，演化方向也跟着变。
    """
    from agent.integration import get_library
    from engine.genetic_enhanced import EnhancedFactorEvolver, eval_expr

    evolver = EnhancedFactorEvolver(gp_kline, library=get_library(), seed=3)
    train = evolver.df
    expr = ("ts_zscore", ("col", "amount"), ("const", 20))
    panel = pd.DataFrame({
        "f": np.asarray(eval_expr(expr, train), dtype=float),
        "y": train["_fwd_ret"].to_numpy(dtype=float),
        "date": train["date"].to_numpy(),
    }).replace([np.inf, -np.inf], np.nan).dropna()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = panel.groupby("date").apply(
            lambda g: g["f"].corr(g["y"]) if g["f"].std() > 0 else np.nan).dropna()
    assert len(ref) > 50, "参考实现得有几天的截面，否则这个对照没有意义"
    assert evolver._fitness(expr, train) == pytest.approx(float(ref.mean()), rel=0, abs=1e-12)


def test_gp_evolve_clusters_warning_free_and_reproducible(gp_kline: pd.DataFrame):
    """演化链路在"告警即错误"下必须干净，且同种子可复现。

    复现性不是学术洁癖：候选表要能跨轮次对比、能进测试基线，靠的就是同参同种子
    给同一串结果。
    """
    from agent.integration import get_library
    from engine.genetic_enhanced import EnhancedFactorEvolver

    def run():
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            evolver = EnhancedFactorEvolver(gp_kline, library=get_library(), seed=7)
            out = evolver.evolve_clusters(generations=2, pop_per_cluster=8, top_k=5,
                                          auto_save=False)
        return evolver, out

    evolver, first = run()
    _, second = run()
    assert first and second, "这个小规模演化应当至少产出几个候选"
    keys = {"name", "code", "cluster", "category", "train_ic", "test_ic",
            "overfit_gap", "fitness"}
    assert keys <= set(first[0]), sorted(first[0])
    assert [(r["name"], r["train_ic"], r["overfit_gap"]) for r in first] == \
        [(r["name"], r["train_ic"], r["overfit_gap"]) for r in second], "同种子必须可复现"
    assert evolver.history and {"gen", "island", "best_ic"} <= set(evolver.history[0])
    json.dumps(first, ensure_ascii=False, allow_nan=False)   # 界面要直接喂给 st.dataframe
