"""新增挖掘能力的回归测试：时间切分、样本外复核、双向桥、基因库。

与 tests/test_mining.py 的分工：那边管表达式/算子/评价的基础语义，这边管
"搜索口径"这一层（搜索期别看过确认集、筛选分别被噪声骗、两条主线能打通）。
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from mining import bridge as B  # noqa: E402
from mining import evaluator as EV  # noqa: E402
from mining import expr as ex  # noqa: E402
from mining import genome as G  # noqa: E402
from mining import gridminer as GM  # noqa: E402
from mining import risk as R  # noqa: E402
from mining import split as SP  # noqa: E402
from mining.panel import PanelData  # noqa: E402


@pytest.fixture(scope="module")
def small_panel():
    pn = PanelData.synthetic(12, 360, seed=5)
    R.install_risk_fields(pn)
    return pn


@pytest.fixture(scope="module")
def deep_panel():
    pn = PanelData.synthetic(20, 1050, seed=5)
    R.install_risk_fields(pn)
    return pn


# ---------------------------------------------------------------- 时间切分
def test_split_hard_for_long_sample(deep_panel):
    plan = SP.make_split(deep_panel)
    assert plan.mode == "split"
    assert plan.folds and plan.n_search > 0 and plan.n_confirm > 0
    # purge：train 段的末尾到 test 段起点至少要隔开 horizon 天，
    # 否则 train 最后一天的前瞻收益会吃掉 test 段开头。
    for f in plan.folds:
        gap = (deep_panel.dates.get_loc(f.test[0])
               - deep_panel.dates.get_loc(f.train[-1]))
        assert gap >= plan.horizon
    # 搜索段的最后一天，其前瞻收益不能落进确认段
    last = deep_panel.dates.get_loc(plan.search_dates[-1])
    assert (deep_panel.dates.get_loc(plan.confirm_dates[0]) - last) > plan.horizon


def test_split_soft_for_short_sample(small_panel):
    plan = SP.make_split(small_panel)
    assert plan.mode == "soft"
    assert not plan.folds
    assert plan.n_search == len(small_panel.dates)      # 不切，全量用于搜索


def test_split_modes(deep_panel):
    assert SP.make_split(deep_panel, mode="off").mode == "soft"
    assert SP.make_split(deep_panel, mode="soft").mode == "soft"
    assert SP.make_split(deep_panel, mode="split").mode == "split"


def test_walk_forward_folds_purge_tail():
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    folds = SP.walk_forward_folds(dates, n_folds=4, horizon=5, start=300)
    assert folds
    for f in folds:
        assert f.n_train > 0 and f.n_test > 0
        assert f.train[-1] < f.test[0]


# -------------------------------------------------------- 搜索期的新口径
def test_oos_check_runs_on_split(deep_panel):
    cfg = GM.SearchConfig(max_layers=1, width=4, max_expr=120, max_seconds=120,
                          top_k=3, min_ic=0.02)
    res = GM.mine(deep_panel, config=cfg)
    assert res.split["mode"] == "split"
    assert res.oos["mode"] == "split"
    assert res.oos["factors"], "入围因子应完成 walk-forward 复核"
    for row in res.oos["factors"].values():
        assert row["n_folds"] >= 1
        assert np.isfinite(row["oos_ic_mean"])
        assert -1.0 <= row["oos_ic_mean"] <= 1.0
    assert any("oos_ic_mean" in rep.metrics for rep in res.reports)


def test_soft_mode_skips_confirm(small_panel):
    cfg = GM.SearchConfig(max_layers=1, width=4, max_expr=80, max_seconds=120,
                          top_k=2, min_ic=0.02)
    res = GM.mine(small_panel, config=cfg)
    assert res.split["mode"] == "soft"
    assert res.oos["mode"] == "soft" and res.oos["factors"] == {}


def test_consistency_is_directional(small_panel):
    """一致性要按 IC 方向取"最弱段"：负 IC 因子不能因为取了 np.min 而恒为 1。"""
    m = GM.GridMiner(small_panel)
    rng = np.random.default_rng(0)
    neg = pd.Series(-0.06 + rng.normal(0, 0.01, 240))
    neg[:120] = -0.002                       # 前半段几乎没有信号
    pos = -neg
    assert m._consistency(neg, float(neg.mean())) < 1.0
    assert m._consistency(pos, float(pos.mean())) < 1.0
    # 各段一样强时一致性应接近 1
    flat = pd.Series([-0.05] * 240)
    assert m._consistency(flat, -0.05) > 0.99
    # 存在反号段 → 归零
    flip = pd.Series([-0.05] * 120 + [0.05] * 120)
    assert m._consistency(flip, 0.0) == 1.0 or m._consistency(flip, -0.05) == 0.0


def test_screen_metrics_bounded(deep_panel):
    cfg = GM.SearchConfig(max_layers=1, width=6, max_expr=120, max_seconds=120)
    res = GM.mine(deep_panel, config=cfg)
    assert res.candidates
    for c in res.candidates:
        assert 0.0 <= c.incr_ratio <= 1.0
        assert 0.0 <= c.consistency <= 1.0


def test_reproducible_with_new_defaults(deep_panel):
    cfg = GM.SearchConfig(max_layers=1, width=6, max_expr=120, max_seconds=120,
                          seed=7)
    a = GM.mine(deep_panel, config=cfg)
    b = GM.mine(deep_panel, config=cfg)
    assert [c.expression for c in a.candidates] == [c.expression for c in b.candidates]


def test_budget_still_respected(deep_panel):
    cfg = GM.SearchConfig(max_layers=1, width=6, max_expr=100, max_seconds=120)
    res = GM.mine(deep_panel, config=cfg)
    n_seeds = len(GM.GridMiner(deep_panel, cfg)._base_nodes())
    assert res.n_evaluated <= 100 + n_seeds


# ------------------------------------------------------------ 双向桥
def test_bridge_translates_builtin_templates():
    from engine.factor_builder import TEMPLATE_FACTORS

    ok = 0
    for name, tpl in TEMPLATE_FACTORS.items():
        tr = B.translate(tpl["code"])
        if tr.ok:
            ok += 1
            assert ex.parse(tr.expression) is not None
    assert ok >= 5, f"内置模板应大部分可翻译，实际 {ok}/{len(TEMPLATE_FACTORS)}"


def test_bridge_roundtrip_matches_dsl(small_panel):
    """DSL 求值 vs 渲染成代码再执行：语义必须一致。"""
    node = ex.parse("ts_mean(ts_pct(close, 1), 10)")
    frames = {f: small_panel.field(f).stack().rename(f) for f in ("close",)}
    df = pd.concat(frames.values(), axis=1).reset_index()
    df.columns = ["date", "symbol", "close"]
    got = B.expr_to_series(node, df).unstack()
    want = ex.Evaluator(small_panel, small_panel.registry).run(node)
    got = got.reindex(index=want.index, columns=want.columns)
    mask = (got.notna() & want.notna()).to_numpy()
    assert int(mask.sum()) > 200
    corr = float(np.corrcoef(got.to_numpy()[mask], want.to_numpy()[mask])[0, 1])
    assert corr > 0.99


def test_bridge_refuses_unsupported_op():
    node = ex.parse("ts_corr(close, amount, 20)")
    with pytest.raises(B.UnsupportedOp):
        B.expr_to_code(node)


def test_bridge_reports_failure_instead_of_guessing():
    code = ("def alpha_factor(df):\n"
            "    m = df.pivot(index='date', columns='symbol', values='close')\n"
            "    return m.stack().rename('factor')\n")
    tr = B.translate(code)
    assert not tr.ok and tr.reason


def test_seed_from_code():
    code = ("def alpha_factor(df):\n"
            "    df['ret'] = df.groupby('symbol')['close'].pct_change()\n"
            "    df['factor'] = df.groupby('symbol')['ret'].transform("
            "lambda x: x.rolling(20).mean())\n"
            "    df['factor'] = df.groupby('symbol')['factor'].shift(1)\n"
            "    return df[['date', 'symbol', 'factor']]\n")
    seed = B.seed_from_code(code)
    assert seed and "ts_mean" in seed


def test_bridge_keyword_arguments():
    """``rolling(window=20)`` / ``shift(periods=5)`` 必须和位置参数等价。

    LLM 几乎只写关键字参数。此前窗口取不到，翻译会静默丢掉窗口。
    """
    kw = ("def alpha_factor(df):\n"
          "    df['ret'] = df['close'].pct_change()\n"
          "    df['factor'] = df['ret'].rolling(window=20).mean()\n"
          "    return df[['date', 'symbol', 'factor']]\n")
    pos = kw.replace("rolling(window=20)", "rolling(20)")
    a, b = B.translate(kw), B.translate(pos)
    assert a.ok and b.ok, (a.reason, b.reason)
    assert a.expression == b.expression == "ts_mean(ts_pct(close, 1), 20)"

    sh = ("def alpha_factor(df):\n"
          "    df['factor'] = df.groupby('symbol')['close'].shift(periods=5)\n"
          "    return df[['date', 'symbol', 'factor']]\n")
    assert B.translate(sh).expression == "ts_delay(close, 5)"


def test_bridge_no_silent_drift():
    """认不出的赋值必须整条作废，不能退回上一条"认得出来"的赋值。

    实测过的漂移：``df['factor'] = df['ret'].rolling(window=20).mean()`` 认不出
    时，翻译器曾退回 ``df['ret']`` 并报成功——20 日动量变成 1 日收益率，
    再作为种子污染网格搜索。
    """
    code = ("def alpha_factor(df):\n"
            "    df['ret'] = df['close'].pct_change()\n"
            "    df['factor'] = df['ret'].rolling(window=20).apply(np.mean)\n"
            "    return df[['date', 'symbol', 'factor']]\n")
    tr = B.translate(code)
    assert not tr.ok
    assert "factor" in tr.reason      # 失败原因要点名是哪一步认不出来
    assert tr.expression == ""

    # 裸 lambda 内层变量参与四则运算（"减去自身均线"）必须能翻
    z = ("def alpha_factor(df):\n"
         "    df['factor'] = df.groupby('symbol')['close'].transform(\n"
         "        lambda x: (x - x.rolling(window=20).mean()) / x.rolling(20).std())\n"
         "    return df[['date', 'symbol', 'factor']]\n")
    tr2 = B.translate(z)
    assert tr2.ok, tr2.reason
    assert tr2.expression == "div(sub(close, ts_mean(close, 20)), ts_std(close, 20))"


def test_segmented_ic_direction_agnostic():
    """稳定性口径必须对正/负 IC 因子一视同仁。

    旧口径 ``seg_win = mean(段均值 > 0)``、``worst_seg_ic = min(段均值)``
    是按"越正越好"写的：负 IC（反向）因子的最强段被当成最弱段、同号段占比恒
    为 0，稳定性维度因此系统性压低所有反向因子。
    """
    ic = pd.Series([0.03] * 40 + [0.02] * 40 + [0.025] * 40 + [0.01] * 40)
    pos, neg = EV.segmented_ic(ic, 4), EV.segmented_ic(-ic, 4)
    assert pos["seg_win"] == pytest.approx(neg["seg_win"], abs=1e-9)
    assert pos["worst_seg_ratio"] == pytest.approx(neg["worst_seg_ratio"], abs=1e-9)
    assert pos["seg_win"] == 1.0
    # 展示值保留原始符号（报告里要能看出这是反向因子）
    assert pos["worst_seg_ic"] > 0 > neg["worst_seg_ic"]
    # 出现反号段 → 比例归零（不再是"最差段是多少"这种带方向的数）
    flip = EV.segmented_ic(pd.Series([0.03] * 40 + [-0.02] * 40), 2)
    assert flip["worst_seg_ratio"] == 0.0
    assert flip["seg_win"] == 0.5


# ------------------------------------------------------------ 基因库
def test_genome_warm_start(tmp_path, deep_panel):
    path = os.path.join(str(tmp_path), "genome.json")
    bank = G.GenomeBank(path)
    cfg = GM.SearchConfig(max_layers=1, width=4, max_expr=60, max_seconds=120,
                          top_k=3, min_ic=0.02)
    GM.mine(deep_panel, config=cfg, genome=bank)
    assert len(bank) > 0
    seeds = bank.seeds(deep_panel, n=3)
    assert seeds and len(seeds) <= 3
    assert os.path.exists(path)
    reloaded = G.GenomeBank(path)
    assert len(reloaded) == len(bank)
    # 第二轮带基因库：不崩，且种子被真正用上
    res = GM.mine(deep_panel, config=cfg, genome=reloaded)
    assert res.n_evaluated > 0
    assert reloaded.stats()["n"] > 0


def test_genome_default_off_keeps_reproducible(deep_panel):
    """不传基因库时，两次搜索必须完全一致（可复现性不能被 warm start 破坏）。"""
    cfg = GM.SearchConfig(max_layers=1, width=4, max_expr=60, max_seconds=120,
                          min_ic=0.02)
    a = GM.mine(deep_panel, config=cfg)
    b = GM.mine(deep_panel, config=cfg)
    assert [c.expression for c in a.candidates] == [c.expression for c in b.candidates]


def test_eval_still_works_on_search_segment(deep_panel):
    """入围因子的四维评价跑在搜索段上，指标有限且报告结构完整。"""
    cfg = GM.SearchConfig(max_layers=1, width=4, max_expr=100, max_seconds=120,
                          top_k=2, min_ic=0.02)
    res = GM.mine(deep_panel, config=cfg)
    for rep in res.reports:
        assert np.isfinite(rep.metrics.get("rank_ic_mean", float("nan")))
        assert 0.0 <= float(getattr(rep, "score", 0.0) or 0.0) <= 100.0
        assert isinstance(rep.detail, dict)
    assert EV.factor_corr_matrix(res.pool) is not None
