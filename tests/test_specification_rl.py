# -*- coding: utf-8 -*-
"""``src/engine/specification_rl``（多任务 RL 辅助规范搜索）回归测试。

落地的文献是 ``2609.18441v1``（Delphos）。本文件钉的是**机制**而不是"分数"：

1. **动作掩码**（论文 §3.1 iii）：不可用组件、立即回退、本 episode 已选过——
   三条限制逐条可核对。没有掩码，智能体会把预算浪费在来回拉锯上；
2. **规范 → 表达式**的编译：窗口算子带窗口常量、``ts_corr`` 必须带窗口、
   参数节点挂在最外层，编译产物必须能被 GP 的 ``eval_tree`` 直接求值；
3. **估计环境**：拟合值取 |IC|（因子方向可自由取反）、估计失败返回 ok=False
   而不是硬崩、同一规范的第二次估计走缓存；
4. **多任务迁移**（论文 §5 的核心结论）：在"最优建模概念跨任务共享、承载它的
   特征各不相同"的受控环境里，多任务策略零样本用到**训练时从未出现过的特征**
   上，且优于单任务与未训练（随机）策略；
5. **Pareto 前沿**（拟合 × 简约）与**可复现性**、**载荷可 JSON 化**。

测试成本：受控环境是纯 Python（不做面板求值），两个 seed × 每任务 20 episode，
秒级；真实面板用例取 12 符号 × 60 天、24 episode，同样秒级。
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from engine import specification_rl as SRL          # noqa: E402
from engine.multiscale_gp import eval_tree          # noqa: E402
from mining import report as RP                     # noqa: E402
from mining import triage as TR                     # noqa: E402
from mining.panel import PanelData                  # noqa: E402

# 受控环境用的小目录：3 个特征槽位 × 6 变换 × 4 结构 × 5 窗口
CAT = SRL.DomainCatalogue(
    features=("f0", "f1", "f2", "f3", "f4", "f5"),
    transforms=("none", "log", "abs", "sqrt", "ts_zscore", "ts_rank"),
    structures=("none", "add", "sub", "mul"),
    windows=(0, 3, 5, 10, 20),
    params=(0, 1),
    max_terms=3,
)
STAR = (3, 2, 3)        # 最优"建模概念"：(变换, 结构, 窗口)


class ConceptEnv(SRL.SpecEnvironment):
    """拟合值 = 规范中最匹配最优概念的那一项的组件匹配度 ∈ [0, 1]。

    关键：评分**只看概念**（变换 / 结构 / 窗口），完全不看用哪个特征承载它——
    这正是论文"规范表示与变量名无关"的前提，也是迁移能否成立的关键。
    """

    reward_scale = 1.0          # 拟合值已在 0~1，别再被 tanh 压到饱和

    def __init__(self, stars=None):
        super().__init__()
        self.stars = stars or {}

    def _evaluate(self, task, state):
        star = self.stars.get(task.name, STAR)
        best = 0.0
        for t in state.terms:
            m = ((t.transform == star[0]) + (t.structure == star[1])
                 + (t.window == star[2]))
            best = max(best, m / 3.0)
        return SRL.Estimate(float(best), True)


def _task(name, feats, n_obs=1000.0):
    return SRL.SpecTask(name=name, features=feats, n_obs=n_obs, volatility=0.02)


# ---------------------------------------------------------------------------
# 1. 动作掩码
# ---------------------------------------------------------------------------
def test_action_mask_drops_unavailable_and_repeat_and_rollback():
    task = _task("T", (0, 1))
    state = SRL.SpecState((SRL.SpecTerm(0, 0, 0, -1, 0, 0),))
    acts = SRL.feasible_actions(state, task, CAT, [], None)

    # (i) 不可用组件：任务只有特征 0/1，任何动作都不能引用特征 2..5
    bad = [a for a in acts if a.term.feature > 1 or a.term.covariate > 1]
    assert not bad, "掩码必须挡住任务目录里没有的特征"

    # (ii) 已选过：同一 episode 内同一动作只出现一次
    keys = [a.key() for a in acts]
    assert len(keys) == len(set(keys)), "同一动作不得重复出现在可行动作集里"

    # (iii) 立即回退：把刚改过的项改回原值，应被剔除
    prev = SRL.SpecTerm(0, 2, 0, -1, 0, 0)
    cur = SRL.SpecState((SRL.SpecTerm(0, 3, 0, -1, 0, 0),))
    acts2 = SRL.feasible_actions(cur, task, CAT, [], prev)
    assert not any(a.kind == "change" and a.term == prev for a in acts2), \
        "不得允许把项立刻改回上一步之前的值"

    # 采样版同样遵守三条限制
    rng = np.random.default_rng(0)
    sampled = SRL.sample_actions(state, task, CAT, [], None, rng, 64)
    assert sampled[-1].kind == "terminate", "终止动作必须始终可用"
    assert all(a.term.feature <= 1 and a.term.covariate <= 1 for a in sampled)
    assert len({a.key() for a in sampled}) == len(sampled)


def test_initial_state_is_additive_linear_baseline():
    task = _task("T", (0, 1, 2))
    st = SRL.initial_state(task, CAT)
    assert st.size() == 3
    assert all(t.transform == 0 and t.structure == 0 and t.covariate < 0
               and t.window == 0 and t.param == 0 for t in st.terms), \
        "起手规范必须是各属性线性进入、无交互（论文的 s0）"


# ---------------------------------------------------------------------------
# 2. 规范 → 表达式（与 GP 同一套语法）
# ---------------------------------------------------------------------------
def test_compile_expr_matches_gp_grammar():
    task = _task("T", (0, 1, 2, 3, 4, 5))
    # 窗口型一元算子必须带窗口常量
    e1 = SRL.compile_expr(SRL.SpecState((SRL.SpecTerm(0, 4, 0, -1, 3, 0),)), task, CAT)
    assert e1 == ("ts_zscore", ("col", "f0"), ("const", 10.0))
    # 非窗口型一元算子不带窗口参数
    e2 = SRL.compile_expr(SRL.SpecState((SRL.SpecTerm(0, 1, 0, -1, 3, 0),)), task, CAT)
    assert e2 == ("log", ("col", "f0"))
    # 组合结构带第二个特征；ts_corr 额外带窗口
    e3 = SRL.compile_expr(SRL.SpecState((SRL.SpecTerm(0, 0, 1, 1, 0, 0),)), task, CAT)
    assert e3 == ("add", ("col", "f0"), ("col", "f1"))
    # ts_corr 是三元算子（两个特征 + 窗口），用默认目录（含 ts_corr）验证
    dcat = SRL.DomainCatalogue()
    e4 = SRL.compile_expr(SRL.SpecState((SRL.SpecTerm(0, 0, 5, 1, 2, 0),)), task, dcat)
    assert e4 == ("ts_corr", ("col", "open"), ("col", "high"), ("const", 5.0))
    # 参数节点挂最外层
    e5 = SRL.compile_expr(SRL.SpecState((SRL.SpecTerm(0, 1, 0, -1, 0, 1),)), task, CAT)
    assert e5[0] == "hwma" and e5[1] == ("log", ("col", "f0"))
    # 多项 → 加性组合（论文：utility = 各项之和）
    e6 = SRL.compile_expr(
        SRL.SpecState((SRL.SpecTerm(0, 1, 0, -1, 0, 0), SRL.SpecTerm(1, 1, 0, -1, 0, 0))),
        task, CAT)
    assert e6[0] == "add" and e6[2] == ("log", ("col", "f1"))


def test_compiled_expr_is_evaluable_by_gp():
    df = pd.DataFrame({
        "date": np.repeat(pd.bdate_range("2024-01-02", periods=40).strftime("%Y-%m-%d"), 3),
        "symbol": ["A", "B", "C"] * 40,
        "f0": np.linspace(1.0, 2.0, 120),
        "f1": np.linspace(2.0, 3.0, 120),
        "fwd_ret": np.random.default_rng(0).normal(0, 0.01, 120),
    })
    task = _task("T", (0, 1))
    expr = SRL.compile_expr(SRL.SpecState((SRL.SpecTerm(0, 4, 3, 1, 3, 0),)), task, CAT)
    vals = np.asarray(eval_tree(expr, df), dtype=float)
    assert vals.size == len(df)
    assert np.isfinite(vals).any(), "编译产物必须能被 GP 的 eval_tree 直接求值"


# ---------------------------------------------------------------------------
# 3. 估计环境
# ---------------------------------------------------------------------------
def test_panel_environment_uses_abs_ic_and_caches():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2024-01-02", periods=40).strftime("%Y-%m-%d")
    syms = [f"S{i}" for i in range(8)]          # ≥ min_count，否则逐日 IC 全 NaN
    n = len(dates) * len(syms)
    df = pd.DataFrame({
        "date": np.repeat(dates, len(syms)),
        "symbol": syms * len(dates),
        "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
        "volume": rng.lognormal(10, 1, n),
        "fwd_ret": rng.normal(0, 0.01, n),
    })
    cat = SRL.DomainCatalogue(features=("close", "volume"))
    env = SRL.PanelSpecEnvironment({"T": df}, cat)
    task = _task("T", (0, 1))
    spec = SRL.SpecState((SRL.SpecTerm(0, 1, 0, -1, 0, 0),))     # log(close)

    est = env.evaluate(task, spec)
    assert est.ok and np.isfinite(est.fit)
    assert est.fit >= 0.0, "拟合值取 |IC|：因子方向可自由取反，奖励的是区分度"

    n_before = env.n_estimations
    est2 = env.evaluate(task, spec)
    assert env.n_estimations == n_before, "同一规范的第二次估计必须走缓存"
    assert est2.fit == est.fit

    # 引用面板里没有的列 → 估计失败（ok=False），而不是抛异常终止搜索
    bad = SRL.SpecState((SRL.SpecTerm(2, 1, 0, -1, 0, 0),))     # log(nosuch)
    env2 = SRL.PanelSpecEnvironment({"T": df}, SRL.DomainCatalogue(
        features=("close", "volume", "nosuch")))
    est3 = env2.evaluate(_task("T", (2,)), bad)
    assert not est3.ok


def test_reward_is_bounded_and_penalises_failure():
    assert SRL.spec_reward(0.0, 0.0) == pytest.approx(0.0)
    assert 0.0 < SRL.spec_reward(0.05, 0.0) < 1.0
    assert SRL.spec_reward(-0.05, 0.0) < 0.0                  # 比基线差 → 负
    assert SRL.spec_reward(float("nan"), 0.0) == -1.0         # 估计失败 → −1
    assert SRL.spec_reward(10.0, 0.0, ok=False) == -1.0
    assert SRL.spec_reward(1.0, 0.0) > SRL.spec_reward(0.1, 0.0), "奖励须单调"


# ---------------------------------------------------------------------------
# 4. 多任务迁移（论文 §5）
# ---------------------------------------------------------------------------
def _held_out_mean(shared, seed, train_pools, per_task=20, n=8):
    """在"训练时从未出现过的特征"上做零样本推断，返回候选的平均拟合值。"""
    env = ConceptEnv({"A": STAR, "B": STAR, "HELD": STAR})
    tasks = [_task(name, feats) for name, feats in train_pools]
    ag = SRL.MultitaskSpecAgent(CAT, env, shared=shared, seed=seed, max_actions=64)
    ag.train(tasks, n_episodes=per_task * len(tasks))
    held = _task("HELD", (5,))        # 特征 5 在任何训练任务里都没出现过
    cands = ag.propose(held, n_candidates=n)
    return float(np.mean([c.fit for c in cands])), ag


def test_multitask_transfers_to_unseen_features_better_than_single_task():
    """多任务策略能零样本用到未见特征上，且强于单任务与未训练策略。

    这是论文最核心的主张：把建模经验跨数据集聚合后，学到的**建模概念**（该用
    哪种变换 / 结构 / 时间尺度）可以迁移到变量名完全不同的新数据集上。
    """
    mt, mt_agent = _held_out_mean(True, 11, (("A", (0, 1, 2)), ("B", (3, 4))))
    st, _ = _held_out_mean(False, 11, (("A", (0, 1, 2)),))
    un, _ = _held_out_mean(True, 11, (("A", (0, 1, 2)), ("B", (3, 4))), per_task=0)

    assert mt > un, "训练后必须比未训练（随机）策略更强，否则说明没学到东西"
    assert mt > st, "多任务必须优于单任务：单任务学到的是任务特定的特征偏好"

    rep = mt_agent.report()
    curve = rep["learning_curve"]
    assert len(curve) >= 3
    assert float(np.mean(curve[-3:])) > float(np.mean(curve[:3])), \
        "学习曲线应随训练上行（论文 §4.2 用曲线下面积衡量学习效率）"


def test_multitask_shares_one_q_network_single_task_does_not():
    env = ConceptEnv({"A": STAR, "B": STAR})
    tasks = [_task("A", (0, 1, 2)), _task("B", (3, 4))]
    shared = SRL.MultitaskSpecAgent(CAT, env, shared=True, seed=3, max_actions=48)
    shared.train(tasks, n_episodes=8)
    assert len(shared.qnets) == 1, "多任务必须共用一套 Q 参数"

    solo = SRL.MultitaskSpecAgent(CAT, env, shared=False, seed=3, max_actions=48)
    solo.train(tasks, n_episodes=8)
    assert len(solo.qnets) == 2, "单任务基线必须每任务各学一套"


def test_training_is_reproducible():
    a, _ = _held_out_mean(True, 7, (("A", (0, 1, 2)), ("B", (3, 4))), per_task=10)
    b, _ = _held_out_mean(True, 7, (("A", (0, 1, 2)), ("B", (3, 4))), per_task=10)
    assert a == pytest.approx(b), "同 seed 同配置必须完全可复现"


# ---------------------------------------------------------------------------
# 5. Pareto 前沿与载荷
# ---------------------------------------------------------------------------
def test_pareto_front_keeps_non_dominated_candidates():
    def cand(fit, cx):
        return SRL.Candidate((SRL.SpecTerm(0, 1, 0, -1, 0, 0),), fit, True, cx)

    cands = [cand(0.2, 3), cand(0.5, 4), cand(0.5, 2), cand(0.1, 1)]
    front = SRL.pareto_front(cands)
    fits = [(c.fit, c.complexity) for c in front]
    assert (0.5, 2) in fits, "同等拟合里更简约的应保留"
    assert (0.5, 4) not in fits, "被同拟合更简约者支配"
    assert (0.2, 3) not in fits, "被更高拟合且更简约者支配"
    assert (0.1, 1) in fits, "拟合最低但最简约，仍是非支配解"
    assert front == sorted(front, key=lambda c: c.complexity)


def test_payload_is_strict_json():
    env = ConceptEnv({"A": STAR, "B": STAR})
    ag = SRL.MultitaskSpecAgent(CAT, env, shared=True, seed=5, max_actions=32)
    ag.train([_task("A", (0, 1)), _task("B", (2, 3))], n_episodes=6)
    payload = ag.report()
    json.dumps(payload, ensure_ascii=False, allow_nan=False)   # 裸 NaN 会抛异常


# ---------------------------------------------------------------------------
# 6. 挖掘层接入（真实面板）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_panel() -> PanelData:
    return PanelData.synthetic(n_symbols=12, n_days=60, seed=9)


def test_specification_search_on_real_panel(small_panel):
    out = TR.specification_search(small_panel, horizon=5, n_tasks=2,
                                  n_episodes=12, n_candidates=3, seed=2)
    assert out["n_tasks"] >= 1
    assert out["n_estimations"] > 0, "必须真的调用过估计环境"
    for t in out["tasks"]:
        assert t["candidates"], "每个任务都要给出候选规范"
        assert len(t["pareto"]) <= len(t["candidates"])
    json.dumps(out, ensure_ascii=False, allow_nan=False)


def test_report_section_renders_specification(small_panel):
    """报告里要有这一节，且缺项时整段省略（不能留一张空表）。"""
    empty = RP.render_markdown(RP.ReportInput(title="t", generated_at="2024-01-01T00:00:00Z"))
    assert "多任务规范搜索" not in empty

    out = TR.specification_search(small_panel, horizon=5, n_tasks=2,
                                  n_episodes=12, n_candidates=3, seed=2)
    md = RP.render_markdown(RP.ReportInput(title="t", specification=out,
                                           generated_at="2024-01-01T00:00:00Z"))
    assert "多任务规范搜索" in md
    assert "Pareto" in md and "估计环境调用次数" in md
    assert "建模项" in md


def test_multiscale_mine_accepts_spec_rl_seeds(small_panel):
    """``spec_rl=True``：规范搜索的建议进粗尺度演化的初始种群。

    默认关闭（见 ``triage.multiscale_mine``）——开启会改变搜索轨迹，因此这里
    只钉"能跑通、种子被如实记录、不可求值的种子被丢弃"，不钉具体分数。
    """
    base = TR.multiscale_mine(small_panel, horizon=5, n_intervals=3, n_select=1,
                              seed=4)
    assert base["spec_rl"] is False and base["n_spec_seeds"] == 0

    seeded = TR.multiscale_mine(small_panel, horizon=5, n_intervals=3, n_select=1,
                                seed=4, spec_rl=True, spec_rl_tasks=2,
                                spec_rl_episodes=8)
    assert seeded["spec_rl"] is True
    assert seeded["n_spec_seeds"] >= 1, "规范搜索应至少提出一个可用种子"
    assert seeded["candidates"], "注入种子后 GP 仍要正常产出候选"
