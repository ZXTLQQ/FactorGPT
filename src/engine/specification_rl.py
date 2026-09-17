"""多任务强化学习辅助的因子规范搜索（src/engine/specification_rl.py）。

落地文献 ``2609.18441v1``（*Multitask Reinforcement Learning for Assisting Choice
Model Specification*，Delphos）。该文把"离散选择模型的设定（specification）"写成
一个序贯决策问题：智能体对当前模型施加一串**建模动作**（加项 / 改项 / 终止），
由**估计环境**反馈"拟合、收敛、行为合理性"，从而学会"哪些建模决策倾向于产出
好模型"。其关键洞察与因子挖掘**同构**：

    不要为每个数据集从零开始试错。建模经验是可迁移的——
    把规范表示成"与变量名无关"的建模项集合，用同一个策略在多个任务上训练，
    学到的决策就能零样本用到没见过的新数据集上。

对应关系（本模块的实现）：

======================================  ==================================================
论文机制                                    本模块
======================================  ==================================================
选择模型设定（utility specification）        因子规范：一组"建模项"构成的表达式
建模项 (k, t, g, v)                          ``SpecTerm``：(特征, 变换, 组合结构, 协变量特征)
域目录 C = {K, T, G, V}                      ``DomainCatalogue``：特征 / 变换 / 结构 / 窗口 / 参数化
任务 τ（一个数据集）                          ``SpecTask``：一个时间区间 / 一个股票域 / 一个 horizon
估计环境（Apollo 估计 + 拟合反馈）           ``SpecEnvironment``：编译 → 求值 →  panel IC
状态 s_e = {x_l}（项集合，变长无序）         ``SpecState``：term 元组；DeepSet 均值聚合
DeepSet-Q（φ 嵌入 + ρ 置换不变聚合）        ``DeepSetQ``：项 one-hot → mean pooling → 线性 Q
任务上下文 x_τ（各组件可用性）               ``SpecTask.context``：可用性多重热编码 + 任务统计量
动作 a = add / change / terminate            ``SpecAction``：同构，附带**动作掩码**
动作掩码（不可用 / 回退 / 已选）             ``feasible_actions``：三条限制逐一落地
奖励 tanh((LL − LL0)/N_obs)                  ``spec_reward``：``tanh((IC − IC0)/scale)``，失败 −1
共享经验回放 + 平衡小批量                    ``ReplayBuffer``：按任务分层，各任务等量取样
目标网络 Q(·;θ⁻)                             ``DeepSetQ.target``：每 ``target_sync`` 步同步
ϵ-greedy 逐步衰减                            训练循环中线性衰减
Pareto 前沿（拟合 × 简约）                   ``pareto_front``
推断：零样本用于未见数据集                   ``MultitaskSpecAgent.propose(task)``
======================================  ==================================================

**与论文的两处刻意差异**（写在这里以免被当成"照搬"）：

1. 论文用深度 Q 网络（神经网络 + 经验回放 + 目标网络）。本模块用**线性**实现的
   DeepSet-Q：项嵌入 → 置换不变聚合 → 与任务上下文、动作嵌入拼接 → 线性打分。
   置换不变性、任务条件化、目标网络、平衡回放这些**让迁移成立的机制**全部保留，
   去掉的是非线性拟合能力——本项目不引入深度学习栈，且任务规模（几十个任务、
   几千次估计）下线性近似更稳、更可解释、可单测。
2. 论文的动作是 ``add(k,t,g,v)`` 一次成型，可行动作数随目录规模组合爆炸（本项目
   默认目录下约 2 万个）。因此智能体实际走的是 :func:`sample_actions`——按合法
   分布直接采样 ``max_actions`` 个动作，**三条掩码限制在采样与接纳时分别落地**，
   代价是探索不再是严格的全空间贪心。（完整枚举版 :func:`feasible_actions` 保留
   给小目录与测试用。）
3. 论文用 TD(0) + 目标网络。本模块默认用**折扣回报（reward-to-go）**做回归目标：
   轨迹只有 ``max_steps`` 步，而早期探索阶段多数 episode 都是被步数截断的，一步
   TD 在这么短的轨迹上传不回信号，线性 Q 上的 ``max`` 又会系统性高估——实测表现
   是"训练后反而不如未训练"。``bootstrap_truncated=True`` 可切回论文的目标网络
   bootstrap（实测 held-out 迁移均分 0.75 → 0.49，故默认关闭）。

迁移效果（受控合成环境，最优"建模概念"在各任务间共享、承载它的特征各不相同，
held-out 任务使用训练时从未出现过的特征；各任务 episode 数相同）：

    per_task=15   多任务 0.52   单任务 0.17   未训练（随机）0.32
    per_task=30   多任务 0.75   单任务 0.19   未训练（随机）0.32

即"把建模经验跨任务聚合"确实能零样本用到新数据集上，而单任务学到的部分是有害的
任务特定偏好（低于纯随机）——这正是论文 §5 的结论。

与 :mod:`engine.multiscale_gp` 的关系：本模块**不改动** GP 的算子语义——它产出的是
"建议的建模项集合"，由 :func:`compile_expr` 编译成 GP 同一套表达式树（复用
``eval_tree`` / ``panel_ic``），因此可作为 GP 的**种子种群**（把搜索导向高价值区域）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import param_ops
from .ic_utils import panel_ic
from .multiscale_gp import eval_tree

# ---------------------------------------------------------------------------
# 域目录（论文的 C = {K, T, G, V}）
# ---------------------------------------------------------------------------
#: 默认可用特征（与 ``genetic_enhanced._COLS`` 一致）
DEFAULT_FEATURES: Tuple[str, ...] = (
    "open", "high", "low", "close", "volume", "amount", "pct_chg")

#: 变换 T：索引 0 恒为 ``none``（该项不参与变换）
DEFAULT_TRANSFORMS: Tuple[str, ...] = (
    "none", "log", "abs", "sqrt", "sign", "neg", "delta",
    "ts_zscore", "ts_rank", "rol", "ts_min", "ts_max")

#: 组合结构 G：索引 0 为 ``none``；其余为与协变量特征的组合方式
DEFAULT_STRUCTURES: Tuple[str, ...] = ("none", "add", "sub", "mul", "div", "ts_corr")

#: 时间尺度档位 W：索引 0 为 ``none``（无窗口）
DEFAULT_WINDOWS: Tuple[int, ...] = (0, 3, 5, 10, 20, 60)

#: 参数化记忆算子开关 P：0 否，1 是（``hwma``，对应论文的"偏好异质性"位置）
DEFAULT_PARAMS: Tuple[int, ...] = (0, 1)

_WINDOWED_UNARY = {"ts_zscore", "ts_rank", "rol", "ts_min", "ts_max"}
_DEFAULT_WINDOW = 20

#: 奖励的归一化尺度。论文用 ``tanh(ΔLL / N_obs)``——除以观测数是为了让不同
#: 样本量的数据集可比；IC **本身已按截面标准化**，不再随样本数线性膨胀，因此
#: 这里除以的是"单个截面 IC 的典型量级"（默认 0.05），作用同为把奖励压进 (−1, 1)。
DEFAULT_REWARD_SCALE = 0.05


def _safe_mean(values: Any) -> float:
    """忽略非有限值的均值；全空返回 NaN（不触发 numpy 空切片告警）。

    项目 ``pytest.ini`` 设了 ``filterwarnings = error``，"Mean of empty slice"
    会把测试打成失败，所以聚合一律走这里。
    """
    a = np.asarray(values, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


# ---------------------------------------------------------------------------
# 建模项 / 状态 / 任务
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpecTerm:
    """一个建模项 ``(k, t, g, v, w, p)``。

    论文的项是 ``(k, t, g, v)``（属性、变换、taste 结构、协变量交互）。这里多出
    的两维是因子语境特有的：``w`` 时间尺度档位（滚动窗口），``p`` 是否套参数化
    时序记忆算子 ``hwma``（对应"偏好异质性"那一类决策）。
    """

    feature: int = 0                 # k：属性（特征）索引
    transform: int = 0               # t：变换，0 = none
    structure: int = 0               # g：与协变量的组合结构，0 = none
    covariate: int = -1              # v：协变量特征索引，−1 = none
    window: int = 0                  # w：时间尺度档位，0 = none
    param: int = 0                   # p：是否参数化（hwma）

    def with_(self, **kw: Any) -> "SpecTerm":
        data = {"feature": self.feature, "transform": self.transform,
                "structure": self.structure, "covariate": self.covariate,
                "window": self.window, "param": self.param}
        data.update(kw)
        return SpecTerm(**data)

    def complexity(self) -> int:
        """该项的复杂度：非 none 的组件数（用于 Pareto 前沿的"简约"一侧）。"""
        return int(self.transform != 0) + int(self.structure != 0) + \
            int(self.covariate >= 0) + int(self.window != 0) + int(self.param != 0)


@dataclass(frozen=True)
class SpecState:
    """规范状态 = 建模项的**集合**（论文式 (4)）。

    用元组承载只是实现方便：编码走置换不变的均值聚合，因此顺序不改变语义，
    :meth:`key` 给出集合语义的规范化键（供去重与缓存）。
    """

    terms: Tuple[SpecTerm, ...] = ()

    def key(self) -> Tuple[SpecTerm, ...]:
        return tuple(sorted(self.terms, key=lambda t: (t.feature, t.transform,
                                                       t.structure, t.covariate,
                                                       t.window, t.param)))

    def complexity(self) -> int:
        return int(sum(t.complexity() for t in self.terms))

    def size(self) -> int:
        return len(self.terms)


@dataclass
class SpecTask:
    """一个规范任务 τ（论文：一个数据集 = 一个任务）。

    ``features`` 是该任务可用的特征索引子集（不同任务的变量名可以完全不同，
    索引只是"建模概念"的槽位——这正是论文能跨数据集迁移的前提）。
    ``context`` 是任务上下文向量 x_τ：组件可用性 + 任务统计量（样本量、波动、
    可用特征数），**不含任务 one-hot**，因此训练后能直接用于未见任务。
    """

    name: str = "task"
    features: Tuple[int, ...] = ()
    n_obs: float = 0.0
    volatility: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def context(self, n_features: int) -> np.ndarray:
        avail = np.zeros(n_features, dtype=float)
        for i in self.features:
            if 0 <= i < n_features:
                avail[i] = 1.0
        stat = np.array([
            math.log1p(max(0.0, float(self.n_obs))) / 12.0,
            float(self.volatility),
            len(self.features) / max(1, n_features),
        ], dtype=float)
        return np.concatenate([avail, stat])


@dataclass(frozen=True)
class SpecAction:
    """建模动作：``add`` / ``change`` / ``terminate``（论文式 (6)）。"""

    kind: str = "terminate"          # "add" | "change" | "terminate"
    term: SpecTerm = SpecTerm()
    index: int = -1                  # change 作用到第几个已有项

    def key(self) -> Tuple[Any, ...]:
        return (self.kind, self.index, self.term)


@dataclass
class DomainCatalogue:
    """域目录 C：跨任务共享的建模组件（论文 §3.1）。"""

    features: Tuple[str, ...] = DEFAULT_FEATURES
    transforms: Tuple[str, ...] = DEFAULT_TRANSFORMS
    structures: Tuple[str, ...] = DEFAULT_STRUCTURES
    windows: Tuple[int, ...] = DEFAULT_WINDOWS
    params: Tuple[int, ...] = DEFAULT_PARAMS
    max_terms: int = 4

    def feature_name(self, idx: int) -> str:
        return self.features[idx] if 0 <= idx < len(self.features) else ""


# ---------------------------------------------------------------------------
# 编码：DeepSet（项嵌入 → 置换不变聚合）
# ---------------------------------------------------------------------------
def term_embedding(term: SpecTerm, cat: DomainCatalogue) -> np.ndarray:
    """单个建模项的嵌入 φ(x_l)：各字段 one-hot 拼接（论文式 (8) 的 φ）。"""
    nf = len(cat.features)
    parts = [
        _onehot(term.feature, nf),
        _onehot(term.transform, len(cat.transforms)),
        _onehot(term.structure, len(cat.structures)),
        _onehot(term.covariate + 1, nf + 1),          # −1(none) 落到最后一位
        _onehot(term.window, len(cat.windows)),
        _onehot(term.param, len(cat.params)),
    ]
    return np.concatenate(parts)


def encode_state(state: SpecState, cat: DomainCatalogue) -> np.ndarray:
    """DeepSet 聚合 Z(s)：``ρ = mean`` 而非论文式 (8) 的 sum。

    sum 对项数敏感（项多的规范范数天然更大），会让 Q 值随"加了几项"漂移；
    mean 同样是置换不变的，但把状态范数与项数解耦，线性 Q 更稳。
    """
    if not state.terms:
        return np.zeros(_term_dim(cat), dtype=float)
    emb = np.stack([term_embedding(t, cat) for t in state.terms])
    return emb.mean(axis=0)


def encode_action(action: SpecAction, cat: DomainCatalogue,
                  max_terms: int) -> np.ndarray:
    """动作嵌入 ψ(a)：类型 one-hot + 项嵌入 + 作用位置 one-hot。"""
    kind = _onehot(["add", "change", "terminate"].index(action.kind), 3)
    term = term_embedding(action.term, cat)
    pos = _onehot(action.index + 1, max_terms + 1) if action.kind == "change" \
        else np.zeros(max_terms + 1, dtype=float)
    return np.concatenate([kind, term, pos])


def _onehot(idx: int, size: int) -> np.ndarray:
    out = np.zeros(max(1, int(size)), dtype=float)
    i = int(idx)
    if 0 <= i < out.size:
        out[i] = 1.0
    return out


def _term_dim(cat: DomainCatalogue) -> int:
    nf = len(cat.features)
    return (nf + len(cat.transforms) + len(cat.structures) + (nf + 1)
            + len(cat.windows) + len(cat.params))


# ---------------------------------------------------------------------------
# 动作空间与掩码（论文 §3.1 iii）
# ---------------------------------------------------------------------------
def feasible_actions(state: SpecState, task: SpecTask, cat: DomainCatalogue,
                     used: Sequence[SpecAction],
                     previous: Optional[SpecTerm] = None) -> List[SpecAction]:
    """可行动作集合：论文的三条限制逐一落地。

    1. **不可用组件**：任务目录里没有的特征不能进项；
    2. **立即回退**：不能把某项改回上一步刚改过来的样子（来回拉锯无信息增益）；
    3. **已选过**：同一 episode 内同一动作只出现一次（论文 action masking）。

    另外：项数达到 ``cat.max_terms`` 后不再允许 ``add``（对应"简约"约束）。
    """
    feats = [i for i in task.features if 0 <= i < len(cat.features)]
    if not feats:
        return [SpecAction("terminate")]
    used_keys = {a.key() for a in used}
    out: List[SpecAction] = []

    if len(state.terms) < cat.max_terms:
        for k in feats:
            for t in range(1, len(cat.transforms)):
                for g in range(len(cat.structures)):
                    for v in [-1] + feats:
                        for w in range(len(cat.windows)):
                            for p in range(len(cat.params)):
                                if g == 0 and v >= 0:
                                    continue        # 无组合结构时协变量无意义
                                if g == len(cat.structures) - 1 and w == 0:
                                    continue        # ts_corr 必须带窗口
                                a = SpecAction("add", SpecTerm(k, t, g, v, w, p))
                                if a.key() not in used_keys:
                                    out.append(a)

    for i, term in enumerate(state.terms):
        for t in range(len(cat.transforms)):
            for g in range(len(cat.structures)):
                for v in [-1] + feats:
                    for w in range(len(cat.windows)):
                        for p in range(len(cat.params)):
                            if g == 0 and v >= 0:
                                continue
                            if g == len(cat.structures) - 1 and w == 0:
                                continue
                            new = term.with_(transform=t, structure=g,
                                             covariate=v, window=w, param=p)
                            if previous is not None and new == previous:
                                continue            # 立刻改回上一步之前的值
                            if new == term:
                                continue            # 改了等于没改
                            a = SpecAction("change", new, i)
                            if a.key() not in used_keys:
                                out.append(a)

    out.append(SpecAction("terminate"))
    return out


def _valid_term(term: SpecTerm, cat: DomainCatalogue) -> bool:
    """项的内部一致性：无组合结构时不带协变量；``ts_corr`` 必须带窗口。"""
    if term.structure == 0 and term.covariate >= 0:
        return False
    if term.structure == len(cat.structures) - 1 and term.window == 0:
        return False
    return True


def _random_term(feats: Sequence[int], cat: DomainCatalogue,
                 rng: np.random.Generator) -> SpecTerm:
    k = int(feats[int(rng.integers(0, len(feats)))])
    g = int(rng.integers(0, len(cat.structures)))
    w = int(rng.integers(0, len(cat.windows)))
    if g == len(cat.structures) - 1 and w == 0:
        w = int(rng.integers(1, len(cat.windows)))
    v = -1
    if g != 0 and rng.random() < 0.7:
        v = int(feats[int(rng.integers(0, len(feats)))])
    term = SpecTerm(feature=k,
                    transform=int(rng.integers(0, len(cat.transforms))),
                    structure=g, covariate=v, window=w,
                    param=int(rng.integers(0, len(cat.params))))
    return term if _valid_term(term, cat) else term.with_(covariate=-1 if g == 0 else v)


def sample_actions(state: SpecState, task: SpecTask, cat: DomainCatalogue,
                   used: Sequence[SpecAction], previous: Optional[SpecTerm],
                   rng: np.random.Generator, k: int) -> List[SpecAction]:
    """可行动作的**随机子采样**版（智能体实际走这条路径）。

    完整组合目录在本项目规模下是几万个动作（7 特征 × 12 变换 × 6 结构 × 8 协变量
    × 6 窗口 × 2 参数），逐步枚举再筛选会吃掉几乎全部墙钟时间。这里改为按合法
    分布直接采样 ``k`` 个动作，**同时保留论文的三条掩码限制**（不可用组件在采样
    层面排除、回退与已选动作在接纳时剔除），最后一并附上 terminate。
    """
    feats = [i for i in task.features if 0 <= i < len(cat.features)]
    if not feats:
        return [SpecAction("terminate")]
    used_keys = {a.key() for a in used}
    out: List[SpecAction] = []
    tries, limit = 0, max(8, int(k) * 8)
    can_add = len(state.terms) < cat.max_terms
    while len(out) < max(1, int(k)) and tries < limit:
        tries += 1
        if can_add and (not state.terms or rng.random() < 0.5):
            action = SpecAction("add", _random_term(feats, cat, rng))
        else:
            i = int(rng.integers(0, len(state.terms)))
            new = _random_term(feats, cat, rng).with_(feature=state.terms[i].feature)
            if new == state.terms[i] or (previous is not None and new == previous):
                continue
            action = SpecAction("change", new, i)
        if action.key() in used_keys:
            continue
        used_keys.add(action.key())
        out.append(action)
    out.append(SpecAction("terminate"))
    return out


# ---------------------------------------------------------------------------
# 奖励（论文式 (7)）
# ---------------------------------------------------------------------------
def spec_reward(fit: float, baseline: float, ok: bool = True,
                scale: float = DEFAULT_REWARD_SCALE) -> float:
    """``tanh((fit − baseline) / scale)``；估计失败固定 −1。

    论文的 ``tanh(ΔLL / N_obs)`` 除的是观测数；这里 IC 已按截面标准化，除以的是
    IC 的典型量级（``scale``），作用相同：让不同任务的奖励落到同一尺度的 (−1, 1)。
    """
    if not ok or not np.isfinite(fit):
        return -1.0
    base = baseline if np.isfinite(baseline) else 0.0
    s = scale if np.isfinite(scale) and scale > 0 else DEFAULT_REWARD_SCALE
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        val = float(np.tanh((float(fit) - base) / s))
    return val if np.isfinite(val) else -1.0


# ---------------------------------------------------------------------------
# 估计环境
# ---------------------------------------------------------------------------
@dataclass
class Estimate:
    """估计环境的一次返回：拟合值、是否收敛、诊断信息。"""

    fit: float = float("nan")
    ok: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


class SpecEnvironment:
    """估计环境接口：``evaluate(task, state)`` 返回 :class:`Estimate`。

    子类只需实现 :meth:`evaluate`。自带**结果缓存**——同一 (任务, 规范) 不重复
    估计，因为论文最核心的收益指标就是"少做几次失败的估计尝试"。
    """

    #: 奖励归一化尺度（论文里由观测数 N 提供）。子类可按任务改写——拟合值的量纲
    #: 由环境决定，让 reward 永远饱和到 ±1 就等于没有学习信号。
    reward_scale: float = DEFAULT_REWARD_SCALE

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, Tuple[SpecTerm, ...]], Estimate] = {}
        self.n_estimations: int = 0

    def evaluate(self, task: SpecTask, state: SpecState) -> Estimate:
        key = (task.name, state.key())
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        est = self._evaluate(task, state)
        self.n_estimations += 1
        self._cache[key] = est
        return est

    def baseline(self, task: SpecTask, state: SpecState) -> float:
        """基线拟合值（论文用 null model 作下界）。默认 0（IC 的自然零点）。"""
        return 0.0

    def _evaluate(self, task: SpecTask, state: SpecState) -> Estimate:
        raise NotImplementedError


class PanelSpecEnvironment(SpecEnvironment):
    """面板估计环境：规范 → 表达式树 → 求值 → 逐截面 IC 均值。

    求值走 :func:`engine.multiscale_gp.eval_tree`（``hwma`` 原生支持、其余算子
    委派给 ``genetic_enhanced.eval_expr``），IC 走 :func:`engine.ic_utils.panel_ic`，
    因此与 GP 主链路**同一套语义**，不存在第二套求值口径。
    """

    def __init__(self, panels: Dict[str, pd.DataFrame], cat: DomainCatalogue,
                 y_col: str = "fwd_ret", min_count: int = 5,
                 date_col: str = "date") -> None:
        super().__init__()
        self.panels = panels
        self.cat = cat
        self.y_col = y_col
        self.min_count = int(min_count)
        self.date_col = date_col

    def _evaluate(self, task: SpecTask, state: SpecState) -> Estimate:
        df = self.panels.get(task.name)
        if df is None or not state.terms:
            return Estimate(float("nan"), False, {"reason": "no_panel_or_empty_spec"})
        try:
            expr = compile_expr(state, task, self.cat)
        except (RuntimeError, ValueError, TypeError):
            return Estimate(float("nan"), False, {"reason": "compile_failed"})
        try:
            with np.errstate(all="ignore"):
                vals = np.asarray(eval_tree(expr, df), dtype=float)
        except (ValueError, TypeError, KeyError, FloatingPointError):
            return Estimate(float("nan"), False, {"reason": "eval_failed"})
        y = df[self.y_col].to_numpy(dtype=float) if self.y_col in df.columns else None
        if y is None or vals.size != y.size:
            return Estimate(float("nan"), False, {"reason": "size_mismatch"})
        try:
            ic = panel_ic(vals, y, df[self.date_col].to_numpy(),
                          min_count=self.min_count)
        except (ValueError, TypeError, KeyError):
            return Estimate(float("nan"), False, {"reason": "ic_failed"})
        # 拟合值取 |IC|：因子方向可以自由取反（多头/空头只是符号），
        # 因此要奖励的是"区分度"而不是"恰好为正"。
        fit = abs(_safe_mean(np.asarray(ic, dtype=float)))
        if not np.isfinite(fit):
            return Estimate(float("nan"), False, {"reason": "ic_all_nan"})
        return Estimate(float(fit), True, {"n_dates": int(len(ic))})


# ---------------------------------------------------------------------------
# 规范 → 表达式树（与 GP 同一套语法）
# ---------------------------------------------------------------------------
def compile_expr(state: SpecState, task: SpecTask, cat: DomainCatalogue) -> Any:
    """把规范编译成表达式树：各项先各自成子树，再按论文的"线性加性"逐项相加。"""
    subs: List[Any] = []
    for term in state.terms:
        subs.append(_compile_term(term, task, cat))
    if not subs:
        raise ValueError("空规范无法编译")
    expr = subs[0]
    for s in subs[1:]:
        expr = ("add", expr, s)
    return expr


def _compile_term(term: SpecTerm, task: SpecTask, cat: DomainCatalogue) -> Any:
    name = cat.feature_name(term.feature)
    if not name:
        raise ValueError(f"特征索引越界: {term.feature}")
    node: Any = ("col", name)

    tname = cat.transforms[term.transform] if 0 <= term.transform < len(cat.transforms) else "none"
    if tname != "none":
        if tname in _WINDOWED_UNARY:
            w = cat.windows[term.window] if 0 <= term.window < len(cat.windows) and \
                cat.windows[term.window] > 0 else _DEFAULT_WINDOW
            node = (tname, node, ("const", float(w)))
        else:
            node = (tname, node)

    gname = cat.structures[term.structure] if 0 <= term.structure < len(cat.structures) else "none"
    if gname != "none" and term.covariate >= 0:
        other = cat.feature_name(term.covariate)
        if other:
            if gname == "ts_corr":
                w = cat.windows[term.window] if 0 <= term.window < len(cat.windows) and \
                    cat.windows[term.window] > 0 else _DEFAULT_WINDOW
                node = (gname, node, ("col", other), ("const", float(w)))
            else:
                node = (gname, node, ("col", other))

    if term.param:
        node = param_ops.make_node(node, param_ops.HyperbolicParams())
    return node


# ---------------------------------------------------------------------------
# DeepSet-Q（线性实现：共享表征 + 任务条件化 + 目标网络）
# ---------------------------------------------------------------------------
class DeepSetQ:
    """线性 DeepSet-Q：``Q(s, a, τ) = w · [Z(s) ‖ x_τ ‖ ψ(a)]``。

    保留论文让迁移成立的两个机制：**置换不变的规范聚合**（不同任务项数不同也能
    落到同一表征空间）与**任务上下文条件化**（同一策略按任务可用性调整打分）；
    去掉的是非线性拟合能力（见模块 docstring）。
    """

    def __init__(self, dim: int, lr: float = 0.05, seed: int = 42) -> None:
        self.dim = int(dim)
        self.lr = float(lr)
        rng = np.random.default_rng(int(seed))
        self.w = rng.normal(0.0, 0.01, size=self.dim)
        self.target = self.w.copy()
        self._steps = 0

    def q_values(self, phi: np.ndarray) -> np.ndarray:
        """``phi`` 形如 (n_actions, dim)；返回各动作的 Q。"""
        if phi.size == 0:
            return np.zeros(0, dtype=float)
        return np.asarray(phi @ self.w, dtype=float).reshape(-1)

    def q_target(self, phi: np.ndarray) -> np.ndarray:
        if phi.size == 0:
            return np.zeros(0, dtype=float)
        return np.asarray(phi @ self.target, dtype=float).reshape(-1)

    def update(self, phi: np.ndarray, targets: np.ndarray) -> float:
        """一步（小批量）梯度下降，返回该批的均方误差。

        梯度按范数裁剪到 1：线性 Q 在 one-hot 拼接的高维特征上容易出现个别维度
        的尖峰梯度，不裁剪会把权重整体带跑（表现为"训练后反而不如随机"）。
        """
        if phi.size == 0:
            return float("nan")
        pred = phi @ self.w
        err = np.asarray(targets, dtype=float) - pred
        grad = -(2.0 / phi.shape[0]) * (phi.T @ err)
        norm = float(np.linalg.norm(grad))
        if np.isfinite(norm) and norm > 1.0:
            grad = grad / norm
        self.w -= self.lr * grad
        self._steps += 1
        return float(np.mean(err ** 2))

    def sync_target(self) -> None:
        self.target = self.w.copy()


# ---------------------------------------------------------------------------
# 经验回放（按任务分层的平衡采样）
# ---------------------------------------------------------------------------
@dataclass
class Transition:
    task: str
    phi: np.ndarray
    phi_next: Optional[np.ndarray]
    reward: float
    done: bool


class ReplayBuffer:
    """跨任务**共享**的经验回放（论文 §3.3）。

    采样时对各任务取**相同条数**的平衡小批量——论文明确提到：否则轨迹更长的
    数据集会主导参数更新。
    """

    def __init__(self, capacity: int = 20000) -> None:
        self.capacity = int(capacity)
        self.buffers: Dict[str, List[Transition]] = {}

    def push(self, tr: Transition) -> None:
        buf = self.buffers.setdefault(tr.task, [])
        buf.append(tr)
        if len(buf) > self.capacity:
            del buf[0]

    def __len__(self) -> int:
        return sum(len(b) for b in self.buffers.values())

    def balanced_batch(self, size: int, rng: np.random.Generator) -> List[Transition]:
        tasks = [t for t, b in self.buffers.items() if b]
        if not tasks:
            return []
        per = max(1, int(size) // len(tasks))
        out: List[Transition] = []
        for t in tasks:
            buf = self.buffers[t]
            idx = rng.choice(len(buf), size=min(per, len(buf)), replace=False)
            out.extend(buf[int(i)] for i in idx)
        return out


# ---------------------------------------------------------------------------
# 多任务智能体
# ---------------------------------------------------------------------------
@dataclass
class EpisodeLog:
    task: str
    reward: float
    fit: float
    ok: bool
    n_terms: int
    complexity: int
    spec: Tuple[SpecTerm, ...]


@dataclass
class Candidate:
    spec: Tuple[SpecTerm, ...]
    fit: float
    ok: bool
    complexity: int
    expr: Any = None

    def to_dict(self, cat: Optional[DomainCatalogue] = None) -> Dict[str, Any]:
        return {
            "terms": [_term_to_dict(t, cat) for t in self.spec],
            "fit": _py_float(self.fit),
            "ok": bool(self.ok),
            "complexity": int(self.complexity),
        }


def _term_to_dict(term: SpecTerm, cat: Optional[DomainCatalogue]) -> Dict[str, Any]:
    f = cat.feature_name(term.feature) if cat else str(term.feature)
    c = cat.feature_name(term.covariate) if (cat and term.covariate >= 0) else None
    return {
        "feature": f,
        "transform": cat.transforms[term.transform] if cat else term.transform,
        "structure": cat.structures[term.structure] if cat else term.structure,
        "covariate": c,
        "window": int(cat.windows[term.window]) if cat else int(term.window),
        "param": bool(term.param),
    }


def _py_float(x: Any) -> Any:
    """NaN / ±inf 统一成 ``None``：JSON 里不允许裸 ``NaN``。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


class MultitaskSpecAgent:
    """多任务规范搜索智能体（论文的 Delphos）。

    ``shared=True``（默认）时**所有任务共用一套 Q 参数**（论文的多任务设定）；
    ``shared=False`` 时每个任务各学一套（论文对照的 single-task 基线）——同一个
    类就能跑出"多任务 vs 单任务"的对照实验。
    """

    def __init__(
        self,
        cat: DomainCatalogue,
        env: SpecEnvironment,
        *,
        shared: bool = True,
        gamma: float = 0.9,
        lr: float = 0.1,
        eps_start: float = 0.9,
        eps_end: float = 0.05,
        max_steps: int = 4,
        max_actions: int = 256,
        batch_size: int = 16,
        target_sync: int = 20,
        reward_scale: float = DEFAULT_REWARD_SCALE,
        bootstrap_truncated: bool = False,
        seed: int = 42,
    ) -> None:
        self.cat = cat
        self.env = env
        self.shared = bool(shared)
        self.gamma = float(gamma)
        self._lr = float(lr)
        self.max_steps = max(1, int(max_steps))
        self.max_actions = max(8, int(max_actions))
        self.batch_size = max(2, int(batch_size))
        self.target_sync = max(1, int(target_sync))
        self.reward_scale = float(reward_scale)
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.bootstrap_truncated = bool(bootstrap_truncated)
        self._rng = np.random.default_rng(int(seed))
        dim = _term_dim(cat) + (len(cat.features) + 3) + 1 + \
            (3 + _term_dim(cat) + cat.max_terms + 1)
        self._dim = int(dim)
        self.qnets: Dict[str, DeepSetQ] = {}
        self.buffer = ReplayBuffer()
        self.history: List[EpisodeLog] = []
        self.n_updates = 0

    # -- Q 网络（共享或按任务独立） --
    def _net(self, task: str) -> DeepSetQ:
        key = "__shared__" if self.shared else task
        net = self.qnets.get(key)
        if net is None:
            net = DeepSetQ(self._dim, lr=self._lr,
                           seed=int(self._rng.integers(1, 10 ** 6)))
            self.qnets[key] = net
        return net

    # -- 特征构造 --
    def _phi(self, task: SpecTask, state: SpecState,
             actions: Sequence[SpecAction]) -> np.ndarray:
        z = encode_state(state, self.cat)
        ctx = task.context(len(self.cat.features))
        # 项数也进上下文：均值聚合把范数与项数解耦了，但"这个规范有几项"本身是
        # 决策相关信息（对应论文的"简约"一侧），得显式告诉策略。
        size = np.array([state.size() / max(1, self.cat.max_terms)])
        rows = [np.concatenate([z, ctx, size,
                                encode_action(a, self.cat, self.cat.max_terms)])
                for a in actions]
        return np.asarray(rows, dtype=float) if rows else np.zeros((0, self._dim))

    # -- 一条 episode --
    def _run_episode(self, task: SpecTask, greedy: bool = False,
                     eps: float = 0.0) -> EpisodeLog:
        state = initial_state(task, self.cat)
        used: List[SpecAction] = []
        previous: Optional[SpecTerm] = None
        acc = 0.0
        discount = 1.0
        est = Estimate(float("nan"), False)
        traj: List[Tuple[np.ndarray, Optional[np.ndarray]]] = []
        final_r = 0.0
        truncated = False
        for _ in range(self.max_steps):
            actions = sample_actions(state, task, self.cat, used, previous,
                                     self._rng, self.max_actions)
            net = self._net(task.name)
            phi = self._phi(task, state, actions)
            idx = self._select(net, phi, greedy, eps)
            action = actions[int(idx)]
            used.append(action)

            if action.kind == "terminate":
                est = self.env.evaluate(task, state)
                base = self.env.baseline(task, state)
                final_r = spec_reward(est.fit, base, est.ok, self._scale())
                if not greedy:
                    self.buffer.push(Transition(task.name, phi[int(idx)], None,
                                                final_r, True))
                acc += discount * final_r
                break

            if action.kind == "add":
                previous = None
                state = SpecState(state.terms + (action.term,))
            else:
                i = action.index
                if 0 <= i < len(state.terms):
                    previous = state.terms[i]
                    terms = list(state.terms)
                    terms[i] = action.term
                    state = SpecState(tuple(terms))
                else:
                    previous = None

            if not greedy:
                n_actions_next = sample_actions(state, task, self.cat, used, previous,
                                                self._rng, self.max_actions)
                phi_next = self._phi(task, state, n_actions_next)
                traj.append((phi[int(idx)], phi_next))
            discount *= self.gamma

        if not used or used[-1].kind != "terminate":
            truncated = True
            est = self.env.evaluate(task, state)
            base = self.env.baseline(task, state)
            final_r = spec_reward(est.fit, base, est.ok, self._scale())
            acc += final_r * discount

        # 稀疏奖励只落在终止步。episode 最多 ``max_steps`` 步，用**折扣回报**
        # （reward-to-go）而不是一步 TD 目标：线性 Q 上的 ``max`` 会系统性高估，
        # 而 4 步的短轨迹根本传不回有效信号（实测表现为"训练后不如随机"）。
        # 这是本模块对论文 TD(0) 目标的唯一偏离，理由写在模块 docstring 里。
        if not greedy:
            n_traj = len(traj)
            for step, (p, pn) in enumerate(traj):
                r2g = final_r * (self.gamma ** (n_traj - step))
                last = step == n_traj - 1
                # 被步数截断的 episode，末端可选地补上目标网络的 bootstrap（θ⁻）。
                # 默认关闭：本项目的轨迹只有 ``max_steps`` 步、且早期探索阶段多数
                # episode 都是截断的，此时 bootstrap 的高估会主导目标值——实测
                # 开启后 held-out 迁移均分从 0.75 掉到 0.49。开关留着做对照。
                if last and truncated and self.bootstrap_truncated:
                    self.buffer.push(Transition(task.name, p, pn, r2g, False))
                else:
                    self.buffer.push(Transition(task.name, p, None, r2g, True))
            self._optimise()

        return EpisodeLog(task.name, float(acc), float(est.fit), bool(est.ok),
                          state.size(), state.complexity(), state.key())

    def _scale(self) -> float:
        """奖励尺度：环境给的优先（拟合值量纲由环境定义），否则用构造参数。"""
        s = getattr(self.env, "reward_scale", None)
        return float(s) if s and np.isfinite(s) and s > 0 else self.reward_scale

    def _select(self, net: DeepSetQ, phi: np.ndarray,
                greedy: bool, eps: float) -> int:
        if phi.shape[0] == 0:
            return 0
        if not greedy and self._rng.random() < eps:
            return int(self._rng.integers(0, phi.shape[0]))
        q = net.q_values(phi)
        best = np.argmax(q)
        ties = np.flatnonzero(np.isclose(q, q[int(best)], rtol=0, atol=1e-12))
        return int(ties[int(self._rng.integers(0, ties.size))])

    # -- 训练 --
    def train(self, tasks: Sequence[SpecTask], n_episodes: int = 60) -> Dict[str, Any]:
        """在多个任务上**联合**训练（论文 §3.3）。

        任务按轮转顺序取样，保证各任务的 episode 数相同（平衡暴露）。
        """
        tasks = list(tasks)
        if not tasks:
            return self.report()
        per_task = max(1, int(n_episodes) // len(tasks))
        total = per_task * len(tasks)
        order: List[SpecTask] = []
        for _ in range(per_task):
            order.extend(tasks)
        for i, task in enumerate(order):
            frac = i / max(1, total - 1)
            eps = self.eps_start + (self.eps_end - self.eps_start) * frac
            log = self._run_episode(task, greedy=False, eps=eps)
            self.history.append(log)
            self._optimise()
        return self.report()

    def _optimise(self) -> None:
        if len(self.buffer) < self.batch_size:
            return
        batch = self.buffer.balanced_batch(self.batch_size, self._rng)
        if not batch:
            return
        by_task: Dict[str, List[Transition]] = {}
        for tr in batch:
            by_task.setdefault(tr.task, []).append(tr)
        for task_name, rows in by_task.items():
            net = self._net(task_name)
            phi = np.stack([r.phi for r in rows])
            targets = np.empty(len(rows), dtype=float)
            for j, r in enumerate(rows):
                if r.done or r.phi_next is None or r.phi_next.size == 0:
                    targets[j] = r.reward
                    continue
                # phi_next 是"下一步可行动作"的特征矩阵，所以 max_a' Q 一次算完
                nxt = net.q_target(r.phi_next)
                nxt = nxt[np.isfinite(nxt)]
                # 目标网络（θ⁻）给出下一步的最大 Q；全为 NaN 时退化为纯奖励
                targets[j] = r.reward + self.gamma * float(nxt.max()) if nxt.size else r.reward
            net.update(phi, targets)
            self.n_updates += 1
            if self.n_updates % self.target_sync == 0:
                net.sync_target()

    # -- 推断：零样本用于未见任务 --
    def propose(self, task: SpecTask, n_candidates: int = 6) -> List[Candidate]:
        """用训练好的策略在（可能没见过的）任务上提候选规范。"""
        seen: Dict[Tuple[SpecTerm, ...], Candidate] = {}
        for _ in range(max(1, int(n_candidates))):
            log = self._run_episode(task, greedy=True)
            if log.spec in seen:
                continue
            expr = None
            try:
                expr = compile_expr(SpecState(log.spec), task, self.cat)
            except (ValueError, RuntimeError, TypeError):
                expr = None
            seen[log.spec] = Candidate(log.spec, log.fit, log.ok, log.complexity, expr)
        return sorted(seen.values(), key=lambda c: (-(c.fit if np.isfinite(c.fit) else -1e9),
                                                    c.complexity))

    # -- 指标（论文 §4.2）--
    def report(self, task: Optional[str] = None) -> Dict[str, Any]:
        logs = [h for h in self.history if task is None or h.task == task]
        rewards = [h.reward for h in logs if np.isfinite(h.reward)]
        fits = [h.fit for h in logs if np.isfinite(h.fit)]
        curve = _learning_curve(rewards)
        return {
            "shared": bool(self.shared),
            "n_episodes": len(logs),
            "mean_reward": _py_float(_safe_mean(rewards)),
            "max_reward": _py_float(max(rewards) if rewards else float("nan")),
            "mean_fit": _py_float(_safe_mean(fits)),
            "best_fit": _py_float(max(fits) if fits else float("nan")),
            "auc": _py_float(_safe_mean(curve)),
            "convergence_rate": _py_float(
                sum(1 for h in logs if h.ok) / len(logs) if logs else float("nan")),
            "novelty": _py_float(
                len({h.spec for h in logs}) / len(logs) if logs else float("nan")),
            "n_estimations": int(self.env.n_estimations),
            "n_updates": int(self.n_updates),
            "learning_curve": [_py_float(x) for x in curve],
        }


def _learning_curve(rewards: Sequence[float], bins: int = 10) -> List[float]:
    """学习曲线 AUC：论文用"曲线下面积"衡量学习效率。"""
    n = len(rewards)
    if n == 0:
        return []
    k = max(1, min(int(bins), n))
    step = n / k
    return [_safe_mean(rewards[int(i * step):max(int((i + 1) * step), int(i * step) + 1)])
            for i in range(k)]


# ---------------------------------------------------------------------------
# 初始状态（论文：线性加性基线）
# ---------------------------------------------------------------------------
def initial_state(task: SpecTask, cat: DomainCatalogue) -> SpecState:
    """起手规范：可用特征以**线性形式**各自成项（论文 s0：属性线性进入、无交互）。"""
    terms = tuple(SpecTerm(feature=i) for i in task.features
                  if 0 <= i < len(cat.features))
    return SpecState(terms[:max(1, cat.max_terms)] or (SpecTerm(feature=0),))


# ---------------------------------------------------------------------------
# Pareto 前沿（拟合 × 简约）
# ---------------------------------------------------------------------------
def pareto_front(cands: Sequence[Candidate]) -> List[Candidate]:
    """论文 §4.2：拟合更好且更简约者支配；返回非支配候选（按复杂度升序）。"""
    ok = [c for c in cands if c.ok and np.isfinite(c.fit)]
    front: List[Candidate] = []
    for c in ok:
        dominated = any((o.fit >= c.fit and o.complexity <= c.complexity and
                         (o.fit > c.fit or o.complexity < c.complexity)) for o in ok
                        if o is not c)
        if not dominated:
            front.append(c)
    return sorted(front, key=lambda c: (c.complexity, -(c.fit if np.isfinite(c.fit) else -1e9)))


# ---------------------------------------------------------------------------
# 便捷入口：从面板构造任务 + 训练 + 推断
# ---------------------------------------------------------------------------
def tasks_from_panels(panels: Dict[str, pd.DataFrame], cat: DomainCatalogue,
                      y_col: str = "fwd_ret") -> List[SpecTask]:
    """按面板名构造任务：可用特征取面板中真实存在的列，统计量从面板估计。

    ``volatility`` 用面板收益的标准差（缺失时记 NaN→0），``n_obs`` 用行数——
    两者都进任务上下文 x_τ，让策略知道"这个任务有多少样本、多吵"。
    """
    tasks: List[SpecTask] = []
    for name, df in panels.items():
        feats = tuple(i for i, f in enumerate(cat.features) if f in df.columns)
        vol = float("nan")
        if y_col in df.columns:
            vol = float(np.nanstd(df[y_col].to_numpy(dtype=float)))
        tasks.append(SpecTask(name=name, features=feats,
                              n_obs=float(len(df)),
                              volatility=0.0 if not np.isfinite(vol) else vol))
    return tasks


def specification_search(
    panels: Dict[str, pd.DataFrame],
    *,
    cat: Optional[DomainCatalogue] = None,
    y_col: str = "fwd_ret",
    n_episodes: int = 40,
    n_candidates: int = 6,
    shared: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """一次跑齐：构造任务 → 多任务训练 → 在每个任务上提候选 + Pareto 前沿。

    返回严格可 JSON 化的字典（NaN 一律转 ``None``），并给出"估计次数"这一
    论文核心成本口径。
    """
    cat = cat or DomainCatalogue()
    env = PanelSpecEnvironment(panels, cat, y_col=y_col)
    tasks = tasks_from_panels(panels, cat, y_col=y_col)
    agent = MultitaskSpecAgent(cat, env, shared=shared, seed=int(seed))
    train_report = agent.train(tasks, n_episodes=int(n_episodes))
    per_task = []
    for t in tasks:
        cands = agent.propose(t, n_candidates=int(n_candidates))
        front = pareto_front(cands)
        per_task.append({
            "task": t.name,
            "n_features": len(t.features),
            "candidates": [c.to_dict(cat) for c in cands],
            "pareto": [c.to_dict(cat) for c in front],
            "best_fit": _py_float(cands[0].fit if cands else float("nan")),
        })
    return {
        "shared": bool(shared),
        "n_tasks": len(tasks),
        "train": train_report,
        "tasks": per_task,
        "n_estimations": int(env.n_estimations),
        "catalogue": {
            "features": list(cat.features),
            "transforms": list(cat.transforms),
            "structures": list(cat.structures),
            "windows": [int(w) for w in cat.windows],
        },
    }


def spec_seeds(
    panels: Dict[str, pd.DataFrame],
    *,
    cat: Optional[DomainCatalogue] = None,
    y_col: str = "fwd_ret",
    n_episodes: int = 40,
    n_candidates: int = 6,
    seed: int = 42,
) -> List[Any]:
    """把多任务策略提出的规范编译成**表达式树**，供 GP 作种子种群。

    这是本模块与 :mod:`engine.multiscale_gp` 的接缝：RL 负责"往哪搜"（把搜索
    导向高价值区域），GP 负责"搜多细"（在该区域内做结构演化）。返回的是 GP
    同一套 tuple 表达式树，可直接进 ``HierarchicalFactorMiner`` 的初始种群。
    """
    cat = cat or DomainCatalogue()
    env = PanelSpecEnvironment(panels, cat, y_col=y_col)
    tasks = tasks_from_panels(panels, cat, y_col=y_col)
    if not tasks:
        return []
    agent = MultitaskSpecAgent(cat, env, shared=True, seed=int(seed))
    agent.train(tasks, n_episodes=int(n_episodes))
    seeds: List[Any] = []
    for t in tasks:
        for c in pareto_front(agent.propose(t, n_candidates=int(n_candidates))):
            if c.expr is not None:
                seeds.append(c.expr)
    # 去重（不同任务可能提出同一规范）——保持顺序，避免随机化破坏可复现性
    uniq: List[Any] = []
    seen = set()
    for e in seeds:
        k = repr(e)
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    return uniq
