"""分层多尺度遗传因子挖掘（src/engine/multiscale_gp.py）。

落地文献 ``2502.14141v3`` 的 deep hierarchical policy gradient method。该方法解的是
"资源受限下连续时间最优控制"，其核心洞察与因子挖掘**同构**：

    不要一上来就用最细的时间网格和最大的算力求解全局；
    先在粗网格上求解，再用"粗解是否可信"的判据挑出少数失真区间，
    只在这些区间上加密求解，并用粗尺度的价值函数作为终端代价把两个尺度缝起来。

对应关系（本模块的实现）：

======================================  ==================================================
论文机制                                    本模块
======================================  ==================================================
时间离散 δ（粗 N₁ / 细 N₂）                 评估粒度：粗 = 低频末截面，细 = 逐日截面
粗策略 θ₁                                      粗尺度演化得到的精英表达式
经验训练域 𝒟_tr(T_i)                         区间内**因子暴露的经验分布**（稳健标准化后）
Hausdorff 距离 d_H^i                            相邻区间暴露分布漂移（一维精确 Hausdorff）
价值函数均方差 MSD_i                          相邻区间**暴露→收益 profile** 的均方差
经验选子集 I ⊂ [N₁]                           得分最高的 ``n_select`` 个区间
终端项 χ(T_{i+1},x;ρ₁)                         细尺度适应度里的**粗尺度延续价值**（含代理模型）
资源分配 (M, δ, 复杂度)                        种群规模 / 评估粒度 / 表达式深度
Theorem 3.1 的效率增益                         可核对的评估预算报告 ``ResourceReport.speedup``
======================================  ==================================================

与 :mod:`engine.genetic_enhanced` 的关系：表达式语法、求值、代码生成**完全复用**既有算子集，
本模块不复制 emitter，因此不存在"两套语义漂移"；新增能力为：

1. ``hwma`` 参数化时序算子（来自文献 ``2607.23068v1`` §3.1）：结构由演化决定、5 个参数
   在区间内用坐标下降拟合 —— 即"外演化 + 内拟合"的双层搜索；
2. 真正的子树交叉 + 深度受控变异（既有实现只做同位置子树交换）；
3. 延续价值代理模型（岭回归）：把粗尺度价值与已观测的"结构→细尺度 IC"映射混合，
   对应论文用回归拟合价值函数 χ 的做法；
4. 区间诊断（d_H + MSD）与资源报告，以及样本外 IC 复核。

**成本口径**：``ResourceReport`` 的"评估次数"按论文的资源抽象计数（个体数 × 评估位置数），
其中"暴力细网格"基准 = 每个区间各自独立做细尺度演化所需的评估量。``eval_expr`` 内部对整块
面板向量化求值（滚动窗口需要完整预热历史，按区间裁表会让区间开头的因子值出错），
因此**墙钟时间的节省小于评估次数的节省**；``total_evals`` 另行给出真实去重求值次数，
两个口径都摆出来，不把前者说成后者。
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import param_ops
from .genetic_enhanced import (
    MAX_DEPTH,
    _ROL_W,
    eval_expr,
    expr_to_code,
    expr_to_expr_str,
    random_expr,
)
from .ic_utils import block_last_positions, ic_stats, period_codes

# ---------------------------------------------------------------------------
# 表达式树工具（与 genetic_enhanced 的算子白名单严格对应）
# ---------------------------------------------------------------------------
_UNARY = {"log", "abs", "sqrt", "sign", "neg", "delta"}
_WINDOWED_UNARY = {"ts_zscore", "ts_rank", "rol", "ts_min", "ts_max"}
_BINARY = {"add", "sub", "mul", "div"}
_TERNARY = {"ts_corr"}
_PARAM = {param_ops.OP_NAME}

# 暴露标准化后的截尾阈值：把 d_H / MSD 的度量空间限制在有界区间内（见 _robust_scale）
_Z_CLIP = 5.0


def _child_positions(kind: str, node: tuple) -> Tuple[int, ...]:
    """返回节点的子表达式下标（按**结构**判定，不按算子白名单）。

    遗传编程产出的窗口算子有两种合法形态：``("ts_rank", child, ("const", w))``
    （带窗口位）与 ``("ts_rank", child)``（窗口留空、求值时兜底为默认值）。按
    白名单写死 ``(1, 2)`` 会在第二种形态上直接越界；而参数位 ``("const", w)``
    与 ``("params", θ)`` 不是子表达式。因此统一规则 = 第 1 位之后的、非参数位的
    元组元素。新增算子时本模块无需同步修改。
    """
    return tuple(i for i, x in enumerate(node)
                 if i > 0 and isinstance(x, tuple) and x and x[0] not in ("const", "params"))


def contains_param_op(expr: Any) -> bool:
    """表达式树中是否含 ``hwma`` 参数化时序算子。"""
    if not isinstance(expr, tuple) or not expr:
        return False
    if expr[0] in _PARAM:
        return True
    return any(contains_param_op(expr[i]) for i in _child_positions(expr[0], expr))


def tree_depth(expr: Any) -> int:
    """表达式深度（终结符为 0）。"""
    if not isinstance(expr, tuple) or not expr:
        return 0
    kids = _child_positions(expr[0], expr)
    if not kids:
        return 0
    return 1 + max(tree_depth(expr[i]) for i in kids)


def count_nodes(expr: Any) -> int:
    if not isinstance(expr, tuple) or not expr:
        return 0
    return 1 + sum(count_nodes(expr[i]) for i in _child_positions(expr[0], expr))


def _window_value(node: tuple, idx: int) -> Optional[float]:
    """读取 ``("const", w)`` 形式的窗口参数；不是常量位则返回 None。

    ``random_expr`` 产出的二元算子既可能是 ``("ts_min", a, b)``（第 2 位是子表达式），
    也可能是带窗口位的形态 —— 直接 ``float(node[idx][1])`` 会在前者上抛 TypeError，
    因此必须先确认该位是常量元组。
    """
    if len(node) > idx:
        arg = node[idx]
        if isinstance(arg, tuple) and arg and arg[0] == "const":
            try:
                return float(arg[1])
            except (TypeError, ValueError):
                return None
    return None


def tree_features(expr: Any) -> Dict[str, float]:
    """表达式的低成本结构特征（供延续价值的代理模型使用）。"""
    windows: List[float] = []

    def _walk(e: Any) -> None:
        if not isinstance(e, tuple) or not e:
            return
        kind = e[0]
        if kind in _WINDOWED_UNARY:
            w = _window_value(e, 2)
            if w is not None:
                windows.append(w)
        if kind in _TERNARY:
            w = _window_value(e, 3)
            if w is not None:
                windows.append(w)
        if kind in _PARAM:
            windows.append(float(param_ops.node_params(e).max_lag))
        for i in _child_positions(kind, e):
            _walk(e[i])

    _walk(expr)
    return {
        "depth": float(tree_depth(expr)),
        "nodes": float(count_nodes(expr)),
        "n_windows": float(len(windows)),
        "mean_window": float(np.mean(windows)) if windows else 0.0,
        "uses_param_op": 1.0 if contains_param_op(expr) else 0.0,
    }


def _nanmean(values: Any) -> float:
    """忽略非有限值的均值；全空时返回 NaN 而**不**触发 numpy 空切片告警。

    项目 ``pytest.ini`` 设了 ``filterwarnings = error``，"Mean of empty slice" 这类
    告警会直接把测试打成失败，因此所有聚合都必须走这个安全版本。
    """
    a = np.asarray(values, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def _py(o: Any) -> Any:
    """把 numpy 标量递归转成 Python 原生类型（供 JSON / UI 消费）。"""
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.floating):
        v = float(o)
        return None if not np.isfinite(v) else round(v, 6)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ---------------------------------------------------------------------------
# 距离与价值函数（论文 Step 2 的两个判据）
# ---------------------------------------------------------------------------
def hausdorff_1d(a: Any, b: Any) -> float:
    """一维点集之间的精确 Hausdorff 距离。

    ``H(A,B) = max{ sup_{x∈A} inf_{y∈B} |x−y| , sup_{y∈B} inf_{x∈A} |x−y| }``。
    一维情形可用排序 + 二分精确求出，无需近似 —— 论文用它度量"相邻两段时间上
    经验状态域（此处即因子暴露域）的漂移程度"。
    """
    x = np.sort(np.asarray(a, dtype=float).reshape(-1))
    y = np.sort(np.asarray(b, dtype=float).reshape(-1))
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")

    def _directed(u: np.ndarray, v: np.ndarray) -> float:
        idx = np.searchsorted(v, u)
        lo = np.clip(idx - 1, 0, v.size - 1)
        hi = np.clip(idx, 0, v.size - 1)
        return float(np.minimum(np.abs(u - v[lo]), np.abs(u - v[hi])).max())

    return max(_directed(x, y), _directed(y, x))


def exposure_edges(f_a: Any, f_b: Any, n_buckets: int) -> np.ndarray:
    """用相邻两段的合并暴露样本定分桶边界，保证两段落在**同一网格**上。"""
    both = np.concatenate([np.asarray(f_a, dtype=float).reshape(-1),
                           np.asarray(f_b, dtype=float).reshape(-1)])
    both = both[np.isfinite(both)]
    if both.size < n_buckets + 1:
        return np.array([])
    edges = np.unique(np.quantile(both, np.linspace(0, 1, n_buckets + 1)))
    if edges.size < 3:
        return np.array([])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def value_profile(f: Any, y: Any, edges: np.ndarray,
                  min_count: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """价值函数 χ(f) ≈ E[y | f 落在第 k 桶]（论文价值函数的非参数实现）。

    Returns:
        ``(profile, counts)``：长度等于桶数；样本不足的桶为 NaN。
    """
    f = np.asarray(f, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n_b = max(0, edges.size - 1)
    profile = np.full(n_b, np.nan)
    counts = np.zeros(n_b, dtype=float)
    if n_b == 0 or f.size != y.size:
        return profile, counts
    ok = np.isfinite(f) & np.isfinite(y)
    if not ok.any():
        return profile, counts
    idx = np.clip(np.digitize(f[ok], edges[1:-1], right=False), 0, n_b - 1)
    yy = y[ok]
    cnt = np.bincount(idx, minlength=n_b).astype(float)
    ssum = np.bincount(idx, weights=yy, minlength=n_b)
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(cnt >= min_count, ssum / np.maximum(cnt, 1.0), np.nan)
    return profile, cnt


def profile_msd(p_a: np.ndarray, p_b: np.ndarray, c_a: np.ndarray, c_b: np.ndarray,
                min_count: int = 3) -> float:
    """两个价值函数在**共同支撑**上的均方差（论文 MSD 的实现）。

    只在两段都够样本的桶上比较 —— 论文也是在"凸包交集"上计算，否则用无支撑的桶
    去比较等价于比较噪声。
    """
    if p_a.size == 0 or p_a.size != p_b.size:
        return float("nan")
    shared = (c_a >= min_count) & (c_b >= min_count) & np.isfinite(p_a) & np.isfinite(p_b)
    if not shared.any():
        return float("nan")
    d = p_a[shared] - p_b[shared]
    return float(np.mean(d * d))


# ---------------------------------------------------------------------------
# 配置与结果结构
# ---------------------------------------------------------------------------
@dataclass
class ScaleSpec:
    """单个尺度的资源分配（论文里的 δ / M / 网络复杂度）。"""

    freq: str = "M"          # 粗尺度评估粒度（"M" 月 / "W" 周 / "D" 日）
    name: str = "coarse"
    generations: int = 3
    pop_size: int = 24
    elite: int = 4
    max_depth: int = MAX_DEPTH

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "freq": self.freq, "generations": self.generations,
                "pop_size": self.pop_size, "elite": self.elite, "max_depth": self.max_depth}


@dataclass
class IntervalStat:
    """区间诊断（论文 Step 2 的输出）。"""

    index: int
    label: str
    start: str
    end: str
    n_dates: int
    ic: float = float("nan")
    d_h: float = float("nan")
    msd: float = float("nan")
    score: float = float("nan")
    eligible: bool = False
    selected: bool = False
    ic_refined: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "label": self.label, "start": self.start,
                "end": self.end, "n_dates": self.n_dates, "ic": self.ic, "d_h": self.d_h,
                "msd": self.msd, "score": self.score, "eligible": self.eligible,
                "selected": self.selected, "ic_refined": self.ic_refined}


@dataclass
class ResourceReport:
    """资源报告（论文 Theorem 3.1 效率增益的可核对版本）。"""

    coarse_evals: int = 0
    fine_evals: int = 0
    brute_force_evals: int = 0
    selected_intervals: int = 0
    total_intervals: int = 0
    coarse_dates: int = 0
    fine_dates: int = 0

    @property
    def speedup(self) -> float:
        """相对"每个区间都独立做细尺度演化"的评估次数倍数。"""
        used = self.coarse_evals + self.fine_evals
        if used <= 0 or self.brute_force_evals <= 0:
            return float("nan")
        return float(self.brute_force_evals) / float(used)

    @property
    def eval_saving(self) -> float:
        sp = self.speedup
        return float(1.0 - 1.0 / sp) if np.isfinite(sp) and sp > 0 else float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return _py({
            "coarse_evals": self.coarse_evals,
            "fine_evals": self.fine_evals,
            "brute_force_evals": self.brute_force_evals,
            "selected_intervals": self.selected_intervals,
            "total_intervals": self.total_intervals,
            "coarse_dates": self.coarse_dates,
            "fine_dates": self.fine_dates,
            "speedup": self.speedup,
            "eval_saving": self.eval_saving,
        })


# ---------------------------------------------------------------------------
# 遗传算子（真正的子树交叉 + 深度受控变异）
# ---------------------------------------------------------------------------
def _subtree_paths(expr: Any, path: Tuple[int, ...] = ()) -> List[Tuple[int, ...]]:
    """枚举所有子树路径（含根，根路径为空元组）。"""
    out = [path]
    if isinstance(expr, tuple) and expr:
        for i in _child_positions(expr[0], expr):
            out.extend(_subtree_paths(expr[i], path + (i,)))
    return out


def _at(expr: Any, path: Tuple[int, ...]) -> Any:
    node = expr
    for i in path:
        node = node[i]
    return node


def _replace(expr: Any, path: Tuple[int, ...], sub: Any) -> Any:
    if not path:
        return sub
    head = list(expr)
    head[path[0]] = _replace(expr[path[0]], path[1:], sub)
    return tuple(head)


def subtree_crossover(a: Any, b: Any, rng: random.Random,
                      max_depth: int = MAX_DEPTH) -> Any:
    """从 ``b`` 取一棵随机子树替换 ``a`` 的随机子树（真正的子树交叉）。

    既有实现只做"同位置子树交换"，探索能力受限；此处保留深度约束：替换后若超深，
    重试若干次，仍不行就回退到未交叉的 ``a``（宁可少一次交叉，也不产生越界个体）。
    """
    pa, pb = _subtree_paths(a), _subtree_paths(b)
    if not pa or not pb:
        return a
    for _ in range(4):
        child = _replace(a, rng.choice(pa), _at(b, rng.choice(pb)))
        if tree_depth(child) <= max_depth:
            return child
    return a


def mutate(expr: Any, rng: random.Random, max_depth: int = MAX_DEPTH,
           param_op_prob: float = 0.0) -> Any:
    """随机子树变异；以 ``param_op_prob`` 的概率把子树包进参数化时序算子。"""
    paths = _subtree_paths(expr)
    if not paths:
        return expr
    path = rng.choice(paths)
    if param_op_prob > 0 and rng.random() < param_op_prob:
        node = param_ops.make_node(_at(expr, path), param_ops.HyperbolicParams(
            theta1=rng.uniform(0.5, 1.5), theta2=rng.uniform(0.1, 1.2),
            theta3=rng.uniform(0.5, 2.0), theta4=rng.uniform(0.1, 1.0),
            theta5=rng.uniform(0.05, 0.6), max_lag=int(rng.choice(_ROL_W)),
        ))
        cand = _replace(expr, path, node)
        if tree_depth(cand) <= max_depth:
            return cand
    fresh = random_expr(rng, depth=max(0, tree_depth(expr) - 1))
    cand = _replace(expr, path, fresh)
    return cand if tree_depth(cand) <= max_depth else expr


# ---------------------------------------------------------------------------
# 求值与代码生成（支持 hwma 的"超集"求值器）
# ---------------------------------------------------------------------------
def eval_tree(expr: Any, df: pd.DataFrame) -> pd.Series:
    """表达式求值：``hwma`` 原生支持，其余算子**原样委派**给既有求值器。

    实现要点：把每个 ``hwma`` 子树先算成临时列，再把树上该位置替换成对该列的引用，
    最后整体交给 :func:`engine.genetic_enhanced.eval_expr`。这样标准算子的语义
    （除零置 NaN、窗口预热等细节）与既有实现逐位一致，不存在两套语义。
    """
    if not contains_param_op(expr):
        return eval_expr(expr, df)

    work = df
    counter = itertools.count()

    def _ensure(name: str, vals: np.ndarray) -> None:
        nonlocal work
        if work is df:
            work = df.copy()
        work[name] = vals

    def _rewrite(e: Any) -> Any:
        if param_ops.is_param_op(e):
            child = _rewrite(e[1])                      # 自底向上：子节点先落地
            child_vals = eval_expr(child, work)
            name = f"__hwma_{next(counter)}__"
            _ensure(name, param_ops.eval_node(np.asarray(child_vals, dtype=float),
                                              df["symbol"].to_numpy(), e))
            return ("col", name)
        if not contains_param_op(e):
            return e
        if isinstance(e, tuple):
            return tuple(_rewrite(x) if isinstance(x, tuple) else x for x in e)
        return e

    return eval_expr(_rewrite(expr), work)


_HWMA_HELPER = '''def _hwma(x, sym, w, b):
    """双曲加权 + tanh 饱和的滞后变换（与 engine.param_ops.transform 等价）。"""
    num = 0.0
    den = 0.0
    for k in range(1, len(w) + 1):
        c = w[k - 1] * np.tanh(b[k - 1] * x.groupby(sym).shift(k))
        ok = np.isfinite(c)
        num = num + np.where(ok, c, 0.0)
        den = den + ok.astype(float)
    return np.where(den > 0, num / np.maximum(den, 1.0), np.nan)
'''


def tree_to_code(expr: Any, name: str = "ms_gp_factor") -> str:
    """生成自包含因子代码（复用既有模板，含 shift(1) 防前视）。

    含 ``hwma`` 时：往模板里注入 ``_hwma`` 助手，再把每个参数化子树展开成
    ``df['__hwma_k'] = _hwma(...)`` 语句 —— 参数以常量内嵌，代码脱离本模块仍可复现。

    Raises:
        RuntimeError: 既有模板结构发生变化导致无法安全注入（宁可显式报错，
            也不要静默产出语义不对的代码）。
    """
    if not contains_param_op(expr):
        return expr_to_code(expr, name)

    stmts: List[str] = []
    counter = itertools.count()

    def _rewrite(e: Any) -> Any:
        if param_ops.is_param_op(e):
            child = _rewrite(e[1])
            child_str = expr_to_expr_str(child)
            var = f"__hwma_{next(counter)}"
            w, b = param_ops.code_arrays(param_ops.node_params(e))
            stmts.append(f"    df['{var}'] = _hwma(({child_str}), df['symbol'], "
                         f"np.array({w}), np.array({b}))")
            return ("col", var)
        if not contains_param_op(e):
            return e
        if isinstance(e, tuple):
            return tuple(_rewrite(x) if isinstance(x, tuple) else x for x in e)
        return e

    rewritten = _rewrite(expr)
    body = expr_to_expr_str(rewritten)
    code = expr_to_code(rewritten, name)
    needle = f"    f = {body}\n"
    if needle not in code:
        raise RuntimeError(
            "multiscale_gp.tree_to_code: 无法定位 expr_to_code 模板中的表达式语句，"
            "模板可能已变更，请同步本函数")
    code = code.replace(needle, "\n".join(stmts) + "\n" + needle, 1)
    return code.replace("def alpha_factor(df):",
                        _HWMA_HELPER + "\ndef alpha_factor(df):", 1)


# ---------------------------------------------------------------------------
# 面板辅助
# ---------------------------------------------------------------------------
def attach_forward_return(kline: pd.DataFrame, price_col: str = "close",
                          horizon: int = 1) -> pd.DataFrame:
    """按标的计算前瞻收益（严格组内 ``shift(−h)``，不会跨股票串行）。

    单独提供该辅助函数是因为：前瞻标签一旦算错（跨标的 shift、或多/少一天），
    整套挖掘结果都会变成前视污染的产物，而这类错误在 IC 上表现为"异常漂亮"，
    是最难自查的一类问题。末尾 ``horizon`` 期没有前瞻收益，置 NaN 而非 0。
    """
    df = kline.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    h = max(1, int(horizon))
    price = df[price_col].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["fwd_ret"] = price.groupby(df["symbol"]).shift(-h) / price - 1.0
    tail = df.groupby("symbol").cumcount(ascending=False) < h
    df.loc[tail.to_numpy(), "fwd_ret"] = np.nan
    return df.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------
DEFAULT_LEVELS: Tuple[ScaleSpec, ...] = (
    ScaleSpec(name="coarse", freq="M", generations=3, pop_size=24, elite=4),
    ScaleSpec(name="fine", freq="D", generations=3, pop_size=16, elite=4),
)


class HierarchicalFactorMiner:
    """分层多尺度遗传因子挖掘器。

    典型用法::

        panel = attach_forward_return(kline)
        miner = HierarchicalFactorMiner(panel, n_intervals=8, n_select=2)
        out = miner.mine()
        out["resource"]["speedup"]     # 相对暴力细网格的评估预算倍数
        out["intervals"]               # 每段的 d_H / MSD / 是否被选中
        out["candidates"][0]["code"]   # 可直接落库的因子代码
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        *,
        y_col: str = "fwd_ret",
        levels: Optional[Sequence[ScaleSpec]] = None,
        n_intervals: int = 8,
        n_select: int = 2,
        terminal_weight: float = 0.35,
        seed: int = 42,
        param_op_prob: float = 0.25,
        param_fit_rounds: int = 2,
        n_buckets: int = 10,
        min_ic_samples: int = 5,
        use_value_surrogate: bool = True,
        test_ratio: float = 0.2,
    ) -> None:
        if y_col not in panel.columns:
            raise ValueError(f"面板缺少标签列 {y_col}（可用 attach_forward_return 生成）")
        missing = {"date", "symbol"} - set(panel.columns)
        if missing:
            raise ValueError(f"面板缺少列：{sorted(missing)}")

        self.y_col = y_col
        self.panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
        self.levels = tuple(levels) if levels else DEFAULT_LEVELS
        self.n_intervals = max(2, int(n_intervals))
        self.n_select = max(1, min(int(n_select), self.n_intervals))
        self.terminal_weight = float(np.clip(terminal_weight, 0.0, 1.0))
        self.seed = int(seed)
        self.param_op_prob = float(np.clip(param_op_prob, 0.0, 1.0))
        self.param_fit_rounds = max(0, int(param_fit_rounds))
        self.n_buckets = max(3, int(n_buckets))
        self.min_ic_samples = max(2, int(min_ic_samples))
        self.use_value_surrogate = bool(use_value_surrogate)
        self.test_ratio = float(np.clip(test_ratio, 0.0, 0.5))

        df = self.panel
        self.y = df[y_col].to_numpy(dtype=float)
        self.symbols = df["symbol"].to_numpy()
        self.dates = pd.to_datetime(df["date"]).to_numpy()
        self.uniq_dates = np.sort(pd.unique(self.dates))
        self.n_dates = int(self.uniq_dates.size)
        self.date_codes = pd.Categorical(self.dates,
                                         categories=self.uniq_dates).codes.astype(np.int64)
        if self.date_codes.min() < 0:
            raise ValueError("date 列存在缺失值，无法建立截面编码")

        # 训练 / 样本外切分：区间划分、演化、代理模型全部只用训练段；
        # 测试段仅在最后用于复核，否则"选哪些区间"这一步就会把测试信息吃进去。
        n_test = int(round(self.n_dates * self.test_ratio))
        self.test_codes = np.arange(self.n_dates - n_test, self.n_dates) if n_test > 0 \
            else np.array([], dtype=int)
        self.train_codes = np.arange(0, self.n_dates - n_test)
        if self.train_codes.size < 4:
            self.train_codes = np.arange(self.n_dates)
            self.test_codes = np.array([], dtype=int)

        self._ic_cache: Dict[Any, np.ndarray] = {}
        self._f_cache: Dict[Any, np.ndarray] = {}
        self._eval_count = 0          # 真实表达式求值次数（缓存未命中才自增）
        self._surrogate: Optional[Dict[str, Any]] = None
        self._surrogate_samples: List[Tuple[Dict[str, float], float]] = []
        self._rng = random.Random(self.seed)

    # ---------------- 基础求值 ----------------
    def _factor_values(self, expr: Any) -> np.ndarray:
        """整块面板一次求值得到因子值（缓存）。"""
        key = repr(expr)
        hit = self._f_cache.get(key)
        if hit is not None:
            return hit
        vals = np.asarray(eval_tree(expr, self.panel), dtype=float).reshape(-1)
        self._eval_count += 1
        self._f_cache[key] = vals
        return vals

    def _ic_vector(self, expr: Any) -> np.ndarray:
        """逐截面 IC 向量（缓存）；与因子值共用一份去重键，故不重复计数。"""
        key = repr(expr)
        hit = self._ic_cache.get(key)
        if hit is not None:
            return hit
        fv = self._factor_values(expr)
        n = self.n_dates
        with np.errstate(invalid="ignore", divide="ignore"):
            valid = np.isfinite(fv) & np.isfinite(self.y)
            cnt = np.bincount(self.date_codes, weights=valid.astype(float), minlength=n)
            safe = np.maximum(cnt, 1.0)
            zf = np.where(valid, fv, 0.0)
            zy = np.where(valid, self.y, 0.0)
            mf = np.bincount(self.date_codes, weights=zf, minlength=n) / safe
            my = np.bincount(self.date_codes, weights=zy, minlength=n) / safe
            dx = np.where(valid, zf - mf[self.date_codes], 0.0)
            dy = np.where(valid, zy - my[self.date_codes], 0.0)
            cov = np.bincount(self.date_codes, weights=dx * dy, minlength=n)
            vx = np.bincount(self.date_codes, weights=dx * dx, minlength=n)
            vy = np.bincount(self.date_codes, weights=dy * dy, minlength=n)
            den = np.sqrt(vx * vy)
            ic = np.where((den > 0) & (cnt >= self.min_ic_samples),
                          cov / np.where(den > 0, den, 1.0), np.nan)
        self._ic_cache[key] = ic
        return ic

    def _coarse_date_codes(self, spec: ScaleSpec) -> np.ndarray:
        """粗尺度评估截面：每个粒度块取末截面（论文的粗网格点）。"""
        if not spec.freq or str(spec.freq).upper() == "D":
            return self.train_codes
        codes = period_codes(self.uniq_dates, spec.freq)
        if codes is None or codes.size == 0:
            return self.train_codes
        idx = np.unique(block_last_positions(codes))
        return np.intersect1d(idx, self.train_codes)

    def _score(self, ic: np.ndarray, idx: Optional[np.ndarray] = None,
               objective: str = "ic") -> float:
        vals = ic if idx is None else ic[idx]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return -1e9
        mean = float(vals.mean())
        if objective == "icir" and vals.size > 1:
            std = float(vals.std(ddof=1))
            if np.isfinite(std) and std > 0:
                return mean / std
        return mean

    def _interval_masks(self) -> List[np.ndarray]:
        """把**训练段**时间轴等分成 ``n_intervals`` 段，返回逐段行掩码。"""
        train = self.train_codes
        edges = np.linspace(0, train.size, self.n_intervals + 1).astype(int)
        masks: List[np.ndarray] = []
        for i in range(self.n_intervals):
            lo, hi = edges[i], edges[i + 1] - 1
            if lo > hi or lo >= train.size:
                masks.append(np.zeros(self.dates.size, dtype=bool))
                continue
            d0, d1 = self.uniq_dates[train[lo]], self.uniq_dates[train[hi]]
            masks.append((self.dates >= d0) & (self.dates <= d1))
        return masks

    def _interval_code_masks(self) -> List[np.ndarray]:
        """区间掩码的"截面码"版本（长度 ``n_dates``），供 IC 子集使用。"""
        return [self._date_mask_rows(m) for m in self._interval_masks()]

    def _date_mask_rows(self, row_mask: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_dates, dtype=bool)
        if row_mask.any():
            out[np.unique(self.date_codes[row_mask])] = True
        return out

    @staticmethod
    def _robust_scale(f: np.ndarray) -> np.ndarray:
        """全局稳健标准化 ``(f − median) / (1.4826·MAD)`` 并截尾到 ``±_Z_CLIP``。

        用全局（而非逐截面）尺度是为了让不同区间落在同一量纲上，d_H 才有可比性；
        用中位数/MAD 而非均值/标准差，是为了不让极端值主导"漂移"的度量。

        **为什么还要截尾**：MAD 是全局量，它挡不住"某个区间里出现单个近无穷值"
        （``div`` 类因子的分母贴近 0 时就会这样）。此时那个点在标准化后仍有 1e12
        量级，Hausdorff 距离会被它一个点独占，区间选择退化成"谁有异常值谁入选"。
        截尾把暴露压进有界度量空间，d_H 的上界固定为 ``2·_Z_CLIP``，从而可跨因子比较。
        """
        a = np.asarray(f, dtype=float).reshape(-1)
        finite = a[np.isfinite(a)]
        if finite.size < 3:
            return a
        med = float(np.median(finite))
        mad = float(np.median(np.abs(finite - med)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1e-12:
            std = float(finite.std())
            scale = std if std > 1e-12 else 1.0
        return np.clip((a - med) / scale, -_Z_CLIP, _Z_CLIP)

    # ---------------- 论文 Step 2：区间诊断与选择 ----------------
    def diagnose_intervals(self, exprs: Sequence[Any]) -> List[IntervalStat]:
        """计算每段的 d_H（暴露漂移）与 MSD（价值函数变化），并选出细尺度子集。"""
        masks = self._interval_masks()
        stats: List[IntervalStat] = []
        per_expr = [{"f": self._robust_scale(self._factor_values(e)),
                     "ic": self._ic_vector(e)} for e in exprs]

        for i, m in enumerate(masks):
            rows = np.where(m)[0]
            start = pd.Timestamp(self.dates[rows[0]]) if rows.size else None
            end = pd.Timestamp(self.dates[rows[-1]]) if rows.size else None
            st = IntervalStat(
                index=i,
                # 标签必须唯一：区间边界不落在自然月起点时，用"起始月"会让相邻两段同名
                label=(f"{start:%Y-%m-%d} ~ {end:%Y-%m-%d}"
                       if start is not None and end is not None else f"I{i + 1}"),
                start=start.strftime("%Y-%m-%d") if start is not None else "",
                end=end.strftime("%Y-%m-%d") if end is not None else "",
                n_dates=int(np.unique(self.dates[rows]).size) if rows.size else 0,
            )
            if rows.size == 0:
                stats.append(st)
                continue

            d_h: List[float] = []
            msd: List[float] = []
            ics: List[float] = []
            for pe in per_expr:
                f_all, ic_all = pe["f"], pe["ic"]
                ics.append(_nanmean(ic_all[self._date_mask_rows(m)]))
                if i + 1 >= len(masks) or not masks[i + 1].any():
                    continue
                m2 = masks[i + 1]
                d_h.append(hausdorff_1d(f_all[m], f_all[m2]))
                edges = exposure_edges(f_all[m], f_all[m2], self.n_buckets)
                if edges.size:
                    pa, ca = value_profile(f_all[m], self.y[m], edges, self.min_ic_samples)
                    pb, cb = value_profile(f_all[m2], self.y[m2], edges, self.min_ic_samples)
                    msd.append(profile_msd(pa, pb, ca, cb, self.min_ic_samples))
            st.ic = _nanmean(ics)
            st.d_h = _nanmean(d_h)
            st.msd = _nanmean(msd)
            st.eligible = bool(np.isfinite(st.d_h) or np.isfinite(st.msd))
            stats.append(st)

        self._select_intervals(stats)
        return stats

    def _select_intervals(self, stats: List[IntervalStat]) -> None:
        """归一化 d_H 与 MSD 合成得分并选出子集（论文 Step 2 收尾）。

        末段没有"下一段"，其 d_H/MSD 为 NaN → 标记为不可选，而不是让 0 分去和
        真实最低分竞争（那会让末段被误选）。
        """
        def _norm(x: np.ndarray) -> np.ndarray:
            ok = np.isfinite(x)
            if not ok.any():
                return np.zeros_like(x)
            lo, hi = float(np.min(x[ok])), float(np.max(x[ok]))
            if not np.isfinite(hi - lo) or hi - lo <= 1e-12:
                return np.zeros_like(x)
            return np.where(ok, (x - lo) / (hi - lo), 0.0)

        nd = _norm(np.array([s.d_h for s in stats], dtype=float))
        nm = _norm(np.array([s.msd for s in stats], dtype=float))
        for i, s in enumerate(stats):
            s.score = float(0.5 * nd[i] + 0.5 * nm[i]) if s.eligible else float("nan")

        order = sorted((s for s in stats if s.eligible), key=lambda s: (-s.score, s.index))
        for s in order[:self.n_select]:
            s.selected = True

    # ---------------- 论文 Step 1 / 3：两尺度演化 ----------------
    def _evolve(self, spec: ScaleSpec, initial: Sequence[Any], score_fn,
                fine_stage: bool = False) -> List[Any]:
        """通用演化循环：返回精英列表（``score_fn`` 越大越好）。"""
        rng = self._rng
        pop: List[Any] = list(initial)[:max(1, spec.pop_size)]
        while len(pop) < spec.pop_size:
            pop.append(random_expr(rng, depth=rng.randint(0, max(0, spec.max_depth - 1))))

        p_param = self.param_op_prob if fine_stage else self.param_op_prob * 0.5
        for _ in range(max(0, spec.generations)):
            ranked = sorted(pop, key=lambda e: -score_fn(e))
            elites = ranked[:max(1, spec.elite)]
            children = list(elites)
            while len(children) < spec.pop_size:
                if len(elites) >= 2 and rng.random() < 0.6:
                    child = subtree_crossover(rng.choice(elites), rng.choice(elites),
                                              rng, spec.max_depth)
                else:
                    child = rng.choice(elites)
                children.append(mutate(child, rng, spec.max_depth, p_param))
            pop = children[:spec.pop_size]

        ranked = sorted(pop, key=lambda e: -score_fn(e))
        return ranked[:max(1, spec.elite)]

    def _refine_params(self, expr: Any, row_mask: np.ndarray) -> Any:
        """对 ``hwma`` 节点做内层参数拟合（结构演化 × 参数拟合的双层搜索）。"""
        if not contains_param_op(expr):
            return expr

        def _walk(e: Any) -> Any:
            if param_ops.is_param_op(e):
                child = _walk(e[1])
                init = param_ops.node_params(e)
                try:
                    vals = np.asarray(eval_expr(child, self.panel), dtype=float)
                    best, _ = param_ops.fit_params(
                        vals[row_mask], self.symbols[row_mask], self.y[row_mask],
                        self.date_codes[row_mask], self.n_dates, init=init,
                        max_rounds=self.param_fit_rounds)
                except (ValueError, TypeError, FloatingPointError):
                    best = init
                return param_ops.make_node(child, best)
            if not contains_param_op(e):
                return e
            if isinstance(e, tuple):
                return tuple(_walk(x) if isinstance(x, tuple) else x for x in e)
            return e

        return _walk(expr)

    def _surrogate_fit(self) -> Optional[Dict[str, Any]]:
        """用已观测的（结构特征 → 细尺度 IC）样本拟合岭回归"价值函数"。"""
        if len(self._surrogate_samples) < 8:
            return None
        keys = sorted(self._surrogate_samples[0][0].keys())
        X = np.array([[s[0][k] for k in keys] for s in self._surrogate_samples], dtype=float)
        y = np.array([s[1] for s in self._surrogate_samples], dtype=float)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X, y = X[ok], y[ok]
        if X.shape[0] < 8:
            return None
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Xd = np.hstack([np.ones((X.shape[0], 1)), (X - mu) / sd])
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            try:
                beta = np.linalg.solve(Xd.T @ Xd + np.eye(Xd.shape[1]), Xd.T @ y)
            except np.linalg.LinAlgError:
                return None
        if not np.isfinite(beta).all():
            return None
        return {"keys": keys, "mu": mu, "sd": sd, "beta": beta}

    def _surrogate_predict(self, expr: Any) -> Optional[float]:
        if self._surrogate is None:
            self._surrogate = self._surrogate_fit()
        s = self._surrogate
        if s is None:
            return None
        feat = tree_features(expr)
        x = np.array([feat[k] for k in s["keys"]], dtype=float)
        val = float(np.concatenate([[1.0], (x - s["mu"]) / s["sd"]]) @ s["beta"])
        return val if np.isfinite(val) else None

    def _continuation(self, expr: Any, ic: np.ndarray,
                      coarse_idx: np.ndarray) -> float:
        """延续价值 χ：粗尺度价值（+ 代理模型预测）—— 论文的终端代价项。"""
        base = self._score(ic, coarse_idx)
        if not self.use_value_surrogate or len(self._surrogate_samples) < 8:
            return base
        pred = self._surrogate_predict(expr)
        return base if pred is None else 0.5 * base + 0.5 * float(pred)

    # ---------------- 主流程 ----------------
    def mine(self) -> Dict[str, Any]:
        """执行完整的分层多尺度挖掘：返回候选因子、区间诊断与资源报告。"""
        if self.n_dates < 4 or not np.isfinite(self.y).any():
            raise ValueError("面板数据过少或标签全为空，无法挖掘")

        report = ResourceReport(total_intervals=self.n_intervals)
        coarse_spec = self.levels[0]
        fine_spec = self.levels[1] if len(self.levels) > 1 else ScaleSpec(
            name="fine", freq="D", generations=coarse_spec.generations,
            pop_size=coarse_spec.pop_size, elite=coarse_spec.elite)
        coarse_idx = self._coarse_date_codes(coarse_spec)
        report.coarse_dates = int(coarse_idx.size)
        if coarse_idx.size < 2:
            coarse_idx = self.train_codes
            report.coarse_dates = int(coarse_idx.size)

        def _coarse_score(expr: Any) -> float:
            return self._score(self._ic_vector(expr), coarse_idx)

        # ---- 粗尺度演化（论文 Step 1）----
        coarse_elites = self._evolve(coarse_spec, [], _coarse_score)
        report.coarse_evals = coarse_spec.pop_size * max(1, coarse_spec.generations)

        # ---- 区间诊断与子集选择（论文 Step 2）----
        stats = self.diagnose_intervals(coarse_elites)
        selected = [s for s in stats if s.selected]
        report.selected_intervals = len(selected)
        masks = self._interval_masks()
        code_masks = self._interval_code_masks()

        # ---- 细尺度演化（论文 Step 3）----
        report.brute_force_evals = (fine_spec.pop_size * max(1, fine_spec.generations)
                                    * self.n_intervals)
        candidates: List[Dict[str, Any]] = []
        for st in selected:
            row_mask = masks[st.index]
            if not row_mask.any():
                continue
            ic_idx = code_masks[st.index]

            def _fine_score(expr: Any, idx: np.ndarray = ic_idx) -> float:
                ic = self._ic_vector(expr)
                local = self._score(ic, idx)
                cont = self._continuation(expr, ic, coarse_idx)
                return (1.0 - self.terminal_weight) * local + self.terminal_weight * cont

            seeds = [self._refine_params(e, row_mask) for e in coarse_elites]
            seeds = [mutate(e, self._rng, fine_spec.max_depth, self.param_op_prob)
                     for e in seeds]
            elites = self._evolve(fine_spec, seeds, _fine_score, fine_stage=True)
            report.fine_dates += int(np.unique(self.dates[row_mask]).size)
            report.fine_evals += fine_spec.pop_size * max(1, fine_spec.generations)

            # 代理模型的训练样本：结构特征 → 该区间上的细尺度 IC
            for e in elites:
                self._surrogate_samples.append(
                    (tree_features(e), self._score(self._ic_vector(e), ic_idx)))
            self._surrogate = None      # 新样本进来，代理模型失效重拟合

            best = elites[0] if elites else None
            if best is None:
                continue
            ic_best = self._ic_vector(best)
            st.ic_refined = self._score(ic_best, ic_idx)
            try:
                code = tree_to_code(best, "ms_gp_factor")
            except (RuntimeError, ValueError):
                code = ""
            candidates.append({
                "_expr": best,
                "expr": repr(best),
                "code": code,
                "interval": st.label,
                "interval_index": st.index,
                "ic_interval": st.ic_refined,
                "ic_interval_coarse": st.ic,
                "fitness": _fine_score(best),
                "params": param_ops.describe(best) if contains_param_op(best) else None,
                "features": tree_features(best),
                "family": _family_of(best),
            })

        # ---- 全样本 / 训练 / 样本外复核 ----
        for c in candidates:
            ic = self._ic_vector(c.pop("_expr"))
            c["ic_full"] = self._score(ic)
            c["ic_train"] = self._score(ic, self.train_codes)
            c["ic_test"] = self._score(ic, self.test_codes) if self.test_codes.size else None
            c["stats"] = ic_stats(ic)

        candidates.sort(key=lambda c: (-(c.get("fitness") or -1e9), c.get("interval_index", 0)))
        top = candidates[:max(1, min(5, len(candidates)))] if candidates else []

        return _py({
            "candidates": top,
            "intervals": [s.to_dict() for s in stats],
            "resource": report.to_dict(),
            "levels": [s.to_dict() for s in self.levels],
            "terminal_weight": self.terminal_weight,
            "n_intervals": self.n_intervals,
            "n_select": self.n_select,
            "test_ratio": self.test_ratio,
            "coarse_freq": coarse_spec.freq,
            "fine_freq": fine_spec.freq,
            "objective": "细尺度区间 IC 与粗尺度延续价值的加权",
            "total_evals": self._eval_count,   # 真实表达式求值次数（去重后）
            "train_dates": int(self.train_codes.size),
            "test_dates": int(self.test_codes.size),
        })


def _family_of(expr: Any) -> str:
    """粗粒度归类（供体系构建时分维度使用）。"""
    kinds = set()

    def _walk(e: Any) -> None:
        if not isinstance(e, tuple) or not e:
            return
        kinds.add(e[0])
        for i in _child_positions(e[0], e):
            _walk(e[i])

    _walk(expr)
    if kinds & _PARAM:
        return "时序记忆"
    if kinds & {"ts_zscore", "ts_rank", "rol", "ts_min", "ts_max"}:
        return "时序形态"
    if kinds & {"ts_corr"}:
        return "价量联动"
    if kinds & {"div", "mul"}:
        return "价量组合"
    return "截面变换"
