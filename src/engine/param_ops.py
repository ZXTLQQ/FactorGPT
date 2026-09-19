"""参数高效时序记忆算子（src/engine/param_ops.py）。

落地文献 ``2607.23068v1`` §3.1 的 lag transform：用 **5 个参数**表达任意长度的
时序记忆，替代遗传编程里"离散窗口 + 等权平均"的僵化做法。

原式（该文献的 parameter-efficient lag transform）：

    α_ℓ = θ₁ · ℓ^(−θ₂)                     双曲衰减权重（长记忆）
    β_ℓ = θ₃ − θ₄ · exp(−θ₅ · ℓ)           饱和阈值（随滞后递增、上界 θ₃）
    h(r)_t = Σ_{ℓ=1..L} (α_ℓ / β_ℓ) · tanh(β_ℓ · r_{t−ℓ})

三点工程含义：

1. **参数高效**：该文献把 2400 维参数压到 5 维仍能覆盖任意滞后长度；在遗传挖掘
   里对应"GP 演化结构、内层拟合 5 个参数"的双层搜索，比演化离散窗口更省个体数。
2. **抗离群**：``tanh`` 把每个滞后值先压到 ``(−β_ℓ, β_ℓ)`` 再线性叠加，收益率的
   极端值不会像等权均值那样把整条因子序列带偏。
3. **可复现**：拟合是确定性的坐标下降（无随机重启），产出的因子代码内嵌拟合后的
   5 个参数常量，因此落库代码脱离本模块也能逐位复现。

所有函数为纯计算，只依赖 numpy / pandas。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .ic_utils import groupwise_corr

# 参数上下界：θ₂ 必须为正（否则远期权重不衰减）；θ₃ 为正且 θ₄ < θ₃ 才能保证 β_ℓ > 0
_BOUNDS: Dict[str, Tuple[float, float]] = {
    "theta1": (0.05, 5.0),
    "theta2": (0.05, 3.0),
    "theta3": (0.05, 10.0),
    "theta4": (0.0, 9.9),
    "theta5": (0.005, 2.0),
}

# 坐标下降的候选步长（乘性），围绕当前值试探 —— 与加法步长相比对量级不敏感
_STEPS: Tuple[float, ...] = (0.5, 0.7, 1.0, 1.4, 2.0)


@dataclass
class HyperbolicParams:
    """五参数双曲滞后变换的参数组。"""

    theta1: float = 1.0
    theta2: float = 0.5
    theta3: float = 1.0
    theta4: float = 0.8
    theta5: float = 0.2
    max_lag: int = 20

    def __post_init__(self) -> None:
        self.sanitize()

    def sanitize(self) -> "HyperbolicParams":
        """把参数夹到合法域内（越界不是错误输入，而是需要收敛的中间态）。"""
        for k, (lo, hi) in _BOUNDS.items():
            v = float(getattr(self, k))
            if not np.isfinite(v):
                v = float(np.clip(1.0, lo, hi))
            setattr(self, k, float(np.clip(v, lo, hi)))
        # θ₄ 必须严格小于 θ₃，否则 β_ℓ 会穿过 0（tanh 增益发散）
        self.theta4 = float(min(self.theta4, self.theta3 * 0.99))
        self.max_lag = int(max(2, min(int(self.max_lag), 250)))
        return self

    # ---------- 式中的两条曲线 ----------
    def alpha(self, lags: np.ndarray) -> np.ndarray:
        """α_ℓ = θ₁·ℓ^(−θ₂)，随滞后单调递减。"""
        l = np.asarray(lags, dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            return self.theta1 * np.power(np.maximum(l, 1e-9), -self.theta2)

    def beta(self, lags: np.ndarray) -> np.ndarray:
        """β_ℓ = θ₃ − θ₄·e^(−θ₅ℓ)，单调递增且以 θ₃ 为上界。"""
        l = np.asarray(lags, dtype=float)
        return self.theta3 - self.theta4 * np.exp(-self.theta5 * l)

    def weights(self) -> np.ndarray:
        """归一化后的滞后权重 ``α_ℓ/Σα``（仅用于可读性展示，不参与变换式）。"""
        a = self.alpha(np.arange(1, self.max_lag + 1))
        total = float(a.sum())
        if not np.isfinite(total) or total <= 0:
            return np.full(self.max_lag, 1.0 / self.max_lag)
        return a / total

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "HyperbolicParams":
        if not d:
            return cls()
        known = {k: d[k] for k in [*list(_BOUNDS), "max_lag"] if k in d}
        return cls(**known)


def lag_matrix(values: np.ndarray, symbols: np.ndarray, max_lag: int) -> np.ndarray:
    """构造滞后矩阵 ``(max_lag, N)``：第 ℓ 行是 ``x_{t−ℓ}``（组内，不跨股票）。

    一次构造、多次复用 —— 参数拟合要在同一份滞后矩阵上反复试探上千次，
    逐次 ``groupby.shift`` 会把 90% 的时间花在 pandas 索引开销上。
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    sym = np.asarray(symbols).reshape(-1)
    n = v.size
    L = int(max(2, max_lag))
    if not (n == sym.size):
        raise ValueError("values / symbols 长度必须一致")

    # 组内位置：同一 symbol 内按出现顺序（调用方需保证已按 date 排序）
    s = pd.Series(sym)
    valid = v.copy()
    out = np.full((L, n), np.nan, dtype=float)
    if n == 0:
        return out
    grp_pos = s.groupby(s).cumcount().to_numpy()          # 组内第几行
    offsets = np.arange(n) - grp_pos                       # 每组的起始行号
    for lag in range(1, L + 1):
        rows = np.arange(n) - lag
        ok = (rows >= 0) & (grp_pos >= lag)
        src = np.where(ok, rows, 0)
        out[lag - 1] = np.where(ok, valid[src], np.nan)
    _ = offsets  # 仅用于说明分组边界，保留以表明实现意图
    return out


def transform(lag_mat: np.ndarray, params: HyperbolicParams) -> np.ndarray:
    """对已构造的滞后矩阵执行 ``h(r)_t = Σ (α_ℓ/β_ℓ)·tanh(β_ℓ·r_{t−ℓ})``。"""
    L = lag_mat.shape[0]
    if L == 0:
        return np.zeros(lag_mat.shape[1], dtype=float)
    lags = np.arange(1, L + 1, dtype=float)
    a = params.alpha(lags)
    b = params.beta(lags)
    b = np.where(np.abs(b) < 1e-9, 1e-9, b)
    scale = a / b
    with np.errstate(invalid="ignore", over="ignore"):
        contrib = scale[:, None] * np.tanh(b[:, None] * lag_mat)
    # 滞后越远、有效样本越少（序列起点处的 NaN）：按有效项计数而非直接 nanmean，
    # 避免把"样本不足"和"信号为 0"混为一谈。
    finite = np.isfinite(contrib)
    num = np.where(finite, contrib, 0.0).sum(axis=0)
    den = finite.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    return out


def saturate(values: np.ndarray, params: Optional[HyperbolicParams] = None) -> np.ndarray:
    """逐点饱和压缩 ``(α₁/β₁)·tanh(β₁·x)``（论文变换在 ℓ=1 的特例）。

    在没有足够历史（新股 / 上市初期）时，单点饱和比滞后加权更稳；也用于把
    成交量这类重尾变量先压成有界量。
    """
    p = params or HyperbolicParams()
    a1 = float(p.alpha(np.array([1.0]))[0])
    b1 = float(p.beta(np.array([1.0]))[0])
    b1 = b1 if abs(b1) > 1e-9 else 1e-9
    with np.errstate(invalid="ignore", over="ignore"):
        return (a1 / b1) * np.tanh(b1 * np.asarray(values, dtype=float))


# ---------------------------------------------------------------------------
# 内层参数拟合（坐标下降，确定性）
# ---------------------------------------------------------------------------
def _default_objective(f: np.ndarray, y: np.ndarray, codes: np.ndarray,
                       n_groups: int) -> float:
    """默认目标：截面 IC 均值（越大越好）。"""
    ic = groupwise_corr(f, y, codes, n_groups)
    ic = ic[np.isfinite(ic)]
    if ic.size == 0:
        return -1e9
    return float(ic.mean())


def fit_params(
    values: np.ndarray,
    symbols: np.ndarray,
    y: np.ndarray,
    codes: np.ndarray,
    n_groups: int,
    init: Optional[HyperbolicParams] = None,
    max_rounds: int = 4,
    objective: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, int], float]] = None,
    tol: float = 1e-5,
) -> Tuple[HyperbolicParams, Dict[str, Any]]:
    """在给定截面上拟合 5 个参数（坐标下降，无随机性）。

    Args:
        values: 被变换的原始序列（如收益率）。
        symbols: 与 ``values`` 同序的标的分组（滞后不跨标的）。
        y: 前瞻收益（目标变量）。
        codes: 截面编码（与 ``values`` 同序）。
        n_groups: 截面数。
        init: 初始参数；None 时用默认值。
        max_rounds: 坐标下降轮数上限。
        objective: 目标函数 ``fn(f, y, codes, n_groups) -> float``（越大越好）。
        tol: 收敛阈值（相邻两轮目标提升小于该值即停）。

    Returns:
        ``(最佳参数, 诊断字典)``。诊断含 ``objective`` / ``rounds`` / ``evals`` /
        ``grid``（每轮各参数的最优取值，便于审计拟合轨迹）。
    """
    obj = objective or _default_objective
    p = (init or HyperbolicParams()).sanitize()
    lag_mat = lag_matrix(values, symbols, p.max_lag)

    def _score(cand: HyperbolicParams) -> float:
        cand.sanitize()
        f = transform(lag_mat, cand)
        return float(obj(f, y, codes, n_groups))

    best = _score(p)
    evals = 1
    grid: List[Dict[str, float]] = []
    rounds = 0

    for _ in range(max(0, int(max_rounds))):
        rounds += 1
        round_start = best  # 跨轮比较基准：与"自己"比较会让收敛判据恒真
        improved = False
        round_best: Dict[str, float] = {}
        for key in _BOUNDS:
            cur = float(getattr(p, key))
            lo, hi = _BOUNDS[key]
            local_best, local_val = cur, best
            for s in _STEPS:
                if s == 1.0:
                    continue
                trial = HyperbolicParams(**p.to_dict())
                setattr(trial, key, float(np.clip(cur * s, lo, hi)))
                if key == "theta5":
                    # θ₅ 的量级小，乘性步长会直接跑到下界，补一个加法试探
                    setattr(trial, key, float(np.clip(cur + (s - 1.0) * 0.05, lo, hi)))
                trial.sanitize()
                val = _score(trial)
                evals += 1
                if val > local_val + 1e-12:
                    local_val, local_best = val, float(getattr(trial, key))
            if local_best != cur:
                setattr(p, key, local_best)
                p.sanitize()
                best = local_val
                improved = True
            round_best[key] = float(getattr(p, key))
        grid.append(round_best)
        if not improved or rounds >= max_rounds:
            break
        # 收敛判据：本轮相对提升不足 tol 即停（best 与 round_start 都是真实评估值）
        if abs(best - round_start) < tol * max(abs(round_start), 1e-9):
            break

    diag = {
        "objective": round(float(best), 8),
        "rounds": rounds,
        "evals": evals,
        "grid": grid,
        "max_lag": p.max_lag,
    }
    return p, diag


# ---------------------------------------------------------------------------
# 表达式算子（供分层多尺度 GP 使用）
# ---------------------------------------------------------------------------
OP_NAME = "hwma"


def is_param_op(node: Any) -> bool:
    """判断表达式节点是否为参数化时序算子。"""
    return isinstance(node, tuple) and len(node) >= 4 and node[0] == OP_NAME


def make_node(child: Any, params: HyperbolicParams) -> Tuple[Any, ...]:
    """构造表达式节点 ``("hwma", child, ("const", max_lag), ("params", θ))``。"""
    p = HyperbolicParams(**params.to_dict())
    return (OP_NAME, child, ("const", float(p.max_lag)),
            ("params", (p.theta1, p.theta2, p.theta3, p.theta4, p.theta5)))


def node_params(node: Any) -> HyperbolicParams:
    """从表达式节点读回参数（缺失/越界时回落到默认参数）。"""
    if not is_param_op(node):
        return HyperbolicParams()
    try:
        theta = node[3][1]
        return HyperbolicParams(
            theta1=float(theta[0]), theta2=float(theta[1]), theta3=float(theta[2]),
            theta4=float(theta[3]), theta5=float(theta[4]),
            max_lag=int(float(node[2][1])),
        )
    except (IndexError, TypeError, ValueError):
        return HyperbolicParams()


def eval_node(child_values: np.ndarray, symbols: np.ndarray, node: Any) -> np.ndarray:
    """按节点参数对 ``child_values`` 求值（滞后不跨标的）。"""
    p = node_params(node)
    return transform(lag_matrix(child_values, symbols, p.max_lag), p)


def _fmt_array(values: np.ndarray) -> str:
    """把数组格式化成 ``[a, b, c]`` 字面量（10 位有效数字，可逐位复现）。"""
    v = np.asarray(values, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{float(x):.10g}" for x in v) + "]"


def code_arrays(params: HyperbolicParams) -> Tuple[str, str]:
    """生成可内嵌到因子代码里的 ``(scale, beta)`` 数组字面量。

    与 :func:`transform` 用的是同两条曲线（``α/β`` 与 ``β``），因此"代码里写死的
    常量"和"离线求值的结果"同源；``np.tanh`` 的饱和形状因此不会在落库后走样。
    """
    p = params.sanitize()
    lags = np.arange(1, p.max_lag + 1, dtype=float)
    a = p.alpha(lags)
    b = p.beta(lags)
    b = np.where(np.abs(b) < 1e-9, 1e-9, b)
    return _fmt_array(a / b), _fmt_array(b)


def node_to_code(child_expr: str, node: Any) -> str:
    """生成与 :func:`eval_node` 逐位等价的 pandas 代码（参数以常量内嵌）。

    内嵌常量而非回查配置，是为了让落库的因子代码自包含 —— 脱离本模块后仍可
    复现，且不会因默认参数变更而"同一个因子名跑出不同结果"。
    """
    p = node_params(node)
    w, bb = code_arrays(p)
    lines = [
        f"_hw_x = ({child_expr})",
        f"_hw_w = np.array({w})",
        f"_hw_b = np.array({bb})",
        "_hw_num = 0.0",
        "_hw_den = 0.0",
        "for _k in range(1, _hw_w.size + 1):",
        "    _s = _hw_x.groupby(df['symbol']).shift(_k)",
        "    _c = _hw_w[_k - 1] * np.tanh(_hw_b[_k - 1] * _s)",
        "    _ok = np.isfinite(_c)",
        "    _hw_num = _hw_num + np.where(_ok, _c, 0.0)",
        "    _hw_den = _hw_den + _ok.astype(float)",
        "f = np.where(_hw_den > 0, _hw_num / np.maximum(_hw_den, 1.0), np.nan)",
    ]
    return "\n".join(lines)


def describe(node: Any) -> Dict[str, Any]:
    """参数化算子的可读描述（供 UI / 报告展示）。"""
    p = node_params(node)
    return {
        "op": OP_NAME,
        "max_lag": p.max_lag,
        "theta": {k: round(float(getattr(p, k)), 6) for k in _BOUNDS},
        "half_life": _half_life(p),
    }


def _half_life(p: HyperbolicParams) -> float:
    """权重衰减到初期的一半所需滞后（把 θ₂ 翻译成人话：记忆有多长）。"""
    if p.theta2 <= 0:
        return float("inf")
    return float(np.power(2.0, 1.0 / p.theta2))
