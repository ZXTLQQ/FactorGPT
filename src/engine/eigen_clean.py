"""因子体系的相关矩阵谱清洗与权重求解（src/engine/eigen_clean.py）。

对应论文 2607.23068v1（神经网络协方差谱清洗 + 端到端 GMV）在「因子体系构建」上的落地。

论文的出发点：样本相关矩阵的特征谱里混着大量**纯噪声方向**，直接把它交给组合优化器，
优化器会把仓位压到"看起来独立、其实只是噪声"的方向上 —— 样本内方差很低、样本外更差。
解法是先把谱清洗干净，再在清洗后的矩阵上端到端求解权重。落到本项目，链路拆成四步，
每一步都可以单独检验：

1. :func:`mp_bounds`             —— Marchenko–Pastur 边界：给出"哪几个特征值纯属噪声"的判据
2. :func:`analyze_spectrum`      —— 谱清洗（噪声特征值抬到噪声均值 / 向单位阵线性收缩）
3. :func:`optimize_weights`      —— 清洗后的矩阵上求权重（最小方差 / 最大分散 / 风险平价 / ICIR 倾斜）
4. :func:`marginal_contributions`、:func:`addition_ranking`
                                 —— 把体系方差按因子拆开，回答"再加一个因子值不值"

设计约定
--------
* 因子矩阵已在逐日截面上做过标准化，因此**相关矩阵即协方差矩阵**（对角为 1）；体系方差
  ``σ² = wᵀRw`` 就是合成因子的横截面方差，不需要额外的量纲转换。
* 权重约束固定为 ``Σw = 1, w ≥ 0``：本项目产出的是选股评分，不是可做空的多空组合。
* 求解优先用 ``scipy.optimize.minimize(SLSQP)``；scipy 缺失时退化为解析解 + 投影，
  保证离线环境下仍能给出可用权重而不是直接报错。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# 权重方案标识（与 factor_system.resolve_weights 的既有方案并存）
W_EQUAL = "equal"
W_MIN_VARIANCE = "min_variance"
W_MAX_DIVERSIFICATION = "max_diversification"
W_RISK_PARITY = "risk_parity"
W_ICIR_TILT = "icir_tilt"

SPECTRAL_MODES: Tuple[str, ...] = (W_MIN_VARIANCE, W_MAX_DIVERSIFICATION, W_RISK_PARITY)
SPECTRAL_LABELS: Dict[str, str] = {
    W_EQUAL: "等权重（基准）",
    W_MIN_VARIANCE: "最小方差（谱清洗后）",
    W_MAX_DIVERSIFICATION: "最大分散化（谱清洗后）",
    W_RISK_PARITY: "风险平价（等风险贡献）",
    W_ICIR_TILT: "ICIR 倾斜（谱清洗后最大化 IR）",
}

# 相关系数矩阵的最小特征值下限：清洗后仍可能出现 -1e-17 级负根，统一抬到该值再重建成矩阵
_EIG_FLOOR = 1e-10


# ---------------------------------------------------------------------------
# 1. Marchenko–Pastur 边界
# ---------------------------------------------------------------------------
def mp_bounds(
    n_obs: int,
    n_factors: int,
    sigma2: float = 1.0,
) -> Tuple[float, float]:
    """Marchenko–Pastur 噪声带上下界 ``(λ₋, λ₊)``。

    设噪声矩阵形状为 ``N × T``（N = 因子数，T = 有效观测截面数），纵横比 ``q = N / T``。
    当 N、T 同阶增长而 q 固定时，样本特征值几乎必然落在

        ``λ₊ = σ²(1 + √q)²``，``λ₋ = σ²(1 − √q)²``（q < 1；q ≥ 1 时 λ₋ = 0）

    落在这个区间**内**的特征值无法与纯噪声区分；只有超过 λ₊ 的方向才是真实信号。
    本项目把"低于 λ₋ 的特征值"视为噪声主导方向：它们代表相关矩阵估计误差，而不是
    因子之间的真实弱相关。

    Args:
        n_obs: 有效观测数 T（逐日截面数量，即相关系数矩阵估计用的样本量）。
        n_factors: 因子数 N。
        sigma2: 噪声方差；相关系数矩阵取 1（对角元素均值）。

    Returns:
        ``(λ₋, λ₊)``。

    Raises:
        ValueError: ``n_obs`` 或 ``n_factors`` 非正。
    """
    if n_obs <= 0 or n_factors <= 0:
        raise ValueError("n_obs 与 n_factors 必须为正整数")
    q = float(n_factors) / float(n_obs)
    root = math.sqrt(q)
    lam_plus = float(sigma2) * (1.0 + root) ** 2
    lam_minus = float(sigma2) * (1.0 - root) ** 2 if q < 1.0 else 0.0
    return lam_minus, lam_plus


def _psd_repair(mat: np.ndarray, floor: float = _EIG_FLOOR) -> np.ndarray:
    """把矩阵修回对称半正定：负特征值抬到 ``floor``，再按原对角元素重标定。

    谱清洗后重建的矩阵理论上半正定，但浮点误差会带来 ``-1e-17`` 级的负根，
    直接丢给下游求解器会得到"协方差非正定"的失败信息（而且不同 BLAS 下时有时无）。
    """
    a = np.asarray(mat, dtype=float)
    a = (a + a.T) / 2.0
    vals, vecs = np.linalg.eigh(a)
    if vals.min() < floor:
        vals = np.clip(vals, floor, None)
        a = (vecs * vals) @ vecs.T
        a = (a + a.T) / 2.0
    d = np.sqrt(np.clip(np.diag(a), floor, None))
    a = a / np.outer(d, d)          # 重标定回单位对角（相关矩阵的硬约束）
    np.fill_diagonal(a, 1.0)
    return np.clip(a, -1.0, 1.0)


def _as_square(mat: Any, default_diag: float = 1.0) -> np.ndarray:
    """把入参（DataFrame / 嵌套列表 / ndarray）转成对称方阵并补齐对角。"""
    a = mat.to_numpy(dtype=float) if isinstance(mat, pd.DataFrame) else np.asarray(mat, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"需要方阵，实际形状 {getattr(a, 'shape', None)}")
    a = np.where(np.isfinite(a), a, 0.0)
    a = (a + a.T) / 2.0
    np.fill_diagonal(a, np.where(np.diag(a) > 0, np.diag(a), default_diag))
    return a


# ---------------------------------------------------------------------------
# 2. 谱清洗
# ---------------------------------------------------------------------------
@dataclass
class SpectrumReport:
    """相关矩阵谱清洗的诊断结果（可 JSON 化）。

    Attributes:
        method: 清洗方式，``mp`` / ``shrink`` / ``mp_shrink`` / ``none``。
        n_factors/n_obs: 因子数与有效观测数（决定 MP 边界）。
        lambda_minus/lambda_plus: MP 噪声带上下界。
        eigenvalues/eigenvalues_clean: 清洗前后的特征值（降序）。
        eigenvectors: 特征向量（列向量，与特征值同序）。
        corr_clean: 清洗并修复后的相关矩阵。
        noise_dim: 被判定为噪声方向的数量。
        shrink: 线性收缩强度（``shrink`` 类方法使用）。
        cond_before/cond_after: 条件数（λmax/λmin，带下限保护）。
        effective_factors_before/after: 参与度口径的有效因子数 ``(Σλ)²/Σλ²``。
        notes: 面向人读的结论片段。
    """

    method: str
    n_factors: int
    n_obs: int
    lambda_minus: float
    lambda_plus: float
    eigenvalues: np.ndarray
    eigenvalues_clean: np.ndarray
    eigenvectors: np.ndarray
    corr_clean: np.ndarray
    noise_dim: int
    shrink: float
    cond_before: float
    cond_after: float
    effective_factors_before: float
    effective_factors_after: float
    names: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # -- 便捷视图 ----------------------------------------------------------
    @property
    def noise_ratio(self) -> float:
        """噪声方向占比。"""
        return float(self.noise_dim) / float(self.n_factors) if self.n_factors else 0.0

    @property
    def n_inband(self) -> int:
        """落在噪声带**内部**的方向数（``λ₋ ≤ λ ≤ λ₊``）。

        这些方向的地位是"统计上无法判定"：既不能像 ``λ < λ₋`` 那样断言是噪声，
        也够不到 ``λ > λ₊`` 的信号门槛。把它们单列出来，是为了不让报告出现
        "非噪声即信号"的假二分 —— 说清有多少方向拿不准，比给一个精确的
        "信号数"更有用。
        """
        if self.eigenvalues.size == 0:
            return 0
        return int(np.sum((self.eigenvalues >= self.lambda_minus)
                          & (self.eigenvalues <= self.lambda_plus)))

    @property
    def n_signal(self) -> int:
        """超出噪声带上界的方向数（``λ > λ₊``）。"""
        return int(np.sum(self.eigenvalues > self.lambda_plus))

    @property
    def inband_ratio(self) -> float:
        """无法判定的方向占比。"""
        return float(self.n_inband) / float(self.n_factors) if self.n_factors else 0.0

    @property
    def signal_eigenvalues(self) -> np.ndarray:
        """超出 MP 上界、可判为真实信号的特征值。"""
        return self.eigenvalues[self.eigenvalues > self.lambda_plus]

    @property
    def corr_clean_df(self) -> pd.DataFrame:
        """清洗后的相关矩阵（带因子名）。"""
        idx = self.names or list(range(self.n_factors))
        return pd.DataFrame(self.corr_clean, index=idx, columns=idx)

    def equal_weight_var(self, cleaned: bool = True) -> float:
        """等权组合的体系方差 ``wᵀRw``（``w = 1/N``）。

        **注意方向**：清洗把噪声方向的特征值从"接近 0"抬到噪声均值，等于把样本相关
        矩阵里那些"虚假的独立方向"补回来了，所以等权方差通常**上升**（本例 0.028→0.031）。
        反过来读也成立：样本矩阵给出的低成本、高分散化，有相当一部分是估计误差送的。
        诊断时应该看的是"清洗前后差多少"，而不是"清洗后更小"。
        """
        if self.n_factors < 1:
            return float("nan")
        w = np.full(self.n_factors, 1.0 / self.n_factors)
        vals = self.eigenvalues_clean if cleaned else self.eigenvalues
        return float(w @ ((self.eigenvectors * vals) @ self.eigenvectors.T) @ w)

    def as_dict(self) -> Dict[str, Any]:
        """转成可直接放进 JSON / Streamlit session 的普通字典。"""
        return {
            "method": self.method,
            "n_factors": self.n_factors,
            "n_obs": self.n_obs,
            "q": (self.n_factors / self.n_obs) if self.n_obs else float("nan"),
            "lambda_minus": self.lambda_minus,
            "lambda_plus": self.lambda_plus,
            "eigenvalues": [float(v) for v in self.eigenvalues],
            "eigenvalues_clean": [float(v) for v in self.eigenvalues_clean],
            "noise_dim": self.noise_dim,
            "noise_ratio": self.noise_ratio,
            "n_inband": self.n_inband,
            "inband_ratio": self.inband_ratio,
            "n_signal": self.n_signal,
            "shrink": self.shrink,
            "cond_before": self.cond_before,
            "cond_after": self.cond_after,
            "effective_factors_before": self.effective_factors_before,
            "effective_factors_after": self.effective_factors_after,
            "names": list(self.names),
            "notes": list(self.notes),
        }

    def summary_lines(self) -> List[str]:
        """紧凑文本摘要（供 AI 咨询窗口拼上下文用）。"""
        return list(self.notes)


def _effective_factors(vals: np.ndarray) -> float:
    """参与度（participation ratio）口径的有效因子数 ``(Σλ)²/Σλ²``。

    比"方差解释率平方和倒数"更稳健：它等价于"谱的均匀程度"，值域 [1, N]，
    清洗前后可比。
    """
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    denom = float(np.sum(v ** 2))
    if denom <= 0:
        return 0.0
    return float(v.sum() ** 2 / denom)


def _cond(vals: np.ndarray) -> float:
    """条件数 ``λmax / λmin``（用绝对值并加下限，避免 0 除）。"""
    v = np.abs(np.asarray(vals, dtype=float))
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    lo = max(float(v.min()), _EIG_FLOOR)
    return float(v.max() / lo)


def analyze_spectrum(
    corr: Any,
    n_obs: int,
    method: str = "mp",
    shrink: float = 0.2,
    sigma2: Optional[float] = None,
    names: Optional[Sequence[str]] = None,
) -> SpectrumReport:
    """对因子相关矩阵做谱清洗，返回诊断报告。

    三种清洗方式：

    * ``mp``：把 ``λ < λ₋`` 的特征值统一抬到这些噪声特征值的均值 —— 保留矩阵迹
      （总方差不变），只把"噪声方向之间的强弱差异"抹平。论文里最小侵入的一档。
    * ``shrink``：向平均特征值线性收缩 ``λ ← (1−δ)λ + δ·λ̄``，等价于
      ``R ← (1−δ)R + δ·λ̄·I``（Ledoit–Wolf 式）。对**所有**方向生效，能同时压制
      偏大的样本特征值，代价是会削弱真实信号。
    * ``mp_shrink``：先 MP 抹平噪声，再整体收缩 —— 噪声主导时最稳，信号强时最保守。

    Args:
        corr: 相关矩阵（DataFrame 或 ndarray，方阵、对角为 1）。
        n_obs: 有效观测截面数（估计该相关矩阵用了多少期样本）。
        method: ``mp`` / ``shrink`` / ``mp_shrink`` / ``none``。
        shrink: 线性收缩强度 δ ∈ [0, 1)，仅在 ``shrink`` / ``mp_shrink`` 下生效。
        sigma2: 噪声方差；默认取矩阵对角均值（相关矩阵即 1）。
        names: 因子名列表（与矩阵顺序一致），用于报告与下游展示。

    Returns:
        :class:`SpectrumReport`。
    """
    r = _as_square(corr)
    n = r.shape[0]
    method = (method or "mp").lower()
    sigma2 = float(np.mean(np.diag(r))) if sigma2 is None else float(sigma2)
    lam_minus, lam_plus = mp_bounds(int(max(n_obs, 1)), n, sigma2)

    vals, vecs = np.linalg.eigh(r)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    clean = vals.copy()

    noise_mask = vals < lam_minus
    noise_dim = int(noise_mask.sum())
    shrink = float(min(max(shrink, 0.0), 0.999))
    mean_lam = float(np.mean(vals)) if n else sigma2

    if method in ("mp", "mp_shrink") and noise_dim > 0:
        clean[noise_mask] = float(np.mean(vals[noise_mask]))
    if method in ("shrink", "mp_shrink"):
        clean = (1.0 - shrink) * clean + shrink * mean_lam
    clean = np.clip(clean, _EIG_FLOOR, None)
    # 清洗必须保持迹不变，否则"体系总方差"会被悄悄改掉，下游比较失去意义
    if n and float(clean.sum()) > 0:
        clean *= float(vals.sum()) / float(clean.sum())

    r_clean = _psd_repair((vecs * clean) @ vecs.T)

    notes: List[str] = [
        f"纵横比 q=N/T={n}/{int(max(n_obs, 1))}={n / max(n_obs, 1):.3f}，"
        f"MP 噪声带 [{lam_minus:.3f}, {lam_plus:.3f}]",
        f"谱结构：{int((vals > lam_plus).sum())} 个信号方向（超过上界）、"
        f"{int(((vals >= lam_minus) & (vals <= lam_plus)).sum())} 个带内方向（无法判定）、"
        f"{noise_dim} 个明确噪声方向（低于下界）",
    ]
    if method in ("shrink", "mp_shrink"):
        notes.append(f"线性收缩强度 δ={shrink:.2f}（向平均特征值 {mean_lam:.3f} 收缩）")
    notes.append(
        f"条件数 {_cond(vals):.1f} → {_cond(clean):.1f}，"
        f"有效因子数 {_effective_factors(vals):.2f} → {_effective_factors(clean):.2f}"
    )

    return SpectrumReport(
        method=method,
        n_factors=int(n),
        n_obs=int(max(n_obs, 1)),
        lambda_minus=float(lam_minus),
        lambda_plus=float(lam_plus),
        eigenvalues=vals,
        eigenvalues_clean=clean,
        eigenvectors=vecs,
        corr_clean=r_clean,
        noise_dim=noise_dim,
        shrink=shrink,
        cond_before=_cond(vals),
        cond_after=_cond(clean),
        effective_factors_before=_effective_factors(vals),
        effective_factors_after=_effective_factors(clean),
        names=[str(x) for x in (names if names is not None else range(n))],
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 3. 权重求解
# ---------------------------------------------------------------------------
@dataclass
class WeightSolution:
    """一组权重解及其风险/收益特征。"""

    mode: str
    weights: np.ndarray
    names: List[str] = field(default_factory=list)
    success: bool = True
    message: str = ""
    variance: float = float("nan")
    vol: float = float("nan")
    diversification_ratio: float = float("nan")
    expected_icir: float = float("nan")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "label": SPECTRAL_LABELS.get(self.mode, self.mode),
            "weights": {n: float(w) for n, w in zip(self.names, self.weights)},
            "success": self.success,
            "message": self.message,
            "variance": self.variance,
            "vol": self.vol,
            "diversification_ratio": self.diversification_ratio,
            "expected_icir": self.expected_icir,
        }


def _simplex_bounds(n: int, max_weight: Optional[float]) -> List[Tuple[float, float]]:
    hi = 1.0 if max_weight is None else float(min(max(max_weight, 1.0 / n), 1.0))
    return [(0.0, hi)] * n


def _project_simplex(v: np.ndarray, max_weight: Optional[float] = None) -> np.ndarray:
    """欧氏投影到 ``{w ≥ 0, Σw = 1}``（scipy 不可用时的兜底求解器用）。"""
    x = np.clip(np.asarray(v, dtype=float).reshape(-1), 0.0, None)
    if x.sum() <= 0:
        x = np.ones_like(x)
    x = x / x.sum()
    if max_weight is not None and max_weight < 1.0 and np.any(x > max_weight):
        x = np.minimum(x, max_weight)
        for _ in range(50):     # 把超出的部分按剩余容量再分配
            gap = 1.0 - x.sum()
            if abs(gap) < 1e-12:
                break
            room = np.clip(max_weight - x, 0.0, None)
            if room.sum() <= 0:
                break
            x = x + gap * room / room.sum()
            x = np.clip(x, 0.0, max_weight)
        x = x / x.sum()
    return x


def _projected_gradient(
    cov: np.ndarray,
    grad_fn: Any,
    n: int,
    max_weight: Optional[float],
    iters: int = 400,
    lr: float = 0.05,
) -> np.ndarray:
    """投影梯度下降兜底（无 scipy 时使用）。``grad_fn(w)`` 返回目标梯度。"""
    w = np.full(n, 1.0 / n)
    step = lr / max(float(np.max(np.abs(cov))), 1e-9)
    for _ in range(iters):
        g = grad_fn(w)
        w = _project_simplex(w - step * g, max_weight)
    return w


def optimize_weights(
    cov: Any,
    mode: str = W_MIN_VARIANCE,
    icir: Optional[Sequence[float]] = None,
    max_weight: Optional[float] = None,
    names: Optional[Sequence[str]] = None,
    risk_aversion: float = 1.0,
) -> WeightSolution:
    """在（通常已谱清洗的）协方差矩阵上求解因子权重。

    支持方案：

    * ``equal`` —— 等权，作为对照基准。
    * ``min_variance`` —— ``min wᵀCw``，论文里 GMV 的直接对应物。
    * ``max_diversification`` —— ``max (wᵀσ)/√(wᵀCw)``，最大化分散化比率。
      **本项目中因子已截面标准化（各因子波动都≈1），此时最大分散化与最小方差同解**，
      保留它是为了在传入非等波动矩阵（如原始量纲因子）时仍可用；界面上不要把两者
      并列当成"两套独立方案"。
    * ``risk_parity`` —— 各因子对体系方差的**风险贡献相等**（ERC）。
    * ``icir_tilt`` —— ``max (wᵀμ)/√(wᵀCw)``，在控制体系波动的前提下放大信号，
      需要传入 ``icir``（作为 μ 的代理；ICIR 本身已按波动缩放，故只作倾斜方向）。

    Args:
        cov: 协方差矩阵（谱清洗后的相关矩阵即可，因因子已截面标准化）。
        mode: 见上。
        icir: 各因子的 ICIR，仅 ``icir_tilt`` 需要。
        max_weight: 单因子上限（``None`` 表示不限，但至少不会超过 1）。
        names: 因子名。
        risk_aversion: 保留参数，用于后续扩展均值-方差有效前沿。

    Returns:
        :class:`WeightSolution`；求解失败时 ``success=False`` 且退回等权，绝不返回
        未归一化或含 NaN 的权重。
    """
    c = _as_square(cov)
    n = c.shape[0]
    nm = [str(x) for x in (names if names is not None else range(n))]
    if n == 0:
        return WeightSolution(mode=mode, weights=np.zeros(0), names=[], success=False,
                              message="空矩阵")
    if n == 1:
        return WeightSolution(mode=mode, weights=np.ones(1), names=nm, success=True,
                              variance=float(c[0, 0]), vol=float(math.sqrt(max(c[0, 0], 0.0))),
                              diversification_ratio=1.0)

    vols = np.sqrt(np.clip(np.diag(c), 0.0, None))
    risk_aversion = float(max(risk_aversion, 0.0))

    def _obj(w: np.ndarray) -> float:
        var = float(w @ c @ w)
        if mode == W_MIN_VARIANCE:
            return var
        if mode == W_MAX_DIVERSIFICATION:
            return -float(w @ vols) / math.sqrt(max(var, 1e-18))
        if mode == W_ICIR_TILT:
            mu = mu_vec
            return -float(w @ mu) / math.sqrt(max(var, 1e-18)) + 0.0
        if mode == W_RISK_PARITY:
            rc = w * (c @ w)
            return float(np.sum((rc - var / n) ** 2) * 1e6)
        return var

    mu_raw = np.asarray(icir, dtype=float).reshape(-1) if icir is not None else np.ones(n)
    if mu_raw.size != n:
        mu_raw = np.ones(n)
    mu_raw = np.where(np.isfinite(mu_raw), mu_raw, 0.0)
    mu_vec = np.abs(mu_raw)

    bounds = _simplex_bounds(n, max_weight)
    cons = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    w0 = np.full(n, 1.0 / n)
    if mode == W_ICIR_TILT and mu_vec.sum() > 0:
        w0 = _project_simplex(mu_vec, max_weight)

    if mode == W_EQUAL:
        w = np.full(n, 1.0 / n)
        if max_weight is not None and max_weight < 1.0:
            w = _project_simplex(w, max_weight)
        success, msg = True, "等权基准"
    else:
        success, msg, w = False, "", w0
        try:
            from scipy.optimize import minimize

            res = minimize(_obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                           options={"maxiter": 500, "ftol": 1e-12})
            if res.success and np.all(np.isfinite(res.x)):
                w, success, msg = np.asarray(res.x, dtype=float), True, "SLSQP 收敛"
            else:
                msg = str(getattr(res, "message", "SLSQP 未收敛"))
        except Exception as e:  # noqa: BLE001  —— scipy 缺失或数值异常都走兜底
            msg = f"{type(e).__name__}: {e}"

        if not success:
            if mode == W_MIN_VARIANCE:
                # 解析解 w ∝ C⁻¹1，再投影回单纯形（无约束解在因子高相关时可能为负）
                try:
                    inv = np.linalg.pinv(c)
                    w = _project_simplex(inv @ np.ones(n), max_weight)
                    success = True
                    msg = "解析解 C⁻¹1 + 单纯形投影（SCIPY 不可用）"
                except Exception:  # noqa: BLE001
                    w = np.full(n, 1.0 / n)
                    msg = f"解析解失败，退回等权（{msg}）"
            else:
                grads = {
                    W_MAX_DIVERSIFICATION: lambda w_: -(vols / math.sqrt(max(float(w_ @ c @ w_), 1e-18)))
                    + float(w_ @ vols) * (c @ w_) / max(float(w_ @ c @ w_), 1e-18) ** 1.5,
                    W_ICIR_TILT: lambda w_: -mu_vec / math.sqrt(max(float(w_ @ c @ w_), 1e-18))
                    + float(w_ @ mu_vec) * (c @ w_) / max(float(w_ @ c @ w_), 1e-18) ** 1.5,
                    W_RISK_PARITY: lambda w_: 2e6 * (c @ w_) * (w_ * (c @ w_) - float(w_ @ c @ w_) / n).sum()
                    + 2e6 * (w_ * (c @ w_) - float(w_ @ c @ w_) / n) * (c @ w_),
                }
                w = _projected_gradient(c, grads.get(mode, lambda w_: c @ w_), n, max_weight)
                msg = f"投影梯度下降（{msg}）"

    w = np.clip(np.where(np.isfinite(w), w, 0.0), 0.0, None)
    if w.sum() <= 0:
        w = np.full(n, 1.0 / n)
    w = w / w.sum()

    var = float(w @ c @ w)
    vol = math.sqrt(max(var, 0.0))
    div = float(w @ vols) / vol if vol > 0 else float("nan")
    ir = float(w @ mu_raw) / vol if vol > 0 else float("nan")
    return WeightSolution(mode=mode, weights=w, names=nm, success=success, message=msg,
                          variance=var, vol=vol, diversification_ratio=div, expected_icir=ir)


def compare_schemes(
    cov: Any,
    modes: Sequence[str] = (W_EQUAL, W_MIN_VARIANCE, W_MAX_DIVERSIFICATION, W_RISK_PARITY, W_ICIR_TILT),
    icir: Optional[Sequence[float]] = None,
    names: Optional[Sequence[str]] = None,
    max_weight: Optional[float] = None,
) -> List[WeightSolution]:
    """并排求解多套方案，供界面做"选哪套权重"的对照表。"""
    return [optimize_weights(cov, m, icir=icir, names=names, max_weight=max_weight) for m in modes]


# ---------------------------------------------------------------------------
# 4. 风险分解与"加因子值不值"
# ---------------------------------------------------------------------------
@dataclass
class RiskDecomposition:
    """体系方差按因子的分解结果。"""

    table: pd.DataFrame
    portfolio_var: float
    portfolio_vol: float
    diversification_ratio: float
    effective_n_risk: float
    hhi: float
    top_risk: str = ""
    top_risk_pct: float = float("nan")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table.reset_index(names="factor").to_dict(orient="records"),
            "portfolio_var": self.portfolio_var,
            "portfolio_vol": self.portfolio_vol,
            "diversification_ratio": self.diversification_ratio,
            "effective_n_risk": self.effective_n_risk,
            "hhi": self.hhi,
            "top_risk": self.top_risk,
            "top_risk_pct": self.top_risk_pct,
        }

    def summary_lines(self) -> List[str]:
        """紧凑文本摘要（供 AI 咨询窗口拼上下文用）。"""
        lines = [
            f"体系风险：σ={self.portfolio_vol:.4f}（方差 {self.portfolio_var:.6f}），"
            f"分散化比率 {self.diversification_ratio:.3f}，"
            f"有效风险来源 {self.effective_n_risk:.2f} 个，HHI={self.hhi:.3f}",
        ]
        if self.top_risk:
            lines.append(f"最大风险贡献：{self.top_risk} 占 {self.top_risk_pct * 100:.1f}%")
        return lines


def marginal_contributions(
    cov: Any,
    weights: Sequence[float],
    names: Optional[Sequence[str]] = None,
) -> RiskDecomposition:
    """把体系方差按因子拆开（欧拉分解），回答"风险到底来自谁"。

    定义 ``σ_p = √(wᵀCw)``，边际风险贡献 ``MRC_i = (Cw)_i / σ_p``，
    风险贡献 ``RC_i = w_i·MRC_i``，且严格满足 ``Σ RC_i = σ_p``；
    方差贡献 ``CC_i = w_i·(Cw)_i``，满足 ``Σ CC_i = σ_p²``。
    """
    c = _as_square(cov)
    n = c.shape[0]
    nm = [str(x) for x in (names if names is not None else range(n))]
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != n:
        raise ValueError(f"权重长度 {w.size} 与矩阵维度 {n} 不一致")
    w = np.where(np.isfinite(w), w, 0.0)
    if w.sum() > 0:
        w = w / w.sum()

    vol_i = np.sqrt(np.clip(np.diag(c), 0.0, None))
    cw = c @ w
    var = float(w @ cw)
    vol = math.sqrt(max(var, 0.0))
    mrc = cw / vol if vol > 0 else np.zeros(n)
    rc = w * mrc
    cc = w * cw
    rc_pct = rc / vol if vol > 0 else np.zeros(n)
    cc_pct = cc / var if var > 0 else np.zeros(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_pf = np.divide(cw, vol * vol_i, out=np.zeros(n), where=(vol * vol_i) > 0)
    vol_share = w * vol_i / vol if vol > 0 else np.zeros(n)

    table = pd.DataFrame(
        {
            "weight": w,
            "vol": vol_i,
            "mrc": mrc,
            "risk_contrib": rc,
            "risk_pct": rc_pct,
            "var_contrib": cc,
            "var_pct": cc_pct,
            "vol_share": vol_share,
            "corr_with_system": np.clip(corr_pf, -1.0, 1.0),
        },
        index=pd.Index(nm, name="factor"),
    ).sort_values("risk_pct", ascending=False)

    pct = np.clip(np.asarray(table["risk_pct"], dtype=float), 0.0, None)
    total = float(pct.sum())
    ratios = pct / total if total > 0 else np.full(len(pct), 1.0 / max(len(pct), 1))
    with np.errstate(divide="ignore"):
        eff_n = float(np.exp(-np.sum(ratios * np.log(np.where(ratios > 0, ratios, 1.0)))))
    hhi = float(np.sum(np.square(ratios)))
    div = float(w @ vol_i) / vol if vol > 0 else float("nan")

    top = str(table.index[0]) if len(table) else ""
    top_pct = float(table["risk_pct"].iloc[0]) if len(table) else float("nan")
    return RiskDecomposition(
        table=table,
        portfolio_var=var,
        portfolio_vol=vol,
        diversification_ratio=div,
        effective_n_risk=eff_n,
        hhi=hhi,
        top_risk=top,
        top_risk_pct=top_pct,
    )


def marginal_addition(
    cov: Any,
    weights: Sequence[float],
    index: int,
    icir: Optional[Sequence[float]] = None,
    names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """测算「把第 ``index`` 个因子加进现有体系」的边际价值。

    做法是解析的最优混合：把现有体系视作资产 ``p``（波动 σ_p、与候选因子的协方差
    ``cov(p,k)``），求 ``min_α Var((1−α)·p + α·k)`` 得到

        ``α* = (σ_p² − cov(p,k)) / (σ_p² + σ_k² − 2·cov(p,k))``

    再把 ``α*`` 截到 [0, 1]（不做空、不放大到超过 100%）。这比"重解一次全权重"
    便宜得多，而且给出的是**可解释的**加仓比例，而不是一组不可比的权重数字。

    Returns:
        含 ``alpha``（最优纳入比例）、``vol_before`` / ``vol_after``、
        ``vol_reduction``（绝对）、``vol_reduction_pct``、``corr_with_system``、
        ``ir_before`` / ``ir_after``（若给了 ``icir``）的字典。
    """
    c = _as_square(cov)
    n = c.shape[0]
    if not (0 <= index < n):
        raise ValueError(f"index {index} 超出范围 [0, {n})")
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != n:
        raise ValueError(f"权重长度 {w.size} 与矩阵维度 {n} 不一致")
    w = np.where(np.isfinite(w), w, 0.0)
    if w.sum() > 0:
        w = w / w.sum()

    var_p = float(w @ c @ w)
    cov_pk = float(w @ c[:, index])
    var_k = float(c[index, index])
    denom = var_p + var_k - 2.0 * cov_pk
    alpha = (var_p - cov_pk) / denom if abs(denom) > 1e-15 else 0.0
    alpha = float(min(max(alpha, 0.0), 1.0))

    var_new = (1 - alpha) ** 2 * var_p + 2 * alpha * (1 - alpha) * cov_pk + alpha ** 2 * var_k
    vol_p = math.sqrt(max(var_p, 0.0))
    vol_n = math.sqrt(max(var_new, 0.0))
    out = {
        "index": float(index),
        "alpha": alpha,
        "vol_before": vol_p,
        "vol_after": vol_n,
        "vol_reduction": vol_p - vol_n,
        "vol_reduction_pct": (1.0 - vol_n / vol_p) * 100.0 if vol_p > 0 else 0.0,
        "cov_with_system": cov_pk,
        "corr_with_system": (cov_pk / (vol_p * math.sqrt(var_k))
                             if vol_p > 0 and var_k > 0 else 0.0),
    }
    if icir is not None:
        mu = np.asarray(icir, dtype=float).reshape(-1)
        if mu.size == n:
            mu = np.where(np.isfinite(mu), mu, 0.0)
            ir_p = float(w @ mu) / vol_p if vol_p > 0 else float("nan")
            ir_n = float((1 - alpha) * (w @ mu) + alpha * mu[index]) / vol_n if vol_n > 0 else float("nan")
            out["ir_before"] = ir_p
            out["ir_after"] = ir_n
            out["ir_gain"] = ir_n - ir_p
    if names is not None and 0 <= index < len(names):
        out["name"] = names[index]
    return out


def addition_ranking(
    cov: Any,
    weights: Sequence[float],
    names: Optional[Sequence[str]] = None,
    icir: Optional[Sequence[float]] = None,
    exclude: Sequence[int] = (),
    top_k: int = 10,
) -> List[Dict[str, float]]:
    """对所有候选因子按"把仓位重新分配给它后体系波动下降幅度"排序。

    这是"因子体系还值不值得扩建"的直接回答。语义要看因子是否已在体系内：

    * **未在体系内**（``in_system=False``）—— 新增价值：降幅大 = 它能补体系没覆盖的风险
      方向；降幅 ≈ 0 = 与体系已有方向重复，加进来只是增加维护成本。
    * **已在体系内**（``in_system=True``）—— 增配价值：该因子当前权重偏低（相对其分散化
      贡献），值得从别的因子那里挪仓位过来。

    两者的数值口径完全一致（都是解析最优混合的 ``vol_reduction_pct``），所以可以放在
    同一张表里排序。
    """
    c = _as_square(cov)
    n = c.shape[0]
    nm = [str(x) for x in (names if names is not None else range(n))]
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != n:
        raise ValueError(f"权重长度 {w.size} 与矩阵维度 {n} 不一致")
    w = np.where(np.isfinite(w), w, 0.0)
    if w.sum() > 0:
        w = w / w.sum()

    rows: List[Dict[str, float]] = []
    skip = set(int(i) for i in exclude)
    for i in range(n):
        if i in skip:
            continue
        try:
            d = marginal_addition(c, w, i, icir=icir, names=nm)
        except (ValueError, ZeroDivisionError):
            continue
        d["name_idx"] = float(i)
        d["weight_now"] = float(w[i])
        d["in_system"] = float(w[i] > 0)
        rows.append(d)
    rows.sort(key=lambda d: -float(d.get("vol_reduction_pct", 0.0)))
    return rows[: max(int(top_k), 1)]


def diversification_bias(
    weights: Sequence[float],
    corr_raw: Any,
    corr_clean: Any,
    names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """量化样本相关矩阵的「虚假分散化」：同一组权重在两个矩阵下的风险差多少。

    论文的核心经验事实在这里可以逐项验证 —— 直接用样本相关矩阵做组合优化，会得到
    一个"样本内看起来很分散、样本外并不成立"的权重。把同一组权重分别放到原始矩阵与
    清洗后矩阵上评估，差值就是估计误差白送的那部分分散化：

        ``fake_div_pct = (div_clean − div_raw) / div_raw``

    为正说明原始矩阵**高估**了分散化程度（这是常态）；同理 ``vol_gap_pct`` 为正说明
    原始矩阵**低估**了体系波动。

    Returns:
        含 ``vol_raw`` / ``vol_clean`` / ``vol_gap_pct`` / ``div_raw`` / ``div_clean`` /
        ``fake_div_pct`` / ``var_raw`` / ``var_clean`` 的字典。
    """
    r_raw = _as_square(corr_raw)
    r_cln = _as_square(corr_clean)
    n = r_raw.shape[0]
    if r_cln.shape[0] != n:
        raise ValueError("两个矩阵维度不一致")
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != n:
        raise ValueError(f"权重长度 {w.size} 与矩阵维度 {n} 不一致")
    w = np.where(np.isfinite(w), w, 0.0)
    if w.sum() > 0:
        w = w / w.sum()

    vol_i = np.sqrt(np.clip(np.diag(r_raw), 0.0, None))
    var_raw = float(w @ r_raw @ w)
    var_cln = float(w @ r_cln @ w)
    vol_raw = math.sqrt(max(var_raw, 0.0))
    vol_cln = math.sqrt(max(var_cln, 0.0))
    gross = float(w @ vol_i)      # Σ w_i σ_i，分散化比率的分母只在体系侧变化

    def _div(vol: float) -> float:
        return gross / vol if vol > 0 else float("nan")

    div_raw, div_cln = _div(vol_raw), _div(vol_cln)
    fake = ((div_cln - div_raw) / div_raw * 100.0
            if div_raw and math.isfinite(div_raw) else float("nan"))
    return {
        "vol_raw": vol_raw,
        "vol_clean": vol_cln,
        "vol_gap_pct": (vol_cln - vol_raw) / vol_raw * 100.0 if vol_raw > 0 else 0.0,
        "var_raw": var_raw,
        "var_clean": var_cln,
        "div_raw": div_raw,
        "div_clean": div_cln,
        "fake_div_pct": fake,
        "names": [str(x) for x in (names if names is not None else range(n))],
    }


# ---------------------------------------------------------------------------
# 5. 一站式入口
# ---------------------------------------------------------------------------
def analyze_factor_system_spectrum(
    matrix: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    icir: Optional[Dict[str, float]] = None,
    method: str = "mp",
    shrink: float = 0.2,
    corr: Optional[pd.DataFrame] = None,
    min_periods: int = 30,
    max_weight: Optional[float] = None,
    n_obs: Optional[int] = None,
) -> Dict[str, Any]:
    """从因子矩阵一路做到「谱清洗 → 权重对照 → 风险分解 → 边际价值排序」。

    Args:
        matrix: 因子矩阵，索引 ``(date, symbol)``，每列一个因子（已截面标准化）。
        weights: 当前体系的权重（缺省等权），用于风险分解与候选排序。
        icir: ``{因子名: ICIR}``，用于 ICIR 倾斜方案与 IR 口径的边际价值。
        method/shrink: 谱清洗方式，见 :func:`analyze_spectrum`。
        corr: 已有的相关矩阵（避免重复计算）；缺省时用 ``matrix.corr`` 现算。
        min_periods: 计算相关矩阵的最少重叠样本数。
        max_weight: 单因子上限。
        n_obs: 有效观测数 T（手动指定，覆盖自动推断）。

    Returns:
        ``{"ok", "reason"?, "n_obs", "n_rows", "names", "spectrum", "corr_clean",
        "solutions", "risk", "risk_raw", "bias", "addition", "icir"}``；
        因子数不足 2 时 ``ok=False`` 并给出 ``reason``。

    Note:
        自动推断的有效观测数取**截面数（日期数）**而不是面板行数：同一截面内的股票
        有共同的市场因子，行与行远不独立，用行数会让 ``q=N/T`` 偏小、MP 噪声带过窄，
        等于把噪声当信号（清洗几乎不生效）。宁可偏保守，也不要假阴性。
    """
    valid = matrix.dropna(axis=1, how="all") if isinstance(matrix, pd.DataFrame) else pd.DataFrame()
    if valid.shape[1] < 2:
        return {"ok": False, "reason": "因子数不足 2，无法做谱清洗与权重对照",
                "spectrum": None, "solutions": [], "risk": None, "addition": [], "n_obs": 0}

    names = [str(c) for c in valid.columns]
    if corr is None:
        corr = valid.corr(method="pearson", min_periods=max(int(min_periods), 5)).fillna(0.0)
        np.fill_diagonal(corr.values, 1.0)
    corr = corr.astype(float).fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)

    if n_obs is None:
        idx0 = valid.index.get_level_values(0) if isinstance(valid.index, pd.MultiIndex) else None
        n_obs = int(pd.Index(idx0).nunique()) if idx0 is not None else int(len(valid))
    n_rows = int(len(valid))

    spec = analyze_spectrum(corr, n_obs=max(int(n_obs), 1), method=method, shrink=shrink, names=names)
    clean_df = spec.corr_clean_df

    icir_vec: Optional[List[float]] = None
    if icir:
        icir_vec = [float(icir.get(nm, 0.0) or 0.0) for nm in names]

    w_map = weights or {nm: 1.0 / len(names) for nm in names}
    w_vec = np.array([float(w_map.get(nm, 0.0) or 0.0) for nm in names], dtype=float)
    if w_vec.sum() <= 0:
        w_vec = np.full(len(names), 1.0 / len(names))
    w_vec = w_vec / w_vec.sum()

    solutions = compare_schemes(clean_df, icir=icir_vec, names=names, max_weight=max_weight)
    bias = diversification_bias(w_vec, corr.to_numpy(), clean_df.to_numpy(), names=names)
    risk_raw = marginal_contributions(corr.to_numpy(), w_vec, names=names)
    risk_clean = marginal_contributions(clean_df.to_numpy(), w_vec, names=names)
    addition = addition_ranking(
        clean_df.to_numpy(), w_vec, names=names, icir=icir_vec, exclude=[], top_k=len(names),
    )

    return {
        "ok": True,
        "n_obs": int(n_obs),
        "n_rows": n_rows,
        "names": names,
        "spectrum": spec,
        "spectrum_dict": spec.as_dict(),
        "corr_raw": corr,
        "corr_clean": clean_df,
        "solutions": solutions,
        "solutions_dict": [s.as_dict() for s in solutions],
        "weights_current": {nm: float(v) for nm, v in zip(names, w_vec)},
        "risk_raw": risk_raw,
        "risk": risk_clean,
        "bias": bias,
        "addition": addition,
        "icir": icir_vec,
    }
