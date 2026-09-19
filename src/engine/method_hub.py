"""因子挖掘方法论库（src/engine/method_hub.py）。

把「选什么方法」从拍脑袋变成有依据的选择。本模块提供四类此前缺失的分析视角，
并给出一个可解释的智能选择器：

1. 贝叶斯视角
   - ``bayesian_ic_weights``：把因子历史 IC 当作带噪观测，用正态-正态共轭做
     后验收缩。样本少 / IC 波动大的因子自动被降权，避免「偶然高 IC」被放大。
   - ``bayesian_tpe_search``：TPE（Tree-structured Parzen Estimator）贝叶斯优化，
     用于因子权重、超参数的小样本昂贵目标搜索；numpy/scipy 实现，无新增依赖。
   - ``bayesian_model_average``：多套合成方案的贝叶斯模型平均（按后验权重融合）。

2. 博弈论视角
   - ``shapley_attribution``：把「多因子合成」看作合作博弈，用 Shapley 值分配
     每个因子对合成信号的贡献。k≤8 精确枚举 2^k，k>8 用置换蒙特卡洛近似。
   - ``nash_bargaining_weights``：多目标（IC / 换手 / 稳健性）下的纳什议价解，
     给出兼顾各方、且不会被单一目标绑架的权重。

3. 主成分分析视角
   - ``pca_decomposition``：SVD 实现的 PCA，输出主成分载荷、解释方差与
     「统计因子」；用于识别因子体系的真实维度（多少个因子是冗余的）。
   - ``effective_rank``：有效秩（参与率熵），量化因子体系的分散程度。

4. 智能选择
   - ``select_method``：依据样本量、因子数、平均 |IC|、共线性、信噪比给出
     推荐方法与理由，避免「因子很少却上树模型」这类错配。

设计原则：零新增依赖（numpy/pandas/scipy/sklearn 均为既有依赖），
所有函数对空输入与退化输入返回空结果而非抛异常。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # scipy 为可选依赖（统计检验用）
    from scipy import stats as _st
    _HAS_SCIPY = True
except Exception:  # noqa: BLE001
    _HAS_SCIPY = False


# ======================================================================
# 1. 贝叶斯视角
# ======================================================================

def bayesian_ic_weights(
    ic: pd.DataFrame,
    prior_mean: float = 0.0,
    prior_strength: float = 20.0,
) -> pd.Series:
    """贝叶斯收缩后的因子权重（正态-正态共轭）。

    把因子 i 的历史 IC 序列视为 N(mu_i, sigma_i²) 的观测，先验为
    N(prior_mean, sigma_i² / prior_strength)。后验均值：

        w_i = (n_i / (n_i + tau)) * ic_mean_i + (tau / (n_i + tau)) * prior_mean

    其中 tau = prior_strength 为「先验等效样本量」。样本越少、先验越强，
    权重越被拉向先验均值——等价于对短历史因子做保守打折。

    Args:
        ic: 行=时间（IC 序列），列=因子名。NaN 视为缺失观测。
        prior_mean: 先验均值（默认 0，即「因子无信息」）。
        prior_strength: 先验等效样本量，越大收缩越强。

    Returns:
        按后验均值排序（降序）的权重 Series。
    """
    if ic is None or len(ic) == 0:
        return pd.Series(dtype=float)
    out: Dict[str, float] = {}
    for col in ic.columns:
        s = pd.to_numeric(ic[col], errors="coerce").dropna()
        n = len(s)
        if n == 0:
            out[col] = 0.0
            continue
        shrink = prior_strength / (n + prior_strength)
        out[col] = float((1.0 - shrink) * s.mean() + shrink * prior_mean)
    res = pd.Series(out, dtype=float)
    return res.sort_values(ascending=False)


def bayesian_tpe_search(
    objective: Callable[[List[float]], float],
    bounds: Sequence[Tuple[float, float]],
    n_iter: int = 30,
    n_startup: int = 12,
    n_candidates: int = 8,
    maximize: bool = True,
    seed: int = 42,
) -> Dict:
    """TPE 贝叶斯优化：小样本昂贵目标（因子权重/超参）的顺序搜索。

    每次迭代按 gamma=25% 分位把历史观测分成「好/坏」两组，分别在两组的
    经验分布上拟合逐维截断高斯，然后采样若干候选、取密度比 g(x)/l(x) 最大者
    作为下一评估点（即标准 TPE 的期望改进代理）。

    Returns:
        {best_x, best_y, n_evals, history(list of (x, y))}
    """
    if not bounds:
        return {"best_x": [], "best_y": None, "n_evals": 0, "history": []}
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)

    xs: List[List[float]] = []
    ys: List[float] = []

    def _eval(x: List[float]) -> float:
        y = float(objective(x))
        xs.append(list(x))
        ys.append(y)
        return y

    # 启动阶段：拉丁超立方分层采样。注意每轮都要重新随机「维度↔分层」的对应关系，
    # 否则第 j 维永远只会落在第 j 个分层里（w0 只能取负半轴之类的系统性偏斜）。
    for _ in range(max(1, n_startup)):
        strata = (np.arange(dim) + rng.random(dim)) / dim
        frac = strata[rng.permutation(dim)]
        _eval(list(lo + frac * (hi - lo)))

    for _ in range(max(0, n_iter)):
        y_arr = np.asarray(ys, dtype=float)
        sign = -1.0 if maximize else 1.0
        order = np.argsort(sign * y_arr)
        n_best = max(1, int(np.ceil(0.25 * len(y_arr))))
        good = np.asarray(xs, dtype=float)[order[:n_best]]
        bad = np.asarray(xs, dtype=float)[order[n_best:]]

        mu_g = good.mean(axis=0)
        sd_g = np.maximum(good.std(axis=0), (hi - lo) * 0.02 + 1e-9)
        mu_b = bad.mean(axis=0) if len(bad) else good.mean(axis=0)
        sd_b = np.maximum(bad.std(axis=0) if len(bad) > 1 else sd_g,
                          (hi - lo) * 0.02 + 1e-9)

        # 采样候选并取密度比最大者（TPE 的 EI 代理）
        best_cand, best_ratio = None, -np.inf
        for _ in range(max(1, n_candidates)):
            if _HAS_SCIPY:
                cand = _st.truncnorm.rvs(
                    (lo - mu_g) / sd_g, (hi - mu_g) / sd_g,
                    loc=mu_g, scale=sd_g, random_state=rng,
                )
            else:  # 无 scipy：拒绝采样兜底
                cand = mu_g + rng.normal(size=dim) * sd_g
            cand = np.clip(cand, lo, hi)
            if _HAS_SCIPY:
                log_g = _st.norm.logpdf(cand, mu_g, sd_g).sum()
                log_l = _st.norm.logpdf(cand, mu_b, sd_b).sum()
            else:
                log_g = -0.5 * (((cand - mu_g) / sd_g) ** 2).sum()
                log_l = -0.5 * (((cand - mu_b) / sd_b) ** 2).sum()
            ratio = log_g - log_l
            if ratio > best_ratio:
                best_ratio, best_cand = ratio, cand
        _eval([float(v) for v in (best_cand if best_cand is not None else mu_g)])

    y_arr = np.asarray(ys, dtype=float)
    idx = int(np.argmax(y_arr)) if maximize else int(np.argmin(y_arr))
    return {
        "best_x": xs[idx],
        "best_y": float(y_arr[idx]),
        "n_evals": len(ys),
        "history": list(zip(xs, ys)),
    }


def bayesian_model_average(
    predictions: pd.DataFrame,
    scores: pd.Series,
    temperature: float = 1.0,
) -> pd.Series:
    """贝叶斯模型平均：按各候选方案的评分经 softmax 得到后验权重并融合。

    Args:
        predictions: 列=候选方案，行=样本（各方案的预测/合成值）。
        scores: 每个候选方案的评分（如验证集 IC），索引与列名对齐。
        temperature: softmax 温度，越小越集中于最优方案。

    Returns:
        融合后的序列。
    """
    if predictions is None or predictions.empty or scores is None or scores.empty:
        return pd.Series(dtype=float)
    cols = [c for c in predictions.columns if c in scores.index]
    if not cols:
        return pd.Series(dtype=float)
    s = pd.to_numeric(scores.loc[cols], errors="coerce").fillna(0.0).astype(float)
    z = (s - s.mean()) / (s.std() or 1.0) / max(1e-6, temperature)
    w = np.exp(z - z.max())
    w = w / w.sum()
    return (predictions[cols].astype(float) * w).sum(axis=1)


# ======================================================================
# 2. 博弈论视角
# ======================================================================

def _rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    """秩相关（Spearman），退化输入返回 0。"""
    if a.size < 3 or b.size != a.size:
        return 0.0
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def shapley_attribution(
    factors: pd.DataFrame,
    target: pd.Series,
    max_exact: int = 8,
    n_permutations: int = 256,
    seed: int = 42,
) -> pd.Series:
    """多因子合成的 Shapley 值归因（合作博弈的贡献分配）。

    特征函数 v(S) = 用子集 S 等权标准化合成后的信号与 target 的 RankIC。
    Shapley 值是唯一满足「有效性 + 对称性 + 可加性」的分配方式，因此比
    「单因子 IC 排序」更能反映因子在**组合中**的真实贡献（含互补与冗余）。

    k <= max_exact 时精确枚举 2^k 个子集；否则用随机置换蒙特卡洛近似。

    Returns:
        每个因子的 Shapley 值（按降序排列），其和≈ v(全体)。
    """
    if factors is None or factors.empty or target is None:
        return pd.Series(dtype=float)
    x = factors.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(target, errors="coerce")
    ok = x.notna().all(axis=1) & y.notna()
    x, y = x[ok], y[ok]
    k = x.shape[1]
    if k == 0 or len(x) < 5:
        return pd.Series(dtype=float)

    names = list(x.columns)
    mat = x.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    # 逐列标准化（只有被纳入的因子参与，保证 v(S) 可比）
    mat = (mat - mat.mean(axis=0)) / np.where(mat.std(axis=0) == 0, 1.0, mat.std(axis=0))

    cache: Dict[Tuple[int, ...], float] = {}

    def v(mask: Tuple[int, ...]) -> float:
        if not mask:
            return 0.0
        if mask in cache:
            return cache[mask]
        sig = mat[:, list(mask)].mean(axis=1)
        val = _rank_ic(sig, yv)
        cache[mask] = val
        return val

    if k <= max_exact:
        phi = np.zeros(k)
        from math import factorial
        for i in range(k):
            others = [j for j in range(k) if j != i]
            for size in range(len(others) + 1):
                weight = (factorial(size) * factorial(k - size - 1)) / factorial(k)
                for combo in _combinations(others, size):
                    base = tuple(sorted(combo))
                    with_i = tuple(sorted(combo + (i,)))  # noqa: RUF005  热路径，保持可读
                    phi[i] += weight * (v(with_i) - v(base))
    else:  # 蒙特卡洛置换近似
        rng = np.random.default_rng(seed)
        phi = np.zeros(k)
        for _ in range(int(n_permutations)):
            perm = rng.permutation(k)
            cur: Tuple[int, ...] = ()
            for j in perm:
                nxt = tuple(sorted(cur + (j,)))  # noqa: RUF005  热路径，保持可读
                phi[j] += v(nxt) - v(cur)
                cur = nxt
        phi /= float(n_permutations)

    return pd.Series(dict(zip(names, phi))).sort_values(ascending=False)


def _combinations(items: List[int], size: int):
    """轻量组合枚举（避免依赖 itertools.combinations 的对象开销）。"""
    if size == 0:
        yield ()
        return
    if size > len(items):
        return
    for idx in range(len(items) - size + 1):
        for rest in _combinations(items[idx + 1:], size - 1):
            yield (items[idx],) + rest  # noqa: RUF005  热路径，保持可读


def nash_bargaining_weights(
    utilities: pd.DataFrame,
    weights_grid: int = 201,
    seed: int = 42,
) -> pd.Series:
    """多目标纳什议价解：max prod_i (u_i(w) - d_i)。

    utilities 的每一列是一个目标（如 IC、负换手、稳健性），每一行是某个候选
    方案（或某个因子）在各目标上的效用。权重 w 作用于方案，得到各目标的
    加权效用 u_i(w)；议价解要求「没有人能在不损害他人的情况下变得更好」，
    因此不会像「只看 IC」那样被单一目标绑架。

    Args:
        utilities: 行=方案/因子，列=目标（已归一化到同向、越大越好）。
        weights_grid: 权重搜索粒度（在单纯形上做分层采样 + 局部优化）。

    Returns:
        各方案/因子的议价权重（和为 1）。
    """
    if utilities is None or utilities.empty:
        return pd.Series(dtype=float)
    u = utilities.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    # 列内归一化到 [0,1]，保证各目标可比
    lo = u.min(axis=0)
    hi = u.max(axis=0)
    u = (u - lo) / np.where(hi - lo == 0, 1.0, hi - lo)
    n, m = u.shape
    if n == 0 or m == 0:
        return pd.Series(dtype=float)
    # 分歧点 d：各目标的最差可行效用（取各列最小值，略作下移保证 log 有定义）
    d = u.min(axis=0) - 1e-3

    best_w, best_val = None, -np.inf
    rng = np.random.default_rng(seed)
    # 单纯形分层采样 + 随机重启的局部收缩搜索
    starts = [np.ones(n) / n]
    for _ in range(8):
        r = rng.dirichlet(np.ones(n) * 2.0)
        starts.append(r)
    for w0 in starts:
        w = w0.copy()
        step = 0.3
        for _ in range(60):
            val = _nash_obj(w, u, d)
            improved = False
            for _try in range(12):
                cand = w + rng.normal(scale=step * 0.1, size=n)
                cand = np.clip(cand, 0.0, None)
                s = cand.sum()
                if s <= 0:
                    continue
                cand /= s
                cv = _nash_obj(cand, u, d)
                if cv > val:
                    w, val, improved = cand, cv, True
            if not improved:
                step *= 0.5
                if step < 1e-4:
                    break
        if val > best_val:
            best_val, best_w = val, w
    w = best_w if best_w is not None else np.ones(n) / n
    return pd.Series(w, index=utilities.index).sort_values(ascending=False)


def _nash_obj(w: np.ndarray, u: np.ndarray, d: np.ndarray) -> float:
    """纳什积的对数；存在非正效用时给大惩罚（保证解在可行域内）。"""
    uw = w @ u
    gap = uw - d
    if np.any(gap <= 0):
        return -1e9
    return float(np.sum(np.log(gap)))


# ======================================================================
# 3. 主成分分析视角
# ======================================================================

def pca_decomposition(
    x: pd.DataFrame,
    n_components: Optional[int] = None,
    center: bool = True,
    scale: bool = True,
) -> Dict:
    """PCA 分解：识别因子体系的真实维度与冗余结构。

    Returns:
        {
          explained_variance_ratio: 各主成分解释方差占比,
          cumulative: 累计解释方差,
          loadings: 主成分载荷（行=原因子，列=PC）,
          components: 主成分得分（行=样本，列=PC）,
          n_components_90: 解释 90% 方差所需的主成分数,
        }
    """
    if x is None or x.empty:
        return {}
    m = x.apply(pd.to_numeric, errors="coerce")
    m = m.dropna()
    if len(m) < 3 or m.shape[1] < 1:
        return {}
    a = m.to_numpy(dtype=float)
    if center:
        a = a - a.mean(axis=0)
    if scale:
        sd = a.std(axis=0)
        a = a / np.where(sd == 0, 1.0, sd)
    # SVD 实现 PCA（等价于对协方差矩阵做特征分解，数值更稳）
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    var = (s ** 2) / max(1, len(a) - 1)
    total = var.sum()
    ratio = var / total if total > 0 else var * 0.0
    cum = np.cumsum(ratio)
    n90 = int(np.searchsorted(cum, 0.90) + 1) if len(cum) else 0
    k = n_components or len(ratio)
    k = int(min(max(1, k), len(ratio)))
    comp = u[:, :k] * s[:k]
    return {
        "explained_variance_ratio": pd.Series(ratio, index=[f"PC{i + 1}" for i in range(len(ratio))]),
        "cumulative": pd.Series(cum, index=[f"PC{i + 1}" for i in range(len(cum))]),
        "loadings": pd.DataFrame(vt[:k].T, index=m.columns,
                                 columns=[f"PC{i + 1}" for i in range(k)]),
        "components": pd.DataFrame(comp, index=m.index,
                                   columns=[f"PC{i + 1}" for i in range(k)]),
        "n_components_90": n90,
    }


def effective_rank(x: pd.DataFrame) -> float:
    """有效秩（参与率 PR 的指数）：衡量因子体系的分散程度。

    取值 1~k，越接近 k 表示各因子贡献越均衡；接近 1 表示信息几乎集中在一个方向上
    （典型症状：一堆因子其实是同一个因子的变体）。
    """
    res = pca_decomposition(x)
    if not res:
        return float("nan")
    r = res["explained_variance_ratio"].to_numpy(dtype=float)
    r = r[r > 0]
    if r.size == 0:
        return float("nan")
    return float(np.exp(-(r * np.log(r)).sum()))


# ======================================================================
# 4. 智能方法选择
# ======================================================================

@dataclass
class MethodProfile:
    """用于方法选择的数据画像。"""

    n_samples: int = 0                 # 样本量（截面×时间或时间长度）
    n_factors: int = 0                 # 因子数
    mean_abs_ic: float = 0.0           # 平均 |IC|
    ic_std: float = 0.0                # IC 时序标准差
    mean_pairwise_corr: float = 0.0    # 因子两两平均相关
    effective_rank: float = 0.0        # 有效秩（可选）
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "n_samples": self.n_samples,
            "n_factors": self.n_factors,
            "mean_abs_ic": self.mean_abs_ic,
            "ic_std": self.ic_std,
            "mean_pairwise_corr": self.mean_pairwise_corr,
            "effective_rank": self.effective_rank,
        }


def select_method(profile: MethodProfile) -> Dict:
    """依据数据画像推荐合成/挖掘方法，并给出可读理由。

    规则（可解释优先，不做黑箱打分）：
      - 样本极少（<200）或因子极少（<=2）：线性加权，任何非线性都会过拟合；
      - 信噪比低（|IC|/IC_std < 0.3）：贝叶斯收缩 + 等权，先降噪再谈增益；
      - 共线性高（平均相关 > 0.6 或 有效秩/因子数 < 0.4）：先 PCA 去冗余再合成；
      - 样本充足（>=1000）且因子数适中（3~30）：树模型捕捉非线性；
      - 样本充足且因子多（>30）：PCA 降维 + 树模型；
      - 存在明显状态切换（由调用方在 notes 中给出 'regime'）：contextual 建模。

    Returns:
        {primary, reason, pipeline, alternatives}
    """
    p = profile
    snr = abs(p.mean_abs_ic) / p.ic_std if p.ic_std > 0 else 0.0
    redundant = p.mean_pairwise_corr > 0.6 or (
        p.effective_rank > 0 and p.n_factors > 0
        and p.effective_rank / p.n_factors < 0.4
    )
    regime = any("regime" in str(n).lower() for n in p.notes)

    if p.n_samples < 200 or p.n_factors <= 2:
        return {
            "primary": "linear_ic",
            "reason": f"样本量 {p.n_samples} / 因子数 {p.n_factors} 偏小，非线性模型自由度过高易过拟合；"
                      f"先用线性 IC 加权建立可解释 baseline。",
            "pipeline": ["winsorize", "standardize", "linear_ic"],
            "alternatives": ["bayesian_shrinkage"],
        }
    if snr < 0.3:
        return {
            "primary": "bayesian_shrinkage",
            "reason": f"信噪比 |IC|/std ≈ {snr:.2f} 偏低，先做贝叶斯收缩抑制噪声放大，"
                      f"再评估是否值得引入非线性。",
            "pipeline": ["winsorize", "standardize", "bayesian_shrinkage", "linear_ic"],
            "alternatives": ["equal_weight", "pca_denoise"],
        }
    if redundant:
        return {
            "primary": "pca_denoise",
            "reason": f"因子共线性偏高（平均相关 {p.mean_pairwise_corr:.2f}，"
                      f"有效秩 {p.effective_rank:.2f}/{p.n_factors}），先做 PCA 去冗余再合成，"
                      f"否则加权结果会被重复计数的同类因子主导。",
            "pipeline": ["winsorize", "standardize", "pca_denoise", "linear_ic"],
            "alternatives": ["shapley_prune", "tree_model"],
        }
    if regime:
        return {
            "primary": "contextual",
            "reason": "存在明显状态切换（regime），同一套权重在不同市场状态下会互相抵消；"
                      "按状态分 context 建模可保留各自的有效信号。",
            "pipeline": ["winsorize", "standardize", "contextual_split", "linear_ic_per_context"],
            "alternatives": ["tree_model"],
        }
    if p.n_samples >= 1000 and 3 <= p.n_factors <= 30:
        return {
            "primary": "tree_model",
            "reason": f"样本量 {p.n_samples} 充足、因子数 {p.n_factors} 适中，树模型能显式刻画"
                      f"非线性与交互；须与线性 baseline 对照，增益不足则退回线性（可解释性更优）。",
            "pipeline": ["winsorize", "standardize", "linear_ic_baseline", "tree_model", "compare"],
            "alternatives": ["bayesian_tpe_weights", "linear_ic"],
        }
    if p.n_samples >= 1000 and p.n_factors > 30:
        return {
            "primary": "pca_then_tree",
            "reason": f"因子数 {p.n_factors} 较多，直接上树模型维度灾难与共线性并存；"
                      f"先 PCA 压到主成分再建模。",
            "pipeline": ["winsorize", "standardize", "pca_denoise", "tree_model"],
            "alternatives": ["lasso_screen", "tree_model"],
        }
    return {
        "primary": "linear_ic",
        "reason": "未触发任何特殊条件，采用线性 IC 加权作为稳健默认。",
        "pipeline": ["winsorize", "standardize", "linear_ic"],
        "alternatives": ["bayesian_shrinkage", "equal_weight"],
    }


def profile_from_ic(
    ic: pd.DataFrame,
    factors: Optional[pd.DataFrame] = None,
    notes: Optional[List[str]] = None,
) -> MethodProfile:
    """从 IC 序列（可选：因子截面矩阵）构造数据画像，供 ``select_method`` 使用。"""
    if ic is None or ic.empty:
        return MethodProfile(notes=notes or [])
    num = ic.apply(pd.to_numeric, errors="coerce")
    mean_abs = float(num.abs().mean().mean()) if num.size else 0.0
    ic_std = float(num.std().mean()) if num.size else 0.0
    corr = 0.0
    if factors is not None and not factors.empty and factors.shape[1] > 1:
        c = factors.apply(pd.to_numeric, errors="coerce").corr().to_numpy(dtype=float)
        off = c[~np.eye(c.shape[0], dtype=bool)]
        corr = float(np.nanmean(np.abs(off))) if off.size else 0.0
    return MethodProfile(
        n_samples=int(num.shape[0] * max(1, num.shape[1])),
        n_factors=int(num.shape[1]),
        mean_abs_ic=mean_abs,
        ic_std=ic_std,
        mean_pairwise_corr=corr,
        effective_rank=effective_rank(factors) if factors is not None and not factors.empty else 0.0,
        notes=notes or [],
    )
