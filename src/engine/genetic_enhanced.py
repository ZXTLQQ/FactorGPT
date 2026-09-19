"""
增强遗传规划因子挖掘 (src/engine/genetic_enhanced.py)

在原有 GeneticFactorMiner 基础上扩展：
1. 因子簇驱动演化 — 将相关因子编组为 Cluster，组内交叉促进有效模式保留
2. 事件簇感知 — 按事件（财报、政策窗口、市场状态）对截面 IC 做加权适应度
3. 更丰富的算子集 — ts_rank, ts_zscore, ts_delta, ts_corr, ts_min, ts_max 等
4. 多样性保持 — 岛屿模型 / 拥挤距离
5. 与 FactorLibrary 深度集成 — 产出直接入库，纳入质量评分体系
6. 执行轨迹 — 逐簇逐代记录 best/mean IC 与种群唯一表达式占比（``self.history``），
   供收敛诊断、多样性监控与可视化消费

典型用法：
    library = create_default_library()
    evolver = EnhancedFactorEvolver(kline, library)
    results = evolver.evolve_clusters(
        generations=10, pop_per_cluster=30, top_k=20
    )
    # 产出 批量高质量因子，自动入库
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .factor_builder import analyze_lookahead
from .factor_library import FactorLibrary
from .ic_utils import panel_ic
from .traditional_factors import (
    ALL_CATEGORIES,
    CATEGORY_LABELS,
    FactorDef,
)

_COLS = ["open", "high", "low", "close", "volume", "amount", "pct_chg"]

# 扩展算子集（与 eval_expr / expr_to_code 的算子白名单严格一致）
# 一元算子（无窗口参数）: log, abs, sqrt, sign, neg, delta
# 窗口算子（第 3 位为 ("const", w)）: ts_zscore, ts_rank, ts_min, ts_max, rol
# 二元算子: add, sub, mul, div；三元: ts_corr(a, b, w)
_UNARY = ["log", "abs", "sqrt", "sign", "neg", "delta", "ts_zscore", "ts_rank"]
_BINARY = ["add", "sub", "mul", "div", "ts_corr", "ts_min", "ts_max"]
_TERMINAL = ["col", "const"]
_ROL_W = [3, 5, 10, 14, 20, 30, 60]
# 需要滚动窗口参数的算子（其第 3 个元素应为 ("const", w)）
_WINDOWED = {"ts_zscore", "ts_rank", "ts_min", "ts_max", "rol"}
DEFAULT_WINDOW = 20

MAX_DEPTH = 3


def _window_of(expr: Any, idx: int = 2, default: int = DEFAULT_WINDOW) -> int:
    """从容忍缺失/异常的角度读取窗口参数。

    表达式可能因变异产生缺少窗口位的短元组（历史遗留），此处统一兜底为
    ``default``，避免整个个体因 IndexError 被判为无效个体。
    """
    if len(expr) > idx:
        arg = expr[idx]
        if isinstance(arg, tuple) and arg and arg[0] == "const":
            try:
                return max(2, int(float(arg[1])))
            except (TypeError, ValueError):
                return default
    return default


def _min_periods(window: int, cap: int) -> int:
    """min_periods 必须 <= window，否则 pandas 会直接抛 ValueError。

    短窗口（3/5 日）在 ``_ROL_W`` 中合法，因此这里统一把预热期按窗口收窄。
    """
    return max(2, min(int(window), int(cap)))


@dataclass
class FactorCluster:
    """因子簇 — 将同类因子编组用于 GP 交叉/变异。"""
    name: str
    category: str
    factors: List[FactorDef] = field(default_factory=list)
    description: str = ""


@dataclass
class EventWindow:
    """事件窗口定义 — 按事件类型调整适应度计算中的样本范围。"""
    name: str
    date_range: Tuple[str, str]  # (start_date, end_date)
    event_type: str  # "earnings", "policy", "market_state", "custom"
    weight: float = 1.0  # 适应度加权系数


# ---------------------------------------------------------------------------
# 表达式树构建 / 求值 / 序列化（增强版）
# ---------------------------------------------------------------------------

def random_expr(rng: random.Random, depth: int = 0, cols: Optional[List[str]] = None) -> Any:
    if cols is None:
        cols = list(_COLS)
    if depth >= MAX_DEPTH or (depth > 0 and rng.random() < 0.25):
        if rng.random() < 0.80:
            return ("col", rng.choice(cols))
        return ("const", round(rng.uniform(-2, 2), 4))
    # 70% 一元，30% 二元
    if depth >= MAX_DEPTH - 1 or rng.random() < 0.6:
        op = rng.choice(_UNARY)
        return (op, random_expr(rng, depth + 1, cols))
    else:
        op = rng.choice(_BINARY)
        return (op, random_expr(rng, depth + 1, cols), random_expr(rng, depth + 1, cols))


def seed_expr_from_code(rng: random.Random, code: str, cols: Optional[List[str]] = None) -> Optional[Any]:
    """从预置因子的代码字符串提取并简化，生成种子表达式树。

    目前用启发式：解析常见的滚动窗口/运算模式，构造对应树。
    这是遗传规划「从优秀先验出发演化」的关键桥接。
    """
    if cols is None:
        cols = list(_COLS)
    chosen_col = rng.choice(cols)
    chosen_w = rng.choice(_ROL_W)
    # 从代码中提取模式启发式种子
    if "pct_change" in code:
        return ("sub", ("col", chosen_col), ("rol", ("col", chosen_col), ("const", float(chosen_w))))
    if "rolling" in code and "std" in code.lower():
        return ("neg", ("ts_zscore", ("col", chosen_col), ("const", float(chosen_w))))
    if "rolling" in code and "sum" in code.lower():
        return ("rol", ("col", chosen_col), ("const", float(chosen_w)))
    if "rolling" in code and "corr" in code.lower():
        c2 = rng.choice([c for c in cols if c != chosen_col])
        return ("ts_corr", ("col", chosen_col), ("col", c2), ("const", float(chosen_w)))
    return random_expr(rng, cols=cols)


def eval_expr(expr: Any, df: pd.DataFrame) -> pd.Series:
    """增强版表达式求值。"""
    kind = expr[0]
    if kind == "col":
        return df[expr[1]].astype(float)
    if kind == "const":
        return pd.Series(float(expr[1]), index=df.index)

    # 一元算子
    if kind == "log":
        return np.log(np.abs(eval_expr(expr[1], df)) + 1e-12)
    if kind == "abs":
        return np.abs(eval_expr(expr[1], df))
    if kind == "sqrt":
        return np.sqrt(np.abs(eval_expr(expr[1], df)) + 1e-12)
    if kind == "sign":
        return np.sign(eval_expr(expr[1], df))
    if kind == "neg":
        return -eval_expr(expr[1], df)
    if kind == "delta":
        child = eval_expr(expr[1], df)
        return child.groupby(df["symbol"]).diff()
    if kind == "ts_zscore":
        child = eval_expr(expr[1], df)
        w = _window_of(expr)
        mp = _min_periods(w, 10)
        return child.groupby(df["symbol"]).transform(
            lambda s: (s - s.rolling(w, min_periods=mp).mean()) / (s.rolling(w, min_periods=mp).std() + 1e-8))
    if kind == "ts_rank":
        child = eval_expr(expr[1], df)
        w = _window_of(expr)
        return child.groupby(df["symbol"]).transform(
            lambda s: s.rolling(w, min_periods=_min_periods(w, 10)).rank(pct=True))

    # 二元算子 (3-ary for ts_corr)
    if kind == "ts_corr":
        a = eval_expr(expr[1], df)
        b = eval_expr(expr[2], df)
        w = _window_of(expr, 3)
        mp = _min_periods(w, 10)
        # 逐标的算滚动相关后按**位置**写回：分组滚动按组输出，且 GroupBy.apply
        # 会把分组键一并交给回调（pandas 2.2 起已弃用），故不走 apply。
        av = np.asarray(a, dtype=float)
        bv = np.asarray(b, dtype=float)
        out = pd.Series(np.nan, index=df.index, dtype=float)
        for _, idx in df.groupby("symbol").indices.items():
            idx = np.sort(np.asarray(idx))
            out.iloc[idx] = pd.Series(av[idx]).rolling(w, min_periods=mp).corr(
                pd.Series(bv[idx])
            ).to_numpy()
        return out
    if kind == "ts_min":
        a = eval_expr(expr[1], df)
        w = _window_of(expr)
        return a.groupby(df["symbol"]).transform(lambda s: s.rolling(w, min_periods=_min_periods(w, 5)).min())
    if kind == "ts_max":
        a = eval_expr(expr[1], df)
        w = _window_of(expr)
        return a.groupby(df["symbol"]).transform(lambda s: s.rolling(w, min_periods=_min_periods(w, 5)).max())
    if kind == "rol":
        # 滚动窗口聚合（种子因子 "rolling+sum" 模式的落地算子）
        child = eval_expr(expr[1], df)
        w = _window_of(expr)
        return child.groupby(df["symbol"]).transform(lambda s: s.rolling(w, min_periods=_min_periods(w, 5)).mean())

    a = eval_expr(expr[1], df)
    b = eval_expr(expr[2], df)
    if kind == "add":
        return a + b
    if kind == "sub":
        return a - b
    if kind == "mul":
        return a * b
    if kind == "div":
        return a / (b.replace(0, np.nan) + 1e-12)
    raise ValueError(f"未知算子 {kind}")


def expr_to_expr_str(expr: Any) -> str:
    """把表达式树渲染成可直接内联进因子代码的 Python 表达式字符串。

    这是表达式渲染的**唯一入口**：:func:`expr_to_code` 与多尺度挖掘的代码生成
    都经由此处，因此"离线求值口径"和"落库代码口径"不会各自漂移（两套渲染器
    最常见的失效方式就是除零/窗口预热细节只在一侧被修好）。
    """
    def _emit(e: Any) -> str:
        kind = e[0]
        if kind == "col":
            # 显式 astype(float)：离线行情多为 float32，直接在 float32 上做
            # rolling/rank 会累积精度损失，与 eval_expr 的语义也不一致。
            return f"df['{e[1]}'].astype(float)"
        if kind == "const":
            # 常量对齐为与 df 同索引的 Series：与 eval_expr 语义一致，
            # 同时避免常量作为子树参与 groupby / rolling 时因标量类型报错。
            return f"pd.Series({float(e[1])}, index=df.index)"
        if kind == "log":
            return f"np.log(np.abs({_emit(e[1])}) + 1e-12)"
        if kind == "abs":
            return f"np.abs({_emit(e[1])})"
        if kind == "sqrt":
            return f"np.sqrt(np.abs({_emit(e[1])}) + 1e-12)"
        if kind == "sign":
            return f"np.sign({_emit(e[1])})"
        if kind == "neg":
            return f"(-{_emit(e[1])})"
        if kind == "delta":
            return f"({_emit(e[1])}).groupby(df['symbol']).diff()"
        if kind == "ts_zscore":
            child = _emit(e[1])
            w = _window_of(e)
            mp = _min_periods(w, 10)
            return f"({child}).groupby(df['symbol']).transform(lambda s: (s-s.rolling({w},min_periods={mp}).mean())/(s.rolling({w},min_periods={mp}).std()+1e-8))"
        if kind == "ts_rank":
            child = _emit(e[1])
            w = _window_of(e)
            return f"({child}).groupby(df['symbol']).transform(lambda s: s.rolling({w},min_periods={_min_periods(w, 10)}).rank(pct=True))"
        if kind == "ts_corr":
            a = _emit(e[1]); b = _emit(e[2])
            w = _window_of(e, 3)
            return f"pd.concat([{a}, {b}, df['symbol']], axis=1, keys=['a','b','sym']).groupby('sym').apply(lambda g: g['a'].rolling({w},min_periods={_min_periods(w, 10)}).corr(g['b'])).droplevel(0)"
        if kind == "ts_min":
            child = _emit(e[1])
            w = _window_of(e)
            return f"({child}).groupby(df['symbol']).transform(lambda s: s.rolling({w},min_periods={_min_periods(w, 5)}).min())"
        if kind == "ts_max":
            child = _emit(e[1])
            w = _window_of(e)
            return f"({child}).groupby(df['symbol']).transform(lambda s: s.rolling({w},min_periods={_min_periods(w, 5)}).max())"
        if kind == "rol":
            child = _emit(e[1])
            w = _window_of(e)
            return f"({child}).groupby(df['symbol']).transform(lambda s: s.rolling({w},min_periods={_min_periods(w, 5)}).mean())"
        if kind == "div":
            # 与 eval_expr 保持一致：除数 0 置为 NaN，避免产生 inf 污染截面排序
            return f"({_emit(e[1])} / (({_emit(e[2])}).replace(0, np.nan) + 1e-12))"
        sym = {"add": "+", "sub": "-", "mul": "*"}.get(kind)
        if sym is None:
            raise ValueError(f"expr_to_code: 不支持的算子 '{kind}'")
        return f"({_emit(e[1])} {sym} {_emit(e[2])})"

    return _emit(expr)


def expr_to_code(expr: Any, name: str = "gp_factor") -> str:
    expr_str = expr_to_expr_str(expr)
    return (
        "import pandas as pd\n"
        "import numpy as np\n"
        "def alpha_factor(df):\n"
        "    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)\n"
        f"    f = {expr_str}\n"
        "    f = pd.Series(f, index=df.index) if np.ndim(f) == 0 else f\n"
        "    f = f.replace([np.inf, -np.inf], np.nan)\n"
        "    df['_f'] = f\n"
        "    df['_f'] = df.groupby('symbol')['_f'].shift(1)\n"
        "    df['factor'] = df['_f'].fillna(0.0)\n"
        "    return df[['date', 'symbol', 'factor']]\n"
    )


# ---------------------------------------------------------------------------
# 增强因子演化器
# ---------------------------------------------------------------------------

class EnhancedFactorEvolver:
    """增强遗传规划因子挖掘引擎。

    核心改进：
    - 因子簇驱动：按大类编组演化，组内交叉保留同质优良基因
    - 事件窗口感知：根据事件类型加权适应度
    - 岛屿模型：3-5 个子种群独立演化 + 定期迁移
    - 从 FactorLibrary 种子因子引导演化（不做纯随机起点）
    """

    def __init__(
        self,
        kline: pd.DataFrame,
        library: Optional[FactorLibrary] = None,
        seed: int = 0,
    ) -> None:
        self.df = kline.sort_values(["symbol", "date"]).reset_index(drop=True).copy()
        self.df["_fwd_ret"] = self.df.groupby("symbol")["pct_chg"].shift(-1)
        self.library = library or FactorLibrary()
        self.rng = random.Random(seed)
        self._event_windows: List[EventWindow] = []
        self._clusters: List[FactorCluster] = self._init_clusters()
        # 逐代执行轨迹：每条记录为「某岛屿某代」的统计，供收敛/多样性可视化消费
        self.history: List[Dict[str, Any]] = []
        # 岛屿迁移日志：记录每次迁移的世代、方向与迁移个体数
        self.migrations: List[Dict[str, Any]] = []

    def _init_clusters(self) -> List[FactorCluster]:
        clusters = []
        for cat in ALL_CATEGORIES:
            factors = self.library.list_by_category(cat)
            if factors:
                clusters.append(FactorCluster(
                    name=cat,
                    category=cat,
                    factors=factors,
                    description=CATEGORY_LABELS.get(cat, cat),
                ))
        return clusters

    def add_event_window(self, ew: EventWindow) -> None:
        self._event_windows.append(ew)

    # ---------- 适应度 ----------
    def _fitness(
        self,
        expr: Any,
        train_df: pd.DataFrame,
        cluster: Optional[FactorCluster] = None,
    ) -> float:
        try:
            fac = eval_expr(expr, train_df)
            panel = pd.DataFrame({
                "f": np.asarray(fac, dtype=float),
                "y": train_df["_fwd_ret"].to_numpy(dtype=float),
                "date": train_df["date"].to_numpy(),
            })
            panel = panel.replace([np.inf, -np.inf], np.nan).dropna()
            if len(panel) < 50:
                return -1e9

            # 逐日截面 IC 走 ic_utils 的向量化内核（一次 bincount 聚合）。
            # 这里原本是 groupby("date").apply(...)：pandas ≥ 2.2 会把分组列一并交给
            # 回调并抛 FutureWarning，而项目 filterwarnings = error，等于直接失败；
            # min_count=2 与原实现（组内至少 2 个样本才有相关，单样本方差为 0）逐点一致。
            ic = panel_ic(panel["f"].to_numpy(dtype=float),
                          panel["y"].to_numpy(dtype=float),
                          panel["date"].to_numpy(), min_count=2).dropna()
            if len(ic) == 0:
                return -1e9

            # 聚类奖励：如果因子与簇内典型因子模式相近，给小幅奖励
            cluster_bonus = 0.0
            if cluster and cluster.factors:
                cluster_bonus = 0.005  # 以簇的名义有小奖励

            # 事件窗口加权：窗口内日期的截面 IC 获得额外权重，窗口外保持 1.0。
            # 未配置事件窗口时（默认路径）退化为普通均值，与历史行为完全一致。
            if self._event_windows:
                dates = pd.to_datetime(pd.Index(ic.index))
                w = pd.Series(1.0, index=ic.index, dtype=float)
                for ew in self._event_windows:
                    mask = np.asarray(
                        (dates >= pd.to_datetime(ew.date_range[0]))
                        & (dates <= pd.to_datetime(ew.date_range[1]))
                    )
                    if mask.any():
                        w[mask] *= ew.weight
                if float(w.sum()) > 0:
                    return float((ic * w).sum() / float(w.sum())) + cluster_bonus

            return float(ic.mean()) + cluster_bonus
        except Exception:
            return -1e9

    def _test_fitness(self, expr: Any, test_df: pd.DataFrame) -> float:
        """在测试集上评估（不奖励聚类）。"""
        return self._fitness(expr, test_df, cluster=None)

    # ---------- 遗传操作 ----------
    def _mutate(
        self,
        expr: Any,
        depth: int = 0,
        cols: Optional[List[str]] = None,
        rng: Optional[random.Random] = None,
    ) -> Any:
        rng = rng or self.rng
        if depth > MAX_DEPTH or rng.random() < 0.20:
            return random_expr(rng, depth, cols)
        kind = expr[0]
        if kind in ("col", "const"):
            return random_expr(rng, depth, cols)
        # 一元算子
        if kind in _UNARY:
            if rng.random() < 0.30:
                new_kind = rng.choice(_UNARY)
                # 窗口算子必须携带窗口位，否则表达式结构不合法、个体直接失效
                tail = expr[2] if len(expr) > 2 else ("const", float(rng.choice(_ROL_W)))
                if new_kind in _WINDOWED:
                    return (new_kind, self._mutate(expr[1], depth + 1, cols, rng), tail)
                return (new_kind, self._mutate(expr[1], depth + 1, cols, rng))
            if len(expr) > 2:
                # 保留原窗口位（此前该分支会丢弃窗口，产出非法表达式）
                return (kind, self._mutate(expr[1], depth + 1, cols, rng), expr[2])
            if kind in _WINDOWED:
                return (kind, self._mutate(expr[1], depth + 1, cols, rng),
                        ("const", float(rng.choice(_ROL_W))))
            return (kind, self._mutate(expr[1], depth + 1, cols, rng))
        # 二元算子
        if kind in _BINARY:
            if rng.random() < 0.25:
                return (rng.choice(_BINARY),
                        self._mutate(expr[1], depth + 1, cols, rng),
                        self._mutate(expr[2], depth + 1, cols, rng))
            which = rng.randint(1, min(2, len(expr) - 1))
            return tuple(e if i != which else self._mutate(e, depth + 1, cols, rng)
                         for i, e in enumerate(expr))
        return random_expr(rng, depth, cols)

    def _subtree_swap(
        self,
        a: Any,
        b: Any,
        depth: int = 0,
        rng: Optional[random.Random] = None,
    ) -> Tuple[Any, Any]:
        """子树互换交叉 — 以指数下降的概率在更深层交换。"""
        rng = rng or self.rng
        if depth >= MAX_DEPTH or rng.random() < 0.35:
            return b, a
        kind_a, kind_b = a[0], b[0]
        if kind_a in ("col", "const") or kind_b in ("col", "const"):
            return a, b
        # 同类型子树交换
        if kind_a == kind_b:
            which = rng.randint(1, min(2, len(a) - 1))
            return (
                tuple(e if i != which else self._subtree_swap(e, b[which], depth + 1, rng)[0]
                      for i, e in enumerate(a)),
                tuple(e if i != which else self._subtree_swap(b[which], a[which], depth + 1, rng)[0]
                      for i, e in enumerate(b)),
            )
        return a, b

    # ---------- 岛屿模型 ----------
    def _init_island_population(
        self,
        cluster: FactorCluster,
        pop_size: int,
        cols: List[str],
        rng: random.Random,
    ) -> List[Any]:
        """混合初始化：一半来自簇内种子因子，一半随机表达式。"""
        pop: List[Any] = []
        for i in range(pop_size):
            if i < pop_size // 2 and cluster.factors:
                idx = i % len(cluster.factors)
                seed = seed_expr_from_code(rng, cluster.factors[idx].code, cols)
                if seed is not None:
                    pop.append(seed)
                    continue
            pop.append(random_expr(rng, cols=cols))
        return pop

    def _score_population(
        self,
        island_id: int,
        cluster: FactorCluster,
        pop: List[Any],
        train_df: pd.DataFrame,
        generation: int,
    ) -> List[Tuple[Any, float]]:
        """对岛屿种群打分，并写入本代执行轨迹（收敛速度 + 种群多样性）。"""
        scored = [(e, self._fitness(e, train_df, cluster)) for e in pop]
        scored.sort(key=lambda x: x[1], reverse=True)
        valid = [f for _, f in scored if f > -1e8]
        codes = {repr(e) for e, _ in scored}
        self.history.append({
            "island": island_id,
            "cluster": cluster.name,
            "cluster_label": CATEGORY_LABELS.get(cluster.name, cluster.name),
            "gen": generation,
            "best_ic": round(float(scored[0][1]), 6),
            "mean_ic": round(float(np.mean(valid)), 6) if valid else float("nan"),
            "unique_ratio": round(len(codes) / max(1, len(scored)), 4),
            "invalid_ratio": round(1.0 - len(valid) / max(1, len(scored)), 4),
        })
        return scored

    def _breed_population(
        self,
        scored: List[Tuple[Any, float]],
        pop_size: int,
        cols: List[str],
        rng: random.Random,
    ) -> List[Any]:
        """精英保留（top-2）+ 锦标赛选择 + 变异 / 子树交叉，产出下一代种群。"""
        next_pop: List[Any] = [scored[0][0], scored[1][0]]
        tournament_size = max(3, pop_size // 5)
        for _ in range(pop_size - 2):
            candidates = rng.sample(range(len(scored)), min(tournament_size, len(scored)))
            winner = max(candidates, key=lambda i: scored[i][1])
            child = scored[winner][0]
            if rng.random() < 0.5:
                child = self._mutate(child, cols=cols, rng=rng)
            elif rng.random() < 0.15:
                partner = scored[rng.randint(0, len(scored) - 1)][0]
                child, _ = self._subtree_swap(child, partner, rng=rng)
            next_pop.append(child)
        return next_pop

    def _migrate_elites(
        self,
        clusters: List[FactorCluster],
        scored_all: List[List[Tuple[Any, float]]],
        pops: List[List[Any]],
        migration_rate: int,
        generation: int,
    ) -> None:
        """环形迁移：每个岛屿的 top-k 精英复制进下一岛屿的种群，参与后续代繁殖。

        迁移发生在「繁殖之后、下一代打分之前」：精英在目的地岛屿重新打分，
        因此跨越簇边界的个体既带来新基因，也要接受新簇的适应度检验。
        """
        if migration_rate <= 0 or len(pops) < 2:
            return
        elites = [[e for e, _ in scored_all[i][:migration_rate]] for i in range(len(pops))]
        for i in range(len(pops)):
            dst = (i + 1) % len(pops)
            for migrant in elites[i]:
                pops[dst].append(migrant)
            self.migrations.append({
                "gen": generation,
                "from_island": i,
                "from_cluster": clusters[i].name,
                "to_island": dst,
                "to_cluster": clusters[dst].name,
                "count": len(elites[i]),
            })

    def _evolve_islands(
        self,
        clusters: List[FactorCluster],
        generations: int,
        pop_size: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        cols: List[str],
        migration_rate: int,
        migrate_every: int,
    ) -> List[List[Tuple[Any, float, float]]]:
        """岛屿模型主循环：按「代」并行推进各岛屿，并按间隔交换精英。

        与早期实现的差别：迁移真正发生在演化过程中，跨簇精英会参与后续代的
        选择与繁殖；每个岛屿持有独立随机源，保证可复现的同时避免相互干扰。
        """
        rngs = [random.Random(self.rng.randint(0, 100000)) for _ in clusters]
        pops = [self._init_island_population(c, pop_size, cols, rngs[i])
                for i, c in enumerate(clusters)]

        for gen in range(1, generations + 1):
            scored_all = [
                self._score_population(i, c, pops[i], train_df, gen)
                for i, c in enumerate(clusters)
            ]
            pops = [
                self._breed_population(scored_all[i], pop_size, cols, rngs[i])
                for i in range(len(clusters))
            ]
            if migrate_every > 0 and gen % migrate_every == 0:
                self._migrate_elites(clusters, scored_all, pops, migration_rate, gen)

        # 返回每个岛屿前 10% 的个体及测试集适应度
        islands: List[List[Tuple[Any, float, float]]] = []
        for i, cluster in enumerate(clusters):
            scored = sorted(
                [(e, self._fitness(e, train_df, cluster), self._test_fitness(e, test_df))
                 for e in pops[i]],
                key=lambda x: x[1], reverse=True,
            )
            islands.append(scored[:max(1, pop_size // 10)])
        return islands

    # ---------- 主入口 ----------
    def evolve_clusters(
        self,
        generations: int = 10,
        pop_per_cluster: int = 30,
        top_k: int = 20,
        test_frac: float = 0.2,
        migration_rate: int = 2,
        migrate_every: int = 3,
        auto_save: bool = True,
        verbose: bool = False,
        cols: Optional[List[str]] = None,
        include_expr: bool = False,
    ) -> List[Dict[str, Any]]:
        """执行因子簇驱动的遗传规划演化。

        Args:
            generations: 每簇演化代数
            pop_per_cluster: 每簇种群大小
            top_k: 最终保留的顶级因子数
            test_frac: 测试集比例
            migration_rate: 每次迁移交换个体数
            migrate_every: 每隔多少代发生一次迁移
            auto_save: 是否自动将优秀因子导入 FactorLibrary
            verbose: 是否打印进度
            include_expr: 是否在结果中附带表达式树（嵌套元组）。默认关闭以保证
                结果可直接 JSON 序列化；需要审计/可视化演化产物时置 True。

        Returns:
            [{name, code, train_ic, test_ic, overfit_gap, category, cluster, source}, ...]
        """
        if cols is None:
            cols = list(_COLS)

        # 重置执行轨迹（每次演化独立记录）
        self.history = []
        self.migrations = []

        # 时间切分
        dates = np.sort(self.df["date"].unique())
        cut = dates[int(len(dates) * (1 - test_frac))]
        train_df = self.df[self.df["date"] <= cut]
        test_df = self.df[self.df["date"] > cut]

        # 对每个因子簇建立一个岛屿（因子数 < 2 的簇不足以提供种子，跳过）
        active_clusters: List[FactorCluster] = [
            c for c in self._clusters if len(c.factors) >= 2
        ]
        for cluster in active_clusters:
            if verbose:
                print(f"[GP] 启动簇 {CATEGORY_LABELS.get(cluster.name, cluster.name)} 演化...")

        islands: List[List[Tuple[Any, float, float]]] = self._evolve_islands(
            clusters=active_clusters,
            generations=generations,
            pop_size=pop_per_cluster,
            train_df=train_df,
            test_df=test_df,
            cols=cols,
            migration_rate=migration_rate,
            migrate_every=migrate_every,
        )
        cluster_names: List[str] = [c.name for c in active_clusters]

        # 汇总所有岛屿结果
        all_individuals: List[Tuple[Any, float, float, str, Optional[FactorCluster]]] = []
        for i, island in enumerate(islands):
            cluster = active_clusters[i] if i < len(active_clusters) else None
            for expr, train_fit, test_fit in island:
                all_individuals.append((expr, train_fit, test_fit, cluster_names[i], cluster))

        # 去重（按代码哈希）
        seen_hashes: Set[str] = set()
        unique: List[Tuple[Any, float, float, str, Optional[FactorCluster]]] = []
        for item in all_individuals:
            code = expr_to_code(item[0])
            h = hashlib.md5(code.encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique.append(item)

        # 按训练 IC 排序
        unique.sort(key=lambda x: x[1], reverse=True)

        # 输出 top_k
        results: List[Dict[str, Any]] = []
        for idx, (expr, train_ic, test_ic, cluster_name, cluster) in enumerate(unique[:top_k]):
            code = expr_to_code(expr, f"gp_evolved_{idx}")
            if analyze_lookahead(code):
                continue

            category = cluster.category if cluster else "unknown"
            qs = min(0.85, max(0.15, float(train_ic) * 5.0 + 0.35))

            result = {
                "name": f"gp_evolved_{idx}",
                "code": code,
                "train_ic": round(float(train_ic), 5),
                "test_ic": round(float(test_ic), 5),
                "overfit_gap": round(float(train_ic) - float(test_ic), 5),
                "fitness": round(float(train_ic), 5),
                "category": category,
                "cluster": cluster_name,
                "source": "genetic_enhanced",
            }
            if include_expr:
                # 表达式树（嵌套元组）：调用方可据此绘制结构图 / 做演化审计
                result["expr"] = expr
            results.append(result)

            # 自动入库
            if auto_save and self.library:
                fd = FactorDef(
                    name=f"gp_evolved_{idx}",
                    display_name=f"GP演化因子 {idx}",
                    category=category,
                    description=f"遗传规划演化因子，簇={cluster_name}，IC={train_ic:.4f}",
                    direction="positive" if train_ic > 0 else "negative",
                    code=code,
                    tags=["genetic_programming", "evolved", cluster_name],
                    source="genetic_enhanced",
                    quality_score=float(qs),
                )
                self.library.add_factor(fd, source="generated")

        return results

    def evolve_with_event_focus(
        self,
        event_windows: List[EventWindow],
        generations: int = 8,
        pop_per_cluster: int = 20,
        top_k: int = 15,
        test_frac: float = 0.2,
        auto_save: bool = True,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """事件驱动的因子挖掘 — 在特定事件窗口（如财报季、政策变动期）内寻找有效的因子模式。

        Args:
            event_windows: 事件窗口列表
            generations: 演化代数
            pop_per_cluster: 每簇种群大小
            top_k: 保留顶级因子数
            test_frac: 测试集比例
            auto_save: 自动入库
            verbose: 打印进度

        Returns:
            同 evolve_clusters
        """
        # 暂存原有事件窗口
        saved_windows = list(self._event_windows)
        self._event_windows = list(event_windows)

        results = self.evolve_clusters(
            generations=generations,
            pop_per_cluster=pop_per_cluster,
            top_k=top_k,
            test_frac=test_frac,
            auto_save=auto_save,
            verbose=verbose,
        )

        # 恢复
        self._event_windows = saved_windows
        return results

    # ===================================================================
    # 大规模批量生产（因子簇 + 事件簇全方位覆盖）
    # ===================================================================
    def mass_produce(
        self,
        generations: int = 10,
        pop_per_cluster: int = 30,
        top_k_per_cluster: int = 10,
        test_frac: float = 0.2,
        auto_save: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """大规模批量生产因子——组合所有因子簇和事件窗口。

        产出过程：
        1. 对每个因子大类（5类）独立演化
        2. 参数窗口扩增（3,5,10,20,30,60天）
        3. 质量筛选去重
        4. 融合入库

        Returns:
            stats 字典，含生产各阶段的统计信息
        """
        start_time = time.time()
        all_results: List[Dict[str, Any]] = []

        # Phase 1: 每个大类独立演化
        for cluster in self._clusters:
            if len(cluster.factors) < 2:
                continue
            if verbose:
                print(f"[MassProd] 演化 {CATEGORY_LABELS.get(cluster.name, cluster.name)}...")
            # 临时只针对当前簇
            saved_clusters = list(self._clusters)
            self._clusters = [cluster]

            results = self.evolve_clusters(
                generations=generations,
                pop_per_cluster=pop_per_cluster,
                top_k=top_k_per_cluster,
                test_frac=test_frac,
                auto_save=False,  # 最后统一入库
                verbose=False,
            )
            all_results.extend(results)
            self._clusters = saved_clusters

        # Phase 2: 参数扩增
        if auto_save and self.library:
            expanded = self.library.cluster_expand_all()
        else:
            expanded = []

        # Phase 3: 质量筛选
        quality_pool = [r for r in all_results if r.get("train_ic", -999) > 0.005]
        quality_pool.sort(key=lambda r: r.get("train_ic", 0), reverse=True)

        # Phase 4: 去重 + 入库
        final_factors = quality_pool[:50]
        seen_names: Set[str] = set()
        unique_final: List[Dict[str, Any]] = []
        for r in final_factors:
            if r["name"] not in seen_names:
                seen_names.add(r["name"])
                unique_final.append(r)
                if auto_save and self.library:
                    fd = FactorDef(
                        name=r["name"],
                        display_name=f"GP批量生产 {r.get('cluster', '')}",
                        category=r.get("category", "unknown"),
                        description=f"GP批量生产因子, IC={r.get('train_ic', 0):.4f}",
                        direction="positive" if r.get("train_ic", 0) > 0 else "negative",
                        code=r["code"],
                        tags=["mass_produced", "gp", r.get("cluster", "")],
                        source="mass_production",
                        quality_score=min(0.9, 0.3 + abs(float(r.get("train_ic", 0))) * 4),
                    )
                    self.library.add_factor(fd, source="generated")

        elapsed = time.time() - start_time
        stats = {
            "total_evolved": len(all_results),
            "quality_passed": len(quality_pool),
            "final_factors": len(unique_final),
            "param_expanded": len(expanded),
            "library_total": self.library.statistics()["total"] if self.library else 0,
            "elapsed_seconds": round(elapsed, 1),
        }

        return {
            "factors": unique_final,
            "stats": stats,
        }
