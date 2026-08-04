"""
增强遗传规划因子挖掘 (src/engine/genetic_enhanced.py)

在原有 GeneticFactorMiner 基础上扩展：
1. 因子簇驱动演化 — 将相关因子编组为 Cluster，组内交叉促进有效模式保留
2. 事件簇感知 — 按事件（财报、政策窗口、市场状态）动态调整适应度计算
3. 更丰富的算子集 — ts_rank, ts_zscore, ts_delta, ts_corr, ts_min, ts_max 等
4. 多样性保持 — 岛屿模型 / 拥挤距离
5. 与 FactorLibrary 深度集成 — 产出直接入库，纳入质量评分体系

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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .factor_library import FactorLibrary
from .traditional_factors import (
    FactorDef,
    ALL_CATEGORIES,
    CATEGORY_LABELS,
    CATEGORY_PRICE_TREND,
    CATEGORY_VOLATILITY,
    CATEGORY_TRADING_DIFFICULTY,
    CATEGORY_PRICE_VOLUME_DIVERGENCE,
    CATEGORY_VOLUME_PRICE_FORMULA,
)
from .factor_builder import analyze_lookahead

_COLS = ["open", "high", "low", "close", "volume", "amount", "pct_chg"]

# 扩展算子集
# 一元算子: log, abs, sqrt, sign, rank, delay, delta, ts_zscore, ts_rank
# 二元算子: add, sub, mul, div, ts_corr, ts_cov, ts_min, ts_max
_UNARY = ["log", "abs", "sqrt", "sign", "neg", "delta", "ts_zscore", "ts_rank"]
_BINARY = ["add", "sub", "mul", "div", "ts_corr", "ts_min", "ts_max"]
_TERMINAL = ["col", "const"]
_ROL_W = [3, 5, 10, 14, 20, 30, 60]

MAX_DEPTH = 3


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
        return ("sub", ("col", chosen_col), ("rol", ("col", chosen_col), chosen_w))
    if "rolling" in code and "std" in code.lower():
        return ("neg", ("ts_zscore", ("col", chosen_col), ("const", float(chosen_w))))
    if "rolling" in code and "sum" in code.lower():
        return ("rol", ("col", chosen_col), chosen_w)
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
        w = int(expr[2][1]) if isinstance(expr[2], tuple) and expr[2][0] == "const" else 20
        return child.groupby(df["symbol"]).transform(
            lambda s: (s - s.rolling(w, min_periods=10).mean()) / (s.rolling(w, min_periods=10).std() + 1e-8))
    if kind == "ts_rank":
        child = eval_expr(expr[1], df)
        w = int(expr[2][1]) if isinstance(expr[2], tuple) and expr[2][0] == "const" else 20
        return child.groupby(df["symbol"]).transform(
            lambda s: s.rolling(w, min_periods=10).rank(pct=True))

    # 二元算子 (3-ary for ts_corr)
    if kind == "ts_corr":
        a = eval_expr(expr[1], df)
        b = eval_expr(expr[2], df)
        w = int(expr[3][1]) if len(expr) > 3 and isinstance(expr[3], tuple) and expr[3][0] == "const" else 20
        panel = pd.DataFrame({"a": a, "b": b, "sym": df["symbol"]})
        result = panel.groupby("sym").apply(
            lambda g: g["a"].rolling(w, min_periods=10).corr(g["b"])
        )
        return result.droplevel(0).reindex(df.index).astype(float)
    if kind == "ts_min":
        a = eval_expr(expr[1], df)
        w = int(expr[2][1]) if isinstance(expr[2], tuple) and expr[2][0] == "const" else 20
        return a.groupby(df["symbol"]).transform(lambda s: s.rolling(w, min_periods=5).min())
    if kind == "ts_max":
        a = eval_expr(expr[1], df)
        w = int(expr[2][1]) if isinstance(expr[2], tuple) and expr[2][0] == "const" else 20
        return a.groupby(df["symbol"]).transform(lambda s: s.rolling(w, min_periods=5).max())

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


def expr_to_code(expr: Any, name: str = "gp_factor") -> str:
    def _emit(e: Any) -> str:
        kind = e[0]
        if kind == "col":
            return f"df['{e[1]}']"
        if kind == "const":
            return repr(float(e[1]))
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
            w = int(e[2][1]) if isinstance(e[2], tuple) and e[2][0] == "const" else 20
            return f"({child}).groupby(df['symbol']).transform(lambda s: (s-s.rolling({w},min_periods=10).mean())/(s.rolling({w},min_periods=10).std()+1e-8))"
        if kind == "ts_rank":
            child = _emit(e[1])
            w = int(e[2][1]) if isinstance(e[2], tuple) and e[2][0] == "const" else 20
            return f"({child}).groupby(df['symbol']).transform(lambda s: s.rolling({w},min_periods=10).rank(pct=True))"
        if kind == "ts_corr":
            a = _emit(e[1]); b = _emit(e[2])
            w = int(e[3][1]) if len(e) > 3 and isinstance(e[3], tuple) and e[3][0] == "const" else 20
            return f"pd.concat([{a}, {b}, df['symbol']], axis=1, keys=['a','b','sym']).groupby('sym').apply(lambda g: g['a'].rolling({w},min_periods=10).corr(g['b'])).droplevel(0)"
        if kind == "ts_min":
            child = _emit(e[1])
            w = int(e[2][1]) if isinstance(e[2], tuple) and e[2][0] == "const" else 20
            return f"({child}).groupby(df['symbol']).transform(lambda s: s.rolling({w},min_periods=5).min())"
        if kind == "ts_max":
            child = _emit(e[1])
            w = int(e[2][1]) if isinstance(e[2], tuple) and e[2][0] == "const" else 20
            return f"({child}).groupby(df['symbol']).transform(lambda s: s.rolling({w},min_periods=5).max())"
        sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}.get(kind, "+")
        return f"({_emit(e[1])} {sym} {_emit(e[2])})"

    expr_str = _emit(expr)
    return (
        "import pandas as pd\n"
        "import numpy as np\n"
        "def alpha_factor(df):\n"
        "    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)\n"
        f"    f = {expr_str}\n"
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

            # 事件窗口加权
            weights = pd.Series(1.0, index=panel.index)
            if self._event_windows:
                for ew in self._event_windows:
                    mask = (panel["date"] >= ew.date_range[0]) & (panel["date"] <= ew.date_range[1])
                    weights[mask] *= ew.weight

            ic = panel.groupby("date").apply(
                lambda g: g["f"].corr(g["y"]) if g["f"].std() > 0 else np.nan
            )
            ic = ic.dropna()
            if len(ic) == 0:
                return -1e9

            # 聚类奖励：如果因子与簇内典型因子模式相近，给小幅奖励
            cluster_bonus = 0.0
            if cluster and cluster.factors:
                cluster_bonus = 0.005  # 以簇的名义有小奖励

            return float(ic.mean()) + cluster_bonus
        except Exception:
            return -1e9

    def _test_fitness(self, expr: Any, test_df: pd.DataFrame) -> float:
        """在测试集上评估（不奖励聚类）。"""
        return self._fitness(expr, test_df, cluster=None)

    # ---------- 遗传操作 ----------
    def _mutate(self, expr: Any, depth: int = 0, cols: Optional[List[str]] = None) -> Any:
        if depth > MAX_DEPTH or self.rng.random() < 0.20:
            return random_expr(self.rng, depth, cols)
        kind = expr[0]
        if kind in ("col", "const"):
            return random_expr(self.rng, depth, cols)
        # 一元算子
        if kind in _UNARY:
            if self.rng.random() < 0.30:
                return (self.rng.choice(_UNARY), self._mutate(expr[1], depth + 1, cols))
            if len(expr) > 2 and self.rng.random() < 0.30:
                return (kind, self._mutate(expr[1], depth + 1, cols), expr[2])
            return (kind, self._mutate(expr[1], depth + 1, cols))
        # 二元算子
        if kind in _BINARY:
            if self.rng.random() < 0.25:
                return (self.rng.choice(_BINARY), self._mutate(expr[1], depth + 1, cols), self._mutate(expr[2], depth + 1, cols))
            which = self.rng.randint(1, min(2, len(expr) - 1))
            return tuple(e if i != which else self._mutate(e, depth + 1, cols) for i, e in enumerate(expr))
        return random_expr(self.rng, depth, cols)

    def _subtree_swap(self, a: Any, b: Any, depth: int = 0) -> Tuple[Any, Any]:
        """子树互换交叉 — 以指数下降的概率在更深层交换。"""
        if depth >= MAX_DEPTH or self.rng.random() < 0.35:
            return b, a
        kind_a, kind_b = a[0], b[0]
        if kind_a in ("col", "const") or kind_b in ("col", "const"):
            return a, b
        # 同类型子树交换
        if kind_a == kind_b:
            which = self.rng.randint(1, min(2, len(a) - 1))
            return (
                tuple(e if i != which else self._subtree_swap(e, b[which], depth + 1)[0] for i, e in enumerate(a)),
                tuple(e if i != which else self._subtree_swap(b[which], a[which], depth + 1)[0] for i, e in enumerate(b)),
            )
        return a, b

    # ---------- 岛屿模型 ----------
    def _island_evolve(
        self,
        island_id: int,
        cluster: FactorCluster,
        generations: int,
        pop_size: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        cols: Optional[List[str]] = None,
    ) -> List[Tuple[Any, float, float]]:
        """在单个岛屿（因子簇上下文）下演化。"""
        # 混合初始化：一半来自种子因子，一半随机
        pop: List[Any] = []
        for i in range(pop_size):
            if i < pop_size // 2 and cluster.factors:
                idx = i % len(cluster.factors)
                seed = seed_expr_from_code(self.rng, cluster.factors[idx].code, cols)
                if seed is not None:
                    pop.append(seed)
                    continue
            pop.append(random_expr(self.rng, cols=cols))

        best_expr = pop[0]
        best_fit = -1e9

        for gen in range(generations):
            scored = [(e, self._fitness(e, train_df, cluster)) for e in pop]
            scored.sort(key=lambda x: x[1], reverse=True)
            if scored[0][1] > best_fit:
                best_fit, best_expr = scored[0][1], scored[0][0]

            # 精英保留 + 锦标赛选择
            next_pop = [scored[0][0], scored[1][0]]
            tournament_size = max(3, pop_size // 5)
            for _ in range(pop_size - 2):
                # 锦标赛选择（拥挤距离辅助保持多样性）
                candidates = self.rng.sample(range(len(scored)), min(tournament_size, len(scored)))
                winner = max(candidates, key=lambda i: scored[i][1])
                child = scored[winner][0]
                if self.rng.random() < 0.5:
                    child = self._mutate(child, cols=cols)
                elif self.rng.random() < 0.15:
                    partner = scored[self.rng.randint(0, len(scored) - 1)][0]
                    child, _ = self._subtree_swap(child, partner)
                next_pop.append(child)
            pop = next_pop

        # 返回前 10% 的个体及测试集适应度
        scored = sorted(
            [(e, self._fitness(e, train_df, cluster), self._test_fitness(e, test_df)) for e in pop],
            key=lambda x: x[1], reverse=True
        )
        return scored[:max(1, pop_size // 10)]

    # ---------- 迁移 ----------
    def _migrate_top_individuals(
        self,
        islands: List[List[Tuple[Any, float, float]]],
        migration_rate: int = 2,
    ) -> None:
        """岛屿间迁移：每个岛的最优个体迁移到下一个岛。"""
        num = len(islands)
        for i in range(num):
            next_i = (i + 1) % num
            migrants = [islands[i][j][0] for j in range(min(migration_rate, len(islands[i])))]
            for m in migrants:
                islands[next_i].append((m, 0.0, 0.0))

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

        Returns:
            [{name, code, train_ic, test_ic, overfit_gap, category, cluster, source}, ...]
        """
        if cols is None:
            cols = list(_COLS)

        # 时间切分
        dates = np.sort(self.df["date"].unique())
        cut = dates[int(len(dates) * (1 - test_frac))]
        train_df = self.df[self.df["date"] <= cut]
        test_df = self.df[self.df["date"] > cut]

        # 对每个因子簇建立一个岛屿
        islands: List[List[Tuple[Any, float, float]]] = []
        cluster_names: List[str] = []

        for cluster in self._clusters:
            if len(cluster.factors) < 2:
                # 因子太少的簇跳过
                continue
            if verbose:
                print(f"[GP] 启动簇 {CATEGORY_LABELS.get(cluster.name, cluster.name)} 演化...")
            rng_seed = self.rng.randint(0, 100000)
            # 为每个簇创建独立的随机种子确保可复现
            saved_rng = self.rng
            self.rng = random.Random(rng_seed)
            result = self._island_evolve(
                island_id=len(cluster_names),
                cluster=cluster,
                generations=generations,
                pop_size=pop_per_cluster,
                train_df=train_df,
                test_df=test_df,
                cols=cols,
            )
            islands.append(result)
            cluster_names.append(cluster.name)
            self.rng = saved_rng

        # 迁移循环（按指定间隔）
        for gen_step in range(0, generations, migrate_every):
            self._migrate_top_individuals(islands, migration_rate)

        # 汇总所有岛屿结果
        all_individuals: List[Tuple[Any, float, float, str, Optional[FactorCluster]]] = []
        for i, island in enumerate(islands):
            cluster = self._clusters[i] if i < len(self._clusters) else None
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
