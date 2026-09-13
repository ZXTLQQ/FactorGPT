"""符号回归 / 遗传编程因子发现 (src/engine/genetic_factors.py)

与 LLM 生成互补的因子发现方式：用遗传编程演化因子表达式树，以截面 IC 为适应度。
相比 LLM 生成，遗传编程的结果更可复现、更可解释、不依赖网络/大模型，两者互相印证
会很有说服力。

特点：
- 零重依赖（仅 numpy / pandas），所有表达式求值都用 pandas groupby，结果可直接
  转成符合沙箱契约的 alpha_factor 代码字符串；
- 生成代码统一对最终因子 shift(1)，并经过 analyze_lookahead 静态前视检查，杜绝未来函数；
- 适应度用 IC（因子与下一日收益的截面相关均值），训练集/测试集可分离以评估过拟合。

典型用法：
    miner = GeneticFactorMiner(kline)
    results = miner.mine(generations=12, pop_size=60, top_k=5)
    # results: [{name, code, ic, icir, fitness}, ...]，code 可直接交给 FactorSandbox.run
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .factor_builder import analyze_lookahead

_COLS = ["open", "high", "low", "close", "volume", "amount", "pct_chg"]
_ROL_W = [3, 5, 10, 20, 60]
_MAX_DEPTH = 3


def _groupwise_corr(f: np.ndarray, y: np.ndarray, codes: np.ndarray,
                    n_groups: int) -> np.ndarray:
    """按组（截面）计算 Pearson 相关，**向量化**：一次 bincount 聚合完所有截面。

    等价于 ``panel.groupby("date").apply(lambda g: g["f"].corr(g["y"]))``，
    但适应度函数每一代要被调用上千次，``groupby.apply`` 的 Python 层开销是主要
    瓶颈；显式分组码还避免了 pandas 未来版本剔除分组列带来的语义变化。
    组内方差为 0（样本 < 2 或常数截面）时返回 NaN。
    """
    cnt = np.bincount(codes, minlength=n_groups).astype(float)
    safe = np.maximum(cnt, 1.0)
    mf = np.bincount(codes, weights=f, minlength=n_groups) / safe
    my = np.bincount(codes, weights=y, minlength=n_groups) / safe
    df_ = f - mf[codes]
    dy_ = y - my[codes]
    cov = np.bincount(codes, weights=df_ * dy_, minlength=n_groups)
    vx = np.bincount(codes, weights=df_ * df_, minlength=n_groups)
    vy = np.bincount(codes, weights=dy_ * dy_, minlength=n_groups)
    den = np.sqrt(vx * vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = cov / den
    return np.where((den > 0) & np.isfinite(out), out, np.nan)


class GeneticFactorMiner:
    """用遗传编程演化因子表达式树，适应度 = 样本内 IC。"""

    def __init__(self, kline: pd.DataFrame, seed: int = 0) -> None:
        self.df = kline.copy()
        if "date" not in self.df.columns:
            self.df = self.df.reset_index()
        # 未来收益（下一日 pct_chg），作为 IC 目标
        self.df = self.df.sort_values(["symbol", "date"]).reset_index(drop=True)
        self.df["_fwd_ret"] = self.df.groupby("symbol")["pct_chg"].shift(-1)
        self.rng = random.Random(seed)

    # ---------- 表达式树：构建 / 求值 / 序列化 ----------
    def _random_expr(self, depth: int = 0) -> Any:
        if depth >= _MAX_DEPTH or (depth > 0 and self.rng.random() < 0.3):
            if self.rng.random() < 0.75:
                return ("col", self.rng.choice(_COLS))
            return ("const", round(self.rng.uniform(-1, 1), 3))
        op = self.rng.choice(["add", "sub", "mul", "div", "log", "abs", "rol"])
        if op == "rol":
            child = self._random_expr(depth + 1)
            return ("rol", child, self.rng.choice(_ROL_W))
        if op in ("log", "abs"):
            return (op, self._random_expr(depth + 1))
        return (op, self._random_expr(depth + 1), self._random_expr(depth + 1))

    def _eval(self, expr, df: pd.DataFrame) -> pd.Series:
        """把表达式树求值为与 df 行对齐的 Series（所有滚动都在 symbol 分组内）。"""
        kind = expr[0]
        if kind == "col":
            return df[expr[1]].astype(float)
        if kind == "const":
            return pd.Series(float(expr[1]), index=df.index)
        if kind == "rol":
            child = self._eval(expr[1], df)
            w = expr[2]
            return child.groupby(df["symbol"]).transform(lambda s: s.rolling(w, min_periods=1).mean())
        if kind == "log":
            return np.log(np.abs(self._eval(expr[1], df)) + 1e-12)
        if kind == "abs":
            return np.abs(self._eval(expr[1], df))
        a = self._eval(expr[1], df)
        b = self._eval(expr[2], df)
        if kind == "add":
            return a + b
        if kind == "sub":
            return a - b
        if kind == "mul":
            return a * b
        if kind == "div":
            return a / (b.replace(0, np.nan) + 1e-12)
        raise ValueError(f"未知算子 {kind}")

    def _to_code(self, expr) -> str:
        kind = expr[0]
        if kind == "col":
            return f"df['{expr[1]}']"
        if kind == "const":
            return repr(float(expr[1]))
        if kind == "rol":
            return f"df.groupby('symbol')[{self._to_code(expr[1])[4:-2]}].transform(lambda s: s.rolling({expr[2]}, min_periods=1).mean())" if False else f"({self._to_code(expr[1])}).groupby(df['symbol']).transform(lambda s: s.rolling({expr[2]}, min_periods=1).mean())"
        if kind == "log":
            return f"np.log(np.abs({self._to_code(expr[1])}) + 1e-12)"
        if kind == "abs":
            return f"np.abs({self._to_code(expr[1])})"
        sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}.get(kind, "+")
        return f"({self._to_code(expr[1])} {sym} {self._to_code(expr[2])})"

    def _to_pandas_expr(self, expr) -> str:
        """生成作用在 df 上的 pandas 表达式字符串。"""
        kind = expr[0]
        if kind == "col":
            return f"df['{expr[1]}']"
        if kind == "const":
            return repr(float(expr[1]))
        if kind == "rol":
            return f"({self._to_pandas_expr(expr[1])}).groupby(df['symbol']).transform(lambda s: s.rolling({expr[2]}, min_periods=1).mean())"
        if kind == "log":
            return f"np.log(np.abs({self._to_pandas_expr(expr[1])}) + 1e-12)"
        if kind == "abs":
            return f"np.abs({self._to_pandas_expr(expr[1])})"
        sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}.get(kind, "+")
        return f"({self._to_pandas_expr(expr[1])} {sym} {self._to_pandas_expr(expr[2])})"

    def _expr_to_code(self, expr, name: str) -> str:
        expr_str = self._to_pandas_expr(expr)
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

    # ---------- 适应度 ----------
    def _fitness(self, expr, train_df: pd.DataFrame) -> float:
        try:
            fac = self._eval(expr, train_df)
            # 直接用 train_df 同序的 date 列构建 panel，避免用默认索引去 loc 取 date 的错位
            panel = pd.DataFrame({
                "f": np.asarray(fac, dtype=float),
                "y": train_df["_fwd_ret"].to_numpy(dtype=float),
                "date": train_df["date"].to_numpy(),
            })
            panel = panel.replace([np.inf, -np.inf], np.nan).dropna()
            if len(panel) < 50:
                return -1e9
            codes, uniq = pd.factorize(panel["date"].to_numpy())
            ic = _groupwise_corr(panel["f"].to_numpy(dtype=float),
                                 panel["y"].to_numpy(dtype=float),
                                 codes, len(uniq))
            ic = ic[np.isfinite(ic)]
            if ic.size == 0:
                return -1e9
            return float(ic.mean())

        except Exception:
            return -1e9

    # ---------- 遗传操作 ----------
    def _mutate(self, expr, depth: int = 0) -> Any:
        if depth > _MAX_DEPTH or self.rng.random() < 0.25:
            return self._random_expr(depth)
        if expr[0] in ("col", "const"):
            return self._random_expr(depth)
        if expr[0] == "rol":
            return ("rol", self._mutate(expr[1], depth + 1), expr[2])
        if expr[0] in ("log", "abs"):
            return (expr[0], self._mutate(expr[1], depth + 1))
        which = self.rng.randint(1, 2)
        new = list(expr)
        new[which] = self._mutate(expr[which], depth + 1)
        return tuple(new)

    def _crossover(self, a, b) -> Tuple[Any, Any]:
        # 简单整树交叉：以小概率直接互换子树
        if self.rng.random() < 0.5:
            return b, a
        return a, b

    def mine(
        self,
        generations: int = 12,
        pop_size: int = 60,
        top_k: int = 5,
        test_frac: float = 0.2,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """演化搜索，返回 top_k 个因子的 {name, code, ic, icir, fitness}。"""
        # 按时间切分（每个 symbol 都用前段训练、后段测试），避免按行切分导致的截面错配与标签泄漏
        dates = np.sort(self.df["date"].unique())
        cut = dates[int(len(dates) * (1 - test_frac))]
        train_df = self.df[self.df["date"] <= cut]
        test_df = self.df[self.df["date"] > cut]

        pop = [self._random_expr() for _ in range(pop_size)]
        best_expr = None
        best_fit = -1e9
        for gen in range(generations):
            scored = [(e, self._fitness(e, train_df)) for e in pop]
            scored.sort(key=lambda x: x[1], reverse=True)
            if scored[0][1] > best_fit:
                best_fit, best_expr = scored[0][1], scored[0][0]
            if verbose:
                print(f"[genetic] gen {gen+1}/{generations} bestIC={scored[0][1]:.4f}")
            # 精英保留
            next_pop = [scored[0][0], scored[1][0]]
            while len(next_pop) < pop_size:
                i, j = self.rng.sample(range(len(scored)), 2)
                c1, c2 = self._crossover(scored[i][0], scored[j][0])
                if self.rng.random() < 0.4:
                    c1 = self._mutate(c1)
                if self.rng.random() < 0.4:
                    c2 = self._mutate(c2)
                next_pop.append(c1)
                if len(next_pop) < pop_size:
                    next_pop.append(c2)
            pop = next_pop

        # 最终评估 top 个体（按训练适应度排序）
        scored = sorted(((e, self._fitness(e, train_df)) for e in pop), key=lambda x: x[1], reverse=True)
        results: List[Dict[str, Any]] = []
        for idx, (expr, fit) in enumerate(scored[:top_k]):
            # 测试集 IC（过拟合体检）
            test_fit = self._fitness(expr, test_df)
            code = self._expr_to_code(expr, f"genetic_{idx}")
            if analyze_lookahead(code):
                continue  # 理论上不会触发，双保险
            results.append({
                "name": f"genetic_{idx}",
                "code": code,
                "train_ic": round(fit, 5),
                "test_ic": round(test_fit, 5),
                "overfit_gap": round(fit - test_fit, 5),
                "fitness": round(fit, 5),
            })
        return results
