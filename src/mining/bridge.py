"""LLM 代码 ↔ 符号表达式的双向桥（src/mining/bridge.py）。

两条主线此前各挖各的：``agent`` 让 LLM 写 ``alpha_factor(df)``（Python），
``mining.gridminer`` 在符号空间里做分层网格搜索（DSL 表达式树），产出互不相干。
这座桥把它们接起来：

* :func:`code_to_expr`——把 LLM 写的 Python 翻译成 DSL 子树，作为**种子**交给
  网格搜索做局部精修（换窗口、换算子、加中性化）。翻译是**有损**的：识别不了
  的模式直接返回 ``None``，主线绝不能依赖它成功。
* :func:`expr_to_code`——把符号因子渲染成能在 ``FactorSandbox`` 里跑的代码，
  于是挖出来的因子能走同一套回测、也能交给 LLM 做归因解释。

有损是有意的：LLM 写的代码千变万化，硬要全覆盖只会得到一堆"看起来翻译成功、
实际语义漂移"的表达式。这里只认**高频可验证**的模式，其余老实说"翻不了"。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, List, Optional

from . import expr as ex
from . import ops

__all__ = ["Translation", "code_to_expr", "expr_to_code", "translate"]

# --------------------------------------------------------------------------
# DSL -> Python
# --------------------------------------------------------------------------
# 直接映射到 pandas rolling 方法的时序算子
_TS_METHOD: Dict[str, str] = {
    "ts_mean": "mean", "ts_std": "std", "ts_sum": "sum", "ts_min": "min",
    "ts_max": "max", "ts_median": "median", "ts_skew": "skew",
    "ts_count": "count",
}

# 需要辅助函数实现的时序算子：名 -> (helper 名, 是否带窗口)
_TS_APPLY: Dict[str, str] = {
    "ts_rank": "_ts_rank", "ts_product": "_ts_prod",
    "ts_decay_linear": "_ts_decay", "ts_slope": "_ts_slope",
    "ts_rsquare": "_ts_rsq", "ts_resi": "_ts_resi", "ts_ir": "_ts_ir",
    "ts_argmax": "_ts_argmax", "ts_argmin": "_ts_argmin",
}

_ELEM_EXPR: Dict[str, str] = {
    "neg": "-({a})", "abs": "({a}).abs()", "log": "np.log({a})",
    "log1p": "np.log1p({a})", "sqrt": "np.sqrt({a})", "square": "({a}) ** 2",
    "sign": "np.sign({a})", "tanh": "np.tanh({a})", "inv": "1.0 / ({a})",
    "clip3": "({a}).clip(-3.0, 3.0)",
}

_ELEM2_EXPR: Dict[str, str] = {
    "add": "({a}) + ({b})", "sub": "({a}) - ({b})", "mul": "({a}) * ({b})",
    "div": "({a}) / ({b})", "max2": "np.maximum({a}, {b})",
    "min2": "np.minimum({a}, {b})", "pow2": "({a}) ** ({b})",
    "geom_mean": "np.sign(({a}) * ({b})) * np.sqrt((({a}) * ({b})).abs())",
    "harm_mean": "2.0 * ({a}) * ({b}) / (({a}) + ({b}))",
    "spread": "({a}) / (({a}) + ({b}))",
}

_HELPERS = '''def _cs_zscore(x):
    a = np.asarray(x, dtype=float)
    m, s = np.nanmean(a), np.nanstd(a)
    return (a - m) / s if s > 0 else np.zeros_like(a)


def _cs_scale(x):
    a = np.asarray(x, dtype=float)
    d = np.nansum(np.abs(a))
    return a / d if d > 0 else np.zeros_like(a)


def _cs_winsor(x):
    a = np.asarray(x, dtype=float)
    med = np.nanmedian(a)
    mad = np.nanmedian(np.abs(a - med))
    if mad <= 0:
        return a
    return np.clip(a, med - 5 * mad, med + 5 * mad)


def _ts_rank(x):
    return (x[-1] >= np.asarray(x[:-1])).mean() if len(x) > 1 else np.nan


def _ts_prod(x):
    return np.nanprod(np.asarray(x, dtype=float))


def _ts_decay(x):
    a = np.asarray(x, dtype=float)
    w = np.arange(1, len(a) + 1, dtype=float)
    return float(np.nansum(a * w) / np.nansum(w))


def _ts_slope(x):
    a = np.asarray(x, dtype=float)
    m = ~np.isnan(a)
    if m.sum() < 2:
        return np.nan
    y, t = a[m], np.arange(len(a))[m].astype(float)
    return float(np.polyfit(t, y, 1)[0])


def _ts_rsq(x):
    a = np.asarray(x, dtype=float)
    m = ~np.isnan(a)
    if m.sum() < 3:
        return np.nan
    y, t = a[m], np.arange(len(a))[m].astype(float)
    c = np.corrcoef(t, y)
    return float(c[0, 1] ** 2) if np.isfinite(c[0, 1]) else 0.0


def _ts_resi(x):
    a = np.asarray(x, dtype=float)
    m = ~np.isnan(a)
    if m.sum() < 2:
        return np.nan
    y, t = a[m], np.arange(len(a))[m].astype(float)
    b, c = np.polyfit(t, y, 1)
    return float(a[-1] - (b * (len(a) - 1) + c))


def _ts_ir(x):
    a = np.asarray(x, dtype=float)
    s = np.nanstd(a)
    return float(np.nanmean(a) / s) if s > 0 else 0.0


def _ts_argmax(x):
    a = np.asarray(x, dtype=float)
    return float(len(a) - 1 - np.nanargmax(a)) if len(a) else np.nan


def _ts_argmin(x):
    a = np.asarray(x, dtype=float)
    return float(len(a) - 1 - np.nanargmin(a)) if len(a) else np.nan
'''


class UnsupportedOp(Exception):
    """渲染成代码时遇到不支持的算子（ts2 / cs2 / neutral 等）。"""


def expr_to_code(node: ex.Node, *, with_helpers: bool = True) -> str:
    """把 DSL 表达式渲染成可在 ``FactorSandbox`` 里执行的 ``alpha_factor`` 代码。

    :raises UnsupportedOp: 遇到 ts2 / cs2 / neutral 这类需要多列联合统计的算子。
    """
    lines: List[str] = []
    counter = [0]

    def _v() -> str:
        counter[0] += 1
        return f"v{counter[0] - 1}"

    def _emit(node: ex.Node) -> str:
        if isinstance(node, ex.Field):
            name = _v()
            lines.append(f"{name} = df[{node.name!r}]")
            return name
        if isinstance(node, ex.Const):
            name = _v()
            lines.append(f"{name} = pd.Series({float(node.value)}, index=df.index)")
            return name
        if isinstance(node, ex.Neutral):
            raise UnsupportedOp("neutral（中性化）暂不渲染为代码")
        if isinstance(node, ex.Call):
            return _emit_call(node)
        raise UnsupportedOp(f"未知节点类型 {type(node).__name__}")

    def _emit_call(c: ex.Call) -> str:
        try:
            fam = ops.get_op(c.name).family
        except KeyError:                    # pragma: no cover - 未注册算子
            fam = ""
        if fam == "ts2":
            raise UnsupportedOp(f"{c.name}（时序二元）暂不渲染为代码")
        if fam == "cs2":
            raise UnsupportedOp(f"{c.name}（横截面二元）暂不渲染为代码")
        args = [_emit(a) for a in c.args]
        out = _v()
        if c.name in _TS_METHOD:
            w = int(c.window or 5)
            lines.append(f"{out} = {args[0]}.groupby(df['symbol'])"
                         f".transform(lambda x: x.rolling({w})"
                         f".{_TS_METHOD[c.name]}())")
        elif c.name == "ts_delay":
            lines.append(f"{out} = {args[0]}.groupby(df['symbol'])"
                         f".shift({int(c.window or 1)})")
        elif c.name == "ts_delta":
            lines.append(f"{out} = {args[0]} - {args[0]}.groupby(df['symbol'])"
                         f".shift({int(c.window or 1)})")
        elif c.name in ("ts_pct", "ts_ret"):
            lines.append(f"{out} = {args[0]} / {args[0]}.groupby(df['symbol'])"
                         f".shift({int(c.window or 1)}) - 1.0")
        elif c.name in ("ts_yoy", "ts_qoq"):
            lines.append(f"{out} = {args[0]} / {args[0]}.groupby(df['symbol'])"
                         f".shift({int(c.window or 250)}) - 1.0")
        elif c.name == "ts_zscore":
            w = int(c.window or 20)
            lines.append(f"{out} = {args[0]}.groupby(df['symbol']).transform("
                         f"lambda x: (x - x.rolling({w}).mean()) "
                         f"/ x.rolling({w}).std())")
        elif c.name == "ts_max_diff":
            w = int(c.window or 20)
            lines.append(f"{out} = {args[0]} - {args[0]}.groupby(df['symbol'])"
                         f".transform(lambda x: x.rolling({w}).max())")
        elif c.name == "ts_min_diff":
            w = int(c.window or 20)
            lines.append(f"{out} = {args[0]} - {args[0]}.groupby(df['symbol'])"
                         f".transform(lambda x: x.rolling({w}).min())")
        elif c.name == "ema":
            w = int(c.window or 10)
            lines.append(f"{out} = {args[0]}.groupby(df['symbol']).transform("
                         f"lambda x: x.ewm(span={w}, adjust=False).mean())")
        elif c.name in _TS_APPLY:
            w = int(c.window or 20)
            lines.append(f"{out} = {args[0]}.groupby(df['symbol']).transform("
                         f"lambda x: x.rolling({w}).apply("
                         f"{_TS_APPLY[c.name]}, raw=False))")
        elif c.name == "rank_cs":
            lines.append(f"{out} = {args[0]}.groupby(df['date']).rank(pct=True)")
        elif c.name == "zscore_cs":
            lines.append(f"{out} = {args[0]}.groupby(df['date'])"
                         f".transform(_cs_zscore)")
        elif c.name == "demean_cs":
            lines.append(f"{out} = {args[0]} - {args[0]}.groupby(df['date'])"
                         f".transform('mean')")
        elif c.name == "scale_cs":
            lines.append(f"{out} = {args[0]}.groupby(df['date'])"
                         f".transform(_cs_scale)")
        elif c.name == "winsorize_cs":
            lines.append(f"{out} = {args[0]}.groupby(df['date'])"
                         f".transform(_cs_winsor)")
        elif c.name == "quantile_cs":
            k = int(c.window or 10)
            lines.append(f"{out} = {args[0]}.groupby(df['date']).transform("
                         f"lambda s: pd.qcut(s.rank(method='first'), {k}, "
                         f"labels=False))")
        elif c.name in _ELEM_EXPR:
            lines.append(f"{out} = {_ELEM_EXPR[c.name].format(a=args[0])}")
        elif c.name in _ELEM2_EXPR:
            lines.append(f"{out} = {_ELEM2_EXPR[c.name].format(a=args[0], b=args[1])}")
        else:
            raise UnsupportedOp(f"算子 {c.name} 暂无代码模板")
        return out

    last = _emit(node)
    body = "\n    ".join(lines)
    head = "import numpy as np\nimport pandas as pd\n"
    if with_helpers:
        head += "\n\n" + _HELPERS
    return (f"{head}\n\n\ndef alpha_factor(df):\n"
            f"    df = df.sort_values(['symbol', 'date']).copy()\n"
            f"    {body}\n"
            f"    df['factor'] = {last}\n"
            f"    return df[['date', 'symbol', 'factor']]\n")


def expr_to_series(node: ex.Node, df: Any) -> Any:
    """执行 :func:`expr_to_code` 生成的代码，返回 ``(date, symbol)`` 索引的因子。

    不走 ``FactorSandbox`` 的前视检查是**故意**的：DSL 侧的因果性由
    ``expr.lookback`` 与算子语义保证（t 日收盘可得的因子配 t+1 起的收益），
    而沙箱那条"价格列必须 shift(1)"的规则是为 LLM 写的自由代码准备的。
    两套口径不能互相套用，否则要么误报要么把因子整体滞后一天。
    """
    import re

    import numpy as np  # noqa: F401 - 供 exec 的代码使用
    import pandas as pd

    code = expr_to_code(node)
    for f in node.fields():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(f)):
            raise ValueError(f"字段名不合法，拒绝生成代码: {f!r}")
    g: Dict[str, Any] = {"np": np, "pd": pd}
    exec(compile(code, "<bridge>", "exec"), g)      # noqa: S102 - 模板自生成
    out = g["alpha_factor"](df)
    if isinstance(out, pd.DataFrame):
        return pd.Series(out["factor"].to_numpy(),
                         index=pd.MultiIndex.from_arrays(
                             [out["date"].to_numpy(), out["symbol"].to_numpy()]),
                         name="factor")
    return out


def seed_from_code(code: str, known_fields: Optional[Any] = None) -> Optional[str]:
    """LLM 因子代码 → 可作网格搜索种子的 DSL 文本；翻不了返回 ``None``。

    这是双向桥的正向落点：LLM 提出的因子被翻译成符号表达式后，网格搜索可以
    在它周围做局部精修（换窗口、换算子、加中性化），而不是把它当成黑盒。
    """
    tr = translate(code, known_fields)
    return tr.expression if tr.ok else None


# --------------------------------------------------------------------------
# Python -> DSL
# --------------------------------------------------------------------------
_NP_UNARY: Dict[str, str] = {
    "log": "log", "log1p": "log1p", "sqrt": "sqrt", "abs": "abs",
    "sign": "sign", "tanh": "tanh",
}

_ROLLING_MAP: Dict[str, str] = {
    "mean": "ts_mean", "std": "ts_std", "sum": "ts_sum", "min": "ts_min",
    "max": "ts_max", "median": "ts_median", "skew": "ts_skew",
    "count": "ts_count",
}

_ROLLING_APPLY: Dict[str, str] = {
    "_slope": "ts_slope", "slope": "ts_slope", "_ts_slope": "ts_slope",
    "_rank": "ts_rank", "_prod": "ts_product", "_ts_prod": "ts_product",
    "_decay": "ts_decay_linear", "_ts_decay": "ts_decay_linear",
    "_rsq": "ts_rsquare", "_ts_rsq": "ts_rsquare",
    "_resi": "ts_resi", "_ts_resi": "ts_resi",
    "_ir": "ts_ir", "_ts_ir": "ts_ir",
    "_argmax": "ts_argmax", "_argmin": "ts_argmin",
}


def _is_rolling(call: ast.AST) -> bool:
    return (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and call.func.attr == "rolling")


def _is_groupby(call: ast.AST) -> bool:
    return (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and call.func.attr == "groupby")


def _str_const(e: ast.AST) -> Optional[str]:
    if isinstance(e, ast.Constant) and isinstance(e.value, str):
        return e.value
    return None


@dataclass
class Translation:
    """一次 Python → DSL 翻译的结果（含失败原因，便于落盘审计）。"""

    ok: bool
    node: Optional[ex.Node] = None
    expression: str = ""
    reason: str = ""
    unmatched: List[str] = dc_field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "expression": self.expression,
                "reason": self.reason, "unmatched": list(self.unmatched)}


class _Translator:
    """只认高频可验证的模式；认不出的记一笔 ``unmatched`` 并放弃整条翻译。"""

    def __init__(self, known_fields: Optional[Any] = None) -> None:
        self.env: Dict[str, ex.Node] = {}
        self.unmatched: List[str] = []
        self.known = known_fields

    # -- 入口 --
    def run(self, tree: ast.AST) -> Optional[ex.Node]:
        fn = None
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name == "alpha_factor":
                fn = n
                break
        if fn is None:
            # 没有函数定义时，直接看模块级语句（有些 LLM 直接写脚本）
            stmts = getattr(tree, "body", [])
        else:
            stmts = fn.body
        for st in stmts:
            self._stmt(st)
        return self.env.get("factor") or self._last()

    def _last(self) -> Optional[ex.Node]:
        return next(reversed(self.env.values())) if self.env else None

    # -- 语句 --
    def _stmt(self, st: ast.AST) -> None:
        if isinstance(st, ast.Assign) and len(st.targets) == 1:
            tgt = st.targets[0]
            node = self._expr(st.value)
            key = self._target_key(tgt)
            if key is not None and node is not None:
                self.env[key] = node
            elif node is None:
                self.unmatched.append(ast.dump(st.value)[:60])
        elif isinstance(st, ast.Return):
            return          # 返回 df[['date','symbol','factor']] 不携带新信息
        elif isinstance(st, (ast.Import, ast.ImportFrom, ast.Expr)):
            return
        else:
            self.unmatched.append(type(st).__name__)

    @staticmethod
    def _target_key(tgt: ast.AST) -> Optional[str]:
        if isinstance(tgt, ast.Name):
            return tgt.id
        if (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name)
                and isinstance(tgt.slice, ast.Constant)
                and isinstance(tgt.slice.value, str)):
            return tgt.slice.value          # df['ret'] = ...
        return None

    # -- 表达式 --
    def _expr(self, e: ast.AST, inner: Optional[str] = None) -> Optional[ex.Node]:
        if isinstance(e, ast.Name):
            if inner is not None and e.id == inner:
                return _INNER
            return self.env.get(e.id)
        if isinstance(e, ast.Constant):
            return ex.Const(float(e.value)) if isinstance(e.value, (int, float)) else None
        if isinstance(e, ast.Subscript):        # df['close']
            col = self._df_col(e)
            if col is None:
                return None
            return self.env.get(col) or ex.Field(col)
        if isinstance(e, ast.Attribute):        # df.close
            if isinstance(e.value, ast.Name) and e.value.id == "df":
                return self.env.get(e.attr) or ex.Field(e.attr)
            return None
        if isinstance(e, ast.Call):
            return self._call(e, inner)
        if isinstance(e, ast.UnaryOp):
            v = self._expr(e.operand, inner)
            if v is None:
                return None
            if isinstance(e.op, ast.USub):
                return ex.Call("neg", (v,))
            if isinstance(e.op, ast.UAdd):
                return v
            return None
        if isinstance(e, ast.BinOp):
            a = self._expr(e.left, inner)
            b = self._expr(e.right, inner)
            if a is None or b is None:
                return None
            name = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul",
                    ast.Div: "div", ast.Pow: "pow2"}.get(type(e.op))
            if name is None or a is _INNER or b is _INNER:
                return None
            return ex.Call(name, (a, b))
        self.unmatched.append(type(e).__name__)
        return None

    @staticmethod
    def _df_col(e: ast.Subscript) -> Optional[str]:
        if isinstance(e.value, ast.Name) and e.value.id == "df" \
                and isinstance(e.slice, ast.Constant) \
                and isinstance(e.slice.value, str):
            return e.slice.value
        return None

    # -- 调用 --
    def _call(self, e: ast.Call, inner: Optional[str]) -> Optional[ex.Node]:
        f = e.func
        if isinstance(f, ast.Attribute):
            attr = f.attr
            owner = f.value
            if attr == "transform":                 # groupby(..)[x].transform(λ)
                return self._transform(owner, e, inner)
            if attr == "pct_change":
                base = self._base_of(owner, inner)
                return ex.Call("ts_pct", (base,), 1) if base is not None else None
            if attr == "shift" and e.args:
                base = self._base_of(owner, inner)
                n = _int_const(e.args[0])
                if base is None or n is None:
                    return None
                return ex.Call("ts_delay", (base,), n)
            if attr == "apply" and isinstance(owner, ast.Call) \
                    and _is_rolling(owner):
                return self._rolling(owner, inner, apply_node=e)
            if attr in _ROLLING_MAP and isinstance(owner, ast.Call) \
                    and _is_rolling(owner):
                return self._rolling(owner, inner, method=attr)
            if attr == "rolling":                   # 裸 rolling(w) 没有终点方法
                return None
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
                and f.value.id == "np":
            op = _NP_UNARY.get(f.attr)
            if op and len(e.args) == 1:
                v = self._expr(e.args[0], inner)
                return ex.Call(op, (v,)) if v is not None else None
        return None

    def _transform(self, owner: ast.AST, outer: ast.Call,
                   inner: Optional[str]) -> Optional[ex.Node]:
        """``groupby(...)[x].transform(lambda x: <body>)``。"""
        base = self._base_of(owner, inner)
        lam = outer.args[0] if outer.args else None
        if base is None or not isinstance(lam, ast.Lambda) \
                or len(lam.args.args) != 1:
            return None
        body = self._expr(lam.body, lam.args.args[0].arg)
        if body is None:
            return None
        return _sub_inner(body, base)

    def _rolling(self, e: ast.Call, inner: Optional[str],
                 method: Optional[str] = None,
                 apply_node: Optional[ast.Call] = None) -> Optional[ex.Node]:
        """``x.rolling(w).method()`` / ``x.rolling(w).apply(fn)``。"""
        owner = e.func.value if isinstance(e.func, ast.Attribute) else None
        base = self._base_of(owner, inner)
        w = _int_const(e.args[0]) if e.args else None
        if base is None or w is None:
            return None
        if apply_node is not None:
            fn = apply_node.args[0] if apply_node.args else None
            nm = fn.id if isinstance(fn, ast.Name) else None
            op = _ROLLING_APPLY.get(nm or "")
            return ex.Call(op, (base,), w) if op else None
        op = _ROLLING_MAP.get(method or "")
        return ex.Call(op, (base,), w) if op else None

    def _base_of(self, owner: ast.AST, inner: Optional[str]) -> Optional[ex.Node]:
        """从 ``df.groupby('symbol')['close']`` / ``df['close']`` / ``df.close``
        / ``x``（λ 内层变量）/ ``x.rolling(w).m()`` 里取底数。
        """
        if isinstance(owner, ast.Name):
            if inner is not None and owner.id == inner:
                return _INNER
            return self.env.get(owner.id)
        if isinstance(owner, ast.Subscript):
            col = _str_const(owner.slice)
            if col is None:
                return None
            if isinstance(owner.value, ast.Name) and owner.value.id == "df":
                return self.env.get(col) or ex.Field(col)
            if _is_groupby(owner.value):
                return self.env.get(col) or ex.Field(col)
            return None
        if isinstance(owner, ast.Attribute):
            if isinstance(owner.value, ast.Name) and owner.value.id == "df":
                return self.env.get(owner.attr) or ex.Field(owner.attr)
            if _is_groupby(owner.value):
                return self.env.get(owner.attr) or ex.Field(owner.attr)
            return None
        if isinstance(owner, ast.Call):
            if isinstance(owner.func, ast.Attribute):
                if owner.func.attr == "rolling":
                    return self._rolling(owner, inner)
                if owner.func.attr == "groupby":
                    return self._base_of(owner.func.value, inner)
            return None
        return None


class _InnerMarker:
    """占位：``transform(lambda x: ...)`` 里的内层变量，收尾时替换成底数。"""

    kind = "inner"


_INNER = _InnerMarker()


def _sub_inner(node: ex.Node, base: ex.Node) -> ex.Node:
    if node is _INNER:
        return base
    if isinstance(node, ex.Call):
        return ex.Call(node.name, tuple(_sub_inner(a, base) for a in node.args),
                       node.window)
    if isinstance(node, ex.Neutral):
        return ex.Neutral(_sub_inner(node.inner, base), node.controls,
                          node.min_stocks)
    return node


def _int_const(e: ast.AST) -> Optional[int]:
    if isinstance(e, ast.Constant) and isinstance(e.value, int):
        return int(e.value)
    if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub) \
            and isinstance(e.operand, ast.Constant):
        return -int(e.operand.value)
    return None


def code_to_expr(code: str,
                 known_fields: Optional[Any] = None) -> Optional[ex.Node]:
    """把 LLM 写的 ``alpha_factor(df)`` 翻译成 DSL 表达式树；翻不了返回 ``None``。"""
    if not code or not code.strip():
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    tr = _Translator(known_fields)
    node = tr.run(tree)
    if node is None or node is _INNER:
        return None
    flat = _sub_inner(node, ex.Field("__base__"))
    if "__base__" in flat.fields():
        return None                 # 还有没替换掉的内层变量 → 语义漂移，放弃
    return flat


def translate(code: str, known_fields: Optional[Any] = None) -> Translation:
    """带诊断的翻译：成功给表达式文本，失败给原因与未识别结构。"""
    if not code or not code.strip():
        return Translation(False, reason="代码为空")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return Translation(False, reason=f"语法错误：{e}")
    tr = _Translator(known_fields)
    node = tr.run(tree)
    if node is None or node is _INNER:
        return Translation(False, reason="没有可识别的因子赋值链",
                           unmatched=tr.unmatched)
    flat = _sub_inner(node, ex.Field("__base__"))
    if "__base__" in flat.fields():
        return Translation(False, reason="存在未解析的内层变量（语义可能漂移）",
                           unmatched=tr.unmatched)
    try:
        ex.validate(flat)
    except Exception as e:          # noqa: BLE001 - 校验失败也按"翻不了"处理
        return Translation(False, reason=f"翻译结果未通过类型校验：{e}",
                           unmatched=tr.unmatched)
    return Translation(True, node=flat, expression=flat.render(),
                       reason="翻译成功", unmatched=tr.unmatched)
