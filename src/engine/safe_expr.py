"""安全表达式求值（src/engine/safe_expr.py）。

为「用户/LLM 提供的因子表达式」提供受限求值：以 AST 白名单替代裸 eval，
只放行算术/比较/下标/方法调用等计算型节点，拒绝 import、赋值、推导式、
以及任何以 `_` 开头的属性（__class__ / __globals__ / __subclasses__ 等逃逸链）。

典型用途：
    safe_eval_expr("df['close'] / df['close'].shift(1) - 1", {"df": df, "np": np})

与 factor_builder 的沙箱（进程/白名单双隔离）互补：那里管「整段代码」，
这里管「单行表达式」，两者共用同一套「拒绝 dunder 属性」的核心约束。
"""

from __future__ import annotations

import ast
from typing import Any, Dict, Mapping, Optional

# 允许出现的 AST 节点类型（计算型白名单，语句/导入/推导式一律不在其中）
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.Invert,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
    ast.In, ast.NotIn,
    ast.Load, ast.Name, ast.Constant, ast.Call, ast.Attribute, ast.Subscript,
    ast.Slice, ast.Tuple, ast.List, ast.Dict, ast.Set, ast.keyword,
)

# 允许作为裸名调用的内置函数（不含 eval/exec/open/__import__/compile）
_ALLOWED_BUILTINS: Dict[str, Any] = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len, "round": round,
    "pow": pow, "divmod": divmod, "float": float, "int": int, "str": str,
    "bool": bool, "sorted": sorted, "list": list, "tuple": tuple, "dict": dict,
    "set": set, "zip": zip, "map": map, "filter": filter, "enumerate": enumerate,
    "any": any, "all": all, "isinstance": isinstance, "print": print,
    "True": True, "False": False, "None": None,
}


class UnsafeExpressionError(ValueError):
    """表达式不安全（含被禁止的语法或逃逸尝试）。"""


def _check(node: ast.AST, names: Mapping[str, Any]) -> None:
    """递归校验 AST：不在白名单节点内、或触碰 dunder 属性即拒绝。"""
    if not isinstance(node, _ALLOWED_NODES):
        raise UnsafeExpressionError(
            f"表达式含不允许的语法：{type(node).__name__}（仅支持算术/比较/下标/方法调用）"
        )

    if isinstance(node, ast.Attribute):
        # 核心防线：任何以 _ 开头的属性都拒绝（切断 __class__.__bases__ 逃逸链）
        if node.attr.startswith("_"):
            raise UnsafeExpressionError(f"禁止访问私有/魔术属性：.{node.attr}")
    elif isinstance(node, ast.Name):
        if node.id not in names and node.id not in _ALLOWED_BUILTINS:
            raise UnsafeExpressionError(
                f"未定义的名称：{node.id}（可用变量：{sorted(names)}）"
            )
        if node.id.startswith("_"):
            raise UnsafeExpressionError(f"禁止访问以下划线开头的名称：{node.id}")
    elif isinstance(node, ast.Call):
        # 禁止把 dunder 当函数调用，如 __import__('os')
        if isinstance(node.func, ast.Name) and node.func.id.startswith("__"):
            raise UnsafeExpressionError(f"禁止调用：{node.func.id}")

    for child in ast.iter_child_nodes(node):
        _check(child, names)


def safe_eval_expr(expr: str, names: Optional[Mapping[str, Any]] = None,
                   max_expr_chars: int = 2000) -> Any:
    """在白名单约束下求值单个表达式，返回求值结果。

    Args:
        expr: 表达式字符串（不含语句/赋值/导入）。
        names: 注入的变量环境，如 {"df": df, "np": np}。
        max_expr_chars: 表达式长度上限，防止超长 payload。

    Raises:
        UnsafeExpressionError: 语法不在白名单或命中逃逸模式。
        ValueError: 表达式为空、超长或语法错误。
    """
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("表达式为空。")
    src = expr.strip()
    if len(src) > max_expr_chars:
        raise ValueError(f"表达式过长（{len(src)} > {max_expr_chars} 字符）。")
    if "\n" in src or ";" in src or "=" in src.replace("==", "").replace("!=", "") \
            .replace("<=", "").replace(">=", ""):
        raise UnsafeExpressionError("表达式必须为单行，且不含语句/赋值。")

    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"表达式语法错误：{e}") from e

    env: Dict[str, Any] = dict(_ALLOWED_BUILTINS)
    env.update(names or {})
    _check(tree, env)

    try:
        # 前置 _check() 已完成 AST 白名单与 dunder 拦截，此处为受限求值
        return eval(compile(tree, "<safe_expr>", "eval"), {"__builtins__": {}}, env)  # noqa: S307
    except UnsafeExpressionError:
        raise
    except Exception as e:
        raise ValueError(f"表达式求值失败：{type(e).__name__}: {e}") from e


def safe_factor_expr(df: "Any", expr: str, extra_names: Optional[Mapping[str, Any]] = None) -> Any:
    """把表达式求值为与 df 等长的因子值（供非结构化数据列映射使用）。

    自动注入 df / np / pd，并对结果做长度校验与数值化，避免返回标量或异形对象
    静默污染下游回测。
    """
    import numpy as np  # 局部导入：本模块被沙箱周边引用，避免顶层加重依赖
    import pandas as pd

    names: Dict[str, Any] = {"df": df, "np": np, "pd": pd}
    names.update(extra_names or {})
    value = safe_eval_expr(expr, names)

    if isinstance(value, (pd.Series, pd.DataFrame)):
        if len(value) != len(df):
            raise ValueError(
                f"表达式结果长度 {len(value)} 与数据行数 {len(df)} 不一致。"
            )
        if isinstance(value, pd.DataFrame):
            if value.shape[1] != 1:
                raise ValueError("表达式须返回单列结果。")
            value = value.iloc[:, 0]
        return pd.to_numeric(value, errors="coerce")

    if isinstance(value, (int, float, np.number)):
        return pd.Series([float(value)] * len(df), index=df.index)

    raise ValueError(
        f"表达式须返回标量或与数据等长的序列，实际得到 {type(value).__name__}。"
    )
