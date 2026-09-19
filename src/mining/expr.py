"""强类型表达式树：量价与基本面共处同一因子空间（中信建投《量价 X 基本面
因子挖掘统一框架》落地）。

研报的核心主张是：**先统一类型，再统一搜索**。若量价字段与财务字段各自一套
算子、各自一套命名，跨域组合只能在人脑里做，无法进网格搜索。本模块给出这一层：

- ``parse`` / ``render``：因子表达式的文本 DSL（``zscore_cs(ts_pct(close, 20))``），
  与 Python 语法一致，用 ``ast`` 解析，天然支持四则运算与嵌套。
- **静态类型检查**：每个节点都带 ``(dimension, semantics, role)`` 三元类型。
  规则源自量纲与语义，目的只有一个 —— 让"无意义组合"在**求值之前**被拒绝，
  而不是等到 IC 出来才发现算的是垃圾：

  1. 加减（``add``/``sub``）要求**同维度**，或双方都属无量纲族
     （``ratio``/``score``/``growth``/``count``）。价格加金额这类错误被挡在这一层。
  2. ``flag`` 类字段不得进入加减与对数/开方（它只能当掩码，用 ``mul`` 或
     ``ts_sum`` 使用）。
  3. **跨域（量价 × 基本面）组合只走显式通道**：乘、除、``spread``、
     ``geom_mean``/``harm_mean``、二元时序（``ts_corr``/``ts_beta`` 等）。
     这是研报里"混合因子"的正规写法，避免把两个量纲不同的东西硬加在一起。
  4. **中性化算子只允许出现在最外层**（研报明确约束）。``neutral(f, size, beta)``
     一旦出现在子节点里即报错 —— 否则"对规模中性化之后再做时序平滑"会
     把中性化悄悄破坏掉，是最隐蔽的错。

- ``Evaluator``：带子表达式缓存求值。共享子式只算一次，这是网格搜索能把
  上万条表达式跑完的前提。
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from . import ops
from .ops import OpSpec
from .panel import (
    DIM_AMOUNT,
    DIM_COUNT,
    DIM_FLAG,
    DIM_GROWTH,
    DIM_PRICE,
    DIM_RATIO,
    DIM_SCORE,
    DIM_VOLUME,
    FieldRegistry,
    PanelData,
    default_registry,
    residualize_cs,
)

__all__ = [
    "COMMUTATIVE",
    "DIMENSIONLESS",
    "NEUTRAL_OP",
    "Call",
    "Const",
    "Evaluator",
    "ExprError",
    "ExprParseError",
    "ExprTypeError",
    "Field",
    "Neutral",
    "Node",
    "TypeInfo",
    "describe",
    "evaluate",
    "infer_type",
    "lookback",
    "parse",
    "render",
    "validate",
]


class ExprError(Exception):
    """表达式相关异常基类。"""


class ExprParseError(ExprError):
    """DSL 文本无法解析。"""


class ExprTypeError(ExprError):
    """类型校验不通过（维度/语义/角色冲突，或中性化位置非法）。"""


# 无量纲族：彼此可加减（都已是可比的无量纲量）
DIMENSIONLESS: Set[str] = {DIM_RATIO, DIM_SCORE, DIM_GROWTH, DIM_COUNT}

# 可交换算子：规范化 key 时子节点排序，从而消除 add(a,b) 与 add(b,a) 的重复
COMMUTATIVE: Set[str] = {"add", "mul", "max2", "min2", "geom_mean", "harm_mean"}

NEUTRAL_OP = "neutral"
ROLE_MIXED = "MIX"

_DIM_LABELS = {
    DIM_PRICE: "价格", DIM_VOLUME: "成交量", DIM_AMOUNT: "成交额",
    DIM_RATIO: "比率", DIM_SCORE: "得分", DIM_COUNT: "计数",
    DIM_FLAG: "标志", DIM_GROWTH: "增速",
}
_DIM_CN = lambda d: _DIM_LABELS.get(d, d)

_BINOP = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}
_MAX_DEPTH = 12


# --------------------------------------------------------------------------
# 类型
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TypeInfo:
    """表达式节点的静态类型：量纲 + 语义 + 角色（量价/基本面/另类/暴露）。"""

    dimension: str
    semantics: str
    role: str

    def is_dimensionless(self) -> bool:
        return self.dimension in DIMENSIONLESS

    def to_dict(self) -> Dict[str, str]:
        return {"dimension": self.dimension, "semantics": self.semantics,
                "role": self.role}

    def __str__(self) -> str:  # pragma: no cover - 展示用
        return f"{_DIM_CN(self.dimension)}/{self.semantics}/{self.role}"


def _merge_role(roles: Iterable[str]) -> str:
    uniq = {r for r in roles if r}
    if not uniq:
        return ""
    if len(uniq) == 1:
        return next(iter(uniq))
    return "".join(sorted(uniq))  # 例如量价+基本面 → "FM"


def _merge_sem(sems: Iterable[str]) -> str:
    uniq = {s for s in sems if s}
    if not uniq:
        return ""
    return next(iter(uniq)) if len(uniq) == 1 else "mix"


# --------------------------------------------------------------------------
# 节点
# --------------------------------------------------------------------------
class Node:
    """表达式节点基类（不可变语义：所有字段构造后不再修改）。"""

    kind = "node"

    # -- 求值 --
    def evaluate(self, panel: PanelData,
                 registry: Optional[FieldRegistry] = None,
                 cache: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        raise NotImplementedError

    # -- 类型 --
    def type_of(self, registry: FieldRegistry) -> TypeInfo:
        raise NotImplementedError

    # -- 展示 / 序列化 --
    def render(self) -> str:
        raise NotImplementedError

    def key(self) -> str:
        return self.render()

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    # -- 结构信息 --
    def walk(self) -> Iterable["Node"]:
        yield self

    def depth(self) -> int:
        return 1

    def fields(self) -> Set[str]:
        out: Set[str] = set()
        for n in self.walk():
            if isinstance(n, Field):
                out.add(n.name)
        return out

    def ops_used(self) -> List[str]:
        return sorted({n.name for n in self.walk() if isinstance(n, Call)})

    def size(self) -> int:
        return sum(1 for _ in self.walk())

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.__class__.__name__} {self.render()}>"

    def __str__(self) -> str:
        return self.render()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())


@dataclass(frozen=True)
class Const(Node):
    """常数节点（窗口以外的常数使用很少，仅用于 round-trip 与模板）。"""

    value: float
    kind = "const"

    def evaluate(self, panel: PanelData, registry=None, cache=None) -> pd.DataFrame:
        return pd.DataFrame(float(self.value), index=panel.dates,
                            columns=panel.symbols)

    def type_of(self, registry: FieldRegistry) -> TypeInfo:
        return TypeInfo(DIM_SCORE, "", "")

    def render(self) -> str:
        v = float(self.value)
        return str(int(v)) if v.is_integer() else repr(v)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "const", "value": float(self.value)}


@dataclass(frozen=True)
class Field(Node):
    """字段叶子：类型完全来自字段注册表（这就是"统一"的落点）。"""

    name: str
    kind = "field"

    def evaluate(self, panel: PanelData, registry=None,
                 cache=None) -> pd.DataFrame:
        return panel.field(self.name)

    def type_of(self, registry: FieldRegistry) -> TypeInfo:
        m = registry.get(self.name)
        return TypeInfo(m.dimension, m.semantics, m.role)

    def render(self) -> str:
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "field", "name": self.name}


@dataclass(frozen=True)
class Call(Node):
    """算子调用：``Call("ts_pct", [Field("close")], 20)``。"""

    name: str
    args: Tuple[Node, ...]
    window: Optional[int] = None
    kind = "call"

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))

    # -- 求值 --
    def evaluate(self, panel: PanelData, registry=None,
                 cache=None) -> pd.DataFrame:
        vals = [a.evaluate(panel, registry, cache) for a in self.args]
        return ops.call_op(self.name, vals, self.window)

    # -- 类型 --
    def spec(self) -> OpSpec:
        return ops.get_op(self.name)

    def type_of(self, registry: FieldRegistry) -> TypeInfo:
        spec = self.spec()
        if spec.window and self.window is None:
            raise ExprTypeError(f"{self.name} 缺少窗口参数")
        if not spec.window and self.window is not None:
            raise ExprTypeError(f"{self.name} 不接受窗口参数")
        if len(self.args) != spec.arity:
            raise ExprTypeError(
                f"{self.name} 需要 {spec.arity} 个参数，收到 {len(self.args)}")
        ts = [a.type_of(registry) for a in self.args]
        return _infer_call(spec, ts)

    # -- 展示 --
    def render(self) -> str:
        parts = [a.render() for a in self.args]
        if self.window is not None:
            parts.append(str(int(self.window)))
        return f"{self.name}({', '.join(parts)})"

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "call", "name": self.name,
                "window": self.window, "args": [a.to_dict() for a in self.args]}

    def walk(self) -> Iterable[Node]:
        yield self
        for a in self.args:
            yield from a.walk()

    def depth(self) -> int:
        return 1 + max((a.depth() for a in self.args), default=0)

    def key(self) -> str:
        sub = [a.key() for a in self.args]
        if self.name in COMMUTATIVE:
            sub = sorted(sub)
        body = ",".join(sub)
        if self.window is not None:
            body = f"{body}|w={int(self.window)}"
        return f"{self.name}({body})"

    @property
    def arity(self) -> int:
        return len(self.args)


@dataclass(frozen=True)
class Neutral(Node):
    """中性化 / 正交化：``neutral(f, size, beta)``。

    统一框架要求它**只出现在最外层** —— 见模块文档第 4 条。
    """

    inner: Node
    controls: Tuple[Node, ...]
    min_stocks: int = 20
    standardize: bool = True
    kind = "neutral"

    def __post_init__(self) -> None:
        object.__setattr__(self, "controls", tuple(self.controls))
        if not self.controls:
            raise ExprTypeError("neutral 至少需要一个控制变量")

    def evaluate(self, panel: PanelData, registry=None,
                 cache=None) -> pd.DataFrame:
        y = self.inner.evaluate(panel, registry, cache)
        xs = [c.evaluate(panel, registry, cache) for c in self.controls]
        return residualize_cs(y, xs, min_stocks=self.min_stocks,
                              standardize=self.standardize)

    def type_of(self, registry: FieldRegistry) -> TypeInfo:
        it = self.inner.type_of(registry)
        for c in self.controls:
            c.type_of(registry)  # 控件本身也必须类型合法
        # 标准化后残差必为无量纲得分；语义与角色沿用被中性化的因子
        return TypeInfo(DIM_SCORE, it.semantics, _merge_role([it.role]))

    def render(self) -> str:
        return f"{NEUTRAL_OP}({', '.join([self.inner.render()] + [c.render() for c in self.controls])})"

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "neutral", "inner": self.inner.to_dict(),
                "controls": [c.to_dict() for c in self.controls],
                "min_stocks": self.min_stocks,
                "standardize": self.standardize}

    def walk(self) -> Iterable[Node]:
        yield self
        yield from self.inner.walk()
        for c in self.controls:
            yield from c.walk()

    def depth(self) -> int:
        return 1 + max([self.inner.depth()] + [c.depth() for c in self.controls])

    def key(self) -> str:
        cs = ",".join(sorted(c.key() for c in self.controls))
        return f"{NEUTRAL_OP}({self.inner.key()};{cs})"


# --------------------------------------------------------------------------
# 类型推导规则
# --------------------------------------------------------------------------
def _dim_ok_binary(a: TypeInfo, b: TypeInfo) -> bool:
    if a.dimension == b.dimension:
        return True
    return a.is_dimensionless() and b.is_dimensionless()


def _infer_call(spec: OpSpec, ts: List[TypeInfo]) -> TypeInfo:
    """按算子族推导输出类型；不合法则抛 ``ExprTypeError``。"""
    fam = spec.family
    head = ts[0]
    role = _merge_role(t.role for t in ts)

    if fam == "elem":
        if spec.name in ("log", "log1p", "sqrt") and head.dimension == DIM_FLAG:
            raise ExprTypeError(
                f"{spec.name} 不接受标志类字段（{_DIM_CN(head.dimension)}）")
        if spec.name == "inv":
            return TypeInfo(DIM_RATIO, head.semantics, head.role)
        if spec.name in ("sqrt", "square", "tanh", "clip3", "sign"):
            return TypeInfo(DIM_SCORE, head.semantics, head.role)
        return TypeInfo(head.dimension, head.semantics, head.role)

    if fam == "elem2":
        a, b = ts[0], ts[1]
        if spec.name in ("add", "sub", "max2", "min2"):
            if not _dim_ok_binary(a, b):
                raise ExprTypeError(
                    f"{spec.name} 两侧量纲不可加：{_DIM_CN(a.dimension)} vs "
                    f"{_DIM_CN(b.dimension)}；跨域/跨量纲组合请改用 "
                    f"mul/div/spread/geom_mean/harm_mean，或先做 zscore_cs 归一")
            if DIM_FLAG in (a.dimension, b.dimension):
                raise ExprTypeError(f"{spec.name} 不接受标志类字段")
            dim = a.dimension if a.dimension == b.dimension else DIM_SCORE
            return TypeInfo(dim, _merge_sem([a.semantics, b.semantics]), role)
        if spec.name == "mul":
            pair = {a.dimension, b.dimension}
            dim = DIM_AMOUNT if pair == {DIM_PRICE, DIM_VOLUME} else DIM_SCORE
            return TypeInfo(dim, _merge_sem([a.semantics, b.semantics]), role)
        if spec.name == "pow2":
            return TypeInfo(DIM_SCORE, head.semantics, role)
        # div / safe_div / spread / geom_mean / harm_mean → 显式跨域通道
        return TypeInfo(DIM_RATIO, _merge_sem(t.semantics for t in ts), role)

    if fam == "cs":
        return TypeInfo(DIM_SCORE, head.semantics, head.role)

    if fam == "cs2":
        # 二元横截面（DGTW 分组调整等）：第二参数只作分组依据，不参与量纲
        if head.is_dimensionless() and ts[1].dimension == DIM_FLAG:
            raise ExprTypeError(f"{spec.name} 的分组变量不能是标志类字段")
        return TypeInfo(DIM_SCORE, head.semantics, head.role)

    if fam == "ts":
        if spec.name in ("ts_rank", "ts_decay_linear", "ts_zscore"):
            return TypeInfo(DIM_SCORE, head.semantics, head.role)
        if spec.name in ("ts_ir", "ts_rsquare"):
            return TypeInfo(DIM_SCORE, head.semantics, head.role)
        if spec.name in ("ts_count",):
            return TypeInfo(DIM_COUNT, head.semantics, head.role)
        if spec.dim is not None:
            return TypeInfo(spec.dim, spec.sem or head.semantics, head.role)
        return TypeInfo(head.dimension, head.semantics, head.role)

    if fam == "ts2":
        sem = _merge_sem(t.semantics for t in ts)
        dim = spec.dim or (DIM_SCORE if spec.name == "ts_reg_rsq" else DIM_RATIO)
        return TypeInfo(dim, sem, role)

    raise ExprTypeError(f"未知算子族: {fam}")


# --------------------------------------------------------------------------
# 解析（复用 Python 语法，用 ast 解析）
# --------------------------------------------------------------------------
def parse(text: str) -> Node:
    """把 DSL 文本解析成表达式树。

    支持：字段名、数字、算子调用、``+ - * /``、一元负号、``neutral(f, x1, x2, ...)``。
    窗口参数写成该调用的**最后一个位置整数**（``ts_pct(close, 20)``）。
    """
    if not isinstance(text, str) or not text.strip():
        raise ExprParseError("表达式为空")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:  # pragma: no cover - 语法错误信息直传
        raise ExprParseError(f"解析失败: {exc.msg} @ {text!r}") from exc
    node = _from_ast(tree.body, 0)
    return node


def _from_ast(a: ast.AST, depth: int) -> Node:
    if depth > _MAX_DEPTH:
        raise ExprParseError(f"表达式嵌套超过 {_MAX_DEPTH} 层")
    if isinstance(a, ast.Constant):
        if isinstance(a.value, bool) or not isinstance(a.value, (int, float)):
            raise ExprParseError(f"不支持的常量: {a.value!r}")
        return Const(float(a.value))
    if isinstance(a, ast.UnaryOp):
        if isinstance(a.op, ast.USub):
            inner = _from_ast(a.operand, depth + 1)
            return Const(-inner.value) if isinstance(inner, Const) \
                else Call("neg", (inner,))
        if isinstance(a.op, ast.UAdd):
            return _from_ast(a.operand, depth + 1)
        raise ExprParseError(f"不支持的一元运算符: {type(a.op).__name__}")
    if isinstance(a, ast.BinOp):
        op = _BINOP.get(type(a.op))
        if op is None:
            raise ExprParseError(f"不支持的二元运算符: {type(a.op).__name__}")
        return Call(op, (_from_ast(a.left, depth + 1),
                         _from_ast(a.right, depth + 1)))
    if isinstance(a, ast.Name):
        return Field(a.id)
    if isinstance(a, ast.Attribute):
        return Field(a.attr)
    if isinstance(a, ast.Call):
        fname = _call_name(a.func)
        args = [_from_ast(x, depth + 1) for x in a.args]
        if a.keywords:
            raise ExprParseError(f"{fname} 不支持关键字参数（窗口请写成位置整数）")
        if fname == NEUTRAL_OP:
            if len(args) < 2:
                raise ExprParseError("neutral(因子, 控制变量...) 至少两个参数")
            return Neutral(inner=args[0], controls=tuple(args[1:]))
        return _build_call(fname, args)
    if isinstance(a, (ast.Tuple, ast.List)):
        raise ExprParseError("表达式里不支持列表/元组字面量")
    raise ExprParseError(f"不支持的语法节点: {type(a).__name__}")


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    raise ExprParseError("算子名必须是标识符")


def _build_call(name: str, args: List[Node]) -> Call:
    try:
        spec = ops.get_op(name)
    except KeyError as exc:
        raise ExprParseError(f"未知算子 {name!r}（{exc}）") from None
    window: Optional[int] = None
    if spec.window and len(args) == spec.arity + 1 \
            and isinstance(args[-1], Const) \
            and float(args[-1].value).is_integer():
        window = int(args[-1].value)
        args = args[:-1]
    if spec.window and window is None:
        raise ExprParseError(f"{name} 需要窗口参数，例如 {name}(x, 20)")
    if not spec.window and window is not None:
        raise ExprParseError(f"{name} 不需要窗口参数")
    if len(args) != spec.arity:
        raise ExprParseError(
            f"{name} 需要 {spec.arity} 个参数，收到 {len(args)}")
    if window is not None and window < 1:
        raise ExprParseError(f"{name} 的窗口必须为正整数，收到 {window}")
    return Call(name, tuple(args), window)


def render(node: Node) -> str:
    return node.render()


def from_dict(d: Dict[str, Any]) -> Node:
    t = d.get("type")
    if t == "const":
        return Const(float(d["value"]))
    if t == "field":
        return Field(str(d["name"]))
    if t == "call":
        return Call(str(d["name"]), tuple(from_dict(a) for a in d["args"]),
                    None if d.get("window") is None else int(d["window"]))
    if t == "neutral":
        return Neutral(from_dict(d["inner"]),
                       tuple(from_dict(c) for c in d["controls"]),
                       int(d.get("min_stocks", 20)),
                       bool(d.get("standardize", True)))
    raise ExprParseError(f"无法反序列化的节点类型: {t!r}")


# --------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------
def infer_type(node: Node, registry: Optional[FieldRegistry] = None) -> TypeInfo:
    reg = registry or default_registry()
    _check_neutral_position(node, top=True)
    return node.type_of(reg)


def validate(node: Node, registry: Optional[FieldRegistry] = None,
             require_fields: bool = True) -> List[str]:
    """返回错误信息列表（空列表 = 通过）。供网格搜索批量过滤而不抛异常。"""
    reg = registry or default_registry()
    errs: List[str] = []
    try:
        _check_neutral_position(node, top=True)
        for f in node.fields():
            if not reg.has(f):
                errs.append(f"未注册字段 {f!r}")
        if require_fields and not node.fields():
            errs.append("表达式不含任何字段")
        if not errs:
            node.type_of(reg)
    except ExprError as exc:
        errs.append(str(exc))
    except (KeyError, ValueError) as exc:
        errs.append(f"{type(exc).__name__}: {exc}")
    return errs


# 前置窗口比"窗口本身"更长一档的算子：delay 类需要 n 天之前的完整取值，
# 而 ts_mean 之类只要 w-1 天。这个区分直接决定覆盖率门槛该按哪个分母去算。
_EXTRA_LOOKBACK = {
    "ts_delay", "ts_delta", "ts_pct", "ts_max_diff", "ts_min_diff",
    "ts_yoy", "ts_qoq",
}
_EXTRA_LOOKBACK_WINDOW = {"ts_corr", "ts_cov", "ts_beta", "ts_reg_rsq", "ts_reg_resi"}


def lookback(node: Node) -> int:
    """表达式**必然**需要的前置交易日数：前 ``lookback`` 天一定是 NaN。

    用途是把"覆盖率"拆成两个口径：原始覆盖率（NaN 占比，含必然的预热期）
    与**扣除预热后的覆盖率**（分母只算开始产出之后的交易日）。没有这个区分，
    TTM（四个季度延迟＝约 750 个交易日）这类长回看因子在只有两三年数据的
    面板上会被覆盖率门槛直接判死——而被判死的理由其实是"数据不够长"，
    不是"因子不行"。
    """
    if isinstance(node, (Field, Const)):
        return 0
    if isinstance(node, Neutral):
        lb = lookback(node.inner)
        for c in node.controls:
            lb = max(lb, lookback(c))
        return lb
    if isinstance(node, Call):
        lb = 0
        for a in node.args:
            lb = max(lb, lookback(a))
        spec = node.spec()
        if not spec.window or node.window is None:
            return lb
        w = max(int(node.window), 1)
        if node.name in _EXTRA_LOOKBACK:
            return lb + w
        if spec.family == "ts2" or node.name in _EXTRA_LOOKBACK_WINDOW:
            return lb + w
        if spec.family == "ts":
            return lb + max(w - 1, 1)
        return lb
    return 0


def _check_neutral_position(node: Node, top: bool) -> None:
    if isinstance(node, Neutral):
        if not top:
            raise ExprTypeError(
                "中性化算子 neutral 只允许出现在表达式最外层；"
                "请写成 neutral(完整因子, 控制变量...)")
        _check_neutral_position(node.inner, top=False)
        for c in node.controls:
            _check_neutral_position(c, top=False)
        return
    if isinstance(node, Call):
        for a in node.args:
            _check_neutral_position(a, top=False)


def describe(node: Node, registry: Optional[FieldRegistry] = None) -> Dict[str, Any]:
    """结构化描述（供报告/列表展示）。"""
    reg = registry or default_registry()
    errs = validate(node, reg)
    out: Dict[str, Any] = {
        "expr": node.render(),
        "key": node.key(),
        "depth": node.depth(),
        "n_nodes": node.size(),
        "fields": sorted(node.fields()),
        "ops": node.ops_used(),
        "n_ops": len(node.ops_used()),
        "neutralized": isinstance(node, Neutral),
        "valid": not errs,
        "errors": errs,
    }
    if not errs:
        t = infer_type(node, reg)
        out["type"] = t.to_dict()
        out["dimension_cn"] = _DIM_CN(t.dimension)
    roles = sorted({reg.get(f).role for f in node.fields() if reg.has(f)})
    out["roles"] = roles
    out["cross_domain"] = len(roles) > 1
    return out


# --------------------------------------------------------------------------
# 求值（带子表达式缓存）
# --------------------------------------------------------------------------
class Evaluator:
    """表达式求值器：按规范 key 缓存子表达式，网格搜索里共享子式只算一次。"""

    def __init__(self, panel: PanelData,
                 registry: Optional[FieldRegistry] = None,
                 cache: bool = True,
                 budget_bytes: int = 256 * 1024 * 1024) -> None:
        self.panel = panel
        self.registry = registry or panel.registry or default_registry()
        self.use_cache = bool(cache)
        self.budget_bytes = int(budget_bytes)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._bytes = 0
        self.hits = 0
        self.misses = 0

    def run(self, node: Node) -> pd.DataFrame:
        return self._eval(node)

    def _remember(self, key: str, frame: pd.DataFrame) -> None:
        nbytes = int(frame.shape[0]) * int(frame.shape[1]) * 8
        if nbytes > self.budget_bytes:      # 单条就超预算，不入缓存
            return
        while self._bytes + nbytes > self.budget_bytes and self._cache:
            old, old_frame = next(iter(self._cache.items()))
            self._bytes -= int(old_frame.shape[0]) * int(old_frame.shape[1]) * 8
            del self._cache[old]
        self._cache[key] = frame
        self._bytes += nbytes

    def _eval(self, node: Node) -> pd.DataFrame:
        k = node.key() if self.use_cache else None
        if k is not None and k in self._cache:
            self.hits += 1
            return self._cache[k]
        self.misses += 1
        if isinstance(node, (Const, Field)):
            out = node.evaluate(self.panel, self.registry, None)
        elif isinstance(node, Call):
            vals = [self._eval(a) for a in node.args]
            out = ops.call_op(node.name, vals, node.window)
        elif isinstance(node, Neutral):
            y = self._eval(node.inner)
            xs = [self._eval(c) for c in node.controls]
            out = residualize_cs(y, xs, min_stocks=node.min_stocks,
                                 standardize=node.standardize)
        else:  # pragma: no cover
            raise ExprTypeError(f"未知节点: {node!r}")
        out = out.reindex(index=self.panel.dates, columns=self.panel.symbols)
        if k is not None:
            self._remember(k, out)
        return out

    def clear(self) -> None:
        self._cache.clear()
        self._bytes = 0
        self.hits = self.misses = 0

    @property
    def cached(self) -> int:
        return len(self._cache)

    def __call__(self, node: Node) -> pd.DataFrame:
        return self._eval(node)


def evaluate(node: Node, panel: PanelData,
             registry: Optional[FieldRegistry] = None) -> pd.DataFrame:
    """一次性求值（无跨调用缓存）。"""
    return Evaluator(panel, registry, cache=False).run(node)


# --------------------------------------------------------------------------
# 度量（供报告显示因子"用了几步、跨了几个域"）
# --------------------------------------------------------------------------
def complexity(node: Node) -> Dict[str, float]:
    """表达式复杂度代理指标：节点数、深度、不同算子数、字段数。

    网格搜索用它与 IC 一起构成"简约性惩罚"，避免选出又长又难解释的表达式。
    """
    return {
        "n_nodes": float(node.size()),
        "depth": float(node.depth()),
        "n_ops": float(len(node.ops_used())),
        "n_fields": float(len(node.fields())),
        "log_nodes": float(math.log1p(node.size())),
    }
