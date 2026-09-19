"""失败模式库（src/agent/failure_memory.py）。

``nodes.learn_factor`` 只把**通过校验**的因子写进学习库——Agent 于是只记得
成功、不记得失败，同一类错误（前视、SVD 不收敛、整列 NaN、超时）会一遍遍
重犯；而 ``validate_and_compute`` 里那几条修复提示是硬编码的，不会随使用演化。

这里把失败按**模式**聚类存下来，生成阶段再作为负面约束注入 prompt，于是提示
会随使用自动长出来，而不是靠人一次次手动补。

落盘 ``data/failure_memory.json``；库损坏或写不进去时静默降级为空——失败记忆
是增益，不该成为主线的故障点。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict as dc_asdict
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["DEFAULT_PATH", "PATTERNS", "FailureCase", "FailureMemory", "classify"]

DEFAULT_PATH = os.path.join("data", "failure_memory.json")

# (关键字, 模式名, 修复提示)
PATTERNS: List[Tuple[Tuple[str, ...], str, str]] = [
    (("前视", "未来信息", "lookahead", "shift(-"), "lookahead",
     "因子值只能用到 t 日及之前的信息：对 close/open/high/low/volume 运算后必须 "
     "shift(1)，禁止 shift(-n)，禁止用全样本均值/标准差做标准化。"),
    (("SVD", "LinAlg", "did not converge", "singular", "lstsq", "矩阵"), "numerics",
     "回归/中性化前先 dropna 并过滤 inf；std 可能为 0 时先 replace(0, np.nan)；"
     "自变量近常数直接返回 0。优先用 scipy.stats.linregress 或对 rank 做相减式中性化。"),
    (("NaN", "inf", "divide", "zero", "空值", "整列"), "nan",
     "rolling/ewm 之后要处理首个窗口的 NaN；避免 0 除；截面运算前确认当日有效"
     "样本数 >= 5，否则返回 NaN 而不是硬算。"),
    (("timeout", "超时", "Timeout"), "timeout",
     "避免逐行 apply 与 groupby.apply 里再套循环：改用 transform / 向量化运算，"
     "窗口统计优先 rolling(...).mean() 这类内置方法。"),
    (("alpha_factor",), "no_entry",
     "必须定义 alpha_factor(df) 并返回 df[['date','symbol','factor']]，"
     "不要返回单列或修改入参以外的列。"),
    (("常数", "constant", "标准差为 0", "全 0", "degenerate"), "constant",
     "因子退化成常数：通常是窗口大于可用长度、把同一列相减了两次，或截面样本太少。"),
    (("column", "KeyError", "列"), "missing_column",
     "只用输入 df 里真实存在的列；需要中间变量就用 df['xxx'] = ... 先定义再引用。"),
]


def classify(error: str) -> Tuple[str, str]:
    """把一条报错归到某个已知失败模式，返回 (模式名, 修复提示)。"""
    text = str(error or "")
    for keys, kind, hint in PATTERNS:
        if any(k.lower() in text.lower() for k in keys):
            return kind, hint
    return "other", "先在脑子里跑一遍边界：首窗口 NaN、单日样本不足、0 除、窗口过长。"


@dataclass
class FailureCase:
    """一类失败：累计次数、最近一次报错、几条触发代码样本。"""

    kind: str
    hint: str
    count: int = 1
    last: str = ""
    updated: str = ""
    samples: List[str] = dc_field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dc_asdict(self)


class FailureMemory:
    """失败模式库：记录、排序、渲染成 prompt 片段。"""

    def __init__(self, path: str = DEFAULT_PATH, capacity: int = 50) -> None:
        self.path = path
        self.capacity = int(capacity)
        self.cases: Dict[str, FailureCase] = {}
        self.load()

    # -- 持久化 --
    def load(self) -> "FailureMemory":
        if not os.path.exists(self.path):
            return self
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            return self
        for d in (raw.get("cases") or []):
            try:
                c = FailureCase(**{k: v for k, v in d.items()
                                   if k in FailureCase.__dataclass_fields__})
            except (TypeError, ValueError):
                continue
            self.cases[c.kind] = c
        return self

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".",
                        exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({"updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "n": len(self.cases),
                           "cases": [c.to_dict() for c in self.cases.values()]},
                          fh, ensure_ascii=False, indent=2)
        except OSError:                     # pragma: no cover - 写不进就算了
            pass

    # -- 记录 --
    def record(self, error: str, code: str = "") -> Optional[FailureCase]:
        kind, hint = classify(error)
        c = self.cases.get(kind)
        if c is None:
            c = FailureCase(kind=kind, hint=hint, count=0, samples=[])
            self.cases[kind] = c
        c.count = int(c.count) + 1
        c.last = str(error)[:300]
        c.updated = time.strftime("%Y-%m-%d %H:%M:%S")
        c.hint = hint
        if code:
            c.samples.append(str(code)[-500:])
            c.samples = c.samples[-3:]
        if len(self.cases) > self.capacity:
            self._trim()
        self.save()
        return c

    def _trim(self) -> None:
        keep = sorted(self.cases.values(), key=lambda c: -int(c.count))[:self.capacity]
        self.cases = {c.kind: c for c in keep}

    # -- 读取 --
    def top(self, n: int = 3, min_count: int = 1) -> List[FailureCase]:
        ranked = sorted((c for c in self.cases.values()
                         if int(c.count) >= min_count),
                        key=lambda c: -int(c.count))
        return ranked[:max(0, n)]

    def prompt_block(self, n: int = 3) -> str:
        """渲染成"负面约束"提示块；没有历史失败时返回空串（不污染 prompt）。"""
        tops = self.top(n=n)
        if not tops:
            return ""
        lines = ["[历史失败约束] 以下错误此前在这台机器上反复出现，务必主动避开："]
        for i, c in enumerate(tops, 1):
            lines.append(f"{i}. {c.kind}（累计 {c.count} 次）：{c.hint}")
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        return {"n": len(self.cases),
                "total": int(sum(int(c.count) for c in self.cases.values())),
                "top": [(c.kind, int(c.count)) for c in self.top(5, 0)]}

    def clear(self) -> None:
        self.cases = {}
        self.save()

    def __len__(self) -> int:
        return len(self.cases)
