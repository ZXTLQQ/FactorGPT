"""跨任务因子基因库（src/mining/genome.py）。

符号搜索这一侧此前是**无记忆**的：``GridMiner._seen`` 只管单次搜索内去重，
进程一结束就忘干净，下一次挖掘又从 registry 的原始字段重新长一遍——同样的
数据、同样的预算，每次都付一遍"从头发现动量/波动/流动性"的学费。

基因库把"哪些子树曾经好用"持久化下来，下次挖掘作为种子注入，于是搜索能站在
上次的肩膀上。这是"越挖越快"唯一能拿数字证明的形式：同一批数据连续挖第二轮，
达到同等 top-k 质量所需的预算应当下降。

刻意保持**默认关闭**：注入种子会改变候选集，进而破坏"同种子同结果"的可复现
约定（``tests/test_mining.py`` 里那条可复现性断言正盯着这个）。只有调用方
显式传入 :class:`GenomeBank` 才生效。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict as dc_asdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from . import expr as ex
from . import ops

__all__ = ["DEFAULT_PATH", "Gene", "GenomeBank", "fingerprint"]

DEFAULT_PATH = os.path.join("data", "factor_genome.json")


def fingerprint(panel: Any) -> str:
    """面板数据指纹：同一批数据（标的数、时间跨度、字段集）才算"同一批"。

    指纹不同不代表基因没用——跨数据集的通用骨架同样值得注入，只是优先级低。
    """
    try:
        dates = panel.dates
        fields = sorted(str(f) for f in panel.fields)
        raw = f"{len(panel.symbols)}|{str(dates[0])[:10]}|{str(dates[-1])[:10]}" \
              f"|{len(dates)}|{','.join(fields)}"
    except Exception:              # pragma: no cover - 面板接口异常时退化为匿名
        raw = "unknown"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


@dataclass
class Gene:
    """一条"曾经好用"的因子基因。"""

    expression: str
    key: str
    ic_mean: float = 0.0
    icir: float = 0.0
    coverage: float = 0.0
    consistency: float = 1.0
    incr_ratio: float = 1.0
    family: str = ""
    n_nodes: int = 1
    panel: str = ""
    hits: int = 1                  # 被重复挖到的次数（越高说明骨架越稳）
    updated: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dc_asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Gene":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)


def _family_of(node: ex.Node) -> str:
    if isinstance(node, ex.Neutral):
        return "neutral"
    if isinstance(node, ex.Call):
        try:
            return ops.get_op(node.name).family
        except KeyError:              # pragma: no cover - 未注册算子
            return "elem"
    return "field"


class GenomeBank:
    """因子基因库：读 / 写 / 取种子。落盘为一个 JSON 文件。"""

    def __init__(self, path: str = DEFAULT_PATH, capacity: int = 400) -> None:
        self.path = path
        self.capacity = int(capacity)
        self.genes: Dict[str, Gene] = {}
        self.load()

    # -- 持久化 --
    def load(self) -> "GenomeBank":
        if not os.path.exists(self.path):
            return self
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            return self                     # 基因库损坏不该拖垮挖掘
        for d in (raw.get("genes") or []):
            try:
                g = Gene.from_dict(d)
            except (TypeError, ValueError):
                continue
            self.genes[g.key] = g
        return self

    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".",
                    exist_ok=True)
        payload = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "n": len(self.genes),
                   "genes": [g.to_dict() for g in self.genes.values()]}
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    # -- 写入 --
    def add_search(self, res: Any, panel: Any, top_n: int = 20) -> int:
        """把一次搜索的优质候选并入基因库，返回新增/更新的条数。"""
        fp = fingerprint(panel)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        n = 0
        for c in (getattr(res, "candidates", None) or [])[:max(0, top_n)]:
            try:
                key = c.node.key()
                g = Gene(expression=c.expression, key=key,
                         ic_mean=float(c.ic_mean), icir=float(c.icir),
                         coverage=float(c.coverage),
                         consistency=float(getattr(c, "consistency", 1.0)),
                         incr_ratio=float(getattr(c, "incr_ratio", 1.0)),
                         family=_family_of(c.node), n_nodes=int(c.n_nodes),
                         panel=fp, updated=stamp)
            except Exception:               # pragma: no cover - 单个坏基因跳过
                continue
            old = self.genes.get(key)
            if old is not None:
                old.hits = int(old.hits) + 1
                old.ic_mean = float(old.ic_mean) * 0.5 + g.ic_mean * 0.5
                old.updated = stamp
                if fp == old.panel:
                    old.panel = fp
            else:
                self.genes[key] = g
            n += 1
        if len(self.genes) > self.capacity:
            self._trim()
        self.save()
        return n

    def add_expression(self, expression: str, *, ic_mean: float = 0.0,
                       panel: Any = None, **kw: Any) -> Optional[Gene]:
        """手动登记一条因子（如 LLM 翻译出来的种子）。"""
        try:
            node = ex.parse(expression)
        except Exception:
            return None
        g = Gene(expression=node.render(), key=node.key(), ic_mean=float(ic_mean),
                 family=_family_of(node), n_nodes=int(node.size()),
                 panel=fingerprint(panel) if panel is not None else "",
                 updated=time.strftime("%Y-%m-%d %H:%M:%S"), **kw)
        self.genes[g.key] = g
        self.save()
        return g

    def _trim(self) -> None:
        """超容量时按 |IC| × 命中次数淘汰（保留最"值钱"的骨架）。"""
        ranked = sorted(self.genes.values(),
                        key=lambda g: abs(g.ic_mean) * (1 + 0.1 * g.hits),
                        reverse=True)
        self.genes = {g.key: g for g in ranked[:self.capacity]}

    # -- 读取 --
    def seeds(self, panel: Optional[Any] = None, n: int = 8,
              min_ic: float = 0.0) -> List[str]:
        """取 warm-start 种子：同数据集的优先，其次跨数据集的通用骨架。"""
        fp = fingerprint(panel) if panel is not None else ""
        pool = [g for g in self.genes.values()
                if abs(g.ic_mean) >= min_ic and g.expression]
        same = sorted((g for g in pool if g.panel == fp),
                      key=lambda g: -abs(g.ic_mean))
        rest = sorted((g for g in pool if g.panel != fp),
                      key=lambda g: -abs(g.ic_mean) * (1 + 0.1 * g.hits))
        ranked = same + rest
        out: List[str] = []
        for g in ranked:
            if g.expression not in out:
                out.append(g.expression)
            if len(out) >= max(0, n):
                break
        return out

    def stats(self) -> Dict[str, Any]:
        if not self.genes:
            return {"n": 0}
        ics = [abs(g.ic_mean) for g in self.genes.values()]
        fam: Dict[str, int] = {}
        for g in self.genes.values():
            fam[g.family] = fam.get(g.family, 0) + 1
        return {"n": len(self.genes), "mean_abs_ic": float(sum(ics) / len(ics)),
                "max_abs_ic": float(max(ics)), "families": fam,
                "panels": len({g.panel for g in self.genes.values()})}

    # -- 维护 --
    def clear(self) -> None:
        self.genes = {}
        self.save()

    def __len__(self) -> int:
        return len(self.genes)

    def expressions(self) -> Sequence[str]:
        return [g.expression for g in self.genes.values()]
