"""分层算子网格搜索（山西证券《基于算子网格搜索、Numba 加速的多维度评价体系》落地）。

研报给的是"工程纪律"而不是"一个模型"：**预算可控、逐层收敛、剪枝在前、
多维度评价在后**。本模块把它实现成一条可复现的流水线：

1. **分层**（layer）：第 1 层只做单字段变换；第 2 层做双因子组合；第 3 层起
   对已有候选再做变换与中性化。层数越深表达式越复杂，而真正有新信息往往
   在第 1~2 层就被抓到 —— 深层主要用来验证"没有更便宜的表达方式了"。
2. **预算**：``max_expr`` 限制总求值次数，``max_seconds`` 限制总时长。
   候选生成量超预算时按固定随机种子**无放回抽样**，保证同配置可复现。
3. **剪枝**（按代价从低到高依次执行）：
   - 类型校验不通过（量纲/语义/中性化位置）→ 直接丢弃，**不消耗求值预算**；
   - 求值异常、覆盖率过低、逐日退化 → 丢弃；
   - |IC| 低于门槛 → 丢弃（这是最关键的剪枝：垃圾因子如果活到下一层，
     会被继续变换成更多垃圾）；
   - 与已入选者高度相关 → 保留更"强"的那条（同分时保留更简单的）。
4. **早停**：连续 ``patience`` 层最佳筛选分没有提升即停止；预算/时间耗尽亦停。
5. **多维度评价在后**：搜索期只用 **RankIC + 覆盖率** 做廉价筛选
   （向量化 IC，见 ``evaluator.ICScreener``）；只有最终入围的 ``top_k``
   才跑完整的四维评分与风险体检。这是"上万条候选跑得完"的关键。

筛选分（搜索期）定义为 ``|IC| × min(|ICIR|, 3) × 覆盖系数 × 复杂度惩罚``：
量级与稳定性并重，且对表达式长度做惩罚 —— 研报强调"同样 IC 下越简单越可信"。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import evaluator as EV
from . import expr as ex
from . import ops
from .panel import FieldRegistry, PanelData, default_registry

__all__ = [
    "BIN_OPS",
    "CS_OPS",
    "DEFAULT_WINDOWS",
    "TS2_OPS",
    "TS_OPS",
    "UNARY_OPS",
    "Candidate",
    "GridMiner",
    "LayerStats",
    "SearchConfig",
    "SearchResult",
    "mine",
]


# --------------------------------------------------------------------------
# 默认算子集（分层搜索的"网格"）
# --------------------------------------------------------------------------
UNARY_OPS: Tuple[str, ...] = ("neg", "abs", "log", "log1p", "sqrt", "square",
                              "sign", "tanh", "inv")
CS_OPS: Tuple[str, ...] = ("rank_cs", "zscore_cs", "demean_cs", "winsorize_cs",
                           "scale_cs")
TS_OPS: Tuple[str, ...] = ("ts_mean", "ts_std", "ts_sum", "ts_min", "ts_max",
                           "ts_rank", "ts_argmax", "ts_argmin", "ts_skew",
                           "ts_median", "ts_decay_linear", "ts_ir",
                           "ts_zscore", "ts_slope", "ts_rsquare", "ts_resi",
                           "ts_delta", "ts_pct", "ts_max_diff", "ts_min_diff",
                           "ema")
TS2_OPS: Tuple[str, ...] = ("ts_corr", "ts_cov", "ts_beta", "ts_reg_rsq",
                            "ts_reg_resi")
BIN_OPS: Tuple[str, ...] = ("add", "sub", "mul", "div", "spread", "geom_mean",
                            "harm_mean", "max2", "min2")
DEFAULT_WINDOWS: Tuple[int, ...] = (5, 10, 20, 60)


# --------------------------------------------------------------------------
# 配置与结果结构
# --------------------------------------------------------------------------
@dataclass
class SearchConfig:
    """搜索配置（全部落盘到结果里，保证"同样的配置得到同样的因子"）。"""

    horizon: int = 5                     # 筛选所用持有期
    max_layers: int = 3
    width: int = 40                      # 每层保留条数（宽度）
    max_expr: int = 8000                 # 总求值预算（条）
    max_seconds: float = 600.0
    patience: int = 1                    # 连续多少层无提升即早停
    min_ic: float = 0.010                # 筛选阶段的 |RankIC| 门槛
    min_coverage: float = 0.60
    prune_corr: float = 0.95             # 与已入选者 |相关| 超过即视为重复
    prune_corr_k: int = 8                # 每次最多比对几条已入选者
    dedup_corr: float = 0.999            # 跨层去重相关性
    complexity_penalty: float = 0.03     # 复杂度惩罚系数
    coverage_ref: float = 0.80           # 覆盖率达到该值即不扣分
    windows: Sequence[int] = DEFAULT_WINDOWS
    unary_ops: Sequence[str] = UNARY_OPS
    cs_ops: Sequence[str] = CS_OPS
    ts_ops: Sequence[str] = TS_OPS
    ts2_ops: Sequence[str] = TS2_OPS
    bin_ops: Sequence[str] = BIN_OPS
    cs2_ops: Sequence[str] = ("dgtw_cs",)      # 二元横截面（分组调整）
    dgtw_groups: Sequence[int] = (3, 5, 10)
    pair_k: int = 6                      # 两两组合时每层使用的"精英"条数
    neutral_controls: Sequence[str] = ("size",)
    allow_neutral: bool = True
    top_k: int = 20                      # 最终进入四维评价的条数
    with_risk: bool = True
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__.items())
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d


@dataclass
class Candidate:
    """一条通过筛选的候选因子。"""

    node: ex.Node
    layer: int
    score: float
    ic_mean: float
    icir: float
    ic_win: float
    coverage: float
    n_nodes: int
    values: Optional[pd.DataFrame] = None

    @property
    def expression(self) -> str:
        return self.node.render()

    def to_dict(self, with_values: bool = False) -> Dict[str, Any]:
        d = {"expression": self.expression, "key": self.node.key(),
             "layer": self.layer, "score": self.score, "ic_mean": self.ic_mean,
             "icir": self.icir, "ic_win": self.ic_win,
             "coverage": self.coverage, "n_nodes": self.n_nodes}
        return d


@dataclass
class LayerStats:
    """单层统计（用来解释"为什么停在这一层"）。"""

    layer: int
    generated: int = 0
    screened: int = 0
    kept: int = 0
    pruned_invalid: int = 0
    pruned_error: int = 0
    pruned_coverage: int = 0
    pruned_low_ic: int = 0
    pruned_dup: int = 0
    pruned_corr: int = 0
    replaced: int = 0
    best_score: float = 0.0
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class SearchResult:
    """搜索结果：候选表 + 入围因子的完整评价报告 + 逐层审计。"""

    candidates: List[Candidate] = dc_field(default_factory=list)
    reports: List[EV.FactorReport] = dc_field(default_factory=list)
    history: List[LayerStats] = dc_field(default_factory=list)
    config: Optional[Dict[str, Any]] = None
    panel_info: Optional[Dict[str, Any]] = None
    elapsed: float = 0.0
    n_evaluated: int = 0
    stop_reason: str = ""

    @property
    def pool(self) -> Dict[str, pd.DataFrame]:
        """入围因子池（供下一次挖掘做增量 IC 与去重基准）。"""
        return {r.name: r.detail["factor"] for r in self.reports
                if r.ok and "factor" in r.detail}

    def table(self) -> pd.DataFrame:
        rows = [c.to_dict() for c in self.candidates]
        df = pd.DataFrame(rows)
        return df.sort_values("score", ascending=False).reset_index(drop=True) \
            if len(df) else df

    def report_table(self) -> pd.DataFrame:
        return EV.rank_reports(self.reports)

    def history_table(self) -> pd.DataFrame:
        return pd.DataFrame([h.to_dict() for h in self.history])

    def top(self, k: int = 10) -> List[Candidate]:
        return sorted(self.candidates, key=lambda c: -c.score)[:k]

    def best_reports(self, k: int = 10) -> List[EV.FactorReport]:
        return sorted([r for r in self.reports if r.ok],
                      key=lambda r: -r.score)[:k]

    def to_dict(self) -> Dict[str, Any]:
        return {"config": self.config, "panel": self.panel_info,
                "elapsed": round(self.elapsed, 3), "n_evaluated": self.n_evaluated,
                "stop_reason": self.stop_reason,
                "history": [h.to_dict() for h in self.history],
                "candidates": [c.to_dict() for c in self.candidates],
                "reports": [r.to_dict() for r in self.reports]}


# --------------------------------------------------------------------------
# 搜索引擎
# --------------------------------------------------------------------------
class GridMiner:
    """分层算子网格搜索。"""

    def __init__(self, panel: PanelData,
                 config: Optional[SearchConfig] = None,
                 registry: Optional[FieldRegistry] = None,
                 seeds: Sequence[str] = (),
                 pool: Optional[Dict[str, pd.DataFrame]] = None) -> None:
        self.panel = panel
        self.cfg = config or SearchConfig()
        self.registry = registry or panel.registry or default_registry()
        self.seeds = list(seeds)
        self.pool = dict(pool or {})
        self.evaluator = ex.Evaluator(panel, self.registry)
        self.screener = EV.ICScreener(panel, min_stocks=20)
        self._rng = np.random.default_rng(self.cfg.seed)
        info = panel.describe()
        info["numba"] = ops.numba_available()
        self._result = SearchResult(config=self.cfg.to_dict(), panel_info=info)
        self._seen: Dict[str, Candidate] = {}

    # -- 基础工具 --
    def _base_nodes(self) -> List[ex.Node]:
        """种子：原始字段 + 用户给定表达式（去重后）。"""
        nodes: List[ex.Node] = []

        def _add(n: ex.Node) -> None:
            k = n.key()
            if all(x.key() != k for x in nodes):
                nodes.append(n)

        for name in self.registry.names():
            # 只有面板真能提供（或能派生的）字段才参与搜索
            if name in self.panel:
                _add(ex.Field(name))
        for text in self.seeds:
            try:
                node = ex.parse(text)
            except ex.ExprError:
                continue
            if not ex.validate(node, self.registry):
                _add(node)
        return nodes

    def _screen(self, node: ex.Node) -> Tuple[Optional[Candidate], str]:
        """廉价筛选：返回 (候选, 剪枝原因)。原因非空表示被剪。"""
        errs = ex.validate(node, self.registry)
        if errs:
            return None, "invalid"
        try:
            values = self.evaluator.run(node)
        except Exception:
            return None, "error"
        self._result.n_evaluated += 1
        # 覆盖率门槛用**扣除必然预热期**后的口径：长回看表达式（如 TTM 需要
        # 750 个交易日）不该因为"数据还不够长"被当成垃圾剪掉。
        warmup = ex.lookback(node)
        st = self.screener.screen(values, self.cfg.horizon, warmup=warmup)
        ic = st["ic_mean"]
        if not np.isfinite(ic):
            return None, "coverage"
        if st["coverage_adj"] < self.cfg.min_coverage:
            return None, "coverage"
        if abs(ic) < self.cfg.min_ic:
            return None, "low_ic"
        icir = st["icir"] if np.isfinite(st["icir"]) else 0.0
        cov_f = min(1.0, st["coverage_adj"] / max(self.cfg.coverage_ref, 1e-9))
        score = (abs(ic) * min(abs(icir), 3.0) * cov_f
                 / (1.0 + self.cfg.complexity_penalty * node.size()))
        return Candidate(node=node, layer=0, score=float(score),
                         ic_mean=float(ic), icir=float(icir),
                         ic_win=float(st["ic_win"]) if np.isfinite(st["ic_win"]) else 0.0,
                         coverage=float(st["coverage_adj"]), n_nodes=node.size(),
                         values=values), ""

    # -- 候选生成 --
    def _wrap(self, node: ex.Node, layer: int) -> Iterator[ex.Node]:
        for op in self.cfg.unary_ops:
            yield ex.Call(op, (node,))
        for op in self.cfg.cs_ops:
            yield ex.Call(op, (node,))
        for op in self.cfg.ts_ops:
            for w in self.cfg.windows:
                yield ex.Call(op, (node,), int(w))

    def _pair(self, a: ex.Node, b: ex.Node) -> Iterator[ex.Node]:
        for op in self.cfg.bin_ops:
            yield ex.Call(op, (a, b))
        for op in self.cfg.ts2_ops:
            for w in self.cfg.windows:
                yield ex.Call(op, (a, b), int(w))
        try:                                 # 分组变量需无量纲（score/ratio）
            if b.type_of(self.registry).is_dimensionless():
                for op in self.cfg.cs2_ops:
                    for g in self.cfg.dgtw_groups:
                        yield ex.Call(op, (a, b), int(g))
        except ex.ExprError:
            pass

    def _expand(self, elite: List[ex.Node], bases: List[ex.Node],
                layer: int) -> List[ex.Node]:
        """按层生成候选（返回列表，便于按预算抽样）。"""
        out: List[ex.Node] = []
        for n in elite:
            out.extend(self._wrap(n, layer))
        if layer >= 2:
            k = min(self.cfg.pair_k, len(elite))
            top = elite[:k]
            for i, a in enumerate(top):
                # 只做 i<j 的组合：a 与自身配对会生成 min2(a,a) 这类
                # 与 a 完全等价的冗余表达式，白白吃掉求值预算
                for b in top[i + 1:]:
                    out.extend(self._pair(a, b))
                if layer == 2:
                    # 第 2 层允许"精英 × 原始字段"：抑制组合在精英之间反复
                    # 近亲繁殖，导致候选之间相关性过高
                    for b in bases[:max(4, k)]:
                        if b.key() != a.key():
                            out.extend(self._pair(a, b))
        if layer >= 3 and self.cfg.allow_neutral:
            for n in elite[:max(2, self.cfg.pair_k // 2)]:
                ctrls = tuple(ex.Field(c) for c in self.cfg.neutral_controls
                              if c in self.panel)
                if ctrls:
                    out.append(ex.Neutral(n, ctrls))
        return out

    # -- 主流程 --
    def run(self) -> SearchResult:
        cfg = self.cfg
        t0 = time.perf_counter()
        bases = self._base_nodes()
        if not bases:
            self._result.stop_reason = "无可用种子字段"
            self._result.elapsed = time.perf_counter() - t0
            return self._result

        # 第 0 层：种子筛选（种子不参与 |IC| 门槛，它们只是原料）
        elite: List[Candidate] = []
        for n in bases:
            errs = ex.validate(n, self.registry)
            if errs:
                continue
            try:
                values = self.evaluator.run(n)
            except Exception:
                continue
            self._result.n_evaluated += 1
            st = self.screener.screen(values, cfg.horizon,
                                      warmup=ex.lookback(n))
            ic = st["ic_mean"] if np.isfinite(st["ic_mean"]) else 0.0
            elite.append(Candidate(
                node=n, layer=0, score=0.0, ic_mean=float(ic),
                icir=float(st["icir"]) if np.isfinite(st["icir"]) else 0.0,
                ic_win=float(st["ic_win"]) if np.isfinite(st["ic_win"]) else 0.0,
                coverage=float(st["coverage_adj"]), n_nodes=n.size(),
                values=values))

        best_ever = -np.inf
        stale = 0
        cur = elite
        for layer in range(1, cfg.max_layers + 1):
            lt0 = time.perf_counter()
            stats = LayerStats(layer=layer)
            elite_nodes = [c.node for c in sorted(cur, key=lambda c: -c.score)]
            gen = self._expand(elite_nodes, bases, layer)
            stats.generated = len(gen)
            remaining = max(0, cfg.max_expr - self._result.n_evaluated)
            if remaining == 0:
                self._result.stop_reason = "求值预算耗尽"
                break
            if len(gen) > remaining:
                idx = self._rng.choice(len(gen), size=remaining, replace=False)
                gen = [gen[i] for i in sorted(idx)]

            kept: List[Candidate] = []
            seen_keys: Dict[str, int] = {}
            for node in gen:
                if time.perf_counter() - t0 > cfg.max_seconds:
                    self._result.stop_reason = "时间预算耗尽"
                    break
                cand, reason = self._screen(node)
                stats.screened += 1
                if cand is None:
                    if reason == "invalid":
                        stats.pruned_invalid += 1
                    elif reason == "error":
                        stats.pruned_error += 1
                    elif reason == "coverage":
                        stats.pruned_coverage += 1
                    else:
                        stats.pruned_low_ic += 1
                    continue
                cand.layer = layer
                k = node.key()
                if k in seen_keys or k in self._seen:
                    stats.pruned_dup += 1
                    continue
                # 相关性剪枝：只和已入选的头部比（把比较次数限制在常数级）
                dup = False
                for other in sorted(kept, key=lambda c: -c.score)[:cfg.prune_corr_k]:
                    r = float(EV.rank_ic(cand.values, other.values,
                                         min_stocks=20).abs().mean())
                    if np.isfinite(r) and r > cfg.prune_corr:
                        if cand.score > other.score:
                            kept.remove(other)   # 同源因子只留更强的那条
                            kept.append(cand)
                            seen_keys[k] = 1
                            stats.replaced += 1
                        else:
                            stats.pruned_corr += 1
                        dup = True
                        break
                if dup:
                    continue
                seen_keys[k] = 1
                kept.append(cand)

            kept.sort(key=lambda c: -c.score)
            if len(kept) > cfg.width:
                kept = kept[:cfg.width]
            stats.kept = len(kept)
            stats.best_score = float(kept[0].score) if kept else 0.0
            stats.seconds = round(time.perf_counter() - lt0, 3)
            self._result.history.append(stats)

            if not kept:
                self._result.stop_reason = "本层无候选通过剪枝"
                break
            if stats.best_score > best_ever + 1e-6:
                best_ever = stats.best_score
                stale = 0
            else:
                stale += 1
                if stale > cfg.patience:
                    cur = kept
                    self._result.stop_reason = (
                        f"连续 {stale} 层最佳筛选分未提升，提前停止")
                    break
            cur = kept
            for c in kept:
                self._seen[c.node.key()] = c
        else:
            self._result.stop_reason = "达到最大层数"

        self._result.candidates = cur
        # 最终入围：完整四维评价（+ 风险体检）。逐条把已评价的因子加入 pool，
        # 这样"增量 IC"衡量的是"相对前面已经选中的因子还有多少新信息"。
        pool = dict(self.pool)
        for c in cur[:max(0, cfg.top_k)]:
            rep = EV.evaluate(c.values, self.panel, name=c.expression,
                              expression=c.expression, pool=pool or None,
                              with_risk=cfg.with_risk, node=c.node)
            self._result.reports.append(rep)
            pool[c.expression] = c.values
        self._result.elapsed = time.perf_counter() - t0
        if (self._result.n_evaluated >= cfg.max_expr
                and not self._result.stop_reason.startswith("求值预算")):
            self._result.stop_reason += "（求值预算已耗尽）"
        return self._result


# --------------------------------------------------------------------------
# 便捷入口
# --------------------------------------------------------------------------
def mine(panel: PanelData,
         config: Optional[SearchConfig] = None,
         seeds: Sequence[str] = (),
         pool: Optional[Dict[str, pd.DataFrame]] = None,
         registry: Optional[FieldRegistry] = None) -> SearchResult:
    """一行完成一次分层网格搜索。"""
    return GridMiner(panel, config=config, seeds=seeds, pool=pool,
                     registry=registry).run()
