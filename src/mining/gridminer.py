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

import math
import time
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import evaluator as EV
from . import expr as ex
from . import ops
from . import split as SP
from .panel import FieldRegistry, PanelData, default_registry, residualize_cs

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

    # -- 样本外确认（purged walk-forward，见 ``mining.split``）--
    split_mode: str = "auto"             # auto | split | soft | off
    confirm_frac: float = 0.25           # 确认段占比（时间顺序的末段）
    n_confirm_folds: int = 3             # 确认段内切几个 walk-forward 折叠
    embargo: Optional[int] = None        # 折叠之间的额外隔离天数（默认 = horizon）
    min_days_for_split: int = 488        # 不足（≈两年，244×2）时退化为 soft：不硬切
    # -- 筛选分口径：分段一致性 --
    seg_consistency: bool = True         # 抑制"只在某一段成立"的噪声因子
    n_screen_segments: int = 4
    w_consistency: float = 0.50          # 一致性对筛选分的权重（0 = 关闭）
    # -- 筛选分口径：增量信息前移 --
    incremental: bool = True             # 用"对已选池正交后的残差 IC"打折扣
    w_incremental: float = 0.50
    max_incr_pool: int = 12              # 残差化基准池最多保留几条
    # -- 自适应预算（按算子族 UCB 分配）--
    adaptive_budget: bool = True
    budget_explore: float = 1.0          # UCB 探索系数（0 = 纯贪心）
    # -- 相关性剪枝加速 --
    prune_sample_step: int = 4           # 剪枝比对时每隔 k 天取一天（1 = 不抽样）
    # -- 基因库 warm start（默认关闭，见 mining.genome）--
    genome_seeds: int = 6                # 传了 gene bank 时最多注入几条种子

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
    # -- 新增口径（搜索期就能看出"稳不稳""新不新"）--
    consistency: float = 1.0             # 分段一致性 ∈[0,1]：最差段 IC / 全段 IC
    incr_ratio: float = 1.0              # 残差 IC / 原始 IC ∈[0,1]：1 = 全新信息
    ic_resid: float = 0.0                # 对已选池正交后的 IC（残差 IC）

    @property
    def expression(self) -> str:
        return self.node.render()

    def to_dict(self, with_values: bool = False) -> Dict[str, Any]:
        d = {"expression": self.expression, "key": self.node.key(),
             "layer": self.layer, "score": self.score, "ic_mean": self.ic_mean,
             "icir": self.icir, "ic_win": self.ic_win,
             "coverage": self.coverage, "n_nodes": self.n_nodes,
             "consistency": self.consistency, "incr_ratio": self.incr_ratio,
             "ic_resid": self.ic_resid}
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
    split: Optional[Dict[str, Any]] = None     # 时间切分方案（见 mining.split）
    oos: Optional[Dict[str, Any]] = None       # purged walk-forward 复核结果

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
                "stop_reason": self.stop_reason, "split": self.split, "oos": self.oos,
                "history": [h.to_dict() for h in self.history],
                "candidates": [c.to_dict() for c in self.candidates],
                "reports": [r.to_dict() for r in self.reports]}

    @property
    def oos_table(self) -> pd.DataFrame:
        """样本外复核表（IS vs OOS 对照）。无确认集时返回空表。"""
        rows = (self.oos or {}).get("factors") or {}
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(list(rows.values()))
        return df.sort_values("oos_ic_mean", ascending=False).reset_index(drop=True) \
            if "oos_ic_mean" in df.columns and len(df) else df


# --------------------------------------------------------------------------
# 搜索引擎
# --------------------------------------------------------------------------
class GridMiner:
    """分层算子网格搜索。"""

    def __init__(self, panel: PanelData,
                 config: Optional[SearchConfig] = None,
                 registry: Optional[FieldRegistry] = None,
                 seeds: Sequence[str] = (),
                 pool: Optional[Dict[str, pd.DataFrame]] = None,
                 genome: Optional[Any] = None) -> None:
        self.panel = panel
        self.cfg = config or SearchConfig()
        self.genome = genome     # 因子基因库（默认 None = 关闭，见 mining.genome）
        self.registry = registry or panel.registry or default_registry()
        self.seeds = list(seeds)
        self.pool = dict(pool or {})
        # 时间切分：生成/筛选/早停只在 search 段进行，confirm 段留给最后复核。
        # 样本不足时 plan.mode == "soft"，search_panel 即完整面板，不硬切。
        self.plan = SP.make_split(panel, confirm_frac=self.cfg.confirm_frac,
                                  n_folds=self.cfg.n_confirm_folds,
                                  horizon=self.cfg.horizon,
                                  embargo=self.cfg.embargo,
                                  min_days=self.cfg.min_days_for_split,
                                  mode=self.cfg.split_mode)
        self.search_panel = (panel.slice_dates(end=self.plan.search_end)
                             if self.plan.has_confirm else panel)
        self.evaluator = ex.Evaluator(self.search_panel, self.registry)
        self.full_evaluator = ex.Evaluator(panel, self.registry)   # 复核期用
        self.screener = EV.ICScreener(self.search_panel, min_stocks=20)
        self._rng = np.random.default_rng(self.cfg.seed)
        info = panel.describe()
        info["numba"] = ops.numba_available()
        info["split"] = self.plan.to_dict()
        self._result = SearchResult(config=self.cfg.to_dict(), panel_info=info,
                                    split=self.plan.to_dict())
        self._seen: Dict[str, Candidate] = {}
        # 增量信息基准：已入选因子的值（search 段），用于残差 IC
        self._pool_values: List[pd.DataFrame] = []
        self._comp: Optional[pd.DataFrame] = None
        self._comp_stamp: int = -1
        # 自适应预算：按 (层, 算子族) 统计筛选分产出
        self._fam: Dict[int, Dict[str, List[float]]] = {}

    # -- 基础工具 --
    def _slice(self, frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """把外部传入的因子值对齐到搜索段（评价只在搜索段口径下进行）。"""
        if frame is None or self.search_panel is self.panel:
            return frame
        return frame.reindex(index=self.search_panel.dates,
                             columns=self.search_panel.symbols)
    def _base_nodes(self) -> List[ex.Node]:
        """种子：原始字段 + 用户给定表达式 + 基因库 warm start（去重后）。"""
        nodes: List[ex.Node] = []

        def _add(n: ex.Node) -> None:
            k = n.key()
            if all(x.key() != k for x in nodes):
                nodes.append(n)

        for name in self.registry.names():
            # 只有面板真能提供（或能派生的）字段才参与搜索
            if name in self.panel:
                _add(ex.Field(name))
        seed_texts = list(self.seeds)
        if self.genome is not None:
            try:                            # 基因库损坏不该拖垮挖掘
                seed_texts += list(self.genome.seeds(
                    self.panel, n=int(self.cfg.genome_seeds)))
            except Exception:               # pragma: no cover - 防御
                pass
        for text in seed_texts:
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
        ic_series = self.screener.ic(values, self.cfg.horizon)
        st = EV.ic_stats(ic_series)
        st.update(EV.data_quality(values, warmup))
        ic = st["ic_mean"]
        if not np.isfinite(ic):
            return None, "coverage"
        if st["coverage_adj"] < self.cfg.min_coverage:
            return None, "coverage"
        if abs(ic) < self.cfg.min_ic:
            return None, "low_ic"
        icir = st["icir"] if np.isfinite(st["icir"]) else 0.0
        cov_f = min(1.0, st["coverage_adj"] / max(self.cfg.coverage_ref, 1e-9))
        # 分段一致性：全段 IC 可能由某一段极端行情贡献，最差段与全段反号的
        # 直接把一致性打到 0（其余用 |最差段| / |全段| 线性衰减）。
        cons = self._consistency(ic_series, ic) if self.cfg.seg_consistency else 1.0
        eff_ic = abs(ic) * (1.0 - self.cfg.w_consistency
                            + self.cfg.w_consistency * cons)
        score = (eff_ic * min(abs(icir), 3.0) * cov_f
                 / (1.0 + self.cfg.complexity_penalty * node.size()))
        # 增量信息：与已入选池正交后的残差 IC 占原始 IC 的比例，越接近 1 越"新"
        ic_res, ratio = float(ic), 1.0
        if self.cfg.incremental:
            ic_res, ratio = self._incremental(values, ic)
            score *= (1.0 - self.cfg.w_incremental
                      + self.cfg.w_incremental * ratio)
        return Candidate(node=node, layer=0, score=float(score),
                         ic_mean=float(ic), icir=float(icir),
                         ic_win=float(st["ic_win"]) if np.isfinite(st["ic_win"]) else 0.0,
                         coverage=float(st["coverage_adj"]), n_nodes=node.size(),
                         values=values, consistency=float(cons),
                         incr_ratio=float(ratio),
                         ic_resid=float(ic_res) if np.isfinite(ic_res) else 0.0), ""

    # -- 筛选口径的两个修正项 --
    def _consistency(self, ic_series: pd.Series, ic: float) -> float:
        """分段一致性 ∈[0,1]：最差段 IC 相对全段 IC 的比值（反号记 0）。"""
        if not np.isfinite(ic) or abs(ic) < 1e-12:
            return 1.0
        seg = EV.segmented_ic(ic_series, int(self.cfg.n_screen_segments))
        means = [m for m in (seg.get("segment_means") or [])
                 if np.isfinite(m)]
        if not means:                     # 样本太短切不出段 → 不惩罚
            return 1.0
        # "最弱段"要按方向取：IC 为正取最小段，IC 为负取最大段（越接近 0 越弱）。
        # 直接用 segmented_ic 的 worst_seg_ic（= np.min）会把负 IC 因子的
        # **最强**段当成最弱段，一致性恒等于 1，等于这项修正没生效。
        worst = min(means) if ic > 0 else max(means)
        if worst * ic <= 0.0:             # 存在反号段 → 一致性归零
            return 0.0
        return float(min(1.0, abs(worst) / abs(ic)))

    def _composite(self) -> Optional[pd.DataFrame]:
        """已入选因子的等权复合（截面标准化后平均），作为残差化的对照变量。"""
        if not self._pool_values:
            return None
        if self._comp is not None and self._comp_stamp == len(self._pool_values):
            return self._comp
        comp = sum(ops.cs_zscore(v) for v in self._pool_values) / len(self._pool_values)
        self._comp = comp
        self._comp_stamp = len(self._pool_values)
        return comp

    def _incremental(self, values: pd.DataFrame,
                     ic: float) -> Tuple[float, float]:
        """相对已选池还剩下多少新信息，返回 (残差 IC, 增量占比 ∈[0,1])。

        两个口径取**更保守**的那个：

        * ``1 − |ρ(候选, 池复合)|``——直接度量"有多像已经选过的东西"。RankIC
          是秩相关，对尺度不敏感，因此"残差化后 IC 只掉 7%"这种事很常见；
          用相关性口径才抓得住同构因子。
        * ``|残差 IC| / |原始 IC|``——对池复合做截面正交化后还剩多少预测力，
          这是"增量信息"的严格定义。

        只对**通过了 |IC| 门槛**的候选调用（数量远小于总预算），代价可控。
        """
        comp = self._composite()
        if comp is None or not np.isfinite(ic) or abs(ic) < 1e-12:
            return float(ic), 1.0
        corr = float(EV.rank_ic(values, comp, min_stocks=20).abs().mean())
        red = 1.0 - min(1.0, corr) if np.isfinite(corr) else 1.0
        try:
            resid = residualize_cs(values, [comp], min_stocks=20,
                                   standardize=True)
        except Exception:                 # pragma: no cover - 数值兜底
            return float(ic), float(red)
        icr = self.screener.ic(resid, self.cfg.horizon)
        m = float(icr.mean()) if icr.notna().any() else float("nan")
        if not np.isfinite(m):
            return float(ic), float(red)
        return m, float(min(red, min(1.0, abs(m) / abs(ic))))

    # -- 候选生成 --
    def _allocate(self, gen: List[ex.Node], remaining: int,
                  layer: int) -> List[ex.Node]:
        """按算子族的产出分配预算（UCB），替掉"均匀随机丢掉 3/4 候选"。

        第一层各族都没有历史，UCB 退化为均匀；之后产出高的族拿更多配额，
        没试过的族靠探索项保底。配额取整的余量/缺口用未选中的候选补齐或截断，
        因此**总条数严格等于预算**，可复现性不受影响。
        """
        cfg = self.cfg
        if not cfg.adaptive_budget:
            idx = self._rng.choice(len(gen), size=remaining, replace=False)
            return [gen[i] for i in sorted(idx)]
        buckets: Dict[str, List[int]] = {}
        for i, n in enumerate(gen):
            buckets.setdefault(self._family(n), []).append(i)
        stat = self._fam.setdefault(layer, {})
        seen = [v for scores in stat.values() for v in scores]
        prior = float(np.mean(seen)) if seen else 0.0
        n_total = 1 + len(seen)
        ucb: Dict[str, float] = {}
        for fam in buckets:
            sc = stat.get(fam) or []
            mean = float(np.mean(sc)) if sc else prior
            ucb[fam] = mean + float(cfg.budget_explore) * math.sqrt(
                2.0 * math.log(n_total) / (len(sc) + 1))
        lo = min(ucb.values()) if ucb else 0.0
        w = {f: max(u - lo, 0.0) + 1e-6 for f, u in ucb.items()}
        tot = sum(w.values()) or 1.0
        picked: List[int] = []
        taken: set = set()
        for fam, idxs in buckets.items():
            share = min(len(idxs), round(remaining * w[fam] / tot))
            if share <= 0:
                continue
            chosen = [int(i) for i in self._rng.choice(idxs, size=share,
                                                       replace=False)]
            picked.extend(chosen)
            taken.update(chosen)
        if len(picked) < remaining:
            rest = [i for i in range(len(gen)) if i not in taken]
            extra = min(remaining - len(picked), len(rest))
            if extra > 0:
                picked.extend(int(i) for i in self._rng.choice(
                    rest, size=extra, replace=False))
        elif len(picked) > remaining:
            picked = [int(i) for i in self._rng.choice(picked, size=remaining,
                                                       replace=False)]
        return [gen[i] for i in sorted(picked)]

    def _add_to_incr_pool(self, values: Optional[pd.DataFrame]) -> None:
        """新入选的因子进入残差化基准池（只保留最近若干条）。"""
        if values is None:
            return
        self._pool_values.append(values)
        cap = max(1, int(self.cfg.max_incr_pool))
        if len(self._pool_values) > cap:
            self._pool_values = self._pool_values[-cap:]

    def _family(self, node: ex.Node) -> str:
        """候选所属算子族（自适应预算按族分配配额）。"""
        if isinstance(node, ex.Neutral):
            return "neutral"
        if isinstance(node, ex.Call):
            try:
                return ops.get_op(node.name).family
            except KeyError:               # pragma: no cover - 未注册算子
                return "elem"
        return "field"
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
                gen = self._allocate(gen, remaining, layer)

            kept: List[Candidate] = []
            seen_keys: Dict[str, int] = {}
            step = max(1, int(cfg.prune_sample_step))
            for node in gen:
                if time.perf_counter() - t0 > cfg.max_seconds:
                    self._result.stop_reason = "时间预算耗尽"
                    break
                cand, reason = self._screen(node)
                stats.screened += 1
                # 自适应预算的反馈信号：该族这条候选的产出（被剪即 0 分）
                self._fam.setdefault(layer, {}).setdefault(
                    self._family(node), []).append(
                        0.0 if cand is None else float(cand.score))
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
                # 相关性剪枝：只和已入选的头部比（把比较次数限制在常数级）。
                # 比对时对日期等距抽样（prune_sample_step）：横截面相关的天与天
                # 之间高度冗余，抽 1/k 的天数把这部分开销降一个量级，排序结论不变。
                dup = False
                for other in sorted(kept, key=lambda c: -c.score)[:cfg.prune_corr_k]:
                    r = float(EV.rank_ic(cand.values.iloc[::step],
                                         other.values.iloc[::step],
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
                self._add_to_incr_pool(cand.values)

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
        # 最终入围：完整四维评价（+ 风险体检）。评价口径固定在**搜索段**（与
        # 筛选同源，否则 IS 指标会被确认段污染）；逐条把已评价的因子加入 pool，
        # 这样"增量 IC"衡量的是"相对前面已经选中的因子还有多少新信息"。
        pool = {k: self._slice(v) for k, v in self.pool.items()}
        for c in cur[:max(0, cfg.top_k)]:
            rep = EV.evaluate(c.values, self.search_panel, name=c.expression,
                              expression=c.expression, pool=pool or None,
                              with_risk=cfg.with_risk, node=c.node)
            self._result.reports.append(rep)
            pool[c.expression] = c.values
        # 确认段开封：搜索已经结束，这里只做一次复核，**不允许**因为复核结果
        # 回头改搜索（改了就等于把确认段又变成了搜索段）。
        self._result.oos = self._walk_forward_check(cur[:max(0, cfg.top_k)])
        self._attach_oos()
        self._result.elapsed = time.perf_counter() - t0
        if (self._result.n_evaluated >= cfg.max_expr
                and not self._result.stop_reason.startswith("求值预算")):
            self._result.stop_reason += "（求值预算已耗尽）"
        return self._result

    # -- 样本外复核（purged walk-forward）--
    def _walk_forward_check(self, cands: List[Candidate]) -> Dict[str, Any]:
        """在确认段的每个 fold 上重算 IC，给出 IS / OOS 对照。

        复核要在**完整面板**上重新求值——搜索期只算了搜索段的值。代价只有
        ``top_k`` 次求值，可以忽略。
        """
        cfg = self.cfg
        if not self.plan.has_confirm:
            return {"mode": "soft", "reason": self.plan.reason, "folds": [],
                    "factors": {}, "summary": {"n": 0}}
        fwd = self.panel.fwd(cfg.horizon)
        factors: Dict[str, Dict[str, Any]] = {}
        for c in cands:
            try:
                full = self.full_evaluator.run(c.node)
            except Exception:                 # pragma: no cover - 求值兜底
                continue
            per = [SP.fold_ic(full, fwd, f) for f in self.plan.folds]
            ics = [p["ic_mean"] for p in per if np.isfinite(p["ic_mean"])]
            if not ics:
                continue
            oos = float(np.mean(ics))
            is_ic = float(c.ic_mean)
            decay = (1.0 - oos / is_ic) if abs(is_ic) > 1e-9 else float("nan")
            factors[c.expression] = {
                "expression": c.expression,
                "is_ic": round(is_ic, 6),
                "oos_ic_mean": round(oos, 6),
                "oos_icir": round(float(np.mean([p["icir"] for p in per])), 6),
                "oos_ic_win": round(float(np.mean([p["ic_win"] for p in per])), 4),
                "oos_fold_ic": [round(float(p["ic_mean"]), 6) for p in per],
                "oos_fold_pos": int(sum(1 for v in ics if v > 0)),
                "n_folds": len(ics),
                "decay": float(decay) if np.isfinite(decay) else None,
                "sign_flip": bool(is_ic * oos < 0),
            }
        ics_all = [f["oos_ic_mean"] for f in factors.values()]
        decays = [f["decay"] for f in factors.values()
                  if f.get("decay") is not None]
        return {
            "mode": "split",
            "reason": self.plan.reason,
            "plan": self.plan.to_dict(),
            "folds": [f.to_dict() for f in self.plan.folds],
            "factors": factors,
            "summary": {
                "n": len(factors),
                "oos_pos_rate": (float(np.mean([v > 0 for v in ics_all]))
                                 if ics_all else None),
                "median_decay": float(np.median(decays)) if decays else None,
                "worst_decay": float(np.max(decays)) if decays else None,
                "sign_flip_rate": (float(np.mean([f["sign_flip"]
                                                  for f in factors.values()]))
                                   if factors else None),
            },
        }

    def _attach_oos(self) -> None:
        """把复核结果回写到报告（只做标注，不改评分——评分是搜索段的判断）。"""
        rows = (self._result.oos or {}).get("factors") or {}
        if not rows:
            return
        for rep in self._result.reports:
            row = rows.get(rep.name) or rows.get(rep.expression)
            if not row:
                continue
            rep.metrics["is_ic"] = float(row.get("is_ic", float("nan")))
            rep.metrics["oos_ic_mean"] = float(row.get("oos_ic_mean", float("nan")))
            rep.metrics["oos_icir"] = float(row.get("oos_icir", float("nan")))
            rep.metrics["oos_decay"] = (float(row["decay"])
                                        if row.get("decay") is not None
                                        else float("nan"))
            rep.detail["oos"] = row
            if row.get("sign_flip"):
                rep.detail["oos_warning"] = (
                    "样本外 IC 与样本内反号：这条极可能是搜索期从噪声里挑出来的")
            elif row.get("decay") is not None and row["decay"] > 0.60:
                rep.detail["oos_warning"] = (
                    f"样本外 IC 衰减 {row['decay']:.0%}：搜索期 IC 里有相当"
                    "一部分是选择偏差而非真实预测力")


# --------------------------------------------------------------------------
# 便捷入口
# --------------------------------------------------------------------------
def mine(panel: PanelData,
         config: Optional[SearchConfig] = None,
         seeds: Sequence[str] = (),
         pool: Optional[Dict[str, pd.DataFrame]] = None,
         registry: Optional[FieldRegistry] = None,
         genome: Optional[Any] = None) -> SearchResult:
    """一行完成一次分层网格搜索。

    :param genome: 因子基因库（``mining.genome.GenomeBank``）。传入后：搜索前
        用它做 warm start，搜索后把优质候选写回。**默认关闭**——注入种子会
        改变候选集，进而打破"同种子同结果"的可复现约定。
    """
    m = GridMiner(panel, config=config, seeds=seeds, pool=pool,
                  registry=registry, genome=genome)
    res = m.run()
    if genome is not None:
        try:
            genome.add_search(res, panel, top_n=int(m.cfg.top_k))
        except Exception:                   # pragma: no cover - 写库失败不影响结果
            pass
    return res
