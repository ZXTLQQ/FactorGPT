"""时间切分与样本外确认（purged walk-forward）。

为什么必须有这一段：网格搜索在同一段数据上既筛选又评价，8 000 条候选里
纯噪声也能稳定跑出几条"显著"因子。这是**选择偏差**——多重检验校正
（Sidak / BH-FDR，见 ``engine.significance``）校正的是 p 值，校不掉
"这条因子是被同一段数据挑出来的"这一事实。

本模块给搜索一个它没见过的确认集，分三层：

1. **硬切分**：时间轴切成 ``search`` / ``confirm`` 两段。搜索（候选生成、
   筛选、分层、早停）只看 search 段；confirm 段在整个搜索结束后才开封。
2. **purged walk-forward**：confirm 段内再切 k 个 fold，逐 fold 复核。
   每个 fold 的 train 段末尾删掉 ``horizon`` 天——否则 train 最后一天的前瞻
   收益会用到 test 段的开头，构成前视泄漏；再删 ``embargo`` 天做缓冲。
   test 段尾部同样 purge，让每个 fold 的标签完全落在自己的块内。
3. **退化路径**：样本不足（默认 < 2 年，约 480 个交易日）时自动退回
   ``soft`` 模式——不硬切分，改用分段 IC 一致性 + bootstrap 判显著。样本本来
   就少，再切掉四分之一只会让两段都不可信。

设计上的一个取舍：搜索期**不**对每个 fold 各跑一遍（那是 3~5 倍的算力）。
搜索只在 search 段跑一次，复核只对最终入围的 ``top_k`` 条做——复核要回答的
是"这批因子能不能泛化"，而不是"哪条最好"（后者仍然是 search 段的判断）。
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_MIN_DAYS",
    "TRADING_DAYS_PER_YEAR",
    "Fold",
    "SplitPlan",
    "make_split",
    "walk_forward_folds",
]

TRADING_DAYS_PER_YEAR = 244
DEFAULT_MIN_DAYS = 2 * TRADING_DAYS_PER_YEAR      # 480 ≈ 两年
_MIN_SEARCH = 60                                   # 搜索段至少这么多天
_MIN_FOLD_TEST = 10                                # 单个 fold 的 test 至少这么多天


def _iso(ts: Any) -> str:
    if ts is None:
        return ""
    try:
        return str(pd.Timestamp(ts).date())
    except (ValueError, TypeError):   # pragma: no cover - 防御
        return ""


@dataclass
class Fold:
    """一个 purged walk-forward 折叠。

    ``train`` / ``test`` 是真正的日期索引（供调用方切片），其余字段供落盘审计。
    """

    name: str
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex
    purge: int = 0
    embargo: int = 0

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "train_start": _iso(self.train[0]) if self.n_train else "",
            "train_end": _iso(self.train[-1]) if self.n_train else "",
            "test_start": _iso(self.test[0]) if self.n_test else "",
            "test_end": _iso(self.test[-1]) if self.n_test else "",
            "n_train": self.n_train,
            "n_test": self.n_test,
            "purge": int(self.purge),
            "embargo": int(self.embargo),
        }


@dataclass
class SplitPlan:
    """一次挖掘的时间切分方案（随结果落盘，保证"同样的切分可复现"）。"""

    mode: str = "soft"             # split | soft
    reason: str = ""
    horizon: int = 5
    purge: int = 0
    embargo: int = 0
    confirm_frac: float = 0.25
    n_folds: int = 3
    search_dates: Optional[pd.DatetimeIndex] = None
    confirm_dates: Optional[pd.DatetimeIndex] = None
    folds: List[Fold] = dc_field(default_factory=list)

    @property
    def has_confirm(self) -> bool:
        return self.mode == "split" and bool(self.folds)

    @property
    def n_search(self) -> int:
        return len(self.search_dates) if self.search_dates is not None else 0

    @property
    def n_confirm(self) -> int:
        return len(self.confirm_dates) if self.confirm_dates is not None else 0

    @property
    def search_end(self) -> str:
        return _iso(self.search_dates[-1]) if self.n_search else ""

    @property
    def search_start(self) -> str:
        return _iso(self.search_dates[0]) if self.n_search else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "horizon": int(self.horizon),
            "purge": int(self.purge),
            "embargo": int(self.embargo),
            "confirm_frac": float(self.confirm_frac),
            "n_folds": int(self.n_folds),
            "search_start": self.search_start,
            "search_end": self.search_end,
            "n_search": self.n_search,
            "n_confirm": self.n_confirm,
            "folds": [f.to_dict() for f in self.folds],
        }


def walk_forward_folds(dates: Sequence[Any], n_folds: int = 3, horizon: int = 5,
                       embargo: Optional[int] = None, start: int = 0,
                       min_train: int = _MIN_SEARCH,
                       purge_test_tail: bool = True) -> List[Fold]:
    """在 ``dates[start:]`` 上切 k 个 purged walk-forward 折叠。

    :param start: 确认段的起始下标（fold 只覆盖这一段；train 仍取它之前的全部）。
    :param purge_test_tail: 是否把 test 段尾部 ``horizon`` 天也 purge 掉
        （让每个 fold 的前瞻收益标签完全落在该 fold 内部）。
    :return: 折叠列表；训练段过短的折叠会被直接丢掉而不是硬凑。
    """
    idx = pd.DatetimeIndex(dates)
    n = len(idx)
    purge = int(max(1, horizon))
    emb = int(purge if embargo is None else max(0, embargo))
    start = int(max(0, min(start, n - 1)))
    span = n - start
    if span < 2 * purge + _MIN_FOLD_TEST:
        return []

    edges = [start + round(span * i / max(1, n_folds)) for i in range(n_folds + 1)]
    out: List[Fold] = []
    for i in range(n_folds):
        a, b = edges[i], edges[i + 1]
        if b - a < _MIN_FOLD_TEST:
            continue
        # test 段尾部 purge：最后 horizon 天的标签要用到 fold 之后的数据
        test_end = b - (purge if purge_test_tail else 0)
        if test_end - a < _MIN_FOLD_TEST:
            continue
        train_end = a - purge - emb
        if train_end < min_train:
            continue        # 训练段太短，这个 fold 没有意义
        out.append(Fold(name=f"fold{i + 1}", train=idx[:train_end],
                        test=idx[a:test_end], purge=purge, embargo=emb))
    return out


def make_split(panel: Any, *, confirm_frac: float = 0.25, n_folds: int = 3,
               horizon: int = 5, embargo: Optional[int] = None,
               min_days: int = DEFAULT_MIN_DAYS, mode: str = "auto",
               max_search_loss: float = 0.45) -> SplitPlan:
    """按时间切出 search / confirm，并给出 confirm 段内的 walk-forward 折叠。

    :param mode: ``auto``（样本够就硬切，不够退 soft）/ ``split``（强制硬切，
        切不出来才退 soft）/ ``soft``（从不硬切）/ ``off``（同 soft）。
    :param max_search_loss: search 段因切分损失的天数占比上限，超过则退 soft。
    """
    idx = pd.DatetimeIndex(panel.dates)
    n = len(idx)
    purge = int(max(1, horizon))
    emb = int(purge if embargo is None else max(0, embargo))
    plan = SplitPlan(horizon=int(horizon), purge=purge, embargo=emb,
                     confirm_frac=float(confirm_frac), n_folds=int(n_folds))

    def _soft(reason: str) -> SplitPlan:
        plan.mode = "soft"
        plan.reason = reason
        plan.search_dates = idx
        plan.confirm_dates = idx[:0]
        plan.folds = []
        return plan

    if mode in ("soft", "off"):
        return _soft("调用方关闭了硬切分，仅用分段一致性 + bootstrap 判显著")
    if n < min_days:
        return _soft(f"样本仅 {n} 个交易日（< {min_days} ≈ 两年）："
                     "切掉确认段后两段都不够长，改用分段一致性 + bootstrap")

    n_confirm = round(n * confirm_frac)
    # 确认段要能切出 n_folds 个有意义的 fold，否则宁可多留一点给它
    need = int(n_folds * (2 * purge + emb + _MIN_FOLD_TEST * 2))
    if n_confirm < need:
        n_confirm = min(max(n_confirm, need), round(n * max_search_loss))
    n_search = n - n_confirm
    # 搜索段尾部 purge：最后 horizon 天的前瞻收益会落进 confirm 段
    search_end = n_search - purge
    if search_end < max(_MIN_SEARCH, int(n * (1.0 - max_search_loss))):
        return _soft(f"可用样本 {n} 天，切出确认段后搜索段只剩 {search_end} 天"
                     f"（< 要求的 {max(_MIN_SEARCH, int(n * (1.0 - max_search_loss)))}），"
                     "改用分段一致性 + bootstrap")

    folds = walk_forward_folds(idx, n_folds=int(n_folds), horizon=int(horizon),
                               embargo=emb, start=n_search, min_train=_MIN_SEARCH)
    if not folds:
        return _soft("确认段切不出任何有效 fold（训练段过短），改用分段一致性 + bootstrap")

    plan.mode = "split"
    plan.reason = (f"搜索段 {search_end} 天（尾部 purge {purge} 天），"
                   f"确认段 {n - n_search} 天切 {len(folds)} 个 walk-forward 折叠")
    plan.search_dates = idx[:search_end]
    plan.confirm_dates = idx[n_search:]
    plan.folds = folds
    return plan


def fold_ic(factor: pd.DataFrame, fwd: pd.DataFrame, fold: Fold,
            min_stocks: int = 20) -> Dict[str, float]:
    """单折叠上的样本外 IC 统计（只取 test 段，标签与因子值都按 fold 对齐）。"""
    from . import evaluator as EV  # 局部导入：evaluator 依赖 panel，避免循环

    f = factor.reindex(index=fold.test)
    y = fwd.reindex(index=fold.test)
    ic = EV.rank_ic(f, y, min_stocks=min_stocks)
    st = EV.ic_stats(ic)
    st.update(EV.segmented_ic(ic, 2))
    return {
        "ic_mean": float(st["ic_mean"]),
        "icir": float(st["icir"]) if np.isfinite(st["icir"]) else 0.0,
        "ic_win": float(st["ic_win"]) if np.isfinite(st["ic_win"]) else 0.0,
        "n_days": float(st["n_days"]),
    }
