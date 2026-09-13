# -*- coding: utf-8 -*-
"""概念数量因子（西部证券《因子手工作坊系列(7)：概念数量因子》落地）。

研报把"个股属于多少个概念"当成一类另类数据：
所属概念越多，说明这家公司被越多题材覆盖（关注度/资金容纳度），
越少的则越"稀缺"。围绕这个计数，衍生出一整族因子：

===========  ==================================================
因子          定义
===========  ==================================================
CN           概念数量（成员关系计数）
ACN          异常概念数量 = CN − 其历史均值（**新增**概念才是事件）
稀缺度        其所属概念的平均成员数的倒数（越"小众"越稀缺）
热度          其所属概念的成员近期收益均值（可加衰减）
IN / AIN     对规模（或行业）中性化后的 CN / ACN
DGTW         按市值分箱后组内标准化（剥离市值的非线性影响）
===========  ==================================================

**成员关系必须按区间生效，不能 ffill。** 概念成分有加入/剔除时点；用
"公告日 ffill" 的思路处理成分会立刻引入前后不一致（西部证券在报告里专门
强调了成分数据的时点问题）。本模块用差分数组（``+1/-1`` 后累加）精确统计
每个交易日每个标的的所属概念数，复杂度 O(成员关系数)。

概念热度定义为"成员股票近 ``heat_window`` 日收益的横截面均值"，是**可复算**
的量（不依赖外部情绪数据），因此不存在前视问题。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import expr as ex
from . import ops
from .panel import (DIM_COUNT, DIM_SCORE, ROLE_ALT, SEM_CONCEPT, SEM_SENTIMENT,
                    FieldMeta, FieldRegistry, PanelData)

__all__ = [
    "CONCEPT_FIELDS", "register_concept_fields", "synthetic_concepts",
    "install_concepts", "concept_membership_counts", "CONCEPT_FACTOR_LIBRARY",
    "concept_library_errors", "dgtw_adjust",
]

CONCEPT_FIELDS: Tuple[Tuple[str, str, str, str], ...] = (
    ("concept_count", DIM_COUNT, SEM_CONCEPT, "所属概念数量 CN（区间成分计数）"),
    ("concept_member_size", DIM_COUNT, SEM_CONCEPT,
     "所属概念的平均成员数（越小越稀缺）"),
    ("concept_heat", DIM_SCORE, SEM_SENTIMENT,
     "所属概念热度（成员近 N 日收益均值）的横截面均值"),
)


def register_concept_fields(registry: FieldRegistry) -> FieldRegistry:
    registry.register_fields(CONCEPT_FIELDS, source="concept", role=ROLE_ALT)
    return registry


# --------------------------------------------------------------------------
# 成员关系 → 面板计数（区间生效，差分数组，无 ffill）
# --------------------------------------------------------------------------
def _bounds(dates: pd.DatetimeIndex, start: Any, end: Any) -> Tuple[int, int]:
    """区间 → 交易日下标区间 ``[s, e)``；用 DatetimeIndex.searchsorted 避免
    numpy datetime64 单位不一致（us/ns）带来的静默错位。"""
    s = int(dates.searchsorted(pd.Timestamp(start), side="left"))
    e = int(dates.searchsorted(pd.Timestamp(end), side="right"))
    return max(s, 0), min(e, len(dates))


def _spans_of_concept(dates: pd.DatetimeIndex, grp: pd.DataFrame,
                      sym_pos: Dict[str, int],
                      last: pd.Timestamp) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for row in grp.itertuples():
        i = sym_pos.get(row.symbol)
        if i is None:
            continue
        end = row.end_date if pd.notna(row.end_date) else last
        s, e = _bounds(dates, row.start_date, end)
        if e > s:
            out.append((s, e, i))
    return out


def concept_membership_counts(membership: pd.DataFrame, panel: PanelData,
                              concept_weight: Optional[pd.Series] = None
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """区间成分关系 → (CN 面板, 所属概念平均成员数面板)。

    ``membership`` 需含 ``symbol`` / ``concept`` / ``start_date`` / ``end_date``；
    ``end_date`` 可为 NaT（表示仍在生效）。返回两个 ``(T, N)`` 的 numpy 数组，
    有意用 ndarray 而非 DataFrame：这是逐概念广播的中间结果，直接进面板会
    反复触发对齐开销。

    ``concept_weight`` 可选，按概念给不同权重（默认每条关系权重 1）——
    权重只在 CN 上生效，因为"数量"本就允许加权计数。
    """
    need = {"symbol", "concept", "start_date"}
    missing = need - set(membership.columns)
    if missing:
        raise ValueError(f"成分表缺少列: {sorted(missing)}")
    dates = panel.dates
    sym_pos = {s: i for i, s in enumerate(panel.symbols)}
    n_t, n_s = len(dates), len(panel.symbols)
    if "end_date" not in membership.columns:
        membership = membership.assign(end_date=pd.NaT)

    cnt = np.zeros((n_t, n_s), dtype=np.float64)
    member_cnt = np.zeros((n_t, n_s), dtype=np.float64)
    for concept, grp in membership.groupby("concept", sort=True):
        spans = _spans_of_concept(dates, grp, sym_pos, dates[-1])
        if not spans:
            continue
        # 该概念在每个交易日的成员数 m_c(t)：差分数组精确计数
        delta = np.zeros(n_t + 1, dtype=np.float64)
        for s, e, _ in spans:
            delta[s] += 1.0
            delta[e] -= 1.0
        m_c = np.cumsum(delta)[:n_t]
        w = 1.0 if concept_weight is None else float(
            concept_weight.get(concept, 1.0))
        for s, e, i in spans:
            member_cnt[s:e, i] += m_c[s:e]      # 所属概念当前有多少成员
            cnt[s:e, i] += w                    # 该股票当前属于多少个概念
    with np.errstate(invalid="ignore", divide="ignore"):
        avg_size = member_cnt / np.where(cnt > 0, cnt, np.nan)
    return cnt, np.where(np.isfinite(avg_size), avg_size, np.nan)


def install_concepts(panel: PanelData, membership: pd.DataFrame,
                     registry: Optional[FieldRegistry] = None,
                     heat_window: int = 20,
                     concept_weight: Optional[pd.Series] = None
                     ) -> Dict[str, pd.DataFrame]:
    """把概念成分关系装进面板：``concept_count`` / ``concept_member_size`` /
    ``concept_heat`` 三个 T 角色字段。"""
    reg = registry or panel.registry
    register_concept_fields(reg)
    cnt, avg_size = concept_membership_counts(membership, panel, concept_weight)
    dates, syms = panel.dates, panel.symbols
    out: Dict[str, pd.DataFrame] = {}

    out["concept_count"] = pd.DataFrame(cnt, index=dates, columns=syms)
    out["concept_member_size"] = pd.DataFrame(avg_size, index=dates, columns=syms)
    out["concept_heat"] = _concept_heat(membership, panel, heat_window)

    meta = {row[0]: row for row in CONCEPT_FIELDS}
    for name, frame in out.items():
        row = meta[name]
        panel.add_field(name, frame,
                        FieldMeta(name, row[1], row[2], ROLE_ALT,
                                  "concept", row[3]))
    return out


def _concept_heat(membership: pd.DataFrame, panel: PanelData,
                  heat_window: int = 20) -> pd.DataFrame:
    """个股概念热度 = 其所属各概念热度的均值；概念热度 = 该概念当前成员的
    近 ``heat_window`` 日收益均值。

    这里的口径必须是"**概念的**热度再回到个股"，而不是直接取个股自身收益：
    后者会退化成动量因子的复制品，热度这个 β 也就没有独立含义了。
    只用已实现收益，回看窗口，因此天然 PIT 安全。
    """
    ret = panel.field("ret")
    mom = ops.rolling_unary(ret, max(2, int(heat_window)), "mean")
    mom_np = mom.to_numpy(dtype=np.float64)
    mom_np = np.where(np.isfinite(mom_np), mom_np, np.nan)
    n_t, n_s = mom_np.shape
    total = np.zeros((n_t, n_s), dtype=np.float64)
    cnt = np.zeros((n_t, n_s), dtype=np.float64)
    sym_pos = {s: i for i, s in enumerate(panel.symbols)}
    for _, grp in membership.groupby("concept", sort=True):
        spans = _spans_of_concept(panel.dates, grp, sym_pos, panel.dates[-1])
        if not spans:
            continue
        mask = np.zeros((n_t, n_s), dtype=bool)
        for s, e, i in spans:
            mask[s:e, i] = True
        m = mask.sum(axis=1)                                # 概念成员数 m_c(t)
        ssum = np.nansum(np.where(mask, mom_np, np.nan), axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            heat_c = ssum / np.where(m > 0, m, np.nan)      # 概念热度
        for s, e, i in spans:
            total[s:e, i] += np.where(np.isfinite(heat_c[s:e]), heat_c[s:e], 0.0)
            cnt[s:e, i] += np.isfinite(heat_c[s:e]).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        heat = total / np.where(cnt > 0, cnt, np.nan)
    return pd.DataFrame(heat, index=panel.dates, columns=panel.symbols)


# --------------------------------------------------------------------------
# 合成概念成分（离线演示 / 测试）
# --------------------------------------------------------------------------
def synthetic_concepts(panel: PanelData, n_concepts: int = 40,
                       alpha: float = 1.2, niche_pref: float = 1.15,
                       seed: int = 13, churn: float = 0.03,
                       min_size: int = 3, max_size: int = 26) -> pd.DataFrame:
    """合成概念成分表（含概念诞生/退场时点）。

    同时埋两条可检验结构：

    1. **关注度**：成员概率与面板的潜在状态 ``mu`` 正相关 → 预期收益好的
       公司被更多概念覆盖 → ``CN`` 方向为**正**。
    2. **稀缺性**：概念规模越大，其成员的"质量权重"越低（``niche_pref``
       把大概念的质量系数压到 0 附近）→ 高质量公司更多落进小众概念 →
       ``稀缺度``（所属概念平均成员数的倒数）方向为**正**。

    另外保留"概念数量与市值正相关"这一机械关系（成员概率含规模项），
    用来验证 DGTW / 中性化是否真的把市值那部分剥离了。
    """
    rng = np.random.default_rng(seed)
    dates = panel.dates
    syms = panel.symbols
    n = len(syms)
    mu = panel.latent.get("mu") if panel.latent else None
    if mu is None:
        quality = rng.normal(0, 1, size=n)
    else:
        quality = np.nanmean(np.asarray(mu, dtype=np.float64), axis=0)
        quality = (quality - quality.mean()) / (quality.std() + 1e-12)
    amount = panel.fields.get("amount")
    size_z = (np.log(amount.mean(axis=0).to_numpy()) if amount is not None
              else np.zeros(n))
    size_z = (size_z - size_z.mean()) / (size_z.std() + 1e-12)

    rows: List[Dict[str, Any]] = []
    span = len(dates)
    for c in range(n_concepts):
        birth_frac = rng.uniform(0.0, 0.6)
        death_frac = rng.uniform(0.7, 1.2)
        b = int(birth_frac * span)
        d = min(int(death_frac * span), span - 1)
        if d <= b + 5:
            d = span - 1
        m_target = int(rng.integers(min_size, max_size + 1))
        coef = float(alpha) - float(niche_pref) * m_target / float(max_size)
        score = coef * quality + 0.35 * size_z
        prob = (1.0 / (1.0 + np.exp(-score))) * rng.uniform(0.85, 1.15)
        members = rng.random(n) < prob
        if members.sum() < min_size:
            members[np.argsort(-prob)[:min_size]] = True
        if members.sum() > m_target:      # 按概率裁剪到目标规模
            keep = np.argsort(-np.where(members, prob, -1.0))[:m_target]
            members = np.zeros(n, dtype=bool)
            members[keep] = True
        for i in np.flatnonzero(members):
            # 少数成员的加入/退出时点被随机提前/推后（churn）
            s = b + int(rng.integers(0, max(1, int(churn * span))))
            e = d - int(rng.integers(0, max(1, int(churn * span))))
            s = min(max(s, 0), span - 1)
            e = max(min(e, span - 1), s + 1)
            rows.append({"symbol": syms[i], "concept": f"C{c:03d}",
                         "start_date": dates[s], "end_date": dates[e]})
    return (pd.DataFrame(rows)
            .sort_values(["concept", "symbol"])
            .reset_index(drop=True))


# --------------------------------------------------------------------------
# 因子库
# --------------------------------------------------------------------------
CONCEPT_FACTOR_LIBRARY: Dict[str, str] = {
    # 原始计数
    "CN": "concept_count",
    "CN_rank": "rank_cs(concept_count)",
    "CN_z": "zscore_cs(concept_count)",
    # 异常概念数（新增概念才是事件）
    "ACN_60": "sub(concept_count, ts_mean(concept_count, 60))",
    "ACN_120": "sub(concept_count, ts_mean(concept_count, 120))",
    "ACN_z": "ts_zscore(concept_count, 60)",
    "ACN_change": "ts_delta(concept_count, 20)",
    # 稀缺度（所属概念平均成员数越少越稀缺）
    "稀缺度": "inv(concept_member_size)",
    "稀缺度_rank": "rank_cs(inv(concept_member_size))",
    # 热度及其衰减
    "热度": "zscore_cs(concept_heat)",
    "热度衰减": "zscore_cs(ts_decay_linear(concept_heat, 20))",
    "热度变化": "zscore_cs(ts_delta(concept_heat, 5))",
    # 中性化版本（IN / AIN）
    "CN_规模中性": "neutral(concept_count, size)",
    "ACN_规模中性": "neutral(ts_zscore(concept_count, 60), size)",
    "CN_波动中性": "neutral(concept_count, size, resid_vol)",
    # DGTW 市值分组调整
    "CN_DGTW5": "dgtw_cs(concept_count, size, 5)",
    "ACN_DGTW5": "dgtw_cs(sub(concept_count, ts_mean(concept_count, 60)), size, 5)",
    # 与其他信号组合
    "CN_x_热度": "mul(rank_cs(concept_count), rank_cs(concept_heat))",
    "CN_减_热度": "sub(rank_cs(concept_count), rank_cs(concept_heat))",
    "ACN_x_流动性": "mul(rank_cs(sub(concept_count, ts_mean(concept_count, 60))),"
                    " rank_cs(amount))",
}


def concept_library_errors(registry: FieldRegistry) -> Dict[str, str]:
    bad: Dict[str, str] = {}
    for name, text in CONCEPT_FACTOR_LIBRARY.items():
        try:
            node = ex.parse(text)
        except ex.ExprError as exc:
            bad[name] = f"解析失败: {exc}"
            continue
        errs = ex.validate(node, registry)
        if errs:
            bad[name] = "; ".join(errs)
    return bad


def dgtw_adjust(factor: pd.DataFrame, size: pd.DataFrame,
                n_groups: int = 5) -> pd.DataFrame:
    """DGTW 市值调整的**后处理**入口（因子已在面板外算好时使用）。"""
    return ops.cs_dgtw(factor, size, n_groups)
