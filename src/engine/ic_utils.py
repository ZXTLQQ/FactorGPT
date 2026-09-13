"""截面 IC 计算内核（src/engine/ic_utils.py）。

把「因子值 × 未来收益」的截面相关计算统一成一处向量化实现，供以下模块共用：

* :mod:`engine.param_ops`    —— 参数化时序算子的内层参数拟合；
* :mod:`engine.multiscale_gp` —— 分层多尺度遗传挖掘的粗/细尺度适应度；
* :mod:`engine.significance` —— 平稳块 bootstrap 的输入 IC 序列；
* :mod:`engine.universe`     —— 选股域前后对比。

全部为纯计算函数：不读配置、不碰网络、不依赖 Streamlit。NaN 语义上，调用方
无需预先 dropna —— 非有限值会被排除在均值/协方差之外（等效于按有效样本重算），
而不是被当成 0 参与计算（后者会把"缺失"静默变成"中性"，是最危险的失真假象）。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def groupwise_corr(
    f: np.ndarray,
    y: np.ndarray,
    codes: np.ndarray,
    n_groups: int,
    min_count: int = 2,
) -> np.ndarray:
    """按组（截面）计算 Pearson 相关，一次 ``bincount`` 聚合完成。

    Args:
        f: 因子值。
        y: 同期（前瞻）收益。
        codes: 组编码，``0..n_groups-1`` 的整数（通常由 ``pd.factorize(date)`` 得到）。
        n_groups: 组数。
        min_count: 组内有效样本数下限，低于该值的组返回 NaN。

    Returns:
        长度 ``n_groups`` 的相关系数数组；组内样本不足或方差为 0 时该组为 NaN。
    """
    f = np.asarray(f, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    codes = np.asarray(codes, dtype=np.int64).reshape(-1)
    if not (f.size == y.size == codes.size):
        raise ValueError("f / y / codes 长度必须一致")
    if n_groups <= 0 or f.size == 0:
        return np.full(max(n_groups, 0), np.nan, dtype=float)
    if codes.min() < 0:
        raise ValueError("codes 必须为非负整数")

    # 非有限值不参与任何统计量：对和的贡献置 0，同时不计入有效样本数。
    valid = np.isfinite(f) & np.isfinite(y)
    fv = np.where(valid, f, 0.0)
    yv = np.where(valid, y, 0.0)

    def _acc(w: np.ndarray) -> np.ndarray:
        return np.bincount(codes, weights=w, minlength=n_groups)[:n_groups]

    cnt = _acc(valid.astype(float))
    safe = np.maximum(cnt, 1.0)
    mf = _acc(fv) / safe
    my = _acc(yv) / safe
    df_ = np.where(valid, fv - mf[codes], 0.0)
    dy_ = np.where(valid, yv - my[codes], 0.0)
    cov = _acc(df_ * dy_)
    vx = _acc(df_ * df_)
    vy = _acc(dy_ * dy_)

    with np.errstate(invalid="ignore", divide="ignore"):
        den = np.sqrt(vx * vy)
        out = cov / den
    return np.where((den > 0) & (cnt >= min_count) & np.isfinite(out), out, np.nan)


def panel_ic(
    f: np.ndarray,
    y: np.ndarray,
    group_key: np.ndarray,
    min_count: int = 5,
) -> pd.Series:
    """把面板数据折算成 IC 时间序列。

    Args:
        f: 因子值（与 ``group_key`` 同序同长）。
        y: 前瞻收益。
        group_key: 截面分组键（通常为日期），任意可比较类型。
        min_count: 每个截面最少有效样本数。

    Returns:
        以 ``group_key`` 的排序唯一值为索引、IC 为值的 Series（时间升序）。
    """
    f = np.asarray(f, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    key = np.asarray(group_key).reshape(-1)
    codes, uniq = pd.factorize(key, sort=True)
    ic = groupwise_corr(f, y, codes, len(uniq), min_count=min_count)
    return pd.Series(ic, index=pd.Index(uniq, name="date"), name="ic", dtype=float)


def ic_stats(ic: Any) -> Dict[str, float]:
    """IC 序列的汇总统计量。

    Returns:
        ``{"ic", "ic_std", "icir", "t_stat", "positive_ratio", "n"}``。

        ``icir = mean/std``（不做年化，与 :class:`engine.backtest.FactorBacktester`
        的口径一致）；``t_stat = icir * sqrt(n)`` 用于粗略显著性判断。
        样本不足（``n == 0`` 或 std == 0）时返回 NaN，而不是 0 —— 0 会被下游
        误读成"不显著"，而 NaN 会被显式过滤掉。
    """
    s = pd.Series(ic, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    n = int(s.size)
    nan = float("nan")
    if n == 0:
        return {"ic": nan, "ic_std": nan, "icir": nan, "t_stat": nan,
                "positive_ratio": nan, "n": 0}
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if n > 1 else nan
    if not np.isfinite(std) or std <= 0:
        return {"ic": mean, "ic_std": nan, "icir": nan, "t_stat": nan,
                "positive_ratio": float((s > 0).mean()), "n": n}
    icir = mean / std
    return {
        "ic": mean,
        "ic_std": std,
        "icir": icir,
        "t_stat": icir * float(np.sqrt(n)),
        "positive_ratio": float((s > 0).mean()),
        "n": n,
    }


def period_codes(dates: Any, freq: str = "M") -> Optional[np.ndarray]:
    """把时间轴按粒度折叠成块，返回与输入等长的块序号（非法 ``freq`` 返回 None）。

    实现上用 ``DatetimeIndex.to_period`` + ``factorize``，**不用** ``resample``：
    当前 pandas/numpy 组合下 ``resample`` 会同时抛出两条告警（``'M' is deprecated``
    与 ``generic unit for NumPy timedelta is deprecated``），而项目 ``pytest.ini``
    设的是 ``filterwarnings = error`` —— 一次 resample 就足以把整条测试链判失败。
    """
    arr = np.asarray(dates).reshape(-1)
    if arr.size == 0:
        return np.array([], dtype=np.int64)
    try:
        per = pd.DatetimeIndex(pd.to_datetime(arr)).to_period(str(freq))
    except (ValueError, TypeError):
        return None
    codes, _ = pd.factorize(per, sort=True)
    return np.asarray(codes, dtype=np.int64)


def block_last_positions(codes: np.ndarray) -> np.ndarray:
    """每个块内**最后一个**位置（按时间升序，输入需已按时间排序）。

    这是"粗网格点"的取法：块内末截面包含该块全部历史信息，取块首会把区间
    开端的信息量算低。后写覆盖前写，因此 ``last[c] = arange`` 天然得到末位置。
    """
    c = np.asarray(codes, dtype=np.int64).reshape(-1)
    if c.size == 0:
        return np.array([], dtype=int)
    if c.min() < 0:
        raise ValueError("codes 必须为非负整数")
    last = np.full(int(c.max()) + 1, -1, dtype=int)
    last[c] = np.arange(c.size, dtype=int)
    return last[last >= 0]


def block_mean(ic: pd.Series, freq: str = "M") -> pd.Series:
    """把逐日 IC 序列按时间粒度（``freq``）聚合成块均值。

    这是多尺度挖掘里"粗尺度"评估的实现：把高频 IC 剖面压缩成低频块，
    块内均值代表该尺度下该区间的信号水平。索引取块内最后一个日期，
    与 ``resample(...).mean()`` 的"期末标签"口径一致。
    """
    if ic is None or len(ic) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(ic, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(pd.Index(s.index), errors="coerce")
    if idx.isna().any():
        # 非时间索引（如整数区间）时不做聚合，直接返回原序列
        return s
    codes = period_codes(idx.to_numpy(), freq)
    if codes is None:
        return pd.Series(s.to_numpy(), index=idx).dropna()
    out = pd.Series(s.to_numpy(), index=idx).groupby(codes).mean().dropna()
    if out.empty:
        return pd.Series(dtype=float)
    last = block_last_positions(codes)
    out.index = pd.DatetimeIndex(idx.to_numpy()[last])
    return out


def downside_ratio(ic: Any) -> float:
    """IC 为负的比例（衡量信号方向的可持续性，越低越好）。"""
    s = pd.Series(ic, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return float("nan")
    return float((s < 0).mean())


def split_windows(dates: np.ndarray, n_windows: int) -> list:
    """把时间轴等分成 ``n_windows`` 段，返回 ``[(start, end), ...]``（含端点）。"""
    uniq = np.sort(pd.unique(np.asarray(dates)))
    if uniq.size == 0 or n_windows <= 0:
        return []
    if n_windows == 1:
        return [(uniq[0], uniq[-1])]
    edges = np.linspace(0, uniq.size, n_windows + 1).astype(int)
    out = []
    for i in range(n_windows):
        lo, hi = edges[i], edges[i + 1] - 1
        if lo > hi:
            continue
        out.append((uniq[lo], uniq[hi]))
    return out
