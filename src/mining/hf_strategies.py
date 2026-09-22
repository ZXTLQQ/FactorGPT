"""四类高频策略信号 + 统一的信号评估器。

四种策略的经济逻辑完全不同，不要指望一个因子通吃：

1. **被动做市**（:func:`market_making_edge`）
   赚的是价差，风险是**逆向选择**：成交往往发生在价格即将对你不利变动时。
   因此做市信号的期望优势必须写成
   ``P(成交) × 预期价差收益 − P(成交) × 逆向选择损失``，
   只看成交概率会让策略变成"专门在最坏的时候成交"。

2. **跨期套利**（:func:`calendar_spread_signal`）
   同一品种不同月份合约的价差存在期限结构 mean reversion。
   注意这里做的是**统计套利**，价差偏离本身不代表定价错误，
   必须先用滚动窗口计算 z-score，不能直接用价差的绝对值。

3. **短周期趋势**（:func:`ofi_trend_signal`）
   实测证据最强的一路：OFI 分位第 10 档未来收益 +0.75 tick、
   胜率从第 1 档的 39.8% 升到 55.6%，且随预测周期衰减
   （IC 0.137 → 0.100 → 0.077，对应 5/10/20 tick），典型的信息优势短命特征。

4. **事件驱动**（:func:`event_signal`）
   成交量脉冲、持仓突变、报价抽空（spread 突然变宽）三类事件。
   这类信号的样本极少但幅度大，评估时**必须**看事件后的收益、
   而不是整体胜率，否则会被大量"没事件发生"的样本稀释成 0。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from data.hf_adapter import estimate_tick_size
from mining.hf import _rolling_stat, _to_timedelta

logger = logging.getLogger(__name__)


def _zscore(s: pd.Series, w: int, sid=None, min_periods: Optional[int] = None):
    """会话内滚动 z-score。跨会话计算均值/标准差会把午休两边的状态混在一起。"""
    mp = min_periods if min_periods is not None else max(20, w // 2)
    mu = pd.Series(_rolling_stat(s, w, "mean", sid, min_periods=mp), index=s.index)
    sd = pd.Series(_rolling_stat(s, w, "std", sid, min_periods=mp), index=s.index)
    return (s - mu) / sd.replace(0, np.nan)


# --------------------------------------------------------------------------
# 3. 短周期趋势
# --------------------------------------------------------------------------
def ofi_trend_signal(feat: pd.DataFrame, snap: pd.DataFrame,
                     smooth: int = 10, z_win: int = 200,
                     entry: float = 2.0) -> pd.Series:
    """OFI 平滑后取 z-score，超过阈值产生 ±1 持仓信号。

    OFI 的原始值非常尖峰（大部分时刻为 0，偶尔一个巨大的量），
    直接用原值会被少数噪声主导，因此先做短窗口平滑再标准化。
    """
    sid = snap["session_id"].to_numpy() if "session_id" in snap.columns else None
    ofi = pd.Series(_rolling_stat(feat["ofi"].to_numpy(dtype="float64"),
                                  smooth, "sum", sid, min_periods=smooth), index=feat.index)
    z = _zscore(ofi, z_win, sid)
    sig = pd.Series(0.0, index=feat.index)
    sig[z > entry] = 1.0
    sig[z < -entry] = -1.0
    sig[~np.isfinite(z)] = np.nan
    sig.attrs["z"] = z
    return sig


# --------------------------------------------------------------------------
# 4. 事件驱动
# --------------------------------------------------------------------------
def event_signal(snap: pd.DataFrame, tick: Optional[float] = None,
                 vol_z: float = 4.0, spread_z: float = 4.0,
                 oi_z: float = 4.0, z_win: int = 200) -> dict:
    """检测三类微观结构事件，返回 {事件名: 布尔序列}。

      · ``vol_burst``   成交量脉冲（相对于自身近期水平的极端放量）
      · ``spread_blowup``  报价抽空：做市商集体撤单导致价差瞬间放大
      · ``oi_shock``    持仓突变（增仓/减仓，往往伴随趋势启动）

    全部按 session 内 z-score 判定，阈值默认 4σ——事件驱动要的是稀有而可靠，
    阈值调到 2σ 会得到一堆假信号。
    """
    sid = snap["session_id"].to_numpy() if "session_id" in snap.columns else None
    out = {}
    if "d_volume" in snap.columns:
        dv = pd.Series(np.nan_to_num(snap["d_volume"].to_numpy(dtype="float64")), index=snap.index)
        out["vol_burst"] = _zscore(dv, z_win, sid) > vol_z
    tick = float(tick or estimate_tick_size(snap["bp1"].to_numpy()))
    sp = pd.Series((snap["sp1"].astype("float64") - snap["bp1"].astype("float64")) / tick,
                   index=snap.index)
    out["spread_blowup"] = _zscore(sp, z_win, sid) > spread_z
    if "d_open_int" in snap.columns:
        oi = pd.Series(np.nan_to_num(snap["d_open_int"].to_numpy(dtype="float64")), index=snap.index)
        zh = _zscore(oi, z_win, sid)
        out["oi_shock"] = zh.abs() > oi_z
        out["oi_dir"] = np.sign(zh)
    return out


# --------------------------------------------------------------------------
# 2. 跨期套利（统计套利）
# --------------------------------------------------------------------------
def calendar_spread_signal(term: pd.DataFrame, near: str, far: str,
                           w: int = 300, entry: float = 2.0,
                           exit_: float = 0.5) -> dict:
    """近月-远月价差的均值回归信号。

    返回带状态的字典：``spread`` / ``z`` / ``position``。
    持仓用**带滞回**的规则生成（entry 进场、exit_ 出场），
    否则会在阈值附近反复来回打脸，每次切换都要付一次手续费。
    """
    if near not in term.columns or far not in term.columns:
        return {"ok": False, "reason": f"缺合约 {near}/{far}"}
    spread = (term[near] - term[far]).dropna()
    if len(spread) < w + 10:
        return {"ok": False, "reason": f"样本不足 {len(spread)}"}
    z = (spread - spread.rolling(w, min_periods=w // 2).mean()) \
        / spread.rolling(w, min_periods=w // 2).std()
    pos = pd.Series(0.0, index=spread.index)
    cur = 0.0
    for i, (_, zv) in enumerate(z.items()):
        if not np.isfinite(zv):
            pos.iloc[i] = np.nan
            continue
        if cur == 0:
            if zv > entry:
                cur = -1.0        # 价差过宽 → 空近月多远月
            elif zv < -entry:
                cur = 1.0
        else:
            if abs(zv) <= exit_:
                cur = 0.0
        pos.iloc[i] = cur
    return {"ok": True, "spread": spread, "z": z, "position": pos}


# --------------------------------------------------------------------------
# 1. 被动做市：期望优势 = 成交概率 × (价差收益 − 逆向选择)
# --------------------------------------------------------------------------
def market_making_edge(p_fill: pd.Series, half_spread_ticks: pd.Series,
                       adverse_ticks: pd.Series) -> pd.Series:
    """挂单的期望净优势（tick 单位）。

    ``edge = P(fill) × [ 捕获的半价差 − 成交后的逆向选择损失 ]``

    ``adverse_ticks`` 用「成交后若干 tick 的价格不利变动」度量，
    实际使用时应由历史成交样本估计（见 :func:`estimate_adverse_selection`）。
    只最大化成交概率是做市最典型的死法：那样你会专门在价格即将穿透你的报价时成交。
    """
    return p_fill * (half_spread_ticks - adverse_ticks)


def estimate_adverse_selection(filled_orders: pd.DataFrame, snap: pd.DataFrame,
                               tick: Optional[float] = None,
                               horizon_ticks: int = 20) -> dict:
    """用**实际成交过的**委托估计逆向选择成本。

    对每一笔成交单，看成交之后价格往哪个方向走：
    买单成交后价格下跌 = 逆向损失；卖单成交后价格上涨 = 逆向损失。
    返回平均逆向损失（tick）与样本数——这个数字直接决定做市报价要多宽。
    """
    tick = float(tick or estimate_tick_size(snap["bp1"].to_numpy()))
    if not len(filled_orders):
        return {"ok": False, "reason": "无成交样本"}
    mid = pd.Series((snap["bp1"].astype("float64") + snap["sp1"].astype("float64")) / 2.0,
                    index=snap.index)
    fut = mid.shift(-horizon_ticks)
    o = filled_orders.sort_values("ts").reset_index(drop=True)
    # 取成交时刻之后最接近的快照：这里**必须**用 forward，测的就是成交后的价格
    grid = pd.DataFrame({"ts": snap["ts"].to_numpy(), "fut": fut.to_numpy()}).sort_values("ts")
    m = pd.merge_asof(o[["ts"]], grid, on="ts", direction="forward",
                      tolerance=_to_timedelta("5s"))
    now = pd.merge_asof(o[["ts"]], pd.DataFrame({"ts": snap["ts"].to_numpy(),
                                                 "mid": mid.to_numpy()}).sort_values("ts"),
                        on="ts", direction="backward", tolerance=_to_timedelta("5s"))
    move_ticks = (m["fut"] - now["mid"]) / tick
    side = o["side"].astype(float).to_numpy() if "side" in o.columns else np.ones(len(o))
    adverse = -(side * move_ticks.to_numpy())      # 对自己不利的变动取正值
    return {"ok": True, "mean_adverse_ticks": float(np.nanmean(adverse)),
            "median_adverse_ticks": float(np.nanmedian(adverse)),
            "n": int(np.isfinite(adverse).sum()),
            "share_positive": float(np.nanmean(adverse > 0))}


# --------------------------------------------------------------------------
# 统一评估器
# --------------------------------------------------------------------------
def evaluate_signal(signal: pd.Series, forward_ret_ticks: pd.Series,
                    cost_ticks: float = 1.0, hold_ticks: int = 10,
                    win: Optional[int] = None) -> dict:
    """评估任意信号在扣成本后的净表现，**按 holding 周期降采样**避免重复计数。

    这是高频回测最容易出错的地方：如果每个 tick 都按当时的信号"成交"一次，
    同一个 10 tick 的价格变动会被重复计入 10 次，收益率凭空放大一个数量级。
    这里按 ``hold_ticks`` 取样：每 hold_ticks 个 tick 只允许一次新的持仓决策。
    """
    s = pd.Series(signal).astype(float)
    y = pd.Series(forward_ret_ticks).astype(float)
    m = s.notna() & y.notna() & np.isfinite(s) & np.isfinite(y)
    s, y = s[m], y[m]
    take = np.arange(0, len(s), max(1, hold_ticks or 1))
    s, y = s.iloc[take], y.iloc[take]
    traded = s != 0
    if traded.sum() < 20:
        return {"ok": False, "reason": f"有效交易样本不足 ({int(traded.sum())})"}
    # 只在持仓方向**发生变化**时付手续费，持仓延续不重复扣费
    chg = (s.diff().fillna(s) != 0)
    pnl = s[traded] * y[traded] - cost_ticks * chg[traded].astype(float)
    n = len(pnl)
    mean = float(pnl.mean())
    sd = float(pnl.std(ddof=1)) if n > 1 else float("nan")
    return {
        "ok": True, "n_trades": int(n), "n_signal_ticks": int(traded.sum()),
        "mean_pnl_ticks": mean, "gross_pnl_ticks": float((s[traded] * y[traded]).mean()),
        "win_rate": float((pnl > 0).mean()),
        "sharpe_per_trade": float(mean / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan"),
        "total_pnl_ticks": float(pnl.sum()),
        "cost_ticks": cost_ticks,
    }


__all__ = [
    "calendar_spread_signal",
    "estimate_adverse_selection",
    "evaluate_signal",
    "event_signal",
    "market_making_edge",
    "ofi_trend_signal",
]
