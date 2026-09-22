"""高频 L2 订单簿因子（``hf`` 家族）——把五档快照翻译成有预测力的特征。

前置：特征由 :class:`data.hf_adapter.HFDataSource` 产出，**必须**已经处理好
会话边界（``session_id`` / ``new_session``）与时间还原（``ts``）。本模块不再重复处理，
但会在计算标签时**再次**校验边界，因为这是最容易出错、后果最严重的一步。

方法论上的两个诚实声明
----------------------
1. **这份数据是 500ms 快照，不是逐笔。** 经典 OFI（Cont, Kukanov, Stoikov 2014）
   定义在事件流上：每一次下单/撤单/成交都是一个观测。这里两个快照之间发生的事情
   被压缩成一个净变化 ΔQ，**无法区分**它是成交、撤单还是新挂单。
   因此 :func:`queue_change_ofi` 计算的是「队列净变化」，不是严格意义的 OFI，
   实证研究里它能用，但不要把它当成 OFI 原义去写论文。

2. **快照的时间间隔并不均匀。** 实测 au2608 的 ``dt_ms`` 中位数 500ms，
   但最大到了 900500ms（15 分钟）——远月合约长时间没有新快照。
   所以标签一律按 **tick 数**（未来若干个快照）而不是按秒定义，
   否则「未来 5 秒」在不同时段会横跨完全不同的市场状态。

因子列表与用途
--------------
对外暴露三类能力：

- :func:`build_l2_features` / :func:`make_forward_labels`：**单合约**特征矩阵与未来标签；
- :func:`register_hf_fields` / :func:`install_hf_features`：把这些字段接入
  统一挖掘框架的 :class:`~mining.panel.FieldRegistry` 与 :class:`~mining.panel.PanelData`，
  从而能和日频量价、基本面、另类文本在同一个表达式树里做跨域组合；
- :mod:`mining.hf_models` / :mod:`mining.hf_strategies`：订单簿信号的监督建模与四类策略回测。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from data.hf_adapter import estimate_tick_size

logger = logging.getLogger(__name__)

_LEVEL = 5


def _levels(df: pd.DataFrame, side: str, tick: float) -> tuple:
    """取出某一侧的价/量矩阵 ``(N, 5)``。

    远端档位经常没人挂，价格与量都是 NaN。这里**按档位独立**处理：
    缺失的档位量视为 0（等价于「该价位无人挂单」），而不是整行丢弃——
    丢行会让样本被「是否有远档报价」这个无关事实筛选，引入隐性偏差。
    """
    p = np.column_stack([df.get(f"{side}p{i}", pd.Series(np.nan, index=df.index)).to_numpy(dtype="float64")
                         for i in range(1, _LEVEL + 1)])
    q = np.column_stack([df.get(f"{side}v{i}", pd.Series(np.nan, index=df.index)).to_numpy(dtype="float64")
                         for i in range(1, _LEVEL + 1)])
    q = np.nan_to_num(q, nan=0.0)
    key = np.where(np.isfinite(p), np.round(p / tick), np.nan)
    return p, q, key


def queue_change_ofi(df: pd.DataFrame, tick: Optional[float] = None) -> pd.DataFrame:
    """多档「队列净变化」（Q-change / 快照 OFI）。

    对每个价位 key = round(price / tick)：若该价位在上一快照已存在，取其当时的量作基准；
    否则基准为 0（表示新增报价）。买卖两侧分别汇总后相减：

        ofi_bid = Σ_p [ q_bid(p, t) - q_bid(p, t-1) ]
        ofi_ask = Σ_p [ q_ask(p, t) - q_ask(p, t-1) ]
        ofi     = ofi_bid - ofi_ask

    经济含义：买方队列增厚、卖方队列变薄都代表买压，故取 ofi_bid - ofi_ask。

    为什么按**价格**匹配而不是按档位序号匹配：价格变动时，第 i 档会被整体平移，
    按序号比较会把「价位 A 的量」错配到「价位 B」。按价格 key 匹配天然免疫这一点。

    会话边界：``new_session`` 为真的行，上一快照不存在（或相隔数小时），
    ΔQ 一律置 NaN——否则午休两小时的队列变化会变成一个巨大的假信号。
    """
    tick = float(tick or estimate_tick_size(df["bp1"].to_numpy()))
    n = len(df)
    out = {}
    for side in ("b", "s"):
        p, q, key = _levels(df, side, tick)
        prev_key = np.vstack([np.full((1, _LEVEL), np.nan), key[:-1]])
        prev_q = np.vstack([np.zeros((1, _LEVEL)), q[:-1]])
        # 逐个价位：当前第 i 档价格，在上一快照的五档里能否找到同样的价格
        per_level = np.zeros((n, _LEVEL))
        for i in range(_LEVEL):
            cur_k = key[:, i]
            cur_q = q[:, i]
            match = prev_key == cur_k[:, None]
            hit = match.any(axis=1)
            idx = np.argmax(match, axis=1)
            base = np.where(hit, np.take_along_axis(prev_q, idx[:, None], axis=1).ravel(), 0.0)
            # 价格为 NaN 的档位本身就是「没有这一档」，量变化应视为 0（而非 NaN）
            valid = np.isfinite(cur_k)
            per_level[:, i] = np.where(valid, cur_q - base, 0.0)
        # 会话边界切断
        if "new_session" in df.columns:
            ns = df["new_session"].to_numpy()
            per_level[ns] = np.nan
        out[f"dq_{side}"] = per_level.sum(axis=1)
        for i in range(_LEVEL):
            out[f"dq_{side}{i+1}"] = per_level[:, i]
    res = pd.DataFrame(out, index=df.index)
    res["ofi"] = res["dq_b"] - res["dq_s"]
    return res


def build_l2_features(df: pd.DataFrame, tick: Optional[float] = None,
                      include_ofi: bool = True) -> pd.DataFrame:
    """从单合约快照序列构造高频因子矩阵。

    返回的每一列都保证：**第 t 行只用到第 t 个及之前快照的信息**。
    滚动窗口默认是 ``min_periods`` 完整的过去窗口，因此不包含未来。

    NaN 语义（很重要）：跨会话的第一行、以及各种「需要历史」的指标的起始若干行
    会是 NaN。下游做 IC / 建模时应**显式丢弃**这些行，而不是用 0 填充——
    填充 0 会让「刚开盘」被当成「最均衡的状态」，这是假的。
    """
    tick = float(tick or estimate_tick_size(df["bp1"].to_numpy()))
    f = pd.DataFrame(index=df.index)
    bid_p, bid_q, _ = _levels(df, "b", tick)
    ask_p, ask_q, _ = _levels(df, "s", tick)

    bp1, sp1 = bid_p[:, 0], ask_p[:, 0]
    bv1, sv1 = bid_q[:, 0], ask_q[:, 0]
    mid = (bp1 + sp1) / 2.0
    spread = sp1 - bp1

    # ---- 价格/价差 ----
    f["mid"] = mid
    f["spread"] = spread
    f["spread_ticks"] = spread / tick                       # 以 tick 为单位，跨品种可比
    f["rel_spread"] = np.where(mid > 0, spread / mid, np.nan)
    f["log_spread"] = np.log(np.where(spread > 0, spread, np.nan))

    # ---- 微观价格：以最优档对手力量加权的中价 ----
    tot1 = bv1 + sv1
    micro = np.where(tot1 > 0, (bp1 * sv1 + sp1 * bv1) / np.where(tot1 == 0, np.nan, tot1), np.nan)
    f["microprice"] = micro
    f["micro_dev_ticks"] = (micro - mid) / tick             # 微观价格偏离中价几个 tick

    # ---- 多档加权中价 / 失衡 ----
    decay = np.array([0.5 ** i for i in range(_LEVEL)])     # 越远的档信息含量越低
    bid_dep = bid_q.sum(axis=1)
    ask_dep = ask_q.sum(axis=1)
    f["bid_depth"] = bid_dep
    f["ask_depth"] = ask_dep
    f["depth_ratio"] = np.where(ask_dep > 0, bid_dep / np.where(ask_dep == 0, np.nan, ask_dep), np.nan)
    f["depth_sum"] = bid_dep + ask_dep
    f["obi_l1"] = _safe_ratio(bv1 - sv1, bv1 + sv1)
    f["obi_all"] = _safe_ratio(bid_dep - ask_dep, bid_dep + ask_dep)
    # 逐档失衡的指数衰减加权
    for L in (2, 3, 5):
        num = np.zeros(len(df))
        den = np.zeros(len(df))
        for i in range(L):
            num += decay[i] * (bid_q[:, i] - ask_q[:, i])
            den += decay[i] * (bid_q[:, i] + ask_q[:, i])
        f[f"obi_w{L}"] = _safe_ratio(num, den)

    # ---- 挂单结构的形状 ----
    f["best_share_bid"] = _safe_ratio(bv1, bid_dep)         # 最优档占比：越大越集中在盘口
    f["best_share_ask"] = _safe_ratio(sv1, ask_dep)
    f["l1_over_l5_bid"] = _safe_ratio(bv1, np.maximum(bid_q[:, _LEVEL - 1], 1e-12))
    # 盘口倾斜度：远端价差相对近端价差（衡量簿的形状是凸还是凹）
    f["book_slope"] = _safe_ratio(
        (np.nanmax(np.where(np.isfinite(ask_p), ask_p, np.nan), axis=1) - bp1), spread)

    # ---- 全市场买卖总量（区别于五档，含五档之外的委托） ----
    for a, b, name in (("total_bid_vol", "total_ask_vol", "obi_total"),
                       ("avg_bid_price", "avg_ask_price", "avg_spread")):
        if a in df.columns and b in df.columns:
            av = df[a].to_numpy(dtype="float64")
            bv = df[b].to_numpy(dtype="float64")
            if name == "obi_total":
                f[name] = _safe_ratio(av - bv, av + bv)
            else:
                f[name] = (bv - av) / tick

    # ---- 订单流：队列净变化 OFI ----
    if include_ofi:
        try:
            ofi = queue_change_ofi(df, tick=tick)
            f = pd.concat([f, ofi], axis=1)
        except Exception as e:  # noqa: BLE001
            logger.warning("[HF] OFI 计算失败，跳过该类因子: %s", e)

    # ---- 成交 / 持仓 / 波动 ----
    sid = df["session_id"].to_numpy() if "session_id" in df.columns else None
    if "d_volume" in df.columns:
        dv = df["d_volume"].to_numpy(dtype="float64")
        f["d_volume"] = dv
        f["trade_flag"] = (dv > 0).astype(float)
        f["volume_impulse_20"] = _rolling_stat(dv, 20, "sum", sid)
        f["trade_rate_20"] = _rolling_stat((dv > 0).astype(float), 20, "mean", sid)
    if "d_turnover" in df.columns:
        f["turnover_20"] = _rolling_stat(df["d_turnover"].to_numpy(dtype="float64"), 20, "sum", sid)
    if "d_open_int" in df.columns:
        f["oi_change"] = df["d_open_int"]
        f["oi_change_20"] = _rolling_stat(df["d_open_int"].to_numpy(dtype="float64"), 20, "sum", sid)

    ret = np.diff(mid, prepend=np.nan) / tick               # 以 tick 为单位的中价变动
    ret[np.isinf(ret)] = np.nan
    f["ret_tick"] = ret
    f["mid_move_ticks"] = ret
    # 已实现波动（过去 20 tick）：波动率是几乎所有高频策略的核心调节变量
    f["rvol_20"] = _rolling_stat(ret, 20, "std", sid, min_periods=15)
    f["rvol_100"] = _rolling_stat(ret, 100, "std", sid, min_periods=50)
    f["trend_20"] = _rolling_stat(ret, 20, "sum", sid)      # 短周期趋势：过去各 tick 变动之和
    f["reversal_5"] = -_rolling_stat(ret, 5, "sum", sid)    # 反转因子

    # 用最后一笔成交价的 tick rule 推断主动方向（无逐笔时的次优解）
    if "last_price" in df.columns:
        lp = df["last_price"].to_numpy(dtype="float64")
        dl = np.diff(lp, prepend=np.nan)
        sign = np.sign(dl)
        f["trade_sign"] = sign
        if "d_volume" in df.columns:
            f["signed_flow_20"] = _rolling_stat(sign * np.nan_to_num(dv), 20, "sum", sid)

    f.attrs["tick_size"] = tick
    return f


def make_forward_labels(df: pd.DataFrame, horizon_ticks: int = 10,
                        tick: Optional[float] = None,
                        kind: str = "ret") -> pd.Series:
    """构造**未来**标签：``horizon_ticks`` 个快照之后的中价收益。

    ``horizon_ticks=10`` 在 500ms 快照下约为未来 5 秒。之所以用 tick 数而非秒，
    是因为快照间隔本身不均匀（远月合约可能出现长达 15 分钟的空档）。

    防泄漏的三道保险：
      1. 用 ``shift(-h)`` 取未来价格——**只在这里出现负数 shift**，别处不得再有；
      2. 跨越会话边界的样本置 NaN（否则午休前后的收益会变成一个既非 True 也非 False 的噪声标签）；
      3. 中间存在任一 new_session 的样本也置 NaN。

    kind:
      · ``ret``   未来中价收益 / tick（连续值，用于回归 / IC）
      · ``dir``   ±1 方向 + 0 表示几乎没动（用 0.5 tick 的死区过滤噪声）
      · ``ternary``  经典三分标签：涨 / 跌 / 平
    """
    tick = float(tick or estimate_tick_size(df["bp1"].to_numpy()))
    mid = pd.Series((df["bp1"].to_numpy(dtype="float64")
                     + df["sp1"].to_numpy(dtype="float64")) / 2.0, index=df.index)
    fut = mid.shift(-horizon_ticks)
    ret = (fut - mid) / tick

    # 保险 2/3：t 与 t+h 必须落在同一会话，且中途不能跨会话
    if "session_id" in df.columns:
        sid = df["session_id"].to_numpy()
        same = np.zeros(len(df), dtype=bool)
        fut_sid = np.full(len(df), -1)
        valid = np.arange(len(df)) + horizon_ticks < len(df)
        fut_sid[valid] = sid[np.arange(len(df))[valid] + horizon_ticks]
        same = fut_sid == sid
        # 中途不得出现 new_session
        if "new_session" in df.columns:
            ns = df["new_session"].to_numpy().astype(float)
            cs = np.cumsum(ns)
            fut_cs = np.full(len(df), np.nan)
            fut_cs[valid] = cs[np.arange(len(df))[valid] + horizon_ticks]
            gap_free = (fut_cs - cs) == 0
            same = same & gap_free
        ret = ret.where(same)
    else:
        logger.warning("[HF] 输入缺少 session_id，无法做会话边界防护，标签可能跨午休。")

    if kind == "dir":
        return np.sign(ret).where(np.abs(ret) >= 0.5, 0.0)
    if kind == "ternary":
        out = pd.Series(np.nan, index=df.index)
        out[ret > 0.5] = 1.0
        out[ret < -0.5] = -1.0
        out[np.abs(ret) <= 0.5] = 0.0
        return out
    return ret


def evaluate_factor_ic(features: pd.DataFrame, label: pd.Series,
                       method: str = "spearman") -> pd.DataFrame:
    """逐因子算 IC（截面/时间序列版）与显著性，返回排序后的表。

    这里是**时间序列 IC**：同一合约内，因子值与未来收益的相关。
    因子挖掘阶段用它快速判断哪些因子在这份数据上真的有信息，
    避免把一堆没人看的因子塞进 install()。
    """
    rows = []
    y = pd.Series(label).astype("float64")
    for col in features.columns:
        x = pd.to_numeric(features[col], errors="coerce")
        m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        n = int(m.sum())
        if n < 200:
            continue
        xv, yv = x[m].to_numpy(), y[m].to_numpy()
        if np.std(xv) == 0 or np.std(yv) == 0:
            continue
        try:
            ic = pd.Series(xv).corr(pd.Series(yv), method=method)
        except Exception:  # noqa: BLE001
            continue
        if pd.isna(ic):
            continue
        # t 统计量：IC * sqrt(n-2) / sqrt(1-IC^2)
        t = ic * np.sqrt(n - 2) / np.sqrt(max(1e-12, 1 - ic ** 2))
        rows.append({"factor": col, "ic": float(ic), "abs_ic": abs(float(ic)),
                     "t_stat": float(t), "n": n,
                     "p_hint": "|t|>2" if abs(t) > 2 else ""})
    res = pd.DataFrame(rows)
    if len(res):
        res = res.sort_values("abs_ic", ascending=False).reset_index(drop=True)
    return res


# ---------- 辅助 ----------
def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    den = np.asarray(den, dtype="float64")
    num = np.asarray(num, dtype="float64")
    out = np.where(np.abs(den) > 1e-12, num / np.where(np.abs(den) <= 1e-12, 1.0, den), np.nan)
    return np.where(np.isfinite(out), out, np.nan)


def _rolling_stat(values, w: int, stat: str = "sum", sid=None,
                  min_periods: Optional[int] = None):
    """会话内滚动统计。**滚动窗口绝对不能跨会话**。

    「不能跨会话」有两层意思：

    1. 不是未来泄漏——它用的是过去数据；
    2. 但是跨会话会把午休前后的两小时塞进同一个 20 tick 窗口，
       午休前后的市场状态毫不相干，混在一起会让波动率、成交量脉冲这类因子失真。
       实测本数据午休间隔可达 2 小时，而正常快照间隔只有 500ms，
       这个 4 个数量级的差异足以让所有「短期」因子失去意义。

    因此这里统一按 ``session_id`` 分组后再滚动。
    """
    s = pd.Series(values)
    if s.dtype.kind == "f":
        s = s.replace([np.inf, -np.inf], np.nan)
    mp = min_periods if min_periods is not None else w
    if sid is None:
        r = getattr(s.rolling(w, min_periods=mp), stat)()
    else:
        r = getattr(s.groupby(np.asarray(sid)).rolling(w, min_periods=mp), stat)()
        r = r.reset_index(level=0, drop=True).reindex(s.index)
    return r.to_numpy() if stat != "std" else np.asarray(r)


def register_hf_fields(registry=None):
    """把「撮合前」（pre-trade）的高频订单簿字段注册进 :class:`FieldRegistry`。

    这里的 ``role`` 用 ``ROLE_ALT`` 而不是 ``ROLE_MARKET``：它不是 K 线派生出来的
    量价信息，而是**订单簿层面的另类信息**，和日频量价属于不同信息源，
    便于在统一框架里做「撮合前 ⊕ 撮合后」的双源融合。

    ``dimension`` 决定它能否进表达式树参与无量纲运算（详见 panel.FieldMeta），
    例如 ``ofi`` 是「手数量纲」、不能直接和价格做差，这是静态类型检查能拦住的错。
    """
    from .panel import (
        DIM_COUNT,
        DIM_FLAG,
        DIM_PRICE,
        DIM_RATIO,
        DIM_VOLUME,
        ROLE_ALT,
        SEM_LIQUIDITY,
        SEM_MOMENTUM,
        SEM_RISK,
        SEM_SENTIMENT,
        SEM_VALUE,
    )

    reg = registry if registry is not None else _default_registry()
    reg.register_fields([
        # -- 五档原始量价 --
        ("bp1", DIM_PRICE, SEM_VALUE, "买一价（最优买价）"),
        ("sp1", DIM_PRICE, SEM_VALUE, "卖一价（最优卖价）"),
        ("bv1", DIM_VOLUME, SEM_LIQUIDITY, "买一挂单量（手）"),
        ("sv1", DIM_VOLUME, SEM_LIQUIDITY, "卖一挂单量（手）"),
        ("bid_depth", DIM_VOLUME, SEM_LIQUIDITY, "五档买方挂单总量"),
        ("ask_depth", DIM_VOLUME, SEM_LIQUIDITY, "五档卖方挂单总量"),
        # -- 衍生形态 --
        ("mid", DIM_PRICE, SEM_VALUE, "微观中价 =（买一+卖一）/2"),
        ("microprice", DIM_PRICE, SEM_VALUE, "按最优档量加权的价格（微观价格）"),
        ("spread_ticks", DIM_COUNT, SEM_LIQUIDITY, "报价价差，tick 单位"),
        ("obi_l1", DIM_RATIO, SEM_SENTIMENT, "最优档买卖失衡 (bv1-sv1)/(bv1+sv1)"),
        ("obi_w5", DIM_RATIO, SEM_SENTIMENT, "五档量加权买卖失衡"),
        ("micro_dev_ticks", DIM_RATIO, SEM_SENTIMENT, "微观价格偏离中价，tick 单位"),
        ("ofi", DIM_VOLUME, SEM_SENTIMENT, "队列净变化（快照 OFI），手"),
        ("rvol_20", DIM_RATIO, SEM_RISK, "20 tick 已实现波动（会话内，中价收益标准差）"),
        ("trend_20", DIM_RATIO, SEM_MOMENTUM, "20 tick 中价动量"),
        ("reversal_5", DIM_RATIO, SEM_MOMENTUM, "5 tick 中价反转"),
        ("signed_flow_20", DIM_VOLUME, SEM_MOMENTUM, "20 tick 主动方向净成交量"),
        ("queue_decay", DIM_RATIO, SEM_LIQUIDITY, "队列深度衰减：远档/近档（>1 表示远档更厚）"),
        ("large_wall", DIM_FLAG, SEM_LIQUIDITY, "是否存在显著大单墙（0/1）"),
        ("ofi_vs_wall", DIM_VOLUME, SEM_SENTIMENT, "OFI 与大单墙的联合信号：单边挂大单时的 OFI 才有效"),
    ], source="hf", role=ROLE_ALT)
    return reg


def _default_registry():
    from .panel import FieldRegistry
    return FieldRegistry()


def _to_timedelta(value) -> pd.Timedelta:
    """统一构造 timedelta。

    pandas 2.3 + numpy 2.5 的组合下 ``pd.Timedelta("600s")`` / ``Timedelta(seconds=600)``
    都会抛 ``DeprecationWarning: The 'generic' unit for NumPy timedelta is deprecated``
    （后续 numpy 版本会直接报错），而本项目把告警一律当错误（见 ``pytest.ini``）。
    因此走 ``Timedelta(0) + to_offset(...)`` 这条不掉坑的路径，
    与 ``panel.day_delta``（用 ``np.timedelta64``）同源同理。
    """
    if isinstance(value, pd.Timedelta):
        return value
    return pd.Timedelta(0) + to_offset(value)


def install_hf_features(panel, long_df, fields=None, tolerance="600s",
                        registry=None) -> list:
    """把 (ts, symbol, 因子...) 长表按 **前向填充到 bar 时刻** 装进面板。

    为什么必须前向填充而不是 join：**高频因子是状态量**（某一时刻订单簿的失衡），
    bar 之间缺失时应当沿用最近一次观测；但如果缺失超过 ``tolerance``
    （默认 10 分钟，远大于 500ms 快照间隔，足以判定是会话断裂/停盘），
    就置 NaN——否则午休的旧快照会被一路填到下午开盘。

    ``long_df`` 需含 ``ts`` 与 ``symbol`` 两列，其余数值列即高频因子。
    """
    names = list(fields) if fields else [
        c for c in long_df.columns
        if c not in ("ts", "symbol") and pd.api.types.is_numeric_dtype(long_df[c])]
    reg = registry or panel.registry
    register_hf_fields(reg)
    installed = []
    dates = pd.DatetimeIndex(pd.to_datetime(list(panel.dates)))
    tol = _to_timedelta(tolerance)
    src = long_df.copy()
    src["ts"] = pd.to_datetime(src["ts"])
    for name in names:
        if name not in src.columns:
            continue
        wide = src.pivot_table(index="ts", columns="symbol",
                               values=name, aggfunc="last").sort_index()
        if wide.shape[1] == 0:
            continue
        idx = dates
        uni = wide.index.union(idx)
        full = wide.reindex(uni).ffill().reindex(idx)
        # 超出耐受窗口视为陈旧：先求「每个 bar 时刻最后一次**真实观测**的时刻」，
        # 再用它与 bar 时刻的时间差做掩蔽。注意这里必须逐列求最后观测时刻——
        # 用 union 索引当观测时刻会退化成「永远不陈旧」（bar 时刻本身一定在并集里）。
        seen = wide.notna().reindex(uni, fill_value=False)
        obs_t = pd.DataFrame(
            {c: pd.Series(uni, index=uni).where(seen[c]) for c in wide.columns}
        ).ffill().reindex(idx)
        # rsub(axis=0)：逐列做「bar 时刻 − 最后观测时刻」
        age = obs_t.rsub(pd.Series(idx, index=idx), axis=0)
        full = full.where(age <= tol)
        full = full.reindex(columns=panel.symbols)
        meta = reg.get(name) if reg.has(name) else None
        if meta is None:
            from .panel import DIM_RATIO, ROLE_ALT, SEM_SENTIMENT, FieldMeta
            meta = FieldMeta(name, DIM_RATIO, SEM_SENTIMENT, ROLE_ALT, "hf",
                             "外部高频因子")
        panel.add_field(name, full, meta)
        installed.append(name)
    return installed
