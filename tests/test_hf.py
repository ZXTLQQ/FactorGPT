"""高频（L2 五档快照）模块的契约测试。

全部用例使用**合成数据**，不依赖外部 parquet 文件，因此 CI / 克隆后可直接跑。
覆盖的都是「错了会静默污染结果」的地方：

- SortTime 跨午夜时钟还原（错一个小时＝夜盘整体平移一天）
- 会话边界切断（午休被卷进差分/滚动/未来标签）
- OFI 按价格而非档位匹配（价格平移时不错配）
- tick size 吸附（float32 差分会估出 0.01996 而非 0.02）
- 因子无前视（第 t 行只用 ≤ t 的信息）
- 高频字段进入统一面板后的陈旧掩蔽（跨会话不得 forward-fill）
- 回测只在换手时扣手续费（每 tick 扣一次会把所有策略判死）
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data.hf_adapter import _sorttime_to_wallclock, estimate_tick_size  # noqa: E402
from mining import hf as HF  # noqa: E402
from mining.hf_strategies import evaluate_signal  # noqa: E402
from mining.panel import (  # noqa: E402
    DIM_PRICE,
    ROLE_ALT,
    ROLE_MARKET,
    SEM_VALUE,
    FieldRegistry,
    PanelData,
)

TICK = 0.02


# --------------------------------------------------------------------------
# 合成订单簿
# --------------------------------------------------------------------------
def book(n: int = 40, base: float = 100.0, tick: float = TICK,
         sessions=(0,), start_sid: int = 0) -> pd.DataFrame:
    """构造一个「买价低于卖价、五档等距」的最小可算订单簿。"""
    rows = []
    for i in range(n):
        sid = start_sid
        r = {"ts": pd.Timestamp("2025-12-08 09:00:00") + pd.to_timedelta(500 * i, unit="ms"),
             "session_id": sid, "new_session": i == 0}
        mid = base + tick * i
        for lv in range(1, 6):
            r[f"bp{lv}"] = mid - tick * lv
            r[f"bv{lv}"] = float(100 - 10 * lv)
            r[f"sp{lv}"] = mid + tick * lv
            r[f"sv{lv}"] = float(90 + 10 * lv)
        rows.append(r)
    return pd.DataFrame(rows)


def book_two_sessions(n_per: int = 8) -> pd.DataFrame:
    """两段会话：第二段在第一段结束后很久才开始（模拟午休/隔夜）。"""
    a = book(n_per, base=100.0)
    b = book(n_per, base=100.0 + TICK * 50)
    b["session_id"] = 1
    b["ts"] = pd.Timestamp("2025-12-08 13:30:00") + pd.to_timedelta(
        np.arange(n_per) * 500, unit="ms")
    b["new_session"] = np.arange(n_per) == 0
    out = pd.concat([a, b], ignore_index=True)
    out.attrs["sizes"] = (n_per, n_per)
    return out


# --------------------------------------------------------------------------
# 1. SortTime → 墙上时钟
# --------------------------------------------------------------------------
def test_sorttime_cross_midnight_and_day_session() -> None:
    rows = [
        # 夜盘跨午夜：SortTime 25:00 → 次日 01:00，自然日取 CalDate（已含进位）
        (250000000, 20251206, 20251208, "N"),
        # 日盘：自然日取 Date
        (90000500, 20251206, 20251208, "M"),
        # SortTime 39:00（次日下午收盘）→ %24 = 15:00
        (390000000, 20251206, 20251208, "E"),
    ]
    ts = _sorttime_to_wallclock(
        np.array([r[0] for r in rows]),
        np.array([r[1] for r in rows]),
        np.array([r[2] for r in rows]),
        np.array([r[3] for r in rows]))
    want = [pd.Timestamp("2025-12-06 01:00:00"),
            pd.Timestamp("2025-12-08 09:00:00.500"),
            pd.Timestamp("2025-12-08 15:00:00")]
    assert list(pd.DatetimeIndex(ts)) == want


def test_sorttime_night_continues_without_extra_day() -> None:
    """夜盘跨午夜：墙上时钟 HH%24，自然日直接取 CalDate（**不再**额外 +1 天）。

    CalDate 本身已带午夜进位，这正是最初写错的地方：多加一天会让夜盘整体
    平移到 12-07，与委托流水零交集。
    """
    st = np.array([210000000, 230000000, 250000000, 263000000], dtype="int64")
    cal = np.array([20251205, 20251205, 20251206, 20251206], dtype="int64")
    ts = _sorttime_to_wallclock(st, cal, np.full(4, 20251208), np.array(["N"] * 4))
    got = [pd.Timestamp(t).strftime("%Y-%m-%d %H:%M") for t in ts]
    assert got == ["2025-12-05 21:00", "2025-12-05 23:00",
                   "2025-12-06 01:00", "2025-12-06 02:30"], got
    assert (pd.Series(ts).diff().dropna() > pd.Timedelta(0)).all(), "夜盘时间轴必须单调递增"


# --------------------------------------------------------------------------
# 2. tick size 估计
# --------------------------------------------------------------------------
def test_estimate_tick_size_snaps_to_quote_unit() -> None:
    rng = np.random.default_rng(7)
    prices = np.round(400 + np.cumsum(rng.choice([-1, 0, 1], size=4000)) * 0.02, 4)
    got = estimate_tick_size(prices)
    assert abs(got - 0.02) < 1e-6, f"tick 估计 {got}，应为 0.02"


def test_estimate_tick_size_short_series_is_nan() -> None:
    assert np.isnan(estimate_tick_size(np.array([1.0, 1.0, 1.0])))


# --------------------------------------------------------------------------
# 3. 会话边界
# --------------------------------------------------------------------------
def test_forward_labels_are_cut_at_session_boundary() -> None:
    d = book_two_sessions(n_per=6)
    y = HF.make_forward_labels(d, horizon_ticks=1, tick=TICK)
    n_per = 6
    # 第一段内部：中价每 tick 涨 1 个 tick sized step → 标签恒为 +1.0
    inner = y.iloc[0:n_per - 1].to_numpy()
    assert np.allclose(inner, 1.0), f"段内标签异常: {inner}"
    # 最后一行（未来在段外）与跨段的分界必须 NaN
    assert np.isnan(y.iloc[n_per - 1])
    # 第二段首行开始重新可用（同一段内）
    assert not np.isnan(y.iloc[n_per])
    # 整段尾部必有 h 个 NaN
    assert np.isnan(y.to_numpy()[-1])


def test_forward_labels_horizon_spanning_session_is_nan() -> None:
    d = book_two_sessions(n_per=4)
    # 步长恰好等于每段长度 → 每个样本的 t+h 都落在段外，应全部 NaN
    y = HF.make_forward_labels(d, horizon_ticks=4, tick=TICK)
    assert int(y.notna().sum()) == 0, y.to_numpy()


def test_rolling_stat_is_session_scoped() -> None:
    sid = np.array([0, 0, 0, 1, 1, 1])
    vals = np.array([1.0, 1.0, 1.0, 100.0, 100.0, 100.0])
    got = HF._rolling_stat(vals, w=2, stat="sum", sid=sid)
    want = np.array([np.nan, 2.0, 2.0, np.nan, 200.0, 200.0])
    assert np.allclose(got, want, equal_nan=True), f"跨会话滚动污染: {got}"


def test_build_l2_features_first_row_of_session_is_nan() -> None:
    d = book_two_sessions(n_per=10)
    f = HF.build_l2_features(d, tick=TICK)
    # 依赖历史的量（OFI 队列变化、成交量脉冲）在会话首行必须是 NaN
    for col in ("ofi", "dq_b", "dq_s"):
        assert col in f.columns, f"缺少因子 {col}"
        assert np.isnan(f[col].iloc[0])
        assert np.isnan(f[col].iloc[10]), f"{col} 在第二段首行应切断"


# --------------------------------------------------------------------------
# 4. OFI：按价格匹配
# --------------------------------------------------------------------------
def flat_book(n: int = 3, mid: float = 100.0, tick: float = TICK) -> pd.DataFrame:
    """价格**不漂移**的订单簿：每档数量固定，方便手算 ΔQ。"""
    rows = []
    for i in range(n):
        r = {"ts": pd.Timestamp("2025-12-08 09:00:00") + pd.to_timedelta(500 * i, unit="ms"),
             "session_id": 0, "new_session": i == 0}
        for lv in range(1, 6):
            r[f"bp{lv}"] = mid - tick * lv
            r[f"bv{lv}"] = 10.0
            r[f"sp{lv}"] = mid + tick * lv
            r[f"sv{lv}"] = 10.0
        rows.append(r)
    return pd.DataFrame(rows)


def test_queue_change_ofi_accounts_queue_delta() -> None:
    d = flat_book(3)
    d.loc[1, "bv1"] = 20.0          # 买一 +10 手 → ofi = +10
    d.loc[2, "bv1"] = 20.0          # 买盘维持（相对第 1 行 ΔQ=0）
    d.loc[2, "sv1"] = 15.0          # 卖一 +5 手  → ofi = -5
    res = HF.queue_change_ofi(d, tick=TICK)
    assert np.isnan(res["ofi"].iloc[0]), "会话首行没有可比的前值，必须 NaN"
    assert abs(res["ofi"].iloc[1] - 10.0) < 1e-9
    assert abs(res["ofi"].iloc[2] - (-5.0)) < 1e-9
    per = sum(res[f"dq_b{i}"].iloc[1] for i in range(1, 6))
    assert abs(per - res["dq_b"].iloc[1]) < 1e-9, "分档 ΔQ 之和必须等于总量"


def test_queue_change_ofi_matches_by_price_not_by_level() -> None:
    """在买一**上方**插一档新价：按价格匹配应得 +7，按档位序号匹配会得到 -3。"""
    d = flat_book(2)
    for lv in range(5, 1, -1):
        d.loc[1, f"bp{lv}"] = d.loc[0, f"bp{lv - 1}"]
        d.loc[1, f"bv{lv}"] = d.loc[0, f"bv{lv - 1}"]
    d.loc[1, "bp1"] = d.loc[0, "bp1"] + TICK
    d.loc[1, "bv1"] = 7.0
    res = HF.queue_change_ofi(d, tick=TICK)
    assert abs(res["dq_b"].iloc[1] - 7.0) < 1e-9, res["dq_b"].iloc[1]


def test_distal_missing_levels_count_as_nothing() -> None:
    d = flat_book(2)
    for lv in (4, 5):
        d.loc[:, [f"bp{lv}", f"bv{lv}", f"sp{lv}", f"sv{lv}"]] = np.nan
    res = HF.queue_change_ofi(d, tick=TICK)
    assert np.isfinite(res["dq_b"].iloc[1]), "远端档缺失不应把整行变成 NaN"
    assert abs(res["dq_b"].iloc[1]) < 1e-9, "量未变化则 ΔQ 应为 0"


# --------------------------------------------------------------------------
# 5. 无前视
# --------------------------------------------------------------------------
def test_build_l2_features_has_no_lookahead() -> None:
    d = book(n=60)
    f1 = HF.build_l2_features(d, tick=TICK)
    d2 = d.copy()
    cut = 30
    # 把「未来」的买一量放大 1000 倍：cut 之前的所有因子值必须一模一样
    d2.loc[cut:, "bv1"] = d2.loc[cut:, "bv1"] * 1000.0
    d2.loc[cut:, "sv1"] = d2.loc[cut:, "sv1"] * 1000.0
    f2 = HF.build_l2_features(d2, tick=TICK)
    for col in f1.columns:
        a, b = f1[col].iloc[:cut].to_numpy(dtype=float), f2[col].iloc[:cut].to_numpy(dtype=float)
        assert np.allclose(a, b, equal_nan=True), f"因子 {col} 使用了未来信息"


def test_labels_use_only_negative_shift_in_one_place() -> None:
    """负向 shift 只允许出现在 make_forward_labels。"""
    with open(os.path.join(ROOT, "src", "mining", "hf.py"), encoding="utf-8") as fh:
        src = fh.read()
    assert "shift(-" in src
    # 特征构造函数体内不得出现未来窗
    body = src.split("def make_forward_labels")[1]
    assert "shift(-" in body
    feats = src.split("def build_l2_features")[1].split("def make_forward_labels")[0]
    assert "shift(-" not in feats, "build_l2_features 里出现了负向 shift"


# --------------------------------------------------------------------------
# 6. 高频字段进统一面板
# --------------------------------------------------------------------------
def _tiny_panel():
    dates = pd.date_range("2025-12-08", periods=3, freq="D")
    rows = []
    for i, day in enumerate(dates):
        for sym in ("AAA", "BBB"):
            rows.append({"date": day.strftime("%Y-%m-%d"), "symbol": sym,
                         "open": 10.0 + i, "high": 10.5 + i, "low": 9.5 + i,
                         "close": 10.2 + i, "volume": 1e6})
    reg = FieldRegistry()
    HF.register_hf_fields(reg)
    reg.register_fields([("close", DIM_PRICE, SEM_VALUE, "收盘价")],
                        source="market", role=ROLE_MARKET)
    return PanelData.from_kline(pd.DataFrame(rows), forward_periods=(1,), registry=reg)


def test_register_hf_fields_marks_pretrade_as_alt_role() -> None:
    reg = FieldRegistry()
    HF.register_hf_fields(reg)
    assert reg.get("ofi").role == ROLE_ALT, "订单簿字段应是撮合前另类信息"
    assert reg.get("bp1").role == ROLE_ALT
    assert reg.get("obi_w5").role == ROLE_ALT
    assert HF.register_hf_fields(reg).has("microprice")


def test_install_hf_features_masks_stale_observations() -> None:
    panel = _tiny_panel()
    dates = list(panel.dates)
    long_df = pd.DataFrame({
        "ts": [dates[0], dates[1], dates[2], dates[0], dates[2]],
        "symbol": ["AAA", "AAA", "AAA", "BBB", "BBB"],
        "obi_w5": [0.1, 0.2, 0.3, -0.1, -0.3],
        "my_hf": [1.0, 2.0, 3.0, -1.0, -3.0],
    })
    got = HF.install_hf_features(panel, long_df, fields=["obi_w5", "my_hf"],
                                 tolerance="600s")
    assert set(got) == {"obi_w5", "my_hf"}
    w = panel.field("obi_w5")
    assert w.shape == (len(panel.dates), len(panel.symbols))
    assert abs(w.loc[dates[1], "AAA"] - 0.2) < 1e-9, "同一时刻应精确落地"
    assert np.isnan(w.loc[dates[1], "BBB"]), "BBB 缺失该时点，且超出耐受窗口，应掩蔽为 NaN"
    assert abs(w.loc[dates[2], "BBB"] + 0.3) < 1e-9, "恢复观测后应重新有效"
    # 未注册的外部因子走默认元数据，但仍按同样语义装入
    assert abs(panel.field("my_hf").loc[dates[2], "BBB"] + 3.0) < 1e-9


def test_install_hf_features_is_usable_in_expression() -> None:
    """高频字段必须能进统一表达式树并被类型校验接受。"""
    from mining import expr as ex
    panel = _tiny_panel()
    dates = list(panel.dates)
    long_df = pd.DataFrame({
        "ts": dates * 2,
        "symbol": ["AAA"] * 3 + ["BBB"] * 3,
        "obi_w5": [0.1, 0.2, 0.3, -0.1, -0.2, -0.3],
    })
    HF.install_hf_features(panel, long_df, fields=["obi_w5"])
    node = ex.parse("zscore_cs(obi_w5)")
    assert ex.validate(node, panel.registry) == []
    # 人工量大纲 vs 信号量纲做差应被静态类型检查拦住
    errs = ex.validate(ex.parse("sub(ofi, close)"), panel.registry)
    assert errs, "量纲检查失效：手数成交量 - 价格 竟然通过"


# --------------------------------------------------------------------------
# 7. 回测口径
# --------------------------------------------------------------------------
def test_evaluate_signal_charges_cost_only_on_position_flip() -> None:
    n = 20
    sig = pd.Series([1.0] * 10 + [-1.0] * 10)
    y = sig.copy()                       # 每笔毛收益恰好 +1 tick
    r = evaluate_signal(sig, y, cost_ticks=0.5, hold_ticks=1)
    assert r["ok"] is True
    assert r["n_trades"] == n
    assert abs(r["gross_pnl_ticks"] - 1.0) < 1e-9
    # 只有首仓 + 一次反手付费：总成本 1 tick / 20 笔
    assert abs(r["mean_pnl_ticks"] - (1.0 - 1.0 / n)) < 1e-9, r
    assert abs(r["win_rate"] - 1.0) < 1e-9


def test_evaluate_signal_downsamples_by_hold_ticks() -> None:
    sig = pd.Series(np.r_[np.ones(200), -np.ones(200)])
    y = pd.Series(np.r_[np.ones(200), -np.ones(200)])
    r = evaluate_signal(sig, y, cost_ticks=0.0, hold_ticks=10)
    assert r["ok"] is True
    assert r["n_trades"] == 40, f"hold_ticks=10 时 400 tick 只应有 40 次决策，实际 {r['n_trades']}"


def test_evaluate_signal_rejects_tiny_sample() -> None:
    r = evaluate_signal(pd.Series([1.0, -1.0, 1.0]), pd.Series([1.0, 1.0, 1.0]))
    assert r["ok"] is False and "样本" in r["reason"]


# --------------------------------------------------------------------------
# 8. IC 工具
# --------------------------------------------------------------------------
def test_evaluate_factor_ic_ranks_informative_feature_first() -> None:
    n = 2000
    rng = np.random.default_rng(11)
    x = rng.standard_normal(n)
    y = pd.Series(x + 0.1 * rng.standard_normal(n))   # 强相关
    z = pd.Series(rng.standard_normal(n))             # 纯噪声
    feats = pd.DataFrame({"good": x, "noise": z})
    ic = HF.evaluate_factor_ic(feats, y)
    assert ic.iloc[0]["factor"] == "good"
    assert abs(ic.iloc[0]["ic"]) > 0.9
    assert abs(ic.iloc[0]["t_stat"]) > 20
    assert len(ic) == 2
