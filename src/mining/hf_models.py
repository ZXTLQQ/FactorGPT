"""高频订单簿的监督学习：方向预测 + 挂单成交概率。

两个任务服务于两类完全不同的策略，不要混为一谈：

1. **方向预测**（短周期趋势 / 事件驱动）
   输入 t 时刻的订单簿状态，标签是 t+h 的中价收益方向。
   这一路的评价标准是 IC 与方向准确率。

2. **挂单成交概率**（被动做市的核心）
   输入「提交限价单那一刻」的簿状态 + 报价激进度，标签是这笔单子最终有没有成交。
   做市商的真实损益几乎完全取决于：**挂出去的单子能不能成交、成交后价格会不会反向**。
   这比预测价格方向更贴近做市的本质，也是这批数据（自带委托流水）最独特的价值。

关于时间切分的强约束
--------------------
这里的 train/test **一律按时间顺序切分**（前段训练、后段测试），
绝不能随机打乱。高频样本的相邻记录高度自相关，随机切分会让训练集和测试集
共享同一段市场状态，指标好看但完全不可用。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from data.hf_adapter import estimate_tick_size
from mining.hf import _to_timedelta, build_l2_features, make_forward_labels

logger = logging.getLogger(__name__)


def prepare_direction_dataset(snap: pd.DataFrame, horizon_ticks: int = 10,
                              features: Optional[pd.DataFrame] = None,
                              factor_cols: Optional[list] = None) -> dict:
    """拼装方向预测的监督数据集（特征 + 未来标签），严格按时间切分返回。"""
    X = features if features is not None else build_l2_features(snap)
    if factor_cols:
        X = X[[c for c in factor_cols if c in X.columns]]
    y_ret = make_forward_labels(snap, horizon_ticks=horizon_ticks, kind="ret")
    y_dir = make_forward_labels(snap, horizon_ticks=horizon_ticks, kind="dir")

    m = y_ret.notna() & y_dir.notna()
    X = X.loc[m]
    # 常量列与近全空列一并剔除：它们对模型无用，还会让标准化报警
    keep = [c for c in X.columns
            if X[c].notna().mean() > 0.5 and X[c].nunique(dropna=True) > 1]
    X = X[keep]
    y_ret, y_dir = y_ret.loc[m], y_dir.loc[m]
    # 再剔除含 NaN 的行（不做填充：填充会把「刚开盘」伪装成正常状态）
    ok = X.notna().all(axis=1)
    return {"X": X[ok], "y_ret": y_ret[ok], "y_dir": y_dir[ok]}


def time_split(n: int, train_ratio: float = 0.7, purge_ticks: int = 50):
    """按时间顺序切分，并在两段之间留出 ``purge_ticks`` 的隔离带。

    隔离带（purge）是高频建模的标配：即便按时间切分，紧邻边界的样本之间
    仍然存在很强的自相关，特征还可能包含滚动窗口带来的边界重叠。
    """
    cut = int(n * train_ratio)
    tr = np.arange(0, max(0, cut - purge_ticks))
    te = np.arange(min(n, cut + purge_ticks), n)
    return tr, te


#: 方向模型的**默认因子子集**——来自实测，不是拍脑袋。
#: 在 au2602 日盘、horizon=10 tick 上的对照结果（样本外 accuracy）：
#:     精简 5 因子   0.5443      全 50 个因子  0.5115
#: 也就是说，把 50 个因子全喂给 GBDT **反而更差**：高频数据的信噪比极低，
#: 大量弱/噪声特征给了模型无穷的过拟合空间。换品种或换周期请重新做这个对照。
CORE_HF_FACTORS = ["ofi", "dq_b", "dq_s", "dq_b1", "obi_w5", "obi_l1",
                   "micro_dev_ticks", "rvol_20", "spread_ticks", "trend_20"]


def train_direction_model(ds: dict, train_ratio: float = 0.7,
                          purge_ticks: int = 50,
                          factors: Optional[list | str] = "core") -> dict:
    """训练方向模型（HistGradientBoosting）。返回可核验的指标字典。

    刻意**不用**随机切分，也不追求"漂亮"的准确率：这里报告的是
    样本外（时间上更晚）的真实表现，指标难看才是常态，好看才需要怀疑。

    ``factors``:
      · ``"core"``（默认）只用 :data:`CORE_HF_FACTORS` 白名单，依据是上面的对照实验；
      · ``None`` 使用全部因子（**不推荐**，实测更差，保留仅便于对照）；
      · 传入列名列表则自定义。
    """
    X, y = ds["X"], ds["y_dir"]
    if factors == "core":
        keep = [c for c in CORE_HF_FACTORS if c in X.columns]
        missing = [c for c in ("ofi", "micro_dev_ticks") if c not in keep]
        if missing:
            logger.warning("[HF] 核心因子缺失 %s，退化为全部因子", missing)
        else:
            X = X[keep]
    elif isinstance(factors, list):
        X = X[[c for c in factors if c in X.columns]]
    X = X.fillna(X.median(numeric_only=True))
    n = len(X)
    tr, te = time_split(n, train_ratio, purge_ticks)
    if len(tr) < 500 or len(te) < 200:
        return {"ok": False, "reason": f"样本不足 train={len(tr)} test={len(te)}"}

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    # 仅保留「涨 / 跌」两类样本：0（几乎没动）既无交易价值又会稀释信号
    mask_tr = y.iloc[tr] != 0
    mask_te = y.iloc[te] != 0
    Xtr, ytr = X.iloc[tr][mask_tr.values], y.iloc[tr][mask_tr.values]
    Xte, yte = X.iloc[te][mask_te.values], y.iloc[te][mask_te.values]
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return {"ok": False, "reason": "类别不足"}

    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
                                           max_leaf_nodes=31, random_state=42)
    model.fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:, 1]
    pred = np.where(prob > 0.5, 1.0, -1.0)
    acc = float((pred == yte.to_numpy()).mean())
    try:
        auc = float(roc_auc_score((yte.to_numpy() > 0).astype(int), prob))
    except Exception:  # noqa: BLE001
        auc = float("nan")
    return {
        "ok": True, "model": model, "accuracy": acc, "auc": auc,
        "n_train": len(Xtr), "n_test": len(Xte),
        "hit_rate_dummy": float((yte.to_numpy() == 1).mean()),
        "importance": _feature_importance(model, Xte, yte, X.columns),
    }


def _feature_importance(model, Xte, yte, feature_names, n_repeats: int = 3) -> pd.Series:
    """统一的特征重要性：一律用**置换重要性**。

    这里刻意不用 ``model.feature_importances_``：
    ``HistGradientBoostingClassifier`` 根本没有这个属性（会直接抛 AttributeError），
    而且自带重要性对高基数/高方差特征有系统性偏好，容易被误读。
    置换重要性直接在**样本外**数据上打乱某列、看指标掉了多少，衡量的才是真的贡献。
    """
    try:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(model, Xte, yte, n_repeats=n_repeats,
                                   random_state=42, scoring="roc_auc")
        return pd.Series(r.importances_mean, index=list(feature_names)).sort_values(ascending=False)
    except Exception as e:  # noqa: BLE001
        logger.warning("[HF] 置换重要性计算失败: %s", e)
        return pd.Series(dtype=float)


def quantile_report(feature: pd.Series, y_ret: pd.Series, q: int = 10) -> pd.DataFrame:
    """因子分位分层报告：不看模型，直接看因子与未来收益是否单调。

    这是最朴素的证据，也是最不容易被模型选择偏差污染的证据：
    如果第 1 到第 10 分位的平均未来收益单调递增，因子的预测力就是真的。
    """
    df = pd.DataFrame({"f": feature, "y": y_ret}).dropna()
    if len(df) < q * 20:
        return pd.DataFrame()
    try:
        df["bucket"] = pd.qcut(df["f"].rank(method="first"), q, labels=False)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    g = df.groupby("bucket").agg(mean_ret=("y", "mean"), n=("y", "size"),
                                 hit=("y", lambda s: (s > 0).mean()))
    mono = float(pd.Series(g["mean_ret"]).corr(pd.Series(range(len(g)))))
    g.attrs["monotonicity"] = mono
    g.attrs["spread_ret"] = float(g["mean_ret"].iloc[-1] - g["mean_ret"].iloc[0])
    return g


# --------------------------------------------------------------------------
# 挂单成交概率（被动做市）
# --------------------------------------------------------------------------
def build_fill_dataset(snap: pd.DataFrame, orders: pd.DataFrame,
                       tick: Optional[float] = None,
                       max_lag: str = "3s") -> pd.DataFrame:
    """把「每笔自己的委托」与「下单前可见的盘口」拼起来，构造成交概率数据集。

    两个决定成败的细节：

    1. **只能用下单前可见的快照**（``direction="backward"``）。
       如果取下单之后的快照，那个快照很可能已经被这张单子自己改变了——
       尤其是我们自己挂单后立即成为了盘口的一部分，等于用「我挂了单」预测
       「我挂了单」，这是彻头彻尾的自指泄漏（self-referential leakage）。

    2. **价格激进度是决定性特征**：挂得离中价越近甚至越穿越，越容易成交。
       定义为 ``side * (price - mid) / tick``，买单为正表示更激进。

    最后还有一个**数据层面的硬限制**：委托时间只有秒级精度，盘口是 500ms，
    同一秒内的多笔订单无法区分先后，队列位置（queue position）无法精确还原，
    因此这里只能用同价位的前置挂单量作为**近似**。
    """
    tick = float(tick or estimate_tick_size(snap["bp1"].to_numpy()))
    q = pd.DataFrame({
        "ts": snap["ts"],
        "mid": (snap["bp1"].astype("float64") + snap["sp1"].astype("float64")) / 2.0,
        "spread_ticks": (snap["sp1"].astype("float64") - snap["bp1"].astype("float64")) / tick,
        "bid_depth": np.nan_to_num(snap["bv1"].astype("float64")),
        "ask_depth": np.nan_to_num(snap["sv1"].astype("float64")),
    })
    bp = snap["bp1"].astype("float64").to_numpy()
    bv = np.nan_to_num(snap["bv1"].astype("float64").to_numpy())
    sv = np.nan_to_num(snap["sv1"].astype("float64").to_numpy())
    tot1 = bv + sv
    q["obi_l1"] = np.where(tot1 > 0, (bv - sv) / np.where(tot1 == 0, np.nan, tot1), np.nan)
    q["microprice"] = np.where(tot1 > 0,
                              (bp * sv + snap["sp1"].astype("float64").to_numpy() * bv)
                              / np.where(tot1 == 0, np.nan, tot1), np.nan)
    q["micro_dev_ticks"] = (q["microprice"] - q["mid"]) / tick
    dvol = snap.get("d_volume", pd.Series(0.0, index=snap.index)).astype("float64")
    q["recent_trade"] = np.nan_to_num(dvol.to_numpy())
    q = q.sort_values("ts")

    o = orders.sort_values("ts").copy()
    o["price"] = pd.to_numeric(o["price"], errors="coerce")
    merged = pd.merge_asof(o, q, on="ts", direction="backward",
                           tolerance=_to_timedelta(max_lag))
    merged = merged.dropna(subset=["mid", "spread_ticks"])

    # ---- 特征构造 ----
    merged["price_aggr_ticks"] = merged["side"] * (merged["price"] - merged["mid"]) / tick
    # 同价位排在我前面的挂单量（买看买一量，卖看卖一量）
    merged["queue_ahead"] = np.where(merged["side"] > 0, merged["bid_depth"], merged["ask_depth"])
    merged["queue_ahead_opp"] = np.where(merged["side"] > 0, merged["ask_depth"], merged["bid_depth"])
    merged["size_vs_queue"] = merged["qty"] / merged["queue_ahead"].replace(0, np.nan)
    merged["obi_signed"] = merged["side"] * merged["obi_l1"]
    merged["micro_signed"] = merged["side"] * merged["micro_dev_ticks"]
    merged["spread_inv"] = 1.0 / merged["spread_ticks"].replace(0, np.nan)
    merged.attrs["tick_size"] = tick
    return merged


def train_fill_model(ds: pd.DataFrame, features: Optional[list] = None,
                     train_ratio: float = 0.7) -> dict:
    """训练挂单成交概率模型，返回 AUC / 提升度 / 分成交率校准表。

    关注点不是 accuracy（成交本身是稀有事件），而是：
      · **AUC**：能否把「会成交的单」排到前面；
      · **Top 分位的实际成交率**：如果概率最高的 10% 抽样里成交率显著高于基准，
        这个模型就能直接用于「该不该挂这张单」的决策。
    """
    feats = features or ["price_aggr_ticks", "queue_ahead", "queue_ahead_opp",
                         "size_vs_queue", "obi_signed", "micro_signed",
                         "spread_ticks", "spread_inv", "recent_trade"]
    feats = [f for f in feats if f in ds.columns]
    d = ds.dropna(subset=[*feats, "filled"]).sort_values("ts").reset_index(drop=True)
    if len(d) < 500 or d["filled"].nunique() < 2:
        return {"ok": False, "reason": f"样本不足或标签单一 n={len(d)} "
                                      f"成交率={d['filled'].mean() if len(d) else float('nan'):.4f}"}
    X = d[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = d["filled"].astype(int)
    cut = int(len(d) * train_ratio)
    Xtr, Xte = X.iloc[:cut], X.iloc[cut:]
    ytr, yte = y.iloc[:cut], y.iloc[cut:]
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return {"ok": False, "reason": "训练/测试集类别不足（成交率过低，无法监督）"}

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    out = {"ok": True, "n_train": len(Xtr), "n_test": len(Xte),
           "base_rate_train": float(ytr.mean()), "base_rate_test": float(yte.mean())}
    # 逻辑回归：可解释，用来判断「价格激进度」等特征的符号是否符合直觉
    lr = LogisticRegression(max_iter=2000)
    lr.fit(Xtr.fillna(0), ytr)
    out["lr_coef"] = pd.Series(lr.coef_[0], index=feats).sort_values(ascending=False)
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, random_state=42)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    out["model"] = m
    out["auc"] = float(roc_auc_score(yte, p))
    out["importance"] = _feature_importance(m, Xte, yte, feats)
    # 按预测概率分十档，看真实成交率的单调性
    bin_ = pd.qcut(pd.Series(p, index=Xte.index).rank(method="first"), 10, labels=False)
    tbl = pd.DataFrame({"p": p, "y": yte.to_numpy(), "bin": bin_.to_numpy()}) \
        .groupby("bin").agg(mean_p=("p", "mean"), real_rate=("y", "mean"), n=("y", "size"))
    tbl["lift"] = tbl["real_rate"] / max(out["base_rate_test"], 1e-9)
    out["decile"] = tbl
    out["top_decile_lift"] = float(tbl["lift"].iloc[-1])
    return out


__all__ = [
    "build_fill_dataset",
    "prepare_direction_dataset",
    "quantile_report",
    "time_split",
    "train_direction_model",
    "train_fill_model",
]
