"""因子体系合成与诊断引擎（src/engine/factor_system.py）。

把「一堆单因子」升级为「一个可解释、可回测的因子体系」：

    因子挑选 → 方向对齐 → 截面标准化 → 维度归类 → 权重合成 → 体系回测 → 全景诊断

对外主入口：

* :func:`load_market_panel`  —— 准备回测用的行情面板（优先走本地缓存，可离线）。
* :func:`compute_factor_matrix` —— 批量执行因子代码，产出对齐后的因子矩阵。
* :func:`resolve_weights`    —— 五种权重方案（等权/质量/IC/ICIR/维度均衡/手动）。
* :func:`analyze_system`     —— 一次性产出体系回测 + 相关性 + 主成分 + 衰减 + 分散化诊断。

所有函数均为纯计算，不依赖 Streamlit，便于脚本/测试复用。
"""

from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .backtest import FactorBacktester
from .factor_builder import FactorSandbox

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 权重方案标识
WEIGHT_EQUAL = "equal"
WEIGHT_QUALITY = "quality"
WEIGHT_IC = "ic"
WEIGHT_ICIR = "icir"
WEIGHT_DIMENSION = "dimension_balanced"
WEIGHT_MANUAL = "manual"

WEIGHT_LABELS: Dict[str, str] = {
    WEIGHT_EQUAL: "等权重",
    WEIGHT_QUALITY: "质量分加权",
    WEIGHT_IC: "IC 绝对值加权",
    WEIGHT_ICIR: "ICIR 加权",
    WEIGHT_DIMENSION: "维度均衡（先分维度等权，维度内再等权）",
    WEIGHT_MANUAL: "手动指定",
}

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_CACHE = os.path.join(_PROJECT_ROOT, "data", "cache")

# 无成分股接口时的兜底股票池（大市值、流动性好、行业分散）
_FALLBACK_SYMBOLS = [
    "600519", "000858", "601318", "600036", "000333", "601899", "600276", "000001",
    "600030", "002415", "600887", "000651", "601166", "600028", "601088", "002594",
    "300750", "600900", "601288", "000063", "600009", "601012", "002304", "600585",
    "601601", "600048", "000725", "600104", "601668", "603288",
]


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class SystemMember:
    """因子体系中的一个成分因子。"""

    factor_name: str
    display_name: str = ""
    dimension: str = "未分类"
    category: str = ""
    source: str = "static"
    direction: str = "positive"
    weight: float = 0.0
    quality: float = 0.5
    code: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SystemMember":
        return cls(
            factor_name=d.get("factor_name") or d.get("name", ""),
            display_name=d.get("display_name", "") or d.get("factor_name", ""),
            dimension=d.get("dimension", "未分类"),
            category=d.get("category", ""),
            source=d.get("source", "static"),
            direction=d.get("direction", "positive"),
            weight=float(d.get("weight", 0.0) or 0.0),
            quality=float(d.get("quality", 0.5) or 0.0),
            code=d.get("code", ""),
            meta=d.get("meta", {}) or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_name": self.factor_name,
            "display_name": self.display_name,
            "dimension": self.dimension,
            "category": self.category,
            "source": self.source,
            "direction": self.direction,
            "weight": self.weight,
            "quality": self.quality,
            "code": self.code,
            "meta": self.meta,
        }


# ---------------------------------------------------------------------------
# 1. 行情面板
# ---------------------------------------------------------------------------
def load_market_panel(
    n_symbols: int = 60,
    days: int = 400,
    index_code: str = "000906",
    cache_dir: Optional[str] = None,
    prefer_cache: bool = True,
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """准备因子体系回测所需的行情长表。

    取数优先级：本地整矿缓存 ``real_ore.pkl`` → 在线拉取（AKShare/新浪/Tushare）
    → 合成数据兜底。始终返回一个可用的面板，保证离线环境下界面不会白屏。

    Returns:
        ``(kline, meta)``；kline 列含 date/symbol/open/high/low/close/volume/amount/pct_chg，
        meta 说明数据来源、股票数、日期范围。
    """
    cache_dir = cache_dir or _DEFAULT_CACHE

    if prefer_cache and not force_refresh:
        kline = _load_from_ore_cache(cache_dir)
        if kline is not None and not kline.empty:
            kline = _limit_panel(kline, n_symbols, days)
            return kline, _panel_meta(kline, "本地整矿缓存 real_ore.pkl")

    kline = _load_online(n_symbols, days, index_code)
    if kline is not None and not kline.empty:
        return kline, _panel_meta(kline, f"在线行情（{index_code} 成分）")

    if not prefer_cache:  # 在线失败后仍尝试一次缓存
        kline = _load_from_ore_cache(cache_dir)
        if kline is not None and not kline.empty:
            kline = _limit_panel(kline, n_symbols, days)
            return kline, _panel_meta(kline, "本地整矿缓存 real_ore.pkl（在线回退）")

    kline = build_synthetic_panel(n_symbols=min(n_symbols, 40), days=min(days, 300))
    return kline, _panel_meta(kline, "合成数据（离线兜底，仅供流程演示）")


def _panel_meta(kline: pd.DataFrame, source: str) -> Dict[str, Any]:
    dates = sorted(kline["date"].astype(str).unique())
    return {
        "source": source,
        "n_symbols": int(kline["symbol"].nunique()),
        "n_dates": len(dates),
        "start": dates[0] if dates else "",
        "end": dates[-1] if dates else "",
        "period": f"{dates[0]} ~ {dates[-1]}" if dates else "",
    }


def _limit_panel(kline: pd.DataFrame, n_symbols: int, days: int) -> pd.DataFrame:
    """裁剪面板规模，控制回测耗时。"""
    df = kline.copy()
    df["date"] = df["date"].astype(str)
    dates = sorted(df["date"].unique())
    if days and len(dates) > days:
        df = df[df["date"].isin(set(dates[-days:]))]
    syms = sorted(df["symbol"].unique())
    if n_symbols and len(syms) > n_symbols:
        # 按数据完整度取前 n 只，避免大量停牌股稀释截面
        counts = df.groupby("symbol")["date"].count().sort_values(ascending=False)
        df = df[df["symbol"].isin(set(counts.index[:n_symbols]))]
    return _ensure_columns(df)


def _ensure_columns(kline: pd.DataFrame) -> pd.DataFrame:
    """补齐因子代码常用列，避免个别因子因缺列报错。"""
    df = kline.copy()
    df["date"] = df["date"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    if "amount" not in df.columns and {"close", "volume"}.issubset(df.columns):
        df["amount"] = df["close"] * df["volume"]
    if "pct_chg" not in df.columns:
        df["pct_chg"] = df.groupby("symbol")["close"].pct_change() * 100
    for col in ("open", "high", "low"):
        if col not in df.columns:
            df[col] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df


def _load_from_ore_cache(cache_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(cache_dir, "real_ore.pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            ore = pickle.load(f)
        parts = [
            p for p in (getattr(ore, "train_kline", None), getattr(ore, "test_kline", None))
            if p is not None and not p.empty
        ]
        if not parts:
            return None
        return _ensure_columns(pd.concat(parts, ignore_index=True))
    except Exception:
        return None


def _load_online(n_symbols: int, days: int, index_code: str) -> Optional[pd.DataFrame]:
    try:
        from data.neo_adapter import get_data_source
    except Exception:
        try:
            from src.data.neo_adapter import get_data_source  # type: ignore
        except Exception:
            return None
    try:
        # 数据源走工厂：默认 legacy（本地自爬方案保留），config.yaml 设 data.source=neodata 时切稳定源
        fetcher = get_data_source()
        symbols = fetcher.get_index_constituents(index_code) or []
        symbols = [str(s).zfill(6) for s in symbols][:n_symbols]
        if not symbols:
            symbols = _FALLBACK_SYMBOLS[:n_symbols]
        end = datetime.now()
        start = end - timedelta(days=int(days * 1.6) + 60)
        kline = fetcher.get_daily_kline(
            symbols, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d")
        )
        if kline is None or kline.empty:
            return None
        return _limit_panel(_ensure_columns(kline), n_symbols, days)
    except Exception:
        return None


def build_synthetic_panel(n_symbols: int = 30, days: int = 250, seed: int = 42) -> pd.DataFrame:
    """构造带轻微横截面可预测性的合成面板，用于离线演示完整流程。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=datetime.now(), periods=days).strftime("%Y-%m-%d").tolist()
    symbols = [f"S{i:05d}" for i in range(n_symbols)]

    rows = []
    beta = rng.normal(1.0, 0.25, n_symbols)
    market = rng.normal(0.0003, 0.011, days)
    for j, sym in enumerate(symbols):
        idio = rng.normal(0, 0.016, days)
        ret = beta[j] * market + idio
        close = 20.0 * np.cumprod(1 + ret)
        vol = rng.lognormal(14, 0.5, days)
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": sym,
                    "close": close,
                    "open": close * (1 + rng.normal(0, 0.003, days)),
                    "high": close * (1 + np.abs(rng.normal(0, 0.008, days))),
                    "low": close * (1 - np.abs(rng.normal(0, 0.008, days))),
                    "volume": vol,
                }
            )
        )
    return _ensure_columns(pd.concat(rows, ignore_index=True))


# ---------------------------------------------------------------------------
# 2. 因子矩阵
# ---------------------------------------------------------------------------
def cross_sectional_zscore(series: pd.Series, winsor: float = 3.0) -> pd.Series:
    """逐日截面 z-score + 去极值，让不同量纲的因子可以直接加权相加。"""
    if series is None or series.empty:
        return series
    df = series.rename("v").reset_index()
    date_col = df.columns[0]
    grp = df.groupby(date_col)["v"]
    mean = grp.transform("mean")
    std = grp.transform("std")
    z = (df["v"] - mean) / std.replace(0, np.nan)
    if winsor and winsor > 0:
        z = z.clip(-winsor, winsor)
    out = pd.Series(z.to_numpy(), index=series.index, name=series.name)
    return out


def compute_factor_matrix(
    kline: pd.DataFrame,
    members: Sequence[SystemMember],
    sandbox: Optional[FactorSandbox] = None,
    standardize: bool = True,
    align_direction: bool = True,
    progress: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """批量执行因子代码，返回 ``(因子矩阵, 失败原因)``。

    因子矩阵索引为 ``(date, symbol)``，每列一个因子；已按 ``direction`` 对齐方向
    （negative 因子取负号），并做过逐日截面标准化。

    Args:
        progress: 可选回调 ``fn(done, total, name)``，用于界面进度条。
    """
    sandbox = sandbox or FactorSandbox({"engine": {"sandbox": {"subprocess": False, "timeout": 60}}})
    cols: Dict[str, pd.Series] = {}
    errors: Dict[str, str] = {}
    total = len(members)

    for i, m in enumerate(members, start=1):
        if progress:
            try:
                progress(i, total, m.display_name or m.factor_name)
            except Exception:
                pass
        if not m.code or not m.code.strip():
            errors[m.factor_name] = "缺少因子代码"
            continue
        try:
            s = sandbox.run(m.code, kline)
        except Exception as e:  # 单因子失败不影响体系
            errors[m.factor_name] = f"{type(e).__name__}: {e}"
            continue
        if s is None or s.dropna().empty:
            errors[m.factor_name] = "因子输出全为空值"
            continue
        s = s.astype(float).replace([np.inf, -np.inf], np.nan)
        if align_direction and m.direction == "negative":
            s = -s
        if standardize:
            s = cross_sectional_zscore(s)
        cols[m.factor_name] = s

    if not cols:
        return pd.DataFrame(), errors
    matrix = pd.DataFrame(cols)
    matrix.index.names = ["date", "symbol"]
    return matrix, errors


def synthesize(matrix: pd.DataFrame, weights: Dict[str, float],
               min_valid_ratio: float = 0.3) -> pd.Series:
    """按权重合成体系因子。

    对每一行按「实际非空因子的权重」重新归一，避免个别因子缺值把整行拖成 NaN；
    非空因子占比低于 ``min_valid_ratio`` 的样本点直接丢弃。
    """
    if matrix.empty:
        return pd.Series(dtype=float)
    cols = [c for c in matrix.columns if c in weights]
    if not cols:
        return pd.Series(dtype=float)

    w = np.array([float(weights[c]) for c in cols], dtype=float)
    if np.allclose(w.sum(), 0):
        w = np.ones(len(cols), dtype=float)
    w = w / np.abs(w).sum()

    sub = matrix[cols]
    mask = sub.notna().to_numpy()
    vals = np.nan_to_num(sub.to_numpy(dtype=float), nan=0.0)

    weighted = vals @ w
    wsum = mask @ np.abs(w)
    valid_ratio = mask @ np.abs(w) / np.abs(w).sum()

    out = np.where(wsum > 0, weighted / np.where(wsum == 0, 1.0, wsum) * np.abs(w).sum(), np.nan)
    out = np.where(valid_ratio >= min_valid_ratio, out, np.nan)
    return pd.Series(out, index=matrix.index, name="factor")


# ---------------------------------------------------------------------------
# 3. 权重方案
# ---------------------------------------------------------------------------
def resolve_weights(
    members: Sequence[SystemMember],
    mode: str = WEIGHT_EQUAL,
    factor_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """按方案计算归一化权重（总和为 1）。

    Args:
        factor_stats: IC / ICIR 加权时需要，形如 ``{factor_name: {"ic":..,"icir":..}}``。
    """
    names = [m.factor_name for m in members]
    if not names:
        return {}

    if mode == WEIGHT_MANUAL:
        raw = np.array([max(float(m.weight), 0.0) for m in members], dtype=float)
    elif mode == WEIGHT_QUALITY:
        raw = np.array([max(float(m.quality), 0.01) for m in members], dtype=float)
    elif mode in (WEIGHT_IC, WEIGHT_ICIR):
        key = "ic" if mode == WEIGHT_IC else "icir"
        stats = factor_stats or {}
        raw = np.array(
            [abs(float((stats.get(n) or {}).get(key, 0.0) or 0.0)) for n in names], dtype=float
        )
        if np.allclose(raw.sum(), 0):
            raw = np.ones(len(names), dtype=float)
    elif mode == WEIGHT_DIMENSION:
        dims: Dict[str, List[int]] = {}
        for i, m in enumerate(members):
            dims.setdefault(m.dimension or "未分类", []).append(i)
        raw = np.zeros(len(names), dtype=float)
        per_dim = 1.0 / max(len(dims), 1)
        for idxs in dims.values():
            for i in idxs:
                raw[i] = per_dim / len(idxs)
    else:  # WEIGHT_EQUAL
        raw = np.ones(len(names), dtype=float)

    total = raw.sum()
    if total <= 0:
        raw = np.ones(len(names), dtype=float)
        total = raw.sum()
    return {n: float(v / total) for n, v in zip(names, raw)}


# ---------------------------------------------------------------------------
# 4. 全景诊断
# ---------------------------------------------------------------------------
def _safe(v: Any, default: float = float("nan")) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def evaluate_single_factors(
    kline: pd.DataFrame,
    matrix: pd.DataFrame,
    n_quantiles: int = 5,
    forward_periods: int = 1,
) -> Dict[str, Dict[str, float]]:
    """逐个因子回测，返回 ``{factor_name: 指标}``。"""
    bt = FactorBacktester(n_quantiles=n_quantiles, forward_periods=forward_periods)
    out: Dict[str, Dict[str, float]] = {}
    for col in matrix.columns:
        s = matrix[col].dropna()
        if s.empty:
            continue
        try:
            m = bt.evaluate(kline, s, verbose=False)
        except Exception as e:
            out[col] = {"error": str(e)}
            continue
        if m.get("error"):
            out[col] = {"error": m["error"]}
            continue
        out[col] = {
            "ic": _safe(m.get("ic")),
            "rank_ic": _safe(m.get("rank_ic")),
            "icir": _safe(m.get("icir")),
            "ic_positive_ratio": _safe(m.get("ic_positive_ratio")),
            "long_short_sharpe": _safe(m.get("long_short_sharpe")),
            "long_short_return": _safe(m.get("long_short_return")),
            "max_drawdown": _safe(m.get("max_drawdown")),
            "turnover": _safe(m.get("turnover")),
            "coverage": _safe(m.get("coverage")),
        }
    return out


def correlation_analysis(matrix: pd.DataFrame) -> Dict[str, Any]:
    """因子相关性 + 主成分结构。

    Returns:
        ``corr``（相关矩阵 DataFrame）、``mean_abs_corr``、``max_abs_corr``、
        ``explained_variance``（各主成分方差解释比）、``pc1_loadings``、
        ``effective_factors``（有效因子数 = 1/Σ(方差解释比²)）。
    """
    valid = matrix.dropna(axis=1, how="all")
    if valid.shape[1] < 2:
        return {
            "corr": pd.DataFrame(),
            "mean_abs_corr": float("nan"),
            "max_abs_corr": float("nan"),
            "explained_variance": [],
            "pc1_loadings": {},
            "effective_factors": float(valid.shape[1]),
            "redundant_pairs": [],
        }

    corr = valid.corr(method="pearson", min_periods=30)
    corr = corr.fillna(0.0)

    tri = corr.where(~np.eye(len(corr), dtype=bool))
    abs_vals = tri.abs().stack()
    mean_abs = float(abs_vals.mean()) if len(abs_vals) else float("nan")
    max_abs = float(abs_vals.max()) if len(abs_vals) else float("nan")

    # 高相关因子对（|rho| >= 0.8）
    redundant: List[Dict[str, Any]] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = float(corr.iloc[i, j])
            if abs(r) >= 0.8:
                redundant.append({"a": cols[i], "b": cols[j], "corr": r})
    redundant.sort(key=lambda d: -abs(d["corr"]))

    # 主成分（对相关矩阵做特征分解，等价于标准化后的 PCA）
    try:
        eigvals, eigvecs = np.linalg.eigh(corr.to_numpy())
        order = np.argsort(eigvals)[::-1]
        eigvals = np.clip(eigvals[order], 0, None)
        eigvecs = eigvecs[:, order]
        total = eigvals.sum()
        ratios = (eigvals / total).tolist() if total > 0 else []
        pc1 = eigvecs[:, 0] if eigvecs.shape[1] else np.zeros(len(cols))
        if pc1.sum() < 0:  # 统一符号，便于解读
            pc1 = -pc1
        pc1_loadings = {c: float(v) for c, v in zip(cols, pc1)}
        eff = float(1.0 / np.sum(np.square(ratios))) if ratios else float(len(cols))
    except Exception:
        ratios, pc1_loadings, eff = [], {}, float(len(cols))

    return {
        "corr": corr,
        "mean_abs_corr": mean_abs,
        "max_abs_corr": max_abs,
        "explained_variance": ratios,
        "pc1_loadings": pc1_loadings,
        "effective_factors": eff,
        "redundant_pairs": redundant[:12],
    }


def diversification_curve(
    kline: pd.DataFrame,
    matrix: pd.DataFrame,
    factor_stats: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
    max_points: int = 12,
    n_quantiles: int = 5,
    forward_periods: int = 1,
) -> List[Dict[str, float]]:
    """分散化收益曲线：按 |ICIR| 从高到低逐个纳入因子，观察体系 ICIR 的边际增益。

    用于回答「这个体系还值不值得再加因子」。
    """
    ranked = sorted(
        [c for c in matrix.columns if c in factor_stats and "error" not in factor_stats[c]],
        key=lambda c: -abs(_safe(factor_stats[c].get("icir"), 0.0)),
    )
    if not ranked:
        return []
    bt = FactorBacktester(n_quantiles=n_quantiles, forward_periods=forward_periods)
    step = max(1, len(ranked) // max_points)
    ks = sorted({*range(1, len(ranked) + 1, step), len(ranked)})

    curve: List[Dict[str, float]] = []
    for k in ks:
        sub = ranked[:k]
        w = {c: weights.get(c, 1.0 / k) for c in sub}
        comp = synthesize(matrix[sub], w)
        comp = comp.dropna()
        if comp.empty:
            continue
        try:
            m = bt.evaluate(kline, comp, verbose=False)
        except Exception:
            continue
        curve.append(
            {
                "n_factors": int(k),
                "ic": _safe(m.get("ic"), 0.0),
                "icir": _safe(m.get("icir"), 0.0),
                "sharpe": _safe(m.get("long_short_sharpe"), 0.0),
            }
        )
    return curve


def ic_decay(
    kline: pd.DataFrame,
    composite: pd.Series,
    periods: Sequence[int] = (1, 2, 3, 5, 10, 20),
    n_quantiles: int = 5,
) -> List[Dict[str, float]]:
    """体系因子的 IC 衰减曲线：不同持有期下的 IC / RankIC。"""
    out: List[Dict[str, float]] = []
    for p in periods:
        try:
            bt = FactorBacktester(n_quantiles=n_quantiles, forward_periods=int(p))
            m = bt.evaluate(kline, composite, verbose=False)
        except Exception:
            continue
        if m.get("error"):
            continue
        out.append(
            {
                "period": int(p),
                "ic": _safe(m.get("ic"), 0.0),
                "rank_ic": _safe(m.get("rank_ic"), 0.0),
                "icir": _safe(m.get("icir"), 0.0),
            }
        )
    return out


def dimension_summary(
    members: Sequence[SystemMember],
    weights: Dict[str, float],
    factor_stats: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    """按维度聚合权重与表现，输出体系的「结构画像」。"""
    buckets: Dict[str, Dict[str, Any]] = {}
    for m in members:
        d = m.dimension or "未分类"
        b = buckets.setdefault(
            d, {"dimension": d, "n_factors": 0, "weight": 0.0, "ics": [], "icirs": [], "names": []}
        )
        b["n_factors"] += 1
        b["weight"] += float(weights.get(m.factor_name, 0.0))
        b["names"].append(m.display_name or m.factor_name)
        st = factor_stats.get(m.factor_name) or {}
        if "error" not in st:
            if np.isfinite(_safe(st.get("ic"))):
                b["ics"].append(_safe(st.get("ic")))
            if np.isfinite(_safe(st.get("icir"))):
                b["icirs"].append(_safe(st.get("icir")))

    out: List[Dict[str, Any]] = []
    for b in buckets.values():
        out.append(
            {
                "dimension": b["dimension"],
                "n_factors": b["n_factors"],
                "weight": round(b["weight"], 6),
                "mean_ic": float(np.mean(b["ics"])) if b["ics"] else float("nan"),
                "mean_icir": float(np.mean(b["icirs"])) if b["icirs"] else float("nan"),
                "factors": b["names"],
            }
        )
    out.sort(key=lambda d: -d["weight"])
    return out


def analyze_system(
    kline: pd.DataFrame,
    members: Sequence[SystemMember],
    weight_mode: str = WEIGHT_EQUAL,
    n_quantiles: int = 5,
    forward_periods: int = 1,
    standardize: bool = True,
    run_decay: bool = True,
    run_diversification: bool = True,
    progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """一次性完成因子体系的全景分析。

    Returns:
        含 ``composite_metrics`` / ``factor_stats`` / ``weights`` / ``dimensions``
        / ``correlation`` / ``decay`` / ``diversification`` / ``errors`` 的字典。
        失败时返回 ``{"error": ...}``。
    """
    members = [m if isinstance(m, SystemMember) else SystemMember.from_dict(m) for m in members]
    if not members:
        return {"error": "因子体系为空，请先选择因子"}

    matrix, errors = compute_factor_matrix(
        kline, members, standardize=standardize, progress=progress
    )
    if matrix.empty:
        return {"error": "所有因子均计算失败，请检查因子代码或行情数据", "errors": errors}

    ok_members = [m for m in members if m.factor_name in matrix.columns]

    # 单因子回测（IC/ICIR 权重方案依赖它）
    factor_stats = evaluate_single_factors(kline, matrix, n_quantiles, forward_periods)
    weights = resolve_weights(ok_members, weight_mode, factor_stats)

    composite = synthesize(matrix, weights).dropna()
    if composite.empty:
        return {"error": "体系因子合成后无有效值", "errors": errors}

    bt = FactorBacktester(n_quantiles=n_quantiles, forward_periods=forward_periods)
    comp_raw = bt.evaluate(kline, composite, verbose=False)
    if comp_raw.get("error"):
        return {"error": comp_raw["error"], "errors": errors}

    ic_series = comp_raw.pop("_ic_series", pd.Series(dtype=float))
    ls_series = comp_raw.pop("_ls_series", pd.Series(dtype=float))
    quantile_cum = comp_raw.pop("quantile_cum", {})

    composite_metrics = {
        k: (_safe(v) if isinstance(v, (int, float, np.floating)) else v)
        for k, v in comp_raw.items()
    }

    corr_info = correlation_analysis(matrix)
    dims = dimension_summary(ok_members, weights, factor_stats)

    decay = ic_decay(kline, composite, n_quantiles=n_quantiles) if run_decay else []
    divers = (
        diversification_curve(kline, matrix, factor_stats, weights,
                              n_quantiles=n_quantiles, forward_periods=forward_periods)
        if run_diversification and len(matrix.columns) > 1
        else []
    )

    return {
        "composite_metrics": composite_metrics,
        "factor_stats": factor_stats,
        "weights": weights,
        "members": [m.to_dict() for m in ok_members],
        "dimensions": dims,
        "correlation": corr_info,
        "decay": decay,
        "diversification": divers,
        "errors": errors,
        "ic_series": ic_series,
        "ls_series": ls_series,
        "quantile_cum": quantile_cum,
        "quantile_stats": composite_metrics.get("quantile_stats", {}),
        "n_factors": len(ok_members),
        "weight_mode": weight_mode,
    }


# ---------------------------------------------------------------------------
# 5. 结论生成
# ---------------------------------------------------------------------------
def build_findings(result: Dict[str, Any]) -> List[Dict[str, str]]:
    """把诊断数字翻译成人话结论，用于仪表盘顶部的「核心发现」。"""
    findings: List[Dict[str, str]] = []
    cm = result.get("composite_metrics", {}) or {}
    corr = result.get("correlation", {}) or {}
    dims = result.get("dimensions", []) or []
    stats = result.get("factor_stats", {}) or {}

    ic = _safe(cm.get("ic"), 0.0)
    icir = _safe(cm.get("icir"), 0.0)
    sharpe = _safe(cm.get("long_short_sharpe"), 0.0)

    # 1. 体系整体成色
    if abs(ic) >= 0.03 and abs(icir) >= 0.4:
        findings.append({
            "tone": "ok",
            "text": f"体系整体有效：合成因子 IC={ic:.4f}、ICIR={icir:.2f}、"
                    f"多空夏普={sharpe:.2f}，达到可进入组合构建阶段的水准。",
        })
    elif abs(ic) >= 0.015:
        findings.append({
            "tone": "warn",
            "text": f"体系信号偏弱：IC={ic:.4f}、ICIR={icir:.2f}。建议提高因子质量门槛，"
                    f"或补充与现有维度低相关的新因子。",
        })
    else:
        findings.append({
            "tone": "warn",
            "text": f"体系当前几乎无横截面预测力（IC={ic:.4f}）。优先排查方向设置是否正确、"
                    f"样本区间是否过短，再考虑更换因子。",
        })

    # 2. 相关性 / 冗余
    mac = _safe(corr.get("mean_abs_corr"), 0.0)
    eff = _safe(corr.get("effective_factors"), 0.0)
    n = int(result.get("n_factors", 0))
    if n >= 2:
        if mac >= 0.5:
            findings.append({
                "tone": "warn",
                "text": f"因子同质化严重：平均绝对相关 {mac:.2f}，{n} 个因子的有效维度仅约 "
                        f"{eff:.1f} 个，加因子的边际收益已很低。",
            })
        else:
            findings.append({
                "tone": "info",
                "text": f"因子分散度良好：平均绝对相关 {mac:.2f}，{n} 个因子折合 "
                        f"{eff:.1f} 个有效独立维度。",
            })
    pairs = corr.get("redundant_pairs") or []
    if pairs:
        p = pairs[0]
        findings.append({
            "tone": "warn",
            "text": f"存在高度重复因子：<b>{p['a']}</b> 与 <b>{p['b']}</b> 相关系数 "
                    f"{p['corr']:.2f}，建议二选一或做残差正交化。",
        })

    # 3. 维度结构
    if dims:
        top = dims[0]
        if top["weight"] >= 0.5 and len(dims) > 1:
            findings.append({
                "tone": "warn",
                "text": f"权重集中在「{top['dimension']}」维度（占比 {top['weight']*100:.0f}%），"
                        f"体系风格暴露单一，遇到风格切换时回撤风险较高。",
            })
        else:
            findings.append({
                "tone": "info",
                "text": f"体系覆盖 {len(dims)} 个维度，最大维度「{top['dimension']}」权重 "
                        f"{top['weight']*100:.0f}%，结构相对均衡。",
            })

    # 4. 最强 / 最弱因子
    valid = {k: v for k, v in stats.items() if "error" not in v and np.isfinite(_safe(v.get("icir")))}
    if valid:
        best = max(valid.items(), key=lambda kv: abs(_safe(kv[1].get("icir"), 0)))
        worst = min(valid.items(), key=lambda kv: abs(_safe(kv[1].get("icir"), 0)))
        findings.append({
            "tone": "info",
            "text": f"贡献最强因子 <b>{best[0]}</b>（ICIR={_safe(best[1].get('icir')):.2f}）；"
                    f"最弱因子 <b>{worst[0]}</b>（ICIR={_safe(worst[1].get('icir')):.2f}），"
                    f"可考虑剔除后重跑对比。",
        })

    # 5. 衰减
    decay = result.get("decay") or []
    if len(decay) >= 2:
        first, last = decay[0], decay[-1]
        ic0, icn = abs(_safe(first.get("ic"), 0)), abs(_safe(last.get("ic"), 0))
        if ic0 > 0:
            keep = icn / ic0
            tone = "ok" if keep >= 0.5 else "warn"
            findings.append({
                "tone": tone,
                "text": f"持有期 {last['period']} 日时 IC 仍保留 {keep*100:.0f}%"
                        f"（{first['period']}日 {first['ic']:.4f} → {last['period']}日 {last['ic']:.4f}），"
                        f"{'适合中低频调仓' if keep >= 0.5 else '信号衰减快，需要较高换手'}。",
            })

    # 6. 失败因子提醒
    errors = result.get("errors") or {}
    if errors:
        findings.append({
            "tone": "warn",
            "text": f"有 {len(errors)} 个因子计算失败被自动剔除："
                    f"{'、'.join(list(errors.keys())[:4])}{'…' if len(errors) > 4 else ''}。",
        })

    return findings
