"""面板数据 + 字段注册表 + PIT 对齐：量价与基本面因子的统一载体。

三篇研报的共同前提是「数据先统一，因子才能统一」（中信建投：《量价 X 基本面
因子挖掘统一框架》）。本模块给出这一层：

- ``PanelData``：**行=交易日、列=标的** 的宽表面板集合。量价字段（开高低收、
  成交量额、收益）与基本面字段（PIT 投影后的财务科目）、另类字段（概念计数）
  共用同一容器，因此表达式树、算子库、评价体系对三类因子完全同构。
- ``FieldRegistry``：字段的**维度 / 语义 / 角色**三元元数据。角色区分
  ``M``（量价）、``F``（基本面）、``T``（另类文本）与 ``X``（风险暴露），
  是"统一框架"里跨域组合与中性化的依据。
- ``asof_align``：财务数据的 **point-in-time** 投影。财务值只有在
  ``ann_date + lag_days`` 之后才允许出现在因子中，从机制上杜绝前视偏差
  （中信建投稿明确要求；这是财务类因子唯一真正致命的风险）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ── 维度（量纲） ──
DIM_PRICE = "price"
DIM_VOLUME = "volume"
DIM_AMOUNT = "amount"
DIM_RATIO = "ratio"
DIM_SCORE = "score"
DIM_COUNT = "count"
DIM_FLAG = "flag"
DIM_GROWTH = "growth"

# ── 语义（经济含义） ──
SEM_VALUE = "value"
SEM_GROWTH = "growth"
SEM_QUALITY = "quality"
SEM_LEVERAGE = "leverage"
SEM_VALUATION = "valuation"
SEM_MOMENTUM = "momentum"
SEM_LIQUIDITY = "liquidity"
SEM_RISK = "risk"
SEM_SIZE = "size"
SEM_CONCEPT = "concept"
SEM_SENTIMENT = "sentiment"

# ── 角色 ──
ROLE_MARKET = "M"
ROLE_FUNDAMENTAL = "F"
ROLE_ALT = "T"
ROLE_EXPOSURE = "X"

ROLE_LABELS = {
    ROLE_MARKET: "量价（market）",
    ROLE_FUNDAMENTAL: "基本面（fundamental）",
    ROLE_ALT: "另类文本（text/alt）",
    ROLE_EXPOSURE: "风险暴露（exposure）",
}


@dataclass(frozen=True)
class FieldMeta:
    """字段元数据：表达式树静态类型检查与跨域组合的判据。"""

    name: str
    dimension: str
    semantics: str
    role: str = ROLE_MARKET
    source: str = ""
    doc: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "dimension": self.dimension,
                "semantics": self.semantics, "role": self.role,
                "source": self.source, "doc": self.doc}


class FieldRegistry:
    """字段元数据注册表（可合并：量价 + 基本面 + 另类字段各注册自己的部分）。"""

    def __init__(self, fields: Optional[Iterable[FieldMeta]] = None) -> None:
        self._meta: Dict[str, FieldMeta] = {}
        for m in fields or ():
            self.register(m)

    # -- 基本操作 --
    def register(self, meta: FieldMeta) -> FieldMeta:
        self._meta[meta.name] = meta
        return meta

    def register_fields(self, rows: Iterable[Tuple[str, str, str, str]],
                        source: str = "", role: str = ROLE_MARKET) -> None:
        """批量注册：(name, dimension, semantics, doc) 或 (name, dim, sem, role, doc)。"""
        for row in rows:
            if len(row) == 4:
                name, dim, sem, doc = row
                r = role
            else:
                name, dim, sem, r, doc = row  # type: ignore[misc]
            self.register(FieldMeta(name, dim, sem, r, source, doc))

    def get(self, name: str) -> FieldMeta:
        if name not in self._meta:
            raise KeyError(
                f"未注册字段 {name!r}；已注册: {sorted(self._meta)}")
        return self._meta[name]

    def has(self, name: str) -> bool:
        return name in self._meta

    def names(self, role: Optional[str] = None,
              dimension: Optional[str] = None) -> List[str]:
        out = []
        for n, m in self._meta.items():
            if role and m.role != role:
                continue
            if dimension and m.dimension != dimension:
                continue
            out.append(n)
        return sorted(out)

    def merge(self, other: "FieldRegistry") -> "FieldRegistry":
        new = FieldRegistry(self._meta.values())
        new._meta.update(other._meta)
        return new

    def copy(self) -> "FieldRegistry":
        return FieldRegistry(self._meta.values())

    def to_records(self) -> List[Dict[str, str]]:
        return [m.to_dict() for m in self._meta.values()]

    @classmethod
    def from_records(cls, rows: Iterable[Dict[str, str]]) -> "FieldRegistry":
        return cls(FieldMeta(**{k: r.get(k, "") for k in
                                ("name", "dimension", "semantics", "role",
                                 "source", "doc")}) for r in rows)

    def __contains__(self, name: object) -> bool:
        return name in self._meta

    def __len__(self) -> int:
        return len(self._meta)


def default_registry() -> FieldRegistry:
    """量价基础字段（离线数据即含这些列）。"""
    reg = FieldRegistry()
    reg.register_fields([
        ("open", DIM_PRICE, SEM_VALUE, "开盘价"),
        ("high", DIM_PRICE, SEM_VALUE, "最高价"),
        ("low", DIM_PRICE, SEM_VALUE, "最低价"),
        ("close", DIM_PRICE, SEM_VALUE, "收盘价"),
        ("vwap", DIM_PRICE, SEM_VALUE, "成交均价 = 成交额/成交量"),
        ("volume", DIM_VOLUME, SEM_LIQUIDITY, "成交量"),
        ("amount", DIM_AMOUNT, SEM_LIQUIDITY, "成交额"),
        ("turnover", DIM_RATIO, SEM_LIQUIDITY, "换手率（若离线数据提供）"),
        ("rel_volume", DIM_RATIO, SEM_LIQUIDITY, "相对成交量 = 成交量/20日均量"),
        ("ret", DIM_RATIO, SEM_MOMENTUM, "日收益（收盘价环比）"),
        ("amplitude", DIM_RATIO, SEM_RISK, "振幅 =（最高-最低）/收盘"),
    ], source="market", role=ROLE_MARKET)
    reg.register_fields([
        ("size", DIM_SCORE, SEM_SIZE, "规模暴露 = log(20日均成交额) 横截面标准化"),
        ("beta", DIM_SCORE, SEM_RISK, "市场 β（滚动，横截面标准化）"),
        ("momentum", DIM_SCORE, SEM_MOMENTUM, "动量暴露（20 日收益，横截面标准化）"),
        ("resid_vol", DIM_SCORE, SEM_RISK, "特质波动（对市场回归残差的滚动波动）"),
        ("liquidity", DIM_SCORE, SEM_LIQUIDITY, "流动性（Amihud 非流动性，横截面标准化）"),
        ("reversal", DIM_SCORE, SEM_RISK, "反转暴露（近 5 日收益取负）"),
    ], source="risk", role=ROLE_EXPOSURE)
    return reg


# --------------------------------------------------------------------------
# 派生字段：不在原始数据里、但可由已有字段算出（中性化控件常用这些）
# --------------------------------------------------------------------------
def _derive_size(panel: "PanelData") -> pd.DataFrame:
    from . import ops
    return ops.cs_zscore(np.log(ops.rolling_unary(panel.field("amount"), 20, "mean")))


DERIVED_FIELDS: Dict[str, Any] = {"size": _derive_size}


def register_derived(name: str, fn: Any) -> None:
    """注册派生字段：``fn(panel) -> DataFrame``，按需计算并缓存。"""
    DERIVED_FIELDS[name] = fn


# --------------------------------------------------------------------------
# PIT（point-in-time）对齐
# --------------------------------------------------------------------------
def day_delta(lag_days: int) -> np.timedelta64:
    """``lag_days`` 个自然日的显式 timedelta。

    ``pd.Timedelta(days=n)`` 在本项目的 pandas 2.3 + numpy 2.5 组合下会抛
    ``DeprecationWarning: The 'generic' unit for NumPy timedelta is deprecated``
    （numpy 侧后续版本将直接报错），因此统一走 ``np.timedelta64(n, "D")``。
    """
    return np.timedelta64(int(lag_days), "D")


def asof_align(long_df: pd.DataFrame, dates: Sequence[pd.Timestamp],
               lag_days: int = 1, value_col: str = "value",
               symbol_col: str = "symbol", date_col: str = "ann_date") -> pd.DataFrame:
    """把「公告日 + 数值」长表投影成交易日面板（严格无前视）。

    规则：某字段在交易日 ``d`` 的可用取值 = 所有 ``ann_date <= d - lag_days``
    的记录中**最新一条**的值；``ann_date + lag_days`` 之前一律为 NaN。

    ``lag_days`` 用于模拟披露时滞（公告当晚/次日才可用，默认 1 天）。
    """
    if long_df.empty:
        return pd.DataFrame(index=pd.DatetimeIndex(dates), dtype=float)
    df = long_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["_eff_date"] = df[date_col] + day_delta(lag_days)
    wide = (df.pivot_table(index="_eff_date", columns=symbol_col,
                           values=value_col, aggfunc="last")
              .sort_index())
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    # 先把生效日并入索引再 ffill：保证生效日当天即可用（含非交易日生效的情形）
    full = wide.reindex(wide.index.union(idx)).ffill()
    return full.reindex(idx)


def coverage(frame: pd.DataFrame) -> pd.Series:
    """逐日有效值占比（数据质量维度之一）。"""
    if frame.shape[1] == 0:
        return pd.Series(dtype=float)
    return frame.notna().sum(axis=1) / float(frame.shape[1])


def residualize_cs(target: pd.DataFrame, controls: Sequence[pd.DataFrame],
                   min_stocks: int = 20, standardize: bool = True,
                   add_const: bool = True) -> pd.DataFrame:
    """逐日横截面 OLS 残差化（正交化）。

    用途（中信建投《量价 X 基本面统一框架》）：把量价因子对基本面因子正交化
    （或反向），只保留**增量信息**；以及中性化（对规模/行业正交化）——统一框架
    要求中性化算子只出现在表达式最外层，正交化的实现即本函数。

    目标与控件都会先做横截面标准化（``standardize=True``），使残差与控件量纲无关。
    """
    if not controls:
        return target.astype(float)
    from . import ops  # 局部导入避免模块级循环

    y = (ops.cs_zscore(target) if standardize else target).astype(float)
    xs = [(ops.cs_zscore(c.reindex_like(target)) if standardize
           else c.reindex_like(target)).astype(float) for c in controls]
    y_np = y.to_numpy(dtype=np.float64)
    x_np = [x.to_numpy(dtype=np.float64) for x in xs]
    out = np.full_like(y_np, np.nan, dtype=np.float64)
    for i in range(y_np.shape[0]):
        yy = y_np[i]
        cols = [x[i] for x in x_np]
        parts = ([np.ones_like(yy)] if add_const else []) + cols
        X = np.column_stack(parts)
        mask = np.isfinite(yy) & np.all(np.isfinite(X), axis=1)
        if int(mask.sum()) < max(min_stocks, X.shape[1] + 2):
            continue
        Xm, ym = X[mask], yy[mask]
        try:
            beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover
            continue
        out[i, mask] = ym - Xm @ beta
    return pd.DataFrame(out, index=y.index, columns=y.columns)


# --------------------------------------------------------------------------
# 面板容器
# --------------------------------------------------------------------------
class PanelData:
    """统一因子面板：多字段宽表 + 多周期前瞻收益 + 字段注册表。"""

    def __init__(self, fields: Dict[str, pd.DataFrame],
                 registry: Optional[FieldRegistry] = None,
                 forward_periods: Sequence[int] = (1, 5, 20),
                 name: str = "", industry: Optional[pd.Series] = None) -> None:
        if not fields:
            raise ValueError("PanelData 至少需要一个字段")
        aligned = self._align(fields)
        self.fields: Dict[str, pd.DataFrame] = aligned
        self.dates: pd.DatetimeIndex = next(iter(aligned.values())).index
        self.symbols: List[str] = list(next(iter(aligned.values())).columns)
        self.registry = registry or default_registry()
        self.forward_periods = tuple(int(h) for h in forward_periods)
        self.name = name
        self.industry = industry  # symbol -> 行业名（可选，供行业中性化）
        # 合成数据暴露的潜在状态（真实数据为空）。持有 ``mu``（真实预期收益
        # 状态）的用意是：让合成财报、合成概念数据能挂到**同一个**驱动源上，
        # 从而"跨域因子能挖到真信号"这件事可被检验，而不是靠噪声蒙。
        self.latent: Dict[str, Any] = {}
        self._fwd: Dict[int, pd.DataFrame] = {}
        self._build_forward_returns()

    # -- 构建 --
    @staticmethod
    def _align(fields: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        idx = None
        cols = None
        for f in fields.values():
            idx = f.index if idx is None else idx.union(f.index)
            cols = f.columns if cols is None else cols.union(f.columns)
        out: Dict[str, pd.DataFrame] = {}
        for k, f in fields.items():
            out[k] = f.reindex(index=idx, columns=cols).astype(float)
        return out

    def _build_forward_returns(self) -> None:
        close = self.fields.get("close")
        if close is None:
            return
        for h in self.forward_periods:
            self._fwd[h] = close.shift(-h) / close - 1.0

    @classmethod
    def from_kline(cls, kline: pd.DataFrame,
                   forward_periods: Sequence[int] = (1, 5, 20),
                   registry: Optional[FieldRegistry] = None,
                   name: str = "") -> "PanelData":
        """长表行情（date/symbol/OHLCV，可含 amount/turnover）→ 宽表面板。"""
        df = kline.copy()
        df["date"] = pd.to_datetime(df["date"])
        need = {"open", "high", "low", "close", "volume"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"行情数据缺少列: {sorted(missing)}")

        def _pivot(col: str) -> Optional[pd.DataFrame]:
            if col not in df.columns:
                return None
            return df.pivot_table(index="date", columns="symbol", values=col,
                                  aggfunc="last").sort_index()

        fields: Dict[str, pd.DataFrame] = {}
        for col in ("open", "high", "low", "close", "volume", "amount",
                    "turnover"):
            wide = _pivot(col)
            if wide is not None:
                fields[col] = wide
        if "turnover" in fields:
            fields["rel_volume"] = fields["turnover"]
        if "amount" not in fields:
            fields["amount"] = fields["volume"] * fields["close"]
        fields["vwap"] = (fields["amount"] / fields["volume"].where(
            fields["volume"].abs() > 1e-12))
        fields["ret"] = fields["close"].pct_change()
        if "turnover" not in fields:
            vol_ma = fields["volume"].rolling(20, min_periods=5).mean()
            fields["rel_volume"] = fields["volume"] / vol_ma.where(vol_ma > 0)
        fields["amplitude"] = ((fields["high"] - fields["low"])
                               / fields["close"].where(fields["close"] > 0))
        return cls(fields, registry=registry, forward_periods=forward_periods,
                   name=name)

    @classmethod
    def synthetic(cls, n_symbols: int = 60, n_days: int = 420, seed: int = 42,
                  forward_periods: Sequence[int] = (1, 5, 20),
                  name: str = "synthetic") -> "PanelData":
        """可复现合成面板（离线演示 / 测试用）。

        刻意埋入**四条方向明确、可独立检验**的横截面结构。之所以要四条而不是
        一条，是因为网格搜索引擎必须同时满足两件事：能挖到真信号，且能分辨
        方向与衰减——只有一条正信号时，任何"总挑 IC 最高的表达式"的实现都能
        蒙对。

        1. **横截面动量（正）**：每只股票有缓慢变化的预期收益状态 ``mu``
           （AR(1)，rho=0.96），过去 60 日收益能部分揭示 ``mu``，故对长周期
           前瞻收益为正 IC。
        2. **低波动异象（负）**：特质波动越高的股票日频预期收益越低，
           故 ``ts_std(ret, w)`` / 振幅类因子为负 IC。
        3. **流动性（负）**：成交量与近期绝对收益正相关，故成交额类因子
           与未来收益负相关。
        4. **短期反转（负且随持有期衰减）**：收益含 ``-phi*u_{t-1}`` 的
           买卖价差反弹成分，使近 5 日收益负向预测次日收益，且 IC 随
           持有期明显衰减（用于检验 IC 衰减与半衰期指标）。

        数据生成不含未来信息：任一时点可见的字段只依赖该时点及之前的信息。
        """
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2021-01-04", periods=n_days)
        symbols = [f"SYN{i:03d}" for i in range(n_symbols)]

        # 市场因子：波动聚集 + 轻微自相关
        mkt_ret = np.empty(n_days)
        vol = 0.011
        for t in range(n_days):
            vol = 0.94 * vol + 0.06 * 0.011 + 0.0004 * rng.standard_normal()
            vol = float(min(max(vol, 0.004), 0.05))
            mkt_ret[t] = (0.15 * mkt_ret[t - 1] if t else 0.0) \
                + rng.standard_normal() * vol

        beta = rng.uniform(0.6, 1.4, size=n_symbols)
        idio_vol = rng.uniform(0.008, 0.028, size=n_symbols)
        vol_z = (idio_vol - idio_vol.mean()) / (idio_vol.std() + 1e-12)

        # 结构 1：缓慢变化的预期收益状态（动量来源）
        # sd_mu 与 phi 需匹配：价差反弹（结构 4）会在**长形成窗口**里污染动量，
        # 若 mu 太弱，60 日动量会被反转成分抹成负 IC。
        rho, sd_mu = 0.96, 0.0018
        mu = np.zeros((n_days, n_symbols))
        for t in range(n_days):
            prev = mu[t - 1] if t else 0.0
            mu[t] = rho * prev + sd_mu * math.sqrt(1.0 - rho ** 2) \
                * rng.standard_normal(n_symbols)

        # 结构 4：价差反弹成分（短期反转来源）
        u = rng.standard_normal((n_days, n_symbols)) * idio_vol[None, :]
        bounce = np.zeros_like(u)
        bounce[1:] = -0.18 * u[:-1]

        # 结构 2：低波动异象 → 高波动股票日频预期收益更低
        drift = mu - 0.0004 * vol_z[None, :]
        rets = drift + beta[None, :] * mkt_ret[:, None] + u + bounce

        ret_df = pd.DataFrame(rets, index=dates, columns=symbols)
        close = 10.0 * (1.0 + ret_df).cumprod()
        # 结构 3：成交量随近期绝对收益放大（成交与波动同源）
        absorb = np.abs(ret_df).rolling(5, min_periods=1).mean().to_numpy()
        rel = idio_vol[None, :] / idio_vol.mean()
        volume = (1e6 * rel * (1.0 + 6.0 * absorb)
                  * (1.0 + 0.20 * rng.standard_normal((n_days, n_symbols))))
        volume = np.clip(volume, 1.0, None)
        amount = volume * close.to_numpy()
        spread = np.abs(rng.standard_normal((n_days, n_symbols))) * 0.004

        fields = {
            "open": close * (1.0 - spread / 2),
            "high": close * (1.0 + spread),
            "low": close * (1.0 - spread),
            "close": close,
            "volume": pd.DataFrame(volume, index=dates, columns=symbols),
            "amount": pd.DataFrame(amount, index=dates, columns=symbols),
        }
        panel = cls.from_kline(cls._to_long(fields),
                               forward_periods=forward_periods, name=name)
        # 暴露潜在状态：合成财报（fundamental.py）与合成概念（concept.py）
        # 都挂到 ``mu`` 上，保证跨域结构可检验
        panel.latent = {"mu": mu, "idio_vol": idio_vol, "vol_z": vol_z,
                        "beta": beta, "mkt_ret": mkt_ret}
        return panel

    @staticmethod
    def _to_long(fields: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """宽表集合 → 长表（date/symbol/各字段），供 from_kline 复用同一入口。"""
        long: Optional[pd.DataFrame] = None
        for name, wide in fields.items():
            piece = (wide.rename_axis(index="date", columns="symbol")
                         .reset_index()
                         .melt(id_vars="date", var_name="symbol",
                               value_name=name))
            long = piece if long is None else long.merge(
                piece, on=["date", "symbol"], how="outer")
        assert long is not None
        return long

    # -- 访问 --
    def field(self, name: str) -> pd.DataFrame:
        """取字段面板；若为已注册的派生字段（如 ``size``）则按需计算并缓存。"""
        if name in self.fields:
            return self.fields[name]
        if name in DERIVED_FIELDS:
            frame = (DERIVED_FIELDS[name](self)
                     .reindex(index=self.dates, columns=self.symbols)
                     .astype(float))
            self.fields[name] = frame
            return frame
        raise KeyError(f"面板中没有字段 {name!r}；现有: {sorted(self.fields)}；"
                       f"可派生: {sorted(DERIVED_FIELDS)}")

    def register_derived(self, name: str, fn: Any) -> None:
        """为派生字段登记计算函数（中性化控件据此按名引用）。"""
        register_derived(name, fn)

    def __contains__(self, name: object) -> bool:
        return name in self.fields or name in DERIVED_FIELDS

    def add_field(self, name: str, frame: pd.DataFrame,
                  meta: Optional[FieldMeta] = None) -> None:
        self.fields[name] = frame.reindex(index=self.dates,
                                          columns=self.symbols).astype(float)
        if meta is not None:
            self.registry.register(meta)

    def fwd(self, horizon: int = 1) -> pd.DataFrame:
        """前瞻 h 日收益（用于 IC/分组/风险回归的标签）。"""
        if horizon not in self._fwd:
            close = self.fields.get("close")
            if close is None:
                raise KeyError("面板无 close 字段，无法构造前瞻收益")
            self._fwd[horizon] = close.shift(-horizon) / close - 1.0
        return self._fwd[horizon]

    @property
    def n_dates(self) -> int:
        return len(self.dates)

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    def market_ret(self) -> pd.Series:
        """等权市场收益（风险模型的市场因子）。"""
        return self.field("ret").mean(axis=1)

    def size_exposure(self) -> pd.DataFrame:
        """规模暴露（对数成交额横截面标准化）：中性化的默认控制变量。"""
        if "amount" not in self.fields:
            raise KeyError("面板无 amount 字段，无法构造规模暴露")
        return self.field("size")

    def slice_dates(self, start: Optional[str] = None,
                    end: Optional[str] = None) -> "PanelData":
        mask = pd.Series(True, index=self.dates)
        if start:
            mask &= self.dates >= pd.Timestamp(start)
        if end:
            mask &= self.dates <= pd.Timestamp(end)
        sub = self.__class__.__new__(self.__class__)
        sub.fields = {k: v.loc[mask] for k, v in self.fields.items()}
        sub.dates = self.dates[mask]
        sub.symbols = list(self.symbols)
        sub.registry = self.registry
        sub.forward_periods = self.forward_periods
        sub.name = self.name
        sub.industry = self.industry
        sub._fwd = {h: f.loc[mask] for h, f in self._fwd.items()}
        if not sub._fwd:  # 面板未预置前瞻收益时按切片重算
            sub._build_forward_returns()
        return sub

    def describe(self) -> Dict[str, Any]:
        rows = {}
        for name, frame in self.fields.items():
            meta = self.registry.get(name) if self.registry.has(name) else None
            rows[name] = {
                "role": meta.role if meta else "-",
                "dimension": meta.dimension if meta else "-",
                "semantics": meta.semantics if meta else "-",
                "coverage": round(float(coverage(frame).mean()), 4),
                "nan_ratio": round(float(frame.isna().mean().mean()), 4),
            }
        return {"panel": self.name, "n_dates": self.n_dates,
                "n_symbols": self.n_symbols,
                "forward_periods": list(self.forward_periods),
                "roles": {r: len(self.registry.names(role=r))
                          for r in ROLE_LABELS},
                "fields": rows}
