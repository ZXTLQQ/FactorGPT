"""内置 Benchmark 与因子动物园（P2）：横向对比主流因子，验证增量信息。

学生量化项目常被质疑「你的因子是不是只是已知因子的翻版？」。本模块内置一组
学术界/业界公认的经典因子（动量、反转、波动率、流动性、偏度、振幅），与
精炼厂合成的复合因子做横向对比，并量化复合因子在剔除这些已知因子后的
「增量信息」（incremental IC），证明我们产出的因子并非重复已知因子。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine.backtest import FactorBacktester

logger = logging.getLogger("factor_gpt.factor_zoo")


class FactorZoo:
    """经典因子动物园：提供 Baseline 因子并与复合因子对比。"""

    ZOO_FACTORS = ["mom_20", "reversal_5", "vol_20", "liquidity_amihud", "skew_20", "hl_range_5"]

    def build(self, kline: pd.DataFrame) -> dict:
        """从行情长表派生经典因子，返回 name -> (date,symbol) 索引 Series。"""
        needed = ["date", "symbol", "close"]
        for c in ("volume", "amount", "open", "high", "low"):
            if c in kline.columns:
                needed.append(c)
        df = kline[needed].copy()
        if "volume" not in df.columns:
            df["volume"] = 1.0
        if "amount" not in df.columns:
            df["amount"] = df["close"] * df["volume"]
        df["date"] = df["date"].astype(str)
        df = df.sort_values(["symbol", "date"])
        g = df.groupby("symbol")
        df["ret"] = g["close"].pct_change()
        df["mom_20"] = g["close"].transform(lambda x: x.pct_change(20))
        df["reversal_5"] = -g["close"].transform(lambda x: x.pct_change(5))
        df["vol_20"] = g["ret"].transform(lambda x: x.rolling(20).std())
        df["liquidity_amihud"] = (
            (df["ret"].abs() / df["volume"].replace(0, np.nan))
            .groupby(df["symbol"]).transform(lambda x: x.rolling(21).mean())
        )
        df["skew_20"] = g["ret"].transform(lambda x: x.rolling(20).skew())
        if "high" in df.columns and "low" in df.columns:
            df["hl_range_5"] = (
                g["high"].transform(lambda x: x.rolling(5).max())
                - g["low"].transform(lambda x: x.rolling(5).min())
            ) / g["close"].transform(lambda x: x.rolling(5).mean())
        else:
            df["hl_range_5"] = np.nan

        pool = {}
        idx = ["date", "symbol"]
        for col in self.ZOO_FACTORS:
            s = df.set_index(idx)[col].dropna()
            if not s.empty:
                pool[col] = s
        return pool

    def compare_to_zoo(self, composite: pd.Series, kline: pd.DataFrame) -> dict:
        """对比复合因子与动物园因子，量化增量信息。

        Returns:
            dict 含 zoo_icir（各基准 ICIR）、composite_icir、max_zoo_icir（最强基准）、
            incremental_icir（剔除全部基准后的残差因子 ICIR）、max_abs_corr（与最强基准相关性）。
        """
        bt = FactorBacktester()
        zoo = self.build(kline)

        zoo_icir = {}
        for name, s in zoo.items():
            m = bt.evaluate(kline, s)
            zoo_icir[name] = m.get("icir")

        comp_m = bt.evaluate(kline, composite)
        comp_icir = comp_m.get("icir")

        # 相关性 + 增量 IC（残差正交化）
        common = composite.index
        for s in zoo.values():
            common = common.intersection(s.index)
        if len(common) < 50:
            logger.warning("因子动物园样本不足，跳过增量信息计算")
            return {
                "zoo_icir": zoo_icir,
                "composite_icir": comp_icir,
                "max_zoo_icir": (max(zoo_icir.values()) if zoo_icir else float("nan")),
                "incremental_icir": float("nan"),
                "max_abs_corr": float("nan"),
                "note": "样本不足",
            }

        y = composite.reindex(common).to_numpy(dtype=float)
        X = np.column_stack([zoo[n].reindex(common).to_numpy(dtype=float) for n in zoo if n in zoo_icir])
        X = np.nan_to_num(X)
        # 截面去均值后正交化（去除已知因子线性信息）
        Xc = X - X.mean(axis=0, keepdims=True)
        yc = y - y.mean()
        try:
            beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
            residual = yc - Xc @ beta
        except Exception:  # noqa: BLE001
            residual = yc
        # 残差因子 ICIR：以残差作为新因子，重算 per-date rank IC
        resid_series = pd.Series(residual, index=common)
        resid_icir = bt.evaluate(kline, resid_series).get("icir")

        # 与最强基准的相关性
        max_corr = 0.0
        for n in zoo:
            if n in zoo_icir:
                zz = zoo[n].reindex(common).to_numpy(dtype=float)
                if np.std(zz) > 0 and np.std(y) > 0:
                    c = float(np.corrcoef(y, zz)[0, 1])
                    max_corr = max(max_corr, abs(c))

        return {
            "zoo_icir": zoo_icir,
            "composite_icir": comp_icir,
            "max_zoo_icir": float(max(zoo_icir.values())) if zoo_icir else float("nan"),
            "incremental_icir": resid_icir,
            "max_abs_corr": max_corr,
            "has_incremental_info": bool(
                (not np.isnan(resid_icir)) and resid_icir > 0
                and (np.isnan(comp_icir) or resid_icir > 0.5 * comp_icir if not np.isnan(comp_icir) else True)
            ),
        }
