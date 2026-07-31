"""
数据清洗器 (DataCleaner)

提供金融因子数据清洗与预处理工具，包括去空值、缩尾处理、中性化、
标准化和停牌过滤等功能。所有方法返回清洗后的新 DataFrame，不修改原始数据。
"""
import logging

from typing import Optional, List, Union

import pandas as pd
import numpy as np


class DataCleaner:
    """金融数据清洗与预处理工具集。

    提供因子工程中常用的数据清洗流水线，包括缺失值处理、异常值处理、
    标准化、行业市值中性化等。
    """

    # ------------------------------------------------------------------
    # K线数据清洗
    # ------------------------------------------------------------------

    @staticmethod
    def clean_minute_kline(df: pd.DataFrame) -> pd.DataFrame:
        """清洗分钟K线数据：去空值、剔除无成交记录、计算分钟收益率。

        与日线数据不同，单分钟零成交量不直接视为停牌（可能是交易不活跃），
        仅剔除开盘/收盘价格异常为空的记录。

        处理步骤：
        1. 删除核心价格列为空的行
        2. 剔除开盘/收盘异常为 0 的记录
        3. 按 symbol 分组，在每个交易日内计算分钟对数收益率
        4. 删除 ret 为 NA 的行

        Args:
            df: 分钟K线 DataFrame，需包含 date / time / symbol /
                open / high / low / close / volume 列。

        Returns:
            清洗后的 DataFrame，新增 ret 列（分钟对数收益率）。
        """
        if df.empty:
            return df

        df = df.copy()

        # 1) 删除核心价格列为空的行
        core_cols = ["open", "high", "low", "close"]
        existing = [c for c in core_cols if c in df.columns]
        if existing:
            df = df.dropna(subset=existing)

        # 2) 剔除开盘/收盘为 0 的异常记录（可能数据缺失）
        if "open" in df.columns and "close" in df.columns:
            df = df[(df["open"] > 0.01) & (df["close"] > 0.01)]

        # 3) 日期时间排序
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # 4) 按 symbol + date 分组计算分钟对数收益率
        if "close" in df.columns and "symbol" in df.columns:
            df = df.sort_values(["symbol", "date", "time"] if "time" in df.columns else ["symbol", "date"])
            # 在同一 symbol 内，跨日时收益率重置为 NaN（避免跨日收益率污染）
            df["ret"] = df.groupby("symbol")["close"].transform(
                lambda x: np.log(x / x.shift(1))
            )
            # 每天第一根K线的 ret 设为 NaN（避免跨日计算）
            if "date" in df.columns:
                day_first_mask = df.groupby(["symbol", "date"]).cumcount() == 0
                df.loc[day_first_mask, "ret"] = np.nan

        # 5) 删除 ret 为 NA 的行
        if "ret" in df.columns:
            df = df.dropna(subset=["ret"])

        return df.reset_index(drop=True)

    @staticmethod
    def minute_to_daily_factors(
        df: pd.DataFrame,
        factor_funcs: Optional[dict] = None,
    ) -> pd.DataFrame:
        """将分钟K线数据聚合成日内因子（daily factors）。

        从分钟级数据中提取日内特征，如日内波动率、开盘缺口、尾盘效应等。

        Args:
            df: 清洗后的分钟K线 DataFrame，需包含 date / symbol / time /
                open / high / low / close / volume / ret 列。
            factor_funcs: 自定义因子计算函数字典，key 为因子名，
                          value 为函数（接收分组后的 minute_df，返回标量）。
                          为 None 时使用预设因子集。

        Returns:
            DataFrame，每行一个股票-日期，列包含 date / symbol
            及各日内因子值。index 为 (date, symbol) 的 MultiIndex。

        Examples:
            >>> cleaner = DataCleaner()
            >>> daily_factors = cleaner.minute_to_daily_factors(minute_df)
            >>> print(daily_factors.columns)
            Index(['intraday_vol', 'open_gap', 'close_effect', ...])
        """
        if df.empty:
            return pd.DataFrame()

        df = df.copy()

        # 预设日内因子
        if factor_funcs is None:

            def intraday_volatility(group: pd.DataFrame) -> float:
                """日内波动率：当日分钟收益率标准差 * sqrt(240)"""
                if "ret" in group.columns and len(group) > 1:
                    return float(group["ret"].std() * np.sqrt(240))
                return np.nan

            def open_gap(group: pd.DataFrame) -> float:
                """开盘跳空：首笔开盘价相对昨日收盘的偏离"""
                if "open" in group.columns and "close" in group.columns:
                    first_open = group["open"].iloc[0]
                    last_close = group["close"].iloc[-1]
                    if last_close > 0:
                        return float(np.log(first_open / last_close))
                return np.nan

            def afternoon_effect(group: pd.DataFrame) -> float:
                """午盘效应：下午成交量占比"""
                if "time" not in group.columns or "volume" not in group.columns:
                    return np.nan
                group = group.copy()
                # 提取小时
                hours = pd.to_datetime(group["time"], format="%H:%M").dt.hour
                afternoon_mask = hours >= 13
                total_vol = group["volume"].sum()
                if total_vol > 0:
                    return float(group.loc[afternoon_mask, "volume"].sum() / total_vol)
                return np.nan

            def tail_effect(group: pd.DataFrame) -> float:
                """尾盘效应：最后30分钟收益率"""
                if "ret" not in group.columns or "time" not in group.columns:
                    return np.nan
                group = group.copy()
                times = pd.to_datetime(group["time"], format="%H:%M")
                tail_mask = times >= pd.Timestamp("14:30")
                if tail_mask.sum() > 0:
                    return float(group.loc[tail_mask, "ret"].sum())
                return np.nan

            def volume_concentration(group: pd.DataFrame) -> float:
                """成交量集中度：最高成交量分钟 / 平均分钟成交量"""
                if "volume" not in group.columns or len(group) < 2:
                    return np.nan
                avg_vol = group["volume"].mean()
                if avg_vol > 0:
                    return float(group["volume"].max() / avg_vol)
                return np.nan

            factor_funcs = {
                "intraday_volatility": intraday_volatility,
                "open_gap": open_gap,
                "afternoon_effect": afternoon_effect,
                "tail_effect": tail_effect,
                "volume_concentration": volume_concentration,
            }

        # 按 symbol + date 分组计算因子
        results = []
        for (symbol, date), group in df.groupby(["symbol", "date"]):
            row = {"symbol": symbol, "date": date}
            for name, func in factor_funcs.items():
                try:
                    row[name] = func(group)
                except Exception:
                    row[name] = np.nan
            results.append(row)

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        result_df["date"] = pd.to_datetime(result_df["date"])
        result_df = result_df.set_index(["date", "symbol"]).sort_index()
        return result_df

    # ------------------------------------------------------------------
    # K线数据清洗
    # ------------------------------------------------------------------

    @staticmethod
    def winsorize(
        series: Union[pd.Series, np.ndarray],
        pct: float = 0.01,
    ) -> pd.Series:
        """对数值序列进行缩尾处理（Winsorize）。

        将序列中低于 pct 分位数和高于 (1-pct) 分位数的值，
        替换为对应的分位数值，以消除极端异常值的影响。

        Args:
            series: 输入数值序列。
            pct: 缩尾百分比，默认 0.01（即 1%~99% 缩尾）。

        Returns:
            缩尾后的 Series，保持原始索引。
        """
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        series = series.copy()
        lower = series.quantile(pct)
        upper = series.quantile(1 - pct)
        # 仅对非 NaN 的值做缩尾
        mask = series.notna()
        series.loc[mask] = series.loc[mask].clip(lower=lower, upper=upper)
        return series

    # ------------------------------------------------------------------
    # 行业 + 市值中性化
    # ------------------------------------------------------------------

    @staticmethod
    def neutralize(
        factor: pd.Series,
        industry: pd.Series,
        mkt_cap: Optional[pd.Series] = None,
    ) -> pd.Series:
        """行业 + 市值中性化处理。

        使用 OLS 回归将因子对行业哑变量和市值做回归，取残差作为中性化后的因子值。

        Args:
            factor: 原始因子值 Series，index 为股票代码。
            industry: 行业分类 Series，index 为股票代码，值为行业名称/代码。
            mkt_cap: 市值 Series（对数），index 为股票代码。为 None 时仅做行业中性化。

        Returns:
            中性化后的因子值 Series，index 对齐输入。
        """
        try:
            from statsmodels.api import OLS, add_constant
        except ImportError:
            print("[DataCleaner] statsmodels 未安装，降级为手动去均值中性化")
            return DataCleaner._neutralize_simple(factor, industry, mkt_cap)

        # 对齐索引
        common_idx = factor.index.intersection(industry.index)
        if mkt_cap is not None:
            common_idx = common_idx.intersection(mkt_cap.index)

        if len(common_idx) < 10:
            logging.debug("[DataCleaner] 中性化可用样本不足（<10），返回原始因子")
            return factor

        y = factor.loc[common_idx]

        # 行业哑变量
        ind = industry.loc[common_idx]
        ind_dummies = pd.get_dummies(ind, drop_first=True).astype(float)

        # 构建设计矩阵 X
        if mkt_cap is not None:
            cap = mkt_cap.loc[common_idx]
            # 市值取对数
            cap_log = np.log(cap.replace(0, np.nan))
            cap_log = cap_log.fillna(cap_log.median())
            X = pd.concat([ind_dummies, cap_log.rename("ln_mkt_cap")], axis=1)
        else:
            X = ind_dummies

        X = add_constant(X)

        # 去除含 NaN 的行
        mask = y.notna() & X.notna().all(axis=1)
        y_clean = y[mask]
        X_clean = X[mask]

        if len(y_clean) < 10:
            logging.debug("[DataCleaner] 中性化有效样本不足（<10），返回原始因子")
            return factor

        try:
            model = OLS(y_clean, X_clean).fit()
            residuals = model.resid
            result = pd.Series(np.nan, index=factor.index)
            result.loc[residuals.index] = residuals
            return result
        except Exception as e:
            print(f"[DataCleaner] 中性化回归失败: {e}，降级为简单中性化")
            return DataCleaner._neutralize_simple(factor, industry, mkt_cap)

    @staticmethod
    def _neutralize_simple(
        factor: pd.Series,
        industry: pd.Series,
        mkt_cap: Optional[pd.Series] = None,
    ) -> pd.Series:
        """简单中性化：行业组内减去均值，再对市值做截面去均值。"""
        result = factor.copy()

        # 行业中性化
        common_idx = factor.index.intersection(industry.index)
        ind_aligned = industry.loc[common_idx]
        group_mean = factor.loc[common_idx].groupby(ind_aligned).transform("mean")
        result.loc[common_idx] = factor.loc[common_idx] - group_mean

        # 市值中性化（简单去均值回归）
        if mkt_cap is not None:
            common_idx2 = result.index.intersection(mkt_cap.index)
            if len(common_idx2) > 10:
                cap_log = np.log(mkt_cap.loc[common_idx2].replace(0, np.nan))
                cap_log = cap_log.fillna(cap_log.median())
                from statsmodels.api import OLS, add_constant
                y = result.loc[common_idx2].dropna()
                x = cap_log.loc[y.index]
                mask = y.notna() & x.notna()
                if mask.sum() > 10:
                    X = add_constant(x[mask])
                    model = OLS(y[mask], X).fit()
                    result.loc[y[mask].index] = model.resid

        return result

    # ------------------------------------------------------------------
    # Z-score 标准化
    # ------------------------------------------------------------------

    @staticmethod
    def standardize(series: pd.Series) -> pd.Series:
        """Z-score 标准化：减去均值后除以标准差。

        Args:
            series: 输入数值 Series。

        Returns:
            标准化后的 Series，均值为 0，标准差为 1。
        """
        series = series.copy()
        mask = series.notna()
        if mask.sum() < 2:
            return series
        mean_val = series.loc[mask].mean()
        std_val = series.loc[mask].std()
        if std_val == 0 or np.isnan(std_val):
            return series
        series.loc[mask] = (series.loc[mask] - mean_val) / std_val
        return series

    # ------------------------------------------------------------------
    # 停牌股票过滤
    # ------------------------------------------------------------------

    @staticmethod
    def filter_suspended(
        df: pd.DataFrame,
        threshold: float = 0.3,
    ) -> pd.DataFrame:
        """过滤长期停牌股票。

        计算每只股票的零成交量天数占比，超过 threshold 的股票视为长期停牌并剔除。

        Args:
            df: K线 DataFrame，需包含 symbol / volume / date 列。
            threshold: 停牌阈值，零成交量天数占比超过此值则过滤，默认 0.3。

        Returns:
            过滤后的 DataFrame，剔除长期停牌股票的所有记录。
        """
        if df.empty or "volume" not in df.columns or "symbol" not in df.columns:
            return df

        # 每只股票零成交量天数
        zero_vol = df.groupby("symbol")["volume"].apply(
            lambda x: (x == 0).sum()
        )
        total_days = df.groupby("symbol").size()
        zero_ratio = zero_vol / total_days

        bad_symbols = zero_ratio[zero_ratio > threshold].index.tolist()
        if bad_symbols:
            print(
                f"[DataCleaner] 过滤长期停牌股票 {len(bad_symbols)} 只: "
                f"{bad_symbols[:5]}{'...' if len(bad_symbols) > 5 else ''}"
            )

        keep = ~df["symbol"].isin(bad_symbols)
        return df[keep].reset_index(drop=True)
