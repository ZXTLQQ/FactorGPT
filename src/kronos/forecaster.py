"""Kronos 预测因子核心实现 (src/kronos/forecaster.py)。

设计要点
--------
1. `KronosForecaster` 负责加载真实 Kronos 模型(NeoQuasar/Kronos-mini 等)并对单只
   标的的历史 OHLCV 预测未来收益; 在真实模型不可用(缺依赖/权重下载失败)时自动降级为
   stub —— 一个纯几何动量的可复现代理, 使流水线不依赖 GPU/网络也能跑通。
2. `predict_panel` 对面板数据(多标的 × 多日期)逐标的、逐交易日滚动预测, 同时给出
   真实未来收益, 便于直接评估 IC / 多空收益。
3. `attach_kronos_factor` 把 Kronos 预测收益作为一个新因子 `KRONOS_PRED` 接入
   FactorGPT 的 OreStock 因子池, 供后续 refinery 流水线使用。

Kronos 上游接口约定(morrisluo/kronos, 基于 shiyu-coder/Kronos)
-------------------------------------------------------------
- 模型代码位于 `third_party/kronos/model/`(由本仓库随附, 来自 GitHub fork),
  `from model import Kronos, KronosTokenizer, KronosPredictor`。
- `Kronos` / `KronosTokenizer` 为 PyTorchModelHubMixin, 经 HuggingFace 加载:
      tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
      model     = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
      predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
- `predictor.predict(df, x_timestamp, y_timestamp, pred_len, T, top_k, top_p,
   sample_count, verbose)` 输入需含小写列 open/high/low/close(volume/amount 缺失则补 0),
   x_timestamp/y_timestamp 为 pandas datetime 序列, 返回未来 pred_len 步的 OHLCV DataFrame。
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger("factor_gpt.kronos")

# Kronos 要求的输入列(顺序无关, 但列名须一致)
KRONOS_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]

# stub 默认回看窗口(用于几何动量代理)
_STUB_LOOKBACK = 20

# Kronos 内部 tokenizer 所在的 HuggingFace 仓库(与模型权重分开托管)
_DEFAULT_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"


class KronosForecaster:
    """Kronos 预测因子封装: 真实模型优先, 失败时降级 stub。"""

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.model_name = cfg.get("model_name", "NeoQuasar/Kronos-mini")
        self.tokenizer_name = cfg.get("tokenizer_name", _DEFAULT_TOKENIZER)
        self.device = cfg.get("device", "auto")
        self.horizon = int(cfg.get("horizon", 5))
        self.future_return_window = int(
            cfg.get("future_return_window", self.horizon)
        )
        self.cache_dir = cfg.get("cache_dir")
        self.hf_endpoint = cfg.get("hf_endpoint", "")
        self.fallback_to_stub = bool(cfg.get("fallback_to_stub", True))

        self._model = None
        self._using_stub = False
        self._tried_load = False

    # ------------------------------------------------------------------
    # 真实模型加载(惰性)
    # ------------------------------------------------------------------
    def _load_real(self) -> None:
        if self._tried_load:
            return
        self._tried_load = True
        # 允许从第三方目录(third_party/vendor)导入 Kronos 模型代码
        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for _p in (
            os.path.join(_root, "third_party", "kronos"),
            os.path.join(_root, "vendor", "kronos"),
        ):
            if os.path.isdir(_p) and _p not in sys.path:
                sys.path.insert(0, _p)
        try:
            import torch  # noqa: F401
            # 本地随附的 Kronos 模型代码: third_party/kronos/model/
            from model import Kronos, KronosPredictor, KronosTokenizer
        except Exception as e:  # pragma: no cover - 依赖缺失
            if self.fallback_to_stub:
                logger.warning(
                    "Kronos 依赖(model 包/torch)未就绪, 降级为 stub: %s", e
                )
                self._using_stub = True
                return
            raise
        # 设置 HuggingFace 镜像(国内网络友好)
        if self.hf_endpoint:
            os.environ["HF_ENDPOINT"] = self.hf_endpoint
        device = self.device
        if device in ("auto", "", None):
            device = "cuda" if (torch.cuda.is_available()) else "cpu"
        try:
            tokenizer = KronosTokenizer.from_pretrained(
                self.tokenizer_name, cache_dir=self.cache_dir or None
            )
            model = Kronos.from_pretrained(
                self.model_name, cache_dir=self.cache_dir or None
            )
            self._model = KronosPredictor(
                model, tokenizer, device=device, max_context=512
            )
            logger.info(
                "Kronos 真实模型已加载: %s (tokenizer=%s, device=%s)",
                self.model_name,
                self.tokenizer_name,
                device,
            )
        except Exception as e:  # pragma: no cover - 下载/加载失败
            if self.fallback_to_stub:
                logger.warning("Kronos 权重加载失败, 降级为 stub: %s", e)
                self._using_stub = True
            else:
                raise

    @property
    def using_stub(self) -> bool:
        """当前是否处于 stub(降级)模式。"""
        self._load_real()
        return self._using_stub

    # ------------------------------------------------------------------
    # 输入规范化
    # ------------------------------------------------------------------
    def _normalize_input(self, prices: pd.DataFrame, lookback: int | None = None) -> pd.DataFrame:
        """把任意 OHLCV 面板/序列整理为 Kronos 要求的输入 DataFrame。

        自动识别日期/价格/成交量列, 取最后 `lookback` 行(默认 60), 列名映射为
        Date/Open/High/Low/Close/Volume 且 Date 转为 datetime。
        """
        df = prices.copy()
        lower_map = {c.lower(): c for c in df.columns}
        date_col = lower_map.get("date") or lower_map.get("datetime") or "Date"
        open_col = lower_map.get("open")
        high_col = lower_map.get("high")
        low_col = lower_map.get("low")
        close_col = lower_map.get("close")
        vol_col = lower_map.get("volume") or lower_map.get("vol")

        out = pd.DataFrame()
        out["Date"] = pd.to_datetime(df[date_col], errors="coerce")
        out["Open"] = pd.to_numeric(df[open_col], errors="coerce") if open_col else np.nan
        out["High"] = pd.to_numeric(df[high_col], errors="coerce") if high_col else np.nan
        out["Low"] = pd.to_numeric(df[low_col], errors="coerce") if low_col else np.nan
        out["Close"] = pd.to_numeric(df[close_col], errors="coerce")
        out["Volume"] = (
            pd.to_numeric(df[vol_col], errors="coerce") if vol_col else 0.0
        )
        out = out.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
        # 用 close 近似缺失的 OHLC
        for c in ("Open", "High", "Low"):
            out[c] = out[c].fillna(out["Close"])
        out["High"] = out[["High", "Close"]].max(axis=1)
        out["Low"] = out[["Low", "Close"]].min(axis=1)
        if lookback is not None and len(out) > lookback:
            out = out.iloc[-lookback:].reset_index(drop=True)
        return out

    # ------------------------------------------------------------------
    # 预测核心
    # ------------------------------------------------------------------
    def predict_forward_return(self, prices: pd.DataFrame, lookback: int = 60) -> float:
        """对单只标的的历史 OHLCV 预测未来 `future_return_window` 根 K 线的累计收益率。

        Args:
            prices: 含日期/OHLC/成交量的 DataFrame(升序), 至少含 close 列。
            lookback: 喂给模型的回看根数(默认 60)。

        Returns:
            预测的未来累计收益率(float)。失败时返回 stub 值(或抛错, 取决于配置)。
        """
        self._load_real()
        norm = self._normalize_input(prices, lookback=lookback)
        if len(norm) < 2:
            return 0.0
        if self._using_stub:
            return self._stub_forward_return(norm)

        try:
            pred_len = int(self.future_return_window)
            pred_df = self._predict_via_model(norm, pred_len=pred_len)
            pred_close = pd.to_numeric(pred_df["close"], errors="coerce").dropna().values
            if len(pred_close) == 0:
                raise ValueError("Kronos 预测结果为空")
            last_close = float(norm["Close"].iloc[-1])
            if last_close == 0:
                raise ValueError("最后收盘价为 0")
            # 取预测窗口末端的收盘价, 作为未来 horizon 步的预测收盘价
            target = float(pred_close[-1])
            return float(target / last_close - 1.0)
        except Exception as e:  # pragma: no cover - 推理异常
            if self.fallback_to_stub:
                logger.warning("Kronos 真实预测异常, 降级 stub: %s", e)
                return self._stub_forward_return(norm)
            raise

    def _predict_via_model(self, norm: pd.DataFrame, pred_len: int) -> pd.DataFrame:
        """调用上游 KronosPredictor 对未来 pred_len 步 OHLCV 做自回归预测。"""
        df = pd.DataFrame(
            {
                "open": norm["Open"].astype(float).to_numpy(),
                "high": norm["High"].astype(float).to_numpy(),
                "low": norm["Low"].astype(float).to_numpy(),
                "close": norm["Close"].astype(float).to_numpy(),
                "volume": norm["Volume"].astype(float).to_numpy(),
            }
        )
        x_ts = norm["Date"].reset_index(drop=True)
        last_date = x_ts.iloc[-1]
        # 未来时间戳: 按日递进(pred_len 步)。Kronos 训练于 5 分钟 K 线, 日线属分布外,
        # 但推理仍可运行(时间嵌入取 minute/hour=0); 此处以日频生成未来时间戳。
        y_ts = pd.date_range(last_date, periods=pred_len + 1, freq="D")[1:].to_series().reset_index(drop=True)
        pred_df = self._model.predict(
            df=df,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=pred_len,
            T=1.0,
            top_k=0,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )
        return pred_df

    def _stub_forward_return(self, norm: pd.DataFrame) -> float:
        """降级代理: 纯几何动量(近 N 日累计收益), 无前瞻泄漏。"""
        close = norm["Close"].to_numpy(dtype=float)
        w = min(_STUB_LOOKBACK, len(close) - 1)
        if w <= 0:
            return 0.0
        return float(close[-1] / close[-1 - w] - 1.0)

    # ------------------------------------------------------------------
    # 面板滚动预测
    # ------------------------------------------------------------------
    def predict_panel(
        self,
        kline: pd.DataFrame,
        symbol_col: str = "symbol",
        date_col: str = "date",
        lookback: int = 60,
        n_eval: int | None = None,
    ) -> pd.DataFrame:
        """对面板数据逐标的、逐交易日滚动预测未来收益。

        对每个标的, 在每一个评估日 t 用截至 t 的历史(最后 `lookback` 根)预测未来
        `future_return_window` 日收益, 同时记录真实未来收益用于评估。

        Args:
            kline: 面板 DataFrame, 含 symbol/date/close(及可选 open/high/low/volume)。
            n_eval: 每个标的只评估最后 n_eval 个交易日(降低真实模型推理成本);
                    None 表示评估全部日期(适合 stub/小样本)。

        Returns:
            DataFrame: [symbol, date, kronos_pred_ret, fwd_ret_realized]
        """
        kline = kline.copy()
        lower_map = {c.lower(): c for c in kline.columns}
        s_col = lower_map.get(symbol_col.lower(), symbol_col)
        d_col = lower_map.get(date_col.lower(), date_col)
        kline[d_col] = pd.to_datetime(kline[d_col], errors="coerce")
        kline = kline.dropna(subset=[d_col]).sort_values([s_col, d_col])

        close_col = lower_map.get("close")
        records = []
        for sym, g in kline.groupby(s_col):
            g = g.sort_values(d_col).reset_index(drop=True)
            n = len(g)
            if n < lookback + self.future_return_window + 1:
                continue
            idx_dates = g[d_col].tolist()
            eval_positions = (
                range(n) if n_eval is None else range(max(0, n - n_eval), n)
            )
            for t in eval_positions:
                hist = g.iloc[: t + 1]
                pred_ret = self.predict_forward_return(hist, lookback=lookback)
                # 真实未来收益(无前瞻)
                tgt_idx = min(t + self.future_return_window, n - 1)
                c0 = float(g[close_col].iloc[t])
                c1 = float(g[close_col].iloc[tgt_idx])
                real_ret = (c1 / c0 - 1.0) if c0 != 0 else 0.0
                records.append(
                    {
                        "symbol": sym,
                        "date": idx_dates[t],
                        "kronos_pred_ret": pred_ret,
                        "fwd_ret_realized": real_ret,
                    }
                )
        return pd.DataFrame(records)


# ----------------------------------------------------------------------
# 接入 FactorGPT 因子池
# ----------------------------------------------------------------------
def attach_kronos_factor(ore, cfg: dict | None = None, n_eval: int | None = None) -> object:
    """把 Kronos 预测收益作为新因子 `KRONOS_PRED` 接入 OreStock 因子池。

    在 refinery PART-01(数据底座) 之后调用即可让后续 RPN 求值引擎把 Kronos 预测
    当作一个候选因子参与筛选与合成。

    Args:
        ore: pipeline.schema.OreStock 实例(test_kline / train_kline 至少其一非空)。
        cfg: 完整 config 字典(读取 kronos 段); 为空则使用默认配置。
        n_eval: 每只标的仅评估最后 n_eval 个交易日(None=全部)。
                真实模型推理成本较高，未显式指定时自动限制为 30 日以控制耗时。

    Returns:
        传入的 ore(已就地更新 factor_pool), 便于链式调用。
    """
    kcfg = (cfg or {}).get("kronos", {}) if isinstance(cfg, dict) else {}
    forecaster = KronosForecaster(kcfg)

    kline = getattr(ore, "test_kline", None)
    if kline is None or len(kline) == 0:
        kline = getattr(ore, "train_kline", None)
    if kline is None or len(kline) == 0:
        logger.warning("attach_kronos_factor: 无可用 kline, 跳过")
        return ore

    # 真实模型推理成本高: 未指定时, 仅评估每只标的最后 30 个交易日(训练集+测试集合计)
    if n_eval is None:
        n_eval = 30 if not forecaster.using_stub else None
    preds = forecaster.predict_panel(kline, n_eval=n_eval)
    if preds.empty:
        logger.warning("attach_kronos_factor: Kronos 预测为空, 跳过")
        return ore

    # 组装为以 (date, symbol) 为索引的因子序列
    factor = preds.set_index(["date", "symbol"])["kronos_pred_ret"].rename("KRONOS_PRED")
    factor.index = factor.index.set_levels(
        [pd.to_datetime(factor.index.levels[0]), factor.index.levels[1]]
    )

    pool = dict(ore.factor_pool or {})
    pool["KRONOS_PRED"] = factor
    ore.factor_pool = pool
    logger.info(
        "Kronos 因子 KRONOS_PRED 已接入因子池 (mode=%s, 样本=%d)",
        "stub" if forecaster.using_stub else "real",
        len(factor),
    )
    return ore
