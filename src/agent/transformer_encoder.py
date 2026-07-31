"""PART-02 采矿作业层 · 研磨车间 × 向量化表征 (TransformerEncoder)。

使用 Transformer 特征处理器（d_model=128，2 层，5 头自注意力）对逐股的时间序列
原始特征进行序列建模，输出向量化表征，并可直接派生一个候选 alpha 因子。

  * 真实实现基于 PyTorch nn.TransformerEncoder；
  * 若环境无 torch，自动降级为「确定性线性投影 + 时序均值」的 numpy 实现，
    保证流水线在任意机器上均可运行（演示/CI 场景）。
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("factor_gpt.transformer")

DATE = "date"
SYMBOL = "symbol"

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False
    nn = None
    torch = None


if _HAS_TORCH:
    class _TorchEncoder(nn.Module):
        def __init__(self, feat_dim: int, d_model: int = 128, nhead: int = 5, num_layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.input_proj = nn.Linear(feat_dim, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x):
            # x: (B, T, feat_dim)
            h = self.input_proj(x)
            h = self.encoder(h)
            score = self.head(h).squeeze(-1)  # (B, T)
            return score
else:
    _TorchEncoder = None  # torch 缺失时由 numpy 降级路径处理，不定义真实类（避免导入期崩溃）


class TransformerEncoder:
    """时序特征 → 向量化表征 / 候选因子。"""

    def __init__(self, d_model: int = 128, nhead: int = 5, num_layers: int = 2, dropout: float = 0.1):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self._model = None
        self._feat_dim = None

    # -- 序列构造 --------------------------------------------------------- #
    def _to_sequences(self, panel: pd.DataFrame, feature_cols) -> Dict[str, np.ndarray]:
        seqs: Dict[str, np.ndarray] = {}
        feat_dim = len(feature_cols)
        for sym, g in panel.groupby(SYMBOL, sort=False):
            g = g.sort_values(DATE)
            arr = g[feature_cols].to_numpy(dtype=float)
            if arr.shape[1] != feat_dim:
                continue
            if np.isnan(arr).any():
                arr = np.nan_to_num(arr, nan=0.0)
            seqs[sym] = arr
        self._feat_dim = feat_dim
        return seqs

    def _ensure_model(self, feat_dim: int):
        if not _HAS_TORCH:
            return
        # PyTorch 要求 d_model 能被 nhead 整除；当规格（如 d_model=128, nhead=5）不兼容时，
        # 自动将 nhead 下调到不超过设定值的最大整除因子，最大限度保留原始设计意图。
        eff_heads = self.nhead
        while eff_heads > 1 and self.d_model % eff_heads != 0:
            eff_heads -= 1
        if eff_heads != self.nhead:
            logger.warning("Transformer: d_model=%d 无法被 nhead=%d 整除，自动调整为 nhead=%d",
                           self.d_model, self.nhead, eff_heads)
        if self._model is None or self._feat_dim != feat_dim:
            self._model = _TorchEncoder(feat_dim, self.d_model, eff_heads, self.num_layers, self.dropout)
            self._model.eval()
            self._feat_dim = feat_dim

    # -- 向量化表征 ------------------------------------------------------- #
    def encode(self, panel: pd.DataFrame, feature_cols) -> Dict[str, np.ndarray]:
        """返回每只股票的序列最后一维隐状态（d_model 维表征）。"""
        seqs = self._to_sequences(panel, feature_cols)
        out: Dict[str, np.ndarray] = {}
        if not _HAS_TORCH:
            # 降级：确定性线性投影（无训练，仅作表征）
            rng = np.random.default_rng(0)
            w = rng.standard_normal((self._feat_dim, self.d_model)) * 0.1
            for sym, arr in seqs.items():
                h = arr @ w
                out[sym] = h.mean(axis=0)
            return out
        self._ensure_model(self._feat_dim)
        with torch.no_grad():
            for sym, arr in seqs.items():
                x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1, T, F)
                h = self._model.input_proj(x)
                h = self._model.encoder(h)
                out[sym] = h[0, -1].numpy()
        return out

    # -- 派生候选因子 ----------------------------------------------------- #
    def derive_factor(self, panel: pd.DataFrame, feature_cols) -> pd.Series:
        """将序列打分映射为 (date, symbol) 多级索引的候选因子（截面 rank 标准化）。"""
        seqs = self._to_sequences(panel, feature_cols)
        idx = []
        scores = []
        if not _HAS_TORCH:
            rng = np.random.default_rng(1)
            w = rng.standard_normal(self._feat_dim) * 0.1
            for sym, arr in seqs.items():
                s = arr @ w
                for t in range(arr.shape[0]):
                    idx.append((None, sym))  # 时间维由 panel 顺序恢复
                    scores.append(s[t])
            # 重新对齐到 panel 的 (date, symbol)
            panel_sorted = panel.sort_values([SYMBOL, DATE])
            s = pd.Series(scores, index=panel_sorted.index)
            s.index = pd.MultiIndex.from_arrays(
                [panel_sorted[DATE].values, panel_sorted[SYMBOL].values], names=[DATE, SYMBOL]
            )
            return _cs_rank(s)
        self._ensure_model(self._feat_dim)
        panel_sorted = panel.sort_values([SYMBOL, DATE]).reset_index(drop=True)
        with torch.no_grad():
            all_x = []
            lens = []
            for sym, arr in seqs.items():
                all_x.append(arr)
                lens.append(arr.shape[0])
            # 逐股处理（避免 padding 复杂化）
            scores = np.zeros(len(panel_sorted))
            pos = 0
            for arr in seqs.values():
                x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
                sc = self._model(x)[0].numpy()  # (T,)
                scores[pos:pos + len(sc)] = sc
                pos += len(sc)
        s = pd.Series(scores, index=panel_sorted.index)
        s.index = pd.MultiIndex.from_arrays(
            [panel_sorted[DATE].values, panel_sorted[SYMBOL].values], names=[DATE, SYMBOL]
        )
        return _cs_rank(s)


def _cs_rank(s: pd.Series) -> pd.Series:
    """截面 rank 标准化到 ~N(0,1)。"""
    return s.groupby(level=0, group_keys=False).apply(lambda x: x.rank(pct=True)).clip(1e-6, 1 - 1e-6)
