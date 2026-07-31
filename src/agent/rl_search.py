"""PART-02 采矿作业层 · RL 算法 (MaskablePPO 因子组合搜索)。

强化学习负责自动搜索「最优因子组合」：
  * 动作空间 = 从因子池中逐个选取基础因子（组合上限 max_len）；
  * 动作屏蔽 (MaskablePPO) = 已选因子不可重复选取、超过长度则终止，避免错误探索；
  * 奖励 = 组合因子的 Rank ICIR（信息比率），最大化 Alpha + 控制换手；
  * 后端：若安装 stable-baselines3 + sb3_contrib，则使用真正的 MaskablePPO 训练；
          否则降级为「动作屏蔽 + 集束搜索（beam search）」启发式，同样尊重屏蔽规则。

无论哪种后端，对外暴露统一的 `run()` 接口，产出候选因子列表（含 RPN 指标）。
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from pipeline.schema import CandidateFactor

logger = logging.getLogger("factor_gpt.rl")

DATE = "date"
SYMBOL = "symbol"

try:
    from stable_baselines3 import PPO  # noqa: F401
    from sb3_contrib import MaskablePPO  # type: ignore
    _HAS_SB3 = True
except Exception:  # noqa: BLE001
    _HAS_SB3 = False
    MaskablePPO = None

try:
    import gymnasium as gym  # 仅 SB3/MaskablePPO 后端需要
    _HAS_GYM = True
except Exception:  # noqa: BLE001
    gym = None
    _HAS_GYM = False


def _fast_icir(factor: pd.Series, kline: pd.DataFrame, fwd: int = 1) -> float:
    """轻量 Rank ICIR 代理（不跑完整回测，供 RL 高频调用）。

    注意：kline 为列型 DataFrame（含 date/symbol 列），此处先 set_index 对齐到因子的
    (date, symbol) 多级索引，避免前向收益错位。
    """
    if factor is None or len(factor) == 0:
        return 0.0
    kl = kline.set_index([DATE, SYMBOL])
    fr = kl.groupby(level=SYMBOL)["close"].pct_change(fwd)
    df = pd.concat([factor.rename("f"), fr.rename("r")], axis=1).dropna()
    if len(df) < 30:
        return 0.0
    ic = df.groupby(level=DATE, group_keys=False).apply(lambda x: x["f"].corr(x["r"]))
    ic = ic.dropna()
    if ic.std(ddof=1) == 0 or len(ic) < 5:
        return 0.0
    return float(ic.mean() / ic.std(ddof=1))


def _combine(selected_series: list, names: list) -> pd.Series:
    """等权合并多个因子并截面 rank 标准化。"""
    out = None
    for s in selected_series:
        out = s if out is None else out + s
    if out is None:
        return pd.Series(dtype=float)
    return out.groupby(level=0, group_keys=False).rank(pct=True)


if _HAS_GYM:
    class FactorEnv(gym.Env):
        """动作屏蔽因子组合搜索环境（gymnasium 兼容，可直接接入 MaskablePPO）。

        动作空间 = Discrete(n)，逐个选取因子；观测 = [已选掩码(n) | 归一化 ICIR] 的 Box。
        action_masks() 屏蔽已选因子与超长组合，是 MaskablePPO 的核心。
        """

        metadata = {"render_modes": []}

        def __init__(self, factor_pool: dict, kline: pd.DataFrame, max_len: int = 5, fwd: int = 1):
            super().__init__()
            self.pool_names = list(factor_pool.keys())
            self.pool = factor_pool
            self.kline = kline
            self.max_len = min(max_len, len(self.pool_names))
            self.fwd = fwd
            self.n = len(self.pool_names)
            self.action_space = gym.spaces.Discrete(self.n)
            self.observation_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(self.n + 1,), dtype=np.float32
            )
            self.selected = []

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.selected = []
            return self._obs(0.0), {}

        def _obs(self, icir: float):
            mask = np.zeros(self.n, dtype=np.float32)
            for i in self.selected:
                mask[i] = 1.0
            return np.concatenate([mask, [np.clip(icir, -3, 3) / 3.0]]).astype(np.float32)

        def action_masks(self) -> np.ndarray:
            """MaskablePPO 核心：屏蔽已选与超长动作。"""
            masks = np.ones(self.n, dtype=bool)
            if len(self.selected) >= self.max_len:
                masks[:] = False
            else:
                for i in self.selected:
                    masks[i] = False
            return masks

        def step(self, action: int):
            if action in self.selected or len(self.selected) >= self.max_len:
                # 屏蔽外的非法动作：视作终止（零奖励）
                return self._obs(0.0), 0.0, True, False, {}
            self.selected.append(int(action))
            factor = _combine(
                [self.pool[self.pool_names[i]] for i in self.selected], self.selected
            )
            icir = _fast_icir(factor, self.kline, self.fwd)
            terminated = len(self.selected) >= self.max_len
            # 奖励塑形：ICIR 本身（终端），过程步给微小探索激励
            reward = float(icir if terminated else icir * 0.1)
            return self._obs(icir), reward, terminated, False, \
                {"icir": icir, "selected": list(self.selected)}
else:
    class FactorEnv:
        """gymnasium 缺失时的简易环境（仅支撑启发式，不支撑 SB3）。"""

        def __init__(self, factor_pool: dict, kline: pd.DataFrame, max_len: int = 5, fwd: int = 1):
            self.pool_names = list(factor_pool.keys())
            self.pool = factor_pool
            self.kline = kline
            self.max_len = min(max_len, len(self.pool_names))
            self.fwd = fwd
            self.n = len(self.pool_names)
            self.selected = []

        def reset(self, *, seed=None, options=None):
            self.selected = []
            return self._obs(0.0), {}

        def _obs(self, icir: float):
            mask = np.zeros(self.n, dtype=np.float32)
            for i in self.selected:
                mask[i] = 1.0
            return np.concatenate([mask, [np.clip(icir, -3, 3) / 3.0]]).astype(np.float32)

        def action_masks(self) -> np.ndarray:
            masks = np.ones(self.n, dtype=bool)
            if len(self.selected) >= self.max_len:
                masks[:] = False
            else:
                for i in self.selected:
                    masks[i] = False
            return masks

        def step(self, action: int):
            if action in self.selected or len(self.selected) >= self.max_len:
                return self._obs(0.0), 0.0, True, False, {}
            self.selected.append(int(action))
            factor = _combine(
                [self.pool[self.pool_names[i]] for i in self.selected], self.selected
            )
            icir = _fast_icir(factor, self.kline, self.fwd)
            terminated = len(self.selected) >= self.max_len
            reward = float(icir if terminated else icir * 0.1)
            return self._obs(icir), reward, terminated, False, \
                {"icir": icir, "selected": list(self.selected)}


class FactorRLSearch:
    """MaskablePPO 因子组合搜索（SB3 可选 + 启发式降级）。"""

    def __init__(self, max_len: int = 5, fwd: int = 1, backend: str = "auto"):
        self.max_len = max_len
        self.fwd = fwd
        self.use_sb3 = (backend == "sb3") or (backend == "auto" and _HAS_SB3)

    def run(self, factor_pool: dict, kline: pd.DataFrame, n_candidates: int = 6,
            timesteps: int = 4000) -> list:
        if self.use_sb3:
            try:
                return self._run_sb3(factor_pool, kline, n_candidates, timesteps)
            except Exception as e:  # noqa: BLE001
                logger.warning("SB3 路径失败，降级启发式: %s", e)
        return self._run_heuristic(factor_pool, kline, n_candidates)

    def _run_heuristic(self, factor_pool: dict, kline: pd.DataFrame, n_candidates: int) -> list:
        """集束搜索：每步仅展开「未被屏蔽」的动作，保留 top-K 组合。"""
        names = list(factor_pool.keys())
        env = FactorEnv(factor_pool, kline, self.max_len, self.fwd)
        beam = [([], 0.0)]  # (selected_indices, icir)
        for _ in range(self.max_len):
            new_beam = []
            for sel, _ in beam:
                used = set(sel)
                for i in range(env.n):
                    if i in used:
                        continue  # 动作屏蔽：不重复选取
                    factor = _combine([factor_pool[names[j]] for j in sel + [i]], sel + [i])
                    icir = _fast_icir(factor, kline, self.fwd)
                    new_beam.append((sel + [i], icir))
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:max(n_candidates, 3)]
        candidates = []
        seen = set()
        for sel, icir in beam:
            key = tuple(sorted(sel))
            if not sel or key in seen:
                continue
            seen.add(key)
            factor = _combine([factor_pool[names[j]] for j in sel], sel)
            names_sel = [names[j] for j in sel]
            candidates.append(CandidateFactor(
                name="RL_" + "_".join(names_sel)[:24],
                source="rl",
                series=factor,
                description=f"RL 组合因子，成分={names_sel}，组合上限={self.max_len}",
                metrics={"icir_proxy": icir, "n_components": len(sel)},
            ))
            if len(candidates) >= n_candidates:
                break
        return candidates

    def _run_sb3(self, factor_pool, kline, n_candidates, timesteps):
        from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
        from sb3_contrib.common.wrappers import ActionMasker

        env = FactorEnv(factor_pool, kline, self.max_len, self.fwd)
        # 观测是单一 Box（非 Dict），故用 MaskableActorCriticPolicy；
        # ActionMasker 在每步将 action_masks() 注入策略，确保不选取被屏蔽动作。
        masked_env = ActionMasker(env, lambda e: e.action_masks())
        model = MaskablePPO(MaskableActorCriticPolicy, masked_env, verbose=0)
        model.learn(total_timesteps=timesteps)
        # 用训练后策略采样若干轨迹，取 ICIR 最高的若干组合
        best = []
        for _ in range(max(n_candidates * 4, 8)):
            obs, _ = env.reset()
            terminated = False
            while not terminated:
                action, _ = model.predict(
                    obs, action_masks=env.action_masks(), deterministic=True
                )
                obs, _, terminated, _, _ = env.step(int(action))
            sel = list(env.selected)
            if not sel:
                continue
            factor = _combine([factor_pool[env.pool_names[i]] for i in sel], sel)
            best.append((sel, _fast_icir(factor, kline, self.fwd), factor))
        best.sort(key=lambda x: x[1], reverse=True)
        out = []
        seen = set()
        for sel, icir, factor in best:
            key = tuple(sorted(sel))
            if key in seen:
                continue
            seen.add(key)
            out.append(CandidateFactor(
                name="RL_" + "_".join(env.pool_names[i] for i in sel)[:24],
                source="rl",
                series=factor,
                description=f"SB3-MaskablePPO 组合因子，成分={[env.pool_names[i] for i in sel]}",
                metrics={"icir_proxy": icir, "n_components": len(sel)},
            ))
            if len(out) >= n_candidates:
                break
        return out
