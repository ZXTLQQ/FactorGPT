"""六阶段因子冶炼流水线 · 因子精炼厂 (Refinery Pipeline)。

系统以工业冶炼为隐喻，将量化因子开发抽象为从「矿石开采」到「成品交付」的六道工序：

  PART-01  矿石原料仓   ：FeatureForge 构建数据底座（合成/真实行情）
  PART-02  采矿作业层   ：LLM 矿场 + MaskablePPO 强化学习 + Transformer 向量化表征（三维生成）
  PART-03  研磨车间     ：RPN 引擎对候选因子做 Rank IC/IR/ICIR 量化评估 + 稳定性校验
  PART-04  三级筛选     ：LASSO 去冗余 → 人机协同 → TOP 10% 截断
  PART-05  合金配比     ：AlphaPool 合成（ICIR 加权 + 正交化 + leave-one-out）
  PART-06  提交         ：MethodologyReport 自动产出方法学总结并一键导出

数据流：矿石 → 冶炼厂 → 三维生成 → RPN 评估 → 三级过滤 → 合金配比 → 方法学报告。
对外暴露 `run()`，支持离线演示（无需 LLM/网络）与完整生产（接入 LLM 矿场）。
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from agent.rl_search import FactorRLSearch
from agent.transformer_encoder import TransformerEncoder
from data.feature_forge import FeatureForge, MINUTE_FEATURES
from engine.backtest import FactorBacktester
from engine.rpn_engine import RPNConfig, RPNEngine
from pipeline.alpha_pool import AlphaPool, AlphaPoolConfig
from pipeline.factor_zoo import FactorZoo
from pipeline.methodology import MethodologyReport
from pipeline.screener import Screener, ScreenerConfig
from pipeline.schema import CandidateFactor, OreStock, RefineryResult

logger = logging.getLogger("factor_gpt.refinery")

DATE = "date"
SYMBOL = "symbol"


@dataclass
class RefineryContext:
    """PART-01~03 完成后的中间态快照。

    人机协同筛选（PART-04 第二级）需要「暂停等待人工评审」，而 Web UI 是无状态的
    请求-响应模型，无法在一次调用里阻塞等待。故将流水线拆为两段：
    `run_to_review()` 产出本上下文并暂停，人工在界面上勾选保留/剔除后，
    再由 `resume_from_review()` 携带评审结果续跑 PART-04~06。
    """
    requirement: str
    ore: OreStock
    candidates: List[CandidateFactor]
    trace: List[dict] = field(default_factory=list)
    t0: float = 0.0


@dataclass
class RefineryConfig:
    # PART-01
    n_symbols: int = 200
    train_days: int = 500
    test_days: int = 120
    start: str = "2019-01-01"
    n_workers: int = 4
    seed: int = 42
    # PART-02
    transformer: dict = field(default_factory=lambda: {"d_model": 128, "nhead": 5, "num_layers": 2})
    rl_max_len: int = 5
    rl_candidates: int = 6
    rl_backend: str = "auto"     # auto | sb3 | heuristic；auto 在装了 sb3_contrib 时启用真 MaskablePPO
    n_pool_seed: int = 12          # 直接从因子池取样作为候选的基数
    # PART-03
    rpn: RPNConfig = field(default_factory=RPNConfig)
    # PART-04
    screener: ScreenerConfig = field(default_factory=ScreenerConfig)
    # PART-05
    alpha_pool: AlphaPoolConfig = field(default_factory=AlphaPoolConfig)
    # 总控
    offline: bool = True           # 离线演示：跳过 LLM 矿场
    use_real_data: bool = False     # true 时 PART-01 矿石改用真实行情（需网络/akshare）
    cache_dir: str = "data/cache"  # 整矿缓存目录（与 data.cache_dir 一致），网络波动时回退
    cache_only: bool = False       # true 时仅读取已预备整矿缓存，绝不触网（现场答辩防断网）
    max_llm_factors: int = 1
    output_dir: str = "output"
    # P1 多模态
    multimodal: bool = False            # true 时真实数据矿石纳入基本面/估值/资金流/新闻情绪多模态因子
    # P0 组合级回测与过拟合检验（默认开启，作为交付证据）
    run_portfolio: bool = True


class RefineryPipeline:
    """因子精炼厂主流水线。"""

    def __init__(self, config: Optional[RefineryConfig] = None):
        self.config = config or RefineryConfig()

    # -- 总入口 ---------------------------------------------------------- #
    def run(self, requirement: str = "", review_callback: Optional[Callable] = None) -> RefineryResult:
        """一次性跑完六道工序（无人值守模式）。

        人机协同筛选若需真人介入，请改用 `run_to_review()` + `resume_from_review()`
        两段式调用；此处 `review_callback` 仍保留，便于脚本注入自动化评审规则。
        """
        ctx = self.run_to_review(requirement)
        return self.resume_from_review(ctx, review_callback=review_callback)

    # -- 第一段：PART-01~03（产出候选并暂停，等待人工评审） ---------------- #
    def run_to_review(self, requirement: str = "") -> RefineryContext:
        t0 = time.time()
        trace = []

        # PART-01 矿石原料仓
        ore = self._stage01_ore()

        # Kronos 预测因子接入：在 PART-01 数据底座之后，把 Kronos 预测的未来收益
        # 作为候选因子 KRONOS_PRED 并入因子池，供后续 RPN 评估与 AlphaPool 合成使用。
        # 真实模型不可用（依赖缺失 / 权重下载失败）时按配置降级 stub 或跳过，不影响主流程。
        self._maybe_attach_kronos(ore)

        trace.append(_stage("PART-01 矿石原料仓", time.time() - t0,
                            f"universe={len(ore.universe)} 因子池={len(ore.factor_pool)} "
                            f"分钟特征={ore.raw_features.shape[1]-2}"))

        # PART-02 三维生成
        candidates = self._stage02_generate(ore, requirement)
        trace.append(_stage("PART-02 采矿作业层", time.time() - t0,
                            f"候选因子={len(candidates)} (来源: "
                            f"{_count_sources(candidates)})"))

        # PART-03 研磨车间（RPN 评估，填充 ICIR/稳定性/换手）
        candidates = self._stage03_evaluate(candidates, ore)
        trace.append(_stage("PART-03 研磨车间(RPN)", time.time() - t0,
                            "完成 Rank IC/IR/ICIR + 稳定性评估"))

        return RefineryContext(requirement=requirement, ore=ore,
                               candidates=candidates, trace=trace, t0=t0)

    # -- 第二段：PART-04~06（携带人工评审结果续跑） ------------------------ #
    def resume_from_review(self, ctx: RefineryContext,
                           review_callback: Optional[Callable] = None,
                           keep_names: Optional[Iterable[str]] = None) -> RefineryResult:
        """从人机协同评审点续跑。

        `keep_names` 为人工在界面上勾选保留的因子名集合；给定后会构造评审回调，
        真实作用于 PART-04 第二级筛选的 `screened` 结果（而非仅作展示）。
        """
        t0 = ctx.t0 or time.time()
        trace = ctx.trace
        ore = ctx.ore
        candidates = ctx.candidates
        requirement = ctx.requirement

        if keep_names is not None:
            keep_set = {str(n) for n in keep_names}

            def review_callback(cands, _keep=keep_set):  # noqa: F811
                # 评审回调约定返回「保留的因子名列表」
                return [c.name for c in cands if c.name in _keep]

        # PART-04 三级筛选
        screener = Screener(self.config.screener)
        screened = screener.screen(candidates, ore.train_kline, review_callback)
        trace.append(_stage("PART-04 三级筛选", time.time() - t0,
                            f"{len(candidates)} → {len(screened)}"))

        # PART-05 合金配比：权重在训练集拟合，结构诊断（LOO）在训练集评估
        ap = AlphaPool(self.config.alpha_pool)
        composite = ap.optimize(screened, ore.train_kline)
        loo = ap.leave_one_out(screened, ore.train_kline)

        # 样本外评估集：复合因子的性能（ICIR/组合/DSR/动物园）一律在测试集评估，
        # 避免「训练集内定权又评估」导致的样本内乐观偏差。测试集缺失时回退训练集并标注。
        if ore.test_kline is not None and not ore.test_kline.empty:
            eval_kline = ore.test_kline
            eval_set = "test"
        else:
            eval_kline = ore.train_kline
            eval_set = "train"
            logger.warning("test_kline 缺失，复合因子性能回退到训练集评估（样本内，存在乐观偏差）")

        comp_metrics = RPNEngine(self.config.rpn).evaluate(composite, eval_kline)
        trace.append(_stage("PART-05 合金配比(AlphaPool)", time.time() - t0,
                            f"复合ICIR({'OOS' if eval_set == 'test' else 'IS'})={comp_metrics.get('icir', 0):.3f} "
                            f"LOO依赖={loo.get('most_dependent_factor')}"))

        # P0 组合级回测 + 过拟合检验 + 因子动物园（一律在样本外测试集评估）
        robustness = None
        portfolio = None
        cost_sens = None
        ic_year = None
        zoo = None
        bench_cmp = None
        if self.config.run_portfolio:
            bt = FactorBacktester(
                forward_periods=self.config.rpn.forward_periods,
                n_quantiles=self.config.rpn.n_quantiles,
                risk_free_rate=self.config.rpn.risk_free_rate,
            )
            ic_year = bt.ic_by_year(eval_kline, composite)
            bench = bt.equal_weight_benchmark(eval_kline)
            portfolio = bt.realistic_portfolio(eval_kline, composite, benchmark_ret=bench)
            cost_sens = bt.cost_sensitivity(eval_kline, composite)
            robustness = ap.robustness_check(
                composite, eval_kline, n_trials=max(10, len(candidates)))
            zoo = FactorZoo().compare_to_zoo(composite, eval_kline)
            bench_cmp = {k: portfolio.get("metrics", {}).get(k) for k in
                         ("benchmark_info_ratio", "benchmark_alpha_ann", "benchmark_beta", "ann_return")}
            trace.append(_stage(
                f"P0 组合回测/过拟合/动物园({'OOS' if eval_set == 'test' else 'IS'})", time.time() - t0,
                f"组合年化={portfolio.get('metrics', {}).get('ann_return', 0):.3f} "
                f"夏普={portfolio.get('metrics', {}).get('sharpe', 0):.2f} "
                f"DSR={robustness.get('deflated_sharpe_ratio')} "
                f"动物园增量IC={zoo.get('incremental_icir')}"))

        # PART-06 方法学总结
        reporter = MethodologyReport(self.config.output_dir, self.config.rpn)
        result = RefineryResult(
            ore=ore, candidates=candidates, screened=screened,
            composite=composite, composite_metrics=comp_metrics,
            loo_result=loo, stage_trace=trace,
            robustness=robustness, portfolio=portfolio, cost_sensitivity=cost_sens,
            ic_by_year=ic_year, benchmark_comparison=bench_cmp, factor_zoo=zoo,
            multimodal_factors=ore.meta.get("multimodal_factors"),
            eval_set=eval_set,
            screen_audit=dict(getattr(screener, "audit", {}) or {}),
        )

        # P1 产品化交付：导出因子表达式 / 调仓 CSV / 可解释 HTML+PDF 报告
        try:
            from pipeline.exporter import Exporter
            delivered = Exporter(self.config.output_dir).export_all(result)
            result.report_path = delivered.get("html")
            trace.append(_stage(
                "P1 产品交付导出", time.time() - t0,
                f"HTML={os.path.basename(delivered.get('html', ''))} "
                f"PDF={'有' if delivered.get('pdf') else '无'} "
                f"JSON={'有' if delivered.get('json') else '无'}"))
        except Exception as e:  # noqa: BLE001
            logger.warning("产品交付导出失败（不影响主流程）: %s", e)
        report_path = reporter.generate(result, requirement)
        result.report_path = report_path
        trace.append(_stage("PART-06 方法学总结", time.time() - t0, f"report={report_path}"))

        logger.info("精炼厂流水线完成，总耗时 %.1fs", time.time() - t0)
        return result

    # -- Kronos 预测因子接入 ------------------------------------------- #
    def _maybe_attach_kronos(self, ore: OreStock) -> None:
        """若 config.yaml 的 kronos.enabled 为 true，将 Kronos 预测因子接入因子池。

        Kronos (morrisluo/kronos, HuggingFace: NeoQuasar/Kronos-*) 是 K 线时序预测
        基础模型，这里用它预测未来收益构造选股信号 KRONOS_PRED。真实模型不可用
        （依赖缺失/权重下载失败）时按配置降级为 stub 或跳过，不影响主流水线。
        """
        try:
            from llm.client import load_config
            kcfg = (load_config() or {}).get("kronos", {})
        except Exception:
            return
        if not kcfg.get("enabled"):
            return
        try:
            try:
                from kronos import attach_kronos_factor
            except ImportError:
                from src.kronos import attach_kronos_factor
        except Exception as e:  # noqa: BLE001
            logger.warning("Kronos 模块导入失败，跳过 Kronos 因子: %s", e)
            return
        # 合成数据(use_real_data=False)时 Kronos 预测为空, 真实权重派不上用场;
        # 强制 allow_download=False, 避免触发 HuggingFace 权重下载(可能卡数分钟)。
        if not self.config.use_real_data:
            kcfg = {**kcfg, "allow_download": False}
        try:
            attach_kronos_factor(ore, {"kronos": kcfg})
            logger.info("[refinery] Kronos 预测因子 KRONOS_PRED 已接入因子池")
        except Exception as e:  # noqa: BLE001
            logger.warning("Kronos 因子接入失败(已跳过): %s", e)

    # -- PART-01 --------------------------------------------------------- #
    def _stage01_ore(self) -> OreStock:
        if self.config.use_real_data:
            # cache_only：完全离线，仅读取已预备的整矿缓存（现场答辩防断网）
            if self.config.cache_only:
                cached = self._load_real_ore_cache()
                if cached is not None:
                    logger.info("cache_only 模式：直接加载本地预备整矿缓存")
                    return cached
                logger.warning("cache_only 但无预备整矿缓存，回退合成数据")
            else:
                try:
                    return self._build_real_ore()
                except Exception as e:  # noqa: BLE001
                    logger.warning("真实数据矿石构建失败: %s", e)
                    cached = self._load_real_ore_cache()
                    if cached is not None:
                        logger.info("从本地预备整矿缓存加载（网络不可用）")
                        return cached
                    logger.warning("无预备整矿缓存，回退合成数据")
        ff = FeatureForge(n_workers=self.config.n_workers, seed=self.config.seed)
        return ff.build_synthetic_universe(
            n_symbols=self.config.n_symbols,
            train_days=self.config.train_days,
            test_days=self.config.test_days,
            start=self.config.start,
        )

    def prepare_real_ore(self) -> OreStock:
        """预备真实数据：联网拉取并缓存整矿，便于现场离线回退。

        运行 scripts/prefetch_data.py 即调用本方法；成功后在 data/cache/real_ore.pkl
        落盘整个矿石包（行情/行业/市值/因子池/特征），后续即使断网也可由 cache_only 加载。
        """
        ore = self._build_real_ore()
        self._save_real_ore_cache(ore)
        return ore

    def _save_real_ore_cache(self, ore: OreStock) -> None:
        try:
            os.makedirs(self.config.cache_dir, exist_ok=True)
            path = os.path.join(self.config.cache_dir, "real_ore.pkl")
            with open(path, "wb") as f:
                pickle.dump(ore, f)
            logger.info("整矿缓存已写入 %s", path)
        except Exception as e:  # noqa: BLE001
            logger.warning("整矿缓存写入失败: %s", e)

    def _load_real_ore_cache(self) -> Optional[OreStock]:
        p = os.path.join(self.config.cache_dir, "real_ore.pkl")
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    ore = pickle.load(f)
                if getattr(ore, "train_kline", None) is not None and getattr(ore, "factor_pool", None):
                    # 防御：旧缓存可能含重复 (date, symbol) 行，统一去重避免下游 reindex 崩溃
                    for attr in ("train_kline", "test_kline"):
                        df = getattr(ore, attr, None)
                        if df is not None and not df.empty:
                            setattr(ore, attr, df.drop_duplicates(subset=["date", "symbol"], keep="last"))
                    for name, s in list((ore.factor_pool or {}).items()):
                        if s is not None and s.index.duplicated().any():
                            ore.factor_pool[name] = s[~s.index.duplicated(keep="last")]
                    logger.info("已加载本地整矿缓存 %s", p)
                    return ore
            except Exception as e:  # noqa: BLE001
                logger.warning("整矿缓存读取失败: %s", e)
        return None

    def _build_real_ore(self) -> OreStock:
        """从 AKShare 拉取真实行情构建矿石（训练/测试切分 + 真实行业/市值）。

        因子池与分钟级特征均由真实日 K 派生，确保与真实 (date, symbol) 索引对齐，
        使 RPN 评估、中性化均作用于真实收益。网络/数据源不可用时由调用方回退合成数据。
        """
        from data.neo_adapter import get_data_source

        cfg = self.config
        # 数据源走工厂：默认 legacy（本地自爬方案保留），config.yaml 设 data.source=neodata 时切稳定源
        fetcher = get_data_source()
        dates_all = pd.bdate_range(cfg.start, periods=cfg.train_days + cfg.test_days)
        end = dates_all[-1].strftime("%Y-%m-%d")

        symbols = fetcher.get_index_constituents("000906") or []
        symbols = [str(s).zfill(6) for s in symbols][: min(cfg.n_symbols, 100)]
        if not symbols:
            symbols = ["600519", "000858", "601318", "600036", "000333",
                       "601899", "600276", "000001"][: min(cfg.n_symbols, 8)]

        kline = fetcher.get_daily_kline(symbols, start=cfg.start, end=end)
        if kline is None or kline.empty:
            raise RuntimeError("真实行情为空，无法构建矿石")

        # 补齐精炼厂下游可能引用的列（成交额/涨跌幅代理）
        kline = kline.copy()
        # 防御：真实行情偶发重复 (date, symbol) 行，会导致下游 factor series 多索引非唯一，
        # 使 screener/alpha_pool 的 reindex 抛 "cannot handle a non-unique multi-index!"。统一去重。
        kline = kline.drop_duplicates(subset=["date", "symbol"], keep="last")
        if "amount" not in kline.columns:
            kline["amount"] = kline["close"] * kline["volume"]
        if "pct_chg" not in kline.columns:
            kline["pct_chg"] = kline.groupby("symbol")["close"].pct_change() * 100

        uniq = sorted(kline["date"].astype(str).unique())
        train_dates = set(uniq[: cfg.train_days])
        train_kline = kline[kline["date"].astype(str).isin(train_dates)].copy()
        test_kline = kline[~kline["date"].astype(str).isin(train_dates)].copy()
        if train_kline.empty or test_kline.empty:
            raise RuntimeError("训练/测试切分后为空，请增大 train_days/test_days")

        industry_series, mkt_cap_series = fetcher.get_industry_and_cap(symbols)

        industry_df = pd.DataFrame({"symbol": symbols})
        industry_df["ind_0"] = industry_series.reindex(symbols).fillna("未知").astype(str).values
        for i in range(1, 9):
            industry_df[f"ind_{i}"] = 0
        style_df = pd.DataFrame({"symbol": symbols})
        size = np.log(mkt_cap_series.reindex(symbols).replace(0, np.nan)).fillna(0.0).values
        style_df["size"] = size
        style_df["value"] = 0.0
        style_df["momentum"] = 0.0

        factor_pool = self._build_real_factor_pool(kline, train_dates)
        raw_features = self._build_real_raw_features(kline, train_dates)

        # P1 多模态：纳入基本面/估值/资金流/新闻情绪因子（失败自动降级）
        multimodal_factors: List[str] = []
        if self.config.multimodal:
            try:
                from data.multimodal import build_multimodal_factors
                mm = build_multimodal_factors(kline, symbols)
                if mm:
                    factor_pool.update(mm)
                    multimodal_factors = list(mm.keys())
                    logger.info("多模态因子已并入因子池：%s", multimodal_factors)
            except Exception as e:  # noqa: BLE001
                logger.warning("多模态因子构建失败，已跳过: %s", e)

        ore = OreStock(
            universe=symbols,
            train_kline=train_kline,
            test_kline=test_kline,
            industry=industry_df,
            style=style_df,
            raw_features=raw_features,
            factor_pool=factor_pool,
            meta={"source": "real", "multimodal_factors": multimodal_factors},
        )
        self._save_real_ore_cache(ore)
        return ore

    @staticmethod
    def _build_real_factor_pool(kline: pd.DataFrame, train_dates) -> Dict[str, pd.Series]:
        """基于真实日 K 派生因子池（与真实 (date, symbol) 对齐）。"""
        df = kline.copy()
        df["date"] = df["date"].astype(str)
        df = df.sort_values(["symbol", "date"])
        g = df.groupby("symbol")
        df["ret"] = g["close"].pct_change()
        df["log_ret"] = np.log(df["close"]).diff()
        df["mom_5"] = g["close"].transform(lambda x: x.pct_change(5))
        df["mom_20"] = g["close"].transform(lambda x: x.pct_change(20))
        df["mom_60"] = g["close"].transform(lambda x: x.pct_change(60))
        df["vol_5"] = g["ret"].transform(lambda x: x.rolling(5).std())
        df["vol_20"] = g["ret"].transform(lambda x: x.rolling(20).std())
        df["vol_60"] = g["ret"].transform(lambda x: x.rolling(60).std())
        df["reversal_5"] = -df["mom_5"]
        df["reversal_20"] = -df["mom_20"]
        df["turnover_5"] = g["volume"].transform(lambda x: x.rolling(5).mean())
        df["turnover_20"] = g["volume"].transform(lambda x: x.rolling(20).mean())
        df["amihud_21"] = (df["ret"].abs() /
                           df["volume"].replace(0, np.nan)) \
            .groupby(df["symbol"]).transform(lambda x: x.rolling(21).mean())
        df["skew_10"] = g["ret"].transform(lambda x: x.rolling(10).skew())
        df["skew_20"] = g["ret"].transform(lambda x: x.rolling(20).skew())
        df["hl_range_5"] = (g["high"].transform(lambda x: x.rolling(5).max()) -
                            g["low"].transform(lambda x: x.rolling(5).min())) / \
            g["close"].transform(lambda x: x.rolling(5).mean())
        df["ret_1"] = df["ret"]

        keep = [c for c in df.columns if c not in
                ("date", "symbol", "open", "high", "low", "close", "volume", "amount", "pct_chg")]
        pool = {}
        for col in keep:
            s = df.set_index(["date", "symbol"])[col].dropna()
            if s.empty:
                continue
            pool[col] = s
        if len(pool) < 12:
            raise RuntimeError(f"真实因子池不足 12 个（实际 {len(pool)}），请检查行情字段")
        return pool

    @staticmethod
    def _build_real_raw_features(kline: pd.DataFrame, train_dates) -> pd.DataFrame:
        """派生 28 维分钟级特征（以日 K 近似），列名对齐 MINUTE_FEATURES。"""
        df = kline.copy()
        df["date"] = df["date"].astype(str)
        df = df.sort_values(["symbol", "date"])
        g = df.groupby("symbol")
        base = df[["date", "symbol"]].copy()
        for i in range(1, len(MINUTE_FEATURES) + 1):
            if i <= 10:
                base[f"min_ret_{i}"] = g["close"].transform(lambda x: x.pct_change(i - 1))
            else:
                base[f"min_ret_{i}"] = g["close"].transform(lambda x: x.pct_change(i - 5))
        base = base.dropna()
        return base

    # -- PART-02 --------------------------------------------------------- #
    def _stage02_generate(self, ore: OreStock, requirement: str) -> List[CandidateFactor]:
        candidates: List[CandidateFactor] = []

        # (a) Transformer 向量化表征 → 派生候选因子
        te = TransformerEncoder(**self.config.transformer)
        try:
            tf_factor = te.derive_factor(ore.raw_features, MINUTE_FEATURES)
            candidates.append(CandidateFactor(
                name="Transformer_SeqAlpha", source="transformer", series=tf_factor,
                description="Transformer(d_model=128,2层,5头) 对分钟级序列建模派生的注意力 alpha",
            ))
        except Exception as e:  # noqa: BLE001
            logger.warning("Transformer 派生因子失败: %s", e)

        # (b) MaskablePPO 因子组合搜索
        rl = FactorRLSearch(
            max_len=self.config.rl_max_len,
            fwd=self.config.rpn.forward_periods,
            backend=self.config.rl_backend,
        )
        try:
            candidates += rl.run(ore.factor_pool, ore.train_kline, n_candidates=self.config.rl_candidates)
        except Exception as e:  # noqa: BLE001
            logger.warning("RL 搜索失败: %s", e)

        # (c) 因子池直接取样（基础候选）
        for name, s in list(ore.factor_pool.items())[: self.config.n_pool_seed]:
            candidates.append(CandidateFactor(name=name, source="pool", series=s,
                                              description="因子池基础候选"))

        # (d) LLM 矿场（可选，需 API 且数据同 universe）
        if not self.config.offline:
            candidates += self._llm_mine(ore, requirement)

        return candidates

    def _llm_mine(self, ore: OreStock, requirement: str) -> List[CandidateFactor]:
        """接入现有 FactorAgent 作为 LLM 矿场（需 LLM API 与行情数据）。"""
        try:
            from ..agent import FactorAgent
            from llm.client import load_config
            cfg = load_config()
            agent = FactorAgent(cfg)
            res = agent.run(requirement, max_iterations=int(cfg.get("agent", {}).get("max_iterations", 6)))
            fs = res.get("state", {}).get("final_factor_series")
            if fs is not None:
                # 仅当与精炼厂 universe 对齐时才纳入
                aligned = fs.reindex(ore.train_kline.set_index([DATE, SYMBOL]).index)
                if aligned.notna().sum() > 50:
                    st = res.get("state", {})
                    return [CandidateFactor(
                        name="LLM_" + requirement[:16], source="llm",
                        code=st.get("code"), series=aligned,
                        description=requirement, metrics=res.get("metrics", {}),
                        rationale=st.get("factor_rationale", ""),
                        references=st.get("factor_references", []),
                    )]
            logger.warning("LLM 矿场因子与精炼厂 universe 不对齐，已跳过")
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM 矿场调用失败（已跳过，不影响其余工序）: %s", e)
        return []

    # -- PART-03 --------------------------------------------------------- #
    def _stage03_evaluate(self, candidates: List[CandidateFactor], ore: OreStock) -> List[CandidateFactor]:
        rpn = RPNEngine(self.config.rpn)
        cand_map = {c.name: c.series for c in candidates if c.series is not None}
        metrics_map = rpn.evaluate_batch(cand_map, ore.train_kline)
        for c in candidates:
            if c.name in metrics_map:
                c.metrics.update(metrics_map[c.name])
        return candidates


def _stage(name: str, elapsed: float, note: str) -> dict:
    return {"stage": name, "elapsed_s": round(elapsed, 2), "note": note}


def build_refinery_config(d: Optional[dict] = None) -> RefineryConfig:
    """从 config.yaml 的 `refinery` 段落构造 RefineryConfig（容错、缺省补全）。"""
    d = d or {}
    rpn_d = d.get("rpn", {}) or {}
    rpn = RPNConfig(
        n_quantiles=rpn_d.get("n_quantiles", 5),
        forward_periods=rpn_d.get("forward_periods", 1),
        commission=rpn_d.get("commission", 0.001),
        risk_free_rate=rpn_d.get("risk_free_rate", 0.03),
        w_turnover_penalty=rpn_d.get("w_turnover_penalty", 0.2),
        parallel=rpn_d.get("parallel", True),
        n_workers=rpn_d.get("n_workers", 4),
    )
    scr_d = d.get("screener", {}) or {}
    screener = ScreenerConfig(
        use_lasso=scr_d.get("use_lasso", True),
        use_human_collab=scr_d.get("use_human_collab", True),
        topk_ratio=scr_d.get("topk_ratio", 0.1),
        min_keep=scr_d.get("min_keep", 3),
    )
    ap_d = d.get("alpha_pool", {}) or {}
    alpha_pool = AlphaPoolConfig(
        ortho=ap_d.get("ortho", True),
        loo=ap_d.get("loo", True),
        iterative=ap_d.get("iterative", True),
        n_iter=ap_d.get("n_iter", 20),
    )
    return RefineryConfig(
        n_symbols=d.get("n_symbols", 200),
        train_days=d.get("train_days", 500),
        test_days=d.get("test_days", 120),
        start=d.get("start", "2019-01-01"),
        n_workers=d.get("n_workers", 4),
        seed=d.get("seed", 42),
        transformer=d.get("transformer", {"d_model": 128, "nhead": 5, "num_layers": 2}),
        rl_max_len=d.get("rl_max_len", 5),
        rl_candidates=d.get("rl_candidates", 6),
        rl_backend=d.get("rl_backend", "auto"),
        n_pool_seed=d.get("n_pool_seed", 12),
        offline=d.get("offline", True),
        use_real_data=bool(d.get("use_real_data", False)),
        cache_dir=d.get("cache_dir", "data/cache"),
        cache_only=bool(d.get("cache_only", False)),
        max_llm_factors=d.get("max_llm_factors", 1),
        output_dir=d.get("output_dir", "output"),
        multimodal=bool(d.get("multimodal", False)),
        run_portfolio=bool(d.get("run_portfolio", True)),
        rpn=rpn, screener=screener, alpha_pool=alpha_pool,
    )


def _count_sources(candidates: List[CandidateFactor]) -> str:
    from collections import Counter
    cnt = Counter(c.source for c in candidates)
    return ", ".join(f"{k}={v}" for k, v in cnt.items())
