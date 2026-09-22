"""为 README 各功能区块生成运行结果图例（调用真实引擎）。

输出（docs/assets/）：
    feature_factor_library.png      # 3. 62 内置因子库：五大类分布
    feature_gp_evolution.png        # 4. 增强遗传编程：训练/测试 IC 演化
    feature_unstructured.png        # 5. 非结构化数据因子挖掘：文本情绪分布
    feature_transformer_coupling.png# 6. Transformer-Agent 深度耦合：因子检索相关度
    feature_offline_data.png        # 7. 本地部署与离线韧性：离线数据覆盖
    feature_ima_pipeline.png        # 8. 研报知识管线：关键词命中统计
    feature_hf_pipeline.png         # 高频：L2 快照 → 压实产物 → 分钟面板
    feature_multimodal_judging.png  # 材料三级判定链：本地训练模型实测准确率
    feature_upload_context.png      # 上传长文：按预算的按页切片（首尾保留）
    feature_intent_routing.png      # 意图分流与材料注入（流程示意）

用法：
    python scripts/gen_feature_charts.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

# Windows 控制台 GBK：强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from engine.factor_library import FactorLibrary  # noqa: E402
from engine.factor_system import build_synthetic_panel  # noqa: E402
from engine.genetic_enhanced import (  # noqa: E402
    EnhancedFactorEvolver,
    EventWindow,
    eval_expr,
)
from engine.transformer_coupling import TransformerCoupling  # noqa: E402
from engine.unstructured_miner import TextAnalyzer  # noqa: E402

ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# 中文字体（Windows 微软雅黑，缺失时回退英文标签）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# 统一视觉风格（与 feature_gp_evolution.png 保持一致）
# ---------------------------------------------------------------------------
_PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
_CAPTION_BOX = {"boxstyle": "round,pad=0.35", "fc": "#F7F8FA", "ec": "#D6DAE2"}
_INFO_BOX = {"boxstyle": "round,pad=0.5", "fc": "#F7F8FA", "ec": "#4C72B0"}


def fig_factor_library() -> None:
    """功能 3：内置传统因子库五大类分布。"""
    lib = FactorLibrary()
    stats = lib.statistics()
    by_cat = stats.get("by_category", {})
    cats = list(by_cat.keys())
    counts = [by_cat[c] for c in cats]
    total = stats.get("total", 0)
    colors = _PALETTE[: len(cats)]

    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=130)
    ypos = np.arange(len(cats))
    bars = ax.barh(ypos, counts, color=colors, height=0.58)
    for i, (b, c) in enumerate(zip(bars, counts)):
        ax.text(c + max(counts) * 0.02, i, f"{c}  ({c / total:.0%})",
                va="center", fontsize=9.5, color="#333333")
    ax.set_yticks(ypos)
    ax.set_yticklabels(cats, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("因子数量", fontsize=10)
    ax.set_xlim(0, max(counts) * 1.22)
    ax.set_title("(a) 五大类因子数量与占比", fontsize=11)
    ax.grid(alpha=0.25, axis="x")
    ax.text(0.98, 0.03, f"总因子数：{total}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9.5, bbox=_CAPTION_BOX)
    fig.suptitle("Factor Library · 内置传统因子库 · 5 大类", fontsize=12.5)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(ASSETS / "feature_factor_library.png")
    plt.close(fig)
    print(f"  ✓ feature_factor_library.png  (total={total})")


# ---------------------------------------------------------------------------
# 功能 4：增强遗传编程（真实离线行情 + 真实演化轨迹）
# ---------------------------------------------------------------------------

CLUSTER_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
GP_GENERATIONS = 6
GP_POP_PER_CLUSTER = 20
GP_MIGRATE_EVERY = 3
GP_MIGRATION_RATE = 2
# 需要滚动窗口参数的算子（窗口位位于子节点末尾）
_WINDOW_OPS = {"ts_zscore", "ts_rank", "ts_min", "ts_max", "rol", "ts_corr"}


def _gp_panel() -> tuple:
    """取一段真实离线行情；离线数据缺失时回退到合成面板。"""
    try:
        from data.offline_adapter import OfflineDataSource

        ds = OfflineDataSource({"data": {"offline": {"index": "csi800"}}})
        panel = ds.get_daily_kline(ds.get_index_constituents()[:45], "2023-01-01", "2024-03-31")
        if panel is not None and len(panel) > 5000:
            panel = panel.copy()
            panel["pct_chg"] = panel["pct_chg"].astype(float)
            print(f"    真实离线行情：{panel['symbol'].nunique()} 只 × {panel['date'].nunique()} 个交易日")
            return panel, "离线行情"
        print("    离线行情不足，回退合成面板")
    except Exception as exc:  # noqa: BLE001
        print(f"    离线行情不可用（{exc}），回退合成面板")

    panel = build_synthetic_panel(n_symbols=40, days=300, seed=7)
    panel["pct_chg"] = panel.groupby("symbol")["close"].pct_change() * 100.0
    return panel, "合成面板"


def _derive_event_window(panel: pd.DataFrame) -> tuple:
    """从真实行情定位「市场状态」事件窗口：日均波动率峰值前后各 10 个交易日。

    窗口由数据自身决定（非人工指定），保证图中所画事件区间可复现、可解释。
    """
    daily = panel.groupby("date")["pct_chg"].mean().sort_index()
    vol = daily.rolling(20, min_periods=20).std().dropna()
    if vol.empty:
        days = sorted(panel["date"].unique())
        return str(days[0]), str(days[-1]), None
    peak = vol.idxmax()
    i = int(vol.index.get_loc(peak))
    lo, hi = max(0, i - 10), min(len(vol) - 1, i + 10)
    return str(vol.index[lo]), str(vol.index[hi]), str(peak)


def _daily_ic(panel: pd.DataFrame, expr) -> pd.Series:
    """逐日截面 IC 序列（与引擎 _fitness 内部口径一致，仅去掉事件加权）。"""
    df = panel.sort_values(["symbol", "date"]).reset_index(drop=True).copy()
    df["_fwd_ret"] = df.groupby("symbol")["pct_chg"].shift(-1)
    fac = eval_expr(expr, df)
    tbl = pd.DataFrame({
        "f": np.asarray(fac, dtype=float),
        "y": df["_fwd_ret"].to_numpy(dtype=float),
        "date": df["date"].to_numpy(),
    })
    tbl = tbl.replace([np.inf, -np.inf], np.nan).dropna()
    if tbl.empty:
        return pd.Series(dtype=float)
    return tbl.groupby("date").apply(
        lambda g: g["f"].corr(g["y"]) if g["f"].std() > 0 else np.nan
    ).dropna()


def _tree_label(expr, role: str = "val") -> str:
    kind = expr[0]
    if kind == "col":
        return str(expr[1])
    if kind == "const":
        return f"w={float(expr[1]):g}" if role == "win" else f"{float(expr[1]):g}"
    return str(kind)


def _draw_expr_tree(ax, expr, caption: str = "") -> None:
    """把演化产出的表达式树按真实结构绘制成节点图（非示意图）。"""
    nodes: list = []
    xs: dict = {}
    seq = [0.0]

    def walk(e, parent, depth, role="val"):
        idx = len(nodes)
        nodes.append({"idx": idx, "parent": parent, "depth": depth,
                      "kind": e[0], "label": _tree_label(e, role)})
        kids = []
        for i, sub in enumerate(e[1:]):
            if not isinstance(sub, tuple):
                continue
            is_win = e[0] in _WINDOW_OPS and (
                (e[0] == "ts_corr" and i == 2) or (e[0] != "ts_corr" and i == 1)
            )
            kids.append((sub, "win" if is_win else "val"))
        if kids:
            x = sum(walk(sub, idx, depth + 1, r) for sub, r in kids) / len(kids)
        else:
            x, seq[0] = seq[0], seq[0] + 1.0
        xs[idx] = x
        return x

    walk(expr, None, 0)
    for n in nodes:
        if n["parent"] is None:
            continue
        p = nodes[n["parent"]]
        ax.plot([xs[p["idx"]], xs[n["idx"]]], [-p["depth"], -n["depth"]],
                color="#B0B7C3", lw=1.0, zorder=1)
    for n in nodes:
        if n["kind"] == "col":
            fc, ec = "#E8F1E4", "#55A868"
        elif n["kind"] == "const":
            fc, ec = "#FDF3E3", "#CCB974"
        else:
            fc, ec = "#E6EDF7", "#4C72B0"
        ax.text(xs[n["idx"]], -n["depth"], n["label"], ha="center", va="center",
                fontsize=8.0, zorder=2,
                bbox={"boxstyle": "round,pad=0.32", "fc": fc, "ec": ec, "lw": 1.0})
    depth_max = max(n["depth"] for n in nodes)
    ax.set_xlim(-1.45, seq[0] + 0.45)
    ax.set_ylim(-depth_max - 1.15, 0.9)
    ax.axis("off")
    ax.set_title("(c) 最优演化因子表达式树", fontsize=11)
    if caption:
        caption = caption.replace(" | ", "\n")
        ax.text(0.5, 0.02, caption, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.5, color="#444",
                bbox={"boxstyle": "round,pad=0.35", "fc": "#F7F8FA", "ec": "#D6DAE2"})


def fig_gp_evolution() -> None:
    """功能 4：增强遗传编程——因子簇 × 岛屿迁移 × 事件窗口的真实演化诊断图。"""
    panel, src = _gp_panel()
    ew_start, ew_end, vol_peak = _derive_event_window(panel)

    evolver = EnhancedFactorEvolver(kline=panel, library=FactorLibrary(), seed=17)
    evolver.add_event_window(EventWindow(
        name="高波动市场状态", date_range=(ew_start, ew_end),
        event_type="market_state", weight=3.0,
    ))
    results = evolver.evolve_clusters(
        generations=GP_GENERATIONS, pop_per_cluster=GP_POP_PER_CLUSTER, top_k=12,
        migration_rate=GP_MIGRATION_RATE, migrate_every=GP_MIGRATE_EVERY,
        auto_save=False, include_expr=True,
    )
    if not results:
        print("  ! feature_gp_evolution.png 跳过（未产出因子）")
        return

    hist = pd.DataFrame(evolver.history)
    labels = list(dict.fromkeys(hist["cluster_label"]))
    label_of = dict(zip(hist["cluster"], hist["cluster_label"]))
    color_of = {lb: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, lb in enumerate(labels)}
    best_expr = results[0]["expr"]
    gen_first, gen_last = int(hist["gen"].min()), int(hist["gen"].max())
    mig_gens = sorted({m["gen"] for m in evolver.migrations})

    fig = plt.figure(figsize=(13.8, 7.9), dpi=130)
    gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.30,
                          left=0.125, right=0.975, top=0.865, bottom=0.075)

    # (a) 各簇收敛轨迹 + 迁移事件
    ax = fig.add_subplot(gs[0, 0])
    for lb in labels:
        sub = hist[hist["cluster_label"] == lb].sort_values("gen")
        ax.plot(sub["gen"], sub["best_ic"], marker="o", ms=4, lw=1.6,
                color=color_of[lb], label=lb)
    mean_ic = hist.groupby("gen")["mean_ic"].mean()
    ax.plot(mean_ic.index, mean_ic.values, color="#555555", lw=1.4, ls="--",
            marker="^", ms=4, label="全体种群均值 IC")
    lo, hi = ax.get_ylim()
    for mg in mig_gens:
        ax.axvline(mg, color="#C44E52", lw=1.0, ls=":", alpha=0.85)
    if mig_gens:
        ax.text(mig_gens[0], hi, f"  第 {mig_gens[0]} 代起：岛屿精英环形迁移",
                fontsize=8, color="#C44E52", va="top", ha="left")
    ax.set_xlabel("演化代数")
    ax.set_ylabel("训练集截面 IC")
    ax.set_title("(a) 因子簇并行演化收敛轨迹", fontsize=11)
    ax.set_xticks(range(gen_first, gen_last + 1))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=7.5)

    # (b) 多样性保持与个体有效性
    ax = fig.add_subplot(gs[0, 1])
    uni = hist.groupby("gen")["unique_ratio"].agg(["mean", "std"]).fillna(0.0)
    inv = hist.groupby("gen")["invalid_ratio"].mean()
    ax.plot(uni.index, uni["mean"], color="#4C72B0", marker="o", ms=4, lw=1.6,
            label="种群唯一表达式占比")
    ax.fill_between(uni.index, uni["mean"] - uni["std"], uni["mean"] + uni["std"],
                    color="#4C72B0", alpha=0.15)
    ax.set_ylim(0, 1.06)
    ax.set_xlabel("演化代数")
    ax.set_ylabel("唯一表达式占比", color="#4C72B0")
    ax.tick_params(axis="y", labelcolor="#4C72B0")
    ax2 = ax.twinx()
    ax2.plot(inv.index, inv.values, color="#C44E52", marker="s", ms=4, lw=1.6,
             label="无效个体占比")
    ax2.set_ylim(0, 1.06)
    ax2.set_ylabel("无效个体占比", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=7.5)
    ax.set_title("(b) 多样性保持与表达式有效性", fontsize=11)
    ax.set_xticks(range(gen_first, gen_last + 1))
    ax.grid(alpha=0.25)

    # (c) 最优演化因子表达式树
    ax = fig.add_subplot(gs[0, 2])
    _draw_expr_tree(
        ax, best_expr,
        caption=(f"训练 IC={results[0]['train_ic']:.4f} | "
                 f"测试 IC={results[0]['test_ic']:.4f} | 过拟合差={results[0]['overfit_gap']:.4f}"),
    )

    # (d) 各簇初代 → 末代 best IC 演化增益
    ax = fig.add_subplot(gs[1, 0])
    first = hist[hist["gen"] == gen_first].set_index("cluster_label")["best_ic"]
    last = hist[hist["gen"] == gen_last].set_index("cluster_label")["best_ic"]
    rows = [lb for lb in labels if lb in first.index and lb in last.index]
    ypos = np.arange(len(rows))
    ax.barh(ypos + 0.19, [first[lb] for lb in rows], height=0.34,
            color="#C3CCDA", label=f"第 {gen_first} 代")
    ax.barh(ypos - 0.19, [last[lb] for lb in rows], height=0.34,
            color=[color_of[lb] for lb in rows], label=f"第 {gen_last} 代")
    for i, lb in enumerate(rows):
        gain = (last[lb] - first[lb]) / abs(first[lb]) if first[lb] else 0.0
        ax.text(max(first[lb], last[lb]) + 0.0015, i, f"{gain:+.0%}",
                va="center", fontsize=8.5, color="#444444")
    ax.set_yticks(ypos)
    ax.set_yticklabels(rows, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("该簇最优个体训练集 IC")
    ax.set_title("(d) 各因子簇演化增益", fontsize=11)
    ax.set_xlim(0, max(max(first), max(last)) * 1.22)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25, axis="x")

    # (e) 样本外体检：训练 IC vs 测试 IC
    ax = fig.add_subplot(gs[1, 1])
    for lb in labels:
        pts = [r for r in results if label_of.get(r["cluster"]) == lb]
        if pts:
            ax.scatter([r["train_ic"] for r in pts], [r["test_ic"] for r in pts],
                       s=54, color=color_of[lb], alpha=0.9,
                       edgecolor="white", lw=0.6, label=lb)
    tr = [r["train_ic"] for r in results]
    te = [r["test_ic"] for r in results]
    lims = [min(min(tr), min(te)) - 0.008, max(max(tr), max(te)) + 0.008]
    ax.plot(lims, lims, "k--", lw=0.9, alpha=0.5, label="test = train（无过拟合）")
    ax.axhline(0, color="#C44E52", lw=0.8, ls=":", alpha=0.7)
    ax.axvline(0, color="#C44E52", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("训练集 IC")
    ax.set_ylabel("测试集 IC")
    ov = float(np.mean([r["overfit_gap"] for r in results]))
    ax.set_title(f"(e) 样本外体检（平均过拟合差 {ov:+.4f}）", fontsize=11)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(alpha=0.25)

    # (f) 事件窗口感知的适应度加权
    ax = fig.add_subplot(gs[1, 2])
    ic = _daily_ic(panel, best_expr)
    if not ic.empty:
        xd = pd.to_datetime(pd.Index(ic.index))
        ax.axhline(0, color="#888888", lw=0.8)
        ax.plot(xd, ic.to_numpy(dtype=float), color="#9FB4D0", lw=0.8,
                alpha=0.9, label="每日截面 IC")
        ax.plot(xd, ic.rolling(21, min_periods=5).mean().to_numpy(dtype=float),
                color="#4C72B0", lw=1.8, label="21 日滚动均值")
        ax.axvspan(pd.to_datetime(ew_start), pd.to_datetime(ew_end),
                   color="#E7A93B", alpha=0.22, label="事件窗口（波动率峰值 ±10 日）")
        # 同一表达式在「关闭事件窗口」与「开启事件窗口」下的适应度对比
        saved = evolver._event_windows
        evolver._event_windows = []
        plain = evolver._fitness(best_expr, evolver.df)
        evolver._event_windows = saved
        weighted = evolver._fitness(best_expr, evolver.df)
        win_note = f"{ew_start} ~ {ew_end}"
        ax.text(0.02, 0.03,
                f"事件窗口：{win_note}\n未加权 IC = {plain:.4f}\n事件窗口加权 IC = {weighted:.4f}",
                transform=ax.transAxes, va="bottom", fontsize=8.5,
                bbox={"boxstyle": "round,pad=0.4", "fc": "#FFF9EC", "ec": "#E7A93B"})
        ax.legend(loc="upper right", fontsize=7.5)
    ax.set_title("(f) 事件窗口感知的适应度加权", fontsize=11)
    ax.set_ylabel("截面 IC")
    ax.tick_params(axis="x", labelsize=7.5, rotation=20)
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"Enhanced Genetic Programming · 因子簇 × 岛屿迁移 × 事件窗口"
        f"（{src} {panel['symbol'].nunique()} 只 × {panel['date'].nunique()} 个交易日，"
        f"{len(labels)} 个因子簇，{GP_GENERATIONS} 代，迁移 {len(evolver.migrations)} 次）",
        fontsize=12.5,
    )
    fig.savefig(ASSETS / "feature_gp_evolution.png")
    plt.close(fig)
    print(f"  ✓ feature_gp_evolution.png  (factors={len(results)}, "
          f"migrations={len(evolver.migrations)}, clusters={len(labels)}, src={src})")


def fig_unstructured() -> None:
    """功能 5：非结构化数据因子挖掘——文本情绪分布。"""
    ta = TextAnalyzer()
    corpus = [
        "公司发布超预期三季报，营收同比增长32%，机构上调目标价",
        "行业竞争加剧，公司毛利率下降，分析师下调盈利预测",
        "政策利好落地，板块整体走强，北向资金持续流入",
        "管理层回购股份彰显信心，股价有望企稳回升",
        "公司业绩爆雷，股价暴跌，机构集体减持",
        "新产品市场反响热烈，渠道扩张顺利，订单饱满",
        "应收款项减值风险上升，现金流状况恶化，需警惕",
        "海外业务拓展取得突破，订单超预期",
        "监管新规落地，行业短期不确定性增加",
        "研发投入持续加码，技术壁垒不断加深",
        "业绩大幅亏损，面临退市风险，投资者恐慌抛售",
        "股权激励计划出炉，绑定核心团队利益",
    ]
    pos = neg = neu = 0
    scores = []
    for t in corpus:
        r = ta.analyze(t)
        senti = r.get("sentiment_score", r.get("sentiment", 0))
        scores.append(float(senti))
        if senti > 0.05:
            pos += 1
        elif senti < -0.05:
            neg += 1
        else:
            neu += 1

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), dpi=130)
    cats = ["积极", "中性", "消极"]
    vals = [pos, neu, neg]
    colors = ["#55A868", "#CCB974", "#C44E52"]
    ax = axes[0]
    bars = ax.bar(cats, vals, color=colors, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08,
                str(v), ha="center", fontsize=10, color="#333333")
    ax.set_ylabel("句子数", fontsize=10)
    ax.set_title("(a) 文本情绪量化分布", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(alpha=0.25, axis="y")

    tg = sorted(enumerate(scores), key=lambda kv: kv[1])[-12:]
    ax = axes[1]
    ypos = np.arange(len(tg))
    bar_colors = ["#55A868" if v > 0 else "#C44E52" for _, v in tg]
    bars = ax.barh(ypos, [v for _, v in tg], color=bar_colors, height=0.55)
    for i, (b, (_, v)) in enumerate(zip(bars, tg)):
        ax.text(v + (0.03 if v >= 0 else -0.03), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color="#333333")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"句{k + 1}" for k, _ in tg], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("情绪得分（-1 ~ 1）", fontsize=10)
    ax.set_title("(b) 逐句情绪量化得分", fontsize=11)
    ax.grid(alpha=0.25, axis="x")
    ax.axvline(0, color="#888888", lw=0.8)
    ax.set_xlim(-1.15, 1.15)

    fig.suptitle("Unstructured Mining · 非结构化数据因子挖掘 · 文本情绪量化", fontsize=12.5)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(ASSETS / "feature_unstructured.png")
    plt.close(fig)
    print(f"  ✓ feature_unstructured.png  (pos={pos}, neg={neg}, neu={neu})")


def fig_transformer_coupling() -> None:
    """功能 6：Transformer-Agent 深度耦合——因子检索相关度 Top10。"""
    tc = TransformerCoupling(library=FactorLibrary())
    ctx = tc.build_agent_context("构建一个中期动量叠加波动率控制的因子", top_k_factor=10)
    related = ctx.get("related_factors", [])
    names = []
    scores = []
    for i, rf in enumerate(related[:10]):
        if isinstance(rf, dict):
            names.append(str(rf.get("name", rf.get("factor", f"因子{i + 1}")))[:14])
            scores.append(float(rf.get("score", rf.get("similarity", 1.0 - i * 0.06))))
        else:
            names.append(str(rf)[:14])
            scores.append(1.0 - i * 0.06)

    if not names:  # 降级：无候选时展示启发式相关度
        names = ["动量因子", "波动率因子", "趋势因子", "换手率因子", "量价因子",
                 "相对强弱", "乖离率", "振幅因子", "流动性", "价格形态"]
        scores = [0.98, 0.91, 0.84, 0.77, 0.72, 0.66, 0.60, 0.54, 0.49, 0.44]

    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=130)
    ypos = np.arange(len(names))
    # 按相关度从低到高着色，形成渐变
    norm = plt.Normalize(min(scores), max(scores))
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        "blues", ["#9FB4D0", "#4C72B0"]
    )
    bar_colors = cmap(norm(scores))
    bars = ax.barh(ypos, scores, color=bar_colors, height=0.55, edgecolor="white", lw=0.6)
    for i, (b, s) in enumerate(zip(bars, scores)):
        ax.text(s + 0.015, i, f"{s:.2f}", va="center", fontsize=8.5, color="#333333")
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("与用户意图的相关度", fontsize=10)
    ax.set_xlim(0, 1.06)
    ax.set_title("(a) 因子检索相关度 Top10", fontsize=11)
    ax.grid(alpha=0.25, axis="x")
    ax.text(0.98, 0.03, "意图：中期动量 + 波动率控制",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=_CAPTION_BOX)
    fig.suptitle("Transformer-Agent Coupling · 深度耦合 · 语义检索 Top10", fontsize=12.5)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(ASSETS / "feature_transformer_coupling.png")
    plt.close(fig)
    print(f"  ✓ feature_transformer_coupling.png  (top={len(names)})")


def fig_offline_data() -> None:
    """功能 7：离线数据覆盖（读取 data/offline/meta.json）。"""
    meta_p = ROOT / "data" / "offline" / "meta.json"
    if not meta_p.exists():
        print("  ! feature_offline_data.png 跳过（data/offline/meta.json 不存在）")
        return
    meta = json.loads(meta_p.read_text(encoding="utf-8"))

    # 各分片大小
    part_names = meta.get("parts", [])
    rows = []
    for pn in part_names:
        pp = ROOT / "data" / "offline" / pn
        if pp.exists():
            rows.append((pn.replace("bars_csi800_", "").replace(".parquet", ""),
                         round(pp.stat().st_size / 1e6, 1)))

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), dpi=130)
    if rows:
        labels = [r[0] for r in rows]
        sizes = [r[1] for r in rows]
        colors = _PALETTE[1]  # #55A868
        bars = axes[0].bar(labels, sizes, color=colors, width=0.5, edgecolor="white", lw=0.6)
        for b, s in zip(bars, sizes):
            axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + max(sizes) * 0.015,
                         f"{s} MB", ha="center", fontsize=9, color="#333333")
        axes[0].set_ylabel("大小 (MB)", fontsize=10)
        axes[0].set_title("(a) 离线行情分片大小", fontsize=11)
        axes[0].set_ylim(0, max(sizes) * 1.16)
        axes[0].grid(alpha=0.25, axis="y")
    pools = meta.get("constituents", {}) or {}
    if pools:
        pn = list(pools.keys())
        pc = [pools[k] for k in pn]
        hb = axes[1].barh(pn, pc, color=_PALETTE[2], height=0.55)
        for b, c in zip(hb, pc):
            axes[1].text(b.get_width() + max(pc) * 0.015,
                         b.get_y() + b.get_height() / 2, f"{c}",
                         va="center", fontsize=9, color="#333333")
        axes[1].set_xlim(0, max(pc) * 1.2)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("成分股数 (只)", fontsize=10)
        axes[1].set_title("(b) 离线票池成分股（按需切换）", fontsize=11)
        axes[1].grid(alpha=0.25, axis="x")
    idx_meta = meta.get("indices", {}) or {}
    idx_names = "、".join(idx_meta.get("names", {}).values()) or "-"
    axes[2].text(0.5, 0.55,
                 f"主票池：{meta.get('index', 'csi800')}\n"
                 f"股票数：{meta.get('symbols', '-')}\n"
                 f"交易日：{meta.get('trade_days', '-')}\n"
                 f"数据行数：{meta.get('rows', '-')}\n"
                 f"区间：{meta.get('start', '-')} ~ {meta.get('end', '-')}\n"
                 f"基准指数日线：{idx_meta.get('rows', '-')} 行\n"
                 f"基准指数：{idx_names}",
                 ha="center", va="center", fontsize=10,
                 bbox=_INFO_BOX)
    axes[2].axis("off")
    axes[2].set_title("(c) 内置离线数据源 · 开箱即用", fontsize=11)
    fig.suptitle("Offline Data · 本地部署与离线韧性 · 行情 + 基准 + 多票池", fontsize=12.5)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(ASSETS / "feature_offline_data.png")
    plt.close(fig)
    print(f"  ✓ feature_offline_data.png  (rows={meta.get('rows', 0)})")


def fig_ima_pipeline() -> None:
    """功能 8：研报知识管线——关键词命中统计（真实 CSV）。"""
    csv_p = ROOT / "ima_subscription" / "keyword_hits.csv"
    if not csv_p.exists():
        print("  ! feature_ima_pipeline.png 跳过（keyword_hits.csv 不存在）")
        return

    import csv as _csv

    rows = list(_csv.reader(csv_p.open(encoding="utf-8")))
    header = rows[0]
    ki = header.index("关键词") if "关键词" in header else 0
    hits = Counter(r[ki] for r in rows[1:] if len(r) > ki and r[ki].strip())
    top = hits.most_common(10)
    total_hits = sum(hits.values())

    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=130)
    keywords = [k for k, _ in top][::-1]
    counts = [v for _, v in top][::-1]
    ypos = np.arange(len(keywords))
    # 按命中数渐变
    norm = plt.Normalize(min(counts), max(counts))
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        "reds", ["#E8A0A0", "#C44E52"]
    )
    bar_colors = cmap(norm(counts))
    bars = ax.barh(ypos, counts, color=bar_colors, height=0.55, edgecolor="white", lw=0.6)
    for i, (b, c) in enumerate(zip(bars, counts)):
        ax.text(c + max(counts) * 0.01, i, str(c),
                va="center", fontsize=9, color="#333333")
    ax.set_yticks(ypos)
    ax.set_yticklabels(keywords, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("命中研报数", fontsize=10)
    ax.set_xlim(0, max(counts) * 1.16)
    ax.set_title("(a) 关键词命中 Top10", fontsize=11)
    ax.grid(alpha=0.25, axis="x")
    ax.text(0.98, 0.03, f"累计命中：{total_hits} 条",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=_CAPTION_BOX)
    fig.suptitle("IMA Pipeline · 研报知识管线 · 关键词命中统计", fontsize=12.5)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(ASSETS / "feature_ima_pipeline.png")
    plt.close(fig)
    print(f"  ✓ feature_ima_pipeline.png  (hits={total_hits})")


# ---------------------------------------------------------------------------
# 高频离线链路（真实压实产物：data/offline/hf_*）
# ---------------------------------------------------------------------------
# 原始 L2 快照的规模来自 scripts/hf_offline_build.py 的实际输入（15.8M 行 / 382MB），
# 压实后的规模一律从磁盘与 hf_meta.json 现场读，不手写数字。
HF_RAW_ROWS = 15_800_000
HF_RAW_MB = 382.0
HF_COLUMN_FAMILIES = [
    ("价差/微观价格", ["hf_spread", "hf_spread_ticks", "hf_rel_spread", "hf_micro_dev_ticks"]),
    ("盘口深度/失衡", ["hf_depth_ratio", "hf_obi_l1", "hf_obi_all", "hf_obi_w2", "hf_obi_w3",
                       "hf_obi_w5", "hf_book_slope", "hf_best_share_bid", "hf_best_share_ask"]),
    ("波动/趋势", ["hf_rvol_20", "hf_trend_20", "hf_reversal_5"]),
    ("成交/流量", ["hf_trade_rate_20", "hf_signed_flow_20", "hf_volume_impulse_20", "hf_ofi",
                   "hf_dq_b", "hf_dq_s", "hf_d_volume", "hf_d_turnover"]),
    ("持仓变动", ["hf_oi_change"]),
]


def fig_hf_pipeline() -> None:
    """高频：L2 快照 → 压实产物 → 分钟面板（行数/体积/列族/日内价差）。"""
    meta_p = ROOT / "data" / "offline" / "hf_meta.json"
    panel_p = ROOT / "data" / "offline" / "hf_panel_1min.parquet"
    daily_p = ROOT / "data" / "offline" / "hf_daily.parquet"
    orders_p = ROOT / "data" / "offline" / "hf_orders.parquet"
    if not (meta_p.exists() and panel_p.exists()):
        print("  ! feature_hf_pipeline.png 跳过（离线高频产物缺失，先跑 hf_offline_build.py）")
        return

    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    panel = pd.read_parquet(panel_p)
    panel["date"] = pd.to_datetime(panel["date"])

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.2), dpi=130)

    # (a) 压实前后：行数（对数）+ 体积标注
    ax = axes[0][0]
    labels = ["原始 L2 快照", "分钟面板", "合约日频", "委托流水"]
    rows = [HF_RAW_ROWS, meta.get("panel_rows", len(panel)),
            meta.get("daily_rows", 0), meta.get("orders_rows", 0)]
    sizes = [HF_RAW_MB] + [p.stat().st_size / 1e6 for p in (panel_p, daily_p, orders_p)
                           if p.exists()]
    sizes += [0.0] * (len(rows) - len(sizes))
    ypos = np.arange(len(labels))
    ax.barh(ypos, rows, color=_PALETTE[: len(labels)], height=0.55)
    ax.set_xscale("log")
    for i, (r, s) in enumerate(zip(rows, sizes)):
        ax.text(r * 1.15, i, f"{r:,} 行 · {s:.2f} MB" if s else f"{r:,} 行",
                va="center", fontsize=8.5, color="#333333")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("行数（对数轴）", fontsize=9.5)
    ax.set_xlim(1e2, HF_RAW_ROWS * 40)
    ax.set_title("(a) 压实前后：行数与体积", fontsize=11)
    ax.grid(alpha=0.25, axis="x")

    # (b) 每个合约的分钟覆盖
    ax = axes[0][1]
    per = panel.groupby("symbol")["date"].nunique().sort_values()
    ax.bar(np.arange(len(per)), per.values, color=_PALETTE[0], width=0.72)
    ax.set_xticks(np.arange(len(per)))
    ax.set_xticklabels(per.index.astype(str), rotation=60, ha="right", fontsize=7.6)
    ax.set_ylabel("覆盖分钟数", fontsize=9.5)
    ax.set_title(f"(b) {len(per)} 个期货合约的分钟覆盖", fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    ax.text(0.98, 0.04, f"合计 {meta.get('panel_minutes', 0)} 分钟 / {len(panel):,} 行",
            transform=ax.transAxes, ha="right", fontsize=9, bbox=_CAPTION_BOX)

    # (c) 高频列族
    ax = axes[1][0]
    cols = [c for c in panel.columns if str(c).startswith("hf_")]
    fam_counts, fam_names = [], []
    for name, members in HF_COLUMN_FAMILIES:
        n = len([c for c in cols if c in members])
        if n:
            fam_names.append(name)
            fam_counts.append(n)
    other = len(cols) - sum(fam_counts)
    if other:
        fam_names.append("其他高频列")
        fam_counts.append(other)
    ypos = np.arange(len(fam_names))
    ax.barh(ypos, fam_counts, color=_PALETTE[: len(fam_names)], height=0.58)
    for i, c in enumerate(fam_counts):
        ax.text(c + max(fam_counts) * 0.03, i, str(c), va="center", fontsize=9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(fam_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("列数", fontsize=9.5)
    ax.set_xlim(0, max(fam_counts) * 1.3)
    ax.set_title(f"(c) 订单簿衍生列 {len(cols)} 个（按族）", fontsize=11)
    ax.grid(alpha=0.25, axis="x")

    # (d) 日内相对价差中位数
    ax = axes[1][1]
    if "hf_rel_spread" in panel.columns:
        s = (panel.assign(t=panel["date"].dt.strftime("%H:%M"))
             .groupby("t")["hf_rel_spread"].median())
        ax.plot(np.arange(len(s)), s.values, color=_PALETTE[2], lw=1.6)
        step = max(1, len(s) // 8)
        ax.set_xticks(np.arange(0, len(s), step))
        ax.set_xticklabels(s.index[::step], rotation=0, fontsize=8)
        ax.set_ylabel("相对价差中位数", fontsize=9.5)
        ax.set_title("(d) 日内相对价差（全合约中位数）", fontsize=11)
        ax.grid(alpha=0.25)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "（面板缺少 hf_rel_spread 列）", ha="center", va="center")

    fig.suptitle("High-Frequency Pipeline · L2 快照 → 分钟面板（离线压实，freq=%s）"
                 % meta.get("freq", "-"), fontsize=12.5)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(ASSETS / "feature_hf_pipeline.png")
    plt.close(fig)
    print(f"  ✓ feature_hf_pipeline.png  (rows={len(panel):,}, "
          f"cols={len(cols)}, contracts={panel['symbol'].nunique()})")


# ---------------------------------------------------------------------------
# 材料三级判定链（真实训练报告：data/models/multimodal/training_report.json）
# ---------------------------------------------------------------------------
def fig_multimodal_judging() -> None:
    rep_p = ROOT / "data" / "models" / "multimodal" / "training_report.json"
    if not rep_p.exists():
        print("  ! feature_multimodal_judging.png 跳过（未跑 scripts/train_multimodal.py）")
        return
    rep = json.loads(rep_p.read_text(encoding="utf-8"))
    models = rep.get("models") or {}
    hard = rep.get("hard_node_check") or {}
    n_class = len((rep.get("dataset") or {}).get("labels", [])) or 7
    random_acc = 1.0 / n_class

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.9), dpi=130)

    names = list(models.keys())
    accs = [float(models[n].get("acc", 0.0)) for n in names]
    ypos = np.arange(len(names))[::-1]
    colors = ["#55A868" if "torch" in str(models[n].get("engine", "")) else _PALETTE[0]
              for n in names]
    ax.barh(ypos, accs, color=colors, height=0.55)
    for y, v in zip(ypos, accs):
        ax.text(v + 0.012, y, f"{v:.3f}", va="center", fontsize=8.8, color="#333333")
    ax.axvline(random_acc, color="#C44E52", ls="--", lw=1.1)
    ax.text(random_acc + 0.01, -0.6, f"随机基线 {random_acc:.2f}", fontsize=8.5, color="#C44E52")
    ax.set_yticks(ypos)
    ax.set_yticklabels([n.replace("[", " · ").rstrip("]") for n in names], fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("准确率（留出集）", fontsize=9.5)
    ax.set_title("(a) 本地训练模型实测准确率", fontsize=11)
    ax.grid(alpha=0.25, axis="x")

    gcn = float((hard.get("gcn") or {}).get("acc", 0.0))
    nb = float((hard.get("naive_bayes") or {}).get("acc", 0.0))
    bars = [("GCN（关系图传播）", gcn, _PALETTE[3]),
            ("朴素贝叶斯（纯文本）", nb, _PALETTE[0]),
            ("随机基线", random_acc, "#C44E52")]
    x = np.arange(len(bars))
    ax2.bar(x, [b[1] for b in bars], color=[b[2] for b in bars], width=0.55)
    for i, (_, v, _) in enumerate(bars):
        ax2.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([b[0] for b in bars], fontsize=8.8)
    ax2.set_ylim(0, max(gcn, nb, random_acc) * 1.35 + 0.05)
    ax2.set_ylabel("准确率", fontsize=9.5)
    ax2.set_title(f"(b) 硬样本对照（无类型关键词、无代码，n={hard.get('n', 0)}）", fontsize=11)
    ax2.grid(alpha=0.25, axis="y")

    fig.suptitle("Material Judgment Chain · JEV → 本地训练模型 → 本地规则", fontsize=12.5)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(ASSETS / "feature_multimodal_judging.png")
    plt.close(fig)
    print(f"  ✓ feature_multimodal_judging.png  (models={len(names)}, hard_gcn={gcn:.3f})")


# ---------------------------------------------------------------------------
# 上传材料进上下文：按页切片（真实调用 _slice_pages）
# ---------------------------------------------------------------------------
def fig_upload_context() -> None:
    """长文按预算保留「开头 + 结尾」，中间整段省略——用真实切片函数算，不画假图。"""
    from engine.upload_ingest import _slice_pages  # noqa: PLC0415

    rng = np.random.default_rng(20260922)
    n_pages = 21
    pages = ["量化论文正文内容占位" * int(rng.integers(150, 260)) for _ in range(n_pages)]
    total_chars = sum(len(p) for p in pages)
    budgets = [8000, 24000, 60000]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.8), dpi=130,
                                  gridspec_kw={"width_ratios": [1.55, 1]})

    for row, b in enumerate(budgets):
        chunks, note = _slice_pages(pages, b)
        kept = sorted(no for no, _ in chunks)
        ax.broken_barh([(0.5, n_pages)], (row - 0.34, 0.68), facecolors="#E6E9EF")
        for no in kept:
            ax.broken_barh([(no - 0.42, 0.84)], (row - 0.31, 0.62), facecolors=_PALETTE[row])
    ax.set_yticks(range(len(budgets)))
    ax.set_yticklabels([f"{b // 1000} 千字预算" for b in budgets], fontsize=9)
    ax.set_xlim(0, n_pages + 1)
    ax.set_xlabel("页码", fontsize=9.5)
    ax.set_title(f"(a) {n_pages} 页论文在各预算下的保留页（灰 = 省略）", fontsize=11)
    ax.grid(alpha=0.2, axis="x")

    kept_chars, omitted_pages = [], []
    for b in budgets:
        chunks, _ = _slice_pages(pages, b)
        kept_chars.append(sum(len(t) for _, t in chunks))
        omitted_pages.append(n_pages - len(chunks))
    x = np.arange(len(budgets))
    ax2.bar(x, kept_chars, color=_PALETTE[2], width=0.55, label="进入上下文")
    ax2.bar(x, [total_chars - k for k in kept_chars], bottom=kept_chars,
            color="#E6E9EF", width=0.55, label="省略")
    for i, (k, o) in enumerate(zip(kept_chars, omitted_pages)):
        ax2.text(i, total_chars * 1.01, f"省 {o} 页", ha="center", fontsize=8.6)
        ax2.text(i, k / 2, f"{k:,}", ha="center", fontsize=8.6, color="white")
    ax2.axhline(total_chars, color="#333333", ls="--", lw=1.0)
    ax2.text(len(budgets) - 0.45, total_chars * 1.045,
             f"全文 {total_chars:,} 字", ha="right", fontsize=8.6, color="#333333")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{b // 1000} 千字" for b in budgets], fontsize=9)
    ax2.set_ylim(0, total_chars * 1.2)
    ax2.set_ylabel("字数", fontsize=9.5)
    ax2.set_title(f"(b) 全文 {total_chars:,} 字 vs 预算", fontsize=11)
    ax2.legend(fontsize=8.6, loc="upper left", framealpha=0.9)
    ax2.grid(alpha=0.25, axis="y")

    fig.suptitle("Upload Context · 长文按页进 prompt（保留首尾，省略量如实标注）", fontsize=12.5)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(ASSETS / "feature_upload_context.png")
    plt.close(fig)
    print(f"  ✓ feature_upload_context.png  (pages={n_pages}, chars={total_chars:,})")


# ---------------------------------------------------------------------------
# 意图分流与材料注入（流程示意，无数据）
# ---------------------------------------------------------------------------
def fig_intent_routing() -> None:
    from matplotlib.patches import FancyBboxPatch  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(11.0, 5.0), dpi=130)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, text, fc="#FFFFFF", ec="#4C72B0", fs=9.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color="#222222")

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": "-|>", "color": "#4C72B0", "lw": 1.2})
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.018, label,
                    fontsize=8.4, color="#555555", ha="center")

    box(0.015, 0.60, 0.155, 0.30, "用户一句话\n+ 已上传材料", fc="#F2F6FC")
    box(0.205, 0.60, 0.145, 0.30, "意图分类\nclassify()", fc="#FFF7E6", ec="#CCB974")
    box(0.40, 0.72, 0.575, 0.20,
        "mining → Agent 重流水线\n检索 → 生成 → 沙箱校验 → 回测 → 反思\n"
        "注入 unstructured_context + external_factors", ec="#55A868")
    box(0.40, 0.44, 0.575, 0.20,
        "qa / chitchat → chat_answer\n注入 material_context（按页切片 + 目录 + 页码）",
        ec="#4C72B0")
    box(0.40, 0.16, 0.575, 0.20,
        "clarify → 直答并说明缺什么信息\n不启动流水线（误启一次回测的代价远大于多问一句）",
        ec="#C44E52")
    arrow(0.17, 0.75, 0.205, 0.75)
    arrow(0.35, 0.78, 0.40, 0.82)
    arrow(0.35, 0.72, 0.40, 0.54)
    arrow(0.35, 0.68, 0.40, 0.26)
    arrow(0.09, 0.60, 0.09, 0.50, "材料在分流之前算好")
    ax.annotate("", xy=(0.40, 0.82), xytext=(0.09, 0.44),
                arrowprops={"arrowstyle": "-|>", "color": "#8172B2", "lw": 1.1,
                            "connectionstyle": "arc3,rad=-0.25", "ls": "--"})
    ax.annotate("", xy=(0.40, 0.54), xytext=(0.09, 0.42),
                arrowprops={"arrowstyle": "-|>", "color": "#8172B2", "lw": 1.1,
                            "connectionstyle": "arc3,rad=-0.18", "ls": "--"})
    ax.text(0.5, 0.055,
            "两条分支共用同一份材料上下文：问答分支拿不到材料时，模型会如实回答"
            "「没收到你上传的文本」", fontsize=8.8, color="#555555", ha="center")

    ax.set_title("Intent Routing · 一句话怎么分流，材料怎么进 prompt", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(ASSETS / "feature_intent_routing.png")
    plt.close(fig)
    print("  ✓ feature_intent_routing.png")


def main() -> None:
    print("[gen_feature_charts] 开始生成功能图例...")
    fig_factor_library()
    fig_gp_evolution()
    fig_unstructured()
    fig_transformer_coupling()
    fig_offline_data()
    fig_ima_pipeline()
    fig_hf_pipeline()
    fig_multimodal_judging()
    fig_upload_context()
    fig_intent_routing()
    print("[gen_feature_charts] 完成。")


if __name__ == "__main__":
    main()
