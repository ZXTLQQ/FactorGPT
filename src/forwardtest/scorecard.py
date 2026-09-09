"""scorecard — 前瞻检验评分卡（与回测完全独立的统计）。

对 Headline Arena 已结算的账本记录计算：
- 方向准确率（命中率）、平均置信度；
- HA 官方计分：正确 50 + confidence*50，错误 50 - confidence*50（取结算记录平均）；
- Brier Score（三分类：方向预测只给主方向置信度，其余两个方向均分剩余概率质量；
  随机基准（恒 1/3）的 Brier = 0.6667，越接近 0 越准）；
- 校准曲线：按预测置信度分桶，比较「桶内平均置信度」与「桶内实际命中率」，
  直观检验乐观偏差——这正对应 HA 官方 per-agent distribution-calibration 的
  directional 版本（官方为宏观 CRPS 的 PIT 校准，方向挑战用本卡代替）。

评分只统计真实提交（mode=live）与 paper 信号（mode=paper，counts_for_score=False）；
dry_run 影子记录仅作流程审计，不进入任何得分统计。
"""

from __future__ import annotations

import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional

from .translator import ASSET_META, DIRECTIONS

SCORING_MODES = ("live", "paper")

# 校准分桶（置信度区间）
CONF_BINS = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]


def ha_directional_score(is_correct: bool, confidence: float) -> float:
    """HA 方向性计分公式：正确 50+50c；错误 50-50c。"""
    c = float(confidence)
    return 50.0 + (c * 50.0 if is_correct else -c * 50.0)


def brier_for_record(record: Dict[str, Any]) -> float:
    """单条 Brier：sum (p_i - o_i)^2，三分类。

    probabilities 由账本写入时生成（主方向置信度 + 其余均分）；
    observed 取结算 result 的 one-hot。
    """
    probs = record.get("probabilities") or {}
    result = ((record.get("settled") or {}).get("result") or "").lower()
    if result not in DIRECTIONS:
        raise ValueError(f"结算 result 非法: {result!r}")
    obs = {d: (1.0 if d == result else 0.0) for d in DIRECTIONS}
    total = 0.0
    for d in DIRECTIONS:
        p = float(probs.get(d, 0.0))
        total += (p - obs[d]) ** 2
    return total


def _favored_hit(record: Dict[str, Any]) -> bool:
    direction = str(record.get("direction", "")).lower()
    result = ((record.get("settled") or {}).get("result") or "").lower()
    if not direction or not result:
        return False
    return direction == result


def settled_scorable(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in records
            if r.get("settled") and r.get("mode") in SCORING_MODES]


def aggregate_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对（结算且可计分）记录做整体统计。"""
    scorable = settled_scorable(records)
    n = len(scorable)
    if n == 0:
        return {"n": 0, "accuracy": None, "mean_confidence": None,
                "mean_ha_score": None, "brier": None, "brier_baseline": 2.0 / 3.0,
                "by_direction": {}, "by_result": {}}
    hits = sum(1 for r in scorable if _favored_hit(r))
    confs = [float(r.get("confidence", 0.5)) for r in scorable]
    ha_scores = [ha_directional_score(_favored_hit(r), float(r.get("confidence", 0.5)))
                 for r in scorable]
    brier = sum(brier_for_record(r) for r in scorable) / n
    by_direction = {d: sum(1 for r in scorable if r.get("direction") == d) for d in DIRECTIONS}
    by_result = {d: sum(1 for r in scorable
                        if ((r.get("settled") or {}).get("result") or "").lower() == d)
                 for d in DIRECTIONS}
    return {
        "n": n, "accuracy": hits / n, "mean_confidence": statistics.fmean(confs),
        "mean_ha_score": statistics.fmean(ha_scores), "brier": brier,
        "brier_baseline": 2.0 / 3.0,
        "by_direction": by_direction, "by_result": by_result,
    }


def calibration_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """校准分桶：桶内平均置信度 vs 实际命中率。"""
    scorable = settled_scorable(records)
    rows: List[Dict[str, Any]] = []
    for lo, hi in CONF_BINS:
        bucket = [r for r in scorable
                  if lo <= float(r.get("confidence", 0.5)) < hi]
        if not bucket:
            rows.append({"bin": f"[{lo:.2f},{hi:.2f})", "n": 0,
                         "mean_forecast": None, "hit_rate": None, "gap": None})
            continue
        mean_conf = statistics.fmean(float(r.get("confidence", 0.5)) for r in bucket)
        hit = sum(1 for r in bucket if _favored_hit(r)) / len(bucket)
        rows.append({"bin": f"[{lo:.2f},{hi:.2f})", "n": len(bucket),
                     "mean_forecast": round(mean_conf, 3),
                     "hit_rate": round(hit, 3), "gap": round(hit - mean_conf, 3)})
    return rows


def _fmt(x: Any) -> str:
    return "-" if x is None else f"{x:.3f}"


def render_markdown(records: List[Dict[str, Any]], ledger_path: str = "") -> str:
    """渲染前瞻检验评分卡（Markdown，可直接落盘为报告）。"""
    all_records = list(records)
    scorable = settled_scorable(all_records)
    agg = aggregate_stats(all_records)
    live_n = sum(1 for r in all_records if r.get("mode") == "live")
    paper_n = sum(1 for r in all_records if r.get("mode") == "paper")
    dry_n = sum(1 for r in all_records if r.get("mode") == "dry_run")
    pending_n = sum(1 for r in all_records if not r.get("settled"))

    lines: List[str] = []
    lines.append("# FactorGPT · Headline Arena 前瞻检验评分卡")
    lines.append("")
    lines.append("> 与回测（IC/IR 历史检验）完全独立的 forward 检验线：预测在结果出现")
    lines.append("> 之前锁定，结算由第三方（headlinearena.com）按真实行情机械完成。")
    lines.append("")
    lines.append("## 一、样本概览")
    lines.append("")
    lines.append(f"- 账本路径：`{ledger_path or '-'}`")
    lines.append(f"- 总记录：{len(all_records)}（live={live_n}，paper={paper_n}，"
                 f"dry_run={dry_n}，待结算={pending_n}）")
    lines.append(f"- **计入评分的已结算预测：{agg['n']}**（mode ∈ {{live, paper}} 且已结算）")
    lines.append("")

    if agg["n"] == 0:
        lines.append("> 暂无已结算可计分样本。运行 `python scripts/ha_forward_run.py run` "
                     "积累预测，随后 `settle` 拉取第三方结算结果，再执行本命令生成评分卡。")
        lines.append("")
        if scorable:
            pass
        return "\n".join(lines)

    lines.append("## 二、整体表现")
    lines.append("")
    lines.append("| 指标 | 数值 | 备注 |")
    lines.append("|------|------|------|")
    lines.append(f"| 方向命中率 | {_fmt(agg['accuracy'])} | 主方向与结算结果一致的比例 |")
    lines.append(f"| 平均置信度 | {_fmt(agg['mean_confidence'])} | 三选一随机基准 = 0.333 |")
    lines.append(f"| HA 官方分（均） | {_fmt(agg['mean_ha_score'])} | 正确 50+50c / 错误 50-50c |")
    lines.append(f"| Brier Score | {_fmt(agg['brier'])} | 越小越准（随机基准 {agg['brier_baseline']:.4f}）|")
    lines.append("")
    lines.append("### 方向分布")
    lines.append("")
    lines.append("| 方向 | 预测次数 | 结算为 bullish | 结算为 bearish | 结算为 neutral |")
    lines.append("|------|------|------|------|------|")
    for d in DIRECTIONS:
        bd = agg["by_direction"].get(d, 0)
        res = agg["by_result"]
        lines.append(f"| {d} | {bd} | {res.get('bullish', 0)} | "
                     f"{res.get('bearish', 0)} | {res.get('neutral', 0)} |")
    lines.append("")

    lines.append("## 三、校准曲线（前瞻观点是否兑现）")
    lines.append("")
    lines.append("| 置信度桶 | 样本数 | 平均预测置信度 | 实际命中率 | 偏差(命中-预测) |")
    lines.append("|---------|------|--------------|-----------|---------------|")
    for row in calibration_rows(all_records):
        if row["n"] == 0:
            lines.append(f"| {row['bin']} | 0 | - | - | - |")
        else:
            lines.append(f"| {row['bin']} | {row['n']} | {row['mean_forecast']:.3f} | "
                         f"{row['hit_rate']:.3f} | {row['gap']:+.3f} |")
    lines.append("")
    lines.append("> 偏差持续为负 → 系统性乐观（置信度虚高）；持续为正 → 系统性保守。")
    lines.append("")

    # 分资产表现
    by_asset: Dict[str, List[Dict[str, Any]]] = {}
    for r in scorable:
        by_asset.setdefault(str(r.get("asset")), []).append(r)
    if by_asset:
        lines.append("## 四、分资产表现")
        lines.append("")
        lines.append("| 资产 | 名称 | 样本 | 命中率 | 均分 | Brier |")
        lines.append("|------|------|------|--------|------|-------|")
        for asset in sorted(by_asset):
            a = aggregate_stats(by_asset[asset])
            meta = ASSET_META.get(asset, {})
            lines.append(f"| {asset} | {meta.get('name', '-')} | {a['n']} | "
                         f"{_fmt(a['accuracy'])} | {_fmt(a['mean_ha_score'])} | {_fmt(a['brier'])} |")
        lines.append("")

    # 结算时间窗统计（可选，反映“前瞻”跨度）
    horizons: List[float] = []
    for r in scorable:
        created = r.get("created_at")
        resolved = (r.get("settled") or {}).get("resolved_at")
        try:
            t0 = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(str(resolved).replace("Z", "+00:00"))
            horizons.append((t1 - t0).total_seconds() / 86400.0)
        except (TypeError, ValueError):
            continue
    if horizons:
        lines.append(f"### 前瞻跨度\n\n平均 {statistics.fmean(horizons):.1f} 天，"
                     f"中位 {statistics.median(horizons):.1f} 天（预测锁定 → 结算）\n")
    lines.append("---")
    lines.append("*本评分卡由 FactorGPT forwardtest 自动生成，仅用于学术/研究用途，"
                 "不构成投资建议。*")
    lines.append("")
    return "\n".join(lines)


def reconcile_pending(ledger_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """列出待结算记录的简要信息（供 settle 步骤驱动）。"""
    return [{"uid": r.get("uid"), "challenge_id": r.get("challenge_id"),
             "asset": r.get("asset"), "created_at": r.get("created_at"),
             "direction": r.get("direction")} for r in ledger_records
            if not r.get("settled") and r.get("challenge_id")]
