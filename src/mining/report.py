# -*- coding: utf-8 -*-
"""因子研究报告生成：把挖掘层的中间产物落成可复核的 Markdown / JSON。

为什么研究层需要"报告"这一层：四篇研报给出的不只是算子，还有**验收口径**。
山西证券的网格搜索要求交代层级轨迹与剪枝原因，天风证券的风险闸门要求逐项
列出 ΔR² / |t| / VIF / 拥挤度，中信建投的 PIT 要求说明"每个字段的可用日"。
这些东西如果只留在内存对象里，评审时就得重跑一遍；写成报告后，结论与证据
被固定在同一份文件里，跑实验的人自己也更容易发现自己算错了。

设计约束：

1. **不引入依赖**，只用标准库 + pandas/numpy；重模块（``evaluator`` /
   ``gridminer`` / ``panel``）仅在类型检查时导入，运行期靠鸭子类型访问，
   因此 ``import mining.report`` 不会把 numba 之类的东西提前拉起来。
2. **缺什么就少写什么**，绝不因为某个输入没给就抛异常——报告生成器在流水线
   末尾运行，此时最不该做的事就是因为"少了一个可选字段"而整条链路失败。
3. **数字必须可读**：NaN / inf 一律渲染成 ``—``，而不是让 ``nan`` 混进文档
   冒充一个真实数值；布尔量渲染成 ``OK`` / ``预警``。
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:                                     # pragma: no cover
    from .evaluator import FactorReport
    from .gridminer import SearchResult
    from .panel import PanelData

__all__ = [
    "ReportInput", "build_from_expressions", "render_markdown", "report_payload",
    "write_report", "DASH",
]

DASH = "—"          # 缺失值占位符：文档里出现 "nan" 会被误读成真实数值


# --------------------------------------------------------------------------
# 0. 渲染原语
# --------------------------------------------------------------------------
def _is_real(v: Any) -> bool:
    """判断是否为"可展示的实数"（排除 NaN / inf / 残缺数字类型）。"""
    if isinstance(v, (bool, np.bool_)):
        return False
    if isinstance(v, (int, float, np.integer, np.floating)):
        try:
            return bool(np.isfinite(float(v)))
        except (TypeError, ValueError):
            return False
    return False


def _fmt(v: Any, nd: int = 3, pct: bool = False, signed: bool = False) -> str:
    """统一的数字渲染：不可展示 → ``—``；``pct`` 乘 100 并带百分号。"""
    if isinstance(v, (bool, np.bool_)):
        return "是" if v else "否"
    if not _is_real(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return DASH
        return str(v)
    x = float(v)
    if pct:
        return f"{x * 100:.{max(nd - 1, 1)}f}%"
    if not signed and abs(x) >= 1e5:
        return f"{x:.4g}"              # VIF 之类的量级指标：别写成十几位小数
    if signed:
        return f"{x:+.{nd}f}"
    return f"{x:.{nd}f}"


def _cfg_fmt(v: Any) -> str:
    """配置项渲染：整数保持整数、浮点去掉无意义尾零、序列逐项列出。

    配置表不能用 :func:`_fmt`——``max_expr=120`` 被写成 ``120.000`` 会让人以为
    这个参数是连续量。
    """
    if isinstance(v, (bool, np.bool_)):
        return "是" if v else "否"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        x = float(v)
        return DASH if not np.isfinite(x) else f"{x:g}"
    if isinstance(v, (list, tuple, set)):
        return "[" + ", ".join(_cfg_fmt(i) for i in v) + "]"
    if isinstance(v, str):
        return v if v.strip() else DASH
    return str(v)


def _escape(text: Any) -> str:
    """转义竖线：``|t|`` 这类内容会把 Markdown 表格的列切碎。"""
    return str(text).replace("|", "\\|")


def _looks_numeric(cell: Any) -> bool:
    s = str(cell).strip()
    if not s or s == DASH:
        return False
    try:
        float(s.replace("%", "").replace(",", ""))
    except ValueError:
        return False
    return True


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]],
           aligns: Optional[Sequence[str]] = None) -> str:
    """Markdown 表格；缺省按列内容判定对齐（无数值 → 左对齐）。"""
    heads = [str(h) for h in headers]
    cols = len(heads)
    if aligns is None:
        aligns = ["r"] * cols
    fixed: List[str] = []
    for j, a in enumerate(aligns):
        has_num = any(_looks_numeric(r[j]) for r in rows if len(r) > j)
        fixed.append("l" if (a == "r" and not has_num) else a)
    sep = [{"l": ":---", "c": ":---:", "r": "---:"}.get(a, "---:")
           for a in fixed]
    out = ["| " + " | ".join(_escape(h) for h in heads) + " |",
           "| " + " | ".join(sep) + " |"]
    for row in rows:
        cells = [_escape(c) for c in row]
        cells += [""] * max(0, cols - len(cells))
        out.append("| " + " | ".join(cells[:cols]) + " |")
    return "\n".join(out)


def _kv_table(items: Sequence[Sequence[Any]], key_head: str = "指标",
              val_head: str = "取值") -> str:
    return _table([key_head, val_head], items, aligns=["l", "r"])


def _bullet(lines: Sequence[str]) -> str:
    return "\n".join(f"- {s}" for s in lines)


# --------------------------------------------------------------------------
# 1. 输入容器
# --------------------------------------------------------------------------
@dataclass
class ReportInput:
    """报告的全部输入；**每一项都是可选的**，缺项对应的章节直接省略。

    ``panel`` 与 ``search`` / ``reports`` 至少给一个，否则报告只剩标题。
    """

    panel: Optional["PanelData"] = None
    title: str = "因子挖掘研究报告"
    reports: Sequence["FactorReport"] = ()
    search: Optional["SearchResult"] = None
    history: Optional[Mapping[str, Any]] = None
    pit_checks: Sequence[Mapping[str, Any]] = ()
    risk: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    pools: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    incremental: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    corr: Optional[pd.DataFrame] = None
    notes: Sequence[str] = ()
    generated_at: str = ""

    def stamp(self) -> str:
        """生成时刻（UTC，ISO8601）。显式传入时为可复现的定值。"""
        if self.generated_at:
            return self.generated_at
        return datetime.now(timezone.utc).replace(
            microsecond=0).isoformat().replace("+00:00", "Z")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """对象/字典双通道取值，避免报告生成器依赖具体是 dataclass 还是 dict。"""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _count(obj: Any) -> int:
    """安全取长度：``pd.Index`` / ``DataFrame`` 上写 ``x or ()`` 会抛真值歧义。"""
    if obj is None:
        return 0
    try:
        return int(len(obj))
    except TypeError:                                # pragma: no cover - 标量兜底
        return 0


def _node_text(node: Any) -> str:
    if node is None:
        return DASH
    for meth in ("render",):
        fn = getattr(node, meth, None)
        if callable(fn):
            try:
                return str(fn())
            except Exception:               # pragma: no cover - 渲染失败不该中断报告
                pass
    return str(node)


# --------------------------------------------------------------------------
# 2. 各章节渲染
# --------------------------------------------------------------------------
def _sec_summary(ri: ReportInput) -> List[str]:
    search = ri.search
    reports = list(ri.reports)
    facts: List[List[str]] = []

    if reports:
        best = max(reports, key=lambda r: float(_get(r, "score", 0.0) or 0.0))
        facts.append(["评价因子数", str(len(reports))])
        facts.append(["最高分因子", str(_get(best, "name", DASH))])
        facts.append(["最高分", _fmt(_get(best, "score"))])
        facts.append(["评级", str(_get(best, "grade", DASH))])
    if search is not None:
        cands = list(_get(search, "candidates", ()) or ())
        facts.append(["网格评估次数", str(_get(search, "n_evaluated", DASH))])
        facts.append(["保留候选数", str(len(cands))])
        facts.append(["停止原因", str(_get(search, "stop_reason", DASH))])
        facts.append(["搜索耗时", f"{_fmt(_get(search, 'elapsed'), 1)} s"])
    if ri.history:
        early = _get(ri.history, "ok", None)
        facts.append(["历史长度检查", "通过" if early else ("不通过" if early is False else DASH)])
    if not facts:
        return []
    return ["## 摘要", "", _kv_table(facts), ""]


def _sec_data(ri: ReportInput) -> List[str]:
    out: List[str] = []
    panel = ri.panel
    info = _get(ri.search, "panel_info", None)

    if panel is not None or info:
        rows = []
        if info:
            for k in ("panel", "n_dates", "n_symbols", "numba"):
                if k in info:
                    rows.append([k, str(info[k])])
            roles = info.get("roles") or {}
            if roles:
                rows.append(["字段角色", ", ".join(f"{k}:{v}" for k, v in sorted(roles.items()))])
        elif panel is not None:
            rows.append(["交易日数", str(_count(_get(panel, "dates", None)))])
            rows.append(["标的数", str(_count(_get(panel, "symbols", None)))])
        if rows:
            out += ["## 数据前置条件", "", _kv_table(rows), ""]

    if ri.history:
        hist = ri.history
        rows = [[str(k), _cfg_fmt(v)]
                for k, v in hist.items() if k not in ("per_field",)]
        if rows:
            out += ["### 历史长度", "", _kv_table(rows), ""]
        per = _get(hist, "per_field", None)
        if isinstance(per, Mapping) and per:
            table = _table(["字段", "可用交易日", "首日可用", "结论"],
                           [[str(f), _fmt(_get(v, "usable_days", _get(v, "n_usable"))),
                             str(_get(v, "first_usable", _get(v, "first_date", DASH))),
                             "通过" if _get(v, "ok", True) else "不足"]
                            for f, v in per.items()],
                           aligns=["l", "r", "r", "l"])
            out += [table, ""]

    if ri.pit_checks:
        rows = []
        for chk in ri.pit_checks:
            ok = _get(chk, "ok", None)
            rows.append([str(_get(chk, "field", DASH)),
                         str(_get(chk, "symbols_checked", DASH)),
                         str(_get(chk, "mismatch_points", DASH)),
                         str(_get(chk, "lookahead_points", DASH)),
                         "通过" if ok else ("不通过" if ok is False else DASH)])
        out += ["### PIT 前视自检（与朴素参考实现逐点对照）", "",
                _table(["字段", "标的数", "错位点数", "前视点数", "结论"], rows,
                       aligns=["l", "r", "r", "r", "l"]), ""]
    return out


def _sec_search(ri: ReportInput) -> List[str]:
    search = ri.search
    if search is None:
        return []
    out: List[str] = ["## 网格搜索", ""]
    hist = list(_get(search, "history", ()) or ())
    if hist:
        rows = []
        for h in hist:
            rows.append([
                str(_get(h, "layer", DASH)),
                str(_get(h, "generated", DASH)),
                str(_get(h, "screened", DASH)),
                str(_get(h, "kept", DASH)),
                str(_get(h, "pruned_invalid", 0) + _get(h, "pruned_error", 0)
                    + _get(h, "pruned_coverage", 0) + _get(h, "pruned_low_ic", 0)
                    + _get(h, "pruned_dup", 0) + _get(h, "pruned_corr", 0)),
                _fmt(_get(h, "best_score")),
                f"{_fmt(_get(h, 'seconds'), 1)}",
            ])
        out += ["### 层级轨迹", "",
                _table(["层", "生成", "入筛", "保留", "剪枝", "最优分", "耗时(s)"], rows,
                       aligns=["r"] * 7), ""]
        out += ["剪枝列合并了量纲非法、求值报错、覆盖率不足、IC 不达标、重复与"
                "相关性过高六类原因；保留数远小于生成数是网格搜索的正常形态，"
                "两级剪枝的意义正在于此。", ""]

    cands = list(_get(search, "candidates", ()) or ())
    if cands:
        rows = []
        for i, c in enumerate(cands, 1):
            rows.append([
                str(i), f"`{_node_text(_get(c, 'node'))}`",
                str(_get(c, "layer", DASH)),
                _fmt(_get(c, "score")),
                _fmt(_get(c, "ic_mean"), 4),
                _fmt(_get(c, "icir"), 2),
                _fmt(_get(c, "ic_win"), pct=True),
                _fmt(_get(c, "coverage"), pct=True),
            ])
        out += ["### 候选因子", "",
                _table(["#", "表达式", "层", "综合分", "IC", "ICIR", "胜率", "覆盖率"],
                       rows, aligns=["r", "l", "r", "r", "r", "r", "r", "r"]), ""]
    if len(out) <= 2:
        return []
    return out


def _sec_factor(rep: "FactorReport", idx: int) -> List[str]:
    name = str(_get(rep, "name", f"factor{idx}"))
    text = str(_get(rep, "expression", "") or "")
    scores = _get(rep, "scores", {}) or {}
    out: List[str] = [f"## 因子 {idx}：{name}", ""]
    if text:
        out += ["```", text, "```", ""]
    head = [["综合分", f"{_fmt(_get(rep, 'score'))}（{_get(rep, 'grade', DASH)}）"]]
    for k, label in (("predictive", "预测能力"), ("stability", "稳定性"),
                     ("quality", "数据质量"), ("correlation", "低相关/低冗余"),
                     ("base", "加权原始分"), ("penalty", "复杂度惩罚")):
        if k in scores:
            head.append([label, _fmt(scores[k])])
    out += [_kv_table(head), ""]

    metrics = _get(rep, "metrics", {}) or {}
    detail = _get(rep, "detail", {}) or {}
    quality = _get(detail, "quality", {}) or {}
    rows: List[List[str]] = []

    def add(label: str, key: str, nd: int = 3, pct: bool = False,
            src: Mapping[str, Any] = metrics) -> None:
        if key in src:
            rows.append([label, _fmt(src[key], nd=nd, pct=pct)])

    add("RankIC 均值", "rank_ic_mean", 4)
    add("RankIC t 值", "rank_ic_t", 2)
    add("RankICIR", "rank_icir", 3)
    add("RankIC 胜率", "rank_ic_win", pct=True)
    add("RankIC 年化", "rank_icr_annual", 2)
    add("Pearson IC 均值", "pearson_ic_mean", 4)
    add("分组单调性", "monotonicity", 3)
    add("多空均值", "group_ls_mean", 4)
    add("多空 t 值", "group_ls_t", 2)
    add("换手率", "turnover", pct=True)
    add("IC 半衰期(天)", "decay_halflife", 2)
    add("最差分段 IC", "worst_seg_ic", 4)
    add("滚动 IC 下界", "rolling_ic_min", 4)
    add("池内最大相关", "pool_max_corr", 3)
    add("增量 IC", "incremental_ic", 4)
    add("覆盖率", "coverage", pct=True, src=quality or metrics)
    add("预热后覆盖率", "coverage_adj", pct=True, src=quality or metrics)
    add("预热窗口(天)", "warmup", 0, src=quality or metrics)
    add("退化截面占比", "degenerate", pct=True, src=quality or metrics)
    add("有效交易日", "n_valid_days", 0, src=quality or metrics)
    if rows:
        out += ["### 关键指标", "", _kv_table(rows), ""]

    grp = _get(detail, "group_returns", None)
    if grp is not None and len(grp):
        vals = list(grp)
        heads = [f"G{i + 1}" for i in range(len(vals))]
        out += ["### 分组收益（升序，G1 = 因子值最低组）", "",
                _table(heads, [[_fmt(v, 4) for v in vals]], aligns=["r"] * len(vals)),
                ""]

    seg = _get(detail, "segment_means", None)
    if seg is not None and len(seg):
        out += ["### 分段 IC 均值", "",
                _table([f"S{i + 1}" for i in range(len(seg))],
                       [[_fmt(v, 4) for v in seg]], aligns=["r"] * len(seg)), ""]

    errs = list(_get(rep, "errors", ()) or ())
    if errs:
        out += ["### 校验告警", "", _bullet([str(e) for e in errs]), ""]
    return out


def _sec_risk(ri: ReportInput) -> List[str]:
    risk = dict(ri.risk or {})
    if not risk:
        for rep in ri.reports:
            detail = _get(rep, "detail", {}) or {}
            blk = _get(detail, "risk", None)
            if blk:
                risk[str(_get(rep, "name", "factor"))] = blk
    if not risk:
        return []
    keys = [("delta_r2", "ΔR²（风格解释力）", 4),
            ("delta_r2_adj", "调整 ΔR²", 4),
            ("r2_base", "基准 R²", 4),
            ("t_mean", "暴露平均 |t|", 3),
            ("t_gt2_ratio", "|t|>2 暴露占比", 3),
            ("vif", "VIF（与池内因子）", 3),
            ("ac_l1", "lag1 自相关", 3),
            ("crowding_score", "拥挤度", 3),
            ("turnover", "换手率", 3),
            ("exposure_hhi", "暴露集中度 HHI", 3)]
    present = [(k, lbl, nd) for k, lbl, nd in keys
               if any(k in (blk or {}) for blk in risk.values())]
    if not present:
        return []
    rows = []
    for name, blk in risk.items():
        rows.append([name] + [_fmt((blk or {}).get(k), nd=nd) for k, _, nd in present])
    return ["## 风险闸门", "",
            _table(["因子"] + [lbl for _, lbl, _ in present], rows,
                   aligns=["l"] + ["r"] * len(present)),
            "",
            "ΔR² 衡量风格暴露对因子收益的解释力，越大说明因子越像已知风格暴露；"
            "自相关与拥挤度共同刻画「拥挤度风险」——高自相关 + 高拥挤意味着"
            "一旦反转，退出成本很高。", ""]


def _sec_relation(ri: ReportInput) -> List[str]:
    out: List[str] = []
    if ri.corr is not None and getattr(ri.corr, "shape", (0, 0))[0]:
        mat = ri.corr
        cols = [str(c) for c in mat.columns]
        rows = []
        for i, idx in enumerate(mat.index):
            rows.append([str(idx)] + [_fmt(mat.iloc[i, j]) for j in range(len(cols))])
        out += ["## 因子相关性", "",
                _table(["因子"] + cols, rows, aligns=["l"] + ["r"] * len(cols)), ""]
    if ri.pools:
        rows = [[name] + [_fmt(v) for v in blk.values()]
                for name, blk in ri.pools.items()]
        heads = list(next(iter(ri.pools.values())).keys())
        out += ["## 池内相关性", "", _table(["因子"] + heads, rows,
                                           aligns=["l"] + ["r"] * len(heads)), ""]
    if ri.incremental:
        heads = list(next(iter(ri.incremental.values())).keys())
        rows = [[name] + [_fmt(v, 4) for v in blk.values()]
                for name, blk in ri.incremental.items()]
        out += ["## 增量信息（相对既有因子池）", "",
                _table(["因子"] + heads, rows, aligns=["l"] + ["r"] * len(heads)),
                "", "增量 IC 为把候选对既有因子的横截面正交化后的 RankIC 均值；"
                    "接近 0 说明该因子已被因子池解释掉。", ""]
    return out


def _sec_repro(ri: ReportInput) -> List[str]:
    rows: List[List[str]] = [["生成时刻", ri.stamp()]]
    cfg = _get(ri.search, "config", None)
    if cfg is not None:
        if dataclasses.is_dataclass(cfg):
            for f in dataclasses.fields(cfg):
                rows.append([f"`{f.name}`", _cfg_fmt(getattr(cfg, f.name))])
        elif isinstance(cfg, Mapping):
            for k, v in cfg.items():
                rows.append([f"`{k}`", _cfg_fmt(v)])
    if ri.notes:
        rows.extend([["备注", str(n)] for n in ri.notes])
    return ["## 复现信息", "", _kv_table(rows), ""]


def render_markdown(ri: ReportInput) -> str:
    """把 ``ReportInput`` 渲染成一份自包含的 Markdown 报告。"""
    parts: List[str] = [f"# {ri.title}", "", f"_生成时刻：{ri.stamp()}_", ""]
    for seg in (_sec_summary(ri), _sec_data(ri), _sec_search(ri)):
        parts += seg
    for i, rep in enumerate(ri.reports, 1):
        parts += _sec_factor(rep, i)
    for seg in (_sec_risk(ri), _sec_relation(ri), _sec_repro(ri)):
        parts += seg
    return "\n".join(parts).rstrip() + "\n"


# --------------------------------------------------------------------------
# 3. JSON 载荷
# --------------------------------------------------------------------------
def _jsonable(obj: Any) -> Any:
    """递归转成可 JSON 序列化的结构；无法识别的对象降级为字符串。"""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return x if np.isfinite(x) else None
    if isinstance(obj, pd.Series):
        return [[str(k), _jsonable(v)] for k, v in obj.items()]
    if isinstance(obj, pd.DataFrame):
        return {"index": [str(i) for i in obj.index],
                "columns": [str(c) for c in obj.columns],
                "data": [[_jsonable(v) for v in row] for row in obj.to_numpy().tolist()]}
    if dataclasses.is_dataclass(obj):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    meth = getattr(obj, "render", None)          # 表达式节点
    if callable(meth):
        try:
            return str(meth())
        except Exception:                        # pragma: no cover
            pass
    return str(obj)


def report_payload(ri: ReportInput) -> Dict[str, Any]:
    """结构化载荷：与 Markdown 同源，供程序消费（存档、对比、前端展示）。"""
    search = ri.search
    payload: Dict[str, Any] = {
        "title": ri.title,
        "generated_at": ri.stamp(),
        "n_reports": len(ri.reports),
        "reports": [
            {"name": _get(r, "name"), "expression": _get(r, "expression"),
             "score": _jsonable(_get(r, "score")), "grade": _get(r, "grade"),
             "scores": _jsonable(_get(r, "scores")),
             "metrics": _jsonable(_get(r, "metrics")),
             "group_returns": _jsonable(_get(_get(r, "detail", {}) or {}, "group_returns")),
             "segment_means": _jsonable(_get(_get(r, "detail", {}) or {}, "segment_means")),
             "ic_series": _jsonable(_get(_get(r, "detail", {}) or {}, "ic_series")),
             "errors": list(_get(r, "errors", ()) or ())}
            for r in ri.reports],
    }
    if search is not None:
        payload["search"] = {
            "n_evaluated": _get(search, "n_evaluated"),
            "elapsed": _jsonable(_get(search, "elapsed")),
            "stop_reason": _get(search, "stop_reason"),
            "panel_info": _jsonable(_get(search, "panel_info")),
            "config": _jsonable(_get(search, "config")),
            "history": _jsonable(_get(search, "history")),
            "candidates": [
                {"expression": _node_text(_get(c, "node")), "layer": _get(c, "layer"),
                 "score": _jsonable(_get(c, "score")),
                 "ic_mean": _jsonable(_get(c, "ic_mean")),
                 "icir": _jsonable(_get(c, "icir")),
                 "ic_win": _jsonable(_get(c, "ic_win")),
                 "coverage": _jsonable(_get(c, "coverage"))}
                for c in (_get(search, "candidates", ()) or ())],
        }
    for key, val in (("history", ri.history), ("pit_checks", ri.pit_checks),
                     ("risk", ri.risk), ("pools", ri.pools),
                     ("incremental", ri.incremental), ("corr", ri.corr),
                     ("notes", ri.notes)):
        if val is not None and (not hasattr(val, "__len__") or len(val)):
            payload[key] = _jsonable(val)
    return payload


def write_report(path: str, ri: ReportInput, json_path: Optional[str] = None
                 ) -> Dict[str, str]:
    """落盘：Markdown 必写，JSON 可选（默认与 Markdown 同名的 ``.json``）。"""
    md = render_markdown(ri)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(md)
    out = {"markdown": os.path.abspath(path)}
    if json_path is None:
        json_path = os.path.splitext(path)[0] + ".json"
    if json_path:
        with open(json_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report_payload(ri), fh, ensure_ascii=False, indent=2)
        out["json"] = os.path.abspath(json_path)
    return out


# --------------------------------------------------------------------------
# 4. 便捷入口
# --------------------------------------------------------------------------
def build_from_expressions(panel: "PanelData", expressions: Sequence[str],
                           with_risk: bool = False,
                           pool: Optional[Mapping[str, pd.DataFrame]] = None,
                           title: str = "因子挖掘研究报告",
                           history: Optional[Mapping[str, Any]] = None,
                           pit_checks: Sequence[Mapping[str, Any]] = (),
                           search: Optional["SearchResult"] = None,
                           notes: Sequence[str] = (),
                           generated_at: str = "",
                           ) -> ReportInput:
    """从 DSL 文本列表直接组装报告输入（解析失败/校验不通过会记进 ``errors``）。

    ``evaluator`` 延迟到函数内部导入：报告模块本身不该把 numba 之类的东西
    变成 ``import mining.report`` 的隐式依赖。
    """
    from . import evaluator as EV

    reports: List[Any] = []
    for i, text in enumerate(expressions, 1):
        try:
            reports.append(EV.evaluate_expr(str(text), panel, pool=dict(pool or {}),
                                            with_risk=with_risk))
        except Exception as exc:                     # 单条失败不影响其余因子
            reports.append(_FailedReport(name=f"factor{i}", expression=str(text),
                                         error=f"{type(exc).__name__}: {exc}"))
    return ReportInput(panel=panel, title=title, reports=reports, history=history,
                       pit_checks=pit_checks, search=search, notes=notes,
                       generated_at=generated_at)


@dataclass
class _FailedReport:
    """评价失败的占位报告：让报告里留下"这条表达式为什么不成立"的记录。"""

    name: str
    expression: str
    error: str
    score: float = float("nan")
    grade: str = "N/A"

    @property
    def errors(self) -> List[str]:
        return [self.error]

    @property
    def scores(self) -> Dict[str, float]:
        return {}

    @property
    def metrics(self) -> Dict[str, float]:
        return {}

    @property
    def detail(self) -> Dict[str, Any]:
        return {}
