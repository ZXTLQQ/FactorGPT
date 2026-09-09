"""runner — 前瞻检验编排层（run 每日循环 / settle 结算回填 / scorecard 评分卡）。

主循环（与 HA 官方推荐 agent 主循环对齐，但面向因子观点而非自由评论）：
1. 确定「宏观观点」：因子描述/主题文本/外部 view JSON -> translator 翻译；
2. 发现开放挑战：公开 GET /eval/challenges?status=open（无鉴权，dry-run 亦可）；
3. 对每个观点挑同资产、最早截止的挑战，清洗成 payload；
4. dry_run（默认）：只写入本地账本（预测锁定于本机）；
   live（显式 --live 且具备凭据）：POST /eval/challenges/{id}/predict 真实提交；
5. settle：遍历待结算记录，GET results 回填第三方结算结果；
6. scorecard：由已结算记录生成与回测独立的评分卡（准确率/Brier/校准）。

全程原则：任何网络/凭据故障都降级为本地 dry_run 影子记录，绝不抛异常中断
因子流水线；真实提交必须显式请求。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .client import HAError, HeadlineArenaClient
from .ledger import ForwardLedger, LedgerError, default_ledger_dir
from .scorecard import render_markdown, reconcile_pending
from .translator import DEFAULT_ASSETS, factor_to_theme, parse_theme, view_to_prediction

# config.yaml headline_arena 段与 runner 默认配置的映射
ENV_AGENT_ID = "HA_AGENT_ID"
ENV_CLIENT_SECRET = "HA_CLIENT_SECRET"
ENV_BASE_URL = "HA_BASE_URL"


def default_ha_config() -> Dict[str, Any]:
    return {
        "enabled": False,              # 影子节点是否挂入 LangGraph（默认关）
        "base_url": "",                # 留空取环境变量 HA_BASE_URL 或官方默认
        "dry_run": True,               # 默认只写本地账本；--live 真实提交
        "assets": list(DEFAULT_ASSETS),
        "default_theme": "",           # 每日循环默认宏观主题文本
        "ledger_dir": "",              # 留空 -> <项目根>/data/forwardtest
        "auto_subscribe": True,        # live 提交前自动订阅资产 scope
        "neutral_when_no_signal": False,  # 主题无方向信号时是否补中性占位
        "max_challenges_per_asset": 1,
    }


def has_credentials() -> bool:
    return bool(os.environ.get(ENV_AGENT_ID) and os.environ.get(ENV_CLIENT_SECRET))


def build_client(ha_config: Optional[Dict[str, Any]] = None) -> HeadlineArenaClient:
    cfg = dict(default_ha_config(), **(ha_config or {}))
    return HeadlineArenaClient(base_url=cfg.get("base_url") or None)


def resolve_ledger(ha_config: Optional[Dict[str, Any]] = None) -> ForwardLedger:
    """解析账本路径。

    ledger_dir 兼容两种形态：
    - 目录（推荐，如 "" / data/forwardtest）-> <目录>/ledger.jsonl；
    - 文件（以 .jsonl/.json 结尾）-> 直接使用该文件路径。
    """
    cfg = dict(default_ha_config(), **(ha_config or {}))
    raw = str(cfg.get("ledger_dir") or "").strip()
    if raw.lower().endswith((".jsonl", ".json")):
        return ForwardLedger(raw)
    base = Path(raw) if raw else default_ledger_dir()
    return ForwardLedger(str(base / "ledger.jsonl"))


# ------------------------------------------------------------------ 挑战发现 #
def _synthetic_challenge(asset: str) -> Dict[str, Any]:
    """离线/无网环境下的挑战占位（clearly 标记 synthetic，不冒充真实挑战）。"""
    return {
        "id": f"demo-{asset.lower()}",
        "asset": asset,
        "question": f"[离线演示占位] {asset} 结算窗口方向",
        "status": "open",
        "deadline": None,
        "open_price": None,
        "dead_zone_pct": 0.5,
        "synthetic": True,
    }


def fetch_open_challenges(
    ha_config: Optional[Dict[str, Any]] = None,
    client: Optional[HeadlineArenaClient] = None,
) -> List[dict]:
    """拉取全部开放方向性挑战（公共端点）。网络失败返回空，由调用方降级。"""
    try:
        c = client or build_client(ha_config)
        return c.open_challenges() or []
    except Exception:  # noqa: BLE001 —— 降级：离线演示不因网络失败而中断
        return []


def _pick_challenge(
    challenges: List[dict], asset: str, used_ids: set, max_per_asset: int = 1,
) -> Optional[dict]:
    """从开放挑战中挑同资产、尚未使用、最早截止的一个。"""
    cands = [c for c in challenges
             if str((c or {}).get("asset") or "").upper() == asset
             and str((c or {}).get("status") or "") == "open"
             and (c or {}).get("id") not in used_ids]
    if not cands:
        return None
    def _dl(c):
        dl = c.get("deadline")
        return (dl or "") or str(c.get("created_at") or "")
    cands.sort(key=_dl)
    return cands[:max_per_asset][0]


# ------------------------------------------------------------------ 观点来源 #
def resolve_theme_text(
    ha_config: Optional[Dict[str, Any]] = None,
    theme: Optional[str] = None,
    factor_desc: Optional[str] = None,
    factor_name: Optional[str] = None,
    factor_metrics: Optional[dict] = None,
) -> Optional[str]:
    """确定宏观主题文本。返回 None 表示"无方向信号，跳过本轮"。"""
    if theme and theme.strip():
        return theme.strip()
    if factor_desc and factor_desc.strip():
        return factor_to_theme(factor_name or "", factor_desc, factor_metrics)
    cfg = dict(default_ha_config(), **(ha_config or {}))
    t = str(cfg.get("default_theme") or "").strip()
    return t or None


def load_views_file(path: str) -> List[dict]:
    """读取外部 view JSON：
    {"views": [{"asset": "GC", "direction": "bullish", "confidence": 0.62,
                "reasoning": "..."}], "theme": "可选"}
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    views = data.get("views") if isinstance(data, dict) else data
    if not isinstance(views, list):
        raise ValueError(f"view 文件 {path} 需包含 views 列表")
    return views


# ------------------------------------------------------------------ 主流程 #
def run_forward(
    ha_config: Optional[Dict[str, Any]] = None,
    *,
    theme: Optional[str] = None,
    factor_desc: Optional[str] = None,
    factor_name: Optional[str] = None,
    factor_metrics: Optional[dict] = None,
    views: Optional[List[dict]] = None,
    assets: Optional[List[str]] = None,
    live: bool = False,
    ledger: Optional[ForwardLedger] = None,
    client: Optional[HeadlineArenaClient] = None,
    reasoning_prefix: str = "",
) -> Dict[str, Any]:
    """核心入口：观点 -> 挑战 -> 预测账本（dry_run 默认，live 需凭据）。

    返回 summary dict：
      {mode, submitted, dry_run, records: [uid...], views, warnings: [...], ledger_path}
    """
    cfg = dict(default_ha_config(), **(ha_config or {}))
    ledger_obj = ledger or resolve_ledger(cfg)
    target_assets = [a.upper() for a in (assets or cfg.get("assets") or DEFAULT_ASSETS)]

    warnings: List[str] = []

    # 1) 观点解析
    if not views:
        text = resolve_theme_text(
            cfg, theme=theme, factor_desc=factor_desc,
            factor_name=factor_name, factor_metrics=factor_metrics)
        if text:
            views = parse_theme(text, assets=target_assets)
        if not views and factor_desc and not theme:
            warnings.append("因子描述未命中宏观方向措辞，本轮不生成影子预测（无信号不制造噪声）")
    if not views:
        return {"mode": "skip", "submitted": 0, "dry_run": 0,
                "records": [], "views": [], "warnings": warnings,
                "ledger_path": str(ledger_obj.path), "reason": "no_signal"}

    # 2) 模式裁决
    creds = has_credentials()
    mode = "live" if (live and creds) else "dry_run"
    if live and not creds:
        warnings.append(
            "请求 --live 但缺少凭据（HA_AGENT_ID/HA_CLIENT_SECRET），已降级为 dry_run"
            "。请先运行 register 子命令注册 agent 并人工认领。")

    # 3) 挑战发现（公共端点；离线时用明确标记的占位挑战）
    used: set = set()
    used.update(r.get("challenge_id") for r in ledger_obj.records
                if r.get("challenge_id"))
    challenges = []
    if mode == "live" and client is None:
        try:
            client = build_client(cfg)
        except Exception:  # noqa: BLE001
            client = None
    if client is not None:
        try:
            challenges = client.open_challenges() or []
        except Exception as e:  # noqa: BLE001 —— 网络失败降级 dry_run + 占位
            warnings.append(f"拉取开放挑战失败，使用离线占位挑战: {type(e).__name__}")
            challenges = []
    if not challenges:
        challenges = [_synthetic_challenge(a) for a in target_assets]

    # 4) 逐观点提交（dry_run 只写账本；live 真实 POST）
    submitted: List[str] = []
    used_ids: set = set(used)
    max_per_asset = int(cfg.get("max_challenges_per_asset", 1))
    for view in views:
        asset = str(view.get("asset") or "").upper()
        if asset not in target_assets:
            continue
        chal = _pick_challenge(challenges, asset, used_ids, max_per_asset)
        if chal is None:
            chal = _synthetic_challenge(asset)
        if not chal.get("synthetic"):
            used_ids.add(str(chal.get("id")))
        try:
            payload = view_to_prediction(view, chal)
        except ValueError as e:
            warnings.append(str(e))
            continue
        base = reasoning_prefix.strip()
        reasoning = payload.get("reasoning", "")
        if base:
            reasoning = f"{base}｜{reasoning}".strip(" ｜")
        direction = payload["direction"]
        confidence = float(payload["confidence"])
        record: Dict[str, Any] = {
            "asset": asset,
            "direction": direction,
            "confidence": confidence,
            "reasoning": reasoning,
            "challenge_id": chal.get("id"),
            "event_id": chal.get("event_id"),
            "question": chal.get("question"),
            "deadline": chal.get("deadline"),
            "open_price": chal.get("open_price"),
            "dead_zone_pct": chal.get("dead_zone_pct"),
            "source": {"theme": theme, "factor_name": factor_name,
                       "factor_description": factor_desc,
                       "synthetic_challenge": bool(chal.get("synthetic"))},
        }
        if factor_name:
            record["factor_name"] = factor_name
        if factor_metrics:
            record["metrics_snapshot"] = {
                k: (round(float(v), 4) if isinstance(v, (int, float)) else str(v))
                for k, v in factor_metrics.items() if not str(k).startswith("_")
            }
        if mode == "live" and client is not None:
            try:
                cfg_assets = cfg.get("auto_subscribe", True)
                if cfg_assets:
                    client.subscribe_assets([asset])
                resp = client.submit_prediction(
                    chal["id"], direction, confidence, reasoning=reasoning or None)
                record.update({
                    "mode": "live",
                    "prediction_id": resp.get("prediction_id"),
                    "counts_for_score": resp.get("counts_for_score"),
                    "submit_response": {
                        k: resp.get(k) for k in ("prediction_id", "counts_for_score", "note")
                        if resp.get(k) is not None},
                })
            except HAError as e:
                warnings.append(f"{asset} 真实提交失败({e})，改记 dry_run")
                record["mode"] = "dry_run"
                record.setdefault("source", {})["live_error"] = str(e)
        else:
            record["mode"] = "dry_run"
        uid = ledger_obj.log_prediction(record)
        submitted.append(uid)

    live_count = sum(1 for u in submitted
                     if ledger_obj.find(uid=u).get("mode") == "live")
    return {
        "mode": mode, "submitted": len(submitted), "dry_run": len(submitted) - live_count,
        "live": live_count, "records": submitted, "views": views,
        "warnings": warnings, "ledger_path": str(ledger_obj.path),
    }


# ------------------------------------------------------------------ 结算回填 #
def settle_ledger(
    ha_config: Optional[Dict[str, Any]] = None,
    ledger: Optional[ForwardLedger] = None,
    client: Optional[HeadlineArenaClient] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """遍历待结算记录，拉取第三方 results 回填（只写 settled 子块）。"""
    cfg = dict(default_ha_config(), **(ha_config or {}))
    ledger_obj = ledger or resolve_ledger(cfg)
    try:
        c = client or build_client(cfg)
    except HAError as e:
        return {"settled": 0, "failed": 0, "skipped": 0, "error": str(e)}
    pending = [r for r in ledger_obj.pending() if r.get("challenge_id")]
    if limit:
        pending = pending[:limit]
    settled_n = failed_n = skipped_n = 0
    notes: List[str] = []
    for rec in pending:
        chal_id = str(rec["challenge_id"])
        try:
            res = c.challenge_results(chal_id)
        except HAError as e:
            failed_n += 1
            notes.append(f"{rec.get('uid')}: {e}")
            continue
        status = str((res or {}).get("status") or "")
        if status != "resolved":
            skipped_n += 1  # 尚未结算，下次再试
            continue
        result = str((res or {}).get("result") or "").lower()
        # 从响应里的预测明细找本 agent 条目（prediction_id 优先，其次 agent_id+direction）
        is_correct: Optional[bool] = None
        score: Optional[float] = None
        preds = res.get("predictions") if isinstance(res, dict) else None
        if isinstance(preds, list):
            mine = None
            pid = rec.get("prediction_id")
            for p in preds:
                if pid and str(p.get("prediction_id") or "") == str(pid):
                    mine = p
                    break
            if mine is None:
                mine = next((p for p in preds
                             if str(p.get("agent_id") or "") == (c.agent_id or "")
                             and str(p.get("direction") or "").lower() == str(rec.get("direction", "")).lower()),
                            None)
            if mine is not None:
                is_correct = bool(mine.get("is_correct"))
                if mine.get("score") is not None:
                    try:
                        score = float(mine["score"])
                    except (TypeError, ValueError):
                        score = None
        if is_correct is None:
            is_correct = result == str(rec.get("direction", "")).lower()
        from .scorecard import ha_directional_score
        if score is None:
            score = ha_directional_score(is_correct, float(rec.get("confidence", 0.5)))
        block = {
            "status": status,
            "result": result,
            "open_price": res.get("open_price"),
            "close_price": res.get("close_price"),
            "resolution_source": res.get("resolution_source"),
            "resolved_at": res.get("resolved_at"),
            "is_correct": bool(is_correct),
            "score": round(score, 4),
        }
        try:
            ledger_obj.settle(str(rec["uid"]), block)
            settled_n += 1
        except LedgerError as e:
            failed_n += 1
            notes.append(f"{rec.get('uid')}: {e}")
    return {"settled": settled_n, "failed": failed_n, "skipped": skipped_n,
            "notes": notes}


# ------------------------------------------------------------------ 评分卡 #
def make_scorecard(
    ha_config: Optional[Dict[str, Any]] = None,
    ledger: Optional[ForwardLedger] = None,
) -> str:
    """生成前瞻检验评分卡 Markdown。"""
    cfg = dict(default_ha_config(), **(ha_config or {}))
    ledger_obj = ledger or resolve_ledger(cfg)
    return render_markdown(ledger_obj.records, ledger_path=str(ledger_obj.path))
