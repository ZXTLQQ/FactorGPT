"""ledger — 前瞻检验账本（Forward Ledger）。

一句话原则：**预测在结果存在之前锁定**。每条记录在提交（或 dry-run 影子模拟）
的瞬间写入本地 JSONL，只允许：
1. 追加新预测记录（locked_prediction）；
2. 结算完成后回填 settled 子块（由第三方 challenge_results 驱动）。

结算回填之后，预测字段（direction/confidence/reasoning/challenge_id/deadline）
视为不可变——这保证了审计时可证明"当时锁定的观点"与"事后回填的结果"先后有序，
构成与回测相互独立、不可事后修改证据链。

账本文件：<ledger_dir>/ledger.jsonl（默认 <项目根>/data/forwardtest/ledger.jsonl）。
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

SCHEMA = 1
# 预测字段一旦结算即锁定，不允许修改
LOCKED_KEYS = ("challenge_id", "asset", "direction", "confidence", "reasoning",
               "deadline", "open_price", "created_at", "mode")

VALID_MODES = ("dry_run", "live", "paper")


class LedgerError(RuntimeError):
    """账本一致性错误（如对已结算记录改写预测字段）。"""


def default_ledger_dir(project_root: Optional[str] = None) -> Path:
    """默认账本目录：<项目根>/data/forwardtest（向上找到含 config.yaml 的目录）。"""
    if project_root:
        return Path(project_root) / "data" / "forwardtest"
    p = Path.cwd()
    for ancestor in (p, *p.parents):
        if (ancestor / "config.yaml").exists():
            return ancestor / "data" / "forwardtest"
    return p / "data" / "forwardtest"


def make_uid(asset: str, ts: Optional[float] = None) -> str:
    """形如 ft-20260909-GC-3f2a9c1e（前缀可读、后缀唯一）。"""
    stamp = time.strftime("%Y%m%d", time.gmtime(ts or time.time()))
    return f"ft-{stamp}-{str(asset).upper()[:8] or 'NA'}-{uuid.uuid4().hex[:8]}"


class ForwardLedger:
    """追加型 JSONL 前瞻检验账本。"""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path) if path else default_ledger_dir() / "ledger.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: List[Dict[str, Any]] = self._load()

    # ------------------------------------------------------------------ IO #
    def _load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        out: List[Dict[str, Any]] = []
        try:
            for line in self.path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
        except OSError:
            return []
        return out

    def _flush(self) -> None:
        tmp = self.path.with_suffix(".jsonl.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                for rec in self._records:
                    f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            os.replace(tmp, self.path)
        except OSError:
            # 只读文件系统等极端情况：尽力写失败不吞掉用户可见错误
            raise

    # ------------------------------------------------------------------ 读 #
    @property
    def records(self) -> List[Dict[str, Any]]:
        return list(self._records)

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        return iter(self._records)

    def find(self, uid: Optional[str] = None, challenge_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        for rec in self._records:
            if uid and rec.get("uid") == uid:
                return rec
            if challenge_id and rec.get("challenge_id") == challenge_id:
                return rec
        return None

    def pending(self, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        """未结算记录（mode 过滤可选）。"""
        return [r for r in self._records
                if not r.get("settled") and (mode is None or r.get("mode") == mode)]

    def settled(self, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        return [r for r in self._records
                if r.get("settled") and (mode is None or r.get("mode") == mode)]

    def by_mode(self, mode: str) -> List[Dict[str, Any]]:
        return [r for r in self._records if r.get("mode") == mode]

    # ------------------------------------------------------------------ 写 #
    def log_prediction(self, record: Dict[str, Any]) -> str:
        """追加一条锁定预测。

        必填：asset / direction / confidence。mode 默认 dry_run。
        返回生成的 uid。方向与置信度立即清洗，杜绝脏数据进入账本。
        """
        from .translator import clamp_confidence, clean_direction

        asset = str(record.get("asset") or "").upper()
        if not asset:
            raise LedgerError("log_prediction 需要 asset（如 GC/ES/CL/ZN）")
        mode = str(record.get("mode") or "dry_run")
        if mode not in VALID_MODES:
            raise LedgerError(f"未知 mode={mode}（可选 {VALID_MODES}）")
        uid = str(record.get("uid") or make_uid(asset))
        if self.find(uid=uid):
            raise LedgerError(f"uid 重复: {uid}")
        entry: Dict[str, Any] = {
            "schema": SCHEMA,
            "uid": uid,
            "mode": mode,
            "created_at": record.get("created_at")
            or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "asset": asset,
            "direction": clean_direction(str(record.get("direction", "neutral"))),
            "confidence": clamp_confidence(record.get("confidence", 0.5)),
            "reasoning": str(record.get("reasoning") or ""),
        }
        for key in ("challenge_id", "event_id", "question", "deadline", "open_price",
                    "dead_zone_pct", "prediction_id", "counts_for_score", "prompt_hash",
                    "source", "factor_name", "factor_description", "metrics_snapshot"):
            if key in record and record[key] is not None:
                entry[key] = record[key]
        # 方向性三分类概率向量（置信度给主方向，其余均分），供 Brier/校准用
        probs = {d: (1.0 - float(entry["confidence"])) / 2.0
                 for d in ("bullish", "bearish", "neutral")}
        probs[entry["direction"]] = float(entry["confidence"])
        entry["probabilities"] = probs
        self._records.append(entry)
        self._flush()
        return uid

    def settle(self, uid: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """结算回填：只写 settled 子块，禁止改写预测字段。

        result 建议字段：status/result/open_price/close_price/resolved_at/is_correct/score。
        """
        rec = self.find(uid=uid)
        if rec is None:
            raise LedgerError(f"未找到记录 uid={uid}")
        if rec.get("settled"):
            raise LedgerError(f"记录 {uid} 已结算，禁止重复回填")
        if result.get("status") and result["status"] != "resolved":
            raise LedgerError(f"挑战未结算（status={result.get('status')}），不能回填")
        rec["settled"] = {k: v for k, v in result.items() if v is not None}
        rec["settled_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._flush()
        return rec

    # ------------------------------------------------------------------ 统计 #
    def stats(self) -> Dict[str, Any]:
        modes = {m: len(self.by_mode(m)) for m in VALID_MODES}
        return {
            "path": str(self.path),
            "total": len(self._records),
            "pending": len(self.pending()),
            "settled": len(self.settled()),
            "by_mode": modes,
            "assets": sorted({r.get("asset") for r in self._records if r.get("asset")}),
        }
