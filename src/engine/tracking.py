"""实验追踪与因子版本管理 (src/engine/tracking.py)

让整个研究过程可审计、可复现，直接回应"为什么信这个因子"的质疑。

后端可切换：
- local（默认）：零依赖，实验记录落盘为 JSONL，自动附带 git commit 与因子代码，
  并提供按因子名的版本历史与最佳记录查询（轻量因子版本管理）；
- mlflow（可选）：若已 `pip install mlflow`，额外把指标推送到 MLflow UI 方便对比。

无论后端如何，本地 JSONL 始终落盘，保证实验结果可审计、可复现。
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional


def _jsonable(obj: Any) -> Any:
    """把 numpy / pandas 等非 JSON 原生类型转成纯 Python（与 nodes._jsonable_metrics 同义）。"""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


class ExperimentTracker:
    """实验追踪器：记录每次因子评估并维护版本历史。"""

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = {}
        if isinstance(config, dict):
            cfg = config.get("experiment_tracking", {}) or {}
        self.backend = (cfg.get("backend") or "local").lower()
        self.dir = cfg.get("dir") or os.path.join("data", "experiments")
        self.experiment = cfg.get("experiment_name") or "factorgpt"
        self.use_mlflow = self.backend == "mlflow"
        self._mlflow = None
        if self.use_mlflow:
            try:
                import mlflow  # 可选依赖
                self._mlflow = mlflow
                mlflow.set_experiment(self.experiment)
            except Exception as e:  # 优雅降级
                print(f"[tracking] mlflow 不可用，回退本地 JSONL：{e}")
                self.use_mlflow = False
        os.makedirs(self.dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.dir, "factor_runs.jsonl")

    # --- git 信息（可审计） ---
    def _git_commit(self) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode("utf-8").strip() or None
        except Exception:
            return None

    # --- 核心：记录一次因子评估 ---
    def log_factor(
        self,
        name: str,
        code: str,
        metrics: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, Any]] = None,
        status: str = "ok",
    ) -> dict:
        """记录一次因子评估。返回写入的记录字典。"""
        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "experiment": self.experiment,
            "name": name,
            "status": status,
            "git_commit": self._git_commit(),
            "params": _jsonable(params or {}),
            "metrics": _jsonable(metrics or {}),
            "tags": _jsonable(tags or {}),
            "code": code,
        }
        if self.use_mlflow:
            try:
                with self._mlflow.start_run(run_name=name) as run:
                    for k, v in (params or {}).items():
                        self._mlflow.log_param(k, _jsonable(v))
                    for k, v in _jsonable(metrics or {}).items():
                        if isinstance(v, (int, float)):
                            self._mlflow.log_metric(k, v)
                    self._mlflow.set_tags(_jsonable(tags or {}))
                    if code:
                        self._mlflow.log_text(code, "factor.py")
            except Exception as e:
                print(f"[tracking] mlflow 记录失败（本地 JSONL 仍保留）：{e}")
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return record

    # --- 因子版本管理：历史与最佳 ---
    def history(self, name: Optional[str] = None) -> List[dict]:
        if not os.path.exists(self.jsonl_path):
            return []
        out: List[dict] = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if name is None or rec.get("name") == name:
                    out.append(rec)
        return out

    def best(self, name: Optional[str] = None, metric: str = "icir") -> Optional[dict]:
        recs = [r for r in self.history(name) if r.get("status") == "ok"]
        valid = [r for r in recs if isinstance(r.get("metrics", {}).get(metric), (int, float))]
        if not valid:
            return None
        return max(valid, key=lambda r: r["metrics"][metric])

    def summary(self) -> str:
        recs = self.history()
        if not recs:
            return "（暂无实验记录）"
        by_name: Dict[str, list] = {}
        for r in recs:
            by_name.setdefault(r.get("name", "?"), []).append(r)
        lines = [f"实验记录共 {len(recs)} 条，涉及 {len(by_name)} 个因子："]
        for n, lst in by_name.items():
            b = self.best(n)
            m = b["metrics"] if b else {}
            lines.append(
                f"- {n}: {len(lst)} 次 | 最佳 IC={m.get('ic')} ICIR={m.get('icir')} "
                f"LS_Sharpe={m.get('long_short_sharpe')}"
            )
        return "\n".join(lines)
