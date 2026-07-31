"""
已学习因子库（LearnedFactorLibrary）

持久化存储「外部导入（如飞书因子字典）+ Agent 自学习」得到的因子知识，
使其可被 RAG 检索（学习）并作为代码模板复用（调用）。

存储格式：JSONL，每行一个因子对象，字段兼容内置 SEED_FACTORS：
  {
    "title":      str,   因子名称
    "category":   str,   因子类别
    "formula":    str,   计算逻辑/公式
    "description":str,   说明
    "code":       str,   可选，可复用的因子代码（alpha_factor 实现）
    "source":     str,   "feishu" | "self_learned" | "external"
    "metrics":    dict,  可选，回测指标（自学习时记录）
  }
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

DEFAULT_LEARNED_PATH = "data/learned_factors.jsonl"


class LearnedFactorLibrary:
    """已学习因子库：基于 JSONL 的轻量持久化存储（无第三方依赖）。"""

    def __init__(self, path: str = DEFAULT_LEARNED_PATH) -> None:
        self.path = path
        self._ensure_dir()
        self._factors: List[Dict] = self._load()

    # ------------------------------------------------------------------
    def _ensure_dir(self) -> None:
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)

    def _load(self) -> List[Dict]:
        if not os.path.exists(self.path):
            return []
        out: List[Dict] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    # 跳过损坏行，避免整库不可用
                    continue
        return out

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for item in self._factors:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------
    def all(self) -> List[Dict]:
        return list(self._factors)

    @property
    def size(self) -> int:
        return len(self._factors)

    def get(self, title: str) -> Optional[Dict]:
        for f in self._factors:
            if f.get("title") == title:
                return f
        return None

    def add(self, factor: Dict) -> bool:
        """新增一条因子；按 title 去重，已存在则合并更新（返回 False 表示更新）。"""
        title = factor.get("title") or factor.get("name")
        if not title:
            return False
        factor = dict(factor)
        factor["title"] = title
        factor.setdefault("source", "external")
        for i, existing in enumerate(self._factors):
            if existing.get("title") == title:
                merged = {**existing, **factor}
                self._factors[i] = merged
                self._save()
                return False
        self._factors.append(factor)
        self._save()
        return True

    def add_many(self, factors: List[Dict]) -> int:
        n = 0
        for f in factors:
            if self.add(f):
                n += 1
        return n

    def by_source(self, source: str) -> List[Dict]:
        return [f for f in self._factors if f.get("source") == source]
