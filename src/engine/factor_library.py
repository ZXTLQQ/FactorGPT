"""
因子库管理器 (src/engine/factor_library.py)

统一管理所有因子：传统预置因子 + LLM生成因子 + 遗传规划挖掘因子 + 用户自定义因子。
提供 CRUD / 分类检索 / 质量评分 / 批量导出 / 因子簇生成 / 统计分析等功能。

因子全生命周期：
  seed_factors（预置 + 用户定义的因子模板）
    -> cluster_expand（依据参数簇/事件簇衍生变体）
    -> genetic_evolve（遗传规划交叉/变异生成新一代）
    -> quality_filter（IC/Sharpe 筛选）
    -> ensemble_fuse（高相关因子融合去重）
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .traditional_factors import (
    FactorDef,
    get_all_factors,
    get_factors_by_category,
    get_factor_by_name,
    search_factors,
    get_factor_stats,
    export_all_to_dict,
    ALL_CATEGORIES,
    CATEGORY_LABELS,
    CATEGORY_PRICE_TREND,
    CATEGORY_VOLATILITY,
    CATEGORY_TRADING_DIFFICULTY,
    CATEGORY_PRICE_VOLUME_DIVERGENCE,
    CATEGORY_VOLUME_PRICE_FORMULA,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# 因子库管理器
# ---------------------------------------------------------------------------

class FactorLibrary:
    """因子库管理器。

    维护三个层级的因子来源：
    1. static — 预置传统因子（traditional_factors.py）
    2. generated — LLM / GP 自动挖掘的因子
    3. user — 用户上传/定义的因子

    搜索优先级：user > generated > static
    """

    def __init__(self, persist_dir: Optional[str] = None) -> None:
        self._static: Dict[str, FactorDef] = {}
        self._generated: Dict[str, FactorDef] = {}
        self._user: Dict[str, FactorDef] = {}
        self._load_static()
        self._persist_dir = persist_dir
        if persist_dir:
            self._load_persisted()

    # ---- 静态因子加载 ----
    def _load_static(self) -> None:
        for f in get_all_factors():
            self._static[f.name] = f

    # ---- 持久化 ----
    def _persist_path(self) -> Path:
        return Path(self._persist_dir or ".") / "factor_library.json"

    def _load_persisted(self) -> None:
        path = self._persist_path()
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for item in data.get("generated", []):
                fd = FactorDef(**item)
                self._generated[fd.name] = fd
            for item in data.get("user", []):
                fd = FactorDef(**item)
                self._user[fd.name] = fd
        except Exception:
            pass

    def _save_persisted(self) -> None:
        if not self._persist_dir:
            return
        path = self._persist_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "generated": [f.to_dict() for f in self._generated.values()],
            "user": [f.to_dict() for f in self._user.values()],
            "saved_at": datetime.now().isoformat(),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    # ===================================================================
    # CRUD
    # ===================================================================
    def add_factor(self, fd: FactorDef, source: str = "generated") -> bool:
        """添加因子。同名拒绝覆盖（提示 rename），返回是否成功。"""
        existing = self.get_factor(fd.name)
        if existing is not None:
            return False
        if source == "user":
            self._user[fd.name] = fd
        else:
            self._generated[fd.name] = fd
        self._save_persisted()
        return True

    def remove_factor(self, name: str) -> bool:
        """删除因子（仅限 generated / user，不可删除静态因子）。"""
        if name in self._generated:
            del self._generated[name]
            self._save_persisted()
            return True
        if name in self._user:
            del self._user[name]
            self._save_persisted()
            return True
        return False

    def update_factor(self, name: str, updates: Dict[str, Any]) -> bool:
        """更新因子元信息（仅限 generated / user）。"""
        target: Optional[Dict[str, FactorDef]] = None
        if name in self._generated:
            target = self._generated
        elif name in self._user:
            target = self._user
        else:
            return False
        fd = target[name]
        for k, v in updates.items():
            if hasattr(fd, k):
                setattr(fd, k, v)
        self._save_persisted()
        return True

    def get_factor(self, name: str) -> Optional[FactorDef]:
        """三源查找。"""
        return self._user.get(name) or self._generated.get(name) or self._static.get(name)

    # ===================================================================
    # 查询 / 检索
    # ===================================================================
    def list_all(self, sources: Optional[List[str]] = None) -> List[FactorDef]:
        """列出所有因子。sources 过滤: ['static', 'generated', 'user']。"""
        result: List[FactorDef] = []
        if sources is None:
            sources = ["user", "generated", "static"]
        if "static" in sources:
            result.extend(self._static.values())
        if "generated" in sources:
            result.extend(self._generated.values())
        if "user" in sources:
            result.extend(self._user.values())
        return result

    def list_by_category(self, category: str) -> List[FactorDef]:
        """按大类列出。"""
        return [f for f in self.list_all() if f.category == category]

    def search(
        self,
        query: str = "",
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_quality: float = 0.0,
        direction: Optional[str] = None,
        sources: Optional[List[str]] = None,
    ) -> List[FactorDef]:
        """多条件检索因子。"""
        pool = self.list_by_category(category) if category else self.list_all(sources=sources)
        results: List[FactorDef] = []
        q = query.lower()
        for f in pool:
            if q and q not in f.name.lower() and q not in f.display_name and q not in f.description.lower():
                continue
            if tags and not all(t in f.tags for t in tags):
                continue
            if f.quality_score < min_quality:
                continue
            if direction and f.direction != direction:
                continue
            results.append(f)
        return results

    def search_by_keywords(self, keywords: List[str]) -> List[FactorDef]:
        """OR 语义关键词搜索。"""
        pool = self.list_all()
        results: List[FactorDef] = []
        for f in pool:
            text = f"{f.name} {f.display_name} {f.description} {' '.join(f.tags)}".lower()
            if any(kw.lower() in text for kw in keywords):
                results.append(f)
        return results

    # ===================================================================
    # 因子簇扩增（参数/事件簇）
    # ===================================================================
    def expand_by_param_grid(
        self,
        base_name: str,
        param_name: str,
        values: List[int],
    ) -> List[FactorDef]:
        """将单个因子的参数（如窗口期）展开为一个因子簇。

        Example:
            library.expand_by_param_grid('momentum_20d', '{w}', [5,10,20,30,60])
            -> 生成 momentum_5d ~ momentum_60d 共5个变体因子。
        """
        base = self.get_factor(base_name)
        if base is None:
            return []

        expanded: List[FactorDef] = []
        for v in values:
            new_name = base_name.replace(param_name, str(v)) if param_name in base_name else f"{base_name}_{v}d"
            new_code = base.code.replace(param_name, str(v))
            new_display = base.display_name.replace(param_name, str(v)) if param_name in base.display_name else f"{base.display_name}({v})"
            new_desc = base.description.replace(param_name, str(v)) if param_name in base.description else f"{base.description} 窗口={v}"
            new_tags = base.tags + [f"window_{v}", "auto_expanded"]

            fd = FactorDef(
                name=new_name,
                display_name=new_display,
                category=base.category,
                description=new_desc,
                direction=base.direction,
                code=new_code,
                tags=new_tags,
                params={**base.params, "window": v},
                source="param_expansion",
                quality_score=base.quality_score * 0.9,  # 衍生因子略降质量预期
            )
            expanded.append(fd)
            self._generated[fd.name] = fd

        self._save_persisted()
        return expanded

    def expand_by_event_transform(
        self,
        base_name: str,
        transforms: List[Tuple[str, Callable[[str], str]]],
    ) -> List[FactorDef]:
        """通过事件变换（如差分、标准化、中性化）生成因子变体。

        Args:
            transforms: [(tag_suffix, transform_func), ...]
                e.g. [('diff', fn_differenced), ('neu', fn_neutralized)]
        """
        base = self.get_factor(base_name)
        if base is None:
            return []

        expanded: List[FactorDef] = []
        for tag_suffix, transform_fn in transforms:
            new_name = f"{base_name}_{tag_suffix}"
            new_code = transform_fn(base.code)
            new_desc = f"{base.description} [{tag_suffix}变换]"
            new_tags = base.tags + [tag_suffix, "event_transform"]

            fd = FactorDef(
                name=new_name,
                display_name=f"{base.display_name}({tag_suffix})",
                category=base.category,
                description=new_desc,
                direction=base.direction,
                code=new_code,
                tags=new_tags,
                source="event_expansion",
                quality_score=base.quality_score * 0.85,
            )
            expanded.append(fd)
            self._generated[fd.name] = fd

        self._save_persisted()
        return expanded

    def cluster_expand_all(
        self,
        windows: Optional[List[int]] = None,
        categories: Optional[List[str]] = None,
    ) -> List[FactorDef]:
        """批量参数扩增：对所有动量/波动类因子用默认窗口簇展开。

        这是「批量生产海量因子」的核心接口之一。
        默认窗口：3,5,10,20,30,60,120
        """
        if windows is None:
            windows = [3, 5, 10, 20, 30, 60, 120]

        cats = categories or [CATEGORY_PRICE_TREND, CATEGORY_VOLATILITY]
        all_expanded: List[FactorDef] = []
        for cat in cats:
            for name in [f.name for f in self.list_by_category(cat)]:
                # 匹配含数字窗口的因子名，如 momentum_20d, vol_20d
                for orig_w in [5, 10, 14, 20, 30, 60]:
                    marker = f"_{orig_w}"
                    if marker not in name:
                        continue
                    base_tag = name.split(marker)[0]
                    for w in windows:
                        new_name = f"{base_tag}_{w}d"
                        if new_name in self._generated or new_name in self._static:
                            continue
                        base = self.get_factor(name)
                        if base is None:
                            continue
                        new_code = base.code.replace(str(orig_w), str(w))
                        fd = FactorDef(
                            name=new_name,
                            display_name=base.display_name.replace(str(orig_w), str(w)),
                            category=cat,
                            description=base.description.replace(str(orig_w), str(w)),
                            direction=base.direction,
                            code=new_code,
                            tags=base.tags + ["cluster_expanded"],
                            source="cluster_expansion",
                            quality_score=base.quality_score * 0.88,
                        )
                        self._generated[fd.name] = fd
                        all_expanded.append(fd)
        self._save_persisted()
        return all_expanded

    # ===================================================================
    # 因子批量评估与排序
    # ===================================================================
    def quality_rank(
        self,
        factors: Optional[List[FactorDef]] = None,
        top_k: int = 50,
    ) -> List[FactorDef]:
        """按质量分排序（可用于 GP 筛选后的初筛）。"""
        pool = factors or self.list_all()
        ranked = sorted(pool, key=lambda f: f.quality_score, reverse=True)
        return ranked[:top_k]

    def diversity_sample(
        self,
        factors: Optional[List[FactorDef]] = None,
        n_per_category: int = 5,
    ) -> List[FactorDef]:
        """按大类均衡抽样，保证覆盖面。"""
        pool = factors or self.list_all()
        by_cat: Dict[str, List[FactorDef]] = {}
        for f in pool:
            by_cat.setdefault(f.category, []).append(f)
        sampled: List[FactorDef] = []
        for cat_factors in by_cat.values():
            ranked = sorted(cat_factors, key=lambda f: f.quality_score, reverse=True)
            sampled.extend(ranked[:n_per_category])
        return sampled

    # ===================================================================
    # 融合去重
    # ===================================================================
    def deduplicate_by_code(
        self, factors: List[FactorDef]
    ) -> List[FactorDef]:
        """基于代码哈希去重（相同代码的去重）。"""
        seen: Set[str] = set()
        unique: List[FactorDef] = []
        for f in factors:
            h = hashlib.md5(f.code.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(f)
        return unique

    def deduplicate_by_name(
        self, factors: List[FactorDef]
    ) -> List[FactorDef]:
        """基于名称去重。"""
        seen: Set[str] = set()
        unique: List[FactorDef] = []
        for f in factors:
            if f.name not in seen:
                seen.add(f.name)
                unique.append(f)
        return unique

    # ===================================================================
    # 导入/导出
    # ===================================================================
    def import_from_dict_list(self, items: List[Dict[str, Any]], source: str = "generated") -> int:
        """从字典列表批量导入因子。返回成功导入数量。"""
        count = 0
        for item in items:
            try:
                fd = FactorDef(**item)
                if self.add_factor(fd, source=source):
                    count += 1
            except Exception:
                continue
        return count

    def export_category_report(self, category: str) -> Dict[str, Any]:
        """导出某大类的完整报告。"""
        factors = self.list_by_category(category)
        return {
            "category": CATEGORY_LABELS.get(category, category),
            "category_key": category,
            "count": len(factors),
            "factors": [f.to_dict() for f in factors],
        }

    def export_full_report(self) -> Dict[str, Any]:
        """全量因子库报告。"""
        all_f = self.list_all()
        static_count = len(self._static)
        generated_count = len(self._generated)
        user_count = len(self._user)
        return {
            "total": len(all_f),
            "by_source": {"static": static_count, "generated": generated_count, "user": user_count},
            "by_category": {
                cat: len(self.list_by_category(cat)) for cat in ALL_CATEGORIES
            },
            "stats": self.statistics(),
            "factors": [f.to_dict() for f in all_f],
        }

    # ===================================================================
    # 统计
    # ===================================================================
    def statistics(self) -> Dict[str, Any]:
        """因子库全局统计。"""
        all_f = self.list_all()
        if not all_f:
            return {"total": 0}
        dirs: Dict[str, int] = {"positive": 0, "negative": 0, "none": 0}
        tags_counter: Dict[str, int] = {}
        quality_scores = []
        for f in all_f:
            dirs[f.direction] = dirs.get(f.direction, 0) + 1
            for t in f.tags:
                tags_counter[t] = tags_counter.get(t, 0) + 1
            quality_scores.append(f.quality_score)
        tags_rank = sorted(tags_counter.items(), key=lambda x: x[1], reverse=True)[:20]
        return {
            "total": len(all_f),
            "by_source": {
                "static": len(self._static),
                "generated": len(self._generated),
                "user": len(self._user),
            },
            "by_category": {
                CATEGORY_LABELS.get(k, k): len(self.list_by_category(k)) for k in ALL_CATEGORIES
            },
            "by_direction": dirs,
            "top_tags": dict(tags_rank),
            "quality": {
                "mean": round(float(np.mean(quality_scores)), 4),
                "max": round(float(np.max(quality_scores)), 4),
                "min": round(float(np.min(quality_scores)), 4),
            },
        }

    # ===================================================================
    # 因子簇定义（事件簇 / 概念簇）
    # ===================================================================
    def define_event_cluster(
        self,
        cluster_name: str,
        factor_names: List[str],
        event_description: str,
    ) -> Dict[str, Any]:
        """定义一个事件簇（如「财报发布季」「政策宽松窗口」），将相关因子打包。"""
        factors = [self.get_factor(n) for n in factor_names if self.get_factor(n)]
        return {
            "cluster_name": cluster_name,
            "event_description": event_description,
            "factor_count": len(factors),
            "factors": [f.to_dict() for f in factors],
            "created_at": datetime.now().isoformat(),
        }

    def define_concept_cluster(
        self,
        concept: str,
        query: str,
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """按概念（如「低波动」「动量」「反转」）自动聚类因子。"""
        factors = self.search(query=query)[:top_k]
        return {
            "concept": concept,
            "query": query,
            "factor_count": len(factors),
            "factors": [f.to_dict() for f in factors],
            "categories_present": list(set(f.category for f in factors)),
        }


# ===================================================================
# 便捷工厂函数
# ===================================================================

def create_default_library(persist_dir: Optional[str] = None) -> FactorLibrary:
    """创建预配置的默认因子库（含静态因子自动加载）。"""
    return FactorLibrary(persist_dir=persist_dir)


def mass_produce_factors(
    library: FactorLibrary,
    windows: Optional[List[int]] = None,
    categories: Optional[List[str]] = None,
    quality_min: float = 0.3,
) -> Tuple[List[FactorDef], Dict[str, Any]]:
    """批量生产因子——组合参数扩增 + 质量筛选。

    这是「海量因子生产」的一键接口。

    Returns:
        (通过的因子列表, 生产统计)
    """
    expanded = library.cluster_expand_all(windows=windows, categories=categories)
    ranked = library.quality_rank(min_quality=quality_min)
    unique = library.deduplicate_by_code(ranked)
    stats = {
        "expanded_count": len(expanded),
        "after_quality_filter": len(ranked),
        "after_dedup": len(unique),
        "total_in_library": library.statistics()["total"],
    }
    return unique, stats
