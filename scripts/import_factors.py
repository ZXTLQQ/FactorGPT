"""
因子导入器（import_factors.py）

将外部因子字典（如飞书多维表格导出的 CSV / Excel，或 JSON / JSONL）批量写入
「已学习因子库」(data/learned_factors.jsonl)，使 FactorGPT Agent 能够检索学习
并复用其中的因子代码。

用法：
    python scripts/import_factors.py <文件路径> [source标签]

支持的输入格式：
    - CSV / TSV        （飞书表格导出最常见）
    - Excel (.xlsx/.xls)
    - JSON / JSONL     （每行或整体为一个因子对象列表）

列名自动识别（中英文均可），识别规则见 COLUMN_ALIASES：
    标题/名称  -> title
    类别/分类  -> category
    公式/计算逻辑 -> formula
    描述/说明  -> description
    代码/实现  -> code   （可选；含代码才能被 Agent 直接"调用"）

示例（飞书导出为 factors.csv 后）：
    python scripts/import_factors.py data/feishu_factors.csv feishu
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 将项目 src 加入模块搜索路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag.learned_library import LearnedFactorLibrary  # noqa: E402

# 目标字段 -> 可能的列名（子串匹配，大小写不敏感）
COLUMN_ALIASES = {
    "title": ["因子名", "因子名称", "名称", "name", "因子", "因子名字", "因子中文名", "中文名称", "title"],
    "category": ["类别", "分类", "类型", "category", "因子类别", "因子类型", "风格"],
    "formula": ["公式", "计算逻辑", "计算方法", "formula", "计算方式", "表达式"],
    "description": ["描述", "说明", "释义", "description", "备注", "因子说明", "含义", "介绍"],
    "code": ["代码", "实现", "code", "python", "因子代码", "实现代码", "计算方法代码"],
    # 飞书「因子字典」结构（因子名字 / 文章链接 / 作者 / 研报来源）
    "author": ["作者", "author", "因子作者", "创建人"],
    "url": ["因子计算步骤和代码", "文章链接", "计算步骤", "链接", "url", "链接地址", "参考链接", "文章地址"],
    "reference": ["研报/参考研报", "研报来源", "来源", "reference", "参考资料", "出处", "研报"],
}

# 回测评价指标列 -> 统一字段名（解析后写入因子对象的 metrics 字典，并并入检索语料）
METRIC_MAP = {
    "IC": "ic",
    "IR": "ir",
    "IC>0的概率": "ic_pos_prob",
    "Sharpe": "sharpe",
    "Turnover": "turnover",
    "Fitness": "fitness",
    "Returns": "returns",
    "DrawDown": "drawdown",
    "Method": "method",
}


def _resolve_columns(columns):
    """将文件列名映射到目标字段名。返回 {目标字段: 文件列名}。

    匹配策略：先「精确匹配」列名（大小写不敏感），再对未命中的目标做
    「子串匹配」；每个原始列只会被分配给一个目标，避免多目标争用同一列
    （如「代码」被「因子计算步骤和代码」误命中）。
    """
    col_lower = {str(c).lower(): c for c in columns}
    used = set()
    resolved = {}

    # 第一轮：精确匹配
    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            key = alias.lower()
            if key in col_lower and col_lower[key] not in used:
                resolved[target] = col_lower[key]
                used.add(col_lower[key])
                break

    # 第二轮：子串匹配（仅对尚未解析的目标，且列未被占用）
    for target, aliases in COLUMN_ALIASES.items():
        if target in resolved:
            continue
        for alias in aliases:
            key = alias.lower()
            hit = next(
                (orig for lc, orig in col_lower.items()
                 if alias in lc and orig not in used),
                None,
            )
            if hit:
                resolved[target] = hit
                used.add(hit)
                break
    return resolved


def _coerce_number(raw: str):
    """将指标文本转为 float（失败则返回 None）。"""
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


def _read_rows(path: str):
    """读取文件为「列表(dict)」，统一字段为原始列名。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv", ".txt"):
        import pandas as pd

        sep = "\t" if ext == ".tsv" else None
        engine = "python" if sep is None else "c"
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, engine=engine)
        return df.to_dict(orient="records"), list(df.columns)
    if ext in (".xlsx", ".xls"):
        import pandas as pd

        # 自动选择最合适的 sheet：优先使用环境变量指定的 sheet，
        # 否则选取「含 title 列且行数最多」的工作表（因子字典常分散在多个 sheet）。
        override = os.environ.get("FACTORGPT_IMPORT_SHEET")
        xls = pd.ExcelFile(path)
        if override and override in xls.sheet_names:
            chosen = override
        else:
            best, best_rows, best_score = None, -1, -1
            for name in xls.sheet_names:
                d = pd.read_excel(xls, sheet_name=name, dtype=str, keep_default_na=False)
                if d.empty:
                    continue
                cols = list(d.columns)
                score = sum(
                    1
                    for t in ("title", "code", "url", "author", "reference")
                    if any(a in str(c).lower() for c in cols for a in COLUMN_ALIASES[t])
                )
                if score > best_score or (score == best_score and len(d) > best_rows):
                    best, best_rows, best_score = name, len(d), score
            chosen = best or xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=chosen, dtype=str, keep_default_na=False)
        print(f"[import_factors] 已选择工作表: {chosen}（共 {len(df)} 行）")
        return df.to_dict(orient="records"), list(df.columns)
    if ext in (".json", ".jsonl"):
        rows = []
        cols = set()
        with open(path, "r", encoding="utf-8") as f:
            if ext == ".jsonl":
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            rows.append(obj)
                            cols.update(obj.keys())
            else:
                data = json.load(f)
                if isinstance(data, dict):
                    data = data.get("factors") or data.get("data") or [data]
                for obj in data:
                    if isinstance(obj, dict):
                        rows.append(obj)
                        cols.update(obj.keys())
        return rows, list(cols)
    raise ValueError(f"不支持的文件类型: {ext}（仅支持 csv/tsv/xlsx/xls/json/jsonl）")


def import_factors(path: str, source: str = "feishu", learned_path: str = None) -> int:
    rows, columns = _read_rows(path)
    resolved = _resolve_columns(columns)
    if "title" not in resolved:
        raise ValueError(
            f"未识别到「因子名称/title」列。已识别映射: {resolved}；原始列: {columns}"
        )

    factors = []
    for row in rows:
        # 兼容 dict 行（json）与 DataFrame 行（str 值）
        def get(k):
            return row.get(resolved[k], "") if k in resolved else ""

        title = str(get("title")).strip()
        if not title:
            continue
        item = {
            "title": title,
            "category": str(get("category")).strip(),
            "formula": str(get("formula")).strip(),
            "description": str(get("description")).strip(),
            "source": source,
        }
        code = str(get("code")).strip()
        if code:
            item["code"] = code
        # 飞书「因子字典」扩展字段：作者 / 文章链接 / 研报来源
        # 同时并入 description，提升检索命中率与可解释性
        author = str(get("author")).strip()
        url = str(get("url")).strip()
        reference = str(get("reference")).strip()
        extra = []
        if author:
            item["author"] = author
            extra.append(f"作者：{author}")
        if url:
            # 仅当确为链接时才作为 url；否则视为描述文本
            if url.lower().startswith(("http://", "https://")):
                item["url"] = url
                extra.append(f"链接：{url}")
            else:
                item["description"] = (item["description"] + "；" if item["description"] else "") + url
        if reference:
            item["reference"] = reference
            extra.append(f"来源：{reference}")
        # 解析回测评价指标（IC/IR/Sharpe/Fitness/Returns/DrawDown/Turnover/...）
        metrics = {}
        for col, key in METRIC_MAP.items():
            raw = str(row.get(col, "")).strip()
            if not raw:
                continue
            val = _coerce_number(raw)
            metrics[key] = val if val is not None else raw
        if metrics:
            item["metrics"] = metrics
            # 将关键指标并入 description，使 Agent 可按「高 IC / 低回撤」等检索
            mtxt = "，".join(f"{k}={metrics[k]}" for k in ("ic", "ir", "sharpe", "turnover", "fitness", "returns", "drawdown", "method") if k in metrics)
            if mtxt:
                extra.append("指标：" + mtxt)
        if extra:
            item["description"] = (item["description"] + "；" if item["description"] else "") + "；".join(extra)
        factors.append(item)

    lib = LearnedFactorLibrary(learned_path) if learned_path else LearnedFactorLibrary()
    n = lib.add_many(factors)
    print(f"导入完成：共解析 {len(factors)} 条，新增 {n} 条，"
          f"更新 {len(factors) - n} 条。学习库现有 {lib.size} 条。")
    if factors:
        print("示例（前 3 条标题）：")
        for it in factors[:3]:
            has_code = "有代码" if it.get("code") else "无代码"
            print(f"  - [{it.get('category') or '未分类'}] {it['title']}（{has_code}）")
    return n


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/import_factors.py <文件路径> [source标签]")
        sys.exit(1)
    file_path = sys.argv[1]
    src_label = sys.argv[2] if len(sys.argv) > 2 else "feishu"
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        sys.exit(1)
    import_factors(file_path, src_label, learned_path=os.environ.get("FACTORGPT_LEARNED_PATH"))
