"""知识库自动更新：从 arXiv 等公开来源拉取最新因子研究，并监测已学习因子的 IC 衰减。

设计目标：
  - arXiv 抓取：使用标准库 urllib 调用 arXiv 公开 API，无需额外依赖、无需鉴权；
  - 微信/研报：当前为占位接口（微信公众号需要外部搜索 API 或订阅凭证，
    研报需要券商/数据商接口），在 fetch_wechat / fetch_research_report 中显式
    抛出 NotImplementedError 并说明接入点，避免「假数据」污染知识库；
  - 因子衰减监测：FactorDecayMonitor 记录每个已学习因子随时间的 IC，
    用最小二乘斜率估计衰减速度，输出可读性摘要，支撑「知识时效性闭环」。

注意：本模块不依赖 torch / streamlit 等重依赖，可独立运行与测试。
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _arxiv_api_url(query: str, max_results: int, sort_by: str) -> str:
    q = urllib.parse.quote(query)
    return (
        "http://export.arxiv.org/api/query"
        f"?search_query=all:{q}&start=0&max_results={int(max_results)}"
        f"&sortBy={sort_by}&sortOrder=descending"
    )


def fetch_arxiv(
    query: str = "factor investing quantitative factor stock selection alpha",
    max_results: int = 20,
    sort_by: str = "submittedDate",
) -> List[Dict]:
    """调用 arXiv API 拉取因子研究论文（无需鉴权）。

    返回 list[dict]，每项含 title / summary / authors / published / updated / url / arxiv_id。
    网络异常时返回空 list（仅告警，不中断主流程）。
    """
    url = _arxiv_api_url(query, max_results, sort_by)
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = resp.read().decode("utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"[auto_update] arXiv 抓取失败：{e}")
        return []

    ns = {"a": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        print(f"[auto_update] arXiv XML 解析失败：{e}")
        return []

    papers: List[Dict] = []
    for entry in root.findall("a:entry", ns):
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        published = entry.findtext("a:published", default="", namespaces=ns) or ""
        updated = entry.findtext("a:updated", default="", namespaces=ns) or ""
        authors = [a.findtext("a:name", default="", namespaces=ns) or "" for a in entry.findall("a:author", ns)]
        id_url = entry.findtext("a:id", default="", namespaces=ns) or ""
        arxiv_id = id_url.rsplit("/", 1)[-1] if id_url else ""
        papers.append({
            "title": title,
            "summary": summary,
            "authors": [a for a in authors if a],
            "published": published[:10],
            "updated": updated[:10],
            "url": id_url,
            "arxiv_id": arxiv_id,
        })
    return papers


def fetch_wechat(query: str) -> List[Dict]:
    """微信公众号抓取（占位）。

    微信公众号文章需要外部搜索 API / 订阅凭证（如微信公众平台接口或第三方检索服务）。
    请接入对应服务后在此实现；当前显式返回占位异常，避免误用假数据。
    """
    raise NotImplementedError(
        "微信公众号抓取需要外部搜索 API / 订阅凭证（例如 wechat-article-search 或自建爬虫），"
        "请接入后在 fetch_wechat 中实现；当前为占位接口。"
    )


def fetch_research_report(query: str) -> List[Dict]:
    """券商/第三方研报抓取（占位）。需要数据商接口凭证，当前显式占位。"""
    raise NotImplementedError(
        "研报抓取需要券商/数据商接口凭证（如 Wind / 聚源 / 通联），请接入后在 fetch_research_report 中实现。"
    )


def format_entries(papers: List[Dict]) -> List[str]:
    """将论文列表格式化为可入库的 markdown 知识文本。"""
    entries = []
    for p in papers:
        authors = "、".join(p.get("authors", [])[:5])
        text = (
            f"# {p['title']}\n\n"
            f"> 来源：arXiv（{p.get('arxiv_id', '')}）｜发布：{p.get('published', '')}｜"
            f"作者：{authors}\n\n"
            f"{p.get('summary', '')}\n\n"
            f"链接：{p.get('url', '')}\n"
        )
        entries.append(text)
    return entries


def save_knowledge(entries: List[str], path: str = "data/auto_knowledge.jsonl") -> int:
    """追加写入知识库（JSON Lines）。返回写入条数。"""
    full = path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    n = 0
    with open(full, "a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps({"text": e, "source": "arxiv"}, ensure_ascii=False) + "\n")
            n += 1
    print(f"[auto_update] 已写入 {n} 条知识到 {full}")
    return n


class FactorDecayMonitor:
    """监测已学习因子的 IC 随时间的衰减，识别「时效性过期」的因子。"""

    def __init__(self, path: str = "data/factor_decay.json"):
        self.path = path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)
        self.records: List[Dict] = []
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.records = json.load(f).get("records", [])
            except Exception:  # noqa: BLE001
                self.records = []

    def record(self, factor_name: str, date: str, ic: float) -> None:
        self.records.append({"factor_name": factor_name, "date": str(date), "ic": float(ic)})
        self._save()

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"records": self.records}, f, ensure_ascii=False, indent=2)

    def trend(self, factor_name: Optional[str] = None) -> Dict:
        recs = self.records
        if factor_name:
            recs = [r for r in recs if r["factor_name"] == factor_name]
        if len(recs) < 3:
            return {"slope": None, "n": len(recs), "mean_ic": None}
        recs = sorted(recs, key=lambda r: r["date"])
        x = np.arange(len(recs), dtype=float)
        y = np.array([r["ic"] for r in recs], dtype=float)
        slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 1 else 0.0
        return {"slope": slope, "n": len(recs), "mean_ic": float(np.nanmean(y))}

    def summary(self) -> str:
        names = sorted({r["factor_name"] for r in self.records})
        if not names:
            return "（暂无因子 IC 衰减记录）"
        lines = ["**因子 IC 衰减监测**"]
        for nm in names:
            t = self.trend(nm)
            slope = t["slope"]
            if slope is None:
                lines.append(f"- {nm}：样本不足（{t['n']} 条）")
            else:
                verdict = "衰减显著" if slope < -1e-4 else ("增强" if slope > 1e-4 else "稳定")
                lines.append(f"- {nm}：斜率 {slope:+.5f}/期，均值 IC {t['mean_ic']:.4f}（{verdict}）")
        return "\n".join(lines)
