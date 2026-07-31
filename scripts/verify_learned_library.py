"""
验证「已学习因子库」的查询（查阅）与学习（调用）闭环。

用途：确认导入外部因子字典 / Agent 自学习得到的因子，能被 FactorGPT 的 RAG
检索器自适应查阅，且含代码的因子能被 Agent 在生成阶段直接复用（调用）。

- 查阅测试：用真实学习库（data/learned_factors.jsonl）检索「因子评价方法论」，
  确认导入的知识被 Agent 自动检索到。
- 学习/调用测试：用临时学习库模拟 Agent 自学习写入一个「含代码」因子，重建检索器后
  检索该因子并确认 retrieve_template 能返回可复用的代码（验证学习闭环与持久化）。

用法：
    python scripts/verify_learned_library.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag.learned_library import LearnedFactorLibrary
from rag.retriever import FactorRetriever

REAL_LEARNED = str(ROOT / "data" / "learned_factors.jsonl")


def test_query():
    """查阅测试：真实学习库中的知识应被检索到。"""
    print("=== 测试一：自适应查阅（查询导入的因子知识）===")
    retr = FactorRetriever(learned=LearnedFactorLibrary(REAL_LEARNED), top_k=3)
    q = "因子评价指标 IC IR 夏普 Sharpe Turnover Fitness 回测"
    docs = retr.retrieve(q, top_k=3)
    hit = any("因子评价方法论" in d for d in docs)
    for i, d in enumerate(docs, 1):
        print(f"  [{i}] {d.splitlines()[0]}")
    print(f"  命中导入知识: {'是 [OK]' if hit else '否 [FAIL]'}")
    return hit


def test_learn_and_reuse():
    """学习/调用测试：自学习写入含代码因子后，检索应能返回可复用代码。"""
    print("\n=== 测试二：学习闭环（自学习写入 -> 检索调用 -> 持久化）===")
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    tmp_path = tmp.name
    tmp.close()

    lib = LearnedFactorLibrary(tmp_path)
    code = (
        "def alpha_factor(df):\n"
        "    df = df.copy()\n"
        "    df['factor'] = df.groupby('symbol')['close'].transform(\n"
        "        lambda x: x.shift(1) / x.shift(21) - 1)\n"
        "    return df[['date', 'symbol', 'factor']]"
    )
    rec = {
        "title": "20日动量因子",
        "category": "动量/反转",
        "formula": "close.shift(1)/close.shift(21)-1",
        "description": "过去20个交易日收益率，衡量价格动量",
        "code": code,
        "source": "self_learned",
    }
    lib.add(rec)
    print(f"  自学习写入因子：{rec['title']}（含代码）")

    # 重建检索器（模拟新会话加载学习库）
    retr = FactorRetriever(learned=LearnedFactorLibrary(tmp_path), top_k=3)
    tpl = retr.retrieve_template("构建一个动量因子", top_k=1)
    has_code = bool(tpl) and bool(tpl[0].get("code"))
    if tpl:
        print(f"  retrieve_template 命中：{tpl[0].get('title')}（含代码: {'是 [OK]' if has_code else '否 [FAIL]'}）")
    else:
        print("  retrieve_template 命中：无 [FAIL]")

    # 持久化验证：重新从磁盘加载
    reloaded = LearnedFactorLibrary(tmp_path)
    persisted = reloaded.get("20日动量因子")
    print(f"  磁盘持久化校验：{'成功 [OK]' if persisted and persisted.get('code') else '失败 [FAIL]'}")

    os.unlink(tmp_path)
    return has_code and bool(persisted)


if __name__ == "__main__":
    real_ok = os.path.exists(REAL_LEARNED) and os.path.getsize(REAL_LEARNED) > 0
    print(f"真实学习库：{REAL_LEARNED}（{'存在且非空 [OK]' if real_ok else '缺失/为空 [FAIL]'}）")
    if not real_ok:
        print("请先运行：python scripts/import_factors.py <因子文件> <source>")
        sys.exit(1)

    r1 = test_query()
    r2 = test_learn_and_reuse()
    print("\n=== 结论 ===")
    print(f"  自适应查阅：{'通过 [OK]' if r1 else '未通过 [FAIL]'}")
    print(f"  学习/调用闭环：{'通过 [OK]' if r2 else '未通过 [FAIL]'}")
    sys.exit(0 if (r1 and r2) else 1)
