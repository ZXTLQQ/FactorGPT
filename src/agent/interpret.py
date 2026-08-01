"""因子可解释性说明卡。

把「黑盒」的 LLM 生成因子转成可读的归因说明：
  1) 风格载荷：因子到底在押注规模 / 动量 / 波动率中的哪一个（来自风险模型）；
  2) 行业中性度：是否对某个行业重度偏配；
  3) 代码逻辑要点：自动抽取代码中的关键算子（rolling / rank / shift / groupby 等）；
  4) （可选）LLM 自然语言解读：用大白话解释经济逻辑与潜在风险。

该模块不依赖网络；LLM 解读为可选项，缺失时仅输出确定性部分。
"""
from __future__ import annotations

from typing import Dict, Optional


def _code_keypoints(code: str) -> list:
    """抽取因子代码中的关键算子，作为逻辑要点。"""
    kp = []
    checks = [
        ("rolling", "使用了滚动窗口（动量 / 波动率类信号）"),
        ("shift(1)", "对信号做 shift(1)（满足无前视契约）"),
        ("shift(-", "检测到负向 shift（疑似前视，需警惕）"),
        ("rank", "做了截面排名（稳健性处理）"),
        ("groupby", "做了分组运算（个股独立处理）"),
        ("pct_change", "基于收益率构造"),
        ("rolling", "滚动窗口"),
    ]
    seen = set()
    for token, desc in checks:
        if token in code and token not in seen:
            seen.add(token)
            kp.append(desc)
    return kp


def factor_interpretability_card(
    name: str,
    code: str,
    metrics: Dict,
    llm=None,
) -> str:
    """生成因子可解释性说明卡（markdown 文本）。"""
    style = metrics.get("_style_exposure") or {}
    ind = metrics.get("_industry_exposure") or {}
    ic = metrics.get("ic")
    ls_sharpe = metrics.get("long_short_sharpe")

    lines = [f"**因子「{name}」可解释性说明卡**", ""]
    lines.append(f"- 样本内 IC：{ic if ic is not None else 'N/A'} ｜ 多空 Sharpe：{ls_sharpe if ls_sharpe is not None else 'N/A'}")

    # 1) 风格载荷
    label = {"size": "规模(size)", "momentum": "动量(momentum)", "volatility": "波动率(volatility)"}
    loadings = []
    for k, v in style.items():
        if isinstance(v, float) and v == v:  # 非 NaN
            direction = "正向载荷" if v > 0.02 else ("负向载荷" if v < -0.02 else "近似中性")
            loadings.append(f"{label.get(k, k)}：{direction}（暴露 {v:+.3f}）")
    if loadings:
        lines.append("\n**1. 风格载荷（因子在押注什么）**")
        lines += [f"- {x}" for x in loadings]
    else:
        lines.append("\n**1. 风格载荷**：无可用暴露（缺少市值/收益数据）")

    # 2) 行业中性度
    if isinstance(ind, dict) and ind:
        if ind.get("neutral"):
            lines.append(f"- 行业：中性（跨行业最大偏配 {ind.get('max_bias', 0):.3f}）")
        else:
            lines.append(f"- 行业：**非中性**（跨行业最大偏配 {ind.get('max_bias', 0):.3f}），"
                         "组合构建时建议加行业中性约束。")

    # 3) 代码逻辑要点
    kp = _code_keypoints(code)
    if kp:
        lines.append("\n**2. 实现逻辑要点**")
        lines += [f"- {x}" for x in kp]

    # 4) LLM 自然语言解读（可选）
    if llm is not None:
        try:
            txt = llm.complete(
                system=(
                    "你是资深量化研究员。请用 2-3 句通俗中文解释下面因子的经济逻辑、"
                    "可能有效的来源，以及需要警惕的过拟合/风险点。不要写代码。"
                ),
                user=f"因子名：{name}\n样本内IC：{ic}\n代码：\n{code}",
                temperature=0.3,
            )
            if txt:
                lines.append("\n**3. 逻辑解读（LLM）**")
                lines.append(txt.strip())
        except Exception:  # noqa: BLE001
            pass

    return "\n".join(lines)
