"""多 LLM 路由 (src/llm/router.py)

性价比最高的因子生成升级：用小模型（draft，如本地 Qwen / Ollama）低成本、高吞吐地
海选候选因子代码，再用强模型（critic，如 DeepSeek / GPT-4o）评估、修复、精炼，
既降 API 成本又提吞吐。当未配置 draft 时，critic 直接作为主生成器。

接入主流程：在 FactorAgentNodes 中传入 LLMRouter 替代单一 LLMClient 即可，
两个方法分别对应"生成"与"反思精炼"两个节点：
    code = router.generate_factor_code(system, user)
    improved = router.review_and_refine(code, metrics, description)
"""
from __future__ import annotations

import json
from typing import Optional

from llm.client import LLMClient, extract_code_block, extract_json


class LLMRouter:
    """draft（小模型海选）+ critic（强模型精炼）的双层路由。"""

    def __init__(
        self,
        draft_config: Optional[dict] = None,
        critic_config: Optional[dict] = None,
        config: Optional[dict] = None,
        draft_client: Optional[LLMClient] = None,
        critic_client: Optional[LLMClient] = None,
    ) -> None:
        # 允许从整体 config["llm"]["router"] 读取 draft / critic 子配置
        if config is not None:
            rc = (config.get("llm", {}) or {}).get("router", {}) or {}
            draft_config = draft_config or rc.get("draft")
            critic_config = critic_config or rc.get("critic")
        # 优先使用直接传入的 client 实例（便于测试 / 复用已构造 client）
        self.draft = draft_client or self._build(draft_config)
        self.critic = critic_client or self._build(critic_config)
        if self.draft is None and self.critic is None:
            raise ValueError("LLMRouter 需要至少配置 draft 或 critic 之一")

    @staticmethod
    def _build(sub: Optional[dict]) -> Optional[LLMClient]:
        if not sub:
            return None
        # 子配置即 llm 段内容，包装成 {"llm": sub} 交给 LLMClient
        cfg = sub if "llm" in sub else {"llm": sub}
        return LLMClient(cfg)

    def _call(self, client: LLMClient, system: str, user: str, temperature: Optional[float]) -> str:
        return client.complete(system=system, user=user, temperature=temperature)

    def generate_factor_code(self, system: str, user: str, temperature: float = 0.4) -> Optional[str]:
        """draft 生成候选 -> critic 精炼。返回最终因子代码字符串。"""
        draft_code: Optional[str] = None
        if self.draft is not None:
            try:
                raw = self._call(self.draft, system, user, temperature)
                draft_code = self._parse_code(raw)
            except Exception as e:
                print(f"[router] draft 生成失败，转 critic 主生成：{e}")
        if self.critic is not None:
            extra = ""
            if draft_code:
                extra = (
                    "\n\n【草稿因子代码（请评估其正确性，修复前视/语法/逻辑问题，"
                    "产出最终可运行版本）】\n" + draft_code
                )
            try:
                raw = self._call(self.critic, system, user + extra, temperature)
                code = self._parse_code(raw)
                return code or draft_code
            except Exception as e:
                print(f"[router] critic 精炼失败，沿用 draft：{e}")
                return draft_code
        return draft_code

    def review_and_refine(self, code: str, metrics: dict, description: str = "", temperature: float = 0.5) -> str:
        """critic 基于指标反思并改进代码；无 critic 则原样返回。"""
        if self.critic is None:
            return code
        show = {k: v for k, v in metrics.items() if not str(k).startswith("_")}
        user = (
            f"【因子需求】{description}\n\n"
            f"【当前因子代码】\n{code}\n\n"
            f"【回测指标】{json.dumps(show, ensure_ascii=False, default=str)}\n\n"
            "请作为严谨的量化研究员，指出该因子的问题（含前视偏差/过拟合/逻辑缺陷），"
            "并产出改进后的代码。仅返回 JSON：{\"reflection\": str, \"code\": str}。"
        )
        try:
            raw = self._call(
                self.critic,
                "你是严谨的量化因子研究员，擅长发现前视偏差与过拟合。",
                user,
                temperature,
            )
            parsed = extract_json(raw)
            if isinstance(parsed, dict) and parsed.get("code"):
                return parsed["code"]
            code2 = extract_code_block(raw)
            return code2 or code
        except Exception as e:
            print(f"[router] review 失败，沿用原代码：{e}")
            return code

    @staticmethod
    def _parse_code(raw: str) -> Optional[str]:
        if not raw:
            return None
        parsed = extract_json(raw)
        if isinstance(parsed, dict) and parsed.get("code"):
            return parsed["code"]
        return extract_code_block(raw)

    def complete(self, system: str, user: str, temperature: Optional[float] = None) -> str:
        """兼容 LLMClient.complete 的接口：draft 海选 + critic 精炼，返回完整 JSON 文本。

        让 FactorAgentNodes 在配置了 router 时无需改动调用方式即可享受双层路由。
        draft 生成候选 JSON（低成本），critic 在其基础上评估并产出最终完整 JSON
        （含 name/description/code/rationale/references 等字段）。
        """
        draft_raw: Optional[str] = None
        if self.draft is not None:
            try:
                draft_raw = self._call(self.draft, system, user, temperature)
            except Exception as e:
                print(f"[router] draft 失败，转 critic 主生成：{e}")
        if self.critic is not None:
            extra = ""
            if draft_raw:
                extra = (
                    "\n\n【草稿因子（请评估其正确性，修复前视/语法/逻辑问题，"
                    "并产出最终完整 JSON，字段包含 name/description/code/rationale/references）】\n"
                    + draft_raw
                )
            try:
                return self._call(self.critic, system, user + extra, temperature)
            except Exception as e:
                print(f"[router] critic 失败，沿用 draft：{e}")
                return draft_raw or ""
        return draft_raw or ""
