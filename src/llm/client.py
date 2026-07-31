"""
LLM 客户端封装（src/llm/client.py）

基于 LangChain 封装 DeepSeek / OpenAI / Qwen 等 OpenAI 兼容接口，
为 FactorGPT 的因子生成、反思等环节提供统一的对话与结构化抽取能力。

设计要点：
- 通过 config.yaml 中的 `llm` 段配置 provider / api_key / model / base_url。
- 提供 `complete(system, user)` 简易接口与 `chat(messages)` 消息接口。
- 提供 `extract_code_block` / `extract_json` 两个工具函数，用于从模型
  返回文本中稳健地抽取 Python 代码块与 JSON。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import yaml

# 代理/直连策略统一由 netutil 管理：默认强制直连（规避不可达系统代理），
# 当 config.yaml 的 proxy 段启用或已设置 HTTP_PROXY 环境变量时，改走代理；
# localhost/127.0.0.1 始终直连，保证本地 Ollama 不受影响。具体策略在模块底部应用。
from netutil import apply_proxy_settings, get_trust_env


class LLMClient:
    """统一的 LLM 调用客户端。

    Attributes:
        provider: 模型供应商，'deepseek' / 'openai' / 'qwen'。
        model: 模型名称。
        api_key: API Key。
        base_url: OpenAI 兼容接口地址。
        temperature: 采样温度。
        _llm: 懒加载的 LangChain ChatModel 实例。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is None:
            config = {}
        llm_cfg = config.get("llm", {}) if isinstance(config, dict) else {}
        self.provider = (llm_cfg.get("provider", "deepseek") or "deepseek").lower()
        self.model = llm_cfg.get("model", "deepseek-chat")
        self.api_key = llm_cfg.get("api_key", "")
        self.base_url = llm_cfg.get("base_url", "")
        self.temperature = float(llm_cfg.get("temperature", 0.3))
        self.timeout = float(llm_cfg.get("timeout", 60.0))
        self._llm = None

        # 环境变量覆盖（默认关闭, 由 config.llm.use_env_override 控制）。
        # 便于不落盘密钥或快速切换后端, 例如:
        #   FACTORGPT_LLM_PROVIDER=ollama
        #   FACTORGPT_LLM_BASE_URL=http://localhost:11434/v1
        #   FACTORGPT_LLM_API_KEY=ollama
        #   FACTORGPT_LLM_MODEL=qwen2.5-coder:7b
        if bool(llm_cfg.get("use_env_override", False)):
            self.provider = os.getenv("FACTORGPT_LLM_PROVIDER", self.provider).lower()
            self.model = os.getenv("FACTORGPT_LLM_MODEL", self.model)
            self.api_key = os.getenv("FACTORGPT_LLM_API_KEY", self.api_key)
            self.base_url = os.getenv("FACTORGPT_LLM_BASE_URL", self.base_url)

        # provider 感知的默认端点 / key 兜底(base_url 未显式配置时生效)。
        if self.provider == "ollama":
            if not self.base_url:
                self.base_url = "http://localhost:11434/v1"
            if not self.api_key:
                self.api_key = "ollama"   # Ollama 不校验 key
        elif self.provider == "deepseek":
            if not self.base_url:
                self.base_url = "https://api.deepseek.com/v1"
        elif self.provider == "qwen":
            if not self.base_url:
                self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        # provider=openai / vllm 等: 直接使用 config 中给定的 base_url

    # ------------------------------------------------------------------
    # 模型构建
    # ------------------------------------------------------------------
    def _build(self):
        if self._llm is not None:
            return self._llm
        # 所有 provider 均走 OpenAI 兼容接口（DeepSeek / OpenAI / Qwen / 本地
        # Ollama / vLLM / Groq / OpenRouter 等任意 OpenAI-compatible 端点皆可）。
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "未安装 langchain-openai，请执行 pip install langchain-openai"
            ) from e
        kwargs = dict(
            model=self.model,
            api_key=self.api_key or "EMPTY",
            temperature=self.temperature,
            timeout=self.timeout,
            max_tokens=2048,
        )
        # 自定义/本地端点往往无需鉴权或 base_url 为空，仅在有值时传入。
        if self.base_url:
            kwargs["base_url"] = self.base_url
        # 强制本地/自定义端点不走系统代理：构建显式禁用 trust_env 的 httpx 客户端，
        # 双保险覆盖 Windows 注册表代理（即使未设置 NO_PROXY 环境变量也直连）。
        try:
            import httpx

            kwargs["http_client"] = httpx.Client(trust_env=get_trust_env())
        except Exception:  # pragma: no cover
            pass
        self._llm = ChatOpenAI(**kwargs)
        return self._llm

    # ------------------------------------------------------------------
    # 运行时切换（支持用户在 UI 中通过 API Key 切换模型/供应商）
    # ------------------------------------------------------------------
    def set_model(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """在运行时更新模型配置并失效已构建的客户端实例。

        未传的参数保持原值。调用后下一次 `chat/complete` 将使用新配置。
        """
        if provider is not None:
            self.provider = provider
        if model is not None:
            self.model = model
        if api_key is not None:
            self.api_key = api_key
        if base_url is not None:
            self.base_url = base_url
        if temperature is not None:
            self.temperature = float(temperature)
        self._llm = None  # 失效缓存，下次调用重新构建

    # ------------------------------------------------------------------
    # 调用接口
    # ------------------------------------------------------------------
    def chat(self, messages: List[Any], temperature: Optional[float] = None) -> str:
        """发送消息列表并返回模型文本回复。

        Args:
            messages: LangChain 消息对象列表（SystemMessage / HumanMessage ...）。
            temperature: 覆盖默认采样温度。

        Returns:
            模型返回的文本字符串。
        """
        llm = self._build()
        if temperature is not None:
            llm = llm.with_config({"temperature": temperature})
        resp = llm.invoke(messages)
        return resp.content if hasattr(resp, "content") else str(resp)

    def complete(self, system: str, user: str, temperature: Optional[float] = None) -> str:
        """以 system + user 两段式发起一次对话。"""
        from langchain_core.messages import HumanMessage, SystemMessage

        return self.chat(
            [SystemMessage(content=system), HumanMessage(content=user)],
            temperature=temperature,
        )

    def available(self) -> bool:
        """探测 API 是否可用（仅做构建级探测，不实际发请求）。"""
        try:
            self._build()
            return True
        except Exception:
            return False


# ----------------------------------------------------------------------
# 文本抽取工具
# ----------------------------------------------------------------------
_CODE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)
_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_code_block(text: str) -> str:
    """从模型返回文本中提取第一个 Python 代码块。

    若没有 Markdown 代码围栏，则退而求其次返回整段文本中疑似代码的部分。

    Args:
        text: 模型返回文本。

    Returns:
        抽取出的代码字符串；若都失败返回空字符串。
    """
    if not text:
        return ""
    matches = _CODE_RE.findall(text)
    if matches:
        return matches[0].strip()
    # 退化策略：直接返回文本（可能本身就是纯代码）
    return text.strip()


def extract_json(text: str) -> Optional[Any]:
    """从模型返回文本中提取 JSON 对象。

    Args:
        text: 模型返回文本。

    Returns:
        解析出的 Python 对象；解析失败返回 None。
    """
    if not text:
        return None
    # 优先尝试代码围栏内
    candidates = _JSON_RE.findall(text)
    candidates.append(text)
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
    # 退而求其次：截取第一个 { 到最后一个 } 之间的内容
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    except (json.JSONDecodeError, TypeError):
        return None
    return None


def _load_dotenv(path: str = ".env") -> None:
    """极简 .env 加载（零依赖）：把 KEY=VALUE 注入 os.environ（不覆盖已有值）。"""
    import os

    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)
    except Exception:  # noqa: BLE001
        pass


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """加载项目根目录下的 config.yaml。

    支持在 YAML 中以 ``${ENV_VAR}`` 形式引用环境变量，便于把密钥等敏感配置
    从 ``config.yaml`` 剥离到 ``.env``（已被 .gitignore 忽略）中，避免明文密钥入库。
    调用前会先尝试加载 ``.env``（若存在）。
    """
    import os
    import re

    _load_dotenv()

    if not os.path.isabs(path):
        # 允许从仓库任意位置调用：向上查找 config.yaml
        here = os.path.dirname(os.path.abspath(__file__))
        for _ in range(5):
            cand = os.path.join(here, path)
            if os.path.exists(cand):
                path = cand
                break
            here = os.path.dirname(here)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # 环境变量插值：${VAR} -> os.environ[VAR]（未设置则保留原样，便于本地直接填值）
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

    def _sub(m: "re.Match[str]") -> str:
        return os.environ.get(m.group(1), m.group(0))

    text = pattern.sub(_sub, text)
    return yaml.safe_load(text)


# 应用代理/直连策略：config.yaml 的 proxy 段优先；否则尊重 HTTP_PROXY 环境变量；
# 两者皆无则强制直连（规避不可达系统代理）。localhost/127.0.0.1 始终直连。
try:
    _app_cfg = load_config()
    apply_proxy_settings((_app_cfg or {}).get("proxy"))
except Exception:  # noqa: BLE001
    apply_proxy_settings(None)
