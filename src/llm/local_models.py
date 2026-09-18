"""
本地大模型探测（src/llm/local_models.py）
=========================================

云端密钥会过期、会被限流、会 401；而机器上往往已经跑着一个 Ollama。本模块把
「本机有没有可用的本地模型」变成一次可回答的查询，让界面能列出真实存在的模型名，
而不是让用户靠记忆手敲 ``qwen2.5-coder:7b``。

只依赖标准库（urllib）：探测是「配置模型」这种高频轻量操作的第一步，不该因为
缺 requests/httpx 就整块功能不可用。所有网络异常都收敛成 ``available=False`` 并
附上原因——探测失败必须能显示在界面上，否则用户只会看到「模型列表为空」而
无从判断是没装模型还是服务没起。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

__all__ = ["DEFAULT_OLLAMA_BASE_URL", "normalize_ollama_base_url", "probe_ollama", "list_ollama_models"]

#: Ollama 的 OpenAI 兼容端点（供 ChatOpenAI 使用）。
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"

#: 探测用的原生 API 端口与 OpenAI 兼容端口一致，仅路径不同。
_TAGS_PATH = "/api/tags"

# 探测结果缓存：界面每次 rerun 都会重绘，不能每帧都打一次本机端口。
_PROBE_CACHE: Dict[str, Dict[str, Any]] = {}


def normalize_ollama_base_url(base_url: str = "") -> str:
    """把用户填的任意形态端点归一成 OpenAI 兼容地址。

    Ollama 只暴露一个端口，但用户可能填 ``http://localhost:11434``、
    ``.../v1`` 甚至 ``.../api/tags``；ChatOpenAI 要的正是 ``.../v1``。
    """
    s = (base_url or "").strip().rstrip("/")
    if not s:
        return DEFAULT_OLLAMA_BASE_URL
    for suffix in ("/api/tags", "/api/chat", "/api/generate", "/v1"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    if not s.endswith("/v1"):
        s += "/v1"
    return s


def _http_get_json(url: str, timeout: float) -> Any:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def probe_ollama(base_url: str = "", timeout: float = 3.0, use_cache: bool = True) -> Dict[str, Any]:
    """探测本机 Ollama 是否可用。

    Returns:
        ``{"available": bool, "models": [str, ...], "base_url": str, "error": str}``
        ``models`` 为 ``ollama list`` 里已拉取的模型名（不含 ":latest" 之外的标签
        过滤），可直接填进「模型名称」输入框。
    """
    endpoint = normalize_ollama_base_url(base_url)
    origin = endpoint[: -len("/v1")]
    cache_key = f"{origin}|{round(timeout, 1)}"
    if use_cache and cache_key in _PROBE_CACHE:
        return dict(_PROBE_CACHE[cache_key])

    out: Dict[str, Any] = {"available": False, "models": [], "base_url": endpoint, "error": ""}
    try:
        data = _http_get_json(origin + _TAGS_PATH, timeout)
        models: List[str] = []
        for m in (data or {}).get("models", []) or []:
            # 新版本字段是 name，旧版本是 model；两者都兜住。
            name = str(m.get("name") or m.get("model") or "").strip()
            if name:
                models.append(name)
        out["available"] = True
        out["models"] = sorted(set(models))
    except urllib.error.URLError as e:
        out["error"] = f"无法连接 {origin}（Ollama 未启动？）：{e}"
    except Exception as e:  # noqa: BLE001 —— 探测失败只影响可选功能
        out["error"] = f"{type(e).__name__}: {e}"

    if use_cache and out["available"]:
        _PROBE_CACHE[cache_key] = dict(out)
    return out


def list_ollama_models(base_url: str = "", timeout: float = 3.0) -> List[str]:
    """可用则返回已安装模型名列表，不可用返回空列表。"""
    return list(probe_ollama(base_url, timeout).get("models") or [])


def preferred_model(models: Optional[List[str]] = None) -> str:
    """从模型列表里挑一个适合代码生成的默认项（编码模型 > 大模型 > 随便一个）。"""
    models = list(models or [])
    if not models:
        return ""
    for kw in ("coder", "code", "qwen", "deepseek"):
        for m in models:
            if kw in m.lower():
                return m
    return models[0]
