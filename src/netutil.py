"""
网络代理工具（src/netutil.py）

统一管理 requests / httpx 的代理与直连策略，供数据抓取（akshare/东方财富/新浪）
与 LLM 客户端（Ollama / 远程 OpenAI 兼容端点）复用：

- 默认（未配置代理）强制直连：设置 NO_PROXY=*，并把 requests.Session.trust_env
  设为 False，规避 Windows 不可达系统代理（注册表）导致的 ProxyError。
- 当 config.yaml 的 ``proxy`` 段启用且给出地址时，改走指定代理；
- 若 config 未启用但已设置 HTTP_PROXY/HTTPS_PROXY 环境变量，则尊重该环境变量。
- 无论哪种模式，localhost / 127.0.0.1 始终直连，保证本地 Ollama 等服务不受影响。

使用方法：
    from netutil import apply_proxy_settings, get_trust_env, patch_requests_session

    apply_proxy_settings(proxy_cfg)   # proxy_cfg 为 config.yaml 的 proxy 段（可为 None）
    patch_requests_session()          # 让 requests.Session 跟随当前策略（仅需一次）
    httpx.Client(trust_env=get_trust_env())  # 给 httpx 客户端用
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

# 可变状态：requests.Session patch 与 httpx 客户端据此决定 trust_env。
_TRUST_ENV: Dict[str, bool] = {"value": False}


def apply_proxy_settings(proxy_cfg: Optional[Dict[str, Any]] = None) -> bool:
    """根据 config.yaml 的 proxy 段（或用户环境变量）应用代理/直连环境变量。

    Args:
        proxy_cfg: config.yaml 解析出的 ``proxy`` 段（可为 None）。期望字段：
            - enabled: bool，是否启用 config 中的代理（默认 False）
            - http:    HTTP 代理地址，如 ``http://127.0.0.1:7890``
            - https:   HTTPS 代理地址，留空时复用 http
            - no_proxy: 始终直连的地址（默认 ``localhost,127.0.0.1``）

    Returns:
        trust_env 取值：True 表示使用代理环境变量，False 表示强制直连。
    """
    proxy_cfg = proxy_cfg or {}
    enabled = bool(proxy_cfg.get("enabled", False))
    http = (proxy_cfg.get("http") or "").strip()
    https = (proxy_cfg.get("https") or http).strip()
    no_proxy = proxy_cfg.get("no_proxy") or "localhost,127.0.0.1"

    if enabled and (http or https):
        # 配置优先：使用 config.yaml 中显式给出的代理地址
        if http:
            os.environ["HTTP_PROXY"] = http
            os.environ["http_proxy"] = http
        if https:
            os.environ["HTTPS_PROXY"] = https
            os.environ["https_proxy"] = https
        # 本地地址始终直连，避免把 Ollama/本地服务也塞进代理链路
        os.environ["NO_PROXY"] = no_proxy
        os.environ["no_proxy"] = no_proxy
        _TRUST_ENV["value"] = True
    else:
        # 配置未启用：尊重用户已有的 HTTP_PROXY 环境变量；否则强制直连避免 ProxyError
        env_http = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
        env_https = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if env_http or env_https:
            os.environ.setdefault("NO_PROXY", no_proxy)
            os.environ.setdefault("no_proxy", no_proxy)
            _TRUST_ENV["value"] = True
        else:
            os.environ["NO_PROXY"] = "*"
            os.environ["no_proxy"] = "*"
            _TRUST_ENV["value"] = False
    return _TRUST_ENV["value"]


def get_trust_env() -> bool:
    """返回当前代理策略下 requests/httpx 应使用的 trust_env 值。"""
    return _TRUST_ENV["value"]


def patch_requests_session() -> None:
    """monkeypatch requests.Session.__init__，使其 trust_env 跟随当前代理策略。

    只需在进程启动时调用一次。对未打补丁或已打补丁都能安全幂等执行。
    """
    try:
        req = __import__("requests")
        if getattr(req.Session.__init__, "_netutil_patched", False):
            return
        orig_init = req.Session.__init__

        def _patched_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            self.trust_env = _TRUST_ENV["value"]

        _patched_init._netutil_patched = True
        req.Session.__init__ = _patched_init
    except Exception:  # noqa: BLE001
        # requests 未安装或打补丁失败时不阻塞主流程
        pass
