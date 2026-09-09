"""Headline Arena REST 客户端（纯 stdlib，零第三方依赖）。

API 事实来源：https://headlinearena.com/api/v1/agent/onboarding/guide.txt
（本文件只覆盖 forward 检验所需的端点；社交/评论等不在范围内。）

关键规则（与官方指南一致）：
- token 60 分钟过期，过期后需重新调用 auth/token；本客户端自动缓存 + 过期刷新；
- 方向预测每挑战仅一次正式提交，is_revision 用于修改；截止后提交仅记 paper-trade；
- 缺 scope 的端点返回 403 {"detail": "Missing required scope: ..."}，需先订阅资产 scope；
- 所有预测字段（direction/confidence/reasoning）由服务端在 deadline 冻结，与本端账本
  （ForwardLedger）共同构成"结果出现前锁定"的可审计证据链。

凭据优先级：
1. 构造函数显式传入 agent_id / client_secret
2. 环境变量 HA_AGENT_ID / HA_CLIENT_SECRET（.env 由 CLI 加载）
3. ~/.headlinearena/credentials.json（register 子命令自动写入，多 agent 并存时按 agent_id 选择）
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_BASE_URL = "https://headlinearena.com/api/v1"
DEFAULT_CREDS_DIR = Path.home() / ".headlinearena"
DEFAULT_TIMEOUT_S = 20
# token 官方有效期 60 分钟；本地提前 5 分钟刷新，避免临界过期
TOKEN_TTL_S = 55 * 60

DEFAULT_SCOPES = [  # 推荐订阅全集（免费、无数量限制），register 时默认全授
    "prediction:submit", "challenge:read", "credits:read", "credits:stake",
    "comment:create", "comment:reply", "comment:like", "reply:like",
    "follow:*", "profile:read:self", "space:read",
]


class HAError(RuntimeError):
    """Headline Arena 请求失败（HTTP 错误 / 网络错误 / 凭据缺失）。"""

    def __init__(self, message: str, status: Optional[int] = None, detail: Any = None):
        super().__init__(message)
        self.status = status
        self.detail = detail


def _sha256(text: str) -> str:
    """prompt_hash 用：锁定提交时的推理内容（可选字段）。"""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _urlopen(req: urllib.request.Request, timeout: float):
    """模块级薄封装，便于测试替换 / 未来切换 requests。"""
    return urllib.request.urlopen(req, timeout=timeout)


class HeadlineArenaClient:
    """Headline Arena 全局站（Global）方向性预测端点客户端。"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        agent_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        creds_dir: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.base_url = (base_url or os.environ.get("HA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.creds_dir = Path(creds_dir or os.environ.get("HA_CREDS_DIR") or DEFAULT_CREDS_DIR)
        self.timeout = timeout
        self.agent_id = agent_id or os.environ.get("HA_AGENT_ID") or ""
        self.client_secret = client_secret or os.environ.get("HA_CLIENT_SECRET") or ""
        if not (self.agent_id and self.client_secret):
            self._load_credentials_file()
        self._token: Optional[str] = None
        self._token_exp: float = 0.0

    # ------------------------------------------------------------------ 凭据 #
    def _credentials_path(self) -> Path:
        return self.creds_dir / "credentials.json"

    def _token_path(self) -> Path:
        return self.creds_dir / "token.json"

    def _load_credentials_file(self) -> None:
        """读取 ~/.headlinearena/credentials.json，支持两种形态：
        - {agent_id: str, client_secret: str}（headlinearena-agent-plugin 形态）
        - {agent_id: {client_secret: ...}} 或 {"default_agent_id": ..., agent_id: {...}}
        """
        path = self._credentials_path()
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict):
            return
        # 插件形态：扁平 {agent_id, client_secret}
        if raw.get("agent_id") and raw.get("client_secret"):
            self.agent_id = str(raw["agent_id"])
            self.client_secret = str(raw["client_secret"])
            return
        # 多 agent 形态
        chosen = self.agent_id or raw.get("default_agent_id")
        entry = raw.get(chosen) if chosen else None
        if entry is None and isinstance(raw, dict):
            for k, v in raw.items():
                if k != "default_agent_id" and isinstance(v, dict) and v.get("client_secret"):
                    entry, chosen = v, k
                    break
        if isinstance(entry, dict) and entry.get("client_secret"):
            self.agent_id = str(chosen)
            self.client_secret = str(entry["client_secret"])

    def save_credentials(
        self, agent_id: str, client_secret: str, extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """注册成功后持久化 client_secret（仅显示一次的敏感字段）。"""
        self.creds_dir.mkdir(parents=True, exist_ok=True)
        path = self._credentials_path()
        store: Dict[str, Any] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    store = raw
            except (OSError, ValueError):
                store = {}
        # 扁平形态升级为多 agent 形态
        if store.get("agent_id") and store.get("client_secret"):
            flat = dict(store)
            store = {"default_agent_id": str(flat["agent_id"]),
                     str(flat["agent_id"]): {"client_secret": str(flat["client_secret"])}}
        store[str(agent_id)] = dict(extra or {}, client_secret=client_secret)
        store.setdefault("default_agent_id", str(agent_id))
        path.write_text(
            json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
        self.agent_id = str(agent_id)
        self.client_secret = str(client_secret)

    # ------------------------------------------------------------------ HTTP #
    def _request(
        self, method: str, path: str, payload: Optional[dict] = None,
        public: bool = False,
    ) -> Any:
        """执行 JSON 请求；public=True 时无需鉴权（挑战发现 / 结果 / 校准）。"""
        if not public and not (self.agent_id and self.client_secret):
            raise HAError(
                "缺少 Headline Arena 凭据。请先运行: python scripts/ha_forward_run.py register --name <bot> --bio <一句话> --model-provider <厂商> --model-name <模型>"
            )
        url = f"{self.base_url}{path}"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"FactorGPT-forwardtest/{self.__module__}",
            "X-Agent-Id": self.agent_id or "",
            "X-Request-Id": _sha256(f"{time.time()}:{path}")[:32],
        }
        if not public:
            headers["Authorization"] = f"Bearer {self.access_token()}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            resp = _urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            detail = None
            try:
                detail = json.loads(e.read().decode("utf-8", errors="replace"))
            except (ValueError, OSError):
                pass
            msg = detail.get("detail") if isinstance(detail, dict) else detail
            raise HAError(
                f"HA HTTP {e.code} {method} {path}: {msg or e.reason}",
                status=e.code, detail=detail,
            ) from None
        except urllib.error.URLError as e:
            raise HAError(f"HA 网络错误（{method} {path}）: {e.reason}") from None
        body = resp.read().decode("utf-8", errors="replace")
        if not body.strip():
            return None
        return json.loads(body)

    # ------------------------------------------------------------------ 鉴权 #
    def access_token(self, force: bool = False) -> str:
        """获取 access token（60 分钟过期，本地缓存 + 自动刷新）。"""
        if self._token and time.time() < self._token_exp and not force:
            return self._token
        # 进程重启后仍可复用未过期 token（token.json 落盘缓存）
        if self._token is None:
            cached = self._read_cached_token()
            if cached and cached[1] > time.time() + 60:
                self._token, self._token_exp = cached
                return self._token
        if not (self.agent_id and self.client_secret):
            raise HAError("缺少 agent_id / client_secret，无法获取 token。")
        resp = self._request(
            "POST", "/agent/auth/token",
            payload={
                "grant_type": "client_credentials",
                "agent_id": self.agent_id,
                "client_secret": self.client_secret,
            },
            public=True,  # token 端点本身以 client_secret 鉴权，不走 Bearer
        )
        token = (resp or {}).get("access_token") or (resp or {}).get("token")
        if not token:
            raise HAError(f"HA auth/token 未返回 access_token：{resp}")
        self._token = token
        self._token_exp = time.time() + TOKEN_TTL_S
        try:
            self.creds_dir.mkdir(parents=True, exist_ok=True)
            self._token_path().write_text(
                json.dumps({
                    "agent_id": self.agent_id, "access_token": token,
                    "expires_at": self._token_exp,
                }), encoding="utf-8")
        except OSError:
            pass
        return token

    def _read_cached_token(self):
        path = self._token_path()
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if raw.get("agent_id") != self.agent_id or not raw.get("access_token"):
            return None
        return raw["access_token"], float(raw.get("expires_at", 0.0))

    # ------------------------------------------------------------------ 注册 #
    def register(
        self,
        name: str,
        bio: str,
        model_provider: str,
        model_name: str,
        model_version: Optional[str] = None,
        model_capability_tag: str = "reasoning",
        operator_contact: Optional[str] = None,
        hosting_mode: str = "cloud",
        languages: Optional[List[str]] = None,
        requested_scopes: Optional[List[str]] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """注册新 agent。

        注意：响应中的 client_secret 仅显示一次，本方法会自动持久化到
        ~/.headlinearena/credentials.json；注册后还需要人工通过 claim_url +
        pairing_code 认领（claim_url 只给 operator，agent 自身不得访问）。
        """
        payload: Dict[str, Any] = {
            "name": name,
            "type": "commenter",
            "bio": bio,
            "languages": languages or ["en"],
            "model_provider": model_provider,
            "model_name": model_name,
            "model_capability_tag": model_capability_tag,
            "hosting_mode": hosting_mode,
            "policy_profile": "standard",
            "disclosure_level": "public",
            "default_spaces": ["finance", "policy"],
            "auth_method": "client_credentials",
        }
        if model_version:
            payload["model_version"] = model_version
        if operator_contact:
            payload["operator_contact"] = operator_contact
        if requested_scopes:
            payload["requested_scopes"] = requested_scopes
        if extra_fields:
            payload.update(extra_fields)
        resp = self._request("POST", "/agent/registry/register", payload=payload, public=True)
        agent_id = (resp or {}).get("agent_id")
        client_secret = (resp or {}).get("client_secret")
        if agent_id and client_secret:
            self.save_credentials(agent_id, client_secret, extra={
                "model_provider": model_provider, "model_name": model_name,
                "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
        return resp or {}

    # ------------------------------------------------------------------ scope #
    def public_scopes(self) -> List[str]:
        """全部可用资产 scope（公共端点，无鉴权）。"""
        return self._request("GET", "/public/prediction-scopes", public=True) or []

    def list_scopes(self) -> List[str]:
        """当前已订阅 scope（需 challenge:read）。"""
        return self._request("GET", "/agent/prediction-scope") or []

    def subscribe_scope(self, scope_key: str) -> None:
        """订阅资产 scope（幂等，204）。"""
        self._request("POST", f"/agent/prediction-scope/{scope_key}")

    def subscribe_assets(self, assets: List[str]) -> List[str]:
        """一次性订阅多个资产 scope，返回实际订阅成功的列表。"""
        ok: List[str] = []
        for code in assets:
            try:
                self.subscribe_scope(code)
                ok.append(code)
            except HAError:
                continue
        return ok

    def add_scope(self, scope_name: str) -> None:
        """手动追加 scope（如 credits:stake），需重新签发 token 后生效。"""
        self._request("POST", "/agent/scopes", payload={"add": [scope_name]})
        self.access_token(force=True)  # scope 变更必须重新签发 token

    # ------------------------------------------------------------------ 挑战 #
    def open_challenges(self) -> List[dict]:
        """全部开放中的方向性挑战（公共端点，无需鉴权即可发现）。"""
        data = self._request("GET", "/eval/challenges?status=open", public=True)
        return data if isinstance(data, list) else (data or {}).get("challenges", [])

    def active_challenges(self) -> List[dict]:
        """仅返回已订阅 scope 的开放挑战（需 challenge:read）。"""
        data = self._request("GET", "/eval/challenges/active")
        return data if isinstance(data, list) else (data or {}).get("challenges", [])

    # ------------------------------------------------------------------ 预测 #
    def submit_prediction(
        self,
        challenge_id: str,
        direction: str,
        confidence: float,
        reasoning: Optional[str] = None,
        is_revision: bool = False,
    ) -> dict:
        """提交方向性预测（需 prediction:submit）。

        每挑战开放期仅一次正式预测；修改传 is_revision=True。
        截止/结算后提交仍成功但 counts_for_score=False（仅记 paper-trade）。
        """
        payload = {
            "direction": direction,
            "confidence": float(confidence),
        }
        if reasoning:
            payload["reasoning"] = reasoning
            payload["prompt_hash"] = _sha256(reasoning)
        if is_revision:
            payload["is_revision"] = True
        return self._request(
            "POST", f"/eval/challenges/{challenge_id}/predict", payload=payload) or {}

    def challenge_results(self, challenge_id: str) -> dict:
        """查询挑战结算结果（公共端点）。"""
        return self._request(
            "GET", f"/eval/challenges/{challenge_id}/results", public=True) or {}

    # ------------------------------------------------------------------ 校准 #
    def leaderboard(self, category: Optional[str] = None) -> List[dict]:
        """全局排行榜；category 可选 commodities/equity/rates/economics/crypto。"""
        path = "/eval/leaderboard"
        if category:
            path += f"?category={category}"
        data = self._request("GET", path, public=True)
        return data if isinstance(data, list) else []

    def agent_calibration(self, agent_id: Optional[str] = None) -> dict:
        """指定 agent 的分布校准（PIT buckets / 区间覆盖率等）。"""
        aid = agent_id or self.agent_id or "self"
        return self._request("GET", f"/eval/agents/{aid}/distribution-calibration", public=True) or {}

    def public_calibration(self) -> dict:
        """全部 agent 的连续分布校准（CRPS 行，公共）。"""
        return self._request("GET", "/eval/distribution-calibration", public=True) or {}
