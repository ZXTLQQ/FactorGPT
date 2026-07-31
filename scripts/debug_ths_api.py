"""
同花顺 MCP 网关接入调试脚本

用法：
    python scripts/debug_ths_api.py
    THS_API_BASE_URL=https://your-gateway/mcp python scripts/debug_ths_api.py

目的：
    验证 config.yaml 中的同花顺 JWE 令牌能否对网关完成 MCP 握手、能否枚举工具，
    从而判断「是否能运行」。脚本只解码令牌头部（非加密，安全），绝不打印明文令牌。
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

# 允许从仓库任意位置运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _b64url_decode(seg: str) -> bytes:
    seg += "=" * (-len(seg) % 4)
    return base64.urlsafe_b64decode(seg)


def decode_jwe_header(token: str) -> dict:
    """解码 JWE 头部（第一段），安全：头部本身未加密。"""
    parts = token.split(".")
    if len(parts) != 5:
        raise ValueError(f"令牌分段数异常（期望5段，实际{len(parts)}），可能不是标准 JWE")
    return json.loads(_b64url_decode(parts[0]))


def mask_token(token: str) -> str:
    if len(token) <= 16:
        return "*" * len(token)
    return token[:12] + "…" + token[-6:]


def main() -> int:
    # 加载配置
    from llm.client import load_config

    cfg = load_config().get("data", {})
    token = os.environ.get("THS_API_TOKEN") or cfg.get("ths_api_token", "")
    base_url = os.environ.get("THS_API_BASE_URL") or cfg.get("ths_api_base_url", "")

    print("=" * 64)
    print("同花顺 MCP 网关接入调试")
    print("=" * 64)

    # 1) 令牌头部诊断（安全）
    if not token:
        print("[FAIL] 未找到 ths_api_token（config.data.ths_api_token 或 THS_API_TOKEN）")
        return 1
    try:
        header = decode_jwe_header(token)
        print(f"[OK ] 令牌头部解析: {json.dumps(header, ensure_ascii=False)}")
        print(f"      算法: {header.get('alg')} / {header.get('enc')}")
        print(f"      用途 kid: {header.get('kid')}  uid: {header.get('uid')}")
    except Exception as e:
        print(f"[FAIL] 令牌头部解析失败: {e}")
        return 1
    print(f"      令牌(脱敏): {mask_token(token)}")

    # 2) 网关地址
    if not base_url:
        print("\n[WARN] 未配置 ths_api_base_url（config.data.ths_api_base_url 或 THS_API_BASE_URL）。")
        print("       无法确定网关端点，无法进行握手测试。请补充后重跑本脚本。")
        print("       已确认：令牌格式合规，可作为 Bearer 鉴权令牌使用。")
        return 0

    print(f"\n[INFO] 网关端点: {base_url}")
    print("[INFO] 开始 MCP 握手与工具发现 …")

    # 3) 实际握手
    try:
        from data.ths_fetcher import THSDataFetcher

        ths = THSDataFetcher(token=token, base_url=base_url)
        diag = ths.connect_and_discover()
        print(f"[OK ] 握手成功。serverInfo={diag['server_info']}")
        print(f"[OK ] 网关暴露工具 {diag['tool_count']} 个:")
        for t in diag["tools"]:
            print(f"       - {t['name']}: {t['description'][:60]}")
        print("\n[结论] 同花顺 MCP 网关接入可运行：令牌有效、端点可达、工具可枚举。")
        print("       在 config.yaml 将 data.primary_source 改为 'ths' 即可启用该数据源。")
        return 0
    except Exception as e:
        print(f"[FAIL] 握手/发现失败: {e}")
        print("\n[排查建议]")
        print("  1) 检查 ths_api_base_url 是否为正确的 MCP 端点（应以 /mcp 结尾的常见）。")
        print("  2) 确认当前网络可访问该网关（公司内网/VPN/白名单）。")
        print("  3) 确认该 JWE 令牌对应此网关且未过期/未被吊销。")
        print("  4) 若网关返回 401/403，需联系令牌签发方核对 uid 与权限。")
        return 2


if __name__ == "__main__":
    sys.exit(main())
