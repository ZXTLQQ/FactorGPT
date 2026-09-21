"""数据源体检：逐源实测连通性，并区分「未配置」与「配置了但不可用」。

背景：数据源失败时上层一律返回空 DataFrame，界面只显示「无数据」，无法区分
「没配 token」「token 无效」「网络不通」「接口变了」。本脚本对每个源做一次
真实探测，把失败原因说清楚。

用法（工作目录须为项目根）::

    python scripts/check_data_sources.py

退出码：0=至少一个实时源可用（或 offline 数据就绪）；1=全部不可用。
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# 与运行时一致：默认强制直连，规避 Windows 不可达系统代理导致的 ProxyError
import netutil  # noqa: E402

netutil.apply_proxy_settings(None)
netutil.patch_requests_session()

import requests  # noqa: E402

OK, FAIL, SKIP = "OK  ", "FAIL", "--  "

_PROBE_SYMBOL = "600519"
_START, _END = "20260801", "20260921"

# Windows 控制台默认非 UTF-8（多为 cp936），不按控制台代码页输出中文会显示为乱码。
if sys.platform == "win32":  # pragma: no cover - 平台相关
    try:
        import ctypes

        sys.stdout.reconfigure(encoding=f"cp{ctypes.windll.kernel32.GetConsoleOutputCP()}",
                               errors="replace")
    except Exception:  # noqa: BLE001
        pass


def _configured(val) -> str:
    """配置值转为「可用真值」：${VAR} 未解析的占位符一律视为未配置。"""
    s = str(val or "").strip()
    return "" if s.startswith("${") else s


def _cfg() -> dict:
    from llm.client import load_config

    return load_config() or {}


def check_offline() -> Tuple[str, str]:
    import glob

    import pandas as pd

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = sorted(glob.glob(os.path.join(root, "data", "offline", "*.parquet")))
    if not files:
        return FAIL, "data/offline/ 下无 parquet"
    rows = 0
    for f in files:
        try:
            rows += len(pd.read_parquet(f))
        except Exception as e:  # noqa: BLE001
            return FAIL, f"{os.path.basename(f)} 读取失败：{e}"
    return OK, f"{len(files)} 个文件 / {rows} 行"


def check_eastmoney() -> Tuple[str, str]:
    import akshare as ak

    t = time.time()
    df = ak.stock_zh_a_hist(symbol=_PROBE_SYMBOL, period="daily",
                            start_date=_START, end_date=_END, adjust="qfq")
    return OK, f"{len(df)} 行 ({time.time() - t:.1f}s)"


def check_sina() -> Tuple[str, str]:
    import akshare as ak

    t = time.time()
    df = ak.stock_zh_a_daily(symbol="sh" + _PROBE_SYMBOL,
                             start_date=_START, end_date=_END, adjust="qfq")
    return OK, f"{len(df)} 行 ({time.time() - t:.1f}s)"


def check_tushare(token: str, base_url: str = "", fallback_url: str = "") -> Tuple[str, str]:
    if not token:
        return SKIP, ("未配置：config.yaml 为占位符且 .env/环境变量无 TUSHARE_TOKEN。"
                      "在 UI「数据源」面板填写并保存，或写入 .env 的 TUSHARE_TOKEN")
    t = time.time()
    try:
        urls = [u for u in (base_url, fallback_url) if u]
        if urls:  # 第三方中转网关：GET {base}/{api}，X-API-Key 鉴权
            from data.fetcher import _TushareGatewayClient

            client = _TushareGatewayClient(urls, token, timeout=20)
            df = client.daily(ts_code=f"{_PROBE_SYMBOL}.SH", start_date="20260901", end_date=_END)
        else:  # 官方 SDK
            import tushare as ts

            df = ts.pro_api(token).daily(ts_code=f"{_PROBE_SYMBOL}.SH",
                                         start_date="20260901", end_date=_END)
    except Exception as e:  # noqa: BLE001
        msg = str(e)[:110]
        if "token" in msg or "40101" in msg or "权限" in msg:
            return FAIL, f"{msg}（token 无效、积分不足，或网关不认该 key）"
        return FAIL, f"{msg}（网络/接口异常，已依次尝试主备网关）"
    # 顺带报最新交易日：中转网关的数据常有 1~3 个交易日滞后，若落后于其它源
    # 应改用其它源取末端数据，否则回测区间末尾会静默缺几根 K 线。
    latest = df["trade_date"].max() if len(df) and "trade_date" in df.columns else "?"
    return OK, f"{len(df)} 行，最新 {latest} ({time.time() - t:.1f}s)"


def check_ths(token: str, base_url: str) -> Tuple[str, str]:
    if not token or not base_url:
        return SKIP, "未配置：需同时填写 ths_api_token 与 ths_api_base_url"
    try:
        r = requests.post(base_url, json={"jsonrpc": "2.0", "id": 1,
                                          "method": "tools/list", "params": {}},
                          headers={"Authorization": f"Bearer {token}"}, timeout=15)
    except Exception as e:  # noqa: BLE001
        return FAIL, f"{type(e).__name__}: {str(e)[:80]}"
    if r.status_code != 200 or "error" in r.text.lower():
        return FAIL, f"HTTP {r.status_code} {r.text[:80]}"
    return OK, f"HTTP 200 ({len(r.text)} 字节)"


def check_neodata(base_url: str) -> Tuple[str, str]:
    import os as _os

    token = _os.environ.get("NEODATA_TOKEN") or ""
    if not token:
        p = os.path.expanduser(os.path.join("~", ".workbuddy", ".neodata_token"))
        if os.path.exists(p):
            try:
                token = open(p, encoding="utf-8").read().strip()
            except Exception:  # noqa: BLE001
                token = ""
    if not token:
        return SKIP, "未配置：无 NEODATA_TOKEN，且无 IDE 会话令牌 ~/.workbuddy/.neodata_token"
    try:
        r = requests.post(base_url or "https://copilot.tencent.com/agenttool/v1/neodata",
                          json={"query": "贵州茅台最新收盘价",
                                "channel": "neodata", "sub_channel": "workbuddy"},
                          headers={"Authorization": f"Bearer {token}"}, timeout=20)
    except Exception as e:  # noqa: BLE001
        return FAIL, f"{type(e).__name__}: {str(e)[:80]}"
    if r.status_code != 200:
        return FAIL, f"HTTP {r.status_code} {r.text[:80]}（令牌约 12 小时过期）"
    return OK, f"HTTP 200 ({len(r.text)} 字节)"


def main() -> int:
    cfg = _cfg()
    data_cfg = cfg.get("data", {}) or {}
    token = _configured(data_cfg.get("tushare_token"))

    print("数据源体检（探测标的 600519）")
    print("-" * 72)
    print(f"data.source      = {data_cfg.get('source', 'legacy')}")
    print(f"tushare 网关     = {_configured(data_cfg.get('tushare_base_url')) or '官方 api.tushare.pro'}"
          f"  备用 = {_configured(data_cfg.get('tushare_fallback_url')) or '无'}"
          f"  token = {'已配置' if token else '未配置'}")
    print(f"data.primary_source = {data_cfg.get('primary_source')}"
          f"    prefer_sina = {data_cfg.get('prefer_sina')}")
    print("-" * 72)

    rows: List[Tuple[str, str, str]] = []

    def run(name: str, fn):
        t = time.time()
        try:
            st, detail = fn()
        except Exception as e:  # noqa: BLE001
            st, detail = FAIL, f"{type(e).__name__}: {str(e)[:110]}"
        rows.append((name, st, f"{detail} ({time.time() - t:.1f}s)"))

    run("offline    本地 parquet", check_offline)
    run("akshare    东方财富", check_eastmoney)
    run("akshare    新浪", check_sina)
    run("tushare    Tushare Pro", lambda: check_tushare(
        token, _configured(data_cfg.get("tushare_base_url")),
        _configured(data_cfg.get("tushare_fallback_url"))))
    run("ths        同花顺 iFinD", lambda: check_ths(
        _configured(data_cfg.get("ths_api_token")), data_cfg.get("ths_api_base_url") or ""))
    run("neodata    NeoData 网关", lambda: check_neodata(
        (data_cfg.get("neodata") or {}).get("base_url") or ""))

    for name, st, detail in rows:
        print(f"[{st}] {name:26s} {detail}")

    ok_real = [r for r in rows if r[1] == OK and not r[0].startswith("offline")]
    print("-" * 72)
    src = str(data_cfg.get("source", "legacy")).lower()
    if src == "offline":
        print("当前 data.source=offline：全链路不触网，以上实时源仅供切换 legacy 后参考。")
        return 0 if rows[0][1] == OK else 1
    print(f"可用实时源 {len(ok_real)} 个。" if ok_real else
          "无可用实时源：请按上方各源的失败原因处理，或将 data.source 置为 offline 用本地数据。")
    return 0 if ok_real else 1


if __name__ == "__main__":
    raise SystemExit(main())
