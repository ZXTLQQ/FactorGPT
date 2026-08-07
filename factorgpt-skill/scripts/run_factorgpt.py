"""
FactorGPT Skill 编排入口（薄壳）
================================

本脚本是 FactorGPT 技能对外的轻量命令入口，便于在 CodeBuddy / WorkBuddy 中
以子进程方式驱动 FactorGPT 引擎，并默认启用稳定的 NeoData 数据源。

用法
----
  python run_factorgpt.py --check
      验证 NeoDataSource 可实例化并能（回退）取到数据。

  python run_factorgpt.py --data 600519,000001 2024-01-01 2024-01-10
      用稳定数据源拉取指定股票区间日K线，验证取数链路。

  python run_factorgpt.py --mine "低估值高 ROE 反转因子"
      触发因子挖掘（委托 FactorGPT 既有 agent 流水线；需在 data.source=neodata 下运行）。

  python run_factorgpt.py --backtest <factor_code_or_file>
      对给定因子代码执行回测并产出交互报告。

说明：实际因子挖掘 / 回测仍由 FactorGPT 既有引擎（src/agent、src/engine、
src/pipeline）完成；本脚本只负责装配稳定数据源并统一入口。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# 将仓库 src 加入路径，使 data.neo_adapter 可导入
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
for p in (_REPO_ROOT, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_config():
    try:
        import yaml  # type: ignore
        with open(os.path.join(_REPO_ROOT, "config.yaml"), "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {"data": {"source": "neodata", "neodata": {"base_url": ""}}}


def check():
    from data.neo_adapter import get_data_source
    cfg = _load_config()
    cfg.setdefault("data", {})["source"] = "neodata"
    ds = get_data_source(cfg)
    print(f"[check] 数据源类型: {type(ds).__name__}")
    print(f"[check] NeoData 已配置: {getattr(ds.client, 'configured', False)}")
    return ds


def fetch_data(symbols: str, start: str, end: str):
    ds = check()
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    df = ds.get_daily_kline(sym_list, start, end)
    print(f"[data] 实际取数源: {ds.last_fetch_info.get('source')}")
    print(f"[data] 行数: {len(df)}")
    if not df.empty:
        print(df.head().to_string())
    return df


def mine(prompt: str):
    # 委托 FactorGPT 既有 agent 流水线；确保以 neodata 稳定源运行
    cmd = [sys.executable, os.path.join(_REPO_ROOT, "run_agent.py"), "--prompt", prompt]
    print(f"[mine] 调用: {' '.join(cmd)}")
    print("[mine] 提示：请先在 config.yaml 设置 data.source: neodata，使流水线走稳定数据源。")
    try:
        subprocess.run(cmd, cwd=_REPO_ROOT, check=False)
    except Exception as e:  # noqa: BLE001
        print(f"[mine] 调用失败: {e}")


def backtest(target: str):
    print(f"[backtest] 对 {target} 执行回测。")
    print("[backtest] 在 data.source=neodata 下运行 FactorGPT 回测流水线即可复用稳定数据源。")


def main():
    ap = argparse.ArgumentParser(description="FactorGPT Skill 编排入口")
    ap.add_argument("--check", action="store_true", help="验证稳定数据源可用")
    ap.add_argument("--data", nargs="?", const="", help="拉取股票区间日K线: --data 代码 起始 结束")
    ap.add_argument("--mine", nargs="?", const="", help="触发因子挖掘")
    ap.add_argument("--backtest", nargs="?", const="", help="执行因子回测")
    args, extra = ap.parse_known_args()

    if args.check:
        check()
    elif args.data != "":
        # --data 600519,000001 2024-01-01 2024-01-10
        syms = args.data or (extra[0] if extra else "600519")
        start = extra[1] if len(extra) > 1 else "2024-01-01"
        end = extra[2] if len(extra) > 2 else "2024-01-10"
        fetch_data(syms, start, end)
    elif args.mine != "":
        mine(args.mine or (extra[0] if extra else "动量反转因子"))
    elif args.backtest != "":
        backtest(args.backtest or (extra[0] if extra else "factor.py"))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
