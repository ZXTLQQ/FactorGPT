#!/usr/bin/env python3
"""ha_forward_run — FactorGPT × Headline Arena 前瞻检验统一入口。

把因子层的宏观观点（利率、风格、商品方向）转成每日概率预测提交给 Headline Arena，
获得一条与回测（IC 历史检验）完全独立的 forward 检验线：预测在结果出现之前锁定，
结算标准在出题时冻结，由第三方按真实行情机械结算（方向 50±50c、宏观 CRPS）。

API 事实来源：https://headlinearena.com/api/v1/agent/onboarding/guide.txt

用法:
  # 1) 首次使用：注册 agent（响应含一次性 client_secret，自动保存；随后人工认领 claim_url）
  python scripts/ha_forward_run.py register --name FactorGPT-GoldBot --bio "黄金宏观因子前瞻检验" \
      --model-provider DeepSeek --model-name deepseek-chat

  # 2) 每日前瞻循环（默认 dry_run：只写本地账本，预测在本地锁定；离线/无凭据也可运行）
  python scripts/ha_forward_run.py run --theme "避险升温，降息预期增强，美元走弱"
  #    指定资产 / 从因子描述推导观点：
  python scripts/ha_forward_run.py run --theme "油价上行，供给收紧" --assets GC,CL,ES,ZN
  python scripts/ha_forward_run.py run --factor-report output/methodology_report.json
  python scripts/ha_forward_run.py run --view-file views.json

  # 3) 真实提交（需凭据 + 已认领；默认自动订阅资产 scope）
  python scripts/ha_forward_run.py run --live --theme "..." --assets GC,CL

  # 4) 结算回填（拉取第三方结算结果写入账本 settled 子块）
  python scripts/ha_forward_run.py settle

  # 5) 前瞻评分卡（准确率 / Brier / 校准曲线，Markdown）
  python scripts/ha_forward_run.py scorecard --out data/forwardtest/scorecard.md

  # 6) 账本状态
  python scripts/ha_forward_run.py status

凭据：.env 的 HA_AGENT_ID / HA_CLIENT_SECRET（或 ~/.headlinearena/credentials.json）。
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_env_file(path: pathlib.Path) -> None:
    """极简 .env 解析（仅 KEY=VALUE 行，忽略注释；不覆盖已有环境变量）。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_project_config() -> dict:
    """加载 config.yaml 的 headline_arena 段（轻量 yaml 解析，yaml 缺失时返回默认）。"""
    import yaml  # 项目自带依赖（CI 亦安装）

    path = PROJECT_ROOT / "config.yaml"
    try:
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, ValueError):
        cfg = {}
    ha = cfg.get("headline_arena") or {}
    return dict(ha) if isinstance(ha, dict) else {}


def _stdout_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def cmd_register(args) -> int:
    from forwardtest.client import HeadlineArenaClient

    client = HeadlineArenaClient()
    print("正在向 Headline Arena 注册 agent（client_secret 将自动保存，仅显示一次）...")
    resp = client.register(
        name=args.name, bio=args.bio,
        model_provider=args.model_provider, model_name=args.model_name,
        model_version=args.model_version, operator_contact=args.operator_contact,
        requested_scopes=(args.scopes.split(",") if args.scopes else None),
    )
    agent_id = resp.get("agent_id", "")
    challenge_id = resp.get("challenge_id", "")
    print("=" * 64)
    print("注册成功。请立即保存以下信息：")
    print(f"  agent_id       : {agent_id}")
    print(f"  challenge_id   : {challenge_id}")
    print("  claim_url      : 见下方 resp.claim_url（只给 operator，agent 自身不得访问）")
    print("=" * 64)
    print("注册挑战题（市场事件分析）与认领指引已随响应返回；")
    print("client_secret 已写入 ~/.headlinearena/credentials.json。")
    print()
    print("认领（claim）由人工完成：打开 claim_url 输入 pairing_code 即可。")
    print("临时状态默认 10 条预测上限，认领后进入正式计分与排行榜。")
    return 0


def cmd_run(args) -> int:
    from forwardtest import runner

    ha_cfg = load_project_config()
    if args.config_file:
        ha_cfg = dict(ha_cfg, **load_project_config())  # keep simple
    view_file_views = None
    factor_desc = factor_name = factor_metrics = None
    theme = args.theme
    if args.view_file:
        view_file_views = runner.load_views_file(args.view_file)
    if args.factor_report:
        import json

        with open(args.factor_report, encoding="utf-8") as f:
            rep = json.load(f)
        factor_desc = (rep.get("description") or rep.get("factor_description")
                       or rep.get("requirement") or "")
        factor_name = rep.get("name") or rep.get("factor_name") or ""
        m = rep.get("metrics") or {}
        if isinstance(m, dict):
            factor_metrics = {k: v for k, v in m.items()
                              if not isinstance(v, (dict, list))}
    assets = None
    if args.assets:
        assets = [a.strip().upper() for a in args.assets.split(",") if a.strip()]

    if args.reasoning_prefix is not None:
        reasoning_prefix = args.reasoning_prefix
    elif args.model_provider:
        reasoning_prefix = f"FactorGPT(model={args.model_provider}/{args.model_name or '?'})"
    else:
        reasoning_prefix = ""
    summary = runner.run_forward(
        ha_cfg, theme=theme, factor_desc=factor_desc, factor_name=factor_name,
        factor_metrics=factor_metrics, views=view_file_views,
        assets=assets, live=args.live, reasoning_prefix=reasoning_prefix,
    )
    print("=" * 64)
    print(f"Headline Arena 前瞻检验 · {'真实提交' if summary['mode']=='live' else 'dry-run 影子'} 模式")
    print("=" * 64)
    for w in summary["warnings"]:
        print(f"[警告] {w}")
    if summary.get("reason") == "no_signal":
        print("未识别到宏观方向信号：本轮未生成预测（无信号不制造噪声）。")
        print("可用 --theme \"利率下行、黄金看多、油价上行...\" 显式给出观点。")
        return 0
    print(f"提交预测 {summary['submitted']} 条"
          f"（live={summary.get('live', 0)} / dry_run={summary['dry_run']}）")
    for view in summary["views"]:
        print(f"  {view['asset']:<5} {view['direction']:<8} "
              f"conf={view['confidence']:.2f}  {view.get('reasoning', '')[:60]}")
    print(f"账本：{summary['ledger_path']}")
    return 0


def cmd_settle(args) -> int:
    from forwardtest import runner

    summary = runner.settle_ledger(load_project_config(), limit=args.limit)
    print("结算回填完成：settled=%d failed=%d skipped=%d" % (
        summary["settled"], summary["failed"], summary["skipped"]))
    for n in summary.get("notes", [])[:10]:
        print(f"  {n}")
    return 0


def cmd_scorecard(args) -> int:
    from forwardtest import runner

    md = runner.make_scorecard(load_project_config())
    out = args.out or (PROJECT_ROOT / "data" / "forwardtest" / "scorecard.md")
    path = pathlib.Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[评分卡已写入] {path}")
    return 0


def cmd_status(args) -> int:
    from forwardtest import runner

    ledger = runner.resolve_ledger(load_project_config())
    st = ledger.stats()
    print("Headline Arena 前瞻检验账本状态：")
    print(f"  账本路径 : {st['path']}")
    print(f"  总记录   : {st['total']}（pending={st['pending']} / settled={st['settled']}）")
    print(f"  按模式   : {st['by_mode']}")
    print(f"  涉及资产 : {', '.join(st['assets']) or '-'}")
    creds = runner.has_credentials()
    print(f"  凭据     : {'已配置 (HA_AGENT_ID / HA_CLIENT_SECRET)' if creds else '未配置（只能 dry_run）'}")
    return 0


def main() -> int:
    _stdout_utf8()
    load_env_file(PROJECT_ROOT / ".env")
    # 与 run_agent.py 一致：把 src 加入 Python 路径
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    parser = argparse.ArgumentParser(
        description="FactorGPT × Headline Arena 前瞻检验（forward testing）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python scripts/ha_forward_run.py run --theme \"避险升温，降息预期增强\"\n"
               "  python scripts/ha_forward_run.py run --live --theme \"油价上行\" --assets GC,CL\n"
               "  python scripts/ha_forward_run.py settle\n"
               "  python scripts/ha_forward_run.py scorecard\n"
               "  python scripts/ha_forward_run.py status")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_reg = sub.add_parser("register", help="注册 HA agent（需人工认领）")
    p_reg.add_argument("--name", required=True)
    p_reg.add_argument("--bio", required=True)
    p_reg.add_argument("--model-provider", required=True, help="真实模型厂商（平台要求如实上报）")
    p_reg.add_argument("--model-name", required=True, help="真实模型名")
    p_reg.add_argument("--model-version", default=None)
    p_reg.add_argument("--operator-contact", default=None, help="邮箱，用于一键认领")
    p_reg.add_argument("--scopes", default=None, help="逗号分隔；缺省=自动授予全部")
    p_reg.set_defaults(func=cmd_register)

    p_run = sub.add_parser("run", help="每日前瞻循环：观点->挑战->预测账本（默认 dry_run）")
    p_run.add_argument("--theme", default=None, help="宏观主题文本（关键词自动翻译）")
    p_run.add_argument("--factor-report", default=None,
                       help="因子方法学报告 JSON（自动折叠成主题，命中宏观措辞才提交）")
    p_run.add_argument("--view-file", default=None, help="外部 view JSON")
    p_run.add_argument("--assets", default=None, help="资产子集，逗号分隔，如 GC,CL,ES,ZN")
    p_run.add_argument("--live", action="store_true", help="真实提交（需凭据+认领）")
    p_run.add_argument("--model-provider", default=None)
    p_run.add_argument("--model-name", default=None)
    p_run.add_argument("--reasoning-prefix", default=None,
                       help="附加到每条 reasoning 前缀；默认带模型指纹")
    p_run.add_argument("--config-file", default=None, help="（保留位）")
    p_run.set_defaults(func=cmd_run)

    p_set = sub.add_parser("settle", help="结算回填：拉取第三方结算结果写入账本")
    p_set.add_argument("--limit", type=int, default=None, help="只处理前 N 条待结算记录")
    p_set.set_defaults(func=cmd_settle)

    p_sc = sub.add_parser("scorecard", help="生成前瞻评分卡 Markdown")
    p_sc.add_argument("--out", default=None, help="输出路径（默认 data/forwardtest/scorecard.md）")
    p_sc.set_defaults(func=cmd_scorecard)

    p_st = sub.add_parser("status", help="账本状态")
    p_st.set_defaults(func=cmd_status)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
