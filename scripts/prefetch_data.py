"""预备真实数据：联网拉取行情/行业/市值并缓存整矿，供现场离线回退。

用法：
    python scripts/prefetch_data.py            # 按 config.yaml 的 refinery 段预备
    python scripts/prefetch_data.py --cache-only-check  # 仅校验本地是否已有可用缓存

说明：
    成功后在 data/cache/ 落盘 real_ore.pkl（整矿）以及 kline_*/index_*/indcap_* 各级缓存。
    现场若担心网络，可直接把 config.yaml 的 data.cache_only / refinery.cache_only 置为 true，
    精炼厂将完全离线加载本脚本预备的数据，绝不触网。
"""

from __future__ import annotations

import os
import sys

# 让脚本可在仓库任意位置运行：将 src 加入 import 路径
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from llm.client import load_config
from pipeline.refinery import build_refinery_config, RefineryPipeline


def main() -> int:
    cfg = load_config()
    rcfg = build_refinery_config(cfg.get("refinery", {}))
    rcfg.offline = True  # 仅预备数据，不调用 LLM 矿场

    pipe = RefineryPipeline(rcfg)
    try:
        ore = pipe.prepare_real_ore()
    except Exception as e:  # noqa: BLE001
        print(f"[prefetch] 预备失败：{e}")
        print("[prefetch] 请检查网络 / akshare 可用性；或确认 data.cache_only 未误开。")
        return 1

    n_factor = len(ore.factor_pool) if ore.factor_pool else 0
    print(f"[prefetch] 预备成功：{len(ore.universe)} 只标的，"
          f"训练 {len(ore.train_kline)} 行 / 测试 {len(ore.test_kline)} 行，"
          f"因子池 {n_factor} 个")
    print(f"[prefetch] 整矿缓存已写入：{os.path.join(rcfg.cache_dir, 'real_ore.pkl')}")
    print("[prefetch] 现场防断网：将 config.yaml 的 data.cache_only 与 refinery.cache_only 置 true 即可离线运行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
