"""现场答辩一键自检 (Preflight Check)：确认断网环境下仍能完整演示。

用法：
    python scripts/preflight_check.py            # 只体检，不改配置
    python scripts/preflight_check.py --offline  # 体检并把 config.yaml 切到离线演示档位

体检项（对应「现场演示防翻车」三件事）：
    1. 计算依赖：torch / stable-baselines3 / sb3-contrib 是否就绪
       —— 决定 PART-02 是跑真 MaskablePPO 还是降级启发式；
    2. 本地大模型：Ollama 端点可达性 + config.yaml 指定模型是否已拉取
       —— 决定断网时 FactorAgent 能否继续生成因子；
    3. 离线数据：data/cache 下整矿与各级缓存是否齐备
       —— 决定 cache_only=true 时精炼厂能否零联网跑通；
    4. 检索依赖：ChromaDB / BGE 向量模型 / 已学习因子库；
    5. 沙箱与追踪：子进程沙箱开关、实验追踪目录。

退出码：0 = 全部通过或仅有可接受降级；1 = 存在会导致演示中断的阻塞项。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

# Windows 控制台默认 GBK，重定向到文件时中文会乱码；统一按 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

OK, WARN, FAIL = "PASS", "WARN", "FAIL"
_ICON = {OK: "[PASS]", WARN: "[WARN]", FAIL: "[FAIL]"}

results: list[tuple[str, str, str]] = []


def record(item: str, status: str, detail: str) -> None:
    results.append((item, status, detail))
    print(f"{_ICON[status]} {item}：{detail}")


def _ver(mod_name: str):
    try:
        mod = __import__(mod_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return None


# ── 1. 计算依赖与 RL 后端 ────────────────────────────────────────────── #
def check_compute() -> None:
    torch_v = _ver("torch")
    sb3_v = _ver("stable_baselines3")
    contrib_v = _ver("sb3_contrib")
    if torch_v and sb3_v and contrib_v:
        record("RL/深度学习依赖", OK,
               f"torch {torch_v} / sb3 {sb3_v} / sb3-contrib {contrib_v}"
               "，PART-02 将运行真实 MaskablePPO 与 Transformer 编码器")
    elif torch_v:
        record("RL/深度学习依赖", WARN,
               f"torch {torch_v} 可用，但 sb3/sb3-contrib 缺失，"
               "PART-02 将降级为启发式集束搜索（功能等价，演示时需主动说明）")
    else:
        record("RL/深度学习依赖", WARN,
               "torch 未安装，Transformer 表征将降级为 numpy 随机投影、"
               "RL 降级为启发式集束搜索；流程仍可跑通，但需在答辩中说明降级")

    if _ver("sklearn"):
        record("LASSO 筛选依赖", OK, f"scikit-learn {_ver('sklearn')}，PART-04 第一级走真实 LassoCV")
    else:
        record("LASSO 筛选依赖", WARN, "scikit-learn 缺失，第一级降级为相关性冗余去重")


# ── 2. 本地大模型（断网保命）────────────────────────────────────────── #
def check_llm(cfg: dict) -> None:
    llm = cfg.get("llm", {}) or {}
    provider = str(llm.get("provider", "")).lower()
    base_url = str(llm.get("base_url", "") or "")
    model = str(llm.get("model", "") or "")

    if provider != "ollama":
        record("LLM 提供方", WARN,
               f"当前 provider={provider or '未配置'}（依赖外网）。"
               "现场建议切到 provider=ollama + base_url=http://localhost:11434/v1 以防断网")
        return

    tags_url = base_url.replace("/v1", "").rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=5) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
        names = [m.get("name", "") for m in payload.get("models", [])]
    except (urllib.error.URLError, OSError, ValueError) as e:
        record("本地 Ollama 服务", FAIL,
               f"{tags_url} 不可达（{e}）。请先运行 `ollama serve`，"
               "否则断网时因子生成会失败")
        return

    record("本地 Ollama 服务", OK, f"{tags_url} 可达，已拉取 {len(names)} 个模型")
    if model in names:
        record("Ollama 目标模型", OK, f"{model} 已就绪，断网可离线生成因子")
    else:
        record("Ollama 目标模型", FAIL,
               f"config.yaml 指定的 {model} 不在本地模型列表 {names}，"
               f"请执行 `ollama pull {model}`")


# ── 3. 离线数据缓存 ─────────────────────────────────────────────────── #
def check_cache(cfg: dict) -> None:
    cache_dir = os.path.join(ROOT, (cfg.get("refinery", {}) or {}).get("cache_dir", "data/cache"))
    if not os.path.isdir(cache_dir):
        record("行情缓存目录", FAIL, f"{cache_dir} 不存在，请先运行 python scripts/prefetch_data.py")
        return

    files = os.listdir(cache_dir)
    ore_path = os.path.join(cache_dir, "real_ore.pkl")
    if os.path.exists(ore_path):
        size_mb = os.path.getsize(ore_path) / 1024 / 1024
        record("整矿缓存 real_ore.pkl", OK,
               f"{size_mb:.1f} MB，cache_only=true 时精炼厂可零联网加载")
    else:
        record("整矿缓存 real_ore.pkl", FAIL,
               "缺失。请运行 python scripts/prefetch_data.py 预备真实行情整矿")

    n_kline = len([f for f in files if f.startswith("kline_")])
    n_index = len([f for f in files if f.startswith("index_")])
    status = OK if n_kline else WARN
    record("分级缓存", status, f"kline 缓存 {n_kline} 份 / 指数成分缓存 {n_index} 份")

    data_only = bool((cfg.get("data", {}) or {}).get("cache_only"))
    ref_only = bool((cfg.get("refinery", {}) or {}).get("cache_only"))
    if data_only and ref_only:
        record("离线开关", OK, "data.cache_only 与 refinery.cache_only 均为 true，当前为完全离线档位")
    else:
        record("离线开关", WARN,
               f"data.cache_only={data_only} / refinery.cache_only={ref_only}。"
               "现场断网前请置为 true（或运行本脚本加 --offline 自动切换）")


# ── 4. 检索依赖 ─────────────────────────────────────────────────────── #
def check_rag(cfg: dict) -> None:
    rag = cfg.get("rag", {}) or {}
    if _ver("chromadb") and _ver("sentence_transformers"):
        record("向量检索依赖", OK, f"chromadb {_ver('chromadb')} + sentence-transformers 就绪")
    else:
        record("向量检索依赖", WARN, "缺少 chromadb/sentence-transformers，RAG 将降级为 jieba+TF-IDF 关键词检索")

    persist = os.path.join(ROOT, str(rag.get("chroma_persist_dir", "./chroma_db")).lstrip("./"))
    if os.path.isdir(persist) and os.listdir(persist):
        record("向量库持久化目录", OK, f"{persist} 已建库，断网可直接检索")
    else:
        record("向量库持久化目录", WARN, f"{persist} 为空，首次使用需联网下载 BGE 模型并建库")

    lib = os.path.join(ROOT, str(rag.get("learned_library_path", "data/learned_factors.jsonl")))
    if os.path.exists(lib):
        with open(lib, "r", encoding="utf-8") as f:
            n = sum(1 for line in f if line.strip())
        record("已学习因子库", OK, f"{n} 条记录可供 RAG 检索复用")
    else:
        record("已学习因子库", WARN, f"{lib} 不存在，冷启动检索命中率会下降")


# ── 5. 沙箱与追踪 ───────────────────────────────────────────────────── #
def check_runtime(cfg: dict) -> None:
    sb = ((cfg.get("engine", {}) or {}).get("sandbox", {}) or {})
    if sb.get("subprocess"):
        record("代码执行沙箱", OK, f"子进程隔离开启，超时 {sb.get('timeout', 30)}s，防死循环卡死界面")
    else:
        record("代码执行沙箱", WARN, "未开启子进程沙箱，LLM 生成的死循环代码可能卡死演示界面")

    kr = cfg.get("kronos", {}) or {}
    if kr.get("enabled"):
        kdir = os.path.join(ROOT, str(kr.get("cache_dir", "./models/kronos")).lstrip("./"))
        if os.path.isdir(kdir) and os.listdir(kdir):
            record("Kronos 权重", OK, f"{kdir} 已缓存，断网可加载")
        elif kr.get("fallback_to_stub"):
            record("Kronos 权重", WARN, "未缓存，但 fallback_to_stub=true，断网会降级 stub 且不中断")
        else:
            record("Kronos 权重", FAIL,
                   f"kronos.enabled=true 且 fallback_to_stub=false，但 {kdir} 无权重，"
                   "断网时会抛错中断；请先联网预热或将 fallback_to_stub 置 true")
    else:
        record("Kronos 权重", OK, "kronos.enabled=false，不参与演示链路")


# ── 一键切离线档位 ──────────────────────────────────────────────────── #
def switch_offline() -> None:
    path = os.path.join(ROOT, "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    replaced = 0
    out_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("cache_only:") and "true" not in stripped:
            indent = line[: len(line) - len(line.lstrip())]
            comment = line.split("#", 1)[1] if "#" in line else ""
            out_lines.append(f"{indent}cache_only: true" + (f"  #{comment}" if comment else ""))
            replaced += 1
        else:
            out_lines.append(line)
    if replaced:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
    print(f"\n[offline] 已将 {replaced} 处 cache_only 置为 true（config.yaml）")


def main() -> int:
    parser = argparse.ArgumentParser(description="FactorGPT 现场答辩一键自检")
    parser.add_argument("--offline", action="store_true", help="体检后把 config.yaml 切到离线演示档位")
    args = parser.parse_args()

    from llm.client import load_config

    cfg = load_config() or {}
    print("=" * 68)
    print("FactorGPT 现场答辩自检 (Preflight Check)")
    print("=" * 68)
    check_compute()
    check_llm(cfg)
    check_cache(cfg)
    check_rag(cfg)
    check_runtime(cfg)

    if args.offline:
        switch_offline()

    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_warn = sum(1 for _, s, _ in results if s == WARN)
    print("-" * 68)
    print(f"合计 {len(results)} 项：通过 {len(results) - n_fail - n_warn}，"
          f"降级告警 {n_warn}，阻塞 {n_fail}")
    if n_fail:
        print("存在阻塞项，现场断网可能中断演示，请按上方提示处理后重跑本脚本。")
    else:
        print("无阻塞项：断网环境下可完整演示「因子挖掘 Agent + 因子精炼厂」闭环。")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
