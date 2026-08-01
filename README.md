# FactorGPT — LLM 金融因子挖掘 Agent

浙江省大学生金融创新大赛（A类：金融创新产品设计）参赛项目。

## 项目简介

FactorGPT 是一个基于大语言模型（LLM）的智能金融因子挖掘 Agent，将自然语言理解与量化金融因子工程深度融合，支持从非结构化数据中自动提取、验证与组合优化金融因子。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 Web 界面（含「🤖 因子挖掘 (Agent)」页面）
streamlit run src/ui/app.py

# 或命令行无界面运行因子挖掘 Agent（离线可用，自动回退到合成数据）
python run_agent.py "请构建一个 20 日动量因子"

# 运行「六阶段因子精炼厂」流水线（离线演示，无需 API/网络）
python run_agent.py --refinery "混合日频与月频，结合短期反转与流动性"
# 接入 LLM 矿场（需配置 API Key 与行情数据）：
python run_agent.py --refinery --no-offline "混合日频与月频，结合短期反转与流动性"
```

## 因子挖掘 Agent 工作流

Agent 基于 LangGraph 编排，形成「检索 → 生成 → 校验 → 评价 → 反思」闭环：

1. **知识检索**：根据用户需求，从因子知识库（ChromaDB+BGE 或 jieba 关键词）检索动量、反转、质量、波动率等经典因子文献作为 LLM 上下文。
2. **因子生成**：LLM 依据需求与知识，生成遵守安全契约的 `alpha_factor(df) -> DataFrame[date,symbol,factor]` 因子代码。若 LLM 不可用，自动回退到内置关键词模板因子。
3. **沙箱校验与计算**：在受限执行沙箱中安全运行代码（白名单导入、禁用危险内置、强制 `shift(1)` 防前视），计算因子值。
4. **因子后处理**：截面缩尾（Winsorize）→ 行业/市值中性化 → 标准化（复用 `DataCleaner`）。
5. **回测评价**：计算 IC、RankIC、ICIR、IC 为正比例、分位数收益、多空对冲收益/夏普/最大回撤、换手率、覆盖率。
6. **反思改进**：若 |IC| 未达阈值，LLM 结合回测指标反思并改进因子定义，循环直至达标或达到最大轮数；最终输出因子代码、指标与报告。

## 因子精炼厂（六阶段冶炼流水线）

在单因子 Agent 之上，系统以「工业冶炼」为隐喻，构建端到端闭环的因子生产线，把因子开发抽象为从「矿石开采」到「成品交付」的六道工序：

| 阶段 | 工序 | 核心组件 | 作用 |
|------|------|----------|------|
| PART-01 | 矿石原料仓（数据底座） | `FeatureForge` | 28 分钟级原始特征 + 50+ 时序/截面因子池 + 9 行业/3 风格维度；多进程并行构建 |
| PART-02 | 采矿作业层（三维生成） | `TransformerEncoder` + `FactorRLSearch` + LLM 矿场 | Transformer(d_model=128,2层,5头) 向量化表征；MaskablePPO 动作屏蔽式因子组合搜索；LLM 矿脉探索 |
| PART-03 | 研磨车间（RPN 引擎） | `RPNEngine` | Rank IC / IR / ICIR 量化有效性度量 + 稳定性评估 + 多进程并行批量求值 |
| PART-04 | 三级筛选（浮选） | `Screener` | 第一级 LASSO 去冗余 → 第二级 人机协同 → 第三级 TOP 10% 截断 |
| PART-05 | 合金配比（AlphaPool） | `AlphaPool` | ICIR 加权 + 正交化合成 + leave-one-out 过拟合检验 + 迭代优化 |
| PART-06 | 提交（方法学总结） | `MethodologyReport` | 自动产出方法学报告（构建逻辑/参数依据/交叉验证），一键导出 MD + JSON |

设计要点（产品壁垒）：
- **可插拔式评估**：`RPNEngine` 暴露指标注册表（`register_metric`），评估指标可灵活扩展。
- **动作屏蔽 RL**：`FactorEnv` 内置 `action_masks()`，屏蔽已选/超长动作避免错误探索；后端可选 `MaskablePPO`（需 `sb3_contrib`）或自动降级「动作屏蔽 + 集束搜索」启发式。
- **强鲁棒性**：`TransformerEncoder` / RL / LASSO 均在依赖缺失时自动降级（numpy 投影 / 集束搜索 / 相关性去重），保证任意环境可跑通。
- **训练/测试隔离**：测试集仅用于最终方法学报告，不参与生成与筛选，杜绝前视泄露。
- **一键复现**：方法学报告连同 JSON 原子产物导出，供审计与复现。

流水线入口：`run_agent.py --refinery`（或 Streamlit「🏭 因子精炼厂」页面）；核心编排见 `src/pipeline/refinery.py`。

## 项目结构

- `src/agent/graph.py` — `FactorAgent`：LangGraph 状态图编排（检索/生成/校验/评价/反思/终局）
- `src/agent/nodes.py` — 各节点实现（知识检索、代码生成、沙箱校验、回测评价、反思、报告）
- `src/agent/state.py` — Agent 状态定义
- `src/engine/factor_builder.py` — 受限代码沙箱、因子后处理流水线、关键词模板因子
- `src/engine/backtest.py` — 因子回测与评价引擎（IC/RankIC/分位数/多空）
- `src/engine/optimizer.py` — 多因子合成（IC 加权、正交化、筛选）
- `src/data/` — 数据获取（`DataFetcher`）与清洗（`DataCleaner`）
- `src/rag/` — 因子知识库（`paper_index` 知识语料 + `retriever` 检索，含 jieba 兜底）
- `src/llm/` — LLM 客户端封装（DeepSeek/OpenAI 兼容）
- `src/ui/app.py` — Streamlit 前端（含因子挖掘 Agent 页面）
- `run_agent.py` — 命令行演示入口（单因子 Agent / 六阶段精炼厂）
- `test_agent_quick.py` — 离线集成测试（桩 LLM + 合成数据）
- `src/pipeline/refinery.py` — 六阶段精炼厂编排（`RefineryPipeline` + `build_refinery_config`）
- `src/pipeline/schema.py` — 阶段间数据契约（`OreStock` / `CandidateFactor` / `RefineryResult`）
- `src/pipeline/screener.py` — PART-04 三级筛选（LASSO + 人机协同 + TOP 10%）
- `src/pipeline/alpha_pool.py` — PART-05 AlphaPool 合成（正交化 + leave-one-out）
- `src/pipeline/methodology.py` — PART-06 方法学总结报告生成与导出
- `src/engine/rpn_engine.py` — PART-03 RPN 求值引擎（可插拔指标 + 稳定性 + 并行）
- `src/data/feature_forge.py` — PART-01 特征冶炼厂（多进程并行因子池构建）
- `src/agent/transformer_encoder.py` — PART-02 Transformer 向量化表征（numpy 降级）
- `src/agent/rl_search.py` — PART-02 MaskablePPO 动作屏蔽因子组合搜索（集束降级）
- `scripts/verify_refinery_pipeline.py` — 精炼厂端到端验证（合成数据，无需网络）

## 配置说明（config.yaml）

- `llm`：模型供应商、API Key、模型名、接口地址（默认 DeepSeek）。
- `data`：`primary_source`（akshare/tushare）、默认指数与日期区间；`force_synthetic: true` 可跳过网络、使用合成数据离线演示。
- `backtest`：分位数个数、手续费率、无风险利率。
- `rag`：`use_vector_store` 控制是否启用 ChromaDB+BGE 向量检索（默认 `true`）；`embedding_model` 为向量模型名（默认 `BAAI/bge-small-zh-v1.5`）；`learned_library_path` 指定已学习因子库路径。

### 国内下载 BGE 向量模型（WinError 10060 解决）

知识库首次启用时会从 HuggingFace 拉取 BGE 向量模型权重。国内网络访问 `huggingface.co` 常被阻断，导致 `WinError 10060` 连接超时。已内置两种解法，无需手动改代码：

1. **镜像自动生效**：`config.yaml` 的 `rag.hf_endpoint` 默认指向 `https://hf-mirror.com`，代码在加载模型前会自动注入 `HF_ENDPOINT` 环境变量并覆盖 `huggingface_hub` 已捕获的常量；启动脚本 `start.ps1` 也会在启动 Streamlit 前注入，双重保险。海外/可直连环境可将 `rag.hf_endpoint` 设为 `""`（空字符串）恢复官方直连。
2. **完全离线降级**：若网络无法访问任何 HuggingFace 源，将 `config.yaml` 的 `rag.use_vector_store` 设为 `false`，知识库会自动降级为 jieba 关键词检索，无需下载任何向量模型，系统照常可用。

模型下载一次后会缓存在 `~/.cache/huggingface`（或 `rag.hf_home` 指定目录），之后运行完全离线、不再触网。
- `agent`：`max_iterations` 反思轮数、`metrics_threshold` 有效因子 IC 阈值。
- `refinery`：六阶段精炼厂配置，含 `offline`（离线演示开关）、`n_symbols/train_days/test_days`（中证1000 子集规模与训练/测试切分）、`n_workers`（多进程并行）、`transformer`（d_model/nhead/num_layers）、`rl_max_len/rl_candidates`（MaskablePPO 组合上限与候选数）、`screener`（LASSO/人机协同/TOP 比例）、`alpha_pool`（正交化/LOO/迭代）、`rpn`（分位数/预测周期/换手惩罚/并行）。置 `refinery.offline: false` 接入 LLM 矿场。

## 学习库（自主学习 + 外部因子字典导入）

Agent 内置「已学习因子库」（`data/learned_factors.jsonl`），实现两个闭环能力：

1. **自主学习**：每轮因子经沙箱校验通过且回测无错误后，自动写入学习库（记录名称、代码、回测指标），后续任务可在检索中复用该因子。
2. **外部导入 / 调用**：可将飞书因子字典等外部因子批量导入；导入后既参与知识检索（学习），其中含代码的因子还会在生成阶段作为可复用模板被 Agent 直接调用（改写/沿用），加速收敛并保证可运行。

导入外部因子字典（飞书表格导出为 CSV/Excel/JSON 均可）：

```bash
python scripts/import_factors.py data/feishu_factors.csv feishu
```

- 列名中英文均可识别：因子名称(title)、类别(category)、公式(formula)、描述(description)、代码(code，可选但建议提供以便"调用")。
- 也可在 Streamlit 的「📚 学习库」页面直接上传文件导入与浏览。
- 导入后的因子立即生效：下一次运行因子挖掘 Agent 即可检索并调用。

## 本地部署：接入 Ollama 与 Kronos

FactorGPT 的 LLM 客户端基于 LangChain `ChatOpenAI`，天然兼容任意 OpenAI 兼容端点。因此**本地 Ollama 只需改 `config.yaml`**，无需改动核心代码；而 **Kronos 是 HuggingFace 上的金融 K 线时序预测基础模型**（非聊天 LLM），项目将其作为「预测因子」集成，而非替代代码生成 LLM。

### 1) 安装并本地部署 Ollama

一键脚本（纯 Python，跨平台；静默安装 + 启动服务 + 拉取代码模型 + 切换 config）：

```bash
python scripts/setup_ollama.py                 # 安装 + 拉取默认 qwen2.5-coder:7b + 切换 config
python scripts/setup_ollama.py --skip-pull     # 仅安装并启动服务, 不拉取模型
python scripts/setup_ollama.py --model llama3.1:8b  # 指定模型
```

> 注意：Ollama 的 Windows 安装器需要**管理员权限**。若当前为非管理员终端，静默安装会失败（退出码 1）——此时请以**管理员身份**重新运行上面的脚本，或手动双击安装包后执行 `ollama pull qwen2.5-coder:7b`。安装器已缓存于 `%TEMP%/OllamaSetup.exe`。

手动步骤：到 https://ollama.com 下载安装；启动后 `ollama pull qwen2.5-coder:7b`（或 `llama3.1:8b`、`deepseek-coder:6.7b` 等代码模型）；服务默认监听 `http://localhost:11434`。

> 本项目 `config.yaml` 的 `llm` 段已默认切换为 `provider: ollama`（模型 `qwen2.5-coder:7b`，端点 `http://localhost:11434/v1`，key 占位 `ollama`）。安装并拉取模型后即可直接 `python run_agent.py "请构建一个 20 日动量因子"` 走本地模型；如需切回云端，把 `provider` 改回 `deepseek` 并恢复 `model`/`base_url`/`api_key` 即可。

### 2) 把 FactorGPT 切到 Ollama

`config.yaml` 中将 `llm` 段改为：

```yaml
llm:
  provider: ollama
  model: qwen2.5-coder:7b          # Ollama 中已拉取的模型名
  base_url: "http://localhost:11434/v1"
  api_key: "ollama"                # 占位, Ollama 不校验
```

也可用环境变量免改文件（先把 `llm.use_env_override` 设为 `true`）：
`FACTORGPT_LLM_PROVIDER=ollama FACTORGPT_LLM_BASE_URL=http://localhost:11434/v1 FACTORGPT_LLM_API_KEY=ollama FACTORGPT_LLM_MODEL=qwen2.5-coder:7b`。
之后 `python run_agent.py "请构建一个 20 日动量因子"` 即走本地模型。

### 3) 接入 Kronos 金融预测模型

检索到的 GitHub 仓库：[morrisluo/kronos](https://github.com/morrisluo/kronos)（fork 自 `shiyu-coder/Kronos`），模型权重在 HuggingFace `NeoQuasar/Kronos-mini`（另有 `small`/`base`，参数量 4.1M/20.5M/93.6M）。它是 decoder-only 的 K 线序列预测模型，输入 `[Date,Open,High,Low,Close,Volume]`，输出未来若干根 K 线，**不是聊天接口、无 GGUF、未进 Ollama 库**，故以「预测因子」方式集成。

集成位置：`src/kronos/`（核心 `forecaster.py`）。

- **快速体验（离线 stub）**：无需 GPU/网络，直接跑演示，自动降级为几何动量代理并评估 IC：
  ```bash
  python scripts/run_kronos_factor.py
  ```
- **启用真实模型**：Kronos 模型代码已随仓库内置在 `third_party/kronos/`（来自 GitHub `morrisluo/kronos` fork，因本机直连 GitHub 受限，已通过 raw 文件落地），推理依赖 `torch`/`transformers`/`einops`/`tqdm`（本项目已具备）。把 `config.yaml` 的 `kronos.enabled` 设为 `true`、`fallback_to_stub` 设为 `false` 即启用真实模型；首次会从 HuggingFace 镜像（`hf-mirror.com`）下载 `NeoQuasar/Kronos-mini` 权重与独立的 `NeoQuasar/Kronos-Tokenizer-base` tokenizer。
- **接入精炼厂因子池**：在 refinery PART-01 之后调用 `from src.kronos import attach_kronos_factor; attach_kronos_factor(ore, cfg)`，即把 `KRONOS_PRED`（Kronos 预测的未来收益）作为候选因子参与 RPN 求值与合成。Kronos 依赖缺失或下载失败时，自动降级 stub 并告警，流水线照常可跑。

> 说明：代码生成环节（因子代码、反思）仍由上面 `llm` 段的模型负责；Ollama 等本地模型仅承担该角色，Kronos 负责「未来价格预测」维度的信号增强。

## Vibe-Trading 集成

FactorGPT 已接入 Vibe-Trading 的自然语言策略范式：

- 知识层：内置 `data/vibe_trading_alpha_catalog.json`（Vibe-Trading / HKUDS 风格量化 Alpha 信号），可在「🚀 Vibe Trading」页面一键注入 RAG 已学习因子库，参与检索与模板复用。
- 工作流层：用自然语言描述交易想法，Agent 借助 Alpha 参考库生成并回测选股因子（describe → factor → backtest）。
- 原生引擎（可选）：`pip install -r requirements.vibe.txt` 后，可在页面 / `--vibe-native` 优先调用 `vibetrading` 包的加密货币策略回测流程；不可用或离线时自动降级到 FactorGPT 引擎。

用法：

- Web：侧边栏选择「🚀 Vibe Trading」，输入策略描述后运行。
- CLI：`python run_agent.py --vibe "低估值高 ROE 的质量因子，行业中性"`。
