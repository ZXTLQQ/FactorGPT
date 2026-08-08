# FactorGPT × 本地 Ollama 接入验证报告

**验证日期**：2026-07-30
**验证目标**：确认本地 Ollama（qwen2.5-coder:7b）已正确接入 FactorGPT 的 LLM 后端，能够端到端驱动因子挖掘 Agent。

## 结论

**Agent 接入本地 Ollama 验证成功。** FactorGPT 通过 `langchain_openai` 客户端调用本地 `http://localhost:11434/v1` 的 `qwen2.5-coder:7b`，真实生成并迭代了因子代码，完成回测（IC/ICIR 等指标）并将因子写入 RAG 学习库。

本次端到端运行（`python run_agent.py "请构建一个20日动量因子，并做行业市值中性化处理"`）的实际产出：

- 因子代码由本地 LLM 生成：包含 20 日动量（剔除最近一日避免未来函数）+ 行业市值中性化 `neutralize_group`，并带有「第 2–6 轮反思」（FactorGPT 的 LLM 多轮自我迭代机制产物，证明 Ollama 被多次调用）。
- 回测指标：`ic=0.0019`、`rank_ic=0.0022`、`icir=0.0170`、`ic_positive_ratio=0.5121`。
- RAG 自学习：因子 `Momentum_20D_Neutralized` 已更新到学习库（BGE 向量检索/存储工作正常）。

## 验证过程中排查并解决的 3 个网络障碍

1. **Windows 系统代理不可达**。Python `requests` 在 Windows 上会读取系统代理注册表；本机系统代理不可达，导致行情拉取 `ProxyError`。已通过禁用代理 + `trust_env=False` + `NO_PROXY=*` 解决（localhost 的 Ollama 调用不受影响）。

2. **huggingface.co 被 DNS 污染到 Facebook 网段**。FactorGPT 启动时加载 RAG 向量模型 `BAAI/bge-small-zh-v1.5`，而 `huggingface.co` 解析到了 `2a03:2880:...`（Facebook IPv6），本机不通 Facebook，导致主线程永久 `SYN_SENT` 挂起。本地缓存的 BGE 权重完整（model.safetensors 91MB 等齐全），设置 `HF_HUB_OFFLINE=1` 后离线加载成功。

3. **行情服务器连接被重置（10054）**。本机网络到东方财富/新浪行情接口在连接层直接被重置（`ConnectionResetError`/`RemoteDisconnected`），非代理、非 UA、非 DNS 问题，属环境硬限制，加 UA/重试均无法绕过。本次验证通过临时开启 `force_synthetic`/`synthetic_on_fail` 让数据层回退到合成数据，从而跳过网络、推进到 LLM 生成阶段。**这两个配置验证后已还原为 `false`。**

## 已知项目 Bug（与接入无关，建议后续修复）

运行末尾报 `[图表] 错误: 'FactorBacktester' object has no attribute 'plot_metrics'`。经核查，`plot_metrics` 方法定义在 `src/engine/backtest.py:480`（`Backtester` 类），但调用处（`src/agent/nodes.py:253`、`src/pipeline/methodology.py:69`）使用的是 `FactorBacktester` 实例，该实例缺少该方法，导致图表未生成。修复方向：让 `FactorBacktester` 继承 `Backtester` 或内聚 `plot_metrics`，不影响 Agent 接入与因子挖掘主流程。

## 后续使用说明

- 确保 Ollama 服务运行：双击桌面/开始菜单的 Ollama 快捷方式（任务栏出现羊驼图标），再执行 `python run_agent.py "你的因子需求"`。`config.yaml` 的 `llm` 段已正确指向本地 Ollama（`provider: ollama`、`model: qwen2.5-coder:7b`、`base_url: http://localhost:11434/v1`）。
- 真实行情回测：本机当前网络到东财/新浪行情服务器连接被重置，单因子回测需配置可访问这些源的网络/代理；或临时将 `config.yaml` 的 `synthetic_on_fail` 设为 `true` 用合成数据演示（已验证可行）。
- 首次推理会加载 7B 模型到内存（约数十秒），之后驻留会更快；纯 CPU 推理生成较长代码偏慢，可换更小模型（如 `qwen2.5-coder:3b`）或安装带 GPU 支持的 Ollama 提速。

## 临时产物清理

- 已删除验证用的临时 wrapper 脚本 `run_agent_local.py`。
- 已还原 `config.yaml` 的 `force_synthetic` / `synthetic_on_fail` 为 `false`。
