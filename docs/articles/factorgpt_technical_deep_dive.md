# FactorGPT 深度解析：用大语言模型构建量化因子工业化生产线的完整实践

> 从自然语言到生产级 Alpha 因子：一个覆盖知识检索、代码生成、沙盒验证、量化回测、反思改进全流程的 LLM Agent 量化因子工厂。

**关键词**: 量化金融, Alpha 因子挖掘, LLM Agent, LangGraph, 因子回测, 沙盒执行, A 股量化, 大模型应用

---

## 一、问题的起点：量化因子研究的三大痛点

每一个从事量化研究的人都经历过这样的循环：读到一篇论文或想到一个投资逻辑 → 手动写 Python 代码 → 跑回测 → 发现效果不好 → 调参数重新来。这个循环反复数次，一个因子从构思到验证往往需要数天到数周。

更深层次的问题有三个：

**高门槛**。因子研究要求同时具备金融领域知识（理解因子逻辑）、数学统计功底（IC 分析、稳定性检验）、编程能力（Pandas 数据处理、回测框架）。任意一项短板都会成为瓶颈。

**迭代慢**。传统流程中，数据清洗、因子编写、参数调优、回测验证是割裂的步骤。一个看似简单的因子（比如"20 日动量反转 + 行业中性化"），实际落地可能需要写 100 多行代码、处理边界情况、debug 数据对齐问题——光是把想法变成可执行的代码就需要半天。

**流程断**。因子生成、评估、筛选、合成、监控这些环节往往是分散的，缺乏一条将原始数据持续转化为可投用因子的流水线。优秀的研究成果常常沉睡在 Jupyter Notebook 里，始终没有真正进入生产环境。

---

## 二、FactorGPT 的解题思路：把因子研究变成精炼厂流水线

FactorGPT 的核心理念是：**因子研究不应该是一个手工创作过程，而应该是一条工业化的精炼流水线**。

想象一下钢铁厂的工作流程：矿石进场 → 高炉冶炼 → 钢水精炼 → 轧制成材 → 质检出库。FactorGPT 将量化因子研究映射为同样的六阶段流程，我们称之为 **"六阶段因子精炼厂"**：

```
矿石仓库(Ore) → 矿场开采(Mining) → 研磨车间(Grinding)
    → 三级筛选(Screening) → 合金配比(Blending) → 方法学报告(Report)
```

每个阶段都是一个独立的工序，有明确的输入、输出和质量控制标准。但 FactoryGPT 最核心的创新不在流水线本身，而在于**用 LLM Agent 驱动整条流水线**。

---

## 三、架构核心：LLM Agent 驱动的闭环因子挖掘

FactorGPT 的智能因子挖掘 Agent 基于 **LangGraph** 构建，实现了 **检索 → 生成 → 验证 → 评估 → 反思** 的闭环工作流。

### 3.1 工作流详解

**第一步：知识检索（Knowledge Retrieval）**

当用户输入"构建一个 20 日动量反转因子"时，Agent 首先不是直接让 LLM 写代码，而是去知识库中检索相关文献。知识库基于 ChromaDB + BGE 嵌入向量，收录了 61 个传统因子，涵盖价格趋势（18个）、波动率（9个）、交易强度（15个）、量价关系（10个）、成交量衍生（9个）五大类。

检索到的相关因子被作为上下文注入 LLM 的提示词中，确保生成代码有学术依据而非凭空编造。

**第二步：代码生成（Factor Code Generation）**

LLM 根据用户意图和知识库上下文生成因子代码。这里有两个关键的约束机制：

1. **安全协议（Safety Protocol）**: LLM 被要求生成一个 `alpha_factor(df) -> DataFrame` 函数，输入和输出都必须遵循规定的 Schema。函数只能使用白名单内的库（numpy, pandas, scipy, statsmodels, sklearn），禁止调用 os、subprocess、eval、exec 等危险接口。

2. **上下文注入**: prompt 中包含了上一个步骤检索到的学术因子文献摘要和参数建议，以及当前市场环境的简要描述。

**第三步：沙盒验证（Sandbox Validation）**

生成的代码不是直接执行的，而是在一个隔离的子进程中运行。沙盒提供三层保护：

- **超时控制**: 因子代码若超过 30 秒未完成则强制终止
- **内存限制**: 限制子进程最大内存使用，防止 OOM
- **超前偏差检测**: 基于 AST 静态分析，检测三种超前偏差：
  - 使用了未来信息（如 `shift(-1)`）
  - 使用了前视变量（如 `fwd_ret`）
  - 价格列（close、pct_chg、ret 等）在算术运算中未做 `shift(1)` 处理

**第四步：因子后处理（Post-processing）**

通过沙盒验证的因子进入标准化后处理流水线：

1. **缩尾处理（Winsorization）**: 对因子值进行 1%/99% 分位数截尾
2. **中性化（Neutralization）**: 对行业哑变量和市值做 OLS 回归取残差
3. **标准化（Standardization）**: 截面 Z-score 标准化

**第五步：回测评估（Backtest Evaluation）**

后处理后的因子进入完整的回测评估体系，产出以下指标：

| 指标类别 | 具体指标 | 说明 |
|---------|---------|-----|
| 信息系数 | IC Mean, Rank IC Mean | 因子值与未来收益的截面相关性 |
| 信息比率 | ICIR | IC 均值 / IC 标准差 |
| 稳定性 | IC > 0 比例, IC 自相关性 | 因子预测能力的稳定性 |
| 分层收益 | 5 分位数组合收益 | 因子的单调性和区分度 |
| 风险指标 | 多空夏普比率, 最大回撤 | 风险调整后表现 |
| 交易成本 | 换手率 | 实际执行可行性 |

**第六步：反思改进（Reflection & Improvement）**

如果因子 IC 未达到设定阈值（默认为 0.02），Agent 会进入反思环节。LLM 被要求分析回测结果中透露的信息（IC 方向是否符合预期、分层是否单调、IC 衰减速度等），然后提出改进方案——可能是调整回看周期、增加中性化维度、或是与另一个因子做组合。

反思结果直接作为新的上下文重新进入代码生成环节，形成一个自动化的因子优化闭环。

### 3.2 State Machine 设计

整个工作流通过 LangGraph 的 StateGraph 实现状态管理：

```
START → retrieve → generate → validate
              ↑                      ↓
              └── reflect ← evaluate ←┘
                               ↓
                            END (if IC达标)
```

每个状态节点都是纯函数，输入输出通过 TypedDict 严格定义。状态对象在所有节点间共享，天然支持断点续传和中间结果持久化。

---

## 四、六阶段因子精炼厂深度解析

如果说智能 Agent 是 FactorGPT 的大脑，那么六阶段精炼厂就是它的身体。让我们逐一深入每个阶段。

### PART-01：矿石仓库（FeatureForge）

"矿石"是 28 个基础特征（价格、成交量、换手率、市值等）的统称。FeatureForge 负责从这 28 个矿种出发，通过时间序列算子（rolling mean, std, skew, kurt, max, min, corr, cov 等）和截面算子（rank, zscore, quantile 等）的笛卡尔积组合，构建出 50+ 个因子池作为后续阶段的原材料。

多进程并行是这一阶段的关键——在 8 核机器上，数千个因子的构建可以在 10 秒内完成。

### PART-02：矿场开采（三点齐发）

开采层采用三种互补的"采矿"手段：

**Transformer 编码器**：将因子序列向量化（d_model=128, 2 层, 5 注意力头），通过自注意力机制自动发现因子之间的非线性交互关系。

**强化学习搜索（MaskablePPO）**：将因子发现建模为序列决策问题——每一步选择一个算子应用到当前表达式树上。PPO 优化器的 mask 机制确保不会选择无效的算子组合（如对两个常数做除法）。

**LLM 矿脉探索**：利用大模型的语言理解和推理能力，在"矿脉"方向上的探索。LLM 会分析当前因子池的覆盖盲区，提出新的开采方向。

三者产出的候选因子汇入下一道工序。

### PART-03：研磨车间（RPN Engine）

RPN（Rank IC Prediction Network）引擎接管了所有候选因子的批量评估。与传统的逐个回测不同，RPN 引擎支持并行批量求值，大幅提升研磨效率。

每个因子的评估报告包含 12 项指标：IC Mean、Rank IC Mean、ICIR、IC > 0 Ratio、IC 自相关性、5 分位数收益、Top-Bottom 利差、多空夏普、最大回撤、年化收益、换手率、覆盖度。

### PART-04：三级筛选（Screener）

候选因子进入三级递进筛选：

1. **LASSO 去冗余**: 对候选因子做 L1 正则化回归，自动剔除无显著增量贡献的因子
2. **人机协同 review**: 保留的人工审核节点——检查因子逻辑合理性、是否存在过拟合迹象
3. **TOP 10% 截断**: 按综合得分取前 10%

通过三级筛选的因子进入最终工序。

### PART-05：合金配比（AlphaPool）

入选因子通过 ICIR 加权合成复合因子，并进行正交化处理确保不重复计量同一信号。最后执行留一法（LOO）过拟合检验，确保合成因子在不同时间切片上表现稳定。

### PART-06：方法学报告

最后一站是自动生成方法学报告——包含因子构建逻辑、参数选择的统计依据、交叉验证结果，以及与其他主流因子的相关性分析。一键导出 Markdown 和 JSON 两种格式。

---

## 五、安全机制：让 AI 写的代码可放心执行

AI 生成代码的最大风险在于不可控——可能包含恶意操作、可能访问系统资源、可能埋藏隐式的未来信息引用。FactorGPT 在安全机制上投入了大量工程精力：

### 5.1 子进程沙盒

所有因子代码在独立子进程中执行，与主进程完全隔离。子进程只能通过 `stdin`/`stdout` pipe 与主进程通信，无法直接访问文件系统或网络。超时和内存限制由操作系统层面强制执行。

```python
# 沙盒执行的核心抽象
def _run_subprocess(code: str, data: pd.DataFrame, 
                    timeout: int = 30, 
                    memory_mb: int = 512) -> pd.DataFrame:
    proc = subprocess.Popen(
        [sys.executable, "-c", sandbox_wrapper],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        env={**os.environ, "SANDBOX_MEMORY_LIMIT_MB": str(memory_mb)},
    )
    proc.stdin.write(pickle.dumps({"code": code, "data": data}))
    proc.stdin.close()
    proc.wait(timeout=timeout)
    return pickle.loads(proc.stdout.read())
```

### 5.2 AST 超前偏差检测

超前偏差（Lookahead Bias）是量化研究中"最危险的 bug"——回测 IC 值很好看，实盘一跑就亏钱，原因往往是在回测中不小心使用了未来信息。

FactorGPT 的 AST 分析器能检测三类超前偏差：

- **负向 shift**: `shift(-N)` 直接使用了未来数据
- **前视变量名**: `fwd_ret`, `next_return`, `future_price` 等变量名
- **未 shift 的价格列**: 在算术/比较运算中直接使用了 `close`、`pct_chg`、`ret` 等价格序列而未做 `shift(1)` 对齐

第三类是最容易忽视的隐式超前偏差。例如：

```python
# ❌ 超前偏差: 使用了当日收盘价预测当日收益
factor = df["close"] / df["close"].shift(20) - 1

# ✅ 正确: 因子使用 T-1 日的信息
factor = df["close"].shift(1) / df["close"].shift(21) - 1
```

### 5.3 白名单机制

LLM 生成的代码如果引用了 `os`、`subprocess`、`importlib`、`pickle`、`eval`、`exec` 等模块或函数，在沙盒阶段就会直接拒绝执行。白名单内的库（numpy, pandas, scipy, statsmodels, sklearn）是唯一允许使用的。

---

## 六、可观测性与离线韧性

### 6.1 在线 Demo 零依赖体验

FactorGPT 提供了完整的 HuggingFace Spaces 在线 Demo，无需任何注册、配置或 API key，直接在浏览器中体验核心功能：

[https://huggingface.co/spaces/ZXTLQQ/factorgpt-demo](https://huggingface.co/spaces/ZXTLQQ/factorgpt-demo)

Demo 包含四个页面：快速体验（一句话生成因子）、因子库（61个传统因子浏览）、回测分析（交互式 IC 图表）、精炼厂（六阶段流水线演示）。

### 6.2 优雅降级策略

FactorGPT 被设计为在各种环境下都能运行：

- **ChromaDB 不可用**？自动降级到 jieba 关键词匹配检索
- **PyTorch 未安装**？Transformer 模块自动降级到 numpy 实现
- **OpenAI API 无密钥**？自动切换到本地 Ollama（qwen2.5-coder:7b）
- **Ollama 也不可用**？内置的模拟 LLM 可以提供基本的模板化因子生成
- **网络不可用**？数据层自动从本地缓存加载，或使用合成数据

这个降级链条确保了 FactorGPT 在任何环境下都不会"罢工"。

### 6.3 预检脚本

内置的 `preflight_check.py` 可以在 10 秒内对系统做全面体检：RL 依赖、本地模型、缓存数据、ChromaDB 可用性、沙盒稳定性——五大风险类别逐项检查，清晰输出 PASS / FAIL / WARN。

```bash
$ python scripts/preflight_check.py --offline

[PASS]  沙盒子进程正常运行
[PASS]  使用工程降级版 (Jieba 关键词匹配)
[WARN]  Ollama 服务未连接 (将用模拟LLM)
[PASS]  合成数据生成正常 (200只股票, 500个交易日)
[PASS]  ChromaDB 缺失 √ 已安全降级
```

---

## 七、工程实践与踩坑经验

### 7.1 依赖管理

量化项目的依赖管理是出了名的问题儿童——不同数据源（akshare/tushare/baostock）的版本互不兼容、CUDA 版本与 PyTorch 版本的耦合、以及国内 pip 源的不稳定性。

FactorGPT 的解决方案是 `requirements.lock.txt`——使用 `pip-compile --generate-hashes` 锁定了 213 个包的确切版本和 SHA256 哈希，确保在任何机器上 reproducible 安装。

### 7.2 多模型适配

LangGraph 的模型抽象层使得 FactorGPT 可以无缝切换 LLM 后端。只需要在 `config.yaml` 中修改两行配置：

```yaml
llm:
  provider: deepseek    # deepseek | openai | qwen | ollama | mock
  model: deepseek-chat
```

所有模型适配逻辑被封装在 LLM Client 层，业务代码（Agent、精炼厂、UI）完全不关心底层使用的是哪个模型。

### 7.3 性能优化

- **数据缓存**: 首次下载的市场数据序列化到本地 parquet，后续秒级加载
- **因子池并行**: 构建阶段使用 multiprocessing.Pool，按 CPU 核数自动并行
- **回测分批**: 大量因子的批量回测支持分片并行

---

## 八、快速上手与部署

### 本地体验

```bash
git clone https://github.com/ZXTLQQ/FactorGPT.git
cd FactorGPT
pip install -r requirements.lock.txt
streamlit run src/ui/app.py
```

### Docker 一键部署

```bash
git clone https://github.com/ZXTLQQ/FactorGPT.git
cd FactorGPT
docker-compose up -d
```

### 在线 Demo

无需任何安装，直接浏览器访问：

👉 [HuggingFace Spaces Demo](https://huggingface.co/spaces/ZXTLQQ/factorgpt-demo)

---

## 九、局限性与未来规划

**当前局限性**：

- 仅支持 A 股市场（数据源基于 AKShare/Tushare）
- 日频因子为主，分钟级频率尚未充分覆盖
- 强化学习搜索在离线模式下退化为启发式规则
- 缺乏实盘交易接口的直接对接

**未来规划**：

- 多市场支持（美股、港股、加密货币）
- 实时因子监控和预警看板
- 因子衰变分析与生命周期管理
- REST API 以支持编程式因子挖掘
- 与主流回测框架（Zipline、Backtrader）的集成

---

## 十、总结

FactorGPT 尝试回答一个问题：**在 LLM 时代，量化因子研究应该是什么样子的？**

我们的答案是：它不应该只是"让 AI 帮忙写几行代码"——那只是工具层面的提升。真正的变革在于**将整个因子研究流程重塑为一条可复现、可监控、可扩展的工业化流水线**，让 AI 不只是一个代码生成器，而是整条流水线的智能调度者。

从知识检索到代码生成，从沙盒安全到回测评估，从反思改进到方法学报告——FactorGPT 的每一步都在试图将量化因子研究从手工作坊升级为智能工厂。

**欢迎 Star & 体验**：[github.com/ZXTLQQ/FactorGPT](https://github.com/ZXTLQQ/FactorGPT)

---

*免责声明：FactorGPT 是面向量化金融教育与研究的学术工具。所有因子输出、回测结果和投资信号仅供参考，不构成任何投资建议。投资有风险，入市需谨慎。*
