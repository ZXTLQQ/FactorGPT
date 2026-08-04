# FactorGPT —— 基于大语言模型的检索增强型量化因子研究智能体

**项目类型**：金融创新产品设计（量化投研 Agent 系统）
**适用赛事**：浙江省大学生金融创新大赛 A 类·金融创新产品设计
**版本**：v3.x（含因子精炼厂六阶段冶炼流水线 / 多 LLM 路由 / 样本外过拟合检验）
**技术栈**：Python 3.10+ · LangGraph · Streamlit · ChromaDB · Stable-Baselines3 · PyTorch · AKShare/Tushare

---

## 一、项目背景与痛点

量化投资的 Alpha 来源高度依赖因子（factor）的挖掘与组合。传统因子研究存在三重显著痛点：第一，研究周期长，一个因子从想法到可回测代码往往需要研究员数小时乃至数天的手工编码与调试；第二，前视偏差（look-ahead bias）高发，新手极易在因子公式中引用未来信息（如未做 `shift(1)` 的收益、直接使用次日收益率标签），导致回测虚高、实盘失效；第三，可复现性与可解释性差，因子思路散落在聊天记录与笔记中，难以沉淀、复用与审计。

与此同时，大语言模型（LLM）在代码生成与领域知识检索上展现出强大能力，但直接让 LLM「写因子」存在两个致命问题：一是模型会编造不可运行或含未来函数的代码；二是缺乏严格的回测与过拟合检验，极易产出「样本内漂亮、样本外崩溃」的伪因子。FactorGPT 正是为系统性解决上述痛点而设计：它把 LLM 的生成能力与量化研究的工程纪律（沙箱校验、分层回测、样本外验证、风险归因）紧密结合，形成一条「想法 → 检索 → 生成 → 校验 → 回测 → 反思 → 可解释交付」的闭环。

---

## 二、产品定位与创新点

FactorGPT 定位于「面向券商/基金/CITIC 类机构的量化因子研究 Copilot」，核心提供两条递进式能力线：

1. **单因子敏捷挖掘（FactorAgent）**：用户用自然语言描述想法（如「结合短期反转与流动性，构建低估值质量因子」），Agent 自动检索知识库、生成因子代码、在沙箱中安全执行、做标准后处理与分层回测，并基于结果进行多轮反思迭代，最终产出带方法学解读与图表的可解释报告。
2. **复合因子冶炼（因子精炼厂 Refinery）**：面向组合层面的高阶需求，将多个基础因子通过「特征锻造 → Transformer 表征 → 强化学习组合搜索 → RPN 公式挖掘 → 浮选筛选 → 因子池管理 → 方法学报告」六阶段流水线，自动冶炼出 ICIR 更优、彼此正交（低相关）的复合因子，并生成可直接交付的调仓清单与可解释 HTML/PDF 报告。

区别于通用「AI 写代码」工具，本产品的四大创新点是：

- **检索增强 + 反思式自纠错（RAG + Reflect）**：因子不凭空生成，先检索知识库（含 606 条已学习因子与经典文献），生成失败或回测不达标时由 LLM 精准定位错误（如 SVD 未收敛、除以 0）进行针对性修复，而非泛泛重写。
- **工程级安全沙箱**：用 AST 静态扫描前视偏差 + 受限 import 白名单 + 受限内置函数 + 独立子进程超时执行，从根本上杜绝危险代码与死循环卡死。
- **过拟合检验内建**：训练/测试集切分、样本外（OOS）独立验证、Deflated Sharpe Ratio、因子动物园（因子增量 ICIR 与最强基准相关性）构成四道防线。
- **离线鲁棒性（比赛保命设计）**：网络/API 不可用时，自动降级到合成数据 + 本地 jieba 关键词检索 + 关键词因子模板兜底，保证「断网也能跑、现场不掉链」。

---

## 三、系统总体架构

系统采用「分层 + 多 Agent 编排」的清晰架构，分为五层：

```
┌────────────────────────────────────────────────────────────┐
│  表现层（Streamlit 交互前端）                                  │
│  概览 / Vibe Trading / 因子挖掘 / Agent对话 / 精炼厂 / 产品交付  │
│  行情中心 / 期货期权 / 基金 / 债券外汇 / 因子监控 / 知识库 / 配置  │
├────────────────────────────────────────────────────────────┤
│  智能体编排层（LangGraph 状态机 + 多 LLM 路由）                  │
│  FactorAgent：retrieve→generate→validate→evaluate→reflect→learn→finalize │
├────────────────────────────────────────────────────────────┤
│  因子工程层（engine）                                          │
│  FactorSandbox(沙箱) · build_pipeline(缩尾/中性化/标准化)        │
│  backtest(IC/RankIC/分层/多空) · risk_model(Barra风格代理)       │
│  TRPO/PPO策略(可插拔) · tracking(实验追踪)                       │
├────────────────────────────────────────────────────────────┤
│  复合因子流水线层（pipeline/refinery.py，六阶段）                │
│  FeatureForge → Transformer+RL → RPN → Screener → AlphaPool → MethodologyReport │
├────────────────────────────────────────────────────────────┤
│  数据/知识层（data · rag · llm）                               │
│  行情(新浪/腾讯/AKShare/Tushare/THS MCP) · 知识库(ChromaDB+BGE) │
│  学习库(606因子xlsx) · LLM(ollama/deepseek/kronos/多模型路由)    │
└────────────────────────────────────────────────────────────┘
```

配置集中在 `config.yaml`，支持运行时切换 LLM 供应商、数据源、精炼厂参数与回测假设（T+1、涨跌停、佣金、流动性门槛等），实现「一处配置、全局生效」。

---

## 四、核心模块一：单因子挖掘智能体（FactorAgent）

`src/agent/graph.py` 用 LangGraph 编排一个有状态工作流，节点实现位于 `src/agent/nodes.py`。完整闭环如下：

1. **retrieve_knowledge（知识检索）**：基于用户需求向量检索知识库，并命中学习库中已验证因子作为可复用模板。
2. **generate_factor（因子生成）**：用强约束 Prompt 要求 LLM 输出严格契约的 `alpha_factor(df) -> df[['date','symbol','factor']]`，并强制给出 economics rationale 与学术 references（Fama-French、Carhart、Jegadeesh & Titman 等），提升方法学可信度。LLM 失败时降级到关键词模板（动量/反转/波动率/规模/流动性/成长）。
3. **validate_and_compute（沙箱校验）**：`FactorSandbox` 先做 AST 前视扫描，再在子进程中执行，输出经缩尾(Winsorize)→行业/市值中性化→Z-score 标准化的因子长表，并计算覆盖率。
4. **evaluate_factor（回测评价）**：调用 `engine/backtest.py` 计算 IC、RankIC、ICIR、IC 正向比率、分位数收益、多空组合年化/夏普/累计收益、最大回撤、换手率；并在样本外独立时段验证；同时做 Barra 风格代理的风险暴露归因，诊断因子是否隐性押注某风格/行业。回测图（IC 序列/分层收益/多空权益）自动落盘。
5. **reflect_and_refine（反思迭代）**：未通过阈值（如 IC 不显著、校验失败）时，LLM 读取上一版错误精准修复，循环直至达标或达到最大轮数；OOS 不参与反思与早停，杜绝用样本外调参导致的过拟合。
6. **learn_factor（自主学习）**：通过校验且无错误的因子写入学习库（`rag/learned_library.py`），后续任务可检索复用，形成持续积累。
7. **finalize（终局报告）**：生成 Markdown 因子报告，并调用 `agent/interpret.py` 产出「因子可解释性说明卡」（逻辑/风险/适用场景/引用文献），以及 `ui/methodologist.py` 的方法学解读。

**沙箱安全机制**（`src/engine/factor_builder.py`）是本模块工程亮点：仅允许 `pandas/numpy/scipy/math/statsmodels/datetime` 导入；内置函数裁剪为最小量化子集；`analyze_lookahead` 用 AST 检测 `shift(0)/shift(-1)` 与 `fwd_ret/future_ret` 等未来变量；`use_subprocess=True` + 超时（默认 30s）在隔离进程中运行代码，避免主进程崩溃。

---

## 五、核心模块二：因子精炼厂（六阶段冶炼流水线）

`src/pipeline/refinery.py` 面向「组合级复合因子」需求，将基础因子池冶炼为更稳健的复合因子，六个阶段如下：

- **PART-01 FeatureForge（特征锻造）**：从知识库/学习库构建基础因子池（默认 12 个种子），统一对齐、去极值与标准化。
- **PART-02 向量化 + 强化学习组合**：用 Transformer（PyTorch）对因子序列做时序表征，再用 MaskablePPO（sb3-contrib）在「因子组合权重」动作空间搜索 ICIR 最优组合；无 torch/sb3 时自动降级为 numpy 表征 + 集束搜索启发式。
- **PART-03 RPNEngine（反波兰式公式挖掘）**：随机生成因子组合表达式（如 `(factorA - factorB) / (factorC + eps)`），`register_metric` 接入 IC/RankIC/ICIR 等评价指标，穷举/采样搜索新奇且有效的非线性组合。
- **PART-04 Screener（浮选筛选）**：用 LASSO（LassoCV）剔除冗余与噪声因子，结合「人工协作」接口（可注入专家约束），保留 Top 10% 候选。
- **PART-05 AlphaPool（因子池管理）**：按 ICIR 加权合成，做正交化（剔除高相关冗余），并用 Leave-One-Out 稳健性评估每个因子对复合结果的边际贡献。
- **PART-06 MethodologyReport（方法学报告）**：自动撰写因子定义、构建逻辑、经济学解释与回测结论。

流水线输出 `RefineryResult`，包含复合因子序列、候选/入选清单、复合 ICIR、组合回测（T+1、涨跌停约束、佣金与流动性门槛建模）、与中证 800 等权基准对比、鲁棒性（Deflated Sharpe Ratio 过拟合判定）、因子动物园（增量 ICIR、与最强基准最大相关性），并写入 `meta_*.json` 结构化元数据，支撑可审计与可复现。

---

## 六、回测与过拟合检验方法论

回测引擎（`src/engine/backtest.py`）基于 **alphalens-reloaded** 实现经典量化评价，核心指标包括：

| 指标类别 | 具体指标 |
|---------|---------|
| 预测力 | IC、RankIC、ICIR（信息比率）、IC 正向占比 |
| 分层 | 分位数平均/年化/累计收益、Sharpe、单调性 |
| 多空 | 多空组合累计收益、年化收益、Sharpe |
| 风险 | 最大回撤、换手率、覆盖率、风险暴露归因 |

**四道过拟合防线**：① 训练/测试集时间切分，样本外不参与反思与早停；② 样本外（OOS）独立验证，对比样本内/外 IC 落差；③ Deflated Sharpe Ratio 对多重检验（Multiple Testing）惩罚，给出「通过/需警惕」verdict；④ 因子动物园（Factor Zoo）检验复合因子相对经典基准的增量 ICIR 与最大相关性，识别「换皮基准因子」。

**风险暴露归因**（`engine/risk_model.py`）以 Barra 风格代理模型，逐日回归因子值对市值、行业、波动率、动量等风格/行业哑变量的暴露，输出 `_risk_report`，帮助判断因子是否只是隐性押注某已知风格。

---

## 七、知识与模型层

**RAG 知识库**（`src/rag/`）：向量库使用 ChromaDB + BGE 中文嵌入模型（国内自动走 `hf-mirror.com` 镜像），未安装 sentence-transformers 时降级为 jieba + TF-IDF 本地关键词检索，秒级响应、零下载。**学习库**（`rag/learned_library.py`）从 `data/learned_factors.xlsx`（606 条已验证因子，含代码与指标）加载，是「检索复用 + 自主学习」闭环的持久化载体。

**多模型接入**（`src/llm/`）：默认本地 `ollama`（deepseek-r1:14b）保障离线；可选 DeepSeek、通义千问、OpenAI 兼容端点、以及 `third_party/kronos/` 集成；`llm/router.py` 提供「小模型海选（draft）+ 强模型精炼（critic）」的多 LLM 路由，调用处零改动即可启用。前端侧边栏支持运行时切换供应商/API Key/Base URL/温度并「测试连接 + 保存配置」。

**数据层**（`src/data/`）：行情接入新浪探针（`sina_probe.py`）、腾讯、AKShare、Tushare，并预留同花顺 MCP 网关（`mcp_ths_gateway`）；前端「行情中心」覆盖 A 股实时/分时/K线、期货期权、ETF/LOF、可转债、外汇与贵金属，数据经 SQLite 短时缓存。`MarketDataFetcher` 内置多源回退链，单源失败自动切换。

---

## 八、产品形态与交付物

系统以 **Streamlit** 多页面交互（13 个导航页）呈现。`📦 产品交付` 页一键运行精炼厂并导出四类交付物：因子表达式 CSV、调仓清单 CSV、可解释 HTML/PDF 报告、结构化 JSON 元数据（`exporter.py`），并与中证 800 等权基准对比展示复合 ICIR、组合年化、夏普、最大回撤、信息比率。报告内嵌因子公式、回测图表、方法学解读与过拟合检验结论，可直接作为比赛路演与机构尽调材料。

为应对比赛现场网络/API 风险，系统设计了**离线演示模式**：`use_real_data=false` 时自动生成可复现合成数据，`cache_only=true` 时复用预抓取快照，配合本地 ollama + jieba 检索 + 关键词模板，保证「零网络也能完整演示闭环」。

---

## 九、工程化与可复现性

- **实验追踪**（`engine/tracking.py`）：每次因子评估记录代码、指标、参数与 git commit，支撑可审计/可复现；可接 MLflow（缺省降级本地 JSONL）。
- **依赖与部署**：`requirements.txt` 区分基础依赖与可选重依赖（torch/sb3/mlflow 等），缺省自动降级；`Dockerfile` 支持容器化部署；`启动.bat`/`start.ps1` 提供一键启动。`requirements.lock.txt` 与 `requirements.vibe.txt` 锁定不同场景依赖。
- **测试**：`tests/`、`verify_mh.py`、`test_agent_quick.py`、`norm_test.py` 覆盖沙箱、归一化与 Agent 快速路径。

---

## 十、创新价值与应用前景

FactorGPT 的价值可归纳为三点：**效率跃迁**（想法到可回测因子从小时级降到分钟级，反思闭环自动纠错）、**纪律内建**（前视扫描 + 样本外 + DSR + 因子动物园，把量化研究的工程纪律做成系统默认行为）、**知识沉淀**（自主学习库让机构因子研究资产可累积复用）。应用前景覆盖券商研究所因子服务、基金量化投研中台、高校金融工程教学实训，以及普惠金融场景下的智能投顾因子生产。

---

## 十一、局限性与未来工作

当前局限：① LLM 生成因子的上限受基座模型金融理解能力约束，复杂微观结构因子仍需专家介入；② 精炼厂 RL/Transformer 在大数据量下训练开销较高，已用启发式降级缓解但未做分布式训练；③ 实盘级交易成本控制（滑点、冲击成本）仍用简化假设。未来工作方向：接入更多另类数据（舆情、产业链），引入因子衰减预警与自动再训练，以及面向监管的可解释性增强（SHAP/归因可视化）。

---

## 十二、参赛亮点总结

FactorGPT 以「LLM 生成能力 × 量化工程纪律」为核心差异，将大语言模型从「会写代码的玩具」升级为「可审计、可回测、可交付的因子研究 Copilot」。其沙箱安全、样本外过拟合检验、检索增强反思闭环、多 LLM 路由与离线鲁棒性五点，既体现了扎实的工程实现，又贴合金融创新「安全、合规、有效」的评审导向，适合作为金融创新产品设计的参赛作品。

---

## 参考文档（项目内）

1. [README.md](README.md) — 项目总览与快速开始
2. [config.yaml](config.yaml) — 全局配置（LLM/数据源/精炼厂/回测假设）
3. [src/agent/graph.py](src/agent/graph.py) — LangGraph 编排图
4. [src/agent/nodes.py](src/agent/nodes.py) — 因子挖掘节点实现
5. [src/engine/factor_builder.py](src/engine/factor_builder.py) — 沙箱与因子后处理
6. [src/pipeline/refinery.py](src/pipeline/refinery.py) — 六阶段冶炼流水线
7. [src/ui/app.py](src/ui/app.py) — Streamlit 交互前端
8. [知识库接入与监控.md](知识库接入与监控.md) — RAG 与学习库设计
9. [OLLAMA_AGENT_VERIFICATION.md](OLLAMA_AGENT_VERIFICATION.md) — 本地模型验证

> 免责声明：本系统所有因子与回测结果仅供研究与演示，不构成任何投资建议。投资有风险，决策需谨慎。
