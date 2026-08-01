# FactorGPT 优化方向与可扩展模块（基于现有代码体检）

> 本文基于 FactorGPT 当前仓库（ZXTLQQ/FactorGPT，提交 472b8fd）的源码结构给出。
> 覆盖：正确性/方法论风险、工程健壮性、性能、可扩展新模块、参赛打磨五个维度。
> 文中引用形如 `src/engine/backtest.py:128` 指向具体代码位置。

## 一、最优先：正确性与方法论风险

### 1. 单因子 Agent 存在过拟合隐患（高优先级）
`src/agent/graph.py` 的闭环是「生成 → 校验 → 评价 → 反思」，而 `_load_data` 加载的同一份 `kline` 同时用于生成阶段的反思迭代和最终评价。也就是说，Agent 是在"看着答案改作业"。`refinery` 流水线已经做了 train/test 切分（`train_days`/`test_days` + LOO），但**单因子 Agent 没有 holdout**。
建议：给 Agent 增加 `train/test` 切分开关——反思迭代只在训练集上评价，最终报告用测试集独立验证；这样既保留闭环，又能在报告里给出"样本外 IC"作为可信度证据（评委很看重这点）。

### 2. turnover 口径不一致
`src/engine/backtest.py` 里 `evaluate()` 用"排名分位差"近似换手率（`:170-173`），而 `realistic_portfolio()` 用真实权重变动算换手（`:374`）。两者数值不可比，报告里同时出现会让人困惑。
建议：统一以 `realistic_portfolio` 的实际换手为唯一口径，或明确标注"近似换手"与"组合实际换手"的区别。

### 3. 前视偏差防护仍是"软约束"
沙箱只白名单了导入与内置函数（`factor_builder.py:29-51`），是否前视完全依赖 LLM 遵守 `shift(1)` 契约（`nodes.py` 的 prompt 第 4 条）。一旦 LLM 漏写 shift，回测会"假有效"且不易察觉。
建议：在沙箱 `_normalize` 之前加一道 **AST 静态检查**——扫描生成的 AST，禁止对未 shift 的 `close`/`ret` 直接构造目标变量，或至少检测 `pct_change` 结果未 shift 即被使用，命中即判校验失败并让 Agent 反思，比纯靠 prompt 可靠。

### 4. 中性化/去极值的逐日 Python 循环
`build_pipeline` 里行业/市值中性化是 `for date, grp in s.groupby(level="date")` 逐日循环（`factor_builder.py:184-194`），在数据量大时慢且容易因单列缺失静默跳过。
建议：改用 `groupby(level="date").apply` + 向量化回归，或预计算行业哑变量后用 `np.linalg.lstsq` 批量残差，速度可提升一个数量级。

## 二、工程健壮性

### 5. 沙箱未做进程隔离与资源限制
`FactorSandbox.run` 直接 `exec` 在宿主进程（`factor_builder.py:80`），仅靠 `__builtins__` 替换。pandas/numpy 仍可能通过 `pd.io`、`np.core` 等路径逃逸到 `os`/文件系统的风险存在；且**无超时、无内存上限**，一段 `while True` 就能卡死界面。
建议：把因子代码放到 **subprocess + 资源限制（Windows Job Object / Linux cgroup / `timeout` + 内存上限）** 中执行，或至少加 `signal` 超时与指令数/迭代上限；同时用 `ast` 解析阻止 `eval/exec/compile/__import__` 的非常规调用。

### 6. 依赖版本未锁定，复现性差
`requirements.txt` 全用 `>=`（如 `torch>=2.1.0`、`langgraph>=0.2.0`），不同环境可能拉到不兼容大版本，评委机器上跑不起来是大风险。
建议：导出 `requirements.lock.txt`（`pip freeze`），并补一个 `Dockerfile` + `docker-compose`，保证"一键复现"；同时把 `chromadb/sentence-transformers` 这类重依赖在离线镜像里预装（你已有 `hf-mirror` 思路，可以延伸到容器）。

### 7. 数据加载失败会"静默回退合成数据"
`graph.py:148-153` 在真实行情获取异常时直接打印并回退合成数据，界面不会报错。现场若网络抖动，评委可能以为跑的是真实数据。
建议：区分"配置要求真实数据但失败"（应明确报错/告警，不静默）与"用户主动选 offline"。可由 `force_synthetic`/`use_real_data` 显式控制，避免误导性。

### 8. 单测覆盖不足
目前只有 `test_agent_quick.py`、`verify_refinery_pipeline.py` 这类集成级脚本。核心数学（`evaluate` 的 IC/RankIC、分位数、多空夏普、`realistic_portfolio` 的涨跌停/停牌逻辑）缺单测。
建议：对照 `alphalens-reloaded` 对 IC/RankIC 做数值一致性测试；对 `realistic_portfolio` 构造已知手算案例做断言（涨停不可买、停牌冻结权重等边界）。这是竞赛答辩"方法学严谨性"的硬支撑。

## 三、性能优化

### 9. `evaluate` 的逐日 `groupby.apply(corr)` 偏慢
`backtest.py:128-129` 对每个交易日做一次 `apply` 相关，样本大时慢。可用预分组后 numpy 向量化（一次性算每截面的 pearson/spearman），或 `pandas.groupby(...).corr()` 的批量实现。
### 10. 并行粒度可下沉
`refinery` 已有多进程（`rpn_engine.evaluate_batch`、`feature_forge`），但单因子 Agent 全程串行。可对 LLM 生成的多个候选因子并行校验+回测，再让反思阶段基于并行结果择优。

## 四、可扩展的新模块（产品壁垒）

### 11. 因子衰减与稳健性监控模块
现有 `ic_by_year` 已按年分解（`:221`），但缺少**持有期 IC 衰减**（因子值随时间衰减速度）和**市场状态切换**（牛/熊/震荡下 IC 差异）。可新增 `robustness.py` 的扩展函数，作为因子"生命力"证据——这正好呼应你 `pipeline/robustness.py` 已搭的骨架。

### 12. 风险模型归因（Barra 风格）
当前组合回测只算收益/回撤，没有风格与行业暴露归因。可加一个 `risk_model` 模块：对合成/真实组合做行业暴露、市值暴露、波动率暴露分解，并在组合优化阶段加"行业中性约束"，更贴近实盘，也更能体现专业性。

### 13. 组合优化升级
`alpha_pool.py` 现在是 ICIR 加权 + 正交化（`config.refinery.alpha_pool`）。可扩展为均值-方差、风险预算（risk parity）、或带约束的 `scipy.optimize` 优化，对比不同合成方式下的样本外表现。

### 14. 多时序基础模型统一接口
`src/kronos/forecaster.py` 已把 Kronos 作为"预测因子"集成。可抽象一个 `BaseForecastFactor` 接口，把 TimesFM、Chronos、Moirai 等时序基础模型统一接入 `attach_*_factor(ore, cfg)`，让精炼厂因子池能横向对比不同预测模型的增量 IC。

### 15. 知识库自动更新与衰减监测
`src/rag/` 的语料是静态 `paper_index`。可加一个定时抓取模块：从 arXiv、金融研报、微信公众号（你的 Deep Research 工作流里有 `wechat-article-search` 能力）拉取最新因子研究，自动入库并标注"提出日期"，同时监测已有因子的实际 IC 是否随市场结构变化而衰减（知识时效性闭环）。

### 16. 实验管理与可复现追踪
当前只有 `data/learned_factors.jsonl` 做自学习。可引入轻量实验追踪（本地 JSON/SQLite 或 MLflow）：每次挖掘/精炼跑记录配置、数据版本、指标、产物路径，支持横向对比与回滚。对竞赛"可复现、可审计"是加分项。

### 17. 实时因子监控看板（Streamlit 扩展）
`src/ui/` 已有 agent/refinery/学习库/市场 hub/方法学家五个页面。可加一个"因子健康"页面：展示因子值分布漂移、近 N 日 IC 实时、组合行业暴露、以及与基准的超额曲线，从"离线挖掘工具"升级为"持续运营平台"。

### 18. 因子可解释性
LLM 生成的因子逻辑对评委/用户是黑盒。可加一层：用 LLM 把生成的因子代码反向解释为自然语言假设，再用 SHAP/特征重要性做数值归因，产出"因子逻辑说明卡"，既增强可信度也利于方法学报告。

### 19. 合成数据注入已知因子结构（对抗验证）
当前 `_synthetic_data`（`graph.py:200`）是纯随机游走，无法检验"系统能否挖出真实存在的因子"。可构造带已知因子结构（如故意埋入动量/规模效应）的合成数据，验证 Agent 能否复现出来——这是证明"Agent 真的在挖掘而非随机"的有力证据。

### 20. 多标的/多周期扩展
`feature_forge.py` 已有 `MINUTE_FEATURES`（分钟级特征），但主链路仍是日频 A 股。可平滑扩展到分钟级、ETF/指数、股指期货，甚至跨市场（港股通），作为"数据底座可拓展性"的展示。

## 五、参赛打磨

- **人机协同筛选落地**：`config.refinery.screener.use_human_collab=true` 目前大概率只是配置位，建议在 Streamlit 真实实现"人在三级筛选中勾选保留/剔除"，这是你 README 里强调的"产品壁垒"之一，必须真做而非占位。
- **对照实验（Ablation）**：在方法学报告里固定展示"无 LLM（仅模板）/ 有 LLM / 有 LLM+Kronos"三组的样本外指标对比，量化每个组件的边际贡献，比单纯堆指标更有说服力。
- **架构图与方法学白皮书**：把六阶段冶炼隐喻画成一张标准流程图，配一段"为什么这样设计能防过拟合"的方法学文字，答辩时直接复用。

## 优先级建议（落地顺序）

1. 先堵正确性漏洞：单因子 Agent 加 holdout（#1）、AST 前视检查（#3）、turnover 口径统一（#2）。
2. 再补工程底线：沙箱进程隔离（#5）、依赖锁定+Docker（#6）、核心单测（#8）。
3. 然后做高区分度的扩展模块：风险归因（#12）、知识库自动更新（#15）、人机协同真落地（参赛打磨）。
4. 最后做锦上添花：监控看板（#17）、可解释性（#18）、多模型接口（#14）。

以上方向均无需推翻现有架构——你的 `RPN_METRICS` 注册表（`rpn_engine.py:49`）、`register_metric` 装饰器、精炼厂六阶段 dataclass 配置（`refinery.py:45`）本身已是很好的可插拔骨架，新指标/新模块基本都能以"注册+配置"方式接入，改造成本低、收益高。
