---
name: factorgpt
description: FactorGPT 量化因子研究助手。当用户需要「挖掘/生成/检验/回测量化选股因子」「构建因子库」「评估因子 IC 与分组收益率」「复现卖方研报的算子与因子」「做算子网格搜索」或做 A股 多因子研究时使用；默认走仓库内置离线数据集（克隆即用、不触网），需要时通过平台 NeoData 金融数据技能获取稳定行情/财务/资金流数据，避免自建 akshare/sina 爬虫断源问题。
---

# FactorGPT 量化因子研究

FactorGPT 是一套因子挖掘—检验—回测—建库引擎。本技能将其封装为可在
CodeBuddy / WorkBuddy 中调用的能力，并把不稳定的自建爬虫替换为平台内置的
``neodata-financial-search`` 技能（token 由平台持久化在 ``~/.workbuddy/.neodata_token``）。

## 何时使用

- 用户要「挖掘/生成一个选股因子」「检验因子有效性（IC/分组收益/多空）」
- 用户要「回测因子」「构建因子库 / 因子动物园」
- 用户做 A股 多因子研究，但不想自己维护 akshare/sina/tushare 爬虫

## 稳定数据源策略（关键）

``config.yaml`` 的 ``data.source`` 默认已是 ``offline``：读取仓库内置的 ``data/offline/``
日K数据集（克隆即用、不触网、无需 Key），适合离线复现与演示。需要覆盖面更广的结构化
数据时，再走平台的 ``neodata-financial-search`` 技能，**不要** 直接调用 akshare / sina
等自建爬虫。FactorGPT 已内置 ``NeoDataSource``
（``src/data/neo_adapter.py``），接口与旧 ``DataFetcher`` 完全一致：

- 启用：在 ``config.yaml`` 设 ``data.source: neodata``，并把代码中的 ``DataFetcher()``
  替换为 ``from data.neo_adapter import get_data_source; get_data_source(config)``。
- 文本 / 研报类需求（如新闻情绪、行业研报）可调用 ``wechat-article-search`` 技能。
- 行情/财务/选股/自选/组合/社区类数据可调用**东财妙想 MX** 技能
  （``skills/mx-*/``，官方 API，配置本地 ``MX_APIKEY``，统一入口
  ``python scripts/mx_query.py <data|search|xuangu|zixuan|moni|poster> "问句"``）。
- 未配置 NeoData 或字段未覆盖时，自动回退 legacy（``data.neodata.fallback_to_legacy``），
  保证过渡期不中断；配置为 ``false`` 则严格只用稳定源。

## 工作流

1. 明确任务类型：单因子挖掘 / 多因子合成 / 因子检验 / 回测 / 建库。
2. 取数：通过上述 ``get_data_source()`` 工厂取数（默认走内置离线数据集；需要更广覆盖时切 ``neodata``），或微信文章技能取文本。
3. 执行：调用 FactorGPT 引擎
   - 交互式：``python run_agent.py``（在 ``data.source: neodata`` 下自动用稳定源）
   - 批处理 / 技能内：``python factorgpt-skill/scripts/run_factorgpt.py --data 600519 2024-01-01 2024-01-10``
     先验证稳定源可用，再按 ``scripts/run_factorgpt.py`` 中的 ``mine/backtest`` 入口跑任务。
4. 产出：复用 FactorGPT 既有 plotly 图表层，生成自包含交互报告
   （模板见 ``factorgpt-skill/assets/report_template.html``）。

## 卖方研报因子层（``src/mining/``，纯本地、无 LLM 依赖）

四篇卖方方法论文献已工程化落地为独立研究层，适合「按研报口径复现因子/算子、并产出可复核报告」类需求：

| 来源 | 落地模块 | 能回答的问题 |
|------|----------|--------------|
| 山西证券《算子网格搜索》 | ``ops`` / ``expr`` / ``gridminer`` / ``evaluator`` / ``report`` | 60 个算子的网格搜索、表达式类型门禁、四维评价（数据质量/预测能力/稳定性/相关性）与统一评分、Markdown 报告 |
| 中信建投《"逐鹿"Alpha》 | ``expr`` / ``panel`` / ``fundamental`` | 量价与基本面共处同一因子空间、TTM/YoY/QoQ、``asof_align`` PIT 投影 |
| 天风证券《因子风险与拥挤》 | ``risk`` | ΔR²、Newey-West 调整 \|t\|、VIF、自相关、拥挤度、组合风险贡献 |
| 西部证券《概念数量因子》 | ``concept`` | 概念区间成员（不回填）、概念数量/稀缺度/热度、DGTW 市值分组 |

```bash
python scripts/mining_report_demo.py      # 一键演示：合成面板 + 网格搜索 + 四维评价 → demo_output/mining_report.md
python -m pytest tests/test_mining.py -q  # 32 项测试，全离线约 15s
```

反前视是结构性保证而非约定：``pit_reference`` 用朴素循环独立复算 ``asof_align`` 的结果并逐点比对，``coverage_adj`` 在 ``lookback`` 暖机窗口之后再统计覆盖率（"历史不足"不会被误判成"因子失效"）。报告中不可测单元格一律渲染为 ``—``，JSON 以 ``allow_nan=False`` 写出，``nan`` 无法冒充真实数值。

## 参考

- 数据契约与字段映射：``factorgpt-skill/references/data_contract.md``
- 东财妙想 MX 技能包说明与安装：``factorgpt-skill/skills/README.md``
- 仓库总览 / 部署 / 配置：``README.md``
- 消融实验与各模块 OOS 贡献：``docs/ablation_report.md``
- 离线数据集口径（时间范围 / 股票数 / 交易日数 / 基准指数 / 票池成分股）：``data/offline/meta.json``
