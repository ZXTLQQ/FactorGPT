---
name: factorgpt
description: FactorGPT 量化因子研究助手。当用户需要「挖掘/生成/检验/回测量化选股因子」「构建因子库」「评估因子 IC 与分组收益率」或做 A股 多因子研究时使用；通过平台 NeoData 金融数据技能获取稳定的行情/财务/资金流数据，避免自建 akshare/sina 爬虫断源问题。
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

所有结构化行情 / 财务 / 基本面数据，优先走平台的 ``neodata-financial-search`` 技能，
**不要** 直接调用 akshare / sina 等自建爬虫。FactorGPT 已内置 ``NeoDataSource``
（``src/data/neo_adapter.py``），接口与旧 ``DataFetcher`` 完全一致：

- 启用：在 ``config.yaml`` 设 ``data.source: neodata``，并把代码中的 ``DataFetcher()``
  替换为 ``from data.neo_adapter import get_data_source; get_data_source(config)``。
- 文本 / 研报类需求（如新闻情绪、行业研报）可调用 ``wechat-article-search`` 技能。
- 未配置 NeoData 或字段未覆盖时，自动回退 legacy（``data.neodata.fallback_to_legacy``），
  保证过渡期不中断；配置为 ``false`` 则严格只用稳定源。

## 工作流

1. 明确任务类型：单因子挖掘 / 多因子合成 / 因子检验 / 回测 / 建库。
2. 取数：通过上述 ``get_data_source()`` 走 NeoData 稳定源（或微信文章技能取文本）。
3. 执行：调用 FactorGPT 引擎
   - 交互式：``python run_agent.py``（在 ``data.source: neodata`` 下自动用稳定源）
   - 批处理 / 技能内：``python factorgpt-skill/scripts/run_factorgpt.py --data 600519 2024-01-01 2024-01-10``
     先验证稳定源可用，再按 ``scripts/run_factorgpt.py`` 中的 ``mine/backtest`` 入口跑任务。
4. 产出：复用 FactorGPT 既有 plotly 图表层，生成自包含交互报告
   （模板见 ``factorgpt-skill/assets/report_template.html``）。

## 参考

- 数据契约与字段映射：``factorgpt-skill/references/data_contract.md``
- 发布与扩展方向：``文档归档/docs/factorgpt_codebuddy_skill_plan.md``
