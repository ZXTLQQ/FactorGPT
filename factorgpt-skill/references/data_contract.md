# FactorGPT 数据契约：legacy DataFetcher → NeoData 字段映射

> 目的：把 FactorGPT 原有自建爬虫（`DataFetcher`）的数据接口，平滑迁移到平台
> ``neodata-financial-search`` 技能。``NeoDataSource``（``src/data/neo_adapter.py``）
> 已对齐方法签名，本文件维护字段级映射，便于接入时核对覆盖度。

## 1. 方法级映射

| 旧方法（DataFetcher）            | NeoDataSource 方法         | NeoData 端点（占位，以 SKILL.md 为准） | 覆盖状态 |
|----------------------------------|----------------------------|----------------------------------------|----------|
| ``get_daily_kline``              | ``get_daily_kline``        | ``/v1/quote/kline``                    | 已实现   |
| ``get_financial_data``           | ``get_financial_data``     | ``/v1/stock/fundamentals``             | 已实现   |
| ``get_industry_classification``  | ``get_industry_classification`` | ``/v1/stock/industry``            | 已实现   |
| ``get_index_constituents``       | ``get_index_constituents`` | ``/v1/index/constituents``             | 已实现   |
| ``get_news_sentiment``           | ``get_news_sentiment``     | ``/v1/news``                           | 已实现   |
| ``get_industry_and_cap``         | ``get_industry_and_cap``   | 待按 SKILL.md 聚合端点接入             | 待接入   |
| ``get_minute_kline``             | ``get_minute_kline``       | 待接入                                 | 回退     |
| ``get_intraday_kline``           | ``get_intraday_kline``     | 待接入                                 | 回退     |
| ``get_market_snapshot``          | ``get_market_snapshot``    | 待接入                                 | 回退     |

## 2. 字段级映射（K线为例）

因子引擎约定列：``date / open / high / low / close / volume / amount / pct_chg / symbol``

| NeoData 原始字段        | FactorGPT 约定字段 | 备注                |
|-------------------------|--------------------|---------------------|
| ``trade_date``/``datetime`` | ``date``       | 统一为日期          |
| ``open``/``high``/``low``/``close`` | 同名   | OHLC 直接对齐       |
| ``vol``                 | ``volume``         | 成交量              |
| ``circ_mv``             | ``amount``         | 成交额（需单位换算）|
| ``change_pct``          | ``pct_chg``        | 涨跌幅              |
| （由代码派生）          | ``symbol``         | 适配器写入          |

## 3. 接入步骤（阶段 0 实测清单）

1. 从平台 ``neodata-financial-search`` 技能 SKILL.md 取得真实 ``base_url`` 与端点路径。
2. 填入 ``config.yaml`` 的 ``data.neodata.base_url``；token 由平台写入
   ``~/.workbuddy/.neodata_token``（或设 ``NEODATA_TOKEN`` 环境变量）。
3. 运行 ``python factorgpt-skill/scripts/run_factorgpt.py --data 600519 2024-01-01 2024-01-10``，
   观察 ``实际取数源`` 应为 ``neodata``。
4. 逐项核对上表「覆盖状态」，对 ``待接入/回退`` 项补全端点或保持 legacy 回退。
5. 将代码中的 ``DataFetcher()`` 调用点逐步替换为 ``get_data_source(config)``。
