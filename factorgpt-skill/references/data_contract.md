# FactorGPT 数据契约：三源字段映射（offline / legacy / NeoData）

> 目的：把 FactorGPT 原有自建爬虫（`DataFetcher`）的数据接口，平滑迁移到平台
> ``neodata-financial-search`` 技能与仓库内置离线数据集。``NeoDataSource``
> （``src/data/neo_adapter.py``）与 ``OfflineDataSource``（``src/data/offline_adapter.py``）
> 均已对齐方法签名，本文件维护字段级映射，便于接入时核对覆盖度。
>
> 三个数据源由 ``config.yaml`` 的 ``data.source`` 选择（``offline`` 为默认值），
> 统一经 ``get_data_source(config)`` 工厂（``data.neo_adapter.DataSourceFactory``）取用，
> 因此上层调用点无需改动。

## 1. 方法级映射

| 旧方法（DataFetcher）            | NeoDataSource 方法         | NeoData 端点（占位，以 SKILL.md 为准） | 覆盖状态 |
|----------------------------------|----------------------------|----------------------------------------|----------|
| ``get_daily_kline``              | ``get_daily_kline``        | ``/v1/quote/kline``                    | 已实现   |
| ``get_financial_data``           | ``get_financial_data``     | ``/v1/stock/fundamentals``             | 已实现   |
| ``get_industry_classification``  | ``get_industry_classification`` | ``/v1/stock/industry``            | 已实现   |
| ``get_index_constituents``       | ``get_index_constituents`` | ``/v1/index/constituents``             | 已实现   |
| ``get_index_daily``              | ``get_index_daily``        | ``/v1/quote/index``                    | 已实现（离线） |
| ``get_trade_calendar``           | ``get_trade_calendar``     | ``/v1/calendar``                       | 已实现（离线） |
| ``get_news_sentiment``           | ``get_news_sentiment``     | ``/v1/news``                           | 已实现   |
| ``get_industry_and_cap``         | ``get_industry_and_cap``   | 无稳定的「行业+市值」批量结构化端点     | 回退 legacy（``neo()`` 显式返回空，不伪造数值） |
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

## 3. 离线数据源契约（offline，默认）

``OfflineDataSource``（``src/data/offline_adapter.py``）读取随仓库分发的 ``data/offline/``
（日K parquet 分片 + 多票池成分股 JSON + 指数日线 parquet + 交易日历 JSON +
微观快照 parquet + ``meta.json``），提供与 ``DataFetcher`` 同构的离线数据。

| 方法 | 离线行为 |
|------|----------|
| ``get_daily_kline`` | 过滤 parquet 后返回 qfq 前复权日K，列同第 2 节 |
| ``get_index_constituents`` | 读 ``constituents_<pool>.json``（支持 ``000300/000905/000906/000852`` 或 ``csi300/csi500/csi800/csi1000``），默认 ``csi800`` |
| ``get_index_daily`` | 读 ``index_daily.parquet``，默认返回中证800（``000906``）指数日线，列同 ``date/close/high/low/volume/amount/pct_chg`` |
| ``get_trade_calendar`` | 读 ``trade_calendar.json``，返回 ``YYYY-MM-DD`` 交易日列表（可按区间裁剪） |
| ``get_industry_and_cap(symbols, level=1)`` | 读 ``micro_snapshot.parquet``，返回 ``(industry, mkt_cap)``：行业取东财 1/2/3 级（``industry`` / ``industry_l2`` / ``industry_l3``），市值取总市值（**元**，构建时刻快照）。索引为 6 位 ``symbol`` 且顺序与入参一致，未命中为 ``NaN``；快照文件缺失时退化为**两个全 NaN 的 pd.Series**（**不得返回 ``None``**，否则调用方解包即崩） |
| ``get_industry_classification(level=1)`` | 由快照汇总的行业板块表（``industry`` / ``n_symbols`` / ``total_mv_100m`` / ``float_mv_100m`` / ``median_pe`` / ``median_pb``，市值单位亿元），按总市值降序 |
| ``get_micro_snapshot(symbols=None)`` | 快照明细（名称/板块/东财三级行业/注册地省份/价/市值/股本/PE/PB/来源/``as_of``），可按代码过滤 |
| ``get_market_snapshot(symbols=None)`` | 由快照拼出的行情快照（``代码/名称/快照价/总市值/流通市值/市盈率-动态/市净率/所属行业/板块/快照日期``），**快照口径非实时** |
| ``get_financial_data`` | 返回空 DataFrame，由上层多模态能力降级，不影响纯量价回测 |
| 新闻情绪 / 分钟K | 返回空，**不尝试联网** |

微观快照由 ``scripts/build_offline_micro.py`` 生成（东财公司概况报表取行业/地区，腾讯批量
报价取价/市值/PE/PB，板块由代码前缀本地判定），粒度是**构建时刻的静态截面**——市值中性化
在历史日期上是近似值，跨期使用需重新生成快照。

两个必须保持的约定：

1. **物理列 vs 契约列**：parquet 中代码列的物理名是 ``instrument``（如 ``sh.600000``），
   适配器用 ``_de_norm_symbol()`` 桥接为对外契约 ``symbol``。适配层之外不得消费
   ``instrument`` —— 该约定已由 ``tests/test_docs_contract.py`` 固化为断言。
2. **复权语义**：复权因子按各自区间的末值归一化折算为前复权价，以对齐 legacy
   ``DataFetcher`` 的默认语义；跨区间拼接数据时需注意归一化基准会随区间变化。

## 4. 接入步骤（阶段 0 实测清单）

1. 从平台 ``neodata-financial-search`` 技能 SKILL.md 取得真实 ``base_url`` 与端点路径。
2. 填入 ``config.yaml`` 的 ``data.neodata.base_url``；token 由平台写入
   ``~/.workbuddy/.neodata_token``（或设 ``NEODATA_TOKEN`` 环境变量）。
3. 运行 ``python factorgpt-skill/scripts/run_factorgpt.py --data 600519 2024-01-01 2024-01-10``，
   观察 ``实际取数源`` 应为 ``neodata``。
4. 逐项核对上表「覆盖状态」，对 ``待接入/回退`` 项补全端点或保持 legacy 回退。
5. 将代码中的 ``DataFetcher()`` 调用点逐步替换为 ``get_data_source(config)``。
