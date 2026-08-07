# FactorGPT 发布为 CodeBuddy / WorkBuddy 技能（Skill）计划

> 目标：把 FactorGPT 做成可在 CodeBuddy / WorkBuddy 中调用的扩展，并通过平台的
> 模型技能（NeoData 金融数据、微信文章搜索）替代自建爬虫，解决 Agent 数据源不稳定的问题。

## 1. 可行性结论

可以做，且推荐形态是 **CodeBuddy / WorkBuddy Marketplace Skill**，而不是通用 VS Code 扩展。

- 数据稳定的来源是平台内置技能（`neodata-financial-search`、`wechat-article-search`），
  它们由平台维护可用性并提供鉴权（token 在 `~/.workbuddy/.neodata_token`）。
- FactorGPT 当前的不稳定根因在 `src/data/`：`DataFetcher`、`IndexQueryService`、
  `ths_fetcher.py`、`market_data.py` 直接调用 akshare / sina / tushare / efinance 爬数据，
  上游改版即中断。只要把这一层替换为调用 NeoData 技能的适配器，稳定性问题即可根治。
- FactorGPT 的引擎（因子挖掘、IC/收益率检验、回测、图表层）与具体数据源解耦良好，
  做成"薄编排层 + 稳定数据适配器"即可，无需重写算法。

## 2. 扩展形态选择

| 形态 | 能否获得稳定数据源 | 结论 |
|------|------------------|------|
| CodeBuddy / WorkBuddy Marketplace Skill | 能（直接调用 NeoData / 微信文章技能） | 推荐，主路径 |
| 纯 VS Code 扩展 (.vsix) | 不能（脱离平台无模型技能） | 仅作外壳，数据稳定需依托 CodeBuddy |
| 自托管 HTTP 服务 + 技能调用 | 能（服务内走 NeoData 技能） | 可选，适合重度算力场景 |

建议主路径：发布为 Marketplace Skill；若需独立 UI，再用一个薄 VS Code 扩展壳承载
（数据仍经 CodeBuddy 技能链路）。

## 3. 目标 Skill 目录结构

```
factorgpt-skill/
├── SKILL.md                 # 触发词 + 工作流（因子挖掘/检验/回测/建库）
├── scripts/
│   ├── neo_adapter.py       # 调用 neodata-financial-search 技能，返回 pandas DataFrame
│   ├── factor_mine.py       # 调 FactorGPT 引擎跑单/多因子挖掘
│   ├── factor_validate.py   # IC、分组收益、多空组合检验
│   └── run_backtest.py      # 回测 + 输出交互式报告
├── references/
│   ├── data_contract.md     # 旧 DataFetcher 字段 -> NeoData 字段映射
│   └── factor_catalog.md    # 内置因子清单与参数
└── assets/
    └── report_template.html # 自包含可视化报告模板
```

## 4. 关键设计：稳定的数据适配器

新建 `NeoDataSource`，实现与现有 `DataFetcher` 完全一致的方法签名，但底层改为调用
NeoData 技能（token 自动读取，无需手动传参），例如：

- `get_stock_list()`        -> NeoData 股票列表
- `get_daily_kline(symbol)` -> NeoData 历史行情 / K 线
- `get_daily_basic()`       -> NeoData 行情衍生（成交额、换手等）
- `get_index_kline()`       -> NeoData 指数行情
- `get_fundamentals()`      -> NeoData 财务 / 基本面
- `get_industry_map()`      -> NeoData 行业分类
- 文本 / 研报类需求          -> `wechat-article-search` 技能

在 `src/data/fetcher.py` 之上加一个工厂：`DataSourceFactory.get("neodata" | "legacy")`。
因子挖掘、检验、回测、可视化逻辑（engine / validation / backtest / chart）保持不动，
只切换数据源实现。

## 5. 分阶段实施

- 阶段 0（验证）：用 NeoData 技能拉取 A股列表、日线、基本面，核对字段覆盖与
  `DataFetcher` 的差异，列出无法覆盖的字段（如个别 akshare 专有数据）。
- 阶段 1（适配器）：实现 `neo_adapter.py` 与 `data_contract.md`，单测通过。
- 阶段 2（编排）：写 `SKILL.md` 与 `scripts/*`，让 Agent 能"挖掘/检验/回测因子"。
- 阶段 3（报告）：接入 `report_template.html`，复用现有 plotly 图表层产出交互报告。
- 阶段 4（发布）：以 Marketplace Skill 形式打包，提供安装说明；必要时补薄 VS Code 壳。

## 6. 主要风险与权衡

- NeoData 字段覆盖未必 100% 等同 akshare（尤其冷门因子源），需用阶段 0 清单兜底。
- FactorGPT 引擎本地运行依赖 pandas/numpy 环境；Skill 内以子进程方式调用本地仓库，
  需约定好仓库路径或随 Skill 附带精简引擎。
- 纯 VS Code 扩展无法复用模型技能链路，数据稳定性目标必须在 CodeBuddy 内达成。

## 7. 扩展方向与补充说明（2026-08-08 更新）

- **VS Code 外壳方案**：纯 `.vsix` 拿不到模型技能链路，数据稳定无从谈起。可行做法是做一个
  薄 VS Code 扩展 / Webview 壳，**内部仍以 CodeBuddy 技能方式驱动 FactorGPT**；
  数据稳定性由 CodeBuddy 内的 NeoData 链路保证，外壳只负责交互呈现。
- **Marketplace 发布**：把 `SKILL.md` + `scripts/` + `references/` + `assets/` 按
  CodeBuddy / WorkBuddy Marketplace 插件规范打包，提交到插件仓库并配置 topics；
  可用 `scripts/set_github_topics.py` 设置 GitHub topics（需 `GITHUB_TOKEN`）。
- **多数据源联邦**：NeoData（结构化行情/财务/资金流）+ `wechat-article-search`（文本/研报）
  + legacy akshare（冷门字段兜底），由 `NeoDataSource` 的 fallback 策略统一调度，
  既稳又能覆盖全字段。
- **离线 / 现场答辩模式**：沿用既有 `cache_only` + `force_synthetic`，配合 NeoData 预热缓存，
  断网也能跑通演示。
- **安全与权限**：NeoData token 由平台托管（`~/.workbuddy/.neodata_token`），不落库、不硬编码；
  适配器只读，不写远端。
- **实测清单**：见 `factorgpt-skill/references/data_contract.md` 阶段 0，先核对字段覆盖再全量切换。
- **已落地代码**：`src/data/neo_adapter.py`（NeoDataSource + DataSourceFactory + `get_data_source` 便捷函数，
  内部自动读取全局 `config.yaml`，保证 `data.source` 开关处处生效）、
  `config.yaml` 的 `data.source` / `data.neodata` 开关、`factorgpt-skill/` 脚手架、本计划文档。

## 8. 调用点切换（2026-08-08 完成）

已将全部业务侧 `DataFetcher()` 实例化 / `DataFetcher.get_*` 类方法调用点切换为工厂 `get_data_source()`，
**本地运行方案完整保留**（默认 `data.source: legacy`，行为与原来 `DataFetcher()` 完全一致）：

- `src/agent/graph.py`：`_load_data` 内两处（行情拉取 + 行业/市值映射），传入完整 `self.config` 以尊重全局开关；
- `src/pipeline/refinery.py`：`_build_real_ore` 矿石构建（保留原 `cfg = self.config` 局部逻辑不变）；
- `src/engine/factor_system.py`：`_load_online` 在线行情加载（保留 `data.fetcher` / `src.data.fetcher` 双导入回退结构）；
- `src/data/market_data.py`：Tushare 二级回退、`stock_news` 个股新闻两处。

切换后自测：`get_data_source()` 默认返回 `DataFetcher`（取到 000906 成分股 800 只）；
设 `data.source: neodata` 且未配置 `base_url` 时返回 `NeoDataSource` 并安全回退 legacy（同样 800 只）。
`neo_adapter.py` 内部对 legacy 的引用（回退与字段约定）保留，未改动本地取数链路。

## 9. NeoData 真实服务对接复核（2026-08-08）

从平台 `neodata-financial-search` 技能 SKILL.md / reference.md 取得真实契约后复核，**结论：真实 NeoData 与最初原型假设不一致，不能直接替代 legacy**。

- **真实契约**：单 POST 端点 `https://copilot.tencent.com/agenttool/v1/neodata`，请求体
  `{"query","channel":"neodata","sub_channel":"workbuddy","data_type":"api"}`；
  成功响应里 `data.apiData.apiRecall[].content` 是**自由文本块**（行情/财务/资金流描述），
  **并非结构化批量行情/财务 REST 接口**（最初原型误假设的 `v1/quote/kline` 等 path 式端点不存在）。
- **影响**：FactorGPT 因子引擎依赖的结构化批量数据——完整日 K 线时序（回测核心）、
  完整指数成分股列表、行业/市值映射、结构化财务报表——NeoData 文本无法稳定提供。
  因此 `neo_adapter.py` 的 `neo()` 解析在多数场景下返回空，必须由 `fallback_to_legacy` 回退 legacy。
- **已落地修正**：
  1. `config.yaml` 的 `data.neodata.base_url` 已填入真实端点；`fallback_to_legacy` **保持 true**（严禁 false），
     并加注详细原因。
  2. `neo_adapter.py` 重写 `NeoDataClient`：正确 POST 真实 NL 端点（不再拼装 `v1/quote/...` 假路径），
     新增 `_nl_query` / `_extract_contents` / `_is_usable`；`NeoDataSource` 各 `neo()` 改为 best-effort 解析文本，
     解析为空/失败/无 token 一律安全回退 legacy；模块与类 docstring 同步如实说明限制。
- **联调受阻**：本环境 `connect_cloud_service` 返回的是 IDE 会话 token（audience=account），
  NeoData 代理需要平台专属 `tempToken`，实测请求返回 **HTTP 401**，无法在本地做全量字段联调。
- **自测（项目 .venv，PYTHONPATH=src）**：
  - 默认 legacy 取 000906 成分股 800 只；
  - 设 `data.source: neodata` + 真实 base_url（token 401）优雅回退 legacy，800 只；
  - neodata 源无 token 立即回退 legacy，800 只；
  - neodata 源取 K 线回退 legacy，58 行。均不崩溃。
- **后续**：待平台 `tempToken` 在本环境可用后，再做结构化解析联调；届时若证明能稳定提供所需字段，
  再评估将 `fallback_to_legacy` 设为 false。README 已同步新增「NeoData Stable Data Source (Experimental)」章节。
