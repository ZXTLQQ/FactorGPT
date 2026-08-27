# 东财妙想 (MX) 金融数据技能包

基于东方财富「妙想」开放 API 的 6 个金融数据技能包，为 FactorGPT 提供稳定的
行情 / 财务 / 资讯 / 选股 / 自选 / 组合 / 社区数据通道，可替代易断流的
akshare / sina 自建爬虫。

> 接口说明文档：<https://marketing.dfcfw.com/res/download/A620260623NIYC2U.md>
> （已归档到 `文档归档/A620260623NIYC2U.md`）

## 技能包清单

| 目录 | 脚本 | 能力 | 典型查询 |
|------|------|------|----------|
| `mx-data` | `mx_data.py` | 行情/财务/板块/指数数据 | "贵州茅台最新收盘价与PE" |
| `mx-search` | `mx_search.py` | 金融资讯/研报搜索 | "半导体行业最新研报" |
| `mx-xuangu` | `mx_xuangu.py` | 智能选股 | "最近5日涨幅超过20%的股票" |
| `mx-zixuan` | `mx_zixuan.py` | 自选股管理 | "查询我的自选股行情" |
| `mx-moni` | `mx_moni.py` | 模拟组合 | "查询我的模拟组合收益" |
| `mx-poster` | `mx_poster.py` | AI 金融社区 | "金融社区热门内容" |

每个技能包均为官方原版（`SKILL.md` + Python 脚本 + `_meta.json`），未做任何改动。

## 安装

本目录即项目的技能包仓库。将其安装到本机智能体（CodeBuddy / WorkBuddy）时，
复制对应技能包到用户的 skills 目录即可：

```bash
# Windows (CodeBuddy)
copy /Y factorgpt-skill\skills\mx-data C:\Users\<用户>\.codebuddy\skills\
# 其余 mx-* 同理

# Linux / macOS
cp -r factorgpt-skill/skills/mx-* ~/.codebuddy/skills/
```

## 配置 API Key（本地，勿提交）

申请妙想 API Key 后，二选一配置：

```bash
# 方式一：永久环境变量（Windows）
setx MX_APIKEY "你的Key"
# 方式二：项目 .env 文件（已 gitignore，不会入库）
#   MX_APIKEY=你的Key
```

仓库内的 `.env.example` 中 `MX_APIKEY=` 保持留空，由使用者自行填写。

## 使用

### 统一入口（推荐，跨平台）

```bash
python scripts/mx_query.py data "上证指数今日行情"
python scripts/mx_query.py search "白酒板块研报"
python scripts/mx_query.py xuangu "市盈率低于10的银行股"
python scripts/mx_query.py --list
```

输出默认写入 `<项目根>/output/mx_data/`（入口自动适配 Windows，无需手动指定）。
API Key 自动从环境变量或项目 `.env` 读取并注入。

### 直接调用官方脚本

```bash
# Linux / macOS（脚本默认输出 /root/.openclaw/workspace/mx_data/output/）
python factorgpt-skill/skills/mx-data/mx_data.py "同花顺最近3年每天的最新价"

# Windows（显式指定输出目录，避免落到 /root/ 路径）
python factorgpt-skill\skills\mx-data\mx_data.py "同花顺最近3年每天的最新价" .\output\mx_data
```

## 注意

- 官方脚本默认输出目录为 Linux 路径（`/root/.openclaw/workspace/...`），
  Windows 下请显式传入输出目录，或统一使用 `scripts/mx_query.py` 入口。
- API 有调用频次限制，请按需查询并缓存结果。
