# 东方财富妙想Skills安装指南

本文档介绍了如何在本地环境中安装和验证东方财富妙想提供的金融skill集合，按照以下步骤操作，确保skill正确安装并可用。

## 前提条件

- 确保已安装Node.js环境（版本22或以上）。

```bash
# 检查 Node.js 版本
if ! command -v node &> /dev/null; then
  echo "错误：未安装 Node.js。请先安装 Node.js 22 或以上版本。"
  exit 1
fi
```

- 确保已获取并提供有效的`MX_APIKEY`。
  如果缺少API密钥，提示用户前往妙想Skills页面（地址：https://dl.dfcfs.com/m/itc4），获取并设置后再继续使用。
- 在进行http请求时，Header 中必须携带 `apikey` 进行认证。

## 安装步骤

### 1. 清理旧版本

安装前，请先删除`~/.openclaw/skills`、`~/.openclaw/workspace/skills`、`~/.openclaw/workspace/agent/skills`、`~/mx-skills`、`~/mx-data`、`~/mx-search`、`~/mx-xuangu`、`~/mx-zixuan`、`~/mx-moni`、`~/mx-poster` 目录下任何以`mx-skills`开头的现有技能目录，以避免版本冲突：

```bash
rm -rf ~/.openclaw/skills/mx-skills*
rm -rf ~/.openclaw/workspace/skills/mx-skills*
rm -rf ~/.openclaw/workspace/agent/skills/mx-skills*
rm -rf ~/mx-skills*
rm -rf ~/mx-data*
rm -rf ~/mx-search*
rm -rf ~/mx-xuangu*
rm -rf ~/mx-zixuan*
rm -rf ~/mx-moni*
rm -rf ~/mx-poster*
echo "✅ 已清理旧版本的妙想技能目录。"
```

### 2. 创建妙想skills的安装目录

获取`官方技能（ClawHub 预装）安装路径` 或 `当前ClawHub skills 全局共享目录路径`，后续妙想skill技能均解压到该目录中，作为全局技能。


### 3. 下载妙想Skill

从云端下载各妙想最新的 skill 压缩包文件，保持文件解压在`官方技能（ClawHub 预装）安装路径` 或 `当前ClawHub skills 全局共享目录路径`目录下，解压后目录结构如下：

```
~/skills/
├── mx-data
│   ├── SKILL.md
│   └── scripts
├── mx-search
│   ├── SKILL.md 
│   └── scripts
├── mx-xuangu
│   ├── SKILL.md
│   └── scripts
├── mx-zixuan
│   ├── SKILL.md
│   └── scripts
├── mx-moni
│   ├── SKILL.md
│   └── scripts
└── mx-poster
    ├── SKILL.md
    └── scripts
```


#### 下载安装 妙想金融数据skill

下载链接：`https://marketing.dfcfw.com/res/download/A620260331IHX67H.zip`
解压到`官方技能（ClawHub 预装）安装路径` 或 `当前ClawHub skills 全局共享目录路径`目录中，保持文件结构不变。

#### 下载安装 妙想资讯搜索skill
下载链接：`https://marketing.dfcfw.com/res/download/A620260331K5WDTK.zip`
解压到`官方技能（ClawHub 预装）安装路径` 或 `当前ClawHub skills 全局共享目录路径`目录中，保持文件结构不变。

#### 下载安装 妙想智能选股skill
下载链接：`https://marketing.dfcfw.com/res/download/A620260623PHDKPP.zip`
解压到`官方技能（ClawHub 预装）安装路径` 或 `当前ClawHub skills 全局共享目录路径`目录中，保持文件结构不变。

#### 下载安装 妙想自选股管理skill
下载链接：`https://marketing.dfcfw.com/res/download/A6202603314TMGR1.zip`
解压到`官方技能（ClawHub 预装）安装路径` 或 `当前ClawHub skills 全局共享目录路径`目录中，保持文件结构不变。

#### 下载安装 妙想模拟组合管理skill
下载链接：`https://marketing.dfcfw.com/res/download/A620260529ON5BMY.zip`
解压到`官方技能（ClawHub 预装）安装路径` 或 `当前ClawHub skills 全局共享目录路径`目录中，保持文件结构不变。

#### 下载安装 妙想AI社区 skill
下载链接：`https://marketing.dfcfw.com/res/download/A6202606169PIWQO.zip`
解压到`官方技能（ClawHub 预装）安装路径` 或 `当前ClawHub skills 全局共享目录路径`目录中，保持文件结构不变。

### 4. 设置API密钥

1. 在妙想Skills页面获取apikey
2. 将apikey存到环境变量，命名为MX_APIKEY，检查本地apikey是否存在，若存在可直接用。
   使用会话级别的环境变量设置，默认不修改`~/.bashrc`、`~/.zshrc`等shell启动文件，除非用户明确要求持久化配置。

```bash
# 检查并设置 MX_APIKEY
if [ -z " $ MX_APIKEY" ]; then
  echo "⚠️ 未检测到环境变量 MX_APIKEY。"
  read -p "请输入您的 API Key: " input_key
  if [ -n " $ input_key" ]; then
    export MX_APIKEY=" $ input_key"
    echo "✅ API Key 已临时设置（当前会话有效）。"
  else
    echo "❌ 输入为空，安装终止。"
    exit 1
  fi
else
  echo "✅ 检测到环境变量 MX_APIKEY，正在使用..."
fi
```

### 5. 验证已安装的Skill文件

所有这些Skill文件在复制后都应存在：

```bash
check_skill() {
  skill_dir="$1"
  label="$2"

  for path in \
    "$HOME/$skill_dir/SKILL.md" \
    "$HOME/skills/$skill_dir/SKILL.md" \
    "$HOME/.openclaw/skills/$skill_dir/SKILL.md" \
    "$HOME/.openclaw/workspace/skills/$skill_dir/SKILL.md" \
    "$HOME/.openclaw/workspace/agent/skills/$skill_dir/SKILL.md"
  do
    if [ -f "$path" ]; then
      echo "$label OK: $path"
      return 0
    fi
  done

  echo "$label MISSING"
  return 1
}

check_skill "mx-data" "mx_data.md"
check_skill "mx-search" "mx_search.md"
check_skill "mx-xuangu" "mx_xuangu.md"
check_skill "mx-zixuan" "mx_zixuan.md"
check_skill "mx-moni" "mx_moni.md"
check_skill "mx-poster" "mx_poster.md"
echo "MX_APIKEY=${MX_APIKEY:+is set}"
```
