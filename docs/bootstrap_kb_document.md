# platform-doc 使用文档

## 概述

`platform-doc` 用于将平台文档（如 StarRocks、Snowflake、Polaris 等）导入知识库，使 Agent 在生成 SQL 时能够查阅数据库平台的官方文档。

完整 pipeline：**抓取 → 解析 → 清洗 → 分块 → 向量化存储**

## 基本语法

```bash
datus-agent platform-doc \
  --namespace <namespace> \
  --source <source> \
  --source-type <github|website|local> \
  --platform <platform_name> \
  [可选参数...]
```

## 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--namespace` | 是 | - | 数据库命名空间（需与 `agent.yml` 中配置一致） |
| `--source` | 是 | - | 文档来源：GitHub 仓库 `owner/repo`、网站 URL 或本地路径 |
| `--source-type` | 否 | `local` | 来源类型：`github`、`website`、`local` |
| `--platform` | 否 | `default` | 平台名称，用于存储和检索时的过滤标识 |
| `--version` | 否 | 自动检测 | 文档版本标签（GitHub 模式下未指定时自动从 release/tag 检测） |
| `--github-ref` | 否 | 默认分支 | GitHub 专用：指定拉取的 git ref（分支名或 tag 名） |
| `--paths` | 否 | `docs README.md` | GitHub 专用：仓库内需要抓取的路径列表 |
| `--chunk-size` | 否 | `1024` | 分块目标大小（字符数） |
| `--max-depth` | 否 | `1` | Website 专用：网页爬取最大深度 |
| `--kb_update_strategy` | 否 | `check` | 更新策略：`check`（校验）、`overwrite`（覆盖）、`incremental`（增量） |
| `--pool_size` | 否 | `4` | 并行处理线程数 |

## 使用示例

### 1. 从 GitHub 仓库导入（默认分支）

```bash
datus-agent platform-doc \
  --namespace starrocks \
  --source StarRocks/starrocks \
  --source-type github \
  --platform starrocks \
  --paths docs/en
```

自动检测版本：从仓库的最新 release/tag 获取版本号作为标签，内容从默认分支（main/master）拉取。

### 2. 从 GitHub 指定 tag 导入

```bash
datus-agent platform-doc \
  --namespace starrocks \
  --source StarRocks/starrocks \
  --source-type github \
  --platform starrocks \
  --version v3.4.0 \
  --github-ref v3.4.0 \
  --paths docs/en
```

`--github-ref v3.4.0` 使 fetcher 从 tag `v3.4.0` 拉取内容，`--version` 作为存储的版本标签。

### 3. 从 GitHub 指定分支导入（多版本仓库）

适用于 Polaris 等将多版本文档存放在特定分支的项目：

```bash
datus-agent platform-doc \
  --namespace starrocks \
  --source apache/polaris \
  --source-type github \
  --platform polaris \
  --github-ref versioned-docs \
  --paths releases
```

当 `--version` 未指定时，系统会从文件路径中自动提取版本号。例如路径 `releases/1.2.0/overview.md` 会被识别为 version `1.2.0`。

### 4. 从官方网站导入

```bash
datus-agent platform-doc \
  --namespace snowflake \
  --source https://docs.snowflake.com/en/sql-reference \
  --source-type website \
  --platform snowflake \
  --version latest \
  --max-depth 2
```

`--max-depth` 控制从入口 URL 开始的爬取深度。

### 5. 从本地目录导入

```bash
datus-agent platform-doc \
  --namespace duckdb \
  --source /path/to/duckdb-docs \
  --source-type local \
  --platform duckdb \
  --version v1.0.0
```

## 导入后的使用

文档导入后，Agent 在对话中自动获得三个工具：

- **`list_document_nav`** -- 浏览文档导航树，按层级结构列出所有文档标题
- **`get_document`** -- 根据标题/层级路径获取完整文档内容
- **`search_document`** -- 基于自然语言关键词进行语义搜索

这些工具在 `chat_system` 和 `sql_system` prompt 模板中自动渲染，Agent 生成 SQL 前会主动查阅平台文档以获取正确的语法和特性信息。

## 注意事项

1. **GitHub 需要 token**：`--source-type github` 需要在环境变量中设置 `GITHUB_TOKEN`，否则 API 请求会受到严格的速率限制（60 次/小时）
2. **`--github-ref` vs `--version`**：`--github-ref` 控制从哪个 git ref 拉取内容，`--version` 控制存储的版本标签。两者可独立设置
3. **增量更新**：使用 `--kb_update_strategy incremental` 可避免重复导入已存在的文档（基于 chunk_id 去重）
4. **`--max-depth` 仅对 website 生效**：GitHub 和 local 模式会递归遍历指定路径下的所有文件，不受此参数影响
