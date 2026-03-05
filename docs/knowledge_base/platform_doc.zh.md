# 平台文档

## 简介

`platform-doc` 用于将数据库/BI 平台的官方文档（例如 Snowflake、StarRocks、Polaris）导入到独立向量库中，使 Agent 在生成 SQL 前可以查阅平台特有语法与能力。

完整流水线：

```text
抓取 → 解析 → 清洗 → 分块 → 向量化 → 存储
```

## 存储范围

- **与 namespace 无关**：文档按平台独立存储，可跨 namespace 复用。
- **默认路径**：`~/.datus/data/document/<platform>/`。
- **选择依据**：工具调用中的 `platform` 参数决定使用哪个文档库。

## 构建文档库

平台文档使用独立的 `platform-doc` 命令初始化（与 `bootstrap-kb` 分开）。

### 命令语法

```bash
datus-agent platform-doc \
  --platform <platform_name> \
  --source <source> \
  --source-type <github|website|local> \
  --update-strategy <check|overwrite> \
  [可选参数]
```

### 参数说明

| 参数                   | 默认值              | 说明                                                          |
|----------------------|------------------|-------------------------------------------------------------|
| `--platform`         | `default`        | 平台名称，用作存储键（建议明确指定）。                                         |
| `--source`           | -                | GitHub 仓库 `owner/repo`、网站 URL 或本地路径。若已在 `agent.yml` 中配置可省略。 |
| `--source-type`      | `local`          | 来源类型：`github`、`website`、`local`                             |
| `--version`          | 自动               | 文档版本标签；不传时会尽量自动检测最新版本。                                      |
| `--github-ref`       | 默认分支             | GitHub 专用：指定分支或 tag。                                        |
| `--paths`            | `docs README.md` | GitHub 专用：仓库内需要抓取的路径列表。                                     |
| `--include-patterns` | -                | 包含规则（本地为 glob，网站为正则）。                                       |
| `--exclude-patterns` | -                | 排除规则（本地为 glob，网站为正则）。                                       |
| `--chunk-size`       | `1024`           | 分块目标大小，字符数（详见下方说明）。                                                |
| `--max-depth`        | `2`              | Website 专用：最大爬取深度。                                          |
| `--pool-size`        | `4`              | 并行处理线程数。                                                    |
| `--update-strategy`  | `check`          | `check`（仅检查）或 `overwrite`（重建）。                              |

#### `--chunk-size` 说明

`--chunk-size` 是一个**软限制**：为保留段落和代码块的语义完整性，单个 chunk 可能超过目标大小（硬上限为 **2048** 字符）。小于 **256** 字符的 chunk 会自动与相邻 chunk 合并。

- **默认值**：1024 字符
- **推荐范围**：512–2048
- **调优方向**：值越大，chunk 越少越粗；值越小，chunk 越多越细

#### 支持的文档类型

- **local / github**：`.md`、`.markdown`、`.html`、`.htm`、`.rst`、`.txt`

### 使用示例

**1）GitHub（默认分支）**

!!! tip
    建议设置 `GITHUB_TOKEN` 环境变量或在 `agent.yml` 中配置 `github_token`，以避免 GitHub API 速率限制。

```bash
datus-agent platform-doc \
  --platform starrocks \
  --source StarRocks/starrocks \
  --source-type github \
  --update-strategy overwrite \
  --paths docs/en
```

**2）GitHub（指定 tag 或分支）**

```bash
datus-agent platform-doc \
  --platform starrocks \
  --source StarRocks/starrocks \
  --source-type github \
  --version 4.0.5 \
  --github-ref 4.0.5 \
  --update-strategy overwrite \
  --paths docs/en
```

**3）GitHub（版本化文档分支）**

```bash
datus-agent platform-doc \
  --platform polaris \
  --source apache/polaris \
  --source-type github \
  --github-ref versioned-docs \
  --update-strategy overwrite \
  --paths 1.2.0 1.3.0
```

**4）官方网站抓取**

```bash
datus-agent platform-doc \
  --platform snowflake \
  --source https://docs.snowflake.com/en/sql-reference \
  --source-type website \
  --version latest \
  --update-strategy overwrite \
  --max-depth 2
```

**5）本地目录**

```bash
datus-agent platform-doc \
  --platform duckdb \
  --source /path/to/duckdb-docs \
  --source-type local \
  --version v1.0.0 \
  --update-strategy overwrite
```

## 在 `agent.yml` 中配置（可选）

可以在 `agent.document` 中为不同平台写入抓取配置，然后只需要传 `--platform` 即可执行。CLI 参数会覆盖 YAML 中的配置。

```yaml
agent:
  document:
    tavily_api_key: ${TAVILY_API_KEY}
    starrocks:
      type: github
      source: StarRocks/starrocks
      paths: ["docs/sql-reference"]
      chunk_size: 1024
      github_token: ${GITHUB_TOKEN}
    snowflake:
      type: website
      source: https://docs.snowflake.com/en/
      max_depth: 3
      include_patterns: ["/en/user-guide", "/en/sql-reference"]
    local_docs:
      type: local
      source: /path/to/docs
      include_patterns: ["*.md"]
      exclude_patterns: ["CHANGELOG.md"]
```

执行：

```bash
datus-agent platform-doc --platform starrocks --github-ref 4.0.5
```

## 在 Datus 中使用工具

文档导入并启用 platform doc tools 后，Agent 会获得四个工具：

| 工具                    | 作用                                       | 关键参数                                         |
|-----------------------|------------------------------------------|----------------------------------------------|
| `list_document_nav`   | 浏览文档导航树，发现可用文档标题。                        | `platform`, `version`                        |
| `get_document`        | 根据层级路径获取单个文档的完整内容。                       | `platform`, `titles`, `version`              |
| `search_document`     | 关键词语义搜索。                                 | `platform`, `keywords`, `version`, `top_n`   |
| `web_search_document` | 可选的 Web 回退搜索（Tavily，需要配置TAVILY_API_KEY）。 | `keywords`, `include_domains`, `max_results` |

如需在自定义节点或子代理中启用这些工具，请在工具配置里加入 `platform_doc_tools`（或具体的 `platform_doc_search_tools.*`）。
**推荐调用顺序**：`list_document_nav` → `get_document` → `search_document` → `web_search_document`（兜底）。

**`get_document` 一次只取一篇文档**，示例：

```text
titles=["DDL", "CREATE TABLE"]
```

若要获取多篇文档，请多次调用。

### CLI 快捷命令（Datus-CLI）

在 `datus-cli` 中可使用：

```bash
!sd
!search_document
```

用于交互式执行 `search_document` 并查看命中文档片段。

## 注意事项与排错

- **GitHub 速率限制**：建议设置 `GITHUB_TOKEN`（或在配置里写 `github_token`）。
- **网站抓取**：`--paths` 对网站无效，请使用 `--include-patterns` / `--exclude-patterns`。
- **无结果**：检查 `platform` 是否与存储一致，若存在多版本请传入正确的 `version`。
- **Web 回退**：`web_search_document` 需要 `TAVILY_API_KEY`。
