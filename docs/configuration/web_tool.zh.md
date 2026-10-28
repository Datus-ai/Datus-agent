# Web 工具

在 chat 会话中，agent 通过一对统一的工具访问公共互联网：

| 工具 | 用途 |
|---|---|
| `web_search` | 搜索互联网，返回带摘要的排序结果 |
| `web_fetch` | 抓取单个 URL，返回其可读的正文文本 |

典型用法是两者配合：先用 `web_search` 找到候选页面（最新的 SQL 方言语法、某条报错信息、产品 changelog），再用 `web_fetch` 拉取最有价值那个 URL 的全文。无论请求由哪个后端处理，两个工具都返回相同的结果结构，下游行为不依赖你所用的 LLM provider。

## 后端选择

Agent 看到的永远是同样的两个工具；由谁来处理请求则按当前激活的 LLM provider 自动决定：

- **厂商原生** —— 如果当前 provider 的 API 自带 web 搜索能力，Datus 会自动注入厂商的托管工具，并跳过本地实现。无需任何配置。
- **本地回退** —— 其他所有 provider：
    - `web_search` 调用 [Tavily](https://tavily.com) 搜索 API —— 需要 API key（见下文）。
    - `web_fetch` 直接通过 HTTP 下载页面并抽取正文 —— 无需 key，不经过任何第三方 API。

各 provider 的当前支持情况：

| Provider | `web_search` | `web_fetch` |
|---|---|---|
| OpenAI（官方 `api.openai.com`） | 厂商原生 | 本地 |
| ChatGPT Codex（官方 `chatgpt.com` 后端） | 厂商原生 | 本地 |
| Claude / Anthropic（官方 `api.anthropic.com`） | 厂商原生 | 本地 |
| 其他所有（DeepSeek、Qwen、自建服务、OpenAI/Claude 兼容中转等） | 本地（Tavily） | 本地 |

两点说明：

- 厂商原生搜索仅在**官方**端点上生效。如果 provider 配置指向自定义 `base_url`（中转、代理或自建网关），托管工具无法保证可用，Datus 会自动回退到本地 Tavily 后端。
- `web_fetch` 目前对所有 provider 都走本地后端。

会话中用 `/model` 切换模型会立即重新解析后端。

## 配置

只有本地 `web_search` 后端需要配置 —— 一个 Tavily API key，按以下顺序解析：

1. `agent.yml` 中的 `agent.document.tavily_api_key`（支持 `${ENV_VAR}` 环境变量替换）
2. 环境变量 `TAVILY_API_KEY`

```yaml title="agent.yml"
agent:
  document:
    tavily_api_key: ${TAVILY_API_KEY}
```

没有 key（且没有厂商原生搜索）时，`web_search` 不会暴露给 agent，日志中会有一条提示。`web_fetch` 始终可用。

## 工具参考

### `web_search`

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `keywords` | 字符串列表 | 必填 | 搜索关键词，如 `["StarRocks materialized view syntax"]` |
| `max_results` | int | `5` | 返回结果数上限（1–20） |
| `include_domains` | 字符串列表 | 无 | 限定结果域名，如 `["docs.snowflake.com"]` |

返回结果：

```json
{
  "query": "StarRocks materialized view syntax",
  "result_count": 5,
  "results": [
    {"title": "...", "url": "...", "snippet": "...", "age": "..."}
  ]
}
```

### `web_fetch`

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `url` | string | 必填 | 要抓取的绝对 `http(s)` URL |
| `max_chars` | int | `20000` | 抽取文本的截断长度（字符数） |

返回结果：

```json
{
  "url": "...",
  "title": "...",
  "content": "...",
  "truncated": false,
  "char_count": 12345
}
```

`truncated` 表示 `content` 是否在 `max_chars` 处被截断；当 agent 需要长页面的后半部分时，可以用更大的上限重新抓取。

## 限制与安全

本地 `web_fetch` 后端强制执行以下约束：

- **仅限公网目标（SSRF 防护）** —— 拒绝访问 `localhost`、`*.local` / `*.internal` 主机名、私有/回环/链路本地 IP 段以及云 metadata 地址。
- **仅支持 HTML / 纯文本** —— 其他内容类型（PDF、图片、二进制）会返回带说明的错误。
- **10 MiB 响应体上限** —— 声明超过此大小的响应在下载前即被拒绝。
- **单次请求 30 秒超时**，自动跟随重定向。
- **去除页面噪音** —— 脚本、样式、导航、页眉页脚会被剥离，只返回正文文本。

## 权限

两个工具同属 `web_tool` 权限类别，在所有内置 profile 中**默认允许**：它们只读取公共互联网，不会发送任何来自你数据库的数据。如需改为确认后执行或禁用，在 `agent.permissions` 下添加规则：

```yaml title="agent.yml"
agent:
  permissions:
    rules:
      - tool: web_tool
        pattern: "*"          # 也可以只写单个工具：web_search / web_fetch
        permission: deny      # allow | ask | deny
```

## 与 `web_search_document` 的区别

不要把 `web_tool.web_search` 和 [`web_search_document`](../knowledge_base/platform_doc.zh.md) 混淆：后者属于知识库平台文档工具组，是本地文档检索无结果时的 Web 兜底搜索。两者都使用 Tavily、共享同一份 API key 配置，但它们是不同的工具，权限类别也各自独立。

## 常见问题排查

| 现象 | 原因 / 处理 |
|---|---|
| 工具列表里始终没有 `web_search` | 未解析到 Tavily key 且无厂商原生搜索 —— 配置 `agent.document.tavily_api_key` 或 `TAVILY_API_KEY`。 |
| `Web search is unavailable: set agent.document.tavily_api_key or TAVILY_API_KEY.` | 同上 —— 调用时 key 不可用。 |
| `Refusing to fetch non-public URL target (SSRF protection)` | URL 指向 localhost / 私有网络。这是有意为之，只允许抓取公网目标。 |
| `Unsupported content-type '...'` | URL 不是 HTML/文本页面。请让 `web_fetch` 指向可读的网页。 |
| `Response too large ... exceeds the 10,485,760-byte web_fetch limit.` | 页面响应体超过 10 MiB。请改抓更小的页面（例如具体文章而非归档页）。 |
