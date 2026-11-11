# Agent 配置

Agent 配置定义 Datus Agent 的核心设置，包括默认目标模型与整个系统可用的 LLM 提供方。

## 结构

### 日志配置

在 `agent.yml` 中配置进程日志：

```yaml
agent:
  logging:
    level: INFO     # DEBUG、INFO、WARNING、ERROR、CRITICAL；不区分大小写
```

级别优先级为显式 `--log-level` / `--debug` → `DATUS_LOG_LEVEL` → `agent.logging.level` → `INFO`。`--debug` 等价于 `--log-level DEBUG`，两者不能同时指定。例如，`datus-api --config conf/agent.yml --log-level WARNING` 会同时将 API 和 Agent 业务日志设为 WARNING。CLI、API、Gateway、MCP 服务、Web Chatbot 和 benchmark 使用相同的解析规则；API 子进程继承最终级别，重建 Agent 不会重置级别。嵌入式应用可在启动时调用 `configure_logging(agent_config=config)`。

文件 handler 同样执行级别过滤，第三方库原有的降噪策略继续保留。日志文件统一使用纯文本格式，事件字段以键值对展示；交互式控制台输出到 stderr。日志与 tracing 独立。可选的 `agent.logging.redact` 支持 `enabled`、`fields` 和正则 `patterns`，默认开启凭据字段脱敏。事件及 trace 采集控制详见[可观测性](../develop/observability.zh.md)。

### 目标模型（target）
在未单独覆盖时，所有节点默认使用 `target` 指向的 LLM 设置：
```yaml
agent:
  target: openai
```

### 响应语言（language）
`language` 是**可选**字段，用于把所有 agentic 节点的**面向用户自然语言输出**统一到指定语言，覆盖：回答文本、总结、澄清提问、通过 `task` 工具调用子 agent 时写入的 prompt、以及写入文件的注释与说明。代码、SQL、表/列标识符、文件路径、URL、JSON key 等机器可读成分保持原样。

若不填写该字段，系统不会向 system prompt 注入任何语言指令，模型将自行根据上下文选择回答语言。

```yaml
agent:
  # 完全省略该字段即可让模型自行决定语言。
  # 填写语言码即可固定所有 agentic 节点的输出语言：
  language: zh   # 常用取值：en, zh, ja, ko, es, fr, de, pt, ru, it
```

内置语言码映射：`en` → English、`zh` / `zh-cn` → Chinese、`zh-tw` → Traditional Chinese、`ja` → Japanese、`ko` → Korean、`es` → Spanish、`fr` → French、`de` → German、`pt` → Portuguese、`ru` → Russian、`it` → Italian。未知代码将原样注入 system prompt，方便扩展。

对映射表内的语言码，除英文的名称行外还会用**该语言本身**再写一遍指令：单独一行英文元指令要和几十轮英文表结构 / SQL 工具输出竞争注意力，小模型容易漂移。未知代码仅保留英文行。

Chat API 请求可通过请求体中的 `language` 字段按任务覆盖该默认值（详见 [Chat API](../API/chat.zh.md)）。CLI 无覆盖参数，直接沿用 yaml 中的默认。

### 模型提供方（models） {#models-configuration}
为智能体配置可用的 LLM 提供方：

**每个提供方条目的必填参数：**

- **提供方键名 (`models.<key>`)** —— 逻辑标识符，由 `agent.target` 和节点 `model` 字段引用（可自定义命名）
- `type`：接口类型（与厂商适配）
- `base_url`：接口基础地址
- `api_key`：访问密钥（支持环境变量）
- `model`：具体模型名
- `ssl_verify`（可选）：该端点的 TLS 校验方式。`true`（默认）、`false`（关闭校验，不推荐）、指向 CA 证书包（PEM）的路径，或直接内联 CA 证书内容（以 `-----BEGIN CERTIFICATE-----` 开头的值），以信任私有/自签名网关 CA。优先级高于 `SSL_VERIFY` / `SSL_CERT_FILE` 环境变量。参见[私有或自签名证书](#private-or-self-signed-certificates)。

```yaml
agent:
  target: provider_name
  models:
    provider_name:
      type: provider_type
      base_url: https://api.example.com/v1
      api_key: ${API_KEY_ENV_VAR}
      model: model-name
```

!!! tip "环境变量"
    建议通过环境变量管理敏感信息：
    ```yaml
    api_key: ${YOUR_API_KEY}
    # 不建议在生产中明文写入
    api_key: "sk-your-actual-key-here"
    ```


### 上下文容量和大工具结果

每次模型请求都会为最大输出和安全余量预留空间，余量为
`max(2048, ceil(context_length * 0.05))`。以下三个配置都是可选项，已有配置无需补齐：

| 配置 | 未填写时的行为 |
| --- | --- |
| `context_length` | 优先读取已有模型规格或供应商目录；仍未知时按 `1048576`（1M）做预算 |
| `max_tokens` | 保留已有输出默认值，未知时默认 `384000`，与内置 DeepSeek V4 Flash 预算对齐。默认输出还会为输入和安全余量留空间，小窗口的默认输出可能更低 |
| `tokenizer_ref` | 已知 OpenAI 模型使用 LiteLLM 计数；其他模型（包括自部署别名）按 UTF-8 字节数保守估算，无需下载 tokenizer |

模型选择与运行时容量检查共用自动刷新的 OpenRouter 目录。首次使用时，Datus 读取当前
Agent home 下的 `cache/openrouter_models.json`，没有有效缓存则获取公共目录，超时为
8 秒。缓存有效期为 10 分钟；刷新失败时保留旧数据，60 秒后才重试。缓存同时保存上下文
和最大输出限制。异步生成和模型列表查询在线程中刷新元数据，事件循环内的容量查询只读
当前缓存快照。显式配置优先于目录元数据，`providers.yml` 和 LiteLLM 规格作为离线
兜底；新数字版本不会通过前缀匹配继承旧版本的限制。
目录公布的输出上限不会自动放大默认输出预算：已有型号保留原默认值，新发现的型号仍以
`384000` 为默认值，在目录上限或剩余上下文不足时调低。公共目录描述其服务供应商的
限制；自定义端点使用这些参数做规划参考，不会用目录输出上限否决显式配置。

默认窗口是本地预算假设，不代表自动探测到了服务端容量。私有部署可以只覆盖需要调整的项；
已知实际窗口较小时应配置实际值，窗口较大或提供准确 tokenizer 时可更充分地利用容量：
```yaml
agent:
  models:
    warehouse_llm:
      type: openai
      base_url: https://llm.example.com/v1
      api_key: ${LLM_API_KEY}
      model: warehouse-model
      context_length: 262144
      max_tokens: 8192
      tokenizer_ref: /opt/model/tokenizer.json
```

`context_length` 和 `max_tokens` 必须是正整数，单次调用的输出上限优先于配置默认值。
`tokenizer_ref` 支持本地 Hugging Face tokenizer JSON、Hugging Face tokenizer 引用，
以及与部署编码一致的 `tiktoken:<encoding>`；本地文件必须能被运行 Datus 的进程读取。
省略配置会使用默认行为，显式填写无效数值仍会报错。tokenizer 无法加载或计数失败时，
会退回可用的模型计数器或 UTF-8 字节估算，并在日志中记录降级原因。
本地计数包含序列化消息、指令、工具定义、
输出格式和消息格式余量；实际部署的模板或路由限制不同，仍可能拒绝请求，需要据此校准配置。
UTF-8 字节估算可能提前触发压缩，不等同于供应商返回的实际 token 用量。
估算超过规划预算不会阻止发送请求；明确配置的部署限制配合可用的匹配 tokenizer 仍执行容量校验，
标准供应商端点也保留目录输出上限校验。
图片优先使用可用计数器，无法计数时每张按 4096 tokens 估算；远程图片引用和无法解析的文件
也按每个 4096 tokens 占位，另计可见文本。二进制内容不参与文本估算，实际发送的内容和引用保持完整。
估算无需注册模型，也不代表探测到了服务端的实际限制。
远程图片和文件引用需要提取内容或使用供应商计数；不能只按 URL 或文件 ID 的长度放行。

OpenAI 兼容接口及原生 OpenAI Responses 路径收到明确的输出上限终止状态时，会报告失败。
工具调用执行会在失败结果和会话中保留已收到的回答正文与用量；reasoning 不会作为回答返回，
未完成的工具调用不会执行。Datus 不会自动重试这类执行；需要继续输出时可沿用原会话。

工具自行管理结果量。Agent 调用 `list_tables`、`list_databases`、`list_schemas`、`query_metrics` 和 `select_records` 默认每页 200 条，`query_lineage` 默认每页 100 条；`page_size` 可调整，没有硬上限。目录和血缘用原工具及 `snapshot_id`、`offset`（血缘还可指定 `section`）续读快照；指标和记录用原工具及 `result_id`、`offset` 续读缓存，不重复执行 SQL。目录/血缘页的 `has_more=true` 时，续读使用 `offset=extra.next_offset`；`extra.offset` 表示当前页的位置。业务 `limit` 与分页大小相互独立。直接 Python/MCP 调用不传分页参数时，保留完整目录/图或压缩查询预览的既有行为。`execute_sql` 和 `execute_reference_template` 保留现有压缩。

不再提供 `read_tool_result` 或通用结果裁剪。旧归档引用会恢复为可通过 `read_file` 按行读取的文件。源结果截断、分析完整性与分页分别表示。关闭 `agent.compact` 不会关闭最终容量检查；压缩后仍无法容纳的输入会在发送到模型前停止。分页大小按完整记录数计算，不按 token 或字节自动缩小，因此很宽的记录页仍可能触发容量检查。

Codex 订阅接口的请求协议没有输出上限字段。该路径保留现有协议，按已知的模型最大输出
（包含供应商目录中的上界）预留容量，不把较小的请求值当作已经生效的限制。
标准 API 路径会显式发送解析后的输出上限。

### 私有或自签名证书 {#private-or-self-signed-certificates}

当 Datus 连接的是内部 LLM 网关或测试环境，而其 TLS 证书由**私有 CA**（不在公共信任库中）签发时，请求会失败并报错：

```text
litellm.InternalServerError: AnthropicException - [SSL: CERTIFICATE_VERIFY_FAILED]
certificate verify failed: self-signed certificate in certificate chain
```

根因是 Python 自带的 `certifi` 信任库只包含公共 CA，且不会读取操作系统信任库，因此私有 CA 永远不会被自动信任。正确做法是让 Datus 信任该 CA 证书包，而不是关闭校验：

```yaml
agent:
  models:
    claude-staging:
      type: claude
      base_url: https://ai.internal.example.com
      api_key: ${ANTHROPIC_API_KEY}
      model: claude-3-7-sonnet
      ssl_verify: /etc/ssl/internal-ca.pem   # 信任私有 CA，校验保持开启
```

`ssl_verify` 也可直接**内联 CA 证书内容**（任何以 `-----BEGIN CERTIFICATE-----`
开头的值），无需在磁盘上放置文件——当配置是由程序生成（例如来自数据库）而非与
`.pem` 一起部署时尤其有用：

```yaml
      ssl_verify: |
        -----BEGIN CERTIFICATE-----
        MIIB...snip...IDAQAB
        -----END CERTIFICATE-----
```

原生客户端会把内联内容加载进内存信任库；litellm 代码路径只接受 CA 证书包的路径，
因此会将其落到一个进程私有目录下、按内容哈希命名的临时文件。

**解析优先级**（取首个命中项）：

```text
ssl_verify（agent.yml）  →  SSL_VERIFY 环境变量  →  SSL_CERT_FILE 环境变量  →  certifi 默认
```

- `ssl_verify` 是项目级配置，优先级高于环境变量。
- 标准的 `SSL_CERT_FILE` / `SSL_VERIFY` 环境变量可作为免配置的替代方案（例如 IT 在机器层面统一下发 CA 路径时）。
- 提取服务端 CA 证书链：
  `openssl s_client -showcerts -connect host:443 </dev/null | openssl x509 -outform PEM > ca.pem`

!!! warning "切勿关闭校验"
    `ssl_verify: false`（或 `SSL_VERIFY=false`）会**完全关闭** TLS 证书校验，使连接暴露于中间人攻击。**请勿关闭校验**，务必通过信任 CA 证书包来解决证书问题。

## 支持的提供方

初始化/配置向导中的 provider，最终都会写入 `agent.models`，并通过 `agent.target` 指定默认模型。也就是说：

- 你在 `datus-agent configure` 里选择的 provider 名称，就是 `agent.models.<provider>` 的键名
- 你也可以在后续手动编辑 `agent.yml`，把某个节点单独切换到另一套模型配置

### 通用 provider

| 提供方 | 典型模型 | 接口类型 | 认证方式 |
|---|---|---|---|
| `openai` | `gpt-5.2`、`gpt-4.1`、`o3` | `openai` | API Key |
| `deepseek` | `deepseek-v4-pro`、`deepseek-v4-flash` | `deepseek` | API Key |
| `claude` | `claude-sonnet-4-5`、`claude-opus-4-5` | `claude` | API Key |
| `kimi` | `kimi-k3`、`kimi-k2.6` | `kimi` | API Key |
| `qwen` | `qwen3-max`、`qwen3-coder-plus` | `openai` | API Key |
| `gemini` | `gemini-2.5-flash`、`gemini-2.5-pro` | `gemini` | API Key |
| `minimax` | `MiniMax-M2.7`、`MiniMax-M2.5` | `minimax` | API Key |
| `glm` | `glm-5`、`glm-4.7` | `glm` | API Key |
| `openrouter` | `anthropic/claude-sonnet-4`、`openai/gpt-4o` | `openrouter` | API Key |

!!! tip "OpenRouter —— 一把 Key，300+ 模型"
    `openrouter` 是统一网关：一个 `OPENROUTER_API_KEY` 即可路由到任意厂商。
    其模型名采用 `vendor/slug` 形式（如 `google/gemini-2.5-pro`）。`/model`
    选择器会拉取 OpenRouter 的完整在线目录，并支持输入关键字过滤。

### 特殊认证 provider

| 提供方 | 接口类型 | 认证方式 | 说明 |
|---|---|---|---|
| `claude_subscription` | `claude` | Claude 订阅 token | 优先自动探测本地订阅凭据，失败时可手动粘贴 `sk-ant-oat01-...` |
| `codex` | `codex` | OAuth | 读取本地 Codex OAuth 凭据并校验连通性 |

### Coding Plan provider

这些 provider 面向编码/规划类 endpoint。虽然它们的名字里带 `coding`，但在配置层面和普通 provider 没有区别，仍然可以作为 `agent.target` 或节点级 `model` 使用。

| 提供方 | 默认模型 | 接口类型 | 说明 |
|---|---|---|---|
| `alibaba_coding` | `qwen3-coder-plus` | `claude` | 阿里云 DashScope 的 Anthropic-compatible coding endpoint |
| `bigmodel_coding` | `glm-5.1` | `claude` | 智谱 BigModel 的 Anthropic-compatible coding endpoint |
| `zai_coding` | `glm-5.1` | `claude` | Z.AI 国际版的 Anthropic-compatible coding endpoint |
| `minimax_coding` | `MiniMax-M2.7` | `claude` | MiniMax 的 Anthropic-compatible coding endpoint |
| `kimi_coding` | `kimi-for-coding` | `claude` | Kimi 的 coding endpoint |

!!! tip "如何选择 coding plan provider"
    如果你更看重通用问答、SQL 生成和成本控制，通常优先选择常规 provider。

    如果你希望默认模型更偏向规划、代码生成、结构化拆解，或者你会频繁使用 [计划模式](../cli/plan_mode.zh.md)，可以额外配置一个 `*_coding` provider，并按节点切换使用。

### 与环境变量的关系

所有 provider 的 `api_key` 都支持环境变量，例如：

```yaml
api_key: ${OPENAI_API_KEY}
```

对于 OpenAI、DeepSeek、Claude、Kimi、Qwen、Gemini、OpenRouter，配置向导会自动提示对应环境变量（如 `${OPENROUTER_API_KEY}`）。对于 `minimax`、`glm` 和各类 `*_coding` provider，你也可以在输入 API Key 时直接填入 `${MINIMAX_API_KEY}`、`${GLM_API_KEY}`、`${KIMI_API_KEY}`、`${DASHSCOPE_API_KEY}` 这类环境变量引用。

另外，当前实现会对少数模型自动补充固定参数覆盖：

- `kimi-k3`、`kimi-k2.6`：自动设置 `temperature: 1.0`、`top_p: 0.95`
- `qwen3-coder-plus`：自动设置 `temperature: 1.0`、`top_p: 0.95`

### 配置示例

=== "OpenAI"
```yaml
openai:
  type: openai
  base_url: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  model: gpt-5.2
```

=== "Anthropic Claude"
```yaml
claude:
  type: claude
  base_url: https://api.anthropic.com
  api_key: ${ANTHROPIC_API_KEY}
  model: claude-sonnet-4-5
```

=== "DeepSeek"
```yaml
deepseek:
  type: deepseek
  base_url: https://api.deepseek.com
  api_key: ${DEEPSEEK_API_KEY}
  model: deepseek-v4-flash
```

=== "Google Gemini"
```yaml
gemini:
  type: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
  api_key: ${GEMINI_API_KEY}
  model: gemini-2.5-flash
```

=== "Kimi (Moonshot)"
```yaml
kimi:
  type: kimi
  base_url: https://api.moonshot.cn/v1
  api_key: ${KIMI_API_KEY}
  model: kimi-k3
```

=== "Qwen (Alibaba)"
```yaml
qwen:
  type: openai
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  api_key: ${DASHSCOPE_API_KEY}
  model: qwen3-max
```

=== "OpenRouter"
```yaml
openrouter:
  type: openrouter
  base_url: https://openrouter.ai/api/v1
  api_key: ${OPENROUTER_API_KEY}
  model: anthropic/claude-sonnet-4   # vendor/slug 形式
```

=== "Alibaba Coding Plan"
```yaml
alibaba_coding:
  type: claude
  base_url: https://coding-intl.dashscope.aliyuncs.com/apps/anthropic
  api_key: ${DASHSCOPE_API_KEY}
  model: qwen3-coder-plus
  temperature: 1.0
  top_p: 0.95
```

=== "Claude Subscription"
```yaml
claude_subscription:
  type: claude
  base_url: https://api.anthropic.com
  api_key: ${CLAUDE_CODE_OAUTH_TOKEN}
  model: claude-sonnet-4-6
  auth_type: subscription
```

=== "Codex"
```yaml
codex:
  type: codex
  base_url: https://chatgpt.com/backend-api/codex
  api_key: ${CODEX_OAUTH_TOKEN}
  model: codex-mini-latest
  auth_type: oauth
```

## Agentic Nodes

`agent.agentic_nodes` 用于配置 chat 和各类 subagent 的行为。

这部分配置会被用于：

- `chat`、`explore`、`gen_sql`、`gen_visual_report`、`gen_dashboard`、`scheduler` 等内置 agentic 节点
- 通过统一 agent TUI（`/agent` 或 `/subagent`）创建的自定义 subagent
- 把自定义名称 alias 到某个内置节点类的高级手工配置

### 常见字段

当前运行时会读取这些常见字段：

- `model`：引用 `agent.models` 中的 provider key
- `system_prompt`：subagent 名称 / 提示词模板基名
- `node_class`：要使用的节点实现，例如 `gen_sql`、`gen_visual_report`、`explore`、`gen_table`、`gen_skill`、`gen_dashboard`、`scheduler`
- `prompt_version`、`prompt_language`
- `agent_description`
- `tools`、`mcp`、`skills`
- `rules`
- `max_turns`
- `workspace_root`
- `scoped_context`
- `subagents`
- dashboard agent 使用的 `bi_platform`
- scheduler 节点使用的 `scheduler_service`

Dosi 是内置语义引擎，无需配置引擎选择。旧的 `agent.services.semantic_layer` 会被忽略并输出警告。

`scoped_kb_path` 已废弃。新配置使用共享的全局存储，并在查询时应用过滤，而不是为每个 subagent 持有独立 scoped KB 目录。

### `subagents` 委派控制

`subagents` 决定该节点是否暴露 `task()` 工具，以及允许委派到哪些 subagent 类型。

- `subagents: "*"`：允许委派到所有可发现的 subagent（排除自己）
- `subagents: explore, gen_sql`：只允许委派到指定子集
- 为空或省略时：
  - `chat` 默认是 `*`
  - 多数其他 agentic 节点默认是 `explore`
  - 显式写成空值时可禁用委派

### `scoped_context`

`scoped_context` 用来限制 subagent 在共享元数据和知识库里可见的范围：

```yaml
scoped_context:
  datasource: finance
  tables: mart.finance_daily, mart.finance_budget
  metrics: finance.revenue.daily_revenue
  sqls: finance.revenue.region_rollup
```

手工写 YAML 时请显式填写 `datasource`。统一 agent TUI 的 Custom Tab 向导会自动从当前数据库填充该值。

### 示例

```yaml
agent:
  agentic_nodes:
    chat:
      model: claude
      max_turns: 50
      subagents: "*"

    finance_report:
      node_class: gen_visual_report
      model: claude
      system_prompt: finance_report
      prompt_version: "1.0"
      prompt_language: en
      agent_description: "财务分析助手"
      tools: semantic_tools.*, db_tools.*, context_search_tools.list_subject_tree
      subagents: explore, gen_sql
      max_turns: 30
      scoped_context:
        datasource: finance
        tables: mart.finance_daily
        metrics: finance.revenue.daily_revenue
        sqls: finance.revenue.region_rollup

    sales_dashboard:
      node_class: gen_dashboard
      model: claude
      bi_platform: superset
      max_turns: 30

    semantic_modeling:
      model: claude
      max_turns: 30

    etl_scheduler:
      node_class: scheduler
      model: claude
      scheduler_service: airflow_prod
      max_turns: 30
```

> 上下文压缩（在长会话中自动运行的 minor/major 压缩，使会话保持在模型上下文窗口内）通过 `agent.compact` 配置。其工作方式与可调参数见[上下文压缩](compact.md)。

## SQL 只读模式 {#sql-read-only}

设置 `agent.sql_read_only`，可对 CLI、API、MCP 和工作流执行统一启用 SQL 只读限制：

```yaml title="agent.yml"
agent:
  sql_read_only: true  # 默认：false
```

启用后会拒绝 SQL 写入、DDL 和跨数据源结果传输，不受交互式权限档位影响。
CTE 或 `EXPLAIN` 中包含的写操作也会被拒绝。此配置无需插件，部署启用后，
单次请求不能将其关闭。

工具调用的交互式确认方式见[工具快捷执行](../cli/execution_command.zh.md)。

## 完整示例
```yaml title="datus-config.yaml"
agent:
  target: alibaba_coding
  models:
    openai:
      type: openai
      base_url: https://api.openai.com/v1
      api_key: ${OPENAI_API_KEY}
      model: gpt-5.2

    gemini:
      type: gemini
      base_url: https://generativelanguage.googleapis.com/v1beta
      api_key: ${GEMINI_API_KEY}
      model: gemini-2.5-flash

    claude:
      type: claude
      base_url: https://api.anthropic.com
      api_key: ${ANTHROPIC_API_KEY}
      model: claude-sonnet-4-5

    deepseek:
      type: deepseek
      base_url: https://api.deepseek.com
      api_key: ${DEEPSEEK_API_KEY}
      model: deepseek-v4-flash

    alibaba_coding:
      type: claude
      base_url: https://coding-intl.dashscope.aliyuncs.com/apps/anthropic
      api_key: ${DASHSCOPE_API_KEY}
      model: qwen3-coder-plus
      temperature: 1.0
      top_p: 0.95
```
