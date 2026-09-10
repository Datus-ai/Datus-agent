# 可观测性

Datus 提供本地执行检查与外部 trace export，用于调试 agent run、workflow run、benchmark、模型调用和工具调用。

## 概览

| 范围 | 机制 | 适用场景 |
| --- | --- | --- |
| 当前 REPL turn | 按 `Ctrl+O` | 本地交互时查看工具调用、SQL、原始输出 |
| 本地文件 | `--save_llm_trace` 或 `save_llm_trace: true` | 不接入托管 tracing 系统时，调试精确 prompt 与模型输出 |
| 外部 trace | `agent.observability.tracing` | 跨运行关联 workflow、benchmark、chat、LiteLLM、OpenAI Agents SDK 和工具 span |

外部 trace export 只通过 `agent.observability.tracing.enabled: true` 启用。只设置 provider API key 不会自动打开 tracing。

## 实现方式

Datus observability 基于 OpenTelemetry trace export：

- Datus 在每个进程中创建一个 OpenTelemetry tracer provider。
- 每个配置的 adapter 都会在这个 provider 上挂载 exporter/span processor。
- 内置的 `langsmith`、`langfuse`、`datadog`、`braintrust`、`otlp` adapter 都发送 OpenTelemetry trace。
- 平台 adapter 是轻量 preset：负责解析平台 endpoint 和鉴权，然后复用共享 OTLP exporter。
- 基础 trace export 不需要安装 LangSmith、Langfuse 或 Datadog 等平台 SDK。
- OpenAI Agents SDK span 通过 OpenInference instrumentation 接入，并合并到 Datus trace tree。
- Datus 通过 OpenTelemetry baggage 和 span attributes 传播 trace 级别的 identity。

同一个进程中可以启用多个 adapter。同一批 span 可以同时发送到 LangSmith、Langfuse、Datadog、Braintrust 和通用 OTLP collector。

Tracing setup 每个进程只初始化一次。修改 tracing 配置或环境变量后，需要重启 Datus 进程。

## 本地观测

### REPL 内联 Trace

在 `datus` REPL 中，运行中或运行后按 `Ctrl+O` 可切换 verbose trace 详情。再次按下，或按 `q`，回到 compact 视图。

### 本地 YAML Trace

使用 `--save_llm_trace` 持久化模型输入输出：

```bash
uv run datus-agent --save_llm_trace run \
  --config conf/agent.yml \
  --datasource local_duckdb \
  --task_db_name duckdb-demo \
  --task "Summarize the tree table"
```

也可以在 custom model 条目上启用：

```yaml
agent:
  models:
    my-internal:
      type: openai
      base_url: https://internal.example.com/v1
      api_key: ${MY_KEY}
      model: internal-gpt-4
      save_llm_trace: true
```

Trace YAML 会写入 `{agent.home}/trajectory/...`。Workflow checkpoint 也保存在同一棵 trajectory 目录下；配置外部 observability 后，保存的 workflow metadata 中可能包含 `trace_id`、`trace_span_id`、`trace_run_id`、`trace_provider` 等稳定 trace 引用字段。

## 基础配置

最短的外部 tracing 配置会启用 tracing，并使用默认的 `langfuse` adapter：

```yaml
agent:
  observability:
    tracing:
      enabled: true
      capture_content: true
```

上面的配置等价于：

```yaml
agent:
  observability:
    tracing:
      enabled: true
      adapters:
        - type: langfuse
```

常用 tracing 字段：

```yaml
agent:
  observability:
    tracing:
      enabled: true
      service_name: datus-agent
      environment: local
      capture_content: true
      capture:
        prompts: true
        responses: true
        reasoning: true
        tool_definitions: true
        tool_args: true
        tool_results: true
        sql: true
        artifacts: true
      redact:
        enabled: true
        fields:
          - api_key
          - password
          - token
          - secret
```

`capture_content` 默认是 `true`，以保持当前调试体验。需要更严格的内容采集策略时，可以设置为 `false`，或在 `capture` 下覆盖单个字段。

## Langfuse

设置 `tracing.enabled: true` 且省略 `adapters` 时，Datus 默认使用 Langfuse adapter。

环境变量：

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://us.cloud.langfuse.com
```

配置：

```yaml
agent:
  observability:
    tracing:
      enabled: true
      service_name: datus-agent
      environment: local
      adapters:
        - type: langfuse
```

Datus 会根据 `LANGFUSE_PUBLIC_KEY` 和 `LANGFUSE_SECRET_KEY` 在内部生成 Langfuse Basic Auth header，不需要单独设置 `LANGFUSE_AUTH_STRING`。

## LangSmith

LangSmith 使用 `LANGSMITH_API_KEY` 或 `LANGCHAIN_API_KEY`；`LANGSMITH_PROJECT` 可选。

环境变量：

```bash
export LANGSMITH_API_KEY=lsv2_...
export LANGSMITH_PROJECT=datus-trace
export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

配置：

```yaml
agent:
  observability:
    tracing:
      enabled: true
      service_name: datus-agent
      environment: local
      adapters:
        - type: langsmith
```

不需要设置 `LANGSMITH_TRACING=true` 这类 legacy LangChain tracing 开关。Datus 通过 `agent.observability.tracing` 控制 trace export。

## Datadog

Datadog 默认使用本地 Agent 的 OTLP HTTP receiver。

Datadog Agent 配置：

```yaml
otlp_config:
  receiver:
    protocols:
      http:
        endpoint: localhost:4318
```

Datus 配置：

```yaml
agent:
  observability:
    tracing:
      enabled: true
      service_name: datus-agent
      environment: local
      adapters:
        - type: datadog
```

如果 Datus 不能访问默认 endpoint，可以显式配置：

```yaml
agent:
  observability:
    tracing:
      enabled: true
      service_name: datus-agent
      environment: local
      adapters:
        - type: datadog
          endpoint: http://127.0.0.1:4318/v1/traces
```

如果 Datus 运行在 Docker 中，而 Datadog Agent 运行在宿主机上，应使用容器内可访问的宿主机地址：

```yaml
agent:
  observability:
    tracing:
      enabled: true
      adapters:
        - type: datadog
          agent_host: host.docker.internal
          agent_port: 4318
```

本地 Agent export 场景下，Datadog API key 配在 Agent 上即可。只有当 Datus 直接发送到一个要求 Datadog API key header 的 endpoint 时，才需要为 Datus 配置 `DD_API_KEY` 或 `DATADOG_API_KEY`。

## 多后端

需要同一个 Datus 进程把 span 同时发送到多个后端时，显式配置多个 adapter：

```yaml
agent:
  observability:
    tracing:
      enabled: true
      service_name: datus-agent
      environment: local
      adapters:
        - type: langsmith
        - type: langfuse
        - type: datadog
          endpoint: http://127.0.0.1:4318/v1/traces
```

## 通用 OTLP

需要完全控制 endpoint/header 时，继续使用 `otlp`：

```yaml
agent:
  observability:
    tracing:
      enabled: true
      adapters:
        - type: otlp
          endpoint: https://collector.example/v1/traces
          headers:
            x-api-key: ${OTLP_API_KEY}
```

## 查找 Trace

运行任意被 trace 的路径：

```bash
uv run datus-agent benchmark \
  --config conf/agent.yml \
  --datasource bird_sqlite \
  --benchmark bird_dev \
  --benchmark_task_ids 14
```

在配置的 provider project 中查看 trace。Trace 名称按操作组织：

| 操作 | Trace 名称形态 |
| --- | --- |
| Workflow run | `workflow/<workflow>` |
| Benchmark task | `benchmark/<benchmark>/<context>/task-<id>` |
| 知识库初始化 | `bootstrap-kb/<datasource>/<components>` |
| Agent session | `agent/<node>` |

Tag 与 metadata 会包含 datasource、workflow、benchmark、task id、run id，以及可用时的 `agent.home`。

Datus 不会在 workflow metadata 中持久化后端特有的 UI URL。使用 `trace_id`、`trace_span_id`、`trace_run_id`、`trace_provider` 等稳定 metadata 字段，将 workflow checkpoint 关联到对应 backend trace。

## 模型调用与 Available tools

每次模型调用有独立的本地 `model_call_id`。对应 generation 每轮保存实际解析出的完整工具定义，包括名称、描述、参数 schema，以及 `tool_choice`、`parallel_tool_calls`，不使用 hash 或前轮引用。SDK 禁用的工具不会出现在列表中。摘要调用标记为 `phase=compact_summary`，摘要没有工具不表示主任务丢失了工具。

工具通过 OpenInference 的 `llm.tools.<index>.tool.json_schema` 和 `llm.invocation_parameters` JSON 中的 `tools` 上报。Langfuse 会将其解析到 generation 的 `input.tools` 及 **Available tools** 展示中。选中 generation 后查看 Available tools，或切换 Input 为 JSON。工具执行节点表示实际执行过的工具；Available tools 表示该轮提供给模型的工具。Anthropic 的原始 `input_schema` 会保留，同时增加 `parameters` 映射以兼容展示，不改变发给模型的请求。

共用 tracing 层会将 LiteLLM 的流式 Response 外壳转换为带 `tool_calls` 的 assistant 消息。OpenInference 的 `input.value` 和 `output.value` 使用包含 `messages` 的 JSON 对象，同时保留索引化消息和用量字段。调用参数中的工具定义与索引化定义遵循相同的采集策略。所有 OTLP adapter 收到相同的 OpenInference 数据，不按后端单独转换输出，也不额外增加一套 GenAI 消息和工具数据。已通过实际 SDK 工具调用验证 Langfuse、LangSmith 和 Datadog 的数据接收结果。Langfuse 在 **Available tools** 展示工具定义，Datadog 在 `Metadata > tools` 展示。各平台的页面布局可以不同。

`capture.tool_definitions` 默认跟随 `capture_content`（通常为 true）。关闭后两处 OpenInference 字段均不包含工具定义，仍保留请求 ID、计数和执行状态。工具定义遵循 tracing 脱敏设置。`datus.llm.tools_capture_state` 区分 `complete`、`redacted`、`disabled`、`truncated`、`failed`、`not_observable`；实际空列表为 complete、数量为零。每轮按 1 MiB 的定义采集预算及 1,536 个索引属性的预算选取工具，然后写入两处 OpenInference 字段；检测到 OTel 属性丢弃或字符串截断时标记不完整，并记录独立的成功采集数量。Datus 的 OTel span 属性数量上限为 4,096，下游采集器可能有其他限制。

采集边界为 `sdk_request`：SDK 完成工具筛选和转换之后，LiteLLM 的供应商适配或网关继续变换之前。它证明 Datus 在此边界提供了哪些工具。验证模型服务最终收到了什么，需要通过返回的 request ID 查询服务端请求日志。Tracing 不影响工具是否可调用。

### 模型服务请求 ID

每个 generation 的 metadata 保存可观测到的 `datus.llm.request_id`，以及 `request_id_source`、`request_id_issuer`、`request_id_status`。ID 原样取自选定响应头或 SDK 文档字段，不使用响应 body ID、LiteLLM completion ID 或本地生成 ID 兜底。确认属于供应商的 ID 同时写入 `provider_request_id`；未确认的地址使用 `remote_request_id` 和 issuer `unknown`。

流式响应在收到 headers 后立即记录 ID，因此后续取消或解析失败仍可关联请求。Codex 直接文本/JSON 调用在认证刷新重试时生成独立调用记录，并通过 `retry_of` 关联。`request_id_coverage=adapter_visible_response` 明确不包含 SDK 或网关内部不可见的重试；拿不到原始值时保留 absent 或 not_observable。错误标记 `failure_stage=before_response` 或 `after_response`，建连失败不会记录响应头耗时。本功能不注入 traceparent 请求头。

已确认自管端点响应头语义时，可配置：

```yaml
agent:
  observability:
    tracing:
      enabled: true
      remote_id_headers:
        gateway.example.com:
          request_id_header: x-upstream-request-id
          trace_id_header: x-upstream-trace-id
          gateway_request_id_header: x-gateway-request-id
          issuer: provider
```

示例声明选中的上游 ID 属于供应商；按实际语义可改为 `gateway` 或 `unknown`。映射仅读取当前 SDK 暴露的 headers，不能取回 SDK 未暴露的响应头。只记录选定 ID，不打印整份 headers。关闭 tracing 时映射仍用于日志。配置变更后重启进程。

### 日志事件

| 级别 | 内容 |
| --- | --- |
| INFO | `llm.started`、流式 `llm.response_received`、`llm.finished`：模型、本地和远端 ID、耗时、已有 usage、状态。`first_event_ms` 是首个 SDK 事件耗时，不一定是首个文本 token。 |
| DEBUG | `llm.tools` 每轮记录名称、数量、调用策略，不打印完整定义；MCP 连接尝试和初始化细节。 |
| INFO / WARNING | 主任务首次 `tools.available`；增减、schema 或调用策略变化时 `tools.changed`，工具减少为 WARNING。比较限定在同一逻辑操作、同一次 agent 执行内；同名并发子 Agent 分开比较。 |
| INFO | `tool.finished` 记录 hook 可观测到的工具结束、耗时、失败状态；SDK 提供 tool-call ID 时关联模型调用。参数和结果继续遵循 trace 内容开关。返回值无法判断成败且没有可观测的 SDK 错误状态时标记为 `returned`，不认定成功。 |
| INFO / WARNING | `compact.started` / `compact.finished` 记录模式、触发原因、已有条目和 token 估算、归档位置、`compact_id`。major compact 生成历史恢复指针但没有 `read_file` 时告警。 |
| WARNING | MCP 可恢复失败、`mcp.degraded`、`mcp.budget_exhausted` 记录服务器、尝试次数和原因。工具变化附带附近能力事件 ID；未证实的变化原因明确为 unknown。 |

可用的 session/trace/span ID 与 `run_id`、`model_call_id`、`compact_id` 关联日志和 trace。完整工具定义保存在 trace 中；不产生实际压缩的 compact 检查不打生命周期日志。重复配置和结果全文、初始化提示已减少；`logger.error` 不再强制附带堆栈，处理异常的边界通过 `logger.exception` 或 `exc_info` 显式记录。

旧的 `--save_llm_trace` YAML 文件保持现有格式；新增字段进入外部 tracing 与结构化日志，不会补写旧 trace 文件。
