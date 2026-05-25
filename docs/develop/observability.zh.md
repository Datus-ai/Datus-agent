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
