# Observability

Datus provides local inspection tools and external trace export for debugging agent runs, workflow runs, benchmarks, and model/tool execution.

## Overview

| Scope | Mechanism | When to use |
| --- | --- | --- |
| Current REPL turn | Press `Ctrl+O` | Inspect tool calls, SQL, and raw outputs while interacting locally |
| Local files | `--save_llm_trace` or `save_llm_trace: true` | Debug exact prompts and model outputs without sending data to a hosted tracing system |
| External traces | `agent.observability.tracing` | Correlate full workflow, benchmark, chat, LiteLLM, OpenAI Agents SDK, and tool spans across runs |

External trace export is enabled only through `agent.observability.tracing.enabled: true`. Setting provider API keys alone does not turn tracing on.

## Implementation

Datus observability is built on OpenTelemetry trace export:

- Datus creates one OpenTelemetry tracer provider per process.
- Each configured adapter attaches an exporter/span processor to that provider.
- The built-in `langsmith`, `langfuse`, `datadog`, `braintrust`, and `otlp` adapters all emit OpenTelemetry traces.
- Platform adapters are lightweight presets. They resolve provider-specific endpoint and authentication settings, then reuse the shared OTLP exporter.
- Basic trace export does not require provider SDKs such as LangSmith, Langfuse, or Datadog SDKs.
- OpenAI Agents SDK spans are instrumented through OpenInference and merged into the Datus trace tree.
- Datus propagates trace-level identity through OpenTelemetry baggage and span attributes.

Multiple adapters can be enabled in the same process. The same spans can be sent to LangSmith, Langfuse, Datadog, Braintrust, and generic OTLP collectors.

Tracing setup is initialized once per process. Restart the Datus process after changing tracing configuration or environment variables.

## Local Inspection

### Inline REPL Trace

In the `datus` REPL, press `Ctrl+O` during or after a turn to toggle verbose trace details. Press it again, or `q`, to return to the compact view.

### Local YAML Traces

Use `--save_llm_trace` to persist model inputs and outputs:

```bash
uv run datus-agent --save_llm_trace run \
  --config conf/agent.yml \
  --datasource local_duckdb \
  --task_db_name duckdb-demo \
  --task "Summarize the tree table"
```

You can also enable it for a custom model entry:

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

Trace YAML files are written under `{agent.home}/trajectory/...`. Workflow checkpoints are saved in the same trajectory tree and, when external observability is configured, may include stable trace reference fields such as `trace_id`, `trace_span_id`, `trace_run_id`, and `trace_provider`.

## Basic Configuration

The shortest external tracing configuration enables tracing and uses the default `langfuse` adapter:

```yaml
agent:
  observability:
    tracing:
      enabled: true
      capture_content: true
```

The configuration above is equivalent to:

```yaml
agent:
  observability:
    tracing:
      enabled: true
      adapters:
        - type: langfuse
```

Common tracing fields:

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

`capture_content` defaults to `true` to preserve current debugging behavior. Set it to `false`, or override individual fields under `capture`, when you need stricter content collection.

## Langfuse

Langfuse is the default adapter when `tracing.enabled: true` is set and `adapters` is omitted.

Environment variables:

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://us.cloud.langfuse.com
```

Configuration:

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

Datus generates the Langfuse Basic Auth header from `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`. You do not need to set `LANGFUSE_AUTH_STRING`.

## LangSmith

LangSmith uses `LANGSMITH_API_KEY` or `LANGCHAIN_API_KEY`. `LANGSMITH_PROJECT` is optional.

Environment variables:

```bash
export LANGSMITH_API_KEY=lsv2_...
export LANGSMITH_PROJECT=datus-trace
export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Configuration:

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

You do not need to set legacy LangChain tracing switches such as `LANGSMITH_TRACING=true`. Datus controls trace export through `agent.observability.tracing`.

## Datadog

Datadog uses the local Agent OTLP HTTP receiver by default.

Datadog Agent configuration:

```yaml
otlp_config:
  receiver:
    protocols:
      http:
        endpoint: localhost:4318
```

Datus configuration:

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

If Datus cannot reach the default endpoint, configure it explicitly:

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

When Datus runs in Docker and the Datadog Agent runs on the host, use the host address that is reachable from the container:

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

For local Agent export, the Datadog API key belongs in the Agent configuration. Datus only needs `DD_API_KEY` or `DATADOG_API_KEY` when it sends to an endpoint that requires a Datadog API key header.

## Multiple Backends

Configure multiple adapters when the same Datus process should send spans to more than one backend:

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

## Generic OTLP

Use `otlp` when you want full control over endpoint and headers:

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

## Finding Traces

Run any traced path:

```bash
uv run datus-agent benchmark \
  --config conf/agent.yml \
  --datasource bird_sqlite \
  --benchmark bird_dev \
  --benchmark_task_ids 14
```

Open the configured provider project to view traces. Traces are named by operation:

| Operation | Trace name shape |
| --- | --- |
| Workflow run | `workflow/<workflow>` |
| Benchmark task | `benchmark/<benchmark>/<context>/task-<id>` |
| Knowledge bootstrap | `bootstrap-kb/<datasource>/<components>` |
| Agent session | `agent/<node>` |

Tags and metadata include datasource, workflow, benchmark, task id, run id, and `agent.home` when available.

Datus does not persist backend-specific UI URLs in workflow metadata. Use stable metadata fields such as `trace_id`, `trace_span_id`, `trace_run_id`, and `trace_provider` to correlate a workflow checkpoint with the corresponding backend trace.

## Model Calls and Available Tools

Each model invocation gets a local `model_call_id`. Its generation records the tools resolved for that invocation, including their names, descriptions, parameter schemas, `tool_choice`, and `parallel_tool_calls`. Definitions are recorded every generation, without hashes or references to earlier generations. Disabled SDK tools are excluded. A summary request is marked `phase=compact_summary`; its empty tool list does not mean the main task lost its tools.

Datus exports definitions using OpenInference `llm.tools.<index>.tool.json_schema` and the `tools` property of the `llm.invocation_parameters` JSON object. Langfuse ingests these into the generation's `input.tools` and its **Available tools** view. Select a generation in the trace, then inspect Available tools or switch Input to JSON. Tool execution spans show what ran; Available tools shows what the model was offered. Anthropic's original `input_schema` is retained and also exposed as `parameters` for this display. The actual model request is unchanged.

The shared tracing layer normalizes LiteLLM's streamed response envelope into assistant messages with `tool_calls`. OpenInference `input.value` and `output.value` contain a JSON object with `messages`, alongside the indexed message and usage attributes. Tool schemas in invocation parameters use the same capture policy as indexed definitions. All OTLP adapters receive this OpenInference representation; no backend-specific output conversion or parallel GenAI message/tool representation is added. Langfuse, LangSmith, and Datadog ingestion has been verified with actual SDK tool calls. Langfuse exposes definitions as **Available tools**; Datadog exposes them under `Metadata > tools`. Platform interfaces can organize these fields differently.

`capture.tool_definitions` defaults to the value of `capture_content` (normally true). Set it to false to omit schemas from both OpenInference fields while keeping request IDs, counts, and execution status. Definitions use the tracing redaction policy. `datus.llm.tools_capture_state` distinguishes `complete`, `redacted`, `disabled`, `truncated`, `failed`, and `not_observable`; an observed empty list is complete with count zero. Definitions are selected with a 1 MiB capture budget and a 1,536 indexed-attribute budget per generation, then exported in both OpenInference fields. Attribute eviction or value truncation is marked incomplete, with a separate captured count. Datus configures an OTel span attribute count limit of 4,096; downstream collectors may impose further limits.

The capture boundary is `sdk_request`: after SDK tool filtering/conversion, before provider-specific LiteLLM or gateway transformations. This proves which definitions Datus offered at that boundary. To verify what a remote model received, correlate the returned request ID with that service's request logs. Tracing never enables or disables runtime tools.

### Request IDs

Each generation's metadata contains `datus.llm.request_id`, its `request_id_source`, `request_id_issuer`, and `request_id_status` when observable. Values are copied verbatim from selected response headers or documented SDK properties. Response-body IDs, LiteLLM completion IDs, and generated local IDs are never substituted. A known provider's ID is also stored as `provider_request_id`; unverified endpoints use `remote_request_id` and issuer `unknown`.

Streaming IDs are captured as soon as response headers are available, so cancellation or parsing errors after that point retain them. Codex direct text/JSON authentication retries produce separate calls linked by `retry_of`. `request_id_coverage=adapter_visible_response` explicitly excludes invisible SDK and gateway retry attempts. Missing headers/properties remain absent or not observable. Errors record `failure_stage=before_response` or `after_response`; a connection failure does not claim response-header latency. No traceparent headers are injected.

For an endpoint whose response-header semantics are known, configure only the relevant ID headers:

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

This example declares that the selected upstream IDs belong to the provider. Use `issuer: gateway` or `unknown` where appropriate. The mapping reads headers exposed by the active SDK; it does not make otherwise hidden headers available. Only selected IDs are recorded, never the entire header set. Mappings also apply to logs when tracing is disabled. Restart the process after configuration changes.

### Lifecycle Logs

| Level | Events and purpose |
| --- | --- |
| INFO | `llm.started`, streaming `llm.response_received`, `llm.finished`: model, local/remote IDs, duration, available usage, status. `first_event_ms` means the first SDK event, not necessarily the first text token. |
| DEBUG | `llm.tools`: each invocation's tool names/count and selection policy, without full definitions. MCP connection attempts and initialization details. |
| INFO / WARNING | `tools.available` on the first task call; `tools.changed` on additions, schema/policy changes, or removals (WARNING). Comparisons are scoped to one logical operation and agent invocation, including separate parallel runs with the same agent name. |
| INFO | `tool.finished`: hook-observed tool completion, duration, failure status and model-call correlation where the SDK provides a tool-call ID. Tool arguments/results follow existing trace content controls. An opaque return value without an observable SDK error state is marked `returned`, not confirmed success. |
| INFO / WARNING | `compact.started` / `compact.finished`: mode, trigger, existing item/token estimates, archive pointer and `compact_id`. Missing `read_file` for major-compact history recovery raises a warning. |
| WARNING | MCP retry failures, `mcp.degraded`, and `mcp.budget_exhausted`: server, attempts and reason. Tool changes carry nearby capability event IDs; the cause remains `unknown` unless established rather than guessed. |

`run_id`, `model_call_id`, `compact_id`, and available session/trace/span IDs connect these records. Full model definitions live in traces, not ordinary log messages. No-op compact checks emit no lifecycle events. Repetitive configuration/result dumps and initialization notices are reduced; `logger.error` no longer implicitly attaches a traceback. Error-handling boundaries request one explicitly with `logger.exception` or `exc_info`.

Legacy `--save_llm_trace` YAML files retain their existing format; these generation-level fields are added to external tracing and structured logs, not retroactively to older trace files.
