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
