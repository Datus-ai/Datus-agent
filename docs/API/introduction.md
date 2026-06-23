# API Introduction

The Datus REST API exposes the agentic chat loop, knowledge-base explorer, database catalog, and semantic-model
management as HTTP endpoints. It is started via the `datus-api` command.

The HTTP service shares the same configuration, knowledge base, and agent capabilities as the `datus` CLI and the
`datus-mcp` MCP server — three entry points backed by one engine.

| Entry point | Best for |
|-------------|----------|
| **CLI** (`datus`) | Local interactive development |
| **MCP** (`datus-mcp`) | Embedding tools in another agent |
| **REST API** (`datus-api`) | Web frontends, services, automation |

## Authentication model

The open-source build ships a header-based identification scheme. There is no token; the caller identifies itself
by sending a `X-Datus-User-Id` header whose value matches `^[A-Za-z0-9_-]+$`. When omitted, requests run under a
default unscoped session; when present, the user id is used to isolate chat sessions and tool state per user.

```
X-Datus-User-Id: alice
```

SQL policies use a separate request principal. When `agent.sql_policy.enabled` is `true`, send business
scope fields in `X-Datus-Principal` as a JSON object, for example `{"market_code":"MKT300"}`. `X-Datus-User-Id`
does not populate SQL policy principal fields. See [SQL Policy](../configuration/sql_policy.md).
Do not include `user_id` in `X-Datus-Principal`; that field is reserved for `X-Datus-User-Id` and will be rejected.

Datasource isolation is controlled separately by the `--datasource` CLI flag (or `DATUS_DATASOURCE` env var) and selects
which datasource from `agent.yml` is used to load databases and knowledge.

## Response envelope

Almost every JSON response is wrapped in a generic `Result[T]` envelope:

```json
{
  "success": true,
  "data":    { ... },
  "errorCode":    null,
  "errorMessage": null
}
```

| Field          | Type     | Description |
|----------------|----------|-------------|
| `success`      | bool     | `true` on success, `false` otherwise |
| `data`         | object   | Endpoint-specific payload, `null` on error |
| `errorCode`    | string   | Stable machine-readable error code |
| `errorMessage` | string   | Human-readable error description |

Streaming endpoints (`/chat/stream`, `/chat/resume`) return `text/event-stream` instead of `Result`. See
[Chat](chat.md) for the SSE event grammar.

## Global URL prefix

All v1 endpoints live under `/api/v1`. Health check (`/health`) and the OpenAPI/Swagger UI (`/docs`, `/openapi.json`)
sit at the application root.

## Troubleshooting

Approaches for diagnosing a misbehaving server:

- **Hung request** — a request that never returns even though `GET /health` still responds is almost always a
  parked coroutine, not a blocked event loop. Send `SIGUSR1` to the server (`kill -USR1 <pid>`) to dump every
  live async task and the innermost frame each is suspended on. See
  [Debugging — SIGUSR1 task dump](debugging.md#sigusr1-task-dump).

## Next steps

- [Deployment](deployment.md) — install and launch the API server
- [Chat](chat.md) — chat endpoints and SSE streaming
- [Knowledge Base](knowledge_base.md) — KB bootstrap and platform doc endpoints with SSE
- [Debugging](debugging.md) — diagnose hung requests with the SIGUSR1 async task dump
