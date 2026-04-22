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

Datasource isolation is controlled separately by the `--database` CLI flag (or `DATUS_DATASOURCE` env var) and selects
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

## Next steps

- [Deployment](deployment.md) — install and launch the API server
- [Chat](chat.md) — chat endpoints and SSE streaming
- [Knowledge Base](knowledge_base.md) — KB bootstrap and platform doc endpoints with SSE
- [Models](models.md) — list available LLM models and provider metadata
