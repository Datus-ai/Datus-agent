# Web Tools

During a chat session the agent can reach the public web through a single pair of tools:

| Tool | Purpose |
|---|---|
| `web_search` | Search the web and return ranked results with snippets |
| `web_fetch` | Fetch one URL and return its readable text content |

The typical flow pairs them: `web_search` finds candidate pages (latest SQL dialect syntax, an error message, a product changelog), then `web_fetch` pulls the full text of the most promising URL. Both tools return the same result structure regardless of which backend served the request, so downstream behavior never depends on your LLM provider.

## Backend selection

The agent always sees the same two tools; what serves them is chosen automatically per active LLM provider:

- **Vendor-native** — if the active provider's API ships a built-in web search capability, Datus injects the vendor's hosted tool and skips the local implementation. No setup is needed.
- **Local fallback** — for every other provider:
    - `web_search` calls the [Tavily](https://tavily.com) search API — requires an API key (see below).
    - `web_fetch` downloads the page directly over HTTP and extracts the main text — no key, no third-party API.

Current support by provider:

| Provider | `web_search` | `web_fetch` |
|---|---|---|
| OpenAI (official `api.openai.com`) | Vendor-native | Local |
| ChatGPT Codex (official `chatgpt.com` backend) | Vendor-native | Local |
| Claude / Anthropic (official `api.anthropic.com`) | Vendor-native | Local |
| Everything else (DeepSeek, Qwen, self-hosted, OpenAI/Claude-compatible relays, …) | Local (Tavily) | Local |

Two caveats:

- Vendor-native search requires the **official** endpoint. If the provider entry points at a custom `base_url` (a relay, proxy, or self-hosted gateway), the hosted tool cannot be relied on and Datus falls back to the local Tavily backend automatically.
- `web_fetch` currently always runs on the local backend, for every provider.

Switching models with `/model` mid-session re-resolves the backends immediately.

## Setup

Only the local `web_search` backend needs configuration — a Tavily API key, resolved in this order:

1. `agent.document.tavily_api_key` in `agent.yml` (supports `${ENV_VAR}` substitution)
2. The `TAVILY_API_KEY` environment variable

```yaml title="agent.yml"
agent:
  document:
    tavily_api_key: ${TAVILY_API_KEY}
```

Without a key (and without vendor-native search), `web_search` is simply not exposed to the agent — a note is written to the log. `web_fetch` is always available.

## Tool reference

### `web_search`

| Argument | Type | Default | Description |
|---|---|---|---|
| `keywords` | list of strings | required | Search queries, e.g. `["StarRocks materialized view syntax"]` |
| `max_results` | int | `5` | Maximum results to return (1–20) |
| `include_domains` | list of strings | none | Restrict results to specific domains, e.g. `["docs.snowflake.com"]` |

Result:

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

| Argument | Type | Default | Description |
|---|---|---|---|
| `url` | string | required | Absolute `http(s)` URL to fetch |
| `max_chars` | int | `20000` | Truncate the extracted text to this many characters |

Result:

```json
{
  "url": "...",
  "title": "...",
  "content": "...",
  "truncated": false,
  "char_count": 12345
}
```

`truncated` reports whether `content` was cut off at `max_chars`; the agent can re-fetch with a larger limit when it needs the tail of a long page.

## Limits and safety

The local `web_fetch` backend enforces:

- **Public targets only (SSRF protection)** — requests to `localhost`, `*.local` / `*.internal` hostnames, private/loopback/link-local IP ranges, and cloud metadata addresses are refused.
- **HTML / plain text only** — other content types (PDFs, images, binaries) are rejected with an explanatory error.
- **10 MiB response cap** — responses that declare a larger body are refused before download.
- **30-second timeout** per request, following redirects.
- **Boilerplate stripping** — scripts, styles, navigation, headers/footers are removed; only the main text is returned.

## Permissions

Both tools belong to the `web_tool` permission category and are **allowed by default** in every built-in profile: they only read the public web and send nothing derived from your databases. To require confirmation or disable them, add a rule under `agent.permissions`:

```yaml title="agent.yml"
agent:
  permissions:
    rules:
      - tool: web_tool
        pattern: "*"          # or a single tool: web_search / web_fetch
        permission: deny      # allow | ask | deny
```

## Relation to `web_search_document`

Do not confuse `web_tool.web_search` with [`web_search_document`](../knowledge_base/platform_doc.md): the latter belongs to the knowledge-base platform-documentation tool group and serves as a web fallback when local document search comes up empty. Both use Tavily and share the same API-key configuration, but they are separate tools with separate permission categories.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `web_search` never appears in the tool list | No Tavily key resolvable and no vendor-native search — set `agent.document.tavily_api_key` or `TAVILY_API_KEY`. |
| `Web search is unavailable: set agent.document.tavily_api_key or TAVILY_API_KEY.` | Same as above — the key disappeared at call time. |
| `Refusing to fetch non-public URL target (SSRF protection)` | The URL points at localhost / a private network. Intentional; only public web targets are fetchable. |
| `Unsupported content-type '...'` | The URL is not an HTML/text page. Point `web_fetch` at a readable page instead. |
| `Response too large ... exceeds the 10,485,760-byte web_fetch limit.` | The page body exceeds 10 MiB. Fetch a smaller page (e.g. a specific article rather than an archive). |
