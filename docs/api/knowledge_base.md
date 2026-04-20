# Knowledge Base API

The knowledge base endpoints manage KB bootstrap operations with real-time progress streaming via Server-Sent Events
(SSE). All endpoints live under `/api/v1/kb`.

## KB Component Bootstrap

### `POST /api/v1/kb/bootstrap`

Start a knowledge base component bootstrap with SSE progress streaming. Components include metadata, semantic model,
metrics, external knowledge, and reference SQL.

**Body**:

| Field                | Type     | Default        | Notes |
|----------------------|----------|----------------|-------|
| `components`         | string[] | _(required)_   | Components to bootstrap: `metadata`, `semantic_model`, `metrics`, `ext_knowledge`, `reference_sql` |
| `strategy`           | string   | `incremental`  | `check` (inspect only), `overwrite` (rebuild), or `incremental` (append/update) |
| `schema_linking_type`| string   | `full`         | Metadata only: `table`, `view`, `mv`, or `full` |
| `catalog`            | string   | `""`           | Metadata catalog filter (Snowflake, StarRocks) |
| `database_name`      | string   | `""`           | Metadata database filter |
| `success_story`      | string?  | `null`         | Project-root-relative path to success-story CSV |
| `subject_tree`       | string[]?| `null`         | Predefined hierarchical categories |
| `sql_dir`            | string?  | `null`         | Project-root-relative directory with `.sql` files |
| `ext_knowledge`      | string?  | `null`         | Project-root-relative CSV for external knowledge |

**Response**: `text/event-stream`. See [SSE event format](#sse-event-format) below.

### `POST /api/v1/kb/bootstrap/{stream_id}/cancel`

Cancel a running KB bootstrap stream.

**Response**: `Result[dict]` with `data = { stream_id, cancelled: true|false }`.

---

## Platform Documentation Bootstrap

### `POST /api/v1/kb/bootstrap-docs`

Start a platform documentation bootstrap with SSE progress streaming. Ingests documentation from GitHub repos, websites,
or local directories into the platform doc vector store.

**Body**:

| Field              | Type     | Default      | Notes |
|--------------------|----------|--------------|-------|
| `platform`         | string   | _(required)_ | Platform name (e.g. `snowflake`, `duckdb`, `starrocks`) |
| `build_mode`       | string   | `overwrite`  | `check` (status only) or `overwrite` (rebuild) |
| `pool_size`        | int      | `4`          | Thread pool size (1–16) |
| `source_type`      | string?  | `null`       | `github`, `website`, or `local` |
| `source`           | string?  | `null`       | GitHub repo `owner/repo`, URL, or local path |
| `version`          | string?  | `null`       | Document version (auto-detected if omitted) |
| `github_ref`       | string?  | `null`       | Git branch / tag / commit for GitHub sources |
| `github_token`     | string?  | `null`       | GitHub API token for authenticated access |
| `paths`            | string[]?| `null`       | File/directory paths to include (GitHub only) |
| `chunk_size`       | int?     | `null`       | Target chunk size in characters |
| `max_depth`        | int?     | `null`       | Max crawl depth for website sources |
| `include_patterns` | string[]?| `null`       | File/URL patterns to include (regex) |
| `exclude_patterns` | string[]?| `null`       | File/URL patterns to exclude (regex) |

Only `platform` is required. All other fields fall back to the matching `DocumentConfig` in `agent.yml`
(`agent.document.<platform>`). See [Platform Documentation — Configure in agent.yml](../knowledge_base/platform_doc.md#configure-in-agentyml-optional) for YAML configuration.

**Response**: `text/event-stream`. See [SSE event format](#sse-event-format) below.

**Example**:

```bash
# Check existing store stats
curl -N -X POST http://localhost:8000/api/v1/kb/bootstrap-docs \
  -H "Content-Type: application/json" \
  -d '{"platform": "starrocks", "build_mode": "check"}'

# Full rebuild from GitHub
curl -N -X POST http://localhost:8000/api/v1/kb/bootstrap-docs \
  -H "Content-Type: application/json" \
  -d '{
    "platform": "starrocks",
    "source": "StarRocks/starrocks",
    "source_type": "github",
    "github_ref": "main",
    "paths": ["docs/en"],
    "build_mode": "overwrite"
  }'
```

### `POST /api/v1/kb/bootstrap-docs/{stream_id}/cancel`

Cancel a running platform doc bootstrap stream.

**Response**: `Result[dict]` with `data = { stream_id, cancelled: true|false }`.

---

## SSE Event Format

Both bootstrap endpoints stream progress as Server-Sent Events. Each event has the form:

```
id: <stream_id>
event: <stage>
data: <json>
```

The JSON payload is a `BootstrapKbEvent`:

| Field       | Type   | Description |
|-------------|--------|-------------|
| `stream_id` | string | Unique stream identifier |
| `component` | string | Component name (`metadata`, `semantic_model`, `platform_doc`, `all`, etc.) |
| `stage`     | string | Lifecycle stage (see below) |
| `message`   | string?| Human-readable status message |
| `error`     | string?| Error message if failed |
| `progress`  | object?| `{ total, completed, failed }` counters |
| `payload`   | object?| Additional data (component-specific results) |
| `timestamp` | string | ISO-format timestamp |

### Lifecycle Stages

| Stage             | Meaning |
|-------------------|---------|
| `task_started`    | Component bootstrap has started |
| `task_validated`  | Items validated, total count confirmed |
| `task_processing` | Component is processing items |
| `task_completed`  | Component completed successfully |
| `task_failed`     | Component failed |
| `item_started`    | Single item processing started |
| `item_completed`  | Single item completed |
| `item_failed`     | Single item failed |

### Example SSE Stream

```
id: abc-123
event: task_started
data: {"stream_id":"abc-123","component":"platform_doc","stage":"task_started","timestamp":"2025-01-01T00:00:00"}

id: abc-123
event: task_validated
data: {"stream_id":"abc-123","component":"platform_doc","stage":"task_validated","progress":{"total":1036,"completed":0,"failed":0},"timestamp":"2025-01-01T00:00:01"}

id: abc-123
event: item_completed
data: {"stream_id":"abc-123","component":"platform_doc","stage":"item_completed","message":"docs/en/intro.md","timestamp":"2025-01-01T00:00:02"}

id: abc-123
event: task_completed
data: {"stream_id":"abc-123","component":"platform_doc","stage":"task_completed","message":"Processed 1036 docs, 5200 chunks","payload":{"platform":"starrocks","version":"3.5.15","total_docs":1036,"total_chunks":5200},"timestamp":"2025-01-01T00:01:30"}
```

### Consuming SSE in JavaScript

```javascript
const evtSource = new EventSource('/api/v1/kb/bootstrap-docs', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ platform: 'starrocks', build_mode: 'overwrite' })
});

evtSource.addEventListener('task_completed', (e) => {
  const data = JSON.parse(e.data);
  console.log(`Done: ${data.payload.total_docs} docs, ${data.payload.total_chunks} chunks`);
  evtSource.close();
});

evtSource.addEventListener('task_failed', (e) => {
  const data = JSON.parse(e.data);
  console.error(`Failed: ${data.error}`);
  evtSource.close();
});
```

### Cancellation

To cancel a running bootstrap, POST to the cancel endpoint with the `stream_id` from the SSE events:

```bash
curl -X POST http://localhost:8000/api/v1/kb/bootstrap-docs/abc-123/cancel
```
