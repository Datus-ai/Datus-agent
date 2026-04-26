# API Deployment

## Install

The API service ships inside the main `datus` package. Install dependencies once:

```bash
uv sync
```

This registers the `datus-api` console script.

## Launch

### Foreground (default)

```bash
datus-api --host 0.0.0.0 --port 8000

# Or via uv
uv run datus-api --host 0.0.0.0 --port 8000
```

### Daemon (background) Mode

Run the API server as a background daemon:

```bash
# Start in background
datus-api --daemon --port 8000

# Check status
datus-api --action status

# Stop
datus-api --action stop

# Restart
datus-api --action restart
```

All daemon commands also work with `uv run`, e.g. `uv run datus-api --daemon --port 8000`.

By default, the PID file is stored at `~/.datus/run/datus-agent-api.pid` and daemon logs are written to `logs/datus-agent-api.log`. You can override these paths:

```bash
datus-api --daemon --pid-file /var/run/datus-api.pid --daemon-log-file /var/log/datus-api.log
```

## CLI arguments

| Flag             | Default                  | Description |
|------------------|--------------------------|-------------|
| `--config`       | (auto-resolved)          | Path to `agent.yml` |
| `--datasource`    | `default`                | Datasource from `agent.yml` |
| `--output-dir`   | `./output`               | Directory for generated artifacts |
| `--log-level`    | `INFO`                   | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |
| `--host`         | `127.0.0.1`              | Bind address |
| `--port`         | `8000`                   | Bind port |
| `--reload`       | off                      | Auto-reload on file change (dev only) |
| `--workers`      | `1`                      | Number of uvicorn worker processes |
| `-v`, `--version`| —                        | Print version and exit |
| `--daemon`       | off                      | Run in background as a daemon |
| `--action`       | `start`                  | Daemon action: `start`, `stop`, `restart`, `status` |
| `--pid-file`     | `~/.datus/run/datus-agent-api.pid` | PID file path |
| `--daemon-log-file` | `logs/datus-agent-api.log` | Daemon log file path |

`--reload` and `--workers > 1` are mutually exclusive; the server will warn and fall back to a single worker.
`--daemon` and `--reload` are mutually exclusive.

## Environment variables

| Variable             | Equivalent flag | Notes |
|----------------------|-----------------|-------|
| `DATUS_CONFIG`       | `--config`      | Empty string triggers default lookup |
| `DATUS_DATASOURCE`    | `--datasource`   | Defaults to `default` |
| `DATUS_OUTPUT_DIR`   | `--output-dir`  | Defaults to `./output` |
| `DATUS_LOG_LEVEL`    | `--log-level`   | Defaults to `INFO` |
| `DATUS_CORS_ORIGINS` | —               | Comma-separated origins, default `*` |

When `DATUS_CORS_ORIGINS` is anything other than `*`, the CORS middleware enables `allow_credentials=true`.

## Configuration resolution priority

`datus-api` resolves the agent configuration file in this order:

1. `--config` flag (or `DATUS_CONFIG`) if explicitly set
2. `./conf/agent.yml` in the current working directory
3. `~/.datus/conf/agent.yml`

This matches the behavior of the `datus` CLI, so you do not need to pass `--config` when using the standard
`~/.datus` install location.

## Built-in endpoints

After startup the following are available regardless of router availability:

| Path             | Description |
|------------------|-------------|
| `GET /`          | Service banner with version pointer |
| `GET /health`    | Health check (no auth required) |
| `GET /docs`      | Swagger UI |
| `GET /openapi.json` | OpenAPI 3 spec |

## Quickstart with curl

```bash
# 1. Start the server
datus-api --port 8000 &

# 2. Health check
curl http://127.0.0.1:8000/health

# 3. List catalogs (identifies as user "alice")
curl -H 'X-Datus-User-Id: alice' \
  'http://127.0.0.1:8000/api/v1/catalog/list'

# 4. Send a streaming chat
curl -N -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -H 'X-Datus-User-Id: alice' \
  -d '{"message": "How many users signed up last week?"}'
```

## Production notes

- Run behind a reverse proxy (nginx/traefik) and terminate TLS upstream.
- Disable response buffering for SSE endpoints on your reverse proxy so events are delivered without delay.
- When running multiple workers, enable sticky sessions at the proxy level so SSE resume requests land on the
  worker that is still holding the task.
