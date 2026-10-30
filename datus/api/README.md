# Datus Agent API Server

A FastAPI-based HTTP server that provides REST API access to the Datus Agent functionality.

## Quick Start

### Foreground Mode
```bash
python datus/api/server.py
```

### Daemon Mode
```bash
# Start the server in background
python datus/api/server.py --daemon

# Check server status
python datus/api/server.py --action status

# Stop the server
python datus/api/server.py --action stop

# Restart the server
python datus/api/server.py --action restart
```

## Configuration

### Basic Options
```bash
python datus/api/server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info
```

### Agent Configuration
```bash
python datus/api/server.py \
  --datasource your_datasource \
  --config conf/agent.yml \
  --max_steps 20 \
  --workflow fixed
```

### Daemon Options
```bash
python datus/api/server.py \
  --daemon \
  --pid-file /custom/path/server.pid \
  --daemon-log-file /custom/path/server.log
```

## Default Paths

- **API Endpoint**: `http://localhost:8000`
- **PID File**: `~/.datus/run/datus-agent-api.pid`
- **Log File**: `logs/datus-agent-api.log`

## API Endpoints

Once the server is running, you can access:

- **Interactive Docs**: `http://localhost:8000/docs`
- **Health Check**: `GET /health`
- **Authentication**: `POST /auth/token`
- **Workflow Execution**: `POST /workflows/run`
- **Feedback Recording**: `POST /workflows/feedback`

## Usage Examples

### Health Check
```bash
curl http://localhost:8000/health
```

### Get Access Token
```bash
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&client_id=your_id&client_secret=your_secret"
```

### Run Workflow (Synchronous)
```bash
curl -X POST "http://localhost:8000/workflows/run" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Find all users from database",
    "datasource": "your_datasource",
    "workflow": "workflow_name",
    "mode": "sync"
  }'
```

### Run Workflow (Streaming)
```bash
curl -X POST "http://localhost:8000/workflows/run" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "task": "Find all users from database",
    "datasource": "your_datasource",
    "workflow": "workflow_name",
    "mode": "async"
  }'
```

### Record Feedback
```bash
curl -X POST "http://localhost:8000/workflows/feedback" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "your_task_id",
    "status": "success"
  }'
```

## Development Mode

For development with auto-reload:
```bash
python datus/api/server.py --reload
```

Note: `--daemon` and `--reload` are mutually exclusive.

## Troubleshooting

### Check if server is running
```bash
python datus/api/server.py --action status
```

### View logs
```bash
tail -f logs/datus-agent-api.log
```

### Force stop
```bash
pkill -f "datus/api/server.py"
```

### Bounded browser chat history

`GET /api/v1/chat/history/page?session_id=...` returns the latest 10 complete
persisted user turns, ordered oldest to newest. Pass `next_cursor` back as
`cursor` to read earlier turns; `has_more` indicates whether another page exists.
`limit` may be supplied (1–100, clamped to the configured server maximum).
Cursors are scoped to the session and retain a snapshot upper row id, so new
turns do not shift older pages. A removed or rewritten cursor anchor returns 409;
reload the latest history after compacting or rewinding a conversation.

Large tool results/errors carry `result.historyContent`; large tool arguments
carry `toolParamsHistoryContent`. Each reference contains `ref`, a bounded
`preview`, and `size_bytes`. Download the complete text through authenticated
`GET /api/v1/chat/history/content?session_id=...&reference=...`. The response is
an attachment with `Cache-Control: private, no-store`. References are resolved
from the same user's project/session storage on each request; no process-local
cache or new database migration is required. Changed or removed content returns
409/404. Historical tool calls are marked `historyOnly` for display rather than
execution; live resumed calls still follow the existing tool review flow.

The legacy `/api/v1/chat/history` endpoint retains its full-history contract.
Under `agent.api`, configure `default_history_turns` (10), `max_history_turns`
(50), and `history_inline_chars` (8000). These settings affect browser transport,
not the model's context or persisted session contents.
