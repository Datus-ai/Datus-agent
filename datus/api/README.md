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
  --namespace your_namespace \
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
    "namespace": "your_namespace",
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
    "namespace": "your_namespace",
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