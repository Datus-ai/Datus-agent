# Datus Agent FastAPI Service

A FastAPI wrapper for the Datus Agent workflow system that provides HTTP API endpoints for SQL query generation and execution.

## Features

- **Synchronous Queries**: Execute queries and wait for results
- **Asynchronous Queries**: Start long-running tasks and monitor progress
- **Streaming Updates**: Real-time task progress via Server-Sent Events
- **Health Monitoring**: Check service and database connectivity
- **Auto Documentation**: Swagger UI and ReDoc integration

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Start the Service

```bash
# Development mode
python -m datus.api.server --host 0.0.0.0 --port 8000 --reload

# Production mode
python -m datus.api.server --host 0.0.0.0 --port 8000 --workers 4
```

### API Documentation

Once started, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Execute SQL query (sync/async) |
| GET | `/task/{task_id}` | Get task status |
| GET | `/task/{task_id}/stream` | Stream task updates |
| GET | `/health` | Health check |
| GET | `/` | API information |

## Example Usage

### Synchronous Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Get total sales for last month",
    "namespace": "default",
    "database": "sales_db",
    "query_type": "sync"
  }'
```

### Asynchronous Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze customer behavior patterns",
    "namespace": "default", 
    "database": "analytics_db",
    "query_type": "async"
  }'
```

## Configuration

Uses the same configuration files as Datus Agent:
- `conf/agent.yml`
- `~/.datus/conf/agent.yml`

## Documentation

See [API Usage Documentation](../../docs/api_usage.md) for detailed usage examples, client code, and deployment instructions.