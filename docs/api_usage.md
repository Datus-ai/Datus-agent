# Datus Agent FastAPI Service Documentation

## Overview

The Datus Agent FastAPI service wraps the original workflow functionality into HTTP APIs, supporting both synchronous and asynchronous query methods. Users can pass parameters (such as namespace, database, etc.) and natural language questions through HTTP requests to get SQL query results.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Service

```bash
# Method 1: Direct use of uvicorn
uvicorn datus.api.service:app --host 0.0.0.0 --port 8000

# Method 2: Use the provided startup script
python -m datus.api.server --host 0.0.0.0 --port 8000

# Development mode (auto-reload)
python -m datus.api.server --host 0.0.0.0 --port 8000 --reload
```

### 3. Access API Documentation

After starting the service, visit the following addresses to view the API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### 1. Execute Query (POST /query)

#### Synchronous Query Example

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Get user count",
    "namespace": "default",
    "database": "my_database",
    "query_type": "sync"
  }'
```

#### Asynchronous Query Example

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Get detailed information for all users",
    "namespace": "default",
    "database": "my_database",
    "query_type": "async"
  }'
```

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| query | string | Yes | Natural language query description |
| namespace | string | Yes | Database namespace |
| database | string | No | Specific database name |
| schema_name | string | No | Database schema name |
| domain | string | No | Business domain context |
| layer1 | string | No | Layer 1 context |
| layer2 | string | No | Layer 2 context |
| external_knowledge | string | No | Additional context or evidence |
| query_type | string | No | Query type: "sync" or "async" (default: "sync") |
| max_steps | integer | No | Maximum workflow steps (default: 10) |

#### Response Format

```json
{
  "task_id": "uuid",
  "status": "completed|failed|error|started",
  "sql": "SELECT COUNT(*) FROM users;",
  "result": [
    {"count": 1000}
  ],
  "metadata": {},
  "error": null,
  "execution_time": 2.5
}
```

### 2. Query Task Status (GET /task/{task_id})

Used to query the execution status of asynchronous tasks:

```bash
curl "http://localhost:8000/task/your-task-id"
```

### 3. Stream Task Updates (GET /task/{task_id}/stream)

Used to get real-time execution progress of asynchronous tasks:

```bash
curl "http://localhost:8000/task/your-task-id/stream"
```

Returns streaming data in Server-Sent Events (SSE) format.

### 4. Health Check (GET /health)

Check service health status:

```bash
curl "http://localhost:8000/health"
```

## Usage Examples

### Python Client Example

```python
import requests
import json

# Synchronous query
def sync_query(query, namespace, database=None):
    url = "http://localhost:8000/query"
    payload = {
        "query": query,
        "namespace": namespace,
        "database": database,
        "query_type": "sync"
    }
    
    response = requests.post(url, json=payload)
    return response.json()

# Asynchronous query
def async_query(query, namespace, database=None):
    url = "http://localhost:8000/query"
    payload = {
        "query": query,
        "namespace": namespace,
        "database": database,
        "query_type": "async"
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    task_id = result["task_id"]
    
    # Query task status
    status_url = f"http://localhost:8000/task/{task_id}"
    status_response = requests.get(status_url)
    return status_response.json()

# Stream task progress
def stream_task_progress(task_id):
    url = f"http://localhost:8000/task/{task_id}/stream"
    
    with requests.get(url, stream=True) as response:
        for line in response.iter_lines():
            if line:
                # Handle SSE format data
                if line.startswith(b'data: '):
                    data = json.loads(line[6:])  # Remove 'data: ' prefix
                    print(f"Event: {data['event_type']}, Data: {data['data']}")

# Usage example
if __name__ == "__main__":
    # Synchronous query example
    result = sync_query("Get user count", "default", "my_database")
    print("Sync query result:", result)
    
    # Asynchronous query example
    async_result = async_query("Analyze user behavior data", "default", "analytics_db")
    print("Async query result:", async_result)
```

### JavaScript Client Example

```javascript
// Synchronous query
async function syncQuery(query, namespace, database = null) {
    const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query: query,
            namespace: namespace,
            database: database,
            query_type: 'sync'
        })
    });
    
    return await response.json();
}

// Asynchronous query
async function asyncQuery(query, namespace, database = null) {
    const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query: query,
            namespace: namespace,
            database: database,
            query_type: 'async'
        })
    });
    
    return await response.json();
}

// Stream task progress
function streamTaskProgress(taskId) {
    const eventSource = new EventSource(`http://localhost:8000/task/${taskId}/stream`);
    
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        console.log('Event:', data.event_type, 'Data:', data.data);
        
        // Close connection when task is completed
        if (data.event_type === 'completed' || data.event_type === 'error') {
            eventSource.close();
        }
    };
    
    eventSource.onerror = function(event) {
        console.error('EventSource failed:', event);
        eventSource.close();
    };
}

// Usage example
(async () => {
    // Synchronous query
    const syncResult = await syncQuery("Get user count", "default", "my_database");
    console.log("Sync query result:", syncResult);
    
    // Asynchronous query
    const asyncResult = await asyncQuery("Analyze sales data trends", "default", "sales_db");
    console.log("Async query started:", asyncResult);
    
    if (asyncResult.status === 'started') {
        // Monitor task progress
        streamTaskProgress(asyncResult.task_id);
    }
})();
```

## Configuration

The service uses the same configuration files as the original Datus Agent:
- `conf/agent.yml`
- `~/.datus/conf/agent.yml`

Ensure proper configuration of database connections, LLM models, and other parameters before starting the service.

## Error Handling

The API uses standard HTTP status codes:
- 200: Success
- 400: Bad request parameters
- 404: Resource not found
- 500: Internal server error

Error response format:
```json
{
    "detail": "Error description message"
}
```

## Performance Considerations

1. **Synchronous vs Asynchronous**: 
   - Synchronous queries are suitable for simple, fast queries
   - Asynchronous queries are suitable for complex, time-consuming analysis tasks

2. **Concurrency Limits**: 
   - The service supports multiple concurrent requests
   - Recommend adjusting worker count based on server resources

3. **Timeout Settings**: 
   - Synchronous queries have default timeout limits
   - Asynchronous queries can handle long-running tasks

## Deployment Recommendations

### Production Environment Deployment

```bash
# Deploy using Gunicorn
pip install gunicorn
gunicorn datus.api.service:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Or use Docker (if Dockerfile exists)
docker build -t datus-api .
docker run -p 8000:8000 datus-api
```

### Nginx Reverse Proxy Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Support SSE
        proxy_buffering off;
        proxy_cache off;
    }
}
```