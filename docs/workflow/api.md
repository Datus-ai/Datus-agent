# Workflow API

## Introduction

The Datus Agent Workflow API Service is a RESTful API that exposes the power of Datus Agent's natural language to SQL capabilities through HTTP endpoints. This service enables applications to integrate intelligent SQL generation, execution, and workflow management into their systems.

## Quick Start

### Starting the Service

```bash
# Start the API service
python -m datus.api.server --host 0.0.0.0 --port 8000

# Start with multiple workers
python -m datus.api.server --workers 4 --port 8000

# Start in daemon mode (background)
python -m datus.api.server --daemon --port 8000
```

## Authentication

### OAuth2 Client Credentials Flow

The API uses OAuth2 client credentials authentication:

#### 1. Get Authentication Token

```bash
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=your_client_id&client_secret=your_client_secret&grant_type=client_credentials"
```

#### 2. Use Token in Requests

```bash
curl -X POST "http://localhost:8000/workflows/run" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{"workflow": "fixed", "namespace": "your_db", "task": "Show me users"}'
```

### Configuration

Create `auth_clients.yml` to configure clients:

```yaml
clients:
  your_client_id: your_client_secret
  another_client: another_secret

jwt:
  secret_key: your-jwt-secret-key-change-in-production
  algorithm: HS256
  expiration_hours: 2
```

## API Endpoints

### Authentication

#### POST /auth/token

Obtain JWT access token.

**Request:**
```http
POST /auth/token
Content-Type: application/x-www-form-urlencoded

client_id=your_client_id&client_secret=your_client_secret&grant_type=client_credentials
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 7200
}
```

### Workflow Execution

#### POST /workflows/run

Execute a workflow to convert natural language to SQL.

**Common Request Parameters:**

| Parameter | Type     | Required | Description |
|-----------|----------|----------|-------------|
| `workflow` | string   | ✅ | Workflow name (nl2sql, reflection, fixed, metric_to_sql) |
| `namespace` | string   | ✅ | Database namespace |
| `task` | string   | ✅ | Natural language task description |
| `mode` | string   | ✅ | Execution mode (sync or async) |
| `task_id` | string   | ❌ | Custom task ID for idempotency |
| `catalog_name` | string   | ❌ | Database catalog |
| `database_name` | string   | ❌ | Database name |
| `schema_name` | string   | ❌ | Schema name |
| `current_date` | string   | ❌ | Reference date for time expressions |
| `subject_path` | string[] | ❌ | Business domain |
| `ext_knowledge` | string   | ❌ | Additional business context |

#### Synchronous Mode (mode: "sync")

**Request Headers:**
```
Authorization: Bearer your_jwt_token
Content-Type: application/json
```

**Request Body:**
```json
{
  "workflow": "nl2sql",
  "namespace": "your_database_namespace",
  "task": "Show me monthly revenue by product category",
  "mode": "sync",
  "catalog_name": "your_catalog",
  "database_name": "your_database"
}
```

**Response:**
```json
{
  "task_id": "client_20240115143000",
  "status": "completed",
  "workflow": "nl2sql",
  "sql": "SELECT DATE_TRUNC('month', order_date) as month, product_category, SUM(amount) as revenue FROM orders WHERE order_date >= '2023-01-01' GROUP BY month, product_category ORDER BY month, revenue DESC",
  "result": [
    {
      "month": "2023-01-01",
      "product_category": "Electronics",
      "revenue": 150000.00
    },
    {
      "month": "2023-01-01",
      "product_category": "Clothing",
      "revenue": 85000.00
    }
  ],
  "metadata": {
    "execution_time": 12.5,
    "nodes_executed": 5,
    "reflection_rounds": 0
  },
  "error": null,
  "execution_time": 12.5
}
```

#### Asynchronous Mode (mode: "async")

**Request Headers:**
```
Authorization: Bearer your_jwt_token
Content-Type: application/json
Accept: text/event-stream
Cache-Control: no-cache
```

**Request Body:**
```json
{
  "workflow": "nl2sql",
  "namespace": "your_database_namespace",
  "task": "Show me monthly revenue by product category",
  "mode": "async",
  "catalog_name": "your_catalog",
  "database_name": "your_database"
}
```

**Response (Server-Sent Events stream):**
```
Content-Type: text/event-stream

event: started
data: {"task_id": "client_20240115143000", "workflow": "nl2sql"}

event: progress
data: {"message": "Initializing workflow", "progress": 10}

event: node_progress
data: {"node": "schema_linking", "status": "processing", "progress": 25}

event: node_detail
data: {"node": "schema_linking", "description": "Analyzing user query and finding relevant tables", "details": {"tables_found": ["orders", "products"]}}

event: sql_generated
data: {"sql": "SELECT DATE_TRUNC('month', order_date) as month, product_category, SUM(amount) as revenue FROM orders GROUP BY month, product_category"}

event: execution_complete
data: {"status": "success", "rows_affected": 24, "execution_time": 2.1}

event: output_ready
data: {"result": [...], "metadata": {...}}

event: done
data: {"task_id": "client_20240115143000", "status": "completed", "total_time": 15.2}
```

### Feedback Submission

#### POST /workflows/feedback

Submit feedback on workflow execution quality.

**Request:**
```json
{
  "task_id": "client_20240115143000",
  "status": "success"
}
```

**Response:**
```json
{
  "task_id": "client_20240115143000",
  "acknowledged": true,
  "recorded_at": "2024-01-15T14:30:15Z"
}
```

## Workflow Types

### reflection
**Intelligent, self-improving SQL generation:**
- Includes reflection for error correction
- Can adapt and retry queries
- Best for complex or uncertain queries

### fixed
**Deterministic SQL generation:**
- Predictable execution path
- No adaptive behavior
- Best for well-understood queries

### metric_to_sql
**Generate SQL from business metrics:**
- Leverages predefined business metrics
- Includes date parsing for temporal queries
- Best for standardized business intelligence

## Configuration

### Server Configuration

```bash
python -m datus.api.server \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --reload \
  --debug
```

### Server Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--host` | Server host address | `127.0.0.1` |
| `--port` | Server port | `8000` |
| `--workers` | Number of worker processes | `1` |
| `--reload` | Auto-reload on code changes | `False` |
| `--debug` | Enable debug mode | `False` |
| `--daemon` | Run in background | `False` |

## Best Practices

### Security
- Use strong, unique JWT secret keys in production
- Rotate client credentials regularly
- Implement rate limiting for production deployments
- Use HTTPS in production environments

### Performance
- Use async mode for long-running queries
- Configure appropriate worker count based on expected load
- Monitor memory usage with multiple workers
- Implement client-side timeouts for sync requests

### Error Handling
- Always check response status codes
- Implement retry logic for transient failures
- Handle streaming disconnections in async mode
- Log detailed error information for debugging

## Conclusion

The Datus Agent Workflow API Service provides a powerful, flexible interface for integrating natural language to SQL capabilities into your applications. With support for multiple execution modes, real-time progress streaming, and comprehensive authentication, it enables developers to build intelligent data analysis applications.