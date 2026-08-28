# Workflow Introduction

Datus Agent Workflow is an intelligent system that transforms natural language questions into SQL queries and executes them against your databases. Think of it as having a data analyst that understands your business needs and can automatically generate the right SQL queries to get you the answers you need.

## Embedded Workflows

Datus Agent offers different embedded workflows for different needs:

### 1. Fixed Workflow
**Best for**: General SQL generation and execution

- **Performance**: Fast and predictable
- **Perfect for**: "List all customers from California" or "Show me total sales for 2023"
- **Use cases**: Direct data retrieval, aggregations, filtering, and multi-table queries

### 2. Metric-to-SQL Workflow
**Best for**: Standardized business reports

- **Consistency**: Uses predefined business metrics
- **Perfect for**: "Show monthly active users for the last quarter" or "Calculate customer churn rate"
- **Use cases**: KPI reporting, standardized metrics, business intelligence

## Nodes: The Building Blocks

Workflows are made up of specialized components called "[nodes](nodes.md)." Each node does one job really well:

- **[Schema Linking](nodes.md#schema-linking-node)**: Finds the right database tables for your question
- **[Generate SQL](nodes.md#generate-sql-node)**: Creates the SQL query
- **[Execute SQL](nodes.md#execute-sql-node)**: Runs the query against your database
- **[Output](nodes.md#output-node)**: Presents the final results to you

## Getting Started

### Basic Usage

```bash
# Ask a simple question
datus-agent run --datasource your_db --task_db_name analytics --task "Show me monthly sales"

# Use a specific workflow type
datus-agent run --datasource your_db --task_db_name analytics --task "Show me complex revenue trends" --workflow fixed

# Use business metrics
datus-agent run --datasource your_db --task_db_name analytics --task "Calculate customer lifetime value" --workflow metric_to_sql
```

### Via API

You can also use workflows through a REST API:

```python
import requests

response = requests.post(
    "http://localhost:8000/workflows/run",
    headers={"Authorization": "Bearer your_token"},
    json={
        "workflow": "fixed",
        "datasource": "your_db",
        "task": "Show me quarterly revenue trends",
        "mode": "sync"
    }
)

result = response.json()
print(result["sql"])    # The generated SQL query
print(result["result"]) # The query results
```

## Next Steps

- Learn about [workflow orchestration](orchestration.md) to understand how workflows are executed
- Explore the [API documentation](api.md) for programmatic access
- Deep dive into [individual nodes](nodes.md) to understand each component
