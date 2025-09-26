# Workflow Orchestration

Workflow orchestration in Datus Agent is the process of defining, managing, and executing sequences of nodes to accomplish data analysis tasks. This guide explains how workflows are structured, configured, and executed to transform natural language requests into SQL queries and results.

## Core Concepts

### 1. Workflow Definition

A workflow is a sequence of nodes that:

- **Has a clear purpose**: Each workflow solves specific types of problems
- **Follows a logical order**: Nodes execute in a predefined sequence
- **Shares data**: Information flows between nodes through a shared context
- **Can be adaptive**: Some workflows can modify themselves during execution

### 2. Workflow Configuration (workflow.yml)

The `workflow.yml` file defines different workflow templates:

```yaml
workflow:
  reflection:
    - schema_linking
    - generate_sql
    - execute_sql
    - reflect
    - output

  fixed:
    - schema_linking
    - generate_sql
    - execute_sql
    - output

  metric_to_sql:
    - schema_linking
    - search_metrics
    - date_parser
    - generate_sql
    - execute_sql
    - output
```

## Built-in Workflow Types

### 1. Reflection Workflow

**Purpose**: Intelligent, self-improving SQL generation with adaptive behavior

**Node Sequence:**
```
Schema Linking → Generate SQL → Execute SQL → Reflect → Output
```

**Key Features:**
- **Self-assessment**: Reflect node evaluates results and decides next steps
- **Adaptive**: Can add new nodes dynamically based on execution results
- **Robust**: Handles complex queries that may require multiple attempts

**Best For:**
- Complex business queries
- Situations where perfect SQL isn't generated on first try
- Queries requiring domain knowledge

**Real-world Example:**

```
User: "Show me quarterly revenue trends by product category,
       excluding returns and considering seasonal adjustments"

Process:
1. Schema Linking: Finds orders, products, categories tables
2. Generate SQL: Creates initial quarterly revenue query
3. Execute SQL: Runs the query
4. Reflect: Notices missing seasonal adjustment logic
5. Add Fix Node: Corrects the query with seasonal calculations
6. Output: Final results with proper seasonal adjustments
```

### 2. Fixed Workflow

**Purpose**: Deterministic SQL generation with predictable execution path

**Node Sequence:**
```
Schema Linking → Generate SQL → Execute SQL → Output
```

**Key Features:**
- **Predictable**: Always follows the same execution path
- **Fast**: No reflection overhead
- **Simple**: Easy to understand and debug
- **Reliable**: Consistent behavior for well-understood problems

**Best For:**
- Simple, straightforward queries
- Well-defined data requirements
- Situations where you know exactly what you need
- Performance-critical applications

**Real-world Example:**

```
User: "List all customers from California"

Process:
1. Schema Linking: Finds customers table with state column
2. Generate SQL: Creates "SELECT * FROM customers WHERE state = 'CA'"
3. Execute SQL: Returns California customers
4. Output: Displays results
```

### 3. Metric-to-SQL Workflow

**Purpose**: Generate SQL from predefined business metrics

**Node Sequence:**
```
Schema Linking → Search Metrics → Date Parser → Generate SQL → Execute SQL → Output
```

**Key Features:**
- **Metric-driven**: Starts with business metrics rather than raw SQL
- **Time-aware**: Includes date parsing for temporal queries
- **Reusable**: Leverages existing metric definitions
- **Standardized**: Ensures consistent business calculations

**Best For:**
- Business intelligence and reporting
- Standardized KPI calculations
- Time-series analysis
- Dashboards and regular reports

**Real-world Example:**

```
User: "Show monthly active users for the last quarter"

Process:
1. Schema Linking: Finds user_activity table
2. Search Metrics: Finds "monthly_active_users" metric definition
3. Date Parser: Determines "last quarter" date range
4. Generate SQL: Creates query using the metric definition
5. Execute SQL: Runs the metric calculation
6. Output: Displays monthly active users by month
```

## Workflow Configuration

### Customizing Workflows

You can create custom workflow templates by modifying `workflow.yml`:

```yaml
workflow:
  custom_analytics:
    - schema_linking
    - search_metrics
    - generate_sql
    - execute_sql
    - compare
    - output

  data_exploration:
    - schema_linking
    - doc_search
    - generate_sql
    - execute_sql
    - reflect
    - output
```

### Workflow Parameters

Workflows can be configured with parameters:

```bash
# Use specific workflow
datus run --namespace your_db --task "your query" --plan reflection

# Use custom workflow
datus run --namespace your_db --task "your query" --plan custom_analytics
```

### Available Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--plan` | Workflow type to execute | `reflection` | `reflection`, `fixed`, `metric_to_sql`, custom |
| `--namespace` | Database namespace | Required | Any configured namespace |
| `--task` | Natural language query | Required | Any string |
| `--max_iterations` | Maximum reflection rounds | `3` | Integer |
| `--save_dir` | Directory to save workflow state | `./save` | Any valid path |

## Advanced Features

### Dynamic Node Addition

Workflows can add nodes dynamically during execution:

```python
# In reflection node
if needs_data_validation:
    workflow.add_node("validation", after="execute_sql")

if requires_aggregation_fix:
    workflow.add_node("fix", after="reflect")
```

### Context Management

Workflows maintain shared context across nodes:

```python
class WorkflowContext:
    sql_contexts: List[SQLContext]      # Generated SQL and results
    table_schemas: List[TableSchema]    # Database schema information
    metrics: List[BusinessMetric]       # Available business metrics
    reflections: List[Reflection]       # Reflection results and improvements
```

### Error Handling and Recovery

Workflows include built-in error handling:

- **Syntax Errors**: Automatically trigger fix nodes
- **Execution Failures**: Retry with modified queries
- **Context Limits**: Automatic context compression
- **Timeout Handling**: Graceful degradation for long-running queries

## Best Practices

### Workflow Selection

1. **Use Fixed for Simple Queries**
   - Direct data retrieval
   - Well-understood requirements
   - Performance-critical scenarios

2. **Use Reflection for Complex Analysis**
   - Multi-table joins
   - Business logic implementation
   - Uncertain or exploratory queries

3. **Use Metric-to-SQL for Standardized Reports**
   - KPI calculations
   - Regular business reports
   - Time-series analysis

### Performance Optimization

- **Parallel Execution**: Some nodes can run in parallel
- **Context Caching**: Reuse schema and metric information
- **Incremental Processing**: Avoid recomputing unchanged parts
- **Resource Management**: Monitor memory and CPU usage

### Debugging and Monitoring

```bash
# Enable debug mode for detailed logging
datus run --namespace your_db --task "your query" --debug

# Save workflow state for inspection
datus run --namespace your_db --task "your query" --save_dir ./debug_session

# Resume from saved state
datus resume --save_dir ./debug_session
```

## Conclusion

Workflow orchestration is the backbone of Datus Agent's intelligent SQL generation capabilities. By understanding the different workflow types and their appropriate use cases, you can leverage the full power of the system to solve complex data analysis problems efficiently and reliably.