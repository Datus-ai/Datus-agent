# Workflow Orchestration

Workflow orchestration in Datus Agent is the process of defining, managing, and executing sequences of nodes to accomplish data analysis tasks. This guide explains how workflows are structured, configured, and executed to transform natural language requests into SQL queries and results.

## Core Concepts

### 1. Workflow Definition

A workflow is a sequence of nodes that:

- **Has a clear purpose**: Each workflow solves specific types of problems
- **Follows a logical order**: Nodes execute in a predefined sequence
- **Shares data**: Information flows between nodes through a shared context

### 2. Workflow Configuration

Datus provides several built-in workflow templates optimized for different use cases:

```yaml
workflow:
  fixed:
    - schema_linking
    - gen_sql
    - execute_sql
    - output

  gen_sql_agentic:
    - gen_sql
    - execute_sql
    - output
```

**Note**: These are built-in workflow templates. To customize workflows or create your own, you need to configure them in `agent.yml` (see [Customizing Workflows](#customizing-workflows) section below).

## Built-in Workflow Types

### 1. Fixed Workflow

**Purpose**: Deterministic SQL generation with predictable execution path

**Node Sequence:**
```
Schema Linking → Generate SQL → Execute SQL → Output
```

**Key Features:**

- **Predictable**: Always follows the same execution path
- **Fast**: Uses a direct generation and execution path
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

### 2. Gen SQL Agentic Workflow

**Purpose**: Generate SQL with database, semantic-layer, metric-search, and other configured tools.

**Node Sequence:**
```
Generate SQL → Execute SQL → Output
```

**Key Features:**

- **Tool-driven**: Uses function tools when metric, date, or documentation context is needed
- **Configurable**: Tool families can be enabled per agentic node
- **Interactive**: The model decides which available tools to call

**Best For:**

- Queries that need semantic metrics or reference SQL
- Temporal questions when `date_parsing_tools.*` is enabled
- Complex SQL generation that benefits from iterative tool use

**Real-world Example:**

```
User: "Show monthly active users for the last quarter"
```

The `gen_sql` agent can search metric definitions through
`context_search_tools.search_metrics` and parse temporal expressions through
`date_parsing_tools.parse_temporal_expressions` when those tools are enabled.

## Workflow Configuration

### Customizing Workflows

You can create custom workflow templates by adding them to your `agent.yml` configuration:

```yaml
agent:
  workflow:
    plan: custom_analytics  # Set your custom plan as default

    custom_analytics:
      - schema_linking
      - gen_sql
      - execute_sql
      - compare
      - output

    data_exploration:
      - schema_linking
      - gen_sql
      - execute_sql
      - output
```

### Advanced Workflow Features

#### Parallel Execution

Workflows support parallel node execution for improved performance:

```yaml
agent:
  workflow:
    plan: bird_para

    bird_para:
      - schema_linking
      - parallel:
        - gen_sql
        - gen_sql
      - selection
      - execute_sql
      - output
```

#### Sub-workflows

You can define reusable sub-workflows:

```yaml
agent:
  workflow:
    plan: main_workflow

    main_workflow:
      - schema_linking
      - parallel:
        - subworkflow1
        - subworkflow2
      - selection
      - execute_sql
      - output

    subworkflow1:
      - gen_sql

    subworkflow2:
      - gen_sql
```

#### Sub-workflows with Custom Configuration

Sub-workflows can reference separate configuration files:

```yaml
agent:
  workflow:
    plan: multi_agent

    multi_agent:
      - schema_linking
      - parallel:
        - agent1_workflow
        - agent2_workflow
      - selection
      - output

    agent1_workflow:
      steps:
        - gen_sql
      config: multi/agent1.yaml

    agent2_workflow:
      steps:
        - gen_sql
      config: multi/agent2.yaml
```

### Workflow Parameters

Workflows can be configured with parameters:

```bash
# Use specific workflow
datus-agent run --datasource <your_datasource> --task "your query" --task_db_name <database> --workflow fixed

# Use custom workflow
datus-agent run --datasource <your_datasource> --task "your query" --task_db_name <database> --workflow custom_analytics
```

### Available Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--workflow` | Workflow type to execute | `fixed` | `fixed`, `chat_agentic`, `gen_sql_agentic`, custom |
| `--datasource` | Database datasource | Required | Any configured datasource |
| `--task_db_name` | Target database name for the task | Required | Any configured database name |
| `--task` | Natural language query | Required | Any string |
| `--max_steps` | Maximum workflow steps | `20` | Integer |


## Best Practices

### Workflow Selection

**Use Fixed for Simple Queries**

- Direct data retrieval
- Well-understood requirements
- Performance-critical scenarios

**Use Gen SQL Agentic for Tool-assisted Queries**

- Metric and semantic-layer discovery
- Temporal queries with date parsing enabled
- Iterative SQL generation


### Debugging and Monitoring

```bash
# Enable debug mode for detailed logging
datus-agent --debug run --datasource <your_datasource> --task "your query" --task_db_name <database>

# Save LLM input/output traces for inspection
datus-agent --save_llm_trace run --datasource <your_datasource> --task "your query" --task_db_name <database>
```

## Conclusion

Workflow orchestration is the backbone of Datus Agent's intelligent SQL generation capabilities. By understanding the different workflow types and their appropriate use cases, you can leverage the full power of the system to solve complex data analysis problems efficiently and reliably.
