# Semantic Model Generation Guide

## Overview

The semantic model generation feature helps you create MetricFlow semantic models from database tables through an AI-powered assistant. The assistant analyzes your table structure and generates comprehensive YAML configuration files that define metrics, dimensions, and relationships.

## What is a Semantic Model?

A semantic model is a YAML configuration that defines:

- **Measures**: Metrics and aggregations (SUM, COUNT, AVERAGE, etc.)
- **Dimensions**: Categorical and time-based attributes
- **Identifiers**: Primary and foreign keys for relationships
- **Data Source**: Connection to your database table

## How It Works

Start Datus CLI with `datus --database <datasource>`, and begin with a subagent command:

```text
  /gen_semantic_model generate a semantic model for table <table_name>
```


### Interactive Generation

When you request a semantic model, the AI assistant:

1. Retrieves your table's DDL (structure)
2. Checks if a semantic model already exists
3. Generates a comprehensive YAML file
4. Validates the configuration using MetricFlow
5. Syncs it to the Knowledge Base after validation passes

### Generation Workflow

```text
User Request → DDL Analysis → YAML Generation → Validation → Storage
```

### Validation and Sync

The agent calls `validate_semantic()` before publishing. If validation fails, it edits the YAML and retries. Once validation passes, `end_semantic_model_generation` publishes the semantic model to the Knowledge Base automatically.

## Configuration

### Agent Configuration

Most configurations are built-in. In `agent.yml`, minimal setup is needed:

```yaml
agent:
  services:
    semantic_layer:
      metricflow: {}     # Key MUST equal the adapter type (e.g. `metricflow`).
                         # If `type:` is given, it must match the key; otherwise Datus raises a config error at startup.

  agentic_nodes:
    gen_semantic_model:
      model: claude      # Optional: defaults to configured model
      max_turns: 30      # Optional: defaults to 30
      semantic_adapter: metricflow   # Optional when only one semantic layer is configured
```

See [Semantic Layer Configuration](../configuration/semantic_layer.md) for the full set of options.

**Built-in configurations** (automatically enabled):
- **Tools**: Database tools, generation tools, and filesystem tools
- **Hooks**: Validation evidence tracking and Knowledge Base sync
- **MCP Server**: MetricFlow validation server
- **System Prompt**: Built-in template; the latest available version is used unless `prompt_version` is set
- **Workspace**: `~/.datus/data/{datasource}/semantic_models`

### Configuration Options

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `model` | No | LLM model to use | Uses default configured model |
| `max_turns` | No | Maximum conversation turns | 30 |

## Semantic Model Structure

### Basic Template

```yaml
data_source:
  name: table_name                    # Required: lowercase with underscores
  description: "Table description"

  sql_table: schema.table_name        # For databases with schemas
  # OR
  sql_query: |                        # For custom queries
    SELECT * FROM table_name

  measures:
    - name: total_amount              # Required
      agg: SUM                        # Required: SUM|COUNT|AVERAGE|etc.
      expr: amount_column             # Column or SQL expression
      create_metric: true             # Auto-create queryable metric
      description: "Total transaction amount"

  dimensions:
    - name: created_date
      type: TIME                      # Required: TIME|CATEGORICAL
      type_params:
        is_primary: true              # One primary time dimension required
        time_granularity: DAY         # Required for TIME: DAY|WEEK|MONTH|etc.

    - name: status
      type: CATEGORICAL
      description: "Order status"

  identifiers:
    - name: order_id
      type: PRIMARY                   # PRIMARY|FOREIGN|UNIQUE|NATURAL
      expr: order_id

    - name: customer
      type: FOREIGN
      expr: customer_id
```

## Summary

The semantic model generation feature provides:

- ✓ Automated YAML generation from table DDL
- ✓ Interactive validation and error fixing
- ✓ Automatic sync after validation passes
- ✓ Knowledge Base integration
- ✓ Duplicate prevention
- ✓ MetricFlow compatibility
