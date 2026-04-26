# Metrics Generation Guide

## Overview

The metrics generation feature helps you convert SQL queries into reusable MetricFlow metric definitions. Using an AI assistant, you can analyze SQL business logic and automatically generate standardized YAML metric configurations that can be queried consistently across your organization.

## What is a Metric?

A **metric** is a reusable business calculation built on top of semantic models. Metrics provide:

- **Consistent Business Logic**: One definition, used everywhere
- **Type Safety**: Validated structure and measure references
- **Metadata**: Display names, formats, business context
- **Composability**: Build complex metrics from simpler ones

**Example**: Instead of writing `SELECT SUM(revenue) / COUNT(DISTINCT customer_id)` repeatedly, define an `avg_customer_revenue` metric once.

## Important Limitations

**⚠️ Single Table Queries Only**

The current version **only supports generating metrics from single-table SQL queries**. Multi-table JOINs are not supported.

**Supported**:
```sql
SELECT SUM(revenue) FROM transactions WHERE status = 'completed'
SELECT COUNT(DISTINCT customer_id) / COUNT(*) FROM orders
```

**Not Supported**:
```sql
SELECT SUM(o.amount)
FROM orders o
JOIN customers c ON o.customer_id = c.id  -- ❌ JOIN not supported
```

## How It Works

Start Datus CLI with `datus --datasource <datasource>`, and use the metrics generation subagent:

```bash
/gen_metrics Generate a metric from this SQL: SELECT SUM(amount) FROM transactions, the coresponding question is total amount of all transactions
```

### Generation Workflow

```
User provides SQL and question → Agent analyzes logic → Finds semantic model → Reads measures →
Checks for duplicates → Generates metric YAML → Appends to file → Validates →
Runs dry-run SQL → Syncs to Knowledge Base
```

### Validation and Sync

Before publishing, the agent must pass both checks:

- `validate_semantic()` validates the semantic model and metric YAML.
- `query_metrics(..., dry_run=True)` verifies that MetricFlow can compile SQL for the generated metric.

After those checks pass, `end_metric_generation` syncs the generated metric to the Knowledge Base automatically.

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
    gen_metrics:
      model: claude      # Optional: defaults to configured model
      max_turns: 40      # Optional: defaults to 30
      semantic_adapter: metricflow   # Optional when only one semantic layer is configured
```

See [Semantic Layer Configuration](../configuration/semantic_layer.md) for the full set of options.

**Built-in configurations** (automatically enabled):
- **Tools**: Generation tools and filesystem tools
- **Hooks**: Validation evidence tracking and Knowledge Base sync
- **MCP Server**: MetricFlow validation server
- **System Prompt**: Built-in template; the latest available version is used unless `prompt_version` is set
- **Workspace**: `~/.datus/data/{datasource}/semantic_models`

### Configuration Options

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `model` | No | LLM model to use | Uses default configured model |
| `max_turns` | No | Maximum conversation turns | 30 |

### Subject Tree Categorization

Subject tree allows organizing metrics by domain and layers for better management. In CLI mode, include it in your question:

**Example with subject_tree:**
```text
/gen_metrics Generate a metric from this SQL: SELECT SUM(amount) FROM transactions, subject_tree: finance/revenue/transactions
```

**Example without subject_tree:**
```text
/gen_metrics Generate a metric from this SQL: SELECT SUM(amount) FROM transactions
```

When subject_tree is provided, the metric will be categorized accordingly (e.g., domain: finance, layer1: revenue, layer2: transactions). If not provided, the agent operates in learning mode and may suggest categories based on existing metrics in the Knowledge Base.



## Usage Examples

### Example 1: Simple Aggregation

**User Input**:
```text
/gen_metrics Generate a metric for total order count
```

**Agent Actions**:
1. Finds `orders.yml` semantic model
2. Reads file to discover `order_count` measure
3. Generates measure proxy metric:

```yaml
---
metric:
  name: total_orders
  description: "Total number of orders"
  type: measure_proxy
  type_params:
    measure: order_count
  locked_metadata:
    tags:
      - "subject_tree: finance/orders/core"
```

### Example 2: Revenue Metric

**User Input**:
```
/gen_metrics Create a metric from this SQL:
SELECT SUM(amount) as total_revenue FROM transactions WHERE status = 'completed'
```

**Agent Actions**:
1. Analyzes SQL aggregation (SUM with filter)
2. Finds or creates `transactions.yml` semantic model
3. Adds a filtered measure to the semantic model and generates a metric that references it:

```yaml
---
data_source:
  name: transactions
  measures:
    - name: completed_revenue
      description: "Revenue from completed transactions"
      agg: SUM
      expr: "CASE WHEN status = 'completed' THEN amount ELSE 0 END"
---
metric:
  name: total_revenue
  description: "Total revenue from completed transactions"
  type: measure_proxy
  type_params:
    measure: completed_revenue
  locked_metadata:
    tags:
      - "subject_tree: finance/revenue/transactions"
```

### Example 3: Count Metric

**User Input**:
```text
/gen_metrics Generate unique customer count metric:
SELECT COUNT(DISTINCT customer_id) FROM orders
```

**Agent Actions**:
1. Locates `orders.yml` semantic model
2. Identifies COUNT DISTINCT aggregation
3. Generates measure proxy metric:

```yaml
---
metric:
  name: unique_customer_count
  description: "Total number of unique customers"
  type: measure_proxy
  type_params:
    measure: unique_customers
  locked_metadata:
    tags:
      - "subject_tree: sales/customers/core"
```

## How Metrics Are Stored

### File Organization

Metrics are stored in separate files from semantic models:

- **Semantic Model**: `{table_name}.yml` - Contains data_source definition with measures and dimensions
- **Metrics**: `metrics/{table_name}_metrics.yml` - Contains metric definitions

**Semantic Model File** (`transactions.yml`):
```yaml
data_source:
  name: transactions
  sql_table: transactions
  measures:
    - name: revenue
      agg: SUM
      expr: amount
    - name: transaction_count
      agg: COUNT
      expr: "1"
  dimensions:
    - name: transaction_date
      type: TIME
```

**Metrics File** (`metrics/transactions_metrics.yml`):
```yaml
metric:
  name: total_revenue
  description: "Total revenue from all transactions"
  type: measure_proxy
  type_params:
    measure: revenue

---
metric:
  name: total_transactions
  description: "Total number of transactions"
  type: measure_proxy
  type_params:
    measure: transaction_count
```

**Why separate files?**
- Clear separation between schema definitions and business metrics
- Easier to manage metrics independently
- Multiple metrics use YAML document separator `---` within the metrics file

### Knowledge Base Storage

After validation and dry-run SQL succeed, the metric is synced to the Knowledge Base with:

1. **Metadata**: Name, description, type, domain/layer classification
2. **LLM Text**: Natural language representation for semantic search
3. **References**: Associated semantic model name
4. **Timestamp**: Creation date

## Summary

The metrics generation feature provides:

✅ **SQL-to-Metric Conversion**: Analyze SQL queries and generate MetricFlow metrics
✅ **Intelligent Type Detection**: Automatically selects the right metric type
✅ **Duplicate Prevention**: Checks for existing metrics before generation
✅ **Validation**: MetricFlow validation ensures correctness
✅ **Publish Gate**: Syncs only after semantic validation and dry-run SQL succeed
✅ **Knowledge Base Integration**: Semantic search for metric discovery
✅ **File Management**: Organizes metrics in dedicated files separate from semantic models
