# Business Metrics Intelligence

Starting from **version 0.2.4**, the Metrics component focuses on creating standardized, queryable business metrics as an independent semantic query layer. Metrics can be executed directly through MetricFlow, rather than serving solely as references for LLM SQL generation.

## Core Value

Solves common enterprise challenges:
- **Duplicate SQL queries**: Query metrics directly instead of rewriting similar SQL
- **Inconsistent definitions**: Standardize metric definitions across teams through executable specifications
- **Manual classification**: Organize metrics with hierarchical subject tree taxonomy
- **Ad-hoc SQL complexity**: Use semantic queries (`query_metrics`) instead of generating SQL for common metrics

## How It Works

Metrics are business-level calculations built on top of semantic models. Starting from version 0.2.4, they operate independently:

- **Metrics** (this document): Standardized KPIs queryable via MetricFlow
- **Semantic Models** (see [semantic_model.md](semantic_model.md)): Schema extensions for ad-hoc SQL generation

Both can be generated from historical SQLs, but metrics focus on reusable business logic while semantic models focus on schema understanding.

## Querying Metrics

Once metrics are defined, query them directly using MetricFlow tools:

```python
# In agent conversation or workflow
# Search for relevant metrics
search_semantic_objects(query="daily active users", kinds=["metric"])

# Execute metric query
query_metrics(
    metrics=["daily_active_users"],
    group_by=["platform", "country"],
    start_time="2024-01-01",
    end_time="2024-01-31"
)
```

**Metrics-First Strategy**: When user queries involve KPIs (e.g., "show me DAU by platform"), the agent will:
1. Search for matching metrics using `search_semantic_objects`
2. Execute via `query_metrics` if found (preferred)
3. Fall back to ad-hoc SQL generation only if no metric exists

This ensures consistent metric definitions across the organization.

## Usage

**Prerequisites**: This command relies on [datus-semantic-metricflow](../adapters/semantic_adapters.md), install it first with `pip install datus-semantic-metricflow`.

### Basic Command

```bash
# From CSV (historical SQLs)
datus-agent bootstrap-kb \
    --namespace <your_namespace> \
    --components metrics \
    --success_story path/to/success_story.csv

# From YAML (semantic models)
datus-agent bootstrap-kb \
    --namespace <your_namespace> \
    --components metrics \
    --semantic_yaml path/to/semantic_model.yaml
```

### Key Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `--namespace` | ✅ | Database namespace | `sales_db` |
| `--components` | ✅ | Components to initialize | `metrics` |
| `--success_story` | ⚠️ | CSV file with historical SQLs and questions (required if no `--semantic_yaml`) | `success_story.csv` |
| `--semantic_yaml` | ⚠️ | Semantic model YAML file (required if no `--success_story`) | `semantic_model.yaml` |
| `--kb_update_strategy` | ❌ | Update strategy | `overwrite`/`incremental` |
| `--subject_tree` | ❌ | Predefined categories (comma-separated) | `Sales/Reporting/Daily,Finance/Revenue/Monthly` |
| `--pool_size` | ❌ | Concurrent thread count | `4` |

### Subject Tree Categorization

Organizes metrics using hierarchical taxonomy: `domain/layer1/layer2` (e.g., `Sales/Reporting/Daily`)

**Two modes:**
- **Predefined**: Use `--subject_tree` to enforce specific categories
- **Learning**: Omit `--subject_tree` to reuse existing categories or create new ones

```bash
# Predefined mode example
--subject_tree "Sales/Reporting/Daily,Finance/Revenue/Monthly"

# Learning mode: omit --subject_tree parameter
```

**Generated Tag Format:**

When metrics are generated, the subject_tree classification is stored in `locked_metadata.tags` with the format `"subject_tree: {domain}/{layer1}/{layer2}"`:

```yaml
metric:
  name: daily_revenue
  type: simple
  type_params:
    measure: revenue
  locked_metadata:
    tags:
      - "Finance"
      - "subject_tree: Sales/Reporting/Daily"
```

**Important for YAML Import:**

When using `--semantic_yaml` to sync metrics from YAML files to lancedb, you must manually add the `locked_metadata.tags` with subject_tree format in your YAML file for successful categorization. The system will not automatically classify metrics imported from YAML - you need to include the tags yourself:

```yaml
metric:
  name: your_metric
  # ... other fields
  locked_metadata:
    tags:
      - "YourDomain"
      - "subject_tree: Domain/Layer1/Layer2"
```

## Data Source Formats

### CSV Format

```csv
question,sql
How many customers have been added per day?,"SELECT ds AS date, SUM(1) AS new_customers FROM customers GROUP BY ds ORDER BY ds;"
What is the total transaction amount?,SELECT SUM(transaction_amount_usd) as total_amount FROM transactions;
```

### YAML Format (Metrics Only)

When importing metrics from YAML files, the metric definition references an existing semantic model:

```yaml
metric:
  name: total_revenue
  description: "Total revenue from all transactions"
  type: simple
  type_params:
    measure: amount  # References measure from semantic model
  filter: "amount > 0"
  locked_metadata:
    tags:
      - "Finance"
      - "subject_tree: Finance/Revenue/Total"
```

**Note**: The underlying semantic model (`data_source` with dimensions/measures) should already exist. See [semantic_model.md](semantic_model.md) for how to define semantic models.

## Summary

The Metrics component establishes a **semantic query layer** that transforms historical SQLs into standardized, executable metric definitions. Unlike traditional semantic layers that only serve as LLM references, Datus metrics can be directly queried through MetricFlow, eliminating the need for ad-hoc SQL generation for common KPIs.

Key differentiators:
- **Executable Metrics**: Query via `query_metrics` instead of generating SQL
- **Metrics-First Strategy**: Agent prioritizes metric queries over ad-hoc SQL
- **Independent from Semantic Models**: Metrics operate as a separate query tool, not embedded in schema definitions
- **Hierarchical Organization**: Subject tree taxonomy for discoverability

This approach ensures consistent metric definitions across teams while reducing query complexity and improving performance.