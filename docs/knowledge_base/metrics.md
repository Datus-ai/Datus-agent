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
search_metrics(query_text="daily active users")

# Execute metric query
query_metrics(
    metrics=["daily_active_users"],
    group_by=["platform", "country"],
    start_time="2024-01-01",
    end_time="2024-01-31"
)
```

**Metrics-First Strategy**: When user queries involve KPIs (e.g., "show me DAU by platform"), the agent will:
1. Search for matching metrics using `search_metrics`
2. Execute via `query_metrics` if found (preferred)
3. Fall back to ad-hoc SQL generation only if no metric exists

This ensures consistent metric definitions across the organization.

## Usage

With `--success_story`, this compatibility component runs the full Dosi-only `semantic_modeling` workflow, including any datasets and relationships required by the generated metrics. Existing YAML import remains a non-LLM compatibility operation.

### Basic Command

```bash
# From CSV (historical SQLs)
datus-agent bootstrap-kb \
    --datasource <your_datasource> \
    --components metrics \
    --success_story path/to/success_story.csv

# From YAML (semantic models)
datus-agent bootstrap-kb \
    --datasource <your_datasource> \
    --components metrics \
    --semantic_yaml path/to/semantic_model.yaml
```

### Key Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `--datasource` | ✅ | Database datasource | `sales_db` |
| `--components` | ✅ | Components to initialize | `metrics` |
| `--success_story` | ⚠️ | CSV file with historical SQLs and questions (required if no `--semantic_yaml`) | `success_story.csv` |
| `--semantic_yaml` | ⚠️ | Semantic model YAML file (required if no `--success_story`) | `semantic_model.yaml` |
| `--kb_update_strategy` | ❌ | Update strategy | `overwrite`/`incremental` |
| `--subject_tree` | ❌ | Predefined categories (comma-separated) | `Sales/Reporting/Daily,Finance/Revenue/Monthly` |
| `--pool_size` | ❌ | Concurrent thread count | `4` |

Combining `metrics` with `semantic_model` or `semantic_modeling` executes one full `semantic_modeling` run. Existing MetricFlow and OSI projects remain query-only for generated changes.

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

**Generated Tag Format (legacy MetricFlow files):**

Existing MetricFlow metric files store the subject_tree classification in `locked_metadata.tags` with the format `"subject_tree: {domain}/{layer1}/{layer2}"`. This shape is shown for reference only — such files remain queryable but can no longer be generated or imported:

```yaml
# Legacy MetricFlow shape — query-only, not importable
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

`--semantic_yaml` accepts Dosi/OSI semantic YAML only, and only in Dosi projects. MetricFlow YAML (`data_source:` / `metric:` documents) is rejected with an explicit error; migrate the project to Dosi and re-author the model with `semantic_modeling`.

## Data Source Formats

### CSV Format

```csv
question,sql
How many customers have been added per day?,"SELECT ds AS date, SUM(1) AS new_customers FROM customers GROUP BY ds ORDER BY ds;"
What is the total transaction amount?,SELECT SUM(transaction_amount_usd) as total_amount FROM transactions;
```

### YAML Format (Metrics Import)

Metric import reads the `metrics` collection of a Dosi/OSI semantic-model document — the same file the datasets live in:

```yaml
semantic_model:
  - name: transactions
    datasets:
      - name: transactions
        source: analytics.transactions
    metrics:
      - name: total_revenue
        description: "Total revenue from all transactions"
        expression: SUM(transactions.amount)
        dataset: transactions
```

Standalone MetricFlow `metric:` documents are legacy and no longer importable. See [semantic_model.md](semantic_model.md) for how to define semantic models.

## Summary

The Metrics component establishes a **semantic query layer** that transforms historical SQLs into standardized, executable metric definitions. Unlike traditional semantic layers that only serve as LLM references, Datus metrics can be directly queried through MetricFlow, eliminating the need for ad-hoc SQL generation for common KPIs.

Key differentiators:

- **Executable Metrics**: Query via `query_metrics` instead of generating SQL
- **Metrics-First Strategy**: Agent prioritizes metric queries over ad-hoc SQL
- **Embedded Definitions, Independent Execution**: Metric definitions live in the semantic model's `metrics` collection; querying them via `query_metrics` is a separate execution path that needs no ad-hoc SQL
- **Hierarchical Organization**: Subject tree taxonomy for discoverability

This approach ensures consistent metric definitions across teams while reducing query complexity and improving performance.
