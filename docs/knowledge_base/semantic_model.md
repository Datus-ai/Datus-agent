# Semantic Model

Starting from **version 0.2.4**, semantic models serve as **schema extensions** that enrich database tables with semantic information. They define table structures (columns, dimensions, measures, entities) to help the agent better understand data for ad-hoc query generation.

## Core Value

Enhances database schema understanding for better SQL generation:
- **Rich column descriptions**: Augment DDL comments with usage patterns and filter examples
- **Dimension & measure classification**: Explicitly mark columns as dimensions (for grouping) or measures (for aggregation)
- **Entity relationships**: Define foreign key relationships between tables for accurate joins
- **Query pattern insights**: Capture how columns are typically filtered (LIKE, IN, FIND_IN_SET, etc.) from historical queries

## How It Works

Semantic models define the foundational schema layer. Starting from version 0.2.4, they operate independently:

- **Semantic Models** (this document): Schema extensions with dimensions, measures, and entity relationships
  - Storage: `semantic_model` table in LanceDB
  - Purpose: Help agent understand table structures for ad-hoc SQL generation

- **Metrics** (see [metrics.md](metrics.md)): Business calculations built on semantic models
  - Storage: `metrics` table in LanceDB
  - Purpose: Standardized KPIs queryable via MetricFlow

Semantic models provide the building blocks (dimensions, measures) that metrics reference.

## Storage Structure

Semantic model objects are stored at field level:

```python
# Stored objects (kind field):
- "table": Table-level metadata
- "column": Column-level metadata with semantic flags
- "entity": Entity definitions for relationships

# Semantic flags for columns:
- is_dimension: Column used for grouping/filtering
- is_measure: Column used for aggregation
- is_entity_key: Column used for table joins
```

## Usage

**Prerequisites**: This command relies on [datus-semantic-metricflow](../adapters/semantic_adapters.md), install it first with `pip install datus-semantic-metricflow`.

### Basic Command

```bash
# From CSV (historical SQLs)
datus-agent bootstrap-kb \
    --namespace <your_namespace> \
    --components semantic_model \
    --success_story path/to/success_story.csv

# From YAML (existing semantic model files)
datus-agent bootstrap-kb \
    --namespace <your_namespace> \
    --components semantic_model \
    --semantic_yaml path/to/semantic_model.yaml
```

### Key Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `--namespace` | ✅ | Database namespace | `sales_db` |
| `--components` | ✅ | Components to initialize | `semantic_model` |
| `--success_story` | ⚠️ | CSV file with historical SQLs (required if no `--semantic_yaml`) | `success_story.csv` |
| `--semantic_yaml` | ⚠️ | Semantic model YAML file (required if no `--success_story`) | `semantic_model.yaml` |
| `--kb_update_strategy` | ✅ | Update strategy | `overwrite`/`incremental` |

## Data Source Formats

### CSV Format

```csv
question,sql
How many orders per customer?,SELECT customer_id, COUNT(*) as order_count FROM orders GROUP BY customer_id;
What is the average order amount?,SELECT AVG(amount) FROM orders WHERE status = 'completed';
```

The agent analyzes these SQLs to:
- Extract table structures
- Identify dimensions (GROUP BY columns) and measures (aggregated columns)
- Discover column usage patterns (WHERE clause filters)
- Infer entity relationships (JOIN patterns)

### YAML Format

MetricFlow-compatible semantic model YAML:

```yaml
data_source:
  name: orders
  description: "Order transactions table"

  identifiers:
    - name: order
      type: PRIMARY
      expr: id
      description: "Unique order identifier"

    - name: customer
      type: FOREIGN
      expr: customer_id
      description: "References customers table"

  dimensions:
    - name: status
      type: CATEGORICAL
      expr: status
      description: "Order status. Use IN for filtering. Common values: 'pending', 'completed', 'cancelled'"

    - name: created_at
      type: TIME
      expr: created_at
      description: "Order creation timestamp"

  measures:
    - name: amount
      agg: sum
      expr: amount
      description: "Order amount in USD"

    - name: order_count
      agg: count
      expr: id
      description: "Count of orders"
```

## How Semantic Models Enhance SQL Generation

When generating SQL, the agent uses semantic models to:

1. **Schema Linking**: Find relevant tables and columns using vector search on descriptions
2. **Proper JOIN Construction**: Use entity relationships to generate correct JOIN conditions
3. **Smart Filtering**: Apply usage patterns (e.g., use `FIND_IN_SET()` for comma-separated tag columns)
4. **Accurate Aggregation**: Select appropriate measures for aggregation functions

Example query flow:

```text
User: "Show me total revenue by customer status"

Agent process:
1. Search semantic objects: "revenue", "customer", "status"
2. Find: orders.amount (measure, agg=sum), customers.status (dimension)
3. Use entity relationship: orders.customer_id → customers.id
4. Generate SQL:
   SELECT c.status, SUM(o.amount) as revenue
   FROM orders o
   JOIN customers c ON o.customer_id = c.id
   GROUP BY c.status
```

## Integration with Context Search

Semantic models are searchable via `@subject` context command:

```bash
# In CLI chat mode
@subject <domain>/<layer1>/<layer2>

# Search for semantic objects
search_semantic_objects(query="customer revenue", kinds=["table", "column"])
```

## Summary

Semantic models extend database schemas with rich semantic information, enabling more accurate ad-hoc SQL generation. Starting from version 0.2.4, they operate independently from metrics:

Key characteristics:
- **Schema extensions**: Augment physical schemas with business semantics
- **Field-level storage**: Store tables, columns, entities separately for flexible search
- **Usage pattern enrichment**: Capture real-world query patterns from historical SQLs
- **Independent from metrics**: Focus on schema understanding, not metric calculations

This separation ensures semantic models remain focused on their core purpose: helping the agent understand your data structure for generating correct, efficient SQL queries.
