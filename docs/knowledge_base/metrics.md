# Business Metrics Intelligence

The Metrics component specifically focuses on automatically extracting and generating business metrics from historical SQL queries, establishing an enterprise-level metrics management system.

## Core Value

### What Problem Does It Solve?

- **Duplicate SQL queries**: Large numbers of similar business query needs in enterprises
- **Inconsistent metric definitions**: Different developers have different understandings of the same business metrics
- **Inefficient data analysis**: SQL needs to be rewritten for each query

### What Value Does It Provide?

- **Metric standardization**: Unified business metric definitions ensuring data consistency
- **Knowledge reuse**: Converting historical SQL experience into reusable metrics library
- **Intelligent querying**: Directly calling predefined metrics through natural language

## How It Works

### Data Hierarchy Structure
Data Table/Historical SQLs → Semantic Model → Business Metrics

#### 1. Semantic Model
- Defines table structure and business meaning
- Contains dimensions, measures, identifiers, and other metadata
- Provides semantic foundation for SQL queries

#### 2. Business Metrics
- Calculable metrics defined based on semantic models
- Contains metric name, description, calculation logic
- Supports complex business calculations

## Usage

### Basic Command

```bash
# Initialize metrics component from success story CSV
datus bootstrap-kb \
    --namespace your_namespace \
    --components metrics \
    --success_story path/to/success_story.csv \
    --metric_meta business_meta
```

```bash
# Initialize metrics component from semantic YAML
datus bootstrap-kb \
    --namespace your_namespace \
    --components metrics \
    --semantic_yaml path/to/semantic_model.yaml \
    --metric_meta business_meta
```

### Key Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `--namespace` | ✅ | Database namespace | `sales_db` |
| `--components` | ✅ | Components to initialize | `metrics` |
| `--success_story` | ⚠️ | A CSV file containing historical SQLs and questions (required if not using `--semantic_yaml`) | `benchmark/semantic_layer/success_story.csv` |
| `--semantic_yaml` | ⚠️ | Semantic model YAML file (required if not using `--success_story`) | `models/semantic_model.yaml` |
| `--metric_meta` | ✅ | Metric metadata | `ecommerce` configuration component in `agent.yml` |
| `--kb_update_strategy` | ✅ | Update strategy | `overwrite`/`incremental` |
| `--pool_size` | ❌ | Concurrent thread count | `4` |

## Data Source Formats

### CSV Format

```csv
question,sql
How many customers have been added per day?,"SELECT ds AS date, SUM(1) AS new_customers FROM customers GROUP BY ds ORDER BY ds;"
What is the total transaction amount?,SELECT SUM(transaction_amount_usd) as total_amount FROM transactions;
```

### YAML Format

```yaml
data_source:
  name: transactions
  description: "Transaction records"
  identifiers:
    - name: transaction_id
      type: primary
  dimensions:
    - name: transaction_date
      type: time
    - name: transaction_type
      type: categorical
  measures:
    - name: amount
      type: double
      agg: sum

metric:
  name: total_revenue
  description: "Total revenue from all transactions"
  constraint: "amount > 0"
```

## Advanced Features

### 1. Vector Search

Metrics support semantic search, finding relevant metrics through natural language descriptions:

```python
# Example search logic
search_results = metrics_store.search("customer retention metrics")
# Returns related customer retention metrics
```

### 2. Hierarchical Organization

Metrics are organized by business hierarchy:

```python
# Metric ID structure
f"{domain}_{layer1}_{layer2}_{semantic_model}_{metric_name}"
# Example: ecommerce_revenue_daily_orders_total_amount
```

### 3. Multi-strategy Updates

- **overwrite**: Completely rebuild metrics library
- **incremental**: Incrementally update new metrics

## Best Practices

### 1. Data Preparation

- Use high-quality success story data
- Ensure SQL queries represent typical business scenarios
- Provide clear business problem descriptions

### 2. Configuration Optimization

```bash
# Recommended configuration
datus bootstrap-kb \
    --namespace your_db \
    --components metrics \
    --success_story clean_success_stories.csv \
    --metric_meta business_meta \
    --kb_update_strategy overwrite \
    --pool_size 4
```

### 3. Maintenance Strategy

- Regularly update success story library
- Monitor metrics generation quality
- Clean up duplicate or outdated metrics

## Summary

The Bootstrap-KB Metrics component is the core functionality of Datus Agent, helping enterprises establish standardized data metrics systems through automated metric generation and management. It not only significantly improves data analysis efficiency but also ensures enterprise data consistency and reusability.

By properly using the Metrics component, enterprises can build a strong foundation for data-driven decision making.