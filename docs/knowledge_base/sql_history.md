# SQL History Intelligence

## Overview

Bootstrap-KB SQL History is a powerful component that processes, analyzes, and indexes SQL query files to create an intelligent searchable repository. It transforms raw SQL files with comments into a structured knowledge base with semantic search capabilities.

## Core Value

### What Problem Does It Solve?

- **SQL Knowledge Silos**: SQL queries scattered across files without organization
- **SQL Reusability**: Difficulty finding existing queries for similar needs
- **Query Discovery**: No efficient way to search SQL by business intent
- **Knowledge Management**: SQL expertise locked in individual developers' minds

### What Value Does It Provide?

- **Intelligent Organization**: Automatically categorizes and classifies SQL queries
- **Semantic Search**: Find SQL queries using natural language descriptions
- **Knowledge Preservation**: Captures SQL expertise in a searchable format
- **Query Reusability**: Easily discover and reuse existing SQL patterns

## Usage

### Basic Command

```bash
# Initialize SQL history component
datus bootstrap-kb \
    --namespace your_namespace \
    --components sql_history \
    --sql_dir /path/to/sql/directory \
    --kb_update_strategy overwrite
```

### Key Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `--namespace` | ✅ | Database namespace | `analytics_db` |
| `--components` | ✅ | Components to initialize | `sql_history` |
| `--sql_dir` | ✅ | Directory containing SQL files | `/sql/queries` |
| `--kb_update_strategy` | ✅ | Update strategy | `overwrite`/`incremental` |
| `--validate-only` | ❌ | Only validate, don't store | `true`/`false` |
| `--pool_size` | ❌ | Concurrent processing threads | `8` |

## SQL File Format

### Expected Format

SQL files should use comment blocks to describe each query:

```sql
-- Daily active users count
-- Count unique users who logged in each day
SELECT
    DATE(created_at) as activity_date,
    COUNT(DISTINCT user_id) as daily_active_users
FROM user_activity
WHERE created_at >= '2025-01-01'
GROUP BY DATE(created_at)
ORDER BY activity_date;

-- Monthly revenue summary
-- Total revenue grouped by month and category
SELECT
    DATE_TRUNC('month', order_date) as month,
    category,
    SUM(amount) as total_revenue,
    COUNT(*) as order_count
FROM orders
WHERE order_date >= '2025-01-01'
GROUP BY DATE_TRUNC('month', order_date), category
ORDER BY month, total_revenue DESC;
```

## Advanced Features

### 1. Multi-Dialect SQL Validation

Support for multiple SQL dialects with automatic detection:

- **MySQL**: Standard MySQL syntax
- **Hive**: Hadoop Hive SQL dialect
- **Spark**: Apache Spark SQL syntax

### 2. Intelligent Classification

Automatically categorizes SQL queries into hierarchical structure:

```json
{
    "subject_tree": "analytics/user_analytics/activity_metrics",
    "tags": ["daily", "users", "engagement"]
}
```

### 3. Vector Search Capabilities

- **Semantic Search**: Find queries by meaning, not just keywords
- **Hybrid Search**: Combine vector search with traditional filtering
- **Relevance Scoring**: Results ranked by semantic relevance

### 4. Incremental Updates

- **Incremental Mode**: Add new queries to existing index
- **Overwrite Mode**: Complete rebuild of the index

## Best Practices

### 1. File Organization

```
/sql_queries/
├── user_analytics.sql
├── financial_reports.sql
├── product_metrics.sql
└── system_monitoring.sql
```

### 2. Comment Standards

```sql
-- Clear, descriptive title
-- Detailed business context and purpose
-- Important assumptions or business rules
SELECT
    column1,
    column2
FROM table_name
WHERE conditions;
```

### 3. Performance Optimization

```bash
# High-performance processing
datus bootstrap-kb \
    --namespace your_db \
    --components sql_history \
    --sql_dir /large_sql_directory \
    --kb_update_strategy incremental \
    --pool_size 16

# Validation only (fast check)
datus bootstrap-kb \
    --namespace your_db \
    --components sql_history \
    --sql_dir /new_sql_files \
    --validate-only
```

### 4. Maintenance Strategy

- **Regular Updates**: Add new SQL files incrementally
- **Quality Checks**: Use validate-only mode for new files
- **Index Optimization**: Periodic full rebuild for large updates

## Usage Examples

### Initial Setup

```bash
# First time setup with complete SQL directory
datus bootstrap-kb \
    --namespace production_db \
    --components sql_history \
    --sql_dir /company/sql_repository \
    --kb_update_strategy overwrite \
    --pool_size 8
```

### Adding New Queries

```bash
# Add new SQL files incrementally
datus bootstrap-kb \
    --namespace production_db \
    --components sql_history \
    --sql_dir /new_sql_queries \
    --kb_update_strategy incremental
```

### Validation

```bash
# Validate SQL files before processing
datus bootstrap-kb \
    --namespace production_db \
    --components sql_history \
    --sql_dir /untested_queries \
    --validate-only
```

## Summary

The Bootstrap-KB SQL History component transforms scattered SQL files into an intelligent, searchable knowledge base. It combines advanced NLP capabilities with robust SQL processing to create a powerful tool for SQL discovery and reuse.

By implementing SQL History, teams can break down knowledge silos and build a collective SQL intelligence asset that grows over time.