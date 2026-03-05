# Reference SQL Intelligence

## Overview

Bootstrap-KB Reference SQL is a powerful component that processes, analyzes, and indexes SQL query files to create an intelligent searchable repository. It transforms raw SQL files with comments into a structured knowledge base with semantic search capabilities.

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
# Initialize Reference SQL component
datus-agent bootstrap-kb \
    --namespace <your_namespace> \
    --components reference_sql \
    --sql_dir /path/to/sql/directory \
    --kb_update_strategy overwrite
```

### Key Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `--namespace` | ✅ | Database namespace | `analytics_db` |
| `--components` | ✅ | Components to initialize | `reference_sql` |
| `--sql_dir` | ✅ | Directory containing SQL files | `/sql/queries` |
| `--kb_update_strategy` | ✅ | Update strategy | `overwrite`/`incremental` |
| `--validate-only` | ❌ | Only validate, don't store |  |
| `--pool_size` | ❌ | Concurrent processing threads, default value is 4 | `8` |
| `--subject_tree` | ❌ | Predefined subject tree categories for classification | `Analytics/User/Activity,Analytics/Revenue/Daily` |

### Subject Tree Categorization

Subject tree provides a hierarchical taxonomy for organizing SQL queries by domain and layers. This helps maintain consistent classification and improves query discoverability.

#### Format

Subject tree categories follow the format: `domain/layer1/layer2`

Example: `Analytics/User/Activity`, `Analytics/Revenue/Daily`, `Reporting/Sales/Monthly`

#### Usage Modes

**1. Predefined Mode (with --subject_tree)**

When you provide predefined categories, SQL queries will be classified using only these categories:

```bash
datus-agent bootstrap-kb \
    --namespace analytics_db \
    --components reference_sql \
    --sql_dir /path/to/sql/queries \
    --kb_update_strategy overwrite \
    --subject_tree "Analytics/User/Activity,Analytics/Revenue/Daily,Reporting/Sales/Monthly"
```

**2. Learning Mode (without --subject_tree)**

When no subject_tree is provided, the system operates in learning mode:
- Reuses existing categories from the Knowledge Base
- Analyzes similar SQL queries to suggest appropriate categories
- Creates new categories as needed based on query content
- Builds taxonomy organically over time

```bash
datus-agent bootstrap-kb \
    --namespace analytics_db \
    --components reference_sql \
    --sql_dir /path/to/sql/queries \
    --kb_update_strategy overwrite
```

**Benefits:**
- **Consistency**: Ensures SQL queries follow organizational taxonomy
- **Discoverability**: Makes queries easier to find through semantic search
- **Knowledge Management**: Organizes SQL expertise in a structured way
- **Pattern Recognition**: Groups similar queries for better reusability

## SQL File Format

### Expected Format

SQL files should use comment blocks to describe each query. Each SQL statement must be terminated with a semicolon (`;`).

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

### Format Requirements

1. **Semicolon Delimiter**: Each SQL statement must end with a semicolon (`;`)
2. **Comment Format**: Use SQL line comments (`--`) to describe queries (optional)
      - Comments immediately preceding a SQL statement will be associated with that query
      - Multiple comment lines are concatenated into a single description
      - Comments can be omitted if not needed
3. **SELECT Queries Only**: Only `SELECT` statements are processed and stored
4. **SQL Dialects**: Supports MySQL, Hive, and Spark SQL dialects
5. **Parameter Placeholders**: The following parameter styles are supported:
      - `#parameter#` - hash-delimited parameters
      - `:parameter` - colon-prefixed parameters
      - `@parameter` - at-sign parameters
      - `${parameter}` - shell-style parameters

### File Organization

- Place all SQL files (`.sql` extension) in a single directory
- Files are processed recursively
- Invalid SQL entries are logged to `sql_processing_errors.log` for review


## Summary

The Bootstrap-KB Reference SQL component transforms scattered SQL files into an intelligent, searchable knowledge base. It combines advanced NLP capabilities with robust SQL processing to create a powerful tool for SQL discovery and reuse.

**Key Features:**
- **Intelligent Organization**: Automatically categorizes SQL queries using subject tree taxonomy
- **Flexible Classification**: Support both predefined and learning modes for categorization
- **Semantic Search**: Find SQL queries using natural language descriptions
- **Knowledge Preservation**: Captures SQL expertise in a structured, searchable format
- **Automatic Context Retrieval**: Analyzes similar queries to suggest appropriate categories

By implementing reference SQL, teams can break down knowledge silos and build a collective SQL intelligence asset that grows over time.