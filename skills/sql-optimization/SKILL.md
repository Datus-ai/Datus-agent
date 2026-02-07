---
name: sql-optimization
description: Optimize SQL queries for better performance with index suggestions, query rewriting, and execution plan analysis
tags: [sql, optimization, performance, database]
version: "1.0.0"
license: Apache-2.0
compatibility:
  datus: ">=0.2.0"
allowed_commands:
  - "python:scripts/*.py"
  - "sh:scripts/*.sh"
---

# SQL Optimization Skill

This skill helps optimize SQL queries by analyzing execution plans, suggesting indexes, and rewriting queries for better performance.

## Features

- **Query Analysis**: Analyze SQL queries and identify performance bottlenecks
- **Index Suggestions**: Recommend indexes based on query patterns
- **Query Rewriting**: Suggest optimized query alternatives
- **Execution Plan**: Parse and explain execution plans

## Usage

### Analyze a Query

```bash
python scripts/analyze_query.py --sql "SELECT * FROM orders WHERE status = 'pending'" --db-type sqlite
```

### Suggest Indexes

```bash
python scripts/suggest_indexes.py --sql-file queries.sql --output suggestions.json
```
