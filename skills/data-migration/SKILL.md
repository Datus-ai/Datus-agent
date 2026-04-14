---
name: data-migration
description: Migrate a table from a source database to a target database with type mapping, DDL generation, data transfer, and reconciliation
tags:
  - data-engineering
  - migration
  - cross-database
  - etl
  - reconciliation
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
---

# Data Migration

Use this skill when migrating data between different database engines (e.g., DuckDB to Greenplum or StarRocks). This skill covers the full lifecycle: source inspection, target DDL generation, data transfer, and post-migration reconciliation.

## When to use this skill

Activate when you need to:

- Migrate a single table from one database to another
- Rebuild a target table with correct cross-database type mapping
- Transfer data across database engines
- Reconcile source vs target after migration

## Prerequisites

- Both source and target databases must be configured and accessible
- Source database must support pandas query execution
- Target database must support DDL and DML operations

## Core workflow

### Phase 1: Inspect Source

- Use `describe_table(database=source)` to get source schema
- Use `read_query(database=source)` to get row count and sample data
- Identify column types, nullable columns, and primary key candidates
- Document the source schema for DDL generation

### Phase 2: Inspect Target

- Check if target schema/database exists using `list_tables(database=target)`
- If target table already exists, use `describe_table(database=target)` to compare
- Determine whether to create new or replace existing

### Phase 3: Build Target DDL

- Map source column types to target dialect (DuckDB types to Greenplum/StarRocks types)
- For StarRocks: determine DUPLICATE KEY columns and DISTRIBUTED BY HASH strategy
- Generate CREATE TABLE DDL appropriate for target database
- Execute DDL with `execute_ddl(sql, database=target)`

### Phase 4: Transfer Data

- Use `transfer_query_result(source_sql, source_database, target_table, target_database, mode)` to move data
- For fresh migration use `mode='replace'` (truncates target first)
- For incremental load use `mode='append'`
- Verify the transfer result (rows_transferred count)

### Phase 5: Reconcile

Run all reconciliation checks using `read_query(database=...)` on both source and target. Checks must be executed in this order:

1. **Row count** - Total row count comparison
2. **Null ratio** - Null count per column comparison
3. **Min/max** - Range comparison for numeric and date columns
4. **Distinct count** - Cardinality comparison for key columns
5. **Duplicate key** - Check for duplicate keys in target
6. **Sample diff** - Key-based row sample comparison (top 10)
7. **Numeric aggregate** - SUM/AVG comparison for numeric columns

### Phase 6: Report

Summarize migration results including:

- Source and target table names
- Rows transferred
- Each reconciliation check with pass/fail status
- Any issues or warnings

## Critical rules

- Always specify the `database` parameter explicitly in every tool call
- Source database is read-only: never execute DDL or write operations against it
- Target database is write-target: all DDL and writes go here
- Reconciliation is mandatory: never skip it after data transfer
- Report all reconciliation results even if some checks fail
- NEVER fall back to a different database if the target is unavailable — STOP and report the error
- Before starting, verify both source and target show `"available": true` in `list_databases()` output

## Checklist

- Migration checklist: [references/checklist.md](references/checklist.md)

## Output expectations

At minimum, return:

- source and target database/table names
- rows transferred
- reconciliation check results with pass/fail
- blocking issues that must be addressed
