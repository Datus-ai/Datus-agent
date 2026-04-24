---
name: data-migration
description: "Activate when the gen_job agent detects that the source and target databases differ. Covers cross-database migration lifecycle - type mapping via adapter Mixin hints, DDL generation, data transfer via transfer_query_result, and mandatory reconciliation."
tags:
  - data-engineering
  - migration
  - cross-database
  - etl
  - reconciliation
version: "1.1.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_job
---

# Data Migration

Use this skill when the `gen_job` agent is asked to move data between DIFFERENT database engines (any source → any target: e.g., DuckDB ↔ Greenplum, MySQL ↔ StarRocks, Postgres ↔ ClickHouse, Hive → Iceberg). When `source_database == target_database` the job stays inside `gen_job` but this skill does NOT apply — in that case follow the `gen-table` and `table-validation` skills for intra-database ETL instead.

This skill covers the full lifecycle: source inspection, target DDL generation (using adapter Mixin hints), data transfer, and post-migration reconciliation.

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

### Phase 3: Build Target DDL (dialect-neutral)

- Call `get_migration_capabilities(database=target)` to read the target dialect's hard requirements, forbids, type_hints, and a reference example_ddl.
  - If the result reports `supported == false`, the target adapter has not implemented migration hints. Proceed in pure-LLM mode, relying on your own knowledge of that dialect.
- For OLAP-like targets (dialect_family indicates OLAP) call `suggest_table_layout(database=target, columns_json=...)` to get distribution / partition / order-by hints.
- Map source types → target types guided by `type_hints`. When ambiguous, prefer widening over narrowing.
- Draft the CREATE TABLE DDL.
- Call `validate_ddl(database=target, ddl=<draft>, target_table=<name>)`. Iterate until `errors == []`.
- Execute the DDL with `execute_ddl(sql, database=target)`.

### Phase 4: Transfer Data

- Use `transfer_query_result(source_sql, source_database, target_table, target_database, mode)` to move data
- For fresh migration use `mode='replace'` (truncates target first)
- For incremental load use `mode='append'`
- Verify the transfer result (rows_transferred count)

### Phase 5: Reconcile

Cross-database reconciliation (row count, null ratio, min/max, distinct
count, duplicate key, sample diff, numeric aggregates) is normally driven
by the `migration-reconciliation` validator skill via ValidationHook at
the end of the agent run — **provided
`agent.validation.skill_validators_enabled` is on** (the default). When
the validator is enabled, focus on correct transfer execution and let
the hook drive reconciliation; if it reports blocking failures they will
be injected back into this conversation so you can fix the transfer and
retry.

**When skill validators are disabled**, the hook cannot reconcile for
you. In that case you MUST run row-count parity and at least one
sanity-check query (e.g. distinct-count on the join key) manually using
`read_query(datasource=<source>)` vs `read_query(datasource=<target>)`
before declaring the migration done.

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
