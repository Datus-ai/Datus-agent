---
name: data-migration
description: "Activate when the gen_job agent detects that the source and target databases differ. Covers cross-database transfer lifecycle - type mapping via adapter Mixin hints, DDL generation, data transfer via transfer_query_result, and lightweight reconciliation."
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

Use this skill when the `gen_job` agent is asked to move data between DIFFERENT database engines (any source → any target: e.g., DuckDB ↔ Greenplum, MySQL ↔ StarRocks, Postgres ↔ ClickHouse, Hive → Iceberg). When `source_database == target_database` the job stays inside `gen_job` and should use the `gen-table` path for intra-database ETL; `table-validation` is an automatic validator (`kind: validator`) invoked by `ValidationHook.on_end` for schema-contract checks, not a selectable workflow step.

This skill covers the full lifecycle: source inspection, target DDL generation (using adapter Mixin hints), data transfer, and post-transfer reconciliation.

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

Cross-database transfer sanity checks are normally driven by the
`transfer-reconciliation` validator skill via ValidationHook at the end
of the agent run — **provided
`agent.validation.skill_validators_enabled` is on** (the default). When
the validator is enabled, focus on correct transfer execution and let the
hook compare tool-reported row counts plus a small target-side sanity
query; if it reports blocking failures they will be injected back into
this conversation so you can fix the transfer and retry.

**When skill validators are disabled**, the hook cannot reconcile for
you. In that case you MUST compare the transfer tool's source/target row
counts and run at least one target-side sanity query before declaring the
transfer done.

### Phase 6: Report

Summarize migration results including:

- Source and target table names
- Rows transferred
- Reconciliation summary with pass/fail status
- Any issues or warnings

## Critical rules

- Always specify the `database` parameter explicitly in every tool call
- Source database is read-only: never execute DDL or write operations against it
- Target database is write-target: all DDL and writes go here
- Reconciliation is mandatory: never skip the row-count and target sanity checks after data transfer
- Report reconciliation results even if some checks fail
- NEVER fall back to a different database if the target is unavailable — STOP and report the error
- Before starting, verify both source and target show `"available": true` in `list_databases()` output

## Execution checklist

Use this checklist as the in-skill runbook. Keep the default path lightweight;
only run expensive reconciliation when the user or project-specific validator
rules ask for it.

### Pre-transfer

1. Confirm the source table exists and has the expected shape.
2. Document source columns, types, nullability, and the source row count
   reported or needed for the transfer.
3. Identify key columns only when they are needed for layout, deduplication,
   or optional reconciliation.
4. Confirm the target database / schema exists, or create it if the target
   connector supports schema creation and the user asked for it.
5. Verify target connectivity and write permissions before executing DDL.

### DDL generation

1. Map source column types to the target dialect using adapter migration
   capabilities when available.
2. Handle unsupported complex types (for example LIST, STRUCT, MAP, BLOB) by
   either reporting a blocking issue or explicitly excluding / serializing them
   with user approval.
3. For StarRocks or OLAP-style targets, choose a safe key and distribution
   layout (`DUPLICATE KEY`, `DISTRIBUTED BY HASH`, buckets, partitions) using
   `suggest_table_layout` when available.
4. Generate the target `CREATE TABLE` DDL.
5. Validate the DDL, execute it on the target, then verify the target schema.

### Data transfer

1. Execute `transfer_query_result`.
2. Verify the tool-reported `rows_transferred` and any reported source /
   target counts.
3. If the transfer fails partially, report rows written and the exact error;
   do not continue as if the migration succeeded.

### Default reconciliation

1. Compare the transfer tool's source/target row counts when both are
   available.
2. Run one small target-side sanity query only when the validator hook is
   disabled or the transfer result is ambiguous.
3. Treat blocking mismatches as retryable transfer failures and fix them before
   reporting success.

### Optional extended reconciliation

Do not run these by default on large systems. They can be added by
project-level validator skills or used when the user explicitly requests a
deep audit:

- null ratio comparison for nullable columns
- min/max comparison for numeric and date columns
- distinct count comparison for key columns
- duplicate-key checks on the target table
- key-based sample diff
- numeric aggregate comparison such as `SUM` or `AVG`

### Reporting

1. Compile check results with pass/fail status.
2. Flag blocking issues separately from warnings.
3. Output the final migration summary with source, target, rows transferred,
   and reconciliation status.

## Output expectations

At minimum, return:

- source and target database/table names
- rows transferred
- reconciliation summary with pass/fail
- blocking issues that must be addressed
