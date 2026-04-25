# Data Pipeline (gen_job)

## Overview

The `gen_job` subagent is a built-in data engineering agent that covers BOTH single-database ETL and cross-database transfer. It builds or updates target tables from one or more source tables, transferring data across database engines when the source and target differ, and validates the result.

The agent auto-detects the path based on the user's prompt: if the source and target are the same database, it runs intra-DB ETL (CREATE TABLE AS SELECT / INSERT from SELECT). If they differ, it runs a cross-DB transfer (via `transfer_query_result`) and activates the `data-migration` skill for lightweight reconciliation.

## Quick Start

Start Datus CLI and use the gen_job subagent via natural language:

```bash
# Single-database ETL
/gen_job Build a summary table from orders and customers tables

# Cross-database transfer
/gen_job Transfer the users table from local_duckdb to greenplum
```

The chat agent can also automatically delegate to gen_job when it detects an ETL or migration task.

## Prerequisites

### Database Configuration

All databases involved must be configured in `agent.yml` under `services.datasources`:

```yaml
agent:
  services:
    datasources:
      local_duckdb:
        type: duckdb
        uri: duckdb:///./sample_data/duckdb-demo.duckdb
        name: demo
      greenplum:
        type: greenplum
        host: 127.0.0.1
        port: 15432
        username: gpadmin
        password: pivotal
        database: test
        schema_name: public
        sslmode: disable
      starrocks:
        type: starrocks
        host: 127.0.0.1
        port: 9030
        username: root
        password: ""
        database: test
        catalog: default_catalog
```

Each database entry has a **logical name** (the YAML key, e.g., `local_duckdb`, `greenplum`). This is the name you reference when specifying source and target databases.

### For Cross-database Transfer

- Both source and target databases must be accessible
- Source database must support pandas query execution (DuckDB, PostgreSQL, etc.)
- Target database adapter must be installed (`datus-greenplum`, `datus-starrocks`, etc.)

## Built-in Tools

gen_job comes with a comprehensive tool set:

| Tool | Purpose |
|------|---------|
| `list_databases` | Discover available databases with type info |
| `list_tables` | List tables in a database |
| `describe_table` | Get column definitions and metadata |
| `read_query` | Execute read-only SQL queries |
| `get_table_ddl` | Get CREATE TABLE statement |
| `execute_ddl` | Execute DDL (CREATE/ALTER/DROP TABLE/SCHEMA) |
| `execute_write` | Execute DML (INSERT/UPDATE/DELETE) |
| `transfer_query_result` | Transfer data between databases |
| `get_migration_capabilities` | Read target dialect's requires/forbids/type_hints from adapter's `MigrationTargetMixin` (see Cross-DB section) |
| `suggest_table_layout` | Ask target adapter to suggest distribution/partition/order keys based on source columns |
| `validate_ddl` | Run target-dialect static validation (and optionally dry-run CREATE+DROP) before executing DDL |
| `read_file` / `write_file` | Read/write SQL artifact files |
| `ask_user` | Interactive confirmation (interactive mode only) |

All database tools accept an optional `database` parameter for explicit routing to the target database.

## Workflow

### Single-database ETL

```
Phase 1: Inspect source and target tables
Phase 2: Generate and execute DDL / DML
Phase 3: Validate result (built-in existence/row count plus explicit schema contract checks)
Phase 4: Summarize
```

### Cross-database Transfer

```
Phase 1: Discover databases (list_databases) and inspect source
Phase 2: Call get_migration_capabilities() on target for dialect hints
         + suggest_table_layout() for OLAP targets
Phase 3: Draft target DDL → validate_ddl() → execute_ddl()
Phase 4: Transfer data (transfer_query_result)
Phase 5: Reconcile row counts and target-side sanity checks
Phase 6: Summarize with reconciliation report
```

## Cross-database Transfer Details

### Database Discovery

The agent first calls `list_databases()` to discover available databases:

```json
[
  {"name": "local_duckdb", "type": "duckdb"},
  {"name": "greenplum", "type": "greenplum"},
  {"name": "starrocks", "type": "starrocks"}
]
```

### Dialect Hints (adapter-driven)

Type mapping and DDL requirements are NOT hardcoded in the agent. Each target adapter implements `MigrationTargetMixin` in `datus-db-core` to declare its own dialect's contract. The agent consumes this via three wrapper tools:

- **`get_migration_capabilities(database=target)`** — returns the target's `dialect_family`, `requires` (clauses the DDL MUST include), `forbids` (patterns the DDL MUST NOT include), `type_hints` (preferred mappings), and a reference `example_ddl`.
- **`suggest_table_layout(database=target, columns_json=...)`** — returns OLAP-specific hints (e.g. `{duplicate_key, distributed_by, buckets}` for StarRocks, `{engine, order_by}` for ClickHouse, `{partitioned_by}` for Hive via Trino).
- **`validate_ddl(database=target, ddl=..., target_table=...)`** — runs structural validation (e.g. StarRocks must have `DUPLICATE KEY` + `DISTRIBUTED BY`; ClickHouse must have `ENGINE` + `ORDER BY`); optionally runs `dry_run_ddl` (CREATE + DROP to a temp table).

Adapters that implement the Mixin (as of this release): StarRocks, Greenplum, PostgreSQL, MySQL, ClickHouse, Trino, Snowflake, Redshift, DuckDB, SQLite, plus a generic OLTP fallback on the SQLAlchemy base. Adapters without a Mixin yet (e.g. BigQuery, Hive, Spark, ClickZetta) fall back to pure-LLM mode — `get_migration_capabilities` returns `{"supported": false, "warning": ...}` and the LLM relies on its own knowledge of the target dialect.

To add hints for a new target dialect, implement `MigrationTargetMixin` in the adapter — no changes to the agent are required.

### Data Transfer

The `transfer_query_result` tool handles data movement:

```
transfer_query_result(
    source_sql="SELECT * FROM users",
    source_database="local_duckdb",
    target_table="public.users_copy",
    target_database="greenplum",
    mode="replace"        # replace (TRUNCATE+INSERT) or append
)
```

Limitations:
- Maximum 1,000,000 rows per transfer
- No transaction rollback on partial failure
- Data is loaded via batch INSERT (not COPY or stream load)

### Reconciliation Checks

After `transfer_query_result`, validation runs lightweight reconciliation:

1. **Tool-reported row count parity** - compare `source_row_count` and `transferred_row_count`.
2. **Target row count** - run one target-side `COUNT(*)` and compare it with `transferred_row_count`.
3. **Target sample** - optionally read a small target sample to confirm the table is queryable.

More expensive checks such as null ratios, min/max, distinct counts, duplicate-key checks, sample diffs, and numeric aggregates should be added as project-specific validator skills when the table contract requires them.

## Optional Configuration

gen_job works out of the box. You can optionally customize it in `agent.yml`:

```yaml
agent:
  agentic_nodes:
    gen_job:
      max_turns: 40       # Default: 40 (cross-DB flows need more turns)
```

## Skills Used

gen_job leverages three skills:

- **gen-table** - Table creation and DDL decisions
- **table-validation** - Explicit schema contract checks
- **data-migration** - Cross-database transfer workflow and lightweight reconciliation

## Examples

### Example 1: Build a Summary Table

```
User: Build a daily_sales_summary table from the orders table,
      aggregating total_amount and order_count by date

gen_job:
  1. Inspects orders table schema
  2. Generates CREATE TABLE daily_sales_summary DDL
  3. Executes INSERT ... SELECT with aggregation
  4. Validates row count and schema
  5. Returns summary
```

### Example 2: Transfer DuckDB to Greenplum

```
User: Migrate the users table from local_duckdb to greenplum,
      target table public.users_copy

gen_job:
  1. Calls list_databases() -> local_duckdb (duckdb) and greenplum (greenplum)
  2. Calls describe_table("users", database="local_duckdb") -> 6 columns
  3. Calls get_migration_capabilities(database="greenplum")
     -> dialect_family="postgres-like", DISTRIBUTED BY recommended
  4. Drafts CREATE TABLE public.users_copy (...) DISTRIBUTED BY (id)
  5. Calls validate_ddl(database="greenplum", ddl=..., target_table="users_copy")
     -> errors=[], proceeds
  6. Executes DDL via execute_ddl(database="greenplum")
  7. Calls transfer_query_result(source_database="local_duckdb",
     target_database="greenplum", mode="replace")
  8. Activates data-migration skill; runs row-count and target-side sanity checks
  9. Returns migration report with pass/fail status
```

### Example 3: Migrate MySQL to StarRocks

```text
User: Migrate orders from mysql_prod to starrocks, target test.orders_copy

gen_job:
  1. Calls get_migration_capabilities(database="starrocks")
     -> requires DUPLICATE KEY + DISTRIBUTED BY HASH + BUCKETS
  2. Calls suggest_table_layout(database="starrocks", columns_json=...)
     -> {"duplicate_key": ["order_id"], "distributed_by": ["order_id"], "buckets": 10}
  3. Drafts CREATE TABLE with the suggested layout + type_hints
  4. validate_ddl() catches a missing NOT NULL on the key column -> LLM fixes it
  5. Executes DDL on starrocks
  6. transfer_query_result + lightweight reconciliation (tool-reported row-count parity + target-side COUNT(*) + optional sample)
```
