# ETL Job (gen_job)

## Overview

The `gen_job` subagent is a built-in data engineering agent for single-database ETL. It builds target tables from source tables within the same database using SQL (CREATE TABLE AS SELECT, INSERT from SELECT, etc.).

For cross-database migration, use the [migration](migration.md) subagent instead.

## Quick Start

Start Datus CLI and use the gen_job subagent via natural language:

```bash
# Single-database ETL
/gen_job Build a summary table from orders and customers tables

# Cross-database migration
/gen_job Migrate the users table from local_duckdb to greenplum
```

The chat agent can also automatically delegate to gen_job when it detects an ETL or migration task.

## Prerequisites

### Database Configuration

All databases involved must be configured in `agent.yml` under `services.databases`:

```yaml
agent:
  services:
    databases:
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

### For Cross-database Migration

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
| `read_file` / `write_file` | Read/write SQL artifact files |
| `ask_user` | Interactive confirmation (interactive mode only) |

All database tools accept an optional `database` parameter for explicit routing to the target database.

## Workflow

### Single-database ETL

```
Phase 1: Inspect source and target tables
Phase 2: Generate and execute DDL / DML
Phase 3: Validate result (row count, schema, data quality)
Phase 4: Summarize
```

### Cross-database Migration

```
Phase 1: Discover databases (list_databases) and inspect source
Phase 2: Generate target DDL with cross-database type mapping
Phase 3: Transfer data (transfer_query_result)
Phase 4: Reconcile source vs target (7 checks)
Phase 5: Summarize with reconciliation report
```

## Cross-database Migration Details

### Database Discovery

The agent first calls `list_databases()` to discover available databases:

```json
[
  {"name": "local_duckdb", "type": "duckdb"},
  {"name": "greenplum", "type": "greenplum"},
  {"name": "starrocks", "type": "starrocks"}
]
```

### Type Mapping

Types are automatically mapped between dialects:

| DuckDB | Greenplum | StarRocks |
|--------|-----------|-----------|
| VARCHAR | VARCHAR | VARCHAR(65533) |
| VARCHAR(n) | VARCHAR(n) | VARCHAR(n) |
| TEXT | TEXT | STRING |
| INTEGER | INTEGER | INT |
| BIGINT | BIGINT | BIGINT |
| DOUBLE | DOUBLE PRECISION | DOUBLE |
| DECIMAL(p,s) | NUMERIC(p,s) | DECIMAL(p,s) |
| BOOLEAN | BOOLEAN | BOOLEAN |
| DATE | DATE | DATE |
| TIMESTAMP | TIMESTAMP | DATETIME |

Unsupported types (LIST, STRUCT, MAP, BLOB, etc.) are reported as errors.

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

After migration, 7 reconciliation checks are run automatically:

1. **Row count** - Total row count comparison
2. **Null ratio** - Null count per column
3. **Min/max** - Range comparison for numeric/date columns
4. **Distinct count** - Cardinality comparison
5. **Duplicate key** - Check for duplicate keys in target
6. **Sample diff** - Key-based row sample comparison
7. **Numeric aggregate** - SUM/AVG comparison

## Optional Configuration

gen_job works out of the box. You can optionally customize it in `agent.yml`:

```yaml
agent:
  agentic_nodes:
    gen_job:
      max_turns: 30       # Default: 30
```

## Skills Used

gen_job leverages three skills:

- **gen-table** - Table creation and DDL decisions
- **table-validation** - Schema and data quality checks
- **data-migration** - Cross-database migration workflow and reconciliation

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

### Example 2: Migrate DuckDB to Greenplum

```
User: Migrate the users table from local_duckdb to greenplum,
      target table public.users_copy

gen_job:
  1. Calls list_databases() -> discovers local_duckdb (duckdb) and greenplum (greenplum)
  2. Calls describe_table("users", database="local_duckdb") -> gets 6 columns
  3. Maps types: INTEGER->INTEGER, VARCHAR->VARCHAR, DECIMAL->NUMERIC, etc.
  4. Executes CREATE TABLE on greenplum
  5. Calls transfer_query_result(source_database="local_duckdb", target_database="greenplum")
  6. Runs 7 reconciliation checks comparing both databases
  7. Returns migration report with pass/fail status
```
