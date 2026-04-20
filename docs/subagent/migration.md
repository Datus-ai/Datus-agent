# Cross-Database Migration (migration)

## Overview

The `migration` subagent is a built-in agent for migrating tables between different database engines. It handles the full lifecycle:

1. Discover and inspect source/target databases
2. Generate target DDL with cross-database type mapping
3. Transfer data via `transfer_query_result`
4. Run mandatory post-migration reconciliation (7 checks)

## Quick Start

```bash
# Migrate a table from DuckDB to Greenplum
/migration Migrate the users table from local_duckdb to greenplum, target table public.users_copy

# Migrate to StarRocks
/migration Migrate orders from local_duckdb to starrocks, target table test.orders_copy
```

## Prerequisites

Both source and target databases must be configured in `agent.yml`:

```yaml
agent:
  services:
    databases:
      local_duckdb:
        type: duckdb
        uri: duckdb:///./sample_data/duckdb-demo.duckdb
      greenplum:
        type: greenplum
        host: 127.0.0.1
        port: 15432
        username: gpadmin
        password: pivotal
        database: test
        sslmode: disable
```

The `database` parameter in all tool calls uses the **logical name** (YAML key), not the engine-internal database name.

## Built-in Tools

| Tool | Purpose |
|------|---------|
| `list_databases` | Discover available databases with type info |
| `list_tables` | List tables in a database |
| `describe_table` | Get column definitions |
| `read_query` | Execute read-only SQL |
| `get_table_ddl` | Get CREATE TABLE statement |
| `execute_ddl` | Execute DDL on target database |
| `execute_write` | Execute DML on target database |
| `transfer_query_result` | Transfer data between databases |
| `read_file` / `write_file` | SQL artifact files |

## Workflow

```
Phase 1: Discover databases (list_databases) → inspect source schema
Phase 2: Generate target DDL with type mapping → create target table
Phase 3: Transfer data (transfer_query_result)
Phase 4: Reconcile source vs target (7 mandatory checks)
Phase 5: Output migration report
```

## Type Mapping

| DuckDB | Greenplum | StarRocks |
|--------|-----------|-----------|
| VARCHAR | VARCHAR | VARCHAR(65533) |
| VARCHAR(n) | VARCHAR(n) | VARCHAR(n) |
| TEXT | TEXT | STRING |
| INTEGER | INTEGER | INT |
| BIGINT | BIGINT | BIGINT |
| DECIMAL(p,s) | NUMERIC(p,s) | DECIMAL(p,s) |
| BOOLEAN | BOOLEAN | BOOLEAN |
| DATE | DATE | DATE |
| TIMESTAMP | TIMESTAMP | DATETIME |

Unsupported types (LIST, STRUCT, MAP, BLOB) are reported as errors.

## Reconciliation Checks

After every migration, 7 checks are run automatically:

1. **Row count** — total row count comparison
2. **Null ratio** — null count per column
3. **Min/max** — range comparison for numeric/date columns
4. **Distinct count** — cardinality comparison
5. **Duplicate key** — check for duplicate keys in target
6. **Sample diff** — key-based row sample comparison
7. **Numeric aggregate** — SUM/AVG comparison

## Limitations

- Maximum 1,000,000 rows per transfer
- No transaction rollback on partial failure
- Data transferred via batch INSERT (not COPY or stream load)

## Optional Configuration

```yaml
agent:
  agentic_nodes:
    migration:
      max_turns: 40       # Default: 40
```

## Skills Used

- **data-migration** — Migration workflow and reconciliation
- **table-validation** — Schema and data quality checks

## Comparison with gen_job

| Feature | gen_job | migration |
|---------|---------|-----------|
| Single-database ETL | Yes | No |
| Cross-database migration | No | Yes |
| `transfer_query_result` | No | Yes |
| Reconciliation | Optional validation | Mandatory 7 checks |
| Default max_turns | 30 | 40 |
