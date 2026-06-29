# Adapter Contract Tests

End-to-end contract tests that exercise `DBFuncTool` / `BIFuncTool` (main repo)
against real database / BI services started from each adapter's own docker-compose.

Each adapter's contract tests are **opt-in** via an env var, because they
require a docker container to be running. Tests skip cleanly when the opt-in
flag is unset.

## Why separate `integration/adapters/`?

`tests/integration/tools/` already covers `DBFuncTool` against SQLite/DuckDB.
This directory specifically exercises the *adapter packages* (`datus-postgresql`,
`datus-mysql`, etc.) — one suite per adapter, each using the adapter repo's
own `docker-compose.yml` as the canonical fixture.

## Running

### PostgreSQL

```bash
# 1. Install the adapter (not a hard dep of Datus-agent)
uv pip install datus-postgresql

# 2. Start the docker container (in the adapter repo)
cd /path/to/datus-db-adapters/datus-postgresql
docker compose up -d
# Wait ~30s for the healthcheck to pass

# 3. Run the contract tests (from Datus-agent repo)
cd /path/to/Datus-agent
ADAPTERS_PG=1 uv run pytest tests/integration/adapters/test_postgresql.py -v

# 4. Tear down
cd /path/to/datus-db-adapters/datus-postgresql
docker compose down -v
```

## Env vars

| Adapter | Opt-in flag | Connection env | Default (matches adapter's docker-compose.yml) |
|---|---|---|---|
| postgresql | `ADAPTERS_PG=1` | `POSTGRESQL_HOST/PORT/USER/PASSWORD/DATABASE` | `localhost:5432 test_user/test_password/test` |
| mysql | `ADAPTERS_MYSQL=1` | `MYSQL_HOST/PORT/USER/PASSWORD/DATABASE` | `localhost:3306 test_user/test_password/test` |
| clickhouse | `ADAPTERS_CH=1` | `CLICKHOUSE_HOST/PORT/USER/PASSWORD/DATABASE` | `localhost:8123 default_user/default_test/default_test` |
| starrocks | `ADAPTERS_SR=1` | `STARROCKS_HOST/PORT/USER/PASSWORD/CATALOG/DATABASE` | `127.0.0.1:9030 root//default_catalog/test` |
| trino | `ADAPTERS_TRINO=1` | `TRINO_HOST/PORT/USER` | `localhost:8080 trino` (uses built-in `tpch.tiny`, no seeding) |
| greenplum | `ADAPTERS_GP=1` | `GREENPLUM_HOST/PORT/USER/PASSWORD/DATABASE/SCHEMA` | `localhost:15432 gpadmin/pivotal/postgres/public` |
| hive | `ADAPTERS_HIVE=1` | `HIVE_HOST/PORT/USERNAME/PASSWORD/DATABASE` | `localhost:10000 hive//default` |
| spark | `ADAPTERS_SPARK=1` | `SPARK_HOST/PORT/USER/PASSWORD/DATABASE/AUTH_MECHANISM` | `localhost:10000 spark//default/NONE` |

### Port conflicts

Several adapters use default ports that are commonly occupied:
- postgresql (5432) — conflicts with any local Postgres / superset-db
- trino (8080) — conflicts with Airflow / many web dev servers
- starrocks (9030) — conflicts with existing StarRocks instances
- hive / spark (10000) — both default to the HiveServer2/Spark Thrift port; run one suite at a time or remap one service

For the Trino adapter, the compose file already supports a `TRINO_HOST_PORT`
override (see its `docker-compose.yml`). For the others, either stop the
conflicting container or use a one-off `docker run` on an alternate port and
override the `*_PORT` env var.

## What gets tested

For each adapter, contract tests cover the public surface of `DBFuncTool`
that the agent actually calls:

- `list_tables` — returns seeded tables
- `describe_table` — returns column metadata
- `read_query` — executes a SELECT and returns compressed rows
- `read_query` read-only guard — rejects DML / multi-statement injection

## Adding a new adapter

1. Copy `test_postgresql.py` as `test_<name>.py`.
2. Replace `datus_postgresql` imports with the new adapter's connector/config.
3. Adjust the seeded DDL to the target dialect (quote style, type names).
4. Pick a new opt-in flag (`ADAPTERS_<NAME>=1`) and document env vars here.
5. Confirm the adapter's `docker-compose.yml` ports don't collide with others
   you run simultaneously.

---

## MetricFlow Semantic Adapter Tests

These suites exercise `MetricFlowAdapter` against real databases:
`validate_semantic`, `list_metrics`, `get_dimensions`, `query_metrics(dry_run=True)`,
and live `query_metrics(...)` behavior including time filters, multi-metric queries,
and `where`-clause SQL generation.

Each suite seeds a minimal `mf_orders` fact table plus `mf_time_spine` (required
by MetricFlow) and cleans up on teardown.

### DuckDB (no Docker)

```bash
ADAPTERS_METRICFLOW_DUCKDB=1 uv run pytest tests/integration/adapters/test_semantic_metricflow_duckdb.py -v
```

No container needed. The database file is created in a pytest tmp directory.

### MySQL (shares container with MySQL Adapter Tests)

```bash
cd /path/to/datus-db-adapters/datus-mysql && docker compose up -d
cd /path/to/Datus-agent
ADAPTERS_METRICFLOW_MYSQL=1 uv run pytest tests/integration/adapters/test_semantic_metricflow_mysql.py -v
```

MetricFlow tables (`mf_orders`, `mf_time_spine`) are created inside the existing
`test` database used by the MySQL Adapter Tests and dropped on teardown.

### PostgreSQL (shares container with PostgreSQL Adapter Tests)

```bash
cd /path/to/datus-db-adapters/datus-postgresql && docker compose up -d
cd /path/to/Datus-agent
ADAPTERS_METRICFLOW_PG=1 uv run pytest tests/integration/adapters/test_semantic_metricflow_postgresql.py -v
```

MetricFlow tables are created in the `mf_nightly` schema within the existing
`test` database and dropped on teardown.

### MetricFlow env vars

| Suite | Opt-in flag | Connection env | Notes |
|---|---|---|---|
| DuckDB | `ADAPTERS_METRICFLOW_DUCKDB=1` | none | DB file auto-generated in tmp dir |
| MySQL | `ADAPTERS_METRICFLOW_MYSQL=1` | same as `ADAPTERS_MYSQL` vars | tables in `test` DB |
| PostgreSQL | `ADAPTERS_METRICFLOW_PG=1` | same as `ADAPTERS_PG` vars | tables in `mf_nightly` schema |
