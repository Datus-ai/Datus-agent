"""
MetricFlow semantic adapter nightly tests -- PostgreSQL backend.

Opt-in (all required):
  * datus-semantic-metricflow must be installed
  * PostgreSQL container must be running (shared with PostgreSQL Adapter Tests suite)
  * set env var: ADAPTERS_METRICFLOW_PG=1

Env overrides (defaults match datus-postgresql/docker-compose.yml):
  POSTGRESQL_HOST=localhost  POSTGRESQL_PORT=5432
  POSTGRESQL_USER=test_user  POSTGRESQL_PASSWORD=test_password  POSTGRESQL_DATABASE=test

MetricFlow tables are isolated in the `mf_nightly` schema within the existing
`test` database and dropped on teardown.
"""

import asyncio
import os

import pytest

from tests.nightly_requirements import import_required, require_opt_in_env

require_opt_in_env("ADAPTERS_METRICFLOW_PG", "tests/integration/adapters/README.md")

datus_semantic_metricflow = import_required(  # noqa: E402
    "datus_semantic_metricflow",
    reason="datus-semantic-metricflow not installed; install it before running this suite",
)

MetricFlowAdapter = datus_semantic_metricflow.MetricFlowAdapter
MetricFlowConfig = datus_semantic_metricflow.MetricFlowConfig

pytestmark = [pytest.mark.integration, pytest.mark.nightly]

_HOST = os.getenv("POSTGRESQL_HOST", "localhost")
_PORT = int(os.getenv("POSTGRESQL_PORT", "5432"))
_USER = os.getenv("POSTGRESQL_USER", "test_user")
_PASSWORD = os.getenv("POSTGRESQL_PASSWORD", "test_password")
_DATABASE = os.getenv("POSTGRESQL_DATABASE", "test")
_SCHEMA = "mf_nightly"

_DATA_TABLE = "mf_orders"
_TIME_SPINE_TABLE = "mf_time_spine"

_SEMANTIC_YAML = f"""\
data_source:
  name: mf_orders
  sql_table: {_SCHEMA}.{_DATA_TABLE}
  identifiers:
    - name: order_id
      type: primary
      expr: id
  measures:
    - name: total_amount
      agg: sum
      expr: amount
    - name: order_count
      agg: count
      expr: id
  dimensions:
    - name: created_at
      type: time
      type_params:
        is_primary: true
        time_granularity: day
---
metric:
  name: total_amount
  type: measure_proxy
  type_params:
    measure: total_amount
---
metric:
  name: order_count
  type: measure_proxy
  type_params:
    measure: order_count
"""

_SAMPLE_ROWS = [
    (1, 10.00, "2020-01-01"),
    (2, 20.00, "2020-01-02"),
    (3, 30.00, "2020-01-03"),
    (4, 40.00, "2020-01-04"),
    (5, 50.00, "2020-01-05"),
]


@pytest.fixture(scope="module")
def mf_config(tmp_path_factory):
    yaml_dir = tmp_path_factory.mktemp("mf_pg_models")
    (yaml_dir / "mf_orders.yaml").write_text(_SEMANTIC_YAML)
    return MetricFlowConfig(
        datasource="mf_nightly",
        db_config={
            "type": "postgres",
            "host": _HOST,
            "port": str(_PORT),
            "username": _USER,
            "password": _PASSWORD,
            "database": _DATABASE,
            "schema": _SCHEMA,
        },
        semantic_models_path=str(yaml_dir),
    )


@pytest.fixture(scope="module")
def seeded_db(mf_config):
    import psycopg2

    conn = psycopg2.connect(
        host=_HOST,
        port=_PORT,
        user=_USER,
        password=_PASSWORD,
        dbname=_DATABASE,
    )
    conn.autocommit = True
    cursor = conn.cursor()
    try:
        cursor.execute(f'DROP TABLE IF EXISTS "{_SCHEMA}"."{_TIME_SPINE_TABLE}" CASCADE')
        cursor.execute(f'DROP TABLE IF EXISTS "{_SCHEMA}"."{_DATA_TABLE}" CASCADE')
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {_SCHEMA}")
        cursor.execute(
            f'CREATE TABLE "{_SCHEMA}"."{_DATA_TABLE}" (id INTEGER PRIMARY KEY, amount DECIMAL(10,2), created_at DATE)'
        )
        values = ", ".join(f"({r[0]}, {r[1]}, '{r[2]}')" for r in _SAMPLE_ROWS)
        cursor.execute(f'INSERT INTO "{_SCHEMA}"."{_DATA_TABLE}" VALUES {values}')
        cursor.execute(f'CREATE TABLE "{_SCHEMA}"."{_TIME_SPINE_TABLE}" (ds DATE NOT NULL)')
        cursor.execute(
            f'INSERT INTO "{_SCHEMA}"."{_TIME_SPINE_TABLE}" '
            "SELECT d::date FROM generate_series('2020-01-01'::date, '2025-12-31'::date, '1 day') d"
        )
    finally:
        cursor.close()

    yield

    cleanup = conn.cursor()
    try:
        cleanup.execute(f'DROP TABLE IF EXISTS "{_SCHEMA}"."{_TIME_SPINE_TABLE}" CASCADE')
        cleanup.execute(f'DROP TABLE IF EXISTS "{_SCHEMA}"."{_DATA_TABLE}" CASCADE')
    finally:
        cleanup.close()
        conn.close()


@pytest.fixture(scope="module")
def mf_adapter(mf_config, seeded_db):
    return MetricFlowAdapter(mf_config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_validate_semantic_passes(mf_adapter):
    result = asyncio.run(mf_adapter.validate_semantic())
    errors = [i for i in result.issues if i.severity == "error"]
    assert result.valid, f"Unexpected validation errors: {errors}"


def test_list_metrics_returns_metric(mf_adapter):
    metrics = asyncio.run(mf_adapter.list_metrics())
    names = {m.name for m in metrics}
    assert len(metrics) >= 1, "Expected at least one metric"
    assert "total_amount" in names, f"'total_amount' not in {sorted(names)}"


def test_get_dimensions_returns_dimension(mf_adapter):
    dims = asyncio.run(mf_adapter.get_dimensions("total_amount"))
    assert len(dims) >= 1, f"Expected at least one dimension, got {dims}"


def test_query_metrics_dry_run_returns_sql(mf_adapter):
    result = asyncio.run(mf_adapter.query_metrics(["total_amount"], dry_run=True))
    sql = result.metadata.get("sql", "")
    assert sql, f"Expected non-empty SQL from dry_run; metadata={result.metadata}"


def test_query_metrics_live(mf_adapter):
    result = asyncio.run(mf_adapter.query_metrics(["total_amount"]))
    assert len(result.data) >= 1, f"Expected data rows; got: {result}"
    assert "total_amount" in result.columns, f"Expected 'total_amount' in columns; got: {result.columns}"


def test_query_metrics_with_time_filter(mf_adapter):
    result = asyncio.run(
        mf_adapter.query_metrics(
            ["total_amount"],
            time_start="2020-01-01",
            time_end="2020-01-03",
        )
    )
    assert len(result.data) >= 1, f"Expected data with time filter; got: {result}"
    total = sum(float(row["total_amount"]) for row in result.data if row.get("total_amount") is not None)
    assert total == pytest.approx(60.0), f"Expected SUM=60 for 2020-01-01..03, got {total}"


def test_query_metrics_multi_metric(mf_adapter):
    result = asyncio.run(mf_adapter.query_metrics(["total_amount", "order_count"]))
    assert len(result.data) >= 1, f"Expected data rows; got: {result}"
    assert "total_amount" in result.columns, f"'total_amount' missing from {result.columns}"
    assert "order_count" in result.columns, f"'order_count' missing from {result.columns}"


def test_query_metrics_where_clause_dry_run(mf_adapter):
    result = asyncio.run(
        mf_adapter.query_metrics(
            ["total_amount"],
            where="metric_time >= '2020-01-04'",
            dry_run=True,
        )
    )
    sql = result.metadata.get("sql", "")
    assert sql, f"Expected non-empty SQL with where clause; metadata={result.metadata}"
    assert "WHERE" in sql.upper(), f"Expected WHERE in generated SQL; got:\n{sql}"
