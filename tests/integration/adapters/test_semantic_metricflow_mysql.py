"""
MetricFlow semantic adapter nightly tests -- MySQL backend.

Opt-in (all required):
  * datus-semantic-metricflow must be installed
  * MySQL container must be running (shared with MySQL Adapter Tests suite)
  * set env var: ADAPTERS_METRICFLOW_MYSQL=1

Env overrides (defaults match datus-mysql/docker-compose.yml):
  MYSQL_HOST=localhost  MYSQL_PORT=3306
  MYSQL_USER=test_user  MYSQL_PASSWORD=test_password  MYSQL_DATABASE=test

In MySQL, schema == database. MetricFlow tables (mf_orders, mf_time_spine)
are created inside the existing `test` database alongside the adapter test
tables and dropped on teardown.
"""

import logging
import os
from datetime import date, timedelta

import pytest

from tests.nightly_requirements import import_required, require_opt_in_env

require_opt_in_env("ADAPTERS_METRICFLOW_MYSQL", "tests/integration/adapters/README.md")

datus_semantic_metricflow = import_required(  # noqa: E402
    "datus_semantic_metricflow",
    reason="datus-semantic-metricflow not installed; install it before running this suite",
)

MetricFlowAdapter = datus_semantic_metricflow.MetricFlowAdapter
MetricFlowConfig = datus_semantic_metricflow.MetricFlowConfig

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.integration, pytest.mark.nightly, pytest.mark.asyncio]

_HOST = os.getenv("MYSQL_HOST", "localhost")
_PORT = int(os.getenv("MYSQL_PORT", "3306"))
_USER = os.getenv("MYSQL_USER", "test_user")
_PASSWORD = os.getenv("MYSQL_PASSWORD", "test_password")
_DATABASE = os.getenv("MYSQL_DATABASE", "test")

_DATA_TABLE = "mf_orders"
_TIME_SPINE_TABLE = "mf_time_spine"

# sql_table uses the full database-qualified name (MySQL has no separate schema concept).
_SEMANTIC_YAML = f"""\
data_source:
  name: mf_orders
  sql_table: {_DATABASE}.{_DATA_TABLE}
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
    yaml_dir = tmp_path_factory.mktemp("mf_mysql_models")
    (yaml_dir / "mf_orders.yaml").write_text(_SEMANTIC_YAML)
    return MetricFlowConfig(
        datasource="mf_nightly",
        db_config={
            "type": "mysql",
            "host": _HOST,
            "port": str(_PORT),
            "username": _USER,
            "password": _PASSWORD,
            "database": _DATABASE,
            "schema": _DATABASE,
        },
        semantic_models_path=str(yaml_dir),
    )


@pytest.fixture(scope="module")
def seeded_db(mf_config):
    import pymysql

    conn = pymysql.connect(
        host=_HOST,
        port=_PORT,
        user=_USER,
        password=_PASSWORD,
        database=_DATABASE,
        charset="utf8mb4",
        autocommit=True,
    )
    # Outer try/finally guarantees the connection is closed even if seeding
    # raises before `yield`; each cursor is scoped with a context manager.
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS `{_DATA_TABLE}`")
            cursor.execute(f"DROP TABLE IF EXISTS `{_TIME_SPINE_TABLE}`")
            cursor.execute(
                f"CREATE TABLE `{_DATA_TABLE}` (id INT NOT NULL, amount DECIMAL(10,2), created_at DATE, PRIMARY KEY (id))"
            )
            values = ", ".join(f"({r[0]}, {r[1]}, '{r[2]}')" for r in _SAMPLE_ROWS)
            cursor.execute(f"INSERT INTO `{_DATA_TABLE}` VALUES {values}")

            cursor.execute(f"CREATE TABLE `{_TIME_SPINE_TABLE}` (ds DATE NOT NULL)")
            spine_values = []
            d = date(2020, 1, 1)
            while d <= date(2025, 12, 31):
                spine_values.append(f"('{d}')")
                d += timedelta(days=1)
            cursor.execute(f"INSERT INTO `{_TIME_SPINE_TABLE}` VALUES {','.join(spine_values)}")

        yield

        with conn.cursor() as cleanup:
            cleanup.execute(f"DROP TABLE IF EXISTS `{_DATA_TABLE}`")
            cleanup.execute(f"DROP TABLE IF EXISTS `{_TIME_SPINE_TABLE}`")
    finally:
        conn.close()


@pytest.fixture(scope="module")
def mf_adapter(mf_config, seeded_db):
    adapter = MetricFlowAdapter(mf_config)
    yield adapter
    # Best-effort connection teardown so the adapter's SQL client does not stay
    # open for the rest of the pytest process; log instead of silently passing.
    try:
        adapter.client.sql_client.close()
    except Exception as exc:  # noqa: BLE001 - teardown is best-effort
        logger.warning("Failed to close MetricFlow SQL client during teardown: %s", exc)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_validate_semantic_passes(mf_adapter):
    result = await mf_adapter.validate_semantic()
    errors = [i for i in result.issues if i.severity == "error"]
    assert result.valid, f"Unexpected validation errors: {errors}"


async def test_list_metrics_returns_metric(mf_adapter):
    metrics = await mf_adapter.list_metrics()
    names = {m.name for m in metrics}
    assert len(metrics) >= 1, "Expected at least one metric"
    assert "total_amount" in names, f"'total_amount' not in {sorted(names)}"


async def test_get_dimensions_returns_dimension(mf_adapter):
    dims = await mf_adapter.get_dimensions("total_amount")
    assert len(dims) >= 1, f"Expected at least one dimension, got {dims}"


async def test_query_metrics_dry_run_returns_sql(mf_adapter):
    result = await mf_adapter.query_metrics(["total_amount"], dry_run=True)
    sql = result.metadata.get("sql", "")
    assert sql, f"Expected non-empty SQL from dry_run; metadata={result.metadata}"


async def test_query_metrics_live(mf_adapter):
    result = await mf_adapter.query_metrics(["total_amount"])
    assert len(result.data) >= 1, f"Expected data rows; got: {result}"
    assert "total_amount" in result.columns, f"Expected 'total_amount' in columns; got: {result.columns}"


async def test_query_metrics_with_time_filter(mf_adapter):
    result = await mf_adapter.query_metrics(
        ["total_amount"],
        time_start="2020-01-01",
        time_end="2020-01-03",
    )
    assert len(result.data) >= 1, f"Expected data with time filter; got: {result}"
    total = sum(float(row["total_amount"]) for row in result.data if row.get("total_amount") is not None)
    assert total == pytest.approx(60.0), f"Expected SUM=60 for 2020-01-01..03, got {total}"


async def test_query_metrics_multi_metric(mf_adapter):
    result = await mf_adapter.query_metrics(["total_amount", "order_count"])
    assert len(result.data) >= 1, f"Expected data rows; got: {result}"
    assert "total_amount" in result.columns, f"'total_amount' missing from {result.columns}"
    assert "order_count" in result.columns, f"'order_count' missing from {result.columns}"


async def test_query_metrics_where_clause_dry_run(mf_adapter):
    result = await mf_adapter.query_metrics(
        ["total_amount"],
        where="metric_time >= '2020-01-04'",
        dry_run=True,
    )
    sql = result.metadata.get("sql", "")
    assert sql, f"Expected non-empty SQL with where clause; metadata={result.metadata}"
    assert "WHERE" in sql.upper(), f"Expected WHERE in generated SQL; got:\n{sql}"
    # Assert the caller-supplied predicate survives: a framework-generated WHERE
    # could otherwise pass this test even if the where= argument were dropped.
    assert "2020-01-04" in sql, f"Expected dry_run SQL to preserve the caller filter; got:\n{sql}"
