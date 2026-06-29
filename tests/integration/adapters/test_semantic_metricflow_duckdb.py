"""
MetricFlow semantic adapter nightly tests -- DuckDB backend.

Opt-in:
  * datus-semantic-metricflow must be installed
  * set env var: ADAPTERS_METRICFLOW_DUCKDB=1

The DuckDB instance lives in a pytest tmp directory; no Docker required.

Env overrides: none (DuckDB is file-based, path is auto-generated)
"""

import logging
import pathlib

import pytest

from tests.nightly_requirements import import_required, require_opt_in_env

require_opt_in_env("ADAPTERS_METRICFLOW_DUCKDB", "tests/integration/adapters/README.md")

datus_semantic_metricflow = import_required(  # noqa: E402
    "datus_semantic_metricflow",
    reason="datus-semantic-metricflow not installed; install it before running this suite",
)

MetricFlowAdapter = datus_semantic_metricflow.MetricFlowAdapter
MetricFlowConfig = datus_semantic_metricflow.MetricFlowConfig

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.integration, pytest.mark.nightly, pytest.mark.asyncio]

_SCHEMA = "mf_nightly"
_DATA_TABLE = "mf_orders"
_TIME_SPINE_TABLE = "mf_time_spine"

_SEMANTIC_YAML = """\
data_source:
  name: mf_orders
  sql_table: mf_nightly.mf_orders
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
    base = tmp_path_factory.mktemp("mf_duckdb")
    yaml_dir = base / "models"
    yaml_dir.mkdir()
    (yaml_dir / "mf_orders.yaml").write_text(_SEMANTIC_YAML)
    db_path = base / "test.duckdb"
    return MetricFlowConfig(
        datasource="mf_nightly",
        db_config={
            "type": "duckdb",
            "database": str(db_path),
            "schema": _SCHEMA,
        },
        semantic_models_path=str(yaml_dir),
    )


@pytest.fixture(scope="module")
def seeded_db(mf_config):
    import duckdb as _duckdb

    db_path = mf_config.db_config["database"]
    conn = _duckdb.connect(db_path)
    try:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {_SCHEMA}")
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {_SCHEMA}.{_DATA_TABLE} (id INTEGER, amount DECIMAL(10,2), created_at DATE)"
        )
        values = ", ".join(f"({r[0]}, {r[1]}, '{r[2]}')" for r in _SAMPLE_ROWS)
        conn.execute(f"INSERT INTO {_SCHEMA}.{_DATA_TABLE} VALUES {values}")
        conn.execute(f"CREATE TABLE IF NOT EXISTS {_SCHEMA}.{_TIME_SPINE_TABLE} (ds DATE NOT NULL)")
        conn.execute(
            f"INSERT INTO {_SCHEMA}.{_TIME_SPINE_TABLE} "
            "SELECT range::DATE FROM range(DATE '2020-01-01', DATE '2026-01-01', INTERVAL '1 day')"
        )
    finally:
        conn.close()

    yield

    # Remove the DuckDB file and its write-ahead log; unlink(missing_ok=True)
    # avoids swallowing unexpected errors the way a bare try/except would.
    for suffix in ("", ".wal"):
        pathlib.Path(db_path + suffix).unlink(missing_ok=True)


@pytest.fixture(scope="module")
def mf_adapter(mf_config, seeded_db):
    adapter = MetricFlowAdapter(mf_config)
    yield adapter
    # Best-effort connection teardown: log (instead of silently passing) so a
    # close failure stays visible without breaking the suite.
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
