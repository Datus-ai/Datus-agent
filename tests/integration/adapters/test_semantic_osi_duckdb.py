"""OSI query-compatibility nightly tests using the MetricFlow DuckDB backend.

Opt-in:
  * datus-semantic-osi[metricflow] must be installed
  * set env var: ADAPTERS_OSI_DUCKDB=1

The suite intentionally exercises only the retained query surface. Semantic
authoring is provided by the default Dosi adapter.
"""

import pytest

from tests.nightly_requirements import import_required, require_opt_in_env

require_opt_in_env("ADAPTERS_OSI_DUCKDB", "tests/integration/adapters/README.md")

import_required(
    "datus_semantic_metricflow",
    reason="OSI's query backend requires datus-semantic-osi[metricflow]",
)
import_required(
    "datus_semantic_osi",
    reason="datus-semantic-osi not installed; install datus-semantic-osi[metricflow]",
)

from datus_semantic_osi.adapter import DatusOSIAdapter  # noqa: E402
from datus_semantic_osi.config import DatusOSIConfig  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.nightly, pytest.mark.asyncio]

_SCHEMA = "osi_nightly"
_DATA_TABLE = "osi_orders"
_TIME_SPINE_TABLE = "mf_time_spine"

_SEMANTIC_YAML = f"""\
version: 0.2.0.dev0
semantic_model:
  - name: order_model
    datasets:
      - name: orders
        source: {_SCHEMA}.{_DATA_TABLE}
        primary_key: [id]
        fields:
          - name: created_at
            expression:
              dialects:
                - dialect: ANSI_SQL
                  expression: created_at
            dimension:
              is_time: true
            custom_extensions:
              - vendor_name: DATUS
                data: '{{"time_granularity": "day", "type": "time"}}'
          - name: status
            expression:
              dialects:
                - dialect: ANSI_SQL
                  expression: status
            dimension: {{}}
          - name: amount
            expression:
              dialects:
                - dialect: ANSI_SQL
                  expression: amount
        custom_extensions:
          - vendor_name: DATUS
            data: '{{"time_dimension": {{"granularity": "day", "name": "created_at"}}}}'
    metrics:
      - name: total_amount
        expression:
          dialects:
            - dialect: ANSI_SQL
              expression: SUM(amount)
        custom_extensions:
          - vendor_name: DATUS
            data: '{{"dataset": "orders"}}'
      - name: order_count
        expression:
          dialects:
            - dialect: ANSI_SQL
              expression: COUNT(DISTINCT id)
        custom_extensions:
          - vendor_name: DATUS
            data: '{{"dataset": "orders"}}'
"""


@pytest.fixture(scope="module")
def osi_adapter(tmp_path_factory):
    base = tmp_path_factory.mktemp("osi_duckdb")
    model_dir = base / "models"
    model_dir.mkdir()
    (model_dir / "orders.yaml").write_text(_SEMANTIC_YAML, encoding="utf-8")

    db_path = base / "test.duckdb"
    import duckdb

    connection = duckdb.connect(str(db_path))
    try:
        connection.execute(f"CREATE SCHEMA {_SCHEMA}")
        connection.execute(
            f"CREATE TABLE {_SCHEMA}.{_DATA_TABLE} (id INTEGER, amount DECIMAL(10, 2), status VARCHAR, created_at DATE)"
        )
        connection.execute(
            f"INSERT INTO {_SCHEMA}.{_DATA_TABLE} VALUES "
            "(1, 10.00, 'completed', DATE '2020-01-01'), "
            "(2, 20.00, 'pending', DATE '2020-01-02'), "
            "(3, 30.00, 'completed', DATE '2020-01-03')"
        )
        connection.execute(f"CREATE TABLE {_SCHEMA}.{_TIME_SPINE_TABLE} (ds DATE NOT NULL)")
        connection.execute(
            f"INSERT INTO {_SCHEMA}.{_TIME_SPINE_TABLE} "
            "SELECT range::DATE FROM range("
            "DATE '2020-01-01', DATE '2026-01-01', INTERVAL '1 day')"
        )
    finally:
        connection.close()

    return DatusOSIAdapter(
        DatusOSIConfig(
            datasource="osi_nightly",
            semantic_models_path=str(model_dir),
            db_config={
                "type": "duckdb",
                "database": str(db_path),
                "schema": _SCHEMA,
            },
        )
    )


async def test_osi_query_contract_validates_and_discovers_metrics(osi_adapter):
    validation = await osi_adapter.validate_semantic()
    errors = [issue for issue in validation.issues if issue.severity == "error"]
    assert validation.valid, f"Unexpected validation errors: {errors}"

    metrics = await osi_adapter.list_metrics()
    assert {metric.name for metric in metrics} >= {"total_amount", "order_count"}
    dimensions = await osi_adapter.get_dimensions("total_amount")
    assert {dimension.name for dimension in dimensions} >= {"created_at", "status"}


async def test_osi_query_contract_dry_run_returns_sql(osi_adapter):
    result = await osi_adapter.query_metrics(["total_amount"], dry_run=True)

    assert result.metadata.get("sql")
    assert _DATA_TABLE in result.metadata["sql"]


async def test_osi_query_contract_executes_live_query(osi_adapter):
    result = await osi_adapter.query_metrics(["total_amount", "order_count"])

    assert result.data
    row = result.data[0]
    assert float(row["total_amount"]) == pytest.approx(60.0)
    assert int(row["order_count"]) == 3
