"""Phase 3 closure: an OSI-authored model is validated/queried via the osi adapter.

When the active semantic adapter is ``osi``, the generation flow's
``validate_semantic`` / ``query_metrics`` calls are served by DatusOSIAdapter,
which compiles OSI -> IR -> MetricFlow YAML and returns business-semantic errors.
This test drives that adapter the same way SemanticTools does (via the registry),
proving the loop closes without the LLM ever writing MetricFlow syntax.

Skipped when ``datus_semantic_osi`` or its MetricFlow backend dependency is not
installed.
"""

import pytest
import sqlglot
from sqlglot import expressions as exp

pytest.importorskip("datus_semantic_osi")
pytest.importorskip("metricflow")

from datus.tools.semantic_tools.registry import semantic_adapter_registry

# Deterministic adapter-closure test, but it depends on the optional
# ``datus_semantic_osi`` / ``metricflow`` packages that are only guaranteed to be
# installed in the nightly environment. Mark it nightly so the broad nightly suite
# runs it (the importorskip above still skips gracefully when the deps are absent).
pytestmark = pytest.mark.nightly

GOOD_OSI = """
version: 0.2.0.dev0
semantic_model:
  - name: shop
    datasets:
      - name: orders
        source: orders
        primary_key: [order_id]
        fields:
          - name: order_date
            expression:
              dialects:
                - dialect: ANSI_SQL
                  expression: order_date
            dimension: {is_time: true}
            custom_extensions:
              - vendor_name: DATUS
                data: '{"type":"time","time_granularity":"day"}'
    metrics:
      - name: order_count
        description: "number of orders"
        expression:
          dialects:
            - dialect: ANSI_SQL
              expression: "COUNT(DISTINCT order_id)"
        custom_extensions:
          - vendor_name: DATUS
            data: '{"dataset":"orders","time_dimension":"order_date"}'
"""

# A detail query must not be modeled as a metric -> business-semantic error.
BAD_OSI = """
version: 0.2.0.dev0
semantic_model:
  - name: shop
    datasets:
      - name: orders
        source: orders
    metrics:
      - name: ranked
        expression:
          dialects:
            - dialect: ANSI_SQL
              expression: "RANK() OVER (ORDER BY amount DESC)"
        custom_extensions:
          - vendor_name: DATUS
            data: '{"dataset":"orders"}'
"""


def _make_adapter(osi_dir):
    from datus_semantic_osi import register
    from datus_semantic_osi.config import DatusOSIConfig

    register()
    config = DatusOSIConfig(semantic_models_path=str(osi_dir), datasource="orders", execution_backend="metricflow")
    return semantic_adapter_registry.create_adapter("osi", config)


def _has_count_distinct_order_id(sql: str) -> bool:
    parsed = sqlglot.parse_one(sql)
    for count in parsed.find_all(exp.Count):
        normalized = count.sql(dialect="mysql").lower().replace("`", "").replace('"', "")
        if "count(distinct order_id)" in normalized:
            return True
    return False


@pytest.mark.asyncio
async def test_good_osi_validates_and_dry_runs(tmp_path):
    (tmp_path / "model.yaml").write_text(GOOD_OSI)
    adapter = _make_adapter(tmp_path)

    result = await adapter.validate_semantic()
    errors = [i.message for i in result.issues if i.severity == "error"]
    assert result.valid, f"errors: {errors}"

    metrics = {m.name for m in await adapter.list_metrics()}
    assert "order_count" in metrics

    q = await adapter.query_metrics(["order_count"], dry_run=True)
    assert _has_count_distinct_order_id(q.metadata["sql"])


@pytest.mark.asyncio
async def test_bad_osi_returns_business_semantic_error(tmp_path):
    (tmp_path / "model.yaml").write_text(BAD_OSI)
    adapter = _make_adapter(tmp_path)

    result = await adapter.validate_semantic()
    assert not result.valid
    messages = " ".join(i.message for i in result.issues).lower()
    # business-semantic phrasing, not MetricFlow YAML internals; either term
    # signals the adapter surfaced a business-level error (intentional either-or).
    assert "window" in messages or "ranking" in messages  # audit-noqa: or_assert
    assert "type_params" not in messages
