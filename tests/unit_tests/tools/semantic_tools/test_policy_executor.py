# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Real local Dosi/SQL execution with fake host policy decisions; no external services."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import duckdb
import pytest

from datus.configuration.agent_config import AgentConfig
from datus.tools.semantic_tools.dosi.runtime import DosiRuntime
from datus.tools.semantic_tools.models import AttributionRequest, AttributionWindow
from datus.tools.semantic_tools.policy_executor import PolicyMetricExecutor
from datus.utils.exceptions import DatusException

MODEL = 'version: 0.2.0.dev0\nsemantic_model:\n  - name: sales\n    datasets:\n      - name: orders\n        source: orders\n        fields:\n          - name: day\n            datatype: Date\n            dimension:\n              is_time: true\n            expression:\n              dialects:\n                - dialect: DUCKDB\n                  expression: day\n          - name: region\n            expression:\n              dialects:\n                - dialect: DUCKDB\n                  expression: region\n          - name: amount\n            expression:\n              dialects:\n                - dialect: DUCKDB\n                  expression: amount\n    metrics:\n      - name: revenue\n        expression:\n          dialects:\n            - dialect: DUCKDB\n              expression: SUM(amount)\n        custom_extensions:\n          - vendor_name: DATUS\n            data: \'{"uid":"metric","dataset":"orders","time_dimension":"day"}\'\n'


@pytest.fixture
def execution(tmp_path, monkeypatch):
    """Create a real local Dosi model and DuckDB containing two independent regions."""
    database = tmp_path / "local.duckdb"
    with duckdb.connect(str(database)) as conn:
        conn.execute("CREATE TABLE orders(day DATE, region VARCHAR, amount DOUBLE)")
        conn.execute(
            "INSERT INTO orders VALUES ('2026-11-02','west',10),('2026-11-03','west',8),('2026-11-02','east',100),('2026-11-03','east',200)"
        )
    config = AgentConfig(
        nodes={},
        models={},
        services={"datasources": {"warehouse": {"type": "duckdb", "uri": str(database)}}},
        home=str(tmp_path),
        project_root=str(tmp_path),
        project_name="policytest",
        skip_init_dirs=True,
        agentic_nodes={},
        channels={},
    )
    config.current_datasource = "warehouse"
    model = config.path_manager.semantic_models_dir / "warehouse" / "sales.yml"
    model.parent.mkdir(parents=True)
    model.write_text(MODEL)
    monkeypatch.chdir(tmp_path)
    return config, PolicyMetricExecutor(config, DosiRuntime.from_agent_config(config)), model


def attribution():
    """Build an attribution request comparing adjacent daily periods."""
    return AttributionRequest(
        metric="revenue",
        dimensions=["orders.region"],
        time_dimension="metric_time",
        baseline=AttributionWindow(start="2026-11-02", end="2026-11-03"),
        current=AttributionWindow(start="2026-11-03", end="2026-11-04"),
    )


def install_policy(monkeypatch, *, kind="metric", deny=None):
    """Install observable policy hooks for scoped execution and independent denial tests."""
    calls = []

    class Policy:
        runtime_names = ()

        def __init__(self, config):
            """Accept the production policy constructor contract without external plugins."""
            pass

        def validate_context(self, context):
            """Record validation and optionally reject the supplied execution context."""
            calls.append(("validate", context))
            return SimpleNamespace(allowed=deny != "context")

        def before_metric_read(self, **kwargs):
            """Apply a region metric filter or reject the metric read."""
            calls.append(("metric", kwargs))
            where = f"orders.region = '{kwargs['policy_context']['region']}'" if kind == "metric" else kwargs["where"]
            return SimpleNamespace(allowed=deny != "metric", where=where)

        def before_sql_read(self, sql, **kwargs):
            """Rewrite each orders source with the SQL row policy or reject it."""
            from sqlglot import exp, parse_one

            calls.append(("sql", kwargs))
            if kind == "sql":
                tree = parse_one(sql)
                for table in list(tree.find_all(exp.Table)):
                    if table.name == "orders":
                        table.replace(
                            parse_one(
                                f"SELECT * FROM orders WHERE region = '{kwargs['policy_context']['region']}'"
                            ).subquery(table.alias_or_name)
                        )
                sql = tree.sql(dialect="duckdb")
            return SimpleNamespace(allowed=deny != "sql", sql=sql, applied_policies=[], reason="denied")

        def after_read_result(self, result, **kwargs):
            """Record the result-policy boundary and optionally reject its rows."""
            calls.append(("result", kwargs))
            return SimpleNamespace(allowed=deny != "result", result=result, applied_policies=[], reason="denied")

    monkeypatch.setattr("datus.tools.semantic_tools.policy_executor.PolicyRuntime", Policy)
    monkeypatch.setattr("datus.tools.policy_runtime.PolicyRuntime", Policy)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["metric", "sql"])
async def test_query_and_attribution_share_scope_and_do_not_mutate_shared_model(execution, monkeypatch, kind):
    """Real query and attribution results obey each region scope without model mutation."""
    _, executor, model = execution
    calls = install_policy(monkeypatch, kind=kind)
    original = model.read_bytes()
    for region, expected, baseline in [("west", 8, 10), ("east", 200, 100)]:
        result = await executor.query_metrics(
            "revenue",
            policy_context={"region": region},
            dimensions=["metric_time"],
            time_start="2026-11-03",
            time_end="2026-11-04",
            time_granularity="day",
        )
        assert len(result.data) == 1
        assert result.data[0]["revenue"] == expected
        explained = await executor.attribute(attribution(), policy_context={"region": region})
        assert explained.total_change.current_value == expected
        assert explained.total_change.baseline_value == baseline
    assert {call[0] for call in calls} == {"validate", "metric", "sql", "result"}
    assert model.read_bytes() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("denied", ["context", "metric", "sql", "result"])
async def test_query_denial_never_returns_rows(execution, monkeypatch, denied):
    """Every query policy stage can independently prevent result disclosure."""
    _, executor, _ = execution
    install_policy(monkeypatch, deny=denied)
    with pytest.raises(DatusException):
        await executor.query_metrics("revenue", policy_context={"region": "west"})


@pytest.mark.asyncio
@pytest.mark.parametrize("denied", ["context", "metric", "sql"])
async def test_attribution_denial_never_returns_evidence(execution, monkeypatch, denied):
    """Each supported attribution policy stage fails closed when denied."""
    _, executor, _ = execution
    install_policy(monkeypatch, deny=denied)
    with pytest.raises(DatusException):
        await executor.attribute(attribution(), policy_context={"region": "west"})


@pytest.mark.parametrize(
    "names,policies",
    [(("unknown",), []), (("sql-policy",), [{"type": "column_mask"}]), (("sql-policy",), [{"type": "result_filter"}])],
)
def test_attribution_rejects_unrepresentable_policy(names, policies):
    """Unknown plugins and non-row policies disable native attribution."""
    config = SimpleNamespace(get_plugin_profile=lambda _: {"policies": policies})
    with pytest.raises(DatusException, match="analysis_runtime_unavailable"):
        PolicyMetricExecutor.validate_attribution_support(config, SimpleNamespace(runtime_names=names))


def test_attribution_accepts_only_known_row_policy_forms():
    """Known SQL and metric row policies remain eligible for native attribution."""
    config = SimpleNamespace(
        get_plugin_profile=lambda _: {"policies": [{"type": "row_filter"}, {"type": "metric_row_filter"}]}
    )
    assert (
        PolicyMetricExecutor.validate_attribution_support(config, SimpleNamespace(runtime_names=("sql-policy",)))
        is None
    )
    assert PolicyMetricExecutor.validate_attribution_support(config, SimpleNamespace(runtime_names=())) is None


@pytest.mark.asyncio
async def test_dry_run_and_missing_dataset_are_denied_before_execution(execution, monkeypatch):
    """Dry-run SQL exposure and unresolved metric scope fail before query execution."""
    _, executor, _ = execution
    install_policy(monkeypatch)
    query = AsyncMock()
    monkeypatch.setattr(executor.runtime, "query_metrics", query)
    with pytest.raises(DatusException, match="dry-run"):
        await executor.query_metrics("revenue", policy_context={}, dry_run=True)
    monkeypatch.setattr(executor.runtime, "artifact_metric_binding", lambda _: {"datasets": []})
    with pytest.raises(DatusException, match="datasets"):
        await executor.query_metrics("revenue", policy_context={})
    query.assert_not_awaited()
