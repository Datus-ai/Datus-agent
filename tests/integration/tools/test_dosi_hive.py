# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Real query_metrics through the native HS2 connector; no business data writes.

Set DOSI_TEST_HIVE_CONNECTIONS to a credential-free/local connections file (or
an externally provisioned secret file). Both Hive and Inceptor use type=hive.
The model's inline datasets keep the expected results independent of live data.
"""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

pytestmark = pytest.mark.nightly


def _expression(sql):
    return {"dialects": [{"dialect": "ANSI_SQL", "expression": sql}]}


def _model():
    source = """
        SELECT 1 AS order_id, CAST('中文' AS STRING) AS label, CAST('2024-01-15' AS DATE) AS order_date,
               CAST(10.50 AS DECIMAL(18,2)) AS amount, 10 AS customer_id
        UNION ALL SELECT 2, CAST('中文' AS STRING), CAST('2024-01-20' AS DATE), CAST(20 AS DECIMAL(18,2)), 10
        UNION ALL SELECT 3, CAST('English' AS STRING), CAST('2024-02-02' AS DATE), CAST(0 AS DECIMAL(18,2)), 20
        UNION ALL SELECT 4, CAST(NULL AS STRING), CAST('2024-02-29' AS DATE), CAST(NULL AS DECIMAL(18,2)), CAST(NULL AS INT)
    """
    fields = [
        {"name": name, "expression": _expression(name)}
        for name in ("order_id", "label", "order_date", "amount", "customer_id")
    ]
    fields[2]["dimension"] = {"is_time": True}
    metrics = [
        {"name": name, "expression": _expression(sql)}
        for name, sql in (
            ("order_count", "COUNT(orders.order_id)"),
            ("revenue", "SUM(orders.amount)"),
            ("unique_customers", "COUNT(DISTINCT orders.customer_id)"),
            ("avg_order_value", "SUM(orders.amount) / COUNT(orders.order_id)"),
            ("min_label", "MIN(orders.label)"),
            ("max_label", "MAX(orders.label)"),
            ("customer_count", "COUNT(customers.customer_id)"),
            ("elapsed_hours", "SUM(DATEDIFF('hour', clock.started_at, clock.ended_at))"),
            ("elapsed_minutes", "SUM(DATEDIFF('minute', clock.started_at, clock.ended_at))"),
            ("elapsed_seconds", "SUM(DATEDIFF('second', clock.started_at, clock.ended_at))"),
            ("scaled_hours", "SUM(2 * DATEDIFF('hour', clock.started_at, clock.ended_at) + 1)"),
        )
    ]
    metrics.append(
        {
            "name": "running_revenue",
            "expression": _expression("SUM(orders.amount)"),
            "custom_extensions": [
                {
                    "vendor_name": "DATUS",
                    "data": json.dumps({"v": "1.2", "window": {"type": "cumulative", "function": "sum"}}),
                }
            ],
        }
    )
    for name, window in [
        ("rolling_revenue", {"type": "rolling", "function": "sum", "periods": 2}),
        ("previous_revenue", {"offset": {"count": 1, "granularity": "month"}, "calculation": "value"}),
    ]:
        metrics.append(
            {
                "name": name,
                "expression": _expression("SUM(orders.amount)"),
                "custom_extensions": [{"vendor_name": "DATUS", "data": json.dumps({"v": "1.2", "window": window})}],
            }
        )
    return {
        "version": "0.2.0.dev0",
        "semantic_model": [
            {
                "name": "hive_acceptance",
                "datasets": [
                    {"name": "orders", "source": source, "primary_key": ["order_id"], "fields": fields},
                    {
                        "name": "customers",
                        "source": "SELECT 10 AS customer_id, '北区' AS region UNION ALL SELECT 20, '南区'",
                        "primary_key": ["customer_id"],
                        "fields": [
                            {"name": name, "expression": _expression(name)} for name in ("customer_id", "region")
                        ],
                    },
                    {
                        "name": "clock",
                        "source": (
                            "SELECT CAST('2024-01-01 00:20:00' AS TIMESTAMP) AS started_at, "
                            "CAST('2024-01-01 00:40:00' AS TIMESTAMP) AS ended_at "
                            "UNION ALL SELECT CAST('2023-12-31 23:59:59.9' AS TIMESTAMP), "
                            "CAST('2024-01-01 00:00:00.1' AS TIMESTAMP)"
                        ),
                        "fields": [
                            {"name": name, "expression": _expression(name)} for name in ("started_at", "ended_at")
                        ],
                    },
                ],
                "relationships": [
                    {
                        "name": "orders_to_customers",
                        "from": "orders",
                        "to": "customers",
                        "from_columns": ["customer_id"],
                        "to_columns": ["customer_id"],
                    }
                ],
                "metrics": metrics,
            }
        ],
    }


@pytest.fixture(scope="module")
def hive_tool(tmp_path_factory):
    path = os.environ.get("DOSI_TEST_HIVE_CONNECTIONS")
    if not path:
        pytest.skip("set DOSI_TEST_HIVE_CONNECTIONS for live HS2 acceptance")
    from datus_hive import register

    from datus.configuration.agent_config import AgentConfig, NodeConfig
    from datus.tools.func_tool.semantic_tools import SemanticTools

    register()
    profiles = yaml.safe_load(Path(path).read_text())["datasources"]
    profile = profiles[os.environ.get("DOSI_TEST_HIVE_CONNECTION", "hive")]
    profile["configuration"] = {**profile.get("configuration", {}), "hive.local.time.zone": "Asia/Kolkata"}
    root = tmp_path_factory.mktemp("dosi_hive")
    config = AgentConfig(
        nodes={"semantic": NodeConfig(model="mock", input=None)},
        home=str(root / "home"),
        project_name="hive_acceptance",
        project_root=str(root / "workspace"),
        target="mock",
        models={"mock": {"type": "openai", "api_key": "unused", "model": "unused", "base_url": "http://127.0.0.1:1"}},
        services={"datasources": {"hive_acceptance": {**profile, "default": True}}},
    )
    config.current_datasource = "hive_acceptance"
    model_dir = Path(config.path_manager.semantic_model_path("hive_acceptance"))
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.yaml").write_text(yaml.safe_dump(_model(), allow_unicode=True))
    with patch("datus.tools.func_tool.semantic_tools.MetricRAG"):
        tool = SemanticTools(agent_config=config)
    assert type(tool.runtime).__module__ == "datus.tools.semantic_tools.dosi.runtime"
    return tool, profile, model_dir


def _rows(tool, result):
    assert result.success == 1, result.error
    page = tool.get_query_metrics_result(result.result["result_id"])
    assert page.success == 1, page.error
    return list(csv.DictReader(io.StringIO(page.result["csv"])))


def test_native_hive_aggregates_and_ratio(hive_tool):
    tool, _, _ = hive_tool
    rows = _rows(tool, tool.query_metrics(metrics=["order_count", "revenue", "unique_customers", "avg_order_value"]))
    assert len(rows) == 1
    assert int(rows[0]["order_count"]) == 4
    assert float(rows[0]["revenue"]) == 30.5
    assert int(rows[0]["unique_customers"]) == 2
    assert float(rows[0]["avg_order_value"]) == 7.625


def test_native_hive_unicode_filter_and_minmax(hive_tool):
    tool, _, _ = hive_tool
    rows = _rows(
        tool,
        tool.query_metrics(
            metrics=["order_count", "revenue", "min_label", "max_label"],
            dimensions=["orders.label"],
            where="orders.label = '中文'",
        ),
    )
    assert len(rows) == 1
    assert rows[0]["label"] == rows[0]["min_label"] == rows[0]["max_label"] == "中文"
    assert int(rows[0]["order_count"]) == 2
    assert float(rows[0]["revenue"]) == 30.5


@pytest.mark.parametrize(
    "predicate",
    [
        "NOT orders.label IS NOT DISTINCT FROM '中文'",
        "NOT (orders.label IS NOT DISTINCT FROM '中文')",
        "orders.label IS DISTINCT FROM '中文'",
    ],
)
def test_native_hive_negated_null_safe_filter(hive_tool, predicate):
    tool, _, _ = hive_tool
    rows = _rows(tool, tool.query_metrics(metrics=["order_count"], where=predicate))
    assert len(rows) == 1
    assert int(rows[0]["order_count"]) == 2


def test_native_hive_wall_clock_date_differences(hive_tool):
    tool, _, _ = hive_tool
    metrics = ["elapsed_hours", "elapsed_minutes", "elapsed_seconds", "scaled_hours"]
    for predicate, expected in [
        (None, [1, 21, 1201, 4]),
        ("clock.started_at = CAST('2024-01-01 00:20:00' AS TIMESTAMP)", [0, 20, 1200, 1]),
    ]:
        rows = _rows(tool, tool.query_metrics(metrics=metrics, where=predicate))
        assert len(rows) == 1
        assert [int(rows[0][metric]) for metric in metrics] == expected


def test_native_hive_month_and_time_range(hive_tool):
    tool, _, _ = hive_tool
    rows = _rows(
        tool,
        tool.query_metrics(
            metrics=["order_count", "revenue"],
            dimensions=["orders.order_date"],
            time_granularity="month",
            time_start="2024-01-01",
            time_end="2024-03-01",
            order_by=["orders.order_date:month"],
        ),
    )
    assert [row["order_date__month"][:7] for row in rows] == ["2024-01", "2024-02"]
    assert [int(row["order_count"]) for row in rows] == [2, 2]
    assert [float(row["revenue"]) for row in rows] == [30.5, 0.0]


def test_native_hive_join_keeps_unmatched_orders(hive_tool):
    tool, _, _ = hive_tool
    rows = _rows(tool, tool.query_metrics(metrics=["order_count"], dimensions=["customers.region"]))
    assert {row["region"]: int(row["order_count"]) for row in rows} == {"北区": 2, "南区": 1, "": 1}


def test_native_hive_cumulative_window(hive_tool):
    tool, _, _ = hive_tool
    rows = _rows(
        tool,
        tool.query_metrics(
            metrics=["running_revenue"],
            dimensions=["orders.order_date"],
            time_granularity="month",
            order_by=["orders.order_date:month"],
        ),
    )
    assert [row["order_date__month"][:7] for row in rows] == ["2024-01", "2024-02"]
    assert [float(row["running_revenue"]) for row in rows] == [30.5, 30.5]


def test_native_hive_merges_metrics_from_different_datasets(hive_tool):
    tool, _, _ = hive_tool
    rows = _rows(tool, tool.query_metrics(metrics=["order_count", "customer_count"], dimensions=["customers.region"]))
    assert {row["region"]: (int(row["order_count"]), int(row["customer_count"])) for row in rows} == {
        "北区": (2, 1),
        "南区": (1, 1),
        "": (1, 0),
    }


def test_native_hive_rolling_and_previous_period(hive_tool):
    tool, _, _ = hive_tool
    rows = _rows(
        tool,
        tool.query_metrics(
            metrics=["rolling_revenue", "previous_revenue"],
            dimensions=["orders.order_date"],
            time_granularity="month",
            order_by=["orders.order_date:month"],
        ),
    )
    assert [float(row["rolling_revenue"]) for row in rows] == [30.5, 30.5]
    assert [row["previous_revenue"] for row in rows] == ["", "30.5"]


def test_native_hive_dry_run_selects_server_target(hive_tool):
    tool, profile, _ = hive_tool
    result = tool.query_metrics(metrics=["revenue"], dry_run=True)
    assert result.success == 1, result.error
    expected = "inceptor" if profile.get("server_type") == "inceptor" else "hive"
    assert (
        tool.runtime._dry_run_dialect(
            __import__("dosi_engine"), tool.runtime._handle_for_metrics(["revenue"]).profile_name
        )
        == expected
    )


def test_native_inceptor_system_varchar2_metrics(hive_tool):
    tool, profile, model_dir = hive_tool
    if profile.get("server_type") != "inceptor":
        pytest.skip("Inceptor-specific system view and VARCHAR2 metadata")
    model = {
        "version": "0.2.0.dev0",
        "semantic_model": [
            {
                "name": "catalog_acceptance",
                "datasets": [
                    {
                        "name": "catalog_columns",
                        "source": "system.columns_v",
                        "fields": [{"name": "column_name", "expression": _expression("column_name")}],
                    }
                ],
                "metrics": [
                    {"name": name, "expression": _expression(sql)}
                    for name, sql in (
                        ("catalog_count", "COUNT(catalog_columns.column_name)"),
                        ("catalog_min", "MIN(catalog_columns.column_name)"),
                        ("catalog_max", "MAX(catalog_columns.column_name)"),
                    )
                ],
            }
        ],
    }
    (model_dir / "catalog.yaml").write_text(yaml.safe_dump(model))
    rows = _rows(tool, tool.query_metrics(metrics=["catalog_count", "catalog_min", "catalog_max"]))
    assert len(rows) == 1
    assert int(rows[0]["catalog_count"]) > 0
    assert rows[0]["catalog_min"] and rows[0]["catalog_max"]
