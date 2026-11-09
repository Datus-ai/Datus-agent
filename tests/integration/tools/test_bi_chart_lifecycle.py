"""Deterministic Agent BI repair contract against a real Superset service."""

import uuid

import pytest
from datus_bi_core import ChartSpec
from rich.console import Console

from datus.cli.bootstrap_bi_picker import BootstrapBiPicker
from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.bi_tools import BIFuncTool
from datus.validation import ChartTarget, ValidationHook
from tests.integration.agent.test_gen_dashboard_agentic import SUPERSET_URL, dashboard_agent_config  # noqa: F401, F811
from tests.integration.tools.test_bi_dashboard import (
    _create_adapter,
    _ensure_superset_test_database,
    _get_or_create_sales_dataset,
)


@pytest.mark.nightly
@pytest.mark.product_e2e
@pytest.mark.asyncio
async def test_table_grouping_repair_and_deleted_chart_validation(dashboard_agent_config):  # noqa: F811
    config = dashboard_agent_config
    adapter = _create_adapter(
        BootstrapBiPicker(config, Console(log_path=False, force_terminal=False)),
        config,
        {"platform": "superset", "dashboard_url": SUPERSET_URL, "api_base_url": SUPERSET_URL},
    )
    tool = BIFuncTool(agent_config=config, bi_service="superset", adapter=adapter)
    hook = ValidationHook("gen_dashboard", None, None, bi_tool=tool, skill_validators_enabled=False)
    title = "P0 Repair " + uuid.uuid4().hex[:8]
    chart_ids = set()
    dashboard = tool.create_dashboard(title)
    assert dashboard.success, dashboard.error
    dashboard_id = str(dashboard.result["id"])

    async def collect(result):
        assert result.success, result.error
        await hook.on_tool_end(None, None, None, result)
        return result.result

    def verify_grouped(chart_id):
        data = adapter.get_chart_data(chart_id, limit=100)
        assert "region" in data.columns, data.columns
        (metric,) = [name for name in data.columns if name != "region"]
        assert sorted((row["region"], float(row[metric])) for row in data.rows) == [
            ("North", 2050.0),
            ("South", 1630.0),
            ("West", 1940.0),
        ]

    try:
        await collect(dashboard)
        dataset = _get_or_create_sales_dataset(adapter, _ensure_superset_test_database(adapter))
        # Recreate the saved configuration from the failed nightly, bypassing the Agent fix.
        broken = adapter.create_chart(
            ChartSpec(
                chart_type="table",
                title=title + " old",
                dataset_id=dataset.id,
                metrics=["SUM(revenue)"],
                dimensions=["region"],
                x_axis="region",
            ),
            dashboard_id=dashboard_id,
        )
        old_id = str(broken.id)
        chart_ids.add(old_id)
        await collect(
            FuncToolResult(
                result={
                    "deliverable_target": ChartTarget(
                        platform="superset",
                        chart_id=old_id,
                        dashboard_id=dashboard_id,
                    ).model_dump()
                }
            )
        )
        assert "region" not in adapter.get_chart_data(old_id).columns
        await collect(tool.update_chart(old_id, dimensions="region"))
        verify_grouped(old_id)
        await collect(tool.update_chart(old_id, title=title + " renamed"))
        verify_grouped(old_id)  # Omitted dimensions preserve saved grouping.
        await collect(tool.update_chart(old_id, dimensions=""))
        assert "region" not in adapter.get_chart_data(old_id).columns
        await collect(tool.update_chart(old_id, dimensions="region"))
        verify_grouped(old_id)

        replacement = await collect(
            tool.create_chart(
                "table",
                title + " replacement",
                dataset_id=str(dataset.id),
                metrics="SUM(revenue)",
                dimensions="region",
                x_axis="region",
                dashboard_id=dashboard_id,
            )
        )
        new_id = str(replacement["id"])
        chart_ids.add(new_id)
        verify_grouped(new_id)
        await collect(tool.add_chart_to_dashboard(new_id, dashboard_id))
        await collect(tool.delete_chart(old_id))
        chart_ids.remove(old_id)
        assert [t.chart_id for t in hook.session_targets if isinstance(t, ChartTarget)] == [new_id]
        assert [str(i) for i in adapter.get_dashboard_info(dashboard_id).chart_ids] == [new_id]
        await hook.on_end(None, None, None)
        assert hook.final_report is not None
        assert not hook.final_report.has_blocking_failure(), hook.final_report.model_dump()
    finally:
        for chart_id in chart_ids:
            adapter.delete_chart(chart_id)
        adapter.delete_dashboard(dashboard_id)
