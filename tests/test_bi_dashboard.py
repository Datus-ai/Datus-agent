# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.


from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml
from rich.console import Console

from datus.cli.bi_dashboard import BiDashboardCommands, DashboardCliOptions
from datus.configuration.agent_config import AgentConfig
from datus.tools.bi_tools.base_adaptor import AuthParam
from datus.tools.bi_tools.dashboard_assembler import ChartSelection, DashboardAssembler
from datus.utils.loggings import configure_logging
from tests.conftest import load_acceptance_config

configure_logging(False, console_output=False)

# ============================================================================
# Helper Functions
# ============================================================================


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL for comparison by:
    - Converting to lowercase
    - Removing extra whitespace
    - Removing trailing semicolons
    - Normalizing line breaks
    - Replacing dynamic timestamps with placeholders
    """
    import re

    if not sql:
        return ""

    # Convert to lowercase
    normalized = sql.lower().strip()

    # Replace multiple whitespace with single space
    normalized = re.sub(r"\s+", " ", normalized)

    # Remove trailing semicolon
    normalized = normalized.rstrip(";").strip()

    # Replace dynamic timestamps in TO_TIMESTAMP functions with placeholder
    # Pattern matches: to_timestamp('YYYY-MM-DD HH:MI:SS.FFFFFF', 'format')
    # The timestamp value changes on each run, so we normalize it
    normalized = re.sub(
        r"to_timestamp\s*\(\s*'[\d\-:\s.]+'",
        "to_timestamp('<TIMESTAMP>'",
        normalized,
    )

    return normalized


def validate_chart_sql(chart_id: str, actual_sql: str, expected_sql: str) -> tuple[bool, str]:
    """
    Validate that actual SQL matches expected SQL.

    Returns:
        (is_valid, error_message)
    """
    normalized_actual = normalize_sql(actual_sql)
    normalized_expected = normalize_sql(expected_sql)

    if normalized_actual == normalized_expected:
        return True, ""

    # Generate detailed error message
    error_msg = f"\n ❌ SQL mismatch for chart {chart_id}:\n"
    error_msg += f"Expected (normalized):\n{normalized_expected}\n\n"
    error_msg += f"Actual (normalized):\n{normalized_actual}\n"

    return False, error_msg


@pytest.fixture(scope="module")
def agent_config() -> AgentConfig:
    """Load agent config with superset namespace."""
    config = load_acceptance_config(namespace="superset")
    return config


@pytest.fixture
def bi_commands(agent_config) -> BiDashboardCommands:
    """Create BiDashboardCommands for E2E tests."""
    console = Console(log_path=False, force_terminal=False)
    return BiDashboardCommands(agent_config, console)


@pytest.fixture(scope="module")
def input_data() -> List[Dict[str, Any]]:
    """Load test data from YAML file."""
    yaml_path = Path(__file__).parent / "data" / "BIDashboardInput.yaml"
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
        # Handle both list format and dict with 'input' key
        if isinstance(data, list):
            return [item["input"] for item in data]
        elif isinstance(data, dict) and "input" in data:
            return [data]
        else:
            pytest.fail(reason=f"Unexpected data type: {type(data)}")
            return []


# ============================================================================
# Partial Integration Tests (Strategic Mocks)
# ============================================================================


class TestPartialIntegration:
    """
    Partial integration tests with STRATEGIC mocking.

    These tests mock ONLY the expensive/slow parts:
    - LLM API calls (mocked)
    - File system (isolated)

    But keep real:
    - Superset API calls
    - Data processing logic
    - Workflow orchestration
    """

    @pytest.mark.acceptance
    def test_workflow_without_llm(
        self,
        bi_commands: BiDashboardCommands,
        agent_config: AgentConfig,
        input_data: List[Dict[str, Any]],
    ):
        """
        Integration test WITHOUT LLM calls.

        This tests:
        ✓ Real Superset API interaction
        ✓ Real data extraction
        ✓ Real workflow orchestration

        But mocks:
        ✗ LLM API calls (too expensive/slow)

        This is faster and cheaper than true E2E but still validates
        the core integration between components.
        """
        from unittest.mock import patch

        for dashboard_item in input_data:
            # Extract configuration
            platform = dashboard_item["platform"]
            dashboard_url = dashboard_item["dashboard_url"]
            api_base_url = dashboard_item["api_base_url"]
            dialect = dashboard_item.get("dialect", "postgresql")

            # Get dashboard config
            dashboard_config = agent_config.dashboard_config.get(platform)
            if not dashboard_config:
                pytest.skip(f"Dashboard config for platform '{platform}' not found")

            # Create BI adaptor
            bi_adaptor = bi_commands._create_adaptor(
                DashboardCliOptions(
                    platform=platform,
                    dashboard_url=dashboard_url,
                    api_base_url=api_base_url,
                    auth_params=AuthParam(
                        username=dashboard_config.username,
                        password=dashboard_config.password,
                        api_key=dashboard_config.api_key,
                        extra=dashboard_config.extra,
                    ),
                    dialect=dialect,
                )
            )

            try:
                # Real Superset extraction
                dashboard_id = bi_adaptor.parse_dashboard_id(dashboard_url)
                dashboard = bi_adaptor.get_dashboard_info(dashboard_id)
                chart_metas = bi_adaptor.list_charts(dashboard_id)
                charts = bi_commands._hydrate_charts(bi_adaptor, dashboard_id, chart_metas)

                # Real data assembly
                assembler = DashboardAssembler(
                    bi_adaptor,
                    default_dialect=dialect,
                )

                charts_with_sql = [c for c in charts if c.query and c.query.sql]

                # Select charts based on valid_charts or default to first 2
                chart_selections = []
                if "valid_charts" in dashboard_item:
                    # Use valid_charts from YAML configuration - match by name (more stable than ID)
                    valid_chart_names = {c["name"] for c in dashboard_item["valid_charts"]}
                    # Build map of chart_name -> expected_sql for validation
                    expected_sqls = {c["name"]: c.get("sql", "") for c in dashboard_item["valid_charts"] if "sql" in c}
                    print(f"\n           Using valid_charts from config: {valid_chart_names}")
                    if expected_sqls:
                        print(f"           Will validate SQL for {len(expected_sqls)} charts")

                    # Select charts that match valid_chart_names
                    for chart in charts_with_sql:
                        if chart.name in valid_chart_names:
                            # Validate SQL if expected SQL is provided
                            if chart.name in expected_sqls:
                                actual_sql = chart.query.sql[0] if chart.query.sql else ""
                                expected_sql = expected_sqls[chart.name]

                                is_valid, error_msg = validate_chart_sql(chart.name, actual_sql, expected_sql)
                                if not is_valid:
                                    print(error_msg)
                                    pytest.fail(
                                        f"SQL validation failed for chart '{chart.name}'. "
                                        f"See output above for details."
                                    )
                                else:
                                    print(f"           ✓ SQL validated for chart '{chart.name}'")

                            chart_selections.append(
                                ChartSelection(chart=chart, sql_indices=list(range(len(chart.query.sql))))
                            )

                    # Verify we found all expected charts
                    selected_chart_names = {cs.chart.name for cs in chart_selections}
                    if selected_chart_names != valid_chart_names:
                        missing_names = valid_chart_names - selected_chart_names
                        print(f"           Warning: Could not find charts with SQL for names: {missing_names}")
                else:
                    # Fallback: Select first 2 charts for testing (to save time/cost)
                    print("\n           No valid_charts specified, using first 2 charts")
                    chart_selections = [
                        ChartSelection(chart=c, sql_indices=list(range(len(c.query.sql)))) for c in charts_with_sql[:2]
                    ]

                assert len(chart_selections) > 0, "Should have at least one chart selected"
                print(f"           Selected {len(chart_selections)} charts for processing")

                datasets = bi_adaptor.list_datasets(dashboard_id)
                result = assembler.assemble(dashboard, chart_selections, chart_selections, datasets)

                # Verify real data extraction worked
                assert len(result.reference_sqls) > 0
                assert len(result.metric_sqls) > 0
                assert len(result.tables) > 0

                # Mock ONLY the LLM calls and slow initialization
                with (
                    patch("datus.cli.bi_dashboard.init_reference_sql") as mock_ref_sql,
                    patch("datus.cli.bi_dashboard.init_semantic_model") as mock_semantic,
                    patch("datus.cli.bi_dashboard.init_metrics") as mock_metrics,
                    patch.object(
                        bi_commands, "_validate_semantic_model", return_value=True
                    ) as _,  # Skip MetricFlowAdapter init
                    patch("builtins.open") as mock_open,
                    patch("yaml.safe_load_all") as mock_yaml_load,
                ):
                    # Setup mocks
                    mock_ref_sql.return_value = {
                        "status": "success",
                        "valid_entries": len(result.reference_sqls),
                        "invalid_entries": 0,
                        "processed_entries": len(result.reference_sqls),
                        "processed_items": [
                            {"subject_tree": f"{platform}/test/metric", "name": f"metric_{i}"}
                            for i in range(len(result.reference_sqls))
                        ],
                    }

                    mock_semantic.return_value = (True, {"semantic_model_count": len(result.metric_sqls)})

                    # Mock metrics with fake YAML file paths
                    fake_metric_files = [f"fake_metric_{i}.yaml" for i in range(len(result.metric_sqls))]
                    mock_metrics.return_value = (
                        True,
                        {
                            "semantic_models": fake_metric_files,
                            "success": True,
                            "response": "Metrics generated successfully",
                            "tokens_used": 1000,
                        },
                    )

                    # Mock YAML file content for metrics
                    # Each file should have its own metric data
                    def mock_yaml_side_effect(_, _result=result, _platform=platform):
                        # Return a generator that yields metric documents
                        for i in range(len(_result.metric_sqls)):
                            yield {
                                "metric": {
                                    "name": f"metric_{i}",
                                    "locked_metadata": {"tags": [f"subject_tree:{_platform}/test/layer{i}"]},
                                }
                            }

                    mock_yaml_load.side_effect = mock_yaml_side_effect
                    # Configure mock_open to return a context manager
                    mock_open.return_value.__exit__.return_value = None

                    # Call the generation methods
                    ref_sqls = bi_commands._gen_reference_sqls(result.reference_sqls, platform, dashboard)

                    semantic_result = bi_commands._gen_semantic_model(result.metric_sqls, platform, dashboard)

                    metrics = bi_commands._gen_metrics(result.metric_sqls, platform, dashboard)

                    # Verify workflow orchestration worked
                    assert len(ref_sqls) > 0
                    assert semantic_result is True
                    assert len(metrics) > 0

                    # Verify LLM methods were called with correct data
                    assert mock_ref_sql.called
                    assert mock_semantic.called
                    assert mock_metrics.called

                print(f"\n✓ Partial integration test passed for {platform}")
                print(f"  - Real Superset extraction: {len(charts)} charts")
                print(f"  - Real data assembly: {len(result.reference_sqls)} SQLs")
                print("  - Mocked LLM calls: 3 calls avoided")

            finally:
                # Clean up
                if hasattr(bi_adaptor, "close"):
                    bi_adaptor.close()
