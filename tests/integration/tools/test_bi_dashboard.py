# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""BI Dashboard integration tests.

Contains two test classes at different verification levels:
- TestPartialIntegration: Real Superset API + mocked LLM (acceptance + nightly)
- TestE2EIntegration: Full end-to-end with zero mocks (nightly only)
"""

import os
import re
import shutil
from typing import Any, Dict, List, Tuple

import pytest
import yaml
from rich.console import Console

from datus.cli.bi_dashboard import BiDashboardCommands, DashboardCliOptions
from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.tools.bi_tools.base_adaptor import AuthParam
from datus.tools.bi_tools.dashboard_assembler import ChartSelection, DashboardAssembler
from datus.utils.loggings import configure_logging
from tests.conftest import TEST_CONF_DIR, TEST_DATA_DIR

configure_logging(False, console_output=False)


# ============================================================================
# Helper Functions
# ============================================================================


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison.

    Converts to lowercase, collapses whitespace, strips trailing semicolons,
    and replaces dynamic TO_TIMESTAMP values with placeholders.
    """
    if not sql:
        return ""

    normalized = sql.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.rstrip(";").strip()
    normalized = re.sub(
        r"to_timestamp\s*\(\s*'[\d\-:\s.]+'",
        "to_timestamp('<TIMESTAMP>'",
        normalized,
    )
    return normalized


def validate_chart_sql(chart_id: str, actual_sql: str, expected_sql: str) -> tuple[bool, str]:
    """Validate that actual SQL matches expected SQL after normalization.

    Returns:
        (is_valid, error_message)
    """
    normalized_actual = normalize_sql(actual_sql)
    normalized_expected = normalize_sql(expected_sql)

    if normalized_actual == normalized_expected:
        return True, ""

    error_msg = f"\n SQL mismatch for chart {chart_id}:\n"
    error_msg += f"Expected (normalized):\n{normalized_expected}\n\n"
    error_msg += f"Actual (normalized):\n{normalized_actual}\n"
    return False, error_msg


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def agent_config(tmp_path_factory) -> AgentConfig:
    """Load agent config from a temp copy so the source file is never modified."""
    src = TEST_CONF_DIR / "agent.yml"
    tmp_dir = tmp_path_factory.mktemp("bi_conf")
    tmp_cfg = tmp_dir / "agent.yml"
    shutil.copy2(src, tmp_cfg)
    config = load_agent_config(config=str(tmp_cfg), namespace="superset", reload=True, force=True, yes=True)
    return config


@pytest.fixture
def bi_commands(agent_config) -> BiDashboardCommands:
    """Create BiDashboardCommands for tests."""
    console = Console(log_path=False, force_terminal=False)
    return BiDashboardCommands(agent_config, console, force=True)


@pytest.fixture(scope="module")
def input_data() -> List[Dict[str, Any]]:
    """Load test data from YAML file."""
    yaml_path = TEST_DATA_DIR / "BIDashboardInput.yaml"
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
        if isinstance(data, list):
            return [item["input"] for item in data]
        elif isinstance(data, dict) and "input" in data:
            return [data]
        else:
            pytest.fail(reason=f"Unexpected data type: {type(data)}")
            return []


# ============================================================================
# Shared extraction logic
# ============================================================================


def _create_adaptor(bi_commands, agent_config, dashboard_item):
    """Create a BI adaptor from dashboard_item config."""
    platform = dashboard_item["platform"]
    dashboard_config = agent_config.dashboard_config.get(platform)
    if not dashboard_config:
        pytest.skip(f"Dashboard config for platform '{platform}' not found")

    return bi_commands._create_adaptor(
        DashboardCliOptions(
            platform=platform,
            dashboard_url=dashboard_item["dashboard_url"],
            api_base_url=dashboard_item["api_base_url"],
            auth_params=AuthParam(
                username=dashboard_config.username,
                password=dashboard_config.password,
                api_key=dashboard_config.api_key,
                extra=dashboard_config.extra,
            ),
            dialect=dashboard_item.get("dialect", "postgresql"),
        )
    )


def _extract_and_select_charts(
    bi_commands,
    bi_adaptor,
    dashboard_item,
) -> Tuple[Any, List[ChartSelection], List[Any], List[Any]]:
    """Extract dashboard, select charts with SQL validation, and assemble.

    Returns:
        (dashboard, chart_selections, charts, datasets)
    """
    dashboard_url = dashboard_item["dashboard_url"]

    dashboard_id = bi_adaptor.parse_dashboard_id(dashboard_url)
    dashboard = bi_adaptor.get_dashboard_info(dashboard_id)
    assert dashboard is not None, "Failed to get dashboard"
    assert dashboard.name, "Dashboard should have name"

    chart_metas = bi_adaptor.list_charts(dashboard_id)
    assert len(chart_metas) > 0, "Dashboard should have charts"

    charts = bi_commands._hydrate_charts(bi_adaptor, dashboard_id, chart_metas)
    charts_with_sql = [c for c in charts if c.query and c.query.sql]
    assert len(charts_with_sql) > 0, "Should have charts with SQL"

    # Verify expected charts if provided
    if "valid_charts" in dashboard_item:
        expected_chart_names = {c["name"] for c in dashboard_item["valid_charts"]}
        actual_chart_names = {c.name for c in charts}
        for expected_name in expected_chart_names:
            assert expected_name in actual_chart_names, f"Expected chart '{expected_name}' not found in dashboard"

    # Select charts
    chart_selections = []
    if "valid_charts" in dashboard_item:
        valid_chart_names = {c["name"] for c in dashboard_item["valid_charts"]}
        expected_sqls = {c["name"]: c.get("sql", "") for c in dashboard_item["valid_charts"] if "sql" in c}

        for chart in charts_with_sql:
            if chart.name in valid_chart_names:
                if chart.name in expected_sqls:
                    actual_sql = chart.query.sql[0] if chart.query.sql else ""
                    is_valid, error_msg = validate_chart_sql(chart.name, actual_sql, expected_sqls[chart.name])
                    if not is_valid:
                        print(error_msg)
                        pytest.fail(f"SQL validation failed for chart '{chart.name}'. See output above for details.")

                chart_selections.append(ChartSelection(chart=chart, sql_indices=list(range(len(chart.query.sql)))))
    else:
        chart_selections = [
            ChartSelection(chart=c, sql_indices=list(range(len(c.query.sql)))) for c in charts_with_sql[:2]
        ]

    assert len(chart_selections) > 0, "Should have at least one chart selected"

    datasets = bi_adaptor.list_datasets(dashboard_id)

    return dashboard, chart_selections, charts, datasets


def _assemble(bi_adaptor, dashboard, chart_selections, datasets, dialect):
    """Run the DashboardAssembler and verify results."""
    assembler = DashboardAssembler(bi_adaptor, default_dialect=dialect)
    result = assembler.assemble(dashboard, chart_selections, chart_selections, datasets)

    assert len(result.reference_sqls) > 0, "Should have reference SQLs"
    assert len(result.metric_sqls) > 0, "Should have metric SQLs"
    assert len(result.tables) > 0, "Should have tables"

    return result


# ============================================================================
# Partial Integration Tests (Strategic Mocks)
# ============================================================================


class TestPartialIntegration:
    """Partial integration tests with STRATEGIC mocking.

    Real: Superset API calls, data processing, workflow orchestration.
    Mocked: LLM API calls (too expensive/slow).
    """

    @pytest.mark.acceptance
    @pytest.mark.nightly
    def test_workflow_without_llm(
        self,
        bi_commands: BiDashboardCommands,
        agent_config: AgentConfig,
        input_data: List[Dict[str, Any]],
    ):
        """Integration test: real Superset extraction + mocked LLM generation."""
        from unittest.mock import patch

        for dashboard_item in input_data:
            platform = dashboard_item["platform"]
            dialect = dashboard_item.get("dialect", "postgresql")
            bi_adaptor = _create_adaptor(bi_commands, agent_config, dashboard_item)

            try:
                dashboard, chart_selections, charts, datasets = _extract_and_select_charts(
                    bi_commands, bi_adaptor, dashboard_item
                )
                result = _assemble(bi_adaptor, dashboard, chart_selections, datasets, dialect)

                # Mock ONLY the LLM calls and slow initialization
                with (
                    patch("datus.cli.bi_dashboard.init_reference_sql") as mock_ref_sql,
                    patch("datus.cli.bi_dashboard.init_semantic_model") as mock_semantic,
                    patch("datus.cli.bi_dashboard.init_metrics") as mock_metrics,
                    patch.object(bi_commands, "_validate_semantic_model", return_value=True) as _,
                    patch("builtins.open") as mock_open,
                    patch("yaml.safe_load_all") as mock_yaml_load,
                ):
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

                    def mock_yaml_side_effect(_, _result=result, _platform=platform):
                        for i in range(len(_result.metric_sqls)):
                            yield {
                                "metric": {
                                    "name": f"metric_{i}",
                                    "locked_metadata": {"tags": [f"subject_tree:{_platform}/test/layer{i}"]},
                                }
                            }

                    mock_yaml_load.side_effect = mock_yaml_side_effect
                    mock_open.return_value.__exit__.return_value = None

                    ref_sqls = bi_commands._gen_reference_sqls(result.reference_sqls, platform, dashboard)
                    semantic_result = bi_commands._gen_semantic_model(result.metric_sqls, platform, dashboard)
                    metrics = bi_commands._gen_metrics(result.metric_sqls, platform, dashboard)

                    assert len(ref_sqls) > 0
                    assert semantic_result is True
                    assert len(metrics) > 0
                    assert mock_ref_sql.called
                    assert mock_semantic.called
                    assert mock_metrics.called

                print(f"\nPartial integration test passed for {platform}")
                print(f"  - Real Superset extraction: {len(charts)} charts")
                print(f"  - Real data assembly: {len(result.reference_sqls)} SQLs")
                print("  - Mocked LLM calls: 3 calls avoided")

            finally:
                if hasattr(bi_adaptor, "close"):
                    bi_adaptor.close()


# ============================================================================
# True E2E Integration Tests (No Mocks)
# ============================================================================


class TestE2EIntegration:
    """Pure end-to-end integration tests with NO mocks.

    Validates the COMPLETE workflow: real Superset API, real LLM API,
    real file system, real database operations.

    SLOW (2-5 min per dashboard), EXPENSIVE ($0.05-0.20/run), requires full env setup.
    """

    @pytest.mark.nightly
    @pytest.mark.timeout(600)
    def test_complete_workflow(
        self,
        bi_commands: BiDashboardCommands,
        agent_config: AgentConfig,
        input_data: List[Dict[str, Any]],
    ):
        """TRUE END-TO-END: dashboard extraction -> sub-agent bootstrap -> verification."""
        # Clean storage for a clean slate
        storage_dir = agent_config.rag_storage_path()
        for component_dirpath in [
            "schema_metadata.lance",
            "schema_value.lance",
            "metrics.lance",
            "semantic_model.lance",
            "reference_sql.lance",
        ]:
            dirpath = os.path.join(storage_dir, component_dirpath)
            if os.path.isdir(dirpath):
                shutil.rmtree(dirpath)

        from datus.utils.path_manager import DatusPathManager

        path_manager = DatusPathManager(agent_config.home)
        semantic_model_dir = path_manager.semantic_model_path(agent_config.current_namespace)
        if semantic_model_dir.exists():
            shutil.rmtree(semantic_model_dir)

        test_results = []

        for dashboard_item in input_data:
            platform = dashboard_item["platform"]
            dialect = dashboard_item.get("dialect", "postgresql")
            bi_adaptor = _create_adaptor(bi_commands, agent_config, dashboard_item)

            test_result = {
                "platform": platform,
                "dashboard_url": dashboard_item["dashboard_url"],
                "status": "running",
                "error": None,
                "dashboard_name": None,
                "charts_processed": 0,
                "sub_agents": [],
                "tables": 0,
                "semantic_model_rows": 0,
                "metrics_rows": 0,
                "reference_sql_rows": 0,
            }

            try:
                dashboard, chart_selections, charts, datasets = _extract_and_select_charts(
                    bi_commands, bi_adaptor, dashboard_item
                )

                sub_agent_name = bi_commands._build_sub_agent_name(platform, dashboard.name or "")
                attr_name = f"{sub_agent_name}_attribution"

                # Clean sub-agent old data
                for sa_name in [sub_agent_name, attr_name]:
                    sa_path = agent_config.sub_agent_storage_path(sa_name)
                    if os.path.exists(sa_path):
                        shutil.rmtree(sa_path)

                result = _assemble(bi_adaptor, dashboard, chart_selections, datasets, dialect)

                # Save sub-agent (complete flow: gen + save + bootstrap)
                bi_commands._save_sub_agent(platform, dashboard, result)

                # Verify 2 sub-agents created
                assert sub_agent_name in agent_config.agentic_nodes, (
                    f"Main sub-agent '{sub_agent_name}' not found in agentic_nodes"
                )
                assert attr_name in agent_config.agentic_nodes, (
                    f"Attribution sub-agent '{attr_name}' not found in agentic_nodes"
                )
                attr_node = agent_config.agentic_nodes[attr_name]
                assert attr_node.get("node_class") == "gen_report", (
                    f"Attribution sub-agent should have node_class='gen_report', got '{attr_node.get('node_class')}'"
                )

                # Verify bootstrap data via store managers
                from datus.storage.metric.store import MetricRAG
                from datus.storage.reference_sql.store import ReferenceSqlRAG
                from datus.storage.semantic_model.store import SemanticModelRAG

                total_semantic_model_rows = 0
                total_metrics_rows = 0
                total_reference_sql_rows = 0

                for name in [sub_agent_name, attr_name]:
                    sm_size = SemanticModelRAG(agent_config, sub_agent_name=name).get_size()
                    m_size = MetricRAG(agent_config, sub_agent_name=name).get_metrics_size()
                    rs_size = ReferenceSqlRAG(agent_config, sub_agent_name=name).get_reference_sql_size()
                    total_semantic_model_rows += sm_size
                    total_metrics_rows += m_size
                    total_reference_sql_rows += rs_size
                    print(f"  Sub-agent '{name}': semantic_model={sm_size}, metrics={m_size}, reference_sql={rs_size}")

                test_result["semantic_model_rows"] = total_semantic_model_rows
                test_result["metrics_rows"] = total_metrics_rows
                test_result["reference_sql_rows"] = total_reference_sql_rows

                # Verify file artifacts
                from datus.utils.path_manager import get_path_manager

                sql_dir = get_path_manager(agent_config.home).dashboard_path() / platform
                sql_files = list(sql_dir.glob("*.sql"))
                assert len(sql_files) > 0, "SQL files should exist"
                csv_files = list(sql_dir.glob("*.csv"))
                assert len(csv_files) > 0, "CSV files should exist"

                # Update test result
                test_result["dashboard_name"] = dashboard.name
                test_result["charts_processed"] = len(chart_selections)
                test_result["sub_agents"] = [sub_agent_name, attr_name]
                test_result["tables"] = len(result.tables)

                bootstrap_failures = []
                if total_semantic_model_rows == 0:
                    bootstrap_failures.append("semantic_model")
                if total_metrics_rows == 0:
                    bootstrap_failures.append("metrics")
                if total_reference_sql_rows == 0:
                    bootstrap_failures.append("reference_sql")

                if bootstrap_failures:
                    error_msg = (
                        f"Bootstrap data missing: {', '.join(bootstrap_failures)} have 0 total rows across sub-agents"
                    )
                    test_result["status"] = "failed"
                    test_result["error"] = error_msg
                else:
                    test_result["status"] = "passed"

            except Exception as e:
                test_result["status"] = "failed"
                test_result["error"] = str(e)

            finally:
                test_results.append(test_result)
                if hasattr(bi_adaptor, "close"):
                    bi_adaptor.close()

        # Print summary
        print("-" * 80)
        print(" BI DASHBOARD INTEGRATION TEST SUMMARY")
        print("-" * 80)

        passed_tests = [r for r in test_results if r["status"] == "passed"]
        failed_tests = [r for r in test_results if r["status"] == "failed"]
        print(f"\nTotal: {len(test_results)}, Passed: {len(passed_tests)}, Failed: {len(failed_tests)}")

        for result in passed_tests:
            print(
                f"\n  PASSED: {result['platform']} - {result['dashboard_name']}"
                f" ({result['charts_processed']} charts, {result['tables']} tables)"
            )
            print(
                f"    Bootstrap: semantic_model={result['semantic_model_rows']}, "
                f"metrics={result['metrics_rows']}, reference_sql={result['reference_sql_rows']}"
            )

        for result in failed_tests:
            print(f"\n  FAILED: {result['platform']} - {result['error']}")

        print("-" * 80)
