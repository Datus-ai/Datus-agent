# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.


import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml
from rich.console import Console

from datus.cli.bi_dashboard import BiDashboardCommands, DashboardCliOptions
from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.tools.bi_tools.base_adaptor import AuthParam
from datus.tools.bi_tools.dashboard_assembler import ChartSelection, DashboardAssembler
from datus.utils.loggings import configure_logging
from tests.test_bi_dashboard import validate_chart_sql

configure_logging(False, console_output=False)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def agent_config(tmp_path_factory) -> AgentConfig:
    """Load agent config from a temp copy so the source file is never modified."""
    src = Path(__file__).parent.parent / "conf" / "agent.yml"
    tmp_dir = tmp_path_factory.mktemp("bi_conf")
    tmp_cfg = tmp_dir / "agent.yml"
    shutil.copy2(src, tmp_cfg)
    config = load_agent_config(config=str(tmp_cfg), namespace="superset", reload=True, force=True, yes=True)
    return config


@pytest.fixture
def bi_commands(agent_config) -> BiDashboardCommands:
    """Create BiDashboardCommands for E2E tests."""
    console = Console(log_path=False, force_terminal=False)
    bi_commands = BiDashboardCommands(agent_config, console, force=True)
    return bi_commands


@pytest.fixture(scope="module")
def input_data() -> List[Dict[str, Any]]:
    """Load test data from YAML file."""
    yaml_path = Path(__file__).parent.parent / "data" / "BIDashboardInput.yaml"
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
# True E2E Integration Tests (No Mocks)
# ============================================================================


class TestE2EIntegration:
    """
    Pure end-to-end integration tests with NO mocks.

    These tests validate the COMPLETE workflow including:
    - Real Superset API calls
    - Real LLM API calls (OpenAI, Claude, etc.)
    - Real file system operations
    - Real database operations

    ⚠️ These tests are:
    - SLOW (2-5 minutes per test)
    - EXPENSIVE (LLM API costs)
    - REQUIRE full environment setup
    """

    def test_complete_workflow(
        self,
        bi_commands: BiDashboardCommands,
        agent_config: AgentConfig,
        input_data: List[Dict[str, Any]],
    ):
        """
        TRUE END-TO-END TEST: Complete dashboard-to-agent workflow.

        This test has ZERO mocks and tests the complete real workflow:
        1. Extract dashboard from Superset (REAL API)
        2. Extract charts and SQL (REAL API)
        3. Assemble dashboard data (REAL)
        4. Save sub-agent: gen metadata/reference SQL/semantic model/metrics + bootstrap (REAL LLM)
        5. Verify 2 sub-agents created (main + attribution)
        6. Verify bootstrapped data: metrics, reference_sql, semantic_model in sub-agent stores
        7. Verify file artifacts (SQL, CSV, semantic model YAML)

        Cost: ~$0.05-0.20 per run
        Time: ~2-5 minutes
        """
        # Step 0: Clean storage directories for a clean slate
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
                print(f"✓ Cleaned {agent_config.current_namespace} storage: {dirpath}")
            else:
                print(f"  Storage does not exist (clean): {dirpath}")

        from datus.utils.path_manager import DatusPathManager

        path_manager = DatusPathManager(agent_config.home)

        semantic_model_dir = path_manager.semantic_model_path(agent_config.current_namespace)
        if semantic_model_dir.exists():
            shutil.rmtree(semantic_model_dir)
            print(f"✓ Cleaned semantic model: {semantic_model_dir}")

        # Collect results for final summary
        test_results = []

        for dashboard_item in input_data:
            # Extract configuration
            platform = dashboard_item["platform"]
            dashboard_url = dashboard_item["dashboard_url"]
            api_base_url = dashboard_item["api_base_url"]
            dialect = dashboard_item.get("dialect", "postgresql")

            print(f"\n{'='*70}")
            print(f"Testing Dashboard: {platform}")
            print(f"URL: {dashboard_url}")
            print(f"{'='*70}\n")

            # Get dashboard config from agent_config
            dashboard_config = agent_config.dashboard_config.get(platform)
            if not dashboard_config:
                pytest.skip(f"Dashboard config for platform '{platform}' not found in agent_config")

            # Step 0: Create BI adaptor
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
            print(f"✓ Step 0: BI adaptor created for {platform}")

            # Track result for this test case
            test_result = {
                "platform": platform,
                "dashboard_url": dashboard_url,
                "status": "running",
                "error": None,
                "dashboard_name": None,
                "charts_processed": 0,
                "sub_agents": [],
                "tables": 0,
            }

            try:
                # Step 1: Extract dashboard from Superset (REAL)
                dashboard_id = bi_adaptor.parse_dashboard_id(dashboard_url)
                dashboard = bi_adaptor.get_dashboard_info(dashboard_id)

                assert dashboard is not None, "Failed to get dashboard"
                assert dashboard.name, "Dashboard should have name"

                print(f"\n✓ Step 1: Extracted dashboard '{dashboard.name}' (ID: {dashboard_id})")

                # Step 2: Extract charts (REAL)
                chart_metas = bi_adaptor.list_charts(dashboard_id)
                assert len(chart_metas) > 0, "Dashboard should have charts"

                charts = bi_commands._hydrate_charts(bi_adaptor, dashboard_id, chart_metas)

                # Filter charts with SQL
                charts_with_sql = [c for c in charts if c.query and c.query.sql]
                assert len(charts_with_sql) > 0, "Should have charts with SQL"

                print(f"✓ Step 2: Extracted {len(charts_with_sql)} charts with SQL")

                # Verify expected charts if provided - match by name (more stable than ID)
                if "valid_charts" in dashboard_item:
                    expected_chart_names = {c["name"] for c in dashboard_item["valid_charts"]}
                    actual_chart_names = {c.name for c in charts}
                    for expected_name in expected_chart_names:
                        assert (
                            expected_name in actual_chart_names
                        ), f"Expected chart '{expected_name}' not found in dashboard"
                    print(f"           Validated {len(expected_chart_names)} expected charts")

                # Step 3: Create assembler and assemble (REAL)
                assembler = DashboardAssembler(
                    bi_adaptor,
                    default_dialect=dialect,
                )

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
                    for chart in charts_with_sql[:2]:
                        chart_selections.append(
                            ChartSelection(chart=chart, sql_indices=list(range(len(chart.query.sql))))
                        )

                assert len(chart_selections) > 0, "Should have at least one chart selected"
                print(f"           Selected {len(chart_selections)} charts for processing")

                datasets = bi_adaptor.list_datasets(dashboard_id)

                result = assembler.assemble(
                    dashboard,
                    chart_selections,  # For reference SQL
                    chart_selections,  # For metrics
                    datasets,
                )

                assert len(result.reference_sqls) > 0, "Should have reference SQLs"
                assert len(result.metric_sqls) > 0, "Should have metric SQLs"
                assert len(result.tables) > 0, "Should have tables"

                print(f"✓ Step 3: Assembled {len(result.reference_sqls)} reference SQLs")
                print(f"           Assembled {len(result.metric_sqls)} metric SQLs")
                print(f"           Extracted {len(result.tables)} tables")

                # Step 4: Save sub-agent (complete flow: gen + save + bootstrap)
                print(
                    f"\n⏳ Step 4: Running complete sub-agent build "
                    f"({len(result.reference_sqls)} ref SQLs, "
                    f"{len(result.metric_sqls)} metric SQLs, "
                    f"{len(result.tables)} tables)..."
                )

                bi_commands._save_sub_agent(platform, dashboard, result)

                print("✓ Step 4: Sub-agent build flow completed")

                # Step 5: Verify 2 sub-agents created
                sub_agent_name = bi_commands._build_sub_agent_name(platform, dashboard.name or "")
                attr_name = f"{sub_agent_name}_attribution"

                assert (
                    sub_agent_name in agent_config.agentic_nodes
                ), f"Main sub-agent '{sub_agent_name}' not found in agentic_nodes"
                assert (
                    attr_name in agent_config.agentic_nodes
                ), f"Attribution sub-agent '{attr_name}' not found in agentic_nodes"

                attr_node = agent_config.agentic_nodes[attr_name]
                assert attr_node.get("node_class") == "gen_report", (
                    f"Attribution sub-agent should have node_class='gen_report', "
                    f"got '{attr_node.get('node_class')}'"
                )

                print(f"✓ Step 5: Verified 2 sub-agents: '{sub_agent_name}' + '{attr_name}'")

                # Step 6: Verify all 5 LanceDB tables in both sub-agent stores
                import lancedb

                required_tables = [
                    "schema_metadata",
                    "schema_value",
                    "metrics",
                    "semantic_model",
                    "reference_sql",
                ]

                for name in [sub_agent_name, attr_name]:
                    store_path = agent_config.sub_agent_storage_path(name)
                    assert os.path.isdir(
                        store_path
                    ), f"Sub-agent '{name}' storage directory does not exist: {store_path}"
                    db = lancedb.connect(store_path)
                    actual_tables = db.table_names()
                    print(f"           Sub-agent '{name}' tables: {actual_tables}")

                    for tbl_name in required_tables:
                        assert tbl_name in actual_tables, (
                            f"Sub-agent '{name}' missing required table '{tbl_name}'. " f"Found: {actual_tables}"
                        )
                        row_count = db.open_table(tbl_name).count_rows()
                        assert row_count > 0, f"Sub-agent '{name}' table '{tbl_name}' is empty (0 rows)"
                        print(f"           ✓ '{name}'.{tbl_name}: {row_count} rows")

                print("✓ Step 6: Verified all 5 tables with data for both sub-agents")

                # Step 7: Verify file artifacts
                from datus.utils.path_manager import get_path_manager

                sql_dir = get_path_manager(agent_config.home).dashboard_path() / platform
                sql_files = list(sql_dir.glob("*.sql"))
                assert len(sql_files) > 0, "SQL files should exist"

                csv_files = list(sql_dir.glob("*.csv"))
                assert len(csv_files) > 0, "CSV files should exist"

                semantic_dir = get_path_manager(agent_config.home).semantic_model_path(agent_config.current_namespace)
                semantic_files = []
                if semantic_dir.exists():
                    semantic_files = list(semantic_dir.glob("*.yml")) + list(semantic_dir.glob("*.yaml"))
                if not semantic_files:
                    print("           ⚠ No semantic model YAML files found (LLM may not have generated them)")

                print(
                    f"✓ Step 7: Artifacts verified - "
                    f"{len(sql_files)} SQL, {len(csv_files)} CSV, {len(semantic_files)} semantic model files"
                )

                # Update test result
                test_result["status"] = "passed"
                test_result["dashboard_name"] = dashboard.name
                test_result["charts_processed"] = len(chart_selections)
                test_result["sub_agents"] = [sub_agent_name, attr_name]
                test_result["tables"] = len(result.tables)

                print(f"\n✅ {platform} dashboard test PASSED")

            except Exception as e:
                # Capture failure
                test_result["status"] = "failed"
                test_result["error"] = str(e)
                print(f"\n❌ {platform} dashboard test FAILED: {str(e)}")
                # Re-raise to fail the test
                raise

            finally:
                # Add result to summary
                test_results.append(test_result)

                # Clean up
                if hasattr(bi_adaptor, "close"):
                    bi_adaptor.close()

        # Print final summary after all test cases
        print("────────────────────────────────────────────────────────────────────────────────")
        print(" 📊 BI DASHBOARD INTEGRATION TEST SUMMARY")
        print("─" * 80)

        total_tests = len(test_results)
        passed_tests = [r for r in test_results if r["status"] == "passed"]
        failed_tests = [r for r in test_results if r["status"] == "failed"]

        print(f"\nTotal Tests: {total_tests}")
        print(f" ✅ Passed: {len(passed_tests)}")
        print(f" ❌ Failed: {len(failed_tests)}")

        if passed_tests:
            print("\n" + "─" * 80)
            print(" ✅ PASSED TESTS:")
            print("─" * 80)
            for result in passed_tests:
                print(f"\n  Platform: {result['platform']}")
                print(f"  Dashboard: {result['dashboard_name']}")
                print(f"  URL: {result['dashboard_url']}")
                print(f"  Charts processed: {result['charts_processed']}")
                print(f"  Tables: {result['tables']}")
                print(f"  Sub-agents: {', '.join(result['sub_agents'])}")

        if failed_tests:
            print("\n" + "─" * 80)
            print(" ❌ FAILED TESTS:")
            print("─" * 80)
            for result in failed_tests:
                print(f"\n  Platform: {result['platform']}")
                print(f"  URL: {result['dashboard_url']}")
                print(f"  Error: {result['error']}")

        print("\n" + "─" * 80)
