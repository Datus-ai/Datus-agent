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


# ============================================================================
# Fixtures
# ============================================================================


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
        3. Generate reference SQL (REAL LLM CALL)
        4. Generate semantic model (REAL LLM CALL)
        5. Generate metrics (REAL LLM CALL)
        6. Create sub-agent files (REAL FILE SYSTEM)
        7. Verify all artifacts created

        Cost: ~$0.05-0.20 per run
        Time: ~2-5 minutes
        """
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
                "reference_sqls": 0,
                "metrics": 0,
                "tables": 0,
                "sql_files": 0,
                "csv_files": 0,
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

                # Step 4: Generate reference SQL with REAL LLM
                print("\n⏳ Step 4: Generating reference SQL (REAL LLM, may take 30-60s)...")
                print(f"           Input: {len(result.reference_sqls)} reference SQL candidates")

                ref_sqls = bi_commands._gen_reference_sqls(result.reference_sqls, platform, dashboard)

                print(f"           Output: {len(ref_sqls)} reference SQL entries")
                if len(ref_sqls) == 0:
                    print("           ⚠️  No reference SQL entries generated!")
                    print("           Possible reasons:")
                    print("           1. LLM call failed")
                    print("           2. All SQLs already exist (incremental mode)")
                    print("           3. Check console output above for errors")

                assert len(ref_sqls) > 0, "Should generate reference SQL entries"
                print(f"✓ Step 4: Generated {len(ref_sqls)} reference SQL entries")

                # Verify SQL file was created
                from datus.utils.path_manager import get_path_manager

                sql_dir = get_path_manager(agent_config.home).dashboard_path() / platform
                sql_files = list(sql_dir.glob("*.sql"))
                assert len(sql_files) > 0, "Should create SQL files"
                print(f"           Created SQL file: {sql_files[0].name}")

                # Step 5: Generate semantic model with REAL LLM
                print("\n⏳ Step 5: Generating semantic model (REAL LLM, may take 30-60s)...")

                semantic_result = bi_commands._gen_semantic_model(result.metric_sqls, platform, dashboard)

                assert semantic_result is True, "Semantic model generation should succeed"
                print("✓ Step 5: Generated semantic model")

                # Step 6: Generate metrics with REAL LLM
                print("\n⏳ Step 6: Generating metrics (REAL LLM, may take 30-60s)...")

                metrics = bi_commands._gen_metrics(result.metric_sqls, platform, dashboard)

                assert metrics is not None, "Should generate metrics"
                if metrics and len(metrics) > 0:
                    print(f"✓ Step 6: Generated {len(metrics)} metrics")
                    for metric in metrics[:3]:  # Show first 3
                        print(f"           - {metric}")
                else:
                    print("✓ Step 6: Metrics generation completed (0 metrics generated)")

                # Step 7: Verify artifacts
                print("\n✓ Step 7: Verifying all artifacts created")

                # Check SQL files
                assert len(sql_files) > 0, "SQL files should exist"
                sql_content = sql_files[0].read_text()
                assert len(sql_content) > 0, "SQL file should have content"
                assert "SELECT" in sql_content.upper(), "SQL file should contain SQL"
                print(f"           - SQL file: {sql_files[0].name} ({len(sql_content)} bytes)")

                # Check CSV files (metrics input)
                csv_files = list(sql_dir.glob("*.csv"))
                if len(csv_files) > 0:
                    csv_content = csv_files[0].read_text()
                    assert "question,sql" in csv_content, "CSV should have header"
                    print(f"           - CSV file: {csv_files[0].name} ({len(csv_content)} bytes)")

                # Check semantic model files
                semantic_dir = get_path_manager(agent_config.home).semantic_model_path(agent_config.current_namespace)
                if semantic_dir.exists():
                    semantic_files = list(semantic_dir.glob("*.yml"))
                    if len(semantic_files) > 0:
                        print(f"           - Semantic files: {len(semantic_files)} files")

                # Update test result with success
                test_result["status"] = "passed"
                test_result["dashboard_name"] = dashboard.name
                test_result["charts_processed"] = len(chart_selections)
                test_result["reference_sqls"] = len(ref_sqls)
                test_result["metrics"] = len(metrics) if metrics else 0
                test_result["tables"] = len(result.tables)
                test_result["sql_files"] = len(sql_files)
                test_result["csv_files"] = len(csv_files)

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
        print("-" * 80)

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
                print(f"  Reference SQLs: {result['reference_sqls']}")
                print(f"  Metrics: {result['metrics']}")
                print(f"  Tables: {result['tables']}")
                print(f"  Artifacts: {result['sql_files']} SQL + {result['csv_files']} CSV files")

        if failed_tests:
            print("\n" + "─" * 80)
            print(" ❌ FAILED TESTS:")
            print("─" * 80)
            for result in failed_tests:
                print(f"\n  Platform: {result['platform']}")
                print(f"  URL: {result['dashboard_url']}")
                print(f"  Error: {result['error']}")

        print("\n" + "-" * 80)
