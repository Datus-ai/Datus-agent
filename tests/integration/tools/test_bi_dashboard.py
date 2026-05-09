# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""BI Dashboard integration tests.

Contains two test classes at different verification levels:
- TestPartialIntegration: Real Superset API + mocked LLM (nightly)
- TestE2EIntegration: Full end-to-end with zero mocks (nightly only)
"""

import os
import re
from typing import Any, Dict, List, Tuple

import pytest
import yaml
from datus_bi_core import AuthParam, ChartSpec, DashboardSpec, DatasetInfo, DatasetSpec
from rich.console import Console

from datus.cli.bootstrap_bi_commands import BootstrapBiCommands
from datus.cli.bootstrap_bi_picker import BootstrapBiPicker, BootstrapBiPlan, DashboardCliOptions
from datus.cli.bootstrap_bi_subagents import build_sub_agent_name
from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.tools.bi_tools.dashboard_assembler import ChartSelection, DashboardAssembler
from datus.utils.loggings import configure_logging, get_logger
from tests.conftest import TEST_CONF_DIR, TEST_DATA_DIR

configure_logging(False, console_output=False)
logger = get_logger(__name__)


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
    """Load agent config from a temp copy, with `home:` redirected to tmp.

    The redirect is critical: without it, `agent_config.rag_storage_path()` and
    `agent_config.path_manager.semantic_model_path()` resolve to the developer's
    real RAG storage (e.g. `.datus_test_data/...`), and the E2E test's cleanup
    block would rmtree that real storage on every run.
    """
    src = TEST_CONF_DIR / "agent.yml"
    tmp_dir = tmp_path_factory.mktemp("bi_conf")
    tmp_cfg = tmp_dir / "agent.yml"
    tmp_home = tmp_path_factory.mktemp("bi_home")
    tmp_project = tmp_path_factory.mktemp("bi_project")

    # Rewrite `home:` AND `project_root:` to point at isolated tmp dirs so
    # every derived path (rag_storage_path resolves from home;
    # semantic_model_path / dashboard_path resolve from project_root) is
    # tmp-scoped. Patching only `home:` leaves project_root pointing at the
    # developer's real `.datus_test_data/workspace/` — E2E cleanup could then
    # rmtree real semantic-model / dashboard storage.
    # Use re.subn and assert a replacement happened — if agent.yml changes
    # shape and the regex misses, the fixture fails loudly instead of silently
    # writing to the real filesystem.
    content = src.read_text()
    # Match with any leading indent — agent.yml nests both keys under `agent:`
    # (two-space indent), so an `^home:` anchor would never match.
    # Preserve the captured indent in each replacement so YAML stays valid.
    content, home_repl = re.subn(
        r"^(\s*)home:\s*\S+",
        lambda m: f"{m.group(1)}home: {tmp_home}",
        content,
        count=1,
        flags=re.MULTILINE,
    )
    assert home_repl == 1, (
        "agent.yml must contain an `agent.home` entry for tmp isolation "
        "(got 0 substitutions — the fixture cannot guarantee safe cleanup)"
    )
    content, project_repl = re.subn(
        r"^(\s*)project_root:\s*\S+",
        lambda m: f"{m.group(1)}project_root: {tmp_project}",
        content,
        count=1,
        flags=re.MULTILINE,
    )
    assert project_repl == 1, (
        "agent.yml must contain an `agent.project_root` entry for tmp "
        "isolation (got 0 substitutions — semantic_model / dashboard paths "
        "would resolve to the real workspace)"
    )
    tmp_cfg.write_text(content)

    config = load_agent_config(config=str(tmp_cfg), datasource="superset", reload=True, force=True, yes=True)

    # Keep this suite hermetic to the Superset compose environment.  The
    # checked-in test config historically pointed at 15432, which collides
    # with the Greenplum adapter service in the nightly runner.
    superset_db = config.services.datasources.get("superset")
    if superset_db is None:
        raise AssertionError("agent.yml must define services.datasources.superset")
    superset_db.host = os.environ.get("SUPERSET_POSTGRES_HOST", superset_db.host)
    superset_db.port = os.environ.get("SUPERSET_POSTGRES_PORT", "5433")
    superset_db.username = "superset"
    superset_db.password = "superset"
    superset_db.database = "superset_examples"
    superset_db.schema = "public"

    return config


@pytest.fixture
def bi_picker(agent_config) -> BootstrapBiPicker:
    """Create a BootstrapBiPicker for tests (used for adapter / chart hydration)."""
    console = Console(log_path=False, force_terminal=False)
    return BootstrapBiPicker(agent_config, console)


@pytest.fixture
def bi_commands(agent_config) -> BootstrapBiCommands:
    """Create BootstrapBiCommands for tests (used to drive _run_plan)."""
    console = Console(log_path=False, force_terminal=False)
    return BootstrapBiCommands(agent_config, console)


@pytest.fixture(scope="module")
def input_data() -> List[Dict[str, Any]]:
    """Load test data from YAML file."""
    yaml_path = TEST_DATA_DIR / "BIDashboardInput.yaml"
    with open(yaml_path, "r") as f:
        superset_url = os.environ.get("SUPERSET_URL")
        data = yaml.safe_load(f)
        if isinstance(data, list):
            items = [item["input"] for item in data]
        elif isinstance(data, dict) and "input" in data:
            items = [data]
        else:
            pytest.fail(reason=f"Unexpected data type: {type(data)}")
            return []

        if superset_url:
            for item in items:
                if item.get("platform") == "superset":
                    item["api_base_url"] = superset_url
                    if item.get("dashboard_url"):
                        item["dashboard_url"] = (
                            f"{superset_url.rstrip('/')}/superset/dashboard/datus-nightly-placeholder/"
                        )

        return items


# ============================================================================
# Shared extraction logic
# ============================================================================


def _create_adapter(bi_picker, agent_config, dashboard_item):
    """Create a BI adapter from dashboard_item config."""
    platform = dashboard_item["platform"]
    dashboard_config = agent_config.dashboard_config.get(platform)
    if not dashboard_config:
        pytest.skip(f"Dashboard config for platform '{platform}' not found")

    return bi_picker._create_adapter(
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


def _items_from_adapter_result(result: Any) -> List[Any]:
    """Return list items from either a PaginatedResult envelope or a plain sequence."""
    if result is None:
        return []
    if hasattr(result, "items"):
        return list(result.items or [])
    return list(result)


def _find_superset_database_id(bi_adapter, db_name: str) -> int | None:
    for db in bi_adapter.list_bi_databases():
        if db.get("name") == db_name:
            db_id = db.get("id")
            return int(db_id) if db_id is not None else None
    return None


def _ensure_superset_test_database(bi_adapter) -> int:
    db_name = "datus_test_examples"
    existing_id = _find_superset_database_id(bi_adapter, db_name)
    if existing_id is not None:
        return existing_id

    created = bi_adapter.register_database(
        db_name,
        "postgresql+psycopg2://superset:superset@postgres:5432/superset_examples",
    )
    created_id = created.get("id")
    if created_id is not None:
        return int(created_id)

    # Superset 4.x can return a creation success payload without echoing the
    # id. Re-query to keep first-run behavior deterministic instead of relying
    # on pytest-rerun.
    existing_id = _find_superset_database_id(bi_adapter, db_name)
    if existing_id is None:
        raise AssertionError(f"Failed to resolve Superset database id for '{db_name}' after creation")
    return existing_id


def _get_or_create_sales_dataset(bi_adapter, db_id: int) -> DatasetInfo:
    table_name = "datus_nightly_bi_sales"
    try:
        data = bi_adapter._request_json("GET", "dataset/")
        for item in data.get("result", []):
            if item.get("table_name") == table_name and item.get("schema") == "public":
                return DatasetInfo(
                    id=item["id"],
                    name=table_name,
                    dialect="postgresql",
                )
    except Exception:
        pass

    return bi_adapter.create_dataset(
        DatasetSpec(
            name=table_name,
            database_id=db_id,
            db_schema="public",
        )
    )


def _ensure_seed_dashboard(bi_adapter, dashboard_item) -> None:
    """Create a deterministic dashboard when the fixture asks for one.

    This keeps the Superset nightly hermetic: no dependency on
    ``superset load-examples`` or live downloads from the examples repository.
    """
    if not dashboard_item.get("seed_dashboard") or dashboard_item.get("_seeded_dashboard_url"):
        if dashboard_item.get("_seeded_dashboard_url"):
            dashboard_item["dashboard_url"] = dashboard_item["_seeded_dashboard_url"]
        return

    db_id = _ensure_superset_test_database(bi_adapter)
    dataset = _get_or_create_sales_dataset(bi_adapter, db_id)
    dashboard = bi_adapter.create_dashboard(DashboardSpec(title="Datus Nightly BI Dashboard"))

    chart_specs = [
        ChartSpec(
            chart_type="table",
            title="Datus Revenue by Region",
            dataset_id=dataset.id,
            metrics=["SUM(revenue)"],
            dimensions=["region"],
        ),
        ChartSpec(
            chart_type="table",
            title="Datus Orders by Channel",
            dataset_id=dataset.id,
            metrics=["SUM(orders)"],
            dimensions=["channel"],
        ),
    ]
    for spec in chart_specs:
        chart = bi_adapter.create_chart(spec)
        bi_adapter.add_chart_to_dashboard(dashboard.id, chart.id)

    dashboard_url = f"{dashboard_item['api_base_url'].rstrip('/')}/superset/dashboard/{dashboard.id}/"
    dashboard_item["dashboard_url"] = dashboard_url
    dashboard_item["_seeded_dashboard_url"] = dashboard_url


def _extract_and_select_charts(
    bi_picker,
    bi_adapter,
    dashboard_item,
) -> Tuple[Any, List[ChartSelection], List[Any], List[Any]]:
    """Extract dashboard, select charts with SQL validation, and assemble.

    Returns:
        (dashboard, chart_selections, charts, datasets)
    """
    _ensure_seed_dashboard(bi_adapter, dashboard_item)
    dashboard_url = dashboard_item["dashboard_url"]

    dashboard_id = bi_adapter.parse_dashboard_id(dashboard_url)
    dashboard = bi_adapter.get_dashboard_info(dashboard_id)
    if dashboard is None:
        raise AssertionError("Failed to get dashboard")
    if not isinstance(dashboard.name, str) or dashboard.name.strip() == "":
        raise AssertionError("Dashboard should have a non-empty name")

    chart_metas = _items_from_adapter_result(bi_adapter.list_charts(dashboard_id))
    assert len(chart_metas) >= 1, "Dashboard should have charts"

    charts = bi_picker._hydrate_charts(bi_adapter, dashboard_id, chart_metas)
    charts_with_sql = [c for c in charts if c.query and c.query.sql]
    assert len(charts_with_sql) >= 1, "Should have charts with SQL"

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
                        pytest.fail(f"SQL validation failed for chart '{chart.name}':\n{error_msg}")

                chart_selections.append(ChartSelection(chart=chart, sql_indices=list(range(len(chart.query.sql)))))
    else:
        chart_selections = [
            ChartSelection(chart=c, sql_indices=list(range(len(c.query.sql)))) for c in charts_with_sql[:2]
        ]

    assert len(chart_selections) >= 1, "Should have at least one chart selected"

    datasets = _items_from_adapter_result(bi_adapter.list_datasets(dashboard_id))

    return dashboard, chart_selections, charts, datasets


def _assemble(bi_adapter, dashboard, chart_selections, datasets, dialect):
    """Run the DashboardAssembler and verify results."""
    assembler = DashboardAssembler(bi_adapter, default_dialect=dialect)
    result = assembler.assemble(dashboard, chart_selections, chart_selections, datasets)

    assert len(result.reference_sqls) >= 1, "Should have reference SQLs"
    assert len(result.metric_sqls) >= 1, "Should have metric SQLs"
    assert len(result.tables) >= 1, "Should have tables"

    return result


# ============================================================================
# Partial Integration Tests (Strategic Mocks)
# ============================================================================


class TestPartialIntegration:
    """Partial integration tests with STRATEGIC mocking.

    Real: Superset API calls, data processing, workflow orchestration.
    Mocked: LLM API calls (too expensive/slow).
    """

    @pytest.mark.nightly
    def test_workflow_without_llm(
        self,
        bi_picker: BootstrapBiPicker,
        bi_commands: BootstrapBiCommands,
        agent_config: AgentConfig,
        input_data: List[Dict[str, Any]],
        tmp_path,
    ):
        """Integration test: real BI extraction + mocked LLM generation.

        We swap the real ``stream_bi_*`` async generators for stubs that
        populate the shared ``BiBuildState`` directly — this exercises
        :meth:`BootstrapBiCommands._run_plan` orchestration (semantic-ok
        gating, ScopedContext assembly, sub-agent persistence) without
        burning LLM tokens.
        """
        import asyncio
        from unittest.mock import patch

        for dashboard_item in input_data:
            platform = dashboard_item["platform"]
            dialect = dashboard_item.get("dialect", "postgresql")
            bi_adapter = _create_adapter(bi_picker, agent_config, dashboard_item)

            try:
                dashboard, chart_selections, charts, datasets = _extract_and_select_charts(
                    bi_picker, bi_adapter, dashboard_item
                )
                result = _assemble(bi_adapter, dashboard, chart_selections, datasets, dialect)

                # Pre-built plan replaces what BootstrapBiPicker.run() would have produced.
                plan = BootstrapBiPlan(
                    options=DashboardCliOptions(
                        platform=platform,
                        dashboard_url=dashboard_item["dashboard_url"],
                        api_base_url=dashboard_item["api_base_url"],
                        auth_params=None,
                        dialect=dialect,
                    ),
                    adapter=bi_adapter,
                    dashboard=dashboard,
                    dashboard_id=bi_adapter.parse_dashboard_id(dashboard_item["dashboard_url"]),
                    chart_selections_ref=chart_selections,
                    chart_selections_metrics=chart_selections,
                    assembled=result,
                    pool_size=2,
                )

                async def _meta_stub(_cfg, *, table_names, pool_size, _state=None):
                    if False:
                        yield  # pragma: no cover

                async def _ref_stub(_cfg, *, reference_sqls, platform, dashboard_name, pool_size, state):
                    state.ref_sqls.extend(f"{platform}/test/metric.metric_{i}" for i in range(len(reference_sqls)))
                    if False:
                        yield  # pragma: no cover

                async def _sem_stub(_cfg, *, sqls, platform, dashboard_name, state):
                    state.semantic_ok = True
                    if False:
                        yield  # pragma: no cover

                async def _metrics_stub(_cfg, *, sqls, platform, dashboard_name, state):
                    state.metrics.extend(f"{platform}/test/layer{i}.metric_{i}" for i in range(len(sqls)))
                    if False:
                        yield  # pragma: no cover

                async def _save_stub(_cfg, **_kwargs):
                    if False:
                        yield  # pragma: no cover

                actions: list = []
                with (
                    patch("datus.cli.bootstrap_bi_commands.stream_bi_metadata", side_effect=_meta_stub),
                    patch("datus.cli.bootstrap_bi_commands.stream_bi_reference_sql", side_effect=_ref_stub),
                    patch("datus.cli.bootstrap_bi_commands.stream_bi_semantic_model", side_effect=_sem_stub),
                    patch("datus.cli.bootstrap_bi_commands.stream_bi_metrics", side_effect=_metrics_stub),
                    patch("datus.cli.bootstrap_bi_commands.stream_bi_save_subagents", side_effect=_save_stub),
                    patch("datus.cli.bootstrap_bi_commands.SubAgentManager"),
                    patch("datus.cli.bootstrap_bi_commands.configuration_manager"),
                    patch("datus.cli.bootstrap_bi_commands.qualify_table_names", return_value=list(result.tables)),
                ):
                    asyncio.run(bi_commands._run_plan(plan, actions))

                # Verify the orchestrator gated correctly: metrics ran (semantic_ok was True),
                # so we expect both ref_sqls and metric identifiers in the resulting actions.
                assert any("Sub-Agent build successful" in (a.messages or "") for a in actions), (
                    f"Sub-agent build did not complete; got actions: {[a.messages for a in actions]}"
                )

                logger.info(
                    "Partial integration test passed for %s — real Superset extraction: %d charts, "
                    "real data assembly: %d SQLs, mocked LLM calls: 3",
                    platform,
                    len(charts),
                    len(result.reference_sqls),
                )

            finally:
                if hasattr(bi_adapter, "close"):
                    bi_adapter.close()


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
        bi_picker: BootstrapBiPicker,
        bi_commands: BootstrapBiCommands,
        agent_config: AgentConfig,
        input_data: List[Dict[str, Any]],
    ):
        """TRUE END-TO-END: dashboard extraction -> sub-agent bootstrap -> verification.

        Storage is already isolated in a tmp dir via the `agent_config` fixture
        (home: redirected to tmp_path_factory), so no cleanup is needed here —
        the storage starts empty for every test module run.
        """
        import asyncio

        test_results = []

        for dashboard_item in input_data:
            platform = dashboard_item["platform"]
            dialect = dashboard_item.get("dialect", "postgresql")
            bi_adapter = _create_adapter(bi_picker, agent_config, dashboard_item)

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
                    bi_picker, bi_adapter, dashboard_item
                )

                sub_agent_name = build_sub_agent_name(platform, dashboard.name or "")
                attr_name = f"{sub_agent_name}_attribution"

                result = _assemble(bi_adapter, dashboard, chart_selections, datasets, dialect)

                # Drive the full streaming pipeline end-to-end.
                plan = BootstrapBiPlan(
                    options=DashboardCliOptions(
                        platform=platform,
                        dashboard_url=dashboard_item["dashboard_url"],
                        api_base_url=dashboard_item["api_base_url"],
                        auth_params=None,
                        dialect=dialect,
                    ),
                    adapter=bi_adapter,
                    dashboard=dashboard,
                    dashboard_id=bi_adapter.parse_dashboard_id(dashboard_item["dashboard_url"]),
                    chart_selections_ref=chart_selections,
                    chart_selections_metrics=chart_selections,
                    assembled=result,
                    pool_size=2,
                )
                actions: list = []
                asyncio.run(bi_commands._run_plan(plan, actions))

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
                    logger.info(
                        "Sub-agent '%s': semantic_model=%d, metrics=%d, reference_sql=%d",
                        name,
                        sm_size,
                        m_size,
                        rs_size,
                    )

                test_result["semantic_model_rows"] = total_semantic_model_rows
                test_result["metrics_rows"] = total_metrics_rows
                test_result["reference_sql_rows"] = total_reference_sql_rows

                # Verify file artifacts
                sql_dir = agent_config.path_manager.dashboard_path() / platform
                sql_files = list(sql_dir.glob("*.sql"))
                assert len(sql_files) >= 1, "SQL files should exist"
                csv_files = list(sql_dir.glob("*.csv"))
                assert len(csv_files) >= 1, "CSV files should exist"

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
                if hasattr(bi_adapter, "close"):
                    bi_adapter.close()

        # Structured summary via logger (use pytest --log-cli-level=INFO to surface).
        passed_tests = [r for r in test_results if r["status"] == "passed"]
        failed_tests = [r for r in test_results if r["status"] == "failed"]
        logger.info(
            "BI dashboard integration summary — total=%d passed=%d failed=%d",
            len(test_results),
            len(passed_tests),
            len(failed_tests),
        )
        for result in passed_tests:
            logger.info(
                "PASSED: %s - %s (%d charts, %d tables); bootstrap semantic_model=%d metrics=%d reference_sql=%d",
                result["platform"],
                result["dashboard_name"],
                result["charts_processed"],
                result["tables"],
                result["semantic_model_rows"],
                result["metrics_rows"],
                result["reference_sql_rows"],
            )
        for result in failed_tests:
            logger.error("FAILED: %s - %s", result["platform"], result["error"])

        failed = [r for r in test_results if r["status"] == "failed"]
        assert not failed, f"Dashboard tests failed: {[r['error'] for r in failed]}"
