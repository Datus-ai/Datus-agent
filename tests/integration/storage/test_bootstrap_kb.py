import pytest

from datus.configuration.agent_config import AgentConfig
from datus.schemas.agent_models import ScopedContext, SubAgentConfig
from datus.storage.metric.store import MetricRAG
from datus.storage.reference_sql import ReferenceSqlRAG
from datus.storage.schema_metadata import SchemaWithValueRAG
from datus.storage.sub_agent_kb_bootstrap import SUPPORTED_COMPONENTS, SubAgentBootstrapper


def _make_sub_agent_config(
    tables="california_schools.*",
    metrics="california_schools",
    sqls="california_schools",
) -> SubAgentConfig:
    scoped_context = ScopedContext(tables=tables, metrics=metrics, sqls=sqls)
    return SubAgentConfig(
        system_prompt="nightly_test",
        agent_description="nightly test agent",
        tools="",
        mcp="",
        scoped_context=scoped_context,
    )


@pytest.mark.nightly
class TestBootstrapKB:
    """N1: bootstrap-kb knowledge base initialization tests."""

    def _register_and_bootstrap(self, agent_config, sub_agent_config, strategy="overwrite"):
        agent_config.agentic_nodes[sub_agent_config.system_prompt] = sub_agent_config
        bootstrapper = SubAgentBootstrapper(sub_agent=sub_agent_config, agent_config=agent_config)
        return bootstrapper, bootstrapper.run(strategy=strategy)

    def _cleanup(self, agent_config, sub_agent_config):
        bootstrapper = SubAgentBootstrapper(sub_agent=sub_agent_config, agent_config=agent_config)
        for comp in SUPPORTED_COMPONENTS:
            bootstrapper._clear_component(comp)

    def test_metadata_component(self, agent_config: AgentConfig):
        """N1-01: metadata single component initialization -- verify LanceDB index created."""
        sub_agent_config = _make_sub_agent_config()
        agent_config.agentic_nodes[sub_agent_config.system_prompt] = sub_agent_config
        bootstrapper = SubAgentBootstrapper(sub_agent=sub_agent_config, agent_config=agent_config)

        result = bootstrapper.run(
            selected_components=["metadata"],
            strategy="overwrite",
        )
        assert result is not None, "Bootstrap result should not be None"
        assert result.storage_path, "Result should have a storage path"

        # Verify metadata component specifically
        metadata_results = [r for r in result.results if r.component == "metadata"]
        assert len(metadata_results) == 1, "Should have exactly one metadata component result"
        assert (
            metadata_results[0].status == "success"
        ), f"Metadata should succeed, got: {metadata_results[0].status} - {metadata_results[0].message}"
        assert metadata_results[0].details.get("stored_tables", 0) > 0, "Should have stored table metadata"

        # Verify LanceDB table was created
        table_schema_rag = SchemaWithValueRAG(agent_config, sub_agent_name="nightly_test")
        assert table_schema_rag.schema_store.table_size() > 0, "LanceDB schema table should have entries"

        # Cleanup
        self._cleanup(agent_config, sub_agent_config)

    def test_metrics_success_story(self, agent_config: AgentConfig):
        """N1-02: metrics initialization from CSV success_story data."""
        sub_agent_config = _make_sub_agent_config(metrics="california_schools")
        _, result = self._register_and_bootstrap(agent_config, sub_agent_config, strategy="overwrite")

        assert result is not None, "Bootstrap result should not be None"
        metrics_results = [r for r in result.results if r.component == "metrics"]
        assert len(metrics_results) >= 1, "Should have metrics component result"
        assert (
            metrics_results[0].status == "success"
        ), f"Metrics should succeed, got: {metrics_results[0].status} - {metrics_results[0].message}"
        stored_metrics = metrics_results[0].details.get("stored_metrics", 0)
        assert stored_metrics > 0, "Should have stored metrics from success_story CSV"

        # Verify through RAG
        metrics_rag = MetricRAG(agent_config, sub_agent_name="nightly_test")
        assert metrics_rag.storage.table_size() > 0, "Metrics RAG should have entries"

        # Cleanup
        self._cleanup(agent_config, sub_agent_config)

    def test_reference_sql(self, agent_config: AgentConfig):
        """N1-05: reference_sql initialization -- SQL file indexing."""
        sub_agent_config = _make_sub_agent_config(sqls="california_schools")
        _, result = self._register_and_bootstrap(agent_config, sub_agent_config, strategy="overwrite")

        assert result is not None, "Bootstrap result should not be None"
        sql_results = [r for r in result.results if r.component == "reference_sql"]
        assert len(sql_results) >= 1, "Should have reference_sql component result"
        assert (
            sql_results[0].status == "success"
        ), f"Reference SQL should succeed, got: {sql_results[0].status} - {sql_results[0].message}"
        stored_sqls = sql_results[0].details.get("stored_sqls", 0)
        assert stored_sqls > 0, "Should have stored reference SQL entries"

        sql_rag = ReferenceSqlRAG(agent_config, sub_agent_name="nightly_test")
        assert sql_rag.reference_sql_storage.table_size() > 0, "Reference SQL RAG should have entries"

        # Cleanup
        self._cleanup(agent_config, sub_agent_config)

    def test_multi_component_plan(self, agent_config: AgentConfig):
        """N1-07: multi-component joint initialization with plan strategy."""
        sub_agent_config = _make_sub_agent_config()
        agent_config.agentic_nodes[sub_agent_config.system_prompt] = sub_agent_config
        bootstrapper = SubAgentBootstrapper(sub_agent=sub_agent_config, agent_config=agent_config)

        result = bootstrapper.run(strategy="plan")
        assert result is not None, "Plan result should not be None"
        assert result.should_bootstrap is True, "Should indicate bootstrap is needed"
        assert len(result.results) == len(
            SUPPORTED_COMPONENTS
        ), f"Should have {len(SUPPORTED_COMPONENTS)} components, got {len(result.results)}"

        # Each component should report plan status
        for comp_result in result.results:
            assert comp_result.status in (
                "plan",
                "skipped",
                "error",
            ), f"Component {comp_result.component} should have plan/skipped/error status, got {comp_result.status}"
            assert comp_result.details is not None, f"Component {comp_result.component} should have details"

        # Metadata should have matches with wildcard pattern
        metadata_results = [r for r in result.results if r.component == "metadata"]
        assert len(metadata_results) == 1, "Should have metadata result"
        assert (
            metadata_results[0].details.get("match_count", 0) >= 3
        ), "Metadata wildcard should match at least 3 tables"

    def test_incremental_update(self, agent_config: AgentConfig):
        """N1-08: incremental update preserves existing data."""
        sub_agent_config = _make_sub_agent_config()

        # First: initial overwrite
        _, result1 = self._register_and_bootstrap(agent_config, sub_agent_config, strategy="overwrite")
        assert result1 is not None, "First bootstrap result should not be None"

        metadata_results1 = [r for r in result1.results if r.component == "metadata"]
        initial_count = metadata_results1[0].details.get("stored_tables", 0) if metadata_results1 else 0

        # Second: incremental update should preserve existing data
        _, result2 = self._register_and_bootstrap(agent_config, sub_agent_config, strategy="incremental")
        assert result2 is not None, "Second bootstrap result should not be None"

        metadata_results2 = [r for r in result2.results if r.component == "metadata"]
        updated_count = metadata_results2[0].details.get("stored_tables", 0) if metadata_results2 else 0

        assert initial_count > 0, "Initial bootstrap should store data"
        assert (
            updated_count >= initial_count
        ), f"Re-bootstrap should not reduce data count: {updated_count} < {initial_count}"

        # Cleanup
        self._cleanup(agent_config, sub_agent_config)

    def test_overwrite_update(self, agent_config: AgentConfig):
        """N1-09: overwrite update replaces all data cleanly."""
        sub_agent_config = _make_sub_agent_config()
        _, result = self._register_and_bootstrap(agent_config, sub_agent_config, strategy="overwrite")

        assert result is not None, "Bootstrap result should not be None"
        assert result.storage_path.endswith(
            "nightly_test"
        ), f"Storage path should end with 'nightly_test', got: {result.storage_path}"

        # Verify components that have data match stored counts
        metadata_rag = SchemaWithValueRAG(agent_config, sub_agent_name="nightly_test")
        # Verify each component that succeeded
        metadata_results = [r for r in result.results if r.component == "metadata"]
        assert len(metadata_results) == 1, "Should have metadata result"
        if metadata_results[0].status == "success":
            expected = metadata_results[0].details.get("stored_tables", 0)
            actual = metadata_rag.schema_store.table_size()
            assert actual == expected, f"Schema store size {actual} should match stored_tables {expected}"

        # All components should succeed with correct scoped context
        assert (
            metadata_results[0].status == "success"
        ), f"Metadata should succeed in overwrite, got: {metadata_results[0].status}"

        metrics_rag = MetricRAG(agent_config, sub_agent_name="nightly_test")
        metrics_results = [r for r in result.results if r.component == "metrics"]
        assert len(metrics_results) == 1, "Should have metrics result"
        assert (
            metrics_results[0].status == "success"
        ), f"Metrics should succeed, got: {metrics_results[0].status} - {metrics_results[0].message}"
        expected = metrics_results[0].details.get("stored_metrics", 0)
        actual = metrics_rag.storage.table_size()
        assert actual == expected, f"Metrics store size {actual} should match stored_metrics {expected}"

        sql_rag = ReferenceSqlRAG(agent_config, sub_agent_name="nightly_test")
        sql_results = [r for r in result.results if r.component == "reference_sql"]
        assert len(sql_results) == 1, "Should have reference_sql result"
        assert (
            sql_results[0].status == "success"
        ), f"Reference SQL should succeed, got: {sql_results[0].status} - {sql_results[0].message}"
        expected = sql_results[0].details.get("stored_sqls", 0)
        actual = sql_rag.reference_sql_storage.table_size()
        assert actual == expected, f"SQL store size {actual} should match stored_sqls {expected}"

        # Cleanup
        self._cleanup(agent_config, sub_agent_config)

    def test_wildcard_scoped_context(self, agent_config: AgentConfig):
        """N1-07b: wildcard patterns in scoped context."""
        sub_agent_config = _make_sub_agent_config(
            tables="california_schools.*",
            metrics="california_schools.*",
            sqls="california_schools.*",
        )
        agent_config.agentic_nodes[sub_agent_config.system_prompt] = sub_agent_config
        bootstrapper = SubAgentBootstrapper(sub_agent=sub_agent_config, agent_config=agent_config)

        result = bootstrapper.run(strategy="plan")
        assert result is not None, "Plan result should not be None"
        assert len(result.results) == len(
            SUPPORTED_COMPONENTS
        ), f"Should have {len(SUPPORTED_COMPONENTS)} component results"

        # Wildcard metadata should match multiple tables
        metadata_result = [r for r in result.results if r.component == "metadata"][0]
        assert (
            metadata_result.details.get("match_count", 0) >= 3
        ), f"Wildcard metadata should match >= 3 tables, got {metadata_result.details.get('match_count', 0)}"

        # Reference SQL component should have plan details with matches
        sql_result = [r for r in result.results if r.component == "reference_sql"][0]
        assert sql_result.details is not None, "Reference SQL plan should have details"
        assert sql_result.status == "plan", f"Reference SQL should have plan status, got {sql_result.status}"
        assert (
            sql_result.details.get("match_count", 0) > 0
        ), f"Reference SQL wildcard should match entries, got {sql_result.details.get('match_count', 0)}"
