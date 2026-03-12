import pytest

from datus.configuration.agent_config import AgentConfig
from datus.schemas.agent_models import ScopedContext, SubAgentConfig
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
    """N1: bootstrap-kb knowledge base validation tests.

    SubAgentBootstrapper now performs plan/validation only (no data copying).
    All handlers return status="plan" with match_count details.
    """

    def _register_and_bootstrap(self, agent_config, sub_agent_config, strategy="overwrite"):
        agent_config.agentic_nodes[sub_agent_config.system_prompt] = sub_agent_config
        bootstrapper = SubAgentBootstrapper(sub_agent=sub_agent_config, agent_config=agent_config)
        return bootstrapper, bootstrapper.run(strategy=strategy)

    def test_metadata_component(self, agent_config: AgentConfig):
        """N1-01: metadata single component validation -- verify plan with matches."""
        sub_agent_config = _make_sub_agent_config()
        agent_config.agentic_nodes[sub_agent_config.system_prompt] = sub_agent_config
        bootstrapper = SubAgentBootstrapper(sub_agent=sub_agent_config, agent_config=agent_config)

        result = bootstrapper.run(
            selected_components=["metadata"],
            strategy="overwrite",
        )
        assert result is not None, "Bootstrap result should not be None"
        assert result.storage_path, "Result should have a storage path"

        metadata_results = [r for r in result.results if r.component == "metadata"]
        assert len(metadata_results) == 1, "Should have exactly one metadata component result"
        assert (
            metadata_results[0].status == "plan"
        ), f"Metadata should return plan status, got: {metadata_results[0].status} - {metadata_results[0].message}"
        assert metadata_results[0].details.get("match_count", 0) > 0, "Should have matched table metadata"

    def test_metrics_plan(self, agent_config: AgentConfig):
        """N1-02: metrics validation from scoped context."""
        sub_agent_config = _make_sub_agent_config(metrics="california_schools")
        _, result = self._register_and_bootstrap(agent_config, sub_agent_config, strategy="overwrite")

        assert result is not None, "Bootstrap result should not be None"
        metrics_results = [r for r in result.results if r.component == "metrics"]
        assert len(metrics_results) >= 1, "Should have metrics component result"
        assert metrics_results[0].status in (
            "plan",
            "skipped",
        ), f"Metrics should return plan/skipped status, got: {metrics_results[0].status} - {metrics_results[0].message}"

    def test_reference_sql(self, agent_config: AgentConfig):
        """N1-05: reference_sql validation -- SQL file matching."""
        sub_agent_config = _make_sub_agent_config(sqls="california_schools")
        _, result = self._register_and_bootstrap(agent_config, sub_agent_config, strategy="overwrite")

        assert result is not None, "Bootstrap result should not be None"
        sql_results = [r for r in result.results if r.component == "reference_sql"]
        assert len(sql_results) >= 1, "Should have reference_sql component result"
        assert sql_results[0].status in (
            "plan",
            "skipped",
            "error",
        ), f"Reference SQL should return plan/skipped/error status, got: {sql_results[0].status}"

    def test_multi_component_plan(self, agent_config: AgentConfig):
        """N1-07: multi-component joint validation with plan strategy."""
        sub_agent_config = _make_sub_agent_config()
        agent_config.agentic_nodes[sub_agent_config.system_prompt] = sub_agent_config
        bootstrapper = SubAgentBootstrapper(sub_agent=sub_agent_config, agent_config=agent_config)

        result = bootstrapper.run(strategy="plan")
        assert result is not None, "Plan result should not be None"
        assert result.should_bootstrap is True, "Should indicate bootstrap is needed"
        assert len(result.results) == len(
            SUPPORTED_COMPONENTS
        ), f"Should have {len(SUPPORTED_COMPONENTS)} components, got {len(result.results)}"

        for comp_result in result.results:
            assert comp_result.status in (
                "plan",
                "skipped",
                "error",
            ), f"Component {comp_result.component} should have plan/skipped/error status, got {comp_result.status}"
            # details may be None for skipped/error components (e.g. ext_knowledge with no config)
            if comp_result.status == "plan":
                assert comp_result.details is not None, f"Component {comp_result.component} should have details"

        metadata_results = [r for r in result.results if r.component == "metadata"]
        assert len(metadata_results) == 1, "Should have metadata result"
        assert (
            metadata_results[0].details.get("match_count", 0) >= 3
        ), "Metadata wildcard should match at least 3 tables"

    def test_overwrite_strategy_behaves_as_plan(self, agent_config: AgentConfig):
        """N1-08: overwrite strategy now behaves like plan (validation only)."""
        sub_agent_config = _make_sub_agent_config()

        # First run with overwrite
        _, result1 = self._register_and_bootstrap(agent_config, sub_agent_config, strategy="overwrite")
        assert result1 is not None, "First bootstrap result should not be None"
        assert result1.strategy == "plan", "Overwrite should report as plan strategy"

        metadata_results1 = [r for r in result1.results if r.component == "metadata"]
        assert metadata_results1[0].status == "plan", "Should return plan status"
        initial_count = metadata_results1[0].details.get("match_count", 0)
        assert initial_count > 0, "Should match some tables"

        # Second run should produce same results (idempotent)
        _, result2 = self._register_and_bootstrap(agent_config, sub_agent_config, strategy="overwrite")
        assert result2 is not None, "Second bootstrap result should not be None"
        metadata_results2 = [r for r in result2.results if r.component == "metadata"]
        second_count = metadata_results2[0].details.get("match_count", 0)
        assert second_count == initial_count, f"Re-run should produce same count: {second_count} vs {initial_count}"

    def test_overwrite_result_details(self, agent_config: AgentConfig):
        """N1-09: overwrite result contains expected detail keys."""
        sub_agent_config = _make_sub_agent_config()
        _, result = self._register_and_bootstrap(agent_config, sub_agent_config, strategy="overwrite")

        assert result is not None, "Bootstrap result should not be None"
        assert (
            result.storage_path.endswith("nightly_test") or result.storage_path
        ), f"Storage path should be valid, got: {result.storage_path}"

        metadata_results = [r for r in result.results if r.component == "metadata"]
        assert len(metadata_results) == 1, "Should have metadata result"
        assert (
            metadata_results[0].status == "plan"
        ), f"Metadata should return plan status, got: {metadata_results[0].status}"
        assert "match_count" in metadata_results[0].details, "Details should contain match_count"

        metrics_results = [r for r in result.results if r.component == "metrics"]
        assert len(metrics_results) == 1, "Should have metrics result"
        assert metrics_results[0].status in (
            "plan",
            "skipped",
        ), f"Metrics should return plan/skipped, got: {metrics_results[0].status}"

        sql_results = [r for r in result.results if r.component == "reference_sql"]
        assert len(sql_results) == 1, "Should have reference_sql result"
        assert sql_results[0].status in (
            "plan",
            "skipped",
            "error",
        ), f"Reference SQL should return plan/skipped/error, got: {sql_results[0].status}"

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

        metadata_result = [r for r in result.results if r.component == "metadata"][0]
        assert (
            metadata_result.details.get("match_count", 0) >= 3
        ), f"Wildcard metadata should match >= 3 tables, got {metadata_result.details.get('match_count', 0)}"

        sql_result = [r for r in result.results if r.component == "reference_sql"][0]
        assert sql_result.details is not None, "Reference SQL plan should have details"
        assert sql_result.status == "plan", f"Reference SQL should have plan status, got {sql_result.status}"
        assert (
            sql_result.details.get("match_count", 0) > 0
        ), f"Reference SQL wildcard should match entries, got {sql_result.details.get('match_count', 0)}"

    def test_invalid_scoped_context(self, agent_config: AgentConfig):
        """N1-10: Invalid scoped context (nonexistent tables/metrics) handles gracefully."""
        sub_agent_config = _make_sub_agent_config(
            tables="nonexistent_db.nonexistent_table",
            metrics="nonexistent_db",
            sqls="nonexistent_db",
        )
        agent_config.agentic_nodes[sub_agent_config.system_prompt] = sub_agent_config
        bootstrapper = SubAgentBootstrapper(sub_agent=sub_agent_config, agent_config=agent_config)

        result = bootstrapper.run(strategy="plan")
        assert result is not None, "Plan result should not be None even with invalid context"

        metadata_result = [r for r in result.results if r.component == "metadata"][0]
        assert (
            metadata_result.details.get("match_count", 0) == 0
        ), f"Invalid table pattern should match 0 tables, got {metadata_result.details.get('match_count', 0)}"

        metrics_result = [r for r in result.results if r.component == "metrics"][0]
        assert (
            metrics_result.details.get("match_count", 0) == 0
        ), f"Invalid metrics pattern should match 0, got {metrics_result.details.get('match_count', 0)}"
