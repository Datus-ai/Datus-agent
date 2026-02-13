import copy

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.schemas.agent_models import ScopedContext, SubAgentConfig
from datus.storage.metric.store import MetricRAG
from datus.storage.reference_sql import ReferenceSqlRAG
from datus.storage.schema_metadata import SchemaWithValueRAG
from datus.storage.sub_agent_kb_bootstrap import SUPPORTED_COMPONENTS, SubAgentBootstrapper
from tests.conftest import load_acceptance_config


@pytest.fixture
def agent_config() -> AgentConfig:
    agent_config = load_acceptance_config(namespace="bird_school")
    agent_config.rag_base_path = "tests/data"
    agent_config.agentic_nodes = copy.deepcopy(agent_config.agentic_nodes)
    return agent_config


class TestBootstrap:
    def _setup_sub_agent_config(
        self,
        tables: str = "california_schools.*",
        metrics: str = "california_schools",
        sqls: str = "california_schools",
    ) -> SubAgentConfig:
        scoped_context = ScopedContext(tables=tables, metrics=metrics, sqls=sqls)
        return SubAgentConfig(
            system_prompt="test",
            agent_description="this is a test agent",
            tools="",
            mcp="",
            scoped_context=scoped_context,
        )

    @pytest.mark.nightly
    def test_plan(self, agent_config: AgentConfig):
        sub_agent_config = self._setup_sub_agent_config()
        scoped_context = sub_agent_config.scoped_context

        agent_config.agentic_nodes[sub_agent_config.system_prompt] = sub_agent_config

        bootstrapper = SubAgentBootstrapper(sub_agent=sub_agent_config, agent_config=agent_config)

        result = bootstrapper.run(strategy="plan")
        assert result
        assert result.storage_path == "tests/data/sub_agents/test"
        component_results = result.results
        assert len(component_results) == len(
            SUPPORTED_COMPONENTS
        ), f"Expected {len(SUPPORTED_COMPONENTS)} components, got {len(component_results)}"

        # Filter by component name instead of positional index
        metadata = [r for r in component_results if r.component == "metadata"][0]
        metrics = [r for r in component_results if r.component == "metrics"][0]
        sql = [r for r in component_results if r.component == "reference_sql"][0]

        assert (
            metadata.details.get("match_count", 0) >= 3
        ), f"Metadata should match >= 3 tables, got {metadata.details.get('match_count', 0)}"
        assert (
            metrics.details.get("match_count", 0) == 1
        ), f"Metrics should match 1, got {metrics.details.get('match_count', 0)}"
        assert sql.details.get("match_count", 0) == 13, f"SQL should match 13, got {sql.details.get('match_count', 0)}"

        # Part 2: wildcard patterns
        scoped_context.tables = "california_schools.*"
        scoped_context.metrics = "california_schools.*"
        scoped_context.sqls = "california_schools.*"

        bootstrapper = SubAgentBootstrapper(sub_agent=sub_agent_config, agent_config=agent_config)

        result = bootstrapper.run(strategy="plan")
        assert result
        component_results = result.results
        assert len(component_results) == len(SUPPORTED_COMPONENTS)

        metadata = [r for r in component_results if r.component == "metadata"][0]
        metrics = [r for r in component_results if r.component == "metrics"][0]
        sql = [r for r in component_results if r.component == "reference_sql"][0]

        assert metadata.details.get("match_count", 0) >= 3
        assert metrics.details.get("match_count", 0) == 1
        assert sql.details.get("match_count", 0) == 13

    @pytest.mark.nightly
    def test_overwrite(self, agent_config: AgentConfig):
        sub_agent_config = self._setup_sub_agent_config()
        agent_config.agentic_nodes[sub_agent_config.system_prompt] = sub_agent_config
        bootstrapper = SubAgentBootstrapper(sub_agent=sub_agent_config, agent_config=agent_config)

        result = bootstrapper.run(strategy="overwrite")
        assert result
        assert result.storage_path == "tests/data/sub_agents/test"

        component_results = result.results

        # Filter by component name instead of positional index
        metadata_result = [r for r in component_results if r.component == "metadata"][0]
        metrics_result = [r for r in component_results if r.component == "metrics"][0]
        sql_result = [r for r in component_results if r.component == "reference_sql"][0]

        # metadata
        table_schema_rag = SchemaWithValueRAG(agent_config, sub_agent_name="test")
        assert metadata_result.details.get("stored_tables", 0) == table_schema_rag.schema_store.table_size()

        # metrics
        metrics_rag = MetricRAG(agent_config, sub_agent_name="test")
        assert metrics_result.details.get("stored_metrics", 0) == metrics_rag.storage.table_size()

        # sql
        sql_rag = ReferenceSqlRAG(agent_config, sub_agent_name="test")
        assert sql_result.details.get("stored_sqls", 0) == sql_rag.reference_sql_storage.table_size()

        for component in SUPPORTED_COMPONENTS:
            bootstrapper._clear_component(component)
