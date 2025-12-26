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
    return agent_config


class TestBootstrap:
    def _setup_sub_agent_config(
        self,
        tables: str = "california_schools",
        metrics: str = "education.schools",
        sqls: str = "education.school_administration",
    ) -> SubAgentConfig:
        scoped_context = ScopedContext(tables=tables, metrics=metrics, sqls=sqls)
        return SubAgentConfig(
            system_prompt="test",
            agent_description="this is a test agent",
            tools="",
            mcp="",
            scoped_context=scoped_context,
        )

    def test_plan(self, agent_config: AgentConfig):
        sub_agent_config = self._setup_sub_agent_config()
        scoped_context = sub_agent_config.scoped_context

        agent_config.agentic_nodes[sub_agent_config.system_prompt] = sub_agent_config

        bootstrapper = SubAgentBootstrapper(sub_agent=sub_agent_config, agent_config=agent_config)

        result = bootstrapper.run(strategy="plan")
        assert result
        assert result.storage_path == "tests/data/sub_agents/test"
        component_results = result.results
        assert len(component_results) == 3
        # metadata
        assert component_results[0].details.get("match_count", 0) >= 3
        # metrics
        assert component_results[1].details.get("match_count", 0) == 5
        # sql
        assert component_results[2].details.get("match_count", 0) == 2

        scoped_context.tables = "california_schools.*"
        scoped_context.metrics = "education.schools.*"
        scoped_context.sqls = "education.school_*"

        bootstrapper = SubAgentBootstrapper(sub_agent=sub_agent_config, agent_config=agent_config)

        result = bootstrapper.run(strategy="plan")
        assert result
        component_results = result.results
        assert len(component_results) == 3
        assert component_results[0].details.get("match_count", 0) >= 3
        assert component_results[1].details.get("match_count", 0) == 5
        assert component_results[2].details.get("match_count", 0) == 7

    def test_overwrite(self, agent_config: AgentConfig):
        sub_agent_config = self._setup_sub_agent_config()
        agent_config.agentic_nodes[sub_agent_config.system_prompt] = sub_agent_config
        bootstrapper = SubAgentBootstrapper(sub_agent=sub_agent_config, agent_config=agent_config)

        result = bootstrapper.run(strategy="overwrite")
        assert result
        assert result.storage_path == "tests/data/sub_agents/test"

        table_schema_rag = SchemaWithValueRAG(agent_config, sub_agent_name="test")
        component_results = result.results
        # metadata
        assert component_results[0].details.get("stored_tables", 0) == table_schema_rag.schema_store.table_size()

        metrics_rag = MetricRAG(agent_config, sub_agent_name="test")
        # metrics
        assert component_results[1].details.get("stored_metrics", 0) == metrics_rag.storage.table_size()
        # sql
        sql_rag = ReferenceSqlRAG(agent_config, sub_agent_name="test")
        assert component_results[2].details.get("stored_sqls", 0) == sql_rag.reference_sql_storage.table_size()

        for component in SUPPORTED_COMPONENTS:
            bootstrapper._clear_component(component)
