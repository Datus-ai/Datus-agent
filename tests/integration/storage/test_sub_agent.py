import copy
import os
from pathlib import Path
from typing import Any, Dict

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.schemas.agent_models import ScopedContext, SubAgentConfig
from datus.storage.metric.store import MetricRAG
from datus.storage.reference_sql import ReferenceSqlRAG
from datus.storage.schema_metadata import SchemaWithValueRAG
from datus.storage.sub_agent_kb_bootstrap import SUPPORTED_COMPONENTS, SubAgentBootstrapper
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.utils.sub_agent_manager import SubAgentManager
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


# =============================================================================
# Sub-agent manager CRUD and scoped KB execution
# =============================================================================


class _StubConfigurationManager:
    """Stub ConfigurationManager for testing."""

    def __init__(self, base_path: Path):
        self._data: Dict[str, Any] = {"agentic_nodes": {}}
        self.config_path = base_path / "agent.yml"

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def update_item(self, key: str, value, delete_old_key: bool = False):
        self._data[key] = value

    def remove_item_recursively(self, key: str, sub_key: str):
        if key in self._data and isinstance(self._data[key], dict):
            self._data[key].pop(sub_key, None)


class _StubPromptManager:
    """Stub PromptManager for testing."""

    def __init__(self, base_path: Path):
        self.templates_dir = base_path
        self.user_templates_dir = base_path

    def copy_to(self, template_name: str, destination_name: str, version: str):
        return f"{destination_name}_{version}.j2"


class _StubAgentConfig:
    """Stub AgentConfig for testing."""

    def __init__(self, base_dir: Path):
        self.rag_base_path = str(base_dir)
        self.db_type = "sqlite"
        self.agentic_nodes = {}

    def sub_agent_storage_path(self, sub_agent_name: str) -> str:
        return os.path.join(self.rag_base_path, "sub_agents", sub_agent_name)


def _build_manager(tmp_path):
    """Helper to build SubAgentManager with stub dependencies."""
    config_mgr = _StubConfigurationManager(tmp_path)
    stub_agent_config = _StubAgentConfig(tmp_path)
    manager = SubAgentManager(configuration_manager=config_mgr, namespace="demo", agent_config=stub_agent_config)
    manager._prompt_manager = _StubPromptManager(tmp_path)
    return manager, config_mgr, stub_agent_config


@pytest.mark.nightly
class TestSubAgentManager:
    """N7: Sub-agent management tests."""

    def test_create_sub_agent(self, tmp_path):
        """N7-01: Create a new sub-agent and verify it persists."""
        manager, config_mgr, stub_agent_config = _build_manager(tmp_path)

        new_config = SubAgentConfig(
            system_prompt="test_agent",
            agent_description="Test agent for nightly tests",
            scoped_context=ScopedContext(tables="california_schools.schools"),
        )

        result = manager.save_agent(new_config, previous_name=None)

        # Verify result structure
        assert isinstance(result, dict), f"save_agent should return dict, got {type(result)}"
        assert "config_path" in result, f"Result should have config_path, got keys: {list(result.keys())}"
        assert result["config_path"] == str(
            config_mgr.config_path
        ), f"config_path should match, got: {result['config_path']}"

        # Verify agent appears in list
        agents = manager.list_agents()
        assert "test_agent" in agents, f"New agent should be in list_agents(), got: {list(agents.keys())}"

        # Verify get_agent returns correct data
        retrieved = manager.get_agent("test_agent")
        assert retrieved is not None, "get_agent should return the created agent"
        assert (
            retrieved["system_prompt"] == "test_agent"
        ), f"system_prompt should match, got: {retrieved['system_prompt']}"
        assert (
            retrieved["agent_description"] == "Test agent for nightly tests"
        ), f"agent_description should match, got: {retrieved['agent_description']}"

    def test_list_and_get_agents(self, tmp_path):
        """N7-02: List and get multiple agents."""
        manager, config_mgr, stub_agent_config = _build_manager(tmp_path)

        # Create three agents
        agent_names = ["agent_alpha", "agent_beta", "agent_gamma"]
        for name in agent_names:
            config = SubAgentConfig(
                system_prompt=name,
                agent_description=f"Description for {name}",
                scoped_context=ScopedContext(tables=f"db.{name}_table"),
            )
            manager.save_agent(config)

        # Test list_agents returns all
        agents = manager.list_agents()
        assert len(agents) >= 3, f"Should have at least 3 agents, got {len(agents)}"
        for name in agent_names:
            assert name in agents, f"Agent '{name}' should be in list"

        # Test get_agent for each
        for name in agent_names:
            retrieved = manager.get_agent(name)
            assert retrieved is not None, f"get_agent('{name}') should not be None"
            assert (
                retrieved["system_prompt"] == name
            ), f"system_prompt should be '{name}', got: {retrieved['system_prompt']}"
            assert f"Description for {name}" in retrieved["agent_description"], f"Description mismatch for {name}"

        # Test get_agent for nonexistent
        nonexistent = manager.get_agent("nonexistent_agent_xyz")
        assert nonexistent is None, "get_agent for nonexistent agent should return None"

    def test_sub_agent_scoped_kb_execution(self, agent_config: AgentConfig):
        """N7-06: Sub-agent with scoped context can query knowledge base."""
        sub_agent_name = "nightly_n7_test"
        scoped_config = SubAgentConfig(
            system_prompt=sub_agent_name,
            agent_description="Nightly N7 test agent",
            tools="",
            mcp="",
            scoped_context=ScopedContext(
                tables="california_schools.*",
                metrics="california_schools",
                sqls="california_schools",
            ),
        )

        # Register in agentic_nodes
        agent_config.agentic_nodes[sub_agent_name] = scoped_config

        bootstrapper = SubAgentBootstrapper(sub_agent=scoped_config, agent_config=agent_config)

        try:
            # Bootstrap with overwrite
            result = bootstrapper.run(strategy="overwrite")
            assert result is not None, "Bootstrap should return a result"

            # Verify at least some components succeeded
            success_count = sum(1 for r in result.results if r.status == "success")
            assert (
                success_count > 0
            ), f"At least one component should succeed, got statuses: {[r.status for r in result.results]}"

            # Create ContextSearchTools with sub-agent name
            ctx_tools = ContextSearchTools(agent_config, sub_agent_name=sub_agent_name)

            # Verify subject tree is accessible
            tree_result = ctx_tools.list_subject_tree()
            assert tree_result.success == 1, f"list_subject_tree should succeed, got error: {tree_result.error}"

            # If reference SQL is available, verify search works
            if ctx_tools.has_reference_sql:
                sql_result = ctx_tools.search_reference_sql("school")
                assert sql_result.success == 1, f"search_reference_sql should succeed, got error: {sql_result.error}"

        finally:
            # Cleanup
            for comp in SUPPORTED_COMPONENTS:
                try:
                    bootstrapper._clear_component(comp)
                except Exception:
                    pass
