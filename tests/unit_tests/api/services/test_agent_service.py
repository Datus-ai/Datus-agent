"""Tests for datus.api.services.agent_service — tool validation and agent constants."""

from pathlib import Path

import pytest

from datus.api.services.agent_service import (
    BUILTIN_SUBAGENTS,
    SUBAGENT_TOOL_REFERENCE,
    VALID_TOOL_CATEGORIES,
    VALID_TOOL_METHODS,
    AgentService,
    _validate_tools,
)


class TestValidateTools:
    """Tests for _validate_tools — pattern validation."""

    def test_exact_category_is_valid(self):
        """Exact category name (e.g. 'db_tools') is valid."""
        for category in VALID_TOOL_CATEGORIES:
            assert _validate_tools([category]) == []

    def test_wildcard_is_valid(self):
        """Wildcard pattern 'category.*' is valid."""
        for category in VALID_TOOL_CATEGORIES:
            assert _validate_tools([f"{category}.*"]) == []

    def test_specific_method_is_valid(self):
        """Specific method 'category.method' is valid if method exists."""
        for category, methods in VALID_TOOL_METHODS.items():
            for method in list(methods)[:2]:  # test first 2 methods per category
                assert _validate_tools([f"{category}.{method}"]) == []

    def test_unknown_category_is_invalid(self):
        """Unknown category returns it as invalid."""
        result = _validate_tools(["nonexistent_tools"])
        assert result == ["nonexistent_tools"]

    def test_unknown_method_is_invalid(self):
        """Valid category but unknown method is invalid."""
        result = _validate_tools(["db_tools.fake_method"])
        assert result == ["db_tools.fake_method"]

    def test_unknown_category_with_method_is_invalid(self):
        """Unknown category with method is invalid."""
        result = _validate_tools(["fake_tools.some_method"])
        assert result == ["fake_tools.some_method"]

    def test_empty_patterns_ignored(self):
        """Empty/whitespace patterns are silently skipped."""
        result = _validate_tools(["", "  ", "db_tools"])
        assert result == []

    def test_multiple_mixed_patterns(self):
        """Mix of valid and invalid patterns returns only invalid."""
        result = _validate_tools(["db_tools", "fake_tools", "db_tools.*", "bad.method"])
        assert "db_tools" not in result
        assert "db_tools.*" not in result
        assert "fake_tools" in result
        assert "bad.method" in result

    def test_empty_list_returns_empty(self):
        """Empty input list returns empty list."""
        assert _validate_tools([]) == []


class TestConstants:
    """Tests for module-level constants."""

    def test_builtin_subagents_has_gen_sql(self):
        """BUILTIN_SUBAGENTS contains gen_sql entry."""
        assert "gen_sql" in BUILTIN_SUBAGENTS
        assert isinstance(BUILTIN_SUBAGENTS["gen_sql"], str)

    def test_builtin_subagents_count(self):
        """BUILTIN_SUBAGENTS has expected number of agents."""
        assert len(BUILTIN_SUBAGENTS) == 6

    def test_valid_tool_categories_non_empty(self):
        """VALID_TOOL_CATEGORIES is non-empty."""
        assert len(VALID_TOOL_CATEGORIES) >= 4

    def test_tool_reference_gen_sql(self):
        """gen_sql tool reference includes all tool categories."""
        assert "gen_sql" in SUBAGENT_TOOL_REFERENCE
        assert set(SUBAGENT_TOOL_REFERENCE["gen_sql"]) == set(VALID_TOOL_METHODS.keys())

    def test_valid_tool_methods_db_tools_has_methods(self):
        """db_tools category has at least one method."""
        assert len(VALID_TOOL_METHODS["db_tools"]) > 0

    def test_valid_tool_methods_filesystem_tools_contains_read_file(self):
        """filesystem_tools contains read_file."""
        assert "read_file" in VALID_TOOL_METHODS["filesystem_tools"]


class TestAgentServiceInit:
    """Tests for AgentService construction."""

    def test_init_succeeds(self):
        """AgentService can be instantiated."""
        svc = AgentService()
        assert svc is not None


class TestGetUseTools:
    """Tests for get_use_tools — tool reference lookup."""

    def test_known_agent_type_returns_tools(self):
        """get_use_tools returns tools for known agent type."""
        result = AgentService.get_use_tools("gen_sql")
        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) > 0

    def test_unknown_agent_type_returns_error(self):
        """get_use_tools returns error for unknown agent type."""
        result = AgentService.get_use_tools("nonexistent")
        assert result.success is False
        assert result.errorCode == "INVALID_AGENT_TYPE"
        assert "nonexistent" in result.errorMessage

    def test_gen_report_returns_tools(self):
        """get_use_tools returns tools for gen_report."""
        result = AgentService.get_use_tools("gen_report")
        assert result.success is True
        assert len(result.data) > 0


@pytest.mark.asyncio
class TestListAgents:
    """Tests for list_agents — enumerate all available agents."""

    async def test_list_includes_builtins(self, real_agent_config):
        """list_agents includes all builtin agents."""
        svc = AgentService()
        result = await svc.list_agents(real_agent_config)
        assert result.success is True
        agent_names = {a["name"] for a in result.data["agents"]}
        for builtin_name in BUILTIN_SUBAGENTS:
            assert builtin_name in agent_names

    async def test_list_contains_builtin_type_entries(self, real_agent_config):
        """At least some agents in the list have type='builtin'."""
        svc = AgentService()
        result = await svc.list_agents(real_agent_config)
        builtin_agents = [a for a in result.data["agents"] if a["type"] == "builtin"]
        assert len(builtin_agents) == len(BUILTIN_SUBAGENTS)

    async def test_list_includes_custom_agents(self, real_agent_config):
        """list_agents includes custom agents from agentic_nodes."""
        svc = AgentService()
        result = await svc.list_agents(real_agent_config)
        assert result.success is True
        # real_agent_config has agentic_nodes from conftest
        agent_names = {a["name"] for a in result.data["agents"]}
        assert len(agent_names) >= len(BUILTIN_SUBAGENTS)


@pytest.mark.asyncio
class TestGetAgent:
    """Tests for get_agent — retrieve single agent config."""

    async def test_get_builtin_agent(self, real_agent_config):
        """get_agent returns builtin agent info."""
        svc = AgentService()
        result = await svc.get_agent("gen_sql", real_agent_config)
        assert result.success is True
        assert result.data["agent"]["name"] == "gen_sql"
        assert result.data["agent"]["type"] == "builtin"

    async def test_get_nonexistent_agent(self, real_agent_config):
        """get_agent returns error for unknown agent."""
        svc = AgentService()
        result = await svc.get_agent("totally_fake_agent", real_agent_config)
        assert result.success is False
        assert result.errorCode == "AGENT_NOT_FOUND"

    async def test_get_custom_agent_from_agentic_nodes(self, real_agent_config):
        """get_agent returns custom agent info from agentic_nodes."""
        svc = AgentService()
        # real_agent_config has agentic_nodes from conftest (e.g. 'gensql', 'chat', etc.)
        nodes = real_agent_config.agentic_nodes or {}
        if nodes:
            first_name = next(iter(nodes))
            result = await svc.get_agent(first_name, real_agent_config)
            assert result.success is True
            assert result.data["agent"]["name"] == first_name


@pytest.mark.asyncio
class TestCreateAgent:
    """Tests for create_agent — agent creation with YAML persistence."""

    async def test_create_agent_success(self, real_agent_config):
        """create_agent creates a new custom agent."""
        import yaml

        from datus.api.models.agent_models import CreateAgentInput

        # Ensure agent.yml exists
        config_path = Path(real_agent_config.home) / "agent.yml"
        if not config_path.exists():
            with open(config_path, "w") as f:
                yaml.dump({"agentic_nodes": {}}, f)

        svc = AgentService()
        request = CreateAgentInput(
            name="test_new_agent",
            type="gen_sql",
            description="Test agent for unit tests",
            tools=["db_tools"],
        )
        result = await svc.create_agent(request, real_agent_config)
        assert result.success is True
        assert result.data["name"] == "test_new_agent"

    async def test_create_agent_duplicate_name_fails(self, real_agent_config):
        """create_agent rejects duplicate agent name."""
        import yaml

        from datus.api.models.agent_models import CreateAgentInput

        config_path = Path(real_agent_config.home) / "agent.yml"
        if not config_path.exists():
            with open(config_path, "w") as f:
                yaml.dump({"agentic_nodes": {}}, f)

        svc = AgentService()
        # Create first
        await svc.create_agent(
            CreateAgentInput(name="dup_agent", type="gen_sql"),
            real_agent_config,
        )
        # Try duplicate
        result = await svc.create_agent(
            CreateAgentInput(name="dup_agent", type="gen_sql"),
            real_agent_config,
        )
        assert result.success is False
        assert result.errorCode == "AGENT_ALREADY_EXISTS"

    async def test_create_agent_builtin_name_fails(self, real_agent_config):
        """create_agent rejects builtin agent names."""
        import yaml

        from datus.api.models.agent_models import CreateAgentInput

        config_path = Path(real_agent_config.home) / "agent.yml"
        if not config_path.exists():
            with open(config_path, "w") as f:
                yaml.dump({"agentic_nodes": {}}, f)

        svc = AgentService()
        result = await svc.create_agent(
            CreateAgentInput(name="gen_sql", type="gen_sql"),
            real_agent_config,
        )
        assert result.success is False
        assert result.errorCode == "AGENT_ALREADY_EXISTS"

    async def test_create_agent_invalid_tools_fails(self, real_agent_config):
        """create_agent rejects invalid tool patterns."""
        import yaml

        from datus.api.models.agent_models import CreateAgentInput

        config_path = Path(real_agent_config.home) / "agent.yml"
        if not config_path.exists():
            with open(config_path, "w") as f:
                yaml.dump({"agentic_nodes": {}}, f)

        svc = AgentService()
        result = await svc.create_agent(
            CreateAgentInput(name="bad_tools_agent", type="gen_sql", tools=["fake_tool_category"]),
            real_agent_config,
        )
        assert result.success is False
        assert result.errorCode == "INVALID_TOOLS"


@pytest.mark.asyncio
class TestEditAgent:
    """Tests for edit_agent — agent update with YAML persistence."""

    async def test_edit_agent_not_found(self, real_agent_config):
        """edit_agent returns error for nonexistent agent."""
        from datus.api.models.agent_models import EditAgentInput

        svc = AgentService()
        result = await svc.edit_agent(
            EditAgentInput(name="nonexistent_agent", description="updated"),
            real_agent_config,
        )
        assert result.success is False
        assert result.errorCode == "AGENT_NOT_FOUND"

    async def test_edit_agent_invalid_tools(self, real_agent_config):
        """edit_agent rejects invalid tool patterns."""
        from datus.api.models.agent_models import EditAgentInput

        svc = AgentService()
        result = await svc.edit_agent(
            EditAgentInput(name="some_agent", tools=["bad_tools.bad"]),
            real_agent_config,
        )
        assert result.success is False
        assert result.errorCode == "INVALID_TOOLS"

    async def test_edit_existing_agent(self, real_agent_config):
        """edit_agent updates existing custom agent."""
        import yaml

        from datus.api.models.agent_models import CreateAgentInput, EditAgentInput

        config_path = Path(real_agent_config.home) / "agent.yml"
        if not config_path.exists():
            with open(config_path, "w") as f:
                yaml.dump({"agentic_nodes": {}}, f)

        svc = AgentService()
        # Create first
        await svc.create_agent(
            CreateAgentInput(name="edit_me", type="gen_sql", description="original"),
            real_agent_config,
        )
        # Edit
        result = await svc.edit_agent(
            EditAgentInput(name="edit_me", description="updated description"),
            real_agent_config,
        )
        assert result.success is True
        # Verify update persisted
        get_result = await svc.get_agent("edit_me", real_agent_config)
        assert get_result.success is True
        assert get_result.data["agent"]["description"] == "updated description"
