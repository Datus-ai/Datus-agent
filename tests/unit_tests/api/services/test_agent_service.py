"""Tests for datus.api.services.agent_service — tool validation and agent constants."""

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from datus.api.services.agent_service import (
    BUILTIN_SUBAGENTS,
    SUBAGENT_TOOL_REFERENCE,
    VALID_TOOL_CATEGORIES,
    VALID_TOOL_METHODS,
    AgentService,
    _normalize_created_at,
    _parse_tools,
    _utc_now_iso,
    _validate_tools,
)
from datus.utils.constants import HIDDEN_SYS_SUB_AGENTS, SYS_SUB_AGENTS

ISO_UTC_Z_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")


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

    def test_builtin_subagents_count(self):
        """BUILTIN_SUBAGENTS has expected number of agents."""
        assert len(BUILTIN_SUBAGENTS) == len(SYS_SUB_AGENTS - HIDDEN_SYS_SUB_AGENTS)

    def test_valid_tool_categories_non_empty(self):
        """VALID_TOOL_CATEGORIES is non-empty."""
        assert len(VALID_TOOL_CATEGORIES) >= 4

    def test_tool_reference_gen_sql(self):
        """gen_sql tool reference includes all tool categories."""
        assert "gen_sql" in SUBAGENT_TOOL_REFERENCE
        assert set(SUBAGENT_TOOL_REFERENCE["gen_sql"]) == set(VALID_TOOL_METHODS.keys())

    def test_valid_tool_methods_db_tools_has_methods(self):
        """db_tools category exposes core query methods."""
        assert "describe_table" in VALID_TOOL_METHODS["db_tools"]
        assert "get_table_ddl" in VALID_TOOL_METHODS["db_tools"]

    def test_valid_tool_methods_filesystem_tools_contains_read_file(self):
        """filesystem_tools contains read_file."""
        assert "read_file" in VALID_TOOL_METHODS["filesystem_tools"]

    def test_valid_tool_methods_semantic_tools_has_methods(self):
        """semantic_tools category is registered with its core methods."""
        assert "semantic_tools" in VALID_TOOL_METHODS
        assert "list_metrics" in VALID_TOOL_METHODS["semantic_tools"]
        assert "get_dimensions" in VALID_TOOL_METHODS["semantic_tools"]


class TestAgentServiceInit:
    """Tests for AgentService construction."""

    def test_init_succeeds(self):
        """AgentService can be instantiated."""
        svc = AgentService()
        assert isinstance(svc, AgentService)


class TestGetUseTools:
    """Tests for get_use_tools — tool reference lookup."""

    def test_known_agent_type_returns_tools(self):
        """get_use_tools returns tools for known agent type."""
        result = AgentService.get_use_tools("gen_sql")
        assert result.success is True
        assert isinstance(result.data, dict)
        assert set(result.data["tools"]) == set(SUBAGENT_TOOL_REFERENCE["gen_sql"])

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
        assert set(result.data["tools"]) == set(SUBAGENT_TOOL_REFERENCE["gen_report"])


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


class TestParseTools:
    """Tests for _parse_tools — normalize yaml tools field to list[str]."""

    def test_comma_separated_string_is_split(self):
        """Comma-separated string is split into trimmed entries."""
        assert _parse_tools("db_tools.*, context_search_tools.*") == [
            "db_tools.*",
            "context_search_tools.*",
        ]

    def test_string_entries_are_trimmed(self):
        """Surrounding whitespace is removed from each split entry."""
        assert _parse_tools("  db_tools.*  ,\tcontext_search_tools.read_query \n") == [
            "db_tools.*",
            "context_search_tools.read_query",
        ]

    def test_list_input_is_trimmed_and_passed_through(self):
        """List input is preserved with each entry trimmed."""
        assert _parse_tools(["db_tools.*", "  ctx.*  "]) == ["db_tools.*", "ctx.*"]

    def test_empty_string_returns_empty_list(self):
        """Empty / whitespace-only string yields []."""
        assert _parse_tools("") == []
        assert _parse_tools("   ") == []
        assert _parse_tools(",,, ,") == []

    def test_none_returns_empty_list(self):
        """None input yields []."""
        assert _parse_tools(None) == []

    def test_unsupported_type_returns_empty_list(self):
        """Non-string non-list inputs (e.g. int, dict) yield [] safely."""
        assert _parse_tools(42) == []
        assert _parse_tools({"db_tools": "*"}) == []


class TestUtcNowIso:
    """Tests for _utc_now_iso — UTC ISO-8601 with Z suffix."""

    def test_format_matches_iso_with_z_suffix(self):
        """Returned string matches ISO-8601 microsecond format ending in Z."""
        value = _utc_now_iso()
        assert ISO_UTC_Z_RE.match(value), f"unexpected format: {value!r}"

    def test_value_is_recent_utc_time(self):
        """Returned timestamp is within a few seconds of now (UTC)."""
        before = datetime.now(timezone.utc)
        value = _utc_now_iso()
        # Parse back: replace trailing Z so fromisoformat accepts it
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        delta = abs((parsed - before).total_seconds())
        assert delta < 5, f"timestamp {value} not close to {before.isoformat()}"


class TestNormalizeCreatedAt:
    """Tests for _normalize_created_at — coerce yaml-loaded values to ISO Z string."""

    def test_none_returns_none(self):
        """Missing value returns None."""
        assert _normalize_created_at(None) is None

    def test_string_passes_through(self):
        """String values pass through unchanged (assumed already ISO)."""
        s = "2026-04-30T09:20:31.545000Z"
        assert _normalize_created_at(s) == s

    def test_aware_datetime_converted_to_z_suffix(self):
        """Timezone-aware datetime converts to ISO Z."""
        dt = datetime(2026, 4, 30, 9, 20, 31, 545000, tzinfo=timezone.utc)
        assert _normalize_created_at(dt) == "2026-04-30T09:20:31.545000Z"

    def test_naive_datetime_assumed_utc(self):
        """Naive datetime is assumed UTC and gets Z suffix."""
        dt = datetime(2026, 4, 30, 9, 20, 31, 545000)
        assert _normalize_created_at(dt) == "2026-04-30T09:20:31.545000Z"

    def test_unsupported_type_returns_none(self):
        """Unsupported types (int, dict, etc.) yield None."""
        assert _normalize_created_at(123) is None
        assert _normalize_created_at({"foo": "bar"}) is None


@pytest.mark.asyncio
class TestGetAgent:
    """Tests for get_agent — retrieve single agent config."""

    EXPECTED_FIELDS = {"id", "name", "type", "description", "created_at", "tools", "rules", "catalogs", "subjects"}

    async def test_get_builtin_agent_has_full_schema(self, real_agent_config):
        """Builtin agent response carries the same field set as custom agents."""
        svc = AgentService()
        result = await svc.get_agent("gen_sql", real_agent_config)
        assert result.success is True
        agent = result.data["agent"]
        assert agent["name"] == "gen_sql"
        assert agent["id"] == "gen_sql"
        assert agent["type"] == "builtin"
        assert set(agent.keys()) == self.EXPECTED_FIELDS
        # All collection fields default to empty list, not None
        assert agent["tools"] == []
        assert agent["rules"] == []
        assert agent["catalogs"] == []
        assert agent["subjects"] == []
        # system_prompt must NOT be present
        assert "system_prompt" not in agent

    async def test_get_nonexistent_agent(self, real_agent_config):
        """get_agent returns error for unknown agent."""
        svc = AgentService()
        result = await svc.get_agent("totally_fake_agent", real_agent_config)
        assert result.success is False
        assert result.errorCode == "AGENT_NOT_FOUND"

    async def test_get_custom_agent_schema_matches_contract(self, real_agent_config):
        """Custom agent response matches the documented schema and omits system_prompt."""
        svc = AgentService()
        nodes = real_agent_config.agentic_nodes or {}
        assert nodes, "real_agent_config fixture must provide agentic_nodes"
        first_name = next(iter(nodes))
        result = await svc.get_agent(first_name, real_agent_config)
        assert result.success is True
        agent = result.data["agent"]
        assert agent["name"] == first_name
        assert agent["id"] == first_name
        assert set(agent.keys()) == self.EXPECTED_FIELDS
        assert "system_prompt" not in agent
        # tools is always a list — never a string
        assert isinstance(agent["tools"], list)
        # collection fields are always lists, never None
        for field in ("rules", "catalogs", "subjects"):
            assert isinstance(agent[field], list)

    async def test_get_custom_agent_parses_tools_string(self, real_agent_config):
        """yaml ``tools: "db_tools.*, ctx.*"`` is returned as a trimmed list."""
        # Inject a custom node with a comma-separated tools string directly into the
        # in-memory config — get_agent reads agent_config.agentic_nodes, not yaml.
        real_agent_config.agentic_nodes["string_tools_agent"] = {
            "type": "gen_sql",
            "description": "agent with string tools",
            "tools": "db_tools.*,  context_search_tools.read_query  ",
        }

        svc = AgentService()
        result = await svc.get_agent("string_tools_agent", real_agent_config)
        assert result.success is True
        assert result.data["agent"]["tools"] == [
            "db_tools.*",
            "context_search_tools.read_query",
        ]

    async def test_get_custom_agent_returns_explicit_created_at(self, real_agent_config):
        """An explicit yaml ``created_at`` is surfaced verbatim in the response."""
        real_agent_config.agentic_nodes["dated_agent"] = {
            "type": "gen_sql",
            "description": "agent with explicit created_at",
            "tools": "db_tools.*",
            "created_at": "2026-04-30T09:20:31.545000Z",
        }

        svc = AgentService()
        result = await svc.get_agent("dated_agent", real_agent_config)
        assert result.success is True
        assert result.data["agent"]["created_at"] == "2026-04-30T09:20:31.545000Z"

    async def test_get_custom_agent_created_at_falls_back_to_file_mtime(self, real_agent_config):
        """When yaml has no ``created_at``, fall back to agent.yml file mtime in ISO-Z."""
        # Ensure agent.yml exists so the mtime fallback can resolve
        config_path = Path(real_agent_config.home) / "agent.yml"
        config_path.write_text("agentic_nodes: {}\n", encoding="utf-8")
        real_agent_config.agentic_nodes["mtime_agent"] = {
            "type": "gen_sql",
            "description": "no explicit created_at",
            "tools": "db_tools.*",
        }

        svc = AgentService()
        result = await svc.get_agent("mtime_agent", real_agent_config)
        assert result.success is True
        created_at = result.data["agent"]["created_at"]
        assert isinstance(created_at, str) and ISO_UTC_Z_RE.match(created_at), f"unexpected created_at: {created_at!r}"


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

    async def test_create_agent_persists_created_at(self, real_agent_config):
        """create_agent writes a UTC ISO-Z created_at into agent.yml."""
        import yaml

        from datus.api.models.agent_models import CreateAgentInput

        config_path = Path(real_agent_config.home) / "agent.yml"
        if not config_path.exists():
            with open(config_path, "w") as f:
                yaml.dump({"agentic_nodes": {}}, f)

        svc = AgentService()
        result = await svc.create_agent(
            CreateAgentInput(name="created_at_agent", type="gen_sql"),
            real_agent_config,
        )
        assert result.success is True

        with open(config_path) as f:
            raw = yaml.safe_load(f)
        entry = raw["agentic_nodes"]["created_at_agent"]
        assert "created_at" in entry
        # Round-trip through the public API: the value surfaces unchanged
        get_result = await svc.get_agent("created_at_agent", real_agent_config)
        assert get_result.success is True
        created_at = get_result.data["agent"]["created_at"]
        # Either yaml stored a string (Z-suffixed) or a datetime that gets normalized
        assert created_at is not None and created_at.endswith("Z")

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
            EditAgentInput(id="nonexistent_id", name="nonexistent_agent", description="updated"),
            real_agent_config,
        )
        assert result.success is False
        assert result.errorCode == "AGENT_NOT_FOUND"

    async def test_edit_agent_invalid_tools(self, real_agent_config):
        """edit_agent rejects invalid tool patterns."""
        from datus.api.models.agent_models import EditAgentInput

        svc = AgentService()
        result = await svc.edit_agent(
            EditAgentInput(id="some_id", name="some_agent", tools=["bad_tools.bad"]),
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
        create_result = await svc.create_agent(
            CreateAgentInput(name="edit_me", type="gen_sql", description="original"),
            real_agent_config,
        )
        agent_id = create_result.data["id"]
        # Edit
        result = await svc.edit_agent(
            EditAgentInput(id=agent_id, name="edit_me", description="updated description"),
            real_agent_config,
        )
        assert result.success is True
        # Verify update persisted
        get_result = await svc.get_agent("edit_me", real_agent_config)
        assert get_result.success is True
        assert get_result.data["agent"]["description"] == "updated description"
