"""Tests for datus.api.services.agent_service — tool validation and agent constants."""

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from datus.api.services.agent_service import (
    _ASK_AGENT_FILESYSTEM_READ_ONLY,
    _TOOL_CATEGORIES_BY_AGENT_TYPE,
    _USER_FACING_TOOL_CATEGORIES,
    BUILTIN_SUBAGENTS,
    SUBAGENT_TOOL_REFERENCE,
    VALID_TOOL_CATEGORIES,
    VALID_TOOL_METHODS,
    AgentService,
    _build_scoped_context,
    _build_tool_types,
    _classify_subject_paths,
    _format_csv,
    _merge_subjects_from_scoped_context,
    _normalize_created_at,
    _parse_csv,
    _parse_tools,
    _strip_leading_slashes,
    _utc_now_iso,
    _validate_tools,
    _validate_tools_for_agent_type,
    sanitize_agentic_node_name,
)
from datus.utils.constants import HIDDEN_SYS_SUB_AGENTS, SYS_SUB_AGENTS

ISO_UTC_Z_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")


@pytest.fixture
def agent_yml_with_singleton(real_agent_config):
    """Pre-seed an ``agent.yml`` at the resolved home and install the
    ``ConfigurationManager`` singleton at that path.

    ``_save_agentic_nodes`` and ``get_agent``'s mtime fallback now route
    through ``configuration_manager()``; tests that exercise create/edit/get
    need the singleton wired up to a tmp-path yaml so writes don't escape
    the temp dir. The fixture also resets the singleton on teardown so
    subsequent tests don't see leaked state.
    """
    from datus.configuration import agent_config_loader

    home = real_agent_config.path_manager.datus_home
    cfg_path = home / "agent.yml"
    if not cfg_path.exists():
        cfg_path.write_text("agent: {}\n", encoding="utf-8")
    agent_config_loader.configuration_manager(config_path=str(cfg_path), reload=True)
    yield cfg_path
    agent_config_loader.CONFIGURATION_MANAGER = None


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


class TestValidateToolsForAgentType:
    """Per-agent-type allowlist gate.

    ``ask_report`` / ``ask_dashboard`` are read-only artifact consultants. The
    gate prevents callers from re-enabling filesystem writes beyond the
    documented allowlist; other agent types rely on syntactic tool validation.
    """

    def test_non_artifact_ask_agents_unrestricted(self):
        """Other agent types fall back to the syntactic _validate_tools
        check only — this helper returns empty for them."""
        for agent_type in ("chat", "gen_sql", "gen_report", "ask_metrics"):
            assert _validate_tools_for_agent_type(["filesystem_tools.write_file"], agent_type) == []
            assert _validate_tools_for_agent_type(["filesystem_tools.*"], agent_type) == []

    def test_ask_report_accepts_read_only_filesystem(self):
        """The three read-side filesystem methods are the load-bearing
        contract — an ask_* agent without them can't even read the
        artifact it's bound to."""
        for tool in ("filesystem_tools.read_file", "filesystem_tools.glob", "filesystem_tools.grep"):
            assert _validate_tools_for_agent_type([tool], "ask_report") == []
            assert _validate_tools_for_agent_type([tool], "ask_dashboard") == []

    def test_ask_report_accepts_full_category_wildcards_for_non_filesystem(self):
        """Wildcards on categories whose full method set is already in the
        allowlist (semantic / context_search / reference_template) must
        pass — those categories have no write methods to suppress."""
        for category in ("semantic_tools", "context_search_tools", "reference_template_tools"):
            assert _validate_tools_for_agent_type([f"{category}.*"], "ask_report") == []
            assert _validate_tools_for_agent_type([f"{category}.*"], "ask_dashboard") == []

    @pytest.mark.parametrize(
        "forbidden",
        [
            "filesystem_tools.write_file",
            "filesystem_tools.edit_file",
            # Wildcard that would include write_file / edit_file in its expansion.
            "filesystem_tools.*",
            # Bare category — same wildcard concern.
            "filesystem_tools",
        ],
    )
    def test_ask_report_rejects_filesystem_write_tools(self, forbidden):
        rejected = _validate_tools_for_agent_type([forbidden], "ask_report")
        assert forbidden in rejected

    @pytest.mark.parametrize(
        "forbidden",
        [
            "filesystem_tools.write_file",
            "filesystem_tools.edit_file",
            "filesystem_tools.*",
            "filesystem_tools",
        ],
    )
    def test_ask_dashboard_rejects_filesystem_write_tools(self, forbidden):
        rejected = _validate_tools_for_agent_type([forbidden], "ask_dashboard")
        assert forbidden in rejected

    def test_ask_report_catalog_excludes_filesystem_writes(self):
        """The ``tool_types`` block returned by ``get_use_tools`` for
        ask_* must not expose write_file / edit_file — the editor picker
        should never surface them as available options."""
        fs_methods = set(SUBAGENT_TOOL_REFERENCE["ask_report"]["tool_types"]["filesystem_tools"]["tools"])
        assert "write_file" not in fs_methods
        assert "edit_file" not in fs_methods
        assert {"read_file", "glob", "grep"}.issubset(fs_methods)

    def test_ask_metrics_allows_valid_tools_outside_default_surface(self):
        tools = [
            "db_tools.read_query",
            "date_parsing_tools.parse_temporal_expressions",
            "semantic_tools.validate_semantic",
            "semantic_tools.*",
            "context_search_tools.search_reference_sql",
        ]
        assert _validate_tools(tools) == []
        assert _validate_tools_for_agent_type(tools, "ask_metrics") == []

    @pytest.mark.parametrize("agent_type", ["ask_report", "ask_dashboard", "ask_metrics"])
    def test_ask_default_tools_all_resolve(self, agent_type):
        """Every entry in ``default_tools`` for ask_* must be recognised by
        ``_validate_tools``.

        Anchors the regression where ``db_tools.execute_sql`` was kept in
        the curated preselect list after the underlying method had been
        renamed to ``read_query``. ``_validate_tools`` rejected it, which
        in turn broke saas-side ``create_agent`` calls that fed the
        preselect through unchanged.
        """
        defaults = SUBAGENT_TOOL_REFERENCE[agent_type]["default_tools"]
        invalid = _validate_tools(defaults)
        assert invalid == [], f"{agent_type} default_tools has unrecognised entries: {invalid}"

    @pytest.mark.parametrize("agent_type", ["ask_report", "ask_dashboard", "ask_metrics"])
    def test_ask_default_tools_pass_agent_type_gate(self, agent_type):
        """``default_tools`` must also satisfy the per-agent-type allowlist
        — preselected tools should never include anything the saas editor
        would later reject (e.g. ``filesystem_tools.write_file``)."""
        defaults = SUBAGENT_TOOL_REFERENCE[agent_type]["default_tools"]
        rejected = _validate_tools_for_agent_type(defaults, agent_type)
        assert rejected == [], f"{agent_type} default_tools rejected by ask gate: {rejected}"


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

    def test_tool_reference_gen_sql_has_saas_shape(self):
        """gen_sql tool reference has the saas {default_tools, tool_types} shape."""
        entry = SUBAGENT_TOOL_REFERENCE["gen_sql"]
        assert set(entry.keys()) == {"default_tools", "tool_types"}
        # gen_sql surfaces only the 3 categories the saas editor renders for
        # the type — db / semantic / context_search. Other categories
        # (filesystem / date_parsing / reference_template) are intentionally
        # hidden so the picker matches the actual default_tools surface.
        assert set(entry["tool_types"].keys()) == {
            "db_tools",
            "semantic_tools",
            "context_search_tools",
        }
        for category, payload in entry["tool_types"].items():
            assert payload == {"tools": sorted(VALID_TOOL_METHODS[category])}
        # gen_sql defaults to db / semantic / context_search wildcards (matches saas)
        assert entry["default_tools"] == [
            "db_tools.*",
            "semantic_tools.*",
            "context_search_tools.*",
        ]

    def test_user_facing_categories_excludes_platform_doc_tools(self):
        """platform_doc_tools is a valid tool but is intentionally hidden from the editor."""
        assert "platform_doc_tools" in VALID_TOOL_METHODS
        assert "platform_doc_tools" not in _USER_FACING_TOOL_CATEGORIES

    def test_filesystem_tools_valid_methods_match_runtime_surface(self):
        """``VALID_TOOL_METHODS["filesystem_tools"]`` is derived from
        ``FilesystemFuncTool.all_tools_name()`` so it auto-tracks the
        runtime instead of drifting (``delete_file`` was previously
        missing from the hand-curated set).

        Pin both directions: every name the runtime advertises ends up
        in the saas catalog, and every name in the catalog is a real
        public method on the class.
        """
        from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

        catalog = VALID_TOOL_METHODS["filesystem_tools"]
        introspected = set(FilesystemFuncTool.all_tools_name())
        assert catalog == introspected

        # The runtime advertises its tool surface in ``available_tools``;
        # the read/write/edit/delete + glob/grep set is the load-bearing
        # contract — make sure introspection captures all of them and
        # filters out BaseTool framework methods.
        expected_runtime_methods = {
            "read_file",
            "write_file",
            "edit_file",
            "delete_file",
            "glob",
            "grep",
        }
        assert introspected == expected_runtime_methods
        for name in expected_runtime_methods:
            assert hasattr(FilesystemFuncTool, name), f"FilesystemFuncTool dropped method {name!r}"
        # BaseTool framework methods must NOT show up in the catalog.
        for framework_method in ("set_tool_context", "get_actions", "call_action"):
            assert framework_method not in catalog

    def test_tool_reference_gen_report_has_saas_defaults(self):
        """gen_report defaults to semantic.* + context_search.list_subject_tree."""
        entry = SUBAGENT_TOOL_REFERENCE["gen_report"]
        assert entry["default_tools"] == [
            "semantic_tools.*",
            "context_search_tools.list_subject_tree",
        ]
        # gen_report exposes db / semantic / context_search plus date_parsing
        # and reference_template — the 5-category surface the saas editor's
        # else-branch rendered before the per-type whitelist moved server-side.
        assert set(entry["tool_types"].keys()) == {
            "db_tools",
            "semantic_tools",
            "context_search_tools",
            "date_parsing_tools",
            "reference_template_tools",
        }

    def test_tool_reference_ask_metrics_has_narrow_defaults(self):
        """ask_metrics defaults to metric QA tools, while custom configs can opt into more."""
        entry = SUBAGENT_TOOL_REFERENCE["ask_metrics"]
        assert entry["default_tools"] == [
            "context_search_tools.search_metrics",
            "context_search_tools.get_metrics",
            "semantic_tools.list_metrics",
            "semantic_tools.get_dimensions",
            "semantic_tools.query_metrics",
            "semantic_tools.attribution_analyze",
            "context_search_tools.list_subject_tree",
        ]
        assert set(entry["tool_types"].keys()) == set(_USER_FACING_TOOL_CATEGORIES)
        for category, payload in entry["tool_types"].items():
            assert payload["tools"] == sorted(VALID_TOOL_METHODS[category])

    def test_tool_reference_chat_has_full_default_set(self):
        """chat default_tools enumerates every non-semantic category as wildcard."""
        entry = SUBAGENT_TOOL_REFERENCE["chat"]
        assert entry["default_tools"] == [
            "db_tools.*",
            "context_search_tools.*",
            "reference_template_tools.*",
            "date_parsing_tools.*",
            "filesystem_tools.*",
            "memory_tools.*",
            "platform_doc_tools.*",
        ]
        # chat is the most permissive agent type — the picker surfaces every
        # user-facing category plus the dedicated memory tools.
        # ``platform_doc_tools`` stays in default_tools but not in tool_types
        # (matches the documented "valid tool, hidden from picker" precedent).
        assert set(entry["tool_types"].keys()) == set(_USER_FACING_TOOL_CATEGORIES) | {"memory_tools"}
        assert "platform_doc_tools" not in entry["tool_types"]

    def test_reference_template_tools_registered(self):
        """reference_template_tools category exposes the 4 expected methods."""
        assert "reference_template_tools" in VALID_TOOL_METHODS
        assert VALID_TOOL_METHODS["reference_template_tools"] == {
            "search_reference_template",
            "get_reference_template",
            "render_reference_template",
            "execute_reference_template",
        }

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
    """Tests for get_use_tools — saas-shape tool reference lookup."""

    def test_gen_sql_returns_default_tools_and_tool_types(self):
        """gen_sql payload matches the saas {default_tools, tool_types} contract."""
        result = AgentService.get_use_tools("gen_sql")
        assert result.success is True
        assert set(result.data.keys()) == {"default_tools", "tool_types"}
        assert result.data["default_tools"] == [
            "db_tools.*",
            "semantic_tools.*",
            "context_search_tools.*",
        ]
        # gen_sql surfaces only 3 categories — the per-type whitelist matches
        # the saas editor's ``gen_sql`` branch in tool-tree.ts.
        assert set(result.data["tool_types"].keys()) == {
            "db_tools",
            "semantic_tools",
            "context_search_tools",
        }
        for category, payload in result.data["tool_types"].items():
            assert list(payload.keys()) == ["tools"]
            assert payload["tools"] == sorted(VALID_TOOL_METHODS[category])

    def test_gen_report_returns_correct_defaults(self):
        """gen_report's default_tools is a curated subset, not the full wildcard list."""
        result = AgentService.get_use_tools("gen_report")
        assert result.success is True
        assert result.data["default_tools"] == [
            "semantic_tools.*",
            "context_search_tools.list_subject_tree",
        ]
        # gen_report's 5-category surface mirrors the saas editor's else
        # branch — no filesystem_tools because the type has no filesystem
        # defaults and the picker historically excluded it.
        assert set(result.data["tool_types"].keys()) == {
            "db_tools",
            "semantic_tools",
            "context_search_tools",
            "date_parsing_tools",
            "reference_template_tools",
        }

    def test_ask_metrics_returns_broad_configurable_tool_types(self):
        """ask_metrics keeps narrow defaults but surfaces valid configurable tools."""
        result = AgentService.get_use_tools("ask_metrics")
        assert result.success is True
        assert result.data["default_tools"] == SUBAGENT_TOOL_REFERENCE["ask_metrics"]["default_tools"]
        assert set(result.data["tool_types"].keys()) == set(_USER_FACING_TOOL_CATEGORIES)
        assert "read_query" in result.data["tool_types"]["db_tools"]["tools"]
        assert "parse_temporal_expressions" in result.data["tool_types"]["date_parsing_tools"]["tools"]
        assert result.data["tool_types"] == SUBAGENT_TOOL_REFERENCE["ask_metrics"]["tool_types"]

    def test_chat_includes_reference_template_tools_in_default(self):
        """chat default_tools wires up reference_template_tools.* (saas parity)."""
        result = AgentService.get_use_tools("chat")
        assert result.success is True
        assert "reference_template_tools.*" in result.data["default_tools"]
        assert "reference_template_tools" in result.data["tool_types"]
        # chat is the most permissive agent type; its tool_types covers every
        # user-facing category plus memory_tools so the editor can surface them.
        assert set(result.data["tool_types"].keys()) == set(_USER_FACING_TOOL_CATEGORIES) | {"memory_tools"}

    @pytest.mark.parametrize("agent_type", ["ask_report", "ask_dashboard"])
    def test_ask_agent_tool_types_includes_filesystem_read_only(self, agent_type):
        """ask_* exposes filesystem_tools as a read-only subset (glob / grep
        / read_file) — the editor picker needs to render the category so
        operators can see (and tweak) the read-side default_tools."""
        result = AgentService.get_use_tools(agent_type)
        assert result.success is True
        tool_types = result.data["tool_types"]
        # ask_* surfaces 6 categories — the 5 from gen_report's else-branch
        # plus filesystem_tools restricted to the read-only methods.
        assert set(tool_types.keys()) == {
            "db_tools",
            "semantic_tools",
            "context_search_tools",
            "reference_template_tools",
            "date_parsing_tools",
            "filesystem_tools",
        }
        assert set(tool_types["filesystem_tools"]["tools"]) == set(_ASK_AGENT_FILESYSTEM_READ_ONLY)


class TestPerAgentTypeCategoryWhitelist:
    """Tests for the per-agent-type editor whitelist that previously lived
    only in the saas frontend (tool-tree.ts).

    The whitelist is now the API's single source of truth: the editor
    renders whatever ``tool_types`` it receives, and the same block gates
    the artifact ask-agent write-path validation via
    :func:`_validate_tools_for_agent_type`.
    """

    def test_every_subagent_type_has_a_whitelist(self):
        """Every entry in :data:`SUBAGENT_TOOL_REFERENCE` must have a
        corresponding category whitelist — otherwise ``_build_tool_types``
        would KeyError at import time."""
        for agent_type in SUBAGENT_TOOL_REFERENCE:
            assert agent_type in _TOOL_CATEGORIES_BY_AGENT_TYPE, (
                f"{agent_type!r} is missing from _TOOL_CATEGORIES_BY_AGENT_TYPE"
            )

    def test_whitelist_only_references_known_categories(self):
        """Every category mentioned in the whitelist must also be a real
        VALID_TOOL_METHODS entry — otherwise ``_build_tool_types`` would
        KeyError when composing ``tool_types``."""
        for agent_type, categories in _TOOL_CATEGORIES_BY_AGENT_TYPE.items():
            for category in categories:
                assert category in VALID_TOOL_METHODS, (
                    f"{agent_type!r} whitelist references unknown category {category!r}"
                )

    def test_build_tool_types_returns_per_type_categories(self):
        """``_build_tool_types(agent_type)`` returns exactly the whitelisted
        categories, in the order declared, with the full method set per
        category."""
        for agent_type, categories in _TOOL_CATEGORIES_BY_AGENT_TYPE.items():
            tool_types = _build_tool_types(agent_type)
            assert list(tool_types.keys()) == list(categories)
            for category in categories:
                payload = tool_types[category]
                assert set(payload.keys()) == {"tools"}
                if agent_type in {"ask_report", "ask_dashboard"} and category == "filesystem_tools":
                    # ask_* filesystem is the read-only subset — write tools
                    # must be absent and the read trio must be present.
                    assert set(payload["tools"]) == set(_ASK_AGENT_FILESYSTEM_READ_ONLY)
                else:
                    assert payload["tools"] == sorted(VALID_TOOL_METHODS[category])

    def test_payload_no_longer_wraps_in_tools_key(self):
        """Old shape {"tools": [...]} is gone — data is now {default_tools, tool_types}."""
        result = AgentService.get_use_tools("gen_sql")
        assert "tools" not in result.data, "legacy 'tools' key must not appear at top level"

    def test_unknown_agent_type_returns_error(self):
        """get_use_tools returns error for unknown agent type."""
        result = AgentService.get_use_tools("nonexistent")
        assert result.success is False
        assert result.errorCode == "INVALID_AGENT_TYPE"
        assert "nonexistent" in result.errorMessage

    def test_known_agent_types_match_subagent_reference(self):
        """Every key in SUBAGENT_TOOL_REFERENCE returns success and has saas shape."""
        for agent_type in SUBAGENT_TOOL_REFERENCE:
            result = AgentService.get_use_tools(agent_type)
            assert result.success is True, f"agent_type {agent_type} should resolve"
            assert set(result.data.keys()) == {"default_tools", "tool_types"}


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


class TestSanitizeAgenticNodeName:
    """Tests for sanitize_agentic_node_name — make a name safe for yaml/path keys."""

    def test_alphanumeric_passthrough(self):
        """Letters / digits / underscore / hyphen pass through unchanged."""
        assert sanitize_agentic_node_name("OrderAnalyst-2") == "OrderAnalyst-2"

    def test_unicode_and_punct_replaced_with_underscore(self):
        """Spaces, dots, slashes, unicode all collapse to underscore."""
        assert sanitize_agentic_node_name("订单 分析/v.1") == "______v_1"

    def test_empty_and_none_become_empty_string(self):
        """Empty / None inputs degrade to '' rather than raising."""
        assert sanitize_agentic_node_name("") == ""
        assert sanitize_agentic_node_name(None) == ""  # type: ignore[arg-type]


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

    async def test_get_custom_agent_schema_matches_contract(self, real_agent_config, agent_yml_with_singleton):
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

    async def test_get_custom_agent_parses_tools_string(self, real_agent_config, agent_yml_with_singleton):
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

    async def test_get_custom_agent_reads_description_from_agent_description_key(self, real_agent_config):
        """yaml's ``agent_description`` key surfaces as the API ``description`` field.

        The runtime stores descriptions under ``agent_description`` (read by
        sub_agent_task_tool, agentic_node, the wizard, and Datus-backend's
        config_loader). The /agent endpoint must surface the same key under
        the API contract's ``description`` field so the editor reads what the
        runtime sees.
        """
        real_agent_config.agentic_nodes["modern_agent"] = {
            "type": "gen_sql",
            "agent_description": "agent stored under the runtime-visible key",
            "tools": "db_tools.*",
            "created_at": "2026-04-30T09:20:31.545000Z",
        }

        svc = AgentService()
        result = await svc.get_agent("modern_agent", real_agent_config)
        assert result.success is True
        assert result.data["agent"]["description"] == "agent stored under the runtime-visible key"

    async def test_get_custom_agent_falls_back_to_legacy_description_key(self, real_agent_config):
        """Older yaml files stored the description under ``description``.

        Until a save migrates the entry, the read path must fall back so
        existing configs keep rendering the right text in the editor.
        """
        real_agent_config.agentic_nodes["legacy_agent"] = {
            "type": "gen_sql",
            "description": "stored under the legacy key only",
            "tools": "db_tools.*",
            "created_at": "2026-04-30T09:20:31.545000Z",
        }

        svc = AgentService()
        result = await svc.get_agent("legacy_agent", real_agent_config)
        assert result.success is True
        assert result.data["agent"]["description"] == "stored under the legacy key only"

    async def test_get_custom_agent_prefers_agent_description_over_legacy(self, real_agent_config):
        """When both keys are present, ``agent_description`` wins.

        This matches the migration semantics: edit_agent writes the new key
        and clears the legacy one, but during the transition both may briefly
        coexist; the runtime sees ``agent_description`` so the API must too.
        """
        real_agent_config.agentic_nodes["mixed_agent"] = {
            "type": "gen_sql",
            "description": "stale legacy text",
            "agent_description": "current text the runtime uses",
            "tools": "db_tools.*",
            "created_at": "2026-04-30T09:20:31.545000Z",
        }

        svc = AgentService()
        result = await svc.get_agent("mixed_agent", real_agent_config)
        assert result.success is True
        assert result.data["agent"]["description"] == "current text the runtime uses"

    async def test_get_custom_agent_created_at_falls_back_to_file_mtime(
        self, real_agent_config, agent_yml_with_singleton
    ):
        """When yaml has no ``created_at``, fall back to the loaded config file's mtime."""
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

    async def test_create_agent_success(self, real_agent_config, agent_yml_with_singleton):
        """create_agent creates a new custom agent."""
        from datus.api.models.agent_models import CreateAgentInput

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

    async def test_create_agent_duplicate_name_fails(self, real_agent_config, agent_yml_with_singleton):
        """create_agent rejects duplicate agent name."""
        from datus.api.models.agent_models import CreateAgentInput

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

    async def test_create_agent_builtin_name_fails(self, real_agent_config, agent_yml_with_singleton):
        """create_agent rejects builtin agent names."""
        from datus.api.models.agent_models import CreateAgentInput

        svc = AgentService()
        result = await svc.create_agent(
            CreateAgentInput(name="gen_sql", type="gen_sql"),
            real_agent_config,
        )
        assert result.success is False
        assert result.errorCode == "AGENT_ALREADY_EXISTS"

    async def test_create_agent_persists_created_at(self, real_agent_config, agent_yml_with_singleton):
        """create_agent writes a UTC ISO-Z created_at into agent.yml."""
        import yaml

        from datus.api.models.agent_models import CreateAgentInput

        svc = AgentService()
        result = await svc.create_agent(
            CreateAgentInput(name="created_at_agent", type="gen_sql"),
            real_agent_config,
        )
        assert result.success is True

        with open(agent_yml_with_singleton) as f:
            raw = yaml.safe_load(f)
        # Production yaml wraps everything under ``agent:`` and
        # ConfigurationManager.save() round-trips that wrapping.
        entry = raw["agent"]["agentic_nodes"]["created_at_agent"]
        assert "created_at" in entry
        # Round-trip through the public API: the value surfaces unchanged
        get_result = await svc.get_agent("created_at_agent", real_agent_config)
        assert get_result.success is True
        created_at = get_result.data["agent"]["created_at"]
        # Either yaml stored a string (Z-suffixed) or a datetime that gets normalized
        assert created_at is not None and created_at.endswith("Z")

    async def test_create_agent_persists_description_as_agent_description(
        self, real_agent_config, agent_yml_with_singleton
    ):
        """API ``description`` lands on yaml's ``agent_description`` key.

        The runtime (sub_agent_task_tool, agentic_node, the wizard, and the
        saas config_loader) reads ``agent_description``. Persisting under
        the API field name ``description`` would be invisible to the
        runtime — this is the bug this PR fixes.
        """
        import yaml

        from datus.api.models.agent_models import CreateAgentInput

        svc = AgentService()
        result = await svc.create_agent(
            CreateAgentInput(name="desc_agent", type="gen_sql", description="hello"),
            real_agent_config,
        )
        assert result.success is True

        with open(agent_yml_with_singleton) as f:
            raw = yaml.safe_load(f)
        # ConfigurationManager.save() round-trips the production ``agent:``
        # wrapping, so the entry lives at agent.agentic_nodes.<name>.
        entry = raw["agent"]["agentic_nodes"]["desc_agent"]
        assert entry.get("agent_description") == "hello"
        # The raw API field name must NOT be persisted — that would be invisible
        # to the runtime.
        assert "description" not in entry

    async def test_create_agent_invalid_tools_fails(self, real_agent_config, agent_yml_with_singleton):
        """create_agent rejects invalid tool patterns."""
        from datus.api.models.agent_models import CreateAgentInput

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

    async def test_edit_agent_not_found(self, real_agent_config, agent_yml_with_singleton):
        """edit_agent returns error for nonexistent agent."""
        from datus.api.models.agent_models import EditAgentInput

        svc = AgentService()
        result = await svc.edit_agent(
            EditAgentInput(id="nonexistent_id", name="nonexistent_agent", description="updated"),
            real_agent_config,
        )
        assert result.success is False
        assert result.errorCode == "AGENT_NOT_FOUND"

    async def test_edit_agent_invalid_tools(self, real_agent_config, agent_yml_with_singleton):
        """edit_agent rejects invalid tool patterns."""
        from datus.api.models.agent_models import EditAgentInput

        svc = AgentService()
        result = await svc.edit_agent(
            EditAgentInput(id="some_id", name="some_agent", tools=["bad_tools.bad"]),
            real_agent_config,
        )
        assert result.success is False
        assert result.errorCode == "INVALID_TOOLS"

    async def test_edit_existing_agent(self, real_agent_config, agent_yml_with_singleton):
        """edit_agent updates existing custom agent."""
        from datus.api.models.agent_models import CreateAgentInput, EditAgentInput

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

    async def test_edit_with_prompt_template_writes_under_resolved_home(
        self, real_agent_config, agent_yml_with_singleton
    ):
        """``edit_agent`` with an explicit ``prompt_template`` invokes
        ``_save_prompt_template``, which must resolve ``agent_config.home``
        through ``path_manager`` so a literal ``~`` does not leak into the
        filesystem write.
        """
        from datus.api.models.agent_models import CreateAgentInput, EditAgentInput

        resolved_home = real_agent_config.path_manager.datus_home
        # Mutate ``agent_config.home`` post-construction to a tilde path —
        # path_manager remains pointed at resolved_home.
        real_agent_config.home = "~/datus-tilde-edit-template-does-not-exist"

        svc = AgentService()
        await svc.create_agent(
            CreateAgentInput(name="prompt_edit_agent", type="gen_sql"),
            real_agent_config,
        )
        edit = await svc.edit_agent(
            EditAgentInput(
                id="prompt_edit_agent",
                name="prompt_edit_agent",
                prompt_template="custom system prompt body",
                prompt_version="1.0",
            ),
            real_agent_config,
        )
        assert edit.success is True
        # Template file landed under the resolved home, not anywhere a
        # literal-tilde expansion would point.
        target = resolved_home / "template" / "prompt_edit_agent_system_1.0.j2"
        assert target.exists() and target.read_text(encoding="utf-8") == "custom system prompt body"

    async def test_template_copy_resolves_tilde_in_home(self, real_agent_config, agent_yml_with_singleton, tmp_path):
        """Regression: ``agent_config.home`` may carry a literal ``~`` (default
        ``~/.datus``). ``_copy_prompt_template`` (called by ``create_agent``)
        must route through ``path_manager.datus_home`` — which is already
        ``Path(home).expanduser().resolve()`` — instead of constructing
        ``Path(agent_config.home) / "template"`` directly.

        Pre-fix, ``Path("~/.datus")`` left the literal tilde in place and
        every subsequent ``os.makedirs`` / ``write_text`` either polluted the
        real home or crashed depending on the OS. With path_manager the
        template lands under the resolved tmp home regardless of how
        ``agent_config.home`` is shaped.
        """
        from datus.api.models.agent_models import CreateAgentInput

        # Override agent_config.home with a literal tilde path AFTER fixture
        # setup. path_manager remains pointed at the resolved tmp home, so
        # the call should resolve through that and succeed.
        resolved_home = real_agent_config.path_manager.datus_home
        real_agent_config.home = "~/datus-tilde-regression-does-not-exist"

        svc = AgentService()
        result = await svc.create_agent(
            CreateAgentInput(name="tilde_template_agent", type="gen_sql"),
            real_agent_config,
        )
        assert result.success is True

        # The template should land under the resolved home, never under a
        # literal tilde-prefixed directory next to CWD. CreateAgentInput
        # defaults prompt_version="1.0", so the file is suffixed accordingly.
        template_file = resolved_home / "template" / "tilde_template_agent_system_1.0.j2"
        assert template_file.exists(), f"template not found at {template_file}"
        # And no literal-tilde directory should have been created on disk.
        assert not (tmp_path / "~").exists()

    async def test_save_targets_loaded_config_path_not_home(self, real_agent_config, tmp_path):
        """Regression: ``--config /custom/path/agent.yml`` must persist to that
        same path, not to ``{datus_home}/agent.yml``.

        The original bug: a user starts the web UI with
        ``datus --web --config ana-docs/conf/agent.yml`` and saves an agent.
        The change landed in ``~/.datus/agent.yml`` instead of the
        ``ana-docs/conf/agent.yml`` they configured. This test reproduces
        that scenario by installing the ConfigurationManager singleton at
        a custom path outside ``datus_home`` and asserting writes follow
        the singleton's ``config_path``.
        """
        import yaml

        from datus.api.models.agent_models import CreateAgentInput
        from datus.configuration import agent_config_loader

        # The "real" config the user passed via --config — not under home.
        custom_dir = tmp_path / "ana-docs" / "conf"
        custom_dir.mkdir(parents=True)
        custom_yaml = custom_dir / "agent.yml"
        custom_yaml.write_text("agent: {}\n", encoding="utf-8")
        agent_config_loader.configuration_manager(config_path=str(custom_yaml), reload=True)
        try:
            svc = AgentService()
            result = await svc.create_agent(
                CreateAgentInput(name="cross_path_agent", type="gen_sql", description="x"),
                real_agent_config,
            )
            assert result.success is True

            # The custom yaml must contain the new agent under the production
            # ``agent.agentic_nodes`` shape.
            saved = yaml.safe_load(custom_yaml.read_text(encoding="utf-8"))
            assert saved["agent"]["agentic_nodes"]["cross_path_agent"]["type"] == "gen_sql"

            # The ``{datus_home}/agent.yml`` location must NOT have been written
            # — that was the bug. Read it (or treat it as empty when absent)
            # and assert the leak is absent under either possible yaml shape.
            home_yaml = real_agent_config.path_manager.datus_home / "agent.yml"
            home_data = yaml.safe_load(home_yaml.read_text(encoding="utf-8")) if home_yaml.exists() else {}
            home_data = home_data or {}
            assert "cross_path_agent" not in (home_data.get("agentic_nodes") or {})
            assert "cross_path_agent" not in (home_data.get("agent", {}).get("agentic_nodes") or {})
        finally:
            agent_config_loader.CONFIGURATION_MANAGER = None

    async def test_edit_agent_persists_description_as_agent_description(
        self, real_agent_config, agent_yml_with_singleton
    ):
        """edit_agent writes the API ``description`` field to ``agent_description``.

        Without this mapping, the editor's update never reaches the runtime
        (which only reads ``agent_description``).
        """
        import yaml

        from datus.api.models.agent_models import CreateAgentInput, EditAgentInput

        svc = AgentService()
        await svc.create_agent(CreateAgentInput(name="edit_desc", type="gen_sql"), real_agent_config)
        result = await svc.edit_agent(
            EditAgentInput(id="edit_desc", name="edit_desc", description="brand new text"),
            real_agent_config,
        )
        assert result.success is True

        with open(agent_yml_with_singleton) as f:
            raw = yaml.safe_load(f)
        entry = raw["agent"]["agentic_nodes"]["edit_desc"]
        assert entry.get("agent_description") == "brand new text"
        assert "description" not in entry

    async def test_edit_agent_clears_legacy_description_key_on_write(self, real_agent_config, agent_yml_with_singleton):
        """When yaml has both keys, edit_agent migrates by clearing the legacy one.

        Older API versions wrote ``description``. The new API writes
        ``agent_description``. If a user edits an entry that was created
        under the old code, both keys can coexist briefly. Writing the new
        key alone (and dropping the legacy one) prevents stale shadow data
        from confusing future reads or downstream tools.
        """
        import yaml

        from datus.api.models.agent_models import EditAgentInput

        # Seed an entry with the legacy ``description`` key only. Inject
        # directly into the in-memory dict so edit_agent finds it; the
        # singleton fixture handles the on-disk yaml.
        real_agent_config.agentic_nodes["legacy_edit"] = {
            "type": "gen_sql",
            "description": "old text from a previous version",
        }

        svc = AgentService()
        result = await svc.edit_agent(
            EditAgentInput(id="legacy_edit", name="legacy_edit", description="migrated text"),
            real_agent_config,
        )
        assert result.success is True

        with open(agent_yml_with_singleton) as f:
            raw = yaml.safe_load(f)
        entry = raw["agent"]["agentic_nodes"]["legacy_edit"]
        assert entry.get("agent_description") == "migrated text"
        # Legacy key must be cleared so downstream readers can't pick up
        # the old text.
        assert "description" not in entry


class TestFormatAndParseCsv:
    """Tests for _format_csv / _parse_csv — list ↔ comma-separated string."""

    def test_list_renders_with_separator(self):
        """A list is joined with ``", "`` so it matches the documented yaml form."""
        rendered = _format_csv(["semantic_tools.*", "db_tools.*", "context_search_tools.list_subject_tree"])
        assert rendered == "semantic_tools.*, db_tools.*, context_search_tools.list_subject_tree"

    def test_string_input_is_normalized(self):
        """A pre-formatted string is re-rendered with the canonical spacing."""
        assert _format_csv("a,  b , c") == "a, b, c"

    def test_empty_inputs_render_empty_string(self):
        """``None`` / empty list / empty string all collapse to ``""``."""
        assert _format_csv(None) == ""
        assert _format_csv([]) == ""
        assert _format_csv("") == ""

    def test_blank_entries_are_dropped(self):
        """Whitespace-only entries don't pollute the rendered string."""
        assert _format_csv(["a", "", "  ", "b"]) == "a, b"

    def test_round_trip_through_parse(self):
        """``_parse_csv(_format_csv(items)) == items`` for trimmed entries."""
        items = ["default_catalog.mart.mf_time_spine", "default_catalog.mart.raw_orders"]
        assert _parse_csv(_format_csv(items)) == items


class TestStripLeadingSlashes:
    """Tests for _strip_leading_slashes — defensive normalizer for catalog inputs.

    Catalogs in the API contract are dot-separated (e.g.
    ``default_catalog.mart.raw_orders``), so the strip is normally a
    no-op. The helper still runs for defensive normalization, and the
    cases here pin down its mechanical behavior on path-style strings.
    """

    def test_each_entry_loses_leading_slash(self):
        """A leading ``/`` is removed from every entry in the list."""
        assert _strip_leading_slashes(["/foo/bar", "/baz"]) == ["foo/bar", "baz"]

    def test_string_input_is_split_and_stripped(self):
        """A pre-formatted comma-separated string is split before stripping."""
        assert _strip_leading_slashes("/foo/bar, /baz") == ["foo/bar", "baz"]

    def test_no_leading_slash_passes_through(self):
        """Dot-form catalog entries (no leading slash) survive unchanged."""
        assert _strip_leading_slashes(["default_catalog.mart.mf_time_spine", "default_catalog.mart.raw_orders"]) == [
            "default_catalog.mart.mf_time_spine",
            "default_catalog.mart.raw_orders",
        ]

    def test_inner_slashes_are_kept(self):
        """Only the leading slash is trimmed — inner separators are preserved."""
        assert _strip_leading_slashes(["/foo/bar"]) == ["foo/bar"]

    def test_slash_only_entry_is_dropped(self):
        """A bare ``/`` collapses to nothing and is filtered out."""
        assert _strip_leading_slashes(["/", "/foo"]) == ["foo"]


class TestBuildScopedContext:
    """Tests for _build_scoped_context — fold catalogs/subject buckets into scoped_context."""

    def test_catalogs_writes_runtime_tables_key(self):
        """API ``catalogs`` lands on the runtime-honored ``tables`` key.

        ``ScopedContext`` has no ``catalogs`` field; the runtime's table-scope
        filter (``ScopedFilterBuilder.build_table_filter``) reads ``tables``,
        so the editor's ``catalogs`` array is the same scope under a
        different surface name. Catalog entries are dot-separated table
        identifiers (``catalog.schema.table``) consumed by the right-aligned
        token parser in ``ScopedFilterBuilder.build_table_filter``.
        """
        result = _build_scoped_context(
            None,
            catalogs=["default_catalog.mart.mf_time_spine", "default_catalog.mart.raw_orders"],
        )
        assert result == {"tables": "default_catalog.mart.mf_time_spine, default_catalog.mart.raw_orders"}

    def test_subject_buckets_write_runtime_keys(self):
        """``subject_buckets`` writes to ``metrics`` / ``sqls``.

        Subject paths are stored verbatim in dot-form; no leading-slash
        stripping happens since the API contract is dot-separated.
        """
        result = _build_scoped_context(
            None,
            catalogs=["default_catalog.mart.mf_time_spine"],
            subject_buckets={
                "metrics": ["Finance.Revenue.M1", "Sales.M2"],
                "sqls": ["Finance.SQL.s1"],
            },
        )
        # Empty buckets are omitted; subjects key never appears.
        assert result == {
            "tables": "default_catalog.mart.mf_time_spine",
            "metrics": "Finance.Revenue.M1, Sales.M2",
            "sqls": "Finance.SQL.s1",
        }

    def test_datasource_is_written_when_provided(self):
        """A non-empty datasource is recorded under ``scoped_context.datasource``."""
        result = _build_scoped_context(
            None,
            datasource="finance",
            catalogs=["default_catalog.mart.mf_time_spine"],
        )
        assert result == {
            "datasource": "finance",
            "tables": "default_catalog.mart.mf_time_spine",
        }

    def test_empty_datasource_clears_existing_binding(self):
        """An empty-string ``datasource`` removes a stale binding from ``base``."""
        base = {"datasource": "old_ds", "tables": "default_catalog.mart.raw_orders"}
        result = _build_scoped_context(base, datasource="")
        assert result == {"tables": "default_catalog.mart.raw_orders"}

    def test_none_datasource_preserves_existing_binding(self):
        """``datasource=None`` leaves any existing binding intact."""
        base = {"datasource": "keep_me"}
        result = _build_scoped_context(base, catalogs=["default_catalog.mart.mf_time_spine"])
        assert result == {"datasource": "keep_me", "tables": "default_catalog.mart.mf_time_spine"}

    def test_catalogs_overwrites_tables_and_drops_legacy_catalogs_key(self):
        """Catalogs fully rewrites ``tables`` and clears any non-runtime ``catalogs`` key.

        Earlier API versions wrote ``scoped_context.catalogs`` directly. Once
        the editor calls edit_agent again with catalogs the helper migrates
        them under ``tables`` so the read path can't see two competing copies.
        """
        base = {
            "tables": "default_catalog.mart.legacy_table",
            "catalogs": "stale_legacy_pattern.*",
            "metrics": "Finance.Revenue.daily_revenue",
        }
        result = _build_scoped_context(base, catalogs=["default_catalog.mart.raw_orders"])
        assert result == {
            "tables": "default_catalog.mart.raw_orders",
            "metrics": "Finance.Revenue.daily_revenue",
        }

    def test_subject_buckets_overwrite_existing_bucket_keys(self):
        """Passing subject_buckets fully rewrites the bucket keys.

        Empty bucket entries clear their key (rule: the editor's subjects array
        is the new full scope for the agent).
        """
        base = {
            "tables": "default_catalog.mart.raw_orders",
            "metrics": "Finance.Revenue.old_metric",
            "sqls": "Sales.stale_query",
        }
        result = _build_scoped_context(
            base,
            subject_buckets={
                "metrics": ["Finance.Revenue.daily_revenue"],
                "sqls": [],
            },
        )
        # tables survives; subject keys are rewritten and empty buckets clear.
        assert result == {
            "tables": "default_catalog.mart.raw_orders",
            "metrics": "Finance.Revenue.daily_revenue",
        }

    def test_explicit_empty_catalogs_clears_existing_tables(self):
        """Sending ``catalogs=[]`` removes the ``tables`` key from a base scoped_context."""
        base = {"tables": "default_catalog.mart.raw_orders", "metrics": "Finance.Revenue.daily_revenue"}
        result = _build_scoped_context(base, catalogs=[])
        assert result == {"metrics": "Finance.Revenue.daily_revenue"}

    def test_none_subject_buckets_preserves_existing_keys(self):
        """``subject_buckets=None`` leaves any existing metrics/sqls intact."""
        base = {"metrics": "Finance.Revenue.daily_revenue", "sqls": "Sales.region_query"}
        result = _build_scoped_context(
            base,
            catalogs=["default_catalog.mart.mf_time_spine"],
            subject_buckets=None,
        )
        assert result == {
            "metrics": "Finance.Revenue.daily_revenue",
            "sqls": "Sales.region_query",
            "tables": "default_catalog.mart.mf_time_spine",
        }

    def test_empty_result_returns_none(self):
        """When the merged dict ends up empty, return ``None`` so callers omit the block."""
        assert _build_scoped_context(None) is None
        assert _build_scoped_context({}, catalogs=[]) is None
        assert _build_scoped_context(None, subject_buckets={"metrics": [], "sqls": []}) is None


class TestMergeSubjectsFromScopedContext:
    """Tests for _merge_subjects_from_scoped_context — flatten subject buckets."""

    def test_subject_buckets_are_concatenated(self):
        """metrics + sqls concatenate into a single subjects list."""
        scoped = {
            "metrics": "Commerce.Orders.Avg, Sales.Region",
            "sqls": "finance.sql_a",
        }
        assert _merge_subjects_from_scoped_context(scoped) == [
            "Commerce.Orders.Avg",
            "Sales.Region",
            "finance.sql_a",
        ]

    def test_stored_entries_are_returned_verbatim(self):
        """Stored dot-form entries are surfaced unchanged — no path rewriting on read."""
        scoped = {"metrics": "finance.revenue.daily, sales.region"}
        assert _merge_subjects_from_scoped_context(scoped) == [
            "finance.revenue.daily",
            "sales.region",
        ]

    def test_duplicates_across_buckets_are_dropped(self):
        """A path that lands in multiple buckets only appears once in subjects."""
        scoped = {"metrics": "shared", "sqls": "shared"}
        assert _merge_subjects_from_scoped_context(scoped) == ["shared"]

    def test_missing_or_empty_buckets_yield_empty_list(self):
        """Empty / non-dict inputs return an empty list."""
        assert _merge_subjects_from_scoped_context(None) == []
        assert _merge_subjects_from_scoped_context({}) == []
        assert _merge_subjects_from_scoped_context({"tables": "t1"}) == []


class TestClassifySubjectPaths:
    """Tests for _classify_subject_paths — bucket subjects into metrics/sqls."""

    def test_no_datasource_falls_back_to_metrics(self, real_agent_config):
        """When the AgentConfig has no datasource bound, every subject defaults to metrics.

        The fallback exists so the editor's input survives a save even when the
        project hasn't bootstrapped its KB yet — losing the user's selection
        silently would be the worse failure mode.
        """
        real_agent_config.current_datasource = ""
        result = _classify_subject_paths(real_agent_config, ["Commerce.Orders.Avg", "Sales.Region"])
        assert result == {
            "metrics": ["Commerce.Orders.Avg", "Sales.Region"],
            "sqls": [],
        }

    def test_empty_input_returns_empty_buckets(self, real_agent_config):
        """An empty subjects list yields an empty bucket dict, not an error."""
        result = _classify_subject_paths(real_agent_config, [])
        assert result == {"metrics": [], "sqls": []}

    def test_storage_init_failure_falls_back_to_metrics(self, real_agent_config, monkeypatch):
        """If the metric / sql stores can't initialize, all subjects bucket as metrics.

        Forcing a storage-init exception (here via a broken ``MetricRAG.__init__``)
        exercises the defensive fallback so the API endpoint never raises a 500
        on a save that the user can otherwise complete.
        """
        from datus.storage.metric.store import MetricRAG

        def broken_init(self, *args, **kwargs):
            raise RuntimeError("storage backend down")

        monkeypatch.setattr(MetricRAG, "__init__", broken_init)
        result = _classify_subject_paths(real_agent_config, ["Commerce.Orders.Avg"])
        assert result["metrics"] == ["Commerce.Orders.Avg"]
        assert result["sqls"] == []

    def test_classifies_via_storage_lookup(self, real_agent_config, monkeypatch):
        """Each path is bucketed by the first store whose ``list_entries`` matches the name.

        Stubbing storages forces a deterministic classification that
        doesn't depend on the test fixture pre-populating real KB data; the
        probe order (metrics → sqls) is part of the contract.
        """
        from datus.storage.metric.store import MetricRAG
        from datus.storage.reference_sql.store import ReferenceSqlRAG

        class _StubStore:
            def __init__(self, owns: set[str]):
                self._owns = owns

            def list_entries(self, node_id, name=None, limit=None):
                return [{"name": name}] if name in self._owns else []

        class _StubTree:
            def get_node_by_path(self, path):
                # Return a stable node_id regardless of path so the storages
                # decide ownership purely by name.
                return {"node_id": 1}

        def fake_metric_init(self, *args, **kwargs):
            self.storage = _StubStore({"my_metric"})

        def fake_sql_init(self, *args, **kwargs):
            self.reference_sql_storage = _StubStore({"my_sql"})

        monkeypatch.setattr(MetricRAG, "__init__", fake_metric_init)
        monkeypatch.setattr(ReferenceSqlRAG, "__init__", fake_sql_init)
        monkeypatch.setattr(
            "datus.storage.registry.get_subject_tree_store",
            lambda project, datasource_id="": _StubTree(),
        )

        result = _classify_subject_paths(
            real_agent_config,
            [
                "Commerce.Orders.my_metric",
                "Finance.my_sql",
                "Unknown.path",
            ],
        )
        # Unknown paths fall back to metrics so the editor's input survives the
        # round-trip — losing the user's selection silently would be worse.
        assert result["metrics"] == ["Commerce.Orders.my_metric", "Unknown.path"]
        assert result["sqls"] == ["Finance.my_sql"]


@pytest.mark.asyncio
class TestSubagentScopedContextRoundTrip:
    """End-to-end checks for the create/edit/get pipeline.

    The contract documented in
    ``docs/subagent/customized_subagent.zh.md``: ``tools`` is persisted as a
    comma-separated string (the runtime calls ``str.split(",")``), and
    ``catalogs`` / ``subjects`` live under ``scoped_context`` so a single
    block describes the subagent's reference scope.
    """

    async def test_create_persists_tools_as_csv_string(self, real_agent_config, agent_yml_with_singleton):
        """Tools list is rendered as the comma-separated yaml form on disk."""
        import yaml

        from datus.api.models.agent_models import CreateAgentInput

        svc = AgentService()
        result = await svc.create_agent(
            CreateAgentInput(
                name="csv_tools_agent",
                type="gen_sql",
                tools=["semantic_tools.*", "db_tools.*", "context_search_tools.list_subject_tree"],
            ),
            real_agent_config,
        )
        assert result.success is True

        with open(agent_yml_with_singleton) as f:
            raw = yaml.safe_load(f)
        entry = raw["agent"]["agentic_nodes"]["csv_tools_agent"]
        assert entry["tools"] == "semantic_tools.*, db_tools.*, context_search_tools.list_subject_tree"

    async def test_create_folds_catalogs_into_scoped_context_and_classifies_subjects(
        self, real_agent_config, agent_yml_with_singleton
    ):
        """API top-level catalogs/subjects land inside ``scoped_context`` on save.

        ``catalogs`` writes through to the runtime-honored ``tables`` key (no
        ``catalogs`` key persists, since ``ScopedContext`` doesn't define one).
        ``subjects`` is *classified* into metrics / sqls — no
        flat ``subjects`` key ever appears on disk because each store owns
        its own scope filter. Without pre-populated KB stores the classifier
        falls back to metrics. ``scoped_context.datasource`` is bound to the
        active datasource so ``SubAgentConfig.is_in_datasource`` agrees.
        """
        import yaml

        from datus.api.models.agent_models import CreateAgentInput

        svc = AgentService()
        result = await svc.create_agent(
            CreateAgentInput(
                name="scoped_create_agent",
                type="gen_sql",
                catalogs=["default_catalog.mart.mf_time_spine", "default_catalog.mart.raw_orders"],
                subjects=["Finance.Revenue.Daily"],
            ),
            real_agent_config,
        )
        assert result.success is True

        with open(agent_yml_with_singleton) as f:
            raw = yaml.safe_load(f)
        entry = raw["agent"]["agentic_nodes"]["scoped_create_agent"]
        # catalogs / subjects must NOT remain at the top level — that was the
        # legacy shape before this change.
        assert "catalogs" not in entry
        assert "subjects" not in entry
        scoped = entry.get("scoped_context")
        assert isinstance(scoped, dict)
        # Active datasource binding is recorded.
        assert scoped["datasource"] == real_agent_config.current_datasource
        # No non-runtime ``catalogs`` key — catalogs write through to ``tables``.
        assert "catalogs" not in scoped
        assert scoped["tables"] == "default_catalog.mart.mf_time_spine, default_catalog.mart.raw_orders"
        # Subjects route to ``metrics`` (the fallback bucket) because the
        # KB stores have no entry named "Daily" under "Finance.Revenue".
        # No flat ``subjects`` key is persisted.
        assert "subjects" not in scoped
        assert scoped["metrics"] == "Finance.Revenue.Daily"

    async def test_edit_migrates_tools_and_scoped_context(self, real_agent_config, agent_yml_with_singleton):
        """An edit normalizes tools to CSV and rewrites scoped_context.

        Legacy-shape entries (list-form tools, top-level catalogs/subjects)
        get migrated: tools become a CSV string, catalogs lands in
        ``scoped_context.catalogs``, and subjects are classified into the
        runtime bucket keys (no flat ``subjects`` key persists).
        """
        import yaml

        from datus.api.models.agent_models import EditAgentInput

        # Seed a legacy-shape entry: list-form tools and top-level catalogs/subjects.
        real_agent_config.agentic_nodes["legacy_scope"] = {
            "type": "gen_sql",
            "tools": ["db_tools.*"],
            "catalogs": ["default_catalog.mart.legacy_table"],
            "subjects": ["Legacy.Subject"],
        }

        svc = AgentService()
        result = await svc.edit_agent(
            EditAgentInput(
                id="legacy_scope",
                name="legacy_scope",
                tools=["semantic_tools.*", "db_tools.*"],
                catalogs=["default_catalog.mart.raw_orders"],
                subjects=["Sales.Region"],
            ),
            real_agent_config,
        )
        assert result.success is True

        with open(agent_yml_with_singleton) as f:
            raw = yaml.safe_load(f)
        entry = raw["agent"]["agentic_nodes"]["legacy_scope"]
        assert entry["tools"] == "semantic_tools.*, db_tools.*"
        # Top-level legacy keys must be cleared so the read path can't see two copies.
        assert "catalogs" not in entry
        assert "subjects" not in entry
        scoped = entry["scoped_context"]
        # Catalogs lands on ``tables`` — no non-runtime ``catalogs`` key persists.
        assert "catalogs" not in scoped
        assert scoped["tables"] == "default_catalog.mart.raw_orders"
        # No flat ``subjects`` key — subjects are classified into runtime buckets.
        assert "subjects" not in scoped
        # Without pre-populated KB stores the classifier defaults to metrics.
        assert scoped["metrics"] == "Sales.Region"
        # The active datasource is rebound on every scope-touching edit.
        assert scoped["datasource"] == real_agent_config.current_datasource

    async def test_edit_preserves_existing_scoped_context_keys(self, real_agent_config, agent_yml_with_singleton):
        """Editing catalogs rewrites ``tables`` but leaves metrics/sqls intact.

        Catalogs maps to the runtime ``tables`` key, so an edit that touches
        catalogs is expected to overwrite that field. The other scope keys
        (``metrics`` / ``sqls``) must survive when
        ``subjects`` is not part of the request.
        """
        import yaml

        from datus.api.models.agent_models import EditAgentInput

        real_agent_config.agentic_nodes["preserve_scope"] = {
            "type": "gen_sql",
            "scoped_context": {
                "tables": "default_catalog.mart.legacy_table",
                "metrics": "Finance.Revenue.daily_revenue",
                "sqls": "Finance.Revenue.region_rollup",
            },
        }

        svc = AgentService()
        result = await svc.edit_agent(
            EditAgentInput(
                id="preserve_scope",
                name="preserve_scope",
                catalogs=["default_catalog.mart.raw_orders"],
            ),
            real_agent_config,
        )
        assert result.success is True

        with open(agent_yml_with_singleton) as f:
            raw = yaml.safe_load(f)
        scoped = raw["agent"]["agentic_nodes"]["preserve_scope"]["scoped_context"]
        # ``tables`` was rewritten by catalogs — there's no separate
        # ``catalogs`` key in ``ScopedContext``. Other scope fields survive.
        assert scoped["tables"] == "default_catalog.mart.raw_orders"
        assert scoped["metrics"] == "Finance.Revenue.daily_revenue"
        assert scoped["sqls"] == "Finance.Revenue.region_rollup"

    async def test_get_returns_tools_as_list_and_extracts_scoped_paths(self, real_agent_config):
        """The read path inverts the storage format the editor expects.

        Catalog entries are recovered from the runtime ``tables`` key; subjects
        are recomposed by merging the runtime buckets (metrics / sqls) into a single flat list. Stored entries surface in
        their canonical dot-separated form unchanged.
        """
        real_agent_config.agentic_nodes["roundtrip_get"] = {
            "type": "gen_sql",
            "tools": "semantic_tools.*, db_tools.*, context_search_tools.list_subject_tree",
            "scoped_context": {
                "datasource": "finance",
                "tables": "default_catalog.mart.mf_time_spine, default_catalog.mart.raw_orders",
                "metrics": "Finance.Revenue.Daily, finance.revenue.weekly",
                "sqls": "Sales.region_query",
            },
            "created_at": "2026-04-30T09:20:31.545000Z",
        }

        svc = AgentService()
        result = await svc.get_agent("roundtrip_get", real_agent_config)
        assert result.success is True
        agent = result.data["agent"]
        assert agent["tools"] == [
            "semantic_tools.*",
            "db_tools.*",
            "context_search_tools.list_subject_tree",
        ]
        # Catalogs come back from ``scoped_context.tables`` (the runtime key)
        # in their canonical dot-separated form.
        assert agent["catalogs"] == [
            "default_catalog.mart.mf_time_spine",
            "default_catalog.mart.raw_orders",
        ]
        # Subjects merge across all three buckets in probe order; entries
        # surface verbatim in their stored dot-form.
        assert agent["subjects"] == [
            "Finance.Revenue.Daily",
            "finance.revenue.weekly",
            "Sales.region_query",
        ]

    async def test_round_trip_create_then_get(self, real_agent_config, agent_yml_with_singleton):
        """Saving and re-reading yields the canonical dot-separated form.

        Even though the on-disk shape changes (subjects gets split into
        ``metrics`` / ``sqls``), the API contract round-
        trips: the editor sees ``subjects`` and ``catalogs`` come back as
        the same flat lists it sent.
        """
        from datus.api.models.agent_models import CreateAgentInput

        svc = AgentService()
        await svc.create_agent(
            CreateAgentInput(
                name="full_round_trip",
                type="gen_sql",
                tools=["semantic_tools.*", "db_tools.*"],
                catalogs=["default_catalog.mart.raw_orders"],
                subjects=["Commerce.Orders.Average_Order_Value.average_gross_order_value"],
            ),
            real_agent_config,
        )

        result = await svc.get_agent("full_round_trip", real_agent_config)
        assert result.success is True
        agent = result.data["agent"]
        assert agent["tools"] == ["semantic_tools.*", "db_tools.*"]
        assert agent["catalogs"] == ["default_catalog.mart.raw_orders"]
        assert agent["subjects"] == ["Commerce.Orders.Average_Order_Value.average_gross_order_value"]

    async def test_edit_clearing_scope_persists_to_yaml(self, real_agent_config, agent_yml_with_singleton):
        """Clearing the entire scope must reach disk, not just the in-memory dict.

        When the only thing the user changes is to wipe their scope
        (``catalogs=[]`` and ``subjects=[]``), ``_build_scoped_context``
        returns ``None`` and ``edit_agent`` removes ``scoped_context`` from
        the live agent dict. The subsequent ``not update_data`` short-circuit
        used to skip ``_save_agentic_nodes``, so the deletion was lost on
        the next config reload — this test pins the fix in place.
        """
        import yaml

        from datus.api.models.agent_models import EditAgentInput

        # Seed an entry that already has a scope on disk. ``datasource`` is
        # intentionally omitted so clearing catalogs/subjects fully empties
        # the scoped_context dict — that's the path where merged is ``None``
        # and the early-return previously skipped the save.
        real_agent_config.agentic_nodes["clear_scope_agent"] = {
            "type": "gen_sql",
            "scoped_context": {
                "tables": "default_catalog.mart.raw_orders",
                "metrics": "Finance.Revenue.Daily",
            },
        }
        agent_yml_with_singleton.write_text(
            yaml.safe_dump(
                {
                    "agent": {
                        "agentic_nodes": {
                            "clear_scope_agent": real_agent_config.agentic_nodes["clear_scope_agent"],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        # Reset the active datasource so the helper doesn't re-bind a fresh
        # datasource value into scoped_context — that would leave the dict
        # non-empty and avoid the bug entirely.
        real_agent_config.current_datasource = ""

        svc = AgentService()
        result = await svc.edit_agent(
            EditAgentInput(
                id="clear_scope_agent",
                name="clear_scope_agent",
                catalogs=[],
                subjects=[],
            ),
            real_agent_config,
        )
        assert result.success is True

        with open(agent_yml_with_singleton) as f:
            raw = yaml.safe_load(f)
        entry = raw["agent"]["agentic_nodes"]["clear_scope_agent"]
        # The whole scoped_context block is gone on disk, not just in memory.
        assert "scoped_context" not in entry

    async def test_edit_classifies_subjects_under_saved_datasource(
        self, real_agent_config, agent_yml_with_singleton, monkeypatch
    ):
        """Classification uses the saved DS binding when ``current_datasource`` is unset.

        Without the fix, ``_classify_subject_paths`` ran with
        ``datasource_id=None`` and fell back to ``agent_config.current_datasource``
        — which is empty in this scenario — so every subject was bucketed to
        ``metrics`` regardless of which store actually owned it. With the fix,
        ``edit_agent`` resolves the effective datasource from the saved
        ``scoped_context.datasource`` first, so the SQL store is actually
        probed and ownership wins.
        """
        from datus.api.models.agent_models import EditAgentInput
        from datus.storage.metric.store import MetricRAG
        from datus.storage.reference_sql.store import ReferenceSqlRAG

        # Capture which datasource_id flows into the classifier so we can
        # assert against the resolution rule directly.
        captured: dict = {}

        class _StubStore:
            def __init__(self, owns: set[str]):
                self._owns = owns

            def list_entries(self, node_id, name=None, limit=None):
                return [{"name": name}] if name in self._owns else []

        class _StubTree:
            def get_node_by_path(self, path):
                return {"node_id": 1}

        def fake_metric_init(self, agent_config, datasource_id=None):
            captured["metric_ds"] = datasource_id
            self.storage = _StubStore({"my_metric"})

        def fake_sql_init(self, agent_config, datasource_id=None):
            self.reference_sql_storage = _StubStore({"my_sql"})

        monkeypatch.setattr(MetricRAG, "__init__", fake_metric_init)
        monkeypatch.setattr(ReferenceSqlRAG, "__init__", fake_sql_init)
        monkeypatch.setattr(
            "datus.storage.registry.get_subject_tree_store",
            lambda project, datasource_id="": _StubTree(),
        )

        # Seed an entry already bound to "finance" via scoped_context, then
        # blank out the runtime's current_datasource so the only available
        # binding is the saved one.
        real_agent_config.agentic_nodes["edit_with_saved_ds"] = {
            "type": "gen_sql",
            "scoped_context": {
                "datasource": "finance",
                "tables": "default_catalog.mart.raw_orders",
            },
        }
        real_agent_config.current_datasource = ""

        svc = AgentService()
        result = await svc.edit_agent(
            EditAgentInput(
                id="edit_with_saved_ds",
                name="edit_with_saved_ds",
                subjects=["Finance.SQL.my_sql", "Sales.unknown"],
            ),
            real_agent_config,
        )
        assert result.success is True
        # The classifier must have been invoked with the saved datasource —
        # without the fix it received None and fell back to the
        # "no datasource → all metrics" branch.
        assert captured.get("metric_ds") == "finance"

        # SQL entries land in their owning bucket; only the truly unmatched
        # name falls back to metrics.
        scoped = real_agent_config.agentic_nodes["edit_with_saved_ds"]["scoped_context"]
        assert scoped["sqls"] == "Finance.SQL.my_sql"
        assert scoped["metrics"] == "Sales.unknown"
        # Saved datasource binding survives the edit.
        assert scoped["datasource"] == "finance"


@pytest.mark.asyncio
class TestDeleteAgent:
    """Tests for delete_agent — agent removal from agent.yml."""

    async def test_delete_agent_removes_entry_and_persists(self, real_agent_config, agent_yml_with_singleton):
        """delete_agent removes the agentic_nodes entry and writes the yaml back."""
        import yaml

        from datus.api.models.agent_models import CreateAgentInput

        svc = AgentService()
        await svc.create_agent(
            CreateAgentInput(name="to_delete", type="gen_sql", description="goodbye"),
            real_agent_config,
        )

        # Sanity: entry is present before delete.
        with open(agent_yml_with_singleton) as f:
            before = yaml.safe_load(f)
        assert "to_delete" in before["agent"]["agentic_nodes"]

        result = await svc.delete_agent("to_delete", real_agent_config)
        assert result.success is True
        assert result.data == {"id": "to_delete", "name": "to_delete"}

        # In-memory map is updated and the rewritten yaml no longer has the entry.
        assert "to_delete" not in (real_agent_config.agentic_nodes or {})
        with open(agent_yml_with_singleton) as f:
            after = yaml.safe_load(f)
        assert "to_delete" not in (after["agent"].get("agentic_nodes") or {})

        # Subsequent get returns AGENT_NOT_FOUND.
        get_result = await svc.get_agent("to_delete", real_agent_config)
        assert get_result.success is False
        assert get_result.errorCode == "AGENT_NOT_FOUND"

    async def test_delete_agent_not_found(self, real_agent_config, agent_yml_with_singleton):
        """delete_agent returns AGENT_NOT_FOUND for unknown ids."""
        svc = AgentService()
        result = await svc.delete_agent("never_existed", real_agent_config)
        assert result.success is False
        assert result.errorCode == "AGENT_NOT_FOUND"

    async def test_delete_agent_rejects_builtin(self, real_agent_config, agent_yml_with_singleton):
        """Builtin sub-agents cannot be deleted via this API."""
        svc = AgentService()
        builtin_name = next(iter(BUILTIN_SUBAGENTS))
        result = await svc.delete_agent(builtin_name, real_agent_config)
        assert result.success is False
        assert result.errorCode == "BUILTIN_AGENT_IMMUTABLE"

    async def test_delete_agent_cleans_prompt_templates(self, real_agent_config, agent_yml_with_singleton):
        """delete_agent best-effort removes ``<name>_system_*.j2`` files."""
        from datus.api.models.agent_models import CreateAgentInput

        svc = AgentService()
        await svc.create_agent(
            CreateAgentInput(name="template_owner", type="gen_sql", prompt_version="1.0"),
            real_agent_config,
        )

        template_dir = real_agent_config.path_manager.datus_home / "template"
        seeded = template_dir / "template_owner_system_1.0.j2"
        assert seeded.exists(), "create_agent should have seeded a template file"

        # Drop in a second version to prove the glob clears all versions.
        extra = template_dir / "template_owner_system_2.0.j2"
        extra.write_text("v2 content", encoding="utf-8")

        # And a template owned by a different agent must NOT be touched.
        unrelated = template_dir / "other_owner_system_1.0.j2"
        unrelated.write_text("unrelated", encoding="utf-8")

        result = await svc.delete_agent("template_owner", real_agent_config)
        assert result.success is True
        assert not seeded.exists()
        assert not extra.exists()
        assert unrelated.exists()

    async def test_delete_agent_template_cleanup_does_not_expand_globs(
        self, real_agent_config, agent_yml_with_singleton
    ):
        """Regression: agent names with glob metacharacters must not be
        passed through to ``Path.glob`` during template cleanup.

        ``_sanitize_path_component`` only strips path separators, so a name
        like ``victim*`` would survive sanitization and — if cleanup used
        ``glob(f"{safe_name}_system_*.j2")`` — sweep every file whose stem
        starts with ``victim``. The literal-match cleanup must compare
        ``startswith(f"{safe_name}_system_")`` so only files that literally
        contain the asterisk in their name are removed.
        """
        agentic_nodes = real_agent_config.agentic_nodes or {}
        agentic_nodes["victim*"] = {"type": "gen_sql"}
        real_agent_config.agentic_nodes = agentic_nodes

        template_dir = real_agent_config.path_manager.datus_home / "template"
        template_dir.mkdir(parents=True, exist_ok=True)

        # Sibling owned by a different agent — its name happens to start with
        # the literal prefix the attacker's glob would have expanded to.
        innocent = template_dir / "victim_system_1.0.j2"
        innocent.write_text("innocent", encoding="utf-8")

        # File whose literal on-disk name matches the attacker's sanitized
        # agent name — the only file that should ever be removed.
        owned = template_dir / "victim*_system_1.0.j2"
        owned.write_text("owned", encoding="utf-8")

        svc = AgentService()
        result = await svc.delete_agent("victim*", real_agent_config)

        assert result.success is True
        assert not owned.exists()
        assert innocent.exists()

    async def test_delete_agent_succeeds_when_template_dir_missing(self, real_agent_config, agent_yml_with_singleton):
        """delete_agent succeeds when the project never created any template
        files — ``_delete_prompt_templates`` hits the ``not is_dir()`` early
        return instead of iterating a missing directory.
        """
        agentic_nodes = real_agent_config.agentic_nodes or {}
        agentic_nodes["no_templates"] = {"type": "gen_sql"}
        real_agent_config.agentic_nodes = agentic_nodes

        # Ensure the template dir genuinely doesn't exist on disk.
        template_dir = real_agent_config.path_manager.datus_home / "template"
        if template_dir.exists():
            for path in template_dir.iterdir():
                if path.is_file():
                    path.unlink()
            template_dir.rmdir()
        assert not template_dir.exists()

        svc = AgentService()
        result = await svc.delete_agent("no_templates", real_agent_config)
        assert result.success is True
        assert "no_templates" not in (real_agent_config.agentic_nodes or {})

    async def test_delete_agent_template_cleanup_skips_directories(self, real_agent_config, agent_yml_with_singleton):
        """Subdirectories under ``template/`` are ignored by the cleanup scan.

        A user can manually drop a directory whose name happens to match the
        ``<safe_name>_system_*`` prefix (e.g., a checkpoint dir). The scan
        must skip non-files so it never tries to ``unlink`` a directory and
        crash the delete.
        """
        agentic_nodes = real_agent_config.agentic_nodes or {}
        agentic_nodes["dir_owner"] = {"type": "gen_sql"}
        real_agent_config.agentic_nodes = agentic_nodes

        template_dir = real_agent_config.path_manager.datus_home / "template"
        template_dir.mkdir(parents=True, exist_ok=True)

        # Real owned template that must be removed.
        owned = template_dir / "dir_owner_system_1.0.j2"
        owned.write_text("owned", encoding="utf-8")

        # Subdirectory whose name matches the prefix — must NOT be unlinked.
        nested = template_dir / "dir_owner_system_extras.j2"
        nested.mkdir()
        (nested / "inside.txt").write_text("keep me", encoding="utf-8")

        svc = AgentService()
        result = await svc.delete_agent("dir_owner", real_agent_config)
        assert result.success is True
        assert not owned.exists()
        # The subdirectory and its contents survived.
        assert nested.is_dir()
        assert (nested / "inside.txt").exists()

    async def test_delete_agent_template_unlink_failure_is_non_fatal(
        self, real_agent_config, agent_yml_with_singleton, monkeypatch
    ):
        """``Path.unlink`` raising ``OSError`` (e.g. read-only fs, permission
        denied) must not block the delete — the warning is logged but the
        agent still leaves the yaml.
        """
        from datus.api.models.agent_models import CreateAgentInput

        svc = AgentService()
        await svc.create_agent(
            CreateAgentInput(name="unlink_fails", type="gen_sql", prompt_version="1.0"),
            real_agent_config,
        )

        template_dir = real_agent_config.path_manager.datus_home / "template"
        seeded = template_dir / "unlink_fails_system_1.0.j2"
        assert seeded.exists()

        original_unlink = Path.unlink

        def _raise_for_owned(self, *args, **kwargs):
            # Only fail for the file we expect cleanup to touch — leave
            # unrelated unlinks (test teardown, other fixtures) alone.
            if self.name.startswith("unlink_fails_system_"):
                raise OSError("simulated permission denied")
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", _raise_for_owned)

        result = await svc.delete_agent("unlink_fails", real_agent_config)
        # The agent is gone from yaml even though template cleanup failed.
        assert result.success is True
        assert "unlink_fails" not in (real_agent_config.agentic_nodes or {})
        # The template file is still on disk because unlink was patched to fail.
        assert seeded.exists()

    async def test_delete_agent_swallows_template_cleanup_exception(
        self, real_agent_config, agent_yml_with_singleton, monkeypatch
    ):
        """If ``_delete_prompt_templates`` itself raises an unexpected
        exception (e.g. ``OSError`` on ``iterdir`` for a corrupted dir),
        ``delete_agent`` logs a warning and still returns success — the
        yaml write is the source of truth for whether the agent exists.
        """
        agentic_nodes = real_agent_config.agentic_nodes or {}
        agentic_nodes["cleanup_explodes"] = {"type": "gen_sql"}
        real_agent_config.agentic_nodes = agentic_nodes

        svc = AgentService()

        def _boom(self_inner, agent_name, agent_config):
            raise RuntimeError("simulated cleanup failure")

        monkeypatch.setattr(AgentService, "_delete_prompt_templates", _boom)

        result = await svc.delete_agent("cleanup_explodes", real_agent_config)
        assert result.success is True
        assert "cleanup_explodes" not in (real_agent_config.agentic_nodes or {})


@pytest.mark.asyncio
class TestDeleteAgentRoute:
    """Tests for the ``DELETE /api/v1/agent/delete`` route wrapper."""

    async def test_route_delegates_to_service(self, real_agent_config, agent_yml_with_singleton):
        """The route function instantiates AgentService and forwards the
        ``agent_id`` query parameter and the service's ``agent_config`` from
        the injected ``svc`` dependency.
        """
        from types import SimpleNamespace

        from datus.api.models.agent_models import CreateAgentInput
        from datus.api.routes.agent_routes import delete_agent as delete_agent_route

        svc = AgentService()
        await svc.create_agent(
            CreateAgentInput(name="route_target", type="gen_sql"),
            real_agent_config,
        )
        assert "route_target" in (real_agent_config.agentic_nodes or {})

        result = await delete_agent_route(
            svc=SimpleNamespace(agent_config=real_agent_config),
            agent_id="route_target",
        )
        assert result.success is True
        assert result.data == {"id": "route_target", "name": "route_target"}
        assert "route_target" not in (real_agent_config.agentic_nodes or {})

    async def test_route_returns_not_found_result(self, real_agent_config, agent_yml_with_singleton):
        """Route surfaces the service's AGENT_NOT_FOUND result verbatim."""
        from types import SimpleNamespace

        from datus.api.routes.agent_routes import delete_agent as delete_agent_route

        result = await delete_agent_route(
            svc=SimpleNamespace(agent_config=real_agent_config),
            agent_id="never_existed",
        )
        assert result.success is False
        assert result.errorCode == "AGENT_NOT_FOUND"


@pytest.mark.asyncio
class TestEditAgentChannels:
    """Tests for edit_agent / delete_agent IM gateway channel persistence."""

    async def _create(self, svc, real_agent_config, name):
        from datus.api.models.agent_models import CreateAgentInput

        await svc.create_agent(CreateAgentInput(name=name, type="gen_sql"), real_agent_config)

    def _edit_with_channels(self, agent_id, channels):
        from datus.api.models.agent_models import ChannelBinding, EditAgentInput

        return EditAgentInput(id=agent_id, channels=ChannelBinding(channels=channels))

    def _slack(self, name="slack-main", enabled=True, secrets=None):
        from datus.api.models.agent_models import ChannelInput

        return ChannelInput(type="slack", name=name, enabled=enabled, secrets=secrets or {})

    async def test_edit_agent_adds_channel(self, real_agent_config, agent_yml_with_singleton):
        """A channel binding lands in the global ``channels`` map and persists."""
        from datus.configuration import agent_config_loader

        svc = AgentService()
        await self._create(svc, real_agent_config, "chan_agent")

        result = await svc.edit_agent(
            self._edit_with_channels(
                "chan_agent",
                [self._slack(secrets={"app_token": "${SLACK_APP_TOKEN}", "bot_token": "${SLACK_BOT_TOKEN}"})],
            ),
            real_agent_config,
        )
        assert result.success is True

        # In-memory config updated.
        entry = real_agent_config.channels_config["slack-main"]
        assert entry == {
            "adapter": "slack",
            "enabled": True,
            "subagent_id": "chan_agent",
            "extra": {"app_token": "${SLACK_APP_TOKEN}", "bot_token": "${SLACK_BOT_TOKEN}"},
        }
        # Persisted to agent.yml under the ``channels`` section.
        persisted = agent_config_loader.configuration_manager().data["channels"]["slack-main"]
        assert persisted["subagent_id"] == "chan_agent"
        # Secrets written verbatim — no encryption.
        assert persisted["extra"]["app_token"] == "${SLACK_APP_TOKEN}"

    async def test_edit_agent_disabled_channel_kept(self, real_agent_config, agent_yml_with_singleton):
        """A disabled channel is still persisted (gateway honours `enabled`)."""
        svc = AgentService()
        await self._create(svc, real_agent_config, "chan_agent")

        result = await svc.edit_agent(
            self._edit_with_channels("chan_agent", [self._slack(enabled=False, secrets={"app_token": "x"})]),
            real_agent_config,
        )
        assert result.success is True
        assert real_agent_config.channels_config["slack-main"]["enabled"] is False

    async def test_edit_agent_empty_list_clears_this_agents_channels(self, real_agent_config, agent_yml_with_singleton):
        """An empty channel list removes this agent's entries."""
        svc = AgentService()
        await self._create(svc, real_agent_config, "chan_agent")
        await svc.edit_agent(
            self._edit_with_channels("chan_agent", [self._slack(secrets={"app_token": "x"})]),
            real_agent_config,
        )
        assert "slack-main" in real_agent_config.channels_config

        result = await svc.edit_agent(self._edit_with_channels("chan_agent", []), real_agent_config)
        assert result.success is True
        assert "slack-main" not in real_agent_config.channels_config

    async def test_edit_agent_channels_scoped_per_agent(self, real_agent_config, agent_yml_with_singleton):
        """Editing one agent's channels leaves another agent's channels intact."""
        svc = AgentService()
        await self._create(svc, real_agent_config, "agent_a")
        await self._create(svc, real_agent_config, "agent_b")
        await svc.edit_agent(
            self._edit_with_channels("agent_a", [self._slack(name="slack-a", secrets={"app_token": "a"})]),
            real_agent_config,
        )
        await svc.edit_agent(
            self._edit_with_channels("agent_b", [self._slack(name="slack-b", secrets={"app_token": "b"})]),
            real_agent_config,
        )

        # Re-edit agent_a — agent_b's channel must survive.
        await svc.edit_agent(
            self._edit_with_channels("agent_a", [self._slack(name="slack-a2", secrets={"app_token": "a2"})]),
            real_agent_config,
        )
        channels = real_agent_config.channels_config
        assert "slack-b" in channels
        assert "slack-a" not in channels  # old entry for agent_a replaced
        assert "slack-a2" in channels

    async def test_edit_agent_channel_name_conflict_with_other_agent(self, real_agent_config, agent_yml_with_singleton):
        """Binding a channel name already owned by another agent is rejected."""
        svc = AgentService()
        await self._create(svc, real_agent_config, "agent_a")
        await self._create(svc, real_agent_config, "agent_b")
        await svc.edit_agent(
            self._edit_with_channels("agent_a", [self._slack(name="shared", secrets={"app_token": "a"})]),
            real_agent_config,
        )

        # agent_b tries to claim the same channel name — must fail, not clobber.
        result = await svc.edit_agent(
            self._edit_with_channels("agent_b", [self._slack(name="shared", secrets={"app_token": "b"})]),
            real_agent_config,
        )
        assert result.success is False
        assert result.errorCode == "CHANNEL_NAME_CONFLICT"
        # agent_a's binding is untouched.
        assert real_agent_config.channels_config["shared"]["subagent_id"] == "agent_a"

    async def test_edit_agent_rejects_blank_channel_name(self, real_agent_config, agent_yml_with_singleton):
        """A whitespace-only channel name is rejected."""
        svc = AgentService()
        await self._create(svc, real_agent_config, "chan_agent")
        result = await svc.edit_agent(
            self._edit_with_channels("chan_agent", [self._slack(name="   ", secrets={"app_token": "x"})]),
            real_agent_config,
        )
        assert result.success is False
        assert result.errorCode == "INVALID_CHANNEL_NAME"

    async def test_delete_agent_removes_its_channels(self, real_agent_config, agent_yml_with_singleton):
        """Deleting an agent drops the channels routed to it."""
        svc = AgentService()
        await self._create(svc, real_agent_config, "chan_agent")
        await svc.edit_agent(
            self._edit_with_channels("chan_agent", [self._slack(secrets={"app_token": "x"})]),
            real_agent_config,
        )
        assert "slack-main" in real_agent_config.channels_config

        result = await svc.delete_agent("chan_agent", real_agent_config)
        assert result.success is True
        assert "slack-main" not in real_agent_config.channels_config
