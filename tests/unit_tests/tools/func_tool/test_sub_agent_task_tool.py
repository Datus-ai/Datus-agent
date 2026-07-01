# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""CI-level tests for SubAgentTaskTool (AgenticNode-based execution)."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import SUBAGENT_COMPLETE_ACTION_TYPE, ActionHistory, ActionRole, ActionStatus
from datus.schemas.agent_models import ScopedContext
from datus.tools.func_tool.sub_agent_task_tool import (
    BUILTIN_SUBAGENT_DESCRIPTIONS,
    NODE_CLASS_MAP,
    SubAgentTaskTool,
)
from datus.utils.constants import SYS_SUB_AGENTS


@pytest.fixture
def mock_agent_config():
    config = Mock(spec=AgentConfig)
    config.db_type = "sqlite"
    config.current_datasource = "test_db"
    config.agentic_nodes = {
        "chat": {"model": "default"},
        "gen_sql": {"model": "default", "system_prompt": "gen_sql", "node_class": "gen_sql"},
        "sales_analyst": {
            "model": "default",
            "node_class": "gen_sql",
            "agent_description": "Sales data specialist",
        },
    }
    config.sub_agent_config.side_effect = lambda name: config.agentic_nodes.get(name)
    return config


@pytest.fixture
def task_tool(mock_agent_config):
    return SubAgentTaskTool(agent_config=mock_agent_config)


# ── Initialization ─────────────────────────────────────────────────


@pytest.mark.ci
class TestInit:
    def test_init(self, task_tool, mock_agent_config):
        assert task_tool.agent_config is mock_agent_config
        assert task_tool._action_bus is None
        assert task_tool._interaction_broker is None

    def test_init_only_requires_agent_config(self, mock_agent_config):
        """No model or tool params needed."""
        tool = SubAgentTaskTool(agent_config=mock_agent_config)
        assert tool.agent_config is mock_agent_config


# ── available_tools ────────────────────────────────────────────────


@pytest.mark.ci
class TestAvailableTools:
    def test_returns_task_function_tool(self, task_tool):
        tools = task_tool.available_tools()
        assert len(tools) == 1
        assert tools[0].name == "task"

    def test_tool_has_correct_schema(self, task_tool):
        tools = task_tool.available_tools()
        schema = tools[0].params_json_schema
        assert "type" in schema["properties"]
        assert "prompt" in schema["properties"]
        assert set(schema["required"]) == {"type", "prompt", "description"}


# ── _get_available_types ───────────────────────────────────────────


@pytest.mark.ci
class TestGetAvailableTypes:
    def test_includes_gen_sql(self, task_tool):
        types = task_tool._get_available_types()
        assert "gen_sql" in types

    def test_includes_custom_subagent(self, task_tool):
        types = task_tool._get_available_types()
        assert "sales_analyst" in types

    def test_excludes_chat(self, task_tool):
        types = task_tool._get_available_types()
        assert "chat" not in types

    def test_includes_agent_without_node_class(self):
        """Subagent without node_class should still be discovered (defaults to gen_sql)."""
        config = Mock(spec=AgentConfig)
        config.current_datasource = "default"
        config.agentic_nodes = {
            "chat": {"model": "default"},
            "custom": {"model": "default"},  # no node_class
        }
        tool = SubAgentTaskTool(agent_config=config)
        types = tool._get_available_types()
        assert "gen_sql" in types
        assert "custom" in types

    def test_excludes_scoped_agent_wrong_datasource(self):
        """Subagent with scoped_context bound to a different datasource should be excluded."""
        config = Mock(spec=AgentConfig)
        config.current_datasource = "default"
        config.agentic_nodes = {
            "chat": {"model": "default"},
            "scoped_agent": {
                "model": "default",
                "node_class": "gen_sql",
                "scoped_context": {"datasource": "other_ds", "tables": "t1"},
            },
        }
        tool = SubAgentTaskTool(agent_config=config)
        types = tool._get_available_types()
        assert "scoped_agent" not in types

    def test_includes_scoped_agent_matching_datasource(self):
        """Subagent with scoped_context matching current datasource should be included."""
        config = Mock(spec=AgentConfig)
        config.current_datasource = "sales"
        config.agentic_nodes = {
            "chat": {"model": "default"},
            "scoped_agent": {
                "model": "default",
                "node_class": "gen_sql",
                "scoped_context": {"datasource": "sales", "tables": "orders"},
            },
        }
        tool = SubAgentTaskTool(agent_config=config)
        types = tool._get_available_types()
        assert "scoped_agent" in types

    def test_includes_agent_without_scoped_context(self):
        """Subagent without scoped_context should not be filtered by datasource."""
        config = Mock(spec=AgentConfig)
        config.current_datasource = "default"
        config.agentic_nodes = {
            "chat": {"model": "default"},
            "global_agent": {
                "model": "default",
                "node_class": "gen_sql",
                "agent_description": "A global agent",
            },
        }
        tool = SubAgentTaskTool(agent_config=config)
        types = tool._get_available_types()
        assert "global_agent" in types

    def test_explicit_list_filters_out_unknown_types(self, caplog):
        """Unknown types in explicit allowed_subagents are skipped with a warning."""
        config = Mock(spec=AgentConfig)
        config.current_datasource = "default"
        config.agentic_nodes = {"chat": {"model": "default"}}
        tool = SubAgentTaskTool(
            agent_config=config,
            allowed_subagents=["gen_sql", "nonexistent_foo", "explore"],
            parent_node_name="chat",
        )

        import logging

        with caplog.at_level(logging.WARNING, logger="datus.tools.func_tool.sub_agent_task_tool"):
            types = tool._get_available_types()

        assert "gen_sql" in types
        assert "explore" in types
        assert "nonexistent_foo" not in types
        assert any("nonexistent_foo" in rec.message for rec in caplog.records)

    def test_explicit_list_excludes_self(self):
        """The parent node name is excluded even if listed in allowed_subagents."""
        config = Mock(spec=AgentConfig)
        config.current_datasource = "default"
        config.agentic_nodes = {"chat": {"model": "default"}}
        tool = SubAgentTaskTool(
            agent_config=config,
            allowed_subagents=["gen_sql", "explore"],
            parent_node_name="gen_sql",
        )
        types = tool._get_available_types()
        assert "gen_sql" not in types
        assert "explore" in types


# ── _resolve_node_type ─────────────────────────────────────────────


@pytest.mark.ci
class TestResolveNodeType:
    def test_gen_sql_with_config(self, task_tool):
        """gen_sql resolves to TYPE_GEN_SQL using config key."""
        node_type, node_name = task_tool._resolve_node_type("gen_sql")
        assert node_type == NodeType.TYPE_GEN_SQL
        assert node_name == "gen_sql"

    def test_gen_sql_without_config(self):
        """gen_sql falls back to TYPE_GEN_SQL with default name."""
        config = Mock(spec=AgentConfig)
        config.agentic_nodes = {}
        tool = SubAgentTaskTool(agent_config=config)
        node_type, node_name = tool._resolve_node_type("gen_sql")
        assert node_type == NodeType.TYPE_GEN_SQL
        assert node_name == "gen_sql"

    def test_custom_type_gen_sql_class(self, task_tool):
        """Custom type with node_class=gen_sql maps to TYPE_GEN_SQL."""
        node_type, node_name = task_tool._resolve_node_type("sales_analyst")
        assert node_type == NodeType.TYPE_GEN_SQL
        assert node_name == "sales_analyst"

    def test_unknown_type_raises(self, task_tool):
        """Unknown type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown subagent type"):
            task_tool._resolve_node_type("nonexistent")

    def test_gen_visual_report_resolves(self, task_tool):
        """gen_visual_report must be registered alongside other built-in subagents.

        Regression test for the bug where the chat agent enumerated
        gen_visual_report in available types (via SYS_SUB_AGENTS) but the
        task tool had no NODE_CLASS_MAP / description / factory entry,
        causing the LLM to report it as 'unavailable'.
        """
        node_type, node_name = task_tool._resolve_node_type("gen_visual_report")
        assert node_type == NodeType.TYPE_GEN_VISUAL_REPORT
        assert node_name == "gen_visual_report"
        assert NODE_CLASS_MAP["gen_visual_report"] == NodeType.TYPE_GEN_VISUAL_REPORT
        assert "gen_visual_report" in BUILTIN_SUBAGENT_DESCRIPTIONS

    def test_ask_metrics_resolves(self, task_tool):
        node_type, node_name = task_tool._resolve_node_type("ask_metrics")

        assert node_type == NodeType.TYPE_ASK_METRICS
        assert node_name == "ask_metrics"

    def test_gen_visual_report_constructs_and_builds_input(self, task_tool, tmp_path):
        """Mirror of ``test_gen_visual_dashboard_constructs_and_builds_input``
        for the report subagent — ``_create_builtin_node`` + the
        matching ``_build_node_input`` branch must produce the right
        node class and input dataclass. The mapping-level
        ``test_gen_visual_report_resolves`` check above won't catch
        constructor signature drift or an input-builder branch
        returning the wrong type.
        """
        from datus.agent.node.gen_visual_report_agentic_node import GenVisualReportAgenticNode
        from datus.schemas.gen_visual_report_models import GenVisualReportNodeInput

        task_tool.agent_config.workspace_root = str(tmp_path)

        node = task_tool._create_builtin_node("gen_visual_report")
        assert isinstance(node, GenVisualReportAgenticNode), f"factory returned wrong type: {type(node).__name__}"
        assert node.configured_node_name == "gen_visual_report"
        assert node.NODE_NAME == "gen_visual_report"
        assert node.ARTIFACT_KIND == "report"
        assert node.execution_mode == "interactive"

        prompt = "produce a quarterly revenue report"
        node_input = task_tool._build_node_input(node, prompt)
        assert isinstance(node_input, GenVisualReportNodeInput), (
            f"input builder returned wrong type: {type(node_input).__name__}"
        )
        assert node_input.user_message == prompt
        # The ``database`` context field denotes a physical database, not a datasource, so the
        # builder must not stuff the datasource name ("test_db") into it; it stays unset.
        assert node_input.database is None

    def test_gen_visual_report_rejects_session_id(self, task_tool):
        """gen_visual_report has the same no-resume contract as
        gen_visual_dashboard — both inherit from
        ``BaseVisualArtifactAgenticNode`` which doesn't accept
        ``session_id``. Pin ValueError + the load-bearing substring
        so a regression to silent-drop trips here."""
        with pytest.raises(ValueError, match="gen_visual_report.*session resume"):
            task_tool._create_builtin_node("gen_visual_report", session_id="some-prior-session")

    def test_gen_visual_dashboard_resolves(self, task_tool):
        """gen_visual_dashboard must be registered alongside the other visual subagent.

        Mirrors :meth:`test_gen_visual_report_resolves` — the dashboard subagent
        is enumerated in ``SYS_SUB_AGENTS`` and exposed via the chat agent's
        task tool, so missing entries in NODE_CLASS_MAP / descriptions /
        factory cause the LLM to report it as 'unavailable'.
        """
        node_type, node_name = task_tool._resolve_node_type("gen_visual_dashboard")
        assert node_type == NodeType.TYPE_GEN_VISUAL_DASHBOARD
        assert node_name == "gen_visual_dashboard"
        assert NODE_CLASS_MAP["gen_visual_dashboard"] == NodeType.TYPE_GEN_VISUAL_DASHBOARD
        assert "gen_visual_dashboard" in BUILTIN_SUBAGENT_DESCRIPTIONS

    def test_gen_visual_dashboard_constructs_and_builds_input(self, task_tool, tmp_path):
        """``_create_builtin_node`` and ``_build_node_input`` must wire
        together for gen_visual_dashboard the same way they do for
        every other builtin subagent. The mapping-level assertions in
        :meth:`test_gen_visual_dashboard_resolves` cover registration
        but won't catch a constructor signature drift or an input-
        builder branch that returns the wrong dataclass — both would
        only surface at runtime when the LLM actually invokes the
        subagent.
        """
        from datus.agent.node.gen_visual_dashboard_agentic_node import GenVisualDashboardAgenticNode
        from datus.schemas.gen_visual_dashboard_models import GenVisualDashboardNodeInput

        # ``_setup_filesystem_tools`` reaches for the workspace root;
        # the fixture's bare Mock would otherwise emit a TypeError-noise
        # log line during construction. Setting it to a real path keeps
        # the captured-log output clean for future log-based tests.
        task_tool.agent_config.workspace_root = str(tmp_path)

        node = task_tool._create_builtin_node("gen_visual_dashboard")
        assert isinstance(node, GenVisualDashboardAgenticNode), f"factory returned wrong type: {type(node).__name__}"
        # Both the configured name (used for node_config lookup) and
        # the class-level NODE_NAME / ARTIFACT_KIND constants are
        # load-bearing for the prompt template / artifact dir routing.
        assert node.configured_node_name == "gen_visual_dashboard"
        assert node.NODE_NAME == "gen_visual_dashboard"
        assert node.ARTIFACT_KIND == "dashboard"
        assert node.execution_mode == "interactive"

        # Cross-component contract: feeding the same prompt into
        # ``_build_node_input`` must yield the dashboard-specific
        # input dataclass with the user_message + scoped datasource
        # populated. Hits the ``isinstance(node, GenVisualDashboardAgenticNode)``
        # branch on the input builder side — without this assertion a
        # branch that's only registered but not wired to build input
        # would slip through.
        prompt = "build a dashboard for daily AOV by store"
        node_input = task_tool._build_node_input(node, prompt)
        assert isinstance(node_input, GenVisualDashboardNodeInput), (
            f"input builder returned wrong type: {type(node_input).__name__}"
        )
        assert node_input.user_message == prompt
        # ``database`` denotes a physical database, not a datasource, so the builder leaves it
        # unset rather than stuffing in ``current_datasource`` ("test_db").
        assert node_input.database is None

    def test_gen_visual_dashboard_rejects_session_id(self, task_tool):
        """``_create_builtin_node`` for gen_visual_dashboard MUST fail
        loud when a session_id is passed — the underlying
        ``BaseVisualArtifactAgenticNode`` constructor has no
        ``session_id`` parameter so silently dropping it (the prior
        behaviour) would let resume loops spawn a fresh session per
        turn while the LLM thinks it picked up an existing one. Pin
        on ValueError + a substring that names both the subagent type
        and the load-bearing reason ("does not support session
        resume") so a regression to silent-drop or to a different
        error class trips here.
        """
        with pytest.raises(ValueError, match="gen_visual_dashboard.*session resume"):
            task_tool._create_builtin_node("gen_visual_dashboard", session_id="some-prior-session")

    def test_resolve_effective_inherits_parent_when_child_empty(self, task_tool):
        parent = MagicMock()
        parent.node_config = {"scoped_context": {"tables": "public.users"}}
        task_tool._parent_node = parent
        # 'sales_analyst' yaml has no scoped_context → child empty → inherit parent
        effective = task_tool._resolve_effective_sub_agent_config("sales_analyst")
        assert isinstance(effective.scoped_context, ScopedContext)
        assert effective.scoped_context.tables == "public.users"

    def test_resolve_effective_accepts_parent_scoped_context_instance(self, task_tool):
        from datus.schemas.agent_models import ScopedContext

        parent = MagicMock()
        parent.node_config = {"scoped_context": ScopedContext(tables="public.orders")}
        task_tool._parent_node = parent
        effective = task_tool._resolve_effective_sub_agent_config("sales_analyst")
        assert effective.scoped_context.tables == "public.orders"

    def test_resolve_effective_no_parent_returns_child_only(self, task_tool):
        # No parent → child config drives everything; sales_analyst has no scope → empty
        task_tool._parent_node = None
        effective = task_tool._resolve_effective_sub_agent_config("sales_analyst")
        assert effective.scoped_context is None or effective.scoped_context.is_empty

    def test_resolve_effective_child_overrides_parent(self, task_tool, mock_agent_config):
        # Add a child with its own scoped_context to the agentic_nodes fixture
        mock_agent_config.agentic_nodes["scoped_child"] = {
            "model": "default",
            "node_class": "gen_sql",
            "scoped_context": {"tables": "public.products"},
        }
        parent = MagicMock()
        parent.node_config = {"scoped_context": {"tables": "public.users"}}
        task_tool._parent_node = parent
        effective = task_tool._resolve_effective_sub_agent_config("scoped_child")
        # Whole-segment override: child wins
        assert effective.scoped_context.tables == "public.products"

    def test_resolve_inherited_memory_returns_parent_for_chat(self, task_tool):
        """Built-in child + chat parent → inherit chat's memory."""
        parent = MagicMock()
        parent.get_node_name.return_value = "chat"
        task_tool._parent_node = parent
        assert task_tool._resolve_inherited_memory_node("gen_sql") == "chat"

    def test_resolve_inherited_memory_returns_parent_for_custom_subagent(self, task_tool):
        """Built-in child + custom subagent parent → inherit custom's memory."""
        parent = MagicMock()
        parent.get_node_name.return_value = "sales_analyst"
        task_tool._parent_node = parent
        # custom parent owns its own memory node → child inherits it
        assert task_tool._resolve_inherited_memory_node("gen_sql") == "sales_analyst"

    def test_resolve_inherited_memory_returns_none_for_feedback(self, task_tool):
        """feedback has its own caller-memory mechanism; never inherit via this path."""
        parent = MagicMock()
        parent.get_node_name.return_value = "chat"
        task_tool._parent_node = parent
        assert task_tool._resolve_inherited_memory_node("feedback") is None

    def test_resolve_inherited_memory_custom_child_inherits_parent(self, task_tool):
        """Every sub-agent (including custom ones) inlines the parent's memory
        read-only when run via task — sub-agents never write memory."""
        parent = MagicMock()
        parent.get_node_name.return_value = "chat"
        task_tool._parent_node = parent
        assert task_tool._resolve_inherited_memory_node("sales_analyst") == "chat"

    def test_resolve_inherited_memory_returns_none_when_no_parent(self, task_tool):
        task_tool._parent_node = None
        assert task_tool._resolve_inherited_memory_node("gen_sql") is None

    def test_resolve_inherited_memory_resolves_builtin_parent_to_chat(self, task_tool):
        """A built-in parent owns no memory of its own — as a main agent it uses
        the shared ``chat`` memory, so its sub-agents inherit ``chat``."""
        parent = MagicMock()
        parent.get_node_name.return_value = "gen_report"
        task_tool._parent_node = parent
        assert task_tool._resolve_inherited_memory_node("gen_sql") == "chat"

    def test_resolve_inherited_memory_swallows_parent_exceptions(self, task_tool):
        """If the parent's get_node_name() raises, fall back to no inheritance."""
        parent = MagicMock()
        parent.get_node_name.side_effect = RuntimeError("parent broken")
        task_tool._parent_node = parent
        assert task_tool._resolve_inherited_memory_node("gen_sql") is None

    def test_node_class_map_coverage(self):
        """NODE_CLASS_MAP contains exactly the expected key→NodeType mappings."""
        expected_map = {
            "gen_sql": NodeType.TYPE_GEN_SQL,
            "chat": NodeType.TYPE_CHAT,
            "ask_metrics": NodeType.TYPE_ASK_METRICS,
            "gen_report": NodeType.TYPE_GEN_REPORT,
            "gen_visual_report": NodeType.TYPE_GEN_VISUAL_REPORT,
            "gen_visual_dashboard": NodeType.TYPE_GEN_VISUAL_DASHBOARD,
            "semantic": NodeType.TYPE_SEMANTIC,
            "sql_summary": NodeType.TYPE_SQL_SUMMARY,
            "explore": NodeType.TYPE_EXPLORE,
            "gen_table": NodeType.TYPE_GEN_TABLE,
            "gen_job": NodeType.TYPE_GEN_JOB,
            "gen_skill": NodeType.TYPE_GEN_SKILL,
            "gen_dashboard": NodeType.TYPE_GEN_DASHBOARD,
            "scheduler": NodeType.TYPE_SCHEDULER,
        }
        assert set(NODE_CLASS_MAP.keys()) == set(expected_map.keys()), (
            f"NODE_CLASS_MAP keys differ: got {set(NODE_CLASS_MAP.keys())}"
        )
        for key, expected_value in expected_map.items():
            assert NODE_CLASS_MAP[key] == expected_value, f"Wrong mapping for key '{key}'"


# ── _build_task_description ────────────────────────────────────────


@pytest.mark.ci
class TestBuildTaskDescription:
    def test_contains_all_types(self, task_tool):
        desc = task_tool._build_task_description()
        assert "gen_sql" in desc
        assert "sales_analyst" in desc

    def test_contains_guidelines(self, task_tool):
        desc = task_tool._build_task_description()
        assert "Guidelines" in desc

    def test_description_routes_by_deliverable_ownership(self, task_tool):
        desc = task_tool._build_task_description()
        assert "requested deliverable" in desc
        assert "owning workflow" in desc
        assert "Task complexity is not the deciding factor" in desc
        assert "For simple questions, handle directly" not in desc

    def test_contains_custom_description(self, task_tool):
        desc = task_tool._build_task_description()
        assert "Sales data specialist" in desc

    def test_explore_description_contains_directions(self, task_tool):
        """Explore description lists 3 exploration directions."""
        desc = task_tool._build_task_description()
        assert "Schema+Sample" in desc
        assert "Knowledge" in desc
        assert "File" in desc

    def test_explore_description_contains_prompt_examples(self, task_tool):
        """Explore description includes prompt examples for each direction."""
        desc = task_tool._build_task_description()
        assert "Prompt example:" in desc

    def test_guidelines_contain_parallel_explore(self, task_tool):
        """Guidelines recommend parallel exploration with direction-specific prompts."""
        desc = task_tool._build_task_description()
        assert "PARALLEL" in desc
        assert "direction-specific prompt" in desc

    def test_migration_subagent_removed(self, task_tool):
        """Migration has been merged into gen_job — it should no longer be a standalone type."""
        desc = task_tool._build_task_description()
        # The literal subagent name 'migration' should not appear as a bullet entry
        assert "- migration:" not in desc

    def test_gen_job_description_mentions_cross_database_migration(self, task_tool):
        """Parent ChatAgenticNode routes to gen_job for migration — description must carry the signal."""
        desc = task_tool._build_task_description()
        # Pin the exact routing keywords the description commits to so the
        # parent agent's routing LLM has a deterministic match.
        haystack = desc.lower()
        assert "gen_job" in haystack
        assert "migration" in haystack
        assert "cross-database migration" in haystack

    def test_gen_report_description_is_explicit_only(self, task_tool):
        """The task tool must not advertise gen_report as automatic root-cause routing."""
        desc = task_tool._build_task_description()
        assert "Legacy Markdown report subagent" in desc
        assert "Use only when the user explicitly asks to use the gen_report subagent" in desc
        assert "do not automatically route attribution" in desc
        assert "Use when the question involves metric attribution" not in desc


# ── node creation (fresh per invocation) ──────────────────────────


@pytest.mark.ci
class TestNodeCreation:
    def test_always_creates_fresh_node(self, task_tool):
        """Each call to _create_node returns a distinct new instance (no caching).

        Uses "explore" because it is in NODE_CLASS_MAP but NOT in SYS_SUB_AGENTS,
        so it goes through the Node.new_instance factory path.
        """
        node_a = Mock(name="node_a")
        node_b = Mock(name="node_b")

        with patch("datus.agent.node.node.Node.new_instance", side_effect=[node_a, node_b]):
            node1 = task_tool._create_node("explore")
            node2 = task_tool._create_node("explore")

        assert node1 is not node2
        assert node1 is node_a
        assert node2 is node_b


# ── _build_node_input ──────────────────────────────────────────────


@pytest.mark.ci
class TestBuildNodeInput:
    def test_gen_sql_node_input(self, task_tool):
        """GenSQLAgenticNode gets GenSQLNodeInput."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput

        # Create a mock that is an instance of GenSQLAgenticNode
        mock_node = Mock(spec=GenSQLAgenticNode)
        mock_node.type = NodeType.TYPE_GEN_SQL

        result = task_tool._build_node_input(mock_node, "Show all users")

        assert isinstance(result, GenSQLNodeInput)
        assert result.user_message == "Show all users"
        assert result.database is None

    def test_ask_metrics_node_input(self, task_tool):
        from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode
        from datus.schemas.ask_metrics_agentic_node_models import AskMetricsNodeInput

        mock_node = Mock(spec=AskMetricsAgenticNode)
        result = task_tool._build_node_input(mock_node, "Show revenue by month")

        assert isinstance(result, AskMetricsNodeInput)
        assert result.user_message == "Show revenue by month"
        assert result.database is None


# ── _convert_to_func_result ───────────────────────────────────────


@pytest.mark.ci
class TestConvertToFuncResult:
    def test_sql_result(self, task_tool):
        """GenSQLNodeResult with sql key."""
        output = {"sql": "SELECT 1", "response": "test query", "tokens_used": 100}
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["sql"] == "SELECT 1"
        assert result.result["response"] == "test query"
        assert result.result["tokens_used"] == 100

    def test_generic_result(self, task_tool):
        """Result without sql key."""
        output = {"response": "Here is the answer", "tokens_used": 50}
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["response"] == "Here is the answer"
        assert result.result["tokens_used"] == 50

    def test_none_output(self, task_tool):
        """None output returns error."""
        result = task_tool._convert_to_func_result(None)
        assert result.success == 0
        assert "No result" in result.error

    def test_empty_dict(self, task_tool):
        """Empty dict returns error."""
        result = task_tool._convert_to_func_result({})
        assert result.success == 0

    def test_content_fallback(self, task_tool):
        """Falls back to 'content' key when no 'response'."""
        output = {"content": "Some content", "tokens_used": 0}
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["response"] == "Some content"

    def test_markdown_report_result(self, task_tool):
        output = {"response": "Metric answer", "markdown_report": "## Metric answer", "tokens_used": 25}

        result = task_tool._convert_to_func_result(output, session_id="ask-metrics-session")

        assert result.success == 1
        assert result.result == {
            "response": "Metric answer",
            "markdown_report": "## Metric answer",
            "tokens_used": 25,
            "session_id": "ask-metrics-session",
        }

    def test_visual_dashboard_result_preserves_documented_fields(self, task_tool):
        """``GenVisualDashboardNodeResult.model_dump()`` carries
        ``dashboard_slug``, ``app_jsx_path``, ``render_file_count``,
        ``template_count`` flat at the top level (no
        ``dashboard_result`` envelope — that key belongs to the legacy
        ``gen_dashboard`` node). The conversion must preserve every
        field the parent LLM is told to expect via
        ``BUILTIN_SUBAGENT_DESCRIPTIONS["gen_visual_dashboard"]``.
        """
        # Shape mirrors ``GenVisualDashboardNodeResult(...).model_dump()``
        # — only the documented + load-bearing fields are inlined here
        # so a regression that adds a new field to the model doesn't
        # spuriously fail this test.
        output = {
            "success": True,
            "response": "Dashboard built.",
            "dashboard_slug": "aov_weekly",
            "app_jsx_path": "dashboards/aov_weekly/render/app.jsx",
            "render_file_count": 4,
            "template_count": 3,
            "tokens_used": 12345,
            "artifact_kind": "dashboard",
            "artifact_mode": "new",
            "name": "AOV Weekly Trend",
        }
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        # Every documented field present + non-discarded.
        assert result.result["response"] == "Dashboard built."
        assert result.result["dashboard_slug"] == "aov_weekly"
        assert result.result["app_jsx_path"] == "dashboards/aov_weekly/render/app.jsx"
        assert result.result["render_file_count"] == 4
        assert result.result["template_count"] == 3
        assert result.result["tokens_used"] == 12345

    def test_visual_dashboard_result_preserves_none_slug_on_partial_run(self, task_tool):
        """When the dashboard run failed before binding, the model
        emits ``dashboard_slug=None`` (still a populated key). The
        conversion must keep that explicit None rather than fall
        through to the generic envelope — otherwise the parent LLM
        can't tell "the subagent kind ran but didn't bind" apart from
        "the subagent kind wasn't even invoked"."""
        output = {
            "success": False,
            "response": "Failed before binding an artifact.",
            "dashboard_slug": None,
            "app_jsx_path": None,
            "render_file_count": 0,
            "template_count": 0,
            "tokens_used": 42,
        }
        # The branch fires regardless of the ``success`` flag because
        # the early ``output.get("success") is False`` check intercepts
        # explicit failures. For partial-runs where the result model
        # was constructed with ``success=True`` but bindings still
        # None, the same branch handles it.
        # We simulate the partial-success case by removing the
        # explicit-failure signal.
        partial = dict(output, success=True)
        result = task_tool._convert_to_func_result(partial)
        assert result.success == 1
        # ``dashboard_slug`` key present with None — operators / parent
        # LLM can disambiguate "ran but no artifact" from missing-key.
        assert "dashboard_slug" in result.result
        assert result.result["dashboard_slug"] is None
        assert result.result["render_file_count"] == 0
        assert result.result["template_count"] == 0


@pytest.mark.acceptance
class TestSubAgentTaskAcceptance:
    """Deterministic custom subagent discovery and delegation contract."""

    @pytest.mark.asyncio
    async def test_custom_subagent_discovery_scope_and_delegation(self, task_tool):
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        parent = MagicMock()
        parent.node_config = {"scoped_context": {"tables": "public.orders"}}
        parent.get_node_name.return_value = "chat"
        parent.proxy_tool_patterns = []
        parent.tool_channel = None
        # A chat parent carries no bound physical database, so the delegated subagent must
        # inherit nothing here (contrast TestSubagentInheritsParentDatabase, where a DB-bound
        # parent does propagate). Pin these so the auto-MagicMock attributes don't masquerade
        # as a real database.
        parent.input.database = None
        parent.input.catalog = None
        parent.input.db_schema = None
        parent.db_func_tool = None
        task_tool._parent_node = parent

        assert "sales_analyst" in task_tool._get_available_types()
        node_type, node_name = task_tool._resolve_node_type("sales_analyst")
        assert node_type == NodeType.TYPE_GEN_SQL
        assert node_name == "sales_analyst"

        effective = task_tool._resolve_effective_sub_agent_config("sales_analyst")
        assert effective.scoped_context.tables == "public.orders"

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.ASSISTANT
        mock_action.output = {
            "response": "Sales report is ready",
            "sql": "SELECT SUM(amount) FROM sales",
            "tokens_used": 42,
            "success": True,
        }
        mock_node = Mock(spec=GenSQLAgenticNode)
        mock_node.type = NodeType.TYPE_GEN_SQL
        mock_node.session_id = "sales_session_1"

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node) as create_node:
            result = await task_tool.task(type="sales_analyst", prompt="Summarize sales")

        assert result.success == 1
        assert result.result["response"] == "Sales report is ready"
        assert result.result["sql"] == "SELECT SUM(amount) FROM sales"
        create_node.assert_called_once_with("sales_analyst", session_id=None)
        assert mock_node.input.user_message == "Summarize sales"
        assert mock_node.input.database is None


# ── task execution ─────────────────────────────────────────────────


@pytest.mark.ci
class TestTaskExecution:
    @pytest.mark.asyncio
    async def test_execute_gen_sql_success(self, task_tool):
        """Successful gen_sql execution through node."""
        # Create mock action with SUCCESS status and GenSQLNodeResult-like output
        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.TOOL
        mock_action.output = {
            "sql": "SELECT 1",
            "response": "test query",
            "tokens_used": 100,
            "success": True,
        }

        mock_node = MagicMock()

        # Make execute_stream an async generator
        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="Show all users")

        assert result.success == 1
        assert result.result["sql"] == "SELECT 1"
        assert result.result["tokens_used"] == 100

    @pytest.mark.asyncio
    async def test_execute_unknown_type(self, task_tool):
        result = await task_tool.task(type="nonexistent", prompt="test")
        assert result.success == 0
        assert "disallowed subagent type" in result.error
        # repr of the offending value is included so hidden characters become visible
        assert "'nonexistent'" in result.error

    @pytest.mark.asyncio
    async def test_execute_type_with_whitespace_is_normalized(self, task_tool):
        """LLM sometimes emits ``" gen_sql "`` — _execute_node must strip before matching."""
        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.TOOL
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 1, "success": True}

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node) as create:
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="  gen_sql\n", prompt="test")

        assert result.success == 1
        # The normalized (stripped) type is what gets passed to _create_node.
        create.assert_called_once_with("gen_sql", session_id=None)

    @pytest.mark.asyncio
    async def test_execute_type_with_quotes_is_normalized(self, task_tool):
        """LLM sometimes wraps the type in quotes — outer quotes must be stripped."""
        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.TOOL
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 1, "success": True}

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node) as create:
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type='"gen_sql"', prompt="test")

        assert result.success == 1
        create.assert_called_once_with("gen_sql", session_id=None)

    @pytest.mark.asyncio
    async def test_execute_missing_type(self, task_tool):
        result = await task_tool.task(type="", prompt="test")
        assert result.success == 0
        assert "Missing required parameter: type" in result.error

    @pytest.mark.asyncio
    async def test_execute_missing_prompt(self, task_tool):
        result = await task_tool.task(type="gen_sql", prompt="")
        assert result.success == 0
        assert "Missing required parameter: prompt" in result.error

    @pytest.mark.asyncio
    async def test_execute_custom_subagent(self, task_tool):
        """Custom subagent type executes through node."""
        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.ASSISTANT
        mock_action.output = {"response": "Sales report", "tokens_used": 50, "success": True}

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="sales_analyst", prompt="Show sales")

        assert result.success == 1
        assert result.result["response"] == "Sales report"

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, task_tool):
        """Node exception is caught and returned as error."""
        with patch.object(task_tool, "_create_node", side_effect=RuntimeError("Node init error")):
            result = await task_tool.task(type="gen_sql", prompt="test")

        assert result.success == 0
        assert "Task execution failed" in result.error

    @pytest.mark.asyncio
    async def test_execute_no_successful_output(self, task_tool):
        """When stream yields no successful actions, returns error."""
        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.PROCESSING
        mock_action.role = ActionRole.ASSISTANT
        mock_action.output = None

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="test")

        assert result.success == 0
        assert "No result" in result.error


# ── ActionBus integration ──────────────────────────────────────────


@pytest.mark.ci
class TestActionBusIntegration:
    def test_set_action_bus(self, task_tool):
        """set_action_bus stores the bus reference."""
        from datus.schemas.action_bus import ActionBus

        bus = ActionBus()
        task_tool.set_action_bus(bus)
        assert task_tool._action_bus is bus

    @pytest.mark.asyncio
    async def test_actions_forwarded_to_bus(self, task_tool):
        """Child actions are put into action_bus with depth=1."""
        from datus.schemas.action_bus import ActionBus
        from datus.schemas.action_history import ActionRole

        bus = ActionBus()
        task_tool.set_action_bus(bus)

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}
        mock_action.depth = 0
        mock_action.parent_action_id = None
        mock_action.role = ActionRole.TOOL  # Non-USER role so it's forwarded
        mock_action.action_type = "tool_call"

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="test")

        assert result.success == 1
        # Forwarded action should be in the bus with depth=1
        forwarded = bus._queue.get_nowait()
        assert forwarded.depth == 1

    @pytest.mark.asyncio
    async def test_actions_have_parent_action_id(self, task_tool):
        """When call_id is provided, child actions get parent_action_id."""
        from datus.schemas.action_bus import ActionBus
        from datus.schemas.action_history import ActionRole

        bus = ActionBus()
        task_tool.set_action_bus(bus)

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}
        mock_action.depth = 0
        mock_action.parent_action_id = None
        mock_action.role = ActionRole.TOOL  # Non-USER role so it's forwarded
        mock_action.action_type = "tool_call"

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="test", call_id="parent_call_123")

        assert result.success == 1
        # Forwarded action should have parent_action_id
        forwarded = bus._queue.get_nowait()
        assert forwarded.parent_action_id == "parent_call_123"

    @pytest.mark.asyncio
    async def test_no_bus_no_error(self, task_tool):
        """Without action_bus, execution still works (no forwarding)."""
        assert task_tool._action_bus is None

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.TOOL
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="test")

        assert result.success == 1


# ── InteractionBroker pass-through ─────────────────────────────────


@pytest.mark.ci
class TestInteractionBrokerPassthrough:
    def test_set_interaction_broker(self, task_tool):
        """set_interaction_broker stores the broker reference."""
        from datus.cli.execution_state import InteractionBroker

        broker = InteractionBroker()
        task_tool.set_interaction_broker(broker)
        assert task_tool._interaction_broker is broker

    def test_inject_broker_updates_node(self, task_tool):
        """_inject_broker replaces the node's interaction_broker."""
        from datus.cli.execution_state import InteractionBroker

        parent_broker = InteractionBroker()
        node = MagicMock()
        node.interaction_broker = InteractionBroker()  # original broker
        node.hooks = None
        node.permission_hooks = None
        node.plan_hooks = None

        task_tool._inject_broker(node, parent_broker)
        assert node.interaction_broker is parent_broker

    def test_inject_broker_updates_hooks(self, task_tool):
        """_inject_broker updates broker on GenerationHooks / PermissionHooks."""
        from datus.cli.execution_state import InteractionBroker

        parent_broker = InteractionBroker()
        original_broker = InteractionBroker()

        # Use spec to limit attributes — prevents false positive on hooks_list
        mock_hooks = Mock(spec=["broker"])
        mock_hooks.broker = original_broker

        mock_perm_hooks = Mock(spec=["broker"])
        mock_perm_hooks.broker = original_broker

        node = MagicMock()
        node.interaction_broker = original_broker
        node.hooks = mock_hooks
        node.permission_hooks = mock_perm_hooks
        node.plan_hooks = None

        task_tool._inject_broker(node, parent_broker)

        assert node.interaction_broker is parent_broker
        assert mock_hooks.broker is parent_broker
        assert mock_perm_hooks.broker is parent_broker

    def test_inject_broker_updates_composite_hooks(self, task_tool):
        """_inject_broker updates broker inside CompositeHooks."""
        from datus.cli.execution_state import InteractionBroker

        parent_broker = InteractionBroker()
        original_broker = InteractionBroker()

        inner_hook_1 = Mock()
        inner_hook_1.broker = original_broker
        inner_hook_2 = Mock()
        inner_hook_2.broker = original_broker

        composite = Mock()
        composite.broker = original_broker
        composite.hooks_list = [inner_hook_1, inner_hook_2]

        node = MagicMock()
        node.interaction_broker = original_broker
        node.hooks = composite
        node.permission_hooks = None
        node.plan_hooks = None

        task_tool._inject_broker(node, parent_broker)

        assert inner_hook_1.broker is parent_broker
        assert inner_hook_2.broker is parent_broker

    @pytest.mark.asyncio
    async def test_with_broker_uses_execute_stream(self, task_tool):
        """When broker is injected, _execute_node calls execute_stream (not
        _with_interactions) so the injected broker is not dual-consumed — but it
        wraps the call in the node's own ``action_bus.merge`` so hook-enqueued
        actions (e.g. ``token_usage``) are still surfaced to the parent."""
        from datus.cli.execution_state import InteractionBroker
        from datus.schemas.action_bus import ActionBus

        parent_broker = InteractionBroker()
        task_tool.set_interaction_broker(parent_broker)

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.TOOL
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}

        mock_node = MagicMock()
        # Real ActionBus so the merge path yields the primary stream (and would
        # surface any bus.put() items).
        mock_node.action_bus = ActionBus()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream = mock_stream
        # execute_stream_with_interactions should NOT be called
        mock_node.execute_stream_with_interactions = MagicMock(side_effect=AssertionError("should not be called"))

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="test")

        assert result.success == 1
        assert result.result["sql"] == "SELECT 1"

    @pytest.mark.asyncio
    async def test_with_broker_surfaces_bus_enqueued_token_usage(self, task_tool):
        """A sub-agent ``token_usage`` action delivered via ``action_bus.put``
        (the TokenUsageHook path) must be forwarded to the parent ActionBus with
        depth=1 — the regression that left the pinned-header counter at 0."""
        from datus.cli.execution_state import InteractionBroker
        from datus.schemas.action_bus import ActionBus

        task_tool.set_interaction_broker(InteractionBroker())
        parent_bus = ActionBus()
        task_tool.set_action_bus(parent_bus)
        forwarded: list = []
        parent_bus.put = lambda action: forwarded.append(action)

        final_action = Mock(spec=ActionHistory)
        final_action.status = ActionStatus.SUCCESS
        final_action.role = ActionRole.ASSISTANT
        final_action.output = {"response": "ok", "tokens_used": 12603}

        usage_action = Mock(spec=ActionHistory)
        usage_action.status = ActionStatus.SUCCESS
        usage_action.role = ActionRole.ASSISTANT
        usage_action.action_type = "token_usage"
        usage_action.output = {"cumulative": {"total_tokens": 12603, "cached_tokens": 8192}}

        node_bus = ActionBus()
        mock_node = MagicMock()
        mock_node.action_bus = node_bus

        async def mock_stream(ahm):
            # Hook delivers token_usage via the node's bus mid-stream.
            node_bus.put(usage_action)
            yield final_action

        mock_node.execute_stream = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                await task_tool.task(type="gen_sql", prompt="test")

        # The token_usage action reached the parent bus, tagged depth=1.
        usage_forwarded = [a for a in forwarded if getattr(a, "action_type", None) == "token_usage"]
        assert usage_forwarded, "sub-agent token_usage was not forwarded to the parent bus"
        assert usage_forwarded[0].depth == 1

    @pytest.mark.asyncio
    async def test_without_broker_uses_execute_stream_with_interactions(self, task_tool):
        """Without broker, _execute_node falls back to execute_stream_with_interactions."""
        assert task_tool._interaction_broker is None

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.TOOL
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream
        # execute_stream should NOT be called
        mock_node.execute_stream = MagicMock(side_effect=AssertionError("should not be called"))

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="test")

        assert result.success == 1

    @pytest.mark.asyncio
    async def test_broker_injected_into_node(self, task_tool):
        """When parent broker is set, _execute_node injects it into the created node."""
        from datus.cli.execution_state import InteractionBroker

        parent_broker = InteractionBroker()
        task_tool.set_interaction_broker(parent_broker)

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.ASSISTANT
        mock_action.output = {"response": "ok", "tokens_used": 0}

        injected_broker = None

        from datus.schemas.action_bus import ActionBus

        mock_node = MagicMock()
        mock_node.hooks = None
        mock_node.permission_hooks = None
        mock_node.plan_hooks = None
        mock_node.action_bus = ActionBus()

        async def mock_stream(ahm):
            nonlocal injected_broker
            injected_broker = mock_node.interaction_broker
            yield mock_action

        mock_node.execute_stream = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                await task_tool.task(type="gen_sql", prompt="test")

        assert injected_broker is parent_broker


# ── SQL file storage result conversion ─────────────────────────────


@pytest.mark.ci
class TestConvertToFuncResultFileStorage:
    """Tests for _convert_to_func_result with file-based SQL results."""

    def test_file_based_sql_result(self, task_tool):
        """Result with sql_file_path returns file-based format."""
        output = {
            "sql_file_path": "sql/session_1/task_1.sql",
            "sql_preview": "SELECT a\nFROM users\n-- ... (55 more lines)",
            "response": "Generated complex query",
            "tokens_used": 200,
        }
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["sql_file_path"] == "sql/session_1/task_1.sql"
        assert result.result["sql_preview"] == output["sql_preview"]
        assert result.result["response"] == "Generated complex query"
        assert result.result["tokens_used"] == 200
        assert "sql" not in result.result
        assert "sql_diff" not in result.result

    def test_file_based_sql_result_with_diff(self, task_tool):
        """Result with sql_file_path and sql_diff includes diff."""
        output = {
            "sql_file_path": "sql/session_1/task_1.sql",
            "sql_preview": "SELECT a, b\nFROM users",
            "sql_diff": "--- a/query.sql\n+++ b/query.sql\n@@ -1 +1 @@\n-SELECT a\n+SELECT a, b",
            "response": "Modified query",
            "tokens_used": 150,
        }
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["sql_file_path"] == "sql/session_1/task_1.sql"
        assert result.result["sql_diff"] == output["sql_diff"]

    def test_inline_sql_still_works(self, task_tool):
        """Short SQL still returns inline format (backward compatible)."""
        output = {"sql": "SELECT 1", "response": "simple query", "tokens_used": 50}
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["sql"] == "SELECT 1"
        assert "sql_file_path" not in result.result

    def test_file_path_takes_priority_over_sql(self, task_tool):
        """When both sql_file_path and sql are present, file path wins."""
        output = {
            "sql": "SELECT full query...",
            "sql_file_path": "sql/session_1/task_1.sql",
            "sql_preview": "SELECT ...",
            "response": "query",
            "tokens_used": 100,
        }
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert "sql_file_path" in result.result
        assert "sql" not in result.result


@pytest.mark.ci
class TestBuildTaskDescriptionFileStorage:
    """Tests for updated _build_task_description with file storage info."""

    def test_description_mentions_file_path(self, task_tool):
        desc = task_tool._build_task_description()
        assert "sql_file_path" in desc

    def test_description_mentions_read_file(self, task_tool):
        desc = task_tool._build_task_description()
        assert "read_file" in desc

    def test_description_mentions_diff(self, task_tool):
        desc = task_tool._build_task_description()
        assert "diff" in desc.lower()


# ── Built-in subagent: _get_available_types ────────────────────────


@pytest.mark.ci
class TestGetAvailableTypesBuiltIn:
    def test_includes_all_builtin_types(self, task_tool):
        """All SYS_SUB_AGENTS appear in available types, except 'feedback'
        which is a top-level node and not task()-delegatable."""
        types = task_tool._get_available_types()
        # feedback is a top-level node and must NEVER be exposed as delegatable.
        assert "feedback" not in types, "feedback must not be exposed as a delegatable subagent"
        for name in SYS_SUB_AGENTS:
            if name == "feedback":
                continue
            assert name in types, f"{name} not found in available types"

    def test_no_duplicates(self, task_tool):
        """No duplicates even if builtin names appear in agentic_nodes."""
        types = task_tool._get_available_types()
        assert len(types) == len(set(types))

    def test_no_duplicates_when_in_agentic_nodes(self):
        """Builtin types in agentic_nodes are not duplicated."""
        config = Mock(spec=AgentConfig)
        config.agentic_nodes = {
            "chat": {"model": "default"},
            "gen_sql_summary": {"model": "default", "node_class": "sql_summary"},
        }
        tool = SubAgentTaskTool(agent_config=config)
        types = tool._get_available_types()
        assert types.count("gen_sql_summary") == 1

    def test_builtin_types_sorted(self, task_tool):
        """Built-in types appear in sorted order after gen_sql (excluding 'feedback',
        which is not task()-delegatable)."""
        types = task_tool._get_available_types()
        builtin_in_list = [t for t in types if t in SYS_SUB_AGENTS]
        expected = sorted(name for name in SYS_SUB_AGENTS if name != "feedback")
        assert builtin_in_list == expected


# ── Built-in subagent: _resolve_node_type ──────────────────────────


@pytest.mark.ci
class TestResolveNodeTypeBuiltIn:
    def test_gen_semantic_model(self, task_tool):
        node_type, node_name = task_tool._resolve_node_type("gen_semantic_model")
        assert node_type == NodeType.TYPE_SEMANTIC
        assert node_name == "gen_semantic_model"

    def test_gen_metrics(self, task_tool):
        node_type, node_name = task_tool._resolve_node_type("gen_metrics")
        assert node_type == NodeType.TYPE_SEMANTIC
        assert node_name == "gen_metrics"

    def test_gen_sql_summary(self, task_tool):
        node_type, node_name = task_tool._resolve_node_type("gen_sql_summary")
        assert node_type == NodeType.TYPE_SQL_SUMMARY
        assert node_name == "gen_sql_summary"

    def test_gen_table(self, task_tool):
        node_type, node_name = task_tool._resolve_node_type("gen_table")
        assert node_type == NodeType.TYPE_GEN_TABLE
        assert node_name == "gen_table"

    def test_ask_metrics(self, task_tool):
        node_type, node_name = task_tool._resolve_node_type("ask_metrics")
        assert node_type == NodeType.TYPE_ASK_METRICS
        assert node_name == "ask_metrics"


# ── Built-in subagent: _create_builtin_node ────────────────────────


@pytest.mark.ci
class TestCreateBuiltinNode:
    @patch("datus.agent.node.gen_semantic_model_agentic_node.GenSemanticModelAgenticNode.__init__", return_value=None)
    def test_gen_semantic_model(self, mock_init, task_tool):
        task_tool._create_builtin_node("gen_semantic_model")
        mock_init.assert_called_once_with(
            agent_config=task_tool.agent_config,
            execution_mode="interactive",
            is_subagent=True,
            session_id=None,
        )

    @patch("datus.agent.node.gen_metrics_agentic_node.GenMetricsAgenticNode.__init__", return_value=None)
    def test_gen_metrics(self, mock_init, task_tool):
        task_tool._create_builtin_node("gen_metrics")
        mock_init.assert_called_once_with(
            agent_config=task_tool.agent_config,
            execution_mode="interactive",
            is_subagent=True,
            session_id=None,
        )

    @patch("datus.agent.node.sql_summary_agentic_node.SqlSummaryAgenticNode.__init__", return_value=None)
    def test_gen_sql_summary(self, mock_init, task_tool):
        task_tool._create_builtin_node("gen_sql_summary")
        mock_init.assert_called_once_with(
            node_name="gen_sql_summary",
            agent_config=task_tool.agent_config,
            execution_mode="interactive",
            is_subagent=True,
            session_id=None,
        )

    @patch("datus.agent.node.gen_table_agentic_node.GenTableAgenticNode.__init__", return_value=None)
    def test_gen_table(self, mock_init, task_tool):
        from unittest.mock import ANY

        task_tool._create_builtin_node("gen_table")
        mock_init.assert_called_once_with(
            agent_config=task_tool.agent_config,
            execution_mode="interactive",
            node_id=ANY,
            is_subagent=True,
            session_id=None,
        )

    @patch("datus.agent.node.gen_job_agentic_node.GenJobAgenticNode.__init__", return_value=None)
    def test_gen_job(self, mock_init, task_tool):
        task_tool._create_builtin_node("gen_job")
        mock_init.assert_called_once_with(
            agent_config=task_tool.agent_config,
            execution_mode="interactive",
            is_subagent=True,
            session_id=None,
        )

    @patch("datus.agent.node.gen_dashboard_agentic_node.GenDashboardAgenticNode.__init__", return_value=None)
    def test_gen_dashboard(self, mock_init, task_tool):
        from unittest.mock import ANY

        task_tool._create_builtin_node("gen_dashboard")
        mock_init.assert_called_once_with(
            agent_config=task_tool.agent_config,
            execution_mode="interactive",
            node_id=ANY,
            is_subagent=True,
            session_id=None,
        )

    @patch("datus.agent.node.scheduler_agentic_node.SchedulerAgenticNode.__init__", return_value=None)
    def test_scheduler(self, mock_init, task_tool):
        from unittest.mock import ANY

        task_tool._create_builtin_node("scheduler")
        mock_init.assert_called_once_with(
            agent_config=task_tool.agent_config,
            execution_mode="interactive",
            node_id=ANY,
            is_subagent=True,
            session_id=None,
        )

    def test_unknown_builtin_raises(self, task_tool):
        with pytest.raises(ValueError, match="Unknown builtin subagent type"):
            task_tool._create_builtin_node("nonexistent")

    def test_create_node_delegates_to_builtin(self, task_tool):
        """_create_node delegates to _create_builtin_node for SYS_SUB_AGENTS."""
        with patch.object(task_tool, "_create_builtin_node", return_value=Mock()) as mock_builtin:
            task_tool._create_node("gen_semantic_model")
            mock_builtin.assert_called_once_with("gen_semantic_model", session_id=None)

    def test_create_node_custom_passes_is_subagent_true(self, task_tool):
        """Custom agents must receive ``is_subagent=True`` via Node.new_instance.

        This enforces 2-level depth at the source: the child never constructs a
        SubAgentTaskTool, so there is nothing to strip post-construction.
        """
        with patch("datus.agent.node.node.Node.new_instance", return_value=Mock()) as mock_new_instance:
            task_tool._create_node("sales_analyst")

        mock_new_instance.assert_called_once()
        call_kwargs = mock_new_instance.call_args.kwargs
        assert call_kwargs["is_subagent"] is True
        assert call_kwargs["node_name"] == "sales_analyst"


# ── _resolve_execution_mode ─────────────────────────────────────────


@pytest.mark.ci
class TestResolveExecutionMode:
    def test_returns_interactive_when_no_parent(self, task_tool):
        assert task_tool._parent_node is None
        assert task_tool._resolve_execution_mode() == "interactive"

    def test_returns_parent_mode_workflow(self, task_tool):
        parent = Mock()
        parent.execution_mode = "workflow"
        task_tool.set_parent_node(parent)
        assert task_tool._resolve_execution_mode() == "workflow"

    def test_returns_parent_mode_interactive(self, task_tool):
        parent = Mock()
        parent.execution_mode = "interactive"
        task_tool.set_parent_node(parent)
        assert task_tool._resolve_execution_mode() == "interactive"

    def test_returns_interactive_when_parent_has_no_execution_mode(self, task_tool):
        parent = Mock(spec=[])  # no attributes
        task_tool._parent_node = parent
        assert task_tool._resolve_execution_mode() == "interactive"

    def test_returns_interactive_for_invalid_mode(self, task_tool):
        parent = Mock()
        parent.execution_mode = "unknown_mode"
        task_tool.set_parent_node(parent)
        assert task_tool._resolve_execution_mode() == "interactive"


@pytest.mark.ci
class TestBuiltinNodeInheritsExecutionMode:
    """Verify _create_builtin_node passes parent's execution_mode to subagent constructors."""

    @pytest.mark.parametrize(
        "subagent_type,init_path",
        [
            (
                "gen_semantic_model",
                "datus.agent.node.gen_semantic_model_agentic_node.GenSemanticModelAgenticNode.__init__",
            ),
            ("gen_metrics", "datus.agent.node.gen_metrics_agentic_node.GenMetricsAgenticNode.__init__"),
            ("gen_sql_summary", "datus.agent.node.sql_summary_agentic_node.SqlSummaryAgenticNode.__init__"),
            ("gen_table", "datus.agent.node.gen_table_agentic_node.GenTableAgenticNode.__init__"),
            ("gen_dashboard", "datus.agent.node.gen_dashboard_agentic_node.GenDashboardAgenticNode.__init__"),
            ("scheduler", "datus.agent.node.scheduler_agentic_node.SchedulerAgenticNode.__init__"),
        ],
    )
    def test_builtin_node_uses_workflow_mode(self, task_tool, subagent_type, init_path):
        parent = Mock()
        parent.execution_mode = "workflow"
        task_tool.set_parent_node(parent)

        with patch(init_path, return_value=None) as mock_init:
            task_tool._create_builtin_node(subagent_type)
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["execution_mode"] == "workflow"

    @pytest.mark.parametrize(
        "subagent_type,init_path",
        [
            ("gen_sql", "datus.agent.node.gen_sql_agentic_node.GenSQLAgenticNode.__init__"),
            ("ask_metrics", "datus.agent.node.ask_metrics_agentic_node.AskMetricsAgenticNode.__init__"),
            ("gen_report", "datus.agent.node.gen_report_agentic_node.GenReportAgenticNode.__init__"),
            ("gen_skill", "datus.agent.node.gen_skill_agentic_node.SkillCreatorAgenticNode.__init__"),
        ],
    )
    def test_builtin_node_with_extra_params_uses_workflow_mode(self, task_tool, subagent_type, init_path):
        parent = Mock()
        parent.execution_mode = "workflow"
        task_tool.set_parent_node(parent)

        with patch(init_path, return_value=None) as mock_init:
            task_tool._create_builtin_node(subagent_type)
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["execution_mode"] == "workflow"


# ── Built-in subagent: _build_node_input ───────────────────────────


@pytest.mark.ci
class TestBuildNodeInputBuiltIn:
    def test_semantic_model_node_input(self, task_tool):
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
        from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

        mock_node = Mock(spec=GenSemanticModelAgenticNode)
        result = task_tool._build_node_input(mock_node, "orders table")
        assert isinstance(result, SemanticNodeInput)
        assert result.user_message == "orders table"
        assert result.database is None

    def test_metrics_node_input(self, task_tool):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
        from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

        mock_node = Mock(spec=GenMetricsAgenticNode)
        result = task_tool._build_node_input(mock_node, "SELECT SUM(amount) FROM orders")
        assert isinstance(result, SemanticNodeInput)
        assert result.user_message == "SELECT SUM(amount) FROM orders"
        assert result.database is None

    def test_sql_summary_node_input(self, task_tool):
        from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode
        from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput

        mock_node = Mock(spec=SqlSummaryAgenticNode)
        result = task_tool._build_node_input(mock_node, "SELECT * FROM users")
        assert isinstance(result, SqlSummaryNodeInput)
        assert result.user_message == "SELECT * FROM users"
        assert result.database is None

    def test_gen_table_node_input(self, task_tool):
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode
        from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

        mock_node = Mock(spec=GenTableAgenticNode)
        result = task_tool._build_node_input(mock_node, "Create wide table from orders and customers")
        assert isinstance(result, SemanticNodeInput)
        assert result.user_message == "Create wide table from orders and customers"
        assert result.database is None

    def test_ask_metrics_node_input(self, task_tool):
        from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode
        from datus.schemas.ask_metrics_agentic_node_models import AskMetricsNodeInput

        mock_node = Mock(spec=AskMetricsAgenticNode)
        result = task_tool._build_node_input(mock_node, "Show active users")

        assert isinstance(result, AskMetricsNodeInput)
        assert result.user_message == "Show active users"
        assert result.database is None


# ── Built-in subagent: _build_task_description ─────────────────────


@pytest.mark.ci
class TestBuildTaskDescriptionBuiltIn:
    def test_contains_all_builtin_types(self, task_tool):
        desc = task_tool._build_task_description()
        # feedback is a top-level node; must NEVER be advertised to the LLM.
        assert "feedback" not in desc, "feedback must not appear in task description"
        for name in SYS_SUB_AGENTS:
            if name == "feedback":
                continue
            assert name in desc, f"{name} not found in task description"

    def test_contains_builtin_descriptions(self, task_tool):
        desc = task_tool._build_task_description()
        for name, builtin_desc in BUILTIN_SUBAGENT_DESCRIPTIONS.items():
            assert builtin_desc in desc, f"Description for {name} not found"

    def test_gen_semantic_model_description_content(self, task_tool):
        desc = task_tool._build_task_description()
        assert "semantic model" in desc.lower()
        assert "semantic_models" in desc

    def test_gen_sql_summary_description_content(self, task_tool):
        desc = task_tool._build_task_description()
        assert "sql_summary_file" in desc


# ── Built-in subagent: _convert_to_func_result ────────────────────


@pytest.mark.ci
class TestConvertToFuncResultBuiltIn:
    def test_semantic_models_result(self, task_tool):
        output = {
            "response": "Generated 2 models",
            "semantic_models": ["models/orders.yml", "models/customers.yml"],
            "tokens_used": 500,
        }
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["semantic_models"] == ["models/orders.yml", "models/customers.yml"]
        assert result.result["response"] == "Generated 2 models"
        assert result.result["tokens_used"] == 500

    def test_semantic_models_empty_list(self, task_tool):
        output = {
            "response": "No models generated",
            "semantic_models": [],
            "tokens_used": 100,
        }
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["semantic_models"] == []

    def test_sql_summary_file_result(self, task_tool):
        output = {
            "response": "Summarized query",
            "sql_summary_file": "knowledge/summaries/query_001.yml",
            "tokens_used": 300,
        }
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["sql_summary_file"] == "knowledge/summaries/query_001.yml"
        assert result.result["response"] == "Summarized query"

    def test_sql_file_path_takes_priority_over_semantic_models(self, task_tool):
        """sql_file_path still takes priority (checked first)."""
        output = {
            "sql_file_path": "sql/session/task.sql",
            "sql_preview": "SELECT ...",
            "semantic_models": ["models/x.yml"],
            "response": "test",
            "tokens_used": 100,
        }
        result = task_tool._convert_to_func_result(output)
        assert "sql_file_path" in result.result
        assert "semantic_models" not in result.result

    def test_dashboard_result(self, task_tool):
        """Dashboard result should preserve dashboard_result dict."""
        output = {
            "response": "Created dashboard",
            "dashboard_result": {"dashboard_id": 42, "url": "http://superset/dashboard/42"},
            "tokens_used": 600,
        }
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["dashboard_result"] == {"dashboard_id": 42, "url": "http://superset/dashboard/42"}
        assert result.result["response"] == "Created dashboard"
        assert result.result["tokens_used"] == 600

    def test_dashboard_result_empty_dict(self, task_tool):
        """Empty dashboard_result dict should still be preserved (not fall to generic)."""
        output = {
            "response": "No dashboard changes",
            "dashboard_result": {},
            "tokens_used": 100,
        }
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["dashboard_result"] == {}

    def test_scheduler_result(self, task_tool):
        """Scheduler result should preserve scheduler_result dict."""
        output = {
            "response": "Job submitted",
            "scheduler_result": {"job_id": "dag_123", "status": "scheduled"},
            "tokens_used": 300,
        }
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["scheduler_result"] == {"job_id": "dag_123", "status": "scheduled"}
        assert result.result["response"] == "Job submitted"
        assert result.result["tokens_used"] == 300

    def test_scheduler_result_empty_dict(self, task_tool):
        """Empty scheduler_result dict should still be preserved."""
        output = {
            "response": "No scheduler changes",
            "scheduler_result": {},
            "tokens_used": 50,
        }
        result = task_tool._convert_to_func_result(output)
        assert result.success == 1
        assert result.result["scheduler_result"] == {}


# ── Built-in subagent: end-to-end task execution ──────────────────


@pytest.mark.ci
class TestTaskExecutionBuiltIn:
    @pytest.mark.asyncio
    async def test_execute_gen_semantic_model(self, task_tool):
        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.ASSISTANT
        mock_action.output = {
            "response": "Generated semantic model for orders",
            "semantic_models": ["models/orders.yml"],
            "tokens_used": 400,
        }

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_semantic_model", prompt="orders table")

        assert result.success == 1
        assert result.result["semantic_models"] == ["models/orders.yml"]
        assert result.result["tokens_used"] == 400

    @pytest.mark.asyncio
    async def test_execute_gen_metrics(self, task_tool):
        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.ASSISTANT
        mock_action.output = {
            "response": "Extracted 3 metrics",
            "tokens_used": 350,
        }

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_metrics", prompt="SELECT SUM(amount) FROM orders")

        assert result.success == 1
        assert result.result["response"] == "Extracted 3 metrics"

    @pytest.mark.asyncio
    async def test_execute_gen_sql_summary(self, task_tool):
        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.ASSISTANT
        mock_action.output = {
            "response": "SQL summarized",
            "sql_summary_file": "knowledge/summaries/query_001.yml",
            "tokens_used": 250,
        }

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql_summary", prompt="SELECT * FROM users WHERE active = 1")

        assert result.success == 1
        assert result.result["sql_summary_file"] == "knowledge/summaries/query_001.yml"


# ── SubAgent complete action ──────────────────────────────────────


@pytest.mark.ci
class TestCompleteAction:
    """Tests for the subagent_complete action emitted by _execute_node."""

    @pytest.mark.asyncio
    async def test_complete_action_emitted_on_success(self, task_tool):
        """After a successful stream, the bus contains a subagent_complete action."""
        from datus.schemas.action_bus import ActionBus

        bus = ActionBus()
        task_tool.set_action_bus(bus)

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}
        mock_action.depth = 0
        mock_action.parent_action_id = None
        mock_action.role = ActionRole.TOOL
        mock_action.action_type = "tool_call"

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                await task_tool.task(type="gen_sql", prompt="test", call_id="call_123")

        # Collect all actions from the bus
        actions = []
        while not bus._queue.empty():
            actions.append(bus._queue.get_nowait())

        complete_actions = [a for a in actions if a.action_type == SUBAGENT_COMPLETE_ACTION_TYPE]
        assert len(complete_actions) == 1
        assert complete_actions[0].status == ActionStatus.SUCCESS
        assert complete_actions[0].parent_action_id == "call_123"
        assert complete_actions[0].depth == 1

    @pytest.mark.asyncio
    async def test_complete_action_emitted_on_failure(self, task_tool):
        """When the stream raises an exception, a FAILED complete action is still emitted."""
        from datus.schemas.action_bus import ActionBus

        bus = ActionBus()
        task_tool.set_action_bus(bus)

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield Mock(
                spec=ActionHistory,
                status=ActionStatus.PROCESSING,
                output=None,
                depth=0,
                parent_action_id=None,
                role=ActionRole.TOOL,
                action_type="tool_call",
            )
            raise RuntimeError("Stream error")

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="test", call_id="call_fail")

        assert result.success == 0

        # Collect all actions from the bus
        actions = []
        while not bus._queue.empty():
            actions.append(bus._queue.get_nowait())

        complete_actions = [a for a in actions if a.action_type == SUBAGENT_COMPLETE_ACTION_TYPE]
        assert len(complete_actions) == 1
        assert complete_actions[0].status == ActionStatus.FAILED
        assert complete_actions[0].parent_action_id == "call_fail"

    @pytest.mark.asyncio
    async def test_complete_action_not_emitted_without_bus(self, task_tool):
        """Without an ActionBus, no complete action is emitted and no error occurs."""
        assert task_tool._action_bus is None

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.TOOL
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="test")

        assert result.success == 1

    @pytest.mark.asyncio
    async def test_complete_action_metadata(self, task_tool):
        """The complete action output contains subagent_type and tool_count."""
        from datus.schemas.action_bus import ActionBus

        bus = ActionBus()
        task_tool.set_action_bus(bus)

        # Two TOOL actions to count
        tool_action_1 = Mock(spec=ActionHistory)
        tool_action_1.status = ActionStatus.SUCCESS
        tool_action_1.output = None
        tool_action_1.depth = 0
        tool_action_1.parent_action_id = None
        tool_action_1.role = ActionRole.TOOL
        tool_action_1.action_type = "describe_table"

        tool_action_2 = Mock(spec=ActionHistory)
        tool_action_2.status = ActionStatus.SUCCESS
        tool_action_2.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}
        tool_action_2.depth = 0
        tool_action_2.parent_action_id = None
        tool_action_2.role = ActionRole.TOOL
        tool_action_2.action_type = "read_query"

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield tool_action_1
            yield tool_action_2

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                await task_tool.task(type="gen_sql", prompt="test", call_id="call_meta")

        # Collect all actions from the bus
        actions = []
        while not bus._queue.empty():
            actions.append(bus._queue.get_nowait())

        complete_actions = [a for a in actions if a.action_type == SUBAGENT_COMPLETE_ACTION_TYPE]
        assert len(complete_actions) == 1
        assert complete_actions[0].output["subagent_type"] == "gen_sql"
        assert complete_actions[0].output["tool_count"] == 2


# ── Description parameter ────────────────────────────────────────


@pytest.mark.ci
class TestDescriptionParameter:
    """Tests for the optional 'description' parameter on the task tool."""

    def test_schema_contains_description_property(self, task_tool):
        """Schema includes a 'description' property."""
        tools = task_tool.available_tools()
        schema = tools[0].params_json_schema
        assert "description" in schema["properties"]
        assert schema["properties"]["description"]["type"] == "string"

    def test_description_is_required(self, task_tool):
        """'description' IS in the required list."""
        tools = task_tool.available_tools()
        schema = tools[0].params_json_schema
        assert "description" in schema["required"]

    @pytest.mark.asyncio
    async def test_description_injected_into_first_user_action(self, task_tool):
        """When description is provided, it is injected into the first USER action's input."""
        from datus.schemas.action_bus import ActionBus

        bus = ActionBus()
        task_tool.set_action_bus(bus)

        user_action = Mock(spec=ActionHistory)
        user_action.role = ActionRole.USER
        user_action.status = ActionStatus.SUCCESS
        user_action.output = None
        user_action.depth = 0
        user_action.parent_action_id = None
        user_action.action_type = "user_message"
        user_action.input = {}
        user_action.messages = "Show all users"

        tool_action = Mock(spec=ActionHistory)
        tool_action.role = ActionRole.TOOL
        tool_action.status = ActionStatus.SUCCESS
        tool_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}
        tool_action.depth = 0
        tool_action.parent_action_id = None
        tool_action.action_type = "tool_call"

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield user_action
            yield tool_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                await task_tool.task(
                    type="gen_sql", prompt="Show all users", description="List all users from database"
                )

        # The user_action's input should now contain _task_description
        assert user_action.input["_task_description"] == "List all users from database"

    @pytest.mark.asyncio
    async def test_description_not_injected_when_empty(self, task_tool):
        """When description is empty, no _task_description is added."""
        from datus.schemas.action_bus import ActionBus

        bus = ActionBus()
        task_tool.set_action_bus(bus)

        user_action = Mock(spec=ActionHistory)
        user_action.role = ActionRole.USER
        user_action.status = ActionStatus.SUCCESS
        user_action.output = None
        user_action.depth = 0
        user_action.parent_action_id = None
        user_action.action_type = "user_message"
        user_action.input = {}
        user_action.messages = "Show all users"

        tool_action = Mock(spec=ActionHistory)
        tool_action.role = ActionRole.TOOL
        tool_action.status = ActionStatus.SUCCESS
        tool_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}
        tool_action.depth = 0
        tool_action.parent_action_id = None
        tool_action.action_type = "tool_call"

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield user_action
            yield tool_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                await task_tool.task(type="gen_sql", prompt="Show all users", description="")

        # No _task_description should be injected
        assert "_task_description" not in user_action.input

    @pytest.mark.asyncio
    async def test_description_only_injected_into_first_user_action(self, task_tool):
        """Description is injected only into the first USER action, not subsequent ones."""
        from datus.schemas.action_bus import ActionBus

        bus = ActionBus()
        task_tool.set_action_bus(bus)

        user_action_1 = Mock(spec=ActionHistory)
        user_action_1.role = ActionRole.USER
        user_action_1.status = ActionStatus.SUCCESS
        user_action_1.output = None
        user_action_1.depth = 0
        user_action_1.parent_action_id = None
        user_action_1.action_type = "user_message"
        user_action_1.input = {}
        user_action_1.messages = "First message"

        user_action_2 = Mock(spec=ActionHistory)
        user_action_2.role = ActionRole.USER
        user_action_2.status = ActionStatus.SUCCESS
        user_action_2.output = None
        user_action_2.depth = 0
        user_action_2.parent_action_id = None
        user_action_2.action_type = "user_message"
        user_action_2.input = {}
        user_action_2.messages = "Second message"

        tool_action = Mock(spec=ActionHistory)
        tool_action.role = ActionRole.TOOL
        tool_action.status = ActionStatus.SUCCESS
        tool_action.output = {"response": "ok", "tokens_used": 10}
        tool_action.depth = 0
        tool_action.parent_action_id = None
        tool_action.action_type = "tool_call"

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield user_action_1
            yield user_action_2
            yield tool_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                await task_tool.task(type="gen_sql", prompt="First message", description="Task goal")

        assert user_action_1.input["_task_description"] == "Task goal"
        assert "_task_description" not in user_action_2.input

    @pytest.mark.asyncio
    async def test_description_injected_when_input_is_none(self, task_tool):
        """When the first USER action has input=None, a dict is created for the description."""
        from datus.schemas.action_bus import ActionBus

        bus = ActionBus()
        task_tool.set_action_bus(bus)

        user_action = Mock(spec=ActionHistory)
        user_action.role = ActionRole.USER
        user_action.status = ActionStatus.SUCCESS
        user_action.output = None
        user_action.depth = 0
        user_action.parent_action_id = None
        user_action.action_type = "user_message"
        user_action.input = None
        user_action.messages = "Show all users"

        tool_action = Mock(spec=ActionHistory)
        tool_action.role = ActionRole.TOOL
        tool_action.status = ActionStatus.SUCCESS
        tool_action.output = {"response": "ok", "tokens_used": 10}
        tool_action.depth = 0
        tool_action.parent_action_id = None
        tool_action.action_type = "tool_call"

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield user_action
            yield tool_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                await task_tool.task(type="gen_sql", prompt="Show all users", description="List users")

        assert user_action.input["_task_description"] == "List users"

    def test_build_task_description_mentions_description_guideline(self, task_tool):
        """The guidelines mention providing a 'description' parameter."""
        desc = task_tool._build_task_description()
        assert "description" in desc.lower()


# ── Proxy tool propagation ─────────────────────────────────────────


@pytest.mark.ci
class TestProxyToolPropagation:
    """Tests for proxy tool config propagation to sub-agent nodes via parent node reference."""

    def test_set_parent_node_stores_reference(self, task_tool):
        """set_parent_node stores the parent node reference."""
        parent_node = MagicMock()
        task_tool.set_parent_node(parent_node)
        assert task_tool._parent_node is parent_node

    def test_default_parent_node_is_none(self, task_tool):
        """By default, _parent_node is None."""
        assert task_tool._parent_node is None

    @pytest.mark.asyncio
    async def test_apply_proxy_tools_called_when_parent_has_patterns(self, task_tool):
        """When parent node has proxy_tool_patterns, apply_proxy_tools is called on the sub-agent node."""
        from datus.tools.proxy.tool_result_channel import ToolResultChannel

        parent_node = MagicMock()
        parent_node.proxy_tool_patterns = ["filesystem_tools.*"]
        parent_node.tool_channel = ToolResultChannel()
        task_tool.set_parent_node(parent_node)

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.TOOL
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                with patch("datus.tools.proxy.proxy_tool.apply_proxy_tools") as mock_apply:
                    result = await task_tool.task(type="gen_sql", prompt="test")

        mock_apply.assert_called_once_with(mock_node, parent_node.proxy_tool_patterns, channel=parent_node.tool_channel)
        assert result.success == 1

    @pytest.mark.asyncio
    async def test_apply_proxy_tools_not_called_when_parent_has_no_patterns(self, task_tool):
        """When parent node has no proxy_tool_patterns, apply_proxy_tools is NOT called."""
        parent_node = MagicMock()
        parent_node.proxy_tool_patterns = None
        task_tool.set_parent_node(parent_node)

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.TOOL
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                with patch("datus.tools.proxy.proxy_tool.apply_proxy_tools") as mock_apply:
                    result = await task_tool.task(type="gen_sql", prompt="test")

        mock_apply.assert_not_called()
        assert result.success == 1

    @pytest.mark.asyncio
    async def test_apply_proxy_tools_not_called_when_no_parent_node(self, task_tool):
        """When _parent_node is None, apply_proxy_tools is NOT called."""
        assert task_tool._parent_node is None

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.TOOL
        mock_action.output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                with patch("datus.tools.proxy.proxy_tool.apply_proxy_tools") as mock_apply:
                    result = await task_tool.task(type="gen_sql", prompt="test")

        mock_apply.assert_not_called()
        assert result.success == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "subagent_type",
        ["gen_semantic_model", "gen_metrics", "gen_sql_summary"],
    )
    async def test_fs_dependent_types_still_call_apply_proxy(self, task_tool, subagent_type):
        """FS-dependent subagents still call apply_proxy_tools (exclusion is internal to proxy_tool)."""
        from datus.tools.proxy.tool_result_channel import ToolResultChannel

        parent_node = MagicMock()
        parent_node.proxy_tool_patterns = ["*"]
        parent_node.tool_channel = ToolResultChannel()
        task_tool.set_parent_node(parent_node)

        mock_action = Mock(spec=ActionHistory)
        mock_action.status = ActionStatus.SUCCESS
        mock_action.role = ActionRole.ASSISTANT
        mock_action.output = {"response": "ok", "tokens_used": 10}

        mock_node = MagicMock()

        async def mock_stream(ahm):
            yield mock_action

        mock_node.execute_stream_with_interactions = mock_stream

        with patch.object(task_tool, "_create_node", return_value=mock_node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                with patch("datus.tools.proxy.proxy_tool.apply_proxy_tools") as mock_apply:
                    result = await task_tool.task(type=subagent_type, prompt="test")

        mock_apply.assert_called_once_with(mock_node, parent_node.proxy_tool_patterns, channel=parent_node.tool_channel)
        assert result.success == 1


# ── session persistence + resume ───────────────────────────────────


def _build_persistent_mock_node(
    *,
    node_name: str = "gen_sql",
    session_id_to_assign: str = "gen_sql_session_abc12345",
    output: dict = None,
    status: ActionStatus = ActionStatus.SUCCESS,
    role: ActionRole = ActionRole.TOOL,
):
    """Construct a MagicMock that behaves like an AgenticNode for task-tool tests.

    The mock yields one action through ``execute_stream_with_interactions`` and
    exposes the AgenticNode surface that ``_execute_node`` touches:
    ``session_id``, ``session_subdir``, ``_session``, ``_session_manager``,
    ``session_manager.session_exists``, ``get_node_name``.
    """
    if output is None:
        output = {"sql": "SELECT 1", "response": "ok", "tokens_used": 10}

    mock_action = Mock(spec=ActionHistory)
    mock_action.status = status
    mock_action.role = role
    mock_action.output = output

    mock_node = MagicMock()
    mock_node.session_id = session_id_to_assign
    mock_node.session_subdir = None
    mock_node._session = None
    mock_node._session_manager = MagicMock()
    mock_node.session_manager = mock_node._session_manager
    mock_node.session_manager.session_exists.return_value = True
    mock_node.get_node_name.return_value = node_name

    async def _mock_stream(ahm):
        yield mock_action

    mock_node.execute_stream_with_interactions = _mock_stream
    mock_node.execute_stream = _mock_stream
    return mock_node


@pytest.mark.ci
class TestSessionPersistence:
    @pytest.mark.asyncio
    async def test_returns_session_id_in_result(self, task_tool):
        """A successful task() result must include the subagent's session_id."""
        node = _build_persistent_mock_node(session_id_to_assign="gen_sql_session_abc12345")
        with patch.object(task_tool, "_create_node", return_value=node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="show tables", description="explore schema")
        assert result.success == 1
        assert result.result["session_id"] == "gen_sql_session_abc12345"
        # Existing keys still intact
        assert result.result["sql"] == "SELECT 1"

    @pytest.mark.asyncio
    async def test_disk_path_includes_parent_session_id(self, task_tool):
        """When the parent has a session_id, node.session_subdir is set so the
        subagent .db nests under {sessions_dir}/{user_scope}/{parent_id}/."""
        parent = MagicMock()
        parent.session_id = "chat_session_parent01"
        parent.proxy_tool_patterns = None
        task_tool.set_parent_node(parent)

        node = _build_persistent_mock_node()
        with patch.object(task_tool, "_create_node", return_value=node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="test")

        assert result.success == 1
        assert node.session_subdir == "chat_session_parent01"

    @pytest.mark.asyncio
    async def test_no_parent_session_falls_back_to_flat_path(self, task_tool):
        """Parent without a session_id falls back to a flat layout (no nesting)."""
        # task_tool fixture has no parent_node set by default.
        node = _build_persistent_mock_node()
        with patch.object(task_tool, "_create_node", return_value=node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="test")
        assert result.success == 1
        # Without a parent session_id, session_subdir stays None.
        assert node.session_subdir is None
        # The returned session_id is still valid so the caller can resume later.
        assert result.result["session_id"] == "gen_sql_session_abc12345"

    @pytest.mark.asyncio
    async def test_resume_loads_prior_session(self, task_tool):
        """A valid session_id passes through and is set on the node before
        _get_or_create_session runs (so AgenticNode loads from disk)."""
        node = _build_persistent_mock_node(session_id_to_assign="gen_sql_session_resume01")
        with patch.object(task_tool, "_create_node", return_value=node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(
                    type="gen_sql",
                    prompt="refine: use INNER JOIN",
                    description="refinement",
                    session_id="gen_sql_session_resume01",
                )
        # session_exists was consulted on the node-owned manager
        node.session_manager.session_exists.assert_called_once_with("gen_sql_session_resume01")
        # The pre-existing id was assigned (caller-supplied wins over auto-gen)
        assert node.session_id == "gen_sql_session_resume01"
        assert result.success == 1
        assert result.result["session_id"] == "gen_sql_session_resume01"

    @pytest.mark.asyncio
    async def test_resume_prefix_mismatch_returns_error(self, task_tool):
        """type='gen_sql' with a session_id from gen_report must be rejected."""
        node = _build_persistent_mock_node()
        with patch.object(task_tool, "_create_node", return_value=node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="x", session_id="gen_report_session_xx123456")
        assert result.success == 0
        assert "belongs to subagent type" in result.error
        assert "gen_report" in result.error

    @pytest.mark.asyncio
    async def test_resume_invalid_format_returns_error(self, task_tool):
        """Path-traversal / illegal characters fail the session_id format check."""
        result = await task_tool.task(type="gen_sql", prompt="x", session_id="../etc/passwd")
        assert result.success == 0
        assert "Invalid session_id format" in result.error

    @pytest.mark.asyncio
    async def test_resume_missing_file_returns_error(self, task_tool):
        """A well-formed, prefix-matching session_id whose .db is absent on disk
        is rejected with a clear 'not found' error rather than silently starting fresh."""
        node = _build_persistent_mock_node()
        node.session_manager.session_exists.return_value = False
        with patch.object(task_tool, "_create_node", return_value=node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="x", session_id="gen_sql_session_missing1")
        assert result.success == 0
        assert "not found on disk" in result.error

    @pytest.mark.asyncio
    async def test_failure_envelope_carries_session_id(self, task_tool):
        """Subagent failures still expose session_id under `result` so the parent
        can resume the partial subagent session (its .db is kept alive on disk)."""
        node = _build_persistent_mock_node(
            session_id_to_assign="gen_sql_session_failed01",
            output={"success": False, "error": "broken"},
            status=ActionStatus.FAILED,
        )
        with patch.object(task_tool, "_create_node", return_value=node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="x")
        assert result.success == 0
        assert result.error  # subagent error surfaced
        assert result.result == {"session_id": "gen_sql_session_failed01"}

    @pytest.mark.asyncio
    async def test_stream_exception_returns_session_id(self, task_tool):
        """Mid-stream exceptions (e.g. MaxTurnsExceeded) must NOT propagate out
        of the task tool. They are converted into a failure envelope that still
        carries the subagent's session_id so the parent can resume the partial
        run with `task(session_id=..., prompt=...)`."""
        node = _build_persistent_mock_node(session_id_to_assign="gen_sql_session_maxturn1")

        async def _exploding_stream(_ahm):
            # Yield nothing — raise immediately, the way Runner propagates
            # MaxTurnsExceeded wrapped as DatusException upward.
            raise RuntimeError("Max turns exceeded: 30")
            yield  # pragma: no cover — make this a generator function

        node.execute_stream = _exploding_stream
        node.execute_stream_with_interactions = _exploding_stream

        with patch.object(task_tool, "_create_node", return_value=node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                result = await task_tool.task(type="gen_sql", prompt="long task")

        assert result.success == 0
        assert "Subagent stream failed" in result.error
        assert "Max turns exceeded" in result.error
        assert result.result == {"session_id": "gen_sql_session_maxturn1"}
        # Session handle release still ran in finally — partial .db survives on disk.
        node._session_manager.close_all_sessions.assert_called_once()
        node._session_manager.delete_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_node_creation_failure_omits_session_id(self, task_tool):
        """When the node never gets created (e.g. _create_node raises), there is
        no session_id to surface — the failure envelope's `result` stays None."""
        with patch.object(task_tool, "_create_node", side_effect=RuntimeError("init boom")):
            result = await task_tool.task(type="gen_sql", prompt="x")
        assert result.success == 0
        assert "Task execution failed" in result.error
        assert result.result is None

    @pytest.mark.asyncio
    async def test_finally_does_not_delete_session(self, task_tool):
        """The finally block must NOT call delete_session — the .db is kept
        alive for resume on later turns."""
        node = _build_persistent_mock_node()
        with patch.object(task_tool, "_create_node", return_value=node):
            with patch.object(task_tool, "_build_node_input", return_value=Mock()):
                await task_tool.task(type="gen_sql", prompt="test")
        # delete_session must never have been called on either the node or its manager.
        node.delete_session.assert_not_called()
        node._session_manager.delete_session.assert_not_called()
        # close_all_sessions IS expected — we release the in-memory handle.
        node._session_manager.close_all_sessions.assert_called_once()

    @pytest.mark.asyncio
    async def test_schema_advertises_session_id_parameter(self, task_tool):
        """The LLM-facing schema must expose `session_id` (optional) so models
        learn they can resume a prior subagent conversation."""
        tools = task_tool.available_tools()
        schema = tools[0].params_json_schema
        assert "session_id" in schema["properties"]
        # Optional — must not be in required list.
        assert "session_id" not in schema.get("required", [])
        # Description guides the LLM to the iterative-refinement use case.
        desc = schema["properties"]["session_id"]["description"]
        assert "continue" in desc.lower()
        assert "type" in desc.lower()  # callout that types must match


@pytest.mark.ci
class TestSubagentInheritsParentDatabase:
    """A subagent spawned via the task tool must inherit the parent node's resolved
    physical database (and catalog/schema where supported).

    Regression guard: for a multi-database glob (``path_pattern``) datasource the default
    database is an arbitrary first-matched file, so a subagent that does NOT inherit the
    parent's database silently explores the wrong one — the root cause of a benchmark task
    delegating to ``explore`` and hanging on an unrelated bird database.
    """

    @staticmethod
    def _parent(database="", catalog="", db_schema="", connector_db=None):
        parent = MagicMock()
        parent.input.database = database
        parent.input.catalog = catalog
        parent.input.db_schema = db_schema
        if connector_db is None:
            parent.db_func_tool = None
        else:
            parent.db_func_tool.connector.database_name = connector_db
        return parent

    def test_parent_db_context_reads_parent_input(self, task_tool):
        task_tool._parent_node = self._parent(database="california_schools", catalog="cat", db_schema="sch")
        assert task_tool._parent_db_context() == {
            "database": "california_schools",
            "catalog": "cat",
            "db_schema": "sch",
        }

    def test_parent_db_context_falls_back_to_connector_database_name(self, task_tool):
        # Parent set no explicit database → use the connector's real database name, not the
        # datasource default (resolve_database_name_for_prompt contract).
        task_tool._parent_node = self._parent(database="", connector_db="california_schools")
        assert task_tool._parent_db_context()["database"] == "california_schools"

    def test_parent_db_context_ignores_mock_connector_database_name(self, task_tool):
        parent = MagicMock()
        parent.input.database = ""
        parent.input.catalog = None
        parent.input.db_schema = None
        task_tool._parent_node = parent

        assert task_tool._parent_db_context() == {
            "database": None,
            "catalog": None,
            "db_schema": None,
        }

    def test_parent_db_context_no_parent_returns_empty(self, task_tool):
        task_tool._parent_node = None
        assert task_tool._parent_db_context() == {}

    def test_apply_db_context_skips_fields_the_input_lacks(self):
        # ExploreNodeInput declares only `database`; catalog/db_schema must be skipped, not error.
        class OnlyDatabase:
            database = None

        obj = SubAgentTaskTool._apply_db_context(OnlyDatabase(), {"database": "db1", "catalog": "c", "db_schema": "s"})
        assert obj.database == "db1"
        assert not hasattr(obj, "catalog")

    def test_apply_db_context_sets_all_declared_fields(self):
        from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput

        ctx = {"database": "california_schools", "catalog": "cat", "db_schema": "sch"}
        gen = SubAgentTaskTool._apply_db_context(GenSQLNodeInput(user_message="x"), ctx)
        assert (gen.database, gen.catalog, gen.db_schema) == ("california_schools", "cat", "sch")

    def test_build_node_input_explore_inherits_parent_database(self, real_agent_config, mock_llm_create):
        from datus.agent.node.explore_agentic_node import ExploreAgenticNode
        from datus.schemas.explore_agentic_node_models import ExploreNodeInput

        tool = SubAgentTaskTool(agent_config=real_agent_config)
        tool._parent_node = self._parent(database="california_schools")
        node = ExploreAgenticNode(
            node_id="t",
            description="d",
            node_type=NodeType.TYPE_EXPLORE,
            agent_config=real_agent_config,
            node_name="explore",
        )
        inp = tool._build_node_input(node, "explore schools")
        assert isinstance(inp, ExploreNodeInput)
        assert inp.database == "california_schools"

    def test_build_node_input_explore_no_parent_leaves_database_unset(self, real_agent_config, mock_llm_create):
        from datus.agent.node.explore_agentic_node import ExploreAgenticNode

        tool = SubAgentTaskTool(agent_config=real_agent_config)
        tool._parent_node = None
        node = ExploreAgenticNode(
            node_id="t",
            description="d",
            node_type=NodeType.TYPE_EXPLORE,
            agent_config=real_agent_config,
            node_name="explore",
        )
        inp = tool._build_node_input(node, "explore schools")
        assert inp.database is None

    def test_refresh_node_db_tools_rebuilds_when_database_inherited(self):
        node = MagicMock()
        node.input.database = "california_schools"
        node.db_func_tool = MagicMock()

        SubAgentTaskTool._refresh_node_db_tools(
            node,
            {"database": "california_schools", "catalog": None, "db_schema": None},
        )

        node.setup_tools.assert_called_once_with()

    def test_refresh_node_db_tools_preserves_ask_user_after_setup(self):
        read_query = Mock()
        read_query.name = "read_query"
        ask_user = Mock()
        ask_user.name = "ask_user"

        node = MagicMock()
        node.input.database = "california_schools"
        node.db_func_tool = MagicMock()
        node.tools = [read_query, ask_user]
        node.ask_user_tool.available_tools.return_value = [ask_user]

        def reset_tools():
            node.tools = [read_query]

        node.setup_tools.side_effect = reset_tools

        SubAgentTaskTool._refresh_node_db_tools(
            node,
            {"database": "california_schools", "catalog": None, "db_schema": None},
        )

        assert [tool.name for tool in node.tools] == ["read_query", "ask_user"]

    def test_refresh_node_db_tools_skips_when_database_absent(self):
        node = MagicMock()
        node.input.database = None
        node.db_func_tool = MagicMock()

        SubAgentTaskTool._refresh_node_db_tools(
            node,
            {"database": None, "catalog": None, "db_schema": None},
        )

        node.setup_tools.assert_not_called()
