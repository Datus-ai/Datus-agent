# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for AgenticNode base class.

CI-level: zero external deps, zero network, zero API keys.
Uses _ConcreteAgenticNode (minimal concrete subclass) and patches LLM + sessions.
"""

import asyncio
import os
from typing import AsyncGenerator, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.agent.node.agentic_node import AgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.token_usage import TokenUsage

_UNSET = object()

# ---------------------------------------------------------------------------
# Concrete subclass for testing (can't instantiate abstract AgenticNode directly)
# ---------------------------------------------------------------------------


class _ConcreteAgenticNode(AgenticNode):
    """Minimal concrete implementation of AgenticNode for testing."""

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        action = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="test",
            messages="test response",
            input_data={},
            output_data={"success": True, "result": "done"},
            status=ActionStatus.SUCCESS,
        )
        yield action


def _make_async_session_mock() -> MagicMock:
    """Build a session-like mock whose async methods are AsyncMocks.

    Used by _manual_compact tests because the production code now awaits
    `clear_session` and `add_items` on `self._session` after generating the
    summary.
    """
    sess = MagicMock()
    sess.clear_session = AsyncMock()
    sess.add_items = AsyncMock()
    return sess


def _make_node(agent_config=None, context_length=_UNSET, **overrides):
    """Create a node with __init__ bypassed for targeted testing."""
    with patch.object(AgenticNode, "__init__", lambda self, *a, **kw: None):
        node = _ConcreteAgenticNode.__new__(_ConcreteAgenticNode)
    # Set minimum required attributes (backing fields for properties first)
    node._agent_config_ref = None
    node._pinned_model = None
    node._node_model_name = None
    node._session = None
    node.session_id = None
    node.tools = []
    node.mcp_servers = {}
    node.actions = []

    node.node_config = {}
    node.agent_config = agent_config
    node.skill_manager = None
    node.skill_func_tool = None
    node.permission_manager = None
    node._permission_callback = None
    node.result = None
    node.input = None
    node.type = "test"
    from datus.cli.execution_state import InteractionBroker, InterruptController
    from datus.schemas.action_bus import ActionBus

    node.action_bus = ActionBus()
    node.interaction_broker = InteractionBroker()
    node.interrupt_controller = InterruptController()
    if context_length is not _UNSET and context_length is not None:
        mock_model = MagicMock()
        mock_model.context_length.return_value = context_length
        node._pinned_model = mock_model
    for k, v in overrides.items():
        setattr(node, k, v)
    return node


# ---------------------------------------------------------------------------
# TestGetNodeName
# ---------------------------------------------------------------------------


class TestGetNodeName:
    def test_concrete_node_name_derived_from_class(self):
        """get_node_name strips 'AgenticNode' suffix and lowercases."""
        node = _make_node()
        assert node.get_node_name() == "_concrete"

    def test_node_name_for_specific_class(self):
        """Verify the naming pattern with a well-named subclass."""
        # GenMetricsAgenticNode -> "gen_metrics" (tested via real class)
        # For our concrete class: _ConcreteAgenticNode -> "_concrete"
        node = _make_node()
        name = node.get_node_name()
        assert isinstance(name, str)
        assert name == "_concrete"


# ---------------------------------------------------------------------------
# TestParseNodeConfig
# ---------------------------------------------------------------------------


class TestParseNodeConfig:
    def test_returns_empty_when_no_agent_config(self):
        node = _make_node()
        result = node._parse_node_config(None, "mynode")
        assert result == {}

    def test_returns_empty_when_node_not_in_config(self):
        cfg = MagicMock(spec=AgentConfig)
        cfg.agentic_nodes = {"other_node": {"model": "gpt-4"}}
        node = _make_node()
        result = node._parse_node_config(cfg, "mynode")
        assert result == {}

    def test_parses_model_from_dict(self):
        cfg = MagicMock(spec=AgentConfig)
        cfg.agentic_nodes = {"mynode": {"model": "gpt-4o", "max_turns": 10}}
        node = _make_node()
        result = node._parse_node_config(cfg, "mynode")
        assert result.get("model") == "gpt-4o"

    def test_normalizes_rules_list(self):
        """Rules with dict items are converted to 'key: value' strings."""
        cfg = MagicMock(spec=AgentConfig)
        cfg.agentic_nodes = {
            "mynode": {
                "rules": [{"always": "be concise"}, "plain rule"],
            }
        }
        node = _make_node()
        result = node._parse_node_config(cfg, "mynode")
        rules = result.get("rules", [])
        assert any("always: be concise" in r for r in rules)
        assert "plain rule" in rules

    def test_returns_empty_when_no_agentic_nodes_attr(self):
        cfg = MagicMock()
        del cfg.agentic_nodes  # remove the attribute
        node = _make_node()
        result = node._parse_node_config(cfg, "mynode")
        assert result == {}


# ---------------------------------------------------------------------------
# TestResolveWorkspaceRoot
# ---------------------------------------------------------------------------


class TestResolveWorkspaceRoot:
    def test_returns_cwd_when_no_config(self):
        node = _make_node()
        result = node._resolve_workspace_root()
        assert result == os.getcwd()

    def test_uses_node_config_workspace_root(self):
        node = _make_node()
        node.node_config = {"workspace_root": "/custom/path"}
        node.agent_config = None
        result = node._resolve_workspace_root()
        assert result == "/custom/path"

    def test_expands_tilde(self, tmp_path):
        node = _make_node()
        node.node_config = {"workspace_root": "~/testdir"}
        node.agent_config = None
        result = node._resolve_workspace_root()
        import os

        assert not result.startswith("~")
        assert os.path.expanduser("~/testdir") == result

    def test_uses_project_root(self):
        node = _make_node()
        node.node_config = {}
        cfg = MagicMock(spec=["project_root"])
        cfg.project_root = "/project/root"
        node.agent_config = cfg
        result = node._resolve_workspace_root()
        assert result == "/project/root"

    def test_vscode_source_short_circuits_to_dot(self):
        """vscode owns its own filesystem; the resolver returns the literal "."
        so the daemon's project_root never leaks to the IDE."""
        node = _make_node()
        node.node_config = {"workspace_root": "/should-be-ignored"}
        cfg = MagicMock(spec=["project_root", "_client_source"])
        cfg.project_root = "/server/cwd"
        cfg._client_source = "vscode"
        node.agent_config = cfg
        assert node._resolve_workspace_root() == "."

    def test_web_source_keeps_real_project_root(self):
        node = _make_node()
        node.node_config = {}
        cfg = MagicMock(spec=["project_root", "_client_source"])
        cfg.project_root = "/server/cwd"
        cfg._client_source = "web"
        node.agent_config = cfg
        assert node._resolve_workspace_root() == "/server/cwd"


# ---------------------------------------------------------------------------
# TestGetSystemPromptWorkspaceRoot — vscode source must render "." literally
# ---------------------------------------------------------------------------


class TestGetSystemPromptWorkspaceRoot:
    """``get_system_prompt`` passes ``workspace_root`` to the Jinja template,
    which renders the "Current sql files root directory" hint. For
    ``_client_source == "vscode"`` we render the literal "." so the daemon's
    project_root never leaks to a remote IDE that owns its own filesystem.
    Other sources (web, CLI, None) keep the resolved project_root.
    """

    def _prepare_node(self, monkeypatch, project_root, client_source):
        captured = {}

        class _FakePromptManager:
            def render_template(self, **kwargs):
                captured.update(kwargs)
                return "rendered"

        monkeypatch.setattr(
            "datus.agent.node.agentic_node.get_prompt_manager",
            lambda **_: _FakePromptManager(),
        )

        node = _make_node()
        # ``_finalize_system_prompt`` reaches into bash_tool / skill_func_tool /
        # memory state that the bypassed __init__ never set up. Short-circuit
        # it: the assertion only cares about what ``render_template`` was
        # given, not the post-processing.
        node._finalize_system_prompt = lambda base_prompt, **_: base_prompt  # type: ignore[method-assign]

        spec_attrs = ["project_root", "prompt_version", "current_datasource"]
        if client_source is not None:
            spec_attrs.append("_client_source")
        cfg = MagicMock(spec=spec_attrs)
        cfg.project_root = project_root
        cfg.prompt_version = None
        cfg.current_datasource = None
        if client_source is not None:
            cfg._client_source = client_source
        node.agent_config = cfg
        return node, captured

    def test_vscode_source_renders_dot(self, monkeypatch):
        node, captured = self._prepare_node(monkeypatch, project_root="/server/cwd", client_source="vscode")
        node._get_system_prompt()
        assert captured["workspace_root"] == "."

    def test_web_source_keeps_real_project_root(self, monkeypatch):
        node, captured = self._prepare_node(monkeypatch, project_root="/server/cwd", client_source="web")
        node._get_system_prompt()
        assert captured["workspace_root"] == "/server/cwd"

    def test_no_client_source_keeps_real_project_root(self, monkeypatch):
        """CLI / no-source path renders the concrete project_root (existing behavior)."""
        node, captured = self._prepare_node(monkeypatch, project_root="/local/project", client_source=None)
        node._get_system_prompt()
        assert captured["workspace_root"] == "/local/project"


# ---------------------------------------------------------------------------
# TestSetupInput (default implementation)
# ---------------------------------------------------------------------------


class TestSetupInputAgenticNode:
    def test_default_setup_input_returns_success(self):
        node = _make_node()
        node.input = BaseInput()
        wf = MagicMock()
        wf.task.catalog_name = "cat"
        wf.task.database_name = "db"
        wf.task.schema_name = "sch"
        wf.context.table_schemas = []
        wf.context.metrics = []
        result = node.setup_input(wf)

        assert result["success"] is True

    def test_default_setup_input_creates_base_input_when_none(self):
        node = _make_node()
        node.input = None
        wf = MagicMock()
        wf.task.catalog_name = "cat"
        wf.task.database_name = "db"
        wf.task.schema_name = "sch"
        wf.context.table_schemas = []
        wf.context.metrics = []
        node.setup_input(wf)

        assert isinstance(node.input, BaseInput)


# ---------------------------------------------------------------------------
# TestSemanticRuntimeDbContext
# ---------------------------------------------------------------------------


class TestSemanticRuntimeDbContext:
    def test_returns_empty_without_agent_config(self):
        node = _make_node(agent_config=None)

        assert node._semantic_runtime_db_context() == {}

    def test_uses_request_runtime_context_and_schema_alias(self):
        cfg = MagicMock()
        cfg.current_datasource = "static_ds"
        cfg.runtime_db_context.return_value = {
            "datasource": "runtime_ds",
            "catalog_name": "runtime_catalog",
            "database_name": "runtime_db",
            "schema_name": "runtime_schema",
        }
        cfg.current_db_config.return_value = MagicMock(
            catalog="configured_catalog",
            database="configured_db",
            schema="configured_schema",
        )
        db_tool = MagicMock()
        db_tool.connector.database_name = "connector_db"
        node = _make_node(agent_config=cfg, db_func_tool=db_tool)
        node.input = MagicMock(catalog="", database="", db_schema="")

        assert node._semantic_runtime_db_context() == {
            "datasource": "runtime_ds",
            "catalog": "runtime_catalog",
            "database": "runtime_db",
            "schema": "runtime_schema",
            "db_schema": "runtime_schema",
        }
        cfg.current_db_config.assert_called_once_with("runtime_ds")

    def test_uses_input_context_when_config_context_fails(self):
        cfg = MagicMock()
        cfg.current_datasource = "static_ds"
        cfg.runtime_db_context.side_effect = RuntimeError("context unavailable")
        cfg.current_db_config.side_effect = RuntimeError("config unavailable")
        node = _make_node(agent_config=cfg, db_func_tool=MagicMock())
        node.input = MagicMock(
            catalog="input_catalog",
            database="input_db",
            db_schema="input_schema",
        )

        assert node._semantic_runtime_db_context() == {
            "datasource": "static_ds",
            "catalog": "input_catalog",
            "database": "input_db",
            "schema": "input_schema",
            "db_schema": "input_schema",
        }


# ---------------------------------------------------------------------------
# TestUpdateContextAgenticNode
# ---------------------------------------------------------------------------


class TestUpdateContextAgenticNode:
    def test_no_result_returns_failure(self):
        node = _make_node()
        node.result = None
        wf = MagicMock()
        result = node.update_context(wf)
        assert result["success"] is False

    def test_result_without_sql_returns_success(self):
        node = _make_node()
        node.result = MagicMock()
        node.result.sql = None
        wf = MagicMock()
        result = node.update_context(wf)
        assert result["success"] is True

    def test_result_with_sql_appends_context(self):
        node = _make_node()
        node.result = MagicMock()
        node.result.sql = "SELECT 1"
        node.result.response = "some explanation"
        wf = MagicMock()
        wf.context.sql_contexts = []
        result = node.update_context(wf)
        assert result["success"] is True
        assert len(wf.context.sql_contexts) == 1


# ---------------------------------------------------------------------------
# TestClearSession
# ---------------------------------------------------------------------------


class TestClearSession:
    def test_clear_session(self):
        node = _make_node()
        node.session_id = "real_session_1"
        mock_sm = MagicMock()
        node._session_manager = mock_sm
        node._session = MagicMock()
        node.clear_session()
        mock_sm.clear_session.assert_called_once_with("real_session_1")
        assert node._session is None

    def test_clear_session_no_session_id(self):
        """Without a session_id, clear_session is a no-op: nothing has been created
        on disk yet so there is nothing to clear. ``_session`` and
        ``session_id`` survive untouched.
        """
        node = _make_node()
        node.session_id = None
        sentinel_session = MagicMock()
        node._session = sentinel_session
        node._session_manager = MagicMock()
        node.clear_session()
        assert node._session is sentinel_session
        assert node.session_id is None
        node._session_manager.clear_session.assert_not_called()


# ---------------------------------------------------------------------------
# TestDeleteSession
# ---------------------------------------------------------------------------


class TestDeleteSession:
    def test_delete_session(self):
        node = _make_node()
        node.session_id = "real_1"
        mock_sm = MagicMock()
        node._session_manager = mock_sm
        node._session = MagicMock()
        node.delete_session()
        mock_sm.delete_session.assert_called_once_with("real_1")
        assert node._session is None
        # session_id is immutable — stays set so logs/tracebacks can still
        # identify which session was deleted. The node itself is unusable.
        assert node.session_id == "real_1"


# ---------------------------------------------------------------------------
# TestSetPermissionCallback
# ---------------------------------------------------------------------------


class TestSetPermissionCallback:
    def test_set_permission_callback_stores_callback(self):
        node = _make_node()
        callback = AsyncMock()
        node.set_permission_callback(callback)
        assert node._permission_callback is callback

    def test_set_permission_callback_forwards_to_permission_manager(self):
        node = _make_node()
        mock_pm = MagicMock()
        node.permission_manager = mock_pm
        callback = AsyncMock()
        node.set_permission_callback(callback)
        mock_pm.set_permission_callback.assert_called_once_with(callback)


# ---------------------------------------------------------------------------
# TestSetupPermissionManager
# ---------------------------------------------------------------------------


class TestSetupPermissionManager:
    """``execution_mode="workflow"`` forces a fresh ``dangerous`` profile.

    Bootstrap, scheduler subagents, and other non-interactive flows must run
    against a known-good baseline regardless of the user's chosen profile —
    otherwise a ``normal`` user gets blocked on every write and a
    ``dangerous`` user silently elevates the workflow.
    """

    def _setup_node(self, *, execution_mode):
        from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule

        # User-side config: a deliberately weird custom profile so we can
        # verify workflow mode ignores it. Default DENY everything; one rule
        # explicitly ALLOWs ``custom_marker_tool``. ``dangerous`` profile would
        # never produce this shape.
        user_config = PermissionConfig(
            default_permission=PermissionLevel.DENY,
            rules=[
                PermissionRule(tool="custom_marker_tool", pattern="*", permission=PermissionLevel.ALLOW),
            ],
        )
        agent_config = MagicMock()
        agent_config.permissions_config = user_config
        agent_config.active_profile_name = "normal"

        node = _make_node(agent_config=agent_config)
        node.execution_mode = execution_mode
        return node, user_config

    def test_workflow_mode_loads_dangerous_profile_ignoring_user_config(self):
        from datus.tools.permission.permission_config import PermissionLevel

        node, user_config = self._setup_node(execution_mode="workflow")
        node._setup_permission_manager()

        assert node.permission_manager.active_profile == "dangerous"
        # The custom user rule must NOT leak into workflow-mode managers.
        assert not any(rule.tool == "custom_marker_tool" for rule in node.permission_manager.global_config.rules), (
            "User profile rules must be ignored when workflow mode forces 'dangerous'"
        )
        # ``dangerous`` profile is default=ALLOW with no rules — the workflow
        # manager must reflect that posture and must NOT carry forward the
        # user's DENY default.
        assert node.permission_manager.global_config.default_permission == PermissionLevel.ALLOW

    def test_interactive_mode_keeps_user_config(self):
        node, user_config = self._setup_node(execution_mode="interactive")
        node._setup_permission_manager()

        # ``normal`` is the user's configured profile, NOT clobbered by the
        # workflow override path.
        assert node.permission_manager.active_profile == "normal"
        # User custom rule is preserved in interactive mode.
        assert any(rule.tool == "custom_marker_tool" for rule in node.permission_manager.global_config.rules)

    def test_no_execution_mode_attr_falls_back_to_user_config(self):
        """Nodes without ``execution_mode`` (legacy / non-agentic) must keep
        the existing behavior — read the user's profile."""
        node, user_config = self._setup_node(execution_mode="interactive")
        # Strip the attribute to simulate a node that never declares execution_mode.
        del node.execution_mode
        node._setup_permission_manager()

        assert node.permission_manager.active_profile == "normal"


# ---------------------------------------------------------------------------
# TestGetAvailableSkillsContext
# ---------------------------------------------------------------------------


class TestGetAvailableSkillsContext:
    def test_returns_empty_when_no_skill_manager(self):
        node = _make_node()
        node.skill_manager = None
        result = node._get_available_skills_context()
        assert result == ""

    def test_calls_skill_manager_generate_xml(self):
        node = _make_node()
        mock_sm = MagicMock()
        mock_sm.parse_skill_patterns.return_value = ["sql-*"]
        mock_sm.generate_available_skills_xml.return_value = "<skills>...</skills>"
        node.skill_manager = mock_sm
        node.node_config = {"skills": "sql-*"}
        result = node._get_available_skills_context()
        assert "<skills>" in result


# ---------------------------------------------------------------------------
# TestCompactDispatch (was TestAutoCompact)
# Exercises the public ``compact()`` API and the legacy ``_auto_compact()``
# wrapper kept for backward compatibility.
# ---------------------------------------------------------------------------


class TestCompactDispatch:
    @pytest.mark.asyncio
    async def test_auto_compact_returns_false_when_no_signal(self):
        # With no model + no session, ``_history_token_ratio_sync`` returns 0
        # and ``_user_turn_count_from_session`` returns 0, so ``compact("auto")``
        # is a noop and the legacy wrapper reports False.
        node = _make_node()
        node.model = None
        result = await node._auto_compact()
        assert result is False

    @pytest.mark.asyncio
    async def test_compact_auto_picks_major_when_token_ratio_high(self):
        node = _make_node()
        node.session_id = "sid"
        node._session = MagicMock()
        # Make the sync ratio exceed major threshold (0.9 default).
        with patch.object(_ConcreteAgenticNode, "_history_token_ratio_sync", return_value=0.95):
            with patch.object(
                node, "_major_compact", new=AsyncMock(return_value={"mode": "major", "success": True})
            ) as m:
                result = await node.compact(mode="auto", reason="ratio_test")
        m.assert_awaited_once_with(reason="ratio_test")
        assert result["mode"] == "major"

    @pytest.mark.asyncio
    async def test_compact_auto_picks_minor_when_user_turns_exceed_keep_window(self):
        node = _make_node()
        node.session_id = "sid"
        node._session = MagicMock()
        with patch.object(_ConcreteAgenticNode, "_history_token_ratio_sync", return_value=0.1):
            with patch.object(
                _ConcreteAgenticNode,
                "_user_turn_count_from_session",
                new=AsyncMock(return_value=99),
            ):
                with patch.object(
                    node, "_minor_compact", new=AsyncMock(return_value={"mode": "minor", "success": True})
                ) as m:
                    result = await node.compact(mode="auto", reason="ratio_test")
        m.assert_awaited_once_with(reason="ratio_test")
        assert result["mode"] == "minor"

    @pytest.mark.asyncio
    async def test_compact_auto_returns_noop_when_no_conditions(self):
        node = _make_node()
        node._session = MagicMock()
        with patch.object(_ConcreteAgenticNode, "_history_token_ratio_sync", return_value=0.1):
            with patch.object(
                _ConcreteAgenticNode,
                "_user_turn_count_from_session",
                new=AsyncMock(return_value=0),
            ):
                result = await node.compact(mode="auto", reason="noop_test")
        assert result["mode"] == "noop"
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_compact_explicit_major_skips_dispatch(self):
        node = _make_node()
        node.session_id = "sid"
        node._session = MagicMock()
        # Even at 0% ratio, explicit mode="major" goes straight to _major_compact.
        with patch.object(_ConcreteAgenticNode, "_history_token_ratio_sync", return_value=0.0):
            with patch.object(
                node, "_major_compact", new=AsyncMock(return_value={"mode": "major", "success": True})
            ) as m:
                await node.compact(mode="major", reason="cli_manual")
        m.assert_awaited_once_with(reason="cli_manual")


# ---------------------------------------------------------------------------
# TestGetSessionInfo
# ---------------------------------------------------------------------------


class TestGetSessionInfo:
    @pytest.mark.asyncio
    async def test_get_session_info_no_session(self):
        node = _make_node()
        node.session_id = None
        info = await node.get_session_info()
        assert info["session_id"] is None
        assert info["active"] is False

    @pytest.mark.asyncio
    async def test_get_session_info_with_session(self):
        node = _make_node(context_length=100000)
        node.session_id = "my_session"
        node._session = MagicMock()
        node.actions = []

        with patch.object(node, "_count_session_tokens", return_value=5000):
            info = await node.get_session_info()

        assert info["session_id"] == "my_session"
        assert info["active"] is True
        assert info["token_count"] == 5000


# ---------------------------------------------------------------------------
# TestManualCompact
# ---------------------------------------------------------------------------


class TestManualCompact:
    """Behavioral tests for ``_major_compact`` (the LLM-driven summarization pass).

    Renamed from ``TestManualCompact`` for historical reasons. Class kept under
    the same name so the test count stays comparable across the refactor.
    """

    @pytest.mark.asyncio
    async def test_major_compact_no_model_returns_failure(self):
        node = _make_node()
        node.model = None
        node._session = MagicMock()
        node.session_id = "sid"
        result = await node._major_compact(reason="t")
        assert result["success"] is False
        assert result["mode"] == "major"

    @pytest.mark.asyncio
    async def test_major_compact_no_session_returns_failure(self):
        node = _make_node()
        node.model = MagicMock()
        node._session = None
        node.session_id = ""
        result = await node._major_compact(reason="t")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_major_compact_success_persists_continuation_message(self):
        node = _make_node()
        node.session_id = "compact_test"
        mock_session = _make_async_session_mock()
        node._session = mock_session
        mock_sm = MagicMock()
        node._session_manager = mock_sm
        mock_model = MagicMock()
        mock_model.generate_with_tools = AsyncMock(
            return_value={"content": "## 1. Primary user request\nMy goal", "usage": {"output_tokens": 100}}
        )
        node.model = mock_model
        # Bypass the disk dump so the test stays hermetic; we only care the
        # continuation message gets persisted with the summary embedded.
        with patch.object(_ConcreteAgenticNode, "_dump_session_history_jsonl", new=AsyncMock(return_value=None)):
            with patch.object(_ConcreteAgenticNode, "_get_archive", return_value=None):
                with patch.object(_ConcreteAgenticNode, "_get_system_prompt", return_value="sys"):
                    result = await node._major_compact(reason="cli_manual")

        assert result["success"] is True
        assert result["mode"] == "major"
        assert "Primary user request" in result["summary"]
        mock_session.clear_session.assert_awaited_once()
        mock_session.add_items.assert_awaited_once()
        items = mock_session.add_items.await_args.args[0]
        assert len(items) == 1
        # Continuation persists as an assistant ``output_text`` block so the
        # next turn sees the summary as a prior assistant utterance — the
        # natural shape for "I summarized previously, now answer the next
        # question". Storing as user role used to confuse /chat/history into
        # rendering a phantom user turn.
        assert items[0]["role"] == "assistant"
        assert items[0]["type"] == "message"
        content_blocks = items[0]["content"]
        assert isinstance(content_blocks, list) and len(content_blocks) == 1
        assert content_blocks[0]["type"] == "output_text"
        assert "Primary user request" in content_blocks[0]["text"]
        # Major compact preserves the session — no delete on the session_manager.
        mock_sm.delete_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_major_compact_lazy_loads_session_after_resume(self):
        """After .resume sets session_id but leaves _session None, major compact
        must still materialize the session before attempting summary persistence.
        """
        node = _make_node()
        node.session_id = "resumed_session"
        node._session = None  # Simulate post-resume state
        mock_model = MagicMock()
        mock_model.generate_with_tools = AsyncMock(
            return_value={"content": "Resumed summary", "usage": {"output_tokens": 50}}
        )
        node.model = mock_model
        mock_sm = MagicMock()
        mock_sm.create_session = MagicMock(return_value=_make_async_session_mock())
        node._session_manager = mock_sm
        with patch.object(_ConcreteAgenticNode, "_dump_session_history_jsonl", new=AsyncMock(return_value=None)):
            with patch.object(_ConcreteAgenticNode, "_get_archive", return_value=None):
                with patch.object(_ConcreteAgenticNode, "_get_system_prompt", return_value="sys"):
                    result = await node._major_compact(reason="resume_test")
        mock_sm.create_session.assert_called_once_with("resumed_session")
        assert result["success"] is True
        assert "Resumed summary" in result["summary"]
        assert node.session_id == "resumed_session"

    @pytest.mark.asyncio
    async def test_major_compact_add_items_failure_returns_failure(self):
        """If add_items raises, surface a failure result and don't crash."""
        node = _make_node()
        node.session_id = "fail_test"
        mock_session = _make_async_session_mock()
        mock_session.add_items.side_effect = RuntimeError("write failed")
        node._session = mock_session
        mock_model = MagicMock()
        mock_model.generate_with_tools = AsyncMock(return_value={"content": "summary", "usage": {"output_tokens": 10}})
        node.model = mock_model
        with patch.object(_ConcreteAgenticNode, "_dump_session_history_jsonl", new=AsyncMock(return_value=None)):
            with patch.object(_ConcreteAgenticNode, "_get_archive", return_value=None):
                with patch.object(_ConcreteAgenticNode, "_get_system_prompt", return_value="sys"):
                    result = await node._major_compact(reason="fail_test")
        assert result["success"] is False
        assert result["mode"] == "major"
        mock_session.clear_session.assert_awaited_once()
        mock_session.add_items.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestCountSessionTokens
# ---------------------------------------------------------------------------


class TestCountSessionTokens:
    @pytest.mark.asyncio
    async def test_count_tokens_no_actions_no_session(self):
        node = _make_node()
        node._session = None
        result = await node._count_session_tokens()
        assert result == 0

    @pytest.mark.asyncio
    async def test_count_tokens_uses_last_call_input_tokens(self):
        """Primary path: last_call_input_tokens from the most recent action's usage."""
        node = _make_node()
        action = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="chat",
            messages="ok",
            input_data={},
            output_data={"usage": {"last_call_input_tokens": 5000, "input_tokens": 8000, "total_tokens": 12000}},
            status=ActionStatus.SUCCESS,
        )
        node.actions.append(action)
        result = await node._count_session_tokens()
        assert result == 5000

    @pytest.mark.asyncio
    async def test_count_tokens_falls_back_to_input_tokens(self):
        """When last_call_input_tokens is 0, fall back to input_tokens."""
        node = _make_node()
        action = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="chat",
            messages="ok",
            input_data={},
            output_data={"usage": {"last_call_input_tokens": 0, "input_tokens": 8000, "total_tokens": 12000}},
            status=ActionStatus.SUCCESS,
        )
        node.actions.append(action)
        result = await node._count_session_tokens()
        assert result == 8000

    @pytest.mark.asyncio
    async def test_count_tokens_falls_back_to_turn_usage(self):
        """When actions have no usage, fall back to last turn in turn_usage table."""
        node = _make_node()
        mock_session = MagicMock()
        mock_session.get_turn_usage = AsyncMock(
            return_value=[
                {"user_turn_number": 1, "total_tokens": 500},
                {"user_turn_number": 2, "total_tokens": 1234},
            ]
        )
        node._session = mock_session
        result = await node._count_session_tokens()
        assert result == 1234

    @pytest.mark.asyncio
    async def test_count_tokens_empty_actions_empty_turn_usage(self):
        node = _make_node()
        mock_session = MagicMock()
        mock_session.get_turn_usage = AsyncMock(return_value=[])
        node._session = mock_session
        result = await node._count_session_tokens()
        assert result == 0

    @pytest.mark.asyncio
    async def test_count_tokens_ignores_subagent_depth_actions(self):
        """Sub-agent (depth>0) ASSISTANT actions must not pollute parent context estimate.

        Regression: the scan must skip child/tool usage so that only root-level
        (depth == 0) assistant actions contribute to the context window estimate.
        Here the only depth>0 assistant has large usage; the parent's estimate
        should fall back to turn_usage (or 0) instead of reading the child's.
        """
        node = _make_node()
        subagent_action = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="chat",
            messages="sub",
            input_data={},
            output_data={"usage": {"last_call_input_tokens": 99999, "input_tokens": 99999, "total_tokens": 99999}},
            status=ActionStatus.SUCCESS,
        )
        subagent_action.depth = 1  # simulate sub-agent nesting
        node.actions.append(subagent_action)

        mock_session = MagicMock()
        mock_session.get_turn_usage = AsyncMock(return_value=[{"user_turn_number": 1, "total_tokens": 321}])
        node._session = mock_session

        result = await node._count_session_tokens()
        # Must NOT return 99999 from the depth>0 action; fall back to turn_usage's 321.
        assert result == 321

    @pytest.mark.asyncio
    async def test_count_tokens_breaks_at_root_user_message(self):
        """Scan stops at the most recent root-level USER action to scope to the current turn.

        An older ASSISTANT action preceding the latest root USER message must
        NOT be used, even if it has usage. This guards against bleed-over from
        the previous turn's usage into the current turn's estimate.
        """
        node = _make_node()
        # Older turn's assistant reply with usage (should be ignored after USER break).
        old_assistant = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="chat",
            messages="old",
            input_data={},
            output_data={"usage": {"last_call_input_tokens": 7777}},
            status=ActionStatus.SUCCESS,
        )
        old_assistant.depth = 0
        # Latest root user message marks the boundary of the current turn.
        latest_user = ActionHistory.create_action(
            role=ActionRole.USER,
            action_type="chat",
            messages="new question",
            input_data={},
            status=ActionStatus.SUCCESS,
        )
        latest_user.depth = 0
        node.actions.extend([old_assistant, latest_user])

        mock_session = MagicMock()
        mock_session.get_turn_usage = AsyncMock(return_value=[{"user_turn_number": 1, "total_tokens": 111}])
        node._session = mock_session

        result = await node._count_session_tokens()
        # Reverse scan hits latest_user first -> break -> fall back to turn_usage (111).
        assert result == 111


# ---------------------------------------------------------------------------
# Concrete subclass for testing
# ---------------------------------------------------------------------------


class _SimpleAgenticNode(AgenticNode):
    """Minimal concrete AgenticNode for unit tests."""

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        action = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="test",
            messages="done",
            input_data={},
            output_data={"success": True, "result": "ok"},
            status=ActionStatus.SUCCESS,
        )
        yield action


def _make_simple_node(context_length=_UNSET, **overrides):
    """Build a minimal _SimpleAgenticNode bypassing __init__."""
    with patch.object(AgenticNode, "__init__", lambda self, *a, **kw: None):
        node = _SimpleAgenticNode.__new__(_SimpleAgenticNode)

    node._agent_config_ref = None
    node._pinned_model = None
    node._node_model_name = None
    node._session = None
    node.session_id = None
    node.tools = []
    node.mcp_servers = {}
    node.actions = []

    node.node_config = {}
    node.agent_config = None
    node.permission_manager = None
    node.skill_manager = None
    node.skill_func_tool = None
    node._permission_callback = None
    node.id = "test_node"
    node.description = "Test"
    node.type = "test"
    node.status = "pending"
    node.result = None
    node.dependencies = []
    node.input = None

    from datus.cli.execution_state import InteractionBroker, InterruptController
    from datus.schemas.action_bus import ActionBus

    node.action_bus = ActionBus()
    node.interaction_broker = InteractionBroker()
    node.interrupt_controller = InterruptController()

    if context_length is not _UNSET and context_length is not None:
        mock_model = MagicMock()
        mock_model.context_length.return_value = context_length
        node._pinned_model = mock_model

    for k, v in overrides.items():
        setattr(node, k, v)
    return node


# ---------------------------------------------------------------------------
# get_node_name
# ---------------------------------------------------------------------------


class TestGetNodeNameExtended:
    def test_removes_agentic_node_suffix(self):
        node = _make_simple_node()
        # _SimpleAgenticNode -> "simple"
        assert node.get_node_name() == "_simple"

    def test_class_without_suffix_returns_lowercase(self):
        class MyCustomNode(AgenticNode):
            async def execute_stream(self, ahm=None):
                return
                yield  # noqa

        with patch.object(AgenticNode, "__init__", lambda self, *a, **kw: None):
            n = MyCustomNode.__new__(MyCustomNode)
        n.node_config = {}
        assert n.get_node_name() == "mycustomnode"


# ---------------------------------------------------------------------------
# _parse_node_config
# ---------------------------------------------------------------------------


class TestParseNodeConfigExtended:
    def test_no_agent_config_returns_empty(self):
        node = _make_simple_node()
        result = node._parse_node_config(None, "chat")
        assert result == {}

    def test_node_not_in_config_returns_empty(self):
        node = _make_simple_node()
        mock_config = MagicMock()
        mock_config.agentic_nodes = {}
        result = node._parse_node_config(mock_config, "chat")
        assert result == {}

    def test_dict_node_config_extracted(self):
        node = _make_simple_node()
        mock_config = MagicMock()
        mock_config.agentic_nodes = {
            "chat": {
                "model": "gpt-4",
                "system_prompt": "You are a SQL assistant",
                "max_turns": 10,
            }
        }
        result = node._parse_node_config(mock_config, "chat")
        assert result.get("model") == "gpt-4"
        assert result.get("system_prompt") == "You are a SQL assistant"
        assert result.get("max_turns") == 10

    def test_rules_dict_normalized_to_string(self):
        node = _make_simple_node()
        mock_config = MagicMock()
        mock_config.agentic_nodes = {
            "gen_sql": {
                "rules": [{"always": "use CTEs"}, "plain rule"],
            }
        }
        result = node._parse_node_config(mock_config, "gen_sql")
        rules = result.get("rules", [])
        assert len(rules) == 2
        assert any("always" in r for r in rules)

    def test_none_values_not_included(self):
        node = _make_simple_node()
        mock_config = MagicMock()
        mock_config.agentic_nodes = {"mynode": {"model": "gpt-4", "system_prompt": None}}
        result = node._parse_node_config(mock_config, "mynode")
        assert result.get("model") == "gpt-4"
        # None system_prompt should not be in result
        assert "system_prompt" not in result


# ---------------------------------------------------------------------------
# _get_tool_category
# ---------------------------------------------------------------------------
# _resolve_workspace_root
# ---------------------------------------------------------------------------


class TestResolveWorkspaceRootExtended:
    def test_default_is_cwd(self):
        node = _make_simple_node()
        result = node._resolve_workspace_root()
        assert result == os.getcwd()

    def test_node_config_workspace_root_used(self):
        node = _make_simple_node(node_config={"workspace_root": "/tmp/ws"})
        result = node._resolve_workspace_root()
        assert result == "/tmp/ws"

    def test_agent_config_project_root_used(self):
        node = _make_simple_node()
        mock_config = MagicMock(spec=["project_root"])
        mock_config.project_root = "/var/data/ws"
        node.agent_config = mock_config
        result = node._resolve_workspace_root()
        assert result == "/var/data/ws"

    def test_tilde_expanded(self):
        node = _make_simple_node(node_config={"workspace_root": "~/myproject"})
        result = node._resolve_workspace_root()
        assert "~" not in result
        assert result.startswith("/")


# ---------------------------------------------------------------------------
# clear_session / delete_session
# ---------------------------------------------------------------------------


class TestSessionManagement:
    def test_clear_session_normal(self):
        node = _make_simple_node()
        mock_sm = MagicMock()
        node._session_manager = mock_sm
        node.session_id = "sess_2"
        node._session = MagicMock()
        node.clear_session()
        mock_sm.clear_session.assert_called_once_with("sess_2")
        assert node._session is None

    def test_clear_session_no_session_id(self):
        """No-op path: without a session_id, clear_session leaves state
        alone so the node is reusable once a session is materialized later."""
        node = _make_simple_node()
        sentinel_session = MagicMock()
        node._session = sentinel_session
        node.session_id = None
        node._session_manager = MagicMock()
        node.clear_session()
        assert node._session is sentinel_session
        assert node.session_id is None
        node._session_manager.clear_session.assert_not_called()

    def test_delete_session_normal(self):
        node = _make_simple_node()
        mock_sm = MagicMock()
        node._session_manager = mock_sm
        node.session_id = "sess_5"
        node._session = MagicMock()
        node.delete_session()
        mock_sm.delete_session.assert_called_once_with("sess_5")
        assert node._session is None
        # session_id is immutable — preserved post-delete for log traceability.
        assert node.session_id == "sess_5"


# ---------------------------------------------------------------------------
# get_session_info
# ---------------------------------------------------------------------------


class TestGetSessionInfoExtended:
    def test_no_session_id_returns_inactive(self):
        node = _make_simple_node()
        result = asyncio.run(node.get_session_info())
        assert result["session_id"] is None
        assert result["active"] is False

    def test_with_session_returns_info(self):
        node = _make_simple_node(context_length=4000)
        node.session_id = "sess_x"
        node._session = MagicMock()
        # Provide usage via actions (primary path for _count_session_tokens)
        action = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="chat",
            messages="ok",
            input_data={},
            output_data={"usage": {"last_call_input_tokens": 500, "input_tokens": 800, "total_tokens": 1200}},
            status=ActionStatus.SUCCESS,
        )
        node.actions.append(action)

        result = asyncio.run(node.get_session_info())
        assert result["session_id"] == "sess_x"
        assert result["active"] is True
        assert result["token_count"] == 500
        assert result["context_length"] == 4000


# ---------------------------------------------------------------------------
# _get_or_create_session
# ---------------------------------------------------------------------------


class TestGetOrCreateSession:
    def test_returns_existing_session(self):
        node = _make_simple_node()
        mock_session = MagicMock()
        node._session = mock_session
        session = node._get_or_create_session()
        assert session is mock_session

    def test_creates_new_session_when_none(self):
        node = _make_simple_node()
        mock_sm = MagicMock()
        mock_session = MagicMock()
        mock_sm.create_session.return_value = mock_session
        node._session_manager = mock_sm
        node.session_id = "my_session"

        session = node._get_or_create_session()
        assert session is mock_session
        mock_sm.create_session.assert_called_once_with("my_session")

    def test_uses_existing_session_id(self):
        """``session_id`` is allocated in ``__init__`` (or supplied by the
        caller) and never mutated thereafter. ``_get_or_create_session`` opens
        the .db file under that id; it does not generate or rotate ids."""
        node = _make_simple_node()
        mock_sm = MagicMock()
        mock_session = MagicMock()
        mock_sm.create_session.return_value = mock_session
        node._session_manager = mock_sm
        node.session_id = "preset_session_xyz"

        node._get_or_create_session()
        assert node.session_id == "preset_session_xyz"
        mock_sm.create_session.assert_called_once_with("preset_session_xyz")


# ---------------------------------------------------------------------------
# update_context
# ---------------------------------------------------------------------------


class TestUpdateContext:
    def test_no_result_returns_failure(self):
        node = _make_simple_node()
        workflow = MagicMock()
        result = node.update_context(workflow)
        assert result["success"] is False

    def test_result_with_sql_appended_to_context(self):
        node = _make_simple_node()
        mock_result = MagicMock()
        mock_result.sql = "SELECT * FROM users"
        mock_result.response = "Query executed"
        node.result = mock_result

        workflow = MagicMock()
        workflow.context.sql_contexts = []

        result = node.update_context(workflow)
        assert result["success"] is True
        assert len(workflow.context.sql_contexts) == 1

    def test_result_without_sql_does_not_append(self):
        node = _make_simple_node()
        mock_result = MagicMock()
        mock_result.sql = None
        node.result = mock_result

        workflow = MagicMock()
        workflow.context.sql_contexts = []

        result = node.update_context(workflow)
        assert result["success"] is True
        assert len(workflow.context.sql_contexts) == 0


# ---------------------------------------------------------------------------
# setup_input
# ---------------------------------------------------------------------------


class TestSetupInput:
    def test_creates_base_input_when_none(self):
        node = _make_simple_node()
        workflow = MagicMock()
        workflow.task.catalog_name = "cat"
        workflow.task.database_name = "db"
        workflow.task.schema_name = "schema"
        workflow.context.table_schemas = []
        workflow.context.metrics = []

        result = node.setup_input(workflow)
        assert result["success"] is True
        assert isinstance(node.input, BaseInput)

    def test_populates_fields_when_input_has_them(self):
        node = _make_simple_node()
        node.input = BaseInput()

        workflow = MagicMock()
        workflow.task.catalog_name = "my_cat"
        workflow.task.database_name = "my_db"
        workflow.task.schema_name = "my_schema"
        workflow.context.table_schemas = ["schema1"]
        workflow.context.metrics = []

        node.setup_input(workflow)
        # Verify setup_input populated the node's input
        assert isinstance(node.input, BaseInput)


# ---------------------------------------------------------------------------
# set_permission_callback
# ---------------------------------------------------------------------------


class TestSetPermissionCallbackExtended:
    def test_stores_callback(self):
        node = _make_simple_node()
        callback = AsyncMock()
        node.set_permission_callback(callback)
        assert node._permission_callback is callback

    def test_forwards_to_permission_manager(self):
        node = _make_simple_node()
        mock_pm = MagicMock()
        node.permission_manager = mock_pm
        callback = AsyncMock()
        node.set_permission_callback(callback)
        mock_pm.set_permission_callback.assert_called_once_with(callback)


# ---------------------------------------------------------------------------
# execute (sync wrapper)
# ---------------------------------------------------------------------------


class TestExecuteSync:
    def test_execute_returns_base_result(self):
        node = _make_simple_node()
        result = node.execute()
        assert isinstance(result, BaseResult)

    def test_execute_success_result(self):
        node = _make_simple_node()
        result = node.execute()
        # The simple node yields success action
        assert isinstance(result, BaseResult)
        assert result.success is True


# ---------------------------------------------------------------------------
# _manual_compact
# ---------------------------------------------------------------------------


class TestManualCompactExtended:
    def test_no_model_returns_failure(self):
        node = _make_simple_node()
        result = asyncio.run(node._major_compact(reason="t"))
        assert result["success"] is False

    def test_success_stores_summary(self):
        node = _make_simple_node()
        mock_model = MagicMock()
        mock_session = _make_async_session_mock()
        node.model = mock_model
        node._session = mock_session
        node.session_id = "sess_compact"

        mock_model.generate_with_tools = AsyncMock(
            return_value={"content": "summary text", "usage": {"output_tokens": 100}}
        )

        with patch.object(_SimpleAgenticNode, "_dump_session_history_jsonl", new=AsyncMock(return_value=None)):
            with patch.object(_SimpleAgenticNode, "_get_archive", return_value=None):
                with patch.object(_SimpleAgenticNode, "_get_system_prompt", return_value="sys"):
                    result = asyncio.run(node._major_compact(reason="t"))
        assert result["success"] is True
        assert "summary text" in result["summary"]
        # Session must be preserved — summary now lives inside the session.
        assert node._session is mock_session
        assert node.session_id == "sess_compact"
        mock_model.generate_with_tools.assert_awaited_once()
        assert mock_model.generate_with_tools.await_args.kwargs["agent_name"] == node.get_node_name()
        mock_session.clear_session.assert_awaited_once()
        mock_session.add_items.assert_awaited_once()


# ---------------------------------------------------------------------------
# _auto_compact
# ---------------------------------------------------------------------------


class TestAutoCompactExtended:
    def test_no_model_returns_false(self):
        node = _make_simple_node()
        result = asyncio.run(node._auto_compact())
        assert result is False

    def test_no_context_length_returns_false(self):
        mock_model = MagicMock()
        mock_model.context_length.return_value = None
        node = _make_simple_node()
        node._pinned_model = mock_model
        result = asyncio.run(node._auto_compact())
        assert result is False

    def test_below_threshold_returns_false(self):
        node = _make_simple_node(context_length=10000)
        # Provide usage via actions (primary path for _count_session_tokens)
        action = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="chat",
            messages="ok",
            input_data={},
            output_data={"usage": {"last_call_input_tokens": 100, "input_tokens": 200, "total_tokens": 300}},
            status=ActionStatus.SUCCESS,
        )
        node.actions.append(action)

        result = asyncio.run(node._auto_compact())
        assert result is False

    def test_above_threshold_triggers_compact(self):
        node = _make_simple_node(context_length=1000)
        node._session = _make_async_session_mock()
        action = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="chat",
            messages="ok",
            input_data={},
            output_data={"usage": {"last_call_input_tokens": 950, "input_tokens": 1500, "total_tokens": 2000}},
            status=ActionStatus.SUCCESS,
        )
        node.actions.append(action)
        node._pinned_model.generate_with_tools = AsyncMock(
            return_value={"content": "summary", "usage": {"output_tokens": 50}}
        )
        node.session_id = "sess_auto"

        with patch.object(_SimpleAgenticNode, "_dump_session_history_jsonl", new=AsyncMock(return_value=None)):
            with patch.object(_SimpleAgenticNode, "_get_archive", return_value=None):
                with patch.object(_SimpleAgenticNode, "_get_system_prompt", return_value="sys"):
                    result = asyncio.run(node._auto_compact())
        assert result is True


# ---------------------------------------------------------------------------
# TestGetLastTurnUsage
# ---------------------------------------------------------------------------


class TestGetLastTurnUsage:
    def test_returns_none_when_no_actions(self):
        node = _make_node()
        node.actions = []
        result = asyncio.run(node.get_last_turn_usage())
        assert result is None

    def test_returns_none_when_no_usage_in_actions(self):
        node = _make_node()
        action = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="test",
            messages="hello",
            input_data={},
            output_data={"response": "ok"},
            status=ActionStatus.SUCCESS,
        )
        node.actions = [action]
        result = asyncio.run(node.get_last_turn_usage())
        assert result is None

    def test_returns_usage_from_last_assistant_action(self):
        node = _make_node(context_length=128000)
        usage_dict = {
            "requests": 2,
            "input_tokens": 1000,
            "output_tokens": 200,
            "total_tokens": 1200,
            "cached_tokens": 500,
            "cache_hit_rate": 0.5,
            "last_call_input_tokens": 600,
        }
        action = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="chat_response",
            messages="result",
            input_data={},
            output_data={"response": "ok", "usage": usage_dict},
            status=ActionStatus.SUCCESS,
        )
        node.actions = [action]
        result = asyncio.run(node.get_last_turn_usage())
        assert isinstance(result, TokenUsage)
        assert result.input_tokens == 1000
        assert result.output_tokens == 200
        assert result.cached_tokens == 500
        # session_total_tokens should use last_call_input_tokens, not cumulative input_tokens
        assert result.session_total_tokens == 600
        assert result.context_length == 128000

    def test_session_total_tokens_falls_back_to_input_tokens(self):
        """When last_call_input_tokens is missing/zero, fallback to input_tokens."""
        node = _make_node(context_length=128000)
        usage_dict = {
            "requests": 1,
            "input_tokens": 1000,
            "output_tokens": 200,
            "total_tokens": 1200,
        }
        action = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="chat_response",
            messages="result",
            input_data={},
            output_data={"response": "ok", "usage": usage_dict},
            status=ActionStatus.SUCCESS,
        )
        node.actions = [action]
        result = asyncio.run(node.get_last_turn_usage())
        assert isinstance(result, TokenUsage)
        assert result.session_total_tokens == 1000

    def test_skips_tool_actions(self):
        node = _make_node(context_length=64000)
        tool_action = ActionHistory.create_action(
            role=ActionRole.TOOL,
            action_type="db_query",
            messages="SELECT 1",
            input_data={},
            output_data={"result": "ok"},
            status=ActionStatus.SUCCESS,
        )
        assistant_action = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="chat_response",
            messages="done",
            input_data={},
            output_data={"response": "done", "usage": {"input_tokens": 500, "output_tokens": 100, "total_tokens": 600}},
            status=ActionStatus.SUCCESS,
        )
        node.actions = [assistant_action, tool_action]
        result = asyncio.run(node.get_last_turn_usage())
        # Should find the assistant action even though tool action is last
        assert isinstance(result, TokenUsage)
        assert result.input_tokens == 500

    def test_ignores_sub_agent_usage(self):
        """Usage from sub-agent actions (depth > 0) should be skipped."""
        node = _make_node(context_length=128000)
        sub_agent_action = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="sub_response",
            messages="sub",
            input_data={},
            output_data={"usage": {"input_tokens": 9999, "output_tokens": 100, "total_tokens": 10099}},
            status=ActionStatus.SUCCESS,
        )
        sub_agent_action.depth = 1
        root_action = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="chat_response",
            messages="main",
            input_data={},
            output_data={"usage": {"input_tokens": 500, "output_tokens": 50, "total_tokens": 550}},
            status=ActionStatus.SUCCESS,
        )
        node.actions = [root_action, sub_agent_action]
        result = asyncio.run(node.get_last_turn_usage())
        assert isinstance(result, TokenUsage)
        assert result.input_tokens == 500  # root action, not sub-agent

    def test_scoped_to_current_turn(self):
        """Should stop at the last root-level user message to avoid returning stale usage."""
        node = _make_node(context_length=128000)
        old_usage = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="chat_response",
            messages="old",
            input_data={},
            output_data={"usage": {"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200}},
            status=ActionStatus.SUCCESS,
        )
        user_msg = ActionHistory.create_action(
            role=ActionRole.USER,
            action_type="message",
            messages="new question",
            input_data={},
            output_data={},
            status=ActionStatus.SUCCESS,
        )
        # Current turn has a tool action but no assistant usage yet
        tool_action = ActionHistory.create_action(
            role=ActionRole.TOOL,
            action_type="db_query",
            messages="SELECT 1",
            input_data={},
            output_data={"result": "ok"},
            status=ActionStatus.SUCCESS,
        )
        node.actions = [old_usage, user_msg, tool_action]
        result = asyncio.run(node.get_last_turn_usage())
        # Should return None because old_usage is from a previous turn
        assert result is None

    def test_context_length_none_defaults_to_zero(self):
        node = _make_node(context_length=None)
        action = ActionHistory.create_action(
            role=ActionRole.ASSISTANT,
            action_type="test",
            messages="r",
            input_data={},
            output_data={"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            status=ActionStatus.SUCCESS,
        )
        node.actions = [action]
        result = asyncio.run(node.get_last_turn_usage())
        assert isinstance(result, TokenUsage)
        assert result.context_length == 0


# ---------------------------------------------------------------------------
# TestResolveLanguageName + TestInjectResponseLanguage
# ---------------------------------------------------------------------------


class TestResolveLanguageName:
    def test_known_codes_return_human_names(self):
        from datus.agent.node.agentic_node import _resolve_language_name

        assert _resolve_language_name("en") == "English"
        assert _resolve_language_name("zh") == "Chinese"
        assert _resolve_language_name("ja") == "Japanese"

    def test_case_insensitive(self):
        from datus.agent.node.agentic_node import _resolve_language_name

        assert _resolve_language_name("EN") == "English"
        assert _resolve_language_name("ZH-CN") == "Chinese"

    def test_unknown_code_returned_as_is(self):
        from datus.agent.node.agentic_node import _resolve_language_name

        assert _resolve_language_name("xx-yy") == "xx-yy"

    def test_empty_falls_back_to_english(self):
        from datus.agent.node.agentic_node import _resolve_language_name

        assert _resolve_language_name("") == "English"
        assert _resolve_language_name(None) == "English"


class TestInjectResponseLanguage:
    """``_inject_response_language`` appends a single language-policy block
    driven by ``agent_config.language`` and is invoked from
    ``_finalize_system_prompt`` for every AgenticNode subclass.
    """

    class _Cfg:
        """Lightweight stand-in for AgentConfig. Using a MagicMock would let
        ``getattr(cfg, "prompt_manager")`` return a MagicMock and short-circuit
        ``get_prompt_manager``, defeating the real Jinja render under test.
        """

        def __init__(self, language):
            self.language = language
            self.prompt_manager = None
            self.path_manager = None

    def _agent_config(self, language="en"):
        return self._Cfg(language)

    def test_explicit_english_appends_english_block(self):
        """Explicit ``language: en`` still pins output to English — the policy
        is only skipped when the setting is unset entirely."""
        node = _make_node(agent_config=self._agent_config("en"))
        result = node._inject_response_language("BASE")
        assert "Response Language" in result
        assert "English (en)" in result
        assert result.startswith("BASE")

    def test_apply_to_covers_generated_artifacts(self):
        """The Response Language directive must cover artifact text — without
        this clause LLMs treat report/dashboard ``name`` and ``description``
        (and other artifact-internal prose) as outside the language scope and
        emit them in the user's prompt language even when the global
        ``agent.language`` is pinned to a different value.
        """
        node = _make_node(agent_config=self._agent_config("en"))
        result = node._inject_response_language("BASE")
        assert "generated artifacts" in result

    def test_chinese_override_uses_chinese_name(self):
        node = _make_node(agent_config=self._agent_config("zh"))
        result = node._inject_response_language("BASE")
        assert "Chinese (zh)" in result

    def test_unknown_code_uses_raw_value(self):
        node = _make_node(agent_config=self._agent_config("xx"))
        result = node._inject_response_language("BASE")
        assert "xx (xx)" in result

    def test_none_language_skips_injection(self):
        """``language=None`` means "let the model decide" — no directive."""
        node = _make_node(agent_config=self._agent_config(None))
        result = node._inject_response_language("BASE")
        assert result == "BASE"

    def test_empty_language_skips_injection(self):
        """Whitespace-only/empty language is treated as unset."""
        node = _make_node(agent_config=self._agent_config(""))
        assert node._inject_response_language("BASE") == "BASE"
        node2 = _make_node(agent_config=self._agent_config("   "))
        assert node2._inject_response_language("BASE") == "BASE"

    def test_missing_attribute_skips_injection(self):
        """agent_config without ``language`` attribute is a no-op."""

        class _Bare:
            pass

        node = _make_node(agent_config=_Bare())
        assert node._inject_response_language("BASE") == "BASE"

    def test_render_failure_returns_base_unchanged(self):
        node = _make_node(agent_config=self._agent_config("zh"))
        with patch("datus.agent.node.agentic_node.get_prompt_manager") as mgr:
            mgr.return_value.render_template.side_effect = RuntimeError("boom")
            result = node._inject_response_language("BASE")
        assert result == "BASE"


# ---------------------------------------------------------------------------
# _ensure_permission_hooks: proxied_tool_names wiring
# ---------------------------------------------------------------------------


class TestEnsurePermissionHooksProxyWiring:
    """The shared ``proxied_tool_names`` set must flow into ``PermissionHooks``."""

    def _prepare_node(self, proxied_tool_names):
        from datus.tools.registry.tool_registry import ToolRegistry

        node = _make_simple_node()
        node.permission_manager = MagicMock()
        node.permission_hooks = None
        node.tool_registry = ToolRegistry()
        node.execution_mode = None
        node.proxied_tool_names = proxied_tool_names
        # ``_make_filesystem_policy`` looks at agent_config / root path; stub.
        node._make_filesystem_policy = MagicMock(return_value=None)
        node._populate_tool_registry = MagicMock()
        return node

    def test_passes_shared_set_reference(self):
        """Constructor receives the *same* set object the node holds."""
        proxied = {"read_file"}
        node = self._prepare_node(proxied)

        with patch("datus.tools.permission.permission_hooks.PermissionHooks") as ph_cls:
            node._ensure_permission_hooks()

        ph_cls.assert_called_once()
        kwargs = ph_cls.call_args.kwargs
        # Must be the same set instance, not a copy — late ``apply_proxy_tools``
        # invocations mutate this set in place.
        assert kwargs["proxied_tool_names"] is proxied

    def test_default_empty_set_is_passed(self):
        """A freshly-built node with no proxied tools still passes an empty set."""
        node = self._prepare_node(set())

        with patch("datus.tools.permission.permission_hooks.PermissionHooks") as ph_cls:
            node._ensure_permission_hooks()

        kwargs = ph_cls.call_args.kwargs
        assert kwargs["proxied_tool_names"] == set()
