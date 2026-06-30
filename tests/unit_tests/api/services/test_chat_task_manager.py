"""Tests for datus.api.services.chat_task_manager — background task management."""

import asyncio
import re
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from datus.api.models.cli_models import (
    IMessageContent,
    SSEDataType,
    SSEEvent,
    SSEMessageData,
    SSEMessagePayload,
    SSEPingData,
)
from datus.api.services.chat_task_manager import (
    ChatTask,
    ChatTaskManager,
    _coalesce_deltas,
    _fill_database_context,
    _is_thinking_delta,
    _should_include_final_response,
)


class TestFillDatabaseContext:
    """Tests for _fill_database_context."""

    def test_no_database_falls_back_to_config(self, real_agent_config):
        original = real_agent_config.current_datasource
        catalog, database, schema = _fill_database_context(real_agent_config, database=None)
        assert real_agent_config.current_datasource == original
        assert catalog is None
        assert database == "california_schools"
        assert schema is None

    def test_empty_database_falls_back_to_config(self, real_agent_config):
        original = real_agent_config.current_datasource
        catalog, database, schema = _fill_database_context(real_agent_config, database="")
        assert real_agent_config.current_datasource == original
        assert catalog is None
        assert database == "california_schools"
        assert schema is None

    def test_explicit_database_does_not_override_datasource(self, real_agent_config):
        original = real_agent_config.current_datasource
        catalog, database, schema = _fill_database_context(real_agent_config, database="explicit_db")
        assert real_agent_config.current_datasource == original
        assert catalog is None
        assert database == "explicit_db"
        assert schema is None

    def test_context_falls_back_to_config_fields(self, real_agent_config):
        real_agent_config.current_db_config().catalog = "configured_catalog"
        real_agent_config.current_db_config().database = "configured_database"
        real_agent_config.current_db_config().schema = "configured_schema"
        original = real_agent_config.current_datasource
        catalog, database, schema = _fill_database_context(real_agent_config)
        assert real_agent_config.current_datasource == original
        assert catalog == "configured_catalog"
        assert database == "configured_database"
        assert schema == "configured_schema"


class TestChatTaskInit:
    """Tests for ChatTask initialization."""

    def test_initial_state(self):
        """ChatTask has correct initial state."""
        mock_task = MagicMock(spec=asyncio.Task)
        task = ChatTask(session_id="sess-1", asyncio_task=mock_task)

        assert task.session_id == "sess-1"
        assert task.asyncio_task is mock_task
        assert task.node is None
        assert task.events == []
        assert task.status == "running"
        assert task.error is None
        assert task.consumer_offset == 0
        assert isinstance(task.created_at, datetime)


class TestChatTaskManagerInit:
    """Tests for ChatTaskManager initialization."""

    def test_starts_empty(self):
        """Manager starts with no tasks."""
        manager = ChatTaskManager()
        assert manager._tasks == {}

    def test_default_source_and_interactive_defaults(self):
        """Defaults: source=None, interactive=True."""
        manager = ChatTaskManager()
        assert manager._default_source is None
        assert manager._default_interactive is True

    def test_default_source_and_interactive_stored(self):
        """Constructor stores explicit defaults."""
        manager = ChatTaskManager(default_source="web", default_interactive=False)
        assert manager._default_source == "web"
        assert manager._default_interactive is False


class TestChatTaskManagerCreateNodeInteractive:
    """Verify _create_node forwards interactive flag to ChatAgenticNode."""

    def test_create_node_passes_interactive_false(self, real_agent_config, mock_llm_create):
        """interactive=False reaches ChatAgenticNode and disables ask_user_tool."""
        manager = ChatTaskManager(default_interactive=False)
        node = manager._create_node(
            real_agent_config,
            subagent_id=None,
            node_id="sess-1",
            user_id=None,
            interactive=False,
        )
        assert node.execution_mode == "workflow"
        assert node.ask_user_tool is None

    def test_create_node_passes_interactive_true(self, real_agent_config, mock_llm_create):
        """interactive=True retains ask_user_tool setup."""
        manager = ChatTaskManager()
        node = manager._create_node(
            real_agent_config,
            subagent_id=None,
            node_id="sess-2",
            user_id=None,
            interactive=True,
        )
        assert node.execution_mode == "interactive"

    def test_has_active_tasks_returns_false_when_empty(self):
        """has_active_tasks is False when no tasks exist."""
        manager = ChatTaskManager()
        assert manager.has_active_tasks() is False

    def test_get_task_returns_none_for_missing(self):
        """get_task returns None for non-existent session."""
        manager = ChatTaskManager()
        assert manager.get_task("nonexistent") is None


class TestApplyPermissionModeOverride:
    """Verify per-request permission profile override semantics.

    These tests intentionally use lightweight stubs instead of a real
    AgentConfig/ChatAgenticNode pair: the override path is a pure
    delegation to ``PermissionManager.switch_profile`` and the
    interesting branches (no-op vs. real switch vs. graceful failure)
    are easier to assert directly.
    """

    def _make_agent_config(self, raw_permissions=None):
        from types import SimpleNamespace

        return SimpleNamespace(
            active_profile_name="normal",
            _raw_permissions=raw_permissions or {},
        )

    def _make_node(self, current_profile="normal", switch_side_effect=None):
        node = MagicMock()
        node.session_id = "sess-x"
        if current_profile is None:
            node.permission_manager = None
            return node
        pm = MagicMock()
        pm.active_profile = current_profile
        if switch_side_effect is not None:
            pm.switch_profile.side_effect = switch_side_effect
        node.permission_manager = pm
        return node

    def test_noop_when_permission_mode_is_none(self):
        """Falsy permission_mode must leave permission_manager alone."""
        manager = ChatTaskManager()
        node = self._make_node()
        manager._apply_permission_mode_override(node, self._make_agent_config(), None)
        node.permission_manager.switch_profile.assert_not_called()

    def test_noop_when_node_has_no_permission_manager(self):
        """Nodes without a permission_manager (e.g. workflow) are tolerated."""
        manager = ChatTaskManager()
        node = self._make_node(current_profile=None)
        assert node.permission_manager is None
        manager._apply_permission_mode_override(node, self._make_agent_config(), "dangerous")
        # Permission manager must remain untouched — the override path is
        # silently skipped when the node never had one to begin with.
        assert node.permission_manager is None

    def test_noop_when_already_on_target_profile(self):
        """Requested profile == active profile must skip the switch."""
        manager = ChatTaskManager()
        node = self._make_node(current_profile="dangerous")
        manager._apply_permission_mode_override(node, self._make_agent_config(), "dangerous")
        node.permission_manager.switch_profile.assert_not_called()

    def test_switches_profile_without_user_overrides(self):
        """When _raw_permissions is empty, switch_profile receives user_overrides=None."""
        manager = ChatTaskManager()
        node = self._make_node(current_profile="normal")
        manager._apply_permission_mode_override(node, self._make_agent_config(), "auto")
        node.permission_manager.switch_profile.assert_called_once_with("auto", user_overrides=None)

    def test_switches_profile_with_user_overrides(self):
        """Non-empty _raw_permissions yields a built user_overrides config."""
        from datus.tools.permission.permission_config import PermissionConfig

        manager = ChatTaskManager()
        node = self._make_node(current_profile="normal")
        raw = {"rules": [{"tool": "db_tools", "pattern": "*", "permission": "ask"}]}
        manager._apply_permission_mode_override(node, self._make_agent_config(raw), "dangerous")

        node.permission_manager.switch_profile.assert_called_once()
        args, kwargs = node.permission_manager.switch_profile.call_args
        assert args == ("dangerous",)
        assert isinstance(kwargs["user_overrides"], PermissionConfig)

    def test_raises_when_user_overrides_build_fails(self, monkeypatch):
        """Fail closed if agent.yml permissions.rules can't be parsed.

        Silently dropping malformed user rules and applying the bare
        profile base would broaden permissions beyond the operator's
        intent, so the override path must surface the error instead.
        """
        manager = ChatTaskManager()
        node = self._make_node(current_profile="normal")

        def _explode(*_args, **_kwargs):
            raise ValueError("malformed rule")

        monkeypatch.setattr(
            "datus.tools.permission.profiles.build_user_overrides",
            _explode,
        )

        with pytest.raises(RuntimeError, match="permission_mode='auto'"):
            manager._apply_permission_mode_override(
                node,
                self._make_agent_config({"rules": [{"bad": "shape"}]}),
                "auto",
            )
        node.permission_manager.switch_profile.assert_not_called()

    def test_swallows_switch_profile_failure(self):
        """A malformed override must not abort the chat turn."""
        from datus.utils.exceptions import DatusException, ErrorCode

        manager = ChatTaskManager()
        node = self._make_node(
            current_profile="normal",
            switch_side_effect=DatusException(code=ErrorCode.COMMON_CONFIG_ERROR),
        )
        # Should not raise.
        manager._apply_permission_mode_override(node, self._make_agent_config(), "auto")
        node.permission_manager.switch_profile.assert_called_once()


class TestChatTaskManagerBehavior:
    """Tests for ChatTaskManager task tracking."""

    def test_has_active_tasks_true_when_running(self):
        """has_active_tasks returns True when a task has running status."""
        manager = ChatTaskManager()
        task = ChatTask(session_id="s1", asyncio_task=MagicMock())
        task.status = "running"
        manager._tasks["s1"] = task
        assert manager.has_active_tasks() is True

    def test_has_active_tasks_false_when_completed(self):
        """has_active_tasks returns False when all tasks are completed."""
        manager = ChatTaskManager()
        task = ChatTask(session_id="s1", asyncio_task=MagicMock())
        task.status = "completed"
        manager._tasks["s1"] = task
        assert manager.has_active_tasks() is False

    def test_get_task_returns_existing(self):
        """get_task returns the task for an existing session."""
        manager = ChatTaskManager()
        task = ChatTask(session_id="s2", asyncio_task=MagicMock())
        manager._tasks["s2"] = task
        assert manager.get_task("s2") is task

    @pytest.mark.asyncio
    async def test_stop_task_missing_returns_false(self):
        """stop_task returns False for non-existent session."""
        manager = ChatTaskManager()
        assert await manager.stop_task("ghost") is False

    @pytest.mark.asyncio
    async def test_shutdown_completes_without_tasks(self):
        """shutdown completes cleanly with no tasks."""
        manager = ChatTaskManager()
        await manager.shutdown()
        assert manager._tasks == {}
        assert manager._completed_tasks == {}
        assert manager.has_active_tasks() is False

    @pytest.mark.asyncio
    async def test_wait_all_tasks_completes_without_tasks(self):
        """wait_all_tasks completes cleanly with no tasks."""
        manager = ChatTaskManager()
        await manager.wait_all_tasks()
        assert manager._tasks == {}
        assert manager._completed_tasks == {}
        assert manager.has_active_tasks() is False

    @pytest.mark.asyncio
    async def test_push_event_appends_to_buffer(self):
        """_push_event adds event to task's event list and notifies."""
        manager = ChatTaskManager()
        task = ChatTask(session_id="s3", asyncio_task=MagicMock())
        manager._tasks["s3"] = task

        from datus.api.models.cli_models import SSEEvent, SSEPingData

        event = SSEEvent(id=1, event="ping", data=SSEPingData(), timestamp="2025-01-01T00:00:00Z")
        await manager._push_event(task, event)
        assert len(task.events) == 1
        assert task.events[0] is event

    @pytest.mark.asyncio
    async def test_degraded_context_warning_emits_nonfatal_message(self):
        """Context degradation should be a normal assistant message, not an error event."""
        manager = ChatTaskManager()
        task = ChatTask(session_id="s-degraded", asyncio_task=MagicMock())
        node = SimpleNamespace(
            degraded_capabilities={
                "context_search_tools": "Context search and @ references are disabled; DB tools remain available."
            }
        )

        next_id = await manager._push_degraded_capability_warnings(task, node, 7)

        assert next_id == 8
        assert len(task.events) == 1
        event = task.events[0]
        assert event.id == 7
        assert event.event == "message"
        assert isinstance(event.data, SSEMessageData)
        assert event.data.type == SSEDataType.CREATE_MESSAGE
        assert event.data.payload.role == "assistant"
        assert event.data.payload.content[0].type == "markdown"
        assert "DB tools remain available" in event.data.payload.content[0].payload["content"]

    @pytest.mark.asyncio
    async def test_run_loop_emits_final_response_when_no_plain_assistant_response(self, real_agent_config):
        """The web stream must surface chat_response when the model only produced tool cards."""
        from datus.api.models.cli_models import StreamChatInput
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
        from datus.utils.trace_context import get_trace_context

        captured = {}

        class FakeNode:
            session_id = "s-final"

            def get_node_name(self):
                return "chat"

            async def execute_stream_with_interactions(self, action_history_manager):
                captured["trace_context"] = get_trace_context()
                yield ActionHistory(
                    action_id="final",
                    role=ActionRole.ASSISTANT,
                    action_type="chat_response",
                    messages="done",
                    input={},
                    output={"response": "1 table: orders"},
                    status=ActionStatus.SUCCESS,
                )

            async def get_last_turn_usage(self):
                return None

        manager = ChatTaskManager()
        manager._create_node = lambda *args, **kwargs: FakeNode()  # type: ignore[method-assign]
        task = ChatTask(session_id="s-final", asyncio_task=MagicMock())

        await manager._run_loop(task, real_agent_config, StreamChatInput(message="tables", session_id="s-final"))

        assert captured["trace_context"].name == "agent/chat"
        assert captured["trace_context"].session_id == "s-final"
        message_events = [event for event in task.events if event.event == "message"]
        assert len(message_events) == 1
        content = message_events[0].data.payload.content[0]
        assert content.type == "markdown"
        assert content.payload["content"] == "1 table: orders"

    @pytest.mark.asyncio
    async def test_run_loop_web_source_proxies_filesystem_writes(self, real_agent_config):
        """source='web' proxies the client-owned write tools (write/edit/delete_file)."""
        from datus.api.models.cli_models import StreamChatInput

        class FakeNode:
            session_id = "s-web-proxy"

            def get_node_name(self):
                return "chat"

            async def execute_stream_with_interactions(self, action_history_manager):
                return
                yield  # pragma: no cover - makes this an async generator

            async def get_last_turn_usage(self):
                return None

        manager = ChatTaskManager()
        manager._create_node = lambda *args, **kwargs: FakeNode()  # type: ignore[method-assign]
        task = ChatTask(session_id="s-web-proxy", asyncio_task=MagicMock())

        with patch("datus.api.services.chat_task_manager.apply_proxy_tools") as mock_apply:
            await manager._run_loop(
                task,
                real_agent_config,
                StreamChatInput(message="hi", source="web", session_id="s-web-proxy"),
            )

        mock_apply.assert_called_once()
        called_node, called_patterns = mock_apply.call_args.args
        assert isinstance(called_node, FakeNode)
        assert called_patterns == ["write_file", "edit_file", "delete_file"]
        assert task.status == "completed"

    def test_include_final_response_rejects_nested_subagent_response(self):
        """Depth>0 sub-agent wrappers must not render as top-level answers."""
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        action = ActionHistory(
            action_id="nested",
            role=ActionRole.ASSISTANT,
            action_type="gen_sql_response",
            messages="nested done",
            input={},
            output={"response": "internal sub-agent answer"},
            status=ActionStatus.SUCCESS,
            depth=1,
        )

        assert _should_include_final_response(action, assistant_response_sent=False) is False

    @pytest.mark.asyncio
    async def test_run_loop_ignores_nested_response_and_emits_parent_response(self, real_agent_config):
        """A forwarded sub-agent *_response must not hide the parent chat_response."""
        from datus.api.models.cli_models import StreamChatInput
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        class FakeNode:
            session_id = "s-nested-response"

            async def execute_stream_with_interactions(self, action_history_manager):
                yield ActionHistory(
                    action_id="nested",
                    role=ActionRole.ASSISTANT,
                    action_type="gen_sql_response",
                    messages="nested done",
                    input={},
                    output={"response": "internal sub-agent answer"},
                    status=ActionStatus.SUCCESS,
                    depth=1,
                )
                yield ActionHistory(
                    action_id="parent",
                    role=ActionRole.ASSISTANT,
                    action_type="chat_response",
                    messages="parent done",
                    input={},
                    output={"response": "top-level parent answer"},
                    status=ActionStatus.SUCCESS,
                    depth=0,
                )

            async def get_last_turn_usage(self):
                return None

        manager = ChatTaskManager()
        manager._create_node = lambda *args, **kwargs: FakeNode()  # type: ignore[method-assign]
        task = ChatTask(session_id="s-nested-response", asyncio_task=MagicMock())

        await manager._run_loop(
            task,
            real_agent_config,
            StreamChatInput(message="delegate", session_id="s-nested-response"),
        )

        message_events = [event for event in task.events if event.event == "message"]
        assert len(message_events) == 1
        content = message_events[0].data.payload.content[0]
        assert content.payload["content"] == "top-level parent answer"

    @pytest.mark.asyncio
    async def test_run_loop_skips_final_response_after_plain_assistant_response(self, real_agent_config):
        """chat_response is a wrapper and must not duplicate a streamed assistant response."""
        from datus.api.models.cli_models import StreamChatInput
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        class FakeNode:
            session_id = "s-dedupe"

            async def execute_stream_with_interactions(self, action_history_manager):
                yield ActionHistory(
                    action_id="plain",
                    role=ActionRole.ASSISTANT,
                    action_type="response",
                    messages="answer",
                    input={},
                    output={"raw_output": "1 table: orders", "is_thinking": False},
                    status=ActionStatus.SUCCESS,
                )
                yield ActionHistory(
                    action_id="final",
                    role=ActionRole.ASSISTANT,
                    action_type="chat_response",
                    messages="done",
                    input={},
                    output={"response": "1 table: orders"},
                    status=ActionStatus.SUCCESS,
                )

            async def get_last_turn_usage(self):
                return None

        manager = ChatTaskManager()
        manager._create_node = lambda *args, **kwargs: FakeNode()  # type: ignore[method-assign]
        task = ChatTask(session_id="s-dedupe", asyncio_task=MagicMock())

        await manager._run_loop(task, real_agent_config, StreamChatInput(message="tables", session_id="s-dedupe"))

        message_events = [event for event in task.events if event.event == "message"]
        assert len(message_events) == 1
        content = message_events[0].data.payload.content[0]
        assert content.payload["content"] == "1 table: orders"

    @pytest.mark.asyncio
    async def test_run_loop_skips_wrapper_after_post_tool_thinking_text(self, real_agent_config):
        """Post-tool visible assistant text suppresses chat_response even if marked as thinking."""
        from datus.api.models.cli_models import StreamChatInput
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        class FakeNode:
            session_id = "s-post-tool-thinking"

            async def execute_stream_with_interactions(self, action_history_manager):
                yield ActionHistory(
                    action_id="complete_tool",
                    role=ActionRole.TOOL,
                    action_type="list_tables",
                    messages="Tool call: list_tables",
                    input={"function_name": "list_tables"},
                    output={"summary": "3 tables: orders, customers, invoices"},
                    status=ActionStatus.SUCCESS,
                )
                yield ActionHistory(
                    action_id="assistant_text",
                    role=ActionRole.ASSISTANT,
                    action_type="message",
                    messages="answer",
                    input={},
                    output={"raw_output": "3 tables: orders, customers, invoices", "is_thinking": True},
                    status=ActionStatus.SUCCESS,
                )
                yield ActionHistory(
                    action_id="final",
                    role=ActionRole.ASSISTANT,
                    action_type="chat_response",
                    messages="done",
                    input={},
                    output={"response": "3 tables: orders, customers, invoices"},
                    status=ActionStatus.SUCCESS,
                )

            async def get_last_turn_usage(self):
                return None

        manager = ChatTaskManager()
        manager._create_node = lambda *args, **kwargs: FakeNode()  # type: ignore[method-assign]
        task = ChatTask(session_id="s-post-tool-thinking", asyncio_task=MagicMock())

        await manager._run_loop(
            task,
            real_agent_config,
            StreamChatInput(message="tables", session_id="s-post-tool-thinking"),
        )

        assistant_messages = [
            event
            for event in task.events
            if event.event == "message"
            and isinstance(event.data, SSEMessageData)
            and any(item.type in ("markdown", "thinking") for item in event.data.payload.content)
        ]
        assert len(assistant_messages) == 1
        assistant_text_events = [
            event
            for event in assistant_messages
            if any(
                item.payload.get("content") == "3 tables: orders, customers, invoices"
                for item in event.data.payload.content
            )
        ]
        assert len(assistant_text_events) == 1

    @pytest.mark.asyncio
    async def test_run_loop_dedupes_duplicate_plain_assistant_messages(self, real_agent_config):
        """Identical visible assistant messages in one turn are emitted once."""
        from datus.api.models.cli_models import StreamChatInput
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        class FakeNode:
            session_id = "s-plain-dedupe"

            async def execute_stream_with_interactions(self, action_history_manager):
                for action_id, action_type in (("assistant_1", "message"), ("assistant_2", "response")):
                    yield ActionHistory(
                        action_id=action_id,
                        role=ActionRole.ASSISTANT,
                        action_type=action_type,
                        messages="answer",
                        input={},
                        output={"raw_output": "3 tables: orders, customers, invoices", "is_thinking": False},
                        status=ActionStatus.SUCCESS,
                    )

            async def get_last_turn_usage(self):
                return None

        manager = ChatTaskManager()
        manager._create_node = lambda *args, **kwargs: FakeNode()  # type: ignore[method-assign]
        task = ChatTask(session_id="s-plain-dedupe", asyncio_task=MagicMock())

        await manager._run_loop(task, real_agent_config, StreamChatInput(message="tables", session_id="s-plain-dedupe"))

        assistant_text_events = [
            event
            for event in task.events
            if event.event == "message"
            and isinstance(event.data, SSEMessageData)
            and any(
                item.payload.get("content") == "3 tables: orders, customers, invoices"
                for item in event.data.payload.content
            )
        ]
        assert len(assistant_text_events) == 1

    @pytest.mark.asyncio
    async def test_stop_task_with_no_node_cancels_asyncio_task(self):
        """stop_task cancels asyncio task when node is not set."""
        manager = ChatTaskManager()
        mock_asyncio_task = MagicMock()
        mock_asyncio_task.done.return_value = False
        task = ChatTask(session_id="s4", asyncio_task=mock_asyncio_task)
        task.node = None
        manager._tasks["s4"] = task

        result = await manager.stop_task("s4")
        assert result is True
        mock_asyncio_task.cancel.assert_called_once()


@pytest.mark.asyncio
class TestStartChat:
    """Tests for start_chat — background task creation."""

    async def test_start_chat_creates_task(self, real_agent_config, mock_llm_create):
        """start_chat creates a ChatTask and returns it."""
        from datus.api.models.cli_models import StreamChatInput

        manager = ChatTaskManager()
        request = StreamChatInput(message="hello", session_id="start-test")
        task = await manager.start_chat(real_agent_config, request)
        assert isinstance(task, ChatTask)
        assert task.session_id == "start-test"
        assert task.status == "running"
        # Clean up
        await manager.shutdown()

    async def test_start_chat_fills_request_database_context(self, real_agent_config, monkeypatch):
        """start_chat fills omitted request database context before the run loop."""
        from datus.api.models.cli_models import StreamChatInput

        real_agent_config.current_db_config().catalog = "configured_catalog"
        real_agent_config.current_db_config().database = "configured_database"
        real_agent_config.current_db_config().schema = "configured_schema"
        captured = {}

        async def fake_run_loop(self, task, agent_config, request, **kwargs):
            captured["catalog"] = request.catalog
            captured["database"] = request.database
            captured["db_schema"] = request.db_schema

        monkeypatch.setattr(ChatTaskManager, "_run_loop", fake_run_loop)
        manager = ChatTaskManager()
        request = StreamChatInput(message="hello", session_id="api-context")
        task = await manager.start_chat(real_agent_config, request)
        await task.asyncio_task

        assert captured == {
            "catalog": "configured_catalog",
            "database": "configured_database",
            "db_schema": "configured_schema",
        }

    async def test_start_chat_duplicate_session_raises(self, real_agent_config, mock_llm_create):
        """start_chat raises ValueError for duplicate session_id."""
        from datus.api.models.cli_models import StreamChatInput

        manager = ChatTaskManager()
        request = StreamChatInput(message="hello", session_id="dup-session")
        await manager.start_chat(real_agent_config, request)

        with pytest.raises(ValueError, match="already running"):
            await manager.start_chat(real_agent_config, StreamChatInput(message="again", session_id="dup-session"))
        await manager.shutdown()

    async def test_start_chat_generates_session_id(self, real_agent_config, mock_llm_create):
        """start_chat generates session_id when not provided."""
        from datus.api.models.cli_models import StreamChatInput

        manager = ChatTaskManager()
        request = StreamChatInput(message="hello")
        task = await manager.start_chat(real_agent_config, request)
        assert re.fullmatch(r"chat_session_[0-9a-f]{8}", task.session_id)
        await manager.shutdown()

    async def test_start_chat_with_subagent(self, real_agent_config, mock_llm_create):
        """start_chat with sub_agent_id creates task with correct session pattern."""
        from datus.api.models.cli_models import StreamChatInput

        manager = ChatTaskManager()
        request = StreamChatInput(message="hello")
        task = await manager.start_chat(real_agent_config, request, sub_agent_id="gen_sql")
        assert "gen_sql" in task.session_id
        await manager.shutdown()

    async def test_start_chat_fills_database_context(self, real_agent_config, mock_llm_create):
        """start_chat fills database context from request."""
        from datus.api.models.cli_models import StreamChatInput

        manager = ChatTaskManager()
        request = StreamChatInput(message="hello", database="california_schools")
        task = await manager.start_chat(real_agent_config, request)
        assert isinstance(task, ChatTask)
        assert real_agent_config.current_datasource == "california_schools"
        await manager.shutdown()

    async def test_stop_running_task_with_node(self, real_agent_config, mock_llm_create):
        """stop_task interrupts a running task that has a node set."""
        from unittest.mock import MagicMock

        from datus.api.models.cli_models import StreamChatInput

        manager = ChatTaskManager()
        request = StreamChatInput(message="hello", session_id="stop-test")
        task = await manager.start_chat(real_agent_config, request)

        # Set a mock node with interrupt_controller
        mock_node = MagicMock()
        mock_node.interrupt_controller.interrupt = MagicMock()
        task.node = mock_node

        result = await manager.stop_task("stop-test")
        assert result is True
        mock_node.interrupt_controller.interrupt.assert_called_once()
        await manager.shutdown()

    async def test_wait_all_tasks_with_running(self, real_agent_config, mock_llm_create):
        """wait_all_tasks waits for running tasks without cancelling."""
        from datus.api.models.cli_models import StreamChatInput

        manager = ChatTaskManager()
        request = StreamChatInput(message="wait test", session_id="wait-test")
        task = await manager.start_chat(real_agent_config, request)

        # wait_all_tasks should return (tasks may finish quickly with mock LLM)
        await manager.wait_all_tasks()
        assert task.asyncio_task.done() is True
        assert manager._tasks == {}
        assert manager.get_task("wait-test") is task
        assert manager.has_active_tasks() is False
        await manager.shutdown()

    async def test_consume_events_yields_ping_when_idle(self, monkeypatch):
        """consume_events yields a ping event when idle past HEARTBEAT_INTERVAL."""
        from datus.api.models.cli_models import SSEEvent, SSEPingData
        from datus.api.services import chat_task_manager as ctm

        monkeypatch.setattr(ctm, "HEARTBEAT_INTERVAL", 0.05)

        manager = ChatTaskManager()
        task = ChatTask(session_id="ping-test", asyncio_task=MagicMock())
        task.status = "running"
        manager._tasks["ping-test"] = task

        gen = manager.consume_events(task, start_from=0)

        # First yield should be a ping triggered by timeout
        first = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
        assert first.event == "ping"

        # Now push a real event and ensure it is consumed next
        real_event = SSEEvent(id=1, event="message", data=SSEPingData(), timestamp="2025-01-01T00:00:00Z")
        await manager._push_event(task, real_event)

        nxt = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
        # May get another ping first if timing races; loop until we see the real event
        while nxt.event == "ping":
            nxt = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
        assert nxt.event == "message"
        assert nxt.id == 1

        # Mark task done so generator exits
        async with task.condition:
            task.status = "completed"
            task.condition.notify_all()
        await gen.aclose()

    async def test_consume_events_from_completed_task(self, real_agent_config, mock_llm_create):
        """consume_events yields buffered events from completed task."""
        from datus.api.models.cli_models import SSEEvent, SSEPingData

        manager = ChatTaskManager()
        task = ChatTask(session_id="consume-test", asyncio_task=MagicMock())
        task.status = "completed"
        event = SSEEvent(id=1, event="ping", data=SSEPingData(), timestamp="2025-01-01T00:00:00Z")
        task.events = [event]
        manager._tasks["consume-test"] = task

        events = []
        async for e in manager.consume_events(task, start_from=0):
            events.append(e)
        assert len(events) == 1
        assert events[0].id == 1


class TestResolveAtContext:
    """Tests for _resolve_at_context — @ reference resolution."""

    def test_resolve_empty_paths_returns_empty(self, real_agent_config):
        """_resolve_at_context with no paths returns empty lists."""
        manager = ChatTaskManager()
        tables, metrics, sqls = manager._resolve_at_context(real_agent_config, None, None, None)
        assert tables == []
        assert metrics == []
        assert sqls == []

    def test_resolve_with_empty_lists(self, real_agent_config):
        """_resolve_at_context with empty lists returns empty results."""
        manager = ChatTaskManager()
        tables, metrics, sqls = manager._resolve_at_context(real_agent_config, [], [], [])
        assert tables == []
        assert metrics == []
        assert sqls == []

    def test_resolve_nonexistent_paths(self, real_agent_config):
        """_resolve_at_context with nonexistent paths returns empty (no crash)."""
        manager = ChatTaskManager()
        tables, metrics, sqls = manager._resolve_at_context(
            real_agent_config,
            ["nonexistent/table/path"],
            ["nonexistent/metric/path"],
            ["nonexistent/sql/path"],
        )
        # Should return empty lists since paths don't exist
        assert isinstance(tables, list)
        assert isinstance(metrics, list)
        assert isinstance(sqls, list)

    def test_resolve_at_context_returns_empty_when_completer_fails(self, real_agent_config):
        manager = ChatTaskManager()

        with patch("datus.api.services.chat_task_manager.AtReferenceCompleter", side_effect=RuntimeError("hf offline")):
            tables, metrics, sqls = manager._resolve_at_context(
                real_agent_config,
                ["db.schema.table"],
                ["Finance.revenue"],
                ["Finance.sql"],
            )

        assert tables == []
        assert metrics == []
        assert sqls == []


class TestCreateNode:
    """Tests for _create_node — agentic node factory."""

    def test_create_gen_sql_node(self, real_agent_config, mock_llm_create):
        """_create_node creates GenSQLAgenticNode for gen_sql."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "gen_sql", "test-session")
        assert isinstance(node, GenSQLAgenticNode)

    def test_create_node_returns_agentic_node(self, real_agent_config, mock_llm_create):
        """_create_node returns an AgenticNode subclass for any valid subagent_id."""
        from datus.agent.node.agentic_node import AgenticNode

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "chat", "test-session")
        assert isinstance(node, AgenticNode)

    def test_create_default_node_for_none(self, real_agent_config, mock_llm_create):
        """_create_node creates an AgenticNode when subagent_id is None."""
        from datus.agent.node.agentic_node import AgenticNode

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, None, "test-session")
        assert isinstance(node, AgenticNode)

    def test_create_gen_semantic_model_node(self, real_agent_config, mock_llm_create):
        """_create_node creates GenSemanticModelAgenticNode for gen_semantic_model."""
        from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "gen_semantic_model", "test-session")
        assert isinstance(node, GenSemanticModelAgenticNode)

    def test_create_gen_metrics_node(self, real_agent_config, mock_llm_create):
        """_create_node creates GenMetricsAgenticNode for gen_metrics."""
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "gen_metrics", "test-session")
        assert isinstance(node, GenMetricsAgenticNode)

    def test_create_gen_report_node(self, real_agent_config, mock_llm_create):
        """gen_report must land on GenReportAgenticNode (regression: previously fell back to GenSQL)."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "gen_report", "test-session")
        assert isinstance(node, GenReportAgenticNode)
        assert not isinstance(node, GenSQLAgenticNode)

    def test_create_gen_table_node(self, real_agent_config, mock_llm_create):
        """gen_table must land on GenTableAgenticNode (regression: previously fell back to GenSQL)."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "gen_table", "test-session")
        assert isinstance(node, GenTableAgenticNode)
        assert not isinstance(node, GenSQLAgenticNode)

    def test_create_explore_node(self, real_agent_config, mock_llm_create):
        """explore must land on ExploreAgenticNode (regression: previously fell back to GenSQL)."""
        from datus.agent.node.explore_agentic_node import ExploreAgenticNode
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "explore", "test-session")
        assert isinstance(node, ExploreAgenticNode)
        assert not isinstance(node, GenSQLAgenticNode)

    def test_create_feedback_node(self, real_agent_config, mock_llm_create):
        """feedback continues to land on FeedbackAgenticNode through the shared factory."""
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "feedback", "test-session")
        assert isinstance(node, FeedbackAgenticNode)

    def test_custom_agent_node_class_gen_report(self, real_agent_config, mock_llm_create):
        """A custom sub_agent with node_class=gen_report must instantiate GenReportAgenticNode."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        real_agent_config.agentic_nodes["my_report_agent"] = {
            "system_prompt": "my_report_agent",
            "node_class": "gen_report",
            "tools": "db_tools.*",
            "max_turns": 5,
        }

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "my_report_agent", "test-session")
        assert isinstance(node, GenReportAgenticNode)
        assert not isinstance(node, GenSQLAgenticNode)

    def test_custom_agent_type_ask_metrics(self, real_agent_config, mock_llm_create):
        """API-created type=ask_metrics agents must instantiate AskMetricsAgenticNode."""
        from datus.agent.node.ask_metrics_agentic_node import AskMetricsAgenticNode
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        real_agent_config.agentic_nodes["my_metric_agent"] = {
            "system_prompt": "my_metric_agent",
            "type": "ask_metrics",
            "tools": "semantic_tools.query_metrics",
            "max_turns": 5,
        }

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "my_metric_agent", "test-session")
        assert isinstance(node, AskMetricsAgenticNode)
        assert not isinstance(node, GenSQLAgenticNode)

    def test_custom_agent_no_node_class_falls_back_to_gen_sql(self, real_agent_config, mock_llm_create):
        """A custom sub_agent without node_class must default to GenSQLAgenticNode."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode

        real_agent_config.agentic_nodes["plain_custom_agent"] = {
            "system_prompt": "plain_custom_agent",
            "tools": "db_tools.*",
            "max_turns": 5,
        }

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "plain_custom_agent", "test-session")
        assert isinstance(node, GenSQLAgenticNode)

    def test_custom_agent_uuid_resolved_to_node_class(self, real_agent_config, mock_llm_create):
        """API passes the UUID under "id"; factory must look up the name and honour node_class."""
        from datus.agent.node.gen_report_agentic_node import GenReportAgenticNode

        real_agent_config.agentic_nodes["my_report_agent"] = {
            "id": "11111111-2222-3333-4444-555555555555",
            "system_prompt": "my_report_agent",
            "node_class": "gen_report",
            "tools": "db_tools.*",
            "max_turns": 5,
        }

        manager = ChatTaskManager()
        node = manager._create_node(
            real_agent_config,
            "11111111-2222-3333-4444-555555555555",
            "test-session",
        )
        assert isinstance(node, GenReportAgenticNode)


class TestCreateNodeInput:
    """Tests for _create_node_input — input model factory."""

    def test_gen_sql_node_input(self, real_agent_config, mock_llm_create):
        """_create_node_input for GenSQLAgenticNode returns GenSQLNodeInput."""
        from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "gen_sql", "test")
        result = manager._create_node_input("test query", node, [], [], [])
        assert isinstance(result, GenSQLNodeInput)
        assert result.user_message == "test query"

    def test_default_node_input(self, real_agent_config, mock_llm_create):
        """_create_node_input for default node returns valid input."""
        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, None, "test")
        result = manager._create_node_input("hello", node, [], [], [])
        assert result.user_message == "hello"

    def test_semantic_model_node_input(self, real_agent_config, mock_llm_create):
        """_create_node_input for GenSemanticModelAgenticNode returns SemanticNodeInput."""
        from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "gen_semantic_model", "test")
        result = manager._create_node_input("generate model", node, [], [], [])
        assert isinstance(result, SemanticNodeInput)

    def test_metrics_node_input(self, real_agent_config, mock_llm_create):
        """_create_node_input for GenMetricsAgenticNode returns SemanticNodeInput."""
        from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "gen_metrics", "test")
        result = manager._create_node_input("generate metrics", node, [], [], [])
        assert isinstance(result, SemanticNodeInput)

    def test_sql_summary_node_input(self, real_agent_config, mock_llm_create):
        """_create_node_input for SqlSummaryAgenticNode returns SqlSummaryNodeInput."""
        from datus.schemas.sql_summary_agentic_node_models import SqlSummaryNodeInput

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "gen_sql_summary", "test")
        result = manager._create_node_input("summarize sql", node, [], [], [])
        assert isinstance(result, SqlSummaryNodeInput)

    def test_node_input_with_db_context(self, real_agent_config, mock_llm_create):
        """_create_node_input passes database context through."""

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "gen_sql", "test")
        result = manager._create_node_input(
            "test",
            node,
            [],
            [],
            [],
            catalog="cat",
            database="db",
            db_schema="schema",
        )
        assert result.catalog == "cat"
        assert result.database == "db"
        assert result.db_schema == "schema"

    def test_node_input_falls_back_to_config_db_context(self, real_agent_config, mock_llm_create):
        """_create_node_input fills missing database context from current config."""
        real_agent_config.current_db_config().catalog = "configured_catalog"
        real_agent_config.current_db_config().database = "configured_database"
        real_agent_config.current_db_config().schema = "configured_schema"

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "gen_sql", "test")
        result = manager._create_node_input("test", node, [], [], [])

        assert result.catalog == "configured_catalog"
        assert result.database == "configured_database"
        assert result.db_schema == "configured_schema"

    def test_feedback_node_input_with_source_session(self, real_agent_config, mock_llm_create):
        """_create_node_input for FeedbackAgenticNode carries source_session_id through."""
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode
        from datus.schemas.feedback_agentic_node_models import FeedbackNodeInput

        manager = ChatTaskManager()
        node = manager._create_node(real_agent_config, "feedback", "test")
        assert isinstance(node, FeedbackAgenticNode)

        result = manager._create_node_input(
            '[The user reacted to this message "reply" with [thumbsup]]',
            node,
            [],
            [],
            [],
            database="db",
            source_session_id="chat_session_xyz",
        )
        assert isinstance(result, FeedbackNodeInput)
        assert result.source_session_id == "chat_session_xyz"
        assert result.database == "db"


# ---------------------------------------------------------------------------
# Helpers for thinking-delta tests
# ---------------------------------------------------------------------------


def _make_thinking_delta(event_id: int, text: str, message_id: str = "m1", data_type=SSEDataType.APPEND_MESSAGE):
    """Create a thinking-delta SSEEvent."""
    return SSEEvent(
        id=event_id,
        event="message",
        data=SSEMessageData(
            type=data_type,
            payload=SSEMessagePayload(
                message_id=message_id,
                role="assistant",
                content=[IMessageContent(type="thinking", payload={"content": text})],
            ),
        ),
        timestamp="2025-01-01T00:00:00Z",
    )


def _make_markdown_event(event_id: int, text: str, message_id: str = "m1"):
    """Create a non-delta markdown message SSEEvent."""
    return SSEEvent(
        id=event_id,
        event="message",
        data=SSEMessageData(
            type=SSEDataType.APPEND_MESSAGE,
            payload=SSEMessagePayload(
                message_id=message_id,
                role="assistant",
                content=[IMessageContent(type="markdown", payload={"content": text})],
            ),
        ),
        timestamp="2025-01-01T00:00:00Z",
    )


def _make_ping_event(event_id: int = -1):
    return SSEEvent(id=event_id, event="ping", data=SSEPingData(), timestamp="2025-01-01T00:00:00Z")


# ---------------------------------------------------------------------------
# _is_thinking_delta tests
# ---------------------------------------------------------------------------


class TestIsThinkingDelta:
    """Tests for _is_thinking_delta identification."""

    def test_positive_append(self):
        """Correctly identifies APPEND_MESSAGE thinking delta."""
        ev = _make_thinking_delta(0, "hello")
        assert _is_thinking_delta(ev) is True

    def test_positive_create(self):
        """Correctly identifies CREATE_MESSAGE thinking delta."""
        ev = _make_thinking_delta(0, "hello", data_type=SSEDataType.CREATE_MESSAGE)
        assert _is_thinking_delta(ev) is True

    def test_negative_markdown(self):
        """Markdown content is not a thinking delta."""
        ev = _make_markdown_event(0, "hello")
        assert _is_thinking_delta(ev) is False

    def test_negative_update_message(self):
        """UPDATE_MESSAGE is not a thinking delta."""
        ev = SSEEvent(
            id=0,
            event="message",
            data=SSEMessageData(
                type=SSEDataType.UPDATE_MESSAGE,
                payload=SSEMessagePayload(
                    message_id="m1",
                    role="assistant",
                    content=[IMessageContent(type="thinking", payload={"content": "x"})],
                ),
            ),
            timestamp="t",
        )
        assert _is_thinking_delta(ev) is False

    def test_negative_ping_event(self):
        """Ping event is not a thinking delta."""
        ev = _make_ping_event()
        assert _is_thinking_delta(ev) is False

    def test_negative_empty_content(self):
        """Empty content list is not a thinking delta."""
        ev = SSEEvent(
            id=0,
            event="message",
            data=SSEMessageData(
                type=SSEDataType.APPEND_MESSAGE,
                payload=SSEMessagePayload(message_id="m1", role="assistant", content=[]),
            ),
            timestamp="t",
        )
        assert _is_thinking_delta(ev) is False


# ---------------------------------------------------------------------------
# _coalesce_deltas tests
# ---------------------------------------------------------------------------


class TestCoalesceDeltas:
    """Tests for _coalesce_deltas batch merging."""

    def test_empty_list(self):
        """Empty list returns empty list."""
        assert _coalesce_deltas([]) == []

    def test_single_delta(self):
        """Single delta is returned unchanged."""
        ev = _make_thinking_delta(0, "hi")
        result = _coalesce_deltas([ev])
        assert len(result) == 1
        assert result[0] is ev  # same object, no copy needed

    def test_merges_consecutive(self):
        """3 consecutive deltas merge into 1 with concatenated text."""
        evts = [_make_thinking_delta(i, f"part{i}") for i in range(3)]
        result = _coalesce_deltas(evts)
        assert len(result) == 1
        merged = result[0]
        assert merged.id == 0  # retains first event's id
        data = merged.data
        assert isinstance(data, SSEMessageData)
        assert data.payload.content[0].payload["content"] == "part0part1part2"

    def test_preserves_non_delta(self):
        """Non-delta events pass through unchanged."""
        md = _make_markdown_event(0, "hello")
        ping = _make_ping_event(1)
        result = _coalesce_deltas([md, ping])
        assert len(result) == 2
        assert result[0] is md
        assert result[1] is ping

    def test_mixed_sequence(self):
        """delta + non-delta + delta → 3 events (non-delta breaks run)."""
        d1 = _make_thinking_delta(0, "a")
        md = _make_markdown_event(1, "break")
        d2 = _make_thinking_delta(2, "b")
        result = _coalesce_deltas([d1, md, d2])
        assert len(result) == 3
        # First is the lone delta (unchanged)
        assert result[0] is d1
        # Second is the markdown
        assert result[1] is md
        # Third is the lone delta (unchanged)
        assert result[2] is d2

    def test_trailing_delta_run(self):
        """Non-delta followed by multiple deltas → 2 events."""
        md = _make_markdown_event(0, "start")
        d1 = _make_thinking_delta(1, "x")
        d2 = _make_thinking_delta(2, "y")
        result = _coalesce_deltas([md, d1, d2])
        assert len(result) == 2
        assert result[0] is md
        data = result[1].data
        assert isinstance(data, SSEMessageData)
        assert data.payload.content[0].payload["content"] == "xy"

    def test_different_message_ids_break_run(self):
        """Consecutive deltas with different message_ids are NOT merged."""
        d1 = _make_thinking_delta(0, "a", message_id="m1")
        d2 = _make_thinking_delta(1, "b", message_id="m1")
        d3 = _make_thinking_delta(2, "c", message_id="m2")
        d4 = _make_thinking_delta(3, "d", message_id="m2")
        result = _coalesce_deltas([d1, d2, d3, d4])
        # Should produce 2 merged events (one per message_id)
        assert len(result) == 2
        data0 = result[0].data
        data1 = result[1].data
        assert isinstance(data0, SSEMessageData)
        assert isinstance(data1, SSEMessageData)
        assert data0.payload.message_id == "m1"
        assert data0.payload.content[0].payload["content"] == "ab"
        assert data1.payload.message_id == "m2"
        assert data1.payload.content[0].payload["content"] == "cd"


# ---------------------------------------------------------------------------
# Integration: consume_events with coalescing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestConsumeEventsCoalescing:
    """Integration test: queued deltas are coalesced during consumption."""

    async def test_consume_events_coalesces_queued_deltas(self, monkeypatch):
        """Multiple queued thinking deltas are yielded as a single merged event."""
        from datus.api.services import chat_task_manager as ctm

        monkeypatch.setattr(ctm, "HEARTBEAT_INTERVAL", 0.05)

        manager = ChatTaskManager()
        task = ChatTask(session_id="coalesce-test", asyncio_task=MagicMock())
        task.status = "running"
        manager._tasks["coalesce-test"] = task

        # Push 3 thinking deltas while consumer is not running
        for i in range(3):
            await manager._push_event(task, _make_thinking_delta(i, f"chunk{i}"))

        # Mark done so consumer exits after draining
        async with task.condition:
            task.status = "completed"
            task.condition.notify_all()

        events = []
        async for e in manager.consume_events(task, start_from=0):
            events.append(e)

        # Should receive 1 merged event instead of 3
        assert len(events) == 1
        data = events[0].data
        assert isinstance(data, SSEMessageData)
        assert data.payload.content[0].payload["content"] == "chunk0chunk1chunk2"

        # cursor should have advanced past all 3 original events
        assert task.consumer_offset == 3


@pytest.mark.asyncio
class TestRunLoopPathManagerContext:
    """Regression: _run_loop must pin agent_config.path_manager into its own context.

    The gateway bridge dispatches messages from a Feishu SDK worker thread via
    ``asyncio.run_coroutine_threadsafe``. That thread never inherited the
    ContextVar set by ``AgentConfig.__init__``, so the spawned task starts
    with an empty ``_current_path_manager`` and downstream stores
    (``BaseSubjectEmbeddingStore`` -> ``get_subject_tree_store``) would fall
    back to a path manager with empty ``project_name``, raising
    ``create_rdb_for_store requires a non-empty project``.
    """

    async def test_run_loop_sets_path_manager_when_context_is_empty(self, real_agent_config, mock_llm_create):
        """Even when the calling context has no path manager, _run_loop binds
        agent_config.path_manager so downstream get_path_manager() callers see
        the right project_name."""
        from datus.api.models.cli_models import StreamChatInput
        from datus.utils.path_manager import (
            _current_path_manager,
            get_path_manager,
            reset_path_manager,
        )

        captured: dict[str, str] = {}

        original_create_node = ChatTaskManager._create_node

        def _capturing_create_node(self, agent_config, subagent_id, node_id, **kwargs):
            # Simulate what BaseSubjectEmbeddingStore.__init__ does at line 692:
            # rely on the ambient ContextVar to find the project name.
            captured["project_name"] = get_path_manager().project_name
            return original_create_node(self, agent_config, subagent_id, node_id, **kwargs)

        manager = ChatTaskManager()
        manager._create_node = _capturing_create_node.__get__(manager, ChatTaskManager)  # type: ignore[method-assign]

        # Wipe the ContextVar to mimic the Feishu-thread dispatch case. The
        # task created by start_chat will inherit this empty context.
        token = _current_path_manager.set(None)
        try:
            assert _current_path_manager.get() is None
            request = StreamChatInput(message="hello", session_id="path-mgr-test")
            task = await manager.start_chat(real_agent_config, request)
            # Wait for the background loop to finish (mock LLM returns immediately).
            await asyncio.wait_for(task.asyncio_task, timeout=5.0)
        finally:
            reset_path_manager(token)
            await manager.shutdown()

        expected = real_agent_config.project_name
        assert expected, "real_agent_config.project_name must be non-empty for this test"
        assert captured.get("project_name") == expected, (
            f"_run_loop did not pin path_manager: got {captured.get('project_name')!r}, expected {expected!r}"
        )

    async def test_run_loop_does_not_leak_path_manager_to_caller(self, real_agent_config, mock_llm_create):
        """The ContextVar.set inside _run_loop must stay scoped to its own task
        and not bleed back into the calling context."""
        from datus.api.models.cli_models import StreamChatInput
        from datus.utils.path_manager import _current_path_manager, reset_path_manager

        manager = ChatTaskManager()
        token = _current_path_manager.set(None)
        try:
            request = StreamChatInput(message="hello", session_id="leak-test")
            task = await manager.start_chat(real_agent_config, request)
            await asyncio.wait_for(task.asyncio_task, timeout=5.0)
            # Caller's view stays untouched: the spawned task has its own context.
            assert _current_path_manager.get() is None
        finally:
            reset_path_manager(token)
            await manager.shutdown()


class _StubGenSQLNode:
    def __init__(self, **kwargs):
        self.node_name = kwargs.get("node_name")
        self.kwargs = kwargs


class TestCreateNodeCustomSubAgent:
    """Tests for _create_node custom sub_agent branch — node_name resolution."""

    def test_custom_subagent_resolves_sanitized_key(self, monkeypatch):
        """Custom sub_agent UUID resolves to sanitized node_name via agentic_nodes."""
        monkeypatch.setattr(
            "datus.agent.node.gen_sql_agentic_node.GenSQLAgenticNode",
            _StubGenSQLNode,
        )
        agent_config = MagicMock()
        agent_config.agentic_nodes = {
            "my_sanitized_name": {"id": "uuid-123", "system_prompt": "my_sanitized_name"},
        }
        node = ChatTaskManager()._create_node(agent_config, "uuid-123", "s1")
        assert isinstance(node, _StubGenSQLNode)
        assert node.node_name == "my_sanitized_name"

    def test_custom_subagent_unknown_falls_back_to_id(self, monkeypatch):
        """Unknown subagent_id is used as-is when no matching entry exists."""
        monkeypatch.setattr(
            "datus.agent.node.gen_sql_agentic_node.GenSQLAgenticNode",
            _StubGenSQLNode,
        )
        agent_config = MagicMock()
        agent_config.agentic_nodes = {
            "my_sanitized_name": {"id": "uuid-123", "system_prompt": "my_sanitized_name"},
        }
        node = ChatTaskManager()._create_node(agent_config, "unknown", "s1")
        assert isinstance(node, _StubGenSQLNode)
        assert node.node_name == "unknown"


class TestStartChatLanguageOverride:
    """``StreamChatInput.language`` must land on the cloned config's
    ``language`` attribute so every downstream AgenticNode sees it.

    We short-circuit the async loop to avoid spinning up real nodes: the
    override happens synchronously inside ``start_chat`` before
    ``_run_loop`` is awaited.
    """

    @pytest.mark.asyncio
    async def test_request_language_overrides_cloned_config(self, real_agent_config, monkeypatch):
        from datus.api.models.cli_models import StreamChatInput

        real_agent_config.language = "en"
        captured = {}

        async def fake_run_loop(self, task, agent_config, request, **kwargs):
            captured["agent_config"] = agent_config

        monkeypatch.setattr(ChatTaskManager, "_run_loop", fake_run_loop)
        manager = ChatTaskManager()
        request = StreamChatInput(message="hi", language="zh")
        task = await manager.start_chat(real_agent_config, request)
        await task.asyncio_task  # drain the fake loop
        assert captured["agent_config"].language == "zh"
        # Source config remains untouched because start_chat deep-copies.
        assert real_agent_config.language == "en"

    @pytest.mark.asyncio
    async def test_missing_language_preserves_yaml_default(self, real_agent_config, monkeypatch):
        from datus.api.models.cli_models import StreamChatInput

        real_agent_config.language = "zh"
        captured = {}

        async def fake_run_loop(self, task, agent_config, request, **kwargs):
            captured["agent_config"] = agent_config

        monkeypatch.setattr(ChatTaskManager, "_run_loop", fake_run_loop)
        manager = ChatTaskManager()
        request = StreamChatInput(message="hi")  # no language field
        task = await manager.start_chat(real_agent_config, request)
        await task.asyncio_task
        assert captured["agent_config"].language == "zh"


class TestStartChatModelOverride:
    """``ChatInput.model`` (format ``provider/model_id``) must override the
    cloned config's active model with the highest priority.
    """

    @pytest.mark.asyncio
    async def test_provider_model_override(self, real_agent_config, monkeypatch):
        from datus.api.models.cli_models import StreamChatInput

        captured = {}

        async def fake_run_loop(self, task, agent_config, request, **kwargs):
            captured["agent_config"] = agent_config

        monkeypatch.setattr(ChatTaskManager, "_run_loop", fake_run_loop)
        manager = ChatTaskManager()
        request = StreamChatInput(message="hi", model="openai/gpt-4.1")
        task = await manager.start_chat(real_agent_config, request)
        await task.asyncio_task
        assert captured["agent_config"]._target_provider == "openai"
        assert captured["agent_config"]._target_model == "gpt-4.1"

    @pytest.mark.asyncio
    async def test_custom_model_override(self, real_agent_config, monkeypatch):
        from datus.api.models.cli_models import StreamChatInput

        captured = {}

        async def fake_run_loop(self, task, agent_config, request, **kwargs):
            captured["agent_config"] = agent_config

        monkeypatch.setattr(ChatTaskManager, "_run_loop", fake_run_loop)
        manager = ChatTaskManager()
        request = StreamChatInput(message="hi", model="custom/mock")
        task = await manager.start_chat(real_agent_config, request)
        await task.asyncio_task
        assert captured["agent_config"].target == "mock"
        assert captured["agent_config"]._target_provider is None
        assert captured["agent_config"]._target_model is None

    @pytest.mark.asyncio
    async def test_model_with_slash_in_id(self, real_agent_config, monkeypatch):
        from datus.api.models.cli_models import StreamChatInput

        captured = {}

        async def fake_run_loop(self, task, agent_config, request, **kwargs):
            captured["agent_config"] = agent_config

        monkeypatch.setattr(ChatTaskManager, "_run_loop", fake_run_loop)
        manager = ChatTaskManager()
        request = StreamChatInput(message="hi", model="openai/org/gpt-4.1")
        task = await manager.start_chat(real_agent_config, request)
        await task.asyncio_task
        assert captured["agent_config"]._target_provider == "openai"
        assert captured["agent_config"]._target_model == "org/gpt-4.1"

    @pytest.mark.asyncio
    async def test_model_without_slash_raises(self, real_agent_config, monkeypatch):
        from datus.api.models.cli_models import StreamChatInput

        async def fake_run_loop(self, task, agent_config, request, **kwargs):
            pass

        monkeypatch.setattr(ChatTaskManager, "_run_loop", fake_run_loop)
        manager = ChatTaskManager()
        request = StreamChatInput(message="hi", model="gpt-4.1")
        with pytest.raises(ValueError, match="expected 'provider/model_id'"):
            await manager.start_chat(real_agent_config, request)

    @pytest.mark.asyncio
    async def test_model_none_preserves_config(self, real_agent_config, monkeypatch):
        from datus.api.models.cli_models import StreamChatInput

        captured = {}

        async def fake_run_loop(self, task, agent_config, request, **kwargs):
            captured["agent_config"] = agent_config

        monkeypatch.setattr(ChatTaskManager, "_run_loop", fake_run_loop)
        manager = ChatTaskManager()
        request = StreamChatInput(message="hi")
        task = await manager.start_chat(real_agent_config, request)
        await task.asyncio_task
        assert captured["agent_config"].target == real_agent_config.target

    @pytest.mark.asyncio
    async def test_model_override_does_not_mutate_source(self, real_agent_config, monkeypatch):
        from datus.api.models.cli_models import StreamChatInput

        original_target = real_agent_config.target
        original_target_provider = real_agent_config._target_provider

        async def fake_run_loop(self, task, agent_config, request, **kwargs):
            pass

        monkeypatch.setattr(ChatTaskManager, "_run_loop", fake_run_loop)
        manager = ChatTaskManager()
        request = StreamChatInput(message="hi", model="openai/gpt-4.1")
        task = await manager.start_chat(real_agent_config, request)
        await task.asyncio_task
        assert real_agent_config.target == original_target
        assert real_agent_config._target_provider == original_target_provider

    @pytest.mark.asyncio
    async def test_custom_model_unknown_raises(self, real_agent_config, monkeypatch):
        from datus.api.models.cli_models import StreamChatInput
        from datus.utils.exceptions import DatusException

        async def fake_run_loop(self, task, agent_config, request, **kwargs):
            pass

        monkeypatch.setattr(ChatTaskManager, "_run_loop", fake_run_loop)
        manager = ChatTaskManager()
        request = StreamChatInput(message="hi", model="custom/nonexistent")
        with pytest.raises(DatusException):
            await manager.start_chat(real_agent_config, request)


class TestStartChatRemoteSourceHardening:
    """Remote front-ends (vscode/web) must not see a server-side BashTool.
    ``filesystem_strict`` is always on for the API surface;
    ``bash_tool_enabled`` is additionally forced off when
    ``effective_source in {vscode, web}``. ``project_root`` is intentionally
    untouched — web keeps its configured root and the read-only property
    falls back to CWD when empty.
    """

    @pytest.mark.asyncio
    async def test_vscode_source_disables_bash_tool(self, real_agent_config, monkeypatch):
        from datus.api.models.cli_models import StreamChatInput

        real_agent_config.bash_tool_enabled = True
        captured = {}

        async def fake_run_loop(self, task, agent_config, request, **kwargs):
            captured["agent_config"] = agent_config

        monkeypatch.setattr(ChatTaskManager, "_run_loop", fake_run_loop)
        manager = ChatTaskManager()
        request = StreamChatInput(message="hi", source="vscode", session_id="vscode-hardening")
        task = await manager.start_chat(real_agent_config, request)
        await task.asyncio_task

        cfg = captured["agent_config"]
        assert cfg.filesystem_strict is True
        assert cfg.bash_tool_enabled is False
        assert cfg._client_source == "vscode"
        # Source config remains untouched because start_chat deep-copies.
        assert real_agent_config.bash_tool_enabled is True

    @pytest.mark.asyncio
    async def test_web_source_disables_bash_tool(self, real_agent_config, monkeypatch):
        from datus.api.models.cli_models import StreamChatInput

        real_agent_config.bash_tool_enabled = True
        captured = {}

        async def fake_run_loop(self, task, agent_config, request, **kwargs):
            captured["agent_config"] = agent_config

        monkeypatch.setattr(ChatTaskManager, "_run_loop", fake_run_loop)
        manager = ChatTaskManager()
        request = StreamChatInput(message="hi", source="web", session_id="web-hardening")
        task = await manager.start_chat(real_agent_config, request)
        await task.asyncio_task

        cfg = captured["agent_config"]
        assert cfg.filesystem_strict is True
        assert cfg.bash_tool_enabled is False
        assert cfg._client_source == "web"

    @pytest.mark.asyncio
    async def test_default_source_vscode_applies_hardening_without_request_source(self, real_agent_config, monkeypatch):
        """A daemon launched with --source vscode hardens every request that
        does not override source explicitly."""
        from datus.api.models.cli_models import StreamChatInput

        real_agent_config.bash_tool_enabled = True
        captured = {}

        async def fake_run_loop(self, task, agent_config, request, **kwargs):
            captured["agent_config"] = agent_config

        monkeypatch.setattr(ChatTaskManager, "_run_loop", fake_run_loop)
        manager = ChatTaskManager(default_source="vscode")
        request = StreamChatInput(message="hi", session_id="default-vscode-hardening")
        task = await manager.start_chat(real_agent_config, request)
        await task.asyncio_task

        cfg = captured["agent_config"]
        assert cfg.bash_tool_enabled is False

    @pytest.mark.asyncio
    async def test_no_source_preserves_bash_settings(self, real_agent_config, monkeypatch):
        """CLI / no-source requests keep whatever bash_tool_enabled value was
        in agent.yml. Only filesystem_strict is unconditionally forced on for
        the API surface."""
        from datus.api.models.cli_models import StreamChatInput

        real_agent_config.bash_tool_enabled = True
        captured = {}

        async def fake_run_loop(self, task, agent_config, request, **kwargs):
            captured["agent_config"] = agent_config

        monkeypatch.setattr(ChatTaskManager, "_run_loop", fake_run_loop)
        manager = ChatTaskManager()  # default_source=None
        request = StreamChatInput(message="hi", session_id="no-source-baseline")
        task = await manager.start_chat(real_agent_config, request)
        await task.asyncio_task

        cfg = captured["agent_config"]
        assert cfg.filesystem_strict is True
        assert cfg.bash_tool_enabled is True
        assert cfg._client_source is None


class TestStartChatDatasourceOverride:
    """A per-request ``datasource`` (e.g. an IM channel override) switches the
    connection profile without leaking into the physical ``database`` slot."""

    @pytest.mark.asyncio
    async def test_request_datasource_switches_current_datasource(self, real_agent_config, monkeypatch):
        import copy as _copy

        from datus.api.models.cli_models import StreamChatInput

        # Inject a second datasource so the switch is observable (not a no-op).
        sources = real_agent_config.services.datasources
        sources["analytics"] = _copy.deepcopy(sources["california_schools"])
        assert real_agent_config.current_datasource == "california_schools"

        captured = {}

        async def fake_run_loop(self, task, agent_config, request, **kwargs):
            captured["agent_config"] = agent_config

        monkeypatch.setattr(ChatTaskManager, "_run_loop", fake_run_loop)
        manager = ChatTaskManager()
        request = StreamChatInput(message="hi", session_id="ds-override", datasource="analytics")
        task = await manager.start_chat(real_agent_config, request)
        await task.asyncio_task

        cfg = captured["agent_config"]
        # The cloned config switched datasource; the datasource name never landed in database
        # (database, if filled by _fill_database_context, is the resolved physical db name).
        assert cfg.current_datasource == "analytics"
        assert request.database != "analytics"
        # Original config untouched because start_chat deep-copies.
        assert real_agent_config.current_datasource == "california_schools"

    @pytest.mark.asyncio
    async def test_invalid_request_datasource_raises(self, real_agent_config, monkeypatch):
        from datus.api.models.cli_models import StreamChatInput
        from datus.utils.exceptions import DatusException

        monkeypatch.setattr(ChatTaskManager, "_run_loop", lambda *a, **k: None)
        manager = ChatTaskManager()
        request = StreamChatInput(message="hi", session_id="ds-invalid", datasource="nonexistent")
        with pytest.raises(DatusException):
            await manager.start_chat(real_agent_config, request)
