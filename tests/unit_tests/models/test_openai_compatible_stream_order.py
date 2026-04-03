# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for stream ordering in OpenAICompatibleModel._generate_with_tools_stream_internal().

Validates that PROCESSING actions are yielded immediately (not buffered) to support
proxy tool mode where external callers need the call-tool event before providing results.

CI level: zero external deps, mock all SDK interactions.
"""

from dataclasses import dataclass, field
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus

# ---------------------------------------------------------------------------
# Lightweight fakes for OpenAI Agents SDK streaming objects
# ---------------------------------------------------------------------------


@dataclass
class FakeTextContent:
    text: str


@dataclass
class FakeRawItemWithContent:
    """Fake raw_item for message_output_item (assistant thinking)."""

    content: list = field(default_factory=list)


@dataclass
class FakeRawItemToolCall:
    """Fake raw_item for tool_call_item."""

    name: str = "test_tool"
    arguments: str = '{"key": "value"}'
    call_id: str = "call_001"


@dataclass
class FakeRawItemToolOutput:
    """Fake raw_item for tool_call_output_item (dict-style or object-style)."""

    call_id: str = "call_001"


@dataclass
class FakeItem:
    type: str
    raw_item: object = None
    output: str = ""


@dataclass
class FakeEvent:
    type: str = "run_item_stream_event"
    item: object = None
    data: object = None


@dataclass
class FakeOutputItemDoneData:
    """Fake data for raw_response_event containing ResponseOutputItemDoneEvent."""

    type: str = "response.output_item.done"
    item: object = None


@dataclass
class FakeMessageItem:
    """Fake message item inside ResponseOutputItemDoneEvent."""

    type: str = "message"
    content: list = field(default_factory=list)


def _make_tool_call_event(call_id="call_001", tool_name="test_tool", arguments='{"key":"value"}'):
    raw = FakeRawItemToolCall(name=tool_name, arguments=arguments, call_id=call_id)
    return FakeEvent(item=FakeItem(type="tool_call_item", raw_item=raw))


def _make_tool_output_event(call_id="call_001", output="result text"):
    raw = FakeRawItemToolOutput(call_id=call_id)
    return FakeEvent(item=FakeItem(type="tool_call_output_item", raw_item=raw, output=output))


def _make_message_event(text="I will now query the database"):
    """Create a RunItemStreamEvent for message_output_item (Phase 3 / fallback)."""
    raw = FakeRawItemWithContent(content=[FakeTextContent(text=text)])
    return FakeEvent(item=FakeItem(type="message_output_item", raw_item=raw))


def _make_raw_message_done_event(text="I will now query the database"):
    """Create a raw_response_event containing ResponseOutputItemDoneEvent for a message.

    This simulates the real SDK behavior where the assistant message's text content
    is available in a raw event BEFORE tool execution starts.
    """
    msg_item = FakeMessageItem(content=[FakeTextContent(text=text)])
    data = FakeOutputItemDoneData(item=msg_item)
    return FakeEvent(type="raw_response_event", data=data)


def _make_raw_other_event():
    """Create a raw_response_event that is NOT a message output item done event."""
    return FakeEvent(type="raw_response_event", data=MagicMock(type="response.output_text.delta"))


# ---------------------------------------------------------------------------
# Helper to drive the streaming generator with a sequence of fake events
# ---------------------------------------------------------------------------


def _build_fake_result(events_list):
    """Build a fake Runner.run_streamed result that yields events then marks complete."""

    class FakeResult:
        def __init__(self):
            self._events = list(events_list)
            self._consumed = False

        @property
        def is_complete(self):
            return self._consumed

        async def stream_events(self):
            for ev in self._events:
                yield ev
            self._consumed = True

        def final_output_as(self, _type):
            return "final"

        def to_input_list(self):
            return []

    return FakeResult()


async def _collect_actions(events_list) -> List[ActionHistory]:
    """Run the streaming method with fake events and collect yielded actions."""
    from datus.models.openai_compatible import OpenAICompatibleModel

    model = object.__new__(OpenAICompatibleModel)
    # Provide minimal attributes that _generate_with_tools_stream_internal needs
    model.model_name = "test-model"
    model._format_tool_result = lambda content, tool_name="": f"result: {content[:20]}"
    model._format_tool_result_from_dict = lambda data, tool_name="": f"result: {str(data)[:20]}"
    model._setup_custom_json_encoder = lambda: None
    model._extract_and_distribute_token_usage = AsyncMock()

    # model_config needed by retry wrapper
    mock_config = MagicMock()
    mock_config.max_retry = 1
    mock_config.retry_interval = 0
    model.model_config = mock_config
    model.default_headers = None

    fake_result = _build_fake_result(events_list)

    action_history_manager = ActionHistoryManager()

    # Patch Runner.run_streamed to return our fake result
    with patch("datus.models.openai_compatible.Runner") as mock_runner:
        mock_runner.run_streamed.return_value = fake_result

        # Patch Agent constructor
        with patch("datus.models.openai_compatible.Agent"):
            # Patch multiple_mcp_servers context manager
            with patch("datus.models.openai_compatible.multiple_mcp_servers") as mock_mcp:
                mock_mcp.return_value.__aenter__ = AsyncMock(return_value={})
                mock_mcp.return_value.__aexit__ = AsyncMock(return_value=False)

                # Patch litellm_adapter
                model.litellm_adapter = MagicMock()
                model.litellm_adapter.get_agents_sdk_model.return_value = MagicMock()
                model.litellm_adapter.provider = "openai"
                model.litellm_adapter.is_thinking_model = False

                actions = []
                async for action in model._generate_with_tools_stream_internal(
                    prompt="test prompt",
                    mcp_servers=None,
                    tools=None,
                    instruction="test instruction",
                    output_type=str,
                    strict_json_schema=False,
                    max_turns=10,
                    session=None,
                    action_history_manager=action_history_manager,
                ):
                    actions.append(action)

    return actions


# ---------------------------------------------------------------------------
# Tests: PROCESSING actions are yielded immediately (no buffering)
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestStreamActionOrdering:
    """Tests for immediate PROCESSING yield behavior."""

    @pytest.mark.asyncio
    async def test_processing_yielded_before_assistant_when_tool_starts_first(self):
        """SDK order: tool_call_item -> message_output -> tool_call_output.
        Expected yield: PROCESSING -> ASSISTANT -> SUCCESS.
        PROCESSING is yielded immediately, not buffered until ASSISTANT.
        """
        events = [
            _make_tool_call_event(call_id="call_A"),
            _make_message_event("Let me think about this"),
            _make_tool_output_event(call_id="call_A"),
        ]

        actions = await _collect_actions(events)

        roles_and_statuses = [(a.role, a.status) for a in actions]
        assert roles_and_statuses == [
            (ActionRole.TOOL, ActionStatus.PROCESSING),  # yielded immediately
            (ActionRole.ASSISTANT, ActionStatus.SUCCESS),  # thinking message
            (ActionRole.TOOL, ActionStatus.SUCCESS),  # tool complete
        ]

    @pytest.mark.asyncio
    async def test_no_message_yields_processing_then_success(self):
        """SDK order: tool_call_item -> tool_call_output (no message).
        Expected yield: PROCESSING -> SUCCESS.
        """
        events = [
            _make_tool_call_event(call_id="call_B"),
            _make_tool_output_event(call_id="call_B"),
        ]

        actions = await _collect_actions(events)

        roles_and_statuses = [(a.role, a.status) for a in actions]
        assert roles_and_statuses == [
            (ActionRole.TOOL, ActionStatus.PROCESSING),
            (ActionRole.TOOL, ActionStatus.SUCCESS),
        ]

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_yielded_immediately(self):
        """SDK order: tool_A_start -> tool_B_start -> message -> output_A -> output_B.
        Expected yield: PROC_A -> PROC_B -> ASSISTANT -> SUCC_A -> SUCC_B.
        """
        events = [
            _make_tool_call_event(call_id="call_A", tool_name="tool_a"),
            _make_tool_call_event(call_id="call_B", tool_name="tool_b"),
            _make_message_event("Running two tools"),
            _make_tool_output_event(call_id="call_A", output="result_a"),
            _make_tool_output_event(call_id="call_B", output="result_b"),
        ]

        actions = await _collect_actions(events)

        roles_and_statuses = [(a.role, a.status) for a in actions]
        assert roles_and_statuses == [
            (ActionRole.TOOL, ActionStatus.PROCESSING),  # tool_a start
            (ActionRole.TOOL, ActionStatus.PROCESSING),  # tool_b start
            (ActionRole.ASSISTANT, ActionStatus.SUCCESS),  # thinking
            (ActionRole.TOOL, ActionStatus.SUCCESS),  # tool_a complete
            (ActionRole.TOOL, ActionStatus.SUCCESS),  # tool_b complete
        ]

        # Verify tool names in order
        assert actions[0].action_type == "tool_a"
        assert actions[1].action_type == "tool_b"

    @pytest.mark.asyncio
    async def test_message_before_tool_start_yields_correct_order(self):
        """SDK order: message_output -> tool_call_item -> tool_call_output.
        Expected yield: ASSISTANT -> PROCESSING -> SUCCESS.
        """
        events = [
            _make_message_event("I will query now"),
            _make_tool_call_event(call_id="call_C"),
            _make_tool_output_event(call_id="call_C"),
        ]

        actions = await _collect_actions(events)

        roles_and_statuses = [(a.role, a.status) for a in actions]
        assert roles_and_statuses == [
            (ActionRole.ASSISTANT, ActionStatus.SUCCESS),
            (ActionRole.TOOL, ActionStatus.PROCESSING),
            (ActionRole.TOOL, ActionStatus.SUCCESS),
        ]

    @pytest.mark.asyncio
    async def test_tool_call_without_output_yields_processing(self):
        """SDK order: tool_call_item (no output, no message).
        Expected: PROCESSING yielded immediately.
        """
        events = [
            _make_tool_call_event(call_id="call_D"),
        ]

        actions = await _collect_actions(events)

        assert len(actions) == 1
        assert actions[0].role == ActionRole.TOOL
        assert actions[0].status == ActionStatus.PROCESSING
        assert actions[0].action_id == "call_D"

    @pytest.mark.asyncio
    async def test_processing_action_id_matches_call_id(self):
        """Verify that PROCESSING actions preserve the correct action_id."""
        events = [
            _make_tool_call_event(call_id="unique_123", tool_name="my_tool"),
            _make_message_event("thinking"),
            _make_tool_output_event(call_id="unique_123"),
        ]

        actions = await _collect_actions(events)

        processing_action = [a for a in actions if a.status == ActionStatus.PROCESSING][0]
        assert processing_action.action_id == "unique_123"
        assert processing_action.action_type == "my_tool"


# ---------------------------------------------------------------------------
# Tests: Raw event early capture (real SDK event ordering)
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestRawEventEarlyCapture:
    """Tests for assistant text capture from raw_response_event.

    In real SDK streaming, the event queue order is:
      tool_call_item -> raw(message_done) -> [tool execution] -> message_output_item -> tool_output_item
    PROCESSING is yielded immediately at tool_call_item. The raw event provides
    assistant text before tool execution completes.
    """

    @pytest.mark.asyncio
    async def test_raw_message_before_tool_output_yields_processing_first(self):
        """Real SDK order: tool_called -> raw(message_done) -> message_output -> tool_output.
        Expected: PROCESSING -> ASSISTANT (from raw) -> SUCCESS.
        The Phase 3 message_output should be skipped (duplicate).
        """
        events = [
            _make_tool_call_event(call_id="call_R1"),
            _make_raw_message_done_event("Now let me generate the SQL query"),
            _make_message_event("Now let me generate the SQL query"),  # duplicate, should be skipped
            _make_tool_output_event(call_id="call_R1"),
        ]

        actions = await _collect_actions(events)

        roles_and_statuses = [(a.role, a.status) for a in actions]
        assert roles_and_statuses == [
            (ActionRole.TOOL, ActionStatus.PROCESSING),  # yielded immediately
            (ActionRole.ASSISTANT, ActionStatus.SUCCESS),  # from raw event (early)
            (ActionRole.TOOL, ActionStatus.SUCCESS),  # tool complete
        ]
        # Only ONE assistant action, no duplicate
        assistant_actions = [a for a in actions if a.role == ActionRole.ASSISTANT]
        assert len(assistant_actions) == 1
        assert "SQL query" in assistant_actions[0].output["raw_output"]
        # Has in-progress tool calls -> is_thinking=True
        assert assistant_actions[0].output["is_thinking"] is True

    @pytest.mark.asyncio
    async def test_raw_message_with_processing_yielded_immediately(self):
        """PROCESSING is yielded immediately, not flushed after ASSISTANT."""
        events = [
            _make_tool_call_event(call_id="call_R2", tool_name="subagent_tool"),
            _make_raw_message_done_event("Thinking about the query"),
            _make_tool_output_event(call_id="call_R2"),
        ]

        actions = await _collect_actions(events)

        # PROCESSING comes first (yielded immediately), then ASSISTANT, then SUCCESS
        assert actions[0].role == ActionRole.TOOL
        assert actions[0].status == ActionStatus.PROCESSING
        assert actions[0].action_type == "subagent_tool"
        assert actions[1].role == ActionRole.ASSISTANT
        assert actions[2].role == ActionRole.TOOL
        assert actions[2].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_raw_message_with_multiple_tool_calls(self):
        """Multiple tool calls are yielded immediately before raw message."""
        events = [
            _make_tool_call_event(call_id="call_X", tool_name="tool_x"),
            _make_tool_call_event(call_id="call_Y", tool_name="tool_y"),
            _make_raw_message_done_event("Processing both tools"),
            _make_message_event("Processing both tools"),  # duplicate
            _make_tool_output_event(call_id="call_X"),
            _make_tool_output_event(call_id="call_Y"),
        ]

        actions = await _collect_actions(events)

        roles_and_statuses = [(a.role, a.status) for a in actions]
        assert roles_and_statuses == [
            (ActionRole.TOOL, ActionStatus.PROCESSING),  # tool_x
            (ActionRole.TOOL, ActionStatus.PROCESSING),  # tool_y
            (ActionRole.ASSISTANT, ActionStatus.SUCCESS),  # from raw
            (ActionRole.TOOL, ActionStatus.SUCCESS),  # tool_x complete
            (ActionRole.TOOL, ActionStatus.SUCCESS),  # tool_y complete
        ]

    @pytest.mark.asyncio
    async def test_raw_non_message_event_is_ignored(self):
        """Raw events that are not message output_item.done should be ignored."""
        events = [
            _make_raw_other_event(),  # should be skipped
            _make_tool_call_event(call_id="call_Z"),
            _make_tool_output_event(call_id="call_Z"),
        ]

        actions = await _collect_actions(events)

        roles_and_statuses = [(a.role, a.status) for a in actions]
        assert roles_and_statuses == [
            (ActionRole.TOOL, ActionStatus.PROCESSING),
            (ActionRole.TOOL, ActionStatus.SUCCESS),
        ]

    @pytest.mark.asyncio
    async def test_raw_message_with_empty_text_is_skipped(self):
        """Raw message events with empty/whitespace text should not create ASSISTANT."""
        msg_item = FakeMessageItem(content=[FakeTextContent(text="   ")])
        data = FakeOutputItemDoneData(item=msg_item)
        events = [
            FakeEvent(type="raw_response_event", data=data),
            _make_tool_call_event(call_id="call_E"),
            _make_tool_output_event(call_id="call_E"),
        ]

        actions = await _collect_actions(events)

        # No ASSISTANT action should be created for whitespace-only text
        roles_and_statuses = [(a.role, a.status) for a in actions]
        assert roles_and_statuses == [
            (ActionRole.TOOL, ActionStatus.PROCESSING),
            (ActionRole.TOOL, ActionStatus.SUCCESS),
        ]

    @pytest.mark.asyncio
    async def test_raw_message_without_tool_call(self):
        """Raw message event without any tool calls should yield ASSISTANT only."""
        events = [
            _make_raw_message_done_event("Here is my analysis"),
            _make_message_event("Here is my analysis"),  # duplicate
        ]

        actions = await _collect_actions(events)

        assert len(actions) == 1
        assert actions[0].role == ActionRole.ASSISTANT
        assert "analysis" in actions[0].output["raw_output"]
        # No pending tool calls -> is_thinking=False
        assert actions[0].output["is_thinking"] is False

    @pytest.mark.asyncio
    async def test_is_thinking_flag_true_when_tool_calls_in_progress(self):
        """is_thinking should be True when ASSISTANT fires with in-progress tool calls."""
        events = [
            _make_tool_call_event(call_id="call_T1"),
            _make_tool_call_event(call_id="call_T2"),
            _make_raw_message_done_event("Let me run these tools"),
            _make_message_event("Let me run these tools"),  # duplicate
            _make_tool_output_event(call_id="call_T1"),
            _make_tool_output_event(call_id="call_T2"),
        ]

        actions = await _collect_actions(events)

        assistant_actions = [a for a in actions if a.role == ActionRole.ASSISTANT]
        assert len(assistant_actions) == 1
        assert assistant_actions[0].output["is_thinking"] is True

    @pytest.mark.asyncio
    async def test_is_thinking_flag_false_when_no_tool_calls(self):
        """is_thinking should be False when ASSISTANT fires without any in-progress tool calls."""
        events = [
            _make_raw_message_done_event("Here is the final answer"),
        ]

        actions = await _collect_actions(events)

        assert len(actions) == 1
        assert actions[0].role == ActionRole.ASSISTANT
        assert actions[0].output["is_thinking"] is False

    @pytest.mark.asyncio
    async def test_is_thinking_flag_via_fallback_message_output_item(self):
        """is_thinking via RunItemStreamEvent fallback path (message_output_item)."""
        events = [
            _make_tool_call_event(call_id="call_F1"),
            _make_message_event("Thinking via fallback"),
            _make_tool_output_event(call_id="call_F1"),
        ]

        actions = await _collect_actions(events)

        assistant_actions = [a for a in actions if a.role == ActionRole.ASSISTANT]
        assert len(assistant_actions) == 1
        assert assistant_actions[0].output["is_thinking"] is True

    @pytest.mark.asyncio
    async def test_is_thinking_flag_false_via_fallback_no_tool(self):
        """is_thinking=False via fallback when no tool calls are pending."""
        events = [
            _make_message_event("Final output via fallback"),
        ]

        actions = await _collect_actions(events)

        assert len(actions) == 1
        assert actions[0].role == ActionRole.ASSISTANT
        assert actions[0].output["is_thinking"] is False

    @pytest.mark.asyncio
    async def test_raw_message_with_multiple_content_parts(self):
        """Raw message with multiple text content parts joins them."""
        msg_item = FakeMessageItem(content=[FakeTextContent(text="Part one."), FakeTextContent(text="Part two.")])
        data = FakeOutputItemDoneData(item=msg_item)
        events = [
            FakeEvent(type="raw_response_event", data=data),
        ]

        actions = await _collect_actions(events)

        assert len(actions) == 1
        assert actions[0].role == ActionRole.ASSISTANT
        assert "Part one.\nPart two." in actions[0].output["raw_output"]
