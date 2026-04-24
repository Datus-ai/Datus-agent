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
class FakeOutputTextDeltaData:
    """Fake data for response.output_text.delta raw event."""

    type: str = "response.output_text.delta"
    delta: str = ""


@dataclass
class FakeContentPartDoneData:
    """Fake data for response.content_part.done raw event."""

    type: str = "response.content_part.done"


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


def _make_raw_text_delta_event(delta="chunk"):
    """Create a raw_response_event for response.output_text.delta."""
    data = FakeOutputTextDeltaData(delta=delta)
    return FakeEvent(type="raw_response_event", data=data)


def _make_raw_content_part_done_event():
    """Create a raw_response_event for response.content_part.done."""
    data = FakeContentPartDoneData()
    return FakeEvent(type="raw_response_event", data=data)


def _make_raw_other_event():
    """Create a raw_response_event that is NOT handled (e.g. response.created)."""
    return FakeEvent(type="raw_response_event", data=MagicMock(type="response.created"))


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
    model.base_url = None

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
                model.litellm_adapter.reasoning_effort_level = None

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
      response.output_text.delta (chunk1) -> delta (chunk2) -> ... ->
      response.content_part.done -> response.output_item.done (message) ->
      tool_call_item -> [tool execution] -> tool_output_item
    Deltas are yielded as thinking_delta, content_part.done emits the final action.
    """

    @pytest.mark.asyncio
    async def test_text_deltas_yield_thinking_delta_actions(self):
        """Text deltas yield thinking_delta actions with accumulated text."""
        events = [
            _make_raw_text_delta_event("Hello "),
            _make_raw_text_delta_event("world"),
            _make_raw_content_part_done_event(),
        ]

        actions = await _collect_actions(events)

        delta_actions = [a for a in actions if a.action_type == "thinking_delta"]
        assert len(delta_actions) == 2
        assert delta_actions[0].output["delta"] == "Hello "
        assert delta_actions[0].output["accumulated"] == "Hello "
        assert delta_actions[1].output["delta"] == "world"
        assert delta_actions[1].output["accumulated"] == "Hello world"
        # All deltas share the same stream ID
        assert delta_actions[0].action_id == delta_actions[1].action_id
        assert delta_actions[0].status == ActionStatus.PROCESSING

        # Final ASSISTANT action from content_part.done
        assistant_actions = [a for a in actions if a.role == ActionRole.ASSISTANT and a.action_type == "response"]
        assert len(assistant_actions) == 1
        assert assistant_actions[0].output["raw_output"] == "Hello world"
        assert assistant_actions[0].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_deltas_before_tool_call_yields_thinking_then_processing(self):
        """Real SDK order: deltas -> content_part.done -> tool_call -> tool_output.
        Expected: thinking_delta(s) -> ASSISTANT -> PROCESSING -> SUCCESS.
        """
        events = [
            _make_raw_text_delta_event("Now let me "),
            _make_raw_text_delta_event("generate the SQL query"),
            _make_raw_content_part_done_event(),
            _make_tool_call_event(call_id="call_R1"),
            _make_message_event("Now let me generate the SQL query"),  # duplicate, should be skipped
            _make_tool_output_event(call_id="call_R1"),
        ]

        actions = await _collect_actions(events)

        # Filter out thinking_delta actions for role/status check
        non_delta = [a for a in actions if a.action_type != "thinking_delta"]
        roles_and_statuses = [(a.role, a.status) for a in non_delta]
        assert roles_and_statuses == [
            (ActionRole.ASSISTANT, ActionStatus.SUCCESS),  # from content_part.done
            (ActionRole.TOOL, ActionStatus.PROCESSING),  # tool call
            (ActionRole.TOOL, ActionStatus.SUCCESS),  # tool complete
        ]
        # Only ONE assistant action, no duplicate
        assistant_actions = [a for a in non_delta if a.role == ActionRole.ASSISTANT]
        assert len(assistant_actions) == 1
        assert "SQL query" in assistant_actions[0].output["raw_output"]

    @pytest.mark.asyncio
    async def test_tool_call_before_deltas_yields_processing_first(self):
        """SDK order: tool_call -> deltas -> content_part.done -> tool_output.
        Expected: PROCESSING -> thinking_delta(s) -> ASSISTANT -> SUCCESS.
        """
        events = [
            _make_tool_call_event(call_id="call_R2", tool_name="subagent_tool"),
            _make_raw_text_delta_event("Thinking about the query"),
            _make_raw_content_part_done_event(),
            _make_message_event("Thinking about the query"),  # duplicate
            _make_tool_output_event(call_id="call_R2"),
        ]

        actions = await _collect_actions(events)

        non_delta = [a for a in actions if a.action_type != "thinking_delta"]
        assert non_delta[0].role == ActionRole.TOOL
        assert non_delta[0].status == ActionStatus.PROCESSING
        assert non_delta[0].action_type == "subagent_tool"
        assert non_delta[1].role == ActionRole.ASSISTANT
        assert non_delta[2].role == ActionRole.TOOL
        assert non_delta[2].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_deltas_with_multiple_tool_calls(self):
        """Multiple tool calls yielded immediately, deltas come after."""
        events = [
            _make_tool_call_event(call_id="call_X", tool_name="tool_x"),
            _make_tool_call_event(call_id="call_Y", tool_name="tool_y"),
            _make_raw_text_delta_event("Processing both tools"),
            _make_raw_content_part_done_event(),
            _make_message_event("Processing both tools"),  # duplicate
            _make_tool_output_event(call_id="call_X"),
            _make_tool_output_event(call_id="call_Y"),
        ]

        actions = await _collect_actions(events)

        non_delta = [a for a in actions if a.action_type != "thinking_delta"]
        roles_and_statuses = [(a.role, a.status) for a in non_delta]
        assert roles_and_statuses == [
            (ActionRole.TOOL, ActionStatus.PROCESSING),  # tool_x
            (ActionRole.TOOL, ActionStatus.PROCESSING),  # tool_y
            (ActionRole.ASSISTANT, ActionStatus.SUCCESS),  # from content_part.done
            (ActionRole.TOOL, ActionStatus.SUCCESS),  # tool_x complete
            (ActionRole.TOOL, ActionStatus.SUCCESS),  # tool_y complete
        ]

    @pytest.mark.asyncio
    async def test_raw_non_message_event_is_ignored(self):
        """Raw events that are not handled types should be ignored."""
        events = [
            _make_raw_other_event(),  # should be skipped
            _make_tool_call_event(call_id="call_Z"),
            _make_tool_output_event(call_id="call_Z"),
        ]

        actions = await _collect_actions(events)

        non_delta = [a for a in actions if a.action_type != "thinking_delta"]
        roles_and_statuses = [(a.role, a.status) for a in non_delta]
        assert roles_and_statuses == [
            (ActionRole.TOOL, ActionStatus.PROCESSING),
            (ActionRole.TOOL, ActionStatus.SUCCESS),
        ]

    @pytest.mark.asyncio
    async def test_whitespace_only_deltas_skipped_at_content_part_done(self):
        """Whitespace-only accumulated text should not create ASSISTANT action."""
        events = [
            _make_raw_text_delta_event("   "),
            _make_raw_content_part_done_event(),
            _make_tool_call_event(call_id="call_E"),
            _make_tool_output_event(call_id="call_E"),
        ]

        actions = await _collect_actions(events)

        non_delta = [a for a in actions if a.action_type != "thinking_delta"]
        roles_and_statuses = [(a.role, a.status) for a in non_delta]
        assert roles_and_statuses == [
            (ActionRole.TOOL, ActionStatus.PROCESSING),
            (ActionRole.TOOL, ActionStatus.SUCCESS),
        ]

    @pytest.mark.asyncio
    async def test_deltas_without_tool_call(self):
        """Text deltas without any tool calls should yield ASSISTANT with is_thinking=False."""
        events = [
            _make_raw_text_delta_event("Here is my "),
            _make_raw_text_delta_event("analysis"),
            _make_raw_content_part_done_event(),
        ]

        actions = await _collect_actions(events)

        assistant_actions = [a for a in actions if a.role == ActionRole.ASSISTANT and a.action_type == "response"]
        assert len(assistant_actions) == 1
        assert "analysis" in assistant_actions[0].output["raw_output"]
        assert assistant_actions[0].output["is_thinking"] is False

    @pytest.mark.asyncio
    async def test_is_thinking_flag_true_when_tool_calls_in_progress(self):
        """is_thinking should be True when ASSISTANT fires with in-progress tool calls."""
        events = [
            _make_tool_call_event(call_id="call_T1"),
            _make_tool_call_event(call_id="call_T2"),
            _make_raw_text_delta_event("Let me run these tools"),
            _make_raw_content_part_done_event(),
            _make_message_event("Let me run these tools"),  # duplicate
            _make_tool_output_event(call_id="call_T1"),
            _make_tool_output_event(call_id="call_T2"),
        ]

        actions = await _collect_actions(events)

        assistant_actions = [a for a in actions if a.role == ActionRole.ASSISTANT and a.action_type == "response"]
        assert len(assistant_actions) == 1
        assert assistant_actions[0].output["is_thinking"] is True

    @pytest.mark.asyncio
    async def test_is_thinking_flag_false_when_no_tool_calls(self):
        """is_thinking should be False when ASSISTANT fires without any in-progress tool calls."""
        events = [
            _make_raw_text_delta_event("Here is the final answer"),
            _make_raw_content_part_done_event(),
        ]

        actions = await _collect_actions(events)

        assistant_actions = [a for a in actions if a.role == ActionRole.ASSISTANT and a.action_type == "response"]
        assert len(assistant_actions) == 1
        assert assistant_actions[0].output["is_thinking"] is False

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
    async def test_fallback_message_done_when_no_deltas(self):
        """Fallback: response.output_item.done type=message works when no deltas were received."""
        events = [
            _make_raw_message_done_event("Fallback message"),
            _make_message_event("Fallback message"),  # duplicate
        ]

        actions = await _collect_actions(events)

        non_delta = [a for a in actions if a.action_type != "thinking_delta"]
        assert len(non_delta) == 1
        assert non_delta[0].role == ActionRole.ASSISTANT
        assert "Fallback" in non_delta[0].output["raw_output"]

    @pytest.mark.asyncio
    async def test_fallback_skipped_when_already_captured_via_deltas(self):
        """Fallback output_item.done is skipped when content was already captured via deltas."""
        events = [
            _make_raw_text_delta_event("Already captured"),
            _make_raw_content_part_done_event(),
            _make_raw_message_done_event("Already captured"),  # should be skipped
            _make_message_event("Already captured"),  # also skipped
        ]

        actions = await _collect_actions(events)

        assistant_actions = [a for a in actions if a.role == ActionRole.ASSISTANT and a.action_type == "response"]
        assert len(assistant_actions) == 1  # Only one from content_part.done

    @pytest.mark.asyncio
    async def test_thinking_delta_stream_id_shared(self):
        """All thinking_delta actions in one stream share the same action_id."""
        events = [
            _make_raw_text_delta_event("chunk1"),
            _make_raw_text_delta_event("chunk2"),
            _make_raw_text_delta_event("chunk3"),
            _make_raw_content_part_done_event(),
        ]

        actions = await _collect_actions(events)

        delta_actions = [a for a in actions if a.action_type == "thinking_delta"]
        assert len(delta_actions) == 3
        ids = {a.action_id for a in delta_actions}
        assert len(ids) == 1  # All share same stream ID
        assert delta_actions[0].action_id.startswith("thinking_stream_")
