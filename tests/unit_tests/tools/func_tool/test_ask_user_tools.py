# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for AskUserTool (batch questions)."""

import asyncio
import json

import pytest
import pytest_asyncio

from datus.cli.execution_state import InteractionBroker
from datus.schemas.action_history import ActionStatus
from datus.tools.func_tool.ask_user_tools import AskUserTool


@pytest_asyncio.fixture
async def broker():
    b = InteractionBroker()
    yield b
    b.close()


@pytest_asyncio.fixture
async def tool(broker):
    return AskUserTool(broker=broker)


class TestAskUserToolValidation:
    """Validation tests for AskUserTool."""

    @pytest.mark.asyncio
    async def test_available_tools(self, tool):
        """available_tools returns one tool named ask_user."""
        tools = tool.available_tools()
        assert len(tools) == 1
        assert tools[0].name == "ask_user"

    @pytest.mark.asyncio
    async def test_empty_list_rejected(self, tool):
        """Empty questions list returns error."""
        result = await tool.ask_user(questions=[])
        assert result.success == 0
        assert "non-empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_non_list_rejected(self, tool):
        """Non-list questions returns error."""
        result = await tool.ask_user(questions="not a list")
        assert result.success == 0
        assert "non-empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_json_string_coerced_to_list(self, broker, tool):
        """JSON string containing a list of questions is auto-parsed."""

        async def simulate_user():
            for _ in range(50):
                if broker.has_pending:
                    pending = list(broker._pending.values())[0]
                    await broker.submit(pending.action_id, json.dumps(["Yes"]))
                    return
                await asyncio.sleep(0.01)

        task = asyncio.create_task(simulate_user())
        # Pass questions as a JSON string (LLM double-serialization)
        questions_str = json.dumps([{"question": "Continue?", "options": ["Yes", "No"]}])
        result = await tool.ask_user(questions=questions_str)
        await task

        assert result.success == 1
        answers = json.loads(result.result)
        assert answers[0]["answer"] == "Yes"

    @pytest.mark.asyncio
    async def test_json_string_non_list_rejected(self, tool):
        """JSON string that parses to a non-list (e.g. dict) is rejected."""
        questions_str = json.dumps({"question": "Not a list"})
        result = await tool.ask_user(questions=questions_str)
        assert result.success == 0
        assert "non-empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_none_rejected(self, tool):
        """None questions returns error."""
        result = await tool.ask_user(questions=None)
        assert result.success == 0

    @pytest.mark.asyncio
    async def test_non_dict_question_rejected(self, tool):
        """Non-dict question item returns error."""
        result = await tool.ask_user(questions=["not a dict"])
        assert result.success == 0
        assert "must be a dict" in result.error

    @pytest.mark.asyncio
    async def test_empty_question_text_rejected(self, tool):
        """Empty question text returns error."""
        result = await tool.ask_user(questions=[{"question": ""}])
        assert result.success == 0
        assert "must not be empty" in result.error

    @pytest.mark.asyncio
    async def test_whitespace_question_rejected(self, tool):
        """Whitespace-only question text returns error."""
        result = await tool.ask_user(questions=[{"question": "   "}])
        assert result.success == 0
        assert "must not be empty" in result.error

    @pytest.mark.asyncio
    async def test_empty_options_treated_as_free_text(self, broker, tool):
        """Empty options list is treated as free-text (same as None)."""

        async def simulate_user():
            for _ in range(50):
                if broker.has_pending:
                    pending = list(broker._pending.values())[0]
                    await broker.submit(pending.action_id, json.dumps(["my answer"]))
                    return
                await asyncio.sleep(0.01)

        task = asyncio.create_task(simulate_user())
        result = await tool.ask_user(questions=[{"question": "Time range?", "options": []}])
        await task

        assert result.success == 1
        answers = json.loads(result.result)
        assert answers[0]["answer"] == "my answer"

    @pytest.mark.asyncio
    async def test_too_few_options_rejected(self, tool):
        """Less than 2 options returns error."""
        result = await tool.ask_user(questions=[{"question": "Pick one?", "options": ["only"]}])
        assert result.success == 0
        assert "2-10" in result.error

    @pytest.mark.asyncio
    async def test_too_many_options_rejected(self, tool):
        """More than 10 options returns error."""
        opts = [chr(ord("a") + i) for i in range(11)]  # 11 options
        result = await tool.ask_user(questions=[{"question": "Pick one?", "options": opts}])
        assert result.success == 0
        assert "2-10" in result.error

    @pytest.mark.asyncio
    async def test_too_many_questions_rejected(self, tool):
        """More than MAX_QUESTIONS returns error."""
        questions = [{"question": f"Q{i}?"} for i in range(AskUserTool.MAX_QUESTIONS + 1)]
        result = await tool.ask_user(questions=questions)
        assert result.success == 0
        assert "at most" in result.error

    @pytest.mark.asyncio
    async def test_empty_option_string_rejected(self, tool):
        """Options containing empty/whitespace-only strings return error."""
        result = await tool.ask_user(questions=[{"question": "Pick?", "options": ["valid", "   "]}])
        assert result.success == 0
        assert "non-empty strings" in result.error

    @pytest.mark.asyncio
    async def test_question_item_objects_accepted(self, tool):
        """QuestionItem Pydantic objects are accepted directly."""
        from datus.tools.func_tool.ask_user_tools import QuestionItem

        result = await tool.ask_user(questions=[QuestionItem(question="", options=["A", "B"])])
        assert result.success == 0
        assert "must not be empty" in result.error

    @pytest.mark.asyncio
    async def test_options_not_list_rejected(self, tool):
        """Non-list options returns error."""
        result = await tool.ask_user(questions=[{"question": "Pick?", "options": "not a list"}])
        assert result.success == 0
        assert "must be a list" in result.error


class TestAskUserToolSingleQuestion:
    """Tests for single-question batch (questions list of length 1)."""

    @pytest.mark.asyncio
    async def test_single_question_with_options(self, broker, tool):
        """Single question with options: user picks option, answer resolved."""

        async def simulate_user():
            for _ in range(50):
                if broker.has_pending:
                    pending = list(broker._pending.values())[0]
                    # Submit JSON array with one answer
                    await broker.submit(pending.action_id, json.dumps(["PostgreSQL"]))
                    return
                await asyncio.sleep(0.01)

        task = asyncio.create_task(simulate_user())
        result = await tool.ask_user(
            questions=[{"question": "Which DB?", "options": ["MySQL", "PostgreSQL", "SQLite"]}]
        )
        await task

        assert result.success == 1
        answers = json.loads(result.result)
        assert len(answers) == 1
        assert answers[0]["answer"] == "PostgreSQL"
        assert answers[0]["question"] == "Which DB?"

    @pytest.mark.asyncio
    async def test_single_question_free_text(self, broker, tool):
        """Single question without options: user types free text."""

        async def simulate_user():
            for _ in range(50):
                if broker.has_pending:
                    pending = list(broker._pending.values())[0]
                    await broker.submit(pending.action_id, json.dumps(["my_table"]))
                    return
                await asyncio.sleep(0.01)

        task = asyncio.create_task(simulate_user())
        result = await tool.ask_user(questions=[{"question": "What table name?"}])
        await task

        assert result.success == 1
        answers = json.loads(result.result)
        assert len(answers) == 1
        assert answers[0]["answer"] == "my_table"


class TestAskUserToolBatch:
    """Tests for multi-question batch."""

    @pytest.mark.asyncio
    async def test_batch_questions(self, broker, tool):
        """Multiple questions: all answers returned together."""

        async def simulate_user():
            for _ in range(50):
                if broker.has_pending:
                    pending = list(broker._pending.values())[0]
                    answers = json.dumps(["MySQL", "Last 30 days", "user_id > 1000"])
                    await broker.submit(pending.action_id, answers)
                    return
                await asyncio.sleep(0.01)

        task = asyncio.create_task(simulate_user())
        result = await tool.ask_user(
            questions=[
                {"question": "Which DB?", "options": ["MySQL", "PostgreSQL", "SQLite"]},
                {"question": "Time range?", "options": ["Last 7 days", "Last 30 days"]},
                {"question": "Filter conditions?"},
            ]
        )
        await task

        assert result.success == 1
        answers = json.loads(result.result)
        assert len(answers) == 3
        assert answers[0]["answer"] == "MySQL"
        assert answers[1]["answer"] == "Last 30 days"
        assert answers[2]["answer"] == "user_id > 1000"

    @pytest.mark.asyncio
    async def test_batch_broker_sends_request_batch_action_type(self, broker, tool):
        """Batch request uses 'request_batch' action_type in the broker."""

        async def simulate_user():
            for _ in range(50):
                if broker.has_pending:
                    pending = list(broker._pending.values())[0]
                    await broker.submit(pending.action_id, json.dumps(["A", "B"]))
                    return
                await asyncio.sleep(0.01)

        # Also consume the queue to check action_type
        actions = []

        async def consume_queue():
            async for action in broker.fetch():
                actions.append(action)

        consumer = asyncio.create_task(consume_queue())
        simulate = asyncio.create_task(simulate_user())

        result = await tool.ask_user(
            questions=[
                {"question": "Q1?", "options": ["A", "B"]},
                {"question": "Q2?", "options": ["C", "D"]},
            ]
        )
        await simulate
        broker.close()
        await consumer

        assert result.success == 1
        # First action should be PROCESSING with request_batch
        processing_actions = [a for a in actions if a.status == ActionStatus.PROCESSING]
        assert len(processing_actions) >= 1
        assert processing_actions[0].action_type == "request_batch"
        # Check contents are in input
        contents_in_input = processing_actions[0].input.get("contents", [])
        assert len(contents_in_input) == 2


class TestAskUserToolEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_cancelled_interaction(self, broker, tool):
        """Broker close while waiting returns cancellation error."""

        async def close_broker():
            await asyncio.sleep(0.05)
            broker.close()

        task = asyncio.create_task(close_broker())
        result = await tool.ask_user(questions=[{"question": "Will be cancelled?", "options": ["Yes", "No"]}])
        await task

        assert result.success == 0
        assert "cancel" in result.error.lower()

    @pytest.mark.asyncio
    async def test_unexpected_exception_returns_error(self):
        """When broker.request raises an unexpected exception, return error."""
        broker = InteractionBroker()
        tool = AskUserTool(broker=broker)
        broker.reset_queue()

        async def broken_request(*args, **kwargs):
            raise RuntimeError("something broke")

        tool._broker.request = broken_request

        result = await tool.ask_user(questions=[{"question": "Test?", "options": ["A", "B"]}])
        assert result.success == 0
        assert "something broke" in result.error

    def test_set_tool_context(self):
        """set_tool_context stores context on the tool."""
        broker = InteractionBroker()
        tool = AskUserTool(broker=broker)
        assert tool._tool_context is None
        tool.set_tool_context({"run_id": "abc"})
        assert tool._tool_context == {"run_id": "abc"}

    @pytest.mark.asyncio
    async def test_fallback_when_response_not_json(self, broker, tool):
        """When broker returns non-JSON string, fallback to using it as-is for single question."""

        async def simulate_user():
            for _ in range(50):
                if broker.has_pending:
                    pending = list(broker._pending.values())[0]
                    await broker.submit(pending.action_id, "plain text answer")
                    return
                await asyncio.sleep(0.01)

        task = asyncio.create_task(simulate_user())
        result = await tool.ask_user(questions=[{"question": "What?"}])
        await task

        assert result.success == 1
        answers = json.loads(result.result)
        assert answers[0]["answer"] == "plain text answer"

    @pytest.mark.asyncio
    async def test_non_list_json_single_question_coerced(self, broker, tool):
        """When broker returns valid JSON that is not a list (e.g. dict), coerce for single question."""

        async def simulate_user():
            for _ in range(50):
                if broker.has_pending:
                    pending = list(broker._pending.values())[0]
                    # Submit a JSON string (not an array)
                    await broker.submit(pending.action_id, json.dumps("direct answer"))
                    return
                await asyncio.sleep(0.01)

        task = asyncio.create_task(simulate_user())
        result = await tool.ask_user(questions=[{"question": "What?"}])
        await task

        assert result.success == 1
        answers = json.loads(result.result)
        assert answers[0]["answer"] == "direct answer"

    @pytest.mark.asyncio
    async def test_non_list_json_multi_question_rejected(self, broker, tool):
        """When broker returns valid JSON that is not a list for multi-question, return error."""

        async def simulate_user():
            for _ in range(50):
                if broker.has_pending:
                    pending = list(broker._pending.values())[0]
                    # Submit a JSON dict (not an array)
                    await broker.submit(pending.action_id, json.dumps({"key": "value"}))
                    return
                await asyncio.sleep(0.01)

        task = asyncio.create_task(simulate_user())
        result = await tool.ask_user(
            questions=[
                {"question": "Q1?", "options": ["A", "B"]},
                {"question": "Q2?", "options": ["C", "D"]},
            ]
        )
        await task

        assert result.success == 0
        assert "Malformed" in result.error

    @pytest.mark.asyncio
    async def test_answer_count_mismatch_rejected(self, broker, tool):
        """When answer count doesn't match question count, return error."""

        async def simulate_user():
            for _ in range(50):
                if broker.has_pending:
                    pending = list(broker._pending.values())[0]
                    # Submit only 1 answer for 2 questions
                    await broker.submit(pending.action_id, json.dumps(["only one"]))
                    return
                await asyncio.sleep(0.01)

        task = asyncio.create_task(simulate_user())
        result = await tool.ask_user(
            questions=[
                {"question": "Q1?", "options": ["A", "B"]},
                {"question": "Q2?", "options": ["C", "D"]},
            ]
        )
        await task

        assert result.success == 0
        assert "Malformed" in result.error

    @pytest.mark.asyncio
    async def test_none_collector_response_rejected(self, broker, tool):
        """When collector returns None (interaction failure), return error instead of wrapping as answer."""

        async def simulate_user():
            for _ in range(50):
                if broker.has_pending:
                    pending = list(broker._pending.values())[0]
                    await broker.submit(pending.action_id, None)
                    return
                await asyncio.sleep(0.01)

        task = asyncio.create_task(simulate_user())
        result = await tool.ask_user(questions=[{"question": "Will fail?"}])
        await task

        assert result.success == 0
        assert "No response" in result.error

    @pytest.mark.asyncio
    async def test_multi_question_non_json_string_rejected(self, broker, tool):
        """Multi-question batch with non-JSON string response returns error."""

        async def simulate_user():
            for _ in range(50):
                if broker.has_pending:
                    pending = list(broker._pending.values())[0]
                    await broker.submit(pending.action_id, "plain text not json")
                    return
                await asyncio.sleep(0.01)

        task = asyncio.create_task(simulate_user())
        result = await tool.ask_user(
            questions=[
                {"question": "Q1?", "options": ["A", "B"]},
                {"question": "Q2?", "options": ["C", "D"]},
            ]
        )
        await task

        assert result.success == 0
        assert "Malformed" in result.error

    @pytest.mark.asyncio
    async def test_choice_key_resolved_to_display_value(self, broker, tool):
        """When user submits a choice key (e.g. '2'), it resolves to the display value."""

        async def simulate_user():
            for _ in range(50):
                if broker.has_pending:
                    pending = list(broker._pending.values())[0]
                    # Submit choice keys instead of display values
                    await broker.submit(pending.action_id, json.dumps(["2"]))
                    return
                await asyncio.sleep(0.01)

        task = asyncio.create_task(simulate_user())
        result = await tool.ask_user(
            questions=[{"question": "Which DB?", "options": ["MySQL", "PostgreSQL", "SQLite"]}]
        )
        await task

        assert result.success == 1
        answers = json.loads(result.result)
        assert answers[0]["answer"] == "PostgreSQL"
