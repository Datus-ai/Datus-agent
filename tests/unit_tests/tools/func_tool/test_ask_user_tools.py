# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for AskUserTool."""

import asyncio

import pytest
import pytest_asyncio

from datus.cli.execution_state import InteractionBroker
from datus.tools.func_tool.ask_user_tools import AskUserTool


@pytest_asyncio.fixture
async def broker():
    b = InteractionBroker()
    yield b
    b.close()


@pytest_asyncio.fixture
async def tool(broker):
    return AskUserTool(broker=broker)


class TestAskUserTool:
    """Tests for AskUserTool."""

    @pytest.mark.asyncio
    async def test_available_tools(self, tool):
        """available_tools returns one tool named ask_user."""
        tools = tool.available_tools()
        assert len(tools) == 1
        assert tools[0].name == "ask_user"

    @pytest.mark.asyncio
    async def test_empty_question_rejected(self, tool):
        """Empty question returns error."""
        result = await tool.ask_user(question="")
        assert result.success == 0
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_whitespace_question_rejected(self, tool):
        """Whitespace-only question returns error."""
        result = await tool.ask_user(question="   ")
        assert result.success == 0

    @pytest.mark.asyncio
    async def test_too_few_options_rejected(self, tool):
        """Less than 2 options returns error."""
        result = await tool.ask_user(question="Pick one?", options=["only"])
        assert result.success == 0
        assert "2-5" in result.error

    @pytest.mark.asyncio
    async def test_too_many_options_rejected(self, tool):
        """More than 5 options returns error."""
        result = await tool.ask_user(question="Pick one?", options=["a", "b", "c", "d", "e", "f"])
        assert result.success == 0
        assert "2-5" in result.error

    @pytest.mark.asyncio
    async def test_ask_with_options_returns_answer(self, broker, tool):
        """User picks an option and the answer text is returned."""

        async def simulate_user():
            # Wait for the interaction to appear, then submit choice "2"
            for _ in range(50):
                if broker.has_pending:
                    pending = list(broker._pending.values())[0]
                    await broker.submit(pending.action_id, "2")
                    return
                await asyncio.sleep(0.01)

        task = asyncio.create_task(simulate_user())
        result = await tool.ask_user(question="Which DB?", options=["MySQL", "PostgreSQL", "SQLite"])
        await task

        assert result.success == 1
        assert result.result == "PostgreSQL"

    @pytest.mark.asyncio
    async def test_ask_free_text_returns_answer(self, broker, tool):
        """Without options the user types free text."""

        async def simulate_user():
            for _ in range(50):
                if broker.has_pending:
                    pending = list(broker._pending.values())[0]
                    await broker.submit(pending.action_id, "my custom answer")
                    return
                await asyncio.sleep(0.01)

        task = asyncio.create_task(simulate_user())
        result = await tool.ask_user(question="What table name?")
        await task

        assert result.success == 1
        assert result.result == "my custom answer"

    @pytest.mark.asyncio
    async def test_ask_with_options_free_text(self, broker, tool):
        """User types custom text instead of picking a predefined option."""

        async def simulate_user():
            for _ in range(50):
                if broker.has_pending:
                    pending = list(broker._pending.values())[0]
                    await broker.submit(pending.action_id, "MongoDB")
                    return
                await asyncio.sleep(0.01)

        task = asyncio.create_task(simulate_user())
        result = await tool.ask_user(question="Which DB?", options=["MySQL", "PostgreSQL"])
        await task

        assert result.success == 1
        assert result.result == "MongoDB"

    @pytest.mark.asyncio
    async def test_cancelled_interaction(self, broker, tool):
        """Broker close while waiting returns cancellation error."""

        async def close_broker():
            await asyncio.sleep(0.05)
            broker.close()

        task = asyncio.create_task(close_broker())
        result = await tool.ask_user(question="Will be cancelled?", options=["Yes", "No"])
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

        result = await tool.ask_user(question="Test?", options=["A", "B"])
        assert result.success == 0
        assert "something broke" in result.error

    def test_set_tool_context(self):
        """set_tool_context stores context on the tool."""
        broker = InteractionBroker()
        tool = AskUserTool(broker=broker)
        assert tool._tool_context is None
        tool.set_tool_context({"run_id": "abc"})
        assert tool._tool_context == {"run_id": "abc"}
