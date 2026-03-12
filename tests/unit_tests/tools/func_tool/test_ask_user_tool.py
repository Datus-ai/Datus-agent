# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for AskUserTool.

Tests cover:
- Tool instantiation and registration
- Free-text and multiple-choice ask_user calls
- Error handling for cancelled interactions
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from datus.cli.execution_state import InteractionBroker, InteractionCancelled
from datus.tools.func_tool.ask_user_tool import AskUserTool


@pytest.fixture
def broker():
    """Create a real InteractionBroker."""
    return InteractionBroker()


@pytest.fixture
def ask_user_tool(broker):
    """Create an AskUserTool with a real broker."""
    return AskUserTool(broker=broker)


# ---------------------------------------------------------------------------
# Instantiation Tests
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestAskUserToolInit:
    """Tests for AskUserTool instantiation."""

    def test_init(self, broker):
        """Test AskUserTool initializes with broker."""
        tool = AskUserTool(broker=broker)
        assert tool.broker is broker

    def test_available_tools(self, ask_user_tool):
        """Test available_tools returns one tool."""
        tools = ask_user_tool.available_tools()
        assert len(tools) == 1
        assert tools[0].name == "ask_user"

    def test_tool_has_description(self, ask_user_tool):
        """Test the registered tool has a description."""
        tools = ask_user_tool.available_tools()
        assert tools[0].description
        assert "question" in tools[0].description.lower() or "ask" in tools[0].description.lower()


# ---------------------------------------------------------------------------
# Async Execution Tests
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestAskUserToolExecution:
    """Tests for ask_user async execution."""

    @pytest.mark.asyncio
    async def test_ask_user_free_text(self, ask_user_tool, broker):
        """Test ask_user with free-text input."""

        async def simulate_user():
            # Wait for the request to be queued
            await asyncio.sleep(0.2)
            # Get the pending interaction and submit
            for action_id, _pending in broker._pending.items():
                await broker.submit(action_id, "my answer")
                break

        task = asyncio.create_task(simulate_user())
        result = await ask_user_tool.ask_user(question="What is your name?")
        await task

        assert result.success == 1
        assert result.result["response"] == "my answer"

    @pytest.mark.asyncio
    async def test_ask_user_multiple_choice(self, ask_user_tool, broker):
        """Test ask_user with multiple-choice input."""
        import json

        choices_json = json.dumps({"y": "Yes - proceed", "n": "No - cancel"})

        async def simulate_user():
            await asyncio.sleep(0.2)
            for action_id, _pending in broker._pending.items():
                await broker.submit(action_id, "y")
                break

        task = asyncio.create_task(simulate_user())
        result = await ask_user_tool.ask_user(
            question="Continue?",
            choices_json=choices_json,
        )
        await task

        assert result.success == 1
        assert result.result["response"] == "y"

    @pytest.mark.asyncio
    async def test_ask_user_cancelled(self):
        """Test ask_user handles InteractionCancelled."""
        broker = MagicMock(spec=InteractionBroker)
        broker.request = AsyncMock(side_effect=InteractionCancelled("cancelled"))

        tool = AskUserTool(broker=broker)
        result = await tool.ask_user(question="test?")

        assert result.success == 0
        assert "cancelled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_ask_user_generic_error(self):
        """Test ask_user handles generic exceptions."""
        broker = MagicMock(spec=InteractionBroker)
        broker.request = AsyncMock(side_effect=RuntimeError("connection lost"))

        tool = AskUserTool(broker=broker)
        result = await tool.ask_user(question="test?")

        assert result.success == 0
        assert "connection lost" in result.error
