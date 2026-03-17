# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/tools/llms_tools/mcp_stream_utils.py"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.tools.llms_tools.mcp_stream_utils import base_mcp_stream


def _make_action(action_type="message", role=ActionRole.ASSISTANT):
    return ActionHistory(
        action_id="test-id-1",
        role=role,
        action_type=action_type,
        status=ActionStatus.SUCCESS,
    )


async def _collect_stream(gen):
    results = []
    async for item in gen:
        results.append(item)
    return results


class TestBaseMcpStream:
    def _make_input_data(self, prompt_version=None):
        mock_input = MagicMock()
        mock_input.prompt_version = prompt_version
        return mock_input

    def test_creates_action_history_manager_when_none_provided(self):
        """When no action_history_manager is provided, one is created internally."""
        action = _make_action()
        mock_model = MagicMock()

        async def fake_stream(**kwargs):
            yield action

        mock_model.generate_with_tools_stream = fake_stream
        input_data = self._make_input_data()

        with patch("datus.tools.llms_tools.mcp_stream_utils.prompt_manager") as mock_pm:
            mock_pm.get_raw_template.return_value = "system instruction"

            results = asyncio.run(
                _collect_stream(
                    base_mcp_stream(
                        model=mock_model,
                        input_data=input_data,
                        tool_config={"max_turns": 5},
                        mcp_servers={},
                        prompt="test prompt",
                        instruction_template="test_template",
                    )
                )
            )

        assert results == [action]

    def test_uses_provided_action_history_manager(self):
        """When action_history_manager is provided, it is used (not a new one)."""
        action = _make_action()
        mock_model = MagicMock()

        async def fake_stream(**kwargs):
            yield action

        mock_model.generate_with_tools_stream = fake_stream
        input_data = self._make_input_data()
        existing_manager = ActionHistoryManager()

        with patch("datus.tools.llms_tools.mcp_stream_utils.prompt_manager") as mock_pm:
            mock_pm.get_raw_template.return_value = "instruction"

            results = asyncio.run(
                _collect_stream(
                    base_mcp_stream(
                        model=mock_model,
                        input_data=input_data,
                        tool_config={},
                        mcp_servers={},
                        prompt="prompt",
                        instruction_template="tmpl",
                        action_history_manager=existing_manager,
                    )
                )
            )

        assert results == [action]

    def test_yields_multiple_actions(self):
        """Multiple actions yielded by model are all forwarded."""
        actions = [_make_action("tool_call"), _make_action("message")]
        mock_model = MagicMock()

        async def fake_stream(**kwargs):
            for a in actions:
                yield a

        mock_model.generate_with_tools_stream = fake_stream
        input_data = self._make_input_data()

        with patch("datus.tools.llms_tools.mcp_stream_utils.prompt_manager") as mock_pm:
            mock_pm.get_raw_template.return_value = "instruction"

            results = asyncio.run(
                _collect_stream(
                    base_mcp_stream(
                        model=mock_model,
                        input_data=input_data,
                        tool_config={},
                        mcp_servers={"server1": MagicMock()},
                        prompt="p",
                        instruction_template="t",
                    )
                )
            )

        assert len(results) == 2
        assert results[0] == actions[0]
        assert results[1] == actions[1]

    def test_max_turns_from_tool_config(self):
        """max_turns from tool_config is forwarded to model."""
        captured_kwargs = {}
        mock_model = MagicMock()

        async def fake_stream(**kwargs):
            captured_kwargs.update(kwargs)
            return
            yield  # make it a generator

        mock_model.generate_with_tools_stream = fake_stream
        input_data = self._make_input_data()

        with patch("datus.tools.llms_tools.mcp_stream_utils.prompt_manager") as mock_pm:
            mock_pm.get_raw_template.return_value = "instruction"

            asyncio.run(
                _collect_stream(
                    base_mcp_stream(
                        model=mock_model,
                        input_data=input_data,
                        tool_config={"max_turns": 15},
                        mcp_servers={},
                        prompt="p",
                        instruction_template="t",
                    )
                )
            )

        assert captured_kwargs.get("max_turns") == 15

    def test_default_max_turns_is_10(self):
        """Default max_turns is 10 when not in tool_config."""
        captured_kwargs = {}
        mock_model = MagicMock()

        async def fake_stream(**kwargs):
            captured_kwargs.update(kwargs)
            return
            yield

        mock_model.generate_with_tools_stream = fake_stream
        input_data = self._make_input_data()

        with patch("datus.tools.llms_tools.mcp_stream_utils.prompt_manager") as mock_pm:
            mock_pm.get_raw_template.return_value = "instruction"

            asyncio.run(
                _collect_stream(
                    base_mcp_stream(
                        model=mock_model,
                        input_data=input_data,
                        tool_config={},
                        mcp_servers={},
                        prompt="p",
                        instruction_template="t",
                    )
                )
            )

        assert captured_kwargs.get("max_turns") == 10

    def test_permission_error_is_reraised(self):
        """403/forbidden errors should be re-raised for fallback handling."""
        mock_model = MagicMock()

        async def fake_stream(**kwargs):
            raise PermissionError("403 Forbidden")
            yield

        mock_model.generate_with_tools_stream = fake_stream
        input_data = self._make_input_data()

        with patch("datus.tools.llms_tools.mcp_stream_utils.prompt_manager") as mock_pm:
            mock_pm.get_raw_template.return_value = "instruction"

            with pytest.raises(PermissionError):
                asyncio.run(
                    _collect_stream(
                        base_mcp_stream(
                            model=mock_model,
                            input_data=input_data,
                            tool_config={},
                            mcp_servers={},
                            prompt="p",
                            instruction_template="t",
                        )
                    )
                )

    def test_rate_limit_error_is_reraised(self):
        """Rate limit / overloaded errors should be re-raised."""
        mock_model = MagicMock()

        async def fake_stream(**kwargs):
            raise RuntimeError("rate limit exceeded")
            yield

        mock_model.generate_with_tools_stream = fake_stream
        input_data = self._make_input_data()

        with patch("datus.tools.llms_tools.mcp_stream_utils.prompt_manager") as mock_pm:
            mock_pm.get_raw_template.return_value = "instruction"

            with pytest.raises(RuntimeError, match="rate limit"):
                asyncio.run(
                    _collect_stream(
                        base_mcp_stream(
                            model=mock_model,
                            input_data=input_data,
                            tool_config={},
                            mcp_servers={},
                            prompt="p",
                            instruction_template="t",
                        )
                    )
                )

    def test_generic_error_is_reraised(self):
        """Other errors should also be re-raised."""
        mock_model = MagicMock()

        async def fake_stream(**kwargs):
            raise ValueError("some unexpected error")
            yield

        mock_model.generate_with_tools_stream = fake_stream
        input_data = self._make_input_data()

        with patch("datus.tools.llms_tools.mcp_stream_utils.prompt_manager") as mock_pm:
            mock_pm.get_raw_template.return_value = "instruction"

            with pytest.raises(ValueError, match="some unexpected error"):
                asyncio.run(
                    _collect_stream(
                        base_mcp_stream(
                            model=mock_model,
                            input_data=input_data,
                            tool_config={},
                            mcp_servers={},
                            prompt="p",
                            instruction_template="t",
                        )
                    )
                )

    def test_empty_stream_yields_nothing(self):
        """When model yields nothing, stream produces no results."""
        mock_model = MagicMock()

        async def fake_stream(**kwargs):
            return
            yield

        mock_model.generate_with_tools_stream = fake_stream
        input_data = self._make_input_data()

        with patch("datus.tools.llms_tools.mcp_stream_utils.prompt_manager") as mock_pm:
            mock_pm.get_raw_template.return_value = "instruction"

            results = asyncio.run(
                _collect_stream(
                    base_mcp_stream(
                        model=mock_model,
                        input_data=input_data,
                        tool_config={},
                        mcp_servers={},
                        prompt="p",
                        instruction_template="t",
                    )
                )
            )

        assert results == []
