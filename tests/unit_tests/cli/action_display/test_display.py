# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/cli/action_display/display.py — public interface."""

import uuid
from datetime import datetime
from io import StringIO

import pytest
from rich.console import Console

from datus.cli.action_display.display import ActionHistoryDisplay, create_action_display
from datus.cli.action_display.renderers import ActionRenderer
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus


def _make_action(
    role: ActionRole,
    status: ActionStatus,
    depth: int = 0,
    action_type: str = "test",
    messages: str = "",
    input_data: dict = None,
    output_data: dict = None,
    start_time: datetime = None,
    end_time: datetime = None,
) -> ActionHistory:
    return ActionHistory(
        action_id=str(uuid.uuid4()),
        role=role,
        messages=messages,
        action_type=action_type,
        input=input_data,
        output=output_data,
        status=status,
        start_time=start_time or datetime.now(),
        end_time=end_time,
        depth=depth,
    )


@pytest.mark.ci
class TestActionHistoryDisplayInit:
    """Test ActionHistoryDisplay initialization."""

    def test_has_renderer(self):
        display = ActionHistoryDisplay()
        assert isinstance(display.renderer, ActionRenderer)

    def test_has_content_generator(self):
        display = ActionHistoryDisplay()
        assert display.content_generator is not None


@pytest.mark.ci
class TestCreateActionDisplay:
    """Test factory function."""

    def test_factory_creates_display(self):
        display = create_action_display()
        assert isinstance(display, ActionHistoryDisplay)

    def test_factory_with_truncation(self):
        display = create_action_display(enable_truncation=False)
        assert display.enable_truncation is False


@pytest.mark.ci
class TestRenderMultiTurnHistory:
    """Test render_multi_turn_history."""

    def test_renders_turns_with_headers(self):
        buf = StringIO()
        console = Console(file=buf, no_color=True)
        display = ActionHistoryDisplay(console)

        turns = [
            (
                "First question",
                [
                    _make_action(
                        ActionRole.TOOL, ActionStatus.SUCCESS, messages="result1", input_data={"function_name": "f1"}
                    ),
                ],
            ),
            (
                "Second question",
                [
                    _make_action(
                        ActionRole.TOOL, ActionStatus.SUCCESS, messages="result2", input_data={"function_name": "f2"}
                    ),
                ],
            ),
        ]
        display.render_multi_turn_history(turns)
        output = buf.getvalue()
        assert "First question" in output
        assert "Second question" in output
        assert "\u2500" in output  # separator


@pytest.mark.ci
class TestDisplayStreamingActions:
    """Test display_streaming_actions returns context."""

    def test_returns_context(self):
        display = ActionHistoryDisplay()
        from datus.cli.action_display.streaming import InlineStreamingContext

        ctx = display.display_streaming_actions([])
        assert isinstance(ctx, InlineStreamingContext)


@pytest.mark.ci
class TestStopRestartLive:
    """Test stop_live and restart_live."""

    def test_stop_live_no_context(self):
        display = ActionHistoryDisplay()
        display.stop_live()  # Should not raise

    def test_restart_live_no_context(self):
        display = ActionHistoryDisplay()
        display.restart_live()  # Should not raise
