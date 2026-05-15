# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Targeted tests for ``ChatCommands._full_screen_reprint``.

Specifically the path the sidebar reflow listener depends on:

* ``in_progress_actions`` provided **alongside** completed history must
  cause the running turn to be appended after the multi-turn replay, so
  the user's currently-streaming output survives the buffer wipe.
* Same parameter on the no-history path is still rendered (no completed
  turns means ``fallback_actions`` was historically the only option; the
  new parameter adds in-progress on top).
* When ``in_progress_actions`` is empty/None the no-op path is preserved.
"""

from __future__ import annotations

from typing import List
from unittest import mock

import pytest

from datus.cli.chat_commands import ChatCommands


def _fake_action() -> mock.MagicMock:
    """A stand-in ``ActionHistory`` — the reprint helper only forwards refs."""
    return mock.MagicMock(name="ActionHistory")


@pytest.fixture
def chat_commands() -> ChatCommands:
    cc = object.__new__(ChatCommands)
    cli = mock.MagicMock()
    cli._tui_output_buffer = mock.MagicMock()
    cli._print_welcome = mock.MagicMock()
    cc.cli = cli
    cc.console = mock.MagicMock()
    cc.all_turn_actions = []
    cc._trace_verbose = False
    return cc


@pytest.fixture
def patched_display():
    """Capture every ``ActionHistoryDisplay`` call so we can assert call order."""
    with mock.patch("datus.cli.chat_commands.ActionHistoryDisplay") as factory:
        instance = mock.MagicMock()
        factory.return_value = instance
        yield instance


class TestFullScreenReprintInProgress:
    def test_in_progress_appended_after_completed_turns(self, chat_commands: ChatCommands, patched_display) -> None:
        chat_commands.all_turn_actions = [("msg1", [_fake_action()])]
        in_progress: List = [_fake_action(), _fake_action()]

        chat_commands._full_screen_reprint(verbose=False, in_progress_actions=in_progress)

        # Multi-turn history rendered first, then in-progress appended.
        patched_display.render_multi_turn_history.assert_called_once()
        patched_display.render_action_history.assert_called_once_with(in_progress, verbose=False)
        # Buffer cleared before reprint begins.
        chat_commands.cli._tui_output_buffer.clear.assert_called_once()
        chat_commands.cli._print_welcome.assert_called_once()

    def test_in_progress_only_when_no_history(self, chat_commands: ChatCommands, patched_display) -> None:
        in_progress: List = [_fake_action()]

        chat_commands._full_screen_reprint(verbose=True, in_progress_actions=in_progress)

        patched_display.render_multi_turn_history.assert_not_called()
        patched_display.render_action_history.assert_called_once_with(in_progress, verbose=True)

    def test_empty_in_progress_is_noop(self, chat_commands: ChatCommands, patched_display) -> None:
        chat_commands._full_screen_reprint(verbose=False, in_progress_actions=[])

        # No history, no fallback, empty in-progress → nothing rendered.
        patched_display.render_multi_turn_history.assert_not_called()
        patched_display.render_action_history.assert_not_called()

    def test_none_in_progress_preserves_legacy_path(self, chat_commands: ChatCommands, patched_display) -> None:
        fallback = [_fake_action()]
        chat_commands._full_screen_reprint(verbose=False, fallback_actions=fallback, in_progress_actions=None)

        # Legacy ``elif fallback_actions`` branch should fire exactly once.
        patched_display.render_action_history.assert_called_once_with(fallback, verbose=False)
