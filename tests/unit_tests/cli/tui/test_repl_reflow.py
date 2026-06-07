# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for ``DatusCLI._compute_pane_width`` and ``_reflow_for_sidebar``.

The reflow path is what gives the user "buffer is cleared and re-rendered
proportionally" when the sidebar appears, and "left pane spans the whole
screen" when it disappears. These tests:

* lock the terminal width via ``shutil.get_terminal_size`` so the formula
  is deterministic;
* assert the formula matches ``DatusApp._sidebar_target_width`` (mirroring
  is the entire reason the helper exists);
* drive ``_reflow_for_sidebar`` through both branches: width change rebuilds
  the Rich Console and calls ``chat_commands._full_screen_reprint`` exactly
  once with the current ``_trace_verbose`` value; same width is a no-op.
"""

from __future__ import annotations

from unittest import mock

from rich.console import Console

from datus.cli.repl import DatusCLI


def _bare_cli() -> DatusCLI:
    """Construct a DatusCLI shell with __init__ bypassed."""
    return object.__new__(DatusCLI)


class TestComputePaneWidth:
    def test_sidebar_visible_subtracts_sidebar_and_scrollbar(self) -> None:
        cli = _bare_cli()
        with mock.patch("shutil.get_terminal_size", return_value=mock.MagicMock(columns=200)):
            # cols=200, sidebar=max(14, 200//5)=40, scrollbar=1, pane=200-40-1=159
            assert cli._compute_pane_width(sidebar_visible=True) == 159

    def test_sidebar_visible_min_sidebar_width_floor(self) -> None:
        cli = _bare_cli()
        with mock.patch("shutil.get_terminal_size", return_value=mock.MagicMock(columns=60)):
            # cols=60, sidebar=max(14, 60//5)=14, scrollbar=1, pane=60-14-1=45
            assert cli._compute_pane_width(sidebar_visible=True) == 45

    def test_sidebar_hidden_subtracts_scrollbar(self) -> None:
        cli = _bare_cli()
        with mock.patch("shutil.get_terminal_size", return_value=mock.MagicMock(columns=200)):
            # scrollbar gutter is always rendered (visible_filter=lambda: True)
            assert cli._compute_pane_width(sidebar_visible=False) == 199

    def test_narrow_terminal_floored_at_twenty(self) -> None:
        cli = _bare_cli()
        with mock.patch("shutil.get_terminal_size", return_value=mock.MagicMock(columns=10)):
            assert cli._compute_pane_width(sidebar_visible=False) == 20
            # cols=10, sidebar=14, scrollbar=1, pane=10-14-1=-5 → floored
            assert cli._compute_pane_width(sidebar_visible=True) == 20


class TestReflowForSidebar:
    def _build_cli(self, width: int = 100) -> DatusCLI:
        cli = _bare_cli()
        buffer = mock.MagicMock(spec=["write", "flush", "isatty", "clear"])
        buffer.isatty.return_value = False
        cli._tui_output_buffer = buffer
        cli.console = Console(file=buffer, width=width, force_terminal=True)
        cli.tui_app = mock.MagicMock()
        cli.chat_commands = mock.MagicMock()
        cli.chat_commands._trace_verbose = False
        cli.chat_commands._current_incremental_actions = None
        return cli

    def test_width_change_mutates_console_in_place(self) -> None:
        cli = self._build_cli(width=159)  # sidebar-visible width @ cols=200
        original_console = cli.console
        with mock.patch("shutil.get_terminal_size", return_value=mock.MagicMock(columns=200)):
            cli._reflow_for_sidebar(sidebar_visible=False)

        # Same instance, new width — so chat_commands.console (which captured
        # this reference) and any ActionHistoryDisplay still see the new width.
        # cols=200, scrollbar=1, sidebar hidden → pane=199
        assert cli.console is original_console
        assert cli.console.width == 199
        cli.chat_commands._full_screen_reprint.assert_called_once_with(verbose=False, in_progress_actions=None)
        cli.tui_app.invalidate.assert_called_once()

    def test_same_width_is_noop(self) -> None:
        cli = self._build_cli(width=199)
        with mock.patch("shutil.get_terminal_size", return_value=mock.MagicMock(columns=200)):
            cli._reflow_for_sidebar(sidebar_visible=False)

        cli.chat_commands._full_screen_reprint.assert_not_called()
        cli.tui_app.invalidate.assert_not_called()
        assert cli.console.width == 199

    def test_reprint_uses_current_trace_verbose(self) -> None:
        cli = self._build_cli(width=100)
        cli.chat_commands._trace_verbose = True
        with mock.patch("shutil.get_terminal_size", return_value=mock.MagicMock(columns=200)):
            cli._reflow_for_sidebar(sidebar_visible=False)

        cli.chat_commands._full_screen_reprint.assert_called_once_with(verbose=True, in_progress_actions=None)

    def test_in_progress_actions_forwarded(self) -> None:
        cli = self._build_cli(width=100)
        in_progress = [mock.MagicMock()]
        cli.chat_commands._current_incremental_actions = in_progress
        with mock.patch("shutil.get_terminal_size", return_value=mock.MagicMock(columns=200)):
            cli._reflow_for_sidebar(sidebar_visible=False)

        cli.chat_commands._full_screen_reprint.assert_called_once_with(verbose=False, in_progress_actions=in_progress)

    def test_missing_chat_commands_clears_buffer_only(self) -> None:
        cli = self._build_cli(width=100)
        del cli.chat_commands  # early-boot scenario
        with mock.patch("shutil.get_terminal_size", return_value=mock.MagicMock(columns=200)):
            cli._reflow_for_sidebar(sidebar_visible=False)

        cli._tui_output_buffer.clear.assert_called_once()
        cli.tui_app.invalidate.assert_called_once()

    def test_missing_buffer_short_circuits(self) -> None:
        cli = self._build_cli(width=100)
        del cli._tui_output_buffer
        with mock.patch("shutil.get_terminal_size", return_value=mock.MagicMock(columns=200)):
            cli._reflow_for_sidebar(sidebar_visible=False)

        # console.width unchanged, no reprint, no invalidate.
        assert cli.console.width == 100
        cli.chat_commands._full_screen_reprint.assert_not_called()
        cli.tui_app.invalidate.assert_not_called()

    def test_reprint_exception_does_not_break_reflow(self) -> None:
        cli = self._build_cli(width=100)
        cli.chat_commands._full_screen_reprint.side_effect = RuntimeError("boom")
        with mock.patch("shutil.get_terminal_size", return_value=mock.MagicMock(columns=200)):
            cli._reflow_for_sidebar(sidebar_visible=False)

        # Width still updated, invalidate still called.
        # cols=200, scrollbar=1, sidebar hidden → pane=199
        assert cli.console.width == 199
        cli.tui_app.invalidate.assert_called_once()
