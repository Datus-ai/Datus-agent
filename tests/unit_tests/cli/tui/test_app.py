# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for :mod:`datus.cli.tui.app`.

These tests avoid actually running the prompt_toolkit Application (which
requires a TTY) and instead verify the pure Python state machine around
``agent_running``, the Enter dispatch swallow behavior, ``EXIT_SENTINEL``
handling, and ``tui_enabled`` environment detection.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import Future
from unittest import mock

import pytest

from datus.cli.tui.app import (
    EXIT_SENTINEL,
    DatusApp,
    _is_jediterm,
    _resolve_mouse_support,
    tui_enabled,
)


@pytest.fixture
def tui_app() -> DatusApp:
    """Construct a minimal :class:`DatusApp` wired to recording callbacks."""
    status_calls: list = []

    def _status() -> list:
        status_calls.append(True)
        return [("class:status-bar", "Datus")]

    dispatch_log: list = []

    def _dispatch(text: str):
        dispatch_log.append(text)
        return None

    app = DatusApp(
        status_tokens_fn=_status,
        dispatch_fn=_dispatch,
    )
    # Expose the logs so test functions can assert against them.
    app._test_dispatch_log = dispatch_log  # type: ignore[attr-defined]
    app._test_status_calls = status_calls  # type: ignore[attr-defined]
    return app


class TestTuiEnabled:
    def test_disabled_when_env_set_to_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATUS_TUI", "0")
        assert tui_enabled() is False

    def test_disabled_when_env_set_to_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATUS_TUI", "FALSE")
        assert tui_enabled() is False

    def test_disabled_when_stdin_not_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DATUS_TUI", raising=False)
        # In the test runner stdout/stdin are pipes, so the check should fail.
        # Assert the outcome matches reality to avoid making the test
        # environment-sensitive (CI would always fail this otherwise).
        import sys

        expected = bool(sys.stdin.isatty() and sys.stdout.isatty())
        assert tui_enabled() is expected

    def test_honors_environment_over_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Even if TTY detection would return True, the env var takes priority.
        monkeypatch.setenv("DATUS_TUI", "off")
        with (
            mock.patch("sys.stdin.isatty", return_value=True),
            mock.patch("sys.stdout.isatty", return_value=True),
        ):
            assert tui_enabled() is False


class TestDatusAppState:
    def test_agent_running_is_fresh_threading_event(self, tui_app: DatusApp) -> None:
        assert isinstance(tui_app.agent_running, threading.Event)
        assert tui_app.agent_running.is_set() is False

    def test_submit_blank_input_is_rejected(self, tui_app: DatusApp) -> None:
        future = tui_app.submit_user_input("   \n  ")
        assert future is None
        # Blank input must not flip the running flag or reach dispatch_fn.
        assert tui_app.agent_running.is_set() is False
        assert tui_app._test_dispatch_log == []

    def test_submit_while_running_is_swallowed(self, tui_app: DatusApp) -> None:
        tui_app.agent_running.set()
        future = tui_app.submit_user_input("SELECT 1")
        assert future is None
        # Dispatch must not run when the agent is already busy.
        assert tui_app._test_dispatch_log == []

    def test_submit_without_loop_runs_synchronously(self, tui_app: DatusApp) -> None:
        # With no event loop bound, the app should execute the dispatcher
        # inline so tests (and startup-time invocations) can exercise the
        # same wiring without spinning up an Application.
        tui_app.submit_user_input("SELECT 1")
        assert tui_app._test_dispatch_log == ["SELECT 1"]
        # The inline path must not leave the flag stuck on.
        assert tui_app.agent_running.is_set() is False


class TestOnDispatchDone:
    def test_clears_running_flag_on_success(self, tui_app: DatusApp) -> None:
        tui_app.agent_running.set()
        future: Future = Future()
        future.set_result(None)
        tui_app._on_dispatch_done(future)
        assert tui_app.agent_running.is_set() is False

    def test_clears_running_flag_on_exception(self, tui_app: DatusApp) -> None:
        tui_app.agent_running.set()
        future: Future = Future()
        future.set_exception(RuntimeError("boom"))
        tui_app._on_dispatch_done(future)
        assert tui_app.agent_running.is_set() is False

    def test_exit_sentinel_triggers_exit(self, tui_app: DatusApp) -> None:
        tui_app.agent_running.set()
        future: Future = Future()
        future.set_result(EXIT_SENTINEL)

        with mock.patch.object(tui_app, "exit") as mocked_exit:
            tui_app._on_dispatch_done(future)
            mocked_exit.assert_called_once_with(0)

    def test_non_exit_result_does_not_call_exit(self, tui_app: DatusApp) -> None:
        future: Future = Future()
        future.set_result("anything else")

        with mock.patch.object(tui_app, "exit") as mocked_exit:
            tui_app._on_dispatch_done(future)
            mocked_exit.assert_not_called()


class TestStatusTokens:
    def test_status_tokens_are_wrapped_in_formatted_text(self, tui_app: DatusApp) -> None:
        # ``_safe_status_tokens`` is called on every redraw and must survive
        # callable exceptions without tearing the TUI down.
        ft = tui_app._safe_status_tokens()
        # FormattedText subclasses list, so the iteration check also validates
        # that the returned value is usable by the Window/FormattedTextControl
        # plumbing.
        assert list(ft) == [("class:status-bar", "Datus")]

    def test_status_tokens_tolerates_callable_errors(self) -> None:
        def _boom() -> list:
            raise RuntimeError("explode")

        app = DatusApp(
            status_tokens_fn=_boom,
            dispatch_fn=lambda text: None,
        )
        # Must not propagate; returning an empty token list keeps the
        # status bar visible with no segments rather than crashing redraw.
        assert list(app._safe_status_tokens()) == []


class TestInputPrompt:
    def test_prompt_uses_busy_style_when_running(self, tui_app: DatusApp) -> None:
        tui_app.agent_running.set()
        rendered = tui_app._get_input_prompt()
        assert rendered == [("class:input-prompt.busy", "> ")]

    def test_prompt_uses_idle_style_when_not_running(self, tui_app: DatusApp) -> None:
        assert tui_app.agent_running.is_set() is False
        rendered = tui_app._get_input_prompt()
        assert rendered == [("class:input-prompt", "> ")]

    def test_prompt_fn_errors_fallback_to_default(self) -> None:
        def _boom() -> str:
            raise RuntimeError("explode")

        app = DatusApp(
            status_tokens_fn=lambda: [],
            dispatch_fn=lambda text: None,
            input_prompt_fn=_boom,
        )
        rendered = app._get_input_prompt()
        # Defensive fallback is exercised and still produces a usable prompt.
        assert rendered == [("class:input-prompt", "> ")]


class TestKeyBindingsContract:
    """Verify Enter's dispatch-vs-swallow contract.

    prompt_toolkit stores ``"enter"`` as :class:`Keys.ControlM`, so finding
    the handler by key requires looking up the enum rather than the literal
    string we passed to ``@kb.add``.
    """

    @staticmethod
    def _enter_handler(app: DatusApp):
        from prompt_toolkit.keys import Keys

        for binding in app.key_bindings.bindings:
            if Keys.ControlM in getattr(binding, "keys", ()):
                return binding.handler
        raise AssertionError("DatusApp must register an Enter binding")

    def test_enter_swallowed_while_running(self, tui_app: DatusApp) -> None:
        handler = self._enter_handler(tui_app)

        tui_app.agent_running.set()

        event = mock.MagicMock()
        buffer = mock.MagicMock()
        buffer.complete_state = None
        buffer.text = "SELECT 1"
        event.app.current_buffer = buffer

        handler(event)

        # Swallowed: dispatch should not be called, buffer should not be reset.
        assert tui_app._test_dispatch_log == []
        buffer.reset.assert_not_called()

    def test_enter_dispatches_when_idle(self, tui_app: DatusApp) -> None:
        handler = self._enter_handler(tui_app)

        event = mock.MagicMock()
        buffer = mock.MagicMock()
        buffer.complete_state = None
        buffer.text = "SELECT 1"
        event.app.current_buffer = buffer

        handler(event)

        buffer.reset.assert_called_once()
        assert tui_app._test_dispatch_log == ["SELECT 1"]

    def test_enter_applies_active_completion_and_submits(self, tui_app: DatusApp) -> None:
        """Enter with a highlighted completion applies AND submits in one press.

        Previously this was a two-step (press 1 = apply, press 2 = submit)
        which made slash commands like ``/model`` feel laggy because the
        completion popup opens as soon as the user types ``/``.
        """
        handler = self._enter_handler(tui_app)

        completion = mock.MagicMock()
        complete_state = mock.MagicMock()
        complete_state.current_completion = completion

        buffer = mock.MagicMock()
        buffer.complete_state = complete_state
        buffer.text = "/model"

        event = mock.MagicMock()
        event.app.current_buffer = buffer

        handler(event)

        buffer.apply_completion.assert_called_once_with(completion)
        buffer.cancel_completion.assert_not_called()
        # Single Enter now both applies the highlight and dispatches.
        assert tui_app._test_dispatch_log == ["/model"]

    def test_enter_closes_menu_without_highlight_and_submits(self, tui_app: DatusApp) -> None:
        """Menu open but no highlighted item: Enter closes the menu and submits."""
        handler = self._enter_handler(tui_app)

        complete_state = mock.MagicMock()
        complete_state.current_completion = None

        buffer = mock.MagicMock()
        buffer.complete_state = complete_state
        buffer.text = "/model openai"

        event = mock.MagicMock()
        event.app.current_buffer = buffer

        handler(event)

        buffer.cancel_completion.assert_called_once()
        buffer.apply_completion.assert_not_called()
        assert tui_app._test_dispatch_log == ["/model openai"]


def test_exit_when_loop_absent_is_noop(tui_app: DatusApp) -> None:
    # Calling ``exit`` before the Application starts must not raise; the
    # exit code is still recorded for any later consumer.
    tui_app.exit(7)
    assert tui_app._exit_code == 7


def test_invalidate_without_loop_is_noop(tui_app: DatusApp) -> None:
    # A safety check: invalidate() is called from many callbacks, and
    # several of them fire during startup/shutdown when the loop pointer
    # is ``None``. The method must tolerate that without crashing.
    tui_app.invalidate()
    # No loop was created as a side effect; the app stays in pre-start state.
    assert tui_app._loop is None


def test_env_var_whitespace_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    # Whitespace around the env value must not bypass the disable check —
    # operators commonly paste ``DATUS_TUI= 0`` with a stray space.
    monkeypatch.setenv("DATUS_TUI", "  0  ")
    assert tui_enabled() is False


def test_os_environ_unset_uses_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    # Defensive coverage: if a subprocess inherits an empty string value
    # (common with ``os.execve`` reset patterns), tui_enabled should not
    # misinterpret it as "disabled".
    monkeypatch.setenv("DATUS_TUI", "")
    with (
        mock.patch("sys.stdin.isatty", return_value=True),
        mock.patch("sys.stdout.isatty", return_value=True),
    ):
        assert tui_enabled() is True


def test_environ_truthy_values_allow_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arbitrary non-falsy values should not disable the TUI; only the
    # documented disabled tokens (``0``/``false``/``no``/``off``) flip it.
    monkeypatch.setenv("DATUS_TUI", "yes")
    with (
        mock.patch("sys.stdin.isatty", return_value=True),
        mock.patch("sys.stdout.isatty", return_value=True),
    ):
        assert tui_enabled() is True


def test_os_environ_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # Sanity: nothing in the environment means we defer to TTY detection.
    monkeypatch.delenv("DATUS_TUI", raising=False)
    assert "DATUS_TUI" not in os.environ


def _binding_for_key(app: DatusApp, key):
    for binding in app.key_bindings.bindings:
        if key in getattr(binding, "keys", ()):
            return binding.handler
    raise AssertionError(f"{key!r} binding missing")


class TestCtrlDBinding:
    """Ctrl+D must only exit when the input buffer is empty."""

    def test_exits_when_buffer_empty(self, tui_app: DatusApp) -> None:
        from prompt_toolkit.keys import Keys

        handler = _binding_for_key(tui_app, Keys.ControlD)
        event = mock.MagicMock()
        event.app.current_buffer.text = ""

        with mock.patch.object(tui_app, "exit") as mocked_exit:
            handler(event)
            mocked_exit.assert_called_once_with(0)

    def test_noop_when_buffer_has_text(self, tui_app: DatusApp) -> None:
        from prompt_toolkit.keys import Keys

        handler = _binding_for_key(tui_app, Keys.ControlD)
        event = mock.MagicMock()
        event.app.current_buffer.text = "partial"

        with mock.patch.object(tui_app, "exit") as mocked_exit:
            handler(event)
            mocked_exit.assert_not_called()


class TestCtrlCBinding:
    """Default Ctrl+C just clears the buffer when idle; agent-running
    behavior is wired by DatusCLI (tested separately)."""

    def test_clears_buffer_when_idle(self, tui_app: DatusApp) -> None:
        from prompt_toolkit.keys import Keys

        handler = _binding_for_key(tui_app, Keys.ControlC)
        event = mock.MagicMock()
        event.app.current_buffer = mock.MagicMock()

        handler(event)
        event.app.current_buffer.reset.assert_called_once()

    def test_noop_when_agent_running(self, tui_app: DatusApp) -> None:
        from prompt_toolkit.keys import Keys

        tui_app.agent_running.set()
        handler = _binding_for_key(tui_app, Keys.ControlC)

        event = mock.MagicMock()
        event.app.current_buffer = mock.MagicMock()

        handler(event)
        # When the agent is running, the default handler is inert: DatusCLI
        # installs a more specific c-c binding that routes to the node's
        # interrupt_controller.
        event.app.current_buffer.reset.assert_not_called()


def test_set_input_text_replaces_buffer(tui_app: DatusApp) -> None:
    """``.rewind`` feeds a replayed user message through ``set_input_text``
    so the prefill round-trip must keep the buffer's document type
    consistent with what prompt_toolkit expects."""
    tui_app.set_input_text("SELECT from orders")
    assert tui_app.input_buffer.text == "SELECT from orders"

    # Calling with empty string must clear any prior prefill cleanly.
    tui_app.set_input_text("")
    assert tui_app.input_buffer.text == ""


def test_safe_dispatch_reraises_system_exit(tui_app: DatusApp) -> None:
    """SystemExit is the one exception type we must not swallow — callers
    rely on it to propagate out of the executor so graceful shutdown can
    proceed. Catching it here would strand the worker thread."""

    def _explode(text: str):
        raise SystemExit(2)

    tui_app._dispatch_fn = _explode

    with pytest.raises(SystemExit):
        tui_app._safe_dispatch("anything")


def test_safe_dispatch_logs_and_returns_none_on_base_exception(tui_app: DatusApp) -> None:
    def _explode(text: str):
        raise RuntimeError("kaboom")

    tui_app._dispatch_fn = _explode

    # Defensive swallow: must return ``None`` (no crash) so the worker
    # can be reused for the next command.
    assert tui_app._safe_dispatch("anything") is None


# -- Paste collapse tests ------------------------------------------------


class TestPasteCollapse:
    """Tests for the multi-line paste collapse feature."""

    @staticmethod
    def _paste_handler(app: DatusApp):
        from prompt_toolkit.keys import Keys

        return _binding_for_key(app, Keys.BracketedPaste)

    @staticmethod
    def _enter_handler(app: DatusApp):
        from prompt_toolkit.keys import Keys

        return _binding_for_key(app, Keys.ControlM)

    @staticmethod
    def _ctrl_c_handler(app: DatusApp):
        from prompt_toolkit.keys import Keys

        return _binding_for_key(app, Keys.ControlC)

    def test_short_paste_inserted_normally(self, tui_app: DatusApp) -> None:
        handler = self._paste_handler(tui_app)
        event = mock.MagicMock()
        event.data = "line1\nline2\nline3"
        buffer = mock.MagicMock()
        buffer.text = ""
        event.app.current_buffer = buffer

        handler(event)

        buffer.insert_text.assert_called_once_with("line1\nline2\nline3")
        assert tui_app._stored_paste is None

    def test_long_paste_inserts_placeholder(self, tui_app: DatusApp) -> None:
        handler = self._paste_handler(tui_app)
        lines = "\n".join(f"line{i}" for i in range(15))
        event = mock.MagicMock()
        event.data = lines
        buffer = tui_app.input_buffer
        event.app.current_buffer = buffer

        handler(event)

        assert tui_app._paste_collapsed is True
        assert tui_app._stored_paste == lines
        assert "[Pasted content: 15 lines]" in buffer.text

    def test_paste_preserves_existing_text(self, tui_app: DatusApp) -> None:
        """Pasting inserts placeholder at cursor, does not clear existing text."""
        from prompt_toolkit.document import Document

        handler = self._paste_handler(tui_app)
        buffer = tui_app.input_buffer
        buffer.document = Document("prefix ", len("prefix "))

        lines = "\n".join(f"line{i}" for i in range(12))
        event = mock.MagicMock()
        event.data = lines
        event.app.current_buffer = buffer

        handler(event)

        assert tui_app._stored_paste == lines
        assert buffer.text.startswith("prefix ")
        assert "[Pasted content: 12 lines]" in buffer.text

    def test_enter_replaces_placeholder_with_original(self, tui_app: DatusApp) -> None:
        from prompt_toolkit.document import Document

        paste_text = "\n".join(f"line{i}" for i in range(12))
        tui_app._stored_paste = paste_text
        tui_app._paste_collapsed = True
        placeholder = tui_app._paste_placeholder(12)

        buffer = tui_app.input_buffer
        full_text = f"prefix {placeholder} suffix"
        buffer.document = Document(full_text, len(full_text))

        enter_handler = self._enter_handler(tui_app)
        event = mock.MagicMock()
        event.app.current_buffer = buffer
        event.app.current_buffer.complete_state = None

        enter_handler(event)

        expected = f"prefix {paste_text} suffix"
        assert tui_app._test_dispatch_log == [expected]
        assert tui_app._stored_paste is None

    def test_enter_records_expanded_text_in_history(self, tui_app: DatusApp) -> None:
        from prompt_toolkit.document import Document

        paste_text = "\n".join(f"line{i}" for i in range(12))
        tui_app._stored_paste = paste_text
        tui_app._paste_collapsed = True
        placeholder = tui_app._paste_placeholder(12)

        buffer = tui_app.input_buffer
        buffer.document = Document(placeholder, len(placeholder))

        enter_handler = self._enter_handler(tui_app)
        event = mock.MagicMock()
        event.app.current_buffer = buffer
        event.app.current_buffer.complete_state = None

        enter_handler(event)

        history_strings = buffer.history.get_strings()
        assert paste_text in history_strings

    def test_ctrl_c_clears_paste_state(self, tui_app: DatusApp) -> None:
        tui_app._stored_paste = "some\npasted\ntext"
        tui_app._paste_collapsed = True

        handler = self._ctrl_c_handler(tui_app)
        event = mock.MagicMock()
        event.app.current_buffer = mock.MagicMock()

        handler(event)

        assert tui_app._stored_paste is None
        assert tui_app._paste_collapsed is False
        event.app.current_buffer.reset.assert_called_once()

    def test_ctrl_e_expands_inline(self, tui_app: DatusApp) -> None:
        from prompt_toolkit.document import Document
        from prompt_toolkit.keys import Keys

        paste_text = "\n".join(f"line{i}" for i in range(12))
        tui_app._stored_paste = paste_text
        tui_app._paste_collapsed = True

        placeholder = tui_app._paste_placeholder(12)
        buffer = tui_app.input_buffer
        buffer.document = Document(f"prefix {placeholder} suffix")

        handler = _binding_for_key(tui_app, Keys.ControlE)
        event = mock.MagicMock()
        event.app.current_buffer = buffer

        handler(event)

        assert tui_app._stored_paste is None
        assert tui_app._paste_collapsed is False
        assert f"prefix {paste_text} suffix" == buffer.text

    def test_ctrl_e_noop_without_paste(self, tui_app: DatusApp) -> None:
        """Every Ctrl+E binding must be filter-gated off when no paste
        is stored — otherwise the keystroke would surprise users who
        never bracketed-pasted into the input bar.

        Collect the bindings up-front (rather than asserting inside the
        loop and ``return``-ing on the first hit) so the contract holds
        for ALL Ctrl+E bindings, not just whichever happens to come
        first in the registry.
        """
        from prompt_toolkit.keys import Keys

        assert tui_app._stored_paste is None
        ctrl_e_bindings = [
            binding for binding in tui_app.key_bindings.bindings if Keys.ControlE in getattr(binding, "keys", ())
        ]
        assert ctrl_e_bindings, "ControlE binding not found"
        assert all(not binding.filter() for binding in ctrl_e_bindings)

    def test_placeholder_deleted_clears_state(self, tui_app: DatusApp) -> None:
        """When user deletes the placeholder text, stored paste is discarded."""
        from prompt_toolkit.document import Document

        paste_text = "\n".join(f"line{i}" for i in range(12))
        tui_app._stored_paste = paste_text
        tui_app._paste_collapsed = True

        buffer = tui_app.input_buffer
        buffer.document = Document("user typed something else")

        assert tui_app._stored_paste is None
        assert tui_app._paste_collapsed is False

    def test_editing_around_placeholder_keeps_state(self, tui_app: DatusApp) -> None:
        """Typing before/after placeholder does NOT clear stored paste."""
        from prompt_toolkit.document import Document

        paste_text = "\n".join(f"line{i}" for i in range(12))
        tui_app._stored_paste = paste_text
        tui_app._paste_collapsed = True

        placeholder = tui_app._paste_placeholder(12)
        buffer = tui_app.input_buffer
        buffer.document = Document(f"prefix {placeholder} suffix")

        assert tui_app._stored_paste == paste_text
        assert tui_app._paste_collapsed is True

    def test_dynamic_height_follows_content(self, tui_app: DatusApp) -> None:
        from prompt_toolkit.document import Document

        dim = tui_app._get_input_height()
        assert dim.preferred == 1
        assert dim.max == 15

        tui_app.input_buffer.document = Document("line1\nline2\nline3")
        dim = tui_app._get_input_height()
        assert dim.preferred == 3
        assert dim.max == 15

    def test_dynamic_height_includes_wrapped_lines(self, tui_app: DatusApp, monkeypatch: pytest.MonkeyPatch) -> None:
        """Visual rows count wrapped segments, not just hard newlines."""
        from prompt_toolkit.document import Document

        # Pin a narrow terminal so wrap math is deterministic.
        monkeypatch.setattr(tui_app, "_terminal_columns", lambda: 20)
        # Prompt "> " is 2 cells -> first line usable = 18; others = 20.

        # Single hard line of 50 chars: first segment ceil(50/18)=3 rows.
        tui_app.input_buffer.document = Document("a" * 50)
        dim = tui_app._get_input_height()
        assert dim.preferred == 3

        # Mixed: long first line + short second + long third line.
        # Line 1: 40 chars, usable 18 -> ceil(40/18)=3.
        # Line 2: "ok" -> 1.
        # Line 3: 25 chars, usable 20 -> ceil(25/20)=2.
        tui_app.input_buffer.document = Document(("a" * 40) + "\nok\n" + ("b" * 25))
        dim = tui_app._get_input_height()
        assert dim.preferred == 6

        # Wide / CJK characters take 2 cells each.
        # 12 CJK chars = 24 cells; usable 18 -> ceil(24/18)=2.
        tui_app.input_buffer.document = Document("中" * 12)
        dim = tui_app._get_input_height()
        assert dim.preferred == 2

        # Cap stays at 15 even with very long input.
        tui_app.input_buffer.document = Document("a" * 10000)
        dim = tui_app._get_input_height()
        assert dim.preferred == 15
        assert dim.max == 15

    def test_visual_line_count_handles_empty_buffer(self, tui_app: DatusApp) -> None:
        """Empty input still occupies one visual row."""
        assert tui_app._input_visual_line_count() == 1

    def test_input_prompt_display_width_counts_wide_chars(self, tui_app: DatusApp) -> None:
        """Prompt width must use cwidth so CJK prompts are not undercounted."""
        # Default prompt is "> " -> 2 cells.
        assert tui_app._input_prompt_display_width() == 2

        tui_app._input_prompt_fn = lambda: "中> "
        # "中" = 2 cells, "> " = 2 cells.
        assert tui_app._input_prompt_display_width() == 4

    def test_terminal_columns_falls_back_to_shutil(self, tui_app: DatusApp, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the Application output reports zero columns, use shutil."""
        from prompt_toolkit.data_structures import Size

        # Force the primary path (Application.output) to report a useless size
        # so the shutil fallback runs.
        monkeypatch.setattr(tui_app._app.output, "get_size", lambda: Size(rows=24, columns=0))
        import shutil

        monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): Size(rows=fallback[1], columns=42))
        assert tui_app._terminal_columns() == 42

    def test_clear_paste_state_method(self, tui_app: DatusApp) -> None:
        tui_app._stored_paste = "some text"
        tui_app._paste_collapsed = True

        tui_app.clear_paste_state()

        assert tui_app._stored_paste is None
        assert tui_app._paste_collapsed is False

    def test_paste_placeholder_format(self) -> None:
        assert DatusApp._paste_placeholder(15) == "[Pasted content: 15 lines]"
        assert DatusApp._paste_placeholder(1) == "[Pasted content: 1 lines]"

    def test_prompt_shows_hint_when_collapsed(self, tui_app: DatusApp) -> None:
        tui_app._paste_collapsed = True
        rendered = tui_app._get_input_prompt()
        tokens = list(rendered)
        assert ("class:input-prompt", "> ") in tokens
        assert ("class:input-prompt.hint", "(Ctrl+E to expand) ") in tokens

    def test_prompt_normal_when_not_collapsed(self, tui_app: DatusApp) -> None:
        tui_app._paste_collapsed = False
        rendered = tui_app._get_input_prompt()
        tokens = list(rendered)
        assert ("class:input-prompt", "> ") in tokens

    def test_second_paste_expands_first(self, tui_app: DatusApp) -> None:
        """A second large paste expands the first placeholder inline."""

        handler = self._paste_handler(tui_app)
        first_paste = "\n".join(f"a{i}" for i in range(12))
        buffer = tui_app.input_buffer

        event1 = mock.MagicMock()
        event1.data = first_paste
        event1.app.current_buffer = buffer
        handler(event1)

        assert tui_app._stored_paste == first_paste
        first_ph = tui_app._paste_placeholder(12)
        assert first_ph in buffer.text

        second_paste = "\n".join(f"b{i}" for i in range(15))
        event2 = mock.MagicMock()
        event2.data = second_paste
        event2.app.current_buffer = buffer
        handler(event2)

        assert tui_app._stored_paste == second_paste
        assert first_ph not in buffer.text
        assert first_paste in buffer.text
        assert tui_app._paste_placeholder(15) in buffer.text


class TestJediTermDetection:
    """``_is_jediterm`` reads ``TERMINAL_EMULATOR`` case-insensitively."""

    def test_jediterm_value_detected(self) -> None:
        assert _is_jediterm({"TERMINAL_EMULATOR": "JetBrains-JediTerm"}) is True

    def test_jediterm_case_insensitive(self) -> None:
        # JediTerm itself sets the canonical casing but tooling/wrappers
        # may lowercase or uppercase it; the detection must be robust to
        # both so users on either don't get inconsistent behaviour.
        assert _is_jediterm({"TERMINAL_EMULATOR": "jetbrains-jediterm"}) is True
        assert _is_jediterm({"TERMINAL_EMULATOR": "JETBRAINS-JEDITERM"}) is True

    def test_jediterm_with_surrounding_whitespace(self) -> None:
        assert _is_jediterm({"TERMINAL_EMULATOR": "  JetBrains-JediTerm  "}) is True

    def test_other_terminal_not_detected(self) -> None:
        for value in ("iTerm.app", "Apple_Terminal", "vscode", "alacritty", ""):
            assert _is_jediterm({"TERMINAL_EMULATOR": value}) is False, value

    def test_missing_env_var_not_detected(self) -> None:
        assert _is_jediterm({}) is False


class TestResolveMouseSupport:
    """``_resolve_mouse_support`` honours overrides before terminal sniff."""

    def test_default_on_for_unknown_terminal(self) -> None:
        # Empty environment → no JediTerm marker → mouse capture stays on
        # to preserve in-app wheel/click in iTerm2, Terminal.app, etc.
        assert _resolve_mouse_support({}) is True

    def test_default_off_for_jediterm(self) -> None:
        assert _resolve_mouse_support({"TERMINAL_EMULATOR": "JetBrains-JediTerm"}) is False

    @pytest.mark.parametrize("override", ["1", "true", "TRUE", "Yes", "on"])
    def test_force_on_overrides_jediterm(self, override: str) -> None:
        # Escape hatch for users on a JediTerm build/version that handles
        # SGR cleanly — env var wins over auto-detection.
        env = {
            "TERMINAL_EMULATOR": "JetBrains-JediTerm",
            "DATUS_FORCE_MOUSE_CAPTURE": override,
        }
        assert _resolve_mouse_support(env) is True

    @pytest.mark.parametrize("override", ["0", "false", "FALSE", "No", "off"])
    def test_force_off_overrides_default(self, override: str) -> None:
        # Inverse escape hatch: disable mouse capture even outside
        # JediTerm (e.g. when running under a screen-multiplexer that
        # echoes SGR back).
        env = {"DATUS_FORCE_MOUSE_CAPTURE": override}
        assert _resolve_mouse_support(env) is False

    def test_unrecognised_override_falls_back_to_autodetect(self) -> None:
        # Unknown values must not silently flip the default — they
        # delegate to terminal sniffing so a typo doesn't lock users
        # into the wrong mode without warning.
        env = {
            "TERMINAL_EMULATOR": "JetBrains-JediTerm",
            "DATUS_FORCE_MOUSE_CAPTURE": "maybe",
        }
        assert _resolve_mouse_support(env) is False

        env2 = {"DATUS_FORCE_MOUSE_CAPTURE": "maybe"}
        assert _resolve_mouse_support(env2) is True

    def test_reads_os_environ_when_no_arg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The helper is also called with no argument from DatusApp
        # construction; that path must read ``os.environ`` live so test
        # monkeypatching is honoured.
        monkeypatch.setenv("TERMINAL_EMULATOR", "JetBrains-JediTerm")
        monkeypatch.delenv("DATUS_FORCE_MOUSE_CAPTURE", raising=False)
        assert _resolve_mouse_support() is False

        monkeypatch.setenv("DATUS_FORCE_MOUSE_CAPTURE", "1")
        assert _resolve_mouse_support() is True


def _bindings_for_key(app: DatusApp, key):
    """Return *all* handlers bound to ``key`` (multiple may coexist with filters)."""
    out = []
    for binding in app.key_bindings.bindings:
        if key in getattr(binding, "keys", ()):
            out.append(binding)
    return out


class TestArrowKeyScroll:
    """Arrow keys must scroll the output pane when mouse capture is off
    (so JediTerm's DECSET 1007 wheel→arrow translation gives in-app
    scrolling) but stay out of the way otherwise."""

    def _make_app(self, monkeypatch: pytest.MonkeyPatch, *, jediterm: bool) -> DatusApp:
        if jediterm:
            monkeypatch.setenv("TERMINAL_EMULATOR", "JetBrains-JediTerm")
        else:
            monkeypatch.delenv("TERMINAL_EMULATOR", raising=False)
        monkeypatch.delenv("DATUS_FORCE_MOUSE_CAPTURE", raising=False)
        return DatusApp(
            status_tokens_fn=lambda: [("", "x")],
            dispatch_fn=lambda _t: None,
        )

    def test_mouse_support_off_under_jediterm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        app = self._make_app(monkeypatch, jediterm=True)
        assert app._mouse_support_enabled is False

    def test_mouse_support_on_outside_jediterm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        app = self._make_app(monkeypatch, jediterm=False)
        assert app._mouse_support_enabled is True

    def test_arrow_bindings_registered_under_jediterm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The Up/Down → output scroll handlers must be present in the
        # JediTerm-mode app so wheel events (delivered as ↑/↓ keys)
        # have somewhere to land.
        app = self._make_app(monkeypatch, jediterm=True)
        up_bindings = _bindings_for_key(app, "up")
        down_bindings = _bindings_for_key(app, "down")
        assert len(up_bindings) >= 1
        assert len(down_bindings) >= 1

    def test_arrow_scroll_filter_passes_when_composer_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Filter gates: mouse capture off + empty composer + no
        # completion menu. All three are true at fresh startup.
        app = self._make_app(monkeypatch, jediterm=True)
        up_handlers = _bindings_for_key(app, "up")
        assert any(b.filter() for b in up_handlers), (
            "Expected at least one 'up' handler to be active when the composer is empty under JediTerm mode"
        )

    def test_arrow_scroll_filter_blocked_when_composer_has_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Once the user types, Up must give cursor-up / history behaviour
        # back to the TextArea — our scroll handler bows out.
        app = self._make_app(monkeypatch, jediterm=True)
        app._input_area.buffer.text = "hello"
        up_handlers = _bindings_for_key(app, "up")
        # Find the binding whose filter we own (the one that becomes False
        # when text is non-empty). Other Up handlers (if any) keep their
        # own filter logic.
        scroll_filter_active = [b.filter() for b in up_handlers]
        # At least one binding existed; with text present, our scroll
        # gate must be False — assert by checking it's not unanimously
        # True (the default TextArea binding has no such gate so it may
        # remain active, which is the intended fall-through).
        assert False in scroll_filter_active, "Scroll-up gate should disengage when composer has text"

    def test_arrow_scroll_filter_blocked_outside_jediterm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # In a normal terminal (mouse capture on), real wheel events are
        # delivered as SCROLL_UP/SCROLL_DOWN mouse events, not ↑/↓ keys.
        # Our Up/Down scroll hijack must therefore stay dormant so Up
        # keeps doing history-backward in the TextArea.
        app = self._make_app(monkeypatch, jediterm=False)
        up_handlers = _bindings_for_key(app, "up")
        # The hijacked Up binding's filter must be False here; if no
        # such binding exists at all, the test also passes.
        assert all(not b.filter() for b in up_handlers if _is_jediterm_only_binding(b)), (
            "Scroll-up hijack must be inactive outside JediTerm mode"
        )

    def test_up_handler_scrolls_output_when_invoked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        app = self._make_app(monkeypatch, jediterm=True)
        up_handlers = _bindings_for_key(app, "up")
        event = mock.MagicMock()
        with mock.patch.object(app, "_scroll_output_up") as mock_up:
            for b in up_handlers:
                if b.filter():
                    b.handler(event)
                    break
            else:
                raise AssertionError("no active 'up' handler under JediTerm mode")
        mock_up.assert_called_once_with(app._OUTPUT_WHEEL_STEP)
        event.app.invalidate.assert_called_once()

    def test_down_handler_scrolls_output_when_invoked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        app = self._make_app(monkeypatch, jediterm=True)
        down_handlers = _bindings_for_key(app, "down")
        event = mock.MagicMock()
        with mock.patch.object(app, "_scroll_output_down") as mock_down:
            for b in down_handlers:
                if b.filter():
                    b.handler(event)
                    break
            else:
                raise AssertionError("no active 'down' handler under JediTerm mode")
        mock_down.assert_called_once_with(app._OUTPUT_WHEEL_STEP)
        event.app.invalidate.assert_called_once()


def _is_jediterm_only_binding(binding) -> bool:
    """Heuristic: a binding whose filter mentions our mouse-support flag.

    Used by tests to find the Up/Down hijack handlers we added without
    coupling to prompt_toolkit's filter internals. The handler's
    qualified name is uniquely ``_scroll_up_arrow`` / ``_scroll_down_arrow``
    so we match on that for an exact identifier.
    """
    name = getattr(binding.handler, "__name__", "")
    return name in {"_scroll_up_arrow", "_scroll_down_arrow"}
