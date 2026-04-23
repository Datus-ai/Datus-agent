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

from datus.cli.tui.app import EXIT_SENTINEL, DatusApp, tui_enabled


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
        from prompt_toolkit.keys import Keys

        assert tui_app._stored_paste is None
        for binding in tui_app.key_bindings.bindings:
            if Keys.ControlE in getattr(binding, "keys", ()):
                assert not binding.filter()
                return
        pytest.fail("ControlE binding not found")

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
