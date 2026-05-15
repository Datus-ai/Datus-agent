# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for the Ctrl+T sidebar toggle / auto-show visibility plumbing.

These tests verify three pieces of behaviour on :class:`DatusApp`:

* ``toggle_sidebar_hidden`` flips ``_sidebar_force_hidden`` and the next
  ``_sidebar_visible`` evaluation respects it.
* Transitions detected inside ``_sidebar_visible`` schedule the listener
  registered via ``set_sidebar_visibility_listener`` on the event loop —
  never inline, because the filter is called on the render path.
* The first ``_sidebar_visible`` evaluation just pins the baseline value
  and does **not** fire the listener (avoids a spurious reflow at boot).

The :class:`DatusApp` is constructed without running the prompt_toolkit
Application; we drive the filter directly via ``_todo_sidebar.filter()``.
"""

from __future__ import annotations

from typing import List
from unittest import mock

import pytest

from datus.cli.tui.app import DatusApp


@pytest.fixture
def has_items_box() -> dict:
    """Mutable box that backs ``todo_has_items_fn`` so tests can flip it."""
    return {"value": False}


@pytest.fixture
def tui_app(has_items_box: dict) -> DatusApp:
    """Build a DatusApp whose sidebar has-items is driven by ``has_items_box``."""
    app = DatusApp(
        status_tokens_fn=lambda: [],
        dispatch_fn=lambda _text: None,
        todo_tokens_fn=lambda: [],
        todo_has_items_fn=lambda: has_items_box["value"],
        todo_line_count_fn=lambda: 1 if has_items_box["value"] else 0,
    )
    # Stub out the terminal size probe so the min-cols guard never trips.
    app._terminal_columns = lambda: 200  # type: ignore[assignment]
    return app


def _filter_value(app: DatusApp) -> bool:
    """Invoke the ``ConditionalContainer.filter`` to evaluate ``_sidebar_visible``."""
    return bool(app._todo_sidebar.filter())


class TestToggleSidebarHidden:
    def test_toggle_flips_force_hidden(self, tui_app: DatusApp) -> None:
        assert tui_app._sidebar_force_hidden is False
        assert tui_app.toggle_sidebar_hidden() is True
        assert tui_app._sidebar_force_hidden is True
        assert tui_app.toggle_sidebar_hidden() is False
        assert tui_app._sidebar_force_hidden is False

    def test_force_hidden_hides_sidebar_even_with_items(self, tui_app: DatusApp, has_items_box: dict) -> None:
        has_items_box["value"] = True
        # Pin baseline so the first call doesn't fire the listener.
        _filter_value(tui_app)
        assert _filter_value(tui_app) is True

        tui_app.toggle_sidebar_hidden()
        assert _filter_value(tui_app) is False


class TestVisibilityListenerScheduling:
    def test_first_evaluation_pins_baseline_without_firing(self, tui_app: DatusApp, has_items_box: dict) -> None:
        loop = mock.MagicMock()
        tui_app._loop = loop
        listener = mock.MagicMock()
        tui_app.set_sidebar_visibility_listener(listener)

        has_items_box["value"] = True
        result = _filter_value(tui_app)

        assert result is True
        loop.call_soon_threadsafe.assert_not_called()
        listener.assert_not_called()
        assert tui_app._last_sidebar_visible is True

    def test_transition_false_to_true_schedules_listener(self, tui_app: DatusApp, has_items_box: dict) -> None:
        loop = mock.MagicMock()
        tui_app._loop = loop
        listener = mock.MagicMock()
        tui_app.set_sidebar_visibility_listener(listener)

        # Pin baseline at False.
        _filter_value(tui_app)
        assert tui_app._last_sidebar_visible is False

        has_items_box["value"] = True
        _filter_value(tui_app)

        loop.call_soon_threadsafe.assert_called_once_with(listener, True)
        # Listener is scheduled on the loop, not invoked synchronously.
        listener.assert_not_called()

    def test_transition_true_to_false_schedules_listener(self, tui_app: DatusApp, has_items_box: dict) -> None:
        loop = mock.MagicMock()
        tui_app._loop = loop
        listener = mock.MagicMock()
        tui_app.set_sidebar_visibility_listener(listener)

        has_items_box["value"] = True
        _filter_value(tui_app)
        assert tui_app._last_sidebar_visible is True

        has_items_box["value"] = False
        _filter_value(tui_app)

        loop.call_soon_threadsafe.assert_called_once_with(listener, False)

    def test_ctrl_t_force_hide_notifies_listener_with_false(self, tui_app: DatusApp, has_items_box: dict) -> None:
        loop = mock.MagicMock()
        tui_app._loop = loop
        listener = mock.MagicMock()
        tui_app.set_sidebar_visibility_listener(listener)

        has_items_box["value"] = True
        _filter_value(tui_app)
        loop.call_soon_threadsafe.reset_mock()

        tui_app.toggle_sidebar_hidden()
        _filter_value(tui_app)

        loop.call_soon_threadsafe.assert_called_once_with(listener, False)

    def test_no_loop_attached_does_not_crash(self, tui_app: DatusApp, has_items_box: dict) -> None:
        listener = mock.MagicMock()
        tui_app.set_sidebar_visibility_listener(listener)
        # No ``_loop`` and no ``_app`` — early-boot scenario.
        tui_app._loop = None

        # Pin baseline then transition: should silently no-op.
        _filter_value(tui_app)
        has_items_box["value"] = True
        _filter_value(tui_app)

        listener.assert_not_called()

    def test_repeated_same_value_does_not_reschedule(self, tui_app: DatusApp, has_items_box: dict) -> None:
        loop = mock.MagicMock()
        tui_app._loop = loop
        listener = mock.MagicMock()
        tui_app.set_sidebar_visibility_listener(listener)

        has_items_box["value"] = True
        _filter_value(tui_app)  # baseline
        _filter_value(tui_app)
        _filter_value(tui_app)

        loop.call_soon_threadsafe.assert_not_called()


class TestSidebarVisibleMinCols:
    def test_narrow_terminal_hides_sidebar(self, tui_app: DatusApp, has_items_box: dict) -> None:
        has_items_box["value"] = True
        tui_app._terminal_columns = lambda: 40  # type: ignore[assignment]
        # First call pins, second call confirms — both should report hidden.
        _filter_value(tui_app)
        assert _filter_value(tui_app) is False

    def test_has_items_exception_treated_as_hidden(self) -> None:
        def _boom() -> bool:
            raise RuntimeError("boom")

        calls: List[bool] = []

        def _has_items() -> bool:
            calls.append(True)
            return _boom()

        app = DatusApp(
            status_tokens_fn=lambda: [],
            dispatch_fn=lambda _text: None,
            todo_tokens_fn=lambda: [],
            todo_has_items_fn=_has_items,
        )
        app._terminal_columns = lambda: 200  # type: ignore[assignment]
        assert _filter_value(app) is False
        assert calls, "has_items_fn should have been invoked"
