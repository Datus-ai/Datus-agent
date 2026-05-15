# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :class:`datus.cli.interaction_app.InteractionApp`.

The prompt_toolkit Application is never started under a pty — instead
we exercise the dual-mode finish hook (``_finish/_layout``), the
embedded-panel builder, the ESC-default fallback (``_esc_result``),
and the answer-collection logic (``_confirm``).
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from datus.cli.interaction_app import InteractionApp, InteractionResult
from datus.cli.tui.wizard_host import EmbeddedWizard
from datus.schemas.interaction_event import InteractionEvent

pytestmark = pytest.mark.ci


def _evt(
    *,
    title: str = "Q",
    content: str = "Pick one",
    choices=None,
    default_choice: str = "",
    allow_free_text: bool = False,
    multi_select: bool = False,
    content_type: str = "markdown",
) -> InteractionEvent:
    return InteractionEvent(
        title=title,
        content=content,
        content_type=content_type,
        choices=choices or {},
        default_choice=default_choice,
        allow_free_text=allow_free_text,
        multi_select=multi_select,
    )


def _make_future():
    loop = asyncio.new_event_loop()
    return loop, loop.create_future()


# ─────────────────────────────────────────────────────────────────────
# Construction & state
# ─────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_basic_init(self):
        app = InteractionApp([_evt(choices={"y": "Yes", "n": "No"})])
        assert app._idx == 0
        assert app._answers == [None]
        assert app._on_done is None

    def test_multiple_events_seed_independent_state(self):
        app = InteractionApp([_evt(), _evt(multi_select=True)])
        assert len(app._answers) == 2
        assert len(app._cursors) == 2
        assert len(app._checked) == 2


# ─────────────────────────────────────────────────────────────────────
# ESC default result fallback — the value returned when the user
# cancels without answering. Required to keep the agent loop unblocked.
# ─────────────────────────────────────────────────────────────────────


class TestEscResult:
    def test_esc_default_uses_default_choice(self):
        app = InteractionApp([_evt(choices={"y": "Yes", "n": "No"}, default_choice="n")])
        result = app._esc_result()
        assert result.answers == [["n"]]

    def test_esc_default_no_default_returns_empty_string(self):
        app = InteractionApp([_evt(choices={"y": "Yes", "n": "No"})])
        result = app._esc_result()
        assert result.answers == [[""]]

    def test_esc_default_preserves_existing_answers(self):
        """An already-answered tab keeps its answer when other tabs
        haven't been confirmed yet — important so the partial state is
        not silently discarded."""
        app = InteractionApp([_evt(choices={"y": "Yes"}), _evt(choices={"a": "A"}, default_choice="a")])
        app._answers[0] = ["y"]
        result = app._esc_result()
        assert result.answers == [["y"], ["a"]]


# ─────────────────────────────────────────────────────────────────────
# Embedded panel + dual-mode finish hook
# ─────────────────────────────────────────────────────────────────────


class TestEmbeddedPanel:
    def test_build_embedded_panel_returns_wizard(self):
        app = InteractionApp([_evt(choices={"y": "Yes", "n": "No"})])
        loop, fut = _make_future()
        try:
            panel = app.build_embedded_panel(fut)
            assert isinstance(panel, EmbeddedWizard)
            assert panel.done_future is fut
            assert app._on_done is not None
            assert panel.first_focus is not None
        finally:
            loop.close()

    def test_finish_resolves_future_with_result(self):
        app = InteractionApp([_evt(choices={"y": "Yes"})])
        loop, fut = _make_future()
        try:
            app.build_embedded_panel(fut)
            result = InteractionResult(answers=[["y"]])
            app._finish(result)
            assert fut.done() and fut.result() is result
        finally:
            loop.close()

    def test_finish_without_on_done_is_noop(self):
        app = InteractionApp([_evt()])
        assert app._on_done is None
        app._finish(InteractionResult(answers=[[""]]))  # No raise.


# ─────────────────────────────────────────────────────────────────────
# Layout resolution — same pattern as the other *_app modules.
# ─────────────────────────────────────────────────────────────────────


class TestLayoutResolution:
    def test_layout_returns_app_layout(self):
        app = InteractionApp([_evt()])
        fake_layout = MagicMock()
        app._app = MagicMock(layout=fake_layout)
        assert app._layout() is fake_layout

    def test_layout_returns_none_when_get_app_raises(self):
        app = InteractionApp([_evt()])
        app._app = None
        with patch("datus.cli.interaction_app.get_app", side_effect=RuntimeError("no app")):
            assert app._layout() is None


# ─────────────────────────────────────────────────────────────────────
# _confirm path: dispatches the answer through ``_finish`` when all
# tabs are answered, otherwise switches to the next tab. Smoke-test
# the single-question happy path.
# ─────────────────────────────────────────────────────────────────────


class TestConfirm:
    def test_confirm_single_choice_question_finishes_with_answer(self):
        app = InteractionApp([_evt(choices={"y": "Yes", "n": "No"})])
        captured: list = []
        app._on_done = lambda r: captured.append(r)
        # Cursor defaults to 0 → "y"
        app._confirm()
        assert len(captured) == 1
        assert captured[0].answers == [["y"]]

    def test_confirm_multi_select_collects_checked_keys(self):
        ev = _evt(choices={"a": "A", "b": "B", "c": "C"}, multi_select=True)
        app = InteractionApp([ev])
        app._checked[0] = {"a", "c"}
        captured: list = []
        app._on_done = lambda r: captured.append(r)
        app._confirm()
        assert captured[0].answers == [["a", "c"]]

    def test_confirm_switches_to_next_tab_when_unanswered(self):
        """Single-event confirm always finishes; with two events the
        first ``_confirm`` records the answer and switches focus to
        the next unanswered tab without resolving the future."""
        app = InteractionApp([_evt(choices={"y": "Yes"}, default_choice="y"), _evt(choices={"a": "A"})])
        captured: list = []
        app._on_done = lambda r: captured.append(r)
        app._confirm()
        # Future not resolved yet, tab moved to the second event.
        assert captured == []
        assert app._idx == 1
        assert app._answers[0] == ["y"]
