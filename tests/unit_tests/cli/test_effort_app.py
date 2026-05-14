# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :class:`datus.cli.effort_app.EffortApp`'s embedded path.

These tests focus on the embedded-wizard contract (the new path used
when a ``DatusApp`` is hosting). The classic standalone Application
path is exercised end-to-end by the dispatcher tests in
``test_effort_commands.py`` via mocked ``run()`` returns.
"""

from __future__ import annotations

import asyncio

import pytest
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window
from rich.console import Console

from datus.cli.effort_app import EffortApp, EffortSelection
from datus.cli.tui.wizard_host import EmbeddedWizard


def _console() -> Console:
    return Console(no_color=True)


def _make_future() -> tuple[asyncio.AbstractEventLoop, asyncio.Future]:
    loop = asyncio.new_event_loop()
    return loop, loop.create_future()


def test_build_embedded_panel_returns_wizard_with_expected_shape():
    app = EffortApp(console=_console(), current_effort="low")
    loop, fut = _make_future()
    try:
        panel = app.build_embedded_panel(fut)
        assert isinstance(panel, EmbeddedWizard)
        assert isinstance(panel.container, HSplit)
        assert isinstance(panel.key_bindings, KeyBindings)
        assert isinstance(panel.first_focus, Window)
        assert panel.first_focus.content is not None
        assert panel.done_future is fut
    finally:
        loop.close()


def test_enter_in_effort_phase_advances_to_scope_phase_without_resolving():
    """Hitting Enter on the effort list moves to the scope phase but
    must NOT resolve ``done_future`` — only the final scope Enter
    delivers a result."""
    from prompt_toolkit.keys import Keys

    app = EffortApp(console=_console(), current_effort="medium")
    loop, fut = _make_future()
    try:
        panel = app.build_embedded_panel(fut)
        # Find the Enter handler.
        bindings = [b for b in panel.key_bindings.bindings if b.keys == (Keys.Enter,)]
        assert bindings, "Enter binding must exist"

        # Simulate Enter with a stub event whose .app has invalidate().
        event = type("E", (), {"app": type("A", (), {"invalidate": lambda self: None})()})()
        bindings[0].handler(event)

        assert not fut.done(), "First Enter should NOT resolve the future"
        # The internal phase moved to scope:
        assert app._phase.value == "scope"
        assert app._selected_code == "medium"
    finally:
        loop.close()


def test_escape_resolves_with_none():
    from prompt_toolkit.keys import Keys

    app = EffortApp(console=_console(), current_effort="medium")
    loop, fut = _make_future()
    try:
        panel = app.build_embedded_panel(fut)
        bindings = [b for b in panel.key_bindings.bindings if b.keys == (Keys.Escape,)]
        assert bindings
        bindings[0].handler(type("E", (), {"app": None})())
        assert fut.done() and fut.result() is None
    finally:
        loop.close()


def test_enter_in_scope_phase_resolves_with_selection():
    """Drive the full flow: first Enter advances the phase, second
    Enter resolves the future with an ``EffortSelection``."""
    from prompt_toolkit.keys import Keys

    app = EffortApp(console=_console(), current_effort="medium")
    loop, fut = _make_future()
    try:
        panel = app.build_embedded_panel(fut)
        enter_bindings = [b for b in panel.key_bindings.bindings if b.keys == (Keys.Enter,)]
        assert enter_bindings
        event = type("E", (), {"app": type("A", (), {"invalidate": lambda self: None})()})()

        enter_bindings[0].handler(event)  # selects effort, advances phase
        assert not fut.done()
        enter_bindings[0].handler(event)  # confirms scope, resolves future

        assert fut.done()
        result = fut.result()
        assert isinstance(result, EffortSelection)
        assert result.code == "medium"
        # Default scope index is 0 → "project"
        assert result.scope == "project"
    finally:
        loop.close()


def test_scope_only_mode_resolves_directly_on_first_enter():
    """Direct ``/effort medium`` skips the effort phase — the wizard
    starts in scope mode and a single Enter resolves immediately."""
    from prompt_toolkit.keys import Keys

    app = EffortApp(console=_console(), scope_only="high")
    loop, fut = _make_future()
    try:
        panel = app.build_embedded_panel(fut)
        enter_bindings = [b for b in panel.key_bindings.bindings if b.keys == (Keys.Enter,)]
        assert enter_bindings
        enter_bindings[0].handler(type("E", (), {"app": None})())

        assert fut.done()
        result = fut.result()
        assert isinstance(result, EffortSelection)
        assert result.code == "high"
        assert result.scope == "project"
    finally:
        loop.close()


def test_ctrl_c_resolves_with_none():
    from prompt_toolkit.keys import Keys

    app = EffortApp(console=_console())
    loop, fut = _make_future()
    try:
        panel = app.build_embedded_panel(fut)
        ctrl_c_bindings = [b for b in panel.key_bindings.bindings if Keys.ControlC in b.keys]
        assert ctrl_c_bindings
        ctrl_c_bindings[0].handler(type("E", (), {"app": None})())
        assert fut.done() and fut.result() is None
    finally:
        loop.close()


def test_panel_kb_is_attached_to_focusable_list_window():
    """Embedded key bindings must hang off the focusable list Window
    (not a parent container), so they activate only while the
    wizard owns focus — preventing pollution of the main TUI's
    bindings."""
    app = EffortApp(console=_console())
    loop, fut = _make_future()
    try:
        panel = app.build_embedded_panel(fut)
        # first_focus is the list Window; its FormattedTextControl's
        # ``key_bindings`` must be the same instance the wizard returned.
        control = panel.first_focus.content
        assert control.key_bindings is panel.key_bindings
    finally:
        loop.close()


def test_resolver_is_idempotent_across_repeated_cancel():
    """A wizard that re-resolves (e.g. user presses ESC twice quickly)
    must not crash the host."""
    from prompt_toolkit.keys import Keys

    app = EffortApp(console=_console())
    loop, fut = _make_future()
    try:
        panel = app.build_embedded_panel(fut)
        esc = [b for b in panel.key_bindings.bindings if b.keys == (Keys.Escape,)][0]
        esc.handler(type("E", (), {"app": None})())
        # Second ESC: must not raise InvalidStateError.
        esc.handler(type("E", (), {"app": None})())
        assert fut.result() is None
    finally:
        loop.close()


@pytest.mark.parametrize(
    "current,expected_idx",
    [("low", 2), ("high", 4), ("", 3), ("not-a-real-level", 3)],
)
def test_default_effort_index_resolves_or_falls_back_to_medium(current, expected_idx):
    """The cursor starts on the user's current effort if known, else on
    'medium' — without this the picker would open on 'off' (index 0)
    and surprise users who already have a preference."""
    app = EffortApp(console=_console(), current_effort=current)
    assert app._default_effort_index() == expected_idx
