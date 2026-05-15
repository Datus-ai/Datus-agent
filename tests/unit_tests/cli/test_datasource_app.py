# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :class:`datus.cli.datasource_app.DatasourceApp`.

Mirrors the pattern in :mod:`tests.unit_tests.cli.test_model_app` and
:mod:`tests.unit_tests.cli.test_skill_app` — never run the prompt_toolkit
``Application`` under a pty. Drive the dual-mode finish hook
(``_finish/_layout/_focus``), the embedded panel builder, and the
standalone ``run()`` path with a mocked ``Application``.
"""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.datasource_app import DatasourceApp, DatasourceSelection, _View
from datus.cli.tui.wizard_host import EmbeddedWizard

pytestmark = pytest.mark.ci


def _stub_agent_config(*, datasources=None, current="") -> MagicMock:
    """Build a minimal AgentConfig stub for DatasourceApp.

    ``datasources`` — mapping ``name → (db_type, is_default)``. The stub
    exposes ``datasource_configs`` (used as a dict of names) and
    ``services.datasources`` returning per-name objects with ``type`` /
    ``default`` attributes; that's what ``_load_datasources`` consumes.
    """
    datasources = datasources or {}
    cfg = MagicMock()
    cfg.datasource_configs = {name: {} for name in datasources}
    services = SimpleNamespace(
        datasources={
            name: SimpleNamespace(type=dtype, default=is_default) for name, (dtype, is_default) in datasources.items()
        }
    )
    cfg.services = services
    cfg.current_datasource = current
    return cfg


def _build(*, datasources=None, current="") -> DatasourceApp:
    cfg = _stub_agent_config(datasources=datasources, current=current)
    return DatasourceApp(cfg, Console(file=io.StringIO(), no_color=True))


def _make_future():
    loop = asyncio.new_event_loop()
    return loop, loop.create_future()


# ─────────────────────────────────────────────────────────────────────
# Construction & data loading
# ─────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_empty_datasources_lists_empty(self):
        app = _build()
        assert app._datasources == []
        assert app._view is _View.DATASOURCE_LIST
        # No on_done before run() / build_embedded_panel.
        assert app._on_done is None

    def test_datasources_are_loaded_from_services(self):
        app = _build(datasources={"alpha": ("sqlite", True), "beta": ("duckdb", False)})
        names = {row[0] for row in app._datasources}
        assert names == {"alpha", "beta"}

    def test_current_datasource_positions_cursor(self):
        app = _build(
            datasources={"alpha": ("sqlite", False), "beta": ("duckdb", False)},
            current="beta",
        )
        assert app._datasources[app._list_cursor][0] == "beta"

    def test_unknown_current_falls_back_to_cursor_zero(self):
        app = _build(datasources={"alpha": ("sqlite", False)}, current="ghost")
        assert app._list_cursor == 0


# ─────────────────────────────────────────────────────────────────────
# Dual-mode finish hook + embedded panel
# ─────────────────────────────────────────────────────────────────────


class TestEmbeddedPanel:
    def test_build_embedded_panel_returns_wizard(self):
        app = _build(datasources={"alpha": ("sqlite", True)})
        loop, fut = _make_future()
        try:
            panel = app.build_embedded_panel(fut)
            assert isinstance(panel, EmbeddedWizard)
            assert panel.done_future is fut
            assert app._on_done is not None
            assert panel.first_focus is app._list_window
        finally:
            loop.close()

    def test_embedded_finish_with_selection_resolves(self):
        app = _build()
        loop, fut = _make_future()
        try:
            app.build_embedded_panel(fut)
            sel = DatasourceSelection(kind="switch", name="alpha")
            app._finish(sel)
            assert fut.done() and fut.result() is sel
        finally:
            loop.close()

    def test_embedded_finish_with_none_cancels(self):
        app = _build()
        loop, fut = _make_future()
        try:
            app.build_embedded_panel(fut)
            app._finish(None)
            assert fut.done() and fut.result() is None
        finally:
            loop.close()


class TestFinishAndLayout:
    def test_finish_without_on_done_is_noop(self):
        app = _build()
        assert app._on_done is None
        app._finish(None)

    def test_finish_invokes_on_done(self):
        app = _build()
        captured: list = []
        app._on_done = lambda r: captured.append(r)
        sel = DatasourceSelection(kind="switch", name="alpha")
        app._finish(sel)
        assert captured == [sel]

    def test_layout_returns_app_layout(self):
        app = _build()
        fake_layout = MagicMock()
        app._app = MagicMock(layout=fake_layout)
        assert app._layout() is fake_layout

    def test_layout_returns_none_when_get_app_raises(self):
        app = _build()
        app._app = None
        with patch("prompt_toolkit.application.get_app", side_effect=RuntimeError("no app")):
            assert app._layout() is None

    def test_focus_no_target_noop(self):
        """``_focus(None)`` returns ``None`` via the early-out guard."""
        app = _build()
        app._app = None
        assert app._focus(None) is None

    def test_focus_dispatches_to_layout(self):
        app = _build()
        fake_layout = MagicMock()
        app._app = MagicMock(layout=fake_layout)
        target = object()
        app._focus(target)
        fake_layout.focus.assert_called_once_with(target)


# ─────────────────────────────────────────────────────────────────────
# Standalone run() — mock Application so the prompt_toolkit main loop
# never starts.
# ─────────────────────────────────────────────────────────────────────


class TestRunStandalone:
    def test_run_returns_selection(self):
        app = _build()
        sel = DatasourceSelection(kind="switch", name="alpha")
        fake_app = MagicMock()
        fake_app.run.return_value = sel
        with patch("datus.cli.datasource_app.Application", return_value=fake_app):
            assert app.run() is sel
        assert app._on_done is None
        assert app._app is None

    def test_run_keyboard_interrupt_returns_none(self):
        app = _build()
        fake_app = MagicMock()
        fake_app.run.side_effect = KeyboardInterrupt
        with patch("datus.cli.datasource_app.Application", return_value=fake_app):
            assert app.run() is None

    def test_run_unexpected_exception_returns_none(self):
        app = _build()
        fake_app = MagicMock()
        fake_app.run.side_effect = RuntimeError("boom")
        with patch("datus.cli.datasource_app.Application", return_value=fake_app):
            assert app.run() is None


class TestRunAsync:
    def test_run_async_returns_selection(self):
        """``run_async`` is the async variant used by the embedded host
        when the parent doesn't ``suspend_input``. The flow mirrors
        ``run`` but awaits ``Application.run_async``."""
        app = _build()
        sel = DatasourceSelection(kind="switch", name="alpha")

        async def _fake_async_run():
            return sel

        fake_app = MagicMock()
        fake_app.run_async = _fake_async_run
        with patch("datus.cli.datasource_app.Application", return_value=fake_app):
            result = asyncio.new_event_loop().run_until_complete(app.run_async())
        assert result is sel

    def test_run_async_keyboard_interrupt_returns_none(self):
        app = _build()

        async def _fake_async_run():
            raise KeyboardInterrupt

        fake_app = MagicMock()
        fake_app.run_async = _fake_async_run
        with patch("datus.cli.datasource_app.Application", return_value=fake_app):
            result = asyncio.new_event_loop().run_until_complete(app.run_async())
        assert result is None
