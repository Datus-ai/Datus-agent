# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :mod:`datus.cli.bootstrap_app` (form-only assertions).

Drives ``_collect_for`` directly rather than running the prompt_toolkit
Application — running the TUI in pytest is unreliable across CI.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from datus.cli.bootstrap_app import (
    PANEL_NAMES,
    BootstrapApp,
    BootstrapPlan,
    TaskSpec,
    _Tab,
    _ValidationError,
)


@pytest.fixture()
def console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120, log_path=False)


@pytest.fixture()
def app(console: Console) -> BootstrapApp:
    return BootstrapApp(console, datasource_default="ssb_sqlite")


# ─────────────────────────────────────────────────────────────────────
# Sanity
# ─────────────────────────────────────────────────────────────────────


def test_panel_names_match_tab_enum() -> None:
    assert PANEL_NAMES == tuple(t.value for t in _Tab)


def test_dataclasses_construct() -> None:
    spec = TaskSpec(name="metadata", options={"datasource": "x"})
    plan = BootstrapPlan(task=spec)
    assert plan.task.name == "metadata"
    assert plan.task.options == {"datasource": "x"}


# ─────────────────────────────────────────────────────────────────────
# Per-tab _collect_for shapes — defaults: overwrite checkbox unchecked
# means build_mode == "incremental"
# ─────────────────────────────────────────────────────────────────────


def test_collect_schema_defaults_incremental(app: BootstrapApp) -> None:
    opts = app._collect_for(_Tab.SCHEMA)
    assert opts == {
        "datasource": "ssb_sqlite",
        "build_mode": "incremental",
    }


def test_collect_schema_overwrite_checked(app: BootstrapApp) -> None:
    app._schema_overwrite.checked = True
    opts = app._collect_for(_Tab.SCHEMA)
    assert opts["build_mode"] == "overwrite"


def test_collect_schema_missing_datasource_raises(app: BootstrapApp) -> None:
    app._schema_datasource.text = "  "
    with pytest.raises(_ValidationError):
        app._collect_for(_Tab.SCHEMA)


def test_collect_sql_full_form(app: BootstrapApp) -> None:
    app._sql_dir.text = "/data/sql"
    app._sql_pool.text = "5"
    app._sql_subject_tree.text = "Finance, Revenue "
    app._sql_overwrite.checked = True
    opts = app._collect_for(_Tab.SQL)
    assert opts == {
        "datasource": "ssb_sqlite",
        "sql_dir": "/data/sql",
        "pool_size": 5,
        "subject_tree": "Finance, Revenue",
        "build_mode": "overwrite",
    }


def test_collect_sql_missing_dir(app: BootstrapApp) -> None:
    with pytest.raises(_ValidationError):
        app._collect_for(_Tab.SQL)


def test_collect_sql_invalid_pool(app: BootstrapApp) -> None:
    app._sql_dir.text = "/data/sql"
    app._sql_pool.text = "0"
    with pytest.raises(_ValidationError):
        app._collect_for(_Tab.SQL)


def test_collect_template_shape(app: BootstrapApp) -> None:
    app._tpl_dir.text = "/data/templates"
    opts = app._collect_for(_Tab.TEMPLATE)
    assert opts == {
        "datasource": "ssb_sqlite",
        "template_dir": "/data/templates",
        "pool_size": 3,
        "subject_tree": "",
        "build_mode": "incremental",
    }


def test_collect_semantic_only_success_story(app: BootstrapApp) -> None:
    app._sem_success_story.text = "/data/success.csv"
    opts = app._collect_for(_Tab.SEMANTIC)
    # No semantic_yaml / from_adapter / catalog / subject_path / source.
    assert opts == {
        "datasource": "ssb_sqlite",
        "success_story": "/data/success.csv",
        "build_mode": "incremental",
    }


def test_collect_semantic_overwrite(app: BootstrapApp) -> None:
    app._sem_success_story.text = "/data/success.csv"
    app._sem_overwrite.checked = True
    assert app._collect_for(_Tab.SEMANTIC)["build_mode"] == "overwrite"


def test_collect_semantic_missing_success_story(app: BootstrapApp) -> None:
    with pytest.raises(_ValidationError):
        app._collect_for(_Tab.SEMANTIC)


def test_collect_metrics_full_form(app: BootstrapApp) -> None:
    app._met_success_story.text = "/data/success.csv"
    app._met_pool.text = "2"
    app._met_subject_tree.text = "Finance"
    opts = app._collect_for(_Tab.METRICS)
    assert opts == {
        "datasource": "ssb_sqlite",
        "success_story": "/data/success.csv",
        "pool_size": 2,
        "subject_tree": "Finance",
        "build_mode": "incremental",
    }


# ─────────────────────────────────────────────────────────────────────
# Removed-fields guard — every tab must produce ONLY the simplified key
# set; if anyone re-adds a stale field they'll trip these assertions.
# ─────────────────────────────────────────────────────────────────────


_EXPECTED_KEYS = {
    _Tab.SCHEMA: {"datasource", "build_mode"},
    _Tab.SQL: {"datasource", "sql_dir", "pool_size", "subject_tree", "build_mode"},
    _Tab.TEMPLATE: {"datasource", "template_dir", "pool_size", "subject_tree", "build_mode"},
    _Tab.SEMANTIC: {"datasource", "success_story", "build_mode"},
    _Tab.METRICS: {"datasource", "success_story", "pool_size", "subject_tree", "build_mode"},
}


@pytest.mark.parametrize("tab", list(_EXPECTED_KEYS.keys()))
def test_no_unexpected_keys_per_tab(app: BootstrapApp, tab: _Tab) -> None:
    """Fill every required field with a placeholder, then assert key set."""
    if tab == _Tab.SQL:
        app._sql_dir.text = "x"
    elif tab == _Tab.TEMPLATE:
        app._tpl_dir.text = "x"
    elif tab == _Tab.SEMANTIC:
        app._sem_success_story.text = "x"
    elif tab == _Tab.METRICS:
        app._met_success_story.text = "x"
    opts = app._collect_for(tab)
    assert set(opts.keys()) == _EXPECTED_KEYS[tab]


# ─────────────────────────────────────────────────────────────────────
# Dual-mode finish hook + embedded panel — mirrors test_effort_app.py.
# ─────────────────────────────────────────────────────────────────────


import asyncio  # noqa: E402

from datus.cli.tui.wizard_host import EmbeddedWizard  # noqa: E402


def _make_future() -> tuple[asyncio.AbstractEventLoop, asyncio.Future]:
    loop = asyncio.new_event_loop()
    return loop, loop.create_future()


class TestEmbeddedPanel:
    def test_build_embedded_panel_returns_wizard(self, app: BootstrapApp) -> None:
        loop, fut = _make_future()
        try:
            panel = app.build_embedded_panel(fut)
            assert isinstance(panel, EmbeddedWizard)
            assert panel.done_future is fut
            # Key bindings instance is attached to the panel container subtree.
            assert panel.key_bindings is not None
            # _on_done must be installed so _finish dispatches through the future.
            assert app._on_done is not None
        finally:
            loop.close()

    def test_embedded_finish_with_result_resolves_future(self, app: BootstrapApp) -> None:
        loop, fut = _make_future()
        try:
            app.build_embedded_panel(fut)
            plan = BootstrapPlan(task=TaskSpec(name="metadata", options={"datasource": "x"}))
            app._finish(plan)
            assert fut.done()
            assert fut.result() is plan
        finally:
            loop.close()

    def test_embedded_finish_with_none_cancels_future(self, app: BootstrapApp) -> None:
        loop, fut = _make_future()
        try:
            app.build_embedded_panel(fut)
            app._finish(None)
            assert fut.done()
            assert fut.result() is None
        finally:
            loop.close()


class TestFinishHook:
    def test_finish_without_on_done_is_noop(self, app: BootstrapApp) -> None:
        """``_on_done`` defaults to ``None`` outside run()/build_embedded_panel.
        Calling ``_finish`` in that state must silently no-op, not raise."""
        assert app._on_done is None
        app._finish(None)  # Must not raise.

    def test_finish_invokes_on_done(self, app: BootstrapApp) -> None:
        captured: list = []
        app._on_done = lambda result: captured.append(result)
        plan = BootstrapPlan(task=TaskSpec(name="metadata", options={}))
        app._finish(plan)
        assert captured == [plan]


class TestLayoutHelpers:
    def test_layout_without_app_returns_none_when_get_app_raises(self, app: BootstrapApp) -> None:
        """``_layout`` falls back to ``get_app()`` when ``_app`` is ``None``.
        When ``get_app()`` raises (e.g. no active Application), ``_layout``
        must swallow the error and return ``None``."""
        from unittest.mock import patch

        app._app = None
        with patch("prompt_toolkit.application.get_app", side_effect=RuntimeError("no app")):
            assert app._layout() is None

    def test_layout_with_app_returns_app_layout(self, app: BootstrapApp) -> None:
        """When the standalone Application is live, ``_layout`` returns
        its layout directly rather than touching ``get_app()``."""
        from unittest.mock import MagicMock

        fake_layout = MagicMock(name="Layout")
        app._app = MagicMock(layout=fake_layout)
        assert app._layout() is fake_layout

    def test_focus_no_target_is_noop(self, app: BootstrapApp) -> None:
        """``_focus(None)`` is the cancel-tab-cycle path used when the
        focus chain is empty (defensive). Returns ``None`` via the
        early-out guard even when ``_layout`` is unresolvable."""
        app._app = None
        assert app._focus(None) is None

    def test_focus_with_target_calls_layout_focus(self, app: BootstrapApp) -> None:
        from unittest.mock import MagicMock

        fake_layout = MagicMock()
        app._app = MagicMock(layout=fake_layout)
        target = object()
        app._focus(target)
        fake_layout.focus.assert_called_once_with(target)


class TestSubmitDispatch:
    def test_submit_with_valid_form_finishes_with_plan(self, app: BootstrapApp) -> None:
        """``_submit`` validates, builds a ``BootstrapPlan`` and dispatches
        via ``_finish`` — verify the full happy path on SCHEMA (datasource
        is pre-seeded by the fixture)."""
        captured: list = []
        app._on_done = lambda r: captured.append(r)
        app._submit()
        assert len(captured) == 1
        assert isinstance(captured[0], BootstrapPlan)
        assert captured[0].task.name == "metadata"
        assert captured[0].task.options == {"datasource": "ssb_sqlite", "build_mode": "incremental"}

    def test_submit_with_invalid_form_records_error(self, app: BootstrapApp) -> None:
        """Validation failures must surface as ``_error_message`` without
        resolving the form (no ``_finish`` call)."""
        captured: list = []
        app._on_done = lambda r: captured.append(r)
        app._schema_datasource.text = ""  # Wipe the required field.
        app._submit()
        assert captured == []
        assert app._error_message  # Some error string.


class TestKeyBindings:
    def test_escape_binding_finishes_with_none(self, app: BootstrapApp) -> None:
        """``Esc`` must cancel the wizard — resolves the embedded future
        with ``None`` when in embedded mode."""
        from prompt_toolkit.keys import Keys

        loop, fut = _make_future()
        try:
            panel = app.build_embedded_panel(fut)
            esc = [b for b in panel.key_bindings.bindings if b.keys == (Keys.Escape,)]
            assert esc, "Escape binding must exist"
            esc[0].handler(type("E", (), {"app": None})())
            assert fut.done() and fut.result() is None
        finally:
            loop.close()

    def test_ctrl_r_binding_submits(self, app: BootstrapApp) -> None:
        """``Ctrl+R`` runs the active tab — drives ``_submit`` directly."""
        from prompt_toolkit.keys import Keys

        loop, fut = _make_future()
        try:
            panel = app.build_embedded_panel(fut)
            ctrl_r = [b for b in panel.key_bindings.bindings if b.keys == (Keys.ControlR,)]
            assert ctrl_r
            ctrl_r[0].handler(type("E", (), {"app": None})())
            # SCHEMA defaults satisfy validation, so future resolves with a plan.
            assert fut.done()
            assert isinstance(fut.result(), BootstrapPlan)
        finally:
            loop.close()

    def test_left_right_arrows_cycle_tab(self, app: BootstrapApp) -> None:
        loop, fut = _make_future()
        try:
            panel = app.build_embedded_panel(fut)
            # Locate "right" and "left" bindings.
            right = [b for b in panel.key_bindings.bindings if b.keys == ("right",)]
            assert right
            assert app._tab == _Tab.SCHEMA
            right[0].handler(type("E", (), {"app": None})())
            assert app._tab == _Tab.SQL
            left = [b for b in panel.key_bindings.bindings if b.keys == ("left",)]
            left[0].handler(type("E", (), {"app": None})())
            assert app._tab == _Tab.SCHEMA
        finally:
            loop.close()


class TestRunStandalone:
    """Drive ``run()`` with a mocked :class:`Application` so the prompt_toolkit
    main loop never starts. Verifies the dual-mode plumbing: ``_on_done``
    gets installed, the result returned by ``Application.run`` propagates
    out, and the finally clause resets state."""

    def test_run_returns_plan_from_app_exit(self, app: BootstrapApp) -> None:
        from unittest.mock import MagicMock, patch

        plan = BootstrapPlan(task=TaskSpec(name="metadata", options={}))
        fake_app = MagicMock()
        fake_app.run.return_value = plan
        with patch("datus.cli.bootstrap_app.Application", return_value=fake_app):
            result = app.run()
        assert result is plan
        assert app._on_done is None  # cleared in finally
        assert app._app is None

    def test_run_returns_none_on_keyboard_interrupt(self, app: BootstrapApp) -> None:
        from unittest.mock import MagicMock, patch

        fake_app = MagicMock()
        fake_app.run.side_effect = KeyboardInterrupt
        with patch("datus.cli.bootstrap_app.Application", return_value=fake_app):
            result = app.run()
        assert result is None
        assert app._on_done is None
        assert app._app is None

    def test_run_swallows_unexpected_exceptions(self, app: BootstrapApp) -> None:
        from unittest.mock import MagicMock, patch

        fake_app = MagicMock()
        fake_app.run.side_effect = RuntimeError("boom")
        with patch("datus.cli.bootstrap_app.Application", return_value=fake_app):
            result = app.run()
        assert result is None
