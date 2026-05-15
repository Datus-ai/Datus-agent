# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :class:`datus.cli.skill_app.SkillApp`.

The Application itself is not exercised under a pty — instead each test
constructs a ``SkillApp`` and drives its state machine by calling the
action methods directly (``_on_install``, ``_on_remove``, ``_cycle_tab``,
``_apply_search_filter``, ...). :meth:`Application.exit` is patched so
we can capture what the app would have returned to its caller.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.skill_app import SkillApp, SkillSelection, _Tab, _View
from datus.tools.skill_tools.skill_config import SkillMetadata

pytestmark = pytest.mark.ci


def _meta(
    name: str,
    *,
    version: str = "1.0.0",
    source: str = "local",
    tags=None,
    description: str = "",
    location: Path = Path("/tmp/skills/example"),
) -> SkillMetadata:
    return SkillMetadata(
        name=name,
        description=description or f"{name} description",
        location=location,
        tags=list(tags or []),
        version=version,
        source=source,
    )


def _manager(marketplace_url: str = "http://localhost:9000") -> MagicMock:
    mgr = MagicMock()
    mgr.config = MagicMock(marketplace_url=marketplace_url, install_dir="~/.datus/skills")
    return mgr


def _build(
    *,
    installed=None,
    marketplace=None,
    seed_tab=None,
    seed_search=None,
) -> SkillApp:
    return SkillApp(
        _manager(),
        Console(file=io.StringIO(), no_color=True),
        installed=installed,
        marketplace=marketplace,
        seed_tab=seed_tab,
        seed_search=seed_search,
    )


# ─────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_default_tab_is_installed(self):
        app = _build()
        assert app._tab is _Tab.INSTALLED
        assert app._view is _View.LIST

    def test_seed_tab_marketplace_selected(self):
        app = _build(seed_tab="marketplace")
        assert app._tab is _Tab.MARKETPLACE

    def test_seed_tab_published_selected(self):
        app = _build(seed_tab="published")
        assert app._tab is _Tab.PUBLISHED

    def test_unknown_seed_falls_back_to_installed(self):
        app = _build(seed_tab="bogus")
        assert app._tab is _Tab.INSTALLED

    def test_published_synthesised_from_marketplace_source(self):
        installed = [
            _meta("local-a", source="local"),
            _meta("mkt-b", source="marketplace"),
            _meta("mkt-c", source="marketplace"),
        ]
        app = _build(installed=installed)
        published_names = [s.name for s in app._published]
        assert published_names == ["mkt-b", "mkt-c"]

    def test_seed_search_is_pre_populated(self):
        app = _build(seed_search="  sales  ")
        assert app._filter_query == "sales"
        assert app._search_input.text == "sales"

    def test_login_url_pre_filled_from_manager_config(self):
        mgr = _manager(marketplace_url="https://town.example.com")
        app = SkillApp(mgr, Console(file=io.StringIO(), no_color=True))
        assert app._login_url.text == "https://town.example.com"


# ─────────────────────────────────────────────────────────────────────
# Client-side filtering
# ─────────────────────────────────────────────────────────────────────


class TestFiltering:
    def test_installed_filter_matches_name(self):
        installed = [_meta("sql-optimization"), _meta("data-profiling")]
        app = _build(installed=installed)
        app._filter_query = "sql"
        visible = app._visible_installed()
        assert [s.name for s in visible] == ["sql-optimization"]

    def test_installed_filter_matches_tags_case_insensitive(self):
        installed = [_meta("foo", tags=["SQL", "reporting"]), _meta("bar", tags=["ml"])]
        app = _build(installed=installed)
        app._filter_query = "sql"
        visible = app._visible_installed()
        assert [s.name for s in visible] == ["foo"]

    def test_marketplace_filter_matches_owner(self):
        marketplace = [
            {"name": "alpha", "owner": "datus", "description": "alpha"},
            {"name": "beta", "owner": "other", "description": "beta"},
        ]
        app = _build(seed_tab="marketplace", marketplace=marketplace)
        app._filter_query = "datus"
        visible = app._visible_marketplace()
        assert [r["name"] for r in visible] == ["alpha"]

    def test_empty_filter_returns_all(self):
        installed = [_meta("a"), _meta("b")]
        app = _build(installed=installed)
        app._filter_query = ""
        assert app._visible_installed() == installed

    def test_apply_search_filter_resets_cursor(self):
        installed = [_meta("a"), _meta("b")]
        app = _build(installed=installed)
        app._list_cursor = 5
        app._list_offset = 3
        app._search_input.text = "b"
        app._view = _View.SEARCH_BAR
        app._apply_search_filter()
        assert app._filter_query == "b"
        assert app._view is _View.LIST
        assert app._list_cursor == 0
        assert app._list_offset == 0

    def test_cancel_search_filter_restores_prior_query(self):
        app = _build()
        app._filter_query = "orig"
        app._search_input.text = "typed-but-not-applied"
        app._view = _View.SEARCH_BAR
        app._cancel_search_filter()
        assert app._search_input.text == "orig"
        assert app._view is _View.LIST


# ─────────────────────────────────────────────────────────────────────
# Tab cycling
# ─────────────────────────────────────────────────────────────────────


class TestTabCycling:
    def test_forward_cycle(self):
        app = _build()
        assert app._tab is _Tab.INSTALLED
        app._cycle_tab(+1)
        assert app._tab is _Tab.MARKETPLACE
        app._cycle_tab(+1)
        assert app._tab is _Tab.PUBLISHED
        app._cycle_tab(+1)
        assert app._tab is _Tab.INSTALLED

    def test_backward_cycle(self):
        app = _build()
        app._cycle_tab(-1)
        assert app._tab is _Tab.PUBLISHED
        app._cycle_tab(-1)
        assert app._tab is _Tab.MARKETPLACE

    def test_cycle_resets_cursor(self):
        app = _build()
        app._list_cursor = 7
        app._list_offset = 3
        app._cycle_tab(+1)
        assert app._list_cursor == 0
        assert app._list_offset == 0


# ─────────────────────────────────────────────────────────────────────
# Action handlers → SkillSelection
# ─────────────────────────────────────────────────────────────────────


class TestActions:
    def test_install_emits_selection_with_version(self):
        marketplace = [{"name": "sql-opt", "latest_version": "2.1", "owner": "datus"}]
        app = _build(seed_tab="marketplace", marketplace=marketplace)
        with patch.object(app, "_finish") as exit_mock:
            app._on_install()
        sel = exit_mock.call_args.args[0]
        assert isinstance(sel, SkillSelection)
        assert sel.kind == "install"
        assert sel.name == "sql-opt"
        assert sel.version == "2.1"

    def test_install_falls_back_to_latest_when_version_missing(self):
        marketplace = [{"name": "sql-opt", "owner": "datus"}]
        app = _build(seed_tab="marketplace", marketplace=marketplace)
        with patch.object(app, "_finish") as exit_mock:
            app._on_install()
        sel = exit_mock.call_args.args[0]
        assert sel.version == "latest"

    def test_install_ignores_non_marketplace_row(self):
        app = _build(installed=[_meta("local-a")])
        # Cursor is on INSTALLED tab; _current_row returns SkillMetadata, not dict.
        with patch.object(app, "_finish") as exit_mock:
            app._on_install()
        exit_mock.assert_not_called()

    def test_update_requires_marketplace_source(self):
        installed = [_meta("local-only", source="local")]
        app = _build(installed=installed)
        with patch.object(app, "_finish") as exit_mock:
            app._on_update()
        exit_mock.assert_not_called()
        assert "not marketplace-sourced" in (app._error_message or "")

    def test_update_on_marketplace_source_emits_selection(self):
        installed = [_meta("mkt-a", source="marketplace")]
        app = _build(installed=installed)
        with patch.object(app, "_finish") as exit_mock:
            app._on_update()
        sel = exit_mock.call_args.args[0]
        assert sel.kind == "update"
        assert sel.name == "mkt-a"

    def test_remove_two_press_confirmation(self):
        installed = [_meta("foo", source="local")]
        app = _build(installed=installed)
        with patch.object(app, "_finish") as exit_mock:
            app._on_remove()
            assert exit_mock.call_count == 0
            assert app._pending_remove == "foo"
            assert "Press r again" in (app._error_message or "")
            app._on_remove()
            assert exit_mock.call_count == 1
        sel = exit_mock.call_args.args[0]
        assert sel.kind == "remove"
        assert sel.name == "foo"
        assert app._pending_remove is None

    def test_logout_emits_selection(self):
        app = _build()
        with patch.object(app, "_finish") as exit_mock:
            app._on_logout()
        sel = exit_mock.call_args.args[0]
        assert sel.kind == "logout"

    def test_refresh_emits_selection(self):
        app = _build(seed_tab="marketplace")
        with patch.object(app, "_finish") as exit_mock:
            app._on_refresh()
        sel = exit_mock.call_args.args[0]
        assert sel.kind == "refresh"


# ─────────────────────────────────────────────────────────────────────
# Login form submission
# ─────────────────────────────────────────────────────────────────────


class TestLoginForm:
    def test_submit_requires_email(self):
        app = _build()
        app._enter_login_form()
        app._login_email.text = ""
        app._login_password.text = "pw"
        with patch.object(app, "_finish") as exit_mock:
            app._submit_login_form()
        exit_mock.assert_not_called()
        assert "Email" in (app._error_message or "")

    def test_submit_requires_password(self):
        app = _build()
        app._enter_login_form()
        app._login_email.text = "me@example.com"
        app._login_password.text = ""
        with patch.object(app, "_finish") as exit_mock:
            app._submit_login_form()
        exit_mock.assert_not_called()
        assert "Password" in (app._error_message or "")

    def test_submit_emits_credentials_with_default_url(self):
        mgr = _manager(marketplace_url="https://town.example.com")
        app = SkillApp(mgr, Console(file=io.StringIO(), no_color=True))
        app._enter_login_form()
        app._login_email.text = "me@example.com"
        app._login_password.text = "secret"
        with patch.object(app, "_finish") as exit_mock:
            app._submit_login_form()
        sel = exit_mock.call_args.args[0]
        assert sel.kind == "login"
        assert sel.email == "me@example.com"
        assert sel.password == "secret"
        assert sel.marketplace_url == "https://town.example.com"

    def test_submit_respects_user_edited_url(self):
        app = _build()
        app._enter_login_form()
        app._login_email.text = "me@example.com"
        app._login_password.text = "secret"
        app._login_url.text = "https://override.example.com/"
        with patch.object(app, "_finish") as exit_mock:
            app._submit_login_form()
        sel = exit_mock.call_args.args[0]
        assert sel.marketplace_url == "https://override.example.com/"


# ─────────────────────────────────────────────────────────────────────
# Detail mapping
# ─────────────────────────────────────────────────────────────────────


class TestDetailMapping:
    def test_detail_fields_for_skill_metadata(self):
        skill = _meta(
            "my-skill",
            version="1.2.3",
            source="marketplace",
            tags=["sql", "perf"],
            description="optimises SQL",
        )
        fields = dict(SkillApp._detail_fields(skill))
        assert fields["Name"] == "my-skill"
        assert fields["Version"] == "1.2.3"
        assert fields["Source"] == "marketplace"
        assert "sql" in fields["Tags"]
        assert fields["Description"] == "optimises SQL"

    def test_detail_fields_for_marketplace_dict(self):
        row = {
            "name": "remote-skill",
            "latest_version": "0.9",
            "owner": "datus",
            "promoted": True,
            "usage_count": 42,
            "tags": ["ml"],
            "description": "remote",
        }
        fields = dict(SkillApp._detail_fields(row))
        assert fields["Name"] == "remote-skill"
        assert fields["Latest Version"] == "0.9"
        assert fields["Promoted"] == "yes"
        assert fields["Usage Count"] == "42"
        assert fields["Tags"] == "ml"

    def test_detail_fields_for_unknown_type_returns_empty(self):
        assert SkillApp._detail_fields("nope") == []


# ─────────────────────────────────────────────────────────────────────
# Dual-mode finish hook + embedded panel + standalone run() with mocked
# Application.
# ─────────────────────────────────────────────────────────────────────


import asyncio  # noqa: E402

from datus.cli.tui.wizard_host import EmbeddedWizard  # noqa: E402


def _make_future():
    loop = asyncio.new_event_loop()
    return loop, loop.create_future()


class TestEmbeddedPanel:
    def test_build_embedded_panel_returns_wizard(self):
        app = _build()
        loop, fut = _make_future()
        try:
            panel = app.build_embedded_panel(fut)
            assert isinstance(panel, EmbeddedWizard)
            assert panel.done_future is fut
            assert app._on_done is not None
            # Embedded panels seed focus on the list window the layout
            # builder wires up — pointing focus at None drops the cursor.
            assert panel.first_focus is app._list_window
        finally:
            loop.close()

    def test_embedded_finish_with_selection_resolves(self):
        app = _build()
        loop, fut = _make_future()
        try:
            app.build_embedded_panel(fut)
            sel = SkillSelection(kind="refresh")
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
        app._finish(None)  # No raise.

    def test_layout_returns_app_layout(self):
        app = _build()
        fake_layout = MagicMock()
        app._app = MagicMock(layout=fake_layout)
        assert app._layout() is fake_layout

    def test_layout_falls_back_to_none_when_get_app_raises(self):
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
        sentinel = object()
        app._focus(sentinel)
        fake_layout.focus.assert_called_once_with(sentinel)


class TestRunStandalone:
    def test_run_returns_selection(self):
        app = _build()
        sel = SkillSelection(kind="refresh")
        fake_app = MagicMock()
        fake_app.run.return_value = sel
        with patch("datus.cli.skill_app.Application", return_value=fake_app):
            assert app.run() is sel
        assert app._on_done is None
        assert app._app is None

    def test_run_keyboard_interrupt_returns_none(self):
        app = _build()
        fake_app = MagicMock()
        fake_app.run.side_effect = KeyboardInterrupt
        with patch("datus.cli.skill_app.Application", return_value=fake_app):
            assert app.run() is None
