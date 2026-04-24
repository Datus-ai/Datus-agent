# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :class:`datus.cli.agent_app.AgentApp`.

The prompt_toolkit :class:`Application` is not exercised under a pty;
instead each test constructs an ``AgentApp`` and drives its action
methods directly, mirroring the pattern in ``tests/unit_tests/cli/
test_model_app.py``. :meth:`Application.exit` is patched so assertions
can inspect the :class:`AgentSelection` the app would have returned.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.agent_app import AgentApp, AgentSelection, _Tab, _View
from datus.utils.constants import HIDDEN_SYS_SUB_AGENTS, SYS_SUB_AGENTS

pytestmark = pytest.mark.ci


# ─────────────────────────────────────────────────────────────────────
# Fixtures / builders
# ─────────────────────────────────────────────────────────────────────


def _stub_agent_config(**overrides):
    """Return a MagicMock that looks enough like :class:`AgentConfig`
    for :class:`AgentApp` to initialize.

    ``models`` drives the Built-in edit form's model picker (custom /
    legacy tier). ``agentic_nodes`` seeds the override display.
    """
    cfg = MagicMock()
    cfg.models = overrides.get("models", {"my-internal": SimpleNamespace(type="openai", model="internal-gpt")})
    cfg.agentic_nodes = overrides.get("agentic_nodes", {})
    cfg.set_agentic_node_override = MagicMock()
    return cfg


def _build(**overrides) -> AgentApp:
    cfg = _stub_agent_config(**overrides)
    return AgentApp(
        agent_config=cfg,
        console=Console(file=io.StringIO(), no_color=True),
        default_agent=overrides.get("default_agent", ""),
        visible_custom_agents=overrides.get("visible_custom_agents"),
        seed_tab=overrides.get("seed_tab"),
    )


# ─────────────────────────────────────────────────────────────────────
# Listing / seeding
# ─────────────────────────────────────────────────────────────────────


class TestListing:
    def test_builtin_list_starts_with_chat(self):
        """The ``chat`` pseudo-row is pinned on top so users can always
        reset the default from the Built-in tab, even when no real
        built-in subagent is currently picked."""
        app = _build()
        assert app._builtin_names[0] == "chat"
        assert set(app._builtin_names[1:]) == SYS_SUB_AGENTS - HIDDEN_SYS_SUB_AGENTS

    def test_hidden_builtins_are_excluded(self):
        app = _build()
        for hidden in HIDDEN_SYS_SUB_AGENTS:
            assert hidden not in app._builtin_names

    def test_custom_list_excludes_sys_agents(self):
        """``agentic_nodes`` may carry SYS override entries; those belong
        only to the Built-in tab, never to Custom."""
        cfg_nodes = {
            "gen_sql": {"system_prompt": "gen_sql", "max_turns": 22},
            "my_custom": {"system_prompt": "my_custom", "tools": "db_tools"},
        }
        app = _build(agentic_nodes=cfg_nodes)
        assert "gen_sql" not in app._custom_names
        assert app._custom_names == ["my_custom"]

    def test_seed_tab_custom_positions_cursor(self):
        cfg_nodes = {"alpha": {}, "beta": {}}
        app = _build(agentic_nodes=cfg_nodes, seed_tab="custom", default_agent="beta")
        app._apply_seed()
        assert app._tab == _Tab.CUSTOM
        assert app._custom_names == ["alpha", "beta"]
        assert app._list_cursor == app._custom_names.index("beta")

    def test_seed_tab_builtin_positions_cursor_on_chat_by_default(self):
        """``default_agent=""`` represents "chat" — the pseudo-row we
        pinned at the top. The initial cursor should highlight it."""
        app = _build(seed_tab="builtin")
        app._apply_seed()
        assert app._tab == _Tab.BUILTIN
        assert app._builtin_names[app._list_cursor] == "chat"

    def test_visible_custom_filter_applies(self):
        cfg_nodes = {"alpha": {}, "beta": {}, "gamma": {}}
        app = _build(agentic_nodes=cfg_nodes, visible_custom_agents={"alpha", "gamma"})
        assert app._custom_names == ["alpha", "gamma"]


# ─────────────────────────────────────────────────────────────────────
# Set-default flow
# ─────────────────────────────────────────────────────────────────────


class TestSetDefault:
    def test_enter_on_builtin_row_sets_default(self):
        """``Enter`` on any Built-in row (including ``chat``) sets that
        agent as the session default. The edit form is NOT opened — that
        flow now lives behind ``e``."""
        app = _build()
        gen_sql_idx = app._builtin_names.index("gen_sql")
        app._list_cursor = gen_sql_idx
        with patch.object(app._app, "exit") as exit_mock:
            app._on_list_enter()
        sel = exit_mock.call_args.kwargs["result"]
        assert isinstance(sel, AgentSelection)
        assert sel.kind == "set_default"
        assert sel.name == "gen_sql"

    def test_enter_on_chat_row_resets_default(self):
        app = _build()
        chat_idx = app._builtin_names.index("chat")
        app._list_cursor = chat_idx
        with patch.object(app._app, "exit") as exit_mock:
            app._on_list_enter()
        sel = exit_mock.call_args.kwargs["result"]
        assert sel.kind == "set_default"
        assert sel.name == "chat"

    def test_enter_on_custom_agent_sets_default(self):
        app = _build(agentic_nodes={"alpha": {}, "beta": {}}, seed_tab="custom")
        app._apply_seed()
        app._tab = _Tab.CUSTOM
        app._list_cursor = 1  # beta
        with patch.object(app._app, "exit") as exit_mock:
            app._on_list_enter()
        sel = exit_mock.call_args.kwargs["result"]
        assert sel.kind == "set_default"
        assert sel.name == "beta"


# ─────────────────────────────────────────────────────────────────────
# Custom tab: add / delete / edit
# ─────────────────────────────────────────────────────────────────────


class TestCustomTabActions:
    def test_enter_on_add_row_starts_wizard(self):
        """The ``+ Add agent…`` sentinel isn't an agent — Enter there
        is expected to behave as "add", since there's nothing to set as
        default."""
        app = _build(agentic_nodes={"alpha": {}}, seed_tab="custom")
        app._apply_seed()
        app._tab = _Tab.CUSTOM
        app._list_cursor = len(app._custom_names)
        with patch.object(app._app, "exit") as exit_mock:
            app._on_list_enter()
        sel = exit_mock.call_args.kwargs["result"]
        assert sel.kind == "new_custom"
        assert sel.return_to_tab == "custom"

    def test_edit_key_on_existing_custom_launches_wizard(self):
        app = _build(agentic_nodes={"alpha": {}}, seed_tab="custom")
        app._apply_seed()
        app._tab = _Tab.CUSTOM
        app._list_cursor = 0
        with patch.object(app._app, "exit") as exit_mock:
            app._on_list_edit()
        sel = exit_mock.call_args.kwargs["result"]
        assert sel.kind == "edit_custom"
        assert sel.name == "alpha"

    def test_delete_requires_two_presses(self):
        """First ``d`` press arms confirmation; the app must *not* exit
        until the user presses ``d`` a second time on the same row."""
        app = _build(agentic_nodes={"alpha": {}}, seed_tab="custom")
        app._apply_seed()
        app._tab = _Tab.CUSTOM
        app._list_cursor = 0
        with patch.object(app._app, "exit") as exit_mock:
            app._on_delete_custom()
            exit_mock.assert_not_called()
            assert app._pending_delete_custom == "alpha"
            app._on_delete_custom()
        sel = exit_mock.call_args.kwargs["result"]
        assert sel.kind == "delete_custom"
        assert sel.name == "alpha"
        assert app._pending_delete_custom is None


# ─────────────────────────────────────────────────────────────────────
# Built-in edit form
# ─────────────────────────────────────────────────────────────────────


class TestBuiltinEdit:
    def test_edit_key_on_builtin_opens_single_field_form(self):
        app = _build()
        # Pick a real built-in (not 'chat').
        gen_sql_idx = app._builtin_names.index("gen_sql")
        app._list_cursor = gen_sql_idx
        with patch.object(app._app.layout, "focus"):
            app._on_list_edit()
        assert app._view == _View.BUILTIN_EDIT
        assert app._edit_target == "gen_sql"
        assert app._max_turns_input.text == ""

    def test_edit_key_on_builtin_preselects_existing_max_turns(self):
        cfg_nodes = {"gen_sql": {"system_prompt": "gen_sql", "max_turns": 42}}
        app = _build(agentic_nodes=cfg_nodes)
        gen_sql_idx = app._builtin_names.index("gen_sql")
        app._list_cursor = gen_sql_idx
        with patch.object(app._app.layout, "focus"):
            app._on_list_edit()
        assert app._max_turns_input.text == "42"

    def test_submit_writes_only_max_turns(self):
        """The UI does not expose ``model`` — the save call must pass
        ``model=None`` (cleared) when there is no pre-existing override
        on disk, regardless of what other fields the form might be
        showing."""
        app = _build()
        gen_sql_idx = app._builtin_names.index("gen_sql")
        app._list_cursor = gen_sql_idx
        with patch.object(app._app.layout, "focus"):
            app._on_list_edit()
        app._max_turns_input.text = "15"
        app._submit_builtin_edit()
        app._cfg.set_agentic_node_override.assert_called_once_with("gen_sql", model=None, max_turns=15)
        assert app._view == _View.AGENT_LIST

    def test_submit_preserves_hand_edited_model_override(self):
        """If the YAML already has a ``model`` override (written by
        hand outside the UI), submitting the max_turns form must not
        silently drop it."""
        cfg_nodes = {"gen_sql": {"system_prompt": "gen_sql", "model": "my-internal", "max_turns": 10}}
        app = _build(agentic_nodes=cfg_nodes)
        gen_sql_idx = app._builtin_names.index("gen_sql")
        app._list_cursor = gen_sql_idx
        with patch.object(app._app.layout, "focus"):
            app._on_list_edit()
        app._max_turns_input.text = "42"
        app._submit_builtin_edit()
        app._cfg.set_agentic_node_override.assert_called_once_with("gen_sql", model="my-internal", max_turns=42)

    def test_submit_clears_max_turns_when_empty(self):
        app = _build()
        gen_sql_idx = app._builtin_names.index("gen_sql")
        app._list_cursor = gen_sql_idx
        with patch.object(app._app.layout, "focus"):
            app._on_list_edit()
        app._max_turns_input.text = ""
        app._submit_builtin_edit()
        app._cfg.set_agentic_node_override.assert_called_once_with("gen_sql", model=None, max_turns=None)

    def test_submit_rejects_non_integer_max_turns(self):
        app = _build()
        gen_sql_idx = app._builtin_names.index("gen_sql")
        app._list_cursor = gen_sql_idx
        with patch.object(app._app.layout, "focus"):
            app._on_list_edit()
        app._max_turns_input.text = "not-a-number"
        app._submit_builtin_edit()
        app._cfg.set_agentic_node_override.assert_not_called()
        assert app._view == _View.BUILTIN_EDIT
        assert "integer" in (app._error_message or "")

    def test_submit_rejects_non_positive_max_turns(self):
        app = _build()
        gen_sql_idx = app._builtin_names.index("gen_sql")
        app._list_cursor = gen_sql_idx
        with patch.object(app._app.layout, "focus"):
            app._on_list_edit()
        app._max_turns_input.text = "0"
        app._submit_builtin_edit()
        app._cfg.set_agentic_node_override.assert_not_called()
        assert app._view == _View.BUILTIN_EDIT

    def test_edit_key_on_chat_row_is_rejected(self):
        """``chat`` has no overridable fields — pressing ``e`` on that
        row surfaces an error instead of opening a half-populated form."""
        app = _build()
        chat_idx = app._builtin_names.index("chat")
        app._list_cursor = chat_idx
        app._on_list_edit()
        assert app._view == _View.AGENT_LIST
        assert "chat" in (app._error_message or "")


# ─────────────────────────────────────────────────────────────────────
# Tab switching
# ─────────────────────────────────────────────────────────────────────


class TestTabCycle:
    def test_cycle_tab_forward_goes_builtin_to_custom(self):
        app = _build(agentic_nodes={"alpha": {}})
        assert app._tab == _Tab.BUILTIN
        app._cycle_tab(+1)
        assert app._tab == _Tab.CUSTOM
        app._cycle_tab(+1)
        assert app._tab == _Tab.BUILTIN

    def test_cycle_tab_backward(self):
        app = _build(agentic_nodes={"alpha": {}})
        assert app._tab == _Tab.BUILTIN
        app._cycle_tab(-1)
        assert app._tab == _Tab.CUSTOM
