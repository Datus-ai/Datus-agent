# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for :class:`datus.cli.service_config_app.ServiceConfigApp`.

Mirrors the test conventions of ``test_model_app.py``: the prompt_toolkit
Application is never run under a pty — each test constructs a
``ServiceConfigApp``, drives its state machine via the ``_on_*`` /
``_submit_form`` action methods and asserts what the app would have
returned to ``ServiceCommands``.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.service_config_app import (
    ServiceConfigApp,
    ServiceConfigSelection,
    _Tab,
    _View,
)


def _stub_agent_config(*, dashboards=None, schedulers=None, semantic=None, active_dash=None, active_sched=None):
    """Minimal in-memory ``AgentConfig`` shim sufficient for the App."""
    cfg = SimpleNamespace()
    cfg.dashboard_config = dashboards or {}
    cfg.scheduler_services = schedulers or {}
    cfg.semantic_layer_configs = semantic or {}
    cfg.active_dashboard = MagicMock(return_value=active_dash)
    cfg.active_scheduler = MagicMock(return_value=active_sched)
    return cfg


def _build_app(**kwargs) -> ServiceConfigApp:
    cfg = _stub_agent_config(**kwargs)
    # ``is_adapter_installed`` is monkey-patched by individual tests when
    # they care; default to True so the type picker doesn't try to pip
    # install.
    with patch("datus.cli.service_config_app.ServiceConfigApp._is_installed", return_value=True):
        app = ServiceConfigApp(cfg, Console(file=io.StringIO(), no_color=True))
    return app


def _dash_cfg(adapter_type="superset", api_base_url="http://x", username="u", password="p"):
    cfg = MagicMock()
    cfg.adapter_type = adapter_type
    cfg.api_base_url = api_base_url
    cfg.username = username
    cfg.password = password
    cfg.api_key = ""
    cfg.extra = {}
    cfg.dataset_db = SimpleNamespace(datasource_ref="serving_db", bi_database_name="examples")
    return cfg


# ─────────────────────────────────────────────────────────────────────
# Tab cycling + LIST view
# ─────────────────────────────────────────────────────────────────────


class TestTabCycle:
    def test_initial_tab_dashboard(self):
        app = _build_app()
        assert app._tab == _Tab.DASHBOARD

    def test_initial_tab_scheduler_when_requested(self):
        cfg = _stub_agent_config()
        with patch("datus.cli.service_config_app.ServiceConfigApp._is_installed", return_value=True):
            app = ServiceConfigApp(cfg, Console(file=io.StringIO()), initial_tab="scheduler")
        assert app._tab == _Tab.SCHEDULER

    def test_cycle_swaps_tabs(self):
        app = _build_app()
        app._cycle_tab(+1)
        assert app._tab == _Tab.SCHEDULER
        app._cycle_tab(+1)
        assert app._tab == _Tab.SEMANTIC
        app._cycle_tab(+1)
        assert app._tab == _Tab.DASHBOARD

    def test_initial_tab_semantic_when_requested(self):
        cfg = _stub_agent_config()
        with patch("datus.cli.service_config_app.ServiceConfigApp._is_installed", return_value=True):
            app = ServiceConfigApp(cfg, Console(file=io.StringIO()), initial_tab="semantic")
        assert app._tab == _Tab.SEMANTIC


class TestListEntries:
    def test_dashboard_list_renders_entries_and_add_row(self):
        app = _build_app(dashboards={"superset": _dash_cfg()})
        # Total rows = entries + 1 (Add row)
        entries = app._entries_for(_Tab.DASHBOARD)
        assert len(entries) == 1
        rendered = app._render_list()
        flat = "".join(text for _, text in rendered)
        assert "superset" in flat
        assert "Add new dashboard" in flat

    def test_scheduler_default_flag_marks_entry(self):
        app = _build_app(
            schedulers={
                "airflow_prod": {"type": "airflow", "default": True},
                "airflow_dev": {"type": "airflow"},
            },
        )
        app._tab = _Tab.SCHEDULER
        entries = app._entries_for(_Tab.SCHEDULER)
        names = {e.name: e.is_default for e in entries}
        assert names == {"airflow_prod": True, "airflow_dev": False}

    def test_project_default_marker_set_from_active_dashboard(self):
        app = _build_app(
            dashboards={"superset": _dash_cfg(), "grafana": _dash_cfg("grafana")},
            active_dash="grafana",
        )
        entries = {e.name: e.is_project_default for e in app._entries_for(_Tab.DASHBOARD)}
        assert entries == {"superset": False, "grafana": True}


# ─────────────────────────────────────────────────────────────────────
# Add → TYPE_PICKER → FORM
# ─────────────────────────────────────────────────────────────────────


class TestAddNewFlow:
    def test_enter_on_add_row_opens_type_picker(self):
        app = _build_app()
        app._list_cursor = 0  # only Add row when no entries
        with patch.object(app._app, "exit"):
            app._on_list_enter()
        assert app._view == _View.TYPE_PICKER
        assert "superset" in app._type_choices

    def test_type_picker_selection_enters_form(self):
        app = _build_app()
        app._enter_type_picker()
        app._type_cursor = 0  # superset
        with patch.object(app._app.layout, "focus"):
            app._on_type_picker_enter()
        assert app._view == _View.FORM
        assert app._form_type == "superset"
        assert app._form_is_edit is False

    def test_form_submit_for_new_bi_emits_save(self):
        app = _build_app()
        app._enter_type_picker()
        app._type_cursor = 0
        with patch.object(app._app.layout, "focus"):
            app._on_type_picker_enter()
        app._fld_name.text = "my_superset"
        app._fld_api_base_url.text = "http://localhost:8088"
        app._fld_username.text = "admin"
        app._fld_password.text = "secret"
        app._fld_datasource_ref.text = "serving_db"
        with patch.object(app._app, "exit") as mock_exit:
            app._submit_form()
        result = mock_exit.call_args.kwargs["result"]
        assert isinstance(result, ServiceConfigSelection)
        assert result.action == "save"
        assert result.section == "bi_platforms"
        assert result.name == "my_superset"
        assert result.payload["type"] == "superset"
        assert result.payload["api_base_url"] == "http://localhost:8088"
        assert result.payload["password"] == "secret"
        assert result.payload["dataset_db"] == {"datasource_ref": "serving_db"}

    def test_form_rejects_blank_api_base_url_for_bi(self):
        app = _build_app()
        app._enter_type_picker()
        with patch.object(app._app.layout, "focus"):
            app._on_type_picker_enter()
        app._fld_name.text = "x"
        app._fld_api_base_url.text = ""  # missing
        with patch.object(app._app, "exit") as mock_exit:
            app._submit_form()
        mock_exit.assert_not_called()
        assert "api_base_url" in (app._error_message or "")

    def test_duplicate_name_rejected_on_add(self):
        app = _build_app(dashboards={"superset": _dash_cfg()})
        app._enter_type_picker()
        with patch.object(app._app.layout, "focus"):
            app._on_type_picker_enter()
        app._fld_name.text = "superset"  # collides
        app._fld_api_base_url.text = "http://x"
        with patch.object(app._app, "exit") as mock_exit:
            app._submit_form()
        mock_exit.assert_not_called()
        assert "already exists" in (app._error_message or "")


# ─────────────────────────────────────────────────────────────────────
# Edit / delete / test / set_default / set_project_default
# ─────────────────────────────────────────────────────────────────────


class TestEditExisting:
    def test_e_loads_existing_and_masks_password(self):
        app = _build_app(dashboards={"superset": _dash_cfg(password="real-secret")})
        app._list_cursor = 0
        with patch.object(app._app.layout, "focus"):
            app._on_edit()
        assert app._view == _View.FORM
        assert app._form_is_edit is True
        # Masked placeholder, not the raw password.
        assert app._fld_password.text != "real-secret"

    def test_submit_with_masked_password_keeps_existing(self):
        from datus.cli.service_config_app import _MASKED_PLACEHOLDER

        app = _build_app(dashboards={"superset": _dash_cfg(password="real-secret")})
        app._list_cursor = 0
        with patch.object(app._app.layout, "focus"):
            app._on_edit()
        # User did not touch the password field.
        assert app._fld_password.text == _MASKED_PLACEHOLDER
        with patch.object(app._app, "exit") as mock_exit:
            app._submit_form()
        result = mock_exit.call_args.kwargs["result"]
        # ``password`` is omitted from payload because the user kept the existing value.
        assert "password" not in result.payload


class TestDeleteAndTest:
    def test_x_emits_delete(self):
        app = _build_app(dashboards={"superset": _dash_cfg()})
        app._list_cursor = 0
        with patch.object(app._app, "exit") as mock_exit:
            app._on_delete()
        result = mock_exit.call_args.kwargs["result"]
        assert result.action == "delete"
        assert result.name == "superset"

    def test_delete_ignored_on_add_row(self):
        app = _build_app()
        app._list_cursor = 0  # only Add row
        with patch.object(app._app, "exit") as mock_exit:
            app._on_delete()
        mock_exit.assert_not_called()

    def test_t_emits_test(self):
        app = _build_app(dashboards={"superset": _dash_cfg()})
        app._list_cursor = 0
        with patch.object(app._app, "exit") as mock_exit:
            app._on_test()
        result = mock_exit.call_args.kwargs["result"]
        assert result.action == "test"


class TestSetGlobalDefault:
    def test_d_emits_set_default_on_dashboard_tab(self):
        """``d`` is now active on every tab — Dashboard included."""
        app = _build_app(dashboards={"superset": _dash_cfg(), "grafana": _dash_cfg("grafana")})
        app._tab = _Tab.DASHBOARD
        app._list_cursor = 0  # grafana sorts before superset alphabetically
        with patch.object(app._app, "exit") as mock_exit:
            app._on_set_default()
        result = mock_exit.call_args.kwargs["result"]
        assert result.action == "set_default"
        assert result.section == "bi_platforms"
        assert result.name == "grafana"

    def test_d_emits_set_default_on_scheduler_tab(self):
        app = _build_app(schedulers={"airflow_prod": {"type": "airflow"}, "airflow_dev": {"type": "airflow"}})
        app._tab = _Tab.SCHEDULER
        app._list_cursor = 0  # airflow_dev sorts before airflow_prod alphabetically
        with patch.object(app._app, "exit") as mock_exit:
            app._on_set_default()
        result = mock_exit.call_args.kwargs["result"]
        assert result.action == "set_default"
        assert result.section == "schedulers"

    def test_d_emits_set_default_on_semantic_tab(self):
        app = _build_app(
            semantic={
                "metricflow": {"type": "metricflow"},
                "dbt": {"type": "dbt"},
            }
        )
        app._tab = _Tab.SEMANTIC
        app._list_cursor = 0  # dbt sorts before metricflow
        with patch.object(app._app, "exit") as mock_exit:
            app._on_set_default()
        result = mock_exit.call_args.kwargs["result"]
        assert result.action == "set_default"
        assert result.section == "semantic_layer"
        assert result.name == "dbt"


class TestSetProjectDefault:
    def test_p_pins_when_not_currently_default(self):
        app = _build_app(
            dashboards={"superset": _dash_cfg(), "grafana": _dash_cfg("grafana")},
            active_dash=None,
        )
        app._list_cursor = 0  # grafana sorts first
        with patch.object(app._app, "exit") as mock_exit:
            app._on_set_project_default()
        result = mock_exit.call_args.kwargs["result"]
        assert result.action == "set_project_default"
        assert result.section == "bi_platforms"
        assert result.name == "grafana"

    def test_p_clears_when_already_pinned(self):
        app = _build_app(dashboards={"superset": _dash_cfg()}, active_dash="superset")
        app._list_cursor = 0
        with patch.object(app._app, "exit") as mock_exit:
            app._on_set_project_default()
        result = mock_exit.call_args.kwargs["result"]
        assert result.action == "set_project_default"
        assert result.name == ""  # clear sentinel


# ─────────────────────────────────────────────────────────────────────
# Type picker — adapter installed indicator
# ─────────────────────────────────────────────────────────────────────


class TestTypePickerInstalled:
    def test_installed_marker_reflects_helper(self):
        cfg = _stub_agent_config()

        # Mark superset as installed, grafana as missing.
        def fake_installed(self, section, type_name):  # noqa: ARG001
            return type_name == "superset"

        with patch(
            "datus.cli.service_config_app.ServiceConfigApp._is_installed",
            new=fake_installed,
        ):
            app = ServiceConfigApp(cfg, Console(file=io.StringIO()))
            app._enter_type_picker()
            rendered = app._render_type_picker()

        flat = "".join(text for _, text in rendered)
        assert "superset" in flat and "(installed)" in flat
        assert "grafana" in flat and "(will pip install)" in flat


# ─────────────────────────────────────────────────────────────────────
# Cursor safety
# ─────────────────────────────────────────────────────────────────────


class TestCursor:
    def test_clamp_with_no_entries_keeps_cursor_on_add_row(self):
        app = _build_app()
        # Only the Add row exists → total = 1, cursor must be 0.
        app._list_cursor = 99
        rendered = app._render_list()  # triggers _clamp_cursor internally
        assert app._list_cursor == 0
        flat = "".join(text for _, text in rendered)
        assert "Add new dashboard" in flat


@pytest.mark.parametrize(
    "section,type_name",
    [
        ("bi_platforms", "superset"),
        ("bi_platforms", "grafana"),
        ("schedulers", "airflow"),
        ("semantic_layer", "metricflow"),
    ],
)
def test_builtin_types_present(section, type_name):
    from datus.cli.service_config_app import _BUILTIN_TYPES

    assert type_name in _BUILTIN_TYPES[section]


def test_schedulers_currently_only_airflow():
    """Lock the scheduler picker to ``airflow`` until another adapter
    package ships. Adding more here is fine; *removing* ``airflow`` would
    silently break the only working scheduler entry."""
    from datus.cli.service_config_app import _BUILTIN_TYPES

    assert _BUILTIN_TYPES["schedulers"] == ("airflow",)


def test_semantic_layer_currently_only_metricflow():
    """Lock the semantic picker to ``metricflow`` until another adapter
    package ships. Removing it would break the sole supported semantic
    layer entry."""
    from datus.cli.service_config_app import _BUILTIN_TYPES

    assert _BUILTIN_TYPES["semantic_layer"] == ("metricflow",)


# ─────────────────────────────────────────────────────────────────────
# Semantic tab — list, type-picker, save, delete
# ─────────────────────────────────────────────────────────────────────


class TestSemanticTab:
    def test_build_semantic_entries_from_configs(self):
        """``is_default`` reflects only the explicit YAML ``default: true``
        flag — the single-entry shortcut is an implicit fallback handled
        by the resolver, not a label we display."""
        app = _build_app(semantic={"metricflow": {"type": "metricflow"}})
        entries = app._entries_for(_Tab.SEMANTIC)
        assert len(entries) == 1
        entry = entries[0]
        assert entry.name == "metricflow"
        assert entry.adapter_type == "metricflow"
        assert entry.is_default is False  # no ``default: true`` in YAML
        assert entry.is_project_default is False

    def test_build_semantic_entries_marks_yaml_default_flag(self):
        app = _build_app(semantic={"metricflow": {"type": "metricflow", "default": True}})
        entry = app._entries_for(_Tab.SEMANTIC)[0]
        assert entry.is_default is True

    def test_render_list_shows_add_new_semantic_row(self):
        app = _build_app()
        app._tab = _Tab.SEMANTIC
        rendered = app._render_list()
        flat = "".join(text for _, text in rendered)
        assert "Add new semantic" in flat

    def test_type_picker_for_semantic_lists_metricflow(self):
        app = _build_app()
        app._tab = _Tab.SEMANTIC
        app._enter_type_picker()
        assert app._type_choices == ["metricflow"]

    def test_type_picker_enter_emits_save_without_form(self):
        app = _build_app()
        app._tab = _Tab.SEMANTIC
        app._enter_type_picker()
        app._type_cursor = 0  # metricflow
        with patch.object(app._app, "exit") as mock_exit:
            app._on_type_picker_enter()
        # FORM view is skipped — selection is emitted straight from the picker.
        assert app._view != _View.FORM
        result = mock_exit.call_args.kwargs["result"]
        assert isinstance(result, ServiceConfigSelection)
        assert result.action == "save"
        assert result.section == "semantic_layer"
        assert result.name == "metricflow"
        assert result.payload == {"type": "metricflow"}

    def test_enter_on_existing_entry_is_noop(self):
        app = _build_app(semantic={"metricflow": {"type": "metricflow"}})
        app._tab = _Tab.SEMANTIC
        app._list_cursor = 0  # existing metricflow row
        with patch.object(app._app, "exit") as mock_exit:
            app._on_list_enter()
        # No FORM, no exit — semantic entries have nothing editable yet.
        assert app._view == _View.LIST
        mock_exit.assert_not_called()

    def test_e_key_ignored_on_semantic_tab(self):
        app = _build_app(semantic={"metricflow": {"type": "metricflow"}})
        app._tab = _Tab.SEMANTIC
        app._list_cursor = 0
        with patch.object(app._app, "exit") as mock_exit:
            app._on_edit()
        assert app._view == _View.LIST
        mock_exit.assert_not_called()

    def test_p_emits_set_project_default_for_semantic(self):
        """Semantic now supports project-level pinning (``set_active_semantic``)
        on par with Dashboard / Scheduler."""
        app = _build_app(semantic={"metricflow": {"type": "metricflow"}})
        app._tab = _Tab.SEMANTIC
        app._list_cursor = 0
        with patch.object(app._app, "exit") as mock_exit:
            app._on_set_project_default()
        result = mock_exit.call_args.kwargs["result"]
        assert result.action == "set_project_default"
        assert result.section == "semantic_layer"
        assert result.name == "metricflow"

    def test_x_emits_delete_for_semantic(self):
        app = _build_app(semantic={"metricflow": {"type": "metricflow"}})
        app._tab = _Tab.SEMANTIC
        app._list_cursor = 0
        with patch.object(app._app, "exit") as mock_exit:
            app._on_delete()
        result = mock_exit.call_args.kwargs["result"]
        assert result.action == "delete"
        assert result.section == "semantic_layer"
        assert result.name == "metricflow"

    def test_t_emits_test_for_semantic(self):
        app = _build_app(semantic={"metricflow": {"type": "metricflow"}})
        app._tab = _Tab.SEMANTIC
        app._list_cursor = 0
        with patch.object(app._app, "exit") as mock_exit:
            app._on_test()
        result = mock_exit.call_args.kwargs["result"]
        assert result.action == "test"
        assert result.section == "semantic_layer"

    def test_footer_hint_includes_default_keys_but_omits_edit(self):
        """``e edit`` is hidden because metricflow has no editable fields,
        but ``d global default`` and ``p project default`` are now shown
        on every tab — Semantic included."""
        app = _build_app(semantic={"metricflow": {"type": "metricflow"}})
        app._tab = _Tab.SEMANTIC
        rendered = app._render_footer_hint()
        flat = "".join(text for _, text in rendered)
        assert "edit" not in flat
        assert "global default" in flat
        assert "project default" in flat
        assert "delete" in flat
        assert "test" in flat
