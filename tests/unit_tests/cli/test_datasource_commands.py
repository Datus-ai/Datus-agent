from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

from datus.cli.datasource_app import DatasourceApp, DatasourceSelection, _View
from datus.cli.datasource_commands import DatasourceCommands

# ── Helpers ───────────────────────────────────────────────────


@dataclass
class _FakeDbConfig:
    type: str = "duckdb"
    uri: str = ""
    host: str = ""
    port: str = ""
    username: str = ""
    password: str = ""
    account: str = ""
    database: str = ""
    schema: str = ""
    warehouse: str = ""
    catalog: str = ""
    logic_name: str = ""
    default: bool = False
    path_pattern: str = ""
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return {
            "type": self.type,
            "uri": self.uri,
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "password": self.password,
            "account": self.account,
            "database": self.database,
            "schema": self.schema,
            "warehouse": self.warehouse,
            "catalog": self.catalog,
            "logic_name": self.logic_name,
            "default": self.default,
            "path_pattern": self.path_pattern,
            "extra": self.extra,
        }


@dataclass
class _FakeServicesConfig:
    datasources: Dict[str, _FakeDbConfig] = field(default_factory=dict)
    semantic_layer: Dict = field(default_factory=dict)
    bi_platforms: Dict = field(default_factory=dict)
    schedulers: Dict = field(default_factory=dict)

    @property
    def default_datasource(self):
        for name, cfg in self.datasources.items():
            if cfg.default:
                return name
        if len(self.datasources) == 1:
            return next(iter(self.datasources))
        return None


def _make_cli(datasources=None, current="local_db"):
    cli = MagicMock()
    cli.console = MagicMock()

    ds = datasources or {
        "local_db": _FakeDbConfig(type="duckdb", uri="duckdb:///test.duckdb", default=True),
        "pg_db": _FakeDbConfig(type="postgresql", host="localhost", port="5432", database="mydb"),
    }
    services = _FakeServicesConfig(datasources=ds)
    cli.agent_config.services = services
    cli.agent_config.current_datasource = current
    cli.agent_config.datasource_configs = {name: {name: cfg} for name, cfg in ds.items()}
    cli.db_manager = MagicMock()
    cli.db_connector = MagicMock()
    cli.cli_context = MagicMock()
    cli.chat_commands = MagicMock()
    cli.reset_session = MagicMock()
    cli._init_connection = MagicMock()
    return cli


# ── DatasourceCommands tests ─────────────────────────────────


class TestCmdDispatch:
    def test_empty_args_runs_menu(self):
        cli = _make_cli()
        cmds = DatasourceCommands(cli)
        with patch.object(cmds, "_run_menu") as mock_menu:
            cmds.cmd("")
            mock_menu.assert_called_once()

    def test_known_datasource_switches(self):
        cli = _make_cli()
        cmds = DatasourceCommands(cli)
        with patch.object(cmds, "_switch") as mock_switch:
            cmds.cmd("pg_db")
            mock_switch.assert_called_once_with("pg_db")

    def test_unknown_datasource_shows_error(self):
        cli = _make_cli()
        cmds = DatasourceCommands(cli)
        cmds.cmd("nonexistent")
        cli.console.print.assert_called()


class TestSwitch:
    def test_switch_updates_context(self):
        cli = _make_cli()
        connector = MagicMock()
        connector.catalog_name = ""
        connector.database_name = "mydb"
        connector.schema_name = "public"
        cli.db_manager.first_conn_with_name.return_value = ("pg_db", connector)

        cmds = DatasourceCommands(cli)
        cmds._switch("pg_db")

        assert cli.agent_config.current_datasource == "pg_db"
        cli.cli_context.update_database_context.assert_called_once()
        cli.reset_session.assert_called_once()

    def test_switch_same_datasource_warns(self):
        cli = _make_cli(current="local_db")
        cmds = DatasourceCommands(cli)
        cmds._switch("local_db")
        cli.console.print.assert_called()
        cli.db_manager.first_conn_with_name.assert_not_called()


class TestRunDelete:
    def test_delete_removes_datasource(self):
        cli = _make_cli()
        cmds = DatasourceCommands(cli)
        with patch.object(cmds, "_save", return_value=True), patch.object(cmds, "_reload_runtime"):
            cmds._run_delete("pg_db")
            assert "pg_db" not in cli.agent_config.services.datasources

    def test_delete_nonexistent_shows_error(self):
        cli = _make_cli()
        cmds = DatasourceCommands(cli)
        cmds._run_delete("nonexistent")
        cli.console.print.assert_called()

    def test_delete_current_auto_switches(self):
        cli = _make_cli(current="local_db")
        cmds = DatasourceCommands(cli)
        with (
            patch.object(cmds, "_save", return_value=True),
            patch.object(cmds, "_reload_runtime"),
            patch.object(cmds, "_switch") as mock_switch,
        ):
            cmds._run_delete("local_db")
            mock_switch.assert_called_once_with("pg_db")


class TestSetDefault:
    def test_set_default_updates_flags(self):
        cli = _make_cli()
        cmds = DatasourceCommands(cli)
        with patch.object(cmds, "_save", return_value=True):
            cmds._set_default("pg_db")
            assert cli.agent_config.services.datasources["pg_db"].default is True
            assert cli.agent_config.services.datasources["local_db"].default is False


class TestInstallPlugin:
    def test_install_success_loads_adapter(self):
        cli = _make_cli()
        cmds = DatasourceCommands(cli)
        mock_module = MagicMock()
        mock_module.register = MagicMock()
        with (
            patch("subprocess.run") as mock_run,
            patch("importlib.invalidate_caches") as mock_invalidate,
            patch("importlib.import_module", return_value=mock_module) as mock_import,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            result = cmds._install_plugin("datus-mysql")
            assert result is True
            mock_invalidate.assert_called_once()
            mock_import.assert_called_once_with("datus_mysql")
            mock_module.register.assert_called_once()

    def test_install_success_adapter_load_fails_still_succeeds(self):
        cli = _make_cli()
        cmds = DatasourceCommands(cli)
        with (
            patch("subprocess.run") as mock_run,
            patch("importlib.invalidate_caches"),
            patch("importlib.import_module", side_effect=ImportError("missing native dep")),
        ):
            mock_run.return_value = MagicMock(returncode=0)
            result = cmds._install_plugin("datus-mysql")
            assert result is True

    def test_install_failure(self):
        cli = _make_cli()
        cmds = DatasourceCommands(cli)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="error msg")
            result = cmds._install_plugin("datus-postgresql")
            assert result is False


class TestMenuLoop:
    def test_menu_switch_exits(self):
        cli = _make_cli()
        cmds = DatasourceCommands(cli)
        sel = DatasourceSelection(kind="switch", name="pg_db")
        with patch.object(cmds, "_run_app", return_value=sel), patch.object(cmds, "_switch") as mock_switch:
            cmds._run_menu()
            mock_switch.assert_called_once_with("pg_db")

    def test_menu_cancel_exits(self):
        cli = _make_cli()
        cmds = DatasourceCommands(cli)
        with patch.object(cmds, "_run_app", return_value=None):
            cmds._run_menu()

    def test_menu_add_submit(self):
        cli = _make_cli()
        cmds = DatasourceCommands(cli)
        sel = DatasourceSelection(
            kind="add_submit",
            db_type="postgresql",
            payload={"type": "postgresql", "_name": "new_pg", "host": "localhost"},
        )
        call_count = [0]

        def side_effect(app):
            call_count[0] += 1
            if call_count[0] == 1:
                return sel
            return None

        with (
            patch.object(cmds, "_run_app", side_effect=side_effect),
            patch.object(cmds, "_handle_add_submit") as mock_add,
        ):
            cmds._run_menu()
            mock_add.assert_called_once_with(sel)

    def test_menu_needs_install_then_continues(self):
        cli = _make_cli()
        cmds = DatasourceCommands(cli)
        sel_install = DatasourceSelection(kind="needs_install", db_type="snowflake")
        call_count = [0]

        def side_effect(app):
            call_count[0] += 1
            if call_count[0] == 1:
                return sel_install
            return None

        with (
            patch.object(cmds, "_run_app", side_effect=side_effect),
            patch.object(cmds, "_install_plugin", return_value=True) as mock_install,
        ):
            cmds._run_menu()
            mock_install.assert_called_once_with("datus-snowflake")


class TestHandleAddSubmit:
    def test_add_submit_saves_and_reloads(self):
        cli = _make_cli()
        cmds = DatasourceCommands(cli)
        sel = DatasourceSelection(
            kind="add_submit",
            db_type="duckdb",
            payload={"type": "duckdb", "_name": "new_duck", "uri": "duckdb:///test.duckdb"},
        )
        with (
            patch.object(cmds, "_test_connectivity", return_value=True),
            patch.object(cmds, "_save", return_value=True),
            patch.object(cmds, "_reload_runtime"),
        ):
            cmds._handle_add_submit(sel)
            assert "new_duck" in cli.agent_config.services.datasources

    def test_add_submit_missing_name(self):
        cli = _make_cli()
        cmds = DatasourceCommands(cli)
        sel = DatasourceSelection(kind="add_submit", db_type="duckdb", payload={"type": "duckdb"})
        cmds._handle_add_submit(sel)
        cli.console.print.assert_called()

    def test_add_submit_connectivity_failure(self):
        cli = _make_cli()
        cmds = DatasourceCommands(cli)
        sel = DatasourceSelection(
            kind="add_submit",
            db_type="duckdb",
            payload={"type": "duckdb", "_name": "new_duck"},
        )
        with patch.object(cmds, "_test_connectivity", return_value=False):
            cmds._handle_add_submit(sel)
            assert "new_duck" not in cli.agent_config.services.datasources


class TestHandleEditSubmit:
    def test_edit_submit_updates_config(self):
        cli = _make_cli()
        cmds = DatasourceCommands(cli)
        sel = DatasourceSelection(
            kind="edit_submit",
            name="pg_db",
            payload={"type": "postgresql", "host": "new-host", "port": 5432},
        )
        with (
            patch.object(cmds, "_test_connectivity", return_value=True),
            patch.object(cmds, "_save", return_value=True),
            patch.object(cmds, "_reload_runtime"),
        ):
            cmds._handle_edit_submit(sel)
            assert "pg_db" in cli.agent_config.services.datasources


class TestTestConnectivity:
    def test_success(self):
        cli = _make_cli()
        cmds = DatasourceCommands(cli)
        with patch("datus.cli.datasource_commands.detect_db_connectivity", return_value=(True, "")):
            assert cmds._test_connectivity("test", {"type": "duckdb"}) is True

    def test_failure(self):
        cli = _make_cli()
        cmds = DatasourceCommands(cli)
        with patch("datus.cli.datasource_commands.detect_db_connectivity", return_value=(False, "refused")):
            assert cmds._test_connectivity("test", {"type": "duckdb"}) is False


class TestSave:
    def test_save_calls_configuration_manager(self):
        cli = _make_cli()
        cmds = DatasourceCommands(cli)
        with patch("datus.configuration.agent_config_loader.configuration_manager") as mock_mgr:
            mock_mgr.return_value.update.return_value = True
            result = cmds._save()
            assert result is True
            mock_mgr.return_value.update.assert_called_once()

    def test_save_returns_false_on_exception(self):
        cli = _make_cli()
        cmds = DatasourceCommands(cli)
        with patch("datus.configuration.agent_config_loader.configuration_manager", side_effect=Exception("boom")):
            result = cmds._save()
            assert result is False


# ── DatasourceApp tests ──────────────────────────────────────


class TestDatasourceAppInit:
    def test_loads_datasources(self):
        cli = _make_cli()
        app = DatasourceApp(cli.agent_config, MagicMock())
        assert len(app._datasources) == 2
        assert app._current == "local_db"

    def test_cursor_on_current(self):
        cli = _make_cli(current="pg_db")
        app = DatasourceApp(cli.agent_config, MagicMock())
        names = [n for n, _, _ in app._datasources]
        assert app._list_cursor == names.index("pg_db")


class TestDatasourceAppViews:
    def test_initial_view_is_list(self):
        cli = _make_cli()
        app = DatasourceApp(cli.agent_config, MagicMock())
        assert app._view == _View.DATASOURCE_LIST

    def test_enter_action_menu(self):
        cli = _make_cli()
        app = DatasourceApp(cli.agent_config, MagicMock())
        app._enter_action_menu("local_db", "duckdb")
        assert app._view == _View.DATASOURCE_ACTIONS
        assert app._active_ds_name == "local_db"

    def test_enter_type_select(self):
        cli = _make_cli()
        app = DatasourceApp(cli.agent_config, MagicMock())
        with patch(
            "datus.tools.db_tools.connector_registry.list_available_adapters",
            return_value={"duckdb": MagicMock(), "sqlite": MagicMock()},
        ):
            app._enter_type_select()
            assert app._view == _View.TYPE_SELECT
            assert len(app._db_types) > 0

    def test_enter_config_form(self):
        cli = _make_cli()
        app = DatasourceApp(cli.agent_config, MagicMock())
        mock_adapter = MagicMock()
        mock_adapter.get_config_fields.return_value = {
            "host": {"required": True},
            "port": {"required": False, "default": 5432, "type": "int"},
            "password": {"required": True, "input_type": "password"},
        }
        with patch(
            "datus.tools.db_tools.connector_registry.list_available_adapters",
            return_value={"postgresql": mock_adapter},
        ):
            app._enter_config_form("postgresql")
            assert app._view == _View.CONFIG_FORM
            assert len(app._form_textareas) == 4
            assert app._form_field_names[0] == "_name"
            assert app._form_container is not None

    def test_enter_config_form_edit_mode(self):
        cli = _make_cli()
        app = DatasourceApp(cli.agent_config, MagicMock())
        mock_adapter = MagicMock()
        mock_adapter.get_config_fields.return_value = {
            "host": {"required": True},
            "port": {"required": False, "default": 5432, "type": "int"},
        }
        with patch(
            "datus.tools.db_tools.connector_registry.list_available_adapters",
            return_value={"postgresql": mock_adapter},
        ):
            app._enter_config_form("postgresql", edit_name="pg_db", existing={"host": "old-host", "port": "5432"})
            assert app._view == _View.CONFIG_FORM
            assert "_name" not in app._form_field_names
            assert app._form_edit_name == "pg_db"
            assert app._form_textareas[0].text == "old-host"

    def test_back_from_actions_to_list(self):
        cli = _make_cli()
        app = DatasourceApp(cli.agent_config, MagicMock())
        app._enter_action_menu("local_db", "duckdb")
        app._enter_datasource_list()
        assert app._view == _View.DATASOURCE_LIST


class TestDatasourceAppRendering:
    def test_render_datasource_list_contains_add(self):
        cli = _make_cli()
        app = DatasourceApp(cli.agent_config, MagicMock())
        lines = app._render_datasource_list()
        texts = "".join(text for _, text in lines)
        assert "Add datasource" in texts

    def test_render_datasource_list_marks_current(self):
        cli = _make_cli(current="local_db")
        app = DatasourceApp(cli.agent_config, MagicMock())
        lines = app._render_datasource_list()
        texts = "".join(text for _, text in lines)
        assert "\u2190 current" in texts

    def test_render_action_menu(self):
        cli = _make_cli()
        app = DatasourceApp(cli.agent_config, MagicMock())
        app._enter_action_menu("local_db", "duckdb")
        lines = app._render_action_menu()
        texts = "".join(text for _, text in lines)
        assert "Edit connection" in texts
        assert "Delete datasource" in texts
        assert "Set as default" in texts

    def test_render_footer_varies_by_view(self):
        cli = _make_cli()
        app = DatasourceApp(cli.agent_config, MagicMock())

        hint_list = app._render_footer_hint()
        hint_text = hint_list[0][1]
        assert "actions" in hint_text

        app._enter_action_menu("local_db", "duckdb")
        hint_actions = app._render_footer_hint()
        hint_text = hint_actions[0][1]
        assert "back" in hint_text

    def test_render_form_footer(self):
        cli = _make_cli()
        app = DatasourceApp(cli.agent_config, MagicMock())
        app._view = _View.CONFIG_FORM
        hint = app._render_footer_hint()
        hint_text = hint.pop()[1]
        assert "Ctrl+S" in hint_text


class TestDatasourceAppFormSubmit:
    def _make_app_with_form(self, edit_name=""):
        cli = _make_cli()
        app = DatasourceApp(cli.agent_config, MagicMock())
        mock_adapter = MagicMock()
        mock_adapter.get_config_fields.return_value = {
            "host": {"required": True},
            "port": {"required": False, "default": 5432, "type": "int"},
        }
        with patch(
            "datus.tools.db_tools.connector_registry.list_available_adapters",
            return_value={"postgresql": mock_adapter},
        ):
            existing = {"host": "old", "port": "5432"} if edit_name else None
            app._enter_config_form("postgresql", edit_name=edit_name, existing=existing)
        return app

    def test_submit_validates_empty_name(self):
        app = self._make_app_with_form()
        app._form_textareas[0].text = ""
        app._submit_form()
        assert app._error_message is not None
        assert "name" in app._error_message.lower()

    def test_submit_validates_invalid_name(self):
        app = self._make_app_with_form()
        app._form_textareas[0].text = "bad name!"
        app._submit_form()
        assert app._error_message is not None

    def test_submit_validates_duplicate_name(self):
        app = self._make_app_with_form()
        app._form_textareas[0].text = "local_db"
        app._submit_form()
        assert app._error_message is not None
        assert "already exists" in app._error_message

    def test_submit_validates_required_fields(self):
        app = self._make_app_with_form()
        app._form_textareas[0].text = "new_ds"
        app._form_textareas[1].text = ""
        app._submit_form()
        assert app._error_message is not None
        assert "required" in app._error_message.lower()

    def test_submit_validates_port_range(self):
        app = self._make_app_with_form()
        app._form_textareas[0].text = "new_ds"
        app._form_textareas[1].text = "localhost"
        app._form_textareas[2].text = "99999"
        # port field name is "port" so validation applies
        # but our test adapter uses "port" key with type "int"
        # Need to adjust: the field is named "port" only if the adapter says so
        # In this test, the field is just "port" with type=int
        app._submit_form()
        # Port validation only triggers when field_name == "port"
        # Our fields are: _name, host, port
        # port has type=int, value=99999 — fails port range check
        assert app._error_message is not None


class TestDatasourceAppCursor:
    def test_clamp_cursor_within_bounds(self):
        cli = _make_cli()
        app = DatasourceApp(cli.agent_config, MagicMock())
        app._list_cursor = 100
        app._clamp_cursor(3)
        assert app._list_cursor == 2

    def test_clamp_cursor_zero_total(self):
        cli = _make_cli()
        app = DatasourceApp(cli.agent_config, MagicMock())
        app._list_cursor = 5
        app._clamp_cursor(0)
        assert app._list_cursor == 0
