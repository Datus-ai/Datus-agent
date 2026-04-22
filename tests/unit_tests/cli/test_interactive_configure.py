# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus/cli/interactive_configure.py``.

Scope: database management only. LLM / provider selection moved to
``datus.cli.model_commands`` and is covered by its own unit tests.
"""

import subprocess
from unittest.mock import MagicMock, patch

import yaml


def _make_configure(tmp_path):
    """Construct an :class:`InteractiveConfigure` with paths pinned to ``tmp_path``."""
    from datus.cli.interactive_configure import InteractiveConfigure

    with patch("datus.cli.interactive_configure.get_path_manager") as mock_pm:
        pm = MagicMock()
        pm.conf_dir = tmp_path
        pm.template_dir = tmp_path / "templates"
        pm.sample_dir = tmp_path / "sample"
        mock_pm.return_value = pm

        cfg = InteractiveConfigure(user_home=str(tmp_path))
        cfg.config_path = tmp_path / "agent.yml"
        return cfg


def _make_adapter_metadata(fields=None):
    """Return a mock adapter metadata object with the supplied config fields."""
    if fields is None:
        fields = {"uri": {"required": True, "input_type": "text"}}
    meta = MagicMock()
    meta.get_config_fields.return_value = fields
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# _prompt_with_back
# ─────────────────────────────────────────────────────────────────────────────


class TestPromptWithBack:
    """Module-level helper driven by prompt_toolkit, patched via sys.modules."""

    def _make_mock_session(self, return_value=None, side_effect=None):
        mock_session_instance = MagicMock()
        if side_effect is not None:
            mock_session_instance.prompt.side_effect = side_effect
        else:
            mock_session_instance.prompt.return_value = return_value
        mock_session_class = MagicMock(return_value=mock_session_instance)
        return mock_session_class, mock_session_instance

    def test_returns_user_input(self):
        from datus.cli.interactive_configure import _prompt_with_back

        session_cls, _ = self._make_mock_session("  my value  ")
        mock_pt = MagicMock()
        mock_pt.PromptSession = session_cls
        with patch.dict("sys.modules", {"prompt_toolkit": mock_pt, "prompt_toolkit.key_binding": mock_pt.key_binding}):
            assert _prompt_with_back("Label") == "my value"

    def test_returns_default_when_empty_input(self):
        from datus.cli.interactive_configure import _prompt_with_back

        session_cls, _ = self._make_mock_session("")
        mock_pt = MagicMock()
        mock_pt.PromptSession = session_cls
        with patch.dict("sys.modules", {"prompt_toolkit": mock_pt, "prompt_toolkit.key_binding": mock_pt.key_binding}):
            assert _prompt_with_back("Label", default="mydefault") == "mydefault"

    def test_returns_back_when_result_is_back_sentinel(self):
        from datus.cli.interactive_configure import _BACK, _prompt_with_back

        session_cls, _ = self._make_mock_session(_BACK)
        mock_pt = MagicMock()
        mock_pt.PromptSession = session_cls
        with patch.dict("sys.modules", {"prompt_toolkit": mock_pt, "prompt_toolkit.key_binding": mock_pt.key_binding}):
            assert _prompt_with_back("Label") == _BACK

    def test_returns_back_on_keyboard_interrupt(self):
        from datus.cli.interactive_configure import _BACK, _prompt_with_back

        session_cls, _ = self._make_mock_session(side_effect=KeyboardInterrupt)
        mock_pt = MagicMock()
        mock_pt.PromptSession = session_cls
        with patch.dict("sys.modules", {"prompt_toolkit": mock_pt, "prompt_toolkit.key_binding": mock_pt.key_binding}):
            assert _prompt_with_back("Label") == _BACK


# ─────────────────────────────────────────────────────────────────────────────
# _init_dirs and _copy_files
# ─────────────────────────────────────────────────────────────────────────────


class TestInitDirsAndCopyFiles:
    def test_init_dirs_calls_ensure_dirs(self, tmp_path):
        cfg = _make_configure(tmp_path)
        mock_pm = MagicMock()
        with patch("datus.cli.interactive_configure.get_path_manager", return_value=mock_pm):
            cfg._init_dirs()
        mock_pm.ensure_dirs.assert_called_once()
        args = mock_pm.ensure_dirs.call_args[0]
        assert "conf" in args
        assert "data" in args
        # Sessions are project-scoped and set up lazily at runtime.
        assert "sessions" not in args

    def test_copy_files_handles_errors_gracefully(self, tmp_path):
        cfg = _make_configure(tmp_path)
        with (
            patch("datus.cli.interactive_configure.copy_data_file", side_effect=Exception("copy failed")),
            patch("datus.cli.interactive_configure.logger.debug") as mock_debug,
        ):
            cfg._copy_files()
        assert mock_debug.call_count == 3

    def test_copy_files_copies_prompts_and_samples(self, tmp_path):
        cfg = _make_configure(tmp_path)
        with patch("datus.cli.interactive_configure.copy_data_file") as mock_copy:
            cfg._copy_files()
        assert mock_copy.call_count >= 2


# ─────────────────────────────────────────────────────────────────────────────
# _load_existing_config
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadExistingConfig:
    def test_service_databases_format_populates_self_databases(self, tmp_path):
        raw = {
            "agent": {
                "services": {
                    "datasources": {"my_db": {"type": "sqlite", "uri": "data/test.sqlite"}},
                    "semantic_layer": {},
                    "bi_platforms": {},
                    "schedulers": {},
                }
            }
        }
        (tmp_path / "agent.yml").write_text(yaml.dump(raw), encoding="utf-8")

        cfg = _make_configure(tmp_path)
        cfg._load_existing_config()

        assert cfg.datasources["my_db"]["type"] == "sqlite"

    def test_legacy_namespace_format_auto_migrates(self, tmp_path):
        raw = {
            "agent": {
                "namespace": {"legacy_db": {"type": "duckdb", "uri": "legacy.duckdb"}},
            }
        }
        (tmp_path / "agent.yml").write_text(yaml.dump(raw), encoding="utf-8")

        migrated = {"datasources": {"legacy_db": {"type": "duckdb", "uri": "legacy.duckdb"}}}
        with patch(
            "datus.configuration.agent_config.ServicesConfig.migrate_from_namespace",
            return_value=migrated,
        ):
            cfg = _make_configure(tmp_path)
            cfg._load_existing_config()

        assert "legacy_db" in cfg.datasources

    def test_missing_config_file_leaves_empty_state(self, tmp_path):
        cfg = _make_configure(tmp_path)
        cfg.config_path = tmp_path / "nonexistent.yml"
        cfg._load_existing_config()
        assert cfg.datasources == {}

    def test_malformed_yaml_leaves_empty_state(self, tmp_path):
        (tmp_path / "agent.yml").write_text(": bad: yaml: {[", encoding="utf-8")
        cfg = _make_configure(tmp_path)
        cfg._load_existing_config()
        assert cfg.datasources == {}

    def test_llm_sections_are_not_loaded(self, tmp_path):
        """``InteractiveConfigure`` no longer tracks ``target`` / ``models``."""
        raw = {
            "agent": {
                "target": "openai",
                "models": {"openai": {"type": "openai", "model": "gpt-4o"}},
                "providers": {"openai": {"api_key": "sk-x"}},
                "services": {"datasources": {"db": {"type": "sqlite", "uri": "x.sqlite"}}},
            }
        }
        (tmp_path / "agent.yml").write_text(yaml.dump(raw), encoding="utf-8")
        cfg = _make_configure(tmp_path)
        cfg._load_existing_config()
        assert cfg.datasources == {"db": {"type": "sqlite", "uri": "x.sqlite"}}
        assert not hasattr(cfg, "target")
        assert not hasattr(cfg, "models")


# ─────────────────────────────────────────────────────────────────────────────
# _show_current_state
# ─────────────────────────────────────────────────────────────────────────────


class TestShowCurrentState:
    def test_empty_state_does_not_raise(self, tmp_path):
        import io

        from rich.console import Console

        cfg = _make_configure(tmp_path)
        cfg.console = Console(file=io.StringIO(), no_color=True)
        cfg._show_current_state()
        output = cfg.console.file.getvalue()
        assert "no datasources" in output.lower()

    def test_database_with_host_field_shows_host(self, tmp_path):
        import io

        from rich.console import Console

        cfg = _make_configure(tmp_path)
        cfg.console = Console(file=io.StringIO(), no_color=True)
        cfg.datasources = {"pg": {"type": "postgresql", "host": "db.example.com"}}
        cfg._show_current_state()
        output = cfg.console.file.getvalue()
        assert "db.example.com" in output

    def test_database_default_marked_with_asterisk(self, tmp_path):
        import io

        from rich.console import Console

        cfg = _make_configure(tmp_path)
        cfg.console = Console(file=io.StringIO(), no_color=True)
        cfg.datasources = {
            "d1": {"type": "sqlite", "uri": "d1.sqlite", "default": True},
            "d2": {"type": "sqlite", "uri": "d2.sqlite"},
        }
        cfg._show_current_state()
        output = cfg.console.file.getvalue()
        assert "*" in output


# ─────────────────────────────────────────────────────────────────────────────
# run() dispatches first-time vs. menu based on whether datasources exist
# ─────────────────────────────────────────────────────────────────────────────


class TestRun:
    def test_first_time_setup_when_no_databases(self, tmp_path):
        cfg = _make_configure(tmp_path)
        with (
            patch.object(cfg, "_init_dirs"),
            patch.object(cfg, "_copy_files"),
            patch.object(cfg, "_load_existing_config"),
            patch.object(cfg, "_first_time_setup", return_value=0) as first_time,
            patch.object(cfg, "_interactive_menu") as menu,
        ):
            assert cfg.run() == 0
            first_time.assert_called_once()
            menu.assert_not_called()

    def test_interactive_menu_when_databases_exist(self, tmp_path):
        cfg = _make_configure(tmp_path)
        cfg.datasources = {"d": {"type": "sqlite", "uri": "x"}}
        with (
            patch.object(cfg, "_init_dirs"),
            patch.object(cfg, "_copy_files"),
            patch.object(cfg, "_load_existing_config"),
            patch.object(cfg, "_interactive_menu", return_value=0) as menu,
        ):
            assert cfg.run() == 0
            menu.assert_called_once()

    def test_keyboard_interrupt_returns_1(self, tmp_path):
        cfg = _make_configure(tmp_path)
        with (
            patch.object(cfg, "_init_dirs", side_effect=KeyboardInterrupt),
        ):
            assert cfg.run() == 1

    def test_exception_returns_1(self, tmp_path):
        cfg = _make_configure(tmp_path)
        with (
            patch.object(cfg, "_init_dirs"),
            patch.object(cfg, "_copy_files"),
            patch.object(cfg, "_load_existing_config", side_effect=RuntimeError("boom")),
            patch("datus.cli.interactive_configure.print_rich_exception"),
        ):
            assert cfg.run() == 1


# ─────────────────────────────────────────────────────────────────────────────
# _interactive_menu (DB-only choice flow)
# ─────────────────────────────────────────────────────────────────────────────


class TestInteractiveMenu:
    def test_done_exits(self, tmp_path):
        cfg = _make_configure(tmp_path)
        with patch("datus.cli.interactive_configure.select_choice", return_value="done"):
            assert cfg._interactive_menu() == 0

    def test_add_database_calls_add_and_save(self, tmp_path):
        cfg = _make_configure(tmp_path)
        with (
            patch("datus.cli.interactive_configure.select_choice", side_effect=["add_database", "done"]),
            patch.object(cfg, "_add_database", return_value=True) as add_db,
            patch.object(cfg, "_save") as save,
        ):
            cfg._interactive_menu()
        add_db.assert_called_once()
        save.assert_called_once()

    def test_delete_database_only_shown_when_databases_exist(self, tmp_path):
        cfg = _make_configure(tmp_path)
        cfg.datasources = {"db": {"type": "sqlite", "uri": "x"}}
        seen_choices = []

        def _capture(_console, choices, default="done"):
            seen_choices.append(dict(choices))
            return "done"

        with patch("datus.cli.interactive_configure.select_choice", side_effect=_capture):
            cfg._interactive_menu()
        assert "delete_database" in seen_choices[0]
        assert "add_database" in seen_choices[0]
        # Legacy LLM actions must not appear.
        assert "add_model" not in seen_choices[0]
        assert "delete_model" not in seen_choices[0]
        assert "set_default_model" not in seen_choices[0]


# ─────────────────────────────────────────────────────────────────────────────
# _add_database
# ─────────────────────────────────────────────────────────────────────────────


class TestAddDatabase:
    def _mock_connector_registry(self, types=("duckdb", "sqlite")):
        registry = MagicMock()
        registry.list_available_adapters.return_value = {t: _make_adapter_metadata() for t in types}
        return registry

    def test_successful_database_addition(self, tmp_path):
        cfg = _make_configure(tmp_path)
        registry = self._mock_connector_registry()
        with (
            patch("datus.cli.interactive_configure.Prompt.ask", side_effect=["my_db"]),
            patch("datus.cli.interactive_configure.select_choice", return_value="duckdb"),
            patch("datus.cli.interactive_configure._prompt_with_back", return_value="duckdb:///my.duckdb"),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(True, "")),
            patch("datus.tools.db_tools.connector_registry", registry),
        ):
            ok = cfg._add_database()
        assert ok is True
        assert cfg.datasources["my_db"]["type"] == "duckdb"
        assert cfg.datasources["my_db"].get("default") is True, "First database should be marked default"

    def test_returns_false_when_db_name_already_exists(self, tmp_path):
        cfg = _make_configure(tmp_path)
        cfg.datasources = {"my_db": {"type": "sqlite", "uri": "x"}}
        registry = self._mock_connector_registry()
        with (
            patch("datus.cli.interactive_configure.Prompt.ask", side_effect=["my_db", "new_db"]),
            patch("datus.cli.interactive_configure.select_choice", return_value="duckdb"),
            patch("datus.cli.interactive_configure._prompt_with_back", return_value="duckdb:///my.duckdb"),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(False, "refused")),
            patch("datus.tools.db_tools.connector_registry", registry),
        ):
            result = cfg._add_database()
        assert result is False
        assert "new_db" not in cfg.datasources

    def test_returns_false_when_connectivity_fails(self, tmp_path):
        cfg = _make_configure(tmp_path)
        registry = self._mock_connector_registry()
        with (
            patch("datus.cli.interactive_configure.Prompt.ask", side_effect=["my_db"]),
            patch("datus.cli.interactive_configure.select_choice", return_value="duckdb"),
            patch("datus.cli.interactive_configure._prompt_with_back", return_value="duckdb:///my.duckdb"),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(False, "refused")),
            patch("datus.tools.db_tools.connector_registry", registry),
        ):
            ok = cfg._add_database()
        assert ok is False
        assert "my_db" not in cfg.datasources


# ─────────────────────────────────────────────────────────────────────────────
# _delete_database
# ─────────────────────────────────────────────────────────────────────────────


class TestDeleteDatabase:
    def test_delete_database_removes_entry(self, tmp_path):
        cfg = _make_configure(tmp_path)
        cfg.datasources = {"db1": {"type": "sqlite", "uri": "x"}, "db2": {"type": "sqlite", "uri": "y"}}
        with (
            patch("datus.cli.interactive_configure.Prompt.ask", return_value="db1"),
            patch("datus.cli.interactive_configure.Confirm.ask", return_value=True),
        ):
            cfg._delete_database()
        assert "db1" not in cfg.datasources
        assert "db2" in cfg.datasources

    def test_delete_database_cancelled(self, tmp_path):
        cfg = _make_configure(tmp_path)
        cfg.datasources = {"db1": {"type": "sqlite", "uri": "x"}}
        with (
            patch("datus.cli.interactive_configure.Prompt.ask", return_value="db1"),
            patch("datus.cli.interactive_configure.Confirm.ask", return_value=False),
        ):
            cfg._delete_database()
        assert "db1" in cfg.datasources


# ─────────────────────────────────────────────────────────────────────────────
# _save
# ─────────────────────────────────────────────────────────────────────────────


class TestSave:
    def test_save_writes_services_structure(self, tmp_path):
        cfg = _make_configure(tmp_path)
        cfg.datasources = {"d": {"type": "sqlite", "uri": "x", "default": True}}
        cfg._save()
        saved = yaml.safe_load(cfg.config_path.read_text(encoding="utf-8"))
        assert saved["agent"]["services"]["datasources"] == cfg.datasources
        # LLM sections must be untouched by this wizard.
        assert "models" not in saved["agent"]
        assert "target" not in saved["agent"]

    def test_save_preserves_llm_sections(self, tmp_path):
        """Existing ``agent.models`` / ``agent.providers`` survive round-trips."""
        raw = {
            "agent": {
                "providers": {"openai": {"api_key": "sk-x"}},
                "models": {"custom": {"type": "openai", "model": "m", "api_key": "k"}},
                "services": {"datasources": {}, "semantic_layer": {}, "bi_platforms": {}, "schedulers": {}},
            }
        }
        cfg = _make_configure(tmp_path)
        cfg.config_path.write_text(yaml.dump(raw), encoding="utf-8")
        cfg.datasources = {"d": {"type": "sqlite", "uri": "x"}}
        cfg._save()
        saved = yaml.safe_load(cfg.config_path.read_text(encoding="utf-8"))
        assert saved["agent"]["providers"] == raw["agent"]["providers"]
        assert saved["agent"]["models"] == raw["agent"]["models"]
        assert "d" in saved["agent"]["services"]["datasources"]

    def test_save_removes_legacy_namespace_key(self, tmp_path):
        raw = {
            "agent": {
                "namespace": {"ns1": {"type": "duckdb"}},
                "services": {"datasources": {}, "semantic_layer": {}, "bi_platforms": {}, "schedulers": {}},
            }
        }
        cfg = _make_configure(tmp_path)
        cfg.config_path.write_text(yaml.dump(raw), encoding="utf-8")
        cfg.datasources = {"d": {"type": "sqlite", "uri": "x"}}
        cfg._save()
        saved = yaml.safe_load(cfg.config_path.read_text(encoding="utf-8"))
        assert "namespace" not in saved["agent"]

    def test_save_sets_default_nodes_when_absent(self, tmp_path):
        cfg = _make_configure(tmp_path)
        cfg.datasources = {"d": {"type": "sqlite", "uri": "x"}}
        cfg._save()
        saved = yaml.safe_load(cfg.config_path.read_text(encoding="utf-8"))
        assert "nodes" in saved["agent"]
        assert "schema_linking" in saved["agent"]["nodes"]


# ─────────────────────────────────────────────────────────────────────────────
# _display_completion
# ─────────────────────────────────────────────────────────────────────────────


class TestDisplayCompletion:
    def test_shows_default_datasource_in_message(self, tmp_path, capsys):
        cfg = _make_configure(tmp_path)
        cfg.datasources = {
            "prod": {"type": "sqlite", "uri": "x", "default": True},
            "other": {"type": "sqlite", "uri": "y"},
        }
        cfg._display_completion()
        captured = capsys.readouterr()
        assert "prod" in captured.out

    def test_shows_generic_message_when_no_databases(self, tmp_path, capsys):
        cfg = _make_configure(tmp_path)
        cfg._display_completion()
        captured = capsys.readouterr()
        assert "datus init" in captured.out.lower()


# ─────────────────────────────────────────────────────────────────────────────
# _install_plugin
# ─────────────────────────────────────────────────────────────────────────────


class TestInstallPlugin:
    def test_returns_true_on_success(self, tmp_path):
        cfg = _make_configure(tmp_path)
        proc = MagicMock()
        proc.returncode = 0
        with patch("subprocess.run", return_value=proc):
            assert cfg._install_plugin("datus-mysql") is True

    def test_returns_false_on_nonzero_exit(self, tmp_path):
        cfg = _make_configure(tmp_path)
        proc = MagicMock()
        proc.returncode = 1
        proc.stderr = "error"
        with patch("subprocess.run", return_value=proc):
            assert cfg._install_plugin("datus-mysql") is False

    def test_returns_false_on_timeout(self, tmp_path):
        cfg = _make_configure(tmp_path)
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="", timeout=1)):
            assert cfg._install_plugin("datus-mysql") is False

    def test_returns_false_on_generic_exception(self, tmp_path):
        cfg = _make_configure(tmp_path)
        with patch("subprocess.run", side_effect=Exception("boom")):
            assert cfg._install_plugin("datus-mysql") is False
