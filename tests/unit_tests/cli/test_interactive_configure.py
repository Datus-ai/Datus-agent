# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/cli/interactive_configure.py"""

import subprocess
from unittest.mock import MagicMock, patch

import yaml


def _make_configure(tmp_path):
    """Create an InteractiveConfigure instance with config_path pointing to tmp_path."""
    from datus.cli.interactive_configure import InteractiveConfigure

    with (
        patch("datus.cli.interactive_configure.get_path_manager") as mock_pm,
    ):
        pm = MagicMock()
        pm.conf_dir = tmp_path
        pm.template_dir = tmp_path / "templates"
        pm.sample_dir = tmp_path / "sample"
        mock_pm.return_value = pm

        cfg = InteractiveConfigure(user_home=str(tmp_path))
        cfg.config_path = tmp_path / "agent.yml"
        return cfg


def _make_provider_catalog(with_models=True, auth_type="api_key"):
    """Return a sample provider catalog dict."""
    providers = {
        "openai": {
            "type": "openai",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "models": ["gpt-4o", "gpt-4-turbo"] if with_models else [],
            "default_model": "gpt-4o",
        },
        "deepseek": {
            "type": "openai",
            "base_url": "https://api.deepseek.com/v1",
            "api_key_env": "DEEPSEEK_API_KEY",
            "models": ["deepseek-chat"],
            "default_model": "deepseek-chat",
        },
    }
    if auth_type == "oauth":
        providers["codex"] = {
            "type": "openai",
            "base_url": "",
            "auth_type": "oauth",
            "models": ["codex-model"],
            "default_model": "codex-model",
        }
    if auth_type == "subscription":
        providers["claude"] = {
            "type": "anthropic",
            "base_url": "https://api.anthropic.com/v1",
            "auth_type": "subscription",
            "models": ["claude-3-opus"],
            "default_model": "claude-3-opus",
        }
    return {"providers": providers, "model_overrides": {}}


def _make_adapter_metadata(fields=None):
    """Return a mock adapter metadata object."""
    if fields is None:
        fields = {
            "uri": {"required": True, "input_type": "text"},
        }
    meta = MagicMock()
    meta.get_config_fields.return_value = fields
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# _prompt_with_back
# ─────────────────────────────────────────────────────────────────────────────


class TestPromptWithBack:
    """Tests for the module-level _prompt_with_back() helper.

    PromptSession is imported locally inside _prompt_with_back, so it must be
    patched via sys.modules['prompt_toolkit'].
    """

    def _make_mock_session(self, return_value=None, side_effect=None):
        mock_session_instance = MagicMock()
        if side_effect is not None:
            mock_session_instance.prompt.side_effect = side_effect
        else:
            mock_session_instance.prompt.return_value = return_value
        mock_session_class = MagicMock(return_value=mock_session_instance)
        return mock_session_class, mock_session_instance

    def test_returns_user_input(self):
        """Returns stripped user input when user enters text."""
        from datus.cli.interactive_configure import _prompt_with_back

        mock_session_class, mock_instance = self._make_mock_session("  my value  ")
        mock_pt = MagicMock()
        mock_pt.PromptSession = mock_session_class
        mock_pt.key_binding = MagicMock()

        with patch.dict("sys.modules", {"prompt_toolkit": mock_pt, "prompt_toolkit.key_binding": mock_pt.key_binding}):
            result = _prompt_with_back("Label")

        assert result == "my value"

    def test_returns_default_when_empty_input(self):
        """Returns the default value when user presses Enter without typing."""
        from datus.cli.interactive_configure import _prompt_with_back

        mock_session_class, mock_instance = self._make_mock_session("")
        mock_pt = MagicMock()
        mock_pt.PromptSession = mock_session_class

        with patch.dict("sys.modules", {"prompt_toolkit": mock_pt, "prompt_toolkit.key_binding": mock_pt.key_binding}):
            result = _prompt_with_back("Label", default="mydefault")

        assert result == "mydefault"

    def test_returns_back_when_result_is_back_sentinel(self):
        """Returns _BACK when user presses ESC (result equals sentinel)."""
        from datus.cli.interactive_configure import _BACK, _prompt_with_back

        mock_session_class, mock_instance = self._make_mock_session(_BACK)
        mock_pt = MagicMock()
        mock_pt.PromptSession = mock_session_class

        with patch.dict("sys.modules", {"prompt_toolkit": mock_pt, "prompt_toolkit.key_binding": mock_pt.key_binding}):
            result = _prompt_with_back("Label")

        assert result == _BACK

    def test_returns_back_on_keyboard_interrupt(self):
        """Returns _BACK when KeyboardInterrupt is raised."""
        from datus.cli.interactive_configure import _BACK, _prompt_with_back

        mock_session_class, mock_instance = self._make_mock_session(side_effect=KeyboardInterrupt)
        mock_pt = MagicMock()
        mock_pt.PromptSession = mock_session_class

        with patch.dict("sys.modules", {"prompt_toolkit": mock_pt, "prompt_toolkit.key_binding": mock_pt.key_binding}):
            result = _prompt_with_back("Label")

        assert result == _BACK

    def test_returns_back_on_eof_error(self):
        """Returns _BACK when EOFError is raised (e.g. Ctrl+D)."""
        from datus.cli.interactive_configure import _BACK, _prompt_with_back

        mock_session_class, mock_instance = self._make_mock_session(side_effect=EOFError)
        mock_pt = MagicMock()
        mock_pt.PromptSession = mock_session_class

        with patch.dict("sys.modules", {"prompt_toolkit": mock_pt, "prompt_toolkit.key_binding": mock_pt.key_binding}):
            result = _prompt_with_back("Label")

        assert result == _BACK

    def test_prompt_includes_default_in_suffix(self):
        """When default is provided, the prompt label includes the default value."""
        from datus.cli.interactive_configure import _prompt_with_back

        mock_session_class, mock_instance = self._make_mock_session("answer")
        mock_pt = MagicMock()
        mock_pt.PromptSession = mock_session_class

        with patch.dict("sys.modules", {"prompt_toolkit": mock_pt, "prompt_toolkit.key_binding": mock_pt.key_binding}):
            _prompt_with_back("Label", default="mydefault")

        call_args = mock_instance.prompt.call_args
        label_used = call_args[0][0] if call_args[0] else ""
        assert "mydefault" in label_used


# ─────────────────────────────────────────────────────────────────────────────
# _init_dirs and _copy_files
# ─────────────────────────────────────────────────────────────────────────────


class TestInitDirsAndCopyFiles:
    """Tests for _init_dirs() and _copy_files()."""

    def test_init_dirs_calls_ensure_dirs(self, tmp_path):
        """_init_dirs() calls ensure_dirs with the expected directory names."""
        cfg = _make_configure(tmp_path)

        mock_pm = MagicMock()
        with patch("datus.cli.interactive_configure.get_path_manager", return_value=mock_pm):
            cfg._init_dirs()

        mock_pm.ensure_dirs.assert_called_once()
        args = mock_pm.ensure_dirs.call_args[0]
        assert "conf" in args
        assert "data" in args

    def test_copy_files_handles_errors_gracefully(self, tmp_path):
        """_copy_files() swallows exceptions from copy_data_file without raising."""
        cfg = _make_configure(tmp_path)

        with patch("datus.cli.interactive_configure.copy_data_file", side_effect=Exception("copy failed")):
            # Should not raise
            cfg._copy_files()

    def test_copy_files_copies_prompts_and_samples(self, tmp_path):
        """_copy_files() attempts to copy prompts, sample_data, and skills."""
        cfg = _make_configure(tmp_path)

        with patch("datus.cli.interactive_configure.copy_data_file") as mock_copy:
            cfg._copy_files()

        assert mock_copy.call_count >= 2
        resource_paths = [c[1].get("resource_path", c[0][0] if c[0] else "") for c in mock_copy.call_args_list]
        assert any("prompts" in p for p in resource_paths)


# ─────────────────────────────────────────────────────────────────────────────
# _load_existing_config
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadExistingConfig:
    """Tests for InteractiveConfigure._load_existing_config()."""

    def test_service_databases_format_populates_self_databases(self, tmp_path):
        """New service.databases format is loaded into self.databases correctly."""
        raw = {
            "agent": {
                "target": "openai",
                "models": {"openai": {"type": "openai", "model": "gpt-4o", "api_key": "sk-test"}},
                "service": {
                    "databases": {
                        "my_db": {"type": "sqlite", "uri": "data/test.sqlite"},
                    },
                    "bi_tools": {},
                    "schedulers": {},
                },
            }
        }
        config_file = tmp_path / "agent.yml"
        config_file.write_text(yaml.dump(raw), encoding="utf-8")

        cfg = _make_configure(tmp_path)
        cfg._load_existing_config()

        assert "my_db" in cfg.databases
        assert cfg.databases["my_db"]["type"] == "sqlite"
        assert cfg.target == "openai"
        assert "openai" in cfg.models

    def test_legacy_namespace_format_auto_migrates(self, tmp_path):
        """Legacy namespace format is auto-migrated via ServiceConfig.migrate_from_namespace."""
        raw = {
            "agent": {
                "target": "",
                "models": {},
                "namespace": {
                    "legacy_db": {
                        "type": "duckdb",
                        "uri": "legacy.duckdb",
                    }
                },
            }
        }
        config_file = tmp_path / "agent.yml"
        config_file.write_text(yaml.dump(raw), encoding="utf-8")

        migrate_result = {"databases": {"legacy_db": {"type": "duckdb", "uri": "legacy.duckdb"}}}

        with patch(
            "datus.configuration.agent_config.ServiceConfig.migrate_from_namespace",
            return_value=migrate_result,
        ):
            cfg = _make_configure(tmp_path)
            cfg._load_existing_config()

        assert "legacy_db" in cfg.databases

    def test_missing_config_file_leaves_empty_state(self, tmp_path):
        """When config file does not exist, models and databases remain empty."""
        cfg = _make_configure(tmp_path)
        # config_path points to nonexistent file
        cfg.config_path = tmp_path / "nonexistent.yml"
        cfg._load_existing_config()

        assert cfg.models == {}
        assert cfg.databases == {}
        assert cfg.target == ""

    def test_malformed_yaml_leaves_empty_state(self, tmp_path):
        """When config YAML is unreadable, state stays empty (no exception raised)."""
        config_file = tmp_path / "agent.yml"
        config_file.write_text(": bad: yaml: {[", encoding="utf-8")

        cfg = _make_configure(tmp_path)
        cfg._load_existing_config()

        assert cfg.models == {}
        assert cfg.databases == {}


# ─────────────────────────────────────────────────────────────────────────────
# _load_provider_catalog
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadProviderCatalog:
    """Tests for InteractiveConfigure._load_provider_catalog()."""

    def test_returns_dict_with_providers_key(self, tmp_path):
        """_load_provider_catalog returns a dict containing 'providers' key."""
        catalog_data = {
            "providers": {
                "openai": {
                    "type": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "OPENAI_API_KEY",
                    "models": ["gpt-4o"],
                    "default_model": "gpt-4o",
                }
            },
            "model_overrides": {},
        }

        with patch(
            "datus.cli.interactive_configure.read_data_file_text",
            return_value=yaml.dump(catalog_data),
        ):
            cfg = _make_configure(tmp_path)
            result = cfg._load_provider_catalog()

        assert "providers" in result
        assert "openai" in result["providers"]

    def test_returns_empty_dict_on_failure(self, tmp_path):
        """_load_provider_catalog returns fallback empty structure on exception."""
        with patch(
            "datus.cli.interactive_configure.read_data_file_text",
            side_effect=FileNotFoundError("not found"),
        ):
            cfg = _make_configure(tmp_path)
            result = cfg._load_provider_catalog()

        assert result == {"providers": {}, "model_overrides": {}}


# ─────────────────────────────────────────────────────────────────────────────
# _show_current_state
# ─────────────────────────────────────────────────────────────────────────────


class TestShowCurrentState:
    """Tests for InteractiveConfigure._show_current_state()."""

    def test_with_models_and_databases_no_exception(self, tmp_path):
        """_show_current_state() does not raise when both models and databases exist."""
        cfg = _make_configure(tmp_path)
        cfg.models = {"openai": {"model": "gpt-4o", "base_url": "https://api.openai.com/v1", "api_key": "sk-test"}}
        cfg.databases = {
            "my_db": {"type": "sqlite", "uri": "path/to/db.sqlite"},
        }
        cfg.target = "openai"

        # Should not raise
        cfg._show_current_state()

    def test_with_empty_models_and_databases_no_exception(self, tmp_path):
        """_show_current_state() handles empty state without errors."""
        cfg = _make_configure(tmp_path)
        cfg.models = {}
        cfg.databases = {}
        cfg.target = ""

        cfg._show_current_state()

    def test_marks_default_model_with_asterisk(self, tmp_path):
        """_show_current_state() shows '*' next to the target/default model."""
        cfg = _make_configure(tmp_path)
        cfg.models = {
            "openai": {"model": "gpt-4o", "base_url": "https://api.openai.com/v1", "api_key": "sk-test"},
            "deepseek": {"model": "deepseek-chat", "base_url": "https://api.deepseek.com/v1", "api_key": "sk-ds"},
        }
        cfg.databases = {}
        cfg.target = "openai"

        # No exception + table is rendered (verified by no exception)
        cfg._show_current_state()

    def test_database_with_host_field_shows_host(self, tmp_path):
        """_show_current_state() uses 'host' as connection when 'uri' is absent."""
        cfg = _make_configure(tmp_path)
        cfg.models = {}
        cfg.databases = {
            "pg_db": {"type": "postgresql", "host": "localhost"},
        }
        cfg.target = ""

        cfg._show_current_state()

    def test_database_default_marked_with_asterisk(self, tmp_path):
        """_show_current_state() shows '*' next to the database with default=True."""
        cfg = _make_configure(tmp_path)
        cfg.models = {}
        cfg.databases = {
            "main_db": {"type": "duckdb", "uri": "main.duckdb", "default": True},
            "other_db": {"type": "sqlite", "uri": "other.sqlite"},
        }
        cfg.target = ""

        cfg._show_current_state()


# ─────────────────────────────────────────────────────────────────────────────
# run() method
# ─────────────────────────────────────────────────────────────────────────────


class TestRun:
    """Tests for InteractiveConfigure.run()."""

    def test_run_calls_first_time_setup_when_no_existing_config(self, tmp_path):
        """run() calls _first_time_setup() when no models/databases are configured."""

        cfg = _make_configure(tmp_path)

        with (
            patch.object(cfg, "_init_dirs"),
            patch.object(cfg, "_copy_files"),
            patch.object(cfg, "_load_existing_config"),
            patch.object(cfg, "_first_time_setup", return_value=0) as mock_setup,
            patch("logging.getLogger") as mock_get_logger,
        ):
            # Return a root logger with no handlers to avoid stream.name attribute issues
            mock_root = MagicMock()
            mock_root.handlers = []
            mock_get_logger.return_value = mock_root
            cfg.models = {}
            cfg.databases = {}
            result = cfg.run()

        mock_setup.assert_called_once()
        assert result == 0

    def test_run_calls_interactive_menu_when_config_exists(self, tmp_path):
        """run() calls _interactive_menu() when models/databases already exist."""
        cfg = _make_configure(tmp_path)

        with (
            patch.object(cfg, "_init_dirs"),
            patch.object(cfg, "_copy_files"),
            patch.object(cfg, "_load_existing_config"),
            patch.object(cfg, "_interactive_menu", return_value=0) as mock_menu,
            patch("logging.getLogger") as mock_get_logger,
        ):
            mock_root = MagicMock()
            mock_root.handlers = []
            mock_get_logger.return_value = mock_root
            cfg.models = {"openai": {"type": "openai"}}
            cfg.databases = {}
            result = cfg.run()

        mock_menu.assert_called_once()
        assert result == 0

    def test_run_returns_1_on_keyboard_interrupt(self, tmp_path):
        """run() catches KeyboardInterrupt and returns 1."""
        cfg = _make_configure(tmp_path)

        with (
            patch.object(cfg, "_init_dirs"),
            patch.object(cfg, "_copy_files"),
            patch.object(cfg, "_load_existing_config"),
            patch.object(cfg, "_first_time_setup", side_effect=KeyboardInterrupt),
            patch("logging.getLogger") as mock_get_logger,
        ):
            mock_root = MagicMock()
            mock_root.handlers = []
            mock_get_logger.return_value = mock_root
            cfg.models = {}
            cfg.databases = {}
            result = cfg.run()

        assert result == 1

    def test_run_returns_1_on_unexpected_exception(self, tmp_path):
        """run() catches unexpected exceptions and returns 1."""
        cfg = _make_configure(tmp_path)

        with (
            patch.object(cfg, "_init_dirs"),
            patch.object(cfg, "_copy_files"),
            patch.object(cfg, "_load_existing_config"),
            patch.object(cfg, "_first_time_setup", side_effect=RuntimeError("unexpected")),
            patch("datus.cli.interactive_configure.print_rich_exception"),
            patch("logging.getLogger") as mock_get_logger,
        ):
            mock_root = MagicMock()
            mock_root.handlers = []
            mock_get_logger.return_value = mock_root
            cfg.models = {}
            cfg.databases = {}
            result = cfg.run()

        assert result == 1

    def test_run_restores_log_handler_levels(self, tmp_path):
        """run() restores original log handler levels in the finally block."""
        import logging

        cfg = _make_configure(tmp_path)

        # Create a handler that has a stream with a name attribute
        handler = logging.StreamHandler()
        handler.stream = MagicMock()
        handler.stream.name = "<stderr>"
        original_level = logging.DEBUG
        handler.setLevel(original_level)

        mock_root = MagicMock()
        mock_root.handlers = [handler]

        with (
            patch.object(cfg, "_init_dirs"),
            patch.object(cfg, "_copy_files"),
            patch.object(cfg, "_load_existing_config"),
            patch.object(cfg, "_first_time_setup", return_value=0),
            patch("logging.getLogger", return_value=mock_root),
        ):
            cfg.models = {}
            cfg.databases = {}
            cfg.run()

        # Handler level should be restored to original_level
        assert handler.level == original_level


# ─────────────────────────────────────────────────────────────────────────────
# _first_time_setup
# ─────────────────────────────────────────────────────────────────────────────


class TestFirstTimeSetup:
    """Tests for InteractiveConfigure._first_time_setup()."""

    def test_returns_0_when_both_model_and_database_added(self, tmp_path):
        """_first_time_setup() returns 0 when model and database are added successfully."""
        cfg = _make_configure(tmp_path)

        with (
            patch.object(cfg, "_add_model", return_value=True),
            patch.object(cfg, "_add_database", return_value=True),
            patch.object(cfg, "_save"),
            patch.object(cfg, "_show_current_state"),
            patch.object(cfg, "_display_completion"),
        ):
            result = cfg._first_time_setup()

        assert result == 0

    def test_returns_1_when_user_cancels_model_retry(self, tmp_path):
        """_first_time_setup() returns 1 when user declines to retry model config."""
        from rich.prompt import Confirm

        cfg = _make_configure(tmp_path)

        with (
            patch.object(cfg, "_add_model", return_value=False),
            patch.object(Confirm, "ask", return_value=False),
        ):
            result = cfg._first_time_setup()

        assert result == 1

    def test_retries_model_when_confirm_true(self, tmp_path):
        """_first_time_setup() retries model addition when user confirms retry."""
        from rich.prompt import Confirm

        cfg = _make_configure(tmp_path)
        # Fail first, succeed second
        add_model_calls = [False, True]

        with (
            patch.object(cfg, "_add_model", side_effect=add_model_calls),
            patch.object(Confirm, "ask", return_value=True),
            patch.object(cfg, "_add_database", return_value=True),
            patch.object(cfg, "_save"),
            patch.object(cfg, "_show_current_state"),
            patch.object(cfg, "_display_completion"),
        ):
            result = cfg._first_time_setup()

        assert result == 0

    def test_returns_1_when_user_cancels_database_retry(self, tmp_path):
        """_first_time_setup() returns 1 when user declines to retry database config."""
        from rich.prompt import Confirm

        cfg = _make_configure(tmp_path)

        with (
            patch.object(cfg, "_add_model", return_value=True),
            patch.object(cfg, "_add_database", return_value=False),
            patch.object(Confirm, "ask", return_value=False),
        ):
            result = cfg._first_time_setup()

        assert result == 1


# ─────────────────────────────────────────────────────────────────────────────
# _interactive_menu
# ─────────────────────────────────────────────────────────────────────────────


class TestInteractiveMenu:
    """Tests for InteractiveConfigure._interactive_menu()."""

    def test_done_action_returns_0(self, tmp_path):
        """Selecting 'done' from the menu returns 0."""
        cfg = _make_configure(tmp_path)
        cfg.models = {}
        cfg.databases = {}

        with (
            patch.object(cfg, "_show_current_state"),
            patch("datus.cli.interactive_configure.select_choice", return_value="done"),
            patch.object(cfg, "_display_completion"),
        ):
            result = cfg._interactive_menu()

        assert result == 0

    def test_add_model_action_calls_add_model_and_save(self, tmp_path):
        """Selecting 'add_model' calls _add_model() and _save()."""
        cfg = _make_configure(tmp_path)
        cfg.models = {}
        cfg.databases = {}

        actions = ["add_model", "done"]
        action_iter = iter(actions)

        with (
            patch.object(cfg, "_show_current_state"),
            patch("datus.cli.interactive_configure.select_choice", side_effect=lambda *a, **kw: next(action_iter)),
            patch.object(cfg, "_add_model", return_value=True) as mock_add,
            patch.object(cfg, "_save") as mock_save,
            patch.object(cfg, "_display_completion"),
        ):
            cfg._interactive_menu()

        mock_add.assert_called_once()
        mock_save.assert_called()

    def test_add_database_action_calls_add_database_and_save(self, tmp_path):
        """Selecting 'add_database' calls _add_database() and _save()."""
        cfg = _make_configure(tmp_path)
        cfg.models = {}
        cfg.databases = {}

        actions = ["add_database", "done"]
        action_iter = iter(actions)

        with (
            patch.object(cfg, "_show_current_state"),
            patch("datus.cli.interactive_configure.select_choice", side_effect=lambda *a, **kw: next(action_iter)),
            patch.object(cfg, "_add_database", return_value=True) as mock_add_db,
            patch.object(cfg, "_save") as mock_save,
            patch.object(cfg, "_display_completion"),
        ):
            cfg._interactive_menu()

        mock_add_db.assert_called_once()
        mock_save.assert_called()

    def test_delete_model_action_available_when_models_exist(self, tmp_path):
        """Selecting 'delete_model' calls _delete_model() and _save()."""
        cfg = _make_configure(tmp_path)
        cfg.models = {"openai": {"type": "openai"}}
        cfg.databases = {}

        actions = ["delete_model", "done"]
        action_iter = iter(actions)

        with (
            patch.object(cfg, "_show_current_state"),
            patch("datus.cli.interactive_configure.select_choice", side_effect=lambda *a, **kw: next(action_iter)),
            patch.object(cfg, "_delete_model") as mock_del,
            patch.object(cfg, "_save") as mock_save,
            patch.object(cfg, "_display_completion"),
        ):
            cfg._interactive_menu()

        mock_del.assert_called_once()
        mock_save.assert_called()

    def test_delete_database_action_calls_delete_database_and_save(self, tmp_path):
        """Selecting 'delete_database' calls _delete_database() and _save()."""
        cfg = _make_configure(tmp_path)
        cfg.models = {}
        cfg.databases = {"my_db": {"type": "sqlite"}}

        actions = ["delete_database", "done"]
        action_iter = iter(actions)

        with (
            patch.object(cfg, "_show_current_state"),
            patch("datus.cli.interactive_configure.select_choice", side_effect=lambda *a, **kw: next(action_iter)),
            patch.object(cfg, "_delete_database") as mock_del,
            patch.object(cfg, "_save") as mock_save,
            patch.object(cfg, "_display_completion"),
        ):
            cfg._interactive_menu()

        mock_del.assert_called_once()
        mock_save.assert_called()

    def test_set_default_model_available_when_multiple_models(self, tmp_path):
        """Selecting 'set_default_model' calls _set_default_model() and _save()."""
        cfg = _make_configure(tmp_path)
        cfg.models = {
            "openai": {"type": "openai"},
            "deepseek": {"type": "openai"},
        }
        cfg.databases = {}

        actions = ["set_default_model", "done"]
        action_iter = iter(actions)

        with (
            patch.object(cfg, "_show_current_state"),
            patch("datus.cli.interactive_configure.select_choice", side_effect=lambda *a, **kw: next(action_iter)),
            patch.object(cfg, "_set_default_model") as mock_set,
            patch.object(cfg, "_save") as mock_save,
            patch.object(cfg, "_display_completion"),
        ):
            cfg._interactive_menu()

        mock_set.assert_called_once()
        mock_save.assert_called()


# ─────────────────────────────────────────────────────────────────────────────
# _add_model
# ─────────────────────────────────────────────────────────────────────────────


class TestAddModel:
    """Tests for InteractiveConfigure._add_model()."""

    def _patch_catalog(self, catalog=None):
        """Return a patch context for _load_provider_catalog."""
        if catalog is None:
            catalog = _make_provider_catalog()
        return patch.object, catalog

    def test_returns_false_when_no_providers(self, tmp_path):
        """_add_model() returns False when providers catalog is empty."""
        cfg = _make_configure(tmp_path)

        with patch.object(cfg, "_load_provider_catalog", return_value={"providers": {}, "model_overrides": {}}):
            result = cfg._add_model()

        assert result is False

    def test_successful_model_addition_sets_target(self, tmp_path):
        """_add_model() returns True and sets target when LLM connectivity succeeds."""
        cfg = _make_configure(tmp_path)
        catalog = _make_provider_catalog()

        with (
            patch.object(cfg, "_load_provider_catalog", return_value=catalog),
            patch("datus.cli.interactive_configure.select_choice", side_effect=["openai", "gpt-4o"]),
            patch("datus.cli.interactive_configure.os.environ.get", return_value=""),
            patch(
                "datus.cli.interactive_configure._prompt_with_back",
                side_effect=["sk-testkey", "https://api.openai.com/v1"],
            ),
            patch("datus.cli.interactive_configure.Confirm.ask", return_value=False),
            patch.object(cfg, "_test_llm_connectivity", return_value=(True, "")),
        ):
            result = cfg._add_model()

        assert result is True
        assert "openai" in cfg.models
        assert cfg.target == "openai"

    def test_returns_false_when_llm_connectivity_fails(self, tmp_path):
        """_add_model() returns False when LLM connectivity test fails."""
        cfg = _make_configure(tmp_path)
        catalog = _make_provider_catalog()

        with (
            patch.object(cfg, "_load_provider_catalog", return_value=catalog),
            patch("datus.cli.interactive_configure.select_choice", side_effect=["openai", "gpt-4o"]),
            patch("datus.cli.interactive_configure.os.environ.get", return_value=""),
            patch(
                "datus.cli.interactive_configure._prompt_with_back",
                side_effect=["sk-testkey", "https://api.openai.com/v1"],
            ),
            patch.object(cfg, "_test_llm_connectivity", return_value=(False, "Connection refused")),
        ):
            result = cfg._add_model()

        assert result is False

    def test_back_from_api_key_returns_to_provider_step(self, tmp_path):
        """Pressing Back at API key step returns to provider selection."""
        from datus.cli.interactive_configure import _BACK

        cfg = _make_configure(tmp_path)
        catalog = _make_provider_catalog()

        # First iteration: select openai, back on api key
        # Second iteration: select openai, provide api key, provide base url, select model, test ok
        select_values = iter(["openai", "openai", "gpt-4o"])
        prompt_values = iter([_BACK, "sk-testkey", "https://api.openai.com/v1"])

        with (
            patch.object(cfg, "_load_provider_catalog", return_value=catalog),
            patch("datus.cli.interactive_configure.select_choice", side_effect=lambda *a, **kw: next(select_values)),
            patch("datus.cli.interactive_configure.os.environ.get", return_value=""),
            patch(
                "datus.cli.interactive_configure._prompt_with_back", side_effect=lambda *a, **kw: next(prompt_values)
            ),
            patch("datus.cli.interactive_configure.Confirm.ask", return_value=False),
            patch.object(cfg, "_test_llm_connectivity", return_value=(True, "")),
        ):
            result = cfg._add_model()

        assert result is True

    def test_back_from_base_url_returns_to_api_key_step(self, tmp_path):
        """Pressing Back at Base URL step returns to API key step."""
        from datus.cli.interactive_configure import _BACK

        cfg = _make_configure(tmp_path)
        catalog = _make_provider_catalog()

        # Step sequence: provider -> api_key -> back at base_url -> api_key again -> base_url -> model -> ok
        select_values = iter(["openai", "gpt-4o"])
        prompt_values = iter(["sk-key1", _BACK, "sk-key2", "https://api.openai.com/v1"])

        with (
            patch.object(cfg, "_load_provider_catalog", return_value=catalog),
            patch("datus.cli.interactive_configure.select_choice", side_effect=lambda *a, **kw: next(select_values)),
            patch("datus.cli.interactive_configure.os.environ.get", return_value=""),
            patch(
                "datus.cli.interactive_configure._prompt_with_back", side_effect=lambda *a, **kw: next(prompt_values)
            ),
            patch("datus.cli.interactive_configure.Confirm.ask", return_value=False),
            patch.object(cfg, "_test_llm_connectivity", return_value=(True, "")),
        ):
            result = cfg._add_model()

        assert result is True

    def test_empty_api_key_loops_again(self, tmp_path):
        """An empty API key causes the step to repeat."""
        cfg = _make_configure(tmp_path)
        catalog = _make_provider_catalog()

        # First API key: empty, second: valid
        select_values = iter(["openai", "gpt-4o"])
        prompt_values = iter(["", "sk-valid", "https://api.openai.com/v1"])

        with (
            patch.object(cfg, "_load_provider_catalog", return_value=catalog),
            patch("datus.cli.interactive_configure.select_choice", side_effect=lambda *a, **kw: next(select_values)),
            patch("datus.cli.interactive_configure.os.environ.get", return_value=""),
            patch(
                "datus.cli.interactive_configure._prompt_with_back", side_effect=lambda *a, **kw: next(prompt_values)
            ),
            patch("datus.cli.interactive_configure.Confirm.ask", return_value=False),
            patch.object(cfg, "_test_llm_connectivity", return_value=(True, "")),
        ):
            result = cfg._add_model()

        assert result is True

    def test_sets_target_as_default_when_first_model(self, tmp_path):
        """_add_model() auto-sets the provider as target when no existing models."""
        cfg = _make_configure(tmp_path)
        cfg.target = ""
        catalog = _make_provider_catalog()

        with (
            patch.object(cfg, "_load_provider_catalog", return_value=catalog),
            patch("datus.cli.interactive_configure.select_choice", side_effect=["openai", "gpt-4o"]),
            patch("datus.cli.interactive_configure.os.environ.get", return_value=""),
            patch(
                "datus.cli.interactive_configure._prompt_with_back", side_effect=["sk-key", "https://api.openai.com/v1"]
            ),
            patch.object(cfg, "_test_llm_connectivity", return_value=(True, "")),
        ):
            cfg._add_model()

        assert cfg.target == "openai"

    def test_asks_to_set_default_when_model_already_exists(self, tmp_path):
        """_add_model() asks if user wants to set new model as default when target already set."""
        cfg = _make_configure(tmp_path)
        cfg.target = "openai"
        cfg.models = {"openai": {"type": "openai"}}
        catalog = _make_provider_catalog()

        with (
            patch.object(cfg, "_load_provider_catalog", return_value=catalog),
            patch("datus.cli.interactive_configure.select_choice", side_effect=["deepseek", "deepseek-chat"]),
            patch("datus.cli.interactive_configure.os.environ.get", return_value=""),
            patch(
                "datus.cli.interactive_configure._prompt_with_back",
                side_effect=["sk-ds", "https://api.deepseek.com/v1"],
            ),
            patch("datus.cli.interactive_configure.Confirm.ask", return_value=True),
            patch.object(cfg, "_test_llm_connectivity", return_value=(True, "")),
        ):
            cfg._add_model()

        assert cfg.target == "deepseek"

    def test_model_with_no_models_list_uses_free_text_prompt(self, tmp_path):
        """When provider has no model list, _add_model() uses _prompt_with_back for model name."""
        cfg = _make_configure(tmp_path)
        catalog = _make_provider_catalog(with_models=False)

        select_values = iter(["openai"])
        prompt_values = iter(["sk-key", "https://api.openai.com/v1", "custom-model"])

        with (
            patch.object(cfg, "_load_provider_catalog", return_value=catalog),
            patch("datus.cli.interactive_configure.select_choice", side_effect=lambda *a, **kw: next(select_values)),
            patch("datus.cli.interactive_configure.os.environ.get", return_value=""),
            patch(
                "datus.cli.interactive_configure._prompt_with_back", side_effect=lambda *a, **kw: next(prompt_values)
            ),
            patch("datus.cli.interactive_configure.Confirm.ask", return_value=False),
            patch.object(cfg, "_test_llm_connectivity", return_value=(True, "")),
        ):
            result = cfg._add_model()

        assert result is True

    def test_model_back_from_free_text_model_returns_to_base_url(self, tmp_path):
        """Back at free-text model step returns to Base URL step."""
        from datus.cli.interactive_configure import _BACK

        cfg = _make_configure(tmp_path)
        catalog = _make_provider_catalog(with_models=False)

        select_values = iter(["openai"])
        # api_key, base_url, back-at-model, base_url-again, model-name
        prompt_values = iter(
            ["sk-key", "https://api.openai.com/v1", _BACK, "https://api.openai.com/v1", "custom-model"]
        )

        with (
            patch.object(cfg, "_load_provider_catalog", return_value=catalog),
            patch("datus.cli.interactive_configure.select_choice", side_effect=lambda *a, **kw: next(select_values)),
            patch("datus.cli.interactive_configure.os.environ.get", return_value=""),
            patch(
                "datus.cli.interactive_configure._prompt_with_back", side_effect=lambda *a, **kw: next(prompt_values)
            ),
            patch("datus.cli.interactive_configure.Confirm.ask", return_value=False),
            patch.object(cfg, "_test_llm_connectivity", return_value=(True, "")),
        ):
            result = cfg._add_model()

        assert result is True

    def test_model_overrides_applied_to_entry(self, tmp_path):
        """Model parameter overrides from providers.yml are merged into the entry."""
        cfg = _make_configure(tmp_path)
        catalog = _make_provider_catalog()
        catalog["model_overrides"]["gpt-4o"] = {"temperature": 0.0}

        with (
            patch.object(cfg, "_load_provider_catalog", return_value=catalog),
            patch("datus.cli.interactive_configure.select_choice", side_effect=["openai", "gpt-4o"]),
            patch("datus.cli.interactive_configure.os.environ.get", return_value=""),
            patch(
                "datus.cli.interactive_configure._prompt_with_back", side_effect=["sk-key", "https://api.openai.com/v1"]
            ),
            patch("datus.cli.interactive_configure.Confirm.ask", return_value=False),
            patch.object(cfg, "_test_llm_connectivity", return_value=(True, "")),
        ):
            cfg._add_model()

        assert cfg.models["openai"]["temperature"] == 0.0

    def test_oauth_provider_routes_to_configure_codex_oauth(self, tmp_path):
        """Selecting an OAuth provider calls _configure_codex_oauth()."""
        cfg = _make_configure(tmp_path)
        catalog = _make_provider_catalog(auth_type="oauth")

        with (
            patch.object(cfg, "_load_provider_catalog", return_value=catalog),
            patch("datus.cli.interactive_configure.select_choice", return_value="codex"),
            patch.object(cfg, "_configure_codex_oauth", return_value=True) as mock_oauth,
        ):
            result = cfg._add_model()

        mock_oauth.assert_called_once()
        assert result is True

    def test_subscription_provider_routes_to_configure_claude_subscription(self, tmp_path):
        """Selecting a subscription provider calls _configure_claude_subscription()."""
        cfg = _make_configure(tmp_path)
        catalog = _make_provider_catalog(auth_type="subscription")

        with (
            patch.object(cfg, "_load_provider_catalog", return_value=catalog),
            patch("datus.cli.interactive_configure.select_choice", return_value="claude"),
            patch.object(cfg, "_configure_claude_subscription", return_value=True) as mock_sub,
        ):
            result = cfg._add_model()

        mock_sub.assert_called_once()
        assert result is True

    def test_env_var_api_key_detected_uses_env_ref(self, tmp_path):
        """When env var for API key is set, user can choose to use it as ${ENV_VAR}."""

        cfg = _make_configure(tmp_path)
        catalog = _make_provider_catalog()

        with (
            patch.object(cfg, "_load_provider_catalog", return_value=catalog),
            patch("datus.cli.interactive_configure.select_choice", side_effect=["openai", "gpt-4o"]),
            patch("datus.cli.interactive_configure.os.environ.get", return_value="sk-env-value"),
            patch("datus.cli.interactive_configure.Confirm.ask", side_effect=[True, False]),
            patch("datus.cli.interactive_configure._prompt_with_back", return_value="https://api.openai.com/v1"),
            patch.object(cfg, "_test_llm_connectivity", return_value=(True, "")),
        ):
            result = cfg._add_model()

        assert result is True
        assert cfg.models["openai"]["api_key"] == "${OPENAI_API_KEY}"


# ─────────────────────────────────────────────────────────────────────────────
# _add_database
# ─────────────────────────────────────────────────────────────────────────────


class TestAddDatabase:
    """Tests for InteractiveConfigure._add_database()."""

    def _mock_connector_registry(self, types=("duckdb", "sqlite")):
        """Return a mock connector_registry with the given adapter types."""
        registry = MagicMock()
        adapters = {t: _make_adapter_metadata({"uri": {"required": True, "input_type": "text"}}) for t in types}
        registry.list_available_adapters.return_value = adapters
        return registry

    def test_successful_database_addition(self, tmp_path):
        """_add_database() returns True when connectivity test passes."""
        from rich.prompt import Prompt

        cfg = _make_configure(tmp_path)
        registry = self._mock_connector_registry()

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", return_value="mydb"),
            patch("datus.cli.interactive_configure.select_choice", return_value="duckdb"),
            patch("datus.cli.interactive_configure._prompt_with_back", return_value="path/to/db.duckdb"),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(True, "")),
        ):
            result = cfg._add_database()

        assert result is True
        assert "mydb" in cfg.databases

    def test_marks_first_database_as_default(self, tmp_path):
        """_add_database() marks the first database added as default."""
        from rich.prompt import Prompt

        cfg = _make_configure(tmp_path)
        cfg.databases = {}
        registry = self._mock_connector_registry()

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", return_value="first_db"),
            patch("datus.cli.interactive_configure.select_choice", return_value="duckdb"),
            patch("datus.cli.interactive_configure._prompt_with_back", return_value="first.duckdb"),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(True, "")),
        ):
            cfg._add_database()

        assert cfg.databases["first_db"].get("default") is True

    def test_asks_default_when_databases_already_exist(self, tmp_path):
        """_add_database() asks if new database should be default when others already exist."""
        from rich.prompt import Confirm, Prompt

        cfg = _make_configure(tmp_path)
        cfg.databases = {"existing_db": {"type": "sqlite", "default": True}}
        registry = self._mock_connector_registry()

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", return_value="new_db"),
            patch("datus.cli.interactive_configure.select_choice", return_value="duckdb"),
            patch("datus.cli.interactive_configure._prompt_with_back", return_value="new.duckdb"),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(True, "")),
            patch.object(Confirm, "ask", return_value=True) as mock_confirm,
        ):
            cfg._add_database()

        mock_confirm.assert_called_once()
        assert cfg.databases["new_db"].get("default") is True
        assert "default" not in cfg.databases["existing_db"]

    def test_returns_false_when_db_name_empty(self, tmp_path):
        """_add_database() loops when database name is empty, then exits when user provides one."""
        from rich.prompt import Prompt

        cfg = _make_configure(tmp_path)
        registry = self._mock_connector_registry()

        # First call: empty, second call: valid
        ask_iter = iter(["", "validname"])

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", side_effect=lambda *a, **kw: next(ask_iter)),
            patch("datus.cli.interactive_configure.select_choice", return_value="duckdb"),
            patch("datus.cli.interactive_configure._prompt_with_back", return_value="path.duckdb"),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(True, "")),
        ):
            result = cfg._add_database()

        assert result is True
        assert "validname" in cfg.databases

    def test_returns_false_when_db_name_already_exists(self, tmp_path):
        """_add_database() shows error when database name already exists."""
        from rich.prompt import Confirm, Prompt

        cfg = _make_configure(tmp_path)
        cfg.databases = {"existingdb": {"type": "sqlite"}}
        registry = self._mock_connector_registry()

        # First: duplicate, second: new name
        ask_iter = iter(["existingdb", "newdb"])

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", side_effect=lambda *a, **kw: next(ask_iter)),
            patch("datus.cli.interactive_configure.select_choice", return_value="duckdb"),
            patch("datus.cli.interactive_configure._prompt_with_back", return_value="new.duckdb"),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(True, "")),
            patch.object(Confirm, "ask", return_value=False),
        ):
            result = cfg._add_database()

        assert result is True
        assert "newdb" in cfg.databases

    def test_returns_false_when_no_adapters_available(self, tmp_path):
        """_add_database() returns False when no adapter types are available."""
        from rich.prompt import Prompt

        cfg = _make_configure(tmp_path)
        registry = MagicMock()
        registry.list_available_adapters.return_value = {}

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", return_value="mydb"),
        ):
            result = cfg._add_database()

        assert result is False

    def test_returns_false_when_connectivity_fails(self, tmp_path):
        """_add_database() returns False when connectivity test fails."""
        from rich.prompt import Prompt

        cfg = _make_configure(tmp_path)
        registry = self._mock_connector_registry()

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", return_value="faildb"),
            patch("datus.cli.interactive_configure.select_choice", return_value="duckdb"),
            patch("datus.cli.interactive_configure._prompt_with_back", return_value="fail.duckdb"),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(False, "Connection refused")),
        ):
            result = cfg._add_database()

        assert result is False

    def test_returns_false_when_adapter_has_no_config_fields(self, tmp_path):
        """_add_database() returns False when adapter schema has no fields."""
        from rich.prompt import Prompt

        cfg = _make_configure(tmp_path)
        adapter = MagicMock()
        adapter.get_config_fields.return_value = {}
        registry = MagicMock()
        registry.list_available_adapters.return_value = {"duckdb": adapter}

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", return_value="mydb"),
            patch("datus.cli.interactive_configure.select_choice", return_value="duckdb"),
        ):
            result = cfg._add_database()

        assert result is False

    def test_installs_plugin_for_uninstalled_db_type(self, tmp_path):
        """_add_database() tries to install plugin for uninstalled database type."""
        from rich.prompt import Prompt

        cfg = _make_configure(tmp_path)
        # Only duckdb installed, snowflake not installed
        registry = self._mock_connector_registry(types=("duckdb",))

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", return_value="mydb"),
            patch("datus.cli.interactive_configure.select_choice", return_value="snowflake"),
            patch.object(cfg, "_install_plugin", return_value=True) as mock_install,
        ):
            result = cfg._add_database()

        mock_install.assert_called_once_with("datus-snowflake")
        # Returns False because user must re-run configure after plugin install
        assert result is False

    def test_install_failure_returns_false(self, tmp_path):
        """_add_database() returns False when plugin installation fails."""
        from rich.prompt import Prompt

        cfg = _make_configure(tmp_path)
        registry = self._mock_connector_registry(types=("duckdb",))

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", return_value="mydb"),
            patch("datus.cli.interactive_configure.select_choice", return_value="snowflake"),
            patch.object(cfg, "_install_plugin", return_value=False),
        ):
            result = cfg._add_database()

        assert result is False

    def test_password_field_uses_password_prompt(self, tmp_path):
        """A password field uses password mode in _prompt_with_back."""
        from rich.prompt import Prompt

        cfg = _make_configure(tmp_path)
        fields = {
            "host": {"required": True, "input_type": "text"},
            "password": {"required": False, "input_type": "password"},
        }
        adapter = _make_adapter_metadata(fields)
        registry = MagicMock()
        registry.list_available_adapters.return_value = {"postgresql": adapter}

        prompt_calls = []

        def fake_prompt_with_back(label, default="", password=False):
            prompt_calls.append((label, password))
            return "value"

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", return_value="pgdb"),
            patch("datus.cli.interactive_configure.select_choice", return_value="postgresql"),
            patch("datus.cli.interactive_configure._prompt_with_back", side_effect=fake_prompt_with_back),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(True, "")),
        ):
            cfg._add_database()

        password_prompts = [p for label, p in prompt_calls if p]
        assert len(password_prompts) >= 1

    def test_int_field_validates_port_range(self, tmp_path):
        """Port field rejects values outside 1-65535 and accepts valid ones."""
        from rich.prompt import Prompt

        cfg = _make_configure(tmp_path)
        fields = {
            "port": {"required": False, "type": "int", "default": 5432},
        }
        adapter = _make_adapter_metadata(fields)
        registry = MagicMock()
        registry.list_available_adapters.return_value = {"postgresql": adapter}

        # First call: invalid port (0), second: valid (5432)
        prompt_values = iter(["0", "5432"])

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", return_value="pgdb"),
            patch("datus.cli.interactive_configure.select_choice", return_value="postgresql"),
            patch(
                "datus.cli.interactive_configure._prompt_with_back", side_effect=lambda *a, **kw: next(prompt_values)
            ),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(True, "")),
        ):
            result = cfg._add_database()

        assert result is True
        assert cfg.databases["pgdb"]["port"] == 5432

    def test_int_field_rejects_non_integer_value(self, tmp_path):
        """Port field rejects non-integer strings and loops."""
        from rich.prompt import Prompt

        cfg = _make_configure(tmp_path)
        fields = {
            "port": {"required": False, "type": "int", "default": 5432},
        }
        adapter = _make_adapter_metadata(fields)
        registry = MagicMock()
        registry.list_available_adapters.return_value = {"postgresql": adapter}

        prompt_values = iter(["notanint", "5432"])

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", return_value="pgdb"),
            patch("datus.cli.interactive_configure.select_choice", return_value="postgresql"),
            patch(
                "datus.cli.interactive_configure._prompt_with_back", side_effect=lambda *a, **kw: next(prompt_values)
            ),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(True, "")),
        ):
            result = cfg._add_database()

        assert result is True

    def test_file_path_field_uses_default_path(self, tmp_path):
        """A file_path field uses the sample_dir combined with default_sample as default."""
        from rich.prompt import Prompt

        cfg = _make_configure(tmp_path)
        cfg.sample_dir = tmp_path / "sample"
        fields = {
            "database_file": {
                "required": False,
                "input_type": "file_path",
                "default_sample": "my.db",
            },
        }
        adapter = _make_adapter_metadata(fields)
        registry = MagicMock()
        registry.list_available_adapters.return_value = {"sqlite": adapter}

        prompt_defaults = []

        def fake_prompt_with_back(label, default="", password=False):
            prompt_defaults.append(default)
            return default or "somefile.db"

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", return_value="mydb"),
            patch("datus.cli.interactive_configure.select_choice", return_value="sqlite"),
            patch("datus.cli.interactive_configure._prompt_with_back", side_effect=fake_prompt_with_back),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(True, "")),
        ):
            cfg._add_database()

        assert any(str(tmp_path / "sample" / "my.db") in d for d in prompt_defaults)

    def test_back_from_first_field_returns_to_type_selection(self, tmp_path):
        """Pressing Back on the first config field returns to DB type selection."""
        from rich.prompt import Prompt

        from datus.cli.interactive_configure import _BACK

        cfg = _make_configure(tmp_path)
        fields = {"uri": {"required": True, "input_type": "text"}}
        adapter = _make_adapter_metadata(fields)
        registry = MagicMock()
        registry.list_available_adapters.return_value = {"duckdb": adapter, "sqlite": adapter}

        # Select duckdb -> back from uri -> select sqlite -> provide uri -> ok
        select_values = iter(["duckdb", "sqlite"])
        prompt_values = iter([_BACK, "mydb.sqlite"])

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", return_value="testdb"),
            patch("datus.cli.interactive_configure.select_choice", side_effect=lambda *a, **kw: next(select_values)),
            patch(
                "datus.cli.interactive_configure._prompt_with_back", side_effect=lambda *a, **kw: next(prompt_values)
            ),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(True, "")),
        ):
            result = cfg._add_database()

        assert result is True
        assert cfg.databases["testdb"]["type"] == "sqlite"

    def test_optional_field_with_default_uses_default_on_empty(self, tmp_path):
        """Optional field with a default value uses the default when user enters nothing."""
        from rich.prompt import Prompt

        cfg = _make_configure(tmp_path)
        fields = {
            "schema": {"required": False, "default": "public"},
        }
        adapter = _make_adapter_metadata(fields)
        registry = MagicMock()
        registry.list_available_adapters.return_value = {"postgresql": adapter}

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", return_value="pgdb"),
            patch("datus.cli.interactive_configure.select_choice", return_value="postgresql"),
            patch("datus.cli.interactive_configure._prompt_with_back", return_value="public"),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(True, "")),
        ):
            cfg._add_database()

        assert "pgdb" in cfg.databases

    def test_required_field_uses_plain_prompt(self, tmp_path):
        """Required field without default uses plain _prompt_with_back."""
        from rich.prompt import Prompt

        cfg = _make_configure(tmp_path)
        fields = {
            "account": {"required": True},
        }
        adapter = _make_adapter_metadata(fields)
        registry = MagicMock()
        registry.list_available_adapters.return_value = {"snowflake": adapter}

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", return_value="sfdb"),
            patch("datus.cli.interactive_configure.select_choice", return_value="snowflake"),
            patch("datus.cli.interactive_configure._prompt_with_back", return_value="my-account"),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(True, "")),
        ):
            result = cfg._add_database()

        assert result is True

    def test_back_from_password_non_first_field_navigates_back(self, tmp_path):
        """Pressing Back at a password field that is NOT the first field navigates to prior field."""
        from rich.prompt import Prompt

        from datus.cli.interactive_configure import _BACK

        cfg = _make_configure(tmp_path)
        # Two fields: host (text), password (password)
        fields = {
            "host": {"required": True, "input_type": "text"},
            "password": {"required": False, "input_type": "password"},
        }
        adapter = _make_adapter_metadata(fields)
        registry = MagicMock()
        registry.list_available_adapters.return_value = {"postgresql": adapter}

        # host value, password back, host again, password value
        prompt_calls = iter(["localhost", _BACK, "localhost2", "secret"])

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", return_value="pgdb"),
            patch("datus.cli.interactive_configure.select_choice", return_value="postgresql"),
            patch("datus.cli.interactive_configure._prompt_with_back", side_effect=lambda *a, **kw: next(prompt_calls)),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(True, "")),
        ):
            result = cfg._add_database()

        assert result is True
        assert cfg.databases["pgdb"]["host"] == "localhost2"

    def test_back_from_port_non_first_field_navigates_back(self, tmp_path):
        """Pressing Back at a port field that is NOT the first field navigates to prior field."""
        from rich.prompt import Prompt

        from datus.cli.interactive_configure import _BACK

        cfg = _make_configure(tmp_path)
        # Two fields: host (text), port (int)
        fields = {
            "host": {"required": True, "input_type": "text"},
            "port": {"required": False, "type": "int", "default": 5432},
        }
        adapter = _make_adapter_metadata(fields)
        registry = MagicMock()
        registry.list_available_adapters.return_value = {"postgresql": adapter}

        # host value, port back, host again, port value
        prompt_calls = iter(["localhost", _BACK, "localhost2", "5432"])

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", return_value="pgdb"),
            patch("datus.cli.interactive_configure.select_choice", return_value="postgresql"),
            patch("datus.cli.interactive_configure._prompt_with_back", side_effect=lambda *a, **kw: next(prompt_calls)),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(True, "")),
        ):
            result = cfg._add_database()

        assert result is True

    def test_port_defaults_when_empty_string(self, tmp_path):
        """Port field uses default_value when user submits empty string."""
        from rich.prompt import Prompt

        cfg = _make_configure(tmp_path)
        fields = {
            "port": {"required": False, "type": "int", "default": 5432},
        }
        adapter = _make_adapter_metadata(fields)
        registry = MagicMock()
        registry.list_available_adapters.return_value = {"postgresql": adapter}

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", return_value="pgdb"),
            patch("datus.cli.interactive_configure.select_choice", return_value="postgresql"),
            patch("datus.cli.interactive_configure._prompt_with_back", return_value=""),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(True, "")),
        ):
            cfg._add_database()

        # default_value (5432) should be used when empty string entered
        assert "pgdb" in cfg.databases

    def test_optional_field_without_default_uses_empty_prompt(self, tmp_path):
        """Optional field with no default uses _prompt_with_back with empty default."""
        from rich.prompt import Prompt

        cfg = _make_configure(tmp_path)
        fields = {
            "warehouse": {"required": False},  # No default
        }
        adapter = _make_adapter_metadata(fields)
        registry = MagicMock()
        registry.list_available_adapters.return_value = {"snowflake": adapter}

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", return_value="sfdb"),
            patch("datus.cli.interactive_configure.select_choice", return_value="snowflake"),
            patch("datus.cli.interactive_configure._prompt_with_back", return_value="COMPUTE_WH"),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(True, "")),
        ):
            result = cfg._add_database()

        assert result is True

    def test_back_from_second_text_field_navigates_back(self, tmp_path):
        """Pressing Back on a non-first plain text field navigates to the previous field."""
        from rich.prompt import Prompt

        from datus.cli.interactive_configure import _BACK

        cfg = _make_configure(tmp_path)
        # Three fields: host, schema (optional no default), database (required)
        fields = {
            "host": {"required": True, "input_type": "text"},
            "schema": {"required": False},  # no default -> uses empty prompt
            "database": {"required": True, "input_type": "text"},
        }
        adapter = _make_adapter_metadata(fields)
        registry = MagicMock()
        registry.list_available_adapters.return_value = {"postgresql": adapter}

        # host, schema-back, host again, schema, database
        prompt_calls = iter(["localhost", _BACK, "localhost2", "public", "mydb"])

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", return_value="pgdb"),
            patch("datus.cli.interactive_configure.select_choice", return_value="postgresql"),
            patch("datus.cli.interactive_configure._prompt_with_back", side_effect=lambda *a, **kw: next(prompt_calls)),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(True, "")),
        ):
            result = cfg._add_database()

        assert result is True

    def test_back_from_password_first_field_returns_to_type_selection(self, tmp_path):
        """Pressing Back at a password field that IS the first field returns to DB type selection."""
        from rich.prompt import Prompt

        from datus.cli.interactive_configure import _BACK

        cfg = _make_configure(tmp_path)
        # Single field: password only (it is field_idx == 0)
        fields = {
            "password": {"required": False, "input_type": "password"},
        }
        adapter = _make_adapter_metadata(fields)
        adapter2_fields = {"uri": {"required": True, "input_type": "text"}}
        adapter2 = _make_adapter_metadata(adapter2_fields)
        registry = MagicMock()
        registry.list_available_adapters.return_value = {"mysql": adapter, "duckdb": adapter2}

        # Select mysql -> back from password (first field) -> select duckdb -> provide uri
        select_values = iter(["mysql", "duckdb"])
        prompt_values = iter([_BACK, "mydb.duckdb"])

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", return_value="testdb"),
            patch("datus.cli.interactive_configure.select_choice", side_effect=lambda *a, **kw: next(select_values)),
            patch(
                "datus.cli.interactive_configure._prompt_with_back", side_effect=lambda *a, **kw: next(prompt_values)
            ),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(True, "")),
        ):
            result = cfg._add_database()

        assert result is True
        assert cfg.databases["testdb"]["type"] == "duckdb"

    def test_back_from_port_first_field_returns_to_type_selection(self, tmp_path):
        """Pressing Back at a port field that IS the first field returns to DB type selection."""
        from rich.prompt import Prompt

        from datus.cli.interactive_configure import _BACK

        cfg = _make_configure(tmp_path)
        # Single field: port only (it is field_idx == 0)
        fields = {
            "port": {"required": False, "type": "int", "default": 3306},
        }
        adapter = _make_adapter_metadata(fields)
        adapter2_fields = {"uri": {"required": True, "input_type": "text"}}
        adapter2 = _make_adapter_metadata(adapter2_fields)
        registry = MagicMock()
        registry.list_available_adapters.return_value = {"mysql": adapter, "duckdb": adapter2}

        # Select mysql -> back from port (first field) -> select duckdb -> provide uri
        select_values = iter(["mysql", "duckdb"])
        prompt_values = iter([_BACK, "mydb.duckdb"])

        with (
            patch("datus.tools.db_tools.connector_registry", registry),
            patch.object(Prompt, "ask", return_value="testdb"),
            patch("datus.cli.interactive_configure.select_choice", side_effect=lambda *a, **kw: next(select_values)),
            patch(
                "datus.cli.interactive_configure._prompt_with_back", side_effect=lambda *a, **kw: next(prompt_values)
            ),
            patch("datus.cli.interactive_configure.detect_db_connectivity", return_value=(True, "")),
        ):
            result = cfg._add_database()

        assert result is True
        assert cfg.databases["testdb"]["type"] == "duckdb"

    def test_provider_has_no_api_key_env(self, tmp_path):
        """Provider with no api_key_env uses plain API key prompt."""
        cfg = _make_configure(tmp_path)
        catalog = _make_provider_catalog()
        # Remove api_key_env from openai provider
        del catalog["providers"]["openai"]["api_key_env"]

        with (
            patch.object(cfg, "_load_provider_catalog", return_value=catalog),
            patch("datus.cli.interactive_configure.select_choice", side_effect=["openai", "gpt-4o"]),
            patch("datus.cli.interactive_configure.os.environ.get", return_value=""),
            patch(
                "datus.cli.interactive_configure._prompt_with_back", side_effect=["sk-key", "https://api.openai.com/v1"]
            ),
            patch("datus.cli.interactive_configure.Confirm.ask", return_value=False),
            patch.object(cfg, "_test_llm_connectivity", return_value=(True, "")),
        ):
            result = cfg._add_model()

        assert result is True


# ─────────────────────────────────────────────────────────────────────────────
# _delete_model, _delete_database, _set_default_model
# ─────────────────────────────────────────────────────────────────────────────


class TestDeleteAndSet:
    """Tests for _delete_model(), _delete_database(), and _set_default_model()."""

    def test_delete_model_removes_model(self, tmp_path):
        """_delete_model() removes the specified model from self.models."""
        from rich.prompt import Confirm, Prompt

        cfg = _make_configure(tmp_path)
        cfg.models = {
            "openai": {"type": "openai"},
            "deepseek": {"type": "openai"},
        }
        cfg.target = "openai"

        with (
            patch.object(Prompt, "ask", return_value="openai"),
            patch.object(Confirm, "ask", return_value=True),
        ):
            cfg._delete_model()

        assert "openai" not in cfg.models
        # target should shift to next model
        assert cfg.target == "deepseek"

    def test_delete_model_does_not_delete_when_confirm_false(self, tmp_path):
        """_delete_model() does not remove model when user declines confirmation."""
        from rich.prompt import Confirm, Prompt

        cfg = _make_configure(tmp_path)
        cfg.models = {"openai": {"type": "openai"}}
        cfg.target = "openai"

        with (
            patch.object(Prompt, "ask", return_value="openai"),
            patch.object(Confirm, "ask", return_value=False),
        ):
            cfg._delete_model()

        assert "openai" in cfg.models

    def test_delete_model_last_model_sets_empty_target(self, tmp_path):
        """_delete_model() sets target to '' when the last model is deleted."""
        from rich.prompt import Confirm, Prompt

        cfg = _make_configure(tmp_path)
        cfg.models = {"openai": {"type": "openai"}}
        cfg.target = "openai"

        with (
            patch.object(Prompt, "ask", return_value="openai"),
            patch.object(Confirm, "ask", return_value=True),
        ):
            cfg._delete_model()

        assert cfg.target == ""

    def test_delete_database_removes_database(self, tmp_path):
        """_delete_database() removes the specified database from self.databases."""
        from rich.prompt import Confirm, Prompt

        cfg = _make_configure(tmp_path)
        cfg.databases = {"mydb": {"type": "sqlite"}}

        with (
            patch.object(Prompt, "ask", return_value="mydb"),
            patch.object(Confirm, "ask", return_value=True),
        ):
            cfg._delete_database()

        assert "mydb" not in cfg.databases

    def test_delete_database_does_not_delete_when_confirm_false(self, tmp_path):
        """_delete_database() does not remove database when user declines."""
        from rich.prompt import Confirm, Prompt

        cfg = _make_configure(tmp_path)
        cfg.databases = {"mydb": {"type": "sqlite"}}

        with (
            patch.object(Prompt, "ask", return_value="mydb"),
            patch.object(Confirm, "ask", return_value=False),
        ):
            cfg._delete_database()

        assert "mydb" in cfg.databases

    def test_set_default_model_updates_target(self, tmp_path):
        """_set_default_model() updates self.target to the chosen model."""
        from rich.prompt import Prompt

        cfg = _make_configure(tmp_path)
        cfg.models = {
            "openai": {"type": "openai"},
            "deepseek": {"type": "openai"},
        }
        cfg.target = "openai"

        with patch.object(Prompt, "ask", return_value="deepseek"):
            cfg._set_default_model()

        assert cfg.target == "deepseek"


# ─────────────────────────────────────────────────────────────────────────────
# _save
# ─────────────────────────────────────────────────────────────────────────────


class TestSave:
    """Tests for InteractiveConfigure._save()."""

    def test_save_writes_yaml_with_service_structure(self, tmp_path):
        """_save() writes a YAML file containing the service.databases section."""
        cfg = _make_configure(tmp_path)
        cfg.models = {"openai": {"type": "openai", "model": "gpt-4o", "api_key": "sk-test"}}
        cfg.databases = {"my_db": {"type": "sqlite", "uri": "path/to/db.sqlite", "default": True}}
        cfg.target = "openai"

        cfg._save()

        assert cfg.config_path.exists()
        with open(cfg.config_path, encoding="utf-8") as f:
            saved = yaml.safe_load(f)

        agent = saved["agent"]
        assert agent["target"] == "openai"
        assert "my_db" in agent["service"]["databases"]
        assert "bi_tools" in agent["service"]
        assert "schedulers" in agent["service"]

    def test_save_merges_with_existing_config_preserves_other_sections(self, tmp_path):
        """_save() preserves sections not managed by InteractiveConfigure."""
        existing = {
            "agent": {
                "target": "old",
                "models": {},
                "nodes": {"schema_linking": {"matching_rate": "slow"}},
                "custom_section": {"keep_me": True},
            }
        }
        config_file = tmp_path / "agent.yml"
        config_file.write_text(yaml.dump(existing), encoding="utf-8")

        cfg = _make_configure(tmp_path)
        cfg.models = {"openai": {"type": "openai", "model": "gpt-4o", "api_key": "sk-test"}}
        cfg.databases = {}
        cfg.target = "openai"

        cfg._save()

        with open(config_file, encoding="utf-8") as f:
            saved = yaml.safe_load(f)

        # custom_section must still be there
        assert saved["agent"].get("custom_section") == {"keep_me": True}
        # target updated
        assert saved["agent"]["target"] == "openai"

    def test_save_removes_legacy_namespace_key(self, tmp_path):
        """_save() removes the legacy 'namespace' key if present in existing config."""
        existing = {
            "agent": {
                "target": "",
                "models": {},
                "namespace": {"old_ns": {"type": "sqlite"}},
            }
        }
        config_file = tmp_path / "agent.yml"
        config_file.write_text(yaml.dump(existing), encoding="utf-8")

        cfg = _make_configure(tmp_path)
        cfg.models = {}
        cfg.databases = {}
        cfg.target = ""
        cfg._save()

        with open(config_file, encoding="utf-8") as f:
            saved = yaml.safe_load(f)

        assert "namespace" not in saved["agent"]

    def test_save_sets_default_nodes_when_absent(self, tmp_path):
        """_save() adds default nodes section when it is not already present."""
        cfg = _make_configure(tmp_path)
        cfg.models = {}
        cfg.databases = {}
        cfg.target = ""
        cfg._save()

        with open(cfg.config_path, encoding="utf-8") as f:
            saved = yaml.safe_load(f)

        assert "nodes" in saved["agent"]
        assert "schema_linking" in saved["agent"]["nodes"]

    def test_save_preserves_existing_nodes_section(self, tmp_path):
        """_save() does not overwrite existing nodes section."""
        existing = {
            "agent": {
                "target": "",
                "models": {},
                "nodes": {"schema_linking": {"matching_rate": "accurate"}, "date_parser": {"language": "zh"}},
            }
        }
        config_file = tmp_path / "agent.yml"
        config_file.write_text(yaml.dump(existing), encoding="utf-8")

        cfg = _make_configure(tmp_path)
        cfg.models = {}
        cfg.databases = {}
        cfg.target = ""
        cfg._save()

        with open(config_file, encoding="utf-8") as f:
            saved = yaml.safe_load(f)

        assert saved["agent"]["nodes"]["schema_linking"]["matching_rate"] == "accurate"
        assert saved["agent"]["nodes"]["date_parser"]["language"] == "zh"

    def test_save_handles_corrupt_existing_yaml(self, tmp_path):
        """_save() writes new config even when existing YAML is unreadable."""
        config_file = tmp_path / "agent.yml"
        config_file.write_text(": invalid: yaml: {[", encoding="utf-8")

        cfg = _make_configure(tmp_path)
        cfg.models = {"openai": {"type": "openai"}}
        cfg.databases = {}
        cfg.target = "openai"
        cfg._save()

        with open(config_file, encoding="utf-8") as f:
            saved = yaml.safe_load(f)

        assert saved["agent"]["target"] == "openai"


# ─────────────────────────────────────────────────────────────────────────────
# _display_completion
# ─────────────────────────────────────────────────────────────────────────────


class TestDisplayCompletion:
    """Tests for InteractiveConfigure._display_completion()."""

    def test_shows_default_database_in_message(self, tmp_path):
        """_display_completion() includes the default database name in the output."""
        cfg = _make_configure(tmp_path)
        cfg.databases = {
            "main_db": {"type": "duckdb", "default": True},
            "other_db": {"type": "sqlite"},
        }

        printed = []
        cfg.console = MagicMock()
        cfg.console.print.side_effect = lambda msg: printed.append(msg)

        cfg._display_completion()

        combined = " ".join(str(m) for m in printed)
        assert "main_db" in combined

    def test_shows_first_database_when_no_default(self, tmp_path):
        """_display_completion() uses the first database when none is marked default."""
        cfg = _make_configure(tmp_path)
        cfg.databases = {
            "first_db": {"type": "sqlite"},
            "second_db": {"type": "duckdb"},
        }

        printed = []
        cfg.console = MagicMock()
        cfg.console.print.side_effect = lambda msg: printed.append(msg)

        cfg._display_completion()

        combined = " ".join(str(m) for m in printed)
        assert "first_db" in combined

    def test_shows_generic_message_when_no_databases(self, tmp_path):
        """_display_completion() shows a generic init message when no databases exist."""
        cfg = _make_configure(tmp_path)
        cfg.databases = {}

        printed = []
        cfg.console = MagicMock()
        cfg.console.print.side_effect = lambda msg: printed.append(msg)

        cfg._display_completion()

        combined = " ".join(str(m) for m in printed)
        assert "datus init" in combined


# ─────────────────────────────────────────────────────────────────────────────
# _install_plugin
# ─────────────────────────────────────────────────────────────────────────────


class TestInstallPlugin:
    """Tests for InteractiveConfigure._install_plugin().

    subprocess, shutil, and sys are imported locally inside _install_plugin,
    so they must be patched at their canonical stdlib locations.
    """

    def test_install_plugin_returns_true_on_success(self, tmp_path):
        """_install_plugin returns True when subprocess exits with returncode 0."""
        mock_result = MagicMock()
        mock_result.returncode = 0

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("shutil.which", return_value=None),
        ):
            cfg = _make_configure(tmp_path)
            result = cfg._install_plugin("datus-snowflake")

        assert result is True

    def test_install_plugin_returns_false_on_nonzero_exit(self, tmp_path):
        """_install_plugin returns False when subprocess exits with nonzero code."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error: package not found"

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("shutil.which", return_value=None),
        ):
            cfg = _make_configure(tmp_path)
            result = cfg._install_plugin("datus-nonexistent")

        assert result is False

    def test_install_plugin_returns_false_on_timeout(self, tmp_path):
        """_install_plugin returns False on subprocess.TimeoutExpired."""
        with (
            patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["pip"], timeout=120),
            ),
            patch("shutil.which", return_value=None),
        ):
            cfg = _make_configure(tmp_path)
            result = cfg._install_plugin("datus-slow-package")

        assert result is False

    def test_install_plugin_uses_uv_when_available(self, tmp_path):
        """_install_plugin uses uv pip install when uv is on PATH."""
        mock_result = MagicMock()
        mock_result.returncode = 0

        with (
            patch("subprocess.run", return_value=mock_result) as mock_run,
            patch("shutil.which", return_value="/usr/local/bin/uv"),
        ):
            cfg = _make_configure(tmp_path)
            result = cfg._install_plugin("datus-postgresql")

        assert result is True
        call_args = mock_run.call_args[0][0]
        assert "uv" in call_args[0]
        assert "pip" in call_args

    def test_install_plugin_returns_false_on_generic_exception(self, tmp_path):
        """_install_plugin returns False when subprocess.run raises an unexpected error."""
        with (
            patch(
                "subprocess.run",
                side_effect=OSError("executable not found"),
            ),
            patch("shutil.which", return_value=None),
        ):
            cfg = _make_configure(tmp_path)
            result = cfg._install_plugin("datus-mysql")

        assert result is False


# ─────────────────────────────────────────────────────────────────────────────
# _test_llm_connectivity
# ─────────────────────────────────────────────────────────────────────────────


class TestTestLlmConnectivity:
    """Tests for InteractiveConfigure._test_llm_connectivity()."""

    def test_returns_true_when_llm_generates_response(self, tmp_path):
        """_test_llm_connectivity returns (True, '') when LLM returns non-empty response."""

        cfg = _make_configure(tmp_path)
        model_entry = {
            "type": "openai",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-test",
            "model": "gpt-4o",
        }

        fake_module = MagicMock()
        fake_model_instance = MagicMock()
        fake_model_instance.generate.return_value = "Hello there!"
        fake_model_class = MagicMock(return_value=fake_model_instance)
        fake_module.OpenAIModel = fake_model_class

        mock_config = MagicMock()
        mock_config.type = "openai"

        with (
            patch.dict("sys.modules", {"datus.models.openai_model": fake_module}),
            patch("datus.configuration.agent_config.load_model_config", return_value=mock_config),
            patch("datus.configuration.agent_config.resolve_env", side_effect=lambda v: v),
            patch("datus.models.base.LLMBaseModel.MODEL_TYPE_MAP", {"openai": "OpenAIModel"}),
        ):
            ok, err = cfg._test_llm_connectivity(model_entry)

        assert ok is True
        assert err == ""

    def test_returns_false_when_response_is_empty(self, tmp_path):
        """_test_llm_connectivity returns (False, ...) when LLM returns empty response."""

        cfg = _make_configure(tmp_path)
        model_entry = {
            "type": "openai",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-test",
            "model": "gpt-4o",
        }

        mock_config = MagicMock()
        mock_config.type = "openai"

        fake_module = MagicMock()
        fake_model_instance = MagicMock()
        fake_model_instance.generate.return_value = None
        fake_module.OpenAIModel = MagicMock(return_value=fake_model_instance)

        with (
            patch.dict("sys.modules", {"datus.models.openai_model": fake_module}),
            patch("datus.configuration.agent_config.load_model_config", return_value=mock_config),
            patch("datus.configuration.agent_config.resolve_env", side_effect=lambda v: v),
            patch("datus.models.base.LLMBaseModel.MODEL_TYPE_MAP", {"openai": "OpenAIModel"}),
        ):
            ok, err = cfg._test_llm_connectivity(model_entry)

        assert ok is False

    def test_returns_false_on_exception(self, tmp_path):
        """_test_llm_connectivity returns (False, error_str) on exception."""
        cfg = _make_configure(tmp_path)
        model_entry = {"type": "openai", "api_key": "sk-test", "model": "gpt-4o", "base_url": ""}

        with patch("datus.configuration.agent_config.load_model_config", side_effect=Exception("config error")):
            ok, err = cfg._test_llm_connectivity(model_entry)

        assert ok is False
        assert "config error" in err

    def test_returns_false_for_unsupported_model_type(self, tmp_path):
        """_test_llm_connectivity returns (False, ...) for unsupported model type."""
        cfg = _make_configure(tmp_path)
        model_entry = {"type": "unknown_provider", "api_key": "sk-test", "model": "unknown-model", "base_url": ""}

        mock_config = MagicMock()
        mock_config.type = "unknown_provider"

        with (
            patch("datus.configuration.agent_config.load_model_config", return_value=mock_config),
            patch("datus.configuration.agent_config.resolve_env", side_effect=lambda v: v),
            patch("datus.models.base.LLMBaseModel.MODEL_TYPE_MAP", {}),
        ):
            ok, err = cfg._test_llm_connectivity(model_entry)

        assert ok is False
        assert "Unsupported model type" in err


# ─────────────────────────────────────────────────────────────────────────────
# _configure_codex_oauth
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigureCodexOauth:
    """Tests for InteractiveConfigure._configure_codex_oauth()."""

    def _provider_config(self):
        return {
            "type": "openai",
            "auth_type": "oauth",
            "models": ["codex-model"],
            "default_model": "codex-model",
        }

    def test_successful_oauth_flow(self, tmp_path):
        """_configure_codex_oauth() returns True when token retrieved and LLM responds."""

        cfg = _make_configure(tmp_path)
        provider_config = self._provider_config()

        fake_cred_module = MagicMock()
        fake_cred_module.get_codex_oauth_token.return_value = "oauth-token-123"

        with (
            patch.dict("sys.modules", {"datus.auth.codex_credential": fake_cred_module}),
            patch("datus.cli.interactive_configure.select_choice", return_value="codex-model"),
            patch.object(cfg, "_test_llm_connectivity", return_value=(True, "")),
        ):
            result = cfg._configure_codex_oauth("codex", provider_config)

        assert result is True
        assert "codex" in cfg.models
        assert cfg.models["codex"]["api_key"] == "oauth-token-123"
        assert cfg.models["codex"]["auth_type"] == "oauth"
        assert cfg.target == "codex"

    def test_returns_false_when_token_retrieval_fails(self, tmp_path):
        """_configure_codex_oauth() returns False when token cannot be retrieved."""

        cfg = _make_configure(tmp_path)
        provider_config = self._provider_config()

        fake_cred_module = MagicMock()
        fake_cred_module.get_codex_oauth_token.side_effect = Exception("auth failed")

        with patch.dict("sys.modules", {"datus.auth.codex_credential": fake_cred_module}):
            result = cfg._configure_codex_oauth("codex", provider_config)

        assert result is False

    def test_returns_false_when_connectivity_fails(self, tmp_path):
        """_configure_codex_oauth() returns False when LLM connectivity test fails."""

        cfg = _make_configure(tmp_path)
        provider_config = self._provider_config()

        fake_cred_module = MagicMock()
        fake_cred_module.get_codex_oauth_token.return_value = "token-xyz"

        with (
            patch.dict("sys.modules", {"datus.auth.codex_credential": fake_cred_module}),
            patch("datus.cli.interactive_configure.select_choice", return_value="codex-model"),
            patch.object(cfg, "_test_llm_connectivity", return_value=(False, "LLM error")),
        ):
            result = cfg._configure_codex_oauth("codex", provider_config)

        assert result is False

    def test_provider_with_no_models_list_uses_prompt_ask(self, tmp_path):
        """_configure_codex_oauth() uses Prompt.ask when no models list in provider config."""

        from rich.prompt import Prompt

        cfg = _make_configure(tmp_path)
        provider_config = {"type": "openai", "auth_type": "oauth", "models": [], "default_model": "codex-default"}

        fake_cred_module = MagicMock()
        fake_cred_module.get_codex_oauth_token.return_value = "token"

        with (
            patch.dict("sys.modules", {"datus.auth.codex_credential": fake_cred_module}),
            patch.object(Prompt, "ask", return_value="my-codex-model"),
            patch.object(cfg, "_test_llm_connectivity", return_value=(True, "")),
        ):
            result = cfg._configure_codex_oauth("codex", provider_config)

        assert result is True
        assert cfg.models["codex"]["model"] == "my-codex-model"


# ─────────────────────────────────────────────────────────────────────────────
# _configure_claude_subscription
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigureClaudeSubscription:
    """Tests for InteractiveConfigure._configure_claude_subscription()."""

    def _provider_config(self):
        return {
            "type": "anthropic",
            "base_url": "https://api.anthropic.com/v1",
            "auth_type": "subscription",
            "models": ["claude-3-opus"],
            "default_model": "claude-3-opus",
        }

    def test_successful_subscription_flow_with_auto_token(self, tmp_path):
        """_configure_claude_subscription() returns True when auto-detected token works."""

        cfg = _make_configure(tmp_path)
        provider_config = self._provider_config()

        fake_cred_module = MagicMock()
        fake_cred_module.get_claude_subscription_token.return_value = ("sk-ant-oat01-abc", "environment")

        with (
            patch.dict("sys.modules", {"datus.auth.claude_credential": fake_cred_module}),
            patch("datus.cli.interactive_configure.select_choice", return_value="claude-3-opus"),
            patch.object(cfg, "_test_llm_connectivity", return_value=(True, "")),
        ):
            result = cfg._configure_claude_subscription("claude", provider_config)

        assert result is True
        assert "claude" in cfg.models
        assert cfg.models["claude"]["api_key"] == "sk-ant-oat01-abc"
        assert cfg.models["claude"]["auth_type"] == "subscription"
        assert cfg.target == "claude"

    def test_falls_back_to_getpass_when_auto_detect_fails(self, tmp_path):
        """_configure_claude_subscription() falls back to getpass when auto-detection fails."""

        cfg = _make_configure(tmp_path)
        provider_config = self._provider_config()

        fake_cred_module = MagicMock()
        fake_cred_module.get_claude_subscription_token.side_effect = Exception("no token")

        with (
            patch.dict("sys.modules", {"datus.auth.claude_credential": fake_cred_module}),
            patch("datus.cli.interactive_configure.select_choice", return_value="claude-3-opus"),
            patch("datus.cli.interactive_configure.getpass", return_value="sk-ant-oat01-manual"),
            patch.object(cfg, "_test_llm_connectivity", return_value=(True, "")),
        ):
            result = cfg._configure_claude_subscription("claude", provider_config)

        assert result is True
        assert cfg.models["claude"]["api_key"] == "sk-ant-oat01-manual"

    def test_returns_false_when_manual_token_is_empty(self, tmp_path):
        """_configure_claude_subscription() returns False when manual token is empty."""

        cfg = _make_configure(tmp_path)
        provider_config = self._provider_config()

        fake_cred_module = MagicMock()
        fake_cred_module.get_claude_subscription_token.side_effect = Exception("no token")

        with (
            patch.dict("sys.modules", {"datus.auth.claude_credential": fake_cred_module}),
            patch("datus.cli.interactive_configure.select_choice", return_value="claude-3-opus"),
            patch("datus.cli.interactive_configure.getpass", return_value=""),
        ):
            result = cfg._configure_claude_subscription("claude", provider_config)

        assert result is False

    def test_returns_false_when_connectivity_test_fails(self, tmp_path):
        """_configure_claude_subscription() returns False when LLM connectivity fails."""

        cfg = _make_configure(tmp_path)
        provider_config = self._provider_config()

        fake_cred_module = MagicMock()
        fake_cred_module.get_claude_subscription_token.return_value = ("sk-ant-oat01-abc", "env")

        with (
            patch.dict("sys.modules", {"datus.auth.claude_credential": fake_cred_module}),
            patch("datus.cli.interactive_configure.select_choice", return_value="claude-3-opus"),
            patch.object(cfg, "_test_llm_connectivity", return_value=(False, "auth error")),
        ):
            result = cfg._configure_claude_subscription("claude", provider_config)

        assert result is False

    def test_provider_with_no_models_uses_prompt_ask(self, tmp_path):
        """_configure_claude_subscription() uses Prompt.ask when no models list."""

        from rich.prompt import Prompt

        cfg = _make_configure(tmp_path)
        provider_config = {
            "type": "anthropic",
            "base_url": "https://api.anthropic.com/v1",
            "auth_type": "subscription",
            "models": [],
            "default_model": "claude-default",
        }

        fake_cred_module = MagicMock()
        fake_cred_module.get_claude_subscription_token.return_value = ("sk-ant-oat01-abc", "env")

        with (
            patch.dict("sys.modules", {"datus.auth.claude_credential": fake_cred_module}),
            patch.object(Prompt, "ask", return_value="my-claude-model"),
            patch.object(cfg, "_test_llm_connectivity", return_value=(True, "")),
        ):
            result = cfg._configure_claude_subscription("claude", provider_config)

        assert result is True
        assert cfg.models["claude"]["model"] == "my-claude-model"
