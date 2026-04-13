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
