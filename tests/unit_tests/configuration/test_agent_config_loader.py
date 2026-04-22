# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/configuration/agent_config_loader.py

CI-level: zero external dependencies, all file I/O is mocked or uses tmp_path.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from datus.configuration.agent_config_loader import (
    ConfigurationManager,
    _apply_project_override,
    configuration_manager,
    load_agent_config,
    load_node_config,
    parse_config_path,
)
from datus.configuration.project_config import ProjectOverride
from datus.utils.exceptions import DatusException

# ---------------------------------------------------------------------------
# parse_config_path
# ---------------------------------------------------------------------------


class TestParseConfigPath:
    def test_explicit_existing_file(self, tmp_path):
        cfg = tmp_path / "agent.yml"
        cfg.write_text("agent: {}")
        result = parse_config_path(str(cfg))
        assert result == cfg

    def test_explicit_non_existing_non_default_raises(self):
        with pytest.raises(DatusException, match="not found"):
            parse_config_path("nonexistent_config.yml")

    def test_explicit_default_fallback(self, tmp_path, monkeypatch):
        """When config_file is 'conf/agent.yml' and doesn't exist anywhere, raises DatusException."""
        # chdir to tmp_path (no conf/agent.yml there) and patch home to nonexistent path
        monkeypatch.chdir(tmp_path)
        with patch("datus.configuration.agent_config_loader.Path.home", return_value=tmp_path / "noexist"):
            with pytest.raises(DatusException):
                parse_config_path("conf/agent.yml")

    def test_local_conf_found(self, tmp_path, monkeypatch):
        """Finds conf/agent.yml in the current working directory."""
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        cfg = conf_dir / "agent.yml"
        cfg.write_text("agent: {}")
        monkeypatch.chdir(tmp_path)
        result = parse_config_path("")
        assert result.name == "agent.yml"

    def test_home_config_found(self, tmp_path, monkeypatch):
        """Falls back to ~/.datus/conf/agent.yml."""
        # Use a directory without local conf
        monkeypatch.chdir(tmp_path)
        home_conf = tmp_path / ".datus" / "conf"
        home_conf.mkdir(parents=True)
        cfg = home_conf / "agent.yml"
        cfg.write_text("agent: {}")

        with patch("datus.configuration.agent_config_loader.Path.home", return_value=tmp_path):
            result = parse_config_path("")
        assert result.name == "agent.yml"

    def test_no_config_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # No conf/agent.yml and patch home to non-existent
        with patch("datus.configuration.agent_config_loader.Path.home", return_value=tmp_path / "noexist"):
            with pytest.raises(DatusException, match="not found"):
                parse_config_path("")


# ---------------------------------------------------------------------------
# ConfigurationManager
# ---------------------------------------------------------------------------


class TestConfigurationManager:
    def _make_config(self, tmp_path, data: dict | None = None) -> Path:
        cfg = tmp_path / "agent.yml"
        content = {"agent": data or {"target": "test"}}
        cfg.write_text(yaml.safe_dump(content))
        return cfg

    def test_load_basic(self, tmp_path):
        cfg = self._make_config(tmp_path, {"target": "test_db"})
        mgr = ConfigurationManager(str(cfg))
        assert mgr.get("target") == "test_db"

    def test_get_with_default(self, tmp_path):
        cfg = self._make_config(tmp_path, {})
        mgr = ConfigurationManager(str(cfg))
        assert mgr.get("missing_key", "default_val") == "default_val"

    def test_update_item_new_key(self, tmp_path):
        cfg = self._make_config(tmp_path, {"a": 1})
        mgr = ConfigurationManager(str(cfg))
        result = mgr.update_item("b", 2, save=True)
        assert result is True
        assert mgr.get("b") == 2
        # Verify persisted
        mgr2 = ConfigurationManager(str(cfg))
        assert mgr2.get("b") == 2

    def test_update_item_merge_dict(self, tmp_path):
        cfg = self._make_config(tmp_path, {"opts": {"x": 1}})
        mgr = ConfigurationManager(str(cfg))
        mgr.update_item("opts", {"y": 2}, save=False)
        assert mgr.get("opts") == {"x": 1, "y": 2}

    def test_update_item_delete_old_key(self, tmp_path):
        cfg = self._make_config(tmp_path, {"a": {"old": True}})
        mgr = ConfigurationManager(str(cfg))
        mgr.update_item("a", {"new": True}, delete_old_key=True, save=False)
        assert mgr.get("a") == {"new": True}

    def test_update_multiple(self, tmp_path):
        cfg = self._make_config(tmp_path, {"a": 1})
        mgr = ConfigurationManager(str(cfg))
        result = mgr.update({"a": 10, "b": 20}, save=False)
        assert result is True
        assert mgr.get("a") == 10
        assert mgr.get("b") == 20

    def test_remove_item_recursively(self, tmp_path):
        cfg = self._make_config(tmp_path, {"outer": {"inner": "value"}})
        mgr = ConfigurationManager(str(cfg))
        result = mgr.remove_item_recursively("outer", "inner")
        assert result is True
        assert "inner" not in mgr.get("outer", {})

    def test_remove_item_missing_path_raises(self, tmp_path):
        cfg = self._make_config(tmp_path, {"a": {}})
        mgr = ConfigurationManager(str(cfg))
        with pytest.raises(DatusException):
            mgr.remove_item_recursively("nonexistent", "key")

    def test_getitem(self, tmp_path):
        cfg = self._make_config(tmp_path, {"key1": "val1"})
        mgr = ConfigurationManager(str(cfg))
        assert mgr["key1"] == "val1"

    def test_setitem(self, tmp_path):
        cfg = self._make_config(tmp_path, {"key1": "val1"})
        mgr = ConfigurationManager(str(cfg))
        mgr["key2"] = "val2"
        assert mgr.get("key2") == "val2"

    def test_load_invalid_yaml(self, tmp_path):
        cfg = tmp_path / "agent.yml"
        cfg.write_text("agent: {invalid: yaml: content")
        # Should not raise — returns empty dict
        mgr = ConfigurationManager(str(cfg))
        assert mgr.data == {}

    def test_save_and_reload(self, tmp_path):
        cfg = self._make_config(tmp_path, {"x": 42})
        mgr = ConfigurationManager(str(cfg))
        mgr.update_item("x", 99, save=True)
        mgr2 = ConfigurationManager(str(cfg))
        assert mgr2.get("x") == 99


# ---------------------------------------------------------------------------
# configuration_manager singleton
# ---------------------------------------------------------------------------


class TestConfigurationManagerSingleton:
    def test_reload_creates_new_instance(self, tmp_path):
        cfg = tmp_path / "agent.yml"
        cfg.write_text(yaml.safe_dump({"agent": {"v": 1}}))
        m1 = configuration_manager(str(cfg), reload=True)
        m2 = configuration_manager(str(cfg), reload=True)
        # reload=True always creates a new instance
        assert m1 is not m2

    def test_no_reload_returns_cached(self, tmp_path):
        cfg = tmp_path / "agent.yml"
        cfg.write_text(yaml.safe_dump({"agent": {"v": 1}}))
        m1 = configuration_manager(str(cfg), reload=True)
        m2 = configuration_manager(str(cfg), reload=False)
        # Without reload, returns the same cached instance
        assert m1 is m2


# ---------------------------------------------------------------------------
# load_node_config
# ---------------------------------------------------------------------------


class TestApplyProjectOverride:
    """_apply_project_override validates & merges ./.datus/config.yml into agent_raw."""

    def _base_raw(self):
        return {
            "target": "openai",
            "models": {"openai": {"type": "openai"}, "deepseek": {"type": "deepseek"}},
            "services": {"datasources": {"db1": {"type": "sqlite"}, "db2": {"type": "duckdb"}}},
        }

    def test_no_override_is_noop(self):
        agent_raw = self._base_raw()
        with patch(
            "datus.configuration.agent_config_loader.load_project_override",
            return_value=None,
        ):
            _apply_project_override(agent_raw)
        assert agent_raw["target"] == "openai"
        assert "project_name" not in agent_raw

    def test_empty_override_is_noop(self):
        agent_raw = self._base_raw()
        with patch(
            "datus.configuration.agent_config_loader.load_project_override",
            return_value=ProjectOverride(),
        ):
            _apply_project_override(agent_raw)
        assert agent_raw["target"] == "openai"

    def test_target_merged_when_valid(self):
        agent_raw = self._base_raw()
        with patch(
            "datus.configuration.agent_config_loader.load_project_override",
            return_value=ProjectOverride(target="deepseek"),
        ):
            _apply_project_override(agent_raw)
        assert agent_raw["target"] == "deepseek"

    def test_invalid_target_raises(self):
        agent_raw = self._base_raw()
        with patch(
            "datus.configuration.agent_config_loader.load_project_override",
            return_value=ProjectOverride(target="nonexistent"),
        ):
            with pytest.raises(DatusException) as exc:
                _apply_project_override(agent_raw)
        assert "target" in str(exc.value)

    def test_project_name_merged(self):
        agent_raw = self._base_raw()
        with patch(
            "datus.configuration.agent_config_loader.load_project_override",
            return_value=ProjectOverride(project_name="my_proj"),
        ):
            _apply_project_override(agent_raw)
        assert agent_raw["project_name"] == "my_proj"

    def test_valid_default_datasource_flips_default_flags(self):
        """default_datasource overlay is applied by flipping databases[*].default
        so AgentConfig.services.default_datasource resolves to the override target
        uniformly across every entry point (REPL, datus-api, SDK)."""
        agent_raw = self._base_raw()
        with patch(
            "datus.configuration.agent_config_loader.load_project_override",
            return_value=ProjectOverride(default_datasource="db2"),
        ):
            _apply_project_override(agent_raw)
        assert agent_raw["services"]["datasources"]["db2"]["default"] is True
        assert agent_raw["services"]["datasources"]["db1"]["default"] is False

    def test_default_datasource_overlay_clears_prior_default(self):
        """A base config marking db1 as default must have that flag cleared
        when the overlay points elsewhere, otherwise default_datasource would
        return the first match (db1) and ignore the overlay."""
        agent_raw = self._base_raw()
        agent_raw["services"]["datasources"]["db1"]["default"] = True
        with patch(
            "datus.configuration.agent_config_loader.load_project_override",
            return_value=ProjectOverride(default_datasource="db2"),
        ):
            _apply_project_override(agent_raw)
        assert agent_raw["services"]["datasources"]["db1"]["default"] is False
        assert agent_raw["services"]["datasources"]["db2"]["default"] is True

    def test_invalid_default_datasource_raises(self):
        agent_raw = self._base_raw()
        with patch(
            "datus.configuration.agent_config_loader.load_project_override",
            return_value=ProjectOverride(default_datasource="ghost_db"),
        ):
            with pytest.raises(DatusException) as exc:
                _apply_project_override(agent_raw)
        assert "default_datasource" in str(exc.value)

    def test_all_three_fields_merged(self):
        agent_raw = self._base_raw()
        override = ProjectOverride(target="deepseek", default_datasource="db1", project_name="p")
        with patch(
            "datus.configuration.agent_config_loader.load_project_override",
            return_value=override,
        ):
            _apply_project_override(agent_raw)
        assert agent_raw["target"] == "deepseek"
        assert agent_raw["project_name"] == "p"

    def test_missing_models_section_invalid_target_raises(self):
        agent_raw = {"services": {"datasources": {"db1": {}}}}
        with patch(
            "datus.configuration.agent_config_loader.load_project_override",
            return_value=ProjectOverride(target="deepseek"),
        ):
            with pytest.raises(DatusException):
                _apply_project_override(agent_raw)

    def test_missing_service_section_invalid_db_raises(self):
        agent_raw = {"models": {"openai": {}}}
        with patch(
            "datus.configuration.agent_config_loader.load_project_override",
            return_value=ProjectOverride(default_datasource="db1"),
        ):
            with pytest.raises(DatusException):
                _apply_project_override(agent_raw)


class TestLoadNodeConfig:
    def test_with_model(self):
        """When 'model' key is present, it's extracted into NodeConfig.model."""
        data = {"model": "gpt-4o"}
        with patch("datus.configuration.agent_config_loader.NodeType.type_input", return_value={}):
            node_cfg = load_node_config("gen_sql", data)
        assert node_cfg.model == "gpt-4o"

    def test_without_model(self):
        """When 'model' key is absent, NodeConfig.model is empty string."""
        data = {}
        with patch("datus.configuration.agent_config_loader.NodeType.type_input", return_value={}):
            node_cfg = load_node_config("gen_sql", data)
        assert node_cfg.model == ""

    def test_none_data(self):
        """When data is None/falsy, NodeConfig.model is empty string."""
        with patch("datus.configuration.agent_config_loader.NodeType.type_input", return_value={}):
            node_cfg = load_node_config("gen_sql", None)
        assert node_cfg.model == ""


# ---------------------------------------------------------------------------
# load_agent_config — default database resolution tail
# ---------------------------------------------------------------------------


class TestLoadAgentConfigResolution:
    """Cover the post-override resolution that guarantees ``current_datasource``
    is populated for every entry point (REPL, datus-api, datus-gateway, SDK),
    regardless of whether ``override_by_args`` ran for the CLI ``action``.
    """

    def _write_base_yaml(self, tmp_path, datasources: dict) -> Path:
        """Write a minimal agent.yml with the given datasources map."""
        cfg = tmp_path / "agent.yml"
        cfg.write_text(
            yaml.safe_dump(
                {
                    "agent": {
                        "home": str(tmp_path),
                        "target": "mock",
                        "models": {
                            "mock": {
                                "type": "openai",
                                "api_key": "mock-api-key",
                                "model": "mock-model",
                                "base_url": "http://localhost:0",
                            }
                        },
                        "services": {"datasources": datasources},
                        "project_root": str(tmp_path / "workspace"),
                    }
                }
            )
        )
        (tmp_path / "workspace").mkdir(exist_ok=True)
        return cfg

    def test_resolves_from_default_flag(self, tmp_path, reset_global_singletons):
        """base has two DBs, one marked ``default: true`` → selected at bootstrap."""
        cfg = self._write_base_yaml(
            tmp_path,
            {
                "db_a": {"type": "sqlite", "uri": str(tmp_path / "a.sqlite"), "default": False},
                "db_b": {"type": "sqlite", "uri": str(tmp_path / "b.sqlite"), "default": True},
            },
        )
        with patch(
            "datus.configuration.agent_config_loader.load_project_override",
            return_value=None,
        ):
            agent_config = load_agent_config(config=str(cfg), home=str(tmp_path), reload=True)
        assert agent_config.current_datasource == "db_b"

    def test_resolves_single_db_auto(self, tmp_path, reset_global_singletons):
        """base has a single DB and no explicit default → auto-selected."""
        cfg = self._write_base_yaml(
            tmp_path,
            {"only_db": {"type": "sqlite", "uri": str(tmp_path / "only.sqlite")}},
        )
        with patch(
            "datus.configuration.agent_config_loader.load_project_override",
            return_value=None,
        ):
            agent_config = load_agent_config(config=str(cfg), home=str(tmp_path), reload=True)
        assert agent_config.current_datasource == "only_db"

    def test_project_overlay_wins_over_base_default(self, tmp_path, reset_global_singletons):
        """``.datus/config.yml::default_datasource`` overrides the base default flag."""
        cfg = self._write_base_yaml(
            tmp_path,
            {
                "db_a": {"type": "sqlite", "uri": str(tmp_path / "a.sqlite"), "default": True},
                "db_b": {"type": "sqlite", "uri": str(tmp_path / "b.sqlite")},
            },
        )
        with patch(
            "datus.configuration.agent_config_loader.load_project_override",
            return_value=ProjectOverride(default_datasource="db_b", target="mock"),
        ):
            agent_config = load_agent_config(config=str(cfg), home=str(tmp_path), reload=True)
        assert agent_config.current_datasource == "db_b"
        assert agent_config.target == "mock"

    def test_raises_when_ambiguous_default(self, tmp_path, reset_global_singletons):
        """Multi-DB + no ``default`` flag + no overlay → startup-time error that
        tells the user how to fix it (run ``datus`` init wizard)."""
        cfg = self._write_base_yaml(
            tmp_path,
            {
                "db_a": {"type": "sqlite", "uri": str(tmp_path / "a.sqlite")},
                "db_b": {"type": "sqlite", "uri": str(tmp_path / "b.sqlite")},
            },
        )
        with patch(
            "datus.configuration.agent_config_loader.load_project_override",
            return_value=None,
        ):
            with pytest.raises(DatusException) as exc:
                load_agent_config(config=str(cfg), home=str(tmp_path), reload=True, action="start")
        msg = str(exc.value)
        assert "./.datus/config.yml" in msg
        # The guidance must name the CLI command so users know how to recover.
        assert "datus" in msg

    def test_service_action_still_resolves(self, tmp_path, reset_global_singletons):
        """``action='service'`` previously skipped the namespace fallback in
        ``override_by_args``; the loader tail must still populate the default."""
        cfg = self._write_base_yaml(
            tmp_path,
            {
                "db_a": {"type": "sqlite", "uri": str(tmp_path / "a.sqlite"), "default": True},
                "db_b": {"type": "sqlite", "uri": str(tmp_path / "b.sqlite")},
            },
        )
        with patch(
            "datus.configuration.agent_config_loader.load_project_override",
            return_value=None,
        ):
            agent_config = load_agent_config(
                config=str(cfg),
                home=str(tmp_path),
                reload=True,
                action="service",
            )
        assert agent_config.current_datasource == "db_a"

    def test_no_databases_is_tolerated(self, tmp_path, reset_global_singletons):
        """Deployments without any configured DB (pure KB / tool-only) must not
        crash at bootstrap; ``current_datasource`` simply stays empty."""
        cfg = self._write_base_yaml(tmp_path, {})
        with patch(
            "datus.configuration.agent_config_loader.load_project_override",
            return_value=None,
        ):
            agent_config = load_agent_config(config=str(cfg), home=str(tmp_path), reload=True)
        assert agent_config.current_datasource == ""
