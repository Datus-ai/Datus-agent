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
    configuration_manager,
    load_node_config,
    parse_config_path,
)
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
