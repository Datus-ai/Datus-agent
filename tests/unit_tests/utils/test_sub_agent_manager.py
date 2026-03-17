import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from datus.schemas.agent_models import ScopedContext, SubAgentConfig
from datus.utils.sub_agent_manager import SubAgentManager


class StubConfigurationManager:
    def __init__(self, base_path: Path):
        self._data: Dict[str, Any] = {"agentic_nodes": {}}
        self.config_path = base_path / "agent.yml"

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def update_item(self, key: str, value, delete_old_key: bool = False):
        self._data[key] = value

    def remove_item_recursively(self, key: str, sub_key: str):
        if key in self._data and isinstance(self._data[key], dict):
            self._data[key].pop(sub_key, None)


class StubPromptManager:
    def __init__(self, base_path: Path):
        self.templates_dir = base_path
        self.user_templates_dir = base_path

    def copy_to(self, template_name: str, destination_name: str, version: str):
        return f"{destination_name}_{version}.j2"


class StubAgentConfig:
    def __init__(self, base_dir: Path):
        self.rag_base_path = str(base_dir)
        self.agentic_nodes = {}

    def rag_storage_path(self) -> str:
        return os.path.join(self.rag_base_path, "global")

    def sub_agent_storage_path(self, sub_agent_name: str) -> str:
        return os.path.join(self.rag_base_path, "sub_agents", sub_agent_name)

    def sub_agent_config(self, sub_agent_name: str):
        return self.agentic_nodes.get(sub_agent_name, {})


def _build_manager(tmp_path):
    config_mgr = StubConfigurationManager(tmp_path)
    agent_config = StubAgentConfig(tmp_path)
    manager = SubAgentManager(configuration_manager=config_mgr, namespace="demo", agent_config=agent_config)
    manager._prompt_manager = StubPromptManager(tmp_path)
    return manager, config_mgr, agent_config


def test_save_agent_rename_preserves_config(tmp_path):
    """When renaming a sub-agent, the old config key is removed and the new key is added."""
    manager, config_mgr, agent_config = _build_manager(tmp_path)

    existing_context = ScopedContext(tables="orders")
    existing_config = SubAgentConfig(system_prompt="old_agent", scoped_context=existing_context)
    config_mgr.update_item("agentic_nodes", {"old_agent": existing_config.as_payload("demo")}, delete_old_key=True)

    updated_context = ScopedContext(tables="orders")
    updated_config = SubAgentConfig(system_prompt="new_agent", scoped_context=updated_context)

    result = manager.save_agent(updated_config, previous_name="old_agent")

    assert result["changed"] is True
    assert "new_agent" in manager.list_agents()
    assert "old_agent" not in manager.list_agents()


def test_remove_agent_removes_config(tmp_path):
    """Removing an agent removes its config entry."""
    manager, config_mgr, agent_config = _build_manager(tmp_path)

    scoped_context = ScopedContext(tables="sales")
    config = SubAgentConfig(system_prompt="cleanup_agent", scoped_context=scoped_context)
    config_mgr.update_item(
        "agentic_nodes",
        {"cleanup_agent": config.as_payload("demo")},
        delete_old_key=True,
    )

    removed = manager.remove_agent("cleanup_agent")

    assert removed is True
    assert "cleanup_agent" not in manager.list_agents()


def test_clear_scoped_kb_is_noop(tmp_path):
    """clear_scoped_kb is now a no-op since sub-agents use global storage."""
    manager, _, _ = _build_manager(tmp_path)
    config = SubAgentConfig(system_prompt="test_agent", scoped_context=ScopedContext(tables="orders"))
    # Should not raise
    manager.clear_scoped_kb(config)


def test_sub_agent_config_with_ext_knowledge(tmp_path):
    """SubAgentConfig with ext_knowledge scoped context is serialized correctly."""
    manager, config_mgr, agent_config = _build_manager(tmp_path)

    context = ScopedContext(ext_knowledge="Finance/Revenue, Sales/*")
    config = SubAgentConfig(system_prompt="knowledge_agent", scoped_context=context)

    assert config.has_scoped_context() is True
    assert config.scoped_context.is_empty is False

    payload = config.as_payload("demo")
    assert "scoped_context" in payload
    assert payload["scoped_context"]["ext_knowledge"] == "Finance/Revenue, Sales/*"


def test_sub_agent_config_ext_knowledge_only_not_empty():
    """ScopedContext with only ext_knowledge is not empty."""
    context = ScopedContext(ext_knowledge="Finance/*")
    config = SubAgentConfig(system_prompt="agent", scoped_context=context)
    assert config.has_scoped_context() is True


# ---------------------------------------------------------------------------
# Stubs (same pattern as existing test file)
# ---------------------------------------------------------------------------


class StubConfigurationManagerExtended:
    def __init__(self, base_path: Path):
        self._data: Dict[str, Any] = {"agentic_nodes": {}}
        self.config_path = base_path / "agent.yml"

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def update_item(self, key: str, value, delete_old_key: bool = False):
        self._data[key] = value

    def remove_item_recursively(self, key: str, sub_key: str):
        if key in self._data and isinstance(self._data[key], dict):
            self._data[key].pop(sub_key, None)


class StubPromptManagerExtended:
    def __init__(self, base_path: Path):
        self.templates_dir = base_path
        self.user_templates_dir = base_path

    def copy_to(self, template_name: str, destination_name: str, version: str):
        return f"{destination_name}_{version}.j2"


class StubAgentConfigExtended:
    def __init__(self, base_dir: Path):
        self.rag_base_path = str(base_dir)
        self.agentic_nodes = {}

    def rag_storage_path(self) -> str:
        return os.path.join(self.rag_base_path, "global")

    def sub_agent_storage_path(self, sub_agent_name: str) -> str:
        return os.path.join(self.rag_base_path, "sub_agents", sub_agent_name)

    def sub_agent_config(self, sub_agent_name: str):
        return self.agentic_nodes.get(sub_agent_name, {})


def _build_managerExtended(tmp_path):
    config_mgr = StubConfigurationManager(tmp_path)
    agent_config = StubAgentConfig(tmp_path)
    manager = SubAgentManager(configuration_manager=config_mgr, namespace="demo", agent_config=agent_config)
    manager._prompt_manager = StubPromptManager(tmp_path)
    return manager, config_mgr, agent_config


# ---------------------------------------------------------------------------
# list_agents
# ---------------------------------------------------------------------------


class TestListAgents:
    def test_empty_agents(self, tmp_path):
        manager, _, _ = _build_managerExtended(tmp_path)
        result = manager.list_agents()
        assert result == {}

    def test_agents_with_system_prompt_defaulted(self, tmp_path):
        """Agents without system_prompt get the key as system_prompt."""
        manager, config_mgr, _ = _build_managerExtended(tmp_path)
        # Inject an agent config without system_prompt key
        config_mgr.update_item("agentic_nodes", {"my_agent": {"prompt_version": "1.0"}})
        agents = manager.list_agents()
        assert agents["my_agent"]["system_prompt"] == "my_agent"

    def test_agents_with_existing_system_prompt(self, tmp_path):
        manager, config_mgr, _ = _build_managerExtended(tmp_path)
        config_mgr.update_item("agentic_nodes", {"my_agent": {"system_prompt": "my_agent", "prompt_version": "1.0"}})
        agents = manager.list_agents()
        assert agents["my_agent"]["system_prompt"] == "my_agent"


# ---------------------------------------------------------------------------
# get_agent
# ---------------------------------------------------------------------------


class TestGetAgent:
    def test_get_existing_agent(self, tmp_path):
        manager, config_mgr, _ = _build_managerExtended(tmp_path)
        config_mgr.update_item("agentic_nodes", {"agent1": {"system_prompt": "agent1", "prompt_version": "1.0"}})
        result = manager.get_agent("agent1")
        assert result is not None
        assert result["system_prompt"] == "agent1"

    def test_get_missing_agent_returns_none(self, tmp_path):
        manager, _, _ = _build_managerExtended(tmp_path)
        result = manager.get_agent("nonexistent")
        assert result is None

    def test_get_agent_returns_deep_copy(self, tmp_path):
        manager, config_mgr, _ = _build_managerExtended(tmp_path)
        config_mgr.update_item("agentic_nodes", {"ag": {"system_prompt": "ag"}})
        result1 = manager.get_agent("ag")
        result2 = manager.get_agent("ag")
        result1["extra"] = "modified"
        assert "extra" not in result2


# ---------------------------------------------------------------------------
# config_path property
# ---------------------------------------------------------------------------


class TestConfigPath:
    def test_config_path_from_manager(self, tmp_path):
        manager, config_mgr, _ = _build_managerExtended(tmp_path)
        assert manager.config_path == config_mgr.config_path


# ---------------------------------------------------------------------------
# save_agent - no change detection
# ---------------------------------------------------------------------------


class TestSaveAgentNoChange:
    def test_unchanged_config_returns_changed_false(self, tmp_path):
        manager, config_mgr, _ = _build_managerExtended(tmp_path)
        context = ScopedContext(tables="orders")
        config = SubAgentConfig(system_prompt="stable_agent", scoped_context=context)
        # Pre-populate with exact payload
        config_mgr.update_item("agentic_nodes", {"stable_agent": config.as_payload("demo")})

        # Save again with same config - previous_name == config.system_prompt
        result = manager.save_agent(config, previous_name="stable_agent")
        assert result["changed"] is False
        assert result["kb_action"] == "unchanged"


# ---------------------------------------------------------------------------
# save_agent - new agent (no previous)
# ---------------------------------------------------------------------------


class TestSaveAgentNew:
    def test_creates_new_agent(self, tmp_path):
        manager, _, _ = _build_managerExtended(tmp_path)
        context = ScopedContext(tables="sales")
        config = SubAgentConfig(system_prompt="new_agent", scoped_context=context)
        result = manager.save_agent(config)
        assert result["changed"] is True
        assert "new_agent" in manager.list_agents()

    def test_prompt_path_set(self, tmp_path):
        manager, _, _ = _build_managerExtended(tmp_path)
        config = SubAgentConfig(system_prompt="agent_x", scoped_context=ScopedContext(tables="t"))
        result = manager.save_agent(config)
        assert result["prompt_path"] is not None

    def test_gen_report_node_uses_gen_report_template(self, tmp_path):
        manager, _, _ = _build_managerExtended(tmp_path)
        # Track which source template was used
        calls = []
        original_copy_to = manager._prompt_manager.copy_to

        def tracking_copy_to(source, dest, version):
            calls.append(source)
            return original_copy_to(source, dest, version)

        manager._prompt_manager.copy_to = tracking_copy_to
        config = SubAgentConfig(
            system_prompt="report_agent",
            node_class="gen_report",
            scoped_context=ScopedContext(tables="t"),
        )
        manager.save_agent(config)
        assert "gen_report_system" in calls

    def test_gen_sql_node_uses_sql_system_template(self, tmp_path):
        manager, _, _ = _build_managerExtended(tmp_path)
        calls = []
        original_copy_to = manager._prompt_manager.copy_to

        def tracking_copy_to(source, dest, version):
            calls.append(source)
            return original_copy_to(source, dest, version)

        manager._prompt_manager.copy_to = tracking_copy_to
        config = SubAgentConfig(
            system_prompt="sql_agent",
            node_class="gen_sql",
            scoped_context=ScopedContext(tables="t"),
        )
        manager.save_agent(config)
        assert "sql_system" in calls


# ---------------------------------------------------------------------------
# save_agent - rename
# ---------------------------------------------------------------------------


class TestSaveAgentRename:
    def test_rename_removes_old_prompt_template(self, tmp_path):
        manager, config_mgr, _ = _build_managerExtended(tmp_path)

        # Create old agent and its prompt file
        old_config = SubAgentConfig(system_prompt="old_name", scoped_context=ScopedContext(tables="t"))
        config_mgr.update_item("agentic_nodes", {"old_name": old_config.as_payload("demo")})

        # Create the old prompt template file
        old_prompt = tmp_path / "old_name_system_1.0.j2"
        old_prompt.write_text("template content")

        new_config = SubAgentConfig(system_prompt="new_name", scoped_context=ScopedContext(tables="t"))
        result = manager.save_agent(new_config, previous_name="old_name")

        assert result["changed"] is True
        assert "new_name" in manager.list_agents()
        assert "old_name" not in manager.list_agents()
        # Old prompt file should be deleted
        assert not old_prompt.exists()


# ---------------------------------------------------------------------------
# remove_agent
# ---------------------------------------------------------------------------


class TestRemoveAgent:
    def test_remove_nonexistent_returns_false(self, tmp_path):
        manager, _, _ = _build_managerExtended(tmp_path)
        result = manager.remove_agent("nonexistent")
        assert result is False

    def test_remove_deletes_prompt_file(self, tmp_path):
        manager, config_mgr, _ = _build_managerExtended(tmp_path)
        config = SubAgentConfig(system_prompt="to_delete", scoped_context=ScopedContext(tables="t"))
        config_mgr.update_item("agentic_nodes", {"to_delete": config.as_payload("demo")})

        # Create the prompt template file
        prompt_file = tmp_path / "to_delete_system_1.0.j2"
        prompt_file.write_text("template")

        result = manager.remove_agent("to_delete")
        assert result is True
        assert "to_delete" not in manager.list_agents()
        assert not prompt_file.exists()

    def test_remove_agent_no_prompt_file(self, tmp_path):
        """Remove succeeds even if prompt file doesn't exist."""
        manager, config_mgr, _ = _build_managerExtended(tmp_path)
        config = SubAgentConfig(system_prompt="no_file_agent", scoped_context=ScopedContext(tables="t"))
        config_mgr.update_item("agentic_nodes", {"no_file_agent": config.as_payload("demo")})
        result = manager.remove_agent("no_file_agent")
        assert result is True


# ---------------------------------------------------------------------------
# _remove_prompt_template
# ---------------------------------------------------------------------------


class TestRemovePromptTemplate:
    def test_removes_existing_file(self, tmp_path):
        manager, _, _ = _build_managerExtended(tmp_path)
        prompt_file = tmp_path / "myagent_system_1.0.j2"
        prompt_file.write_text("content")
        manager._remove_prompt_template("myagent", "1.0")
        assert not prompt_file.exists()

    def test_no_op_when_file_missing(self, tmp_path):
        manager, _, _ = _build_managerExtended(tmp_path)
        # Should not raise
        manager._remove_prompt_template("nonexistent_agent", "1.0")


# ---------------------------------------------------------------------------
# _write_prompt_template
# ---------------------------------------------------------------------------


class TestWritePromptTemplate:
    def test_returns_path_string(self, tmp_path):
        manager, _, _ = _build_managerExtended(tmp_path)
        config = SubAgentConfig(system_prompt="myagent", scoped_context=ScopedContext(tables="t"))
        result = manager._write_prompt_template(config)
        assert isinstance(result, str)
        assert "myagent" in result

    def test_raises_on_ioerror(self, tmp_path):
        manager, _, _ = _build_managerExtended(tmp_path)
        config = SubAgentConfig(system_prompt="myagent", scoped_context=ScopedContext(tables="t"))
        manager._prompt_manager.copy_to = MagicMock(side_effect=IOError("write failed"))
        with pytest.raises(IOError):
            manager._write_prompt_template(config)


# ---------------------------------------------------------------------------
# bootstrap_agent
# ---------------------------------------------------------------------------


class TestBootstrapAgent:
    def test_bootstrap_calls_bootstrapper_run(self, tmp_path):
        manager, _, _ = _build_managerExtended(tmp_path)
        config = SubAgentConfig(system_prompt="boot_agent", scoped_context=ScopedContext(tables="t"))
        mock_result = MagicMock()
        mock_bootstrapper = MagicMock()
        mock_bootstrapper.run.return_value = mock_result
        with patch.object(manager, "_build_bootstrapper", return_value=mock_bootstrapper):
            result = manager.bootstrap_agent(config)
        mock_bootstrapper.run.assert_called_once_with(None, "plan")
        assert result is mock_result

    def test_bootstrap_with_components(self, tmp_path):
        manager, _, _ = _build_managerExtended(tmp_path)
        config = SubAgentConfig(system_prompt="agent", scoped_context=ScopedContext(tables="t"))
        mock_bootstrapper = MagicMock()
        with patch.object(manager, "_build_bootstrapper", return_value=mock_bootstrapper):
            manager.bootstrap_agent(config, components=["schema", "sql"])
        mock_bootstrapper.run.assert_called_once_with(["schema", "sql"], "plan")

    def test_bootstrap_with_strategy(self, tmp_path):
        manager, _, _ = _build_managerExtended(tmp_path)
        config = SubAgentConfig(system_prompt="agent", scoped_context=ScopedContext(tables="t"))
        mock_bootstrapper = MagicMock()
        with patch.object(manager, "_build_bootstrapper", return_value=mock_bootstrapper):
            manager.bootstrap_agent(config, strategy="build")
        mock_bootstrapper.run.assert_called_once_with(None, "build")


# ---------------------------------------------------------------------------
# clear_scoped_kb
# ---------------------------------------------------------------------------


class TestClearScopedKb:
    def test_is_noop(self, tmp_path):
        manager, _, _ = _build_managerExtended(tmp_path)
        config = SubAgentConfig(system_prompt="agent", scoped_context=ScopedContext(tables="t"))
        # Should complete without error and return None
        result = manager.clear_scoped_kb(config)
        assert result is None

    def test_noop_with_none(self, tmp_path):
        manager, _, _ = _build_managerExtended(tmp_path)
        result = manager.clear_scoped_kb(None)
        assert result is None
