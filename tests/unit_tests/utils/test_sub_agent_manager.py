import os
from pathlib import Path
from typing import Any, Dict

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
