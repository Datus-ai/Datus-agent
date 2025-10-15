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

    def sub_agent_storage_path(self, sub_agent_name: str) -> str:
        return os.path.join(self.rag_base_path, "sub_agents", sub_agent_name)


def _build_manager(tmp_path):
    config_mgr = StubConfigurationManager(tmp_path)
    agent_config = StubAgentConfig(tmp_path)
    manager = SubAgentManager(configuration_manager=config_mgr, namespace="demo", agent_config=agent_config)
    manager._prompt_manager = StubPromptManager(tmp_path)
    return manager, config_mgr, agent_config


def test_save_agent_renames_scoped_kb_directory(tmp_path):
    manager, config_mgr, agent_config = _build_manager(tmp_path)

    existing_context = ScopedContext(tables="orders")
    existing_config = SubAgentConfig(system_prompt="old_agent", scoped_context=existing_context)
    existing_config.scoped_kb_path = agent_config.sub_agent_storage_path("old_agent")
    old_path = Path(existing_config.scoped_kb_path)
    old_path.mkdir(parents=True)
    (old_path / "placeholder.txt").write_text("keep me")
    config_mgr.update_item("agentic_nodes", {"old_agent": existing_config.as_payload("demo")}, delete_old_key=True)

    updated_context = ScopedContext(tables="orders")
    updated_config = SubAgentConfig(system_prompt="new_agent", scoped_context=updated_context)
    updated_config.scoped_kb_path = existing_config.scoped_kb_path

    result = manager.save_agent(updated_config, previous_name="old_agent")

    assert result["kb_action"] == "renamed"
    assert not old_path.exists()
    new_path = Path(agent_config.sub_agent_storage_path("new_agent"))
    assert new_path.exists()
    assert (new_path / "placeholder.txt").exists()
    assert updated_config.scoped_kb_path == str(new_path)
    assert "new_agent" in manager.list_agents()
    assert "old_agent" not in manager.list_agents()


def test_remove_agent_clears_scoped_kb_directory(tmp_path):
    manager, config_mgr, agent_config = _build_manager(tmp_path)

    scoped_context = ScopedContext(tables="sales")
    config = SubAgentConfig(system_prompt="cleanup_agent", scoped_context=scoped_context)
    kb_path = Path(agent_config.sub_agent_storage_path("cleanup_agent"))
    config.scoped_kb_path = str(kb_path)
    kb_path.mkdir(parents=True)
    config_mgr.update_item(
        "agentic_nodes",
        {"cleanup_agent": config.as_payload("demo")},
        delete_old_key=True,
    )

    removed = manager.remove_agent("cleanup_agent")

    assert removed is True
    assert not kb_path.exists()
    assert "cleanup_agent" not in manager.list_agents()
