# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import ConfigurationManager
from datus.prompts.prompt_manager import PromptManager, prompt_manager
from datus.schemas.agent_models import SubAgentConfig
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SubAgentManager:
    """Encapsulates sub-agent configuration and prompt management operations."""

    def __init__(
        self,
        configuration_manager: ConfigurationManager,
        namespace: str,
        agent_config: AgentConfig,
    ):
        self._configuration_manager = configuration_manager
        self._prompt_manager: PromptManager = prompt_manager
        self._namespace = namespace
        self._agent_config = agent_config

    @property
    def config_path(self) -> Path:
        return self._configuration_manager.config_path

    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        agents = self._configuration_manager.get("agentic_nodes") or {}
        for name, raw_config in agents.items():
            if not raw_config.get("system_prompt"):
                raw_config["system_prompt"] = name
        return agents

    def get_agent(self, agent_name: str) -> Optional[Dict[str, Any]]:
        agents = self.list_agents()
        config = agents.get(agent_name)
        return deepcopy(config) if config else None

    def save_agent(self, config: SubAgentConfig, previous_name: Optional[str] = None) -> Dict[str, Any]:
        """Persist the given sub-agent configuration.

        Args:
            config: New configuration to persist.
            previous_name: Existing agent name when updating/renaming.
        """

        agents = dict(self.list_agents())
        previous_config: Optional[Dict[str, Any]] = None
        previous_key = previous_name or config.system_prompt
        if previous_key in agents:
            previous_config = agents.get(previous_key)

        payload = config.as_payload(self._namespace)
        result: Dict[str, Any] = {
            "config_path": str(self.config_path),
            "prompt_path": None,
            "changed": True,
            "kb_action": "none",
        }

        if previous_config and previous_name == config.system_prompt and previous_config == payload:
            result["changed"] = False
            result["kb_action"] = "unchanged"
            return result

        # Handle renaming: remove old prompt template and config key
        if previous_name and previous_name != config.system_prompt and previous_config:
            old_prompt_version = str(previous_config.get("prompt_version") or config.prompt_version or "1.0")
            self._remove_prompt_template(previous_name, old_prompt_version)
            agents.pop(previous_name, None)

        agents[config.system_prompt] = config.as_payload(self._namespace)

        self._configuration_manager.update_item("agentic_nodes", agents, delete_old_key=True)
        self._agent_config.agentic_nodes = agents

        prompt_path = self._write_prompt_template(config)
        result["prompt_path"] = prompt_path

        return result

    def remove_agent(self, agent_name: str) -> bool:
        agents = self.list_agents()
        if agent_name not in agents:
            return False
        sub_agent = agents[agent_name]
        try:
            self._configuration_manager.remove_item_recursively("agentic_nodes", agent_name)
            prompt_version = str(sub_agent.get("prompt_version", "1.0"))
            self._remove_prompt_template(agent_name, prompt_version)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to remove agent '%s': %s", agent_name, exc)
            raise
        return True

    def _write_prompt_template(self, config: SubAgentConfig) -> str:
        # Select source template based on node_class
        node_class = config.node_class or "gen_sql"
        if node_class == "gen_report":
            source_template = "gen_report_system"
        else:
            source_template = "sql_system"

        try:
            file_name = self._prompt_manager.copy_to(
                source_template, f"{config.system_prompt}_system", config.prompt_version
            )
        except IOError as exc:
            logger.error("Failed to write prompt template for '%s': %s", config.system_prompt, exc)
            raise
        return str(self._prompt_manager.user_templates_dir / file_name)

    def _remove_prompt_template(self, agent_name: str, prompt_version: str):
        file_name = f"{agent_name}_system_{prompt_version}.j2"
        file_path = self._prompt_manager.user_templates_dir / file_name

        if not file_path.exists():
            return
        try:
            file_path.unlink()
        except OSError as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to delete prompt template '%s': %s", file_path, exc)
        else:
            return
