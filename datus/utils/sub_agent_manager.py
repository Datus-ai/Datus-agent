from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from datus.configuration.agent_config_loader import ConfigurationManager
from datus.prompts.prompt_manager import PromptManager, prompt_manager
from datus.schemas.agent_models import SubAgentConfig
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SubAgentManager:
    """Encapsulates sub-agent configuration and prompt management operations."""

    def __init__(self, configuration_manager: ConfigurationManager, namespace: str):
        self._configuration_manager = configuration_manager
        self._prompt_manager: PromptManager = prompt_manager
        self._namespace = namespace

    @property
    def config_path(self) -> Path:
        return self._configuration_manager.config_path

    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        agents = self._configuration_manager.get("agentic_nodes", {})
        return agents if isinstance(agents, dict) else {}

    def get_agent(self, agent_name: str) -> Optional[Dict[str, Any]]:
        agents = self.list_agents()
        config = agents.get(agent_name)
        return deepcopy(config) if config else None

    def save_agent(self, config: SubAgentConfig, previous_name: Optional[str] = None) -> Dict[str, str]:
        """Persist the given sub-agent configuration.

        Args:
            config: New configuration to persist.
            previous_name: Existing agent name when updating/renaming.
        """
        agent_payload = self._build_agent_payload(config)

        agents = dict(self.list_agents())
        previous_config: Optional[Dict[str, Any]] = None
        if previous_name:
            previous_config = agents.get(previous_name)
        if previous_name and previous_name != config.system_prompt:
            prompt_version = str(config.prompt_version)
            if previous_config:
                prompt_version = str(previous_config.get("prompt_version", prompt_version))
            self._remove_prompt_template(previous_name, prompt_version)
            agents.pop(previous_name, None)
        agents[config.system_prompt] = agent_payload

        self._configuration_manager.update_item("agentic_nodes", agents, delete_old_key=True)
        prompt_path = self._write_prompt_template(config)

        return {
            "config_path": str(self.config_path),
            "prompt_path": prompt_path,
        }

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

    def _build_agent_payload(self, config: SubAgentConfig) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "system_prompt": config.system_prompt,
            "prompt_version": config.prompt_version,
            "prompt_language": config.prompt_language,
            "agent_description": config.agent_description,
            "tools": config.tools,
            "mcp": config.mcp,
            "rules": list(config.rules or []),
        }

        if config.scoped_context:
            scoped_context = config.scoped_context.model_dump(exclude_none=True)
            if scoped_context:
                payload["scoped_context"] = scoped_context

        return payload

    def _write_prompt_template(self, config: SubAgentConfig) -> str:
        try:
            file_name = self._prompt_manager.copy_to(
                "sql_system", f"{config.system_prompt}_system", config.prompt_version
            )
        except IOError as exc:
            logger.error("Failed to write prompt template for '%s': %s", config.system_prompt, exc)
            raise
        return str(self._prompt_manager.templates_dir / file_name)

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
