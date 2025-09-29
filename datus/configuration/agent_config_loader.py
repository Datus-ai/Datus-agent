import os
from pathlib import Path
from typing import Any, Dict

import yaml

from datus.configuration.agent_config import AgentConfig, NodeConfig
from datus.configuration.node_type import NodeType
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def load_node_config(node_type: str, data: dict) -> NodeConfig:
    if data and isinstance(data, dict) and "model" in data.keys():
        model = data.pop("model")
        return NodeConfig(model=model, input=NodeType.type_input(node_type, data, ignore_require_check=True))
    else:
        return NodeConfig(model="", input=NodeType.type_input(node_type, data, ignore_require_check=True))


class ConfigurationManager:
    def __init__(self, config_path: str = ""):
        self.config_path: Path = parse_config_path(config_path)

        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def update(self, updates: Dict[str, Any], delete_old_key: bool = False, save: bool = True) -> bool:
        try:
            for key, value in updates.items():
                self.update_item(key, value, delete_old_key, False)
            if save:
                self.save()
            return True
        except Exception as e:
            print(f"Error updating YAML file: {e}")
            return False

    def update_item(self, key: str, value: Any, delete_old_key: bool = False, save: bool = True) -> bool:
        try:
            if delete_old_key:
                self.data[key] = value
            elif isinstance(value, dict) and key in self.data:
                self.data[key].update(value)
            else:
                self.data[key] = value
            if save:
                self.save()
            return True
        except Exception as e:
            print(f"Error updating YAML file: {e}")
            return False

    def remove_item_recursively(self, *keys) -> bool:
        """
        Delete recursively the corresponding keys.
        Example:
            keys = ['a', 'b', 'c'], The deleted item should be self.data['a']['b']['c']
        """
        if not keys:
            return False
        key_path = []
        temp_data = self.data
        for key in keys[:-1]:
            key_path.append(key)
            if key not in temp_data:
                error_path = ".".join(key_path)
                raise DatusException(
                    ErrorCode.COMMON_FIELD_INVALID,
                    message=f"The key path '{error_path}' does not exist in the configuration data. ",
                )
            temp_data = temp_data[key]
        del temp_data[keys[-1]]
        self.save()
        return True

    def save(self):
        with open(self.config_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(self.data, file, allow_unicode=True, sort_keys=False)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any):
        self.data[key] = value
        self.save()


CONFIGURATION_MANAGER: ConfigurationManager | None = None


def configuration_manager(config_path: str = "", reload: bool = False) -> ConfigurationManager:
    global CONFIGURATION_MANAGER
    if reload or not CONFIGURATION_MANAGER:
        CONFIGURATION_MANAGER = ConfigurationManager(config_path)
    return CONFIGURATION_MANAGER


def parse_config_path(config_file: str = "") -> Path:
    if config_file:
        config_path = Path(config_file).expanduser()
        if config_path.exists():
            return config_path
        elif config_file != "conf/agent.yml":
            # default config file
            raise DatusException(
                code=ErrorCode.COMMON_FILE_NOT_FOUND, message=f"Agent configuration file not found: {config_path}"
            )
    if os.path.exists("conf/agent.yml"):
        return Path("conf/agent.yml")
    home_config = Path.home() / ".datus" / "conf" / "agent.yml"
    if home_config.exists():
        return home_config

    raise DatusException(
        code=ErrorCode.COMMON_FILE_NOT_FOUND,
        message=(
            "Agent configuration file not found. Please configure your `conf/agent.yaml` or `.datus/conf/agent.yml`"
            ". You can also use --config <your_config_file_path>"
        ),
    )


def load_agent_config(reload: bool = False, **kwargs) -> AgentConfig:
    # Check config file in order: kwargs["config"] > conf/agent.yml > ~/.datus/conf/agent.yml
    # Load .env file if it exists
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    config = configuration_manager(config_path=kwargs.get("config", ""), reload=reload)
    agent_raw = config.get("agent")
    nodes = {}
    if "nodes" in agent_raw:
        nodes_raw = agent_raw["nodes"]
        if isinstance(nodes_raw, str):
            if nodes_raw not in NodeType.ACTION_TYPES:
                raise DatusException(
                    ErrorCode.COMMON_FIELD_INVALID,
                    message_args={
                        "field_name": "Node Type",
                        "except_values": set(NodeType.ACTION_TYPES)
                        | set(NodeType.AGENTIC_TYPES)
                        | {NodeType.TYPE_REFLECT},
                        "your_value": nodes_raw,
                    },
                )
        for node_type, cfg in nodes_raw.items():
            if node_type == NodeType.TYPE_REFLECT:
                pass
            elif node_type not in NodeType.ACTION_TYPES and node_type not in NodeType.AGENTIC_TYPES:
                raise DatusException(
                    ErrorCode.COMMON_FIELD_INVALID,
                    message_args={
                        "field_name": "Node Type",
                        "except_values": set(NodeType.ACTION_TYPES)
                        | set(NodeType.AGENTIC_TYPES)
                        | {NodeType.TYPE_REFLECT},
                        "your_value": node_type,
                    },
                )
            nodes[node_type] = load_node_config(node_type, cfg)
        del agent_raw["nodes"]

    agent_config = AgentConfig(nodes=nodes, **agent_raw)
    if kwargs:
        # Filter out the 'config' parameter as it's only used for loading, not for overriding
        override_kwargs = {k: v for k, v in kwargs.items() if k != "config"}
        if override_kwargs:
            agent_config.override_by_args(**override_kwargs)
    if agent_config.db_type in {DBType.SQLITE, DBType.DUCKDB} and not agent_config.current_database:
        current_configs = agent_config.current_db_configs()
        agent_config.current_database = current_configs[list(current_configs.keys())[0]].database
    return agent_config
