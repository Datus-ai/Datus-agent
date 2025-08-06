import os
from pathlib import Path

import yaml

from datus.configuration.agent_config import AgentConfig, NodeConfig
from datus.configuration.node_type import NodeType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def load_node_config(node_type: str, data: dict) -> NodeConfig:
    if data and isinstance(data, dict) and "model" in data.keys():
        model = data.pop("model")
        return NodeConfig(model=model, input=NodeType.type_input(node_type, data, ignore_require_check=True))
    else:
        return NodeConfig(model="", input=NodeType.type_input(node_type, data, ignore_require_check=True))


def load_agent_config(**kwargs) -> AgentConfig:
    # Check config file in order: kwargs["config"] > conf/agent.yml > ~/.datus/conf/agent.yml
    # Load .env file if it exists
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    yaml_path = None
    if config_path := kwargs.get("config"):
        yaml_path = os.path.expanduser(config_path)

    if not yaml_path and os.path.exists("conf/agent.yml"):
        yaml_path = "conf/agent.yml"

    if not yaml_path:
        home_config = Path.home() / ".datus" / "conf" / "agent.yml"
        if os.path.exists(home_config):
            yaml_path = home_config

    if not yaml_path:
        raise DatusException(
            code=ErrorCode.COMMON_FILE_NOT_FOUND,
            message=(
                "Agent configuration file not found. Please configure your `conf/agent.yaml` or `.datus/conf/agent.yml`"
                ". You can also use --config <your_config_file_path>"
            ),
        )

    if not os.path.exists(yaml_path):
        raise DatusException(
            code=ErrorCode.COMMON_FILE_NOT_FOUND,
            message_args={"config_name": "Agent configuration", "file_name": yaml_path},
        )
    with open(yaml_path, "r") as f:
        logger.info(f"Loading agent config from {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    agent_raw = config["agent"]
    nodes = {}
    if "nodes" in agent_raw:
        nodes_raw = agent_raw["nodes"]
        if isinstance(nodes_raw, str):
            if nodes_raw not in NodeType.ACTION_TYPES:
                raise DatusException(
                    ErrorCode.COMMON_FIELD_INVALID,
                    message_args={
                        "field_name": "Node Type",
                        "except_values": set(NodeType.ACTION_TYPES) | {NodeType.TYPE_REFLECT},
                        "your_value": nodes_raw,
                    },
                )
        for node_type, cfg in nodes_raw.items():
            if node_type == NodeType.TYPE_REFLECT:
                pass
            elif node_type not in NodeType.ACTION_TYPES:
                raise DatusException(
                    ErrorCode.COMMON_FIELD_INVALID,
                    message_args={
                        "field_name": "Node Type",
                        "except_values": set(NodeType.ACTION_TYPES) | {NodeType.TYPE_REFLECT},
                        "your_value": node_type,
                    },
                )
            nodes[node_type] = load_node_config(node_type, cfg)
        del agent_raw["nodes"]

    agent_config = AgentConfig(nodes=nodes, **agent_raw)
    if kwargs:
        agent_config.override_by_args(**kwargs)
    return agent_config
