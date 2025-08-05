import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

from datus.configuration.mcp_type import McpType
from datus.utils.json_utils import resolve_json_env
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

MCP_SERVERS = "mcp_servers"
MCP_CONF_DIR = "conf"
MCP_CONF_FILE = "mcp.json"


@dataclass
class ServerConfig:
    # name: str
    # server type: stdio, sse, http
    type: str

    def __init__(self, **kwargs):
        self.type = kwargs.get("type", "")


@dataclass
class StdioServerConfig(ServerConfig):
    # studio config
    command: str = ""
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.command = resolve_json_env(kwargs.get("command", ""))
        self.args = kwargs.get("args", [])
        self.env = kwargs.get("env", {})


@dataclass
class SSEServerConfig(ServerConfig):
    # sse or http config
    url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    # HTTP request timeout unit second
    timeout: int = 30

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.url = kwargs.get("url", "")
        self.headers = kwargs.get("headers", {})
        self.timeout = kwargs.get("timeout", 30)


@dataclass
class StreamedHTTPConfig(ServerConfig):
    # sse or http config
    url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    # HTTP request timeout unit second
    timeout: int = 30

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.url = kwargs.get("url", "")
        self.headers = kwargs.get("headers", {})
        self.timeout = kwargs.get("timeout", 30)


@dataclass
class McpConfig:
    # raw_servers retain  the env variable unresolved, like {VAR} or {VAR:-default}, it is used when updating mcp.json
    __raw_servers: Dict[str, Dict[str, Any]] = Field(default_factory={}, description="Raw server configurations")
    __servers: Dict[str, Dict[str, Any]] = Field(default_factory={}, description="Server configurations")

    _instance: Optional["McpConfig"] = None  # store the unique instance
    _initialized: bool = False  # whether the instance is initialized

    def __new__(cls, *args, **kwargs):
        """overwrite __new__ to ensure a globally unique instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)  # the first call: create instance
        return cls._instance  # not the first call: return existing instance

    def __init__(self, raw_servers: Dict[str, Dict[str, Any]], servers: Dict[str, Dict[str, Any]]):
        """
        initialize the global config from mcp.json file
        """
        if McpConfig._initialized:
            return  # already initialized: skip
        if raw_servers:
            self.__raw_servers = raw_servers
        if servers:
            self.__servers = servers

        McpConfig._initialized = True  # mark as initialized

    def get_servers(self) -> Dict[str, Dict[str, Any]]:
        return self.__servers

    def get_raw_servers(self) -> Dict[str, Dict[str, Any]]:
        return self.__raw_servers

    @classmethod
    def get_instance(cls) -> "McpConfig":
        """get the unique instance of McpConfig"""
        if cls._instance is None:
            cls._instance = McpConfig(*cls.load_mcp_config())  # create instance if not exist
        return cls._instance  # return existing instance

    @classmethod
    def get_server_configs(cls, json_config) -> Dict[str, Dict[str, Any]]:
        server_config = {}
        for server_name, server_data in json_config.items():
            if server_data["type"] not in [McpType.STDIO, McpType.SSE, McpType.STREAM_HTTP]:
                raise ValueError(f"Unknown server type: {server_data.type}")
            server_config[server_name] = server_data
        return server_config

    @classmethod
    def load_mcp_config(cls, **kwargs) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        load mcp config from mcp.json file
        """
        # Check mcp.json file in order: kwargs["mcp_config"] > conf/mcp.json > ~/.datus/conf/mcp.json
        # Load .env file if it exists
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass
        mcp_json_path = None
        if "mcp_config" in kwargs and kwargs["mcp_config"]:
            mcp_json_path = kwargs["mcp_config"]

        if not mcp_json_path and os.path.exists("conf/mcp.json"):
            mcp_json_path = "conf/mcp.json"

        if not mcp_json_path:
            home_config = Path.home() / ".datus" / "conf" / "mcp.json"
            if os.path.exists(home_config):
                mcp_json_path = home_config

        if not mcp_json_path:
            logger.warning(
                "MCP configuration file not found. If you want to register mcp servers in advance to use , "
                "please configure your `conf/mcp.json` or `.datus/conf/mcp.json`"
                ". You can also use --mcp_config <your_config_file_path>"
            )
            return {}, {}
        else:
            with open(mcp_json_path, "r") as f:
                # 1. load mcp_json file
                logger.info(f"Loading mcp config from {mcp_json_path}")
                raw_json_str = ""
                json_str = ""
                for line in f:
                    raw_json_str += line
                    json_str += resolve_json_env(line)
                # raw_json_str contain the env variable unresolved, like {VAR} or {VAR:-default}
                raw_json_config = json.loads(raw_json_str)
                json_config = json.loads(json_str)

                # 2. get server config
                raw_server_configs: Dict[str, Dict[str, Any]] = cls.get_server_configs(
                    raw_json_config.get(MCP_SERVERS, {})
                )
                server_configs: Dict[str, Dict[str, Any]] = cls.get_server_configs(json_config.get(MCP_SERVERS, {}))
                return raw_server_configs, server_configs

    def update_then_save_config(self, new_servers: Dict[str, Dict[str, Any]], file_name: str = "mcp.json"):
        """
        save the config to mcp.json file
        """
        os.makedirs(MCP_CONF_DIR, exist_ok=True)
        config_path = os.path.join(MCP_CONF_DIR, file_name)
        if new_servers:
            self.__raw_servers.update(new_servers)
            self.__servers.update(new_servers)
        servers_dict = {name: server for name, server in self.__raw_servers.items()}
        mcp_servers = {MCP_SERVERS: servers_dict}
        with open(config_path, "w") as f:
            json.dump(mcp_servers, f, indent=2)
        pass
