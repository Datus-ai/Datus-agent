from enum import Enum
from typing import Dict, List

from datus.configuration.mcp_config import McpConfig, ServerConfig
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class Mode(str, Enum):
    ON = "on"
    OFF = "off"


class OperateMCPTools:
    def __init__(self, mcp_config: McpConfig, **kwargs):
        self.mcp_config = mcp_config

    def check_mcp_servers(self, servers: List[str]):
        # todo check logic
        pass

    def list_servers(self) -> Dict[str, bool]:
        # 1. get all servers from mcp config
        # 2. check mcp server status
        connected = True
        return {"xxx": connected}

    def get_mcp_server(self, servers: List[str]) -> Dict[str, ServerConfig]:
        # todo get server details
        # 1. check mcp server status
        # 2. get server details from mcp config
        # 3. get tools & capabilities from mcp server
        return {}

    def get_tools(self, server: str) -> List[str]:
        # todo get tools from mcp server
        return []

    def call_tool(self, server: str, tool: str, **kwargs) -> str:
        # todo call tool from mcp server
        return ""

    def add_or_rm_mcp_server(self, server: str, flag: bool = True, **kwargs) -> str:
        return ""
