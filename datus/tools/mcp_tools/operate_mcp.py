import asyncio
from typing import Any, Dict, List

from datus.configuration.mcp_config import McpConfig
from datus.models.mcp_utils import CALL_TOOL, LIST_SERVERS, LIST_TOOLS, exe_mcp_cli_func
from datus.tools.mcp_tools.mcp_server_client import McpServerClient
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class OperateMCPTools:
    _mcp_config: McpConfig

    def __init__(self, mcp_config: McpConfig, **kwargs):
        self._mcp_config = mcp_config

    async def list_servers(self) -> Dict[str, bool]:
        # 1. get all mcp clients
        server_client_dict = McpServerClient.get_client_pool()
        result = asyncio.run(exe_mcp_cli_func(server_client_dict, LIST_SERVERS))
        return result

    def get_mcp_server(self, server_name: str) -> Dict[str, Any]:
        server_config = {"Server_name": server_name, "Status": False}
        server_info = self._mcp_config.get_servers()[server_name]
        if server_info:
            server_config = server_info
        server_config["Capabilities"] = "tools"
        server_config["Tools"] = len(self.list_tools(server_name))
        server_config["Status"] = True
        return server_config

    def list_tools(self, server_namae: str) -> List[str]:
        server_client_dict = self.get_mcp_client_dict(server_namae)
        result = asyncio.run(exe_mcp_cli_func(server_client_dict, LIST_TOOLS))
        return result[server_namae]

    def call_tool(self, server_namae: str, tool: str, **kwargs):
        # todo call tool from mcp server
        server_client_dict = self.get_mcp_client_dict(server_namae)
        result = asyncio.run(exe_mcp_cli_func(server_client_dict, CALL_TOOL))
        return result[server_namae]

    def add_mcp_client(self, server_namae: str, server_type: str, **kwargs) -> bool:
        McpServerClient.add_mcp_client(server_namae, server_type, **kwargs)
        return True

    def rm_mcp_client(self, server_namae: str) -> bool:
        McpServerClient.rm_mcp_client(server_namae)
        return True

    def get_mcp_client_dict(self, server_name: str):
        client_dict = {}
        client_pool = McpServerClient.get_client_pool()
        if server_name in client_pool:
            client_dict[server_name] = client_pool[server_name]
            return client_dict
        else:
            raise DatusException(ErrorCode.MCP_SERVER_NOT_FOUND, f"Server {server_name} not found")
