import threading
from typing import Any, Dict

from agents.mcp import (
    MCPServerSse,
    MCPServerSseParams,
    MCPServerStdioParams,
    MCPServerStreamableHttp,
    MCPServerStreamableHttpParams,
)

from datus.configuration.agent_config import DbConfig
from datus.configuration.mcp_config import McpConfig
from datus.configuration.mcp_type import McpType
from datus.tools.mcp_server import SilentMCPServerStdio
from datus.utils.exceptions import DatusException, ErrorCode


class McpServerClient:
    _lock = threading.Lock()
    _mcp_server_client_pool: Dict[str, Any] = {}

    @classmethod
    def get_client_pool(cls) -> Dict[str, Any]:
        return cls._mcp_server_client_pool

    @classmethod
    def get_mcp_server_client(cls, server_name: str, db_config: DbConfig):
        if server_name in cls._mcp_server_client_pool and cls._mcp_server_client_pool[server_name]:
            return cls._mcp_server_client_pool[server_name]

        config = McpConfig.get_instance().get_servers().get(server_name)
        if config:
            with cls._lock:
                config = McpConfig.get_instance().get_servers().get(server_name)
                if config:
                    client = cls.create_mcp_server_client(db_config, config)
                    cls._mcp_server_client_pool[server_name] = client
                    return client
        raise DatusException(ErrorCode.MCP_SERVER_NOT_FOUND, f"Server {server_name} not found")

    @classmethod
    def add_mcp_client(cls, server_name: str, server_type: str, **kwargs):
        try:
            McpConfig.check_required_params(server_type, **kwargs)
            client = cls.create_mcp_server_client(DbConfig(), **kwargs)
            cls._mcp_server_client_pool[server_name] = client
        except DatusException as e:
            raise DatusException(ErrorCode.MCP_SERVER_NOT_FOUND, str(e))

    @classmethod
    def create_mcp_server_client(cls, db_config: DbConfig, config: Dict[str, Any]):
        # todo: how to set Dbconfig to mcp server client
        if config["type"] == McpType.STDIO:
            mcp_server_params = MCPServerStdioParams(
                command=config["command"],
                args=config["args"],
                env=config["env"],
            )
            mcp_server_client = SilentMCPServerStdio(params=mcp_server_params, client_session_timeout_seconds=30)
            return mcp_server_client
        elif config["type"] == McpType.SSE:
            mcp_server_params = MCPServerSseParams(
                url=config["url"],
                headers=config["headers"],
                timeout=config["timeout"],
                # The timeout for the SSE connection, in seconds. Defaults to 5 minutes
                sse_read_timeout=300,
            )
            mcp_server_client = MCPServerSse(params=mcp_server_params, client_session_timeout_seconds=30)
            return mcp_server_client
        elif config["type"] == McpType.STREAM_HTTP:
            mcp_server_params = MCPServerStreamableHttpParams(
                url=config["url"],
                headers=config["headers"],
                timeout=config["timeout"],
            )
            mcp_server_client = MCPServerStreamableHttp(params=mcp_server_params, client_session_timeout_seconds=30)
            return mcp_server_client
        else:
            raise DatusException(ErrorCode.MCP_TYPE_NOT_FOUND, f"Unknown server type: {config['type']}")

    @classmethod
    def rm_mcp_client(cls, server_namae):
        if server_namae in McpConfig.get_instance().get_servers():
            McpConfig.get_instance().get_servers().pop(server_namae)
        if server_namae in McpServerClient.get_client_pool():
            McpServerClient.get_client_pool().pop(server_namae)
