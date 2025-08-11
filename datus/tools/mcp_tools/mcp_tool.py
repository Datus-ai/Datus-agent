"""
MCP Tool - Tool class for MCP server management operations.

This module provides the MCPTool class that implements the BaseTool interface
for MCP server management operations. It acts as a wrapper around MCPManager
providing a standardized tool interface.
"""
import json
import re
import shlex
from typing import Any, Dict, List, Optional, Tuple

from datus.tools.base import BaseTool, BaseToolExecResult, ToolAction
from datus.tools.mcp_tools.mcp_manager import MCPManager
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class MCPTool(BaseTool):
    """Tool class for MCP server management operations."""

    tool_name = "mcp_tool"
    tool_description = "Management tool for MCP (Model Context Protocol) servers"

    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize the MCP tool.

        Args:
            config_path: Path to MCP config file
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.manager = MCPManager(config_path=config_path)
        logger.info(f"Initialized MCP Tool with config path: {self.manager.config_path}")

    @ToolAction(description="Add a new MCP server config")
    def add_server(
        self,
        name: str,
        server_type: str,
        **config_params,
    ) -> BaseToolExecResult:
        """
        Add a new MCP server config.

        Args:
            name: Server name/identifier
            server_type: Server type (stdio, sse, http)
            **config_params: Server type specific config parameters

        Returns:
            BaseToolExecResult with operation result
        """
        try:
            # Prepare config data for server creation
            config_data = {"type": server_type, **config_params}

            # Create server config using the factory method
            from .mcp_config import MCPServerConfig

            server_config = MCPServerConfig.from_config_format(name, config_data)

            success, message = self.manager.add_server(server_config)

            result_data = {}
            if success:
                result_data.update(server_config.model_dump())

            return BaseToolExecResult(success=success, message=message, result=result_data if success else None)

        except Exception as e:
            logger.error(f"Error in add_server: {e}")
            return BaseToolExecResult(success=False, message=f"Error adding server: {e}")

    @ToolAction(description="Remove an MCP server config")
    def remove_server(self, name: str) -> BaseToolExecResult:
        """
        Remove an MCP server config.

        Args:
            name: Server name

        Returns:
            BaseToolExecResult with operation result
        """
        try:
            success, message = self.manager.remove_server(name)

            return BaseToolExecResult(
                success=success,
                message=message,
                result={"removed_server": name} if success else None,
            )

        except Exception as e:
            logger.error(f"Error in remove_server: {e}")
            return BaseToolExecResult(success=False, message=f"Error removing server: {e}")

    @ToolAction(description="List MCP server configs")
    def list_servers(self, server_type: Optional[str] = None) -> BaseToolExecResult:
        """
        List MCP server configs with optional filtering.

        Args:
            server_type: Filter by server type (stdio, sse, http)

        Returns:
            BaseToolExecResult with list of server configs
        """
        try:
            servers = self.manager.list_servers(server_type=server_type)

            # Convert to dict format for result
            server_list = []
            for server in servers:
                server_dict = server.model_dump()
                server_list.append(server_dict)

            return BaseToolExecResult(
                success=True,
                message=f"Found {len(server_list)} servers",
                result={"servers": server_list, "total_count": len(server_list)},
            )

        except Exception as e:
            logger.error(f"Error in list_servers: {e}")
            return BaseToolExecResult(success=False, message=f"Error listing servers: {e}")

    @ToolAction(description="Get MCP server config")
    def get_server(self, name: str) -> BaseToolExecResult:
        """
        Get MCP server config by name.

        Args:
            name: Server name

        Returns:
            BaseToolExecResult with server config
        """
        try:
            config = self.manager.get_server_config(name)

            if not config:
                return BaseToolExecResult(success=False, message=f"Server '{name}' not found")

            server_dict = config.model_dump()

            return BaseToolExecResult(success=True, message=f"Retrieved config for server '{name}'", result=server_dict)

        except Exception as e:
            logger.error(f"Error in get_server: {e}")
            return BaseToolExecResult(success=False, message=f"Error getting server: {e}")

    @ToolAction(description="Check connectivity to an MCP server")
    def check_connectivity(self, name: str) -> BaseToolExecResult:
        """
        Check connectivity to an MCP server by attempting to connect and list tools.

        Args:
            name: Server name

        Returns:
            BaseToolExecResult with connectivity status
        """
        try:
            success, message, details = self.manager.check_connectivity(name)

            return BaseToolExecResult(
                success=success,
                message=message,
                result={
                    "name": name,
                    "connectivity": success,
                    "details": details,
                },
            )

        except Exception as e:
            logger.error(f"Error in check_connectivity: {e}")
            return BaseToolExecResult(success=False, message=f"Error checking connectivity: {e}")

    @ToolAction(description="List tools available on an MCP server")
    def list_tools(self, server_name: str) -> BaseToolExecResult:
        """
        List tools available on an MCP server.

        Args:
            server_name: Name of the MCP server

        Returns:
            BaseToolExecResult with list of available tools
        """
        try:
            success, message, tools_list = self.manager.list_tools(server_name)

            if success:
                return BaseToolExecResult(
                    success=True,
                    message=message,
                    result={"server_name": server_name, "tools_count": len(tools_list), "tools": tools_list},
                )
            else:
                return BaseToolExecResult(
                    success=False, message=message, result={"server_name": server_name, "tools_count": 0, "tools": []}
                )

        except Exception as e:
            logger.error(f"Error in list_tools: {e}")
            return BaseToolExecResult(success=False, message=f"Error listing tools: {e}")

    @ToolAction(description="Call a tool on an MCP server")
    def call_tool(
        self, server_name: str, tool_name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> BaseToolExecResult:
        """
        Call a tool on an MCP server.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool (optional)

        Returns:
            BaseToolExecResult with tool execution result
        """
        try:
            if arguments is None:
                arguments = {}

            success, message, result_data = self.manager.call_tool(server_name, tool_name, arguments)

            if success:
                return BaseToolExecResult(
                    success=True,
                    message=message,
                    result={
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "result": result_data,
                    },
                )
            else:
                return BaseToolExecResult(
                    success=False,
                    message=message,
                    result={
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "error": result_data,
                    },
                )

        except Exception as e:
            logger.error(f"Error in call_tool: {e}")
            return BaseToolExecResult(success=False, message=f"Error calling tool: {e}")

    def cleanup(self) -> None:
        """Clean up the MCP tool and manager."""
        if self.manager:
            self.manager.cleanup()


def _parse_header_from_parts(parts: List[str]) -> Dict[str, str]:
    """
    Try strict JSON parse first, then do a best-effort token-level parse.
    """
    STRIP_CHARS = " \"' ,"
    raw = " ".join(parts)
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        norm = raw.replace("'", '"').replace('\\"', '"')
        return json.loads(norm)
    except Exception:
        pass

    # fallback: token-level parse like `Token: 1234a, header2: 1`
    tokens = parts[:]
    if tokens and tokens[0].startswith("{"):
        tokens[0] = tokens[0][1:] or ""
    if tokens and tokens[-1].endswith("}"):
        tokens[-1] = tokens[-1][:-1] or ""
    flattened = []
    for t in tokens:
        if not t:
            continue
        items = re.split(r"\s*,\s*", t)
        for it in items:
            if it:
                flattened.append(it)
    result: Dict[str, str] = {}
    i = 0
    while i < len(flattened):
        part = flattened[i]
        m = re.match(r'^\s*["\']?([A-Za-z0-9_\-]+)["\']?\s*:\s*(.*)$', part)
        if m:
            key = m.group(1)
            rest = m.group(2).strip()
            if rest != "":
                val = rest.strip(STRIP_CHARS)
                result[key] = val
                i += 1
            else:
                if i + 1 < len(flattened):
                    val = flattened[i + 1].strip(STRIP_CHARS)
                    result[key] = val
                    i += 2
                else:
                    i += 1
        else:
            i += 1
    return result


def parse_command_string(s: str) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """
    Parse a command-line string into structured info depending on transport.

    Return: (transport_type, name, payload)
    - For 'studio'/'stdio': payload = {"command": str|None, "args": [...], "env": {...}}
    - For 'sse'/'http':    payload = {"url": str|None, "headers": {...}, "timeout": float}
    Comments are in English as requested.
    """
    tokens = shlex.split(s)
    n = len(tokens)

    # find the transport segment tokens (tokens immediately following --transport until next --option)
    transport_seg: List[Tuple[int, str]] = []
    for i, t in enumerate(tokens):
        if t == "--transport":
            j = i + 1
            seg = []
            while j < n and not tokens[j].startswith("--"):
                seg.append((j, tokens[j]))
                j += 1
            transport_seg = seg
            break

    # default timeout
    timeout: Optional[float] = None
    # parse timeout only for sse/http later; but we still capture its value if present
    for i, t in enumerate(tokens):
        if t == "--timeout" and i + 1 < n:
            try:
                timeout = float(tokens[i + 1])
            except Exception:
                timeout = None

    if not transport_seg:
        raise DatusException(ErrorCode.COMMON_FIELD_INVALID, message="No --transport found.")

    transport_type = transport_seg[0][1].lower()  # first token after --transport is type

    # find first index of --env (if any)
    first_env_idx = None
    for i, t in enumerate(tokens):
        if t == "--env":
            first_env_idx = i
            break

    # parse --env KEY=VAL occurrences (may be multiple)
    env: Dict[str, str] = {}
    i = 0
    while i < n:
        if tokens[i] == "--env" and i + 1 < n:
            kv = tokens[i + 1]
            if "=" in kv:
                k, v = kv.split("=", 1)
                env[k] = v
            i += 2
        else:
            i += 1

    # branch by transport type
    if transport_type in ("studio", "stdio"):
        # determine name and command indices in tokens
        name = transport_seg[1][1] if len(transport_seg) > 1 else None
        # choose command index: prefer the 3rd token after --transport if present, else last seg token
        if len(transport_seg) > 2:
            command_idx = transport_seg[2][0]
            command = transport_seg[2][1]
        else:
            # no explicit command token — fallback: use last token in transport segment as command if any
            command_idx = transport_seg[-1][0]
            command = transport_seg[-1][1]

        # args start immediately after command token
        args_start = command_idx + 1

        # args end at the position of first --env (user requested "command 后面 到 --env 中间的所有 均是args")
        args_end = first_env_idx if first_env_idx is not None else n

        # Ensure indices validity
        if args_start > args_end:
            args = []
        else:
            args = tokens[args_start:args_end]

        return transport_type, name, {"command": command, "args": args, "env": env}

    if transport_type in ("sse", "http"):
        name = transport_seg[1][1] if len(transport_seg) > 1 else None
        url = transport_seg[2][1] if len(transport_seg) > 2 else None

        # if url missing in transport_seg, try explicit --url or any http(s) token
        if not url:
            for i, t in enumerate(tokens):
                if t == "--url" and i + 1 < n:
                    url = tokens[i + 1]
                    break
            if not url:
                for t in tokens:
                    if t.startswith("http://") or t.startswith("https://"):
                        url = t
                        break

        # parse header (only for sse/http)
        header: Dict[str, str] = {}
        for i, t in enumerate(tokens):
            if t == "--header":
                # collect header tokens until token that ends with '}'
                j = i + 1
                parts = []
                while j < n:
                    parts.append(tokens[j])
                    if tokens[j].endswith("}"):
                        break
                    j += 1
                if parts:
                    header = _parse_header_from_parts(parts)
                break

        # default timeout fallback
        if timeout is None:
            timeout = 10.0

        return transport_type, name, {"url": url, "headers": header or {}, "timeout": timeout}
    raise DatusException(
        ErrorCode.COMMON_FIELD_INVALID, message=f"Unsupported transport protocols: {transport_type}, full_params: {s}"
    )
