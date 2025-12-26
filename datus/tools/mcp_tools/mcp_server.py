# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
import threading
from pathlib import Path

from agents.mcp import MCPServerStdio, MCPServerStdioParams

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SilentMCPServerStdio(MCPServerStdio):
    """MCP server wrapper that redirects stderr to suppress logs.

    Uses shell wrapping with explicit environment variable passing to ensure:
    1. stderr is redirected to /dev/null (suppresses all console output)
    2. Environment variables are correctly passed to the subprocess
    """

    def __init__(self, params: MCPServerStdioParams, **kwargs):
        # Get command, args, and env
        if hasattr(params, "command"):
            original_command = params.command
            original_args = params.args or []
            env = params.env or {}
        else:
            original_command = params["command"]
            original_args = params["args"] or []
            env = params.get("env") or {}

        import shlex
        import sys

        if sys.platform == "win32":
            # Windows: Use cmd with stderr redirection
            redirect_cmd = "cmd"
            env_sets = " && ".join(f"set {k}={v}" for k, v in env.items())
            args_str = " ".join(original_args)
            redirect_args = ["/c", f"{env_sets} && {original_command} {args_str} 2>nul"]
        else:
            # Unix/Linux/macOS: Use sh with explicit env exports
            redirect_cmd = "sh"

            # Filter out shell-specific and system internal variables that cause issues
            # Keep user-configured variables (API keys, DB credentials, etc.) and essential system variables
            excluded_prefixes = (
                "BASH_",
                "ZSH_",
                "_",
                "__CF",
                "COMMAND_MODE",
                "SSH_CLIENT",
                "SSH_CONNECTION",
                "SSH_TTY",
            )
            excluded_vars = ("SHLVL", "SHELLOPTS", "PS1", "PS2", "PS3", "PS4", "OLDPWD")

            filtered_env = {k: v for k, v in env.items() if not (k.startswith(excluded_prefixes) or k in excluded_vars)}

            # Build environment variable exports
            env_exports = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in filtered_env.items())

            # Quote command and args properly
            cmd_str = shlex.quote(original_command)
            args_str = " ".join(shlex.quote(str(arg)) for arg in original_args)

            # Combine: export vars, run command, redirect stderr
            full_cmd = f"{env_exports} {cmd_str} {args_str} 2>/dev/null"
            redirect_args = ["-c", full_cmd]

        # Update params
        if hasattr(params, "command"):
            params.command = redirect_cmd
            params.args = redirect_args
            params.env = None  # Already included in the shell command
        else:
            params["command"] = redirect_cmd
            params["args"] = redirect_args
            params["env"] = None

        super().__init__(params, **kwargs)


def find_mcp_directory(mcp_name: str) -> str:
    """Find the MCP directory, whether in development or installed package"""

    relative_path = f"mcp/{mcp_name}"
    if Path(relative_path).exists():
        logger.info(f"Found MCP directory in development: {Path(relative_path).resolve()}")
        return relative_path

    import sys

    for path in sys.path:
        if "site-packages" in path:
            datus_mcp_path = Path(path) / "mcp" / mcp_name
            if datus_mcp_path.exists():
                logger.info(f"Found MCP directory via sys.path: {datus_mcp_path}")
                return str(datus_mcp_path)

    raise FileNotFoundError(
        f"MCP directory '{mcp_name}' not found in development mcp directory or installed datus-mcp package"
    )


class MCPServer:
    _metricflow_mcp_server = None
    _lock = threading.Lock()

    @classmethod
    def get_metricflow_mcp_server(cls, namespace: str):
        if cls._metricflow_mcp_server is None:
            with cls._lock:
                if cls._metricflow_mcp_server is None:
                    directory = os.getenv("METRICFLOW_MCP_DIR")
                    if not directory:
                        try:
                            directory = find_mcp_directory("mcp-metricflow-server")
                        except FileNotFoundError as e:
                            logger.error(f"Could not find MetricFlow MCP directory: {e}")
                            return None
                    logger.info(f"Using MetricFlow MCP server with directory: {directory}")

                    # Verify directory exists
                    if not os.path.exists(directory):
                        logger.error(f"MetricFlow MCP directory does not exist: {directory}")
                        return None

                    # Verify mcp-metricflow-server exists
                    pyproject_path = os.path.join(directory, "pyproject.toml")
                    if not os.path.exists(pyproject_path):
                        logger.error(f"MetricFlow MCP pyproject.toml not found: {pyproject_path}")
                        return None

                    logger.info(f"MetricFlow MCP server directory verified: {directory}")

                    # MetricFlow can now read Datus config directly via DatusConfigHandler
                    # Pass the namespace via --namespace command line argument
                    # Pass current working directory so mf can find ./conf/agent.yml
                    env_dict = os.environ.copy()
                    env_dict["DATUS_PROJECT_ROOT"] = os.getcwd()

                    mcp_server_params = MCPServerStdioParams(
                        command="uv",
                        args=[
                            "--directory",
                            directory,
                            "run",
                            "mcp-metricflow-server",
                            "--namespace",
                            namespace,
                        ],
                        env=env_dict,
                    )
                    cls._metricflow_mcp_server = SilentMCPServerStdio(
                        params=mcp_server_params, client_session_timeout_seconds=20
                    )
        return cls._metricflow_mcp_server
