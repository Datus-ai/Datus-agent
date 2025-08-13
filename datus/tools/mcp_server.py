import asyncio
import os
import threading
from pathlib import Path

from agents import Agent, RunContextWrapper, Usage
from agents.mcp import MCPServerStdio, MCPServerStdioParams

from datus.configuration.agent_config import DbConfig
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SilentMCPServerStdio(MCPServerStdio):
    """Enhanced MCP server wrapper that redirects stdout and stderr to suppress all output
    WARNING: This redirects both stdout and stderr, which may break MCP protocol communication.
    Use with caution and test thoroughly.
    """

    def __init__(self, params: MCPServerStdioParams, **kwargs):
        # Set environment variables to reduce output
        if hasattr(params, "env"):
            if params.env is None:
                params.env = {}

            # Basic environment variables for all MCP servers
            params.env.update(
                {
                    "UV_QUIET": "1",  # Quiet uv tool output
                    "RUST_LOG": "error",  # Reduce Rust logging
                }
            )

            # Additional variables for filesystem MCP server
            if hasattr(params, "args") and any("server-filesystem" in str(arg) for arg in (params.args or [])):
                params.env.update(
                    {
                        "NODE_OPTIONS": "--no-warnings --quiet",
                        "NPM_CONFIG_LOGLEVEL": "silent",
                        "SUPPRESS_NO_CONFIG_WARNING": "1",
                    }
                )

        # Redirect both stdout and stderr using shell redirection
        if hasattr(params, "command") and hasattr(params, "args"):
            original_command = params.command
            original_args = params.args or []

            # Create shell command to redirect both stdout and stderr
            import sys

            if sys.platform == "win32":
                # Windows: redirect both stdout and stderr to nul
                params.command = "cmd"
                params.args = ["/c", f'"{original_command}" {" ".join(original_args)} >nul 2>&1']
            else:
                # Unix/Linux/macOS: redirect both stdout and stderr to /dev/null
                args_str = " ".join(f'"{arg}"' for arg in original_args)
                params.command = "sh"
                params.args = ["-c", f'"{original_command}" {args_str} >/dev/null 2>&1']

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
            datus_mcp_path = Path(path) / "datus-mcp" / mcp_name
            if datus_mcp_path.exists():
                logger.info(f"Found MCP directory via sys.path: {datus_mcp_path}")
                return str(datus_mcp_path)

    raise FileNotFoundError(
        f"MCP directory '{mcp_name}' not found in development mcp directory or installed datus-mcp package"
    )


class MCPServer:
    _snowflake_mcp_server = None
    _starrocks_mcp_server = None
    _sqlite_mcp_server = None
    _duckdb_mcp_server = None
    _metricflow_mcp_server = None
    _filesystem_mcp_server = None
    _lock = threading.Lock()

    @classmethod
    def get_db_mcp_server(cls, db_config: DbConfig):
        """Get the appropriate MCP server based on database type and configuration."""
        if not db_config:
            logger.error("No database configuration provided")
            return None

        if db_config.type in [DBType.SNOWFLAKE, DBType.STARROCKS]:
            return cls._get_server_based_mcp_server(db_config, db_config.database or "")
        elif db_config.type in [DBType.SQLITE, DBType.DUCKDB]:
            return cls._get_file_based_mcp_server(db_config)
        else:
            logger.error(f"Unsupported database type for MCP: {db_config.type}")
            raise ValueError(f"Unsupported database type for MCP: {db_config.type}")

    @classmethod
    def _get_server_based_mcp_server(cls, db_config: DbConfig, database: str):
        """Get MCP server for server-based databases (Snowflake, StarRocks)."""
        if db_config.type == DBType.SNOWFLAKE:
            logger.debug("Initializing Snowflake MCP server")
            return cls.get_snowflake_mcp_server(database, db_config)
        elif db_config.type == DBType.STARROCKS:
            logger.debug("Initializing StarRocks MCP server")
            return cls.get_starrocks_mcp_server(database, db_config)
        else:
            logger.error(f"Unsupported server-based database type: {db_config.type}")
            return None

    @classmethod
    def _get_file_based_mcp_server(cls, db_config: DbConfig):
        """Get MCP server for file-based databases (SQLite, DuckDB)."""
        if not db_config.uri:
            if db_config.type == DBType.SQLITE:
                logger.info("Initializing SQLite MCP server with default database")
                return cls.get_sqlite_mcp_server()
            else:
                logger.info("Initializing DuckDB MCP server with memory database")
                return cls.get_duckdb_mcp_server()

        db_path = cls._extract_db_path_from_uri(db_config.uri, db_config.type)

        if db_config.type == DBType.SQLITE:
            logger.info(f"Initializing SQLite MCP server with database: {db_path}")
            return cls.get_sqlite_mcp_server(db_path=db_path)
        elif db_config.type == DBType.DUCKDB:
            logger.info(f"Initializing DuckDB MCP server with database: {db_path}")
            return cls.get_duckdb_mcp_server(db_path=db_path)
        else:
            logger.error(f"Unsupported file-based database type: {db_config.type}")
            return None

    @classmethod
    def _extract_db_path_from_uri(cls, uri: str, db_type: str) -> str:
        """Extract database path from URI and convert to absolute path."""
        # Remove protocol prefix if present
        if uri.startswith(f"{db_type.lower()}:///"):
            db_path = uri.replace(f"{db_type.lower()}:///", "")
        else:
            db_path = uri

        # Expand user home directory (handle ~ paths)
        db_path = str(Path(db_path).expanduser())

        # Convert relative path to absolute path, if not already absolute
        if not os.path.isabs(db_path):
            db_path = os.path.abspath(db_path)

        return db_path

    @classmethod
    def check_connectivity(cls, db_config: DbConfig):
        """Check MCP server connectivity for a given database configuration."""
        logger.info(f"Checking MCP server connectivity for database type: {db_config.type}")

        try:
            mcp_server = cls.get_db_mcp_server(db_config)
            if not mcp_server:
                logger.error(f"{db_config.type} MCP Server failed to initialize")
                return

            async def test_connection():
                try:
                    await mcp_server.connect()
                    logger.info(f"{db_config.type} MCP Server connected successfully")

                    if hasattr(mcp_server, "list_tools"):
                        # Create minimal agent and run context for the new interface
                        agent = Agent(name="test-agent")
                        run_context = RunContextWrapper(context=None, usage=Usage())
                        tools = await mcp_server.list_tools(run_context, agent)
                        tool_count = len(tools) if tools else 0
                        logger.info(
                            f"{db_config.type} MCP Server has {tool_count} tools available: "
                            f"{[tool.name for tool in tools]}"
                        )

                except Exception as e:
                    logger.error(f"{db_config.type} MCP Server connection failed: {str(e)}")
                finally:
                    # Ensure proper cleanup
                    if hasattr(mcp_server, "cleanup"):
                        try:
                            await mcp_server.cleanup()
                        except Exception:
                            pass

            asyncio.run(test_connection())

        except Exception as e:
            logger.error(f"MCP server check failed: {str(e)}")

    @classmethod
    def get_snowflake_mcp_server(cls, database_name: str, db_config: DbConfig):
        if cls._snowflake_mcp_server is None:
            with cls._lock:
                if cls._snowflake_mcp_server is None:
                    directory = os.environ.get("SNOWFLAKE_MCP_DIR", "mcp/mcp-snowflake-server")
                    if not directory:
                        try:
                            directory = find_mcp_directory("mcp-snowflake-server")
                        except FileNotFoundError as e:
                            logger.error(f"Could not find Snowflake MCP directory: {e}")
                            return None

                    mcp_server_params = MCPServerStdioParams(
                        command="uv",
                        args=["--directory", directory, "run", "mcp_snowflake_server"],
                        env={
                            "SNOWFLAKE_DATABASE": database_name if database_name else db_config.database,
                            "SNOWFLAKE_SCHEMA": db_config.schema,
                            "SNOWFLAKE_WAREHOUSE": db_config.warehouse,
                            "SNOWFLAKE_USER": db_config.username,
                            "SNOWFLAKE_PASSWORD": db_config.password,
                            "SNOWFLAKE_ACCOUNT": db_config.account,
                        },
                    )
                    logger.info(f"Snowflake MCP server params: {mcp_server_params}")
                    cls._snowflake_mcp_server = SilentMCPServerStdio(
                        params=mcp_server_params,
                        client_session_timeout_seconds=120,
                    )
        return cls._snowflake_mcp_server

    @classmethod
    def get_starrocks_mcp_server(cls, database_name: str, db_config: DbConfig):
        if cls._starrocks_mcp_server is None:
            with cls._lock:
                if cls._starrocks_mcp_server is None:
                    directory = os.environ.get("STARROCKS_MCP_DIR", "mcp/mcp-starrocks-server")
                    if not directory:
                        try:
                            directory = find_mcp_directory("mcp-starrocks-server")
                        except FileNotFoundError as e:
                            logger.error(f"Could not find StarRocks MCP directory: {e}")
                            return None

                    mcp_server_params = MCPServerStdioParams(
                        command="uv",
                        args=["--directory", directory, "run", "mcp-server-starrocks"],
                        env={
                            "STARROCKS_DATABASE": database_name if database_name else db_config.database,
                            "STARROCKS_HOST": db_config.host,
                            "STARROCKS_PORT": str(db_config.port),
                            "STARROCKS_USER": db_config.username,
                            "STARROCKS_PASSWORD": db_config.password,
                        },
                    )
                    cls._starrocks_mcp_server = SilentMCPServerStdio(
                        params=mcp_server_params, client_session_timeout_seconds=120  # Increase timeout for StarRocks
                    )
        return cls._starrocks_mcp_server

    @classmethod
    def get_sqlite_mcp_server(cls, db_path: str = "./sqlite_mcp_server.db"):
        # Convert db_path to absolute path to avoid confusion with relative paths
        absolute_db_path = os.path.abspath(db_path)

        # Check if we need to create a new server for a different database
        if cls._sqlite_mcp_server is None or getattr(cls, "_current_sqlite_db_path", None) != absolute_db_path:
            with cls._lock:
                if cls._sqlite_mcp_server is None or getattr(cls, "_current_sqlite_db_path", None) != absolute_db_path:
                    directory = os.environ.get("SQLITE_MCP_DIR", "mcp/mcp-sqlite-server")
                    if not directory:
                        try:
                            directory = find_mcp_directory("mcp-sqlite-server")
                        except FileNotFoundError as e:
                            logger.error(f"Could not find SQLite MCP directory: {e}")
                            return None

                    logger.info(f"Using SQLite database: {absolute_db_path}")

                    mcp_server_params = MCPServerStdioParams(
                        command="uv",
                        args=[
                            "--directory",
                            directory,
                            "run",
                            "mcp-server-sqlite",
                            "--db-path",
                            absolute_db_path,
                        ],
                        env={},  # SQLite doesn't need additional environment variables
                    )
                    cls._sqlite_mcp_server = SilentMCPServerStdio(params=mcp_server_params)
                    cls._current_sqlite_db_path = absolute_db_path
        return cls._sqlite_mcp_server

    @classmethod
    def create_sqlite_mcp_server(cls, db_path: str = "./sqlite_mcp_server.db"):
        """Create a new SQLite MCP server instance without using the shared singleton.

        This is useful for parallel subworkflows to avoid shared lifecycle and cleanup races.
        """
        directory = os.environ.get("SQLITE_MCP_DIR", "mcp/mcp-sqlite-server")
        if not directory:
            try:
                directory = find_mcp_directory("mcp-sqlite-server")
            except FileNotFoundError as e:
                logger.error(f"Could not find SQLite MCP directory: {e}")
                return None

        logger.info(f"Using SQLite database: {db_path}")

        mcp_server_params = MCPServerStdioParams(
            command="uv",
            args=[
                "--directory",
                directory,
                "run",
                "mcp-server-sqlite",
                "--db-path",
                db_path,
            ],
            env={},
        )
        return SilentMCPServerStdio(params=mcp_server_params)

    @classmethod
    def get_duckdb_mcp_server(cls, db_path: str = ":memory:"):
        if cls._duckdb_mcp_server is None:
            with cls._lock:
                if cls._duckdb_mcp_server is None:
                    directory = os.environ.get("DUCKDB_MCP_DIR", "mcp/mcp-duckdb-server")
                    if not directory:
                        try:
                            directory = find_mcp_directory("mcp-duckdb-server")
                        except FileNotFoundError as e:
                            logger.error(f"Could not find DuckDB MCP directory: {e}")
                            return None

                    logger.info(f"Using DuckDB database: {db_path}")

                    mcp_server_params = MCPServerStdioParams(
                        command="uv",
                        args=[
                            "--directory",
                            directory,
                            "run",
                            "mcp-server-motherduck",
                            "--db-path",
                            db_path,
                            "--read-only",
                        ],
                        env={},  # DuckDB doesn't need additional environment variables for local usage
                    )
                    cls._duckdb_mcp_server = SilentMCPServerStdio(
                        params=mcp_server_params, client_session_timeout_seconds=30
                    )
        return cls._duckdb_mcp_server

    @classmethod
    def get_metricflow_mcp_server(cls, database_name: str, db_config: DbConfig):
        if cls._metricflow_mcp_server is None:
            with cls._lock:
                if cls._metricflow_mcp_server is None:
                    directory = os.environ.get("METRICFLOW_MCP_DIR", "mcp/mcp-metricflow-server")
                    mf_path = os.environ.get("MF_PATH", "")
                    mf_project_dir = os.environ.get("MF_PROJECT_DIR", "")
                    mf_verbose = os.environ.get("MF_VERBOSE", "false")
                    if not directory:
                        try:
                            directory = find_mcp_directory("mcp-metricflow-server")
                        except FileNotFoundError as e:
                            logger.error(f"Could not find MetricFlow MCP directory: {e}")
                            return None
                    logger.info(f"Using MetricFlow MCP server with directory: {directory}")

                    env_settings = {
                        "MF_MODEL_PATH": os.getenv("FILESYSTEM_MCP_DIRECTORY", "/tmp"),
                        "MF_PATH": mf_path,
                        "MF_PROJECT_DIR": mf_project_dir,
                        "MF_VERBOSE": mf_verbose,
                    }
                    if db_config.type in (DBType.DUCKDB, DBType.SQLITE):
                        env_settings["MF_DWH_SCHEMA"] = db_config.schema
                        env_settings["MF_DWH_DIALECT"] = db_config.type
                        env_settings["MF_DWH_DB"] = str(Path(db_config.uri).expanduser())
                    elif db_config.type == DBType.STARROCKS:
                        env_settings["MF_DWH_SCHEMA"] = db_config.schema
                        env_settings["MF_DWH_DIALECT"] = DBType.MYSQL
                        env_settings["MF_DWH_HOST"] = db_config.host
                        env_settings["MF_DWH_PORT"] = str(db_config.port)
                        env_settings["MF_DWH_USER"] = db_config.username
                        env_settings["MF_DWH_PASSWORD"] = db_config.password
                        env_settings["MF_DWH_DB"] = database_name

                    mcp_server_params = MCPServerStdioParams(
                        command="uv",
                        args=[
                            "--directory",
                            directory,
                            "run",
                            "mcp-metricflow-server",
                        ],
                        env=env_settings,
                    )
                    cls._metricflow_mcp_server = SilentMCPServerStdio(
                        params=mcp_server_params, client_session_timeout_seconds=20
                    )
        return cls._metricflow_mcp_server

    @classmethod
    def get_filesystem_mcp_server(cls):
        if cls._filesystem_mcp_server is None:
            with cls._lock:
                if cls._filesystem_mcp_server is None:
                    filesystem_mcp_directory = os.environ.get("FILESYSTEM_MCP_DIRECTORY", "/tmp")
                    mcp_server_params = MCPServerStdioParams(
                        command="npx",
                        args=[
                            "--silent",
                            "-y",
                            "@modelcontextprotocol/server-filesystem",
                            filesystem_mcp_directory,
                        ],
                        env={
                            "NODE_OPTIONS": "--no-warnings",
                            "NPM_CONFIG_LOGLEVEL": "silent",
                        },
                    )
                    cls._filesystem_mcp_server = SilentMCPServerStdio(
                        params=mcp_server_params, client_session_timeout_seconds=30
                    )
        return cls._filesystem_mcp_server
