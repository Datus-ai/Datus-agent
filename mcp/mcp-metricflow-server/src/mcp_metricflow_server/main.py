"""Main entry point for MCP MetricFlow Server."""

import logging
from typing import Any

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from .config import MetricFlowConfig
from .tools import register_metricflow_tools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVER_VERSION = "0.1.0"


async def main() -> None:
    """Main server entry point."""
    # Load configuration from environment
    config = MetricFlowConfig.from_env()

    logger.info(f"Starting MCP MetricFlow Server with config: {config.model_dump()}")

    # Create MCP server
    server = Server("mcp-metricflow-server")

    # Register MetricFlow tools
    register_metricflow_tools(server, config)

    # Add server info
    @server.list_resources()
    async def list_resources() -> list[Any]:
        """List available resources."""
        return []

    @server.list_prompts()
    async def list_prompts() -> list[Any]:
        """List available prompts."""
        return []

    # Run the server using stdin/stdout streams
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mcp-metricflow-server",
                server_version=SERVER_VERSION,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
