"""Main entry point for MCP MetricFlow Server."""

import argparse
import logging
import os
from typing import Any

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from .tools import register_metricflow_tools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVER_VERSION = "0.1.0"


async def main() -> None:
    """Main server entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MCP MetricFlow Server")
    parser.add_argument("--namespace", help="Datus namespace to use for configuration")
    args = parser.parse_args()

    namespace = args.namespace

    logger.info(f"Starting MCP MetricFlow Server with namespace: {namespace}")

    # Create MCP server
    server = Server("mcp-metricflow-server")

    # Register MetricFlow tools
    register_metricflow_tools(server, namespace)

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
