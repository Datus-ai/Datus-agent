"""MCP Tools for MetricFlow Server."""

from typing import List, Optional

from mcp.server import Server

from mcp import types

from .prompts import get_prompt
from .server import MetricFlowServer


def register_metricflow_tools(server: Server, namespace: Optional[str] = None) -> None:
    """Register MetricFlow tools with the MCP server.

    Args:
        server: MCP server instance
        namespace: Datus namespace for configuration
    """
    metricflow_server = MetricFlowServer(namespace)

    @server.list_tools()
    async def handle_list_tools() -> List[types.Tool]:
        """List available MetricFlow tools."""
        return [
            types.Tool(
                name="list_metrics",
                description=get_prompt("list_metrics"),
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            types.Tool(
                name="get_dimensions",
                description=get_prompt("get_dimensions"),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of metric names to get dimensions for",
                        }
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="get_entities",
                description=get_prompt("get_entities"),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of metric names to get entities for",
                        }
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="query_metrics",
                description=get_prompt("query_metrics"),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of metric names to query (required)",
                        },
                        "dimensions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of dimensions to group by",
                        },
                        "order_by": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of fields to order by",
                        },
                        "where": {"type": "string", "description": "Optional WHERE clause filter"},
                        "limit": {"type": "integer", "description": "Optional number of rows to limit"},
                        "start_time": {
                            "type": "string",
                            "description": "Optional start time for the query (ISO format)",
                        },
                        "end_time": {"type": "string", "description": "Optional end time for the query (ISO format)"},
                        "explain": {"type": "boolean", "description": "Optional flag to explain the query"},
                    },
                    "required": ["metrics"],
                },
            ),
            types.Tool(
                name="validate_configs",
                description="Validate the MetricFlow project configurations",
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            types.Tool(
                name="get_dimension_values",
                description="Get possible values for a specific dimension associated with a metric",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dimension_name": {"type": "string", "description": "Name of the dimension to get values for"},
                        "metric_name": {
                            "type": "string",
                            "description": "Metric that is associated with the dimension (required)",
                        },
                    },
                    "required": ["dimension_name", "metric_name"],
                },
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict | None
    ) -> List[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        """Handle tool execution requests."""
        if arguments is None:
            arguments = {}

        try:
            if name == "list_metrics":
                result = metricflow_server.list_metrics()
                return [types.TextContent(type="text", text=result)]

            elif name == "get_dimensions":
                metrics = arguments.get("metrics")
                result = metricflow_server.get_dimensions(metrics)
                return [types.TextContent(type="text", text=result)]

            elif name == "get_entities":
                metrics = arguments.get("metrics")
                result = metricflow_server.get_entities(metrics)
                return [types.TextContent(type="text", text=result)]

            elif name == "query_metrics":
                if "metrics" not in arguments:
                    raise ValueError("metrics parameter is required")

                result = metricflow_server.query_metrics(
                    metrics=arguments["metrics"],
                    dimensions=arguments.get("dimensions"),
                    order_by=arguments.get("order_by"),
                    where=arguments.get("where"),
                    limit=arguments.get("limit"),
                    start_time=arguments.get("start_time"),
                    end_time=arguments.get("end_time"),
                    explain=arguments.get("explain", False),
                )
                return [types.TextContent(type="text", text=result)]

            elif name == "validate_configs":
                result = metricflow_server.validate_configs()
                return [types.TextContent(type="text", text=result)]

            elif name == "get_dimension_values":
                if "dimension_name" not in arguments:
                    raise ValueError("dimension_name parameter is required")
                if "metric_name" not in arguments:
                    raise ValueError("metric_name parameter is required")

                result = metricflow_server.get_dimension_values(
                    dimension_name=arguments["dimension_name"], metric_name=arguments["metric_name"]
                )
                return [types.TextContent(type="text", text=result)]

            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]
