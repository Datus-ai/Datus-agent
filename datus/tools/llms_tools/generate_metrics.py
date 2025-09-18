import asyncio
import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from agents import Tool

from datus.configuration.agent_config import DbConfig
from datus.models.base import LLMBaseModel
from datus.prompts.generate_metrics_with_mcp import get_generate_metrics_prompt
from datus.prompts.prompt_manager import prompt_manager
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.schemas.generate_metrics_node_models import GenerateMetricsInput, GenerateMetricsResult, Metric
from datus.tools.llms_tools.mcp_stream_utils import base_mcp_stream
from datus.tools.mcp_server import MCPServer
from datus.utils.json_utils import extract_json_str
from datus.utils.loggings import get_logger
from datus.utils.traceable_utils import optional_traceable

logger = get_logger(__name__)


@optional_traceable()
async def generate_metrics_with_mcp_stream(
    model: LLMBaseModel,
    input_data: GenerateMetricsInput,
    tool_config: Dict[str, Any],
    db_config: DbConfig,
    tools: List[Tool],
    action_history_manager: Optional[ActionHistoryManager] = None,
) -> AsyncGenerator[ActionHistory, None]:
    """Generate metrics with streaming support and action history tracking."""
    if not isinstance(input_data, GenerateMetricsInput):
        raise ValueError("Input must be a GenerateMetricsInput instance")

    prompt = get_generate_metrics_prompt(
        database_type=input_data.sql_task.database_type,
        sql_query=input_data.sql_query,
        description=input_data.sql_task.task,
        prompt_version=input_data.prompt_version,
    )

    # Setup MCP servers
    metricflow_mcp_server = MCPServer.get_metricflow_mcp_server(
        database_name=input_data.sql_task.database_name, db_config=db_config
    )
    filesystem_mcp_server = MCPServer.get_filesystem_mcp_server(path=os.getenv("MF_MODEL_PATH"))
    mcp_servers = {
        "metricflow_mcp_server": metricflow_mcp_server,
        "filesystem_mcp_server": filesystem_mcp_server,
    }

    tool_config["max_turns"] = 30

    async for action in base_mcp_stream(
        model=model,
        input_data=input_data,
        tool_config=tool_config,
        mcp_servers=mcp_servers,
        prompt=prompt,
        tools=tools,
        instruction_template="generate_metrics_system",
        action_history_manager=action_history_manager,
    ):
        yield action


@optional_traceable()
def generate_metrics_with_mcp(
    model: LLMBaseModel,
    input_data: GenerateMetricsInput,
    db_config: DbConfig,
    tools: List[Tool],
    tool_config: Dict[str, Any],
) -> GenerateMetricsResult:
    """Generate metrics for the given SQL query."""
    if not isinstance(input_data, GenerateMetricsInput):
        raise ValueError("Input must be a GenerateMetricsInput instance")

    metricflow_mcp_server = MCPServer.get_metricflow_mcp_server(
        database_name=input_data.sql_task.database_name,
        db_config=db_config,
    )
    filesystem_mcp_server = MCPServer.get_filesystem_mcp_server(path=os.getenv("MF_MODEL_PATH"))

    instruction = prompt_manager.get_raw_template("generate_metrics_system", input_data.prompt_version)
    max_turns = tool_config.get("max_turns", 30)

    prompt = get_generate_metrics_prompt(
        database_type=input_data.sql_task.database_type,
        sql_query=input_data.sql_query,
        description=input_data.sql_task.task,
        prompt_version=input_data.prompt_version,
    )
    try:
        exec_result = asyncio.run(
            model.generate_with_tools(
                prompt=prompt,
                mcp_servers={
                    "metricflow_mcp_server": metricflow_mcp_server,
                    "filesystem_mcp_server": filesystem_mcp_server,
                },
                tools=tools,
                instruction=instruction,
                output_type=str,
                max_turns=max_turns,
            )
        )

        try:
            logger.debug(f"exec_result: {exec_result['content']}")
            content_dict = json.loads(extract_json_str(exec_result["content"]))
            metrics = parse_metrics(content_dict)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse exec_result.content: {e}, exec_result: {exec_result}")
            content_dict = {}
            metrics = []
        # content_dict = exec_result
        # Extract required pieces from the parsed dict
        return GenerateMetricsResult(
            success=True,
            error="",
            sql_queries=content_dict.get("sql_queries", []),
            metrics=metrics,
        )
    except Exception as e:
        # TODO : deal with excceed the max round
        error_msg = str(e)
        logger.error(f"Generate Metrics with MCP failed: {e}")
        return GenerateMetricsResult(
            success=False,
            error=error_msg,
            sql_queries=[],
            metrics=[],
        )


def parse_metrics(content_dict: Dict[str, Any]) -> List[Metric]:
    metrics = []
    metric_list = content_dict.get("metrics", [])
    query_list = content_dict.get("sql_queries", [])
    if not metric_list or not query_list or len(metric_list) != len(query_list):
        logger.error(f"Failed to parse metrics: {content_dict}")
        return metrics
    for metric, query in zip(metric_list, query_list):
        metrics.append(
            Metric(
                name=metric.get("name", ""),
                description=metric.get("description", ""),
                constraint=metric.get("constraint", ""),
                sql_query=query,
            )
        )
    return metrics
