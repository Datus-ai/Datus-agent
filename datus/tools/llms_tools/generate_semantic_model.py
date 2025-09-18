import asyncio
import json
import os
from typing import Any, AsyncGenerator, Dict, Optional

from datus.configuration.agent_config import DbConfig
from datus.models.base import LLMBaseModel
from datus.prompts.generate_semantic_model import get_generate_semantic_model_prompt
from datus.prompts.prompt_manager import prompt_manager
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.schemas.generate_semantic_model_node_models import GenerateSemanticModelInput, GenerateSemanticModelResult
from datus.tools.llms_tools.mcp_stream_utils import base_mcp_stream
from datus.tools.mcp_server import MCPServer
from datus.utils.json_utils import extract_json_str
from datus.utils.loggings import get_logger
from datus.utils.traceable_utils import optional_traceable

logger = get_logger(__name__)


@optional_traceable()
async def generate_semantic_model_with_mcp_stream(
    model: LLMBaseModel,
    table_definition: str,
    input_data: GenerateSemanticModelInput,
    db_config: DbConfig,
    tool_config: Dict[str, Any],
    action_history_manager: Optional[ActionHistoryManager] = None,
) -> AsyncGenerator[ActionHistory, None]:
    """Generate semantic model with streaming support and action history tracking."""
    if not isinstance(input_data, GenerateSemanticModelInput):
        raise ValueError("Input must be a GenerateSemanticModelInput instance")

    prompt = get_generate_semantic_model_prompt(
        database_type=db_config.type,
        table_definition=table_definition,
        prompt_version=input_data.prompt_version,
    )

    # Setup MCP servers
    filesystem_mcp_server = MCPServer.get_filesystem_mcp_server(path=os.getenv("MF_MODEL_PATH"))
    metricflow_mcp_server = MCPServer.get_metricflow_mcp_server(
        database_name=input_data.sql_task.database_name, db_config=db_config
    )
    mcp_servers = {
        "filesystem_mcp_server": filesystem_mcp_server,
        "metricflow_mcp_server": metricflow_mcp_server,
    }
    tool_config["max_turns"] = 20

    async for action in base_mcp_stream(
        model=model,
        input_data=input_data,
        db_config=db_config,
        tool_config=tool_config,
        mcp_servers=mcp_servers,
        prompt=prompt,
        instruction_template="generate_semantic_model_system",
        action_history_manager=action_history_manager,
    ):
        yield action


@optional_traceable()
def generate_semantic_model_with_mcp(
    model: LLMBaseModel,
    table_definition: str,
    input_data: GenerateSemanticModelInput,
    db_config: DbConfig,
    tool_config: Dict[str, Any],
) -> GenerateSemanticModelResult:
    """Generate semantic model for the given SQL query."""
    if not isinstance(input_data, GenerateSemanticModelInput):
        raise ValueError("Input must be a GenerateSemanticModelInput instance")

    filesystem_mcp_server = MCPServer.get_filesystem_mcp_server(path=os.getenv("MF_MODEL_PATH"))
    metricflow_mcp_server = MCPServer.get_metricflow_mcp_server(
        database_name=input_data.sql_task.database_name, db_config=db_config
    )

    instruction = prompt_manager.get_raw_template("generate_semantic_model_system", input_data.prompt_version)
    max_turns = tool_config.get("max_turns", 20)

    prompt = get_generate_semantic_model_prompt(
        database_type=db_config.type,
        table_definition=table_definition,
        prompt_version=input_data.prompt_version,
    )

    try:
        exec_result = asyncio.run(
            model.generate_with_tools(
                prompt=prompt,
                mcp_servers={
                    "filesystem_mcp_server": filesystem_mcp_server,
                    "metricflow_mcp_server": metricflow_mcp_server,
                },
                instruction=instruction,
                output_type=str,
                max_turns=max_turns,
            )
        )

        try:
            logger.debug(f"exec_result: {exec_result['content']}")
            content_dict = json.loads(extract_json_str(exec_result["content"]))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse exec_result.content: {e}, exec_result: {exec_result}")
            content_dict = {}

        return GenerateSemanticModelResult(
            success=True,
            error="",
            table_name=input_data.table_name,
            semantic_model_file=content_dict.get("semantic_model_file", ""),
        )
    except Exception as e:
        logger.error(f"Generate semantic model failed: {e}")
        return GenerateSemanticModelResult(
            success=False,
            error=str(e),
            table_name=input_data.table_name,
            semantic_model_file="",
        )
