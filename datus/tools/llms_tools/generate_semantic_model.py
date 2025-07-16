import asyncio
import json
from typing import Any, AsyncGenerator, Dict, Optional

from langsmith import traceable

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

logger = get_logger(__name__)


@traceable
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

    def generate_semantic_model_prompt(input_data, db_config):
        return get_generate_semantic_model_prompt(
            database_type=db_config.type,
            table_definition=table_definition,
            prompt_version=input_data.prompt_version,
        )

    # Setup MCP servers
    filesystem_mcp_server = MCPServer.get_filesystem_mcp_server()
    mcp_servers = {"filesystem_mcp_server": filesystem_mcp_server}

    async for action in base_mcp_stream(
        model=model,
        input_data=input_data,
        db_config=db_config,
        tool_config=tool_config,
        mcp_servers=mcp_servers,
        prompt_generator=generate_semantic_model_prompt,
        instruction_template="generate_semantic_model_system",
        action_history_manager=action_history_manager,
    ):
        yield action


@traceable
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

    filesystem_mcp_server = MCPServer.get_filesystem_mcp_server()

    instruction = prompt_manager.get_raw_template("generate_semantic_model_system", input_data.prompt_version)
    max_turns = tool_config.get("max_turns", 20)

    prompt = get_generate_semantic_model_prompt(
        database_type=db_config.type,
        table_definition=table_definition,
        prompt_version=input_data.prompt_version,
    )

    try:
        exec_result = asyncio.run(
            model.generate_with_mcp(
                prompt=prompt,
                mcp_servers={
                    "filesystem_mcp_server": filesystem_mcp_server,
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
            semantic_model_meta=input_data.semantic_model_meta,
            semantic_model_file=content_dict.get("semantic_model_file", ""),
        )
    except Exception as e:
        logger.error(f"Generate semantic model failed: {e}")
        return GenerateSemanticModelResult(
            success=False,
            error=str(e),
            semantic_model_meta=input_data.semantic_model_meta,
            semantic_model_file="",
        )
