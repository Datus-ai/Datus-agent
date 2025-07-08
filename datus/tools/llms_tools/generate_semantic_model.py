import asyncio
import json
from typing import Any, Dict

from langsmith import traceable

from datus.configuration.agent_config import DbConfig
from datus.models.base import LLMBaseModel
from datus.prompts.generate_semantic_model import get_generate_semantic_model_prompt
from datus.prompts.prompt_manager import prompt_manager
from datus.schemas.generate_semantic_model_node_models import GenerateSemanticModelInput, GenerateSemanticModelResult
from datus.tools.mcp_server import MCPServer
from datus.utils.json_utils import strip_json_str
from datus.utils.loggings import get_logger

logger = get_logger(__file__)


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
            content_dict = json.loads(strip_json_str(exec_result["content"]))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse exec_result.content: {e}, exec_result: {exec_result}")
            content_dict = {}
        semantic_model_meta = input_data.semantic_model_meta
        semantic_model_meta.table_name = content_dict.get("table_name", "")
        semantic_model_meta.schema_name = content_dict.get("schema_name", "")
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
