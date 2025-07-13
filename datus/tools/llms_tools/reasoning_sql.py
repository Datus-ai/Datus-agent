import asyncio
import json
from typing import Any, Dict

from langsmith import traceable

from datus.configuration.agent_config import DbConfig
from datus.models.base import LLMBaseModel
from datus.prompts.prompt_manager import prompt_manager
from datus.prompts.reasoning_sql_with_mcp import get_reasoning_prompt
from datus.schemas.node_models import ExecuteSQLResult
from datus.schemas.reason_sql_node_models import ReasoningInput, ReasoningResult
from datus.tools.mcp_server import MCPServer
from datus.utils.constants import DBType
from datus.utils.json_utils import llm_result2json
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@traceable
def reasoning_sql_with_mcp(
    model: LLMBaseModel, input_data: ReasoningInput, db_config: DbConfig, tool_config: Dict[str, Any]
) -> ReasoningResult:
    """Generate SQL via MCP, execute it, and return the execution result."""
    if not isinstance(input_data, ReasoningInput):
        logger.error(f"Input type error: expected ReasoningInput, got {type(input_data)}")
        raise ValueError(f"Input must be a ReasoningInput instance, got {type(input_data)}")

    # logger.info(f"@@@@db_config: {db_config}, input_data: {input_data.sql_task.database_name}")
    mcp_server = MCPServer.get_db_mcp_server(db_config, input_data.sql_task.database_name)

    instruction = prompt_manager.get_raw_template("reasoning_system", input_data.prompt_version)
    # update to python 3.12 to enable structured output
    # output_type = tool_config.get(
    # "output_type", {"sql": str, "tables": list, "explanation": str})
    # tool_list =
    max_turns = tool_config.get("max_turns", 10)

    prompt = get_reasoning_prompt(
        database_type=input_data.get("database_type", DBType.SQLITE),
        table_schemas=input_data.table_schemas,
        data_details=input_data.data_details,
        metrics=input_data.metrics,
        question=input_data.sql_task.task,
        context=[sql_context.to_str(input_data.max_sql_return_length) for sql_context in input_data.contexts],
        prompt_version=input_data.prompt_version,
        max_table_schemas_length=input_data.max_table_schemas_length,
        max_data_details_length=input_data.max_data_details_length,
        max_context_length=input_data.max_context_length,
        max_value_length=input_data.max_value_length,
        max_text_mark_length=input_data.max_text_mark_length,
        knowledge_content=input_data.external_knowledge,
    )
    try:
        exec_result = asyncio.run(
            model.generate_with_mcp(
                prompt=prompt,
                mcp_servers={input_data.sql_task.database_name: mcp_server},
                instruction=instruction,
                # if model is OpenAI, json_schema output is supported, use ReasoningSQLResponse
                output_type=str,
                max_turns=max_turns,
            )
        )

        try:
            logger.debug(f"exec_result: {exec_result['content']}")
            content_dict = llm_result2json(exec_result["content"])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse exec_result.content: {e}, exec_result_strip: {exec_result['content']}")
            content_dict = {}
        # content_dict = exec_result
        # Extract required pieces from the parsed dict
        return ReasoningResult(
            success=True,
            sql_query=content_dict.get("sql", ""),
            sql_return="",  # Remove the result from the return to avoid large data return
            sql_contexts=exec_result["sql_contexts"],
        )
    except Exception as e:
        # TODO : deal with excced the max round
        error_msg = str(e)
        logger.error(f"Reasoning SQL with MCP failed: {e}")

        # Re-raise permission/tool-calling errors so fallback can handle them
        if any(indicator in error_msg.lower() for indicator in ["403", "forbidden", "not allowed", "permission"]):
            logger.info("Re-raising permission error for fallback handling")
            raise

        # Return failed result for other errors
        return ExecuteSQLResult(success=False, error=str(e), sql_query="", row_count=0, sql_return="")
