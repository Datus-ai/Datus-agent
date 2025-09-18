import json
from typing import Any, AsyncGenerator, Dict, List, Optional

from agents import Tool

from datus.configuration.agent_config import DbConfig
from datus.models.base import LLMBaseModel
from datus.prompts.compare_sql import compare_sql_prompt
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.schemas.compare_node_models import CompareInput, CompareResult
from datus.tools.llms_tools.mcp_stream_utils import base_mcp_stream
from datus.utils.loggings import get_logger
from datus.utils.traceable_utils import optional_traceable

logger = get_logger(__name__)


def generate_compare_prompt(input_data: CompareInput):
    """Generate comparison prompt with database context."""
    sql_result = input_data.sql_context.sql_return if hasattr(input_data.sql_context, "sql_return") else ""
    sql_error = input_data.sql_context.sql_error if hasattr(input_data.sql_context, "sql_error") else ""

    return compare_sql_prompt(
        sql_task=input_data.sql_task,
        prompt_version=input_data.prompt_version,
        sql_query=input_data.sql_context.sql_query,
        sql_explanation=input_data.sql_context.explanation,
        sql_result=sql_result,
        sql_error=sql_error,
        expectation=input_data.expectation,
    )


@optional_traceable()
def compare_sql(model: LLMBaseModel, input_data) -> CompareResult:
    """Compare SQL query with expectation using the provided model."""
    if not isinstance(input_data, CompareInput):
        raise ValueError("Input must be a CompareInput instance")

    try:
        prompt = generate_compare_prompt(input_data)
        logger.debug(f"Compare SQL prompt: {type(model)}, {prompt}")

        # Generate comparison using the provided model
        comparison_result = model.generate_with_json_output(prompt)
        logger.debug(f"Comparison result: {comparison_result}")

        # Clean and parse the response
        if isinstance(comparison_result, str):
            # Remove markdown code blocks if present
            comparison_result = comparison_result.strip().replace("```json\n", "").replace("\n```", "")
            try:
                result_dict = json.loads(comparison_result)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse comparison result: {e}")
                result_dict = {
                    "explanation": f"Failed to parse model response: {comparison_result}",
                    "suggest": "Please check the model response format",
                }
        else:
            result_dict = comparison_result

        return CompareResult(
            success=True,
            explanation=result_dict.get("explanation", "No explanation provided"),
            suggest=result_dict.get("suggest", "No suggestions provided"),
        )

    except Exception as e:
        logger.error(f"SQL comparison failed: {str(e)}")
        return CompareResult(
            success=False,
            error=str(e),
            explanation="Comparison analysis failed",
            suggest="Please check the input parameters and try again",
        )


@optional_traceable()
async def compare_sql_with_mcp_stream(
    model: LLMBaseModel,
    input_data: CompareInput,
    db_config: DbConfig,
    tool_config: Dict[str, Any],
    tools: Optional[List[Tool]] = None,
    action_history_manager: Optional[ActionHistoryManager] = None,
) -> AsyncGenerator[ActionHistory, None]:
    """Compare SQL query with expectation using MCP streaming support."""
    if not isinstance(input_data, CompareInput):
        logger.error(f"Input type error: expected CompareInput, got {type(input_data)}")
        raise ValueError(f"Input must be a CompareInput instance, got {type(input_data)}")

    # Setup MCP servers

    tool_config["max_turns"] = 30

    async for action in base_mcp_stream(
        model=model,
        input_data=input_data,
        tool_config=tool_config,
        prompt=generate_compare_prompt(input_data=input_data),
        mcp_servers={},
        instruction_template="compare_sql_system_mcp",
        tools=tools,
        action_history_manager=action_history_manager,
    ):
        yield action
