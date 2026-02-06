# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

from agents import Tool

from datus.models.base import LLMBaseModel
from datus.prompts.prompt_manager import prompt_manager
from datus.prompts.reasoning_sql_with_mcp import get_reasoning_prompt
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.schemas.reason_sql_node_models import ReasoningInput, ReasoningResult
from datus.tools.llms_tools.mcp_stream_utils import base_mcp_stream
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.json_utils import llm_result2json, llm_result2sql
from datus.utils.loggings import get_logger
from datus.utils.traceable_utils import optional_traceable

logger = get_logger(__name__)


async def reasoning_sql_with_mcp_stream(
    model: LLMBaseModel,
    input_data: ReasoningInput,
    tool_config: Dict[str, Any],
    tools: List[Tool],
    action_history_manager: Optional[ActionHistoryManager] = None,
) -> AsyncGenerator[ActionHistory, None]:
    """Generate SQL reasoning with streaming support and action history tracking."""
    if not isinstance(input_data, ReasoningInput):
        logger.error(f"Input type error: expected ReasoningInput, got {type(input_data)}")
        raise ValueError(f"Input must be a ReasoningInput instance, got {type(input_data)}")

    prompt = get_reasoning_prompt(
        database_type=input_data.get("database_type", "sqlite"),
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

    # Setup MCP servers
    # If no action history manager provided, create one to track the final result
    if action_history_manager is None:
        action_history_manager = ActionHistoryManager()

    async for action in base_mcp_stream(
        model=model,
        input_data=input_data,
        tool_config=tool_config,
        mcp_servers={},
        prompt=prompt,
        tools=tools,
        instruction_template="reasoning_system",
        action_history_manager=action_history_manager,
    ):
        yield action

    # After streaming completes, extract final result and add to SQLContext
    try:
        # Find the final message/result from action history
        final_message_action = None
        sql_contexts = []

        # Look for actions that contain SQL execution results
        for action in action_history_manager.actions:
            if action.action_type == "read_query" and action.status.value == "success":
                # This is a SQL execution result, create SQLContext from it
                from datus.schemas.node_models import SQLContext

                sql_input = action.input or {}
                sql_output = action.output or {}

                sql_context = SQLContext(
                    sql_query=sql_input.get("sql", ""),
                    explanation="",
                    sql_return=sql_output.get("result", ""),
                    sql_error=sql_output.get("error", ""),
                    row_count=0,
                )
                sql_contexts.append(sql_context)

            elif action.action_type == "message" and action.role.value == "assistant":
                # This could be the final reasoning result
                final_message_action = action

        # Extract the final SQL from the final message if available
        if final_message_action and final_message_action.output:
            raw_output = final_message_action.output.get("raw_output", "")
            if raw_output:
                try:
                    # Parse the final result to extract SQL
                    content_dict = llm_result2json(raw_output)
                    sql_query = content_dict.get("sql", "")

                    if sql_query:
                        # Create SQLContext with the final result SQL
                        from datus.schemas.node_models import SQLContext

                        final_sql_context = SQLContext(
                            sql_query=sql_query,
                            explanation=content_dict.get("explanation", ""),
                            sql_return="",  # Will be filled by execution
                            sql_error="",
                            row_count=0,
                        )
                        sql_contexts.append(final_sql_context)
                        logger.info(f"Added final result SQL to SQLContext: {sql_query[:100]}...")

                except Exception as e:
                    logger.debug(f"Could not parse final message as JSON: {e}")

        # Store sql_contexts in action history manager for later retrieval
        if not hasattr(action_history_manager, "sql_contexts"):
            action_history_manager.sql_contexts = []
        action_history_manager.sql_contexts.extend(sql_contexts)

    except Exception as e:
        logger.warning(f"Failed to extract final result SQL for SQLContext: {e}")
        # Don't fail the entire process, just log the warning


@optional_traceable()
def reasoning_sql_with_mcp(
    model: LLMBaseModel, input_data: ReasoningInput, tools: List[Tool], tool_config: Dict[str, Any]
) -> ReasoningResult:
    """Generate SQL via MCP, execute it, and return the execution result."""
    if not isinstance(input_data, ReasoningInput):
        logger.error(f"Input type error: expected ReasoningInput, got {type(input_data)}")
        raise ValueError(f"Input must be a ReasoningInput instance, got {type(input_data)}")

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
            model.generate_with_tools(
                prompt=prompt,
                mcp_servers={},
                instruction=instruction,
                tools=tools,
                # if model is OpenAI, json_schema output is supported, use ReasoningSQLResponse
                output_type=str,
                max_turns=max_turns,
            )
        )

        logger.debug(f"Reasoning SQL execute result: {exec_result['content']}")

        # Try JSON parsing first
        content_dict = llm_result2json(exec_result["content"])
        if content_dict:
            # Successfully parsed JSON with meaningful SQL content
            logger.info(f"Successfully parsed JSON content: {content_dict}")
            reasoning_result = ReasoningResult(
                success=True,
                sql_query=content_dict.get("sql", ""),
                sql_return="",  # Remove the result from the return to avoid large data return
                sql_contexts=exec_result["sql_contexts"],
            )
            logger.info(
                f"Created ReasoningResult: success={reasoning_result.success}, sql_query={reasoning_result.sql_query}"
            )
            return reasoning_result

        # JSON parsing failed, try SQL extraction.
        # Some LLM can't follow the instruction well, try some failback
        extracted_sql = llm_result2sql(exec_result["content"])
        if extracted_sql:
            # Successfully extracted SQL from code blocks
            logger.info(f"Extract json format failed, but find a sql {extracted_sql} from response")
            return ReasoningResult(
                success=True,
                sql_query=extracted_sql,
                sql_return="",
                sql_contexts=exec_result["sql_contexts"],
            )

        # Both JSON and SQL extraction failed, raise exception
        response_content = exec_result["content"]
        response_preview = response_content[:20] if response_content else ""
        response_length = len(response_content) if response_content else 0
        logger.error(f"Extract json format/sql failed. len:{response_length}, resp:{response_preview}... ")
        raise DatusException(
            ErrorCode.MODEL_ILLEGAL_FORMAT_RESPONSE,
            message_args={"response_preview": response_preview, "response_length": response_length},
        )

    except DatusException:
        raise
    except Exception as e:
        # TODO : deal with exceed the max round
        error_msg = str(e)
        logger.error(f"Reasoning SQL with MCP failed: {e}")

        # Re-raise permission/tool-calling errors so fallback can handle them
        if any(indicator in error_msg.lower() for indicator in ["403", "forbidden", "not allowed", "permission"]):
            logger.info("Re-raising permission error for fallback handling")
            raise

        # Return failed result for other errors
        logger.error(f"Reasoning SQL failed: {e}")
        raise DatusException(
            ErrorCode.NODE_EXECUTION_FAILED,
            message=f"Reasoning SQL failed: {e}",
        )
