import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, AsyncGenerator

from langsmith import traceable

from datus.configuration.agent_config import DbConfig
from datus.models.base import LLMBaseModel
from datus.prompts.generate_semantic_model import get_generate_semantic_model_prompt
from datus.prompts.prompt_manager import prompt_manager
from datus.schemas.generate_semantic_model_node_models import GenerateSemanticModelInput, GenerateSemanticModelResult
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionType
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

    if action_history_manager is None:
        action_history_manager = ActionHistoryManager()

    # Initialize action
    action_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    # Add initial action
    initial_action = ActionHistory(
        action_id=action_id,
        role=ActionRole.MODEL,
        thought="Starting semantic model generation",
        action_type=ActionType.FUNCTION_CALL,
        input={
            "table_definition": table_definition,
            "sql_query": input_data.sql_query,
            "semantic_model_meta": input_data.semantic_model_meta.dict(),
        },
        timestamp=timestamp,
    )
    action_history_manager.add_action(initial_action)
    yield initial_action

    try:
        # Setup MCP server
        filesystem_mcp_server = MCPServer.get_filesystem_mcp_server()

        # Update action with MCP server setup
        mcp_setup_action = ActionHistory(
            action_id=str(uuid.uuid4()),
            role=ActionRole.WORKFLOW,
            thought="Setting up MCP filesystem server",
            action_type=ActionType.FUNCTION_CALL,
            input={"mcp_server": "filesystem"},
            output={"status": "connected"},
            timestamp=datetime.now().isoformat(),
        )
        action_history_manager.add_action(mcp_setup_action)
        yield mcp_setup_action

        # Get prompt and instruction
        instruction = prompt_manager.get_raw_template("generate_semantic_model_system", input_data.prompt_version)
        max_turns = tool_config.get("max_turns", 20)

        prompt = get_generate_semantic_model_prompt(
            database_type=db_config.type,
            table_definition=table_definition,
            prompt_version=input_data.prompt_version,
        )

        # Update action with prompt generation
        prompt_action = ActionHistory(
            action_id=str(uuid.uuid4()),
            role=ActionRole.WORKFLOW,
            thought="Generated prompt for semantic model creation",
            action_type=ActionType.CHAT,
            input={"database_type": db_config.type, "prompt_version": input_data.prompt_version},
            output={"prompt_length": len(prompt), "instruction_length": len(instruction)},
            timestamp=datetime.now().isoformat(),
        )
        action_history_manager.add_action(prompt_action)
        yield prompt_action

        # Execute LLM generation
        llm_action = ActionHistory(
            action_id=str(uuid.uuid4()),
            role=ActionRole.MODEL,
            thought="Executing LLM generation with MCP",
            action_type=ActionType.FUNCTION_CALL,
            input={"prompt": prompt[:100] + "...", "max_turns": max_turns},
            timestamp=datetime.now().isoformat(),
        )
        action_history_manager.add_action(llm_action)
        yield llm_action

        # Execute the actual generation
        exec_result = await model.generate_with_mcp(
            prompt=prompt,
            mcp_servers={
                "filesystem_mcp_server": filesystem_mcp_server,
            },
            instruction=instruction,
            output_type=str,
            max_turns=max_turns,
        )

        # Update LLM action with result
        action_history_manager.update_current_action(
            output={"content_length": len(exec_result.get("content", ""))},
            reflection="LLM generation completed successfully",
        )

        # Parse result
        parse_action = ActionHistory(
            action_id=str(uuid.uuid4()),
            role=ActionRole.WORKFLOW,
            thought="Parsing LLM output to extract semantic model",
            action_type=ActionType.FUNCTION_CALL,
            input={"raw_content": exec_result.get("content", "")[:100] + "..."},
            timestamp=datetime.now().isoformat(),
        )
        action_history_manager.add_action(parse_action)
        yield parse_action

        try:
            logger.debug(f"exec_result: {exec_result['content']}")
            content_dict = json.loads(strip_json_str(exec_result["content"]))

            # Update parse action with success
            action_history_manager.update_current_action(
                output=content_dict,
                reflection="Successfully parsed semantic model JSON",
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse exec_result.content: {e}, exec_result: {exec_result}")
            content_dict = {}

            # Update parse action with error
            action_history_manager.update_current_action(
                output={"error": str(e)},
                reflection=f"Failed to parse JSON: {str(e)}",
            )

        # Final result action
        semantic_model_meta = input_data.semantic_model_meta
        semantic_model_meta.table_name = content_dict.get("table_name", "")
        semantic_model_meta.schema_name = content_dict.get("schema_name", "")

        final_action = ActionHistory(
            action_id=str(uuid.uuid4()),
            role=ActionRole.WORKFLOW,
            thought="Finalizing semantic model result",
            action_type=ActionType.FUNCTION_CALL,
            input={"semantic_model_meta": semantic_model_meta.dict()},
            output={
                "success": True,
                "semantic_model_file": content_dict.get("semantic_model_file", ""),
                "table_name": content_dict.get("table_name", ""),
                "schema_name": content_dict.get("schema_name", ""),
            },
            reflection="Semantic model generation completed successfully",
            timestamp=datetime.now().isoformat(),
        )
        action_history_manager.add_action(final_action)
        yield final_action

    except Exception as e:
        logger.error(f"Generate semantic model failed: {e}")

        # Error action
        error_action = ActionHistory(
            action_id=str(uuid.uuid4()),
            role=ActionRole.WORKFLOW,
            thought="Error occurred during semantic model generation",
            action_type=ActionType.FUNCTION_CALL,
            input={"error": str(e)},
            output={"success": False, "error": str(e)},
            reflection=f"Generation failed: {str(e)}",
            timestamp=datetime.now().isoformat(),
        )
        action_history_manager.add_action(error_action)
        yield error_action


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
