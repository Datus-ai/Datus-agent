from typing import Any, AsyncGenerator, Callable, Dict, Optional

from langsmith import traceable

from datus.configuration.agent_config import DbConfig
from datus.models.base import LLMBaseModel
from datus.prompts.prompt_manager import prompt_manager
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@traceable
async def base_mcp_stream(
    model: LLMBaseModel,
    input_data: Any,
    db_config: DbConfig,
    tool_config: Dict[str, Any],
    mcp_server_getter: Callable,
    prompt_generator: Callable,
    instruction_template: str,
    action_history_manager: Optional[ActionHistoryManager] = None,
) -> AsyncGenerator[ActionHistory, None]:
    """Base MCP streaming function that yields only function call actions.

    Args:
        model: LLM model instance
        input_data: Input data for the operation
        db_config: Database configuration
        tool_config: Tool configuration, which tools do you want to use, and max_truns
        mcp_server_getter: Function to get MCP server
        prompt_generator: Function to generate prompt
        instruction_template: Template name for instruction
        action_history_manager: Optional action history manager

    Yields:
        ActionHistory objects for function calls only
    """
    if action_history_manager is None:
        action_history_manager = ActionHistoryManager()

    try:
        # Setup MCP server
        mcp_server = mcp_server_getter()

        # Get instruction and generate prompt
        instruction = prompt_manager.get_raw_template(instruction_template, input_data.prompt_version)
        max_turns = tool_config.get("max_turns", 10)
        prompt = prompt_generator(input_data, db_config)

        # Determine MCP servers dict based on server type
        if hasattr(input_data, "sql_task") and hasattr(input_data.sql_task, "database_name"):
            mcp_servers = {input_data.sql_task.database_name: mcp_server}
        else:
            mcp_servers = {"filesystem_mcp_server": mcp_server}

        # Stream function calls only
        async for action in model.generate_with_mcp_stream(
            prompt=prompt,
            mcp_servers=mcp_servers,
            instruction=instruction,
            output_type=str,
            max_turns=max_turns,
            action_history_manager=action_history_manager,
        ):
            yield action

    except Exception as e:
        logger.error(f"Base MCP stream failed: {e}")
        # Re-raise permission errors for fallback handling
        error_msg = str(e)
        if any(indicator in error_msg.lower() for indicator in ["403", "forbidden", "not allowed", "permission"]):
            logger.info("Re-raising permission error for fallback handling")
            raise
