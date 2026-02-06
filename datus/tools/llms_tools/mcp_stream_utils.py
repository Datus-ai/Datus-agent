# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from agents import Tool

from datus.models.base import LLMBaseModel
from datus.prompts.prompt_manager import prompt_manager
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


async def base_mcp_stream(
    model: LLMBaseModel,
    input_data: Any,
    tool_config: Dict[str, Any],
    mcp_servers: Dict[str, Any],
    prompt: Union[str, List[dict[str, str]]],
    instruction_template: str,
    tools: Optional[List[Tool]] = None,
    action_history_manager: Optional[ActionHistoryManager] = None,
) -> AsyncGenerator[ActionHistory, None]:
    """Base MCP streaming function that yields only function call actions.

    Args:
        model: LLM model instance
        input_data: Input data for the operation
        tools: database function tools
        tool_config: Tool configuration, which tools do you want to use, and max_truns
        mcp_servers: Dictionary of MCP servers
        prompt: prompt
        instruction_template: Template name for instruction
        action_history_manager: Optional action history manager

    Yields:
        ActionHistory objects for function calls only
    """
    if action_history_manager is None:
        action_history_manager = ActionHistoryManager()

    try:
        # Get instruction and generate prompt
        instruction = prompt_manager.get_raw_template(instruction_template, input_data.prompt_version)
        max_turns = tool_config.get("max_turns", 10)

        logger.info(f"Starting MCP stream with {len(mcp_servers)} servers, max_turns={max_turns}")
        logger.debug(f"MCP servers: {list(mcp_servers.keys())}")
        # Stream function calls only
        async for action in model.generate_with_tools_stream(
            prompt=prompt,
            tools=tools,
            mcp_servers=mcp_servers,
            instruction=instruction,
            output_type=str,
            max_turns=max_turns,
            action_history_manager=action_history_manager,
        ):
            logger.debug(f"Yielding action: {action.action_id}")
            yield action

    except Exception as e:
        logger.error(f"Base MCP stream failed: {e}")
        # Re-raise permission errors for fallback handling
        error_msg = str(e).lower()
        if any(indicator in error_msg for indicator in ["403", "forbidden", "not allowed", "permission"]):
            logger.info("Re-raising permission error for fallback handling")
            raise

        # Handle OpenAI API errors with user-friendly messages
        if any(
            indicator in error_msg
            for indicator in ["overloaded", "rate limit", "timeout", "connection error", "server error"]
        ):
            logger.info("Re-raising OpenAI API error for retry handling")
            raise

        # For other errors, provide generic error message
        logger.error(f"Unexpected error in MCP stream: {e}")
        raise
