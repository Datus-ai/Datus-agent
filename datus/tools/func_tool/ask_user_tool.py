# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
AskUserTool — allows an LLM agent to ask the user a question during execution.

Uses the existing InteractionBroker mechanism for async agent-user interaction.
Supports free-text input and multiple-choice selection.
"""

import json

from datus.cli.execution_state import InteractionBroker, InteractionCancelled
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class AskUserTool:
    """Tool that enables the LLM agent to ask the user questions during execution.

    Wraps the InteractionBroker to provide a simple function-tool interface.
    """

    def __init__(self, broker: InteractionBroker):
        self.broker = broker

    async def ask_user(
        self,
        question: str,
        choices_json: str = "",
        content_type: str = "markdown",
    ) -> FuncToolResult:
        """
        Ask the user a question and wait for their response.

        Use this tool when you need information from the user, such as:
        - A file path (e.g., CSV file with SQL queries)
        - Clarification on requirements
        - Confirmation before proceeding with an action
        - SQL queries, table names, or other domain-specific input

        Args:
            question: The question or prompt to display to the user. Supports markdown formatting.
            choices_json: Optional JSON string of {key: display_text} for multiple-choice selection.
                     If empty, the user can provide free-text input.
                     Example: '{"y": "Yes - proceed", "n": "No - cancel"}'
            content_type: How to render the question. One of "text", "markdown", "yaml", "sql".
                          Defaults to "markdown".

        Returns:
            FuncToolResult with the user's response in result["response"].
        """
        try:
            effective_choices: dict = {}
            if choices_json:
                try:
                    effective_choices = json.loads(choices_json)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Invalid choices_json: {choices_json}, treating as free-text")

            default_choice = ""
            if effective_choices:
                default_choice = next(iter(effective_choices))

            choice, callback = await self.broker.request(
                content=question,
                choices=effective_choices,
                default_choice=default_choice,
                content_type=content_type,
            )

            await callback(f"User responded: {choice}")

            return FuncToolResult(
                success=1,
                result={"response": choice},
            )

        except InteractionCancelled:
            return FuncToolResult(
                success=0,
                error="User cancelled the interaction",
            )
        except Exception as e:
            logger.error(f"AskUserTool error: {e}")
            return FuncToolResult(
                success=0,
                error=str(e),
            )

    def available_tools(self) -> list:
        """Return list of function tools for agent registration."""
        return [trans_to_function_tool(self.ask_user)]
