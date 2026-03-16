# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Ask User tool — allows the agent to pause and ask the user a question.

When the LLM is uncertain about the user's intent or needs clarification,
it can call ``ask_user`` to present a question with options. The tool blocks
until the user responds, then returns the answer so the agent can continue.
"""

from typing import Any, List, Optional

from datus.cli.execution_state import InteractionBroker, InteractionCancelled
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class AskUserTool:
    """Tool that lets the agent ask the user a question with predefined options.

    The tool uses ``InteractionBroker`` to present a question to the user and
    wait for their response. The user can pick one of the provided options or
    type a custom answer.

    Args:
        broker: InteractionBroker instance (shared with permission hooks).
    """

    def __init__(self, broker: InteractionBroker):
        self._broker = broker
        self._tool_context: Any = None

    def set_tool_context(self, ctx: Any) -> None:
        self._tool_context = ctx

    async def ask_user(
        self,
        question: str,
        options: Optional[List[str]] = None,
    ) -> FuncToolResult:
        """Ask the user a question and wait for their response.

        Use this tool when you need clarification from the user before
        proceeding. For example:
        - The user's request is ambiguous and could be interpreted multiple ways
        - You need the user to choose between several approaches
        - You want to confirm an important action before executing it

        Args:
            question: The question to ask the user. Should be clear and specific.
            options: Optional list of 2-5 predefined answer choices.
                     The user can always type a custom answer even when options
                     are provided, so do NOT include an "Other" or "Custom"
                     option — the free-text input is built-in.
                     If omitted, the user provides free-text input.

        Returns:
            FuncToolResult with the user's answer in the ``result`` field.
        """
        if not question or not question.strip():
            return FuncToolResult(success=0, error="Question must not be empty")

        # Build choices dict for the broker
        choices: dict = {}
        if options:
            if len(options) < 2 or len(options) > 5:
                return FuncToolResult(success=0, error="options must contain 2-5 items")
            for i, opt in enumerate(options, 1):
                choices[str(i)] = opt

        # Build display content
        content = f"### Agent Question\n\n{question}"

        try:
            choice, callback = await self._broker.request(
                content=content,
                choices=choices,
                default_choice="1" if choices else "",
                allow_free_text=True,
            )

            # Resolve the display text for the answer
            if choices and choice in choices:
                answer = choices[choice]
            else:
                answer = choice  # free-text input

            await callback(f"User answered: {answer}")

            logger.info(f"AskUserTool: question='{question}', answer='{answer}'")
            return FuncToolResult(success=1, result=answer)

        except InteractionCancelled:
            logger.info("AskUserTool: interaction cancelled by user")
            return FuncToolResult(success=0, error="User cancelled the question")
        except Exception as e:
            logger.error(f"AskUserTool: unexpected error: {e}")
            return FuncToolResult(success=0, error=f"Failed to ask user: {e}")

    def available_tools(self):
        """Return list of FunctionTool instances for this tool group."""
        return [trans_to_function_tool(self.ask_user)]
