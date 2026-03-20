# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Ask User tool — allows the agent to pause and ask the user questions.

When the LLM is uncertain about the user's intent or needs clarification,
it can call ``ask_user`` to present one or more questions with optional
predefined options. The tool blocks until the user responds to all
questions, then returns the answers so the agent can continue.
"""

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from datus.cli.execution_state import InteractionBroker, InteractionCancelled
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class QuestionItem(BaseModel):
    """A single question to ask the user."""

    question: str = Field(description="The question to ask. Should be clear and specific.")
    options: Optional[List[str]] = Field(
        default=None,
        description="2-5 predefined answer choices. The user can always type a custom answer "
        "even when options are provided, so do NOT include an 'Other' or 'Custom' option. "
        "If omitted, the user provides free-text input.",
    )


class AskUserTool:
    """Tool that lets the agent ask the user one or more questions.

    Each question can optionally include predefined options. The tool uses
    ``InteractionBroker`` to present all questions to the user and wait for
    their responses. The user can pick one of the provided options or type
    a custom answer for each question.

    Args:
        broker: InteractionBroker instance (shared with permission hooks).
    """

    MAX_QUESTIONS = 10
    MAX_OPTIONS_PER_QUESTION = 5
    MIN_OPTIONS_PER_QUESTION = 2

    def __init__(self, broker: InteractionBroker):
        self._broker = broker
        self._tool_context: Any = None

    def set_tool_context(self, ctx: Any) -> None:
        self._tool_context = ctx

    async def ask_user(
        self,
        questions: List[QuestionItem],
    ) -> FuncToolResult:
        """Ask the user one or more questions and wait for their responses.

        Use this tool when you need clarification from the user before
        proceeding. For example:
        - The user's request is ambiguous and could be interpreted multiple ways
        - You need the user to choose between several approaches
        - You want to confirm important actions before executing them

        Collect ALL questions you need to ask into a single call rather than
        asking one at a time.

        Args:
            questions: A list of question objects, each with a "question" string
                and optional "options" list of 2-5 predefined answer choices.

        Returns:
            FuncToolResult with the answers in the ``result`` field.
            The result is a JSON array of answer objects, each containing
            "question" and "answer" keys.
        """
        # --- validation ---
        if not questions or not isinstance(questions, list):
            return FuncToolResult(success=0, error="questions must be a non-empty list")

        if len(questions) > self.MAX_QUESTIONS:
            return FuncToolResult(success=0, error=f"questions must contain at most {self.MAX_QUESTIONS} items")

        validated: List[Dict[str, Any]] = []
        for i, q in enumerate(questions):
            # Accept both QuestionItem and plain dict (from tests / non-SDK callers)
            if isinstance(q, QuestionItem):
                q_text = q.question
                options = q.options
            elif isinstance(q, dict):
                q_text = q.get("question", "")
                options = q.get("options")
            else:
                return FuncToolResult(success=0, error=f"questions[{i}] must be a dict")

            if not q_text or not str(q_text).strip():
                return FuncToolResult(success=0, error=f"questions[{i}].question must not be empty")
            if options is not None:
                if not isinstance(options, list):
                    return FuncToolResult(success=0, error=f"questions[{i}].options must be a list")
                if len(options) < self.MIN_OPTIONS_PER_QUESTION or len(options) > self.MAX_OPTIONS_PER_QUESTION:
                    return FuncToolResult(
                        success=0,
                        error=f"questions[{i}].options must contain "
                        f"{self.MIN_OPTIONS_PER_QUESTION}-{self.MAX_OPTIONS_PER_QUESTION} items",
                    )
                # Coerce all options to non-empty strings
                options = [str(opt) for opt in options]
                if any(not opt.strip() for opt in options):
                    return FuncToolResult(success=0, error=f"questions[{i}].options must be non-empty strings")
            validated.append({"question": str(q_text).strip(), "options": options})

        # --- build display content ---
        content = self._build_content(validated)

        try:
            choice, callback = await self._broker.request(
                content=content,
                choices={},
                default_choice="",
                allow_free_text=True,
                action_type="request_batch",
                questions=validated,
            )

            # Reject None response (collector failure)
            if choice is None:
                logger.warning("AskUserTool: collector returned None (interaction failure)")
                return FuncToolResult(success=0, error="No response received from collector")

            # Parse the JSON response from the collector
            try:
                answers = json.loads(choice)
            except (json.JSONDecodeError, TypeError):
                if len(validated) == 1:
                    answers = [choice]
                else:
                    logger.warning("AskUserTool: expected JSON array for multi-question batch response")
                    return FuncToolResult(success=0, error="Malformed batch response from collector")

            # Ensure answers is a list (json.loads may return dict/str/int)
            if not isinstance(answers, list):
                if len(validated) == 1:
                    answers = [answers]
                else:
                    logger.warning(f"AskUserTool: expected list batch response, got type={type(answers).__name__}")
                    return FuncToolResult(success=0, error="Malformed batch response from collector")

            # Validate answer count matches question count
            if len(answers) != len(validated):
                logger.warning(f"AskUserTool: answer count mismatch (expected {len(validated)}, got {len(answers)})")
                return FuncToolResult(success=0, error="Malformed batch response from collector")

            # Build structured result
            result_list = []
            for i, q in enumerate(validated):
                result_list.append({"question": q["question"], "answer": answers[i]})

            result_json = json.dumps(result_list, ensure_ascii=False)
            await callback(f"User answered {len(result_list)} question(s)")

            logger.info(f"AskUserTool: completed batch clarification with {len(validated)} question(s)")
            return FuncToolResult(success=1, result=result_json)

        except InteractionCancelled:
            logger.info("AskUserTool: interaction cancelled by user")
            return FuncToolResult(success=0, error="User cancelled the question")
        except Exception as e:
            logger.error(f"AskUserTool: unexpected error: {e}")
            return FuncToolResult(success=0, error=f"Failed to ask user: {e}")

    @staticmethod
    def _build_content(questions: List[Dict[str, Any]]) -> str:
        """Build markdown display content for the questions."""
        if len(questions) == 1:
            q = questions[0]
            return f"### Agent Question\n\n{q['question']}"

        lines = [f"### Agent Questions ({len(questions)} questions)\n"]
        for i, q in enumerate(questions, 1):
            lines.append(f"**{i}. {q['question']}**")
            if q.get("options"):
                opts = " / ".join(q["options"])
                lines.append(f"   Options: {opts}")
            else:
                lines.append("   _(free text)_")
            lines.append("")
        return "\n".join(lines)

    def available_tools(self):
        """Return list of FunctionTool instances for this tool group."""
        return [trans_to_function_tool(self.ask_user)]
