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
from datus.schemas.interaction_event import InteractionEvent
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class QuestionItem(BaseModel):
    """A single question to ask the user."""

    question: str = Field(description="The question to ask. Should be clear and specific.")
    options: Optional[List[str]] = Field(
        default=None,
        description="2-10 predefined answer choices. The user can always type a custom answer "
        "even when options are provided, so do NOT include an 'Other' or 'Custom' option. "
        "If omitted, the user provides free-text input.",
    )
    multi_select: bool = Field(
        default=False,
        description="When True, the user can select multiple options (checkbox-style). "
        "The answer will be a list of selected option texts. "
        "Only meaningful when options are provided.",
    )
    title: str = Field(
        default="Question",
        description="Short label (1-2 words) for the interaction header tab. "
        "Example: 'Database', 'Approach', 'Confirm'.",
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
    MAX_OPTIONS_PER_QUESTION = 10
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
            questions: A JSON array of question objects (NOT a JSON string).
                Each object has a "question" string, optional "options" list,
                and a "title" label (1-2 words).
                Example: [{"question": "Which DB?", "options": ["MySQL", "PostgreSQL"], "title": "Database"}]

        Returns:
            FuncToolResult with the answers in the ``result`` field.
            The result is a JSON array of answer objects, each containing
            "question" and "answer" keys.
        """
        # --- coerce JSON-string to list (LLMs sometimes double-serialize) ---
        if isinstance(questions, str):
            try:
                questions = json.loads(questions)
            except (json.JSONDecodeError, TypeError):
                return FuncToolResult(success=0, error="questions must be a non-empty list (got unparseable string)")

        # --- validation ---
        if not questions or not isinstance(questions, list):
            return FuncToolResult(success=0, error="questions must be a non-empty list")

        if len(questions) > self.MAX_QUESTIONS:
            return FuncToolResult(success=0, error=f"questions must contain at most {self.MAX_QUESTIONS} items")

        validated: List[Dict[str, Any]] = []
        for i, q in enumerate(questions):
            if isinstance(q, QuestionItem):
                q_text = q.question
                options = q.options
                multi_select = q.multi_select
                q_title = q.title
            elif isinstance(q, dict):
                q_text = q.get("question", "")
                options = q.get("options")
                multi_select = q.get("multi_select", False)
                q_title = q.get("title", "Question")
            else:
                return FuncToolResult(success=0, error=f"questions[{i}] must be a dict")

            if not q_text or not str(q_text).strip():
                return FuncToolResult(success=0, error=f"questions[{i}].question must not be empty")
            if isinstance(options, list) and len(options) == 0:
                options = None
            if options is not None:
                if not isinstance(options, list):
                    return FuncToolResult(success=0, error=f"questions[{i}].options must be a list")
                if len(options) < self.MIN_OPTIONS_PER_QUESTION or len(options) > self.MAX_OPTIONS_PER_QUESTION:
                    return FuncToolResult(
                        success=0,
                        error=f"questions[{i}].options must contain "
                        f"{self.MIN_OPTIONS_PER_QUESTION}-{self.MAX_OPTIONS_PER_QUESTION} items",
                    )
                options = [str(opt) for opt in options]
                if any(not opt.strip() for opt in options):
                    return FuncToolResult(success=0, error=f"questions[{i}].options must be non-empty strings")
            choices_dict = {str(j): opt for j, opt in enumerate(options, 1)} if options else None
            validated.append(
                {
                    "question": str(q_text).strip(),
                    "choices": choices_dict,
                    "multi_select": bool(multi_select) if options else False,
                    "title": str(q_title).strip() or "Question",
                }
            )

        # --- build InteractionEvent list and pass to broker ---
        events = [
            InteractionEvent(
                title=q["title"],
                content=q["question"],
                choices=q["choices"] or {},
                allow_free_text=True,
                multi_select=q["multi_select"],
            )
            for q in validated
        ]

        try:
            raw_answers = await self._broker.request(events)

            # raw_answers is List[List[str]] — one inner list per question
            if len(raw_answers) != len(validated):
                logger.warning(
                    f"AskUserTool: answer count mismatch (expected {len(validated)}, got {len(raw_answers)})"
                )
                return FuncToolResult(success=0, error="Answer count mismatch from collector")

            # Resolve choice keys to display values (e.g. "2" → "PostgreSQL")
            result_list = []
            for i, q in enumerate(validated):
                answer_list = raw_answers[i]
                if q["multi_select"] and q["choices"]:
                    resolved = [q["choices"].get(str(k), str(k)) for k in answer_list]
                    result_list.append({"question": q["question"], "answer": resolved})
                elif q["choices"] and len(answer_list) == 1 and answer_list[0] in q["choices"]:
                    result_list.append({"question": q["question"], "answer": q["choices"][answer_list[0]]})
                else:
                    result_list.append({"question": q["question"], "answer": answer_list[0] if answer_list else ""})

            result_json = json.dumps(result_list, ensure_ascii=False)
            logger.info(f"AskUserTool: completed batch clarification with {len(validated)} question(s)")
            return FuncToolResult(success=1, result=result_json)

        except InteractionCancelled:
            logger.info("AskUserTool: interaction cancelled by user")
            return FuncToolResult(success=0, error="User cancelled the question")
        except Exception as e:
            logger.error(f"AskUserTool: unexpected error: {e}")
            return FuncToolResult(success=0, error=f"Failed to ask user: {e}")

    def available_tools(self):
        """Return list of FunctionTool instances for this tool group."""
        return [trans_to_function_tool(self.ask_user)]
