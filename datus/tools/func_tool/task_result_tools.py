# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Structured task outcome, for runs started by an external orchestrator.

An orchestrator that dispatches work here has to know how the run ended before
it can decide what happens next — was the question answered, does something need
building first, or is the project simply unable to help? Parsing that back out
of prose is unreliable in three separate ways: the model may not say it, the
wording drifts, and a JSON block gets wrapped in fences. A tool call is
schema-validated, arrives as its own frame in the stream, and the schema itself
is far better at steering the model than an instruction to "end with JSON".

The tool is only injected when the request declares an orchestrator origin, so
ordinary IDE chat never sees it.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from agents import FunctionTool
from pydantic import BaseModel, Field

from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Same closed set the caller's deliverables use. Deliberately not a private
# vocabulary: a second enum plus a mapping is one more thing to forget to update
# when a kind is added, and the failure would be silent.
PlanItemKind = Literal["dimension", "table", "metric", "report", "dashboard", "dag"]

TaskOutcome = Literal["answered", "needs_development", "blocked"]


class PlanItem(BaseModel):
    """One thing that has to be built before the request can be answered."""

    kind: PlanItemKind = Field(description="What sort of object this is.")
    name: str = Field(description="Proposed object name, e.g. 'dim_market_maker'.")
    description: str = Field(default="", description="One line on what it is and where it comes from.")


class TaskArtifact(BaseModel):
    """Something produced during the run that the caller can link to."""

    kind: str = Field(description="csv | report | dashboard | metric | table | file")
    ref: str = Field(description="Identifier or path the caller can resolve.")
    title: Optional[str] = Field(default=None, description="Human-readable label.")


class TaskResultTool:
    """Lets a dispatched run declare how it ended."""

    permission_category: str = "tools"

    def __init__(self) -> None:
        self.submitted: Optional[dict] = None

    def available_tools(self) -> List[FunctionTool]:
        return [trans_to_function_tool(self.submit_task_result, strict_mode=False)]

    def submit_task_result(
        self,
        outcome: TaskOutcome,
        summary: str,
        artifacts: Optional[List[TaskArtifact]] = None,
        gap_reasons: Optional[List[str]] = None,
        plan_items: Optional[List[PlanItem]] = None,
        estimate: Optional[str] = None,
    ) -> FuncToolResult:
        """Report how this task ended. Call this exactly once, as your final action.

        Args:
            outcome: One of —
                ``answered``: you produced the answer or the artifact that was asked for.
                ``needs_development``: you cannot answer yet because something has to be
                    built first. Give BOTH ``gap_reasons`` and ``plan_items`` in the same
                    call — they are two halves of one judgement ("because A and B are
                    missing, build X and Y"), and the caller renders them together.
                ``blocked``: you cannot answer and cannot propose a build either — no
                    data source, no permission, out of scope. Give ``gap_reasons``.
                    Do not invent a plan to avoid this outcome; a human is handed the
                    request when you use it, which is the correct result.
            summary: A few sentences the caller reads instead of your full transcript.
                Include the grain and definitions you settled on — you are the only one
                who knows what you had to decide along the way.
            artifacts: Anything produced that the caller can link to.
            gap_reasons: Concretely what is missing. Required for ``needs_development``
                and ``blocked``.
            plan_items: What to build, for ``needs_development``.
            estimate: Rough effort for the plan, e.g. "1.5 person-days".
        """
        if not summary or not summary.strip():
            return FuncToolResult(success=0, error="summary must not be empty")

        if outcome in ("needs_development", "blocked") and not gap_reasons:
            return FuncToolResult(
                success=0,
                error=f"outcome '{outcome}' requires gap_reasons explaining what is missing",
            )

        if outcome == "needs_development" and not plan_items:
            return FuncToolResult(
                success=0,
                error="outcome 'needs_development' requires plan_items describing what to build",
            )

        self.submitted = {
            "outcome": outcome,
            "summary": summary.strip(),
            "artifacts": [a.model_dump() for a in artifacts or []],
            "gap_reasons": list(gap_reasons or []),
            "plan_items": [p.model_dump() for p in plan_items or []],
            "estimate": estimate,
        }
        logger.info(f"task result submitted: outcome={outcome} artifacts={len(self.submitted['artifacts'])}")

        return FuncToolResult(
            result={
                "acknowledged": True,
                "outcome": outcome,
                # The caller reads the tool call itself off the stream and stops the
                # run, so anything said after this is discarded. Say so, rather than
                # letting the model spend a turn on a closing paragraph nobody reads.
                "note": "Result recorded. Stop here — no further output is used.",
            }
        )
