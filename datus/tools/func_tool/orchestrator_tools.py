# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Orchestrator runtime proxy tools.

These tools are intentionally placeholders when executed inside Datus-agent.
An external orchestrator enables them in print mode and proxies the calls back
to its tracker adapter, so Datus-agent can request issue updates without holding tracker
credentials directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents import FunctionTool

from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool


class OrchestratorIssueTools:
    """Issue lifecycle tools owned by an external orchestrator runtime."""

    def available_tools(self) -> List[FunctionTool]:
        return [
            trans_to_function_tool(self.create_issue_comment, strict_mode=False),
            trans_to_function_tool(self.update_issue_status, strict_mode=False),
            trans_to_function_tool(self.request_human_input, strict_mode=False),
            trans_to_function_tool(self.mark_blocked, strict_mode=False),
            trans_to_function_tool(self.finish_mission, strict_mode=False),
        ]

    def create_issue_comment(self, body: str, issue_id: Optional[str] = None) -> FuncToolResult:
        """Create an issue comment through the orchestrator.

        Args:
            body: Markdown comment body to append to the current issue.
            issue_id: Optional tracker issue id. Omit this for the current issue.
        """
        return self._requires_proxy("create_issue_comment", {"issue_id": issue_id, "body": body})

    def update_issue_status(self, status: str, issue_id: Optional[str] = None) -> FuncToolResult:
        """Move an issue to a tracker status through the orchestrator.

        Args:
            status: Target issue status name, for example "In Progress" or "In Review".
            issue_id: Optional tracker issue id. Omit this for the current issue.
        """
        return self._requires_proxy("update_issue_status", {"issue_id": issue_id, "status": status})

    def finish_mission(
        self,
        outcome: str,
        summary: str,
        issue_id: Optional[str] = None,
        next_status: Optional[str] = None,
        artifacts: Optional[List[Dict[str, Any]]] = None,
        validation: Optional[List[Dict[str, Any]]] = None,
    ) -> FuncToolResult:
        """Report the final mission outcome to the orchestrator.

        Args:
            outcome: completed, needs_review, blocked, or failed.
            summary: Short operator-facing summary.
            issue_id: Optional tracker issue id. Omit this for the current issue.
            next_status: Optional tracker status the orchestrator should move the issue to.
            artifacts: Optional generated artifacts or URLs.
            validation: Optional validation checks and results.
        """
        return self._requires_proxy(
            "finish_mission",
            {
                "issue_id": issue_id,
                "outcome": outcome,
                "summary": summary,
                "next_status": next_status,
                "artifacts": artifacts or [],
                "validation": validation or [],
            },
        )

    def request_human_input(
        self,
        question: str,
        issue_id: Optional[str] = None,
        next_status: Optional[str] = None,
    ) -> FuncToolResult:
        """Request human input on the current issue.

        Args:
            question: Specific question or decision needed from the operator.
            issue_id: Optional tracker issue id. Omit this for the current issue.
            next_status: Optional tracker status the orchestrator should move the issue to.
        """
        return self._requires_proxy(
            "request_human_input",
            {
                "issue_id": issue_id,
                "question": question,
                "next_status": next_status,
            },
        )

    def mark_blocked(
        self,
        reason: str,
        issue_id: Optional[str] = None,
        next_status: Optional[str] = None,
        artifacts: Optional[List[Dict[str, Any]]] = None,
        validation: Optional[List[Dict[str, Any]]] = None,
    ) -> FuncToolResult:
        """Mark the current issue as blocked.

        Args:
            reason: Concrete blocker that prevents completion.
            issue_id: Optional tracker issue id. Omit this for the current issue.
            next_status: Optional tracker status the orchestrator should move the issue to.
            artifacts: Optional generated artifacts or URLs.
            validation: Optional validation checks and results.
        """
        return self._requires_proxy(
            "mark_blocked",
            {
                "issue_id": issue_id,
                "reason": reason,
                "next_status": next_status,
                "artifacts": artifacts or [],
                "validation": validation or [],
            },
        )

    def _requires_proxy(self, tool_name: str, payload: Dict[str, Any]) -> FuncToolResult:
        return FuncToolResult(
            success=0,
            error=(
                f"{tool_name} must be proxied by the orchestrator runtime. "
                "Run Datus-agent print mode with --orchestrator-tools and proxy orchestrator_tools.*."
            ),
            result=payload,
        )
