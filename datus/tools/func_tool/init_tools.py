# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Coverage reporting for one ordinary /init chat run; no task scheduling."""

from datus.cli.init_context import InitContext
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool


class InitTools:
    permission_category = "init_tools"

    def __init__(self, run: InitContext):
        self.run = run

    def available_tools(self):
        return [
            trans_to_function_tool(self.plan_init_scope),
            trans_to_function_tool(self.report_init_progress),
            trans_to_function_tool(self.get_init_status),
        ]

    def plan_init_scope(
        self,
        tables: list[str],
        source_paths: list[str] | None = None,
        uncovered: str = "",
        exclude_tables: list[str] | None = None,
    ) -> FuncToolResult:
        """Register or adjust selected coverage, then continue in this chat.

        Use todo tools for work steps. This tool only records which tables
        are in scope and relevant source paths; it does not schedule anything.
        Names should match describe_table's qualified identifiers. Lists can
        span several calls. exclude_tables removes names from current coverage
        without deleting their recorded results or published knowledge.
        Incidental dependency lookups do not select more work. Empty tables
        are valid for file-only work. uncovered describes database areas outside
        this invocation. No plan approval or end-of-chat transition is required.
        """
        try:
            return FuncToolResult(
                result=self.run.select_scope(tables, source_paths or [], uncovered, exclude_tables or [])
            )
        except (OSError, ValueError) as exc:
            return FuncToolResult(success=0, error=str(exc))

    def report_init_progress(
        self,
        summary: str,
        completed_tables: list[str],
        scope_complete: bool = False,
        remaining: str = "",
        next_step: str = "",
    ) -> FuncToolResult:
        """Record actual outcomes and continue freely in the current chat.

        Report only tables whose knowledge you initialized or confirmed current;
        a successful schema lookup alone is not completion. Explicitly completed
        tables join the selected coverage even if not previously registered.
        Use [] for file-only work. Give a short summary, remaining work and next
        step so compaction preserves the task. scope_complete=true explicitly
        reports completion of this invocation's selected scope, never the whole
        database. For file-only work, explain what was completed in summary.
        This report neither ends the run nor certifies business content; errors
        or interruption still make the outcome incomplete. Resolve tool-answerable
        structural questions before handing them to the user. Keep genuine business
        ambiguities in knowledge/pending-questions.md with sources, and distinguish
        checked tables from unverified ones in the published knowledge.
        """
        try:
            return FuncToolResult(
                result=self.run.report_progress(summary, completed_tables, scope_complete, remaining, next_step)
            )
        except ValueError as exc:
            return FuncToolResult(success=0, error=str(exc))

    def get_init_status(self, offset: int = 0, limit: int = 10) -> FuncToolResult:
        """Read a bounded page of coverage, sources and user clarifications.

        offset/limit page this invocation's records, not the database catalog.
        Observed dependencies do not enlarge selected coverage. Metadata checks
        and model-reported knowledge completion are separate. Work steps belong
        to todo_list/read; this is not another execution plan.
        """
        try:
            return FuncToolResult(result=self.run.progress(offset, limit))
        except ValueError as exc:
            return FuncToolResult(success=0, error=str(exc))
