# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
ExplorerTool - Agent-based exploration tool.

Spawns a sub-agent that can use DB/ContextSearch/Filesystem(read-only) tools
to explore and gather information, then returns a concise summary to the
main agent.  The main agent may invoke ``explorer()`` multiple times in
parallel (parallel tool calls) but should avoid calling it iteratively.

Session branching
-----------------
When the tool is invoked the parent session is *branched*: a new session_id
is created in the same SQLite DB and the existing messages are copied over so
the explorer can see conversation history.  The branch is destroyed after the
tool call completes.
"""

from __future__ import annotations

import sqlite3
import uuid
from typing import TYPE_CHECKING, List, Optional

from agents import Tool

from datus.models.base import LLMBaseModel
from datus.prompts.prompt_manager import prompt_manager
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger
from datus.utils.traceable_utils import optional_traceable

if TYPE_CHECKING:
    from agents import SQLiteSession

    from datus.configuration.agent_config import AgentConfig
    from datus.schemas.action_bus import ActionBus
    from datus.tools.func_tool.context_search import ContextSearchTools
    from datus.tools.func_tool.database import DBFuncTool
    from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

logger = get_logger(__name__)


class ExplorerTool:
    """Internal agent-based exploration tool for the main SQL agent.

    The explorer is exposed as a single ``explorer(query_text)`` function tool.
    Internally it creates a temporary *branch* of the parent conversation
    session, builds a sub-agent equipped with DB / context-search / read-only
    filesystem tools, runs it, and returns the final summary.
    """

    PROMPT_TEMPLATE = "explorer_system"

    def __init__(
        self,
        agent_config: "AgentConfig",
        max_turns: int = 50,
    ):
        self.agent_config = agent_config
        self.max_turns = max_turns

        # Parent session – set via set_session() before execution
        self._session: Optional["SQLiteSession"] = None

        # Sub-tool instances – set via set_sub_tools()
        self._db_func_tool: Optional["DBFuncTool"] = None
        self._context_search_tools: Optional["ContextSearchTools"] = None
        self._filesystem_func_tool: Optional["FilesystemFuncTool"] = None

        # Lazy-initialised model
        self._model: Optional[LLMBaseModel] = None

        # ActionBus for forwarding sub-actions to the main stream
        self._action_bus: Optional["ActionBus"] = None

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_session(self, session: "SQLiteSession") -> None:
        """Inject the parent session so branches can be created."""
        self._session = session

    def set_sub_tools(
        self,
        db_func_tool: Optional["DBFuncTool"] = None,
        context_search_tools: Optional["ContextSearchTools"] = None,
        filesystem_root: Optional[str] = None,
    ) -> None:
        """Share sub-tool instances from the parent node.

        ``db_func_tool`` and ``context_search_tools`` are reused directly.
        A *new* read-only ``FilesystemFuncTool`` is created for ``filesystem_root``.
        """
        self._db_func_tool = db_func_tool
        self._context_search_tools = context_search_tools
        if filesystem_root is not None:
            from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

            self._filesystem_func_tool = FilesystemFuncTool(root_path=filesystem_root)

    def set_action_bus(self, bus: "ActionBus") -> None:
        """Inject the :class:`ActionBus` for forwarding sub-actions."""
        self._action_bus = bus

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    def available_tools(self) -> List[Tool]:
        """Return the single ``explorer`` function tool."""
        return [trans_to_function_tool(self.explore)]

    @optional_traceable(name="explore", run_type="tool")
    async def explore(self, query_text: str) -> FuncToolResult:
        """Explore databases, context, and files to gather information.

        Spawns a sub-agent that investigates the available data sources and
        returns a concise summary of its findings.

        Args:
            query_text: Natural-language description of what to explore.
        """
        branch: Optional["SQLiteSession"] = None
        try:
            # 1. Create branch session
            branch = self._create_branch_session()

            # 2. Run exploration via LLMBaseModel.generate_with_tools_stream
            tools = self._build_explorer_tools()
            instruction = self._get_explorer_instruction()
            model = self._get_or_create_model()
            action_history_manager = ActionHistoryManager()

            summary = ""
            async for action in model.generate_with_tools_stream(
                prompt=query_text,
                tools=tools,
                instruction=instruction,
                output_type=str,
                max_turns=self.max_turns,
                session=branch,
                action_history_manager=action_history_manager,
            ):
                # Forward sub-action to the ActionBus (real-time)
                if self._action_bus is not None:
                    self._action_bus.put(action)

                # Extract content from the final successful action
                if action.status == ActionStatus.SUCCESS and action.output:
                    if isinstance(action.output, dict):
                        summary = (
                            action.output.get("content", "")
                            or action.output.get("response", "")
                            or action.output.get("raw_output", "")
                            or summary
                        )

            logger.info(f"Explorer completed: {len(summary)} chars summary")
            return FuncToolResult(success=1, result=summary)

        except Exception as e:
            logger.error(f"Explorer failed: {e}")
            return FuncToolResult(success=0, error=str(e))

        finally:
            if branch is not None:
                self._destroy_branch_session(branch)

    # ------------------------------------------------------------------
    # Session branching
    # ------------------------------------------------------------------

    def _create_branch_session(self) -> "SQLiteSession":
        """Create a branch of the parent session by copying messages."""
        from datus.models.session_manager import ExtendedSQLiteSession

        if self._session is None:
            # No parent session – return a fresh in-memory session
            branch_id = f"explorer_branch_{uuid.uuid4().hex[:8]}"
            logger.debug("No parent session, creating empty branch: %s", branch_id)
            return ExtendedSQLiteSession(branch_id)

        parent_db_path = getattr(self._session, "db_path", ":memory:")
        parent_session_id = getattr(self._session, "session_id", None)
        branch_id = f"explorer_branch_{uuid.uuid4().hex[:8]}"

        # Create branch session in the same DB file
        branch = ExtendedSQLiteSession(branch_id, db_path=parent_db_path)

        # Copy messages from parent to branch
        if parent_session_id:
            try:
                conn = branch._get_connection()
                with conn:
                    # Ensure session record exists
                    conn.execute(
                        f"INSERT OR IGNORE INTO {branch.sessions_table} (session_id) VALUES (?)",
                        (branch_id,),
                    )
                    # Copy messages
                    conn.execute(
                        f"""
                        INSERT INTO {branch.messages_table} (session_id, message_data, created_at)
                        SELECT ?, message_data, created_at
                        FROM {branch.messages_table}
                        WHERE session_id = ?
                        ORDER BY id ASC
                        """,
                        (branch_id, parent_session_id),
                    )
                logger.debug(
                    "Created branch session %s from parent %s (db=%s)",
                    branch_id,
                    parent_session_id,
                    parent_db_path,
                )
            except sqlite3.Error as e:
                logger.warning("Failed to copy parent messages to branch: %s", e)

        return branch

    def _destroy_branch_session(self, branch: "SQLiteSession") -> None:
        """Remove branch session data from the DB."""
        branch_id = getattr(branch, "session_id", None)
        if not branch_id:
            return

        try:
            conn = branch._get_connection()
            with conn:
                conn.execute(
                    f"DELETE FROM {branch.messages_table} WHERE session_id = ?",
                    (branch_id,),
                )
                conn.execute(
                    f"DELETE FROM {branch.sessions_table} WHERE session_id = ?",
                    (branch_id,),
                )
            logger.debug("Destroyed branch session: %s", branch_id)
        except sqlite3.Error as e:
            logger.warning("Failed to destroy branch session %s: %s", branch_id, e)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_model(self) -> LLMBaseModel:
        if self._model is None:
            self._model = LLMBaseModel.create_model(agent_config=self.agent_config)
        return self._model

    def _get_explorer_instruction(self) -> str:
        """Render the explorer system prompt template."""
        try:
            ctx = self._context_search_tools
            return prompt_manager.render_template(
                template_name=self.PROMPT_TEMPLATE,
                version=None,  # use latest
                has_db_tools=self._db_func_tool is not None,
                has_context_search_tools=ctx is not None,
                has_metrics=getattr(ctx, "has_metrics", False),
                has_reference_sql=getattr(ctx, "has_reference_sql", False),
                has_knowledge=getattr(ctx, "has_knowledge", False),
                has_semantic_objects=getattr(ctx, "has_semantic_objects", False),
                has_filesystem_tools=self._filesystem_func_tool is not None,
            )
        except FileNotFoundError:
            logger.warning("Explorer system prompt template not found, using fallback")
            return (
                "You are an exploration agent. Investigate databases, context, "
                "and files to gather information relevant to the query. "
                "Return a concise summary of your findings."
            )

    def _build_explorer_tools(self) -> List[Tool]:
        """Collect tools from the shared sub-tool instances."""
        tools: List[Tool] = []

        if self._db_func_tool is not None:
            tools.extend(self._db_func_tool.available_tools())

        if self._context_search_tools is not None:
            tools.extend(self._context_search_tools.available_tools())

        if self._filesystem_func_tool is not None:
            tools.extend(self._filesystem_func_tool.available_readonly_tools())

        return tools
