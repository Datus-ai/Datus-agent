# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenTableAgenticNode implementation for wide table generation.

This node creates database tables via CTAS (from JOIN SQL) or CREATE TABLE
(from natural-language descriptions). Most of the plumbing lives in the
shared :class:`DeliverableAgenticNode` base; this subclass only declares the
tool set (the unified ``execute_sql`` tool via the standard DB tools) and the
permission profile category map.
"""

import json
from typing import ClassVar, Optional

from datus.agent.node.deliverable_node import DeliverableAgenticNode
from datus.configuration.node_type import NodeType
from datus.tools.func_tool import DBFuncTool
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GenTableAgenticNode(DeliverableAgenticNode):
    """Wide table generation subagent.

    Registers the standard DB tools, including the unified ``execute_sql`` tool
    so the LLM can CREATE TABLE / CTAS. Post-write validation is driven by
    :class:`ValidationHook` in the base class.
    """

    NODE_NAME: ClassVar[str] = "gen_table"
    NODE_TYPE: ClassVar[str] = NodeType.TYPE_GEN_TABLE
    DEFAULT_SKILLS: ClassVar[Optional[str]] = "gen-table"
    PROMPT_TEMPLATE: ClassVar[str] = "gen_table_system"
    ACTION_TYPE: ClassVar[str] = "gen_table_response"
    DEFAULT_MAX_TURNS: ClassVar[int] = 100

    def continuation_target_error(self) -> Optional[str]:
        """Reject a resume that would silently rebind writes to another target."""
        from datus.schemas.database_context import (
            EXECUTION_CONTEXT_META_KEY,
            DatabaseExecutionContext,
            execution_context_for_node,
        )

        saved = self.session_manager.load_session_meta(self.session_id, EXECUTION_CONTEXT_META_KEY)
        try:
            previous = DatabaseExecutionContext.from_mapping(json.loads(saved)) if saved else None
        except (TypeError, ValueError, AttributeError):
            previous = None
        if previous is None or not previous.datasource:
            return "The original table's execution target is unavailable. Establish its exact datasource and database before continuing in a new specialist session."
        current = execution_context_for_node(self.agent_config, self.input)
        if previous != current:
            return (
                f"The table session is bound to {previous.identity}, but this turn targets {current.identity}. "
                "No continuation was executed. Restore the original execution target, or obtain an explicit "
                "decision about the different target before starting a new specialist session."
            )
        return None

    def _setup_domain_tools(self) -> None:
        """Standard DB tools, including the unified ``execute_sql`` tool. No transfer.

        Statement-type permission gating (read auto-allow, write/DDL ask) is
        handled by ``PermissionHooks._handle_sql_permission``.
        """
        try:
            self.db_func_tool = DBFuncTool(
                # Same anchor the filesystem tools use, so a path the model got
                # from glob resolves identically in load_file_as_table.
                filesystem_root=self._resolve_workspace_root(),
                agent_config=self.agent_config,
                sub_agent_name=self._configured_node_name,
            )
            self.tools.extend(self.db_func_tool.available_tools())
            logger.debug("Added database tools (execute_sql) from DBFuncTool")
        except Exception as e:
            logger.exception("Failed to setup database tools")
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"Failed to setup database tools for {self.NODE_NAME}: {e}"},
            ) from e
