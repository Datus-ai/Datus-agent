# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenTableAgenticNode implementation for wide table generation.

This node creates database tables via CTAS (from JOIN SQL) or CREATE TABLE
(from natural-language descriptions). Most of the plumbing lives in the
shared :class:`TableDeliverableAgenticNode` base; this subclass only declares
the tool set (``execute_ddl`` on top of the read-only DB tools) and the
permission profile category map.
"""

from typing import Any, ClassVar, Dict, List, Optional

from datus.agent.node.table_deliverable_node import TableDeliverableAgenticNode
from datus.configuration.node_type import NodeType
from datus.tools.func_tool import DBFuncTool
from datus.tools.func_tool.base import trans_to_function_tool
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GenTableAgenticNode(TableDeliverableAgenticNode):
    """Wide table generation subagent.

    Registers the standard read-only DB tools plus :func:`execute_ddl` so the
    LLM can CREATE TABLE / CTAS. Post-write validation is driven by
    :class:`ValidationHook` in the base class.
    """

    NODE_NAME: ClassVar[str] = "gen_table"
    NODE_TYPE: ClassVar[str] = NodeType.TYPE_GEN_TABLE
    DEFAULT_SKILLS: ClassVar[Optional[str]] = "gen-table, table-validation"
    PROMPT_TEMPLATE: ClassVar[str] = "gen_table_system"
    ACTION_TYPE: ClassVar[str] = "gen_table_response"
    DEFAULT_MAX_TURNS: ClassVar[int] = 20

    def _setup_db_tools(self) -> None:
        """DDL-only: read tools + ``execute_ddl``. No DML / no transfer."""
        try:
            self.db_func_tool = DBFuncTool(
                agent_config=self.agent_config,
                sub_agent_name=self._configured_node_name,
            )
            self.tools.extend(self.db_func_tool.available_tools())
            if hasattr(self.db_func_tool, "execute_ddl"):
                self.tools.append(trans_to_function_tool(self.db_func_tool.execute_ddl))
            logger.debug("Added database tools + execute_ddl from DBFuncTool")
        except Exception as e:
            logger.exception("Failed to setup database tools")
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"Failed to setup database tools for {self.NODE_NAME}: {e}"},
            ) from e

    def _tool_category_map(self) -> Dict[str, List[Any]]:
        """Register db / filesystem tools so write/destructive rules fire."""
        mapping = super()._tool_category_map()
        db_bucket: List[Any] = []
        if getattr(self, "db_func_tool", None):
            db_bucket.extend(self.db_func_tool.available_tools())
            if hasattr(self.db_func_tool, "execute_ddl"):
                db_bucket.append(trans_to_function_tool(self.db_func_tool.execute_ddl))
        if db_bucket:
            mapping["db_tools"] = db_bucket
        if getattr(self, "filesystem_func_tool", None):
            mapping["filesystem_tools"] = list(self.filesystem_func_tool.available_tools())
        if self.ask_user_tool:
            mapping.setdefault("tools", []).extend(self.ask_user_tool.available_tools())
        return mapping
