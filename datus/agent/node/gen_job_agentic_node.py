# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenJobAgenticNode implementation for ETL and cross-database migration jobs.

This node builds target tables from source tables (single-database ETL) and
migrates data across database engines (cross-database migration). Most of the
plumbing lives in the shared :class:`DeliverableAgenticNode` base; this
subclass adds DML (``execute_write``), cross-DB transfer
(``transfer_query_result``), and the three ``MigrationTargetMixin`` wrappers
(``get_migration_capabilities`` / ``suggest_table_layout`` / ``validate_ddl``)
to the DDL-only default set.

Post-transfer reconciliation is driven by the ``transfer-reconciliation``
validator skill via :class:`ValidationHook`, not by this node directly.
"""

from typing import Any, ClassVar, Dict, List, Optional

from datus.agent.node.deliverable_node import DeliverableAgenticNode
from datus.configuration.node_type import NodeType
from datus.tools.func_tool import DBFuncTool
from datus.tools.func_tool.base import trans_to_function_tool
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GenJobAgenticNode(DeliverableAgenticNode):
    """ETL / cross-DB migration subagent.

    In addition to the base DDL + read tools it registers:

    - ``execute_write`` — INSERT / UPDATE / DELETE
    - ``transfer_query_result`` — cross-DB data transfer
    - The three ``MigrationTargetMixin`` wrappers for dialect-aware DDL advice
    """

    NODE_NAME: ClassVar[str] = "gen_job"
    NODE_TYPE: ClassVar[str] = NodeType.TYPE_GEN_JOB
    DEFAULT_SKILLS: ClassVar[Optional[str]] = "gen-table, data-migration"
    PROMPT_TEMPLATE: ClassVar[str] = "gen_job_system"
    ACTION_TYPE: ClassVar[str] = "gen_job_response"
    DEFAULT_MAX_TURNS: ClassVar[int] = 40

    def _setup_domain_tools(self) -> None:
        """Register read tools + DDL + DML + transfer + migration mixin wrappers."""
        try:
            self.db_func_tool = DBFuncTool(
                agent_config=self.agent_config,
                sub_agent_name=self.NODE_NAME,
            )
            self.tools.extend(self.db_func_tool.available_tools())
            if hasattr(self.db_func_tool, "execute_ddl"):
                self.tools.append(trans_to_function_tool(self.db_func_tool.execute_ddl))
            if hasattr(self.db_func_tool, "execute_write"):
                self.tools.append(trans_to_function_tool(self.db_func_tool.execute_write))
            if hasattr(self.db_func_tool, "transfer_query_result"):
                self.tools.append(trans_to_function_tool(self.db_func_tool.transfer_query_result))
            if hasattr(self.db_func_tool, "get_migration_capabilities"):
                self.tools.append(trans_to_function_tool(self.db_func_tool.get_migration_capabilities))
            if hasattr(self.db_func_tool, "suggest_table_layout"):
                self.tools.append(trans_to_function_tool(self.db_func_tool.suggest_table_layout))
            if hasattr(self.db_func_tool, "validate_ddl"):
                self.tools.append(trans_to_function_tool(self.db_func_tool.validate_ddl))
            logger.debug(
                "Added database tools + execute_ddl + execute_write + transfer_query_result "
                "+ migration Mixin wrappers from DBFuncTool"
            )
        except Exception as e:
            logger.exception("Failed to setup database tools")
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"Failed to setup database tools for {self.NODE_NAME}: {e}"},
            ) from e

    def _tool_category_map(self) -> Dict[str, List[Any]]:
        """Register db / filesystem tools for permission profile rule matching.

        DB writes (``execute_ddl`` / ``execute_write`` / ``transfer_query_result``)
        should ASK in ``normal`` and ``auto`` profiles — without this
        mapping they'd fall into the ``tools`` catch-all and bypass
        ``db_tools.*`` rules entirely.
        """
        mapping = super()._tool_category_map()
        db_bucket: List[Any] = []
        if getattr(self, "db_func_tool", None):
            db_bucket.extend(self.db_func_tool.available_tools())
            for attr in (
                "execute_ddl",
                "execute_write",
                "transfer_query_result",
                "get_migration_capabilities",
                "suggest_table_layout",
                "validate_ddl",
            ):
                if hasattr(self.db_func_tool, attr):
                    db_bucket.append(trans_to_function_tool(getattr(self.db_func_tool, attr)))
        if db_bucket:
            mapping["db_tools"] = db_bucket
        if getattr(self, "filesystem_func_tool", None):
            mapping["filesystem_tools"] = list(self.filesystem_func_tool.available_tools())
        if self.ask_user_tool:
            mapping.setdefault("tools", []).extend(self.ask_user_tool.available_tools())
        return mapping
