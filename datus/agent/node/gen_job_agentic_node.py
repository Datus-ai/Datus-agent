# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
GenJobAgenticNode implementation for ETL and cross-database migration jobs.

This node builds target tables from source tables (single-database ETL) and
migrates data across database engines (cross-database migration). Most of the
plumbing lives in the shared :class:`DeliverableAgenticNode` base. Read, DML,
and DDL all flow through the unified ``execute_sql`` tool; this subclass adds
cross-DB transfer (``transfer_query_result``) and the three
``MigrationTargetMixin`` wrappers (``get_migration_capabilities`` /
``suggest_table_layout`` / ``validate_ddl``).

Post-transfer reconciliation is driven by the ``transfer-reconciliation``
validator skill via :class:`ValidationHook`, not by this node directly.
"""

from typing import ClassVar, Optional

from datus.agent.node.deliverable_node import DeliverableAgenticNode
from datus.configuration.node_type import NodeType
from datus.tools.func_tool import DBFuncTool
from datus.tools.func_tool.base import trans_to_function_tool
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GenJobAgenticNode(DeliverableAgenticNode):
    """ETL / cross-DB migration subagent.

    In addition to the base ``execute_sql`` tool (read + DML + DDL) it registers:

    - ``transfer_query_result`` — cross-DB data transfer
    - The three ``MigrationTargetMixin`` wrappers for dialect-aware DDL advice
    """

    NODE_NAME: ClassVar[str] = "gen_job"
    NODE_TYPE: ClassVar[str] = NodeType.TYPE_GEN_JOB
    DEFAULT_SKILLS: ClassVar[Optional[str]] = "gen-table, data-migration"
    PROMPT_TEMPLATE: ClassVar[str] = "gen_job_system"
    ACTION_TYPE: ClassVar[str] = "gen_job_response"
    DEFAULT_MAX_TURNS: ClassVar[int] = 50

    def _setup_domain_tools(self) -> None:
        """Register the unified ``execute_sql`` tool (via ``available_tools``) plus
        cross-DB transfer and migration mixin wrappers.

        Read/DML/DDL all flow through ``execute_sql``; per-statement-type
        permission gating lives in ``PermissionHooks._handle_sql_permission``.
        """
        try:
            self.db_func_tool = DBFuncTool(
                agent_config=self.agent_config,
                sub_agent_name=self.NODE_NAME,
            )
            self.tools.extend(self.db_func_tool.available_tools())
            if hasattr(self.db_func_tool, "transfer_query_result"):
                self.tools.append(trans_to_function_tool(self.db_func_tool.transfer_query_result))
            if hasattr(self.db_func_tool, "get_migration_capabilities"):
                self.tools.append(trans_to_function_tool(self.db_func_tool.get_migration_capabilities))
            if hasattr(self.db_func_tool, "suggest_table_layout"):
                self.tools.append(trans_to_function_tool(self.db_func_tool.suggest_table_layout))
            if hasattr(self.db_func_tool, "validate_ddl"):
                self.tools.append(trans_to_function_tool(self.db_func_tool.validate_ddl))
            logger.debug(
                "Added database tools (execute_sql) + transfer_query_result + migration Mixin wrappers from DBFuncTool"
            )
        except Exception as e:
            logger.exception("Failed to setup database tools")
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={"config_error": f"Failed to setup database tools for {self.NODE_NAME}: {e}"},
            ) from e
