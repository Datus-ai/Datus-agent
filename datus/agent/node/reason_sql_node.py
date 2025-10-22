# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from datetime import datetime
from typing import AsyncGenerator, Dict, Optional

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import SQLContext
from datus.schemas.reason_sql_node_models import ReasoningInput, ReasoningResult
from datus.tools.llms_tools.reasoning_sql import reasoning_sql_with_mcp, reasoning_sql_with_mcp_stream
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ReasonSQLNode(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_history_manager = None

    def execute(self):
        result = self._reason_sql()
        logger.debug(
            f"ReasonSQLNode execute got result type: {type(result)}, success: {getattr(result, 'success', 'N/A')}"
        )
        self.result = result

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute SQL reasoning with streaming support."""
        async for action in self._reason_sql_stream(action_history_manager):
            yield action

    def setup_input(self, workflow: Workflow) -> Dict:
        next_input = ReasoningInput(
            database_type=workflow.task.database_type,
            sql_task=workflow.task,
            table_schemas=workflow.context.table_schemas,
            data_details=workflow.context.table_values,
            metrics=workflow.context.metrics,
            external_knowledge=workflow.task.external_knowledge,
            contexts=workflow.context.sql_contexts[-1:] if workflow.context.sql_contexts else [],
        )
        self.input = next_input
        logger.info(f"Setup reasoning input: {self.input}")
        return {"success": True, "message": "Reasoning input setup complete", "suggestions": [next_input]}

    def update_context(self, workflow: Workflow) -> Dict:
        """Update reasoning results to workflow context."""
        logger.debug(
            f"ReasonSQLNode.update_context called: result_type={type(self.result)}, "
            f"result_success={getattr(self.result, 'success', 'N/A')}"
        )
        try:
            # Check if we have streaming results from action history manager
            if self.action_history_manager and hasattr(self.action_history_manager, "sql_contexts"):
                # Use sql_contexts from streaming execution
                sql_contexts = self.action_history_manager.sql_contexts
                logger.info(f"Using streaming results: {len(sql_contexts)} SQL contexts found")

                # Add successful SQL contexts to workflow context
                for sql_ctx in sql_contexts:
                    if sql_ctx.sql_error == "":  # only add the successful sql context
                        workflow.context.sql_contexts.append(sql_ctx)
                    else:
                        logger.warning(f"Failed context, skip it: {sql_ctx.sql_query}, {sql_ctx.sql_error}")

                return {"success": True, "message": "Updated reasoning context from streaming results"}

            # Fall back to non-streaming result
            result = self.result
            if result and result.success:
                # Add the reasoning process sqls to the sql context
                for sql_ctx in result.sql_contexts:
                    if sql_ctx.sql_error == "":  # only add the successful sql context
                        workflow.context.sql_contexts.append(sql_ctx)
                    else:
                        logger.warning(f"Failed context, skip it: {sql_ctx.sql_query}, {sql_ctx.sql_error}")

                # Add the reasoning result to the sql context
                new_record = SQLContext(
                    sql_query=result.sql_query, sql_return=result.sql_return
                )  # add explanation later
                workflow.context.sql_contexts.append(new_record)
                return {"success": True, "message": "Updated reasoning context"}
            else:
                # reasoning failed, use a final try with generate_sql
                self._regenerate_sql_with_all_context(workflow)
                return {
                    "success": True,
                    "message": "Reasoning failed, regenerated SQL with all context",
                }
        except Exception as e:
            logger.error(f"Failed to update reasoning context: {str(e)}")
            return {"success": False, "message": f"Reasoning context update failed: {str(e)}"}

    def _regenerate_sql_with_all_context(self, workflow: Workflow) -> None:
        """
        Regenerate the SQL with all context
        """
        current_position = workflow.current_node_index

        # Create SQL generation node
        generate_sql_node = Node.new_instance(
            node_id=f"reflect_{workflow.reflection_round}_regenerate_sql",
            description="Generate corrected SQL based on schema analysis",
            node_type="generate_sql",
            input_data=None,
            agent_config=self.agent_config,
            tools=workflow.tools,
        )

        # Create SQL execution node
        execute_sql_node = Node.new_instance(
            node_id=f"reflect_{workflow.reflection_round}_regenerate_execute_sql",
            description="Execute the corrected SQL query",
            node_type="execute_sql",
            input_data=None,
            agent_config=self.agent_config,
            tools=workflow.tools,
        )

        # Add new nodes to workflow
        workflow.add_node(execute_sql_node, current_position + 1)
        workflow.add_node(generate_sql_node, current_position + 1)

    async def _reason_sql_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Reasoning and Exploring the database with streaming support and action history tracking."""
        if not self.model:
            logger.error("Model not available for SQL reasoning")
            return

        # Store the action history manager for later use in update_context
        self.action_history_manager = action_history_manager

        try:
            # Setup reasoning context action
            setup_action = ActionHistory(
                action_id="setup_reasoning",
                role=ActionRole.WORKFLOW,
                messages="Setting up reasoning context with database schemas and data",
                action_type="schema_linking",
                input={
                    "database_type": self.input.database_type,
                    "task": self.input.sql_task.task,
                    "table_schemas_count": len(self.input.table_schemas),
                    "data_details_count": len(self.input.data_details),
                    "metrics_count": len(self.input.metrics),
                    "contexts_count": len(self.input.contexts),
                    "external_knowledge_available": bool(self.input.external_knowledge),
                },
                status=ActionStatus.SUCCESS,
            )
            yield setup_action

            # Update setup action with success
            setup_action.output = {
                "success": True,
                "reasoning_input_prepared": True,
                "database_name": self.input.sql_task.database_name,
                "max_turns": self.input.max_turns,
            }
            setup_action.end_time = datetime.now()

            # Stream the reasoning process

            async for action in reasoning_sql_with_mcp_stream(
                model=self.model,
                input_data=self.input,
                tools=self.tools,
                tool_config={"max_turns": self.input.max_turns},
                action_history_manager=action_history_manager,
            ):
                yield action

        except Exception as e:
            logger.error(f"SQL reasoning streaming error: {str(e)}")
            raise

    def _reason_sql(self) -> ReasoningResult:
        """Reasoning and Exploring the database to refine SQL query.
        Returns:
            ReasoningResult containing the generated SQL query
        """
        try:
            result = reasoning_sql_with_mcp(
                self.model, self.input, self.tools, tool_config={"max_turns": self.input.max_turns}
            )
            logger.debug(
                f"_reason_sql got result from tool: type={type(result)}, success={getattr(result, 'success', 'N/A')}"
            )
            return result
        except Exception as e:
            logger.error(f"SQL reasoning execution error: {str(e)}")
            fallback_result = ReasoningResult(success=False, error=str(e), sql_query="")
            logger.debug(
                f"_reason_sql returning fallback result: type={type(fallback_result)}, "
                f"success={fallback_result.success}"
            )
            return fallback_result
