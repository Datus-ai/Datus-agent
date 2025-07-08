from typing import Dict

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.node_models import SQLContext
from datus.schemas.reason_sql_node_models import ReasoningInput, ReasoningResult
from datus.tools.llms_tools import LLMTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ReasonSQLNode(Node):
    def execute(self):
        self.result = self._reason_sql()

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
        result = self.result
        try:
            # Choose the valuable sql_context from the result
            # Append successful SQL contexts from reasoning result to workflow context
            if result.success:
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
        )

        # Create SQL execution node
        execute_sql_node = Node.new_instance(
            node_id=f"reflect_{workflow.reflection_round}_regenerate_execute_sql",
            description="Execute the corrected SQL query",
            node_type="execute_sql",
            input_data=None,
            agent_config=self.agent_config,
        )

        # Add new nodes to workflow
        workflow.add_node(execute_sql_node, current_position + 1)
        workflow.add_node(generate_sql_node, current_position + 1)

    def _reason_sql(self) -> ReasoningResult:
        """Reasoning and Exploring the database to refine SQL query.
        Returns:
            GenerateSQLResult containing the generated SQL query
        """
        try:
            tool = LLMTool(self.model)
            # TODO: pass the mcp_server to tools, don't repeat init the mcp server
            return tool.reasoning_sql(
                self.input, self.agent_config.current_db_config(db_name=self.input.sql_task.database_name)
            )
        except Exception as e:
            logger.error(f"SQL reasoning execution error: {str(e)}")
            return ReasoningResult(success=False, error=str(e), sql_query="")
