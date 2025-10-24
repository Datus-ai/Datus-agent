# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
import json
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from agents import Tool

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.models.base import LLMBaseModel
from datus.prompts.gen_sql import get_sql_prompt
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import GenerateSQLInput, GenerateSQLResult, SQLContext, SqlTask, TableSchema, TableValue
from datus.storage.schema_metadata import SchemaWithValueRAG
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from datus.utils.time_utils import get_default_current_date
from datus.utils.traceable_utils import optional_traceable

logger = get_logger(__name__)


class GenerateSQLNode(Node):
    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: GenerateSQLInput = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[List[Tool]] = None,
    ):
        super().__init__(node_id, description, node_type, input_data, agent_config, tools)
        self._metadata_rag: SchemaWithValueRAG | None = None

    @property
    def metadata_rag(self) -> SchemaWithValueRAG:
        if not self._metadata_rag:
            self._metadata_rag = SchemaWithValueRAG(self.agent_config)
        return self._metadata_rag

    def execute(self):
        self.result = self._execute_generate_sql()

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute SQL generation with streaming support."""
        async for action in self._generate_sql_stream(action_history_manager):
            yield action

    def setup_input(self, workflow: Workflow) -> Dict:
        if workflow.context.document_result:
            database_docs = "\n Reference documents:\n"
            for _, docs in workflow.context.document_result.docs.items():
                database_docs += "\n".join(docs) + "\n"
        else:
            database_docs = ""
        # irrelevant to current node
        next_input = GenerateSQLInput(
            database_type=workflow.task.database_type,
            sql_task=workflow.task,
            table_schemas=workflow.context.table_schemas,
            data_details=workflow.context.table_values,
            metrics=workflow.context.metrics,
            contexts=workflow.context.sql_contexts,
            external_knowledge=workflow.task.external_knowledge,
            database_docs=database_docs,
        )
        self.input = next_input
        return {"success": True, "message": "Schema appears valid", "suggestions": [next_input]}

    def update_context(self, workflow: Workflow) -> Dict:
        """Update SQL generation results to workflow context."""
        result = self.result
        try:
            # Create new SQL context record and add to context
            new_record = SQLContext(sql_query=result.sql_query, explanation=result.explanation or "")
            workflow.context.sql_contexts.append(new_record)

            # Get and update schema information
            table_schemas, table_values = self._get_schema_and_values(workflow.task, result.tables)
            if len(table_schemas) == len(result.tables) and len(table_values) == len(result.tables):
                workflow.context.table_schemas = table_schemas
                workflow.context.table_values = table_values
                return {"success": True, "message": "Updated SQL generation context"}
            else:
                error_msg = (
                    f"Failed to get schemas and values for tables {result.tables} " f"{workflow.task.database_name}"
                )
                logger.warning(f"{error_msg}, table_schemas: {table_schemas}, table_values: {table_values}")
                return {"success": True, "message": error_msg}
        except Exception as e:
            logger.error(f"Failed to update SQL generation context: {str(e)}")
            return {"success": False, "message": f"SQL generation context update failed: {str(e)}"}

    def _execute_generate_sql(self) -> GenerateSQLResult:
        """Execute SQL generation action to create SQL query.

        Combines input data from previous nodes into a structured format for SQL generation.
        The input data includes:
        - table_schemas: Database schema information from schema linking
        - data_details: Additional data context
        - metrics: Relevant metrics information
        - database: Database type information

        Returns:
            GenerateSQLResult containing the generated SQL query
        """
        if not self.model:
            return GenerateSQLResult(
                success=False,
                error="SQL generation model not provided",
                sql_query="",
                tables=[],
                explanation=None,
            )

        try:
            logger.debug(f"Generate SQL input: {type(self.input)} {self.input}")
            return generate_sql(self.model, self.input)
        except Exception as e:
            logger.error(f"SQL generation execution error: {str(e)}")
            return GenerateSQLResult(success=False, error=str(e), sql_query="", tables=[], explanation=None)

    def _get_schema_and_values(
        self, sql_task: SqlTask, table_names: List[str]
    ) -> Tuple[List[TableSchema], List[TableValue]]:
        """Get table schemas and values using the schema lineage tool."""
        try:
            # Get the schema lineage tool instance
            sql_connector = self._sql_connector(self.input.sql_task.database_name)
            catalog_name = sql_task.catalog_name or sql_connector.catalog_name
            database_name = sql_task.database_name or sql_connector.database_name
            schema_name = sql_task.schema_name or sql_connector.schema_name

            # Use the tool to get schemas and values
            logger.debug(f"Getting schemas and values for tables {table_names} from {database_name}")
            return self.metadata_rag.search_tables(
                tables=table_names,
                catalog_name=catalog_name,
                database_name=database_name,
                schema_name=schema_name,
                dialect=sql_task.database_type,
            )
        except Exception as e:
            logger.warning(f"Failed to get schemas and values for tables {table_names}: {e}")
            return [], []  # Return empty lists if lookup fails

    async def _generate_sql_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Generate SQL with streaming support and action history tracking."""
        if not self.model:
            logger.error("Model not available for SQL generation")
            return

        try:
            # SQL generation preparation action
            prep_action = ActionHistory(
                action_id="sql_generation_prep",
                role=ActionRole.WORKFLOW,
                messages="Preparing SQL generation with schema and context information",
                action_type="sql_preparation",
                input={
                    "database_type": self.input.database_type if hasattr(self.input, "database_type") else "",
                    "table_count": len(self.input.table_schemas)
                    if hasattr(self.input, "table_schemas") and self.input.table_schemas
                    else 0,
                    "has_metrics": bool(hasattr(self.input, "metrics") and self.input.metrics),
                    "has_external_knowledge": bool(
                        hasattr(self.input, "external_knowledge") and self.input.external_knowledge
                    ),
                },
                status=ActionStatus.PROCESSING,
            )
            yield prep_action

            # Update preparation status
            try:
                prep_action.status = ActionStatus.SUCCESS
                prep_action.output = {
                    "preparation_complete": True,
                    "input_validated": True,
                }
            except Exception as e:
                prep_action.status = ActionStatus.FAILED
                prep_action.output = {"error": str(e)}
                logger.warning(f"SQL preparation failed: {e}")

            # SQL generation action
            generation_action = ActionHistory(
                action_id="sql_generation",
                role=ActionRole.WORKFLOW,
                messages="Generating SQL query based on schema and requirements",
                action_type="sql_generation",
                input={
                    "task_description": getattr(self.input.sql_task, "task", "")
                    if hasattr(self.input, "sql_task")
                    else "",
                    "database_type": self.input.database_type if hasattr(self.input, "database_type") else "",
                },
                status=ActionStatus.PROCESSING,
            )
            yield generation_action

            # Execute SQL generation - reuse existing logic
            try:
                result = self._execute_generate_sql()

                generation_action.status = ActionStatus.SUCCESS
                generation_action.output = {
                    "success": result.success,
                    "sql_query": result.sql_query,
                    "tables_involved": result.tables if result.tables else [],
                    "has_explanation": bool(result.explanation),
                }

                # Store result for later use
                self.result = result

            except Exception as e:
                generation_action.status = ActionStatus.FAILED
                generation_action.output = {"error": str(e)}
                logger.error(f"SQL generation error: {str(e)}")
                raise

            # Yield the updated generation action with final status
            yield generation_action

        except Exception as e:
            logger.error(f"SQL generation streaming error: {str(e)}")
            raise


@optional_traceable()
def generate_sql(model: LLMBaseModel, input_data: GenerateSQLInput) -> GenerateSQLResult:
    """Generate SQL query using the provided model."""
    if not isinstance(input_data, GenerateSQLInput):
        raise TypeError("Input data must be a GenerateSQLInput instance")

    sql_query = ""
    try:
        # Format the prompt with schema list
        prompt = get_sql_prompt(
            database_type=input_data.database_type or DBType.SQLITE.value,
            table_schemas=input_data.table_schemas,
            data_details=input_data.data_details,
            metrics=input_data.metrics,
            question=input_data.sql_task.task,
            external_knowledge=input_data.external_knowledge,
            prompt_version=input_data.prompt_version,
            context=[sql_context.to_str() for sql_context in input_data.contexts],
            max_table_schemas_length=input_data.max_table_schemas_length,
            max_data_details_length=input_data.max_data_details_length,
            max_context_length=input_data.max_context_length,
            max_value_length=input_data.max_value_length,
            max_text_mark_length=input_data.max_text_mark_length,
            database_docs=input_data.database_docs,
            current_date=get_default_current_date(input_data.sql_task.current_date),
            date_ranges=getattr(input_data.sql_task, "date_ranges", ""),
        )

        logger.debug(f"Generated SQL prompt:  {type(model)}, {prompt}")
        # Generate SQL using the provided model
        sql_query = model.generate_with_json_output(prompt)
        logger.debug(f"Generated SQL: {sql_query}")

        # Clean and parse the response
        if isinstance(sql_query, str):
            # Remove markdown code blocks if present
            sql_query = sql_query.strip().replace("```json\n", "").replace("\n```", "")
            # Remove SQL comments
            cleaned_lines = []
            for line in sql_query.split("\n"):
                line = line.strip()
                if line and not line.startswith("--"):
                    cleaned_lines.append(line)
            cleaned_sql = " ".join(cleaned_lines)
            try:
                sql_query_dict = json.loads(cleaned_sql)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse cleaned SQL: {cleaned_sql}")
                return GenerateSQLResult(success=False, error="Invalid JSON format", sql_query=sql_query)
        else:
            sql_query_dict = sql_query

        # Return result as GenerateSQLResult
        if sql_query_dict and isinstance(sql_query_dict, dict):
            return GenerateSQLResult(
                success=True,
                error=None,
                sql_query=sql_query_dict.get("sql", ""),
                tables=sql_query_dict.get("tables", []),
                explanation=sql_query_dict.get("explanation"),
            )
        else:
            return GenerateSQLResult(success=False, error="sql generation failed, no result", sql_query=sql_query)
    except json.JSONDecodeError as e:
        logger.error(f"SQL json decode failed: {e}")
        return GenerateSQLResult(success=False, error=str(e), sql_query=str(sql_query))
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        return GenerateSQLResult(success=False, error=str(e), sql_query="")
