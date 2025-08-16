from typing import AsyncGenerator, Dict, Optional

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.schema_linking_node_models import SchemaLinkingInput, SchemaLinkingResult
from datus.storage.ext_knowledge.store import ExtKnowledgeStore
from datus.tools.lineage_graph_tools.schema_lineage import SchemaLineageTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SchemaLinkingNode(Node):
    def execute(self):
        self.result = self._execute_schema_linking()

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute schema linking with streaming support."""
        async for action in self._schema_linking_stream(action_history_manager):
            yield action

    def update_context(self, workflow: Workflow) -> Dict:
        """Update schema linking results to workflow context."""
        result = self.result
        try:
            if len(workflow.context.table_schemas) == 0:
                workflow.context.table_schemas = result.table_schemas
                workflow.context.table_values = result.table_values
            else:
                pass  # if it's not the first schema linking, wait it after execute_sql

            return {"success": True, "message": "Updated schema linking context"}
        except Exception as e:
            logger.error(f"Failed to update schema linking context: {str(e)}")
            return {"success": False, "message": f"Schema linking context update failed: {str(e)}"}

    def setup_input(self, workflow: Workflow) -> Dict:
        logger.info("Setup schema linking input")

        # Search and enhance external knowledge before schema linking
        enhanced_external_knowledge = self._search_external_knowledge(
            workflow.task.task,  # User query
            workflow.task.domain,  # Business domain
            workflow.task.layer1,  # First layer
            workflow.task.layer2,  # Second layer
        )

        # Combine original and searched knowledge
        if enhanced_external_knowledge:
            original_knowledge = workflow.task.external_knowledge
            combined_knowledge = self._combine_knowledge(original_knowledge, enhanced_external_knowledge)
            workflow.task.external_knowledge = combined_knowledge

        # Setup schema linking input
        matching_rate = self.agent_config.schema_linking_rate
        matching_rates = ["fast", "medium", "slow", "from_llm"]
        start = matching_rates.index(matching_rate)
        final_matching_rate = matching_rates[min(start + workflow.reflection_round, len(matching_rates) - 1)]
        logger.debug(f"Final matching rate: {final_matching_rate}")
        next_input = SchemaLinkingInput(
            input_text=workflow.task.task,
            matching_rate=final_matching_rate,
            database_type=workflow.task.database_type,
            database_name=workflow.task.database_name,
            sql_context=None,
            table_type=workflow.task.schema_linking_type,
        )
        self.input = next_input
        return {"success": True, "message": "Schema and external knowledge prepared"}

    def _execute_schema_linking(self) -> SchemaLinkingResult:
        """Execute schema linking action to analyze database schema.
        Input:
            query - The input query to analyze.
        Returns:
            A validated SchemaLinkingResult containing table schemas and values.
        """
        import os

        path = self.agent_config.rag_storage_path()
        logger.debug(f"Checking if rag storage path exists: {path}")
        if not os.path.exists(path):
            logger.info(f"RAG storage path `{path}` does not exist.")
            return self._execute_schema_linking_fallback(SchemaLineageTool())
        else:
            tool = SchemaLineageTool(db_path=path)
            try:
                # Import SchemaLineageTool only when needed
                if tool:
                    result = tool.execute(self.input, self.model)
                    if not result.success:
                        logger.warning(f"Schema linking failed: {result.error}")
                        return self._execute_schema_linking_fallback(tool)
                    logger.info(f"Schema linking result: found {len(result.table_schemas)} tables")
                    if len(result.table_schemas) > 0:
                        return result
                    logger.info("No tables found, using fallback method")
                    return self._execute_schema_linking_fallback(tool)
                else:
                    logger.warning("Schema linking tool not found")
                    return self._execute_schema_linking_fallback(tool)
            except Exception as e:
                logger.warning(f"Schema linking tool initialization failed: {e}")
                return self._execute_schema_linking_fallback(tool)

    def _execute_schema_linking_fallback(self, tool: SchemaLineageTool) -> SchemaLinkingResult:
        # Fallback: directly get tables from current database
        logger.info("Get tables directly from database")
        try:
            # Get database connector through db_manager
            from datus.tools.db_tools.db_manager import db_manager_instance

            db_manager = db_manager_instance(self.agent_config.namespaces)

            # Get current namespace and database connection
            current_namespace = self.agent_config.current_namespace
            database_name = self.input.database_name if hasattr(self.input, "database_name") else ""

            # Get database connector
            connector = db_manager.get_conn(current_namespace, database_name)

            return tool.get_schems_by_db(connector=connector, input_param=self.input)

        except Exception as e:
            logger.warning(f"Schema linking failed: {e}")
            return SchemaLinkingResult(
                success=False,
                error=f"Schema linking failed: {e}",
                schema_count=0,
                value_count=0,
                table_schemas=[],
                table_values=[],
            )

    def _search_external_knowledge(self, user_query: str, domain: str = "", layer1: str = "", layer2: str = "") -> str:
        """Search for relevant external knowledge based on user query and metadata.

        Args:
            user_query: The user's natural language query
            domain: Business domain filter
            layer1: First layer filter
            layer2: Second layer filter

        Returns:
            Formatted string of relevant knowledge entries, empty string if no results or error
        """
        try:
            # Initialize ExtKnowledgeStore
            storage_path = self.agent_config.rag_storage_path()
            ext_knowledge_store = ExtKnowledgeStore(storage_path)

            # Check if ext_knowledge table exists
            if ext_knowledge_store.table_size() == 0:
                logger.debug("External knowledge store is empty, skipping search")
                return ""

            # Execute semantic search
            search_results = ext_knowledge_store.search_knowledge(
                query_text=user_query, domain=domain, layer1=layer1, layer2=layer2, top_n=5
            )

            # Format search results
            if search_results:
                knowledge_items = []
                for result in search_results:
                    knowledge_items.append(f"- {result['terminology']}: {result['explanation']}")

                formatted_knowledge = "\n".join(knowledge_items)
                logger.info(f"Found {len(search_results)} relevant knowledge entries")
                return formatted_knowledge
            else:
                logger.debug("No relevant external knowledge found")
                return ""

        except Exception as e:
            logger.warning(f"Failed to search external knowledge: {str(e)}")
            return ""

    def _combine_knowledge(self, original: str, enhanced: str) -> str:
        """Combine original knowledge and searched knowledge.

        Args:
            original: Original external knowledge from SqlTask
            enhanced: Knowledge retrieved from search

        Returns:
            Combined knowledge string
        """
        parts = []
        if original:
            parts.append(original)
        if enhanced:
            parts.append(f"Relevant Business Knowledge:\n{enhanced}")

        return "\n\n".join(parts)

    async def _schema_linking_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute schema linking with streaming support and action history tracking."""
        try:
            # External knowledge search action
            knowledge_action = ActionHistory(
                action_id="external_knowledge_search",
                role=ActionRole.WORKFLOW,
                messages="Searching external knowledge base for relevant business context",
                action_type="knowledge_search",
                input={
                    "query": self.input.input_text if hasattr(self.input, "input_text") else "",
                    "database_name": self.input.database_name if hasattr(self.input, "database_name") else "",
                },
                status=ActionStatus.PROCESSING,
            )
            yield knowledge_action

            # Execute external knowledge search
            try:
                enhanced_knowledge = self._search_external_knowledge(
                    self.input.input_text if hasattr(self.input, "input_text") else "",
                    "",
                    "",
                    "",  # domain, layer1, layer2 - would need to get from workflow if available
                )
                knowledge_action.status = ActionStatus.SUCCESS
                knowledge_action.output = {
                    "knowledge_found": bool(enhanced_knowledge),
                    "knowledge_length": len(enhanced_knowledge) if enhanced_knowledge else 0,
                }
            except Exception as e:
                knowledge_action.status = ActionStatus.FAILED
                knowledge_action.output = {"error": str(e)}
                logger.warning(f"External knowledge search failed: {e}")

            # Schema linking action
            schema_action = ActionHistory(
                action_id="schema_linking",
                role=ActionRole.WORKFLOW,
                messages="Analyzing database schema and linking relevant tables",
                action_type="schema_linking",
                input={
                    "input_text": self.input.input_text if hasattr(self.input, "input_text") else "",
                    "database_name": self.input.database_name if hasattr(self.input, "database_name") else "",
                    "matching_rate": self.input.matching_rate if hasattr(self.input, "matching_rate") else "medium",
                },
                status=ActionStatus.PROCESSING,
            )
            yield schema_action

            # Execute schema linking - reuse existing logic
            try:
                result = self._execute_schema_linking()

                schema_action.status = ActionStatus.SUCCESS
                schema_action.output = {
                    "success": result.success,
                    "tables_found": len(result.table_schemas) if result.table_schemas else 0,
                    "values_found": len(result.table_values) if result.table_values else 0,
                }

                # Store result for later use
                self.result = result

            except Exception as e:
                schema_action.status = ActionStatus.FAILED
                schema_action.output = {"error": str(e)}
                logger.error(f"Schema linking error: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Schema linking streaming error: {str(e)}")
            raise
