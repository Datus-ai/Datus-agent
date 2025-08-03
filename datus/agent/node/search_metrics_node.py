from typing import Dict

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.generate_semantic_model_node_models import SemanticModelMeta
from datus.schemas.search_metrics_node_models import SearchMetricsInput, SearchMetricsResult
from datus.tools.metric_tools.search_metric import SearchMetricsTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SearchMetricsNode(Node):
    def setup_input(self, workflow: Workflow) -> Dict:
        logger.info("Setup search metrics input")
        semantic_model_meta = SemanticModelMeta(
            catalog_name=workflow.task.catalog_name,
            database_name=workflow.task.database_name,
            schema_name=workflow.task.schema_name,
            layer1=workflow.task.layer1,
            layer2=workflow.task.layer2,
            domain=workflow.task.domain,
        )

        # irrelevant to current node: it should be Start or Reflection node now
        matching_rate = self.agent_config.search_metrics_rate
        matching_rates = ["fast", "medium", "slow"]
        start = matching_rates.index(matching_rate)
        final_matching_rate = matching_rates[min(start + workflow.reflection_round, len(matching_rates) - 1)]
        logger.debug(f"Final matching rate: {final_matching_rate}")

        next_input = SearchMetricsInput(
            input_text=workflow.task.task,
            matching_rate=final_matching_rate,
            semantic_model_meta=semantic_model_meta,
            database_type=workflow.task.database_type,
            sql_contexts=workflow.context.sql_contexts,
        )
        self.input = next_input
        return {"success": True, "message": "Search Metrics appears valid"}

    def execute(self):
        self.result = self._execute_search_metrics()

    def _execute_search_metrics(self) -> SearchMetricsResult:
        """Execute schema linking action to analyze database schema.
        Input:
             query - The input query to analyze.
             catalog_name - The catalog name to use.
             database_name - The database name to use.
             schema_name - The schema name to use.
        Returns:
             A validated SchemaLinkingResult containing table schemas and values.
        """
        import os

        path = self.agent_config.rag_storage_path()
        logger.debug(f"Checking if rag storage path exists: {path}")
        if not os.path.exists(path):
            logger.info("RAG storage path does not exist.")
            return self.get_bad_result("RAG storage path does not exist.")
        else:
            try:
                tool = SearchMetricsTool(db_path=self.agent_config.rag_storage_path())
                if tool:
                    result = tool.execute(self.input, self.model)
                    logger.info(f"Search metric result: found {result} ")
                    if not result:
                        logger.info("No search result , please check your config or table data")
                        return self.get_bad_result("No search result , please check your config or table data")
                    else:
                        return result
                else:
                    logger.warning("Search metrics tool not found")
                    return self.get_bad_result("Search metrics tool not found")

            except Exception as e:
                logger.warning(f"Search metrics tool initialization failed: {e}")
                return self.get_bad_result(str(e))

    def get_bad_result(self, error_msg: str):
        return SearchMetricsResult(
            success=False,
            error=error_msg,
            semantic_model_meta=self.input.semantic_model_meta,
            metrics=[],
            metrics_count=0,
        )

    def update_context(self, workflow: Workflow) -> Dict:
        """Update search metrics results to workflow context."""
        result = self.result
        try:
            if len(workflow.context.metrics) == 0:
                workflow.context.metrics = result.metrics
            else:
                pass  # if it's not the first search metrics, wait it after execute_sql

            return {"success": True, "message": "Updated search metrics context"}
        except Exception as e:
            logger.error(f"Failed to update search metrics context: {str(e)}")
            return {"success": False, "message": f"Search metrics context update failed: {str(e)}"}
