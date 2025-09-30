# -*- coding: utf-8 -*-
from typing import List

from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.storage.metric.store import rag_by_configuration
from datus.tools.tools import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SchemaTools:
    """
    Tools for checking and searching existing semantic models and schemas.

    This class provides tools for searching existing semantic models in LanceDB,
    avoiding duplicate generation and providing context for new model creation.
    """

    def __init__(self, agent_config: AgentConfig):
        self.agent_config = agent_config
        self.metrics_rag = rag_by_configuration(agent_config)

    def available_tools(self) -> List[Tool]:
        """
        Provide tools for checking and searching existing semantic models.

        Returns:
            List of available tools for semantic model search and validation
        """
        return [trans_to_function_tool(func) for func in (self.check_semantic_model_exists,)]

    def check_semantic_model_exists(
        self,
        table_name: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
    ) -> FuncToolResult:
        """
        Check if semantic model already exists in LanceDB.

        Use this tool when you need to:
        - Avoid generating duplicate semantic models
        - Check if a table already has semantic model definition
        - Get existing semantic model content for reference

        Args:
            table_name: Name of the database table
            catalog_name: Catalog name (optional)
            database_name: Database name (optional)
            schema_name: Schema name (optional)

        Returns:
            dict: Check results containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if failed
                - 'result' (dict): Contains:
                    - 'exists' (bool): Whether semantic model exists
                    - 'file_path' (str): Path to existing semantic model file if exists
                    - 'semantic_model' (dict): Existing semantic model content if found
        """
        try:
            # Search for existing semantic models by database name
            # Use search_all_semantic_models which exists in SemanticMetricsRAG
            all_models = self.metrics_rag.search_all_semantic_models(database_name=database_name or "")

            # Filter by exact table name match
            for model in all_models:
                model_table = model.get("table_name", "").lower()
                target_table = table_name.lower()

                # Check exact match
                if model_table == target_table:
                    # Also check schema and catalog if provided
                    if schema_name and model.get("schema_name", "").lower() != schema_name.lower():
                        continue
                    if catalog_name and model.get("catalog_name", "").lower() != catalog_name.lower():
                        continue

                    return FuncToolResult(
                        result={
                            "exists": True,
                            "file_path": model.get("semantic_file_path", ""),
                            "semantic_model_name": model.get("semantic_model_name", ""),
                            "table_name": model.get("table_name", ""),
                            "message": f"Semantic model already exists for table '{table_name}'",
                        }
                    )

            # No match found
            return FuncToolResult(
                result={"exists": False, "message": f"No semantic model found for table '{table_name}'"}
            )

        except Exception as e:
            logger.error(f"Error checking semantic model existence: {e}")
            return FuncToolResult(success=0, error=f"Failed to check semantic model: {str(e)}")
