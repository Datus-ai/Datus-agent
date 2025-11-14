# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
from typing import List

from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.storage.metric.store import SemanticMetricsRAG
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GenerationTools:
    """
    Tools for semantic model generation workflow.

    This class provides tools for checking existing semantic models and
    completing the generation process.
    """

    def __init__(self, agent_config: AgentConfig):
        self.agent_config = agent_config
        self.metrics_rag = SemanticMetricsRAG(agent_config)

    def available_tools(self) -> List[Tool]:
        """
        Provide tools for generation workflow.

        Returns:
            List of available tools for generation workflow
        """
        return [
            trans_to_function_tool(func)
            for func in (
                self.check_semantic_model_exists,
                self.check_metric_exists,
                self.generate_sql_summary_id,
                self.end_generation,
            )
        ]

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

    def check_metric_exists(self, metric_name: str) -> FuncToolResult:
        """
        Check if metric already exists in LanceDB.

        Use this tool when you need to:
        - Avoid generating duplicate metrics
        - Check if a metric already has definition
        - Get existing metric content for reference

        Args:
            metric_name: Name of the metric

        Returns:
            dict: Check results containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if failed
                - 'result' (dict): Contains:
                    - 'exists' (bool): Whether metric exists
                    - 'metric_name' (str): Metric name if exists
                    - 'llm_text' (str): Metric definition text if found
        """
        try:
            # Search for existing metrics by name
            all_metrics_table = self.metrics_rag.metric_storage.search(
                query_txt=metric_name, select_fields=["name", "llm_text"], top_n=3
            )

            # Convert PyArrow Table to list of dicts
            all_metrics = all_metrics_table.to_pylist()

            # Filter by exact metric name match
            for metric in all_metrics:
                stored_name = metric.get("name", "").lower()
                target_name = metric_name.lower()

                # Check exact match
                if stored_name == target_name:
                    return FuncToolResult(
                        result={
                            "exists": True,
                            "metric_name": metric.get("name", ""),
                            "llm_text": metric.get("llm_text", ""),
                            "message": f"Metric already exists: '{metric_name}'",
                        }
                    )

            # No match found
            return FuncToolResult(result={"exists": False, "message": f"No metric found with name '{metric_name}'"})

        except Exception as e:
            logger.error(f"Error checking metric existence: {e}")
            return FuncToolResult(success=0, error=f"Failed to check metric: {str(e)}")

    def end_generation(self, filepath: str) -> FuncToolResult:
        """
        Complete the generation process.

        Call this tool when you have finished generating a YAML file (semantic model, metric, etc.).
        This tool triggers user confirmation workflow for syncing to LanceDB.

        Args:
            filepath: Absolute path to the generated YAML file

        Returns:
            dict: Result containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if failed
                - 'result' (dict): Contains confirmation message and filepath
        """
        try:
            logger.info(f"Generation completed for file: {filepath}")

            return FuncToolResult(
                result={
                    "message": "Generation completed successfully",
                    "filepath": filepath,
                }
            )

        except Exception as e:
            logger.error(f"Error completing generation: {e}")
            return FuncToolResult(success=0, error=f"Failed to complete generation: {str(e)}")

    def generate_sql_summary_id(self, sql_query: str, comment: str = "") -> FuncToolResult:
        """
        Generate a unique ID for SQL summary based on SQL query and comment.

        This tool helps create consistent, unique IDs for SQL summary entries.
        Use this tool when you need to generate an ID for a new SQL summary entry.

        Args:
            sql_query: The SQL query that will be used to generate the ID
            comment: Optional comment/description that helps make the ID more unique

        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure
                  - 'error' (Optional[str]): Error message on failure
                  - 'result' (str): The generated unique ID

        Example:
            result = generate_sql_summary_id(
                sql_query="SELECT * FROM users WHERE active = 1",
                comment="Active users query"
            )
        """
        try:
            from datus.storage.reference_sql.init_utils import gen_reference_sql_id

            # Generate the ID using the same utility as the storage system
            generated_id = gen_reference_sql_id(sql_query, comment)

            logger.info(f"Generated reference SQL ID: {generated_id}")
            return FuncToolResult(result=generated_id)

        except Exception as e:
            logger.error(f"Error generating reference SQL ID: {e}")
            return FuncToolResult(success=0, error=f"Failed to generate ID: {str(e)}")
