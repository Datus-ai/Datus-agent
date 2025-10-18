# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
from typing import List

from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.storage.metric.store import SemanticMetricsRAG
from datus.tools.tools import FuncToolResult, trans_to_function_tool
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
                self.prepare_sql_summary_context,
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
                    - 'description' (str): Metric description if found
        """
        try:
            # Search for existing metrics by name
            all_metrics_table = self.metrics_rag.metric_storage.search(
                query_txt=metric_name, select_fields=["name", "description"], top_n=10
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
                            "description": metric.get("description", ""),
                            "message": f"Metric already exists: '{metric_name}'",
                        }
                    )

            # No match found
            return FuncToolResult(result={"exists": False, "message": f"No metric found with name '{metric_name}'"})

        except Exception as e:
            logger.error(f"Error checking metric existence: {e}")
            return FuncToolResult(success=0, error=f"Failed to check metric: {str(e)}")

    def _get_sql_history_taxonomy(self) -> FuncToolResult:
        """
        Get existing SQL history classification taxonomy.

        Use this tool to retrieve the current classification taxonomy used for SQL histories.
        This helps maintain consistency when classifying new SQL queries.

        Returns:
            dict: Taxonomy containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if failed
                - 'result' (dict): Contains:
                    - 'domains' (list): Available business domains
                    - 'layer1_categories' (list): Primary categories with domain associations
                    - 'layer2_categories' (list): Secondary categories with layer1 associations
                    - 'common_tags' (list): Available tags
        """
        try:
            from datus.storage.sql_history.store import SqlHistoryRAG

            # Get SQL history storage
            storage = SqlHistoryRAG(self.agent_config)
            taxonomy = storage.sql_history_storage.get_existing_taxonomy()

            return FuncToolResult(
                result={
                    "taxonomy": taxonomy,
                    "message": f"Found {len(taxonomy.get('domains', []))} domains, "
                    f"{len(taxonomy.get('layer1_categories', []))} layer1 categories, "
                    f"{len(taxonomy.get('layer2_categories', []))} layer2 categories, "
                    f"{len(taxonomy.get('common_tags', []))} tags",
                }
            )

        except Exception as e:
            logger.error(f"Error getting reference SQL taxonomy: {e}")
            return FuncToolResult(success=0, error=f"Failed to get taxonomy: {str(e)}")

    def _check_sql_name_exists(self, name: str) -> FuncToolResult:
        """
        Check if an SQL history name already exists.

        Use this tool to avoid duplicate SQL history names.

        Args:
            name: SQL history name to check

        Returns:
            dict: Check results containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if failed
                - 'result' (dict): Contains:
                    - 'exists' (bool): Whether the name already exists
                    - 'conflicting_names' (list): List of similar existing names
                    - 'message' (str): Description message
        """
        try:
            from datus.storage.sql_history.store import SqlHistoryRAG

            # Get SQL history storage
            storage = SqlHistoryRAG(self.agent_config)

            # Search for similar names using FTS
            all_items = storage.search_all_sql_history()

            # Check for exact match and collect similar names
            exact_match = False
            similar_names = []

            for item in all_items:
                item_name = item.get("name", "").lower()
                search_name = name.lower()

                if item_name == search_name:
                    exact_match = True
                elif search_name in item_name or item_name in search_name:
                    similar_names.append(item.get("name", ""))

            if exact_match:
                return FuncToolResult(
                    result={
                        "exists": True,
                        "conflicting_names": [name] + similar_names[:5],
                        "message": f"Name '{name}' already exists. Consider using a different name.",
                    }
                )
            elif similar_names:
                return FuncToolResult(
                    result={
                        "exists": False,
                        "conflicting_names": similar_names[:10],
                        "message": f"Name '{name}' is available, but {len(similar_names)} similar names exist.",
                    }
                )
            else:
                return FuncToolResult(
                    result={
                        "exists": False,
                        "conflicting_names": [],
                        "message": f"Name '{name}' is available and unique.",
                    }
                )

        except Exception as e:
            logger.error(f"Error checking SQL name existence: {e}")
            return FuncToolResult(success=0, error=f"Failed to check SQL name: {str(e)}")

    def _get_similar_sql_histories(self, comment: str = "", summary: str = "", top_n: int = 5) -> FuncToolResult:
        """
        Get similar SQL histories based on comment or summary.

        Use this tool to find similar existing SQL histories for reference when
        classifying new SQL queries.

        Args:
            comment: SQL comment for similarity search
            summary: SQL summary for similarity search (if available)
            top_n: Number of similar results to return (default: 5)

        Returns:
            dict: Search results containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if failed
                - 'result' (dict): Contains:
                    - 'similar_items' (list): List of similar SQL histories with fields:
                        name, domain, layer1, layer2, tags, comment, summary
                    - 'count' (int): Number of results found
        """
        try:
            from datus.storage.sql_history.store import SqlHistoryRAG

            # Get SQL history storage
            storage = SqlHistoryRAG(self.agent_config)

            # Use summary for vector search if available, otherwise use comment
            query_text = summary if summary else comment

            if not query_text:
                return FuncToolResult(
                    result={
                        "similar_items": [],
                        "count": 0,
                        "message": "No query text provided for similarity search",
                    }
                )

            # Search using vector similarity on summary field
            similar_items = storage.search_sql_history_by_summary(query_text=query_text, top_n=top_n)

            # Extract relevant fields
            results = []
            for item in similar_items:
                results.append(
                    {
                        "name": item.get("name", ""),
                        "domain": item.get("domain", ""),
                        "layer1": item.get("layer1", ""),
                        "layer2": item.get("layer2", ""),
                        "tags": item.get("tags", ""),
                        "comment": item.get("comment", ""),
                        "summary": item.get("summary", ""),
                    }
                )

            return FuncToolResult(
                result={
                    "similar_items": results,
                    "count": len(results),
                    "message": f"Found {len(results)} similar SQL histories",
                }
            )

        except Exception as e:
            logger.error(f"Error getting similar SQL histories: {e}")
            return FuncToolResult(success=0, error=f"Failed to get similar SQL histories: {str(e)}")

    def prepare_sql_summary_context(
        self,
        sql: str,
        comment: str = "",
        suggested_name: str = "",
    ) -> FuncToolResult:
        """
        One-shot context preparation for SQL summary generation.

        This tool combines multiple preparatory steps into a single call:
        1. Get existing taxonomy (domains, layers, tags)
        2. Find similar SQL histories for reference
        3. Check name uniqueness (if suggested_name provided)

        Use this tool at the beginning of SQL summary workflow to get all necessary
        context in one efficient call.

        Args:
            sql: SQL query to analyze
            comment: SQL comment for similarity search
            suggested_name: Optional suggested name to check for uniqueness

        Returns:
            dict: Comprehensive context containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if failed
                - 'result' (dict): Contains:
                    - 'taxonomy' (dict): Existing classification taxonomy
                    - 'similar_items' (list): Similar SQL histories for reference
                    - 'name_check' (dict): Name uniqueness check result (if suggested_name provided)
                    - 'message' (str): Summary message
        """
        try:
            result = {}
            messages = []

            # 1. Get taxonomy
            taxonomy_result = self._get_sql_history_taxonomy()
            if taxonomy_result.success:
                result["taxonomy"] = taxonomy_result.result
                taxonomy_info = taxonomy_result.result.get("message", "")
                messages.append(f"Taxonomy: {taxonomy_info}")
            else:
                logger.debug(f"Failed to get taxonomy: {taxonomy_result.error}")
                result["taxonomy"] = {"error": taxonomy_result.error}

            # 2. Find similar histories
            similar_result = self._get_similar_sql_histories(
                comment=comment,
                summary=sql[:200],  # Use first 200 chars of SQL as fallback
                top_n=5,
            )
            if similar_result.success:
                result["similar_items"] = similar_result.result.get("similar_items", [])
                similar_count = similar_result.result.get("count", 0)
                messages.append(f"Similar histories: {similar_count}")
            else:
                logger.debug(f"Failed to get similar histories: {similar_result.error}")
                result["similar_items"] = []

            # 3. Check name if provided
            if suggested_name:
                name_result = self._check_sql_name_exists(name=suggested_name)
                if name_result.success:
                    result["name_check"] = name_result.result
                    name_status = name_result.result.get("message", "")
                    messages.append(f"Name check: {name_status}")
                else:
                    logger.debug(f"Failed to check name: {name_result.error}")
                    result["name_check"] = {"error": name_result.error}

            result["message"] = " | ".join(messages)

            return FuncToolResult(result=result)

        except Exception as e:
            logger.error(f"Error preparing SQL summary context: {e}")
            return FuncToolResult(success=0, error=f"Failed to prepare context: {str(e)}")

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
            from datus.storage.sql_history.init_utils import gen_sql_history_id

            # Generate the ID using the same utility as the storage system
            generated_id = gen_sql_history_id(sql_query, comment)

            logger.info(f"Generated reference SQL ID: {generated_id}")
            return FuncToolResult(result=generated_id)

        except Exception as e:
            logger.error(f"Error generating reference SQL ID: {e}")
            return FuncToolResult(success=0, error=f"Failed to generate ID: {str(e)}")
