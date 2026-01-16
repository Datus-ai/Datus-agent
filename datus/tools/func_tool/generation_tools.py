# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
from typing import Dict, List

from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.storage.lancedb_conditions import And, build_where, eq
from datus.storage.metric.store import MetricRAG
from datus.storage.semantic_model.store import SemanticModelRAG
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
        self.metric_rag = MetricRAG(agent_config)
        self.semantic_rag = SemanticModelRAG(agent_config)

    def available_tools(self) -> List[Tool]:
        """
        Provide tools for generation workflow.

        Returns:
            List of available tools for generation workflow
        """
        return [
            trans_to_function_tool(func)
            for func in (
                self.check_semantic_object_exists,  # Updated name to reflect broader scope
                self.check_metric_exists,  # Kept for backward compat or specific metric checks
                self.generate_sql_summary_id,
                self.end_semantic_model_generation,
                self.end_metric_generation,
                self.generate_ext_knowledge_id,
            )
        ]

    def check_semantic_object_exists(
        self,
        object_name: str,
        kind: str = "table",  # table, column, metric
        table_context: str = "",
    ) -> FuncToolResult:
        """
        Check if a semantic object (table, column, metric) already exists in LanceDB.

        Use this tool to avoid duplicating work.

        Args:
            object_name: Name of the object (e.g. "orders", "orders.amount")
            kind: Type of object ("table", "column", "metric")
            table_context: If checking a column/metric, providing the table name helps narrow search.

        Returns:
            dict: Check results containing existence status and details.
        """
        try:
            # Extract the final segment as target name (e.g., "public.orders" -> "orders")
            target_name = object_name.split(".")[-1].lower()

            found_object = None

            if kind == "table":
                # Exact match for table using SQL WHERE condition
                storage = self.semantic_rag.storage
                where = build_where(And([eq("kind", "table"), eq("name", target_name)]))
                results = storage.search_all(where=where, select_fields=["id", "name", "kind"])
                if results:
                    found_object = results[0]
            elif kind == "metric":
                # For metrics, search all and filter by name
                storage = self.metric_rag.storage
                results = storage.search_all(select_fields=["id", "name"])
                for obj in results:
                    if obj.get("name", "").lower() == target_name:
                        found_object = obj
                        break
            else:
                # For column, use vector search + post-filter
                storage = self.semantic_rag.storage
                results = storage.search_objects(
                    query_text=object_name,
                    kinds=[kind],
                    table_name=table_context if table_context else None,
                    top_n=5,
                )
                # Determine target table from explicit context or dotted name
                target_table = None
                if table_context:
                    target_table = table_context.lower()
                elif "." in object_name:
                    target_table = object_name.rsplit(".", 1)[0].lower()

                for obj in results:
                    name_match = obj.get("name", "").lower() == target_name
                    if target_table:
                        table_match = obj.get("table_name", "").lower() == target_table
                        if name_match and table_match:
                            found_object = obj
                            break
                    elif name_match:
                        found_object = obj
                        break

            if found_object:
                return FuncToolResult(
                    result={
                        "exists": True,
                        "id": found_object.get("id"),
                        "name": found_object.get("name"),
                        "kind": found_object.get("kind") or kind,
                        "message": f"Object '{object_name}' ({kind}) already exists.",
                    }
                )

            return FuncToolResult(result={"exists": False, "message": f"No {kind} found for '{object_name}'"})

        except Exception as e:
            logger.error(f"Error checking semantic object existence: {e}")
            return FuncToolResult(success=0, error=f"Failed to check object: {str(e)}")

    # Backward compatibility wrapper
    def check_semantic_model_exists(
        self,
        table_name: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
    ) -> FuncToolResult:
        """Legacy wrapper for checking table existence."""
        return self.check_semantic_object_exists(table_name, kind="table")

    def check_metric_exists(self, metric_name: str) -> FuncToolResult:
        """
        Check if metric already exists.
        """
        return self.check_semantic_object_exists(metric_name, kind="metric")

    def end_semantic_model_generation(self, semantic_model_files: List[str]) -> FuncToolResult:
        """
        Complete semantic model generation process.

        Call this tool when you have finished generating semantic model YAML files.
        This tool triggers user confirmation workflow for syncing to LanceDB.

        Args:
            semantic_model_files: List of absolute paths to generated semantic model YAML files

        Returns:
            dict: Result containing confirmation message and semantic_model_files
        """
        try:
            logger.info(
                f"Semantic model generation completed for {len(semantic_model_files)} files: {semantic_model_files}"
            )

            return FuncToolResult(
                result={
                    "message": f"Semantic model generation completed for {len(semantic_model_files)} file(s)",
                    "semantic_model_files": semantic_model_files,
                }
            )

        except Exception as e:
            logger.error(f"Error completing semantic model generation: {e}")
            return FuncToolResult(success=0, error=f"Failed to complete generation: {str(e)}")

    def end_metric_generation(
        self, metric_file: str, semantic_model_file: str = "", metric_sqls_json: str = ""
    ) -> FuncToolResult:
        """
        Complete metric generation process.

        Call this tool when you have finished generating a metric YAML file.
        This tool triggers user confirmation workflow for syncing to LanceDB.

        Args:
            metric_file: Absolute path to the generated metric YAML file (required)
            semantic_model_file: Absolute path to the primary semantic model file that defines
                                 the measure(s) used by this metric. Optional - provide this
                                 if the semantic model was newly created or updated.
            metric_sqls_json: JSON string mapping metric names to their generated SQL (from query_metrics dry_run).
                              Example: '{"revenue_total": "SELECT SUM(revenue) FROM orders GROUP BY date"}'

        Returns:
            dict: Result containing confirmation message, file paths, and metric SQLs
        """
        import json

        try:
            # Parse JSON string to dict
            metric_sqls: Dict[str, str] = {}
            if metric_sqls_json:
                try:
                    metric_sqls = json.loads(metric_sqls_json)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse metric_sqls_json: {e}")

            logger.info(
                f"Metric generation completed: metric_file={metric_file}, "
                f"semantic_model_file={semantic_model_file}, "
                f"metric_sqls={metric_sqls}"
            )

            return FuncToolResult(
                result={
                    "message": "Metric generation completed",
                    "metric_file": metric_file,
                    "semantic_model_file": semantic_model_file,
                    "metric_sqls": metric_sqls,
                }
            )

        except Exception as e:
            logger.error(f"Error completing metric generation: {e}")
            return FuncToolResult(success=0, error=f"Failed to complete generation: {str(e)}")

    def generate_sql_summary_id(self, sql_query: str, comment: str = "") -> FuncToolResult:
        """
        Generate a unique ID for SQL summary based on SQL query and comment.
        """
        try:
            from datus.storage.reference_sql.init_utils import gen_reference_sql_id

            # Generate the ID using the same utility as the storage system
            generated_id = gen_reference_sql_id(sql_query)

            logger.info(f"Generated reference SQL ID: {generated_id}")
            return FuncToolResult(result=generated_id)

        except Exception as e:
            logger.error(f"Error generating reference SQL ID: {e}")
            return FuncToolResult(success=0, error=f"Failed to generate ID: {str(e)}")

    def generate_ext_knowledge_id(self, search_text: str, subject_path: str = "") -> FuncToolResult:
        """
        Generate a unique ID for external knowledge entry based on search_text and subject path.

        This tool helps create consistent, unique IDs for external knowledge entries.
        Use this tool when you need to generate an ID for a new knowledge entry.

        Args:
            search_text: The business search_text/concept that will be used to generate the ID
            subject_path: Optional subject path string (format: "domain/layer1/layer2")

        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure
                  - 'error' (Optional[str]): Error message on failure
                  - 'result' (str): The generated unique ID

        Example:
            result = generate_ext_knowledge_id(
                search_text="GMV",
                subject_path="Finance/Revenue/Metrics"
            )
        """
        try:
            from datus.storage.ext_knowledge.init_utils import gen_ext_knowledge_id

            # Parse subject_path string into list
            subject_path_list = []
            if subject_path:
                subject_path_list = [part.strip() for part in subject_path.split("/") if part.strip()]

            # Generate the ID using the same utility as the storage system
            generated_id = gen_ext_knowledge_id(subject_path_list, search_text)

            logger.info(f"Generated external knowledge ID: {generated_id}")
            return FuncToolResult(result=generated_id)

        except Exception as e:
            logger.error(f"Error generating external knowledge ID: {e}")
            return FuncToolResult(success=0, error=f"Failed to generate ID: {str(e)}")
