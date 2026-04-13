# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
import os
from typing import Dict, List, Optional

import yaml
from agents import Tool
from datus_storage_base.conditions import And, eq

from datus.configuration.agent_config import AgentConfig
from datus.storage.metric.store import MetricRAG
from datus.storage.semantic_model.store import SemanticModelRAG
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger
from datus.utils.path_manager import get_path_manager

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
                self.check_semantic_object_exists,
                self.generate_sql_summary_id,
                self.end_semantic_model_generation,
                self.end_metric_generation,
            )
        ]

    def check_semantic_object_exists(
        self,
        object_name: str,
        kind: str = "table",  # table, column, metric
        table_context: str = "",
    ) -> FuncToolResult:
        """
        Check if a semantic object (table, column, metric) already exists in vector store.

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
                where = And([eq("kind", "table"), eq("name", target_name)])
                results = storage.search_all(where=where, select_fields=["id", "name", "kind"])
                if results:
                    found_object = results[0]
            elif kind == "metric":
                # Exact match for metric using SQL WHERE condition
                storage = self.metric_rag.storage
                where = eq("name", target_name)
                results = storage.search_all(where=where, select_fields=["id", "name"])
                if results:
                    found_object = results[0]
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

    def end_semantic_model_generation(self, semantic_model_files: List[str]) -> FuncToolResult:
        """
        Complete semantic model generation process.

        Call this tool when you have finished generating semantic model YAML files.
        This tool triggers user confirmation workflow for syncing to vector store.

        Args:
            semantic_model_files: List of generated semantic model YAML file paths.
                Relative file names within the sub-agent's semantic-model workspace
                are preferred (e.g. ``["orders.yml", "customers.yml"]``). Absolute
                paths are also accepted. The downstream hook resolves relative
                entries against the live agent_config namespace.

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
        Complete metric generation process and automatically sync to Knowledge Base.

        Call this tool when you have finished generating a metric YAML file.
        This tool automatically syncs the metric to the vector store (no user confirmation needed).

        Args:
            metric_file: Path to the generated metric YAML file (required).
                Relative paths (e.g. ``"metrics/orders_metrics.yml"``) are preferred
                and resolved against the sub-agent's semantic-model workspace using
                the live ``agent_config.current_namespace``. Absolute paths are also
                accepted and used as-is.
            semantic_model_file: Path to the primary semantic model file that defines
                the measure(s) used by this metric. Optional — provide this if the
                semantic model was newly created or updated. Same relative/absolute
                rules as ``metric_file``.
            metric_sqls_json: JSON string mapping metric names to their generated SQL (from query_metrics dry_run).
                              Example: '{"revenue_total": "SELECT SUM(revenue) FROM orders GROUP BY date"}'

        Returns:
            dict: Result containing confirmation message, file paths, metric SQLs, and sync status
        """
        import json

        try:
            # Parse JSON string to dict
            metric_sqls: Dict[str, str] = {}
            if metric_sqls_json:
                try:
                    parsed = json.loads(metric_sqls_json)
                    if not isinstance(parsed, dict):
                        return FuncToolResult(success=0, error="metric_sqls_json must decode to a JSON object")
                    metric_sqls = parsed
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse metric_sqls_json: {e}")

            logger.info(
                f"Metric generation completed: metric_file={metric_file}, "
                f"semantic_model_file={semantic_model_file}, "
                f"metric_sqls={metric_sqls}"
            )

            # Resolve absolute paths — use agent_config so knowledge_home override is respected
            base_dir = str(
                get_path_manager(agent_config=self.agent_config).semantic_model_path(
                    self.agent_config.current_namespace
                )
            )
            abs_metric = os.path.join(base_dir, metric_file) if not os.path.isabs(metric_file) else metric_file
            abs_semantic = (
                os.path.join(base_dir, semantic_model_file)
                if semantic_model_file and not os.path.isabs(semantic_model_file)
                else semantic_model_file
            )

            # Auto-sync to Knowledge Base
            sync_result = self._sync_metric_to_db(abs_metric, abs_semantic, metric_sqls)

            if not sync_result.get("success"):
                return FuncToolResult(
                    success=0,
                    error=f"Metric file written but KB sync failed: {sync_result.get('error', 'unknown')}",
                    result={
                        "metric_file": metric_file,
                        "semantic_model_file": semantic_model_file,
                        "metric_sqls": metric_sqls,
                        "sync": sync_result,
                    },
                )

            return FuncToolResult(
                result={
                    "message": "Metric generation completed and synced to Knowledge Base",
                    "metric_file": metric_file,
                    "semantic_model_file": semantic_model_file,
                    "metric_sqls": metric_sqls,
                    "sync": sync_result,
                }
            )

        except Exception as e:
            logger.error(f"Error completing metric generation: {e}")
            return FuncToolResult(success=0, error=f"Failed to complete generation: {str(e)}")

    def _sync_metric_to_db(
        self,
        metric_file: str,
        semantic_model_file: Optional[str] = None,
        metric_sqls: Optional[Dict[str, str]] = None,
    ) -> dict:
        """
        Sync metric (and optionally semantic model) to Knowledge Base.

        Reuses GenerationHooks._sync_semantic_to_db() static method.

        Args:
            metric_file: Absolute path to metric YAML file
            semantic_model_file: Optional absolute path to semantic model YAML file
            metric_sqls: Optional dict mapping metric names to generated SQL

        Returns:
            dict with sync result (success, message, or error)
        """
        from datus.cli.generation_hooks import GenerationHooks

        try:
            if not os.path.exists(metric_file):
                return {"success": False, "error": f"Metric file not found: {metric_file}"}

            if semantic_model_file and os.path.exists(semantic_model_file):
                # Combine semantic model + metric into a temp file for sync
                with open(semantic_model_file, "r", encoding="utf-8") as f:
                    semantic_docs = list(yaml.safe_load_all(f))
                with open(metric_file, "r", encoding="utf-8") as f:
                    metric_docs = list(yaml.safe_load_all(f))

                combined_docs = semantic_docs + metric_docs
                import tempfile

                fd, temp_file = tempfile.mkstemp(
                    suffix=".combined.tmp",
                    dir=os.path.dirname(semantic_model_file),
                )
                try:
                    os.close(fd)
                    with open(temp_file, "w", encoding="utf-8") as f:
                        yaml.safe_dump_all(combined_docs, f, allow_unicode=True, sort_keys=False)

                    # First: sync semantic objects from the semantic model file
                    sem_result = GenerationHooks._sync_semantic_to_db(
                        semantic_model_file,
                        self.agent_config,
                        include_semantic_objects=True,
                        include_metrics=False,
                    )
                    if not sem_result.get("success"):
                        return sem_result
                    # Then: sync metrics from combined file
                    result = GenerationHooks._sync_semantic_to_db(
                        temp_file,
                        self.agent_config,
                        include_semantic_objects=False,
                        include_metrics=True,
                        metric_sqls=metric_sqls,
                        original_yaml_path=metric_file,
                    )
                finally:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            else:
                # Sync metric file alone
                result = GenerationHooks._sync_semantic_to_db(
                    metric_file,
                    self.agent_config,
                    metric_sqls=metric_sqls,
                )

            if result.get("success"):
                logger.info(f"Successfully synced metric to KB: {result.get('message')}")
            else:
                logger.error(f"Failed to sync metric to KB: {result.get('error')}")

            return result

        except Exception as e:
            logger.error(f"Error syncing metric to KB: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

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
