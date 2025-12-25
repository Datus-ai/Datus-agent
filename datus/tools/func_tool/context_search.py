# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Sequence

from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.schemas.agent_models import SubAgentConfig
from datus.storage.metric.store import SemanticMetricsRAG
from datus.storage.reference_sql.store import ReferenceSqlRAG
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_NAME = "context_search_tools"
_NAME_METRICS = "context_search_tools.search_metrics"
_NAME_SQL = "context_search_tools.search_reference_sql"


class ContextSearchTools:
    def __init__(self, agent_config: AgentConfig, sub_agent_name: Optional[str] = None):
        self.agent_config = agent_config
        self.sub_agent_name = sub_agent_name
        self.metric_rag = SemanticMetricsRAG(agent_config, sub_agent_name)
        self.reference_sql_store = ReferenceSqlRAG(agent_config, sub_agent_name)

        # Initialize SubjectTreeStore for domain hierarchy
        self.subject_tree = self.metric_rag.metric_storage.subject_tree

        if sub_agent_name:
            self.sub_agent_config = SubAgentConfig.model_validate(self.agent_config.sub_agent_config(sub_agent_name))
        else:
            self.sub_agent_config = None
        self.has_metrics = self.metric_rag.get_metrics_size() > 0
        self.has_reference_sql = self.reference_sql_store.get_reference_sql_size() > 0

    def _show_metrics(self):
        return self.has_metrics and (
            not self.sub_agent_config
            or _NAME in self.sub_agent_config.tool_list
            or _NAME_METRICS in self.sub_agent_config.tool_list
        )

    def _show_sql(self):
        return self.has_reference_sql and (
            not self.sub_agent_config
            or _NAME in self.sub_agent_config.tool_list
            or _NAME_SQL in self.sub_agent_config.tool_list
        )

    def available_tools(self) -> List[Tool]:
        tools = []
        if self.has_metrics:
            for tool in (self.list_subject_tree, self.search_metrics):
                tools.append(trans_to_function_tool(tool))

        if self.has_reference_sql:
            if not self.has_metrics:
                tools.append(trans_to_function_tool(self.list_subject_tree))
            tools.append(trans_to_function_tool(self.search_reference_sql))
        return tools

    def list_subject_tree(self) -> FuncToolResult:
        """
        Get the domain-layer taxonomy from subject_tree store with metrics and SQL counts.

        The response has the structure:
        {
            "<domain>": {
                "<layer1>": {
                    "<layer2>": {
                        "metrics_size": <int, optional>,
                        "sql_size": <int, optional>
                    },
                    ...
                },
                ...
            },
            ...
        }
        """
        try:
            # Get tree structure with node metadata
            tree_structure = self.subject_tree.get_tree_structure()

            # Enrich tree with metrics and SQL counts
            enriched_tree = self._enrich_tree_with_counts(tree_structure)

            logger.debug(f"enriched_tree: {enriched_tree}")

            return FuncToolResult(result=enriched_tree)
        except ValueError as exc:
            return FuncToolResult(success=0, error=str(exc))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to assemble domain taxonomy: %s", exc)
            return FuncToolResult(success=0, error=str(exc))

    def _enrich_tree_with_counts(self, tree_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively enrich tree structure with metrics and SQL counts.

        Args:
            tree_structure: Tree structure from subject_tree.get_tree_structure()

        Returns:
            Enriched tree with metrics_size and sql_size at leaf nodes
        """
        result = {}

        for name, node_info in tree_structure.items():
            node_id = node_info.get("node_id")
            children = node_info.get("children", {})

            if children:
                # Has children - recursively process
                result[name] = self._enrich_tree_with_counts(children)
            else:
                # Leaf node - add counts
                leaf_data = {}

                if node_id:
                    # Count metrics for this node
                    if self._show_metrics():
                        try:
                            from datus.storage.lancedb_conditions import build_where, eq
                            from datus.storage.subject_tree.store import SUBJECT_ID_COLUMN_NAME

                            metrics_storage = self.metric_rag.metric_storage
                            if hasattr(metrics_storage, "table") and metrics_storage.table:
                                where_clause = build_where(eq(SUBJECT_ID_COLUMN_NAME, node_id))
                                metrics_count = metrics_storage.table.count_rows(where_clause)
                                if metrics_count > 0:
                                    leaf_data["metrics_size"] = metrics_count
                        except Exception as e:
                            logger.warning(f"Failed to count metrics for node {node_id}: {e}")

                    # Count SQL for this node
                    if self._show_sql():
                        try:
                            from datus.storage.lancedb_conditions import build_where, eq
                            from datus.storage.subject_tree.store import SUBJECT_ID_COLUMN_NAME

                            sql_storage = self.reference_sql_store.reference_sql_storage
                            if hasattr(sql_storage, "table") and sql_storage.table:
                                where_clause = build_where(eq(SUBJECT_ID_COLUMN_NAME, node_id))
                                sql_count = sql_storage.table.count_rows(where_clause)
                                if sql_count > 0:
                                    leaf_data["sql_size"] = sql_count
                        except Exception as e:
                            logger.warning(f"Failed to count SQL for node {node_id}: {e}")

                result[name] = leaf_data

        return result

    def _collect_metrics_entries(self) -> Sequence[Dict[str, Any]]:
        if not self._show_metrics():
            return []
        try:
            return self.metric_rag.search_all_metrics()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to collect metrics taxonomy: %s", exc)
            return []

    def _collect_sql_entries(self) -> Sequence[Dict[str, Any]]:
        if not self._show_sql():
            return []
        try:
            return self.reference_sql_store.search_all_reference_sql()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to collect SQL taxonomy: %s", exc)
            return []

    def search_metrics(
        self,
        query_text: str,
        subject_path: Optional[List[str]] = None,
        top_n: int = 5,
    ) -> FuncToolResult:
        """
        Search for business metrics and KPIs using natural language queries.

        Args:
            query_text: Natural language description of the metric (e.g., "revenue metrics", "conversion rates")
            subject_path: Optional subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1'])
            top_n: Maximum number of results to return (default 5)

        Returns:
            FuncToolResult with list of matching metrics containing name, description, constraint, and sql_query
        """
        try:
            metrics = self.metric_rag.search_metrics(
                query_text=query_text,
                subject_path=subject_path,
                top_n=top_n,
            )
            logger.debug(f"result of search_metrics: {metrics}")
            return FuncToolResult(success=1, error=None, result=metrics)
        except Exception as e:
            logger.error(f"Failed to search metrics for table '{query_text}': {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def search_reference_sql(
        self, query_text: str, subject_path: Optional[List[str]] = None, top_n: int = 5
    ) -> FuncToolResult:
        """
        Search for reference SQL queries using natural language queries.

        Args:
            query_text: The natural language query text representing the desired SQL intent.
            subject_path: Optional subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1'])
            top_n: The number of top results to return (default 5).

        Returns:
            dict: A dictionary with keys:
                - 'success' (int): 1 if the search succeeded, 0 otherwise.
                - 'error' (str or None): Error message if any.
                - 'result' (list): On success, a list of matching entries, each containing:
                    - 'sql'
                    - 'comment'
                    - 'tags'
                    - 'summary'
                    - 'file_path'
        """
        try:
            result = self.reference_sql_store.search_reference_sql(
                query_text=query_text,
                subject_path=subject_path,
                top_n=top_n,
                selected_fields=["name", "sql", "comment", "summary", "tags"],
            )
            return FuncToolResult(success=1, error=None, result=result)
        except Exception as e:
            logger.error(f"Failed to search reference SQL for `{query_text}`: {e}")
            return FuncToolResult(success=0, error=str(e))
