# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Literal, Optional

from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.schemas.agent_models import SubAgentConfig
from datus.storage.metric.store import MetricRAG
from datus.storage.reference_sql.store import ReferenceSqlRAG
from datus.storage.semantic_model.store import SemanticModelRAG
from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_NAME = "context_search_tools"
_NAME_METRICS = "context_search_tools.search_metrics"
_NAME_GET_METRICS = "context_search_tools.get_metrics"
_NAME_SQL = "context_search_tools.search_reference_sql"
_NAME_GET_SQL = "context_search_tools.get_reference_sql"
_NAME_SEMANTIC = "context_search_tools.search_semantic_objects"


class ContextSearchTools:
    def __init__(self, agent_config: AgentConfig, sub_agent_name: Optional[str] = None):
        self.agent_config = agent_config
        self.sub_agent_name = sub_agent_name
        self.metric_rag = MetricRAG(agent_config, sub_agent_name)
        self.semantic_rag = SemanticModelRAG(agent_config, sub_agent_name)
        self.reference_sql_store = ReferenceSqlRAG(agent_config, sub_agent_name)

        # Initialize SubjectTreeStore for domain hierarchy
        self.subject_tree = self.metric_rag.storage.subject_tree

        if sub_agent_name:
            self.sub_agent_config = SubAgentConfig.model_validate(self.agent_config.sub_agent_config(sub_agent_name))
        else:
            self.sub_agent_config = None
        self.has_metrics = self.metric_rag.get_metrics_size() > 0
        self.has_reference_sql = self.reference_sql_store.get_reference_sql_size() > 0
        self.has_semantic_objects = self.semantic_rag.get_size() > 0

    def _show_metrics(self):
        return self.has_metrics and (
            not self.sub_agent_config
            or _NAME in self.sub_agent_config.tool_list
            or _NAME_METRICS in self.sub_agent_config.tool_list
            or _NAME_GET_METRICS in self.sub_agent_config.tool_list
        )

    def _show_sql(self):
        return self.has_reference_sql and (
            not self.sub_agent_config
            or _NAME in self.sub_agent_config.tool_list
            or _NAME_SQL in self.sub_agent_config.tool_list
            or _NAME_GET_SQL in self.sub_agent_config.tool_list
        )

    def _show_semantic_objects(self):
        return self.has_semantic_objects and (
            not self.sub_agent_config
            or _NAME in self.sub_agent_config.tool_list
            or _NAME_SEMANTIC in self.sub_agent_config.tool_list
        )

    @staticmethod
    def all_tools_name() -> List[str]:
        from datus.utils.class_utils import get_public_instance_methods

        result = []
        for name in get_public_instance_methods(ContextSearchTools).keys():
            if name == "available_tools":
                continue
            result.append(name)
        return result

    def available_tools(self) -> List[Tool]:
        tools = []
        if self.has_metrics:
            for tool in (self.list_subject_tree, self.search_metrics, self.get_metrics):
                tools.append(trans_to_function_tool(tool))

        if self.has_reference_sql:
            if not self.has_metrics:
                tools.append(trans_to_function_tool(self.list_subject_tree))
            tools.append(trans_to_function_tool(self.search_reference_sql))
            tools.append(trans_to_function_tool(self.get_reference_sql))

        if self._show_semantic_objects():
            tools.append(trans_to_function_tool(self.search_semantic_objects))

        return tools

    def list_subject_tree(self) -> FuncToolResult:
        """
        Get the domain-layer taxonomy from subject_tree store with metrics and SQL counts.

        The response has the structure:
        {
            "<domain>": {
                "<layer1>": {
                    "<layer2>": {
                        "metrics": <[name1, name2, ...], optional>,
                        "reference_sql": <[name1, name2, ...], optional>
                    },
                    ...
                },
                ...
            },
            ...
        }
        """
        try:
            # Collect entries from the new subject-path index (decoupled from the metric/sql payload tables).
            metrics_entries = self._collect_metrics_entries()
            sql_entries = self._collect_sql_entries()
            enriched_tree = {}

            _fill_subject_tree(enriched_tree, metrics_entries, "metrics")
            _fill_subject_tree(enriched_tree, sql_entries, "reference_sql")

            _normalize_subject_tree(enriched_tree)

            logger.debug("enriched_tree: %s", enriched_tree)
            return FuncToolResult(result=enriched_tree)
        except ValueError as exc:
            return FuncToolResult(success=0, error=str(exc))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                f"Failed to assemble domain taxonomy: {exc}",
            )
            return FuncToolResult(success=0, error=str(exc))

    def _collect_metrics_entries(self) -> List[Dict[str, Any]]:
        if not self._show_metrics():
            return []
        try:
            return self.metric_rag.search_all_metrics(select_fields=["name"])
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to collect metrics taxonomy: %s", exc)
            return []

    def _collect_sql_entries(self) -> List[Dict[str, Any]]:
        if not self._show_sql():
            return []
        try:
            return self.reference_sql_store.search_all_reference_sql(select_fields=["name"])
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
            logger.error(f"Failed to search metrics for '{query_text}': {e}")
            return FuncToolResult(success=0, error=str(e))

    def get_metrics(self, subject_path: List[str], name: str = "") -> FuncToolResult:
        """
        Search for business metrics and KPIs using natural language queries.

        Args:
            subject_path: Optional subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1'])
            name: The name of the metric

        Returns:
            FuncToolResult with metrics containing name, description, constraint, and sql_query
        """
        try:
            metrics = self.metric_rag.get_metrics_detail(
                subject_path=subject_path,
                name=name,
            )
            logger.debug(f"result of search_metrics: {metrics}")
            if metrics:
                return FuncToolResult(success=1, error=None, result=metrics[0])
            else:
                return FuncToolResult(success=0, error="No matched result", result=None)
        except Exception as e:
            logger.error(f"Failed to get metrics details for `{'/'.join(subject_path)}/{name}`: {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def search_reference_sql(
        self, query_text: str, subject_path: Optional[List[str]] = None, top_n: int = 5
    ) -> FuncToolResult:
        """
        Search for reference SQL queries using natural language queries.
        MUST call `list_subject_tree` first to get the subject path.

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

    def get_reference_sql(self, subject_path: List[str], name: str = "") -> FuncToolResult:
        """
        Get reference SQL query for a domain and layer combination.

        Args:
            subject_path: Optional subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1'])
            name: The name of the reference SQL intent.

        Returns:
            dict: A dictionary with keys:
                - 'success' (int): 1 if the search succeeded, 0 otherwise.
                - 'error' (str or None): Error message if any.
                - 'result' (dict): On success, a list of matching entries, each containing:
                    - 'sql'
                    - 'comment'
                    - 'tags'
                    - 'summary'
                    - 'file_path'
        """
        try:
            result = self.reference_sql_store.get_reference_sql_detail(subject_path=subject_path, name=name)
            if len(result) > 0:
                return FuncToolResult(success=1, error=None, result=result[0])
            return FuncToolResult(success=0, error="No matched result", result=None)
        except Exception as e:
            logger.error(f"Failed to get reference SQL for `{'/'.join(subject_path)}/{name}`: {e}")
            return FuncToolResult(success=0, error=str(e))

    def search_semantic_objects(
        self,
        query_text: str,
        kinds: Optional[List[str]] = None,
        top_n: int = 5,
    ) -> FuncToolResult:
        """
        Search for semantic objects (metrics, columns, tables) using unified storage.

        Args:
            query_text: Natural language query describing what you're looking for
            kinds: List of object kinds to filter by. Options: ["metric", "column", "table", "entity"]
                   If None, searches all kinds
            top_n: Maximum number of results to return (default 5)

        Returns:
            FuncToolResult with list of matching objects containing:
                - kind: Type of object ("metric", "column", "table", "entity")
                - name: Object name
                - description: Detailed description
                - _distance: Similarity score (lower is better)
                - Additional fields specific to object kind (e.g., available_dimensions for metrics)
        """
        try:
            results = self.semantic_rag.storage.search_objects(
                query_text=query_text,
                kinds=kinds,
                top_n=top_n,
            )

            logger.debug(f"search_semantic_objects results: {results}")
            return FuncToolResult(success=1, error=None, result=results)
        except Exception as e:
            logger.error(f"Failed to search semantic objects for '{query_text}': {str(e)}")
            return FuncToolResult(success=0, error=str(e))


def _fill_subject_tree(
    enriched_tree: Dict[str, Any], entries: List[Dict[str, Any]], entry_type: Literal["metrics", "reference_sql"]
):
    for item in entries:
        subject_path = item["subject_path"]
        (domain, layer1, layer2) = subject_path[:3]
        leaf = enriched_tree.setdefault(domain, {}).setdefault(layer1, {}).setdefault(layer2, {})
        leaf.setdefault(entry_type, set()).add(item["name"])


def _normalize_subject_tree(enriched_tree: Dict[str, Any]) -> Dict[str, Any]:
    for _, layer1_map in enriched_tree.items():
        for _, layer2_map in layer1_map.items():
            for _, leaf in layer2_map.items():
                for k in ("metrics", "reference_sql"):
                    v = leaf.get(k)
                    if isinstance(v, set):
                        leaf[k] = sorted(v)
    return enriched_tree
