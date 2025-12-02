# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
from collections import defaultdict
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
            for tool in (self.list_domain_layers_tree, self.search_metrics, self.list_metrics, self.get_metric):
                tools.append(trans_to_function_tool(tool))

        if self.has_reference_sql:
            if not self.has_metrics:
                tools.append(trans_to_function_tool(self.list_domain_layers_tree))
            for tool in (self.search_reference_sql, self.list_reference_sqls, self.get_reference_sql):
                tools.append(trans_to_function_tool(tool))
        return tools

    def list_domain_layers_tree(self) -> FuncToolResult:
        """
        Aggregate the available domain-layer taxonomy across metrics and reference SQL.

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
            domain_tree: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = defaultdict(
                lambda: defaultdict(lambda: defaultdict(dict))
            )
            for metrics_item in self._collect_metrics_entries():
                ContextSearchTools._fill_in_domain_layer_tree(domain_tree, metrics_item, "metrics_size")

            for sql_item in self._collect_sql_entries():
                ContextSearchTools._fill_in_domain_layer_tree(domain_tree, sql_item, "sql_size")

            serializable_tree = {
                domain: {
                    layer1: {layer2: dict(counts) for layer2, counts in layer2_map.items()}
                    for layer1, layer2_map in layer1_map.items()
                }
                for domain, layer1_map in domain_tree.items()
            }
            logger.debug(f"serializable_tree: {serializable_tree}")

            return FuncToolResult(result=serializable_tree)
        except ValueError as exc:
            return FuncToolResult(success=0, error=str(exc))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to assemble domain taxonomy: %s", exc)
            return FuncToolResult(success=0, error=str(exc))

    @classmethod
    def _fill_in_domain_layer_tree(
        cls, domain_tree: Dict[str, Dict[str, Dict[str, Dict[str, int]]]], item: Dict[str, Any], item_type: str
    ):
        domain = (item.get("domain") or "").strip()
        layer1 = (item.get("layer1") or "").strip()
        layer2 = (item.get("layer2") or "").strip()
        layer_map = domain_tree[domain][layer1][layer2]
        layer_map[item_type] = layer_map.get(item_type, 0) + 1

    def _collect_metrics_entries(self) -> Sequence[Dict[str, Any]]:
        if not self._show_metrics():
            return []
        try:
            return self.metric_rag.search_all_metrics(select_fields=["domain", "layer1", "layer2", "name"])
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to collect metrics taxonomy: %s", exc)
            return []

    def _collect_sql_entries(self) -> Sequence[Dict[str, Any]]:
        if not self._show_sql():
            return []
        try:
            return self.reference_sql_store.search_all_reference_sql(
                selected_fields=["domain", "layer1", "layer2", "name"]
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to collect SQL taxonomy: %s", exc)
            return []

    def search_metrics(
        self,
        query_text: str,
        domain: str = "",
        layer1: str = "",
        layer2: str = "",
        top_n: int = 5,
    ) -> FuncToolResult:
        """
        Search for business metrics and KPIs using natural language queries.

        Args:
            query_text: Natural language description of the metric (e.g., "revenue metrics", "conversion rates")
            domain: Optional business domain filter derived from list_domain_layers_tree
            layer1: Optional first-layer subject filter derived from list_domain_layers_tree
            layer2: Optional second-layer subject filter derived from list_domain_layers_tree
            top_n: Maximum number of results to return (default 5)

        Returns:
            FuncToolResult with list of matching metrics containing name, description, constraint, and sql_query
        """
        try:
            metrics = self.metric_rag.search_metrics(
                query_text=query_text,
                domain=domain,
                layer1=layer1,
                layer2=layer2,
                top_n=top_n,
            )
            logger.debug(f"result of search_metrics: {metrics}")
            return FuncToolResult(success=1, error=None, result=metrics)
        except Exception as e:
            logger.error(f"Failed to search metrics for table '{query_text}': {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def search_reference_sql(
        self, query_text: str, domain: str = "", layer1: str = "", layer2: str = "", top_n: int = 5
    ) -> FuncToolResult:
        """
        Search for reference SQL queries using natural language queries.

        Args:
            query_text: The natural language query text representing the desired SQL intent.
            domain: Domain name for the reference SQL intent. Leave empty if not specified in context.
            layer1: Semantic Layer1 for the reference SQL intent. Leave empty if not specified in context.
            layer2: Semantic Layer2 for the reference SQL intent. Leave empty if not specified in context.
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
            result = self.reference_sql_store.search_reference_sql_by_summary(
                query_text=query_text, domain=domain, layer1=layer1, layer2=layer2, top_n=top_n
            )
            return FuncToolResult(success=1, error=None, result=result)
        except Exception as e:
            logger.error(f"Failed to search reference SQL for `{query_text}`: {e}")
            return FuncToolResult(success=0, error=str(e))

    def list_metrics(self, domain: str = "", layer1: str = "", layer2: str = "") -> FuncToolResult:
        """
        List all metrics in the taxonomy tree structure with metric details at leaf nodes.

        Args:
            domain: Optional business domain filter
            layer1: Optional first-layer subject filter
            layer2: Optional second-layer subject filter

        Returns:
            FuncToolResult with tree structure where leaf nodes contain metric lists instead of counts.
            Structure:
            {
                "<domain>": {
                    "<layer1>": {
                        "<layer2>": [
                            {"name": "metric_name", "llm_text": "...", ...},
                            ...
                        ]
                    }
                }
            }
        """
        try:
            metrics = self.metric_rag.search_all_metrics(
                select_fields=["domain", "layer1", "layer2", "name", "llm_text"]
            )

            # Filter by domain/layer if specified
            if domain or layer1 or layer2:
                filtered_metrics = []
                for m in metrics:
                    if domain and (m.get("domain") or "").strip() != domain:
                        continue
                    if layer1 and (m.get("layer1") or "").strip() != layer1:
                        continue
                    if layer2 and (m.get("layer2") or "").strip() != layer2:
                        continue
                    filtered_metrics.append(m)
                metrics = filtered_metrics

            # Build tree structure with metric lists at leaf nodes
            tree: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = defaultdict(
                lambda: defaultdict(lambda: defaultdict(list))
            )

            for metric in metrics:
                d = (metric.get("domain") or "").strip()
                l1 = (metric.get("layer1") or "").strip()
                l2 = (metric.get("layer2") or "").strip()
                tree[d][l1][l2].append(metric)

            # Convert to regular dict for serialization
            result = {
                domain: {
                    layer1: {layer2: metrics_list for layer2, metrics_list in layer2_map.items()}
                    for layer1, layer2_map in layer1_map.items()
                }
                for domain, layer1_map in tree.items()
            }

            return FuncToolResult(result=result)
        except Exception as e:
            logger.error(f"Failed to list metrics: {e}")
            return FuncToolResult(success=0, error=str(e))

    def get_metric(self, name: str) -> FuncToolResult:
        """
        Get detailed information for a specific metric by name.

        Args:
            name: The metric name to retrieve

        Returns:
            FuncToolResult with the metric details including name, description, constraint, sql_query, etc.
        """
        try:
            metric = self.metric_rag.get_metric_by_name(name)
            if metric is None:
                return FuncToolResult(success=0, error=f"Metric '{name}' not found")
            return FuncToolResult(result=metric)
        except Exception as e:
            logger.error(f"Failed to get metric '{name}': {e}")
            return FuncToolResult(success=0, error=str(e))

    def list_reference_sqls(self, domain: str = "", layer1: str = "", layer2: str = "") -> FuncToolResult:
        """
        List all reference SQL queries in the taxonomy tree structure with SQL details at leaf nodes.

        Args:
            domain: Optional business domain filter
            layer1: Optional first-layer subject filter
            layer2: Optional second-layer subject filter

        Returns:
            FuncToolResult with tree structure where leaf nodes contain reference SQL lists instead of counts.
            Structure:
            {
                "<domain>": {
                    "<layer1>": {
                        "<layer2>": [
                            {"name": "sql_name", "sql": "...", "summary": "...", ...},
                            ...
                        ]
                    }
                }
            }
        """
        try:
            sqls = self.reference_sql_store.search_all_reference_sql(
                selected_fields=["domain", "layer1", "layer2", "name", "summary"]
            )

            # Filter by domain/layer if specified
            if domain or layer1 or layer2:
                filtered_sqls = []
                for s in sqls:
                    if domain and (s.get("domain") or "").strip() != domain:
                        continue
                    if layer1 and (s.get("layer1") or "").strip() != layer1:
                        continue
                    if layer2 and (s.get("layer2") or "").strip() != layer2:
                        continue
                    filtered_sqls.append(s)
                sqls = filtered_sqls

            # Build tree structure with SQL lists at leaf nodes
            tree: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = defaultdict(
                lambda: defaultdict(lambda: defaultdict(list))
            )

            for sql in sqls:
                d = (sql.get("domain") or "").strip()
                l1 = (sql.get("layer1") or "").strip()
                l2 = (sql.get("layer2") or "").strip()
                tree[d][l1][l2].append(sql)

            # Convert to regular dict for serialization
            result = {
                domain: {
                    layer1: {layer2: sql_list for layer2, sql_list in layer2_map.items()}
                    for layer1, layer2_map in layer1_map.items()
                }
                for domain, layer1_map in tree.items()
            }

            return FuncToolResult(result=result)
        except Exception as e:
            logger.error(f"Failed to list reference SQLs: {e}")
            return FuncToolResult(success=0, error=str(e))

    def get_reference_sql(self, name: str) -> FuncToolResult:
        """
        Get detailed information for a specific reference SQL by name.

        Args:
            name: The reference SQL name to retrieve

        Returns:
            FuncToolResult with the SQL details including name, sql, comment, tags, summary, file_path, etc.
        """
        try:
            sql = self.reference_sql_store.get_reference_sql_by_name(name)
            if sql is None:
                return FuncToolResult(success=0, error=f"Reference SQL '{name}' not found")
            return FuncToolResult(result=sql)
        except Exception as e:
            logger.error(f"Failed to get reference SQL '{name}': {e}")
            return FuncToolResult(success=0, error=str(e))
