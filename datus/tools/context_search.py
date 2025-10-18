# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.storage.metric.store import SemanticMetricsRAG
from datus.storage.sql_history.store import SqlHistoryRAG
from datus.tools.tools import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ContextSearchTools:
    def __init__(self, agent_config: AgentConfig, sub_agent_name: Optional[str] = None):
        self.agent_config = agent_config
        self.sub_agent_name = sub_agent_name
        self.metric_rag = SemanticMetricsRAG(agent_config, sub_agent_name)
        self.sql_history_store = SqlHistoryRAG(agent_config, sub_agent_name)
        self.has_metrics = self.metric_rag.get_metrics_size() > 0
        self.has_historical_sql = self.sql_history_store.get_sql_history_size() > 0

    def available_tools(self) -> List[Tool]:
        tools = []
        if self.has_metrics:
            for tool in (self.list_domain_layers_tree, self.search_metrics):
                tools.append(trans_to_function_tool(tool))

        if self.has_historical_sql:
            if not self.has_metrics:
                tools.append(trans_to_function_tool(self.list_domain_layers_tree))
            tools.append(trans_to_function_tool(self.search_reference_sql))
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

        Use this tool to prime the agent with valid hierarchical filters before calling `search_metrics` or
        `search_reference_sql`. Counters represent how many records exist for each leaf sourced from metrics and SQL
        reference respectively; keys with missing counts simply have no entries from that store.
        """
        try:
            domain_tree: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = defaultdict(
                lambda: defaultdict(lambda: defaultdict(dict))
            )
            for metrics_item in self._collect_metrics_entries():
                layer_map = domain_tree[metrics_item["domain"]][metrics_item["layer1"]][metrics_item["layer2"]]
                layer_map["metrics_size"] = layer_map.get("metrics_size", 0) + 1

            for sql_item in self._collect_sql_entries():
                layer_map = domain_tree[sql_item["domain"]][sql_item["layer1"]][sql_item["layer2"]]
                layer_map["sql_size"] = layer_map.get("sql_size", 0) + 1

            serializable_tree = {
                domain: {
                    layer1: {layer2: dict(counts) for layer2, counts in layer2_map.items()}
                    for layer1, layer2_map in layer1_map.items()
                }
                for domain, layer1_map in domain_tree.items()
            }

            return FuncToolResult(result=serializable_tree)
        except ValueError as exc:
            return FuncToolResult(success=0, error=str(exc))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to assemble domain taxonomy: %s", exc)
            return FuncToolResult(success=0, error=str(exc))

    def _collect_metrics_entries(self) -> Sequence[Dict[str, Any]]:
        try:
            return self.metric_rag.search_all_metrics(select_fields=["domain", "layer1", "layer2", "name"])
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to collect metrics taxonomy: %s", exc)
            return []

    def _collect_sql_entries(self) -> Sequence[Dict[str, Any]]:
        try:
            return self.sql_history_store.search_all_sql_history(selected_fields=["domain", "layer1", "layer2", "name"])
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to collect SQL taxonomy: %s", exc)
            return []

    def _inject_taxonomy_entries(
        self,
        rows: Sequence[Dict[str, Any]],
        domains: Set[str],
        layer1_map: Dict[str, Set[str]],
        layer2_map: Dict[Tuple[str, str], Set[str]],
    ) -> None:
        for row in rows:
            domain = self._normalize_taxonomy_value(row.get("domain"))
            layer1 = self._normalize_taxonomy_value(row.get("layer1"))
            layer2 = self._normalize_taxonomy_value(row.get("layer2"))

            domain_key = domain
            if domain:
                domains.add(domain)
            else:
                domain_key = ""

            if layer1:
                layer1_map.setdefault(domain_key, set()).add(layer1)
            elif domain_key not in layer1_map:
                layer1_map[domain_key] = layer1_map.get(domain_key, set())

            if layer2:
                layer2_map.setdefault((domain_key, layer1 or ""), set()).add(layer2)
            elif (domain_key, layer1 or "") not in layer2_map:
                layer2_map[(domain_key, layer1 or "")] = layer2_map.get((domain_key, layer1 or ""), set())

    @staticmethod
    def _normalize_taxonomy_value(value: Optional[str]) -> str:
        if value is None:
            return ""
        return str(value).strip()

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
            return FuncToolResult(success=1, error=None, result=metrics)
        except Exception as e:
            logger.error(f"Failed to search metrics for table '{query_text}': {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def search_reference_sql(
        self, query_text: str, domain: str = "", layer1: str = "", layer2: str = "", top_n: int = 5
    ) -> FuncToolResult:
        """
        Perform a vector search to match reference SQL queries by intent.

        **Application Guidance**: If matches are found, MUST reuse the 'sql' directly if it aligns perfectly, or adjust
        minimally (e.g., change table names or add conditions). Avoid generating new SQL.
        Example: If reference SQL is "SELECT * FROM users WHERE active=1" for "active users", reuse or adjust to
        "SELECT * FROM users WHERE active=1 AND join_date > '2023'".

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
            result = self.sql_history_store.search_sql_history_by_summary(
                query_text=query_text, domain=domain, layer1=layer1, layer2=layer2, top_n=top_n
            )
            return FuncToolResult(success=1, error=None, result=result)
        except Exception as e:
            logger.error(f"Failed to search reference SQL for `{query_text}`: {str(e)}")
            return FuncToolResult(success=0, error=str(e))
