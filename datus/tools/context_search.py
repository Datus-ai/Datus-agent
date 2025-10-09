# -*- coding: utf-8 -*-
from typing import List

from agents import Tool

from datus.configuration.agent_config import AgentConfig
from datus.storage.document import DocumentStore
from datus.storage.ext_knowledge.store import rag_by_configuration as ext_knowledge_by_configuration
from datus.storage.metric.store import rag_by_configuration as metrics_rag_by_configuration
from datus.storage.schema_metadata.store import rag_by_configuration as schema_metadata_by_configuration
from datus.storage.sql_history.store import sql_history_rag_by_configuration
from datus.tools.tools import FuncToolResult, trans_to_function_tool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ContextSearchTools:
    def __init__(self, agent_config: AgentConfig):
        self.schema_rag = schema_metadata_by_configuration(agent_config)
        self.metric_rag = metrics_rag_by_configuration(agent_config)
        self.doc_rag = DocumentStore(agent_config.rag_storage_path())
        self.ext_knowledge_rag = ext_knowledge_by_configuration(agent_config)
        self.sql_history_store = sql_history_rag_by_configuration(agent_config)

    def available_tools(self) -> List[Tool]:
        return [
            trans_to_function_tool(func)
            for func in (
                self.list_domains,
                self.list_layers_by_domain,
                self.list_items,
                self.get_metrics,
                self.get_reference_sql,
                self.search_metrics,
                self.search_reference_sql,
                # Temporarily disabled
                # self.search_documents,
                # self.search_external_knowledge,
            )
        ]

    def search_metrics(
        self,
        query_text: str,
        database_name: str = "",
        schema_name: str = "",
        top_n=5,
    ) -> FuncToolResult:
        """
        Search for business metrics and KPIs using natural language queries.

        Args:
            query_text: Natural language description of the metric (e.g., "revenue metrics", "conversion rates")
            database_name: Optional database name to filter metrics
            schema_name: Optional schema name to filter metrics
            top_n: Maximum number of results to return (default 5)

        Returns:
            FuncToolResult with list of matching metrics containing name, description, constraint, and sql_query
        """
        try:
            metrics = self.metric_rag.search_hybrid_metrics(
                query_text=query_text,
                database_name=database_name,
                schema_name=schema_name,
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
        Perform a vector search to match historical SQL queries by intent.

        **Application Guidance**: If matches are found, MUST reuse the 'sql' directly if it aligns perfectly, or adjust
        minimally (e.g., change table names or add conditions). Avoid generating new SQL.
        Example: If historical SQL is "SELECT * FROM users WHERE active=1" for "active users", reuse or adjust to
        "SELECT * FROM users WHERE active=1 AND join_date > '2023'".

        Args:
            query_text: The natural language query text representing the desired SQL intent.
            domain: Domain name for the historical SQL intent. Leave empty if not specified in context.
            layer1: Semantic Layer1 for the historical SQL intent. Leave empty if not specified in context.
            layer2: Semantic Layer2 for the historical SQL intent. Leave empty if not specified in context.
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
            logger.error(f"Failed to search historical SQL for `{query_text}`: {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def list_domains(self) -> FuncToolResult:
        """
        List all business domains available in metrics and SQL history.

        Use this tool when you need to:
        - Discover what business areas are covered in the knowledge base
        - Start exploring the business layer hierarchy
        - Understand the organizational structure of metrics and SQL queries

        Returns:
            dict: Domain list containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if failed
                - 'result' (list): List of unique domain names (sorted)

        Example:
            result: ["sales", "marketing", "finance", "operations"]
        """
        try:
            # Get domains from both metrics and sql_history
            metrics_domains = set(self.metric_rag.get_domains())
            sql_history_domains = set(self.sql_history_store.get_domains())

            # Combine and sort
            all_domains = sorted(metrics_domains | sql_history_domains)

            logger.debug(f"Found {len(all_domains)} unique domains")
            return FuncToolResult(success=1, error=None, result=all_domains)
        except Exception as e:
            logger.error(f"Failed to list domains: {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def list_layers_by_domain(self, domain: str = "") -> FuncToolResult:
        """
        List all layer1/layer2 combinations for a specific domain.

        Shows which semantic layers contain metrics or SQL history items,
        helping you navigate the business hierarchy.

        Use this tool when you need to:
        - Explore subcategories within a business domain
        - Find which layers have metrics or historical SQL
        - Navigate the semantic layer structure

        Args:
            domain: Domain name to filter. Leave empty to get all layers across domains.

        Returns:
            dict: Layer list containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if failed
                - 'result' (list): List of dicts with:
                    - 'layer1': Primary layer name
                    - 'layer2': Secondary layer name
                    - 'has_metrics': Boolean, True if this layer has metrics
                    - 'has_sql_history': Boolean, True if this layer has SQL history

        Example:
            domain="sales" returns:
            [
                {"layer1": "revenue", "layer2": "monthly", "has_metrics": true, "has_sql_history": true},
                {"layer1": "revenue", "layer2": "daily", "has_metrics": true, "has_sql_history": false}
            ]

        Note:
            Returns up to 1000 unique layer combinations. For large datasets,
            specify a domain filter to narrow the results.
        """
        try:
            layers_map = {}  # {(layer1, layer2): {"has_metrics": bool, "has_sql_history": bool}}

            # Get layers from metrics
            metrics_layers = self.metric_rag.get_layers_by_domain(domain)
            for layer1, layer2 in metrics_layers:
                key = (layer1, layer2)
                if key not in layers_map:
                    layers_map[key] = {"has_metrics": False, "has_sql_history": False}
                layers_map[key]["has_metrics"] = True

            # Get layers from sql_history
            sql_history_layers = self.sql_history_store.get_layers_by_domain(domain)
            for layer1, layer2 in sql_history_layers:
                key = (layer1, layer2)
                if key not in layers_map:
                    layers_map[key] = {"has_metrics": False, "has_sql_history": False}
                layers_map[key]["has_sql_history"] = True

            # Format result
            result = [
                {"layer1": layer1, "layer2": layer2, **flags} for (layer1, layer2), flags in sorted(layers_map.items())
            ]

            logger.debug(f"Found {len(result)} unique layers for domain='{domain}'")
            return FuncToolResult(success=1, error=None, result=result)
        except Exception as e:
            logger.error(f"Failed to list layers for domain '{domain}': {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def list_items(self, domain: str, layer1: str, layer2: str, item_type: str) -> FuncToolResult:
        """
        List metric or SQL history names within a specific business layer.

        This is a lightweight navigation tool that shows what items are available
        in a layer without fetching full details.

        Use this tool when you need to:
        - Browse available metrics or SQL history in a specific layer
        - Find items by name before fetching full details
        - Explore what assets exist in a business category

        Args:
            domain: Domain name (e.g., "sales", "marketing")
            layer1: Primary layer name (e.g., "revenue", "customer")
            layer2: Secondary layer name (e.g., "monthly", "daily")
            item_type: Type of items to list - must be either "metrics" or "sql_history"

        Returns:
            dict: Item list containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if failed
                - 'result' (list): For metrics: [{"name": str, "description": str}]
                                   For sql_history: [{"name": str, "summary": str}]

        Example:
            list_items("sales", "revenue", "monthly", "metrics") returns:
            [
                {"name": "total_revenue", "description": "Sum of all sales"},
                {"name": "avg_order_value", "description": "Average value per order"}
            ]

        Note:
            Returns up to 1000 items per layer. If you need full details including
            SQL queries, use get_metrics() or get_reference_sql() instead.
        """
        try:
            if item_type not in ("metrics", "sql_history"):
                return FuncToolResult(
                    success=0, error=f"Invalid item_type: {item_type}. Must be 'metrics' or 'sql_history'"
                )

            if item_type == "metrics":
                result = self.metric_rag.get_metrics_names(domain, layer1, layer2)
            else:  # sql_history
                result = self.sql_history_store.get_sql_history_names(domain, layer1, layer2)

            logger.debug(f"Found {len(result)} {item_type} items in {domain}/{layer1}/{layer2}")
            return FuncToolResult(success=1, error=None, result=result)
        except Exception as e:
            logger.error(f"Failed to list {item_type} for {domain}/{layer1}/{layer2}: {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def get_metrics(self, domain: str, layer1: str, layer2: str, name: str) -> FuncToolResult:
        """
        Get complete definition of a specific metric.

        Retrieves the full metric definition including description and SQL query,
        ready for use in generating SQL statements.

        Use this tool when you need to:
        - Get the SQL query for a specific metric
        - Understand how a metric is calculated
        - Reuse existing metric logic in your SQL generation

        **Application Guidance**: If results are found, MUST prioritize reusing the 'sql_query'
        directly or with minimal adjustments (e.g., add date filters, change table names).

        Args:
            domain: Domain name
            layer1: Primary layer name
            layer2: Secondary layer name
            name: Metric name

        Returns:
            dict: Metric details containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if failed
                - 'result' (list): List with single metric dict containing:
                    - 'name': Metric name
                    - 'description': Metric description
                    - 'sql_query': SQL query to calculate the metric

        Example:
            [{"name": "total_revenue",
              "description": "Sum of all sales",
              "sql_query": "SELECT SUM(amount) FROM orders WHERE date > '2020'"}]
        """
        try:
            result = self.metric_rag.get_metrics_detail(domain, layer1, layer2, name)

            if not result:
                return FuncToolResult(success=0, error=f"Metric not found: {domain}/{layer1}/{layer2}/{name}")

            # Simplify output - only return essential fields
            simplified_result = [
                {"name": item.get("name"), "description": item.get("description"), "sql_query": item.get("sql_query")}
                for item in result
            ]

            logger.debug(f"Retrieved metric: {domain}/{layer1}/{layer2}/{name}")
            return FuncToolResult(success=1, error=None, result=simplified_result)
        except Exception as e:
            logger.error(f"Failed to get metric {domain}/{layer1}/{layer2}/{name}: {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def get_reference_sql(self, domain: str, layer1: str, layer2: str, name: str) -> FuncToolResult:
        """
        Get complete details of a specific historical SQL query.

        Retrieves the full SQL history entry including the SQL statement,
        comments, and summary, ready for reuse or adaptation.

        Use this tool when you need to:
        - Get the SQL code for a historical query
        - Understand what a previous query does
        - Reuse or adapt existing SQL logic

        **Application Guidance**: If matches are found, MUST reuse the 'sql' directly if it
        aligns perfectly, or adjust minimally (e.g., change table names, add conditions).
        Avoid generating new SQL from scratch when historical queries exist.

        Args:
            domain: Domain name
            layer1: Primary layer name
            layer2: Secondary layer name
            name: SQL history item name

        Returns:
            dict: SQL history details containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if failed
                - 'result' (list): List with single SQL history dict containing:
                    - 'name': SQL history name
                    - 'sql': SQL statement
                    - 'comment': Comment explaining the SQL
                    - 'summary': Brief summary of what the SQL does

        Example:
            [{"name": "daily_sales",
              "sql": "SELECT date, SUM(amount) FROM orders GROUP BY date",
              "comment": "Calculate daily sales totals",
              "summary": "Aggregates order amounts by date"}]
        """
        try:
            result = self.sql_history_store.get_sql_history_detail(domain, layer1, layer2, name)

            if not result:
                return FuncToolResult(success=0, error=f"SQL history not found: {domain}/{layer1}/{layer2}/{name}")

            # Simplify output - only return essential fields
            simplified_result = [
                {
                    "name": item.get("name"),
                    "sql": item.get("sql"),
                    "comment": item.get("comment"),
                    "summary": item.get("summary"),
                }
                for item in result
            ]

            logger.debug(f"Retrieved SQL history: {domain}/{layer1}/{layer2}/{name}")
            return FuncToolResult(success=1, error=None, result=simplified_result)
        except Exception as e:
            logger.error(f"Failed to get SQL history {domain}/{layer1}/{layer2}/{name}: {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def search_external_knowledge(
        self, query_text: str, domain: str = "", layer1: str = "", layer2: str = "", top_n: int = 5
    ) -> FuncToolResult:
        """
        Search for business terminology, domain knowledge, and concept definitions.
        This tool helps find explanations of business terms, processes, and domain-specific concepts.

        Use this tool when you need to:
        - Understand business terminology and definitions
        - Learn about domain-specific concepts and processes
        - Get context for business rules and requirements
        - Find explanations of industry-specific terms

        Args:
            query_text: Natural language query about business terms or concepts (e.g., "customer lifetime value",
                "churn rate definition", "fiscal year")
            domain: Business domain to search within (e.g., "finance", "marketing", "operations").
                Leave empty if not specified in context.
            layer1: Primary semantic layer for categorization. Leave empty if not specified in context.
            layer2: Secondary semantic layer for fine-grained categorization. Leave empty if not specified in context.
            top_n: Maximum number of results to return (default 5)

        Returns:
            dict: Knowledge search results containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if search failed
                - 'result' (list): List of knowledge entries with domain, layer1, layer2, terminology, and explanation
        """
        try:
            result = self.ext_knowledge_rag.search_knowledge(
                query_text=query_text, domain=domain, layer1=layer1, layer2=layer2, top_n=top_n
            )
            return FuncToolResult(
                success=1,
                error=None,
                result=result.select(
                    ["domain", "layer1", "layer2", "terminology", "explanation", "created_at"]
                ).to_pylist(),
            )
        except Exception as e:
            logger.error(f"Failed to search external knowledge for query '{query_text}': {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def search_documents(self, query_text: str, top_n: int = 5) -> FuncToolResult:
        """
        Search through project documentation, specifications, and technical documents.
        This tool helps find relevant information from project docs, requirements, and specifications.

        Use this tool when you need to:
        - Find specific information in project documentation
        - Locate requirements and specifications
        - Search through technical documentation
        - Get context from project-related documents

        Args:
            query_text: Natural language query about what you're looking for in documents (e.g., "API specifications",
                "data pipeline requirements", "system architecture")
            top_n: Maximum number of document chunks to return (default 5)

        Returns:
            dict: Document search results containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if search failed
                - 'result' (list): List of document chunks with title, hierarchy, keywords, language, and chunk_text
        """
        try:
            results = self.doc_rag.search_similar_documents(
                query_text=query_text,
                top_n=top_n,
                select_fields=["title", "hierarchy", "keywords", "language", "chunk_text"],
            )
            return FuncToolResult(success=1, error=None, result=results.to_pylist())
        except Exception as e:
            logger.error(f"Failed to search documents for query '{query_text}': {str(e)}")
            return FuncToolResult(success=0, error=str(e))
