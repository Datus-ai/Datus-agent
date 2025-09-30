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
                self.search_table_metadata,
                self.search_metrics,
                self.search_documents,
                self.search_external_knowledge,
                self.search_historical_sql,
            )
        ]

    def search_table_metadata(
        self,
        query_text: str,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        top_n=5,
        simple_sample_data: bool = True,
    ) -> FuncToolResult:
        """
        Search for database table metadata using natural language queries.
        This tool helps find relevant tables by searching through table names, schemas (DDL)
        , and sample data using vector similarity.

        Use this tool when you need to:
        - Find tables related to a specific business concept or domain
        - Discover tables containing certain types of data
        - Locate tables for SQL query development
        - Understand what tables are available in a database

        **Application Guidance**: Analyze results: 1. If table matches (via definition/sample_data), use it. 2.
        If partitioned (e.g., date-based in definition), explore correct partition via DBTools. 3. If no match,
        then use DBTools for broader exploration.

        Args:
            query_text: Natural language description of what you're looking for (e.g., "customer data",
             "sales transactions", "user profiles")
            catalog_name: Optional catalog name to filter search results.
            database_name: Optional database name to filter search results.
            schema_name: Optional schema name to filter search results.
            top_n: Maximum number of results to return (default 5)

        Returns:
            dict: Search results containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if search failed
                - 'result' (dict): Search results with:
                    - 'metadata' (list): Table information including catalog_name, database_name, schema_name,
                         table_name, table_type ('table'/'view'/'mv'), definition (DDL), and identifier
                    - 'sample_data' (list): Sample rows from matching tables with identifier, table_type,
                         and sample_rows
        """
        try:
            metadata, sample_values = self.schema_rag.search_similar(
                query_text,
                catalog_name=catalog_name,
                database_name=database_name,
                schema_name=schema_name,
                table_type="full",
                top_n=top_n,
            )
            result_dict = {"metadata": [], "sample_data": []}
            if metadata:
                result_dict["metadata"] = metadata.select(
                    [
                        "catalog_name",
                        "database_name",
                        "schema_name",
                        "table_name",
                        "table_type",
                        "definition",
                        "identifier",
                        "_distance",
                    ]
                ).to_pylist()

            if sample_values:
                if simple_sample_data:
                    selected_fields = ["identifier", "table_type", "sample_rows", "_distance"]
                else:
                    selected_fields = [
                        "identifier",
                        "catalog_name",
                        "database_name",
                        "schema_name",
                        "table_type",
                        "table_name",
                        "sample_rows",
                        "_distance",
                    ]
                result_dict["sample_data"] = sample_values.select(selected_fields).to_pylist()
            return FuncToolResult(success=1, error=None, result=result_dict)
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    def search_metrics(
        self,
        query_text: str,
        domain: str = "",
        layer1: str = "",
        layer2: str = "",
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
        top_n=5,
    ) -> FuncToolResult:
        """
        Search for business metrics and KPIs using natural language queries.
        This tool finds relevant metrics by searching through metric definitions, descriptions, and SQL logic.

        Use this tool when you need to:
        - Find existing metrics related to a business question
        - Discover KPIs for reporting and analysis
        - Locate metrics for specific business domains
        - Understand how certain metrics are calculated

        **Application Guidance**: If results are found, MUST prioritize reusing the 'sql_query' directly or with minimal
         adjustments (e.g., add date filters). Integrate 'constraint' as mandatory filters in SQL.
         Example: If metric is "revenue" with sql_query="SELECT SUM(sales) FROM orders" and
         constraint="WHERE date > '2020'", use or adjust to "SELECT SUM(sales) FROM orders WHERE date > '2023'".

        Args:
            query_text: Natural language description of the metric you're looking for (e.g., "revenue metrics",
                "customer engagement", "conversion rates")
            domain: Business domain to search within (e.g., "sales", "marketing", "finance").
            layer1: Primary semantic layer for categorization.
            layer2: Secondary semantic layer for fine-grained categorization.
            catalog_name: Optional catalog name to filter metrics.
            database_name: Optional database name to filter metrics.
            schema_name: Optional schema name to filter metrics.
            top_n: Maximum number of results to return (default 5)

        Returns:
            dict: Metric search results containing:
                - 'success' (int): 1 if successful, 0 if failed
                - 'error' (str or None): Error message if search failed
                - 'result' (list): List of matching metrics with name, description, constraint, and sql_query
        """
        try:
            metrics = self.metric_rag.search_hybrid_metrics(
                domain=domain,
                layer1=layer1,
                layer2=layer2,
                query_text=query_text,
                catalog_name=catalog_name,
                database_name=database_name,
                schema_name=schema_name,
                top_n=top_n,
            )
            return FuncToolResult(success=1, error=None, result=metrics)
        except Exception as e:
            logger.error(f"Failed to search metrics for table '{query_text}': {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def search_historical_sql(
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
