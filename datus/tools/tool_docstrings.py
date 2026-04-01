# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
"""
Centralized tool docstrings for all Datus agent tools.

This file contains all tool descriptions in one place for easy management and simplification.
Tool functions import their docstrings from here to maintain consistency.
"""

# ============================================================================
# DATABASE TOOLS (DBFuncTool)
# ============================================================================

SEARCH_TABLE_DOC = """
Search for database tables using natural language queries with vector similarity.

This tool helps find relevant tables by searching through table names, schemas (DDL),
and sample data using semantic search. Use this FIRST before describe_table to
efficiently discover relevant tables.

Use this tool when you need to:
- Find tables related to a specific business concept or domain
- Discover tables containing certain types of data
- Locate tables for SQL query development
- Understand what tables are available in a database

**Application Guidance**:
1. If table matches (via definition/sample_data), use it directly
2. If partitioned (e.g., date-based in definition), explore correct partition via describe_table
3. If no match, use list_tables for broader exploration

Args:
    query_text: Natural language description of what you're looking for
               (e.g., "customer data", "sales transactions", "user profiles")
    catalog: Optional catalog name to filter search results. Leave empty if not specified.
    database: Optional database name to filter search results. Leave empty if not specified.
    db_schema: Optional schema name to filter search results. Leave empty if not specified.
    top_n: Maximum number of results to return (default 5)
    simple_sample_data: If True, return simplified sample data without catalog/database/schema fields

Returns:
    dict: Search results containing:
        - 'success' (int): 1 if successful, 0 if failed
        - 'error' (str or None): Error message if search failed
        - 'result' (dict): Search results with:
            - 'metadata' (list): Table information including catalog_name, database_name, schema_name,
                 table_name, table_type ('table'/'view'/'mv'), definition (DDL), identifier, and _distance
            - 'sample_data' (list): Sample rows from matching tables with identifier, table_type,
                 sample_rows, and _distance

Example:
    search_table("customer information", database="prod", top_n=3)
    Returns tables with DDL and sample data related to customers
"""

LIST_TABLES_DOC = """
List all tables, views, and materialized views in the database.

Args:
    catalog: Optional catalog name to filter tables
    database: Optional database name to filter tables
    schema_name: Optional schema name to filter tables
    include_views: Whether to include views and materialized views in results

Returns:
    dict: A dictionary with the execution result, containing these keys:
          - 'success' (int): 1 for success, 0 for failure.
          - 'error' (Optional[str]): Error message on failure.
          - 'result' (Optional[List[Dict[str, str]]]): A list of table names and type on success.
"""

DESCRIBE_TABLE_DOC = """
Get the complete schema information for a specific table, including column definitions, data types, and
 constraints.

Use this tool when you need to:
- Understand the structure of a specific table
- Get column names, data types, and constraints for SQL query writing
- Analyze table schema for data modeling or analysis
- Verify table structure before running queries

**IMPORTANT**: Only use AFTER search_table if no match or for partitioned tables.
Always prefer search_table first for semantic table discovery.

Args:
    table_name: Name of the table to describe
    catalog: Optional catalog name for precise table identification. Leave empty if not specified.
    database: Optional database name for precise table identification. Leave empty if not specified.
    schema_name: Optional schema name for precise table identification. Leave empty if not specified.

Returns:
    dict: Table schema information containing:
        - 'success' (int): 1 if successful, 0 if failed
        - 'error' (str or None): Error message if operation failed
        - 'result' (list): Detailed table schema including columns, data types, and constraints
"""

READ_QUERY_DOC = """
Execute a SQL query and return the results.

Args:
    sql: The SQL query to execute

Returns:
    dict: A dictionary with the execution result, containing these keys:
          - 'success' (int): 1 for success, 0 for failure.
          - 'error' (Optional[str]): Error message on failure.
          - 'result' (Optional[dict]): Query results on success, including original_rows, original_columns,
           is_compressed, and compressed_data.
"""

GET_TABLE_DDL_DOC = """
Get complete DDL definition for a database table.

Use this tool when you need to:
- Generate semantic models (LLM needs complete DDL for accurate generation)
- Understand table structure including constraints, indexes, and relationships
- Analyze foreign key relationships for semantic model generation

Args:
    table_name: Name of the database table
    catalog: Optional catalog name to filter tables
    database: Optional database name to filter tables
    schema_name: Optional schema name to filter tables

Returns:
    dict: DDL results containing:
        - 'success' (int): 1 if successful, 0 if failed
        - 'error' (str or None): Error message if failed
        - 'result' (dict): Contains:
            - 'identifier' (str): Full table identifier
            - 'catalog_name' (str): Catalog name
            - 'database_name' (str): Database name
            - 'schema_name' (str): Schema name
            - 'table_name' (str): Table name
            - 'definition' (str): Complete CREATE TABLE DDL statement
            - 'table_type' (str): Table type (table, view, etc.)
"""

LIST_CATALOGS_DOC = """
List all catalogs in the database.

Returns:
    dict: A dictionary with the execution result, containing these keys:
          - 'success' (int): 1 for success, 0 for failure.
          - 'error' (Optional[str]): Error message on failure.
          - 'result' (Optional[List[str]]): A list of catalog names on success.
"""

LIST_DATABASES_DOC = """
List all databases in the database system.

Args:
    catalog: Optional catalog name to filter databases (depends on database type)
    include_sys: Whether to include system databases in the results

Returns:
    dict: A dictionary with the execution result, containing these keys:
          - 'success' (int): 1 for success, 0 for failure.
          - 'error' (Optional[str]): Error message on failure.
          - 'result' (Optional[List[str]]): A list of database names on success.
"""

LIST_SCHEMAS_DOC = """
List all schemas within a database. Schemas are logical containers that organize tables and other database
 objects.

Use this tool when you need to:
- Discover what schemas exist in a database
- Navigate to specific schemas for table exploration
- Find schemas related to specific applications or business areas

Args:
    catalog: Optional catalog name to filter schemas. Leave empty if not specified.
    database: Optional database name to filter schemas. Leave empty if not specified.
    include_sys: Whether to include system schemas (default False). Set to True for maintenance tasks.

Returns:
    dict: Schema list containing:
        - 'success' (int): 1 if successful, 0 if failed
        - 'error' (str or None): Error message if operation failed
        - 'result' (list): List of schema names
"""

# ============================================================================
# CONTEXT SEARCH TOOLS (ContextSearchTools)
# ============================================================================

SEARCH_METRICS_DOC = """
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

SEARCH_REFERENCE_SQL_DOC = """
Perform a vector search to match reference SQL queries by intent.

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

LIST_DOMAINS_DOC = """
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

LIST_LAYERS_BY_DOMAIN_DOC = """
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

LIST_ITEMS_DOC = """
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
    SQL queries, use get_metrics() or get_sql_history() instead.
"""

GET_METRICS_DOC = """
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

GET_SQL_HISTORY_DOC = """
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

SEARCH_EXTERNAL_KNOWLEDGE_DOC = """
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

# ============================================================================
# REFERENCE TEMPLATE TOOLS (ReferenceTemplateTools)
# ============================================================================

SEARCH_REFERENCE_TEMPLATE_DOC = """
Perform a vector search to match reference SQL templates by intent.

**Application Guidance**: If matches are found, use `get_reference_template` to get full details,
then call `render_reference_template` with appropriate parameters to generate the final SQL.
Do NOT write SQL from scratch when a matching template exists.

Args:
    query_text: The natural language query text representing the desired SQL intent.
    subject_path: Optional subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1'])
    top_n: The number of top results to return (default 5).

Returns:
    dict: A dictionary with keys:
        - 'success' (int): 1 if the search succeeded, 0 otherwise.
        - 'error' (str or None): Error message if any.
        - 'result' (list): On success, a list of matching templates, each containing:
            - 'name': Template name
            - 'template': Raw Jinja2 SQL template
            - 'parameters': JSON string of parameter definitions
            - 'summary': Brief description
            - 'tags': Associated tags
"""

GET_REFERENCE_TEMPLATE_DOC = """
Get a specific reference template by exact subject path and name.

Returns the full template content and its required parameters, ready for rendering.

Args:
    subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1'])
    name: The exact name of the reference template.

Returns:
    dict: Template details containing:
        - 'success' (int): 1 if successful, 0 if failed
        - 'error' (str or None): Error message if failed
        - 'result' (dict): Template entry with:
            - 'name': Template name
            - 'template': Raw Jinja2 SQL template
            - 'parameters': JSON string of parameter definitions (e.g., [{"name": "start_date"}, ...])
            - 'summary': Brief description
            - 'tags': Associated tags
"""

RENDER_REFERENCE_TEMPLATE_DOC = """
Render a reference template with given parameters to produce final SQL.

The template is identified by subject_path + name, and rendered server-side using Jinja2.
First use `search_reference_template` or `get_reference_template` to find the template
and its required parameters, then call this tool with appropriate parameter values.

Args:
    subject_path: Subject hierarchy path (e.g., ['Finance', 'Revenue', 'Q1'])
    name: The exact name of the reference template.
    params: Dictionary of parameter values to render the template.
            Keys must match the template's parameter names.
            Example: {"start_date": "2024-01-01", "end_date": "2024-12-31", "region": "US"}

Returns:
    dict: Render results containing:
        - 'success' (int): 1 if successful, 0 if failed
        - 'error' (str or None): Error message if failed (includes missing parameter info)
        - 'result' (dict): On success, contains:
            - 'rendered_sql': The final rendered SQL string
            - 'template_name': Name of the template used
            - 'parameters_used': The parameters that were applied
"""

SEARCH_DOCUMENTS_DOC = """
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
