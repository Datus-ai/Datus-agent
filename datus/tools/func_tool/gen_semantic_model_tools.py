# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Semantic Model Generation Tools

This module provides specialized tools for semantic model generation,
including table relationship analysis and column usage pattern detection.
These tools are only used during semantic model generation.
"""

from typing import Any, Dict, List, Optional

from agents import Tool

from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.database import DBFuncTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class GenSemanticModelTools:
    """
    Specialized tools for semantic model generation.

    These tools analyze database structures and historical query patterns
    to help generate comprehensive semantic models.
    """

    def __init__(self, db_tool: DBFuncTool):
        """
        Initialize semantic model tools.

        Args:
            db_tool: Database function tool instance for accessing database info
        """
        self.db_tool = db_tool
        self.agent_config = db_tool.agent_config
        self.sub_agent_name = db_tool.sub_agent_name

    def available_tools(self) -> List[Tool]:
        """Get all available semantic model tools."""
        from datus.tools.func_tool import trans_to_function_tool

        bound_tools = []
        methods_to_convert = [
            self.analyze_table_relationships,
            self.get_multiple_tables_ddl,
            self.analyze_column_usage_patterns,
        ]

        for bound_method in methods_to_convert:
            bound_tools.append(trans_to_function_tool(bound_method))
        return bound_tools

    def analyze_table_relationships(
        self,
        tables: List[str],
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
        sample_sql_queries: int = 20,
    ) -> FuncToolResult:
        """
        Analyze relationships between tables using multiple strategies.

        Discovers foreign key relationships by examining:
        1. DDL FOREIGN KEY constraints (highest confidence)
        2. Historical JOIN patterns from stored SQL queries (medium confidence)
        3. Column name similarity analysis (low confidence fallback)

        Use this tool when generating multi-table semantic models to discover
        how tables are related through foreign key relationships.

        Args:
            tables: List of table names to analyze relationships for
            catalog: Optional catalog override
            database: Optional database override
            schema_name: Optional schema override
            sample_sql_queries: Number of historical SQL queries to analyze for JOIN patterns

        Returns:
            FuncToolResult with result containing:
            {
                "relationships": [
                    {
                        "source_table": "orders",
                        "source_column": "customer_id",
                        "target_table": "customers",
                        "target_column": "id",
                        "confidence": "high|medium|low",
                        "evidence": "foreign_key|join_pattern|column_name"
                    },
                    ...
                ],
                "summary": "Found 3 relationships across 3 tables"
            }
        """
        try:
            relationships = []

            # Strategy 1: Extract FOREIGN KEY from DDL
            fk_relationships = self._extract_foreign_keys_from_ddl(tables, catalog, database, schema_name)
            relationships.extend(fk_relationships)

            # Strategy 2: Analyze historical SQL JOIN patterns
            join_relationships = self._analyze_join_patterns_from_history(tables, sample_sql_queries)
            relationships.extend(join_relationships)

            # Strategy 3: Infer from column names (fallback)
            if not relationships:
                name_relationships = self._infer_from_column_names(tables, catalog, database, schema_name)
                relationships.extend(name_relationships)

            # Deduplicate and sort by confidence
            deduplicated = self._deduplicate_relationships(relationships)

            return FuncToolResult(
                result={
                    "relationships": deduplicated,
                    "summary": f"Found {len(deduplicated)} relationships across {len(tables)} tables",
                }
            )

        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    def get_multiple_tables_ddl(
        self,
        tables: List[str],
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
    ) -> FuncToolResult:
        """
        Batch retrieve DDL for multiple tables.

        More efficient than calling get_table_ddl multiple times.

        Args:
            tables: List of table names
            catalog: Optional catalog override
            database: Optional database override
            schema_name: Optional schema override

        Returns:
            FuncToolResult with result as list of table DDL info:
            [
                {"table_name": "orders", "definition": "CREATE TABLE ...", ...},
                {"table_name": "customers", "definition": "CREATE TABLE ...", ...}
            ]
        """
        try:
            results = []
            for table in tables:
                ddl_result = self.db_tool.get_table_ddl(table, catalog, database, schema_name)
                if ddl_result.success and ddl_result.result:
                    results.append({"table_name": table, **ddl_result.result})
                else:
                    results.append({"table_name": table, "error": ddl_result.error})

            return FuncToolResult(result=results)
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    def analyze_column_usage_patterns(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
        sample_sql_queries: int = 50,
    ) -> FuncToolResult:
        """
        Analyze how columns are used in historical SQL queries.

        Discovers column usage patterns including:
        1. Filter operators (LIKE, IN, FIND_IN_SET, =, >, <, BETWEEN, etc.)
        2. Common filter values or patterns
        3. Usage frequency

        Use this tool when generating semantic models to understand
        how columns are typically queried and filtered.

        Args:
            table_name: Table name to analyze
            columns: Optional list of specific columns to analyze (None = all columns)
            catalog: Optional catalog override
            database: Optional database override
            schema_name: Optional schema override
            sample_sql_queries: Number of historical SQL queries to analyze

        Returns:
            FuncToolResult with result containing:
            {
                "column_patterns": {
                    "status": {
                        "operators": ["=", "IN"],
                        "functions": [],
                        "common_filters": ["status = 1", "status IN (1,2,3)"],
                        "usage_count": 45,
                        "usage_description": "Commonly filtered with =, IN"
                    },
                    "tags": {
                        "operators": ["LIKE"],
                        "functions": ["FIND_IN_SET"],
                        "common_filters": ["FIND_IN_SET('vip', tags)"],
                        "usage_count": 23,
                        "usage_description": "Use FIND_IN_SET() for filtering"
                    }
                },
                "summary": "Analyzed 2 columns from 50 SQL queries"
            }
        """
        try:
            if not self.agent_config:
                return FuncToolResult(
                    success=0, error="Cannot analyze column patterns without agent_config (no SQL history available)"
                )

            import re

            from datus.storage.reference_sql.store import ReferenceSqlRAG

            # Get table schema to know which columns exist
            schema_result = self.db_tool.describe_table(table_name, catalog, database, schema_name)
            if not schema_result.success:
                return FuncToolResult(success=0, error=f"Failed to get table schema: {schema_result.error}")

            # describe_table returns {"columns": [...], "table": {...}}
            table_columns = schema_result.result.get("columns", [])
            all_columns = [col["name"] for col in table_columns]
            target_columns = columns if columns else all_columns

            # Initialize pattern tracking
            column_patterns = {
                col: {
                    "operators": set(),
                    "functions": set(),
                    "common_filters": [],
                    "usage_count": 0,
                    "filter_examples": [],
                }
                for col in target_columns
            }

            # Search for SQL queries containing the table
            sql_rag = ReferenceSqlRAG(self.agent_config, self.sub_agent_name)
            search_results = sql_rag.search_reference_sql(
                query_text=f"SELECT FROM {table_name}", top_n=sample_sql_queries
            )

            logger.info(f"Found {len(search_results)} historical SQL queries for table {table_name}")

            # Pattern definitions for different operators and functions
            operator_patterns = {
                "LIKE": r"\b{col}\b\s+LIKE\s+",
                "IN": r"\b{col}\b\s+IN\s*\(",
                "BETWEEN": r"\b{col}\b\s+BETWEEN\s+",
                "=": r"\b{col}\b\s*=\s*",
                ">": r"\b{col}\b\s*>\s*",
                "<": r"\b{col}\b\s*<\s*",
                ">=": r"\b{col}\b\s*>=\s*",
                "<=": r"\b{col}\b\s*<=\s*",
                "!=": r"\b{col}\b\s*(?:!=|<>)\s*",
            }

            function_patterns = {
                "FIND_IN_SET": r"FIND_IN_SET\s*\([^,]+,\s*\b{col}\b\s*\)",
                "JSON_EXTRACT": r"JSON_EXTRACT\s*\(\s*\b{col}\b\s*,",
                "JSON_CONTAINS": r"JSON_CONTAINS\s*\(\s*\b{col}\b\s*,",
                "REGEXP": r"\b{col}\b\s+REGEXP\s+",
                "MATCH": r"MATCH\s*\(\s*\b{col}\b\s*\)",
            }

            # Helper function to sanitize filter examples by redacting sensitive literals
            def sanitize_example(example: str) -> str:
                """Redact sensitive literals from SQL example snippets."""
                sanitized = example
                # Redact quoted strings (single and double quotes)
                sanitized = re.sub(r"'[^']*'", "'<REDACTED>'", sanitized)
                sanitized = re.sub(r'"[^"]*"', '"<REDACTED>"', sanitized)
                # Redact numeric literals (integers and decimals) after operators
                sanitized = re.sub(r"(?<=[=<>!\s,(\[])\s*\d+\.?\d*(?=\s*[,)\];\s]|$)", " <REDACTED>", sanitized)
                return sanitized

            # Analyze each SQL query
            for sql_entry in search_results:
                sql_text = sql_entry.get("sql", "")

                # Check if this SQL actually uses our target table
                if not sql_text or table_name.lower() not in sql_text.lower():
                    continue

                # Track columns seen in this query to increment usage_count only once per query
                seen_columns_in_query: set = set()

                for col in target_columns:
                    # Check for operators
                    for op, pattern_template in operator_patterns.items():
                        pattern = pattern_template.replace("{col}", re.escape(col))
                        if re.search(pattern, sql_text, re.IGNORECASE):
                            column_patterns[col]["operators"].add(op)

                            # Increment usage_count only once per column per query
                            if col not in seen_columns_in_query:
                                column_patterns[col]["usage_count"] += 1
                                seen_columns_in_query.add(col)

                            # Extract example filter (limit to 150 chars), sanitize before storing
                            match = re.search(rf"\b{re.escape(col)}\b[^,;)]*", sql_text, re.IGNORECASE)
                            if match and len(column_patterns[col]["filter_examples"]) < 3:
                                example = sanitize_example(match.group(0).strip()[:150])
                                if example not in column_patterns[col]["filter_examples"]:
                                    column_patterns[col]["filter_examples"].append(example)

                    # Check for functions
                    for func, pattern_template in function_patterns.items():
                        pattern = pattern_template.replace("{col}", re.escape(col))
                        if re.search(pattern, sql_text, re.IGNORECASE):
                            column_patterns[col]["functions"].add(func)

                            # Increment usage_count only once per column per query
                            if col not in seen_columns_in_query:
                                column_patterns[col]["usage_count"] += 1
                                seen_columns_in_query.add(col)

                            # Extract example function call, sanitize before storing
                            match = re.search(rf"{func}\s*\([^)]*\b{re.escape(col)}\b[^)]*\)", sql_text, re.IGNORECASE)
                            if match and len(column_patterns[col]["filter_examples"]) < 3:
                                example = sanitize_example(match.group(0).strip()[:150])
                                if example not in column_patterns[col]["filter_examples"]:
                                    column_patterns[col]["filter_examples"].append(example)

            # Generate usage descriptions
            result_patterns = {}
            for col, patterns in column_patterns.items():
                if patterns["usage_count"] == 0:
                    continue

                # Convert sets to sorted lists
                operators = sorted(patterns["operators"])
                functions = sorted(patterns["functions"])

                # Generate natural language description
                desc_parts = []
                if functions:
                    desc_parts.append(f"Use {', '.join(functions)}() for queries")
                if operators:
                    op_desc = "Commonly filtered with " + ", ".join(operators)
                    desc_parts.append(op_desc)

                if patterns["filter_examples"]:
                    examples = " | ".join(patterns["filter_examples"][:2])
                    desc_parts.append(f"Example filters: {examples}")

                usage_description = ". ".join(desc_parts) if desc_parts else "Used in queries"

                result_patterns[col] = {
                    "operators": operators,
                    "functions": functions,
                    "common_filters": patterns["filter_examples"][:3],
                    "usage_count": patterns["usage_count"],
                    "usage_description": usage_description,
                }

            logger.info(f"Analyzed {len(result_patterns)} columns with usage patterns")

            return FuncToolResult(
                result={
                    "column_patterns": result_patterns,
                    "summary": f"Analyzed {len(result_patterns)} columns from {len(search_results)} SQL queries",
                }
            )

        except Exception as e:
            logger.exception("Error analyzing column usage patterns")
            return FuncToolResult(success=0, error=str(e))

    # ========== Private helper methods ==========

    def _extract_foreign_keys_from_ddl(
        self, tables: List[str], catalog: str, database: str, schema_name: str
    ) -> List[Dict[str, Any]]:
        """Extract FOREIGN KEY constraints from DDL definitions."""
        import re

        relationships = []
        for table in tables:
            ddl_result = self.db_tool.get_table_ddl(table, catalog, database, schema_name)
            if ddl_result.success and ddl_result.result:
                ddl_text = ddl_result.result.get("definition", "")
                # Match: FOREIGN KEY (column) REFERENCES target_table(target_column)
                fk_pattern = r"FOREIGN\s+KEY\s*\(([^)]+)\)\s*REFERENCES\s+(\w+)\s*\(([^)]+)\)"
                for match in re.finditer(fk_pattern, ddl_text, re.IGNORECASE):
                    relationships.append(
                        {
                            "source_table": table,
                            "source_column": match.group(1).strip(),
                            "target_table": match.group(2).strip(),
                            "target_column": match.group(3).strip(),
                            "confidence": "high",
                            "evidence": "foreign_key",
                        }
                    )
        return relationships

    def _analyze_join_patterns_from_history(self, tables: List[str], sample_size: int) -> List[Dict[str, Any]]:
        """Search historical SQL queries for JOIN patterns."""
        if not self.agent_config:
            return []

        import re

        from datus.storage.reference_sql.store import ReferenceSqlRAG

        sql_rag = ReferenceSqlRAG(self.agent_config, self.sub_agent_name)
        relationships = []

        # Build case-insensitive lookup: lowercased name -> canonical name
        tables_lower_map = {t.lower(): t for t in tables}

        # Search for SQL queries containing each table
        for table in tables:
            try:
                search_results = sql_rag.search_reference_sql(query_text=f"JOIN {table}", top_n=sample_size)

                for sql_entry in search_results:
                    sql_text = sql_entry.get("sql", "")
                    # Match: table1.column1 = table2.column2
                    join_pattern = r"(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)"
                    for match in re.finditer(join_pattern, sql_text, re.IGNORECASE):
                        left_table, left_col, right_table, right_col = match.groups()

                        # Only keep joins involving target tables (case-insensitive)
                        left_lower = left_table.lower()
                        right_lower = right_table.lower()
                        if left_lower in tables_lower_map and right_lower in tables_lower_map:
                            relationships.append(
                                {
                                    "source_table": tables_lower_map[left_lower],
                                    "source_column": left_col,
                                    "target_table": tables_lower_map[right_lower],
                                    "target_column": right_col,
                                    "confidence": "medium",
                                    "evidence": "join_pattern",
                                }
                            )
            except Exception as e:
                logger.warning(f"Failed to search SQL history for table {table}: {e}")

        return relationships

    def _infer_from_column_names(
        self, tables: List[str], catalog: str, database: str, schema_name: str
    ) -> List[Dict[str, Any]]:
        """Infer relationships from column naming patterns."""
        relationships = []
        table_schemas = {}

        # Build case-insensitive lookup: lowercased name -> canonical name
        tables_lower_map = {t.lower(): t for t in tables}

        # Get all table schemas
        for table in tables:
            schema_result = self.db_tool.describe_table(table, catalog, database, schema_name)
            if schema_result.success and schema_result.result:
                table_schemas[table] = schema_result.result.get("columns", [])

        # Check for {table_name}_id -> {table_name}.id patterns
        for source_table, columns in table_schemas.items():
            for column in columns:
                orig_col_name = column.get("name", "")
                col_name = orig_col_name.lower()  # Lowercase for pattern matching

                # Match pattern: {target}_id
                if col_name.endswith("_id"):
                    target_table_lower = col_name[:-3]  # Remove "_id" (already lowercase)

                    if target_table_lower in tables_lower_map:
                        # Get canonical table name with original casing
                        target_table = tables_lower_map[target_table_lower]
                        # Check if target table has "id" column
                        target_columns = table_schemas.get(target_table, [])
                        if any(c.get("name", "").lower() == "id" for c in target_columns):
                            relationships.append(
                                {
                                    "source_table": source_table,
                                    "source_column": orig_col_name,
                                    "target_table": target_table,
                                    "target_column": "id",
                                    "confidence": "low",
                                    "evidence": "column_name",
                                }
                            )

        return relationships

    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate and sort relationships by confidence."""
        seen = set()
        deduplicated = []

        # Sort by confidence (high > medium > low)
        confidence_order = {"high": 0, "medium": 1, "low": 2}
        sorted_rels = sorted(relationships, key=lambda r: confidence_order.get(r["confidence"], 3))

        for rel in sorted_rels:
            key = (rel["source_table"], rel["source_column"], rel["target_table"], rel["target_column"])
            if key not in seen:
                seen.add(key)
                deduplicated.append(rel)

        return deduplicated
