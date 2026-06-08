# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Semantic Discovery Tools

This module provides read-only discovery tools for semantic-layer generation,
including table relationships, column usage evidence, and metric candidates
mined from historical SQL.
"""

from typing import Any, Dict, List, Optional

from agents import Tool

from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.database import DBFuncTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SemanticDiscoveryTools:
    """
    Read-only discovery tools for semantic-layer generation.

    These tools analyze database structures and historical query patterns
    to help generate semantic models and MetricFlow metrics.
    """

    _AGGREGATE_CLASSES = ()

    def __init__(self, db_tool: DBFuncTool):
        """
        Initialize semantic discovery tools.

        Args:
            db_tool: Database function tool instance for accessing database info
        """
        self.db_tool = db_tool
        self.agent_config = db_tool.agent_config
        self.sub_agent_name = db_tool.sub_agent_name

    @classmethod
    def _aggregate_classes(cls):
        if cls._AGGREGATE_CLASSES:
            return cls._AGGREGATE_CLASSES
        from sqlglot import expressions as exp

        cls._AGGREGATE_CLASSES = (exp.Sum, exp.Count, exp.Avg, exp.Min, exp.Max)
        return cls._AGGREGATE_CLASSES

    def available_tools(self) -> List[Tool]:
        """Get all available semantic discovery tools."""
        from datus.tools.func_tool import trans_to_function_tool

        bound_tools = []
        methods_to_convert = [
            self.analyze_table_relationships,
            self.get_multiple_tables_ddl,
            self.analyze_column_usage_patterns,
            self.analyze_metric_candidates_from_history,
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

    def analyze_metric_candidates_from_history(
        self,
        sql_queries: Optional[List[str]] = None,
        sql_entries_json: Optional[str] = "",
        query_text: Optional[str] = "",
        tables: Optional[List[str]] = None,
        sample_sql_queries: int = 50,
        existing_metric_catalog_json: Optional[str] = "",
    ) -> FuncToolResult:
        """
        Mine MetricFlow metric candidates from historical SQL using SQL ASTs.

        This is a read-only discovery tool. It preserves final SELECT output
        expressions as first-class metric candidates and reports base measures
        as dependencies, instead of reducing every SQL to independent base
        measure_proxy metrics.

        Args:
            sql_queries: Raw SQL statements to analyze.
            sql_entries_json: JSON array of reference SQL-like dictionaries.
                Each item may contain sql plus optional name, summary, question,
                filepath, or comment.
            query_text: Query text used to search reference SQL when direct SQL
                inputs are not provided.
            tables: Optional table names used to search reference SQL and filter
                evidence.
            sample_sql_queries: Maximum reference SQL rows to inspect.
            existing_metric_catalog_json: JSON array of existing metric objects.
                Each item should include at least name, and may include type,
                description, and subject_path. Expressions over these metrics
                can be classified as derived metrics.

        Returns:
            FuncToolResult with metric_candidates, direct_metric_candidates,
            derived_metric_candidates, base_measures, identity_metric_references,
            non_metric_evidence, parse_errors, and summary.
        """
        try:
            entries = self._load_metric_mining_entries(
                sql_queries, sql_entries_json, query_text, tables, sample_sql_queries
            )
            existing_metric_catalog = self._load_existing_metric_catalog(existing_metric_catalog_json)

            metric_candidates: Dict[str, Dict[str, Any]] = {}
            base_measures: Dict[str, Dict[str, Any]] = {}
            non_metric_evidence: List[Dict[str, Any]] = []
            identity_metric_references: List[Dict[str, Any]] = []
            support_measure_candidates: List[Dict[str, Any]] = []
            parse_errors: List[Dict[str, Any]] = []
            source_classifications: List[Dict[str, Any]] = []
            derived_datasource_recommendations: List[Dict[str, Any]] = []
            blocked_direct_metric_candidates: List[Dict[str, Any]] = []
            literal_mappings: List[Dict[str, Any]] = []
            time_grain_evidence: List[Dict[str, Any]] = []
            post_aggregation_constraints: List[Dict[str, Any]] = []

            for idx, entry in enumerate(entries):
                sql_text = entry.get("sql", "")
                source_name = entry.get("name") or entry.get("summary") or entry.get("filepath") or f"sql_{idx + 1}"
                if not sql_text:
                    continue

                try:
                    parsed_expressions = self._parse_sql(sql_text)
                except Exception as exc:
                    parse_errors.append({"source": source_name, "index": idx, "error": str(exc)})
                    continue

                entry_candidates: List[Dict[str, Any]] = []
                entry_has_non_metric_evidence = False
                entry_has_metric_evidence = False
                modeling_analysis = self._analyze_query_modeling(parsed_expressions, source_name)
                if modeling_analysis["derived_datasource_recommendations"]:
                    derived_datasource_recommendations.extend(modeling_analysis["derived_datasource_recommendations"])
                preservation_evidence = self._extract_semantic_preservation_evidence(
                    parsed_expressions,
                    source_name,
                )
                literal_mappings.extend(preservation_evidence["literal_mappings"])
                time_grain_evidence.extend(preservation_evidence["time_grain_evidence"])
                post_aggregation_constraints.extend(preservation_evidence["post_aggregation_constraints"])

                found_candidate = False
                for parsed in parsed_expressions:
                    for select in self._iter_selects(parsed):
                        select_tables = self._collect_tables(select)
                        filters = self._collect_filters(select)
                        dimensions = self._collect_dimensions(select)
                        support_projection_aliases = self._support_measure_projection_aliases(select)

                        for projection in select.expressions:
                            candidate = self._candidate_from_projection(
                                projection=projection,
                                source_name=source_name,
                                tables=select_tables,
                                filters=filters,
                                dimensions=dimensions,
                                existing_metric_catalog=existing_metric_catalog,
                            )
                            if not candidate:
                                continue
                            entry_has_metric_evidence = True
                            if candidate.get("evidence_kind") == "identity_metric_reference":
                                identity_metric_references.append(candidate)
                                continue
                            if self._projection_alias_key(projection) in support_projection_aliases:
                                support_candidate = dict(candidate)
                                support_candidate["evidence_kind"] = "support_measure"
                                support_candidate["reason"] = (
                                    "COUNT(*) appears alongside a distinct business count; keep it as a "
                                    "support/base measure instead of a published metric"
                                )
                                support_measure_candidates.append(support_candidate)
                                for measure in support_candidate.get("base_measures", []):
                                    measure = dict(measure)
                                    measure["evidence_kind"] = "support_measure"
                                    self._merge_base_measure(base_measures, measure)
                                continue
                            found_candidate = True
                            entry_candidates.append(candidate)
                            self._merge_metric_candidate(metric_candidates, candidate)
                            for measure in candidate.get("base_measures", []):
                                self._merge_base_measure(base_measures, measure)

                        if (
                            not found_candidate
                            and not entry_has_metric_evidence
                            and (filters or dimensions or select_tables)
                        ):
                            entry_has_non_metric_evidence = True
                            non_metric_evidence.append(
                                {
                                    "source_sql_name": source_name,
                                    "filters": filters,
                                    "dimensions": dimensions,
                                    "tables": select_tables,
                                    "reason": "detail query without aggregate output",
                                }
                            )

                classification = self._classify_source_query(
                    has_candidates=bool(entry_candidates),
                    has_non_metric_evidence=entry_has_non_metric_evidence,
                    derived_datasource_recommendations=modeling_analysis["derived_datasource_recommendations"],
                )
                source_classifications.append(
                    {
                        "source_sql_name": source_name,
                        "classification": classification,
                        "reason": modeling_analysis["classification_reason"],
                    }
                )
                if classification == "metric_plus_derived_datasource":
                    for candidate in entry_candidates:
                        blocked = dict(candidate)
                        blocked["block_reason"] = (
                            "aggregation over ranked/windowed result; create the recommended derived data source first"
                        )
                        blocked_direct_metric_candidates.append(blocked)

            candidates = sorted(
                metric_candidates.values(), key=lambda item: (-item.get("source_count", 1), item["name"])
            )
            measures = sorted(base_measures.values(), key=lambda item: (-item.get("source_count", 1), item["name"]))
            direct_candidates = [
                candidate
                for candidate in candidates
                if candidate.get("metric_type") != "derived"
                and not self._is_blocked_direct_candidate(candidate, blocked_direct_metric_candidates)
            ]
            derived_candidates = [
                candidate
                for candidate in candidates
                if candidate.get("metric_type") == "derived"
                and not self._is_blocked_direct_candidate(candidate, blocked_direct_metric_candidates)
            ]
            modeling_plan = self._build_modeling_plan(derived_datasource_recommendations)
            return FuncToolResult(
                result={
                    "metric_candidates": candidates,
                    "direct_metric_candidates": direct_candidates,
                    "derived_metric_candidates": derived_candidates,
                    "base_measures": measures,
                    "support_measure_candidates": support_measure_candidates,
                    "non_metric_evidence": non_metric_evidence,
                    "identity_metric_references": identity_metric_references,
                    "parse_errors": parse_errors,
                    "query_classification": self._overall_query_classification(
                        source_classifications=source_classifications,
                        parse_errors=parse_errors,
                    ),
                    "source_classifications": source_classifications,
                    "derived_datasource_recommendations": derived_datasource_recommendations,
                    "blocked_direct_metric_candidates": blocked_direct_metric_candidates,
                    "literal_mappings": literal_mappings,
                    "time_grain_evidence": time_grain_evidence,
                    "post_aggregation_constraints": post_aggregation_constraints,
                    "modeling_plan": modeling_plan,
                    "summary": (
                        f"Found {len(candidates)} metric candidates and {len(measures)} base measures "
                        f"from {len(entries)} SQL queries"
                    ),
                }
            )
        except Exception as e:
            logger.exception("Error analyzing metric candidates from history")
            return FuncToolResult(success=0, error=str(e))

    # ========== Private helper methods ==========

    def _classify_source_query(
        self,
        has_candidates: bool,
        has_non_metric_evidence: bool,
        derived_datasource_recommendations: List[Dict[str, Any]],
    ) -> str:
        """Classify how a SQL query should be modeled."""
        if derived_datasource_recommendations:
            return "metric_plus_derived_datasource"
        if has_candidates:
            return "direct_metric"
        if has_non_metric_evidence:
            return "cohort_or_dataset_only"
        return "manual_review_required"

    def _overall_query_classification(
        self,
        source_classifications: List[Dict[str, Any]],
        parse_errors: List[Dict[str, Any]],
    ) -> str:
        """Summarize per-source classifications for the whole tool call."""
        classifications = {item.get("classification") for item in source_classifications}
        if "metric_plus_derived_datasource" in classifications:
            return "metric_plus_derived_datasource"
        if "direct_metric" in classifications:
            return "direct_metric"
        if classifications == {"cohort_or_dataset_only"}:
            return "cohort_or_dataset_only"
        if parse_errors and not source_classifications:
            return "manual_review_required"
        return "manual_review_required"

    def _build_modeling_plan(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build high-level modeling steps for complex SQL patterns."""
        plans = []
        for rec in recommendations:
            rank_alias = rec.get("rank_alias") or "rank_value"
            rank_filter = rec.get("rank_filters") or []
            plans.append(
                {
                    "source_sql_name": rec.get("source_sql_name", ""),
                    "classification": "metric_plus_derived_datasource",
                    "steps": [
                        {
                            "step": "define_base_metrics",
                            "description": "Define reusable metrics/measures used before the ranking step.",
                            "metric_evidence": rec.get("ordering_metric_evidence", []),
                        },
                        {
                            "step": "create_derived_datasource",
                            "description": (
                                "Create a sql_query data source or materialized view that computes the ranked rows."
                            ),
                            "datasource": rec.get("name", ""),
                            "window": rec.get("window", {}),
                        },
                        {
                            "step": "define_final_metric",
                            "description": (
                                f"Define a metric over a generated flag such as `{rank_alias}_selected` "
                                "instead of using the final aggregation directly."
                            ),
                            "rank_filters": rank_filter,
                        },
                    ],
                }
            )
        return plans

    def _analyze_query_modeling(self, parsed_expressions: List[Any], source_name: str) -> Dict[str, Any]:
        """Detect complex SQL shapes that need more than direct metric extraction."""
        recommendations: List[Dict[str, Any]] = []
        for parsed in parsed_expressions:
            cte_projection_map = self._cte_projection_map(parsed)
            for cte_name, select in self._iter_cte_selects(parsed):
                recommendations.extend(
                    self._ranked_datasource_recommendations(
                        source_name=source_name,
                        cte_name=cte_name,
                        select=select,
                        parsed=parsed,
                        cte_projection_map=cte_projection_map,
                    )
                )
            for subquery_name, select in self._iter_inline_subquery_selects(parsed):
                recommendations.extend(
                    self._ranked_datasource_recommendations(
                        source_name=source_name,
                        cte_name=subquery_name,
                        select=select,
                        parsed=parsed,
                        cte_projection_map=cte_projection_map,
                    )
                )

        reason = ""
        if recommendations:
            reason = "rank/window output is filtered or aggregated downstream; model it as a derived data source first"
        return {
            "derived_datasource_recommendations": recommendations,
            "classification_reason": reason,
        }

    def _extract_semantic_preservation_evidence(
        self,
        parsed_expressions: List[Any],
        source_name: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extract generic SQL facts that generation must preserve verbatim."""
        literal_mappings: List[Dict[str, Any]] = []
        time_grain_evidence: List[Dict[str, Any]] = []
        post_aggregation_constraints: List[Dict[str, Any]] = []

        for parsed in parsed_expressions:
            for select in self._iter_selects(parsed, include_nested=True):
                for projection in select.expressions:
                    literal_mapping = self._literal_mapping_from_projection(projection, source_name)
                    if literal_mapping:
                        self._append_unique(literal_mappings, literal_mapping, ["source_sql_name", "alias", "value"])

                    time_projection = self._time_grain_from_projection(projection, source_name)
                    if time_projection:
                        self._append_unique(
                            time_grain_evidence,
                            time_projection,
                            ["source_sql_name", "alias", "expression", "evidence_type"],
                        )

                where = select.args.get("where")
                predicate = getattr(where, "this", None)
                if predicate is not None:
                    for item in self._time_grain_from_filter(predicate, source_name):
                        self._append_unique(
                            time_grain_evidence,
                            item,
                            ["source_sql_name", "expression", "predicate", "evidence_type"],
                        )

                having = select.args.get("having")
                having_predicate = getattr(having, "this", None)
                if having_predicate is not None:
                    post_aggregation = {
                        "source_sql_name": source_name,
                        "constraint": having_predicate.sql(),
                        "clause": "HAVING",
                        "reason": "post-aggregation constraint must be preserved as a query filter or later derived data source",
                    }
                    self._append_unique(
                        post_aggregation_constraints,
                        post_aggregation,
                        ["source_sql_name", "clause", "constraint"],
                    )

        return {
            "literal_mappings": literal_mappings,
            "time_grain_evidence": time_grain_evidence,
            "post_aggregation_constraints": post_aggregation_constraints,
        }

    def _literal_mapping_from_projection(self, projection: Any, source_name: str) -> Optional[Dict[str, Any]]:
        """Return alias <- literal evidence for projections like `'x' AS source_type`."""
        from sqlglot import expressions as exp

        if not isinstance(projection, exp.Alias):
            return None
        expr = projection.this
        if not isinstance(expr, exp.Literal) or not expr.is_string:
            return None
        return {
            "source_sql_name": source_name,
            "alias": projection.alias,
            "value": expr.this,
            "expression": expr.sql(),
            "projection": projection.sql(),
            "preservation_rule": "preserve literal values verbatim; only MetricFlow object names may be normalized",
        }

    def _time_grain_from_projection(self, projection: Any, source_name: str) -> Optional[Dict[str, Any]]:
        """Return time-dimension evidence from projected date expressions."""
        from sqlglot import expressions as exp

        expr = projection.this if isinstance(projection, exp.Alias) else projection
        alias = projection.alias if isinstance(projection, exp.Alias) else projection.alias_or_name
        if not alias:
            return None

        grain = self._time_grain_for_expr(expr)
        reason = self._time_expression_reason(expr, alias, grain)
        if not reason:
            return None

        return {
            "source_sql_name": source_name,
            "alias": alias,
            "expression": expr.sql(),
            "evidence_type": "projected_time_dimension",
            "grain": grain or self._time_grain_from_alias(alias),
            "reason": reason,
        }

    def _time_grain_from_filter(self, predicate: Any, source_name: str) -> List[Dict[str, Any]]:
        """Return time-grain evidence from date-normalizing filter predicates."""
        evidence = []
        predicate_sql = predicate.sql()
        for expr in self._date_grain_expressions(predicate):
            grain = self._time_grain_for_expr(expr) or "DAY"
            evidence.append(
                {
                    "source_sql_name": source_name,
                    "expression": expr.sql(),
                    "predicate": predicate_sql,
                    "evidence_type": "date_filter",
                    "grain": grain,
                    "reason": f"filter normalizes or constrains data at {grain.lower()} grain",
                }
            )
        return evidence

    def _time_expression_reason(self, expr: Any, alias: str, grain: Optional[str] = None) -> str:
        """Return a reason when an expression should be preserved as time grain."""
        normalized_alias = self._normalize_identifier(alias)
        if self._contains_current_date(expr):
            return "projection uses current date as the output time grain"
        if self._contains_date_cast(expr):
            if grain:
                return f"projection truncates or casts a timestamp to {grain.lower()} grain"
            return "projection truncates or casts a timestamp to a time grain"
        if normalized_alias in {"dt", "ds", "date", "part_dt", "metric_time", "create_date", "created_date"}:
            return "projection alias is commonly used as an output time dimension"
        return ""

    def _time_grain_from_alias(self, alias: str) -> Optional[str]:
        """Infer a conservative grain from common date-like output aliases."""
        normalized_alias = self._normalize_identifier(alias)
        if normalized_alias in {"dt", "ds", "date", "part_dt", "metric_time", "create_date", "created_date"}:
            return "DAY"
        return None

    def _time_grain_for_expr(self, expr: Any) -> Optional[str]:
        """Infer the canonical time grain represented by a date expression."""
        from sqlglot import expressions as exp

        for node in expr.walk():
            if isinstance(node, exp.DateTrunc):
                grain = self._date_trunc_grain(node)
                if grain:
                    return grain
            if isinstance(node, (exp.CurrentDate, exp.TsOrDsToDate, exp.Date)) or self._is_current_date_function(node):
                return "DAY"
        return None

    def _date_trunc_grain(self, expr: Any) -> Optional[str]:
        """Extract a MetricFlow-style grain from a DATE_TRUNC expression."""
        unit = expr.args.get("unit")
        unit_text = ""
        if unit is not None:
            unit_text = getattr(unit, "this", "") or unit.sql()
        normalized_unit = self._normalize_identifier(str(unit_text))
        return {
            "s": "SECOND",
            "sec": "SECOND",
            "second": "SECOND",
            "mi": "MINUTE",
            "min": "MINUTE",
            "minute": "MINUTE",
            "h": "HOUR",
            "hh": "HOUR",
            "hour": "HOUR",
            "d": "DAY",
            "dd": "DAY",
            "day": "DAY",
            "date": "DAY",
            "w": "WEEK",
            "wk": "WEEK",
            "week": "WEEK",
            "m": "MONTH",
            "mm": "MONTH",
            "mon": "MONTH",
            "month": "MONTH",
            "q": "QUARTER",
            "qtr": "QUARTER",
            "quarter": "QUARTER",
            "y": "YEAR",
            "yy": "YEAR",
            "yyyy": "YEAR",
            "year": "YEAR",
        }.get(normalized_unit)

    def _date_grain_expressions(self, expr: Any) -> List[Any]:
        """Find date-grain expressions inside a predicate."""
        from sqlglot import expressions as exp

        matches = []
        for node in expr.walk():
            if isinstance(node, (exp.CurrentDate, exp.TsOrDsToDate, exp.DateTrunc, exp.Date)):
                matches.append(node)
        return matches

    def _contains_current_date(self, expr: Any) -> bool:
        """Return true if expression uses CURRENT_DATE/CURDATE."""
        from sqlglot import expressions as exp

        return any(isinstance(node, exp.CurrentDate) or self._is_current_date_function(node) for node in expr.walk())

    def _contains_date_cast(self, expr: Any) -> bool:
        """Return true if expression truncates/casts timestamp-like data to date."""
        from sqlglot import expressions as exp

        return any(isinstance(node, (exp.TsOrDsToDate, exp.DateTrunc, exp.Date)) for node in expr.walk())

    def _is_current_date_function(self, expr: Any) -> bool:
        """Return true for dialect-specific current-date functions."""
        from sqlglot import expressions as exp

        return isinstance(expr, exp.Anonymous) and expr.name.upper() in {"CURDATE", "CURRENT_DATE"}

    def _iter_cte_selects(self, parsed: Any) -> List[tuple]:
        """Return CTE aliases and SELECT bodies."""
        from sqlglot import expressions as exp

        cte_selects = []
        for cte in parsed.find_all(exp.CTE):
            cte_name = self._safe_name(cte.alias_or_name)
            if isinstance(cte.this, exp.Select):
                cte_selects.append((cte_name, cte.this))
        return cte_selects

    def _iter_inline_subquery_selects(self, parsed: Any) -> List[tuple]:
        """Return inline derived-table aliases and SELECT bodies."""
        from sqlglot import expressions as exp

        subquery_selects = []
        for subquery in parsed.find_all(exp.Subquery):
            if not isinstance(subquery.this, exp.Select):
                continue
            subquery_name = self._safe_name(subquery.alias_or_name or "derived_datasource")
            subquery_selects.append((subquery_name, subquery.this))
        return subquery_selects

    def _cte_projection_map(self, parsed: Any) -> Dict[str, Dict[str, str]]:
        """Map CTE output aliases to their SQL expressions."""
        from sqlglot import expressions as exp

        cte_map: Dict[str, Dict[str, str]] = {}
        for cte_name, select in self._iter_cte_selects(parsed):
            projections: Dict[str, str] = {}
            for projection in select.expressions:
                expr = projection.this if isinstance(projection, exp.Alias) else projection
                alias = projection.alias if isinstance(projection, exp.Alias) else projection.alias_or_name
                if alias:
                    projections[self._normalize_identifier(alias)] = expr.sql()
            cte_map[cte_name] = projections
        return cte_map

    def _ranked_datasource_recommendations(
        self,
        source_name: str,
        cte_name: str,
        select: Any,
        parsed: Any,
        cte_projection_map: Dict[str, Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Recommend derived data sources for rank-like window outputs."""
        from sqlglot import expressions as exp

        recommendations = []
        for projection in select.expressions:
            expr = projection.this if isinstance(projection, exp.Alias) else projection
            rank_alias = projection.alias if isinstance(projection, exp.Alias) else projection.alias_or_name
            if not rank_alias:
                continue

            for window in expr.find_all(exp.Window):
                if not self._is_rank_like_window(window):
                    continue

                rank_filters = self._filters_referencing_alias(parsed, rank_alias)
                if not rank_filters:
                    continue

                order_by = self._window_order_by(window)
                recommendations.append(
                    {
                        "source_sql_name": source_name,
                        "name": cte_name,
                        "source_cte": cte_name,
                        "reason": "rank-like window output is filtered downstream",
                        "rank_alias": self._safe_name(rank_alias),
                        "rank_filters": rank_filters,
                        "window": {
                            "function": self._window_function_name(window),
                            "partition_by": [expr.sql() for expr in window.args.get("partition_by") or []],
                            "order_by": order_by,
                        },
                        "ordering_metric_evidence": self._ordering_metric_evidence(order_by, cte_projection_map),
                        "generated_columns": [
                            self._safe_name(proj.alias or proj.alias_or_name)
                            for proj in select.expressions
                            if (proj.alias or proj.alias_or_name)
                        ],
                    }
                )
        return recommendations

    def _is_rank_like_window(self, window: Any) -> bool:
        """Return true for ranking windows that should become derived data sources."""
        from sqlglot import expressions as exp

        return isinstance(window.this, (exp.Rank, exp.DenseRank, exp.RowNumber))

    def _window_function_name(self, window: Any) -> str:
        """Return the SQL window function name."""
        function_name = getattr(window.this, "key", window.this.__class__.__name__).upper()
        return {
            "ROWNUMBER": "ROW_NUMBER",
            "DENSERANK": "DENSE_RANK",
        }.get(function_name, function_name)

    def _window_order_by(self, window: Any) -> List[Dict[str, Any]]:
        """Return ORDER BY expressions used inside a window."""
        order = window.args.get("order")
        if not order:
            return []
        order_by = []
        for ordered in order.expressions:
            order_by.append(
                {
                    "expr": ordered.this.sql(),
                    "direction": "DESC" if ordered.args.get("desc") else "ASC",
                }
            )
        return order_by

    def _filters_referencing_alias(self, parsed: Any, alias: str) -> List[str]:
        """Collect WHERE/HAVING predicates that reference an output alias."""
        filters = []
        alias_normalized = self._normalize_identifier(alias)
        for select in self._iter_selects(parsed) + [select for _, select in self._iter_cte_selects(parsed)]:
            for key in ("where", "having"):
                clause = select.args.get(key)
                predicate = getattr(clause, "this", None)
                if predicate is None:
                    continue
                if self._expression_references_column(predicate, alias_normalized):
                    predicate_sql = predicate.sql()
                    if predicate_sql not in filters:
                        filters.append(predicate_sql)
        return filters

    def _expression_references_column(self, expr: Any, column_name: str) -> bool:
        """Return true if an expression references a column by normalized name."""
        from sqlglot import expressions as exp

        return any(self._normalize_identifier(col.name) == column_name for col in expr.find_all(exp.Column))

    def _ordering_metric_evidence(
        self,
        order_by: List[Dict[str, Any]],
        cte_projection_map: Dict[str, Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Link window ordering columns back to upstream projection expressions when possible."""
        evidence = []
        for item in order_by:
            order_expr = item.get("expr", "")
            order_name = self._normalize_identifier(order_expr.split(".")[-1])
            match = ""
            for projections in cte_projection_map.values():
                if order_name in projections:
                    match = projections[order_name]
                    break
            evidence.append({"name": self._safe_name(order_name), "expression": match or order_expr})
        return evidence

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
        return self._deduplicate_relationships(relationships)

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
                    ast_relationships = self._extract_join_relationships_from_sql(sql_text, tables_lower_map)
                    relationships.extend(ast_relationships)
                    if ast_relationships:
                        continue

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

        return self._deduplicate_relationships(relationships)

    def _load_metric_mining_entries(
        self,
        sql_queries: Optional[List[str]],
        sql_entries_json: Optional[str],
        query_text: Optional[str],
        tables: Optional[List[str]],
        sample_sql_queries: int,
    ) -> List[Dict[str, Any]]:
        """Normalize direct SQL inputs or load reference SQL entries."""
        import json

        entries: List[Dict[str, Any]] = []
        sql_entries: Optional[List[Dict[str, Any]]] = None
        if sql_entries_json:
            loaded = json.loads(sql_entries_json)
            if not isinstance(loaded, list):
                raise ValueError("sql_entries_json must be a JSON array")
            sql_entries = [item for item in loaded if isinstance(item, dict)]
        if sql_entries:
            for idx, entry in enumerate(sql_entries):
                if entry.get("sql"):
                    entries.append({"name": entry.get("name") or f"sql_{idx + 1}", **entry})
        if sql_queries:
            offset = len(entries)
            entries.extend(
                {"name": f"sql_{offset + idx + 1}", "sql": sql} for idx, sql in enumerate(sql_queries) if sql
            )
        if entries:
            return entries

        if not self.agent_config:
            raise ValueError("Cannot search reference SQL without agent_config. Provide sql_queries or sql_entries.")

        from datus.storage.reference_sql.store import ReferenceSqlRAG

        sql_rag = ReferenceSqlRAG(self.agent_config, self.sub_agent_name)
        searches: List[str] = []
        if query_text:
            searches.append(query_text)
        searches.extend(f"SELECT FROM {table}" for table in (tables or []))
        if not searches:
            searches.append("SELECT")

        seen_sql = set()
        for search in searches:
            for entry in sql_rag.search_reference_sql(query_text=search, top_n=sample_sql_queries):
                sql_text = entry.get("sql", "")
                if sql_text and sql_text not in seen_sql:
                    seen_sql.add(sql_text)
                    entries.append(entry)
        return entries

    def _load_existing_metric_catalog(self, existing_metric_catalog_json: Optional[str]) -> Dict[str, Dict[str, Any]]:
        """Parse existing metric catalog JSON keyed by normalized metric name."""
        import json

        if not existing_metric_catalog_json:
            return {}

        loaded = json.loads(existing_metric_catalog_json)
        if not isinstance(loaded, list):
            raise ValueError("existing_metric_catalog_json must be a JSON array")

        catalog: Dict[str, Dict[str, Any]] = {}
        for item in loaded:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            normalized_name = self._normalize_identifier(name)
            catalog[normalized_name] = {
                "name": name,
                "type": item.get("type") or item.get("metric_type") or "",
                "description": item.get("description") or "",
                "subject_path": item.get("subject_path") or item.get("path") or "",
            }
        return catalog

    def _parse_sql(self, sql_text: str):
        """Parse one SQL string into sqlglot expressions."""
        import sqlglot

        errors = []
        for dialect in (None, "mysql", "hive", "spark"):
            try:
                parsed = sqlglot.parse(sql_text, read=dialect) if dialect else sqlglot.parse(sql_text)
                expressions = [expr for expr in parsed if expr is not None]
                if expressions:
                    return expressions
            except Exception as exc:
                dialect_name = dialect or "default"
                errors.append(f"{dialect_name}: {exc}")

        raise ValueError("Failed to parse SQL with supported dialects: " + " | ".join(errors))

    def _iter_selects(self, parsed, include_nested: bool = False) -> List[Any]:
        """Return SELECT nodes in outer-first order."""
        from sqlglot import expressions as exp

        if include_nested:
            selects = []
            for node in parsed.walk():
                if isinstance(node, exp.Select) and not any(select is node for select in selects):
                    selects.append(node)
            if isinstance(parsed, exp.Select) and not any(select is parsed for select in selects):
                selects.insert(0, parsed)
            return selects

        if isinstance(parsed, exp.Select):
            selects = [parsed]
        else:
            selects = list(parsed.find_all(exp.Select))
        return selects

    def _candidate_from_projection(
        self,
        projection: Any,
        source_name: str,
        tables: List[str],
        filters: List[str],
        dimensions: List[str],
        existing_metric_catalog: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Build a metric candidate from a SELECT projection if possible."""
        from sqlglot import expressions as exp

        expr = projection.this if isinstance(projection, exp.Alias) else projection
        alias = projection.alias if isinstance(projection, exp.Alias) else projection.alias_or_name
        name = self._metric_candidate_name(alias or expr.alias_or_name, expr)
        aggregate_classes = self._aggregate_classes()
        aggregates = list(expr.find_all(*aggregate_classes))
        has_aggregates = bool(aggregates)
        columns = {self._normalize_identifier(col.name) for col in expr.find_all(exp.Column)}
        existing_metric_names = set(existing_metric_catalog)
        referenced_metric_names = columns & existing_metric_names

        if not has_aggregates and not referenced_metric_names:
            return None
        if not has_aggregates and referenced_metric_names:
            if not columns <= existing_metric_names:
                return None
            referenced_metrics = self._referenced_metric_items(referenced_metric_names, existing_metric_catalog)
            if not self._has_real_metric_math(expr):
                return {
                    "evidence_kind": "identity_metric_reference",
                    "name": name,
                    "expression": expr.sql(),
                    "source_alias": alias or "",
                    "source_sql_name": source_name,
                    "referenced_metrics": referenced_metrics,
                    "reason": "projection references existing metric without a new business formula",
                }

        metric_type = self._classify_metric_expression(expr, name, aggregates, columns, existing_metric_names)
        measures = [
            self._measure_from_aggregate(agg, alias if len(aggregates) == 1 else "", expr) for agg in aggregates
        ]
        measures = self._deduplicate_items(measures, ["name", "agg", "expr", "filter"])

        score_reasons = []
        confidence = "medium"
        requires_name_translation = self._requires_name_translation(alias)
        if alias:
            score_reasons.append("final SELECT alias provides metric name evidence")
            if requires_name_translation:
                score_reasons.append("source alias requires LLM translation to a business-safe metric name")
        else:
            confidence = "low"
            score_reasons.append("missing final SELECT alias")
        if metric_type in {"ratio", "expr", "derived", "cumulative"}:
            score_reasons.append(f"expression shape maps to {metric_type}")
            if alias and not requires_name_translation:
                confidence = "high"
        if filters:
            score_reasons.append("historical SQL filters preserved as metric evidence")
        if dimensions:
            score_reasons.append("GROUP BY dimensions preserved as query-grain evidence")

        return {
            "name": name,
            "metric_type": metric_type,
            "expression": expr.sql(),
            "source_alias": alias or "",
            "requires_name_translation": requires_name_translation,
            "name_source": "expression_fallback"
            if requires_name_translation
            else ("source_alias" if alias else "expression"),
            "source_sql_name": source_name,
            "base_measures": measures,
            "dimensions": dimensions,
            "filters": filters,
            "tables": tables,
            "confidence": confidence,
            "score_reasons": score_reasons,
            "source_count": 1,
            "referenced_metrics": self._referenced_metric_items(referenced_metric_names, existing_metric_catalog),
        }

    def _support_measure_projection_aliases(self, select: Any) -> set[str]:
        """Return projection aliases that should stay as support measures only.

        A common BI de-duplication query projects ``COUNT(*)`` next to
        ``COUNT(DISTINCT business_key)`` to show raw rows versus deduplicated
        business entities. In that shape, the row count is useful dependency
        evidence but is usually not a standalone business metric.
        """
        aggregate_classes = self._aggregate_classes()
        count_star_aliases: set[str] = set()
        has_distinct_business_count = False

        for projection in getattr(select, "expressions", []) or []:
            expr = self._projection_expr(projection)
            aggregates = list(expr.find_all(*aggregate_classes))
            if len(aggregates) != 1 or not self._is_same_expression(expr, aggregates[0]):
                continue
            aggregate = aggregates[0]
            if self._is_count_star(aggregate):
                alias_key = self._projection_alias_key(projection)
                if alias_key:
                    count_star_aliases.add(alias_key)
                continue
            if self._is_count_distinct(aggregate):
                has_distinct_business_count = True

        if not has_distinct_business_count:
            return set()
        return count_star_aliases

    def _projection_expr(self, projection: Any) -> Any:
        """Return the expression represented by a SELECT projection."""
        from sqlglot import expressions as exp

        return projection.this if isinstance(projection, exp.Alias) else projection

    def _projection_alias_key(self, projection: Any) -> str:
        """Return the normalized key used to identify a projection alias."""
        from sqlglot import expressions as exp

        alias = projection.alias if isinstance(projection, exp.Alias) else projection.alias_or_name
        return self._safe_name(alias or projection.sql())

    def _is_count_star(self, aggregate: Any) -> bool:
        """Return true for COUNT(*)."""
        from sqlglot import expressions as exp

        return isinstance(aggregate, exp.Count) and (aggregate.this is None or isinstance(aggregate.this, exp.Star))

    def _is_count_distinct(self, aggregate: Any) -> bool:
        """Return true for COUNT(DISTINCT ...)."""
        from sqlglot import expressions as exp

        return isinstance(aggregate, exp.Count) and (
            isinstance(aggregate.this, exp.Distinct) or "DISTINCT" in aggregate.sql().upper()
        )

    def _classify_metric_expression(
        self, expr: Any, name: str, aggregates: List[Any], columns: set, existing_metric_names: set
    ) -> str:
        """Classify one projected expression into a MetricFlow metric type."""
        from sqlglot import expressions as exp

        name_lower = name.lower()
        if any(token in name_lower for token in ("running", "rolling", "cumulative", "mtd", "qtd", "ytd")):
            return "cumulative"
        if list(expr.find_all(exp.Window)):
            return "cumulative"
        if columns and columns <= existing_metric_names and not aggregates:
            return "derived"
        has_division = bool(list(expr.find_all(exp.Div)))
        has_additive_math = any(list(expr.find_all(cls)) for cls in (exp.Add, exp.Sub))
        has_math = any(list(expr.find_all(cls)) for cls in (exp.Add, exp.Sub, exp.Mul, exp.Div))
        ratio_name = any(
            token in name_lower for token in ("ratio", "rate", "share", "percent", "percentage", "per", "arpu", "arppu")
        )
        if has_division and has_additive_math:
            return "expr"
        if has_division and (len(aggregates) >= 2 or ratio_name):
            return "ratio"
        if len(aggregates) == 1 and self._is_same_expression(expr, aggregates[0]):
            return "measure_proxy"
        if len(aggregates) > 1 or has_math:
            return "expr"
        return "measure_proxy"

    def _has_real_metric_math(self, expr: Any) -> bool:
        """Return true when an existing-metric expression defines a new formula."""
        from sqlglot import expressions as exp

        math_classes = (exp.Add, exp.Sub, exp.Mul, exp.Div)
        if any(list(expr.find_all(cls)) for cls in math_classes):
            return True
        return bool(list(expr.find_all(exp.Case)))

    def _referenced_metric_items(
        self,
        referenced_metric_names: set,
        existing_metric_catalog: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return catalog entries for referenced metrics in a stable order."""
        items = []
        for normalized_name in sorted(referenced_metric_names):
            metric = existing_metric_catalog.get(normalized_name, {})
            item = {
                "name": metric.get("name") or normalized_name,
                "type": metric.get("type", ""),
                "description": metric.get("description", ""),
                "subject_path": metric.get("subject_path", ""),
            }
            items.append({key: value for key, value in item.items() if value})
        return items

    def _measure_from_aggregate(self, aggregate: Any, alias: str = "", projection_expr: Any = None) -> Dict[str, Any]:
        """Create base measure evidence from an aggregate expression."""
        from sqlglot import expressions as exp

        agg = aggregate.key.upper()
        if isinstance(aggregate, exp.Avg):
            agg = "AVERAGE"
        if isinstance(aggregate, exp.Count) and "DISTINCT" in aggregate.sql().upper():
            agg = "COUNT_DISTINCT"

        measure_expr = "1"
        inner = aggregate.this
        if inner is not None and not isinstance(inner, exp.Star):
            if isinstance(inner, exp.Distinct):
                distinct_exprs = inner.expressions
                measure_expr = distinct_exprs[0].sql() if distinct_exprs else inner.sql()
            else:
                measure_expr = inner.sql()

        if alias:
            name = self._metric_candidate_name(alias, projection_expr or aggregate)
        else:
            name = self._safe_name(f"{agg.lower()}_{measure_expr}")
        return {
            "name": name,
            "agg": agg,
            "expr": measure_expr,
            "filter": "",
            "source_alias": alias or "",
            "requires_name_translation": self._requires_name_translation(alias),
            "source_count": 1,
        }

    def _merge_metric_candidate(self, candidates: Dict[str, Dict[str, Any]], candidate: Dict[str, Any]) -> None:
        """Merge metric candidates with the same normalized name, type, and formula."""
        key = self._metric_candidate_merge_key(candidate)
        existing = candidates.get(key)
        if not existing:
            candidates[key] = candidate
            return

        existing["source_count"] += 1
        existing["source_sql_name"] = ", ".join(
            sorted(set(filter(None, existing["source_sql_name"].split(", ") + [candidate["source_sql_name"]])))
        )
        for field in ("dimensions", "filters", "tables", "score_reasons"):
            existing[field] = sorted(set(existing.get(field, []) + candidate.get(field, [])))
        for measure in candidate.get("base_measures", []):
            self._append_unique(existing["base_measures"], measure, ["name", "agg", "expr", "filter"])
        for metric in candidate.get("referenced_metrics", []):
            self._append_unique(existing["referenced_metrics"], metric, ["name", "type", "subject_path"])

    def _metric_candidate_merge_key(self, candidate: Dict[str, Any]) -> str:
        """Build a stable candidate identity without collapsing distinct formulas."""
        return "::".join(
            [
                candidate.get("name", ""),
                candidate.get("metric_type", ""),
                self._metric_candidate_formula_signature(candidate),
            ]
        )

    def _metric_candidate_formula_signature(self, candidate: Dict[str, Any]) -> str:
        """Return a deterministic signature for a metric expression and its measures."""
        measure_parts = []
        for measure in candidate.get("base_measures", []):
            measure_parts.append("|".join(str(measure.get(field, "")) for field in ("name", "agg", "expr", "filter")))
        return "||".join([candidate.get("expression", ""), *sorted(measure_parts)])

    def _merge_base_measure(self, measures: Dict[str, Dict[str, Any]], measure: Dict[str, Any]) -> None:
        """Merge repeated base measure evidence."""
        key = f"{measure['name']}::{measure['agg']}::{measure['expr']}::{measure.get('filter', '')}"
        existing = measures.get(key)
        if not existing:
            measures[key] = dict(measure)
            return
        existing["source_count"] += 1

    def _extract_join_relationships_from_sql(
        self, sql_text: str, tables_lower_map: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Extract alias-aware join relationships using sqlglot."""
        from sqlglot import expressions as exp

        relationships: List[Dict[str, Any]] = []
        if not sql_text:
            return relationships
        try:
            parsed_expressions = self._parse_sql(sql_text)
        except Exception:
            return relationships

        for parsed in parsed_expressions:
            alias_to_table = self._alias_to_table_map(parsed)
            for eq in parsed.find_all(exp.EQ):
                left = eq.left
                right = eq.right
                if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                    continue
                left_table = self._resolve_column_table(left, alias_to_table, tables_lower_map)
                right_table = self._resolve_column_table(right, alias_to_table, tables_lower_map)
                if not left_table or not right_table or left_table == right_table:
                    continue
                relationships.append(
                    {
                        "source_table": left_table,
                        "source_column": left.name,
                        "target_table": right_table,
                        "target_column": right.name,
                        "confidence": "medium",
                        "evidence": "join_pattern",
                    }
                )
        return self._deduplicate_relationships(relationships)

    def _alias_to_table_map(self, parsed: Any) -> Dict[str, str]:
        """Build alias -> table name mapping for one parsed SQL expression."""
        from sqlglot import expressions as exp

        mapping: Dict[str, str] = {}
        for table in parsed.find_all(exp.Table):
            table_name = table.name
            alias = table.alias_or_name
            if table_name:
                mapping[self._normalize_identifier(table_name)] = table_name
            if alias:
                mapping[self._normalize_identifier(alias)] = table_name
        return mapping

    def _resolve_column_table(
        self, column: Any, alias_to_table: Dict[str, str], tables_lower_map: Dict[str, str]
    ) -> Optional[str]:
        """Resolve a sqlglot Column's table alias to a requested canonical table name."""
        table_key = self._normalize_identifier(column.table)
        resolved = alias_to_table.get(table_key, column.table)
        return tables_lower_map.get(self._normalize_identifier(resolved))

    def _collect_tables(self, select: Any) -> List[str]:
        """Collect physical table names used by one SELECT."""
        from sqlglot import expressions as exp

        return sorted({table.name for table in select.find_all(exp.Table) if table.name})

    def _collect_filters(self, select: Any) -> List[str]:
        """Collect WHERE/HAVING predicates from one SELECT."""
        filters: List[str] = []
        for key in ("where", "having"):
            clause = select.args.get(key)
            if clause is not None and getattr(clause, "this", None) is not None:
                filters.append(clause.this.sql())
        return filters

    def _collect_dimensions(self, select: Any) -> List[str]:
        """Collect GROUP BY expressions from one SELECT."""
        group = select.args.get("group")
        if not group:
            return []
        return [expr.sql() for expr in group.expressions]

    def _safe_name(self, raw: str) -> str:
        """Convert a SQL alias/expression into a MetricFlow-compatible snake_case name."""
        import re

        value = raw.strip().strip('"`[]').lower()
        value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
        if not value:
            value = "metric_candidate"
        if value[0].isdigit():
            value = f"metric_{value}"
        return value

    def _metric_candidate_name(self, alias: str, expr: Any) -> str:
        """Prefer valid SQL aliases, then derive a technical fallback from the expression.

        The fallback is only for deterministic grouping in this read-only tool.
        If the source alias is not already MetricFlow-safe, callers should use
        `source_alias` plus SQL/question context to choose the final business
        metric name.
        """
        if alias:
            alias_name = self._safe_name(alias)
            if alias_name != "metric_candidate":
                return alias_name
        return self._name_from_expression(expr)

    def _requires_name_translation(self, alias: str) -> bool:
        """Return true when the original SQL alias should be named by the LLM."""
        if not alias:
            return False
        return self._safe_name(alias) == "metric_candidate"

    def _name_from_expression(self, expr: Any) -> str:
        """Derive a deterministic MetricFlow-safe name from a SQL expression."""
        aggregate_classes = self._aggregate_classes()
        aggregates = list(expr.find_all(*aggregate_classes))
        if len(aggregates) == 1 and self._is_same_expression(expr, aggregates[0]):
            return self._name_from_aggregate(aggregates[0])
        return self._safe_name(expr.sql())

    def _name_from_aggregate(self, aggregate: Any) -> str:
        """Derive a deterministic name from one aggregate expression."""
        from sqlglot import expressions as exp

        agg = aggregate.key.lower()
        if isinstance(aggregate, exp.Avg):
            agg = "average"
        elif isinstance(aggregate, exp.Count) and "DISTINCT" in aggregate.sql().upper():
            agg = "count_distinct"

        inner = aggregate.this
        if inner is None or isinstance(inner, exp.Star):
            return "count_rows" if agg == "count" else self._safe_name(f"{agg}_value")
        if isinstance(inner, exp.Distinct):
            distinct_exprs = inner.expressions
            inner = distinct_exprs[0] if distinct_exprs else inner
        return self._safe_name(f"{agg}_{inner.sql()}")

    def _normalize_identifier(self, value: str) -> str:
        """Normalize SQL identifiers for comparisons."""
        return (value or "").strip().strip('"`[]').lower()

    def _is_same_expression(self, left: Any, right: Any) -> bool:
        """Compare SQL expressions by rendered SQL text."""
        return left.sql() == right.sql()

    def _deduplicate_items(self, items: List[Dict[str, Any]], key_fields: List[str]) -> List[Dict[str, Any]]:
        """Deduplicate dictionaries by selected fields."""
        result: List[Dict[str, Any]] = []
        for item in items:
            self._append_unique(result, item, key_fields)
        return result

    def _append_unique(self, items: List[Dict[str, Any]], item: Dict[str, Any], key_fields: List[str]) -> None:
        """Append dict item if the selected-field key is not already present."""
        key = tuple(item.get(field) for field in key_fields)
        if all(tuple(existing.get(field) for field in key_fields) != key for existing in items):
            items.append(dict(item))

    def _is_blocked_direct_candidate(
        self,
        candidate: Dict[str, Any],
        blocked_candidates: List[Dict[str, Any]],
    ) -> bool:
        """Return true if any source merged into the candidate requires derived modeling first."""
        candidate_sources = self._source_sql_name_set(candidate.get("source_sql_name", ""))
        for blocked in blocked_candidates:
            if blocked.get("name") != candidate.get("name") or blocked.get("metric_type") != candidate.get(
                "metric_type"
            ):
                continue
            if self._metric_candidate_formula_signature(blocked) != self._metric_candidate_formula_signature(candidate):
                continue
            blocked_sources = self._source_sql_name_set(blocked.get("source_sql_name", ""))
            if candidate_sources & blocked_sources:
                return True
        return False

    def _source_sql_name_set(self, source_sql_name: str) -> set:
        """Split display source names back into comparable source identifiers."""
        return {name.strip() for name in (source_sql_name or "").split(",") if name.strip()}

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
