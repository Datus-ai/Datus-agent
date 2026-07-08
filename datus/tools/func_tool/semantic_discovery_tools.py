# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Semantic Discovery Tools

This module provides read-only discovery tools for semantic-layer generation,
including table relationships, column usage evidence, and metric candidates
mined from historical SQL.
"""

import json
import time
from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal
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

    permission_category: str = "semantic_tools"

    _AGGREGATE_CLASSES = ()

    def __init__(self, db_tool: DBFuncTool, enable_semantic_model_profiler: bool = False):
        """
        Initialize semantic discovery tools.

        Args:
            db_tool: Database function tool instance for accessing database info
            enable_semantic_model_profiler: Whether to expose the optional
                semantic SQL history profiler tool.
        """
        self.db_tool = db_tool
        self.agent_config = db_tool.agent_config
        self.sub_agent_name = db_tool.sub_agent_name
        self.enable_semantic_model_profiler = enable_semantic_model_profiler

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
        if self.enable_semantic_model_profiler:
            methods_to_convert.insert(3, self.profile_semantic_model_evidence)

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
        1. Filter operators from parsed SQL predicates
        2. Function predicates from parsed SQL expressions
        3. Common redacted filter patterns and usage frequency

        Use this tool when generating semantic models to understand
        how one table's columns are typically queried and filtered. For
        multi-table semantic model evidence or sampled data distributions, use
        the skill-gated `profile_semantic_model_evidence` tool.

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
                        "common_filters": ["status = <REDACTED>", "status IN (<REDACTED>)"],
                        "usage_count": 45,
                        "usage_description": "Commonly filtered with =, IN"
                    },
                    "normalized_name": {
                        "operators": ["="],
                        "functions": ["LOWER"],
                        "common_filters": ["LOWER(name) = '<REDACTED>'"],
                        "usage_count": 23,
                        "usage_description": "Commonly filtered with =. Function predicates: LOWER"
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

            from datus.storage.reference_sql.store import ReferenceSqlRAG

            # Get table schema to know which columns exist
            schema_result = self.db_tool.describe_table(table_name, catalog, database, schema_name)
            if not schema_result.success:
                return FuncToolResult(success=0, error=f"Failed to get table schema: {schema_result.error}")

            # describe_table returns {"columns": [...], "table": {...}}
            table_columns = schema_result.result.get("columns", []) if isinstance(schema_result.result, dict) else []
            all_columns = [
                str(col.get("name") or "") for col in table_columns if isinstance(col, dict) and col.get("name")
            ]
            target_columns = [str(col) for col in (columns if columns else all_columns) if col]

            # Search for SQL queries containing the table
            sql_rag = ReferenceSqlRAG(self.agent_config, self.sub_agent_name)
            search_results = sql_rag.search_reference_sql(
                query_text=f"SELECT FROM {table_name}", top_n=sample_sql_queries
            )

            logger.info(f"Found {len(search_results)} historical SQL queries for table {table_name}")
            entries = [
                {
                    "name": entry.get("name") or entry.get("summary") or entry.get("filepath") or f"sql_{idx + 1}",
                    **entry,
                    "sql": str(entry.get("sql") or "").strip(),
                }
                for idx, entry in enumerate(search_results)
                if str(entry.get("sql") or "").strip()
            ]
            table_evidence, parse_errors = self._semantic_profile_sql_evidence(entries, [table_name], 1)
            table_profile = self._semantic_profile_table_evidence_for(table_evidence, table_name)
            result_patterns = self._column_usage_patterns_from_semantic_profile(
                table_profile.get("field_usage_statistics", {}),
                target_columns,
            )

            logger.info(f"Analyzed {len(result_patterns)} columns with usage patterns")

            result = {
                "column_patterns": result_patterns,
                "summary": f"Analyzed {len(result_patterns)} columns from {len(search_results)} SQL queries",
            }
            if parse_errors:
                result["parse_errors"] = parse_errors[:5]
            return FuncToolResult(result=result)

        except Exception as e:
            logger.exception("Error analyzing column usage patterns")
            return FuncToolResult(success=0, error=str(e))

    def profile_semantic_model_evidence(
        self,
        sql_queries: Optional[List[str]] = None,
        sql_entries_json: Optional[str] = "",
        query_text: Optional[str] = "",
        tables: Optional[List[str]] = None,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
        profile_mode: str = "sql_only",
        sample_sql_queries: int = 50,
        max_tables: int = 8,
        max_columns_per_table: int = 10,
        top_n: int = 5,
        max_profile_seconds: int = 30,
    ) -> FuncToolResult:
        """
        Build semantic-model evidence from historical SQL and optional table profiling.

        This read-only tool is intended for semantic model generation when the
        `semantic-sql-history-profiler` skill is loaded. It mines the provided
        SQL for joins, filters, grouping fields, and aggregate candidates, then
        optionally samples bounded column distributions from the connected DB.

        Args:
            sql_queries: Raw historical SQL statements to analyze.
            sql_entries_json: JSON array of dictionaries with `sql` plus optional
                `name`, `question`, or `summary`.
            query_text: Reference SQL search text when direct SQL is not provided.
            tables: Optional table allowlist. Also used as profiling targets.
            catalog: Optional catalog override for describe/profile calls.
            database: Optional database override for describe/profile calls.
            schema_name: Optional schema override for describe/profile calls.
            profile_mode: `none`/`sql_only` skips DB profiling; `lightweight`
                profiles fields seen in SQL evidence; `deep` may also profile
                schema columns up to max_columns_per_table.
            sample_sql_queries: Maximum reference SQL rows to inspect.
            max_tables: Maximum tables to include.
            max_columns_per_table: Maximum columns profiled per table.
            top_n: Maximum categorical top values per column.
            max_profile_seconds: Best-effort wall-clock budget for DB profiling.

        Returns:
            FuncToolResult with table-level evidence. Values sampled from data are
            evidence to summarize compactly in YAML descriptions, not to copy wholesale.
        """
        try:
            mode = (profile_mode or "sql_only").strip().lower()
            has_sql_seed = bool(sql_queries or sql_entries_json or query_text)
            entries = (
                self._load_metric_mining_entries(sql_queries, sql_entries_json, query_text, tables, sample_sql_queries)
                if has_sql_seed
                else []
            )
            table_evidence, parse_errors = self._semantic_profile_sql_evidence(entries, tables, max_tables)

            data_profiled = mode in {"lightweight", "deep"}
            if data_profiled:
                self._ensure_semantic_profile_tables(table_evidence, tables, max_tables)
                self._attach_table_distribution_profiles(
                    table_evidence=table_evidence,
                    mode=mode,
                    catalog=catalog or "",
                    database=database or "",
                    schema_name=schema_name or "",
                    max_tables=max_tables,
                    max_columns_per_table=max_columns_per_table,
                    top_n=top_n,
                    max_profile_seconds=max_profile_seconds,
                )

            return FuncToolResult(
                result={
                    "summary": (
                        f"Profiled semantic evidence for {len(table_evidence)} table(s) "
                        f"from {len(entries)} SQL entr{'y' if len(entries) == 1 else 'ies'}"
                    ),
                    "profile_mode": mode,
                    "data_profiled": data_profiled,
                    "tables": table_evidence,
                    "parse_errors": parse_errors[:5],
                    "yaml_guidance": (
                        "Keep generated YAML concise: use profiling evidence to choose identifiers, "
                        "measures, dimensions, and time columns; include compact distribution notes "
                        "in descriptions when useful, such as observed min/max, percentiles, "
                        "null rate, date span/freshness/duration, low-cardinality distinct counts, "
                        "stable enum mappings, referential coverage, and common business filter "
                        "templates. Do not dump profiling JSON, long top-N lists, or long filter examples."
                    ),
                }
            )
        except Exception as e:
            logger.exception("Error profiling semantic model evidence")
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
            metric_generation_skips: List[Dict[str, Any]] = []
            literal_mappings: List[Dict[str, Any]] = []
            time_grain_evidence: List[Dict[str, Any]] = []
            post_aggregation_constraints: List[Dict[str, Any]] = []

            for idx, entry in enumerate(entries):
                sql_text = entry.get("sql", "")
                source_name = entry.get("name") or entry.get("summary") or entry.get("filepath") or f"sql_{idx + 1}"
                source_context = self._metric_source_context(entry)
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
                found_candidate = False
                modeling_analysis = self._analyze_query_modeling(parsed_expressions, source_name)
                if modeling_analysis["derived_datasource_recommendations"]:
                    derived_datasource_recommendations.extend(modeling_analysis["derived_datasource_recommendations"])
                    metric_generation_skips.extend(modeling_analysis["metric_generation_skips"])
                preservation_evidence = self._extract_semantic_preservation_evidence(
                    parsed_expressions,
                    source_name,
                )
                literal_mappings.extend(preservation_evidence["literal_mappings"])
                time_grain_evidence.extend(preservation_evidence["time_grain_evidence"])
                post_aggregation_constraints.extend(preservation_evidence["post_aggregation_constraints"])
                pop_candidates = self._period_over_period_metric_candidates(
                    parsed_expressions,
                    source_name,
                    source_context,
                    existing_metric_catalog,
                )
                for candidate in pop_candidates["base_metric_candidates"]:
                    entry_has_metric_evidence = True
                    found_candidate = True
                    entry_candidates.append(candidate)
                    self._merge_metric_candidate(metric_candidates, candidate)
                    for measure in candidate.get("base_measures", []):
                        self._merge_base_measure(base_measures, measure)
                for candidate in pop_candidates["period_metric_candidates"]:
                    entry_has_metric_evidence = True
                    found_candidate = True
                    entry_candidates.append(candidate)
                    self._merge_metric_candidate(metric_candidates, candidate)
                window_candidates = self._window_aggregate_metric_candidates(
                    parsed_expressions,
                    source_name,
                    source_context,
                    existing_metric_catalog,
                )
                for candidate in window_candidates["base_metric_candidates"]:
                    entry_has_metric_evidence = True
                    found_candidate = True
                    entry_candidates.append(candidate)
                    self._merge_metric_candidate(metric_candidates, candidate)
                    for measure in candidate.get("base_measures", []):
                        self._merge_base_measure(base_measures, measure)
                for candidate in window_candidates["window_metric_candidates"]:
                    entry_has_metric_evidence = True
                    found_candidate = True
                    entry_candidates.append(candidate)
                    self._merge_metric_candidate(metric_candidates, candidate)
                    for measure in candidate.get("base_measures", []):
                        self._merge_base_measure(base_measures, measure)

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
                                source_context=source_context,
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

                entry_has_llm_review_candidates = any(
                    candidate.get("candidate_classification") == "llm_review_candidate"
                    for candidate in entry_candidates
                )
                entry_has_direct_candidates = any(
                    candidate.get("candidate_classification") != "llm_review_candidate"
                    for candidate in entry_candidates
                )
                classification = self._classify_source_query(
                    has_direct_candidates=entry_has_direct_candidates,
                    has_llm_review_candidates=entry_has_llm_review_candidates,
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
            llm_review_candidates = [
                candidate
                for candidate in candidates
                if candidate.get("candidate_classification") == "llm_review_candidate"
            ]
            direct_candidates = [
                candidate
                for candidate in candidates
                if candidate.get("metric_type") != "derived"
                and candidate.get("candidate_classification") != "llm_review_candidate"
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
                    "llm_review_candidates": llm_review_candidates,
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
                    "metric_generation_skips": metric_generation_skips,
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

    def _semantic_profile_sql_evidence(
        self,
        entries: List[Dict[str, Any]],
        table_filter: Optional[List[str]],
        max_tables: int,
    ) -> tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        """Mine table-level semantic modeling evidence from SQL ASTs."""
        from sqlglot import expressions as exp

        allowed_tables = {self._normalize_identifier(table.split(".")[-1]) for table in table_filter or [] if table}
        table_stats: Dict[str, Dict[str, Any]] = defaultdict(self._new_semantic_profile_table)
        parse_errors: List[Dict[str, Any]] = []

        for idx, entry in enumerate(entries):
            sql_text = str(entry.get("sql") or "").strip()
            source_name = entry.get("name") or entry.get("summary") or entry.get("filepath") or f"sql_{idx + 1}"
            if not sql_text:
                continue
            try:
                parsed_expressions = self._parse_sql(sql_text)
            except Exception as exc:
                parse_errors.append({"source_sql_name": source_name, "error": str(exc)})
                continue

            source = {
                "source_sql_name": str(source_name),
                "question": self._clip_profile_text(str(entry.get("question") or ""), 120),
            }
            for parsed in parsed_expressions:
                cte_names = self._profile_cte_names(parsed)
                alias_to_table = self._profile_alias_to_table_map(parsed, cte_names)
                for select in self._iter_selects(parsed, include_nested=True):
                    select_tables = self._profile_select_tables(select, cte_names)
                    if not select_tables:
                        select_tables = set(alias_to_table.values())
                    if allowed_tables:
                        select_tables = {
                            table
                            for table in select_tables
                            if self._normalize_identifier(table.split(".")[-1]) in allowed_tables
                        }
                    for table in select_tables:
                        self._add_semantic_profile_source(table_stats[table], source)

                    for projection in select.expressions:
                        for column in projection.find_all(exp.Column):
                            table = self._profile_column_table(column, alias_to_table, select_tables)
                            if table:
                                table_stats[table]["fields"][column.name]["selected_count"] += 1

                    self._collect_semantic_profile_groups(select, table_stats, alias_to_table, select_tables)
                    self._collect_semantic_profile_filters(select, table_stats, alias_to_table, select_tables)
                    self._collect_semantic_profile_aggregates(select, table_stats, alias_to_table, select_tables)
                    self._collect_semantic_profile_joins(select, table_stats, alias_to_table)

        sorted_items = sorted(
            table_stats.items(),
            key=lambda item: (-len(item[1]["source_queries"]), item[0]),
        )[: max(max_tables, 1)]
        return {table: self._finalize_semantic_profile_table(stats) for table, stats in sorted_items}, parse_errors

    def _semantic_profile_table_evidence_for(
        self,
        table_evidence: Dict[str, Dict[str, Any]],
        table_name: str,
    ) -> Dict[str, Any]:
        normalized = self._normalize_identifier(table_name.split(".")[-1])
        for candidate, evidence in table_evidence.items():
            if self._normalize_identifier(candidate.split(".")[-1]) == normalized:
                return evidence
        return {}

    def _column_usage_patterns_from_semantic_profile(
        self,
        field_usage_statistics: Dict[str, Dict[str, Any]],
        target_columns: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        fields_by_name = {
            self._normalize_identifier(field_name): field_stats
            for field_name, field_stats in field_usage_statistics.items()
        }
        result: Dict[str, Dict[str, Any]] = {}
        for column in target_columns:
            field_stats = fields_by_name.get(self._normalize_identifier(column))
            if not field_stats:
                continue
            operators = list(field_stats.get("operators") or [])
            functions = list(field_stats.get("functions") or [])
            common_filters = list(field_stats.get("common_filters") or [])[:3]
            usage_count = int(field_stats.get("filter_count") or 0)
            if usage_count <= 0 and not operators and not functions and not common_filters:
                continue

            desc_parts = []
            if operators:
                desc_parts.append("Commonly filtered with " + ", ".join(operators))
            if functions:
                desc_parts.append("Function predicates: " + ", ".join(functions))
            if common_filters:
                desc_parts.append("Example filters: " + " | ".join(common_filters[:2]))

            result[column] = {
                "operators": operators,
                "functions": functions,
                "common_filters": common_filters,
                "usage_count": usage_count,
                "usage_description": ". ".join(desc_parts) if desc_parts else "Used in filter predicates",
            }
        return result

    def _ensure_semantic_profile_tables(
        self,
        table_evidence: Dict[str, Dict[str, Any]],
        tables: Optional[List[str]],
        max_tables: int,
    ) -> None:
        """Add explicit table targets so deep profiling can run without SQL evidence."""
        for table in (tables or [])[: max(max_tables, 1)]:
            table_name = str(table or "").strip()
            if table_name and table_name not in table_evidence:
                table_evidence[table_name] = self._empty_semantic_profile_table()

    def _empty_semantic_profile_table(self) -> Dict[str, Any]:
        return {
            "query_count": 0,
            "source_queries": [],
            "field_usage_statistics": {},
            "common_filter_conditions": [],
            "common_business_filter_templates": [],
            "join_relationships": [],
            "aggregate_expressions": [],
            "group_by_expressions": [],
        }

    def _new_semantic_profile_table(self) -> Dict[str, Any]:
        return {
            "source_queries": {},
            "fields": defaultdict(self._new_semantic_profile_field),
            "common_filters": Counter(),
            "business_filter_templates": Counter(),
            "join_relationships": Counter(),
            "aggregate_expressions": Counter(),
            "group_by_expressions": Counter(),
        }

    def _new_semantic_profile_field(self) -> Dict[str, Any]:
        return {
            "selected_count": 0,
            "filter_count": 0,
            "group_by_count": 0,
            "aggregate_count": 0,
            "operators": Counter(),
            "functions": Counter(),
            "common_filters": Counter(),
        }

    def _add_semantic_profile_source(self, stats: Dict[str, Any], source: Dict[str, str]) -> None:
        source_name = source["source_sql_name"]
        if source_name not in stats["source_queries"] and len(stats["source_queries"]) < 5:
            stats["source_queries"][source_name] = source

    def _profile_cte_names(self, parsed: Any) -> set[str]:
        from sqlglot import expressions as exp

        return {self._normalize_identifier(cte.alias) for cte in parsed.find_all(exp.CTE) if cte.alias}

    def _profile_alias_to_table_map(self, parsed: Any, cte_names: set[str]) -> Dict[str, str]:
        from sqlglot import expressions as exp

        mapping: Dict[str, str] = {}
        for table in parsed.find_all(exp.Table):
            table_name = self._profile_table_name(table)
            if not table_name or self._normalize_identifier(table.name) in cte_names:
                continue
            mapping[self._normalize_identifier(table.name)] = table_name
            mapping[self._normalize_identifier(table_name)] = table_name
            if table.alias_or_name:
                mapping[self._normalize_identifier(table.alias_or_name)] = table_name
        return mapping

    def _profile_table_name(self, table: Any) -> str:
        parts = [part for part in (getattr(table, "catalog", ""), getattr(table, "db", ""), table.name) if part]
        return ".".join(str(part).strip('"`[]') for part in parts if str(part).strip('"`[]'))

    def _profile_select_tables(self, select: Any, cte_names: set[str]) -> set[str]:
        from sqlglot import expressions as exp

        tables = set()
        for table in select.find_all(exp.Table):
            table_name = self._profile_table_name(table)
            if table_name and self._normalize_identifier(table.name) not in cte_names:
                tables.add(table_name)
        return tables

    def _profile_column_table(
        self,
        column: Any,
        alias_to_table: Dict[str, str],
        select_tables: set[str],
    ) -> Optional[str]:
        table_key = self._normalize_identifier(column.table)
        if table_key:
            return alias_to_table.get(table_key)
        if len(select_tables) == 1:
            return next(iter(select_tables))
        return None

    def _profile_tables_for_expression(
        self,
        expression: Any,
        alias_to_table: Dict[str, str],
        select_tables: set[str],
    ) -> set[str]:
        from sqlglot import expressions as exp

        tables = {
            table
            for column in expression.find_all(exp.Column)
            if (table := self._profile_column_table(column, alias_to_table, select_tables))
        }
        if tables:
            return tables
        return set(select_tables) if len(select_tables) == 1 else set()

    def _collect_semantic_profile_groups(
        self,
        select: Any,
        table_stats: Dict[str, Dict[str, Any]],
        alias_to_table: Dict[str, str],
        select_tables: set[str],
    ) -> None:
        from sqlglot import expressions as exp

        group = select.args.get("group")
        if not group:
            return
        for expression in group.expressions:
            expression_sql = self._sanitize_profile_sql(expression.sql())
            for table in self._profile_tables_for_expression(expression, alias_to_table, select_tables):
                table_stats[table]["group_by_expressions"][expression_sql] += 1
            for column in expression.find_all(exp.Column):
                table = self._profile_column_table(column, alias_to_table, select_tables)
                if table:
                    table_stats[table]["fields"][column.name]["group_by_count"] += 1

    def _collect_semantic_profile_filters(
        self,
        select: Any,
        table_stats: Dict[str, Dict[str, Any]],
        alias_to_table: Dict[str, str],
        select_tables: set[str],
    ) -> None:
        from sqlglot import expressions as exp

        for clause_key in ("where", "having", "qualify"):
            clause = select.args.get(clause_key)
            predicate_root = getattr(clause, "this", None)
            if predicate_root is None:
                continue
            for predicate in self._semantic_profile_filter_predicates(predicate_root):
                condition = self._sanitize_profile_sql(predicate.sql())
                operator = self._semantic_profile_operator(predicate)
                function_names = self._semantic_profile_function_names(predicate)
                business_filter_templates = self._semantic_profile_business_filter_templates(
                    predicate=predicate,
                    alias_to_table=alias_to_table,
                    select_tables=select_tables,
                    condition_template=condition,
                    operator=operator,
                    function_names=function_names,
                )
                for table, template in business_filter_templates:
                    table_stats[table]["business_filter_templates"][
                        json.dumps(template, ensure_ascii=False, sort_keys=True)
                    ] += 1
                for table in self._profile_tables_for_expression(predicate, alias_to_table, select_tables):
                    table_stats[table]["common_filters"][condition] += 1
                for column in predicate.find_all(exp.Column):
                    table = self._profile_column_table(column, alias_to_table, select_tables)
                    if not table:
                        continue
                    field = table_stats[table]["fields"][column.name]
                    field["filter_count"] += 1
                    if operator:
                        field["operators"][operator] += 1
                    for function_name in function_names:
                        field["functions"][function_name] += 1
                    field["common_filters"][condition] += 1

    def _collect_semantic_profile_aggregates(
        self,
        select: Any,
        table_stats: Dict[str, Dict[str, Any]],
        alias_to_table: Dict[str, str],
        select_tables: set[str],
    ) -> None:
        from sqlglot import expressions as exp

        for aggregate in select.find_all(*self._aggregate_classes()):
            aggregate_sql = self._sanitize_profile_sql(aggregate.sql())
            aggregate_tables = (
                self._profile_tables_for_expression(aggregate, alias_to_table, select_tables) or select_tables
            )
            for table in aggregate_tables:
                table_stats[table]["aggregate_expressions"][aggregate_sql] += 1
            for column in aggregate.find_all(exp.Column):
                table = self._profile_column_table(column, alias_to_table, select_tables)
                if table:
                    table_stats[table]["fields"][column.name]["aggregate_count"] += 1

    def _collect_semantic_profile_joins(
        self,
        select: Any,
        table_stats: Dict[str, Dict[str, Any]],
        alias_to_table: Dict[str, str],
    ) -> None:
        from sqlglot import expressions as exp

        for eq in select.find_all(exp.EQ):
            left = eq.left
            right = eq.right
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                continue
            left_table = alias_to_table.get(self._normalize_identifier(left.table))
            right_table = alias_to_table.get(self._normalize_identifier(right.table))
            if not left_table or not right_table or left_table == right_table:
                continue
            relationship = json.dumps(
                {
                    "source_table": left_table,
                    "source_column": left.name,
                    "target_table": right_table,
                    "target_column": right.name,
                    "evidence": "historical_sql_join",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            table_stats[left_table]["join_relationships"][relationship] += 1
            table_stats[right_table]["join_relationships"][relationship] += 1

    def _semantic_profile_filter_predicates(self, root: Any) -> List[Any]:
        from sqlglot import expressions as exp

        operator_classes = tuple(self._semantic_profile_operator_map().keys())
        predicates = []
        covered_nodes = set()
        for node in root.walk():
            if isinstance(node, operator_classes):
                predicates.append(node)
                covered_nodes.update(id(child) for child in node.walk())
        for node in root.walk():
            if isinstance(node, exp.Func) and id(node) not in covered_nodes:
                predicates.append(node)
        return predicates

    def _semantic_profile_operator_map(self) -> Dict[type, str]:
        from sqlglot import expressions as exp

        mapping: Dict[type, str] = {}
        for class_name, operator in (
            ("EQ", "="),
            ("NEQ", "!="),
            ("GT", ">"),
            ("GTE", ">="),
            ("LT", "<"),
            ("LTE", "<="),
            ("In", "IN"),
            ("Like", "LIKE"),
            ("ILike", "ILIKE"),
            ("Between", "BETWEEN"),
            ("Is", "IS"),
            ("RegexpLike", "REGEXP"),
        ):
            expression_class = getattr(exp, class_name, None)
            if expression_class is not None:
                mapping[expression_class] = operator
        return mapping

    def _semantic_profile_operator(self, predicate: Any) -> str:
        for expression_class, operator in self._semantic_profile_operator_map().items():
            if isinstance(predicate, expression_class):
                return operator
        return ""

    def _semantic_profile_function_names(self, expression: Any) -> List[str]:
        from sqlglot import expressions as exp

        names = set()
        for func in expression.find_all(exp.Func):
            if isinstance(func, exp.Anonymous):
                name = func.name or func.this
            else:
                sql_name = getattr(func, "sql_name", None)
                if callable(sql_name):
                    name = sql_name()
                else:
                    name = getattr(func, "key", "") or func.__class__.__name__
            if name:
                names.add(str(name).upper())
        return sorted(names)

    def _semantic_profile_business_filter_templates(
        self,
        predicate: Any,
        alias_to_table: Dict[str, str],
        select_tables: set[str],
        condition_template: str,
        operator: str,
        function_names: List[str],
    ) -> List[tuple[str, Dict[str, Any]]]:
        from sqlglot import expressions as exp

        fields_by_table: Dict[str, set[str]] = defaultdict(set)
        for column in predicate.find_all(exp.Column):
            table = self._profile_column_table(column, alias_to_table, select_tables)
            if table:
                fields_by_table[table].add(column.name)
        if not fields_by_table:
            return []

        literal_values = self._semantic_profile_literal_values(predicate)
        usage_kind = self._semantic_profile_filter_usage_kind(operator, function_names)
        templates = []
        for table, fields in fields_by_table.items():
            template = {
                "condition_template": condition_template,
                "fields": sorted(fields),
            }
            if operator:
                template["operator"] = operator
            if function_names:
                template["functions"] = function_names
            if literal_values:
                template["literal_values"] = literal_values
            if usage_kind:
                template["usage_kind"] = usage_kind
            templates.append((table, template))
        return templates

    def _semantic_profile_literal_values(self, expression: Any, max_values: int = 5) -> List[str]:
        from sqlglot import expressions as exp

        values = []
        seen = set()
        for literal in expression.find_all(exp.Literal):
            raw = literal.this
            if raw is None:
                continue
            value = str(raw).strip()
            if not value or len(value) > 40 or value in seen:
                continue
            seen.add(value)
            values.append(self._clip_profile_text(value, 40))
            if len(values) >= max_values:
                break
        return values

    def _semantic_profile_filter_usage_kind(self, operator: str, function_names: List[str]) -> str:
        if function_names:
            return "function_filter"
        if operator in {"LIKE", "ILIKE", "REGEXP"}:
            return "text_search"
        if operator in {"=", "!=", "IN"}:
            return "categorical_filter"
        if operator in {">", ">=", "<", "<=", "BETWEEN"}:
            return "range_filter"
        if operator == "IS":
            return "null_check"
        return ""

    def _finalize_semantic_profile_table(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        fields = {}
        for field, field_stats in sorted(
            stats["fields"].items(),
            key=lambda item: (-self._semantic_profile_field_usage_count(item[1]), item[0]),
        ):
            usage_count = self._semantic_profile_field_usage_count(field_stats)
            if usage_count <= 0:
                continue
            fields[field] = {
                "usage_count": usage_count,
                "selected_count": field_stats["selected_count"],
                "filter_count": field_stats["filter_count"],
                "group_by_count": field_stats["group_by_count"],
                "aggregate_count": field_stats["aggregate_count"],
                "operators": [item for item, _count in field_stats["operators"].most_common()],
                "functions": [item for item, _count in field_stats["functions"].most_common()],
                "common_filters": [item for item, _count in field_stats["common_filters"].most_common(3)],
            }
        return {
            "query_count": len(stats["source_queries"]),
            "source_queries": list(stats["source_queries"].values()),
            "field_usage_statistics": fields,
            "common_filter_conditions": self._counter_to_profile_list(stats["common_filters"], "condition", 8),
            "common_business_filter_templates": self._counter_json_to_profile_list(
                stats["business_filter_templates"], 8
            ),
            "join_relationships": self._counter_json_to_profile_list(stats["join_relationships"], 12),
            "aggregate_expressions": self._counter_to_profile_list(stats["aggregate_expressions"], "expression", 8),
            "group_by_expressions": self._counter_to_profile_list(stats["group_by_expressions"], "expression", 8),
        }

    def _semantic_profile_field_usage_count(self, stats: Dict[str, Any]) -> int:
        return (
            int(stats["selected_count"])
            + int(stats["filter_count"])
            + int(stats["group_by_count"])
            + int(stats["aggregate_count"])
        )

    def _counter_to_profile_list(self, counter: Counter, value_key: str, limit: int) -> List[Dict[str, Any]]:
        return [{value_key: value, "count": count} for value, count in counter.most_common(limit)]

    def _counter_json_to_profile_list(self, counter: Counter, limit: int) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for value, count in counter.most_common(limit):
            try:
                item = json.loads(value)
            except json.JSONDecodeError:
                item = {"evidence": value}
            item["count"] = count
            items.append(item)
        return items

    def _attach_table_distribution_profiles(
        self,
        table_evidence: Dict[str, Dict[str, Any]],
        mode: str,
        catalog: str,
        database: str,
        schema_name: str,
        max_tables: int,
        max_columns_per_table: int,
        top_n: int,
        max_profile_seconds: int,
    ) -> None:
        """Attach bounded data-distribution profiles to table evidence."""
        started_at = time.monotonic()
        for table_name, evidence in list(table_evidence.items())[: max(max_tables, 1)]:
            if time.monotonic() - started_at > max_profile_seconds:
                evidence["data_profile_skipped"] = "max_profile_seconds exceeded"
                continue

            describe = self.db_tool.describe_table(table_name, catalog, database, schema_name)
            if not describe.success:
                evidence["data_profile_error"] = describe.error or "describe_table failed"
                continue

            columns = (describe.result or {}).get("columns") if isinstance(describe.result, dict) else []
            columns = [col for col in columns if isinstance(col, dict) and col.get("name")]
            selected = self._select_columns_for_distribution_profile(
                evidence=evidence,
                columns=columns,
                mode=mode,
                max_columns=max_columns_per_table,
            )
            table_ref = self._profile_table_reference(table_name, catalog, database, schema_name)
            profile = {
                "profile_mode": mode,
                "table_reference": table_ref,
                "columns": {},
            }
            row_count = self._run_profile_scalar_query(f"SELECT COUNT(*) AS row_count FROM {table_ref}", database)
            if row_count:
                profile["row_count"] = row_count.get("row_count")

            for column in selected:
                if time.monotonic() - started_at > max_profile_seconds:
                    profile["partial"] = True
                    break
                column_name = str(column.get("name") or "")
                column_type = str(column.get("type") or "")
                kind = self._profile_column_kind(column_type)
                column_profile = self._profile_single_column(
                    table_ref=table_ref,
                    column_name=column_name,
                    column_type=column_type,
                    kind=kind,
                    database=database,
                    top_n=top_n,
                )
                profile["columns"][column_name] = column_profile

            duration_profiles = self._profile_date_duration_pairs(
                table_ref=table_ref,
                columns=columns,
                database=database,
                deadline=started_at + max_profile_seconds,
            )
            if duration_profiles:
                profile["date_duration_profiles"] = duration_profiles

            join_profiles = self._profile_join_relationship_profiles(
                relationships=evidence.get("join_relationships") or [],
                catalog=catalog,
                database=database,
                schema_name=schema_name,
                deadline=started_at + max_profile_seconds,
            )
            if join_profiles:
                profile["join_relationship_profiles"] = join_profiles

            evidence["data_distribution_profile"] = profile

    def _select_columns_for_distribution_profile(
        self,
        evidence: Dict[str, Any],
        columns: List[Dict[str, Any]],
        mode: str,
        max_columns: int,
    ) -> List[Dict[str, Any]]:
        by_name = {str(col.get("name")): col for col in columns}
        field_usage = evidence.get("field_usage_statistics") or {}
        selected_names = [
            name
            for name, stats in sorted(
                field_usage.items(),
                key=lambda item: (
                    -int(item[1].get("filter_count", 0)),
                    -int(item[1].get("group_by_count", 0)),
                    -int(item[1].get("aggregate_count", 0)),
                    -int(item[1].get("usage_count", 0)),
                    item[0],
                ),
            )
            if name in by_name
        ]
        if mode == "deep":
            selected_set = set(selected_names)
            for col in columns:
                name = str(col.get("name") or "")
                if name and name not in selected_set:
                    selected_names.append(name)
                    selected_set.add(name)
                if len(selected_names) >= max_columns:
                    break
        return [by_name[name] for name in selected_names[: max(max_columns, 1)]]

    def _profile_single_column(
        self,
        table_ref: str,
        column_name: str,
        column_type: str,
        kind: str,
        database: str,
        top_n: int,
    ) -> Dict[str, Any]:
        column_ref = self._quote_sql_identifier(column_name)
        stats_exprs = [
            "COUNT(*) AS row_count",
            f"COUNT({column_ref}) AS non_null_count",
            f"COUNT(DISTINCT {column_ref}) AS distinct_count",
        ]
        if kind in {"numeric", "temporal"}:
            stats_exprs.extend([f"MIN({column_ref}) AS min_value", f"MAX({column_ref}) AS max_value"])
        stats_sql = f"SELECT {', '.join(stats_exprs)} FROM {table_ref}"
        profile = {
            "type": column_type,
            "kind": kind,
            "stats_sql": stats_sql,
        }
        stats = self._run_profile_scalar_query(stats_sql, database)
        if stats:
            self._attach_null_and_distinct_rates(stats)
            profile["stats"] = stats
            if kind == "numeric":
                percentiles = self._profile_numeric_percentiles(
                    table_ref=table_ref,
                    column_ref=column_ref,
                    stats=stats,
                    database=database,
                )
                if percentiles:
                    profile["percentiles"] = percentiles
            if kind == "temporal":
                temporal_summary = self._profile_temporal_summary(stats)
                if temporal_summary:
                    profile["temporal_summary"] = temporal_summary

        if kind in {"categorical", "boolean"} and top_n > 0:
            top_sql = (
                f"SELECT {column_ref} AS value, COUNT(*) AS count "
                f"FROM {table_ref} WHERE {column_ref} IS NOT NULL "
                f"GROUP BY {column_ref} ORDER BY count DESC LIMIT {max(top_n, 1)}"
            )
            top_values = self._run_profile_rows_query(top_sql, database)
            top_values = [row for row in top_values if isinstance(row, dict) and not row.get("error")]
            profile["top_values_sql"] = top_sql
            if top_values:
                profile["top_values"] = [
                    {
                        "value": self._clip_profile_text(str(row.get("value", "")), 120),
                        "count": self._coerce_profile_scalar(row.get("count")),
                    }
                    for row in top_values[:top_n]
                ]
        return profile

    def _attach_null_and_distinct_rates(self, stats: Dict[str, Any]) -> None:
        row_count = self._profile_number(stats.get("row_count"))
        non_null_count = self._profile_number(stats.get("non_null_count"))
        distinct_count = self._profile_number(stats.get("distinct_count"))
        if row_count is not None and non_null_count is not None and row_count > 0:
            null_count = max(row_count - non_null_count, 0)
            stats["null_count"] = int(null_count) if float(null_count).is_integer() else null_count
            stats["null_rate"] = round(null_count / row_count, 6)
            stats["fill_rate"] = round(non_null_count / row_count, 6)
        if non_null_count is not None and distinct_count is not None and non_null_count > 0:
            stats["distinct_ratio"] = round(distinct_count / non_null_count, 6)

    def _profile_numeric_percentiles(
        self,
        table_ref: str,
        column_ref: str,
        stats: Dict[str, Any],
        database: str,
    ) -> Dict[str, Any]:
        non_null_count = self._profile_number(stats.get("non_null_count"))
        if non_null_count is None or non_null_count <= 0:
            return {}
        positions = {
            "p25": self._profile_percentile_position(non_null_count, 0.25),
            "p50": self._profile_percentile_position(non_null_count, 0.50),
            "p75": self._profile_percentile_position(non_null_count, 0.75),
            "p90": self._profile_percentile_position(non_null_count, 0.90),
            "p95": self._profile_percentile_position(non_null_count, 0.95),
        }
        select_exprs = [
            f"MAX(CASE WHEN rn = {position} THEN value END) AS {name}" for name, position in positions.items()
        ]
        sql = (
            "WITH ordered_profile_values AS ("
            f"SELECT {column_ref} AS value, ROW_NUMBER() OVER (ORDER BY {column_ref}) AS rn "
            f"FROM {table_ref} WHERE {column_ref} IS NOT NULL"
            f") SELECT {', '.join(select_exprs)} FROM ordered_profile_values"
        )
        result = self._run_profile_scalar_query(sql, database)
        if not result or result.get("error"):
            return {}
        result["method"] = "exact_position_from_ordered_non_null_values"
        result["positions"] = positions
        return result

    def _profile_percentile_position(self, count: float, percentile: float) -> int:
        return max(1, min(int(count), int(round((count - 1) * percentile)) + 1))

    def _profile_temporal_summary(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        min_date = self._parse_profile_date(stats.get("min_value"))
        max_date = self._parse_profile_date(stats.get("max_value"))
        if not min_date and not max_date:
            return {}
        summary: Dict[str, Any] = {"profiled_at_date": date.today().isoformat()}
        if min_date and max_date:
            summary["span_days"] = (max_date - min_date).days
        if max_date:
            summary["freshness_days_from_profile_date"] = (date.today() - max_date).days
        return summary

    def _profile_date_duration_pairs(
        self,
        table_ref: str,
        columns: List[Dict[str, Any]],
        database: str,
        deadline: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        pairs = self._candidate_temporal_column_pairs(columns)
        profiles = []
        for pair in pairs[:3]:
            if deadline is not None and time.monotonic() > deadline:
                break
            left_column = pair["left_column"]
            right_column = pair["right_column"]
            left_ref = self._quote_sql_identifier(left_column)
            right_ref = self._quote_sql_identifier(right_column)
            sql = (
                f"SELECT {left_ref} AS left_value, {right_ref} AS right_value "
                f"FROM {table_ref} WHERE {left_ref} IS NOT NULL AND {right_ref} IS NOT NULL LIMIT 1000"
            )
            rows = self._run_profile_rows_query(sql, database)
            deltas = []
            negative_count = 0
            for row in rows:
                if row.get("error"):
                    deltas = []
                    break
                left_date = self._parse_profile_date(row.get("left_value"))
                right_date = self._parse_profile_date(row.get("right_value"))
                if not left_date or not right_date:
                    continue
                delta = (right_date - left_date).days
                if delta < 0:
                    negative_count += 1
                deltas.append(delta)
            if not deltas:
                continue
            profile = {
                "left_column": left_column,
                "right_column": right_column,
                "candidate_reason": pair["candidate_reason"],
                "directional": pair["directional"],
                "sample_size": len(deltas),
                "delta_days": self._profile_numeric_summary_from_values(deltas),
            }
            if negative_count:
                profile["negative_delta_count"] = negative_count
            profiles.append(profile)
        return profiles

    def _candidate_temporal_column_pairs(self, columns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        temporal_names = sorted(
            str(column.get("name") or "")
            for column in columns
            if self._profile_column_kind(str(column.get("type") or "")) == "temporal"
        )
        if len(temporal_names) < 2:
            return []

        starts: Dict[str, str] = {}
        ends: Dict[str, str] = {}
        for name in temporal_names:
            boundary = self._temporal_boundary_stem(name)
            if not boundary:
                continue
            side, stem = boundary
            if side == "left":
                starts.setdefault(stem, name)
            else:
                ends.setdefault(stem, name)

        pairs = []
        seen = set()
        for stem, left_name in sorted(starts.items()):
            right_name = ends.get(stem)
            if right_name:
                key = (left_name, right_name)
                if key not in seen:
                    seen.add(key)
                    pairs.append(
                        {
                            "left_column": left_name,
                            "right_column": right_name,
                            "candidate_reason": "shared_stem_boundary_tokens",
                            "directional": True,
                        }
                    )
        if not pairs and len(temporal_names) == 2:
            pairs.append(
                {
                    "left_column": temporal_names[0],
                    "right_column": temporal_names[1],
                    "candidate_reason": "only_two_temporal_columns",
                    "directional": False,
                }
            )
        return pairs

    def _temporal_boundary_stem(self, column_name: str) -> Optional[tuple[str, str]]:
        tokens = self._identifier_tokens(column_name)
        if not tokens:
            return None
        left_tokens = {"start", "begin", "from", "open", "opened"}
        right_tokens = {"end", "finish", "to", "close", "closed", "expire", "expired"}
        for index, token in enumerate(tokens):
            if token in left_tokens:
                return "left", "|".join(tokens[:index] + ["<boundary>"] + tokens[index + 1 :])
            if token in right_tokens:
                return "right", "|".join(tokens[:index] + ["<boundary>"] + tokens[index + 1 :])
        return None

    def _identifier_tokens(self, value: str) -> List[str]:
        import re

        return [token for token in re.split(r"[^A-Za-z0-9]+", str(value).lower()) if token]

    def _profile_join_relationship_profiles(
        self,
        relationships: List[Dict[str, Any]],
        catalog: str,
        database: str,
        schema_name: str,
        deadline: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        profiles = []
        seen = set()
        for relationship in relationships[:5]:
            if deadline is not None and time.monotonic() > deadline:
                break
            source_table = str(relationship.get("source_table") or "")
            source_column = str(relationship.get("source_column") or "")
            target_table = str(relationship.get("target_table") or "")
            target_column = str(relationship.get("target_column") or "")
            key = (source_table, source_column, target_table, target_column)
            if not all(key) or key in seen:
                continue
            seen.add(key)
            source_ref = self._profile_table_reference(source_table, catalog, database, schema_name)
            target_ref = self._profile_table_reference(target_table, catalog, database, schema_name)
            source_col_ref = self._quote_sql_identifier(source_column)
            target_col_ref = self._quote_sql_identifier(target_column)
            sql = (
                "SELECT "
                "COUNT(*) AS source_rows, "
                f"COUNT(src.{source_col_ref}) AS non_null_source_rows, "
                f"COUNT(DISTINCT src.{source_col_ref}) AS distinct_source_keys, "
                f"COUNT(tgt.{target_col_ref}) AS matched_join_rows, "
                f"COUNT(DISTINCT CASE WHEN tgt.{target_col_ref} IS NOT NULL THEN src.{source_col_ref} END) "
                "AS matched_distinct_source_keys "
                f"FROM {source_ref} src LEFT JOIN {target_ref} tgt "
                f"ON src.{source_col_ref} = tgt.{target_col_ref}"
            )
            stats = self._run_profile_scalar_query(sql, database)
            profile = {
                "source_table": source_table,
                "source_column": source_column,
                "target_table": target_table,
                "target_column": target_column,
                "stats_sql": sql,
            }
            if stats:
                profile["stats"] = stats
                non_null_rows = self._profile_number(stats.get("non_null_source_rows"))
                matched_rows = self._profile_number(stats.get("matched_join_rows"))
                distinct_keys = self._profile_number(stats.get("distinct_source_keys"))
                matched_distinct_keys = self._profile_number(stats.get("matched_distinct_source_keys"))
                if non_null_rows and non_null_rows > 0 and matched_rows is not None:
                    fanout_ratio = matched_rows / non_null_rows
                    profile["join_fanout_ratio"] = round(fanout_ratio, 6)
                    if fanout_ratio == 0:
                        profile["join_cardinality_hint"] = "no_observed_matches"
                    elif fanout_ratio <= 1.01:
                        profile["join_cardinality_hint"] = "many_to_one_or_one_to_one"
                    else:
                        profile["join_cardinality_hint"] = "possible_one_to_many_or_non_unique_target"
                if distinct_keys and distinct_keys > 0 and matched_distinct_keys is not None:
                    profile["referential_coverage"] = round(matched_distinct_keys / distinct_keys, 6)
            profiles.append(profile)
        return profiles

    def _profile_numeric_summary_from_values(self, values: List[float]) -> Dict[str, Any]:
        sorted_values = sorted(values)
        count = len(sorted_values)
        return {
            "min": sorted_values[0],
            "p50": sorted_values[self._profile_percentile_position(count, 0.50) - 1],
            "p90": sorted_values[self._profile_percentile_position(count, 0.90) - 1],
            "max": sorted_values[-1],
        }

    def _run_profile_scalar_query(self, sql: str, database: str) -> Dict[str, Any]:
        result = self.db_tool.read_query(sql, database=database)
        if not result.success:
            return {"error": result.error or "query failed"}
        rows = self._profile_result_rows(result.result)
        if not rows:
            return {}
        return {key: self._coerce_profile_scalar(value) for key, value in rows[0].items() if key != "index"}

    def _run_profile_rows_query(self, sql: str, database: str) -> List[Dict[str, Any]]:
        result = self.db_tool.read_query(sql, database=database)
        if not result.success:
            return [{"error": result.error or "query failed"}]
        return self._profile_result_rows(result.result)

    def _profile_result_rows(self, result: Any) -> List[Dict[str, Any]]:
        if isinstance(result, list):
            return [row for row in result if isinstance(row, dict)]
        if isinstance(result, dict):
            compressed = result.get("compressed_data")
            if isinstance(compressed, str) and compressed and compressed != "Empty dataset":
                import csv
                from io import StringIO

                rows = []
                for row in csv.DictReader(StringIO(compressed)):
                    if "index" in row:
                        row.pop("index", None)
                    rows.append(dict(row))
                return rows
            items = result.get("items")
            if isinstance(items, list):
                return [row for row in items if isinstance(row, dict)]
        return []

    def _profile_column_kind(self, column_type: str) -> str:
        normalized = (column_type or "").upper()
        if any(token in normalized for token in ("INT", "NUMBER", "NUMERIC", "DECIMAL", "DOUBLE", "FLOAT", "REAL")):
            return "numeric"
        if any(token in normalized for token in ("DATE", "TIME", "TIMESTAMP")):
            return "temporal"
        if any(token in normalized for token in ("BOOL",)):
            return "boolean"
        if any(token in normalized for token in ("CHAR", "TEXT", "STRING", "VARCHAR", "ENUM")):
            return "categorical"
        return "unknown"

    def _profile_table_reference(self, table_name: str, catalog: str, database: str, schema_name: str) -> str:
        if "." in table_name:
            return table_name
        parts = [part for part in (catalog, database, schema_name, table_name) if part]
        return ".".join(self._quote_sql_identifier(part) for part in parts)

    def _quote_sql_identifier(self, value: str) -> str:
        value = str(value).strip().strip('"`[]')
        if value and value.replace("_", "").isalnum() and not value[0].isdigit():
            return value
        return '"' + value.replace('"', '""') + '"'

    def _sanitize_profile_sql(self, value: str) -> str:
        import re

        sanitized = str(value)
        sanitized = re.sub(r"'(?:''|[^'])*'", "'<REDACTED>'", sanitized)
        sanitized = re.sub(r'"(?:""|[^"])*"', '"<REDACTED>"', sanitized)
        sanitized = re.sub(r"\b\d+(?:\.\d+)?\b", "<REDACTED>", sanitized)
        return self._clip_profile_text(sanitized, 180)

    def _clip_profile_text(self, value: str, max_chars: int) -> str:
        value = " ".join(str(value).split())
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 1].rstrip() + "..."

    def _coerce_profile_scalar(self, value: Any) -> Any:
        if isinstance(value, Decimal):
            numeric = float(value)
            if numeric.is_integer():
                return int(numeric)
            return numeric
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if stripped == "":
            return ""
        try:
            numeric = float(stripped)
        except ValueError:
            return stripped
        if numeric.is_integer():
            return int(numeric)
        return numeric

    def _profile_number(self, value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float, Decimal)):
            return float(value)
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return None

    def _parse_profile_date(self, value: Any) -> Optional[date]:
        if value in (None, ""):
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        text = str(value).strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        for candidate in (normalized, normalized[:19], normalized[:10]):
            try:
                parsed = datetime.fromisoformat(candidate)
                return parsed.date()
            except ValueError:
                try:
                    return date.fromisoformat(candidate[:10])
                except ValueError:
                    continue
        return None

    def _classify_source_query(
        self,
        has_direct_candidates: bool,
        has_llm_review_candidates: bool,
        has_non_metric_evidence: bool,
        derived_datasource_recommendations: List[Dict[str, Any]],
    ) -> str:
        """Classify how a SQL query should be modeled."""
        if derived_datasource_recommendations:
            return "metric_plus_derived_datasource"
        if has_direct_candidates:
            return "direct_metric"
        if has_llm_review_candidates:
            return "llm_review_candidate"
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
        if "llm_review_candidate" in classifications:
            return "llm_review_candidate"
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
        metric_generation_skips: List[Dict[str, Any]] = []
        if recommendations:
            reason = "rank/window output is filtered or aggregated downstream; model it as a derived data source first"
            for rec in recommendations:
                metric_generation_skips.append(
                    {
                        "source_sql_name": rec.get("source_sql_name", source_name),
                        "reason": (
                            "rank/window TopN query returns row-level or post-window results; "
                            "skip during metric generation"
                        ),
                        "sql_shape": "ranked_window",
                        "window": rec.get("window", {}),
                        "rank_alias": rec.get("rank_alias", ""),
                        "rank_filters": rec.get("rank_filters", []),
                    }
                )
        return {
            "derived_datasource_recommendations": recommendations,
            "metric_generation_skips": metric_generation_skips,
            "classification_reason": reason,
        }

    def _period_over_period_metric_candidates(
        self,
        parsed_expressions: List[Any],
        source_name: str,
        source_context: str,
        existing_metric_catalog: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extract fixed period-over-period metric candidates from LAG-style comparisons."""
        from sqlglot import expressions as exp

        base_candidates: Dict[str, Dict[str, Any]] = {}
        period_candidates: Dict[str, Dict[str, Any]] = {}

        for parsed in parsed_expressions:
            projection_index = self._named_projection_index(parsed)
            shift_outputs_by_select: Dict[str, Dict[str, Dict[str, Any]]] = {}

            for select_name, select in self._named_selects_for_period_analysis(parsed):
                source_names = self._direct_source_names(select)
                for projection in select.expressions:
                    expr = projection.this if isinstance(projection, exp.Alias) else projection
                    alias = projection.alias if isinstance(projection, exp.Alias) else projection.alias_or_name
                    if not alias:
                        continue
                    for window in expr.find_all(exp.Window):
                        if window is not expr:
                            continue
                        detail = self._period_shift_output_detail(
                            window=window,
                            alias=alias,
                            source_names=source_names,
                            projection_index=projection_index,
                        )
                        if not detail:
                            continue
                        shift_outputs_by_select.setdefault(select_name, {})[self._normalize_identifier(alias)] = detail
                        base_candidate = self._base_candidate_for_period_shift(
                            detail=detail,
                            source_name=source_name,
                            source_context=source_context,
                            existing_metric_catalog=existing_metric_catalog,
                            projection_index=projection_index,
                        )
                        if base_candidate:
                            base_candidates[self._metric_candidate_merge_key(base_candidate)] = base_candidate

            for select_name, select in self._named_selects_for_period_analysis(parsed):
                source_names = self._direct_source_names(select)
                visible_shifts = self._visible_period_shift_outputs(
                    source_names=source_names,
                    shift_outputs_by_select=shift_outputs_by_select,
                    projection_index=projection_index,
                )
                visible_shifts.update(shift_outputs_by_select.get(select_name, {}))
                for projection in select.expressions:
                    candidate = self._period_over_period_candidate_from_projection(
                        projection=projection,
                        source_name=source_name,
                        source_context=source_context,
                        select_name=select_name,
                        shift_outputs=visible_shifts,
                        source_names=source_names,
                        projection_index=projection_index,
                        existing_metric_catalog=existing_metric_catalog,
                    )
                    if candidate:
                        period_candidates[self._metric_candidate_merge_key(candidate)] = candidate

        return {
            "base_metric_candidates": sorted(base_candidates.values(), key=lambda item: item["name"]),
            "period_metric_candidates": sorted(period_candidates.values(), key=lambda item: item["name"]),
        }

    def _window_aggregate_metric_candidates(
        self,
        parsed_expressions: List[Any],
        source_name: str,
        source_context: str,
        existing_metric_catalog: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extract cumulative/rolling metric candidates from aggregate windows."""
        from sqlglot import expressions as exp

        base_candidates: Dict[str, Dict[str, Any]] = {}
        window_candidates: Dict[str, Dict[str, Any]] = {}

        for parsed in parsed_expressions:
            projection_index = self._named_projection_index(parsed)
            for select_name, select in self._named_selects_for_period_analysis(parsed):
                source_names = self._direct_source_names(select)
                for projection in select.expressions:
                    expr = projection.this if isinstance(projection, exp.Alias) else projection
                    alias = projection.alias if isinstance(projection, exp.Alias) else projection.alias_or_name
                    if not alias:
                        continue
                    windows = list(expr.find_all(exp.Window))
                    if len(windows) != 1:
                        continue
                    detail = self._window_aggregate_output_detail(
                        window=windows[0],
                        alias=alias,
                        source_name=source_name,
                        select_name=select_name,
                        source_names=source_names,
                        projection_index=projection_index,
                        fallback_tables=self._collect_tables(select),
                        fallback_filters=self._collect_filters(select),
                        fallback_dimensions=self._collect_dimensions(select),
                    )
                    if not detail:
                        continue

                    base_candidate = self._base_candidate_for_window_aggregate(
                        detail=detail,
                        source_name=source_name,
                        source_context=source_context,
                        existing_metric_catalog=existing_metric_catalog,
                    )
                    if base_candidate:
                        base_candidates[self._metric_candidate_merge_key(base_candidate)] = base_candidate

                    candidate = self._window_metric_candidate_from_detail(
                        detail=detail,
                        base_candidate=base_candidate,
                        existing_metric_catalog=existing_metric_catalog,
                    )
                    if candidate:
                        window_candidates[self._metric_candidate_merge_key(candidate)] = candidate

        return {
            "base_metric_candidates": sorted(base_candidates.values(), key=lambda item: item["name"]),
            "window_metric_candidates": sorted(window_candidates.values(), key=lambda item: item["name"]),
        }

    def _window_aggregate_output_detail(
        self,
        window: Any,
        alias: str,
        source_name: str,
        select_name: str,
        source_names: List[str],
        projection_index: Dict[str, Dict[str, Dict[str, Any]]],
        fallback_tables: List[str],
        fallback_filters: List[str],
        fallback_dimensions: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Return executable metadata for one aggregate window projection."""
        aggregate = window.this
        if not isinstance(aggregate, self._aggregate_classes()):
            return None
        aggregation = self._window_aggregation_from_aggregate(aggregate)
        if not aggregation:
            return None
        frame = self._window_frame_detail(window, source_names, projection_index)
        if not frame:
            return None

        input_metric = self._window_input_metric(aggregate)
        base_projection_info = self._window_base_projection(
            input_metric=input_metric,
            source_names=source_names,
            projection_index=projection_index,
        )
        base_expr = ""
        base_metric_name = input_metric
        tables = fallback_tables
        filters = fallback_filters
        dimensions = list(fallback_dimensions)
        base_select_name = ""
        if base_projection_info:
            base_expr = base_projection_info["expr"].sql()
            base_metric_name = self._safe_name(input_metric)
            base_select = base_projection_info["select"]
            tables = self._collect_tables(base_select)
            filters = self._collect_filters(base_select)
            dimensions = self._collect_dimensions(base_select)
            base_select_name = base_projection_info.get("select_name", "")
        else:
            base_expr = aggregate.sql()
            if not base_metric_name:
                base_metric_name = self._name_from_aggregate(aggregate)

        for partition_expr in window.args.get("partition_by") or []:
            partition_sql = partition_expr.sql()
            if partition_sql not in dimensions:
                dimensions.append(partition_sql)

        return {
            "name": self._metric_candidate_name(alias, window),
            "source_alias": alias,
            "source_sql_name": source_name,
            "source_select": select_name,
            "source_names": source_names,
            "base_select": base_select_name,
            "base_metric_name": self._safe_name(base_metric_name),
            "base_expression": base_expr,
            "base_projection_info": base_projection_info,
            "aggregate": aggregate,
            "window_aggregation": aggregation,
            "window": frame.get("window", ""),
            "grain_to_date": frame.get("grain_to_date", ""),
            "time_grain": frame.get("time_grain", ""),
            "window_order_by": self._window_order_by(window),
            "window_frame": frame.get("frame", ""),
            "tables": tables,
            "filters": filters,
            "dimensions": dimensions,
        }

    def _window_aggregation_from_aggregate(self, aggregate: Any) -> str:
        """Map an aggregate function inside OVER() to DATUS window_aggregation."""
        from sqlglot import expressions as exp

        if isinstance(aggregate, exp.Sum):
            return "sum"
        if isinstance(aggregate, exp.Avg):
            return "avg"
        if isinstance(aggregate, exp.Min):
            return "min"
        if isinstance(aggregate, exp.Max):
            return "max"
        if isinstance(aggregate, exp.Count):
            return "row_count" if self._is_count_star(aggregate) else "count"
        return ""

    def _window_input_metric(self, aggregate: Any) -> str:
        """Return the aggregate input alias when a window is over a prior metric."""
        from sqlglot import expressions as exp

        inner = getattr(aggregate, "this", None)
        if inner is None or isinstance(inner, exp.Star):
            return ""
        if isinstance(inner, exp.Distinct):
            inner = inner.expressions[0] if inner.expressions else inner
        if isinstance(inner, exp.Column):
            return self._safe_name(inner.name)
        return ""

    def _window_base_projection(
        self,
        input_metric: str,
        source_names: List[str],
        projection_index: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """Find the upstream projection that produced a window input alias."""
        if not input_metric:
            return None
        input_key = self._normalize_identifier(input_metric)
        for source_name in source_names:
            projection = projection_index.get(source_name, {}).get(input_key)
            if projection:
                return projection
        return None

    def _window_frame_detail(
        self,
        window: Any,
        source_names: List[str],
        projection_index: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Optional[Dict[str, str]]:
        """Translate a SQL window frame into DATUS window/grain metadata."""
        grain = self._window_order_grain(window, source_names, projection_index)
        if not grain:
            return None

        spec = window.args.get("spec")
        frame_sql = spec.sql() if spec is not None else ""
        if spec is None:
            return {
                "grain_to_date": grain.lower(),
                "time_grain": grain,
                "frame": frame_sql,
            }

        start = spec.args.get("start")
        start_side = str(spec.args.get("start_side") or "").upper()
        end = str(spec.args.get("end") or "").upper()
        end_side = str(spec.args.get("end_side") or "").upper()

        if str(start).upper() == "UNBOUNDED" and start_side == "PRECEDING" and end == "CURRENT ROW":
            return {
                "grain_to_date": grain.lower(),
                "time_grain": grain,
                "frame": frame_sql,
            }

        if start_side == "PRECEDING" and end == "CURRENT ROW" and not end_side:
            count = self._window_frame_preceding_count(start)
            if count is None:
                return None
            size = count + 1
            unit = grain.lower()
            if size != 1 and not unit.endswith("s"):
                unit = f"{unit}s"
            return {
                "window": f"{size} {unit}",
                "time_grain": grain,
                "frame": frame_sql,
            }

        return None

    def _window_order_grain(
        self,
        window: Any,
        source_names: List[str],
        projection_index: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> str:
        """Infer the time grain used to order a window metric."""
        order_items = self._window_order_by(window)
        if not order_items:
            return ""
        order_expr = order_items[0].get("expr", "")
        order_name = self._normalize_identifier(order_expr.split(".")[-1])
        for source_name in source_names:
            projection = projection_index.get(source_name, {}).get(order_name)
            if projection:
                grain = self._time_grain_for_expr(projection["expr"]) or self._time_grain_from_alias(order_name) or ""
                if grain:
                    return grain
        try:
            parsed_order = self._parse_sql(f"SELECT {order_expr} AS order_expr")[0]
            order_projection = parsed_order.expressions[0]
            expr = order_projection.this
            return self._time_grain_for_expr(expr) or self._time_grain_from_alias(order_name) or ""
        except Exception:
            return self._time_grain_from_alias(order_name) or ""

    def _window_frame_preceding_count(self, start: Any) -> Optional[int]:
        """Return the N in ROWS BETWEEN N PRECEDING AND CURRENT ROW."""
        try:
            return max(int(getattr(start, "this", start)), 0)
        except (TypeError, ValueError):
            return None

    def _base_candidate_for_window_aggregate(
        self,
        detail: Dict[str, Any],
        source_name: str,
        source_context: str,
        existing_metric_catalog: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Return the base aggregate metric required by a window metric."""
        if detail.get("window_aggregation") == "row_count":
            return None
        base_metric_name = self._normalize_identifier(detail.get("base_metric_name", ""))
        if not base_metric_name or base_metric_name in existing_metric_catalog:
            return None

        projection_info = detail.get("base_projection_info")
        if projection_info:
            select = projection_info["select"]
            candidate = self._candidate_from_projection(
                projection=projection_info["projection"],
                source_name=source_name,
                source_context=source_context,
                tables=self._collect_tables(select),
                filters=self._collect_filters(select),
                dimensions=self._collect_dimensions(select),
                existing_metric_catalog=existing_metric_catalog,
            )
            if candidate and candidate.get("evidence_kind") != "identity_metric_reference":
                candidate["source_select"] = projection_info.get("select_name", "")
                return candidate
            return None

        aggregate = detail.get("aggregate")
        if aggregate is None:
            return None
        measure = self._measure_from_aggregate(aggregate, detail.get("base_metric_name", ""), aggregate)
        return {
            "evidence_kind": "metric_projection",
            "candidate_classification": "exact_metric",
            "expression_kind": "aggregate_expr",
            "aggregation_scope": "aggregate",
            "representable_as": "measure_proxy",
            "equivalence": "exact",
            "requires_validation": False,
            "name": detail.get("base_metric_name", ""),
            "metric_type": "measure_proxy",
            "expression": detail.get("base_expression", ""),
            "source_alias": detail.get("base_metric_name", ""),
            "source_sql_name": source_name,
            "source_select": detail.get("source_select", ""),
            "base_measures": [measure],
            "dimensions": detail.get("dimensions", []),
            "filters": detail.get("filters", []),
            "tables": detail.get("tables", []),
            "confidence": "medium",
            "score_reasons": [
                "aggregate window requires a reusable base-period metric",
            ],
            "source_count": 1,
            "referenced_metrics": [],
        }

    def _window_metric_candidate_from_detail(
        self,
        detail: Dict[str, Any],
        base_candidate: Optional[Dict[str, Any]],
        existing_metric_catalog: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the cumulative/rolling metric candidate from window metadata."""
        base_metric_name = self._normalize_identifier(detail.get("base_metric_name", ""))
        referenced_metric_names = {base_metric_name} & set(existing_metric_catalog)
        if base_candidate:
            base_measures = base_candidate.get("base_measures", [])
        elif referenced_metric_names:
            base_measures = []
        else:
            base_measures = [
                self._measure_from_aggregate(detail["aggregate"], detail.get("name", ""), detail["aggregate"])
            ]
        score_reasons = [
            "aggregate window maps to a cumulative metric over base-period values",
            "window frame was extracted from SQL AST",
        ]
        if referenced_metric_names:
            score_reasons.append("base-period metric already exists in the metric catalog")
        if detail.get("grain_to_date"):
            score_reasons.append("unbounded preceding frame maps to grain_to_date")
        if detail.get("window"):
            score_reasons.append("bounded preceding frame maps to a rolling window")

        candidate = {
            "evidence_kind": "window_metric_projection",
            "candidate_classification": "exact_metric",
            "expression_kind": "window_aggregate_expr",
            "aggregation_scope": "window_over_aggregate",
            "representable_as": "cumulative",
            "equivalence": "exact",
            "requires_validation": False,
            "name": detail.get("name", ""),
            "metric_type": "cumulative",
            "metric_kind": "cumulative",
            "expression": detail.get("base_expression", ""),
            "source_alias": detail.get("source_alias", ""),
            "source_sql_name": detail.get("source_sql_name", ""),
            "source_select": detail.get("source_select", ""),
            "base_metric_name": detail.get("base_metric_name", ""),
            "base_measures": base_measures,
            "dimensions": detail.get("dimensions", []),
            "filters": detail.get("filters", []),
            "tables": detail.get("tables", []),
            "window_aggregation": detail.get("window_aggregation", ""),
            "window_order_by": detail.get("window_order_by", []),
            "window_frame": detail.get("window_frame", ""),
            "time_grain": detail.get("time_grain", ""),
            "confidence": "high",
            "score_reasons": score_reasons,
            "source_count": 1,
            "referenced_metrics": self._referenced_metric_items(referenced_metric_names, existing_metric_catalog),
        }
        if detail.get("window"):
            candidate["window"] = detail["window"]
        if detail.get("grain_to_date"):
            candidate["grain_to_date"] = detail["grain_to_date"]
        return candidate

    def _named_selects_for_period_analysis(self, parsed: Any) -> List[tuple]:
        """Return outer and named inner SELECTs with stable names."""
        from sqlglot import expressions as exp

        selects: List[tuple] = []
        if isinstance(parsed, exp.Select):
            selects.append(("__outer__", parsed))
        selects.extend(self._iter_cte_selects(parsed))
        selects.extend(self._iter_inline_subquery_selects(parsed))
        return selects

    def _named_projection_index(self, parsed: Any) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Index named SELECT output aliases for CTE/subquery lookups."""
        from sqlglot import expressions as exp

        index: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for select_name, select in self._named_selects_for_period_analysis(parsed):
            projections: Dict[str, Dict[str, Any]] = {}
            for projection in select.expressions:
                expr = projection.this if isinstance(projection, exp.Alias) else projection
                alias = projection.alias if isinstance(projection, exp.Alias) else projection.alias_or_name
                if alias:
                    projections[self._normalize_identifier(alias)] = {
                        "projection": projection,
                        "expr": expr,
                        "select": select,
                        "select_name": select_name,
                    }
            index[select_name] = projections
        return index

    def _direct_source_names(self, select: Any) -> List[str]:
        """Return direct FROM/JOIN source names for one SELECT."""
        from sqlglot import expressions as exp

        names: List[str] = []

        def add_source(node: Any) -> None:
            if isinstance(node, exp.Table):
                name = self._safe_name(node.name)
            elif isinstance(node, exp.Subquery):
                name = self._safe_name(node.alias_or_name or "derived_datasource")
            else:
                name = self._safe_name(getattr(node, "alias_or_name", "") or getattr(node, "name", ""))
            if name and name not in names:
                names.append(name)

        from_clause = select.args.get("from_") or select.args.get("from")
        if from_clause is not None:
            if getattr(from_clause, "this", None) is not None:
                add_source(from_clause.this)
            for expr in getattr(from_clause, "expressions", []) or []:
                add_source(expr)

        for join in select.args.get("joins") or []:
            if getattr(join, "this", None) is not None:
                add_source(join.this)
        return names

    def _period_shift_output_detail(
        self,
        window: Any,
        alias: str,
        source_names: List[str],
        projection_index: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """Return LAG offset metadata for one window projection."""
        from sqlglot import expressions as exp

        if not self._is_period_shift_window(window):
            return None
        function_name = self._window_function_name(window)
        if function_name != "LAG":
            return None
        input_expr = getattr(window.this, "this", None)
        if not isinstance(input_expr, exp.Column):
            return None
        input_metric = self._safe_name(input_expr.name)
        offset_window = self._infer_period_offset_window(window, source_names, projection_index)
        if not offset_window:
            return None
        return {
            "alias": self._safe_name(alias),
            "input_metric": input_metric,
            "offset_window": offset_window,
            "window_function": function_name,
            "source_names": source_names,
            "window": {
                "function": function_name,
                "order_by": self._window_order_by(window),
            },
        }

    def _is_period_shift_window(self, window: Any) -> bool:
        """Return true for LAG/LEAD windows that can become offset inputs."""
        from sqlglot import expressions as exp

        return isinstance(window.this, (exp.Lag, exp.Lead))

    def _infer_period_offset_window(
        self,
        window: Any,
        source_names: List[str],
        projection_index: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> str:
        """Infer MetricFlow-style offset_window from the window ORDER BY grain."""
        count = self._period_shift_count(window)
        grain = ""
        order_items = self._window_order_by(window)
        if order_items:
            order_expr = order_items[0].get("expr", "")
            order_name = self._normalize_identifier(order_expr.split(".")[-1])
            for source_name in source_names:
                projection = projection_index.get(source_name, {}).get(order_name)
                if projection:
                    grain = (
                        self._time_grain_for_expr(projection["expr"]) or self._time_grain_from_alias(order_name) or ""
                    )
                    if grain:
                        break
            if not grain:
                try:
                    parsed_order = self._parse_sql(f"SELECT {order_expr} AS order_expr")[0]
                    order_projection = parsed_order.expressions[0]
                    expr = order_projection.this
                    grain = self._time_grain_for_expr(expr) or self._time_grain_from_alias(order_name) or ""
                except Exception:
                    grain = self._time_grain_from_alias(order_name) or ""
        if not grain:
            return ""
        unit = grain.lower()
        if count != 1 and not unit.endswith("s"):
            unit = f"{unit}s"
        return f"{count} {unit}"

    def _period_shift_count(self, window: Any) -> int:
        """Return the LAG offset count, defaulting to one period."""
        offset = getattr(window.this, "args", {}).get("offset")
        try:
            value = int(getattr(offset, "this", "") or 1)
        except (TypeError, ValueError):
            value = 1
        return max(value, 1)

    def _base_candidate_for_period_shift(
        self,
        detail: Dict[str, Any],
        source_name: str,
        source_context: str,
        existing_metric_catalog: Dict[str, Dict[str, Any]],
        projection_index: Dict[str, Dict[str, Dict[str, Any]]],
        skip_existing_input: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Return the upstream aggregate candidate required by a period shift."""
        input_metric = self._normalize_identifier(detail.get("input_metric", ""))
        if not input_metric:
            return None
        if skip_existing_input and input_metric in existing_metric_catalog:
            return None
        for source in detail.get("source_names", []):
            projection_info = projection_index.get(source, {}).get(input_metric)
            if not projection_info:
                continue
            select = projection_info["select"]
            candidate = self._candidate_from_projection(
                projection=projection_info["projection"],
                source_name=source_name,
                source_context=source_context,
                tables=self._collect_tables(select),
                filters=self._collect_filters(select),
                dimensions=self._collect_dimensions(select),
                existing_metric_catalog=existing_metric_catalog,
            )
            if candidate and candidate.get("evidence_kind") != "identity_metric_reference":
                candidate["source_select"] = projection_info.get("select_name", source)
                return candidate
        return None

    def _visible_period_shift_outputs(
        self,
        source_names: List[str],
        shift_outputs_by_select: Dict[str, Dict[str, Dict[str, Any]]],
        projection_index: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        """Return period-shift aliases visible through direct CTE/subquery sources."""
        visible: Dict[str, Dict[str, Any]] = {}
        for source_name in source_names:
            source_shifts = shift_outputs_by_select.get(source_name, {})
            for alias in projection_index.get(source_name, {}):
                shift = source_shifts.get(alias)
                if shift:
                    visible[alias] = shift
        return visible

    def _period_over_period_candidate_from_projection(
        self,
        projection: Any,
        source_name: str,
        source_context: str,
        select_name: str,
        shift_outputs: Dict[str, Dict[str, Any]],
        source_names: List[str],
        projection_index: Dict[str, Dict[str, Dict[str, Any]]],
        existing_metric_catalog: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Build a fixed period-over-period candidate from shifted metric output evidence."""
        from sqlglot import expressions as exp

        expr = projection.this if isinstance(projection, exp.Alias) else projection
        alias = projection.alias if isinstance(projection, exp.Alias) else projection.alias_or_name
        if not alias:
            return None

        projection_shift_outputs = dict(shift_outputs)
        source_expression, analysis_expr, inline_shift_aliases = self._period_over_period_analysis_expression(
            expr=expr,
            source_names=source_names,
            projection_index=projection_index,
            shift_outputs=projection_shift_outputs,
        )
        calculation_detail = self._period_over_period_projection_calculation(
            analysis_expr,
            projection_shift_outputs,
        )
        if not calculation_detail:
            return None
        calculation, detail = calculation_detail
        if calculation == "previous_value" and select_name != "__outer__" and inline_shift_aliases:
            return None

        base_candidate = self._base_candidate_for_period_shift(
            detail=detail,
            source_name=source_name,
            source_context=source_context,
            existing_metric_catalog=existing_metric_catalog,
            projection_index=projection_index,
            skip_existing_input=False,
        )
        if not base_candidate:
            return None

        time_grain = self._period_shift_time_grain(detail, projection_index)
        time_dimension = self._period_shift_time_dimension(detail, projection_index)
        if not time_grain or not time_dimension:
            return None

        score_reasons = list(base_candidate.get("score_reasons", []))
        score_reasons.extend(
            [
                "LAG window maps to a fixed period_over_period metric",
                "published metric keeps the base aggregate expression and stores fixed comparison semantics separately",
            ]
        )
        if calculation == "previous_value":
            score_reasons.append("source SQL exposes the prior-period value as a final output")
        else:
            score_reasons.append("final SELECT expression combines current and prior-period values")

        candidate = dict(base_candidate)
        candidate.update(
            {
                "evidence_kind": "period_over_period_metric_projection",
                "candidate_classification": "exact_metric",
                "expression_kind": "period_over_period_expr",
                "aggregation_scope": "period_over_period",
                "representable_as": "period_over_period",
                "equivalence": "exact",
                "requires_validation": False,
                "name": self._metric_candidate_name(alias, expr),
                "metric_type": "period_over_period",
                "expression": base_candidate.get("expression", ""),
                "source_alias": alias,
                "source_expression": source_expression,
                "source_sql_name": source_name,
                "source_select": select_name,
                "source_context": source_context,
                "base_metric_name": detail.get("input_metric", ""),
                "time_dimension": time_dimension,
                "period_over_period": {
                    "time_grain": time_grain,
                    "offset_window": detail.get("offset_window", ""),
                    "calculation": calculation,
                },
                "confidence": "high",
                "score_reasons": score_reasons,
                "source_count": 1,
                "referenced_metrics": [],
            }
        )
        candidate.pop("inputs", None)
        candidate.pop("required_input_metrics", None)
        candidate.pop("offset_window", None)
        return candidate

    def _period_over_period_analysis_expression(
        self,
        expr: Any,
        source_names: List[str],
        projection_index: Dict[str, Dict[str, Dict[str, Any]]],
        shift_outputs: Dict[str, Dict[str, Any]],
    ) -> tuple:
        """Return an expression where inline LAG windows are replaced with stable aliases."""
        from sqlglot import expressions as exp

        source_expression = expr.sql()
        rewritten_expression = source_expression
        inline_shift_aliases: List[str] = []
        for window in expr.find_all(exp.Window):
            detail = self._period_shift_output_detail(
                window=window,
                alias=f"{self._safe_name(getattr(getattr(window.this, 'this', None), 'name', '') or 'metric')}_prev",
                source_names=source_names,
                projection_index=projection_index,
            )
            if detail:
                rewritten_expression = rewritten_expression.replace(window.sql(), detail["alias"])
                normalized_alias = self._normalize_identifier(detail["alias"])
                inline_shift_aliases.append(normalized_alias)
                shift_outputs[normalized_alias] = detail

        if not inline_shift_aliases:
            return source_expression, expr, inline_shift_aliases

        try:
            parsed = self._parse_sql(f"SELECT {rewritten_expression} AS period_over_period_expr")[0]
            analysis_expr = parsed.expressions[0].this
        except Exception:
            analysis_expr = expr
        return rewritten_expression, analysis_expr, inline_shift_aliases

    def _period_over_period_projection_calculation(
        self,
        expr: Any,
        shift_outputs: Dict[str, Dict[str, Any]],
    ) -> Optional[tuple]:
        """Return the fixed PoP calculation and shift detail represented by a projection."""
        from sqlglot import expressions as exp

        column_names = {self._normalize_identifier(col.name) for col in expr.find_all(exp.Column)}
        shift_aliases = sorted(column_names & set(shift_outputs))
        if len(shift_aliases) != 1:
            return None

        shift_alias = shift_aliases[0]
        detail = shift_outputs[shift_alias]
        base_names = {self._normalize_identifier(detail.get("input_metric", ""))}
        previous_names = {shift_alias}

        if self._is_period_column_ref(expr, previous_names):
            return "previous_value", detail
        if self._is_period_percent_change_expr(expr, base_names, previous_names):
            return "percent_change", detail
        if self._is_period_delta_expr(expr, base_names, previous_names):
            return "delta", detail
        if self._is_period_ratio_expr(expr, base_names, previous_names):
            return "ratio", detail
        return None

    def _is_period_column_ref(self, expr: Any, names: set) -> bool:
        """Return true when an expression is a direct reference to one of the names."""
        from sqlglot import expressions as exp

        if expr is None:
            return False
        core_expr = self._unwrap_metric_candidate_expr(expr)
        return isinstance(core_expr, exp.Column) and self._normalize_identifier(core_expr.name) in names

    def _is_period_delta_expr(self, expr: Any, base_names: set, previous_names: set) -> bool:
        """Return true for current-period minus previous-period expressions."""
        from sqlglot import expressions as exp

        core_expr = self._unwrap_metric_candidate_expr(expr)
        if not isinstance(core_expr, exp.Sub):
            return False
        return self._is_period_column_ref(core_expr.args.get("this"), base_names) and self._is_period_column_ref(
            core_expr.args.get("expression"),
            previous_names,
        )

    def _is_period_percent_change_expr(self, expr: Any, base_names: set, previous_names: set) -> bool:
        """Return true for standard (current - previous) / previous expressions."""
        from sqlglot import expressions as exp

        core_expr = self._strip_period_numeric_scale(expr)
        if isinstance(core_expr, exp.Div):
            numerator = self._strip_period_numeric_scale(core_expr.args.get("this"))
            return self._is_period_delta_expr(
                numerator,
                base_names,
                previous_names,
            ) and self._is_period_previous_denominator(core_expr.args.get("expression"), previous_names)
        if isinstance(core_expr, exp.Sub) and self._is_numeric_one(core_expr.args.get("expression")):
            return self._is_period_ratio_expr(core_expr.args.get("this"), base_names, previous_names)
        return False

    def _is_period_ratio_expr(self, expr: Any, base_names: set, previous_names: set) -> bool:
        """Return true for current-period divided by previous-period expressions."""
        from sqlglot import expressions as exp

        core_expr = self._strip_period_numeric_scale(expr)
        if not isinstance(core_expr, exp.Div):
            return False
        numerator = self._strip_period_numeric_scale(core_expr.args.get("this"))
        return self._is_period_column_ref(
            numerator,
            base_names,
        ) and self._is_period_previous_denominator(core_expr.args.get("expression"), previous_names)

    def _is_period_previous_denominator(self, expr: Any, previous_names: set) -> bool:
        """Return true for previous-period denominator references, including NULLIF guards."""
        from sqlglot import expressions as exp

        core_expr = self._unwrap_metric_candidate_expr(expr)
        if isinstance(core_expr, exp.Nullif):
            return self._is_period_column_ref(core_expr.args.get("this"), previous_names)
        return self._is_period_column_ref(core_expr, previous_names)

    def _strip_period_numeric_scale(self, expr: Any) -> Any:
        """Remove numeric percentage scale factors from a comparison expression."""
        from sqlglot import expressions as exp

        core_expr = self._unwrap_metric_candidate_expr(expr)
        if isinstance(core_expr, exp.Mul):
            left = core_expr.args.get("this")
            right = core_expr.args.get("expression")
            if self._is_numeric_literal(left):
                return self._unwrap_metric_candidate_expr(right)
            if self._is_numeric_literal(right):
                return self._unwrap_metric_candidate_expr(left)
        return core_expr

    def _is_numeric_literal(self, expr: Any) -> bool:
        """Return true for numeric SQL literals."""
        from sqlglot import expressions as exp

        return isinstance(expr, exp.Literal) and not expr.is_string

    def _is_numeric_one(self, expr: Any) -> bool:
        """Return true for numeric literal 1."""
        if not self._is_numeric_literal(expr):
            return False
        try:
            return float(getattr(expr, "this", "")) == 1.0
        except (TypeError, ValueError):
            return False

    def _period_shift_time_grain(
        self,
        detail: Dict[str, Any],
        projection_index: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> str:
        """Return the fixed query grain for a period-shift metric."""
        order_projection = self._period_shift_order_projection(detail, projection_index)
        grain = ""
        if order_projection:
            grain = (
                self._time_grain_for_expr(order_projection["expr"])
                or self._time_grain_from_alias(self._period_shift_order_name(detail))
                or ""
            )
        if not grain:
            grain = self._period_time_grain_from_offset_window(detail.get("offset_window", ""))
        return grain.lower()

    def _period_shift_time_dimension(
        self,
        detail: Dict[str, Any],
        projection_index: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> str:
        """Return the time dimension backing a period-shift metric."""
        from sqlglot import expressions as exp

        expressions = []
        order_projection = self._period_shift_order_projection(detail, projection_index)
        if order_projection:
            expressions.append(order_projection["expr"])

        order_expr = self._period_shift_order_expr(detail)
        if order_expr:
            try:
                parsed_order = self._parse_sql(f"SELECT {order_expr} AS order_expr")[0]
                expressions.append(parsed_order.expressions[0].this)
            except Exception:
                pass

        for expr in expressions:
            for column in expr.find_all(exp.Column):
                if column.name:
                    return self._safe_name(column.name)
        return ""

    def _period_shift_order_projection(
        self,
        detail: Dict[str, Any],
        projection_index: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """Return the upstream projection that provides the LAG ORDER BY key."""
        order_name = self._period_shift_order_name(detail)
        if not order_name:
            return None
        for source_name in detail.get("source_names", []):
            projection = projection_index.get(source_name, {}).get(order_name)
            if projection:
                return projection
        return None

    def _period_shift_order_name(self, detail: Dict[str, Any]) -> str:
        """Return the normalized first ORDER BY expression name for a period shift."""
        order_expr = self._period_shift_order_expr(detail)
        if not order_expr:
            return ""
        return self._normalize_identifier(order_expr.split(".")[-1])

    def _period_shift_order_expr(self, detail: Dict[str, Any]) -> str:
        """Return the first ORDER BY expression for a period shift."""
        order_items = detail.get("window", {}).get("order_by") or []
        if not order_items:
            return ""
        return str(order_items[0].get("expr") or "")

    def _period_time_grain_from_offset_window(self, offset_window: str) -> str:
        """Infer a fixed grain from an offset window such as '12 months'."""
        parts = str(offset_window or "").strip().lower().split()
        if len(parts) < 2:
            return ""
        unit = parts[1].rstrip("s")
        if unit in {"second", "minute", "hour", "day", "week", "month", "quarter", "year"}:
            return unit
        return ""

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
        if normalized_alias.startswith("metric_time__"):
            normalized_alias = normalized_alias.split("__", 1)[1]
        grain_aliases = {
            "second": "SECOND",
            "minute": "MINUTE",
            "hour": "HOUR",
            "day": "DAY",
            "date": "DAY",
            "week": "WEEK",
            "month": "MONTH",
            "quarter": "QUARTER",
            "year": "YEAR",
        }
        if normalized_alias in grain_aliases:
            return grain_aliases[normalized_alias]
        for suffix, grain in grain_aliases.items():
            if normalized_alias.endswith(f"_{suffix}"):
                return grain
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

    def _metric_source_context(self, entry: Dict[str, Any]) -> str:
        """Return source text that can disambiguate metric-like SQL projections."""
        import re

        parts = []
        for field in ("question", "name", "summary", "comment", "search_text", "description"):
            value = str(entry.get(field) or "").strip()
            if not value:
                continue
            if field == "name" and re.fullmatch(r"sql_\d+", value):
                continue
            parts.append(value)
        return " ".join(parts)

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
        source_context: str,
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
        if list(expr.find_all(exp.Window)):
            return None
        aggregate_classes = self._aggregate_classes()
        aggregates = list(expr.find_all(*aggregate_classes))
        has_aggregates = bool(aggregates)
        columns = {self._normalize_identifier(col.name) for col in expr.find_all(exp.Column)}
        existing_metric_names = set(existing_metric_catalog)
        referenced_metric_names = columns & existing_metric_names

        if not has_aggregates and not referenced_metric_names:
            llm_review_candidate = self._llm_review_candidate_from_projection(
                expr=expr,
                alias=alias or "",
                source_name=source_name,
                source_context=source_context,
                tables=tables,
                filters=filters,
                dimensions=dimensions,
            )
            if llm_review_candidate:
                return llm_review_candidate
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
            "evidence_kind": "metric_projection",
            "candidate_classification": "exact_metric",
            "expression_kind": self._exact_metric_expression_kind(metric_type, expr, aggregates),
            "aggregation_scope": "metric_reference" if metric_type == "derived" else "aggregate",
            "representable_as": metric_type,
            "equivalence": "exact",
            "requires_validation": False,
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

    def _exact_metric_expression_kind(self, metric_type: str, expr: Any, aggregates: List[Any]) -> str:
        """Return a stable expression-kind label for deterministic metric candidates."""
        from sqlglot import expressions as exp

        if metric_type == "derived":
            return "derived_expr"
        if metric_type == "cumulative":
            return "cumulative_expr"
        if metric_type == "ratio":
            return "aggregate_ratio_expr"
        if metric_type == "expr":
            return "aggregate_expr"
        if any(list(expr.find_all(exp.Case))) or any(list(agg.find_all(exp.Case)) for agg in aggregates):
            return "conditional_aggregate_expr"
        return "aggregate_expr"

    def _llm_review_candidate_from_projection(
        self,
        expr: Any,
        alias: str,
        source_name: str,
        source_context: str,
        tables: List[str],
        filters: List[str],
        dimensions: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Infer a possible metric candidate from row-level SQL expressions.

        This pass is intentionally SQL-first and high recall: it does not prove
        business metric equivalence. Instead, it preserves row-level arithmetic
        as candidate evidence for the gen_metrics LLM to accept, reject, or lift
        into a reusable MetricFlow metric.
        """
        from sqlglot import expressions as exp

        if list(expr.find_all(exp.Window)):
            return None

        core_expr = self._unwrap_metric_candidate_expr(expr)
        if not self._has_row_metric_expression(core_expr):
            return None
        if not self._has_column_operand(core_expr):
            return None

        div_expr = self._first_division_expr(core_expr)
        has_additive_math = any(list(core_expr.find_all(cls)) for cls in (exp.Add, exp.Sub))
        metric_type = "ratio" if div_expr is not None and not has_additive_math else "expr"
        expression_kind = "row_ratio_expr" if metric_type == "ratio" else "row_arithmetic_expr"

        if metric_type == "ratio" and div_expr is not None:
            measures = self._raw_ratio_base_measures(div_expr)
        else:
            measures = self._raw_expression_base_measures(core_expr)
        if not measures:
            return None

        name = self._llm_review_metric_name(
            alias=alias,
            expr=expr,
            metric_type=metric_type,
            source_context=source_context,
            measures=measures,
        )
        has_context = self._has_metric_naming_context(alias=alias, source_context=source_context)
        confidence = "medium" if alias or has_context else "low"
        score_reasons = [
            "row-level arithmetic expression is metric-like but not statically equivalent to an aggregate metric",
            "candidate must be reviewed before lifting into MetricFlow metric algebra",
        ]
        if metric_type == "ratio":
            score_reasons.append("division expression can be reviewed as a possible ratio metric")
        else:
            score_reasons.append("arithmetic expression can be reviewed as a possible expr metric")
        if alias:
            score_reasons.append("final SELECT alias provides naming evidence")
        if has_context:
            score_reasons.append("source question/name/summary provides business naming evidence")
        if filters:
            score_reasons.append("historical SQL filters preserved as metric evidence")
        if dimensions:
            score_reasons.append("GROUP BY dimensions preserved as query-grain evidence")

        return {
            "evidence_kind": "llm_review_projection",
            "candidate_classification": "llm_review_candidate",
            "expression_kind": expression_kind,
            "aggregation_scope": "row",
            "representable_as": metric_type,
            "equivalence": "lifted",
            "requires_validation": True,
            "name": name,
            "metric_type": metric_type,
            "expression": expr.sql(),
            "source_alias": alias or "",
            "source_sql_name": source_name,
            "source_context": source_context,
            "base_measures": measures,
            "dimensions": dimensions,
            "filters": filters,
            "tables": tables,
            "confidence": confidence,
            "requires_name_translation": not bool(alias) or self._requires_name_translation(alias),
            "name_source": "source_alias" if alias else "expression_with_optional_source_context",
            "score_reasons": score_reasons,
            "source_count": 1,
            "referenced_metrics": [],
        }

    def _unwrap_metric_candidate_expr(self, expr: Any) -> Any:
        """Remove expression wrappers that do not change metric lineage."""
        from sqlglot import expressions as exp

        wrapper_classes = tuple(
            cls
            for cls in (getattr(exp, "Cast", None), getattr(exp, "TryCast", None), getattr(exp, "Paren", None))
            if cls
        )
        current = expr
        while True:
            if wrapper_classes and isinstance(current, wrapper_classes):
                inner = current.args.get("this")
            elif isinstance(current, exp.Round):
                inner = current.args.get("this")
            else:
                inner = None
            if inner is None or inner is current:
                return current
            current = inner

    def _has_row_metric_expression(self, expr: Any) -> bool:
        """Return true for row-level arithmetic expressions worth LLM review."""
        from sqlglot import expressions as exp

        math_classes = (exp.Add, exp.Sub, exp.Mul, exp.Div)
        return isinstance(expr, math_classes) or any(list(expr.find_all(cls)) for cls in math_classes)

    def _has_column_operand(self, expr: Any) -> bool:
        """Return true when an expression depends on at least one physical column."""
        from sqlglot import expressions as exp

        return bool(list(expr.find_all(exp.Column)))

    def _first_division_expr(self, expr: Any) -> Optional[Any]:
        """Return the primary division expression, if any."""
        from sqlglot import expressions as exp

        if isinstance(expr, exp.Div):
            return expr
        divisions = list(expr.find_all(exp.Div))
        return divisions[0] if divisions else None

    def _raw_ratio_base_measures(self, div_expr: Any) -> List[Dict[str, Any]]:
        """Build default SUM base-measure evidence for raw ratio operands."""
        measures: List[Dict[str, Any]] = []
        for role, operand in (
            ("numerator", div_expr.args.get("this")),
            ("denominator", div_expr.args.get("expression")),
        ):
            measure_expr = self._raw_ratio_operand_sql(operand)
            if not measure_expr:
                return []
            measures.append(
                {
                    "name": self._safe_name(measure_expr),
                    "agg": "SUM",
                    "expr": measure_expr,
                    "filter": "",
                    "source_alias": "",
                    "requires_name_translation": False,
                    "source_count": 1,
                    "evidence_kind": "row_ratio_operand",
                    "role": role,
                }
            )
        return self._deduplicate_items(measures, ["name", "agg", "expr", "filter", "role"])

    def _raw_ratio_operand_sql(self, operand: Any) -> str:
        """Return SQL for a raw ratio operand, unwrapping safe divide guards and percentage scaling."""
        from sqlglot import expressions as exp

        if operand is None:
            return ""
        operand = self._unwrap_metric_candidate_expr(operand)
        if isinstance(operand, exp.Nullif):
            inner = operand.args.get("this")
            return self._raw_ratio_operand_sql(inner)
        if isinstance(operand, exp.Mul):
            left = operand.args.get("this")
            right = operand.args.get("expression")
            if isinstance(left, exp.Literal) and not left.is_string:
                return self._raw_ratio_operand_sql(right)
            if isinstance(right, exp.Literal) and not right.is_string:
                return self._raw_ratio_operand_sql(left)
        if isinstance(operand, exp.Literal):
            return ""
        if list(operand.find_all(*self._aggregate_classes())):
            return ""
        return operand.sql()

    def _raw_expression_base_measures(self, expr: Any) -> List[Dict[str, Any]]:
        """Build SUM base-measure evidence for row-level arithmetic operands."""
        from sqlglot import expressions as exp

        measures: List[Dict[str, Any]] = []
        for column in expr.find_all(exp.Column):
            measure_expr = column.sql()
            measures.append(
                {
                    "name": self._safe_name(column.name or measure_expr),
                    "agg": "SUM",
                    "expr": measure_expr,
                    "filter": "",
                    "source_alias": "",
                    "requires_name_translation": False,
                    "source_count": 1,
                    "evidence_kind": "row_arithmetic_operand",
                    "role": "operand",
                }
            )
        return self._deduplicate_items(measures, ["name", "agg", "expr", "filter", "role"])

    def _llm_review_metric_name(
        self,
        alias: str,
        expr: Any,
        metric_type: str,
        source_context: str,
        measures: List[Dict[str, Any]],
    ) -> str:
        """Build a readable fallback name for LLM-reviewed candidates."""
        if alias:
            return self._metric_candidate_name(alias, expr)

        if metric_type == "ratio" and len(measures) >= 2:
            numerator = measures[0].get("name", "numerator")
            denominator = measures[1].get("name", "denominator")
            context_lower = source_context.lower()
            if "rate" in context_lower or "percent" in context_lower or "percentage" in context_lower:
                return self._safe_name(f"{numerator}_rate")
            if "share" in context_lower or "proportion" in context_lower:
                return self._safe_name(f"{numerator}_share")
            return self._safe_name(f"{numerator}_per_{denominator}")

        return self._metric_candidate_name("", expr)

    def _has_metric_naming_context(self, alias: str, source_context: str) -> bool:
        """Return true when source text gives useful business naming evidence."""
        import re

        context_text = " ".join(part for part in (alias, source_context) if part).lower()
        return bool(
            re.search(
                r"\b(metrics?|measures?|kpis?|rates?|ratios?|shares?|percent(?:age)?s?|proportions?|arpu|arppu)\b",
                context_text,
            )
        )

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
        input_parts = []
        for item in candidate.get("inputs", []):
            input_parts.append(
                "|".join(
                    [
                        self._normalize_identifier(str(item.get("name", ""))),
                        self._normalize_identifier(str(item.get("alias", ""))),
                        str(item.get("offset_window", "")),
                        str(item.get("offset_to_grain", "")),
                    ]
                )
            )
        return "||".join(
            [
                candidate.get("expression", ""),
                f"offset_window:{candidate.get('offset_window', '')}",
                f"window:{candidate.get('window', '')}",
                f"grain_to_date:{candidate.get('grain_to_date', '')}",
                f"time_grain:{candidate.get('time_grain', '')}",
                f"time_dimension:{candidate.get('time_dimension', '')}",
                f"window_aggregation:{candidate.get('window_aggregation', '')}",
                f"window_order_by:{json.dumps(candidate.get('window_order_by', []), sort_keys=True, default=str)}",
                f"period_over_period:{json.dumps(candidate.get('period_over_period', {}), sort_keys=True, default=str)}",
                *[f"measure:{part}" for part in sorted(measure_parts)],
                *[f"input:{part}" for part in sorted(input_parts)],
            ]
        )

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
