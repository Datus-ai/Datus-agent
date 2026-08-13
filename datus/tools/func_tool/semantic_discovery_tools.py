# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Semantic Discovery Tools

This module provides read-only discovery tools for semantic-layer generation,
including table relationships, column usage evidence, and semantic profiling.
"""

import json
import re
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from agents import Tool
from pydantic import BaseModel, Field

from datus.tools.func_tool.base import FuncToolResult
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import parse_dialect

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig
    from datus.tools.func_tool.database import DBFuncTool

logger = get_logger(__name__)


class SemanticKeyCandidate(BaseModel):
    """One ordered logical-key candidate to verify against the full table."""

    table_name: str = Field(..., description="Table containing the candidate key")
    columns: List[str] = Field(..., description="Complete ordered candidate key columns")


class SemanticDiscoveryTools:
    """
    Read-only discovery tools for semantic-layer generation.

    These tools analyze database structures and historical query patterns
    to help generate semantic models.
    """

    permission_category: str = "semantic_tools"

    _AGGREGATE_CLASSES = ()

    def __init__(
        self,
        db_tool: Optional["DBFuncTool"] = None,
        enable_semantic_model_profiler: bool = False,
        *,
        agent_config: Optional["AgentConfig"] = None,
        sub_agent_name: Optional[str] = None,
        source_sql_provider: Optional[Callable[[], List[Dict[str, Any]]]] = None,
        compact_source_inspection: bool = False,
    ):
        """
        Initialize semantic discovery tools.

        Args:
            db_tool: Optional database tool.
            enable_semantic_model_profiler: Whether to expose the optional
                semantic SQL history profiler tool.
        """
        self.db_tool = db_tool
        self.agent_config = agent_config if agent_config is not None else getattr(db_tool, "agent_config", None)
        self.sub_agent_name = sub_agent_name if sub_agent_name is not None else getattr(db_tool, "sub_agent_name", None)
        self.enable_semantic_model_profiler = enable_semantic_model_profiler
        self.source_sql_provider = source_sql_provider
        self.compact_source_inspection = compact_source_inspection
        self._semantic_source_cache: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}

    def reset_request_cache(self) -> None:
        """Clear source metadata retained during one authoring request."""
        self._semantic_source_cache.clear()

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

        methods_to_convert = []
        if self.db_tool is not None:
            methods_to_convert.extend(
                [
                    self.inspect_semantic_sources,
                    self.validate_semantic_key_candidates,
                ]
            )
            if self.enable_semantic_model_profiler:
                methods_to_convert.append(self.profile_semantic_model_evidence)
        return [trans_to_function_tool(bound_method) for bound_method in methods_to_convert]

    # These compatibility methods remain callable by Python integrations while
    # the LLM tool surface above exposes only the two batched operations.
    def get_multiple_tables_ddl(
        self,
        tables: List[str],
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
    ) -> FuncToolResult:
        """Return table DDLs for legacy Python callers; not an exposed LLM tool."""
        try:
            results = []
            for table in tables:
                ddl_result = self.db_tool.get_table_ddl(table, catalog, database, schema_name)
                if ddl_result.success and ddl_result.result:
                    results.append({"table_name": table, **ddl_result.result})
                else:
                    results.append({"table_name": table, "error": ddl_result.error})
            return FuncToolResult(result=results)
        except Exception as exc:
            return FuncToolResult(success=0, error=str(exc))

    def analyze_table_relationships(
        self,
        tables: List[str],
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
        sample_sql_queries: int = 20,
    ) -> FuncToolResult:
        """Return legacy relationship-only evidence; not an exposed LLM tool."""
        try:
            relationships = self._extract_foreign_keys_from_ddl(
                tables,
                catalog or "",
                database or "",
                schema_name or "",
            )
            relationships.extend(self._analyze_join_patterns_from_history(tables, sample_sql_queries))
            if not relationships:
                relationships.extend(
                    self._infer_from_column_names(
                        tables,
                        catalog or "",
                        database or "",
                        schema_name or "",
                    )
                )
            relationships = self._deduplicate_relationships(relationships)
            return FuncToolResult(
                result={
                    "relationships": relationships,
                    "summary": f"Found {len(relationships)} relationships across {len(tables)} tables",
                }
            )
        except Exception as exc:
            return FuncToolResult(success=0, error=str(exc))

    def validate_semantic_key_candidate(
        self,
        table_name: str,
        columns: List[str],
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
    ) -> FuncToolResult:
        """Verify one legacy key candidate; not an exposed LLM tool."""
        try:
            return self._validate_semantic_key_candidate(
                table_name=table_name,
                columns=columns,
                catalog=catalog or "",
                database=database or "",
                schema_name=schema_name or "",
            )
        except Exception as exc:
            return FuncToolResult(success=0, error=str(exc))

    def analyze_column_usage_patterns(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
        sample_sql_queries: int = 50,
    ) -> FuncToolResult:
        """Return legacy reference-SQL field usage; not an exposed LLM tool."""
        try:
            if not self.agent_config:
                return FuncToolResult(
                    success=0,
                    error="Cannot analyze column patterns without agent_config (no SQL history available)",
                )
            from datus.storage.reference_sql.store import ReferenceSqlRAG

            schema_result = self.db_tool.describe_table(table_name, catalog, database, schema_name)
            if not schema_result.success:
                return FuncToolResult(success=0, error=f"Failed to get table schema: {schema_result.error}")
            table_columns = schema_result.result.get("columns", []) if isinstance(schema_result.result, dict) else []
            all_columns = [
                str(column.get("name") or "")
                for column in table_columns
                if isinstance(column, dict) and column.get("name")
            ]
            target_columns = [str(column) for column in (columns if columns else all_columns) if column]
            sql_rag = ReferenceSqlRAG(self.agent_config, self.sub_agent_name)
            search_results = sql_rag.search_reference_sql(
                query_text=f"SELECT FROM {table_name}",
                top_n=sample_sql_queries,
            )
            entries = [
                {
                    "name": entry.get("name") or entry.get("summary") or entry.get("filepath") or f"sql_{index + 1}",
                    **entry,
                    "sql": str(entry.get("sql") or "").strip(),
                }
                for index, entry in enumerate(search_results)
                if str(entry.get("sql") or "").strip()
            ]
            table_evidence, parse_errors = self._semantic_profile_sql_evidence(entries, [table_name], 1)
            table_profile = self._semantic_profile_table_evidence_for(table_evidence, table_name)
            patterns = self._column_usage_patterns_from_semantic_profile(
                table_profile.get("field_usage_statistics", {}),
                target_columns,
            )
            result = {
                "column_patterns": patterns,
                "summary": f"Analyzed {len(patterns)} columns from {len(search_results)} SQL queries",
            }
            if parse_errors:
                result["parse_errors"] = parse_errors[:5]
            return FuncToolResult(result=result)
        except Exception as exc:
            logger.exception("Error analyzing column usage patterns")
            return FuncToolResult(success=0, error=str(exc))

    def inspect_semantic_sources(
        self,
        tables: List[str],
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
    ) -> FuncToolResult:
        """Inspect all semantic-model source tables in one read-only call.

        Args:
            tables: Complete list of physical tables involved in the model.
            catalog: Optional catalog override.
            database: Optional database override.
            schema_name: Optional schema override.

        This tool batches schema and relationship discovery. It reads each
        table's DDL and enriched schema once per request, uses DDL for declared
        foreign keys, and mines JOIN evidence from the structured SQL supplied
        with the current request. Unified semantic modeling receives a compact
        schema-first result with raw DDL only as fallback; compatibility callers
        retain detailed DDL and SQL-usage evidence. It never searches unrelated
        reference SQL; historical profiling remains a separate opt-in workflow.
        """
        try:
            normalized_tables = self._normalize_semantic_source_tables(tables)
            inspected_tables = []
            for table in normalized_tables:
                cache_key = (
                    str(catalog or "").strip().casefold(),
                    str(database or "").strip().casefold(),
                    str(schema_name or "").strip().casefold(),
                    self._normalize_identifier(table),
                )
                inspected = self._semantic_source_cache.get(cache_key)
                if inspected is None:
                    inspected = {"table_name": table}
                    ddl_result = self.db_tool.get_table_ddl(table, catalog, database, schema_name)
                    if ddl_result.success and isinstance(ddl_result.result, dict):
                        inspected["ddl"] = ddl_result.result
                    else:
                        inspected["ddl_error"] = ddl_result.error or "DDL unavailable"

                    schema_result = self.db_tool.describe_table(table, catalog, database, schema_name)
                    if schema_result.success and isinstance(schema_result.result, dict):
                        inspected["schema"] = schema_result.result
                    else:
                        inspected["schema_error"] = schema_result.error or "Schema unavailable"
                    self._semantic_source_cache[cache_key] = inspected
                inspected_tables.append(inspected)

            source_entries = self._current_source_sql_entries()
            sql_evidence = {}
            if not self.compact_source_inspection:
                sql_evidence, parse_errors = self._semantic_profile_sql_evidence(
                    source_entries,
                    normalized_tables,
                    max(len(normalized_tables), 1),
                )
            else:
                parse_errors = []
            relationships = self._extract_foreign_keys_from_inspected_tables(inspected_tables)
            tables_lower_map = {self._normalize_identifier(table.split(".")[-1]): table for table in normalized_tables}
            for entry in source_entries:
                sql_text = str(entry.get("sql") or "")
                parsed_expressions = None
                if self.compact_source_inspection:
                    try:
                        parsed_expressions = self._parse_sql(sql_text)
                    except Exception as exc:
                        parse_errors.append({"source_sql_name": entry.get("name") or "sql", "error": str(exc)})
                        continue
                relationships.extend(
                    self._extract_join_relationships_from_sql(
                        sql_text,
                        tables_lower_map,
                        parsed_expressions=parsed_expressions,
                    )
                )
            relationships = self._deduplicate_relationships(relationships)
            if not relationships:
                relationships = self._infer_relationships_from_inspected_tables(inspected_tables)

            public_tables = []
            for inspected in inspected_tables:
                public_table = {"table_name": inspected["table_name"]}
                if not self.compact_source_inspection and inspected.get("ddl") is not None:
                    public_table["ddl"] = inspected["ddl"]
                if inspected.get("schema") is not None:
                    public_table["schema"] = inspected["schema"]
                else:
                    public_table["schema_error"] = inspected.get("schema_error") or "Schema unavailable"
                    if inspected.get("ddl") is not None:
                        public_table["ddl"] = inspected["ddl"]
                if inspected.get("ddl_error"):
                    public_table["ddl_error"] = inspected["ddl_error"]
                if not self.compact_source_inspection:
                    table_profile = self._semantic_profile_table_evidence_for(
                        sql_evidence,
                        inspected["table_name"],
                    )
                    public_table["sql_usage"] = {
                        "query_count": table_profile.get("query_count", 0),
                        "field_usage_statistics": table_profile.get("field_usage_statistics", {}),
                        "aggregate_expressions": table_profile.get("aggregate_expressions", []),
                        "group_by_expressions": table_profile.get("group_by_expressions", []),
                        "common_filter_conditions": table_profile.get("common_filter_conditions", []),
                    }
                public_tables.append(public_table)

            return FuncToolResult(
                result={
                    "tables": public_tables,
                    "relationships": relationships,
                    "source_sql_count": len(source_entries),
                    "parse_errors": parse_errors[:5],
                    "summary": (
                        f"Inspected {len(inspected_tables)} table(s) and found "
                        f"{len(relationships)} relationship candidate(s) from the current request"
                    ),
                }
            )
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    def validate_semantic_key_candidates(
        self,
        candidates: List[SemanticKeyCandidate],
        catalog: Optional[str] = "",
        database: Optional[str] = "",
        schema_name: Optional[str] = "",
    ) -> FuncToolResult:
        """Verify all intended logical-key candidates in one read-only call.

        Historical JOINs and column names can suggest key columns, but they do
        not prove uniqueness. Each candidate is checked over all rows visible
        to the current datasource principal for NULL components and duplicate
        ordered key groups. A passing result may be authored as one OSI
        ``unique_keys`` entry. This tool never infers a physical
        ``primary_key``.

        Args:
            candidates: Complete ordered key candidates that the model intends
                to use. Submit all candidates in one call.
            catalog: Optional catalog override.
            database: Optional database override.
            schema_name: Optional schema override.
        """
        try:
            parsed_candidates = [SemanticKeyCandidate.model_validate(candidate) for candidate in candidates or []]
            if not parsed_candidates:
                return FuncToolResult(
                    success=0,
                    error="Provide every logical-key candidate that must be verified.",
                )

            validations = []
            errors = []
            seen = set()
            for candidate in parsed_candidates:
                normalized_columns = self._validate_key_candidate_columns(candidate.columns)
                identity = (
                    self._normalize_identifier(candidate.table_name),
                    tuple(self._normalize_identifier(column) for column in normalized_columns),
                )
                if identity in seen:
                    continue
                seen.add(identity)
                validation = self._validate_semantic_key_candidate(
                    table_name=candidate.table_name,
                    columns=normalized_columns,
                    catalog=catalog or "",
                    database=database or "",
                    schema_name=schema_name or "",
                )
                if validation.success:
                    validations.append(validation.result)
                else:
                    error = {
                        "table": candidate.table_name,
                        "columns": normalized_columns,
                        "error": validation.error or "Candidate key verification failed",
                    }
                    validations.append(error)
                    errors.append(error)

            result = {
                "validations": validations,
                "all_checks_completed": not errors,
                "summary": (
                    f"Verified {len(validations) - len(errors)} of {len(validations)} logical-key candidate(s)"
                ),
            }
            if errors:
                return FuncToolResult(
                    success=0,
                    error="One or more logical-key candidates could not be verified.",
                    result=result,
                )
            return FuncToolResult(result=result)
        except Exception as e:
            return FuncToolResult(success=0, error=str(e))

    def _validate_semantic_key_candidate(
        self,
        *,
        table_name: str,
        columns: List[str],
        catalog: str,
        database: str,
        schema_name: str,
    ) -> FuncToolResult:
        """Run exact full-table checks for one ordered logical-key candidate."""
        normalized_columns = self._validate_key_candidate_columns(columns)
        table_ref = self._key_validation_table_reference(
            table_name=table_name,
            catalog=catalog,
            database=database,
            schema_name=schema_name,
        )
        column_refs = [self._quote_sql_identifier(column, database) for column in normalized_columns]
        null_predicate = " OR ".join(f"{column_ref} IS NULL" for column_ref in column_refs)
        non_null_predicate = " AND ".join(f"{column_ref} IS NOT NULL" for column_ref in column_refs)
        grouped_columns = ", ".join(column_refs)

        summary_sql = (
            "SELECT COUNT(*) AS row_count, "
            f"SUM(CASE WHEN {null_predicate} THEN 1 ELSE 0 END) "
            f"AS null_key_rows FROM {table_ref}"
        )
        summary = self._run_profile_scalar_query(summary_sql, database)
        if summary.get("error"):
            return FuncToolResult(
                success=0,
                error=f"Candidate key summary check failed: {summary['error']}",
            )

        duplicate_sql = (
            "SELECT COUNT(*) AS duplicate_group_count, "
            "COALESCE(SUM(duplicate_count - 1), 0) AS duplicate_row_count "
            "FROM ("
            "SELECT COUNT(*) AS duplicate_count "
            f"FROM {table_ref} WHERE {non_null_predicate} "
            f"GROUP BY {grouped_columns} HAVING COUNT(*) > 1"
            ") duplicate_keys"
        )
        duplicate_stats = self._run_profile_scalar_query(duplicate_sql, database)
        if duplicate_stats.get("error"):
            return FuncToolResult(
                success=0,
                error=f"Candidate key duplicate check failed: {duplicate_stats['error']}",
            )

        row_count = self._required_profile_count(summary, "row_count")
        null_key_rows = self._optional_profile_count(summary, "null_key_rows", default=0)
        duplicate_group_count = self._required_profile_count(duplicate_stats, "duplicate_group_count")
        duplicate_row_count = self._optional_profile_count(duplicate_stats, "duplicate_row_count", default=0)
        is_non_null = null_key_rows == 0
        is_unique = duplicate_group_count == 0
        is_valid_logical_key = row_count > 0 and is_non_null and is_unique

        if row_count == 0:
            recommendation = "none"
            reason = "The table is empty, so the candidate has no supporting data."
        elif not is_non_null:
            recommendation = "none"
            reason = f"{null_key_rows} rows contain NULL in at least one key component."
        elif not is_unique:
            recommendation = "none"
            reason = f"{duplicate_group_count} duplicate key groups contain {duplicate_row_count} excess rows."
        else:
            recommendation = "unique_keys"
            reason = "The ordered columns are non-NULL and unique across the full table."

        return FuncToolResult(
            result={
                "table": table_name,
                "columns": normalized_columns,
                "verification_scope": "full_table",
                "access_scope": "rows visible to the current datasource principal",
                "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                "row_count": row_count,
                "null_key_rows": null_key_rows,
                "duplicate_group_count": duplicate_group_count,
                "duplicate_row_count": duplicate_row_count,
                "is_non_null": is_non_null,
                "is_unique": is_unique,
                "is_valid_logical_key": is_valid_logical_key,
                "recommended_osi_declaration": recommendation,
                "primary_key_inferred": False,
                "reason": reason,
                "verification_sql": {
                    "summary": summary_sql,
                    "duplicates": duplicate_sql,
                },
            }
        )

    def _normalize_semantic_source_tables(self, tables: List[str]) -> List[str]:
        """Return non-empty source table names without case-insensitive duplicates."""
        normalized = []
        seen = set()
        for raw_table in tables or []:
            table = str(raw_table or "").strip()
            identity = self._normalize_identifier(table)
            if not table or not identity or identity in seen:
                continue
            seen.add(identity)
            normalized.append(table)
        if not normalized:
            raise ValueError("Provide at least one physical source table to inspect.")
        return normalized

    def _current_source_sql_entries(self) -> List[Dict[str, Any]]:
        """Read structured SQL supplied with the current request, when available."""
        if self.source_sql_provider is None:
            return []
        try:
            entries = self.source_sql_provider() or []
        except Exception as exc:
            logger.warning("Failed to read request-local SQL evidence: %s", exc)
            return []
        return [
            {
                **entry,
                "name": entry.get("name") or entry.get("source_sql_name") or f"sql_{index + 1}",
                "sql": str(entry.get("sql") or ""),
            }
            for index, entry in enumerate(entries)
            if isinstance(entry, dict) and str(entry.get("sql") or "").strip()
        ]

    def _extract_foreign_keys_from_inspected_tables(
        self,
        inspected_tables: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract declared scalar or composite foreign keys from cached DDL."""
        relationships = []
        fk_pattern = (
            r"FOREIGN\s+KEY\s*\(([^)]+)\)\s*REFERENCES\s+"
            r"([^\s(]+)\s*\(([^)]+)\)"
        )
        for inspected in inspected_tables:
            ddl = inspected.get("ddl")
            ddl_text = str(ddl.get("definition") or "") if isinstance(ddl, dict) else ""
            for match in re.finditer(fk_pattern, ddl_text, re.IGNORECASE):
                source_columns = self._split_constraint_columns(match.group(1))
                target_columns = self._split_constraint_columns(match.group(3))
                if not source_columns or len(source_columns) != len(target_columns):
                    continue
                relationships.append(
                    self._relationship_evidence(
                        source_table=inspected["table_name"],
                        source_columns=source_columns,
                        target_table=match.group(2).strip().strip('`"[]'),
                        target_columns=target_columns,
                        confidence="high",
                        evidence="foreign_key",
                        target_key_status="declared",
                    )
                )
        return self._deduplicate_relationships(relationships)

    def _infer_relationships_from_inspected_tables(
        self,
        inspected_tables: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Infer low-confidence relationships from cached schema column names."""
        table_columns = {}
        tables_lower_map = {}
        for inspected in inspected_tables:
            table_name = inspected["table_name"]
            tables_lower_map[self._normalize_identifier(table_name.split(".")[-1])] = table_name
            schema = inspected.get("schema")
            columns = schema.get("columns", []) if isinstance(schema, dict) else []
            table_columns[table_name] = [column for column in columns if isinstance(column, dict)]

        relationships = []
        for source_table, columns in table_columns.items():
            for column in columns:
                source_column = str(column.get("name") or "")
                normalized_column = self._normalize_identifier(source_column)
                if not normalized_column.endswith("_id"):
                    continue
                target_table = tables_lower_map.get(normalized_column[:-3])
                if not target_table:
                    continue
                if not any(
                    self._normalize_identifier(target.get("name")) == "id"
                    for target in table_columns.get(target_table, [])
                ):
                    continue
                relationships.append(
                    self._relationship_evidence(
                        source_table=source_table,
                        source_columns=[source_column],
                        target_table=target_table,
                        target_columns=["id"],
                        confidence="low",
                        evidence="column_name",
                        target_key_status="candidate_unverified",
                    )
                )
        return self._deduplicate_relationships(relationships)

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
                self._load_sql_evidence_entries(sql_queries, sql_entries_json, query_text, tables, sample_sql_queries)
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
                        "Keep generated YAML concise: use profiling evidence to choose relationship "
                        "candidates, measures, dimensions, and time columns; historical joins do not "
                        "prove keys, so verify complete target column lists with "
                        "one validate_semantic_key_candidates call before adding unique_keys. Include compact distribution notes "
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

        def record(
            source_table: str,
            source_columns: List[str],
            target_table: str,
            target_columns: List[str],
        ) -> None:
            evidence = self._relationship_evidence(
                source_table=source_table,
                source_columns=source_columns,
                target_table=target_table,
                target_columns=target_columns,
                confidence="medium",
                evidence="historical_sql_join",
                target_key_status="candidate_unverified",
            )
            relationship = json.dumps(
                evidence,
                ensure_ascii=False,
                sort_keys=True,
            )
            table_stats[source_table]["join_relationships"][relationship] += 1
            table_stats[target_table]["join_relationships"][relationship] += 1

        grouped_pairs: Dict[tuple[str, str], List[tuple[str, str]]] = defaultdict(list)

        def collect_pair(left: Any, right: Any, joined_table: Optional[str] = None) -> None:
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                return
            left_table = alias_to_table.get(self._normalize_identifier(left.table))
            right_table = alias_to_table.get(self._normalize_identifier(right.table))
            if not left_table or not right_table or left_table == right_table:
                return

            if joined_table and left_table == joined_table:
                source_table, source_column = right_table, right.name
                target_table, target_column = left_table, left.name
            elif joined_table and right_table == joined_table:
                source_table, source_column = left_table, left.name
                target_table, target_column = right_table, right.name
            elif (right_table, left_table) in grouped_pairs:
                source_table, source_column = right_table, right.name
                target_table, target_column = left_table, left.name
            else:
                source_table, source_column = left_table, left.name
                target_table, target_column = right_table, right.name

            pair = (source_column, target_column)
            if pair not in grouped_pairs[(source_table, target_table)]:
                grouped_pairs[(source_table, target_table)].append(pair)

        for join in select.find_all(exp.Join):
            on_expression = join.args.get("on")
            if on_expression is None:
                continue
            joined_table = None
            if isinstance(join.this, exp.Table):
                joined_table = alias_to_table.get(
                    self._normalize_identifier(join.this.alias_or_name)
                ) or alias_to_table.get(self._normalize_identifier(join.this.name))
            for eq in self._collect_conjunctive_equalities(on_expression):
                collect_pair(eq.left, eq.right, joined_table)

        where_expression = select.args.get("where")
        if where_expression is not None:
            for eq in self._collect_conjunctive_equalities(where_expression):
                collect_pair(eq.left, eq.right)

        for (source_table, target_table), pairs in grouped_pairs.items():
            record(
                source_table,
                [pair[0] for pair in pairs],
                target_table,
                [pair[1] for pair in pairs],
            )

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

    @staticmethod
    def _collect_conjunctive_equalities(root: Any) -> List[Any]:
        """Collect equality predicates under conjunctions while skipping OR branches."""
        from sqlglot import expressions as exp

        equalities: List[Any] = []
        if root is None:
            return equalities

        def walk(node: Any) -> None:
            if node is None or not isinstance(node, exp.Expression):
                return
            if isinstance(node, exp.Or):
                return
            if isinstance(node, exp.EQ):
                equalities.append(node)
                return
            if isinstance(node, exp.Where):
                walk(node.this)
                return
            for child in node.iter_expressions():
                walk(child)

        walk(root)
        return equalities

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
        column_ref = self._quote_sql_identifier(column_name, database)
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
            top_sql = self._render_profile_limit(
                f"SELECT {column_ref} AS value, COUNT(*) AS count "
                f"FROM {table_ref} WHERE {column_ref} IS NOT NULL "
                f"GROUP BY {column_ref} ORDER BY count DESC",
                max(top_n, 1),
                database,
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
            left_ref = self._quote_sql_identifier(left_column, database)
            right_ref = self._quote_sql_identifier(right_column, database)
            sql = self._render_profile_limit(
                f"SELECT {left_ref} AS left_value, {right_ref} AS right_value "
                f"FROM {table_ref} WHERE {left_ref} IS NOT NULL AND {right_ref} IS NOT NULL",
                1000,
                database,
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
            target_table = str(relationship.get("target_table") or "")
            source_columns = [
                str(column)
                for column in (relationship.get("source_columns") or [relationship.get("source_column")])
                if column
            ]
            target_columns = [
                str(column)
                for column in (relationship.get("target_columns") or [relationship.get("target_column")])
                if column
            ]
            key = (
                source_table,
                tuple(source_columns),
                target_table,
                tuple(target_columns),
            )
            if (
                not source_table
                or not target_table
                or not source_columns
                or len(source_columns) != len(target_columns)
                or key in seen
            ):
                continue
            seen.add(key)
            source_ref = self._profile_table_reference(source_table, catalog, database, schema_name)
            target_ref = self._profile_table_reference(target_table, catalog, database, schema_name)
            source_col_refs = [self._quote_sql_identifier(column, database) for column in source_columns]
            target_col_refs = [self._quote_sql_identifier(column, database) for column in target_columns]
            join_condition = " AND ".join(
                f"src.{source_column} = tgt.{target_column}"
                for source_column, target_column in zip(source_col_refs, target_col_refs)
            )
            if len(source_columns) == 1:
                source_col_ref = source_col_refs[0]
                target_col_ref = target_col_refs[0]
                sql = (
                    "SELECT "
                    "COUNT(*) AS source_rows, "
                    f"COUNT(src.{source_col_ref}) AS non_null_source_rows, "
                    f"COUNT(DISTINCT src.{source_col_ref}) AS distinct_source_keys, "
                    f"COUNT(tgt.{target_col_ref}) AS matched_join_rows, "
                    f"COUNT(DISTINCT CASE WHEN tgt.{target_col_ref} IS NOT NULL "
                    f"THEN src.{source_col_ref} END) AS matched_distinct_source_keys "
                    f"FROM {source_ref} src LEFT JOIN {target_ref} tgt "
                    f"ON {join_condition}"
                )
            else:
                non_null_source = " AND ".join(f"src.{column} IS NOT NULL" for column in source_col_refs)
                matched_target = " AND ".join(f"tgt.{column} IS NOT NULL" for column in target_col_refs)
                sql = (
                    "SELECT "
                    "COUNT(*) AS source_rows, "
                    f"SUM(CASE WHEN {non_null_source} THEN 1 ELSE 0 END) "
                    "AS non_null_source_rows, "
                    f"SUM(CASE WHEN {matched_target} THEN 1 ELSE 0 END) "
                    "AS matched_join_rows "
                    f"FROM {source_ref} src LEFT JOIN {target_ref} tgt "
                    f"ON {join_condition}"
                )
            stats = self._run_profile_scalar_query(sql, database)
            profile = {
                "source_table": source_table,
                "source_column": source_columns[0],
                "source_columns": source_columns,
                "target_table": target_table,
                "target_column": target_columns[0],
                "target_columns": target_columns,
                "key_arity": len(source_columns),
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
        return ".".join(self._quote_sql_identifier(part, database) for part in parts)

    def _validate_key_candidate_columns(self, columns: List[str]) -> List[str]:
        if not isinstance(columns, list) or not columns:
            raise ValueError("columns must contain at least one candidate key column")
        normalized = [str(column or "").strip() for column in columns]
        if any(not column for column in normalized):
            raise ValueError("candidate key columns must not be empty")
        comparable = [self._normalize_identifier(column) for column in normalized]
        if len(set(comparable)) != len(comparable):
            raise ValueError("candidate key columns must not contain duplicates")
        return normalized

    def _key_validation_table_reference(
        self,
        table_name: str,
        catalog: str,
        database: str,
        schema_name: str,
    ) -> str:
        table_name = str(table_name or "").strip()
        if not table_name:
            raise ValueError("table_name is required")
        if "." in table_name:
            parts = [part.strip() for part in table_name.split(".")]
        else:
            parts = [
                str(part).strip() for part in (catalog, database, schema_name, table_name) if str(part or "").strip()
            ]
        if not parts or any(not part for part in parts):
            raise ValueError("table_name contains an empty qualified-name component")
        return ".".join(self._quote_sql_identifier(part, database) for part in parts)

    def _required_profile_count(self, stats: Dict[str, Any], key: str) -> int:
        value = self._profile_number(self._profile_stat_value(stats, key))
        if value is None or value < 0:
            raise ValueError(f"Candidate key verification did not return `{key}`")
        return int(value)

    def _optional_profile_count(self, stats: Dict[str, Any], key: str, default: int) -> int:
        value = self._profile_number(self._profile_stat_value(stats, key))
        if value is None:
            return default
        if value < 0:
            raise ValueError(f"Candidate key verification returned invalid `{key}`")
        return int(value)

    def _profile_stat_value(self, stats: Dict[str, Any], key: str) -> Any:
        if key in stats:
            return stats[key]
        normalized_key = self._normalize_identifier(key)
        for candidate, value in stats.items():
            if self._normalize_identifier(str(candidate)) == normalized_key:
                return value
        return None

    def _dialect_operations(self, database: str = "") -> Optional[Any]:
        resolver = getattr(self.db_tool, "dialect_operations", None)
        return resolver(database=database) if callable(resolver) else None

    def _render_profile_limit(self, sql: str, limit: int, database: str) -> str:
        operations = self._dialect_operations(database)
        if operations is not None:
            return operations.render_limit(sql, limit)
        return f"{sql} LIMIT {int(limit)}"

    def _quote_sql_identifier(self, value: str, database: str = "") -> str:
        operations = self._dialect_operations(database)
        if operations is not None:
            return operations.quote_identifier(value)
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

    def _extract_foreign_keys_from_ddl(
        self,
        tables: List[str],
        catalog: str,
        database: str,
        schema_name: str,
    ) -> List[Dict[str, Any]]:
        """Extract foreign-key evidence for legacy relationship-only callers."""
        inspected_tables = []
        for table in tables:
            ddl_result = self.db_tool.get_table_ddl(table, catalog, database, schema_name)
            inspected = {"table_name": table}
            if ddl_result.success and isinstance(ddl_result.result, dict):
                inspected["ddl"] = ddl_result.result
            inspected_tables.append(inspected)
        return self._extract_foreign_keys_from_inspected_tables(inspected_tables)

    def _analyze_join_patterns_from_history(
        self,
        tables: List[str],
        sample_size: int,
    ) -> List[Dict[str, Any]]:
        """Search indexed reference SQL for legacy relationship-only callers."""
        if not self.agent_config:
            return []
        from datus.storage.reference_sql.store import ReferenceSqlRAG

        sql_rag = ReferenceSqlRAG(self.agent_config, self.sub_agent_name)
        relationships = []
        tables_lower_map = {self._normalize_identifier(table.split(".")[-1]): table for table in tables}
        for table in tables:
            try:
                search_results = sql_rag.search_reference_sql(
                    query_text=f"JOIN {table}",
                    top_n=sample_size,
                )
                for sql_entry in search_results:
                    relationships.extend(
                        self._extract_join_relationships_from_sql(
                            str(sql_entry.get("sql") or ""),
                            tables_lower_map,
                        )
                    )
            except Exception as exc:
                logger.warning("Failed to search SQL history for table %s: %s", table, exc)
        return self._deduplicate_relationships(relationships)

    def _load_sql_evidence_entries(
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

    def _parse_sql(self, sql_text: str):
        """Parse one SQL string into sqlglot expressions."""
        import sqlglot

        configured_dialect = ""
        current_datasource = str(getattr(self.agent_config, "current_datasource", "") or "").strip()
        current_db_config = getattr(self.agent_config, "current_db_config", None)
        if callable(current_db_config):
            try:
                dialect_value = getattr(
                    current_db_config(current_datasource),
                    "type",
                    "",
                )
                configured_dialect = str(getattr(dialect_value, "value", dialect_value) or "").strip().lower()
            except Exception:
                configured_dialect = ""
        configured_dialect = parse_dialect(configured_dialect) if configured_dialect else ""
        dialects = [
            configured_dialect or None,
            None,
            "mysql",
            "hive",
            "spark",
            "bigquery",
            "snowflake",
            "postgres",
            "sqlite",
            "duckdb",
            "trino",
            "starrocks",
        ]
        errors = []
        seen = set()
        for dialect in dialects:
            if dialect in seen:
                continue
            seen.add(dialect)
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

    def _extract_join_relationships_from_sql(
        self,
        sql_text: str,
        tables_lower_map: Dict[str, str],
        *,
        parsed_expressions: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Extract alias-aware scalar or composite JOIN relationships."""
        from sqlglot import expressions as exp

        relationships: List[Dict[str, Any]] = []
        if not sql_text:
            return relationships
        if parsed_expressions is None:
            try:
                parsed_expressions = self._parse_sql(sql_text)
            except Exception:
                return relationships

        for parsed in parsed_expressions:
            alias_to_table = self._alias_to_table_map(parsed)
            grouped_pairs: Dict[tuple[str, str], List[tuple[str, str]]] = defaultdict(list)

            def collect_pair(
                left: Any,
                right: Any,
                joined_table: Optional[str],
                current_alias_to_table: Dict[str, str],
                current_grouped_pairs: Dict[tuple[str, str], List[tuple[str, str]]],
            ) -> None:
                if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                    return
                left_table = self._resolve_column_table(left, current_alias_to_table, tables_lower_map)
                right_table = self._resolve_column_table(right, current_alias_to_table, tables_lower_map)
                if not left_table or not right_table or left_table == right_table:
                    return

                if joined_table and left_table == joined_table:
                    source_table, source_column = right_table, right.name
                    target_table, target_column = left_table, left.name
                elif joined_table and right_table == joined_table:
                    source_table, source_column = left_table, left.name
                    target_table, target_column = right_table, right.name
                elif (right_table, left_table) in current_grouped_pairs:
                    source_table, source_column = right_table, right.name
                    target_table, target_column = left_table, left.name
                else:
                    source_table, source_column = left_table, left.name
                    target_table, target_column = right_table, right.name

                pair = (source_column, target_column)
                if pair not in current_grouped_pairs[(source_table, target_table)]:
                    current_grouped_pairs[(source_table, target_table)].append(pair)

            for join in parsed.find_all(exp.Join):
                on_expression = join.args.get("on")
                if on_expression is None:
                    continue
                joined_table = self._resolve_join_target_table(join, alias_to_table, tables_lower_map)
                for eq in self._collect_conjunctive_equalities(on_expression):
                    collect_pair(
                        eq.left,
                        eq.right,
                        joined_table,
                        alias_to_table,
                        grouped_pairs,
                    )

            where_expression = parsed.args.get("where")
            if where_expression is not None:
                for eq in self._collect_conjunctive_equalities(where_expression):
                    collect_pair(
                        eq.left,
                        eq.right,
                        None,
                        alias_to_table,
                        grouped_pairs,
                    )

            for (source_table, target_table), pairs in grouped_pairs.items():
                relationships.append(
                    self._relationship_evidence(
                        source_table=source_table,
                        source_columns=[pair[0] for pair in pairs],
                        target_table=target_table,
                        target_columns=[pair[1] for pair in pairs],
                        confidence="medium",
                        evidence="join_pattern",
                        target_key_status="candidate_unverified",
                    )
                )
        return self._deduplicate_relationships(relationships)

    def _resolve_join_target_table(
        self,
        join: Any,
        alias_to_table: Dict[str, str],
        tables_lower_map: Dict[str, str],
    ) -> Optional[str]:
        from sqlglot import expressions as exp

        target = join.this
        if not isinstance(target, exp.Table):
            return None
        for candidate in (target.alias_or_name, target.name):
            resolved = alias_to_table.get(self._normalize_identifier(candidate), candidate)
            canonical = tables_lower_map.get(self._normalize_identifier(resolved))
            if canonical:
                return canonical
        return None

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

    def _normalize_identifier(self, value: str) -> str:
        """Normalize SQL identifiers for comparisons."""
        return (value or "").strip().strip('"`[]').lower()

    def _split_constraint_columns(self, value: str) -> List[str]:
        return [column.strip().strip('`"[]') for column in str(value or "").split(",") if column.strip().strip('`"[]')]

    def _relationship_evidence(
        self,
        *,
        source_table: str,
        source_columns: List[str],
        target_table: str,
        target_columns: List[str],
        confidence: str,
        evidence: str,
        target_key_status: str,
    ) -> Dict[str, Any]:
        if not source_columns or len(source_columns) != len(target_columns):
            raise ValueError("relationship evidence requires equally sized non-empty column lists")
        return {
            "source_table": source_table,
            "source_column": source_columns[0],
            "source_columns": list(source_columns),
            "target_table": target_table,
            "target_column": target_columns[0],
            "target_columns": list(target_columns),
            "key_arity": len(source_columns),
            "confidence": confidence,
            "evidence": evidence,
            "target_key_status": target_key_status,
            "requires_target_key_validation": target_key_status == "candidate_unverified",
        }

    def _infer_from_column_names(
        self,
        tables: List[str],
        catalog: str,
        database: str,
        schema_name: str,
    ) -> List[Dict[str, Any]]:
        """Infer relationships for legacy relationship-only callers."""
        inspected_tables = []
        for table in tables:
            schema_result = self.db_tool.describe_table(table, catalog, database, schema_name)
            inspected = {"table_name": table}
            if schema_result.success and isinstance(schema_result.result, dict):
                inspected["schema"] = schema_result.result
            inspected_tables.append(inspected)
        return self._infer_relationships_from_inspected_tables(inspected_tables)

    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate and sort relationships by confidence."""
        seen = set()
        deduplicated = []

        # Sort by confidence (high > medium > low)
        confidence_order = {"high": 0, "medium": 1, "low": 2}
        sorted_rels = sorted(
            relationships,
            key=lambda r: confidence_order.get(r.get("confidence", ""), 3),
        )

        for rel in sorted_rels:
            normalized = dict(rel)
            source_columns = list(
                normalized.get("source_columns")
                or ([normalized["source_column"]] if normalized.get("source_column") else [])
            )
            target_columns = list(
                normalized.get("target_columns")
                or ([normalized["target_column"]] if normalized.get("target_column") else [])
            )
            if not source_columns or len(source_columns) != len(target_columns):
                continue
            normalized["source_column"] = source_columns[0]
            normalized["source_columns"] = source_columns
            normalized["target_column"] = target_columns[0]
            normalized["target_columns"] = target_columns
            normalized["key_arity"] = len(source_columns)
            key = (
                normalized["source_table"],
                tuple(source_columns),
                normalized["target_table"],
                tuple(target_columns),
            )
            if key not in seen:
                seen.add(key)
                deduplicated.append(normalized)

        return deduplicated
