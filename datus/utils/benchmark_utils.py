# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Benchmark evaluation utilities for comparing SQL execution results.

This module provides utilities for:
- Executing SQL queries and saving results
- Comparing CSV results with gold standards
- Evaluating benchmark accuracy
- Generating evaluation reports
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import pandas as pd
import yaml

from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import extract_table_names, metadata_identifier, parse_metadata

logger = get_logger(__name__)

FAIL_STATUSES = {"pending", "failed", "error"}


@dataclass
class WorkflowOutput:
    node_id: Optional[str]
    success: bool
    status: str
    error: Optional[str] = None


@dataclass
class WorkflowAnalysis:
    task_id: str
    completion_time: Optional[str]
    status: str
    total_nodes: int
    node_types: Dict[str, int] = field(default_factory=dict)
    outputs: list[WorkflowOutput] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def output_nodes(self) -> int:
        return len(self.outputs)

    @property
    def output_success(self) -> int:
        return sum(1 for output in self.outputs if output.success and output.status not in FAIL_STATUSES)

    @property
    def output_failure(self) -> int:
        return self.output_nodes - self.output_success


@dataclass
class ComparisonOutcome:
    match_rate: float = 0.0
    matched_columns: list[tuple[str, str]] = field(default_factory=list)
    unmatched_columns: list[str] = field(default_factory=list)
    unnecessary_columns: list[str] = field(default_factory=list)
    actual_shape: Optional[tuple[int, int]] = None
    expected_shape: Optional[tuple[int, int]] = None
    actual_preview: Optional[str] = None
    expected_preview: Optional[str] = None
    actual_tables: list[str] = field(default_factory=list)
    expected_tables: list[str] = field(default_factory=list)
    matched_tables: list[str] = field(default_factory=list)
    actual_sql_error: Optional[str] = None
    expected_sql_error: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def with_error(cls, message: str) -> "ComparisonOutcome":
        return cls(error=message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_rate": self.match_rate,
            "matched_columns": self.matched_columns,
            "un_matched_columns": self.unmatched_columns,
            "unnecessary_columns": self.unnecessary_columns,
            "actual_shape": self.actual_shape,
            "expected_shape": self.expected_shape,
            "actual_preview": self.actual_preview,
            "expected_preview": self.expected_preview,
            "actual_tables": self.actual_tables,
            "expected_tables": self.expected_tables,
            "matched_tables": self.matched_tables,
            "actual_sql_error": self.actual_sql_error,
            "expected_sql_error": self.expected_sql_error,
            "error": self.error,
        }


@dataclass
class ResultData:
    task_id: str
    source: str
    dataframe: Optional[pd.DataFrame] = None
    error: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.dataframe is not None and self.error is None


@dataclass
class SqlData:
    task_id: str
    source: str
    sql: Optional[str] = None
    tables: list[str] = field(default_factory=list)
    dialect: Optional[str] = None
    error: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.sql is not None and self.error is None


class SqlProvider(Protocol):
    def fetch(self, task_id: str) -> SqlData:
        ...


def csv_str_to_pands(csv_str: str) -> pd.DataFrame:
    with io.StringIO(csv_str) as csv_buffer:
        return pd.read_csv(csv_buffer)


def _normalize_field_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_") if isinstance(name, str) else ""


def _select_from_mapping(mapping: Mapping[str, Any], candidates: Sequence[str]) -> Tuple[Optional[str], Optional[Any]]:
    if not isinstance(mapping, Mapping):
        return None, None
    normalized = {_normalize_field_name(key): key for key in mapping.keys() if isinstance(key, str)}
    for candidate in candidates:
        key_normalized = _normalize_field_name(candidate)
        if key_normalized in normalized:
            actual_key = normalized[key_normalized]
            value = mapping.get(actual_key)
            if value is not None and value != "":
                return actual_key, value
    return None, None


def _unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    result: list[str] = []
    for item in items:
        if not item:
            continue
        normalized = str(item).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def collect_sql_tables(sql_text: Optional[str], dialect: Optional[str] = None) -> list[str]:
    if not sql_text:
        return []

    dialect = dialect or DBType.SNOWFLAKE
    tables: list[str] = []

    try:
        metadata = parse_metadata(sql_text, dialect=dialect)
    except Exception:
        metadata = None

    if metadata:
        table_info = metadata.get("table", {})
        table_name = table_info.get("name")
        if table_name:
            identifier = metadata_identifier(
                catalog_name=table_info.get("catalog_name", ""),
                database_name=table_info.get("database_name", ""),
                schema_name=table_info.get("schema_name", ""),
                table_name=table_name,
                dialect=dialect,
            )
            tables.append(identifier or table_name)

    try:
        extracted_tables = extract_table_names(sql_text, dialect=dialect)
        tables.extend(extracted_tables)
    except Exception:
        # Ignore extraction errors; tables list may remain empty
        pass

    return _unique_preserve_order(tables)


def compute_table_matches(actual_tables: Iterable[str], expected_tables: Iterable[str]) -> list[str]:
    normalized_actual = {_normalize_field_name(table): table for table in actual_tables if table}
    matches: list[str] = []
    for table in expected_tables:
        if not table:
            continue
        normalized = _normalize_field_name(table)
        if normalized in normalized_actual:
            matches.append(table)
    return _unique_preserve_order(matches)


@dataclass
class ComparisonRecord:
    task_id: str
    actual: ResultData
    expected: ResultData
    outcome: Optional[ComparisonOutcome] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "actual_file_exists": self.actual.available,
            "gold_file_exists": self.expected.available,
            "actual_path": self.actual.source,
            "gold_path": self.expected.source,
            "comparison": self.outcome.to_dict() if self.outcome else None,
        }


@dataclass
class TaskEvaluation:
    task_id: str
    analysis: WorkflowAnalysis
    comparisons: list[ComparisonRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_nodes": self.analysis.total_nodes,
            "output_nodes": self.analysis.output_nodes,
            "output_success": self.analysis.output_success,
            "output_failure": self.analysis.output_failure,
            "errors": list(self.analysis.errors),
            "node_types": dict(self.analysis.node_types),
            "completion_time": self.analysis.completion_time,
            "status": self.analysis.status,
            "comparison_results": [record.to_dict() for record in self.comparisons],
        }


@dataclass
class EvaluationReport:
    status: str
    generated_time: str
    summary: Dict[str, Any]
    task_ids: Dict[str, str]
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "generated_time": self.generated_time,
            "summary": self.summary,
            "task_ids": self.task_ids,
            "details": self.details,
        }


class ResultProvider(Protocol):
    def fetch(self, task_id: str) -> ResultData:
        ...


class CsvPerTaskResultProvider(ResultProvider):
    def __init__(self, directory: str):
        self.directory = Path(directory)

    def fetch(self, task_id: str) -> ResultData:
        csv_path = self.directory / f"{task_id}.csv"
        source = str(csv_path)
        if not csv_path.exists():
            return ResultData(task_id=task_id, source=source, error=f"Result file not found: {csv_path}")
        try:
            dataframe = pd.read_csv(csv_path)
        except Exception as exc:  # pragma: no cover - logging side effect
            return ResultData(task_id=task_id, source=source, error=f"Error loading CSV: {exc}")
        return ResultData(task_id=task_id, source=source, dataframe=dataframe)


class SingleFileGoldProvider(ResultProvider):
    def __init__(self, result_file: str, task_id_key: str = "task_id", query_result_key: str = "Expected answer"):
        self.task_id_key = task_id_key
        self.query_result_key = query_result_key
        self.result_file = Path(result_file)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._errors: Dict[str, str] = {}
        self._loaded = False
        self._global_error: Optional[str] = None

    def fetch(self, task_id: str) -> ResultData:
        self._ensure_loaded()
        source = str(self.result_file)
        if self._global_error:
            return ResultData(task_id=task_id, source=source, error=self._global_error)
        if task_id in self._errors:
            return ResultData(task_id=task_id, source=source, error=self._errors[task_id])
        dataframe = self._cache.get(task_id)
        if dataframe is None:
            return ResultData(task_id=task_id, source=source, error=f"Result not found for task_id={task_id}")
        return ResultData(task_id=task_id, source=source, dataframe=dataframe)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.result_file.exists():
            self._global_error = f"Result file not found: {self.result_file}"
            return

        suffix = self.result_file.suffix.lower()
        try:
            if suffix == ".json":
                with self.result_file.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                self._ingest_json_payload(payload)
            elif suffix == ".csv":
                df = pd.read_csv(self.result_file)
                if self.task_id_key not in df.columns:
                    self._global_error = (
                        f"CSV result file must contain a '{self.task_id_key}' column to disambiguate records."
                    )
                    return
                if self.query_result_key not in df.columns:
                    self._global_error = (
                        f"CSV result file must contain a '{self.query_result_key}' column with the expected answers."
                    )
                    return

                for _, row in df.iterrows():
                    task_id = str(row[self.task_id_key])
                    expected_value = str(row.get(self.query_result_key, "") or "")
                    if not expected_value:
                        self._errors[task_id] = f"Empty `{self.query_result_key}` value"
                        continue
                    parsed_df = self._parse_csv_to_df(expected_value)
                    if parsed_df is None:
                        self._errors[task_id] = f"Invalid `{self.query_result_key}` value"
                        continue
                    self._cache[task_id] = parsed_df
            else:
                self._global_error = f"Unsupported result file format: {self.result_file.suffix}"
        except Exception as exc:  # pragma: no cover - logging side effect
            self._global_error = f"Failed to load result file: {exc}"

    def _ingest_json_payload(self, data: Any) -> None:
        handled = False

        if isinstance(data, dict):
            for task_id, values in data.items():
                df = None
                if isinstance(values, dict):
                    df = self._ingest_result_to_df(values)
                elif isinstance(values, str):
                    value_dict = None
                    try:
                        value_dict = json.loads(values)
                    except json.JSONDecodeError:
                        pass
                    if value_dict is not None:
                        df = self._ingest_result_to_df(value_dict)
                    else:
                        df = self._parse_csv_to_df(str(values))
                else:
                    self._global_error = "Unsupported JSON result format"
                if df:
                    self._cache[task_id] = df
                    handled = True

        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    logger.warning(f"Unsupported json line format: {item}")
                    continue
                if self.task_id_key not in item:
                    logger.warning(f"Result file must contain a '{self.task_id_key}' column to disambiguate records.")
                    continue
                if self.query_result_key not in item:
                    logger.warning(f"Result file must contain a '{self.query_result_key}' column to parse result.")
                    continue
                df = self._parse_csv_to_df(str(item[self.query_result_key]))
                if df:
                    self._cache[item[self.task_id_key]] = df
                    handled = True
        else:
            self._global_error = "Unsupported JSON result format"

        if not handled and not self._global_error:
            self._global_error = "Unsupported JSON result format"

    def _ingest_result_to_df(self, data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        if self.query_result_key not in data:
            self._global_error = f"Column `{self.query_result_key}` not found in json line"

        return self._parse_csv_to_df(data[self.query_result_key])

    def _parse_csv_to_df(self, csv_str: str) -> Optional[pd.DataFrame]:
        try:
            return csv_str_to_pands(csv_str)
        except Exception as exc:
            logger.warning(f"Failed to parse result: {exc}")
            self._global_error = (
                f"The value corresponding to the field `{self.query_result_key}` " f"should be in csv format"
            )
            return None


class DirectorySqlProvider(SqlProvider):
    def __init__(self, directory: str, suffix: str = ".sql", dialect: str = DBType.SNOWFLAKE):
        self.directory = Path(directory)
        self.suffix = suffix
        self.dialect = dialect

    def fetch(self, task_id: str) -> SqlData:
        sql_path = self.directory / f"{task_id}{self.suffix}"
        source = str(sql_path)
        if not sql_path.exists():
            return SqlData(
                task_id=task_id, source=source, error=f"SQL file not found: {sql_path}", dialect=self.dialect
            )
        try:
            sql_text = sql_path.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover - logging side effect
            return SqlData(
                task_id=task_id,
                source=source,
                error=f"Failed to read SQL file: {exc}",
                dialect=self.dialect,
            )
        tables = collect_sql_tables(sql_text, self.dialect)
        return SqlData(task_id=task_id, source=source, sql=sql_text, tables=tables, dialect=self.dialect)


class AgentResultSqlProvider(SqlProvider):
    """
    Parser for JSON files output by Datus-Agent
    """

    def __init__(self, result_dir: str, dialect: str = DBType.SNOWFLAKE):
        self.result_dir = Path(result_dir)
        self.dialect = dialect

    def fetch(self, task_id: str) -> SqlData:
        if not self.result_dir.exists():
            return SqlData(
                task_id=task_id,
                source=str(self.result_dir),
                error=f"Result directory not found: {self.result_dir}",
                dialect=self.dialect,
            )

        sql_path = self.result_dir / f"{task_id}.sql"
        if sql_path.exists():
            try:
                sql_text = sql_path.read_text(encoding="utf-8")
            except Exception as exc:
                return SqlData(
                    task_id=task_id,
                    source=str(sql_path),
                    error=f"Failed to read SQL file: {exc}",
                    dialect=self.dialect,
                )
            tables = collect_sql_tables(sql_text, self.dialect)
            return SqlData(task_id=task_id, source=str(sql_path), sql=sql_text, tables=tables, dialect=self.dialect)

        json_path = self.result_dir / f"{task_id}.json"
        if json_path.exists():
            try:
                with json_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                _, sql_value = _select_from_mapping(payload, ["gen_sql_final", "gen_sql"])
                if sql_value is None:
                    raise KeyError("gen_sql")
                sql_text = str(sql_value)
            except Exception as exc:
                return SqlData(
                    task_id=task_id,
                    source=str(json_path),
                    error=f"Failed to load SQL from JSON: {exc}",
                    dialect=self.dialect,
                )
            tables = collect_sql_tables(sql_text, self.dialect)
            return SqlData(task_id=task_id, source=str(json_path), sql=sql_text, tables=tables, dialect=self.dialect)

        return SqlData(
            task_id=task_id,
            source=str(sql_path),
            error=f"No SQL artifacts found for task {task_id} in {self.result_dir}",
            dialect=self.dialect,
        )


class JsonMappingSqlProvider(SqlProvider):
    def __init__(
        self,
        json_path: str,
        task_id_key: str = "task_id",
        sql_key="SQL",
        dialect: str = DBType.SQLITE,
    ):
        self.json_path = Path(json_path)
        self.task_id_key = task_id_key
        self.sql_key = sql_key
        self.dialect = dialect
        self._cache: Dict[str, str] = {}
        self._loaded = False
        self._global_error: Optional[str] = None

    def fetch(self, task_id: str) -> SqlData:
        self._ensure_loaded()
        source = str(self.json_path)
        if self._global_error:
            return SqlData(task_id=task_id, source=source, error=self._global_error, dialect=self.dialect)

        sql_text = self._cache.get(str(task_id))
        if not sql_text:
            return SqlData(
                task_id=task_id,
                source=source,
                error=f"SQL not found for task_id={task_id}",
                dialect=self.dialect,
            )

        tables = collect_sql_tables(sql_text, self.dialect)
        return SqlData(
            task_id=task_id,
            source=source,
            sql=sql_text,
            tables=tables,
            dialect=self.dialect,
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.json_path.exists():
            self._global_error = f"SQL reference file not found: {self.json_path}"
            return
        try:
            with self.json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self._ingest(payload)
        except Exception as exc:  # pragma: no cover - logging side effect
            self._global_error = f"Failed to load SQL reference file: {exc}"
            return

        if not self._cache and not self._global_error:
            self._global_error = f"No SQL entries found in {self.json_path}"

    def _ingest(self, payload: Any) -> None:
        if isinstance(payload, Mapping):
            self._consume_record(payload)
            for value in payload.values():
                self._ingest(value)
        elif isinstance(payload, list):
            for item in payload:
                self._ingest(item)

    def _consume_record(self, record: Mapping[str, Any]) -> None:
        if not isinstance(record, Mapping):
            return
        task_id_value = record.get(self.task_id_key)
        sql_value = record.get(self.sql_key)
        if not sql_value or not task_id_value:
            logger.warning(f"This item must contain {self.task_id_key} and {self.sql_key}")
            return

        task_id = str(task_id_value)
        if task_id in self._cache:
            return
        sql_text = str(sql_value)
        self._cache[task_id] = sql_text


class CsvColumnSqlProvider(SqlProvider):
    def __init__(
        self,
        csv_path: str = "",
        task_id_key: str = "task_id",
        sql_key="SQL",
        dialect: str = DBType.SQLITE,
    ):
        self.csv_path = Path(csv_path)
        self.task_id_key = task_id_key
        self.sql_key = sql_key
        self.dialect = dialect
        self._cache: Dict[str, str] = {}
        self._loaded = False
        self._global_error: Optional[str] = None

    def fetch(self, task_id: str) -> SqlData:
        self._ensure_loaded()
        source = str(self.csv_path)
        if self._global_error:
            return SqlData(task_id=task_id, source=source, error=self._global_error, dialect=self.dialect)

        sql_text = self._cache.get(str(task_id))
        if not sql_text:
            return SqlData(
                task_id=task_id,
                source=source,
                error=f"SQL not found for task_id={task_id}",
                dialect=self.dialect,
            )

        tables = collect_sql_tables(sql_text, self.dialect)
        return SqlData(task_id=task_id, source=source, sql=sql_text, tables=tables, dialect=self.dialect)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.csv_path.exists():
            self._global_error = f"SQL reference file not found: {self.csv_path}"
            return

        try:
            with self.csv_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    self._global_error = f"CSV file {self.csv_path} has no header row"
                    return
                # header_map = {_normalize_field_name(name): name for name in reader.fieldnames if name}
                for row in reader:
                    if not row:
                        continue
                    task_id = str(row.get(self.task_id_key, "")).strip()
                    if not task_id or task_id in self._cache:
                        continue
                    sql_text = row.get(self.sql_key, "")
                    if not sql_text:
                        continue
                    self._cache[task_id] = sql_text
        except Exception as exc:  # pragma: no cover - logging side effect
            self._global_error = f"Failed to load SQL reference CSV: {exc}"
            return

        if not self._cache and not self._global_error:
            self._global_error = f"No SQL entries found in {self.csv_path}"

    @staticmethod
    def _resolve_column(header_map: Dict[str, str], candidates: Sequence[str]) -> Optional[str]:
        for candidate in candidates:
            normalized = _normalize_field_name(candidate)
            if normalized in header_map:
                return header_map[normalized]
        return None


class TrajectoryParser:
    def parse(self, filepath: Path, task_id: str) -> WorkflowAnalysis:
        errors: list[str] = []
        node_types: defaultdict[str, int] = defaultdict(int)
        outputs: list[WorkflowOutput] = []

        try:
            with filepath.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
        except FileNotFoundError:
            errors.append(f"Trajectory file not found: {filepath}")
            return WorkflowAnalysis(
                task_id=task_id,
                completion_time=None,
                status="missing",
                total_nodes=0,
                node_types={},
                outputs=outputs,
                errors=errors,
            )
        except Exception as exc:
            errors.append(f"Failed to read trajectory file: {exc}")
            return WorkflowAnalysis(
                task_id=task_id,
                completion_time=None,
                status="error",
                total_nodes=0,
                node_types={},
                outputs=outputs,
                errors=errors,
            )

        workflow = data.get("workflow") if isinstance(data, dict) else None
        if not workflow:
            errors.append("Invalid trajectory file format")
            return WorkflowAnalysis(
                task_id=task_id,
                completion_time=None,
                status="invalid",
                total_nodes=0,
                node_types={},
                outputs=outputs,
                errors=errors,
            )

        completion_time = workflow.get("completion_time")
        status = workflow.get("status", "unknown")

        nodes = workflow.get("nodes")
        total_nodes = 0

        if nodes:
            for node in nodes:
                if not node:
                    continue
                total_nodes += 1
                node_type = node.get("type", "unknown")
                node_types[node_type] += 1

                if node_type == "output":
                    result = node.get("result") or {}
                    success = bool(result.get("success"))
                    node_status = str(result.get("status", "unknown")).lower()
                    error_message = result.get("error")
                    outputs.append(
                        WorkflowOutput(
                            node_id=node.get("id"),
                            success=success,
                            status=node_status,
                            error=error_message if error_message else None,
                        )
                    )
                    if not success or node_status in FAIL_STATUSES:
                        msg = error_message or f"Output status is '{node_status}'"
                        errors.append(f"node {node.get('id', 'unknown')}: {msg}")
        else:
            node_type = workflow.get("type")
            if node_type:
                total_nodes = 1
                node_types[node_type] += 1
                if node_type == "output":
                    result = workflow.get("result") or {}
                    success = bool(result.get("success"))
                    node_status = str(result.get("status", "unknown")).lower()
                    error_message = result.get("error")
                    outputs.append(
                        WorkflowOutput(
                            node_id=workflow.get("id"),
                            success=success,
                            status=node_status,
                            error=error_message if error_message else None,
                        )
                    )
                    if not success or node_status in FAIL_STATUSES:
                        msg = error_message or f"Output status is '{node_status}'"
                        errors.append(f"workflow: {msg}")

        return WorkflowAnalysis(
            task_id=task_id,
            completion_time=completion_time,
            status=status,
            total_nodes=total_nodes,
            node_types=dict(node_types),
            outputs=outputs,
            errors=errors,
        )


class TableComparator:
    def compare(self, actual_df: pd.DataFrame, expected_df: pd.DataFrame) -> ComparisonOutcome:
        try:
            actual_preview = preview_dataframe(actual_df)
            expected_preview = preview_dataframe(expected_df)
            compares_result = compare_pandas_tables(actual_df, expected_df)
            return ComparisonOutcome(
                match_rate=compares_result["match_rate"],
                matched_columns=compares_result["matched_columns"],
                unmatched_columns=compares_result["un_matched_columns"],
                unnecessary_columns=compares_result["unnecessary_columns"],
                actual_shape=actual_df.shape,
                expected_shape=expected_df.shape,
                actual_preview=actual_preview,
                expected_preview=expected_preview,
            )
        except Exception as exc:  # pragma: no cover - logging side effect
            return ComparisonOutcome.with_error(f"Comparison error: {exc}")


class EvaluationReportBuilder:
    def build(self, evaluations: Mapping[str, TaskEvaluation]) -> EvaluationReport:
        total_files = len(evaluations)
        total_output_nodes = 0
        total_output_success = 0
        total_output_failure = 0

        total_comparisons = 0
        successful_matches = 0
        mismatches = 0
        comparison_errors = 0
        empty_result_errors = 0

        failed_task_ids: set[str] = set()
        matched_task_ids: set[str] = set()
        mismatched_task_ids: set[str] = set()
        empty_result_task_ids: set[str] = set()

        for task_id, evaluation in evaluations.items():
            analysis = evaluation.analysis

            total_output_nodes += analysis.output_nodes
            total_output_success += analysis.output_success
            total_output_failure += analysis.output_failure

            if analysis.output_failure > 0 or analysis.errors:
                failed_task_ids.add(task_id)

            for comparison in evaluation.comparisons:
                outcome = comparison.outcome
                if outcome is None:
                    continue

                total_comparisons += 1

                if outcome.error:
                    if "No columns to parse from file" in outcome.error:
                        empty_result_errors += 1
                        empty_result_task_ids.add(task_id)
                    else:
                        comparison_errors += 1
                    continue

                if outcome.match_rate == 1:
                    successful_matches += 1
                    matched_task_ids.add(task_id)
                else:
                    mismatches += 1
                    mismatched_task_ids.add(task_id)

        success_rate = (total_output_success / total_output_nodes * 100) if total_output_nodes else 0.0
        match_rate = (successful_matches / total_comparisons * 100) if total_comparisons else 0.0

        summary = {
            "total_files": total_files,
            "total_output_nodes": total_output_nodes,
            "total_output_success": total_output_success,
            "total_output_failure": total_output_failure,
            "success_rate": round(success_rate, 2),
            "comparison_summary": {
                "total_comparisons": total_comparisons,
                "successful_matches": successful_matches,
                "mismatches": mismatches,
                "comparison_errors": comparison_errors,
                "empty_result_errors": empty_result_errors,
                "match_rate": round(match_rate, 2),
            },
        }

        task_ids = {
            "failed_task_ids": ",".join(map(str, sorted(failed_task_ids))),
            "matched_task_ids": ",".join(map(str, sorted(matched_task_ids))),
            "mismatched_task_ids": ",".join(map(str, sorted(mismatched_task_ids))),
            "empty_result_task_ids": ",".join(map(str, sorted(empty_result_task_ids))),
        }

        details = {task_id: evaluation.to_dict() for task_id, evaluation in evaluations.items()}

        return EvaluationReport(
            status="success",
            generated_time=datetime.now().isoformat(),
            summary=summary,
            task_ids=task_ids,
            details=details,
        )


class BenchmarkEvaluator:
    def __init__(
        self,
        *,
        trajectory_parser: Optional[TrajectoryParser],
        result_provider: ResultProvider,
        gold_result_provider: ResultProvider,
        result_sql_provider: Optional[SqlProvider] = None,
        gold_sql_provider: Optional[SqlProvider] = None,
        comparator: Optional[TableComparator] = None,
        report_builder: Optional[EvaluationReportBuilder] = None,
    ):
        self.parser = trajectory_parser or TrajectoryParser()
        self.result_provider = result_provider
        self.gold_result_provider = gold_result_provider
        self.result_sql_provider = result_sql_provider
        self.gold_sql_provider = gold_sql_provider
        self.comparator = comparator or TableComparator()
        self.report_builder = report_builder or EvaluationReportBuilder()

    def evaluate_directory(
        self, trajectory_dir: str, target_task_ids: Optional[Iterable[str]] = None
    ) -> EvaluationReport:
        trajectories = collect_latest_trajectory_files(trajectory_dir)
        if target_task_ids:
            target_ids = {str(task_id) for task_id in target_task_ids}
            trajectories = {task_id: path for task_id, path in trajectories.items() if task_id in target_ids}
        return self.evaluate(trajectories)

    def evaluate(self, trajectories: Mapping[str, Path]) -> EvaluationReport:
        evaluations: Dict[str, TaskEvaluation] = {}
        for task_id, trajectory_path in trajectories.items():
            analysis = self.parser.parse(trajectory_path, task_id)
            comparison_records: list[ComparisonRecord] = []

            for output in analysis.outputs:
                if output.success and output.status not in FAIL_STATUSES:
                    comparison_records.append(self._build_comparison_record(task_id))

            evaluations[task_id] = TaskEvaluation(
                task_id=task_id,
                analysis=analysis,
                comparisons=comparison_records,
            )

        return self.report_builder.build(evaluations)

    def _build_comparison_record(self, task_id: str) -> ComparisonRecord:
        actual = self.result_provider.fetch(task_id)
        expected = self.gold_result_provider.fetch(task_id)
        record = ComparisonRecord(task_id=task_id, actual=actual, expected=expected)

        actual_sql = self._fetch_sql_data(self.result_sql_provider, task_id)
        expected_sql = self._fetch_sql_data(self.gold_sql_provider, task_id)

        if not actual.available:
            outcome = ComparisonOutcome.with_error(actual.error or "Actual result unavailable")
            self._apply_sql_metadata(outcome, actual_sql, expected_sql)
            record.outcome = outcome
            return record

        if not expected.available:
            outcome = ComparisonOutcome.with_error(expected.error or "Gold standard unavailable")
            self._apply_sql_metadata(outcome, actual_sql, expected_sql)
            record.outcome = outcome
            return record

        outcome = self.comparator.compare(actual.dataframe, expected.dataframe)
        self._apply_sql_metadata(outcome, actual_sql, expected_sql)
        record.outcome = outcome
        return record

    def _fetch_sql_data(self, provider: Optional[SqlProvider], task_id: str) -> Optional[SqlData]:
        if not provider:
            return None
        try:
            return provider.fetch(task_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to fetch SQL for task %s: %s", task_id, exc)
            return SqlData(task_id=task_id, source="", error=str(exc))

    def _apply_sql_metadata(
        self,
        outcome: ComparisonOutcome,
        actual_sql: Optional[SqlData],
        expected_sql: Optional[SqlData],
    ) -> None:
        actual_tables = []
        expected_tables = []

        if actual_sql:
            if actual_sql.tables:
                actual_tables = _unique_preserve_order(actual_sql.tables)
                outcome.actual_tables = actual_tables
            if actual_sql.error:
                outcome.actual_sql_error = actual_sql.error

        if expected_sql:
            if expected_sql.tables:
                expected_tables = _unique_preserve_order(expected_sql.tables)
                outcome.expected_tables = expected_tables
            if expected_sql.error:
                outcome.expected_sql_error = expected_sql.error

        if not outcome.actual_tables and actual_tables:
            outcome.actual_tables = actual_tables
        if not outcome.expected_tables and expected_tables:
            outcome.expected_tables = expected_tables

        if outcome.actual_tables or outcome.expected_tables:
            outcome.matched_tables = compute_table_matches(outcome.actual_tables, outcome.expected_tables)


def collect_latest_trajectory_files(save_dir: str) -> Dict[str, Path]:
    directory = Path(save_dir)
    if not directory.exists():
        return {}

    file_groups: dict[str, list[tuple[float, Path]]] = defaultdict(list)

    for filepath in directory.glob("*.yaml"):
        task_id, timestamp = parse_trajectory_filename(filepath.name)
        if task_id and timestamp is not None:
            file_groups[task_id].append((timestamp, filepath))

    latest_files: Dict[str, Path] = {}
    for task_id, files in file_groups.items():
        files.sort(key=lambda entry: entry[0], reverse=True)
        latest_files[task_id] = files[0][1]

    return latest_files


def _resolve_existing_directory(base: Path, candidates: Sequence[str]) -> Optional[Path]:
    for relative in candidates:
        candidate = base / relative
        if candidate.exists():
            return candidate
    return None


def detect_benchmark_type(benchmark_path: Path) -> str:
    if benchmark_path.is_file():
        return "sub_agent"
    if (benchmark_path / "dev.json").exists():
        return "bird_dev"
    if _resolve_existing_directory(benchmark_path, ("evaluation_suite/gold/sql", "gold/sql")):
        return "spider2"
    # Fallback to directory naming conventions
    name = benchmark_path.name.lower()
    if "bird" in name:
        return "bird_dev"
    if "spider" in name:
        return "spider2"
    return "unknown"


def _default_sql_dialect(benchmark_type: str) -> str:
    if benchmark_type == "bird_dev":
        return DBType.SQLITE
    return DBType.SNOWFLAKE


def create_result_provider(path: str, task_id_field_name: str = "task_id") -> ResultProvider:
    path_obj = Path(path)
    if path_obj.is_file():
        return SingleFileGoldProvider(str(path_obj), task_id_key=task_id_field_name)
    return CsvPerTaskResultProvider(str(path_obj))


def parse_trajectory_filename(filename: str) -> tuple[Optional[str], Optional[float]]:
    base_name = Path(filename).stem
    last_underscore_idx = base_name.rfind("_")
    if last_underscore_idx == -1:
        return None, None

    task_id = base_name[:last_underscore_idx]
    timestamp_str = base_name[last_underscore_idx + 1 :]
    try:
        return task_id, float(timestamp_str)
    except ValueError:
        return None, None


def compare_pandas_tables(pred_df: pd.DataFrame, gold_df: pd.DataFrame) -> Dict[str, Any]:
    if len(pred_df) != len(gold_df):
        return {
            "match_rate": 0.0,
            "matched_columns": [],
            "un_matched_columns": list(gold_df.columns),
            "unnecessary_columns": list(pred_df.columns),
        }

    matches: list[tuple[str, str]] = []
    matched_pred_cols: set[str] = set()
    unmatched_gold_cols: set[str] = set(gold_df.columns)

    for pred_col in pred_df.columns:
        if pred_col in matched_pred_cols:
            continue
        for gold_col in list(unmatched_gold_cols):
            try:
                if columns_match(pred_df[pred_col], gold_df[gold_col]):
                    matches.append((pred_col, gold_col))
                    matched_pred_cols.add(pred_col)
                    unmatched_gold_cols.discard(gold_col)
                    break
            except Exception:
                continue

    un_matches = list(unmatched_gold_cols)
    unnecessary_columns = [col for col in pred_df.columns if col not in matched_pred_cols]

    total_gold_cols = len(gold_df.columns)
    match_rate = (len(matches) / total_gold_cols) if total_gold_cols > 0 else 1.0

    return {
        "match_rate": round(match_rate, 4),
        "matched_columns": matches,
        "un_matched_columns": un_matches,
        "unnecessary_columns": unnecessary_columns,
    }


def columns_match(series_a: pd.Series, series_b: pd.Series, tol: float = 1e-6) -> bool:
    if len(series_a) != len(series_b):
        return False

    for a, b in zip(series_a, series_b):
        if pd.isna(a) and pd.isna(b):
            continue
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isclose(float(a), float(b), abs_tol=tol):
                return False
        elif a != b:
            return False
    return True


def preview_dataframe(df: pd.DataFrame, max_rows: int = 3, max_cols: int = 5) -> str:
    if df is None:
        return "No data"

    preview_df = df.head(max_rows)
    truncated_cols = False
    if len(df.columns) > max_cols:
        preview_df = preview_df.iloc[:, :max_cols]
        truncated_cols = True

    result_lines = []
    headers = list(preview_df.columns)
    if truncated_cols:
        headers.append("...")
    result_lines.append(" | ".join(str(h) for h in headers))
    result_lines.append("-" * len(result_lines[0]))

    for _, row in preview_df.iterrows():
        row_values = [str(v) for v in row.values]
        if truncated_cols:
            row_values.append("...")
        result_lines.append(" | ".join(row_values))

    if len(df) > max_rows:
        result_lines.append("...")

    return "\n       ".join(result_lines)


def evaluate_benchmark_accuracy(
    benchmark_platform: str,
    benchmark_path: str,
    trajectory_dir: str,
    result_dir: str,
    target_task_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    if not os.path.exists(result_dir):
        logger.warning(f"Result directory not found at {result_dir}")
        return {}

    benchmark_path_obj = Path(benchmark_path)

    result_provider = CsvPerTaskResultProvider(result_dir)
    dialect: str = DBType.SQLITE if benchmark_platform == "bird_dev" else DBType.SNOWFLAKE
    result_sql_provider = AgentResultSqlProvider(result_dir, dialect=dialect)

    if benchmark_platform == "spider2":
        gold_result_dir = benchmark_path_obj / "evaluation_suite" / "gold" / "exec_result"

        if not gold_result_dir.exists():
            logger.warning(f"Gold exec_result directory not found at {gold_result_dir}")
            return {}

        gold_sql_dir = benchmark_path_obj / "evaluation_suite" / "gold" / "sql"
        if not gold_sql_dir.exists():
            logger.warning(f"Gold sql directory not found at {gold_sql_dir}")
            return {}
        gold_sql_provider = DirectorySqlProvider(str(gold_sql_dir), dialect=dialect)
        gold_result_provider = CsvPerTaskResultProvider(str(gold_result_dir))
    else:
        gold_dir = benchmark_path_obj / "gold" / "exec_result"
        gold_result_provider = CsvPerTaskResultProvider(str(gold_dir))
        gold_sql_provider = JsonMappingSqlProvider(
            str(os.path.join(benchmark_path, "dev.json")), task_id_key="question_id", sql_key="SQL", dialect=dialect
        )

    evaluator = BenchmarkEvaluator(
        trajectory_parser=TrajectoryParser(),
        result_provider=result_provider,
        gold_result_provider=gold_result_provider,
        result_sql_provider=result_sql_provider,
        gold_sql_provider=gold_sql_provider,
    )

    report = evaluator.evaluate_directory(trajectory_dir, target_task_ids)
    return report.to_dict()


def evaluate_and_report_accuracy(
    benchmark_platform: str,
    benchmark_path: str,
    trajectory_dir: str,
    result_dir: str,
    target_task_ids: Optional[Iterable[str]] = None,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    accuracy_report = evaluate_benchmark_accuracy(
        benchmark_platform=benchmark_platform,
        benchmark_path=benchmark_path,
        trajectory_dir=trajectory_dir,
        result_dir=result_dir,
        target_task_ids=target_task_ids,
    )

    if accuracy_report.get("status") == "success":
        _log_accuracy_summary(accuracy_report)
        if output_file:
            with open(output_file, "w", encoding="utf-8") as handle:
                json.dump(accuracy_report, handle, ensure_ascii=False, indent=2)
            logger.info(f"Detailed accuracy report saved to: {output_file}")
    else:
        logger.error(f"Accuracy evaluation failed: {accuracy_report.get('message')}")

    return accuracy_report


def evaluate_sub_agent_and_report(
    benchmark_path: str,
    trajectory_dir: str,
    result_path: str,
    target_task_ids: Optional[Iterable[str]] = None,
    output_file: Optional[str] = None,
    db_type: DBType = DBType.SQLITE,
) -> Dict[str, Any]:
    evaluator = BenchmarkEvaluator(
        trajectory_parser=TrajectoryParser(),
        result_provider=CsvPerTaskResultProvider(str(result_path)),
        gold_result_provider=SingleFileGoldProvider(
            benchmark_path, task_id_key="SQL", query_result_key="Expected answer"
        ),
        result_sql_provider=AgentResultSqlProvider(str(result_path), dialect=db_type),
        gold_sql_provider=CsvColumnSqlProvider(benchmark_path, task_id_key="SQL", sql_key="Gold sql", dialect=db_type),
    )

    report = evaluator.evaluate_directory(trajectory_dir, target_task_ids)
    report_dict = report.to_dict()

    if report_dict.get("status") == "success":
        _log_accuracy_summary(report_dict)
        if output_file:
            with open(output_file, "w", encoding="utf-8") as handle:
                json.dump(report_dict, handle, ensure_ascii=False, indent=2)
            logger.info(f"Detailed accuracy report saved to: {output_file}")
    else:
        logger.error(f"Sub-agent accuracy evaluation failed: {report_dict.get('message')}")

    return report_dict


def _log_accuracy_summary(accuracy_report: Dict[str, Any]) -> None:
    summary = accuracy_report["summary"]
    task_ids = accuracy_report.get("task_ids", {})
    comp_summary = summary.get("comparison_summary", {})

    failed_ids = [task_id.strip() for task_id in task_ids.get("failed_task_ids", "").split(",") if task_id.strip()]
    matched_ids = [task_id.strip() for task_id in task_ids.get("matched_task_ids", "").split(",") if task_id.strip()]
    mismatched_ids = [
        task_id.strip() for task_id in task_ids.get("mismatched_task_ids", "").split(",") if task_id.strip()
    ]
    empty_result_ids = [
        task_id.strip() for task_id in task_ids.get("empty_result_task_ids", "").split(",") if task_id.strip()
    ]

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("BENCHMARK ACCURACY EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"Total tasks analyzed: {summary['total_files']}")

    if summary.get("success_rate") is not None:
        report_lines.append(f"Execution success rate: {summary['success_rate']:.1f}%")

    if comp_summary:
        match_rate = comp_summary.get("match_rate", 0)
        report_lines.append(f"Result comparison match rate: {match_rate:.1f}%")

    report_lines.append("")
    report_lines.append("DETAILED STATISTICS")
    report_lines.append("-" * 40)

    if comp_summary:
        total_comparisons = comp_summary.get("total_comparisons", 0)
        successful_matches = comp_summary.get("successful_matches", 0)
        mismatches = comp_summary.get("mismatches", 0)
        comparison_errors = comp_summary.get("comparison_errors", 0)
        empty_result_errors = comp_summary.get("empty_result_errors", 0)

        report_lines.append(f"Total comparisons performed: {total_comparisons}")
        report_lines.append(f"Successful matches: {successful_matches}")
        report_lines.append(f"Mismatches: {mismatches}")
        report_lines.append(f"Comparison errors: {comparison_errors}")
        report_lines.append(f"Empty result errors: {empty_result_errors}")

        if total_comparisons > 0:
            mismatch_rate = (mismatches / total_comparisons) * 100
            error_rate = ((comparison_errors + empty_result_errors) / total_comparisons) * 100
            report_lines.append(f"Mismatch rate: {mismatch_rate:.1f}%")
            report_lines.append(f"Error rate: {error_rate:.1f}%")

    report_lines.append("")
    report_lines.append("TASK BREAKDOWN BY CATEGORY")
    report_lines.append("-" * 40)

    def _format_task_list(task_list: list[str], max_display: int = 10) -> str:
        if not task_list:
            return "None"
        try:
            sorted_list = sorted(task_list, key=int)
        except ValueError:
            sorted_list = sorted(task_list, key=str)
        if len(sorted_list) <= max_display:
            return ", ".join(sorted_list)
        displayed = sorted_list[:max_display]
        return f"{', '.join(displayed)} ... (+{len(sorted_list) - max_display} more)"

    report_lines.append(f"Matched tasks ({len(matched_ids)}):")
    report_lines.append(f"   {_format_task_list(matched_ids)}")
    report_lines.append("")

    report_lines.append(f"Mismatched tasks ({len(mismatched_ids)}):")
    report_lines.append(f"   {_format_task_list(mismatched_ids)}")
    report_lines.append("")

    report_lines.append(f"Failed tasks ({len(failed_ids)}):")
    report_lines.append(f"   {_format_task_list(failed_ids)}")
    report_lines.append("")

    if empty_result_ids:
        report_lines.append(f"Empty result tasks ({len(empty_result_ids)}):")
        report_lines.append(f"   {_format_task_list(empty_result_ids)}")
        report_lines.append("")

    report_lines.append("ADDITIONAL STATISTICS")
    report_lines.append("-" * 40)

    total_tasks = summary["total_files"]
    success_count = len(matched_ids)
    if total_tasks > 0:
        overall_success_rate = (success_count / total_tasks) * 100
        report_lines.append(f"Overall success rate: {overall_success_rate:.1f}%")

    report_lines.append(f"Successful tasks: {success_count}")
    report_lines.append(f"Failed tasks: {len(failed_ids)}")
    report_lines.append(f"Mismatched tasks: {len(mismatched_ids)}")
    if empty_result_ids:
        report_lines.append(f"Empty result tasks: {len(empty_result_ids)}")

    report_lines.append("")
    report_lines.append("=" * 80)

    logger.info("\n%s", "\n".join(report_lines))


def load_bird_dev_tasks(benchmark_path: str) -> List[Dict[str, Any]]:
    file_path = os.path.join(benchmark_path, "dev.json")
    if not os.path.exists(file_path):
        raise DatusException(
            code=ErrorCode.COMMON_FILE_NOT_FOUND,
            message_args={"file_name": file_path, "config_name": "Bird-dev benchmark"},
        )
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in file '{file_path}': {str(e)}")
        raise DatusException(
            ErrorCode.COMMON_JSON_PARSE_ERROR, message_args={"file_path": file_path, "error_detail": str(e)}
        )
