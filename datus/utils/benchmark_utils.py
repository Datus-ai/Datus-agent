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
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple

import pandas as pd
import yaml

from datus.configuration.agent_config import AgentConfig, BenchmarkConfig
from datus.tools.db_tools import BaseSqlConnector
from datus.tools.db_tools.db_manager import db_manager_instance, get_connection
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import extract_table_names

logger = get_logger(__name__)

FAIL_STATUSES = {"pending", "failed", "error"}


@dataclass
class WorkflowArtifacts:
    files: list[str] = field(default_factory=list)
    reference_sqls: list[str] = field(default_factory=list)
    reference_sql_names: list[str] = field(default_factory=list)
    semantic_models: list[str] = field(default_factory=list)
    metrics_names: list[str] = field(default_factory=list)
    metrics_texts: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "files": list(self.files),
            "reference_sqls": list(self.reference_sqls),
            "reference_sql_names": list(self.reference_sql_names),
            "semantic_models": list(self.semantic_models),
            "metrics_names": list(self.metrics_names),
            "metrics_texts": list(self.metrics_texts),
        }


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
    tool_calls: Dict[str, int] = field(default_factory=dict)
    outputs: list[WorkflowOutput] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    artifacts: WorkflowArtifacts = field(default_factory=WorkflowArtifacts)

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
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    actual_shape: Optional[tuple[int, int]] = None
    expected_shape: Optional[tuple[int, int]] = None
    actual_preview: Optional[str] = None
    expected_preview: Optional[str] = None
    actual_tables: list[str] = field(default_factory=list)
    expected_tables: list[str] = field(default_factory=list)
    matched_tables: list[str] = field(default_factory=list)
    actual_sql: Optional[str] = None
    gold_sql: Optional[str] = None
    actual_sql_error: Optional[str] = None
    match_sql_error: Optional[str] = None
    tools_comparison: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @classmethod
    def with_error(cls, message: str) -> "ComparisonOutcome":
        return cls(error=message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_rate": self.match_rate,
            "matched_columns": self.matched_columns,
            "missing_columns": self.missing_columns,
            "extra_columns": self.extra_columns,
            "actual_shape": self.actual_shape,
            "expected_shape": self.expected_shape,
            "actual_preview": self.actual_preview,
            "expected_preview": self.expected_preview,
            "actual_tables": self.actual_tables,
            "expected_tables": self.expected_tables,
            "matched_tables": self.matched_tables,
            "actual_sql_error": self.actual_sql_error,
            "sql_error": self.match_sql_error,
            "actual_sql": self.actual_sql,
            "gold_sql": self.gold_sql,
            "tools_comparison": self.tools_comparison,
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


@dataclass
class GoldArtifacts:
    file_reference: str = ""
    expected_sql: str = ""
    semantic_model: str = ""
    expected_metrics: str = ""


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


_TABLE_IDENTIFIER_STRIP_CHARS = "\"'`[]"


def _clean_table_identifier_part(part: str) -> str:
    cleaned = str(part).strip()
    return cleaned.strip(_TABLE_IDENTIFIER_STRIP_CHARS)


def _parse_table_identifier(table: str) -> Tuple[str, str, bool]:
    """
    Normalize SQL table identifiers and extract the terminal table name.

    Returns a tuple of (normalized_identifier, base_table_name, is_simple_name).
    """
    if table is None:
        return "", "", False

    identifier = str(table).strip().lower()
    if not identifier:
        return "", "", False

    identifier = identifier.lstrip(".")
    if not identifier:
        return "", "", False

    parts = []
    for raw_part in identifier.split("."):
        cleaned = _clean_table_identifier_part(raw_part)
        if cleaned:
            parts.append(cleaned.lower())

    if not parts:
        return "", "", False

    normalized_identifier = ".".join(parts)
    base_name = parts[-1]
    is_simple = len(parts) == 1

    normalized_identifier = _normalize_field_name(normalized_identifier) or normalized_identifier
    base_name = _normalize_field_name(base_name) or base_name

    return normalized_identifier, base_name, is_simple


def _is_empty_table_identifier(table: Any) -> bool:
    if table is None:
        return True
    text = str(table).strip()
    return text == ""


def _trim_trailing_non_empty_tables(tables: Sequence[Any]) -> list[Any]:
    trimmed: list[Any] = []
    for table in reversed(tables):
        if _is_empty_table_identifier(table):
            break
        trimmed.append(table)
    trimmed.reverse()
    return trimmed


def _tables_equivalent(left: Any, right: Any) -> bool:
    left_normalized, left_base, _ = _parse_table_identifier(left)
    right_normalized, right_base, _ = _parse_table_identifier(right)
    if not left_normalized or not right_normalized:
        return False
    if left_normalized == right_normalized:
        return True
    return bool(left_base and right_base and left_base == right_base)


def collect_sql_tables(sql_text: Optional[str], dialect: Optional[str] = None) -> list[str]:
    if not sql_text:
        return []

    dialect = dialect or DBType.SNOWFLAKE
    tables: list[str] = []
    try:
        extracted_tables = extract_table_names(sql_text, dialect=dialect, ignore_empty=True)
        tables.extend(extracted_tables)
    except Exception:
        # Ignore extraction errors; tables list may remain empty
        pass

    return _unique_preserve_order(tables)


def compute_table_matches(actual_tables: Iterable[str], expected_tables: Iterable[str]) -> list[str]:
    actual_list = list(actual_tables) if actual_tables is not None else []
    expected_list = list(expected_tables) if expected_tables is not None else []
    if not actual_list or not expected_list:
        return []

    trailing_actual = _trim_trailing_non_empty_tables(actual_list)
    trailing_expected = _trim_trailing_non_empty_tables(expected_list)

    if not trailing_actual or not trailing_expected:
        return []

    compare_len = min(len(trailing_actual), len(trailing_expected))
    if compare_len <= 0:
        return []
    backward_matches: list[str] = []

    for expected_table in expected_list:
        for actual_table in actual_list:
            if _tables_equivalent(actual_table, expected_table):
                backward_matches.append(expected_table)
                break

    return _unique_preserve_order(backward_matches)


def _normalize_text(value: str) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def _find_matching_candidates(expected: str, candidates: List[str]) -> List[str]:
    normalized_expected = _normalize_text(expected)
    if not normalized_expected:
        return []
    matches: list[str] = []
    for candidate in candidates:
        normalized_candidate = _normalize_text(candidate)
        if not normalized_candidate:
            continue
        if (
            normalized_expected == normalized_candidate
            or normalized_expected in normalized_candidate
            or normalized_candidate in normalized_expected
        ):
            matches.append(candidate)
    return _unique_preserve_order(matches)


def _split_expected_items(value: str) -> list[str]:
    if not value:
        return []
    parts = re.split(r"[;\n,]+", value)
    return [part.strip() for part in parts if part and part.strip()]


def _append_unique(container: list[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _append_unique(container, item)
        return
    if isinstance(value, Mapping):
        return
    text = str(value).strip()
    if not text:
        return
    if text not in container:
        container.append(text)


def _extract_result_payload(output: Any) -> Any:
    if isinstance(output, Mapping):
        raw_output = output.get("raw_output")
        if isinstance(raw_output, Mapping) and "result" in raw_output:
            return raw_output.get("result")
        if raw_output is not None:
            return raw_output
        if "result" in output:
            return output.get("result")
    return output


def _append_file_candidate(artifacts: WorkflowArtifacts, candidate: str) -> None:
    if not candidate:
        return
    _append_unique(artifacts.files, candidate)
    match = re.search(r"file (?:written|created|saved) successfully:\s*(.+)", candidate, re.IGNORECASE)
    if match:
        _append_unique(artifacts.files, match.group(1))


def _collect_file_artifacts(artifacts: WorkflowArtifacts, result_payload: Any) -> None:
    if isinstance(result_payload, str):
        _append_file_candidate(artifacts, result_payload)
    elif isinstance(result_payload, Mapping):
        for key in result_payload.keys():
            _append_file_candidate(artifacts, key)
    elif isinstance(result_payload, (list, tuple, set)):
        for item in result_payload:
            if isinstance(item, str):
                _append_file_candidate(artifacts, item)
    else:
        logger.warning(f"Unable to parse file call: {result_payload}")


def _collect_reference_sql_artifacts(artifacts: WorkflowArtifacts, result_payload: Any) -> None:
    if isinstance(result_payload, str):
        _append_unique(artifacts.reference_sqls, result_payload)
        return

    items: List[Mapping[str, Any]] = []
    if isinstance(result_payload, Mapping):
        items = [result_payload]
    elif isinstance(result_payload, (list, tuple)):
        items = [item for item in result_payload if isinstance(item, Mapping)]

    for item in items:
        sql_text = item.get("sql")
        if sql_text:
            _append_unique(artifacts.reference_sqls, sql_text)
        name = item.get("name")
        if name:
            _append_unique(artifacts.reference_sql_names, name)


def _collect_semantic_model_artifacts(artifacts: WorkflowArtifacts, result_payload: Any) -> None:
    metadata_entries = None
    if isinstance(result_payload, Mapping):
        metadata_entries = result_payload.get("metadata")
    if isinstance(metadata_entries, list):
        for entry in metadata_entries:
            if not isinstance(entry, Mapping):
                continue
            model_name = entry.get("semantic_model_name") or entry.get("description")
            if model_name:
                _append_unique(artifacts.semantic_models, model_name)


def _collect_metric_artifacts(artifacts: WorkflowArtifacts, result_payload: Any) -> None:
    candidates: List[Mapping[str, Any]] = []
    if isinstance(result_payload, Mapping):
        candidates = [result_payload]
    elif isinstance(result_payload, (list, tuple)):
        candidates = [item for item in result_payload if isinstance(item, Mapping)]

    for item in candidates:
        name = item.get("name")
        if name:
            _append_unique(artifacts.metrics_names, name)
        text = item.get("description")
        if text:
            _append_unique(artifacts.metrics_texts, text)


def _extract_artifacts_from_action_history(
    action_history: Any,
    artifacts: WorkflowArtifacts,
    tool_calls: Optional[MutableMapping[str, int]] = None,
) -> None:
    if not action_history:
        return
    for entry in action_history:
        if not isinstance(entry, Mapping):
            continue
        role = str(entry.get("role", "")).lower()
        if role != "tool":
            continue
        input_payload = entry.get("input") or {}
        function_name = str(input_payload.get("function_name", "") or "").lower()
        if "." in function_name:
            function_name = function_name.split(".")[-1]
        if tool_calls is not None and function_name:
            tool_calls[function_name] = tool_calls.get(function_name, 0) + 1

        output_payload = entry.get("output") or {}
        result_payload = _extract_result_payload(output_payload)
        if function_name in {"write_file", "read_file", "read_multiple_files", "search_files"}:
            _collect_file_artifacts(artifacts, result_payload)
        elif function_name in {"search_reference_sql", "get_reference_sql"}:
            _collect_reference_sql_artifacts(artifacts, result_payload)
        elif function_name == "search_table":
            _collect_semantic_model_artifacts(artifacts, result_payload)
        elif function_name in {"search_metrics", "get_metrics"}:
            _collect_metric_artifacts(artifacts, result_payload)


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
            "total_node_count": self.analysis.total_nodes,
            "output_node_count": self.analysis.output_nodes,
            "output_success_count": self.analysis.output_success,
            "output_failure_count": self.analysis.output_failure,
            "errors": list(self.analysis.errors),
            "node_types": dict(self.analysis.node_types),
            "tool_calls": dict(self.analysis.tool_calls),
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
    def __init__(self, directory: str, namespace: Optional[str] = None, run_id: Optional[str] = None):
        """
        Initialize CSV result provider with hierarchical directory structure.

        Args:
            directory: Base directory path
            namespace: Optional namespace subdirectory. If None, uses directory directly.
            run_id: Optional run_id subdirectory. If None and namespace is provided, uses latest run.
        """
        self.base_directory = Path(directory)
        self.namespace = namespace
        self.run_id = run_id

        # Determine actual search directory
        if namespace:
            namespace_dir = self.base_directory / namespace
            if run_id:
                # Specific namespace and run
                self.directory = namespace_dir / run_id
            else:
                # Auto-select latest run
                if namespace_dir.exists() and namespace_dir.is_dir():
                    run_dirs = sorted(
                        [d for d in namespace_dir.iterdir() if d.is_dir()], key=lambda d: d.name, reverse=True
                    )
                    if run_dirs:
                        self.directory = run_dirs[0]
                        logger.info(f"Auto-selected latest run: {run_dirs[0].name}")
                    else:
                        # No run subdirectories found
                        self.directory = namespace_dir
                else:
                    # Namespace directory doesn't exist yet
                    self.directory = namespace_dir
        else:
            # No namespace - use directory directly (for gold results)
            self.directory = self.base_directory

    def fetch(self, task_id: str) -> ResultData:
        csv_path = self.directory / f"{task_id}.csv"
        source = str(csv_path)
        if not csv_path.exists():
            return ResultData(task_id=task_id, source=source, error=f"Result file not found: {csv_path}")
        try:
            dataframe = pd.read_csv(csv_path)
        except Exception as exc:  # pragma: no cover - logging side effect
            logger.warning(f"Failed to read {csv_path}: {exc}")
            return ResultData(task_id=task_id, source=source)
        return ResultData(task_id=task_id, source=source, dataframe=dataframe)


class SingleFileGoldProvider(ResultProvider):
    def __init__(
        self,
        result_file: str,
        connections: Dict[str, BaseSqlConnector],
        task_id_key: str = "task_id",
        sql_key: str = "",
        query_result_key: str = "",
        db_key: str = "",
        allowed_task_ids: Optional[Iterable[str]] = None,
        frame_cache_size: int = 32,
    ):
        self.task_id_key = task_id_key
        self.query_result_key = query_result_key
        self.sql_key = sql_key
        self.db_key = db_key
        self.result_file = Path(result_file)
        self.connections = connections
        self.allowed_task_ids = {str(task_id) for task_id in allowed_task_ids} if allowed_task_ids else None
        self.frame_cache_size = max(frame_cache_size, 1)

        self._raw_expected_results: Dict[str, str] = {}
        self._sql_tasks: Dict[str, Tuple[str, str]] = {}
        self._frame_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._errors: Dict[str, str] = {}
        self._artifacts: Dict[str, GoldArtifacts] = {}
        self._loaded = False
        self._global_error: Optional[str] = None
        self._suffix: Optional[str] = None

    def fetch(self, task_id: str) -> ResultData:
        self._ensure_loaded()
        source = str(self.result_file)
        if self._global_error:
            return ResultData(task_id=task_id, source=source, error=self._global_error)
        if task_id in self._errors:
            return ResultData(task_id=task_id, source=source, error=self._errors[task_id])

        dataframe = self._frame_cache.get(task_id)
        if dataframe is not None:
            self._frame_cache.move_to_end(task_id)
            return ResultData(task_id=task_id, source=source, dataframe=dataframe)

        raw_expected = self._raw_expected_results.get(task_id)
        if raw_expected is not None:
            dataframe = self._convert_to_dataframe(task_id, raw_expected)
            if dataframe is None:
                return ResultData(task_id=task_id, source=source, error=self._errors.get(task_id))
            self._remember_dataframe(task_id, dataframe)
            return ResultData(task_id=task_id, source=source, dataframe=dataframe)

        sql_task = self._sql_tasks.get(task_id)
        if sql_task is not None:
            dataframe = self._execute_gold_sql(task_id, *sql_task)
            if dataframe is None:
                return ResultData(task_id=task_id, source=source, error=self._errors.get(task_id))
            self._remember_dataframe(task_id, dataframe)
            return ResultData(task_id=task_id, source=source, dataframe=dataframe)

        if self._lazy_load_task(task_id):
            return self.fetch(task_id)

        return ResultData(task_id=task_id, source=source, error=f"Result not found for task_id={task_id}")

    def get_artifacts(self, task_id: str) -> GoldArtifacts:
        self._ensure_loaded()
        return self._artifacts.get(task_id, GoldArtifacts())

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.result_file.exists():
            self._global_error = f"Result file not found: {self.result_file}"
            return

        self._suffix = self.result_file.suffix.lower()
        if self._suffix == ".csv":
            self._load_csv_sources()
        elif self._suffix in (".json", ".jsonl"):
            self._load_json_sources()
        else:
            self._global_error = f"Unsupported result file format: {self.result_file.suffix}"

    def _load_csv_sources(self) -> None:
        try:
            with self.result_file.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames or []
                if self.task_id_key not in fieldnames:
                    self._global_error = (
                        f"Gold file must contain a '{self.task_id_key}' column to disambiguate records."
                    )
                    return
                if self.query_result_key and self.query_result_key not in fieldnames:
                    self._global_error = (
                        f"Gold result file must contain '{self.query_result_key}' when query_result_key is provided."
                    )
                    return
                if not self.query_result_key and self.sql_key and self.sql_key not in fieldnames:
                    self._global_error = (
                        f"Gold result file must contain '{self.sql_key}' when query_result_key is not provided."
                    )
                    return

                for row in reader:
                    task_id = row.get(self.task_id_key, "")
                    if task_id is None:
                        task_id = ""
                    task_id = str(task_id).strip()
                    if not task_id:
                        continue
                    if self.allowed_task_ids and task_id not in self.allowed_task_ids:
                        continue
                    self._record_row(task_id, row)
        except Exception as exc:  # pragma: no cover - logging side effect
            self._global_error = f"Failed to load result file: {exc}"

    def _load_json_sources(self) -> None:
        try:
            if self._suffix == ".jsonl":
                with self.result_file.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        if isinstance(record, Mapping):
                            self._record_row_from_mapping(record)
                return
            with self.result_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._global_error = f"Failed to load result file: {exc}"
            return

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, Mapping):
                    self._record_row_from_mapping(item)
        elif isinstance(payload, Mapping):
            if self.task_id_key in payload:
                self._record_row_from_mapping(payload)
            else:
                for value in payload.values():
                    if isinstance(value, Mapping):
                        self._record_row_from_mapping(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, Mapping):
                                self._record_row_from_mapping(item)

    def _record_row_from_mapping(self, record: Mapping[str, Any]) -> None:
        task_id = record.get(self.task_id_key, "")
        if task_id is None:
            task_id = ""
        task_id = str(task_id).strip()
        if not task_id:
            return
        if self.allowed_task_ids and task_id not in self.allowed_task_ids:
            return
        self._record_row(task_id, record)

    def _record_row(self, task_id: str, row: Mapping[str, Any]) -> None:
        expected_value = None
        if self.query_result_key:
            expected_value = row.get(self.query_result_key)

        if expected_value:
            raw_csv = str(expected_value)
            self._raw_expected_results[task_id] = raw_csv
        else:
            sql_text = str(row.get(self.sql_key, "") or "").strip()
            if not sql_text:
                self._errors[task_id] = f"Missing `{self.sql_key or self.query_result_key}` value"
                return
            db_name = str(row.get(self.db_key, "") or "").strip()
            self._sql_tasks[task_id] = (sql_text, db_name)

        self._store_artifacts(task_id, row)

    def _lazy_load_task(self, task_id: str) -> bool:
        if self._suffix == ".csv":
            return self._load_single_csv_row(task_id)
        if self._suffix == ".json":
            return self._load_single_json_entry(task_id)
        if self._suffix == ".jsonl":
            return self._load_single_jsonl_entry(task_id)
        return False

    def _load_single_csv_row(self, task_id: str) -> bool:
        try:
            with self.result_file.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    current_id = row.get(self.task_id_key, "")
                    if current_id is None:
                        current_id = ""
                    current_id = str(current_id).strip()
                    if current_id != task_id:
                        continue
                    self._record_row(task_id, row)
                    return True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(f"Failed to lazily load gold result for {task_id}: {exc}")
        return False

    def _load_single_json_entry(self, task_id: str) -> bool:
        try:
            with self.result_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(f"Failed to lazily load gold result for {task_id}: {exc}")
            return False

        stack = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, Mapping):
                current_id = item.get(self.task_id_key, "")
                if current_id is None:
                    current_id = ""

                if str(current_id).strip() == task_id:
                    self._record_row(task_id, item)
                    return True
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
        return False

    def _load_single_jsonl_entry(self, task_id: str) -> bool:
        try:
            with self.result_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(record, Mapping):
                        continue
                    current_id = record.get(self.task_id_key, "")
                    if current_id is None:
                        current_id = ""
                    if str(current_id).strip() != task_id:
                        continue
                    self._record_row(task_id, record)
                    return True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(f"Failed to lazily load gold result for {task_id}: {exc}")
        return False

    def _execute_gold_sql(self, task_id: str, sql_text: str, db_name: str) -> Optional[pd.DataFrame]:
        connector = get_connection(self.connections, db_name)
        if connector is None:
            error_msg = f"Connector not found for database '{db_name}'"
            self._errors[task_id] = error_msg
            return None
        try:
            exec_result = connector.execute_csv(sql_text)
        except Exception as exc:  # pragma: no cover - defensive logging
            error_msg = f"Execute gold sql failed: {exc}"
            logger.warning(error_msg)
            self._errors[task_id] = error_msg
            return None

        if not getattr(exec_result, "success", False):
            error_msg = f"Execute gold sql failed: {getattr(exec_result, 'error', 'unknown error')}"
            self._errors[task_id] = error_msg
            return None

        expected_value = getattr(exec_result, "sql_return", "") or ""
        if not expected_value:
            error_msg = "Gold SQL execution returned empty result"
            self._errors[task_id] = error_msg
            return None

        dataframe = self._convert_to_dataframe(task_id, expected_value)
        if dataframe is not None:
            # cache raw csv to avoid re-executing on subsequent fetches
            self._raw_expected_results[task_id] = expected_value
        return dataframe

    def _convert_to_dataframe(self, task_id: str, csv_str_value: str) -> Optional[pd.DataFrame]:
        try:
            return csv_str_to_pands(csv_str_value)
        except Exception as exc:  # pragma: no cover - defensive logging
            error_msg = f"Failed to parse expected answer for task {task_id}: {exc}"
            logger.warning(error_msg)
            self._errors[task_id] = error_msg
            return None

    def _remember_dataframe(self, task_id: str, dataframe: pd.DataFrame) -> None:
        self._frame_cache[task_id] = dataframe
        self._frame_cache.move_to_end(task_id)
        while len(self._frame_cache) > self.frame_cache_size:
            self._frame_cache.popitem(last=False)

    def _store_artifacts(self, task_id: str, row: Mapping[str, Any]) -> None:
        if not isinstance(row, Mapping):
            return
        try:
            normalized = {_normalize_field_name(str(k)): row[k] for k in row}
        except Exception:
            normalized = {}
        file_value = str(normalized.get(_normalize_field_name("file"), "") or "").strip()
        expected_sql = str(normalized.get(_normalize_field_name("expected_sql"), "") or "").strip()
        semantic_model = str(normalized.get(_normalize_field_name("semantic_model"), "") or "").strip()
        expected_metrics = str(normalized.get(_normalize_field_name("expected_metrics"), "") or "").strip()
        self._artifacts[task_id] = GoldArtifacts(
            file_reference=file_value,
            expected_sql=expected_sql,
            semantic_model=semantic_model,
            expected_metrics=expected_metrics,
        )


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

    def __init__(
        self,
        result_dir: str,
        namespace: str,
        run_id: Optional[str] = None,
        dialect: str = DBType.SNOWFLAKE,
    ):
        """
        Initialize agent result SQL provider with hierarchical directory structure.

        Args:
            result_dir: Base result directory path
            namespace: Namespace subdirectory (required)
            run_id: Optional run_id subdirectory. If None, uses latest run.
            dialect: SQL dialect
        """
        self.base_result_dir = Path(result_dir)
        self.dialect = dialect
        self.namespace = namespace
        self.run_id = run_id

        # Determine actual search directory
        namespace_dir = self.base_result_dir / namespace
        if run_id:
            # Specific namespace and run
            self.result_dir = namespace_dir / run_id
        else:
            # Auto-select latest run
            if namespace_dir.exists() and namespace_dir.is_dir():
                run_dirs = sorted(
                    [d for d in namespace_dir.iterdir() if d.is_dir()], key=lambda d: d.name, reverse=True
                )
                if run_dirs:
                    self.result_dir = run_dirs[0]
                    logger.info(f"Auto-selected latest run for SQL: {run_dirs[0].name}")
                else:
                    # No run subdirectories found
                    self.result_dir = namespace_dir
            else:
                # Namespace directory doesn't exist yet
                self.result_dir = namespace_dir

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
        suffix = self.json_path.suffix.lower()
        try:
            if suffix == ".jsonl":
                with self.json_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        self._ingest(record)
            else:
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
            for task_id, value in payload.items():
                if self.task_id_key not in value:
                    value[self.task_id_key] = task_id
                self._consume_record(value)
        elif isinstance(payload, list):
            for item in payload:
                self._consume_record(item)

    def _consume_record(self, record: Mapping[str, Any]) -> None:
        if not isinstance(record, Mapping):
            return
        task_id_value = record.get(self.task_id_key)
        sql_value = record.get(self.sql_key)
        if not sql_value or task_id_value is None:
            logger.warning(f"This item must contain {self.task_id_key} and {self.sql_key}, item={record}")
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
        task_id_key: str = "",
        sql_key="",
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
                    task_id = row.get(self.task_id_key, "")
                    if task_id is None:
                        task_id = ""
                    task_id = str(task_id).strip()
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
        artifacts = WorkflowArtifacts()
        tool_calls: defaultdict[str, int] = defaultdict(int)

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
                tool_calls=dict(tool_calls),
                outputs=outputs,
                errors=errors,
                artifacts=artifacts,
            )
        except Exception as exc:
            errors.append(f"Failed to read trajectory file: {exc}")
            return WorkflowAnalysis(
                task_id=task_id,
                completion_time=None,
                status="error",
                total_nodes=0,
                node_types={},
                tool_calls=dict(tool_calls),
                outputs=outputs,
                errors=errors,
                artifacts=artifacts,
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
                tool_calls=dict(tool_calls),
                outputs=outputs,
                errors=errors,
                artifacts=artifacts,
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
                result = node.get("result") or {}
                action_history = result.get("action_history")
                _extract_artifacts_from_action_history(action_history, artifacts, tool_calls)

                if node_type == "output":
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
                    action_history = result.get("action_history")
                    _extract_artifacts_from_action_history(action_history, artifacts, tool_calls)
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
            tool_calls=dict(tool_calls),
            outputs=outputs,
            errors=errors,
            artifacts=artifacts,
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
                missing_columns=compares_result["missing_columns"],
                extra_columns=compares_result["extra_columns"],
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
        match_count = 0
        mismatches = 0
        comparison_error_count = 0
        empty_result_count = 0

        failed_task_ids: set[str] = set()
        matched_task_ids: set[str] = set()
        mismatched_task_ids: set[str] = set()
        empty_result_task_ids: set[str] = set()
        comparison_error_task_ids: set[str] = set()

        # Metrics comparison tracking
        total_with_expected_metrics = 0
        metrics_matched_count = 0
        metrics_mismatched_count = 0
        metrics_matched_task_ids: set[str] = set()
        metrics_mismatched_task_ids: set[str] = set()

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
                        empty_result_count += 1
                        empty_result_task_ids.add(task_id)
                    else:
                        comparison_error_count += 1
                        comparison_error_task_ids.add(task_id)
                    continue

                if outcome.match_rate == 1:
                    match_count += 1
                    matched_task_ids.add(task_id)
                else:
                    mismatches += 1
                    mismatched_task_ids.add(task_id)

                # Check metrics comparison
                if outcome.tools_comparison:
                    metrics_info = outcome.tools_comparison.get("expected_metrics", {})
                    expected = metrics_info.get("expected", [])
                    if expected:
                        total_with_expected_metrics += 1
                        match = metrics_info.get("match", False)
                        if match:
                            metrics_matched_count += 1
                            metrics_matched_task_ids.add(task_id)
                        else:
                            metrics_mismatched_count += 1
                            metrics_mismatched_task_ids.add(task_id)

        success_rate = (total_output_success / total_output_nodes * 100) if total_output_nodes else 0.0
        match_rate = (match_count / total_comparisons * 100) if total_comparisons else 0.0
        metrics_match_rate = (
            (metrics_matched_count / total_with_expected_metrics * 100) if total_with_expected_metrics else 0.0
        )

        summary = {
            "total_files": total_files,
            "total_output_nodes": total_output_nodes,
            "total_output_success": total_output_success,
            "total_output_failure": total_output_failure,
            "success_rate": round(success_rate, 2),
            "comparison_summary": {
                "total_comparisons": total_comparisons,
                "match_count": match_count,
                "mismatch_count": mismatches,
                "comparison_error_count": comparison_error_count,
                "comparison_error_task_ids": ",".join(map(str, sorted(comparison_error_task_ids))),
                "empty_result_count": empty_result_count,
                "match_rate": round(match_rate, 2),
            },
            "metrics_summary": {
                "total_with_expected_metrics": total_with_expected_metrics,
                "metrics_matched_count": metrics_matched_count,
                "metrics_mismatched_count": metrics_mismatched_count,
                "metrics_match_rate": round(metrics_match_rate, 2),
                "metrics_matched_task_ids": ",".join(map(str, sorted(metrics_matched_task_ids))),
                "metrics_mismatched_task_ids": ",".join(map(str, sorted(metrics_mismatched_task_ids))),
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
        self,
        trajectory_dir: str,
        target_task_ids: Iterable[str],
        namespace: str,
        run_id: Optional[str] = None,
    ) -> EvaluationReport:
        trajectories = collect_latest_trajectory_files(trajectory_dir, namespace, run_id)
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
                    comparison_records.append(self._build_comparison_record(task_id, analysis))

            evaluations[task_id] = TaskEvaluation(
                task_id=task_id,
                analysis=analysis,
                comparisons=comparison_records,
            )

        return self.report_builder.build(evaluations)

    def _build_comparison_record(self, task_id: str, analysis: WorkflowAnalysis) -> ComparisonRecord:
        actual = self.result_provider.fetch(task_id)
        expected = self.gold_result_provider.fetch(task_id)
        record = ComparisonRecord(task_id=task_id, actual=actual, expected=expected)

        actual_sql = self._fetch_sql_data(self.result_sql_provider, task_id)
        gold_sql = self._fetch_sql_data(self.gold_sql_provider, task_id)
        expected_artifacts = self._fetch_gold_artifacts(task_id)

        if not actual.available:
            outcome = ComparisonOutcome.with_error(actual.error or "Actual result unavailable")
            self._apply_tools_comparison(outcome, actual_sql, gold_sql, analysis.artifacts, expected_artifacts)
            record.outcome = outcome
            return record

        if not expected.available:
            outcome = ComparisonOutcome.with_error(expected.error or "Gold standard unavailable")
            self._apply_tools_comparison(outcome, actual_sql, gold_sql, analysis.artifacts, expected_artifacts)
            record.outcome = outcome
            return record

        outcome = self.comparator.compare(actual.dataframe, expected.dataframe)
        outcome.actual_sql = actual_sql.sql if actual_sql else None
        outcome.gold_sql = gold_sql.sql if gold_sql else None
        self._apply_tools_comparison(outcome, actual_sql, gold_sql, analysis.artifacts, expected_artifacts)
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

    def _fetch_gold_artifacts(self, task_id: str) -> GoldArtifacts:
        getter = getattr(self.gold_result_provider, "get_artifacts", None)
        if callable(getter):
            try:
                artifacts = getter(task_id)
                if isinstance(artifacts, GoldArtifacts):
                    return artifacts
                if isinstance(artifacts, Mapping):
                    return GoldArtifacts(
                        file_reference=str(artifacts.get("file", "") or ""),
                        expected_sql=str(artifacts.get("expected_sql", "") or ""),
                        semantic_model=str(artifacts.get("semantic_model", "") or ""),
                        expected_metrics=str(artifacts.get("expected_metrics", "") or ""),
                    )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to fetch gold artifacts for %s: %s", task_id, exc)
        return GoldArtifacts()

    def _apply_tools_comparison(
        self,
        outcome: ComparisonOutcome,
        actual_sql: Optional[SqlData],
        gold_sql: Optional[SqlData],
        actual_artifacts: Optional[WorkflowArtifacts],
        expected_artifacts: Optional[GoldArtifacts],
    ) -> None:
        actual_tables = []
        expected_tables = []

        if actual_sql:
            if actual_sql.tables:
                actual_tables = _unique_preserve_order(actual_sql.tables)
                outcome.actual_tables = actual_tables
            if actual_sql.error:
                outcome.actual_sql_error = actual_sql.error

        if gold_sql:
            if gold_sql.tables:
                expected_tables = _unique_preserve_order(gold_sql.tables)
                outcome.expected_tables = expected_tables
            if gold_sql.error:
                outcome.match_sql_error = gold_sql.error

        if not outcome.actual_tables and actual_tables:
            outcome.actual_tables = actual_tables
        if not outcome.expected_tables and expected_tables:
            outcome.expected_tables = expected_tables

        if outcome.actual_tables or outcome.expected_tables:
            outcome.matched_tables = compute_table_matches(outcome.actual_tables, outcome.expected_tables)
        self._apply_trajectory_artifact_comparison(outcome, actual_sql, actual_artifacts, expected_artifacts)

    def _apply_trajectory_artifact_comparison(
        self,
        outcome: ComparisonOutcome,
        actual_sql: Optional[SqlData],
        actual_artifacts: Optional[WorkflowArtifacts],
        expected_artifacts: Optional[GoldArtifacts],
    ) -> None:
        """
        Compare tool calls and return results during execution
        :param outcome:
        :param actual_sql:
        :param actual_artifacts:
        :param expected_artifacts:
        :param expected_sql_data:
        :return:
        """
        actual_artifacts = actual_artifacts or WorkflowArtifacts()
        expected_artifacts = expected_artifacts or GoldArtifacts()

        artifact_results: Dict[str, Any] = {}

        # File comparison
        actual_file_candidates = _unique_preserve_order(actual_artifacts.files)
        expected_file = expected_artifacts.file_reference.strip()
        matched_files = _find_matching_candidates(expected_file, actual_file_candidates) if expected_file else []
        artifact_results["expected_file"] = {
            "expected": expected_file,
            "actual": actual_file_candidates,
            "matched_actual": matched_files,
            "match": bool(matched_files) if expected_file else True,
        }

        # Expected SQL comparison (reference search results)
        sql_candidates_source: List[str] = []
        sql_candidates_source.extend(actual_artifacts.reference_sqls)
        sql_candidates_source.extend(actual_artifacts.reference_sql_names)
        sql_candidates = _unique_preserve_order(sql_candidates_source)
        expected_sql_value = expected_artifacts.expected_sql.strip()
        matched_sql = _find_matching_candidates(expected_sql_value, sql_candidates) if expected_sql_value else []
        artifact_results["expected_sql"] = {
            "expected": expected_sql_value,
            "actual": sql_candidates,
            "matched_actual": matched_sql,
            "match": bool(matched_sql) if expected_sql_value else True,
        }

        # Semantic model comparison
        semantic_candidates = _unique_preserve_order(actual_artifacts.semantic_models)
        expected_semantic = expected_artifacts.semantic_model.strip()
        matched_semantic = (
            _find_matching_candidates(expected_semantic, semantic_candidates) if expected_semantic else []
        )
        artifact_results["expected_semantic_model"] = {
            "expected": expected_semantic,
            "actual": semantic_candidates,
            "matched_actual": matched_semantic,
            "match": bool(matched_semantic) if expected_semantic else True,
        }

        # Metrics comparison
        expected_metrics_raw = expected_artifacts.expected_metrics.strip()
        expected_metric_items = _split_expected_items(expected_metrics_raw)
        metric_candidates = _unique_preserve_order(actual_artifacts.metrics_names + actual_artifacts.metrics_texts)
        matched_metric_expected: list[str] = []
        matched_metric_actual: list[str] = []
        missing_metric_expected: list[str] = []

        for item in expected_metric_items:
            matches = _find_matching_candidates(item, metric_candidates)
            if matches:
                matched_metric_expected.append(item)
                matched_metric_actual.extend(matches)
            else:
                missing_metric_expected.append(item)

        artifact_results["expected_metrics"] = {
            "expected": expected_metric_items,
            "actual": metric_candidates,
            "matched_expected": matched_metric_expected,
            "matched_actual": _unique_preserve_order(matched_metric_actual),
            "missing_expected": missing_metric_expected,
            "match": not missing_metric_expected if expected_metric_items else True,
        }

        outcome.tools_comparison = artifact_results


def list_trajectory_runs(trajectory_dir: str, namespace: Optional[str] = None) -> Dict[str, List[str]]:
    """
    List all available run IDs in the trajectory directory.

    Args:
        trajectory_dir: Base trajectory directory
        namespace: Optional namespace to filter by

    Returns:
        Dict mapping namespace to list of run_ids (sorted by name, newest first)
    """
    directory = Path(trajectory_dir)
    if not directory.exists():
        return {}

    runs: Dict[str, List[str]] = {}

    if namespace:
        # List runs for specific namespace
        namespace_dir = directory / namespace
        if namespace_dir.exists() and namespace_dir.is_dir():
            run_dirs = sorted([d.name for d in namespace_dir.iterdir() if d.is_dir()], reverse=True)
            if run_dirs:
                runs[namespace] = run_dirs
    else:
        # List runs for all namespaces
        for ns_dir in directory.iterdir():
            if ns_dir.is_dir():
                ns_name = ns_dir.name
                run_dirs = sorted([d.name for d in ns_dir.iterdir() if d.is_dir()], reverse=True)
                if run_dirs:
                    runs[ns_name] = run_dirs

    return runs


def list_save_runs(save_dir: str, namespace: Optional[str] = None) -> Dict[str, List[str]]:
    """
    List all available run IDs in the save directory.

    Args:
        save_dir: Base save directory
        namespace: Optional namespace to filter by

    Returns:
        Dict mapping namespace to list of run_ids (sorted by name, newest first)
    """
    directory = Path(save_dir)
    if not directory.exists():
        return {}

    runs: Dict[str, List[str]] = {}

    if namespace:
        # List runs for specific namespace
        namespace_dir = directory / namespace
        if namespace_dir.exists() and namespace_dir.is_dir():
            run_dirs = sorted([d.name for d in namespace_dir.iterdir() if d.is_dir()], reverse=True)
            if run_dirs:
                runs[namespace] = run_dirs
    else:
        # List runs for all namespaces
        for ns_dir in directory.iterdir():
            if ns_dir.is_dir():
                ns_name = ns_dir.name
                run_dirs = sorted([d.name for d in ns_dir.iterdir() if d.is_dir()], reverse=True)
                if run_dirs:
                    runs[ns_name] = run_dirs

    return runs


def collect_latest_trajectory_files(save_dir: str, namespace: str, run_id: Optional[str] = None) -> Dict[str, Path]:
    """
    Collect latest trajectory files from directory.

    Uses hierarchical structure: {save_dir}/{namespace}/{run_id}/*.yaml

    Args:
        save_dir: Base trajectory directory
        namespace: Namespace to filter by (required)
        run_id: Optional run_id to filter by. If None, uses latest run.

    Returns:
        Dict mapping task_id to latest trajectory file path
    """
    directory = Path(save_dir)
    if not directory.exists():
        return {}

    file_groups: dict[str, list[tuple[float, Path]]] = defaultdict(list)

    # Determine search path
    namespace_dir = directory / namespace
    if run_id:
        # Specific namespace and run
        search_path = namespace_dir / run_id
    else:
        # Auto-select latest run
        if namespace_dir.exists() and namespace_dir.is_dir():
            run_dirs = sorted([d for d in namespace_dir.iterdir() if d.is_dir()], key=lambda d: d.name, reverse=True)
            if run_dirs:
                search_path = run_dirs[0]
                logger.info(f"Auto-selected latest run for trajectories: {run_dirs[0].name}")
            else:
                # No run subdirectories found
                search_path = namespace_dir
        else:
            # Namespace directory doesn't exist
            return {}

    # Collect trajectory files from search path
    if search_path.exists():
        for filepath in search_path.glob("*.yaml"):
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


def compare_pandas_tables(actual_df: pd.DataFrame, gold_df: pd.DataFrame) -> Dict[str, Any]:
    if len(actual_df) != len(gold_df):
        return {
            "match_rate": 0.0,
            "matched_columns": [],
            "missing_columns": [],
            "extra_columns": [],
        }

    matches: list[tuple[str, str]] = []
    matched_pred_cols: set[str] = set()
    unmatched_gold_cols: set[str] = set(gold_df.columns)

    for pred_col in actual_df.columns:
        if pred_col in matched_pred_cols:
            continue
        for gold_col in list(unmatched_gold_cols):
            try:
                if columns_match(actual_df[pred_col], gold_df[gold_col]):
                    matches.append((pred_col, gold_col))
                    matched_pred_cols.add(pred_col)
                    unmatched_gold_cols.discard(gold_col)
                    break
            except Exception:
                continue

    un_matches = list(unmatched_gold_cols)
    extra_columns = [col for col in actual_df.columns if col not in matched_pred_cols]

    total_gold_cols = len(gold_df.columns)
    match_rate = (len(matches) / total_gold_cols) if total_gold_cols > 0 else 1.0

    return {
        "match_rate": round(match_rate, 4),
        "matched_columns": matches,
        "missing_columns": un_matches,
        "extra_columns": extra_columns,
    }


def columns_match(series_a: pd.Series, series_b: pd.Series, tol: float = 1e-6) -> bool:
    if len(series_a) != len(series_b):
        return False

    # Sort both series to ensure order-independent comparison
    series_a_sorted = series_a.sort_values(ignore_index=True)
    series_b_sorted = series_b.sort_values(ignore_index=True)

    for a, b in zip(series_a_sorted, series_b_sorted):
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


def _resolve_optional_path(base_path: Path, relative_path: Optional[str]) -> Optional[Path]:
    if not relative_path:
        return None
    expanded = os.path.expandvars(os.path.expanduser(relative_path))
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = base_path / candidate
    return candidate


def _ensure_question_file_path(base_path: Path, config: BenchmarkConfig) -> Path:
    question_path = _resolve_optional_path(base_path, config.question_file)
    if question_path is None:
        raise DatusException(
            code=ErrorCode.COMMON_CONFIG_ERROR,
            message="The `question_file` field of Benchmark configuration is required",
        )
    return question_path


def _build_gold_sql_provider(
    config: BenchmarkConfig,
    base_path: Path,
    question_file_path: Path,
    dialect: str,
) -> Optional[SqlProvider]:
    sql_source = _resolve_optional_path(base_path, config.gold_sql_path) or question_file_path
    if sql_source.is_dir():
        return DirectorySqlProvider(str(sql_source), dialect=dialect)

    suffix = sql_source.suffix.lower()
    task_id_key = config.question_id_key or "_task_id"
    sql_key = config.gold_sql_key or ""

    if suffix in {".json", ".jsonl"}:
        if not sql_key:
            raise DatusException(
                code=ErrorCode.COMMON_FIELD_REQUIRED,
                message="gold_sql_key is required when gold_sql_path points to a JSON/JSONL file",
            )
        return JsonMappingSqlProvider(
            str(sql_source),
            task_id_key=task_id_key,
            sql_key=sql_key,
            dialect=dialect,
        )

    if suffix in {".csv", ".tsv"}:
        if not sql_key:
            raise DatusException(
                code=ErrorCode.COMMON_FIELD_REQUIRED,
                message="gold_sql_key is required when gold_sql_path points to a CSV/TSV file",
            )
        return CsvColumnSqlProvider(
            str(sql_source),
            task_id_key=task_id_key,
            sql_key=sql_key,
            dialect=dialect,
        )

    logger.warning(f"Unsupported gold SQL source format: {sql_source}")
    return None


def _build_gold_result_provider(
    agent_config: AgentConfig,
    config: BenchmarkConfig,
    base_path: Path,
    question_file_path: Path,
    allowed_task_ids: Optional[set[str]],
) -> ResultProvider:
    result_path = _resolve_optional_path(base_path, config.gold_result_path)

    if result_path and result_path.is_dir():
        return CsvPerTaskResultProvider(str(result_path))

    task_id_key = config.question_id_key or "_task_id"
    sql_key = config.gold_sql_key or ""
    query_result_key = config.gold_result_key or ""
    db_key = config.db_key or ""

    if result_path is None:
        gold_sql_path = _resolve_optional_path(base_path, config.gold_sql_path)
        if gold_sql_path and gold_sql_path.is_dir():
            raise DatusException(
                code=ErrorCode.COMMON_VALIDATION_FAILED,
                message="gold_result_path must be provided when gold_sql_path is a directory.",
            )
        result_file = gold_sql_path or question_file_path
    else:
        result_file = result_path

    if not result_file.exists():
        raise DatusException(
            code=ErrorCode.COMMON_FILE_NOT_FOUND,
            message_args={"config_name": "Gold Result", "file_name": str(result_file)},
        )

    if not query_result_key and not sql_key:
        raise DatusException(
            code=ErrorCode.COMMON_FIELD_REQUIRED,
            message="At least one of gold_result_key or gold_sql_key must be provided for gold result evaluation.",
        )

    db_manager = db_manager_instance(agent_config.namespaces)
    connections = db_manager.get_connections(agent_config.current_namespace)

    return SingleFileGoldProvider(
        result_file=str(result_file),
        connections=connections,
        task_id_key=task_id_key,
        sql_key=sql_key,
        query_result_key=query_result_key,
        db_key=db_key,
        allowed_task_ids=allowed_task_ids,
    )


def evaluate_benchmark(
    agent_config: AgentConfig,
    benchmark_platform: str,
    target_task_ids: Optional[Iterable[str]] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    namespace = agent_config.current_namespace
    trajectory_directory = Path(agent_config.trajectory_dir)

    try:
        benchmark_config = agent_config.benchmark_config(benchmark_platform)
    except DatusException as exc:
        logger.error(f"Failed to load benchmark configuration for {benchmark_platform}: {exc}")
        return {}

    task_id_key = benchmark_config.question_id_key or "_task_id"
    if not target_task_ids:
        target_task_ids = {
            str(task.get(task_id_key))
            for task in load_benchmark_tasks(agent_config, benchmark_platform)
            if task.get(task_id_key) is not None
        }

    benchmark_root = Path(agent_config.benchmark_path(benchmark_platform))
    question_file_path = _ensure_question_file_path(benchmark_root, benchmark_config)

    allowed_task_ids = {str(task_id) for task_id in target_task_ids} if target_task_ids else None

    configured_dialect = getattr(agent_config, "db_type", "")
    if isinstance(configured_dialect, DBType):
        dialect = configured_dialect.value
    else:
        dialect = str(configured_dialect or "").strip().lower()
    if not dialect:
        dialect = _default_sql_dialect(benchmark_platform)

    # Use base save directory for providers, they will handle subdirectories
    save_base_dir = Path(agent_config._save_dir)
    agent_result_provider = CsvPerTaskResultProvider(str(save_base_dir), namespace=namespace, run_id=run_id)
    result_sql_provider = AgentResultSqlProvider(
        str(save_base_dir), namespace=namespace, run_id=run_id, dialect=dialect
    )
    try:
        gold_sql_provider = _build_gold_sql_provider(benchmark_config, benchmark_root, question_file_path, dialect)
        gold_result_provider = _build_gold_result_provider(
            agent_config,
            benchmark_config,
            benchmark_root,
            question_file_path,
            allowed_task_ids,
        )
    except DatusException as exc:
        logger.error(f"Failed to prepare gold references for benchmark {benchmark_platform}: {exc}")
        return {}

    evaluator = BenchmarkEvaluator(
        trajectory_parser=TrajectoryParser(),
        result_provider=agent_result_provider,
        gold_result_provider=gold_result_provider,
        result_sql_provider=result_sql_provider,
        gold_sql_provider=gold_sql_provider,
    )

    report = evaluator.evaluate_directory(
        str(trajectory_directory), target_task_ids, namespace=namespace, run_id=run_id
    )
    return report.to_dict()


def evaluate_benchmark_and_report(
    agent_config: AgentConfig,
    benchmark_platform: str,
    target_task_ids: Optional[Iterable[str]] = None,
    output_file: Optional[str] = None,
    log_summary: bool = True,
    run_id: Optional[str] = None,
    summary_report_file: Optional[str] = None,
) -> Dict[str, Any]:
    accuracy_report = evaluate_benchmark(
        agent_config=agent_config,
        benchmark_platform=benchmark_platform,
        target_task_ids=target_task_ids,
        run_id=run_id,
    )

    if accuracy_report.get("status") == "success":
        if log_summary:
            _log_accuracy_summary(accuracy_report, summary_report_file=summary_report_file)
        if output_file:
            with open(output_file, "w", encoding="utf-8") as handle:
                json.dump(accuracy_report, handle, ensure_ascii=False, indent=2)
            logger.info(f" For detailed comparison results, see: {output_file}")
        else:
            logger.info(" If you want to see the details, please pass in the parameter --output_file")
    else:
        logger.error(f"Accuracy evaluation failed: {accuracy_report.get('message')}")

    return accuracy_report


def _log_accuracy_summary(accuracy_report: Dict[str, Any], summary_report_file: Optional[str] = None) -> None:
    summary = accuracy_report.get("summary", {})
    task_ids_section = accuracy_report.get("task_ids", {})
    details_section = accuracy_report.get("details", {})

    def _parse_task_ids(raw_ids: Any) -> list[str]:
        if not raw_ids:
            return []
        if isinstance(raw_ids, (list, tuple, set)):
            iterable = raw_ids
        else:
            iterable = str(raw_ids).split(",")
        parsed = []
        for value in iterable:
            text = str(value).strip()
            if text:
                parsed.append(text)
        return parsed

    def _natural_sort_key(value: str) -> list[Any]:
        parts = re.split(r"(\\d+)", value)
        key: list[Any] = []
        for part in parts:
            if not part:
                continue
            if part.isdigit():
                key.append(int(part))
            else:
                key.append(part.lower())
        return key

    def _sorted_task_ids(task_id_set: Iterable[str]) -> list[str]:
        return sorted({task_id for task_id in task_id_set if task_id}, key=_natural_sort_key)

    def _row_count(shape: Any) -> Optional[int]:
        if isinstance(shape, (list, tuple)) and shape:
            try:
                return int(shape[0])
            except (TypeError, ValueError):
                return None
        return None

    matched_ids = set(_parse_task_ids(task_ids_section.get("matched_task_ids")))
    empty_result_ids = set(_parse_task_ids(task_ids_section.get("empty_result_task_ids")))
    mismatched_ids = set(_parse_task_ids(task_ids_section.get("mismatched_task_ids")))
    failed_task_ids = set(_parse_task_ids(task_ids_section.get("failed_task_ids")))

    if isinstance(details_section, Mapping):
        detail_map = {str(key): value for key, value in details_section.items()}
    else:
        detail_map = {}

    all_task_ids = set(detail_map.keys()) | matched_ids | empty_result_ids | mismatched_ids | failed_task_ids

    total_queries_raw = summary.get("total_files")
    try:
        total_queries = int(total_queries_raw) if total_queries_raw is not None else 0
    except (TypeError, ValueError):
        total_queries = 0
    if total_queries == 0:
        total_queries = len(all_task_ids)

    remaining_failures = (all_task_ids | failed_task_ids | mismatched_ids) - matched_ids - empty_result_ids

    table_mismatch_ids: set[str] = set()
    row_count_mismatch_ids: set[str] = set()
    column_value_mismatch_ids: set[str] = set()

    for task_id in remaining_failures:
        detail = detail_map.get(task_id, {})
        comparison_results = []
        if isinstance(detail, Mapping):
            comparison_results = detail.get("comparison_results") or []
        table_mismatch = False
        row_mismatch = False
        value_mismatch = False

        if not comparison_results:
            value_mismatch = True
        else:
            for record in comparison_results:
                if not isinstance(record, Mapping):
                    continue
                comparison = record.get("comparison")
                if not isinstance(comparison, Mapping):
                    value_mismatch = True
                    continue
                if comparison.get("error"):
                    value_mismatch = True
                    continue

                matched_tables = {table for table in comparison.get("matched_tables") or [] if table}
                actual_tables = {table for table in comparison.get("actual_tables") or [] if table}
                expected_tables = {table for table in comparison.get("expected_tables") or [] if table}
                if matched_tables and expected_tables:
                    if (len(actual_tables) != len(expected_tables)) or (len(matched_tables) != len(expected_tables)):
                        table_mismatch = True
                        break

                actual_rows = _row_count(comparison.get("actual_shape"))
                expected_rows = _row_count(comparison.get("expected_shape"))
                if actual_rows is not None and expected_rows is not None and actual_rows != expected_rows:
                    row_mismatch = True
                    continue

                match_rate_value = comparison.get("match_rate")
                try:
                    match_rate = float(match_rate_value) if match_rate_value is not None else None
                except (TypeError, ValueError):
                    match_rate = None

                if match_rate is not None and match_rate >= 1.0:
                    continue

                missing_cols = comparison.get("missing_columns")

                if missing_cols:
                    value_mismatch = True
                elif match_rate is None or match_rate < 1.0:
                    value_mismatch = True

        if table_mismatch:
            table_mismatch_ids.add(task_id)
            continue
        if row_mismatch:
            row_count_mismatch_ids.add(task_id)
            continue
        if value_mismatch or task_id in mismatched_ids or task_id in failed_task_ids:
            column_value_mismatch_ids.add(task_id)

    unclassified_failures = remaining_failures - table_mismatch_ids - row_count_mismatch_ids - column_value_mismatch_ids
    if unclassified_failures:
        column_value_mismatch_ids.update(unclassified_failures)

    result_mismatch_ids = row_count_mismatch_ids | column_value_mismatch_ids

    passed_count = len(matched_ids)
    no_sql_count = len(empty_result_ids)
    failed_count = len(remaining_failures)
    table_mismatch_count = len(table_mismatch_ids)
    missmatch_row_count = len(row_count_mismatch_ids)
    missmatch_column_count = len(column_value_mismatch_ids)
    result_mismatch_count = len(result_mismatch_ids)

    def _percentage(count: int) -> float:
        if total_queries <= 0:
            return 0.0
        return (count / total_queries) * 100

    def _format_stat_line(label: str, count: int) -> str:
        pct_text = f"({_percentage(count):.0f}%)"
        return f"{label:<40}{count:>5}    {pct_text}"

    def _format_list_line(label: str, items: Iterable[str]) -> str:
        values = _sorted_task_ids(items)
        list_text = ", ".join(values) if values else "None"
        return f"{label:<40}{list_text}"

    # Extract metrics summary
    metrics_summary = summary.get("metrics_summary", {})
    total_with_metrics = metrics_summary.get("total_with_expected_metrics", 0)
    metrics_matched = metrics_summary.get("metrics_matched_count", 0)
    metrics_mismatched = metrics_summary.get("metrics_mismatched_count", 0)
    metrics_match_rate = metrics_summary.get("metrics_match_rate", 0.0)
    metrics_matched_ids_str = metrics_summary.get("metrics_matched_task_ids", "")
    metrics_mismatched_ids_str = metrics_summary.get("metrics_mismatched_task_ids", "")

    metrics_matched_ids = set(_parse_task_ids(metrics_matched_ids_str))
    metrics_mismatched_ids = set(_parse_task_ids(metrics_mismatched_ids_str))

    separator = "─" * 80
    report_lines = [
        separator,
        f" Datus Evaluation Summary (Total: {total_queries} Queries)",
        separator,
        _format_stat_line(" ✅ Passed:", passed_count),
        _format_stat_line(" ⚠️  No SQL / Empty Result:", no_sql_count),
        _format_stat_line(" ❌ Failed:", failed_count),
        _format_stat_line("     • Table Mismatch:", table_mismatch_count),
        _format_stat_line("     • Table Matched (Result Mismatch):", result_mismatch_count),
        _format_stat_line("         - Row Count Mismatch:", missmatch_row_count),
        _format_stat_line("         - Column Value Mismatch:", missmatch_column_count),
        separator,
    ]

    # Add metrics summary if there are tasks with expected metrics
    if total_with_metrics > 0:
        report_lines.extend(
            [
                f" Metrics Evaluation (Total with Expected Metrics: {total_with_metrics})",
                separator,
                _format_stat_line(" ✅ Metrics Matched:", metrics_matched),
                _format_stat_line(" ❌ Metrics Mismatched:", metrics_mismatched),
                f" Metrics Match Rate: {metrics_match_rate:.2f}%",
                separator,
            ]
        )

    report_lines.extend(
        [
            "",
            _format_list_line(" Passed Queries:", matched_ids),
            _format_list_line(" No SQL / Empty Result Queries:", empty_result_ids),
            _format_list_line(" Failed (Table Mismatch):", table_mismatch_ids),
            _format_list_line(" Failed (Row Count Mismatch):", row_count_mismatch_ids),
            _format_list_line(" Failed (Column Value Mismatch):", column_value_mismatch_ids),
        ]
    )

    # Add metrics task IDs if available
    if total_with_metrics > 0:
        report_lines.extend(
            [
                "",
                _format_list_line(" Metrics Matched Queries:", metrics_matched_ids),
                _format_list_line(" Metrics Mismatched Queries:", metrics_mismatched_ids),
            ]
        )

    report_lines.extend(["", separator, ""])

    report_content = "\n".join(report_lines)
    logger.info(f"\n\n{report_content}")

    # Write to summary report file if specified (append mode)
    if summary_report_file:
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(summary_report_file, "a", encoding="utf-8") as f:
                f.write(f"\n\n### Report generated at {timestamp}\n")
                f.write(report_content)
                f.write("\n")
            logger.info(f" Summary report appended to: {summary_report_file}")
        except Exception as e:
            logger.warning(f" Failed to write summary report to file: {e}")


def _ensure_task_identifier(task: Dict[str, Any], task_id_key: str, position: int) -> Dict[str, Any]:
    """
    Guarantee each benchmark task carries an identifier.

    When the configured key is absent or empty we derive the task_id from the row order.
    """
    if not isinstance(task, MutableMapping):
        return task

    task_id_value = task.get(task_id_key)
    if task_id_value in (None, ""):
        task[task_id_key] = str(position)
    return task  # type: ignore[return-value]


def load_benchmark_tasks(agent_config: AgentConfig, benchmark_platform: str) -> Iterable[Dict[str, Any]]:
    benchmark_config = agent_config.benchmark_config(benchmark_platform)
    benchmark_file = _ensure_question_file_path(Path(agent_config.benchmark_path(benchmark_platform)), benchmark_config)
    if not benchmark_file.exists():
        raise DatusException(
            ErrorCode.COMMON_FILE_NOT_FOUND,
            message_args={"config_name": "Benchmarking Task File", "file_name": benchmark_file},
        )

    task_id_key = benchmark_config.question_id_key or "_task_id"

    def _task_iter():
        if benchmark_file.suffix == ".json":
            with benchmark_file.open(mode="r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise DatusException(
                        ErrorCode.COMMON_VALIDATION_FAILED,
                        message="Only supports JSON task files in List format",
                    )
                for item in data:
                    yield item
        elif benchmark_file.suffix in (".csv", ".tsv"):
            delimiter = "\t" if benchmark_file.suffix == ".tsv" else ","
            with benchmark_file.open(mode="r", encoding="utf-8") as f:
                csv_reader = csv.DictReader(f, delimiter=delimiter)
                for row in csv_reader:
                    yield row
        elif benchmark_file.suffix == ".jsonl":
            with benchmark_file.open(mode="r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    yield json.loads(line)
        else:
            raise DatusException(
                ErrorCode.COMMON_VALIDATION_FAILED,
                message=f"Unsupported benchmark file format: {benchmark_file.suffix}",
            )

    for idx, task in enumerate(_task_iter(), start=1):
        if isinstance(task, Mapping):
            yield _ensure_task_identifier(dict(task), task_id_key, idx)  # ensure mutable copy
        else:
            yield task
