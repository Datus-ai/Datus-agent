"""Benchmark how well the agent retrieves the right tables for NL→SQL tasks.

Why this module exists
----------------------
When a benchmark task fails, we need to know *where* it failed:
  - Did schema search miss the gold tables?  → fix retrieval / ranking
  - Did search find the tables but SQL/answer is still wrong? → fix reasoning
  - Did the agent never even call search? → instrumentation / agent policy gap

This file is pure scoring logic. Callers supply:
  - observed search tool calls (RetrievalEvent list)
  - gold expected tables + where they came from
  - whether the final task answer matched (result_match)
  - a TableMatcher so dialect/case/alias rules stay outside this module
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

# Lifecycle of a single task's retrieval score:
#   evaluated      — gold tables exist AND at least one search call was observed
#   not_observed   — gold tables exist BUT agent never called search (can't score)
#   not_evaluable  — no gold tables at all (nothing to compare against)
RetrievalStatus = Literal["evaluated", "not_observed", "not_evaluable"]

# Provenance of the gold table list (useful when debugging bad labels):
#   explicit  — benchmark JSON already lists required tables
#   gold_sql  — we parsed table names out of the gold SQL
#   missing   — neither source available
ExpectedTablesSource = Literal["explicit", "gold_sql", "missing"]

# Failure attribution from the 2×2 of (full_recall, result_match):
#
#   full_recall | result_match | diagnosis
#   ------------|--------------|------------------------------------------
#   True        | True         | success
#   False       | False        | retrieval_likely_bottleneck
#   True        | False        | downstream_reasoning_failure
#   False       | True         | recovered_without_full_retrieval
#   *           | None         | outcome_not_evaluable (no answer label)
Diagnosis = Literal[
    "success",
    "retrieval_likely_bottleneck",
    "downstream_reasoning_failure",
    "recovered_without_full_retrieval",
    "outcome_not_evaluable",
]

# Injected equality/alias rules. Signature: (retrieved_names, expected_names) → matched.
# Keeps this module free of DB-specific name normalization.
TableMatcher = Callable[[Iterable[str], Iterable[str]], list[str]]


@dataclass
class RetrievalEvent:
    """One observed table-search tool call during a single task run.

    A task may trigger several searches (retry, refine query, raise top_n).
    We keep every call so coverage can use the union, while ranking metrics
    intentionally look only at the first successful call's ordered list.
    """

    action_id: str
    query_text: str
    requested_top_n: int | None
    retrieved_tables: list[str]  # ordered as returned by the search backend
    duration_ms: float | None
    success: bool
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "query_text": self.query_text,
            "requested_top_n": self.requested_top_n,
            "retrieved_tables": list(self.retrieved_tables),
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class RetrievalEvaluation:
    """Per-task scorecard: what was expected, what was found, how we diagnose it.

    Two families of metrics live here on purpose:

    1) Coverage (union over ALL search calls)
       table_recall / table_precision / full_recall / missing_tables
       → "Did the agent eventually surface every gold table?"

    2) Ranking (FIRST successful call only, list order preserved)
       recall_at_1/3/5, first_relevant_rank
       → "Was the initial ranking good enough, or only multi-round recovery?"
    """

    status: RetrievalStatus
    expected_tables_source: ExpectedTablesSource
    expected_tables: list[str]
    events: list[RetrievalEvent]
    retrieved_tables: list[str]
    matched_tables: list[str]
    missing_tables: list[str]
    search_call_count: int
    failed_search_call_count: int
    recall_at_1: float | None
    recall_at_3: float | None
    recall_at_5: float | None
    first_relevant_rank: int | None
    table_recall: float | None  # |matched| / |expected|
    table_precision: float | None  # |matched| / |retrieved|
    full_recall: bool | None  # True iff every expected table appears in matched
    result_match: bool | None  # final answer correct? supplied by caller, not computed here
    diagnosis: Diagnosis

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "expected_tables_source": self.expected_tables_source,
            "expected_tables": list(self.expected_tables),
            "events": [event.to_dict() for event in self.events],
            "retrieved_tables": list(self.retrieved_tables),
            "matched_tables": list(self.matched_tables),
            "missing_tables": list(self.missing_tables),
            "search_call_count": self.search_call_count,
            "failed_search_call_count": self.failed_search_call_count,
            "recall_at_1": self.recall_at_1,
            "recall_at_3": self.recall_at_3,
            "recall_at_5": self.recall_at_5,
            "first_relevant_rank": self.first_relevant_rank,
            "table_recall": self.table_recall,
            "table_precision": self.table_precision,
            "full_recall": self.full_recall,
            "result_match": self.result_match,
            "diagnosis": self.diagnosis,
        }


def evaluate_retrieval(
    *,
    events: list[RetrievalEvent],
    expected_tables: list[str],
    expected_tables_source: ExpectedTablesSource,
    result_match: bool | None,
    table_matcher: TableMatcher,
) -> RetrievalEvaluation:
    """Score one task's retrieval quality.

    Control flow is three gates, then score:

      gate 1 — no expected tables  → not_evaluable (can't build a denominator)
      gate 2 — no search events    → not_observed  (instrumentation/agent gap)
      else   — compute coverage + ranking metrics and diagnose
    """
    normalized_expected = _ordered_unique(expected_tables)
    terminal_events = list(events)

    # Gate 1: without gold tables every ratio is undefined.
    if not normalized_expected:
        return RetrievalEvaluation(
            status="not_evaluable",
            expected_tables_source="missing",
            expected_tables=[],
            events=terminal_events,
            retrieved_tables=[],
            matched_tables=[],
            missing_tables=[],
            search_call_count=len(terminal_events),
            failed_search_call_count=sum(not event.success for event in terminal_events),
            recall_at_1=None,
            recall_at_3=None,
            recall_at_5=None,
            first_relevant_rank=None,
            table_recall=None,
            table_precision=None,
            full_recall=None,
            result_match=result_match,
            diagnosis="outcome_not_evaluable",
        )

    # Gate 2: gold exists but nothing was observed — do not pretend recall=0.
    # Treating this as "failed retrieval" would mix agent-policy bugs with
    # ranker quality bugs. Keep them separate via status=not_observed.
    if not terminal_events:
        return RetrievalEvaluation(
            status="not_observed",
            expected_tables_source=expected_tables_source,
            expected_tables=normalized_expected,
            events=[],
            retrieved_tables=[],
            matched_tables=[],
            missing_tables=normalized_expected,
            search_call_count=0,
            failed_search_call_count=0,
            recall_at_1=None,
            recall_at_3=None,
            recall_at_5=None,
            first_relevant_rank=None,
            table_recall=None,
            table_precision=None,
            full_recall=None,
            result_match=result_match,
            diagnosis="outcome_not_evaluable",
        )

    # --- Coverage path: union of tables across every search call -------------
    # Multi-round agents can miss on call #1 and recover later; full_recall
    # should credit that recovery. Ranking metrics below deliberately do not.
    retrieved_tables = _ordered_unique(table for event in terminal_events for table in event.retrieved_tables)
    matched_tables = table_matcher(retrieved_tables, normalized_expected)
    matched_tables = _ordered_unique(matched_tables)
    # Missing check goes back through table_matcher so alias rules apply both ways
    # (e.g. matched holds "db.users" while expected is "users").
    missing_tables = [table for table in normalized_expected if table not in table_matcher(matched_tables, [table])]

    expected_count = len(normalized_expected)
    retrieved_count = len(retrieved_tables)
    table_recall = len(matched_tables) / expected_count
    table_precision = len(matched_tables) / retrieved_count if retrieved_count else 0.0
    full_recall = len(matched_tables) == expected_count

    # --- Ranking path: first successful call only ----------------------------
    # Measures "did the first hit list already surface gold tables high enough?"
    # Failed calls are skipped so a flaky first attempt doesn't poison @k.
    first_successful_tables = next(
        (event.retrieved_tables for event in terminal_events if event.success),
        [],
    )

    recall_at_1 = _recall_at_k(first_successful_tables, normalized_expected, 1, table_matcher)
    recall_at_3 = _recall_at_k(first_successful_tables, normalized_expected, 3, table_matcher)
    recall_at_5 = _recall_at_k(first_successful_tables, normalized_expected, 5, table_matcher)
    first_relevant_rank = _first_relevant_rank(
        first_successful_tables,
        normalized_expected,
        table_matcher,
    )

    return RetrievalEvaluation(
        status="evaluated",
        expected_tables_source=expected_tables_source,
        expected_tables=normalized_expected,
        events=terminal_events,
        retrieved_tables=retrieved_tables,
        matched_tables=matched_tables,
        missing_tables=missing_tables,
        search_call_count=len(terminal_events),
        failed_search_call_count=sum(not event.success for event in terminal_events),
        recall_at_1=recall_at_1,
        recall_at_3=recall_at_3,
        recall_at_5=recall_at_5,
        first_relevant_rank=first_relevant_rank,
        table_recall=table_recall,
        table_precision=table_precision,
        full_recall=full_recall,
        result_match=result_match,
        diagnosis=_diagnose(full_recall, result_match),
    )


def summarize_retrieval(evaluations: list[RetrievalEvaluation]) -> dict[str, Any]:
    """Roll many per-task scorecards into suite-level rates.

    Denominators matter:
      - observation_coverage_pct uses grounded (has gold), not total
        → answers "of labelable tasks, how often did we see search?"
      - recall / precision / full_recall rates use scored only
        → answers "when we could score, how good was retrieval?"
      - diagnosis_counts uses ALL evaluations (including not_evaluable)
        → keeps the full failure taxonomy visible in one histogram
    """
    total_tasks = len(evaluations)
    grounded = [item for item in evaluations if item.status != "not_evaluable"]
    scored = [item for item in evaluations if item.status == "evaluated"]
    not_observed = [item for item in evaluations if item.status == "not_observed"]
    not_evaluable = [item for item in evaluations if item.status == "not_evaluable"]

    diagnosis_counts: dict[str, int] = {}
    for item in evaluations:
        diagnosis_counts[item.diagnosis] = diagnosis_counts.get(item.diagnosis, 0) + 1

    return {
        "total_tasks": total_tasks,
        "grounded_tasks": len(grounded),
        "observed_tasks": len(scored),
        "scored_tasks": len(scored),
        "not_observed_tasks": len(not_observed),
        "not_evaluable_tasks": len(not_evaluable),
        "observation_coverage_pct": _pct(len(scored), len(grounded)),
        "full_recall_count": sum(item.full_recall is True for item in scored),
        "full_recall_rate_pct": _pct(
            sum(item.full_recall is True for item in scored),
            len(scored),
        ),
        "mean_table_recall_pct": _mean_pct(item.table_recall for item in scored),
        "mean_table_precision_pct": _mean_pct(item.table_precision for item in scored),
        "mean_recall_at_1_pct": _mean_pct(item.recall_at_1 for item in scored),
        "mean_recall_at_3_pct": _mean_pct(item.recall_at_3 for item in scored),
        "mean_recall_at_5_pct": _mean_pct(item.recall_at_5 for item in scored),
        "total_search_calls": sum(item.search_call_count for item in evaluations),
        "failed_search_calls": sum(item.failed_search_call_count for item in evaluations),
        "diagnosis_counts": diagnosis_counts,
    }


def extract_retrieval_events(action_history: list[Mapping[str, Any]]) -> list[RetrievalEvent]:
    events: list[RetrievalEvent] = []
    seen_action_ids: set[str] = set()

    for action in action_history:
        record = _as_mapping(action)
        action_id = str(record.get("action_id") or "")
        if not action_id or action_id in seen_action_ids:
            continue
        if not _is_terminal_action(record):
            continue

        tool_name, arguments = _tool_name_and_arguments(record)
        if not _is_search_table_tool(tool_name):
            continue

        seen_action_ids.add(action_id)
        output = _extract_output_payload(record.get("output"))
        success = _is_success_status(record.get("status")) and not _payload_has_error(output)

        events.append(
            RetrievalEvent(
                action_id=action_id,
                query_text=_query_text(arguments),
                requested_top_n=_requested_top_n(arguments),
                retrieved_tables=_metadata_tables(output),
                duration_ms=_duration_ms(record.get("start_time"), record.get("end_time")),
                success=success,
                error=None if success else _error_message(output),
            )
        )

    return events


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, Mapping) else {}
    if hasattr(value, "dict"):
        dumped = value.dict()
        return dumped if isinstance(dumped, Mapping) else {}
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _is_terminal_action(record: Mapping[str, Any]) -> bool:
    if record.get("output") is not None:
        return True
    status = str(record.get("status") or "").lower()
    return status in {"completed", "complete", "success", "succeeded", "failed", "error"}


def _tool_name_and_arguments(record: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    input_payload = _as_mapping(record.get("input"))
    tool_name = str(
        input_payload.get("function_name")
        or input_payload.get("tool_name")
        or input_payload.get("name")
        or ""
    )
    arguments = _as_mapping(input_payload.get("arguments"))
    return tool_name, arguments


def _is_search_table_tool(tool_name: str) -> bool:
    return tool_name == "search_table" or tool_name.endswith(".search_table")


def _extract_output_payload(output: Any) -> Mapping[str, Any]:
    output_mapping = _as_mapping(output)
    result = output_mapping.get("result")
    if isinstance(result, Mapping):
        return result

    raw_output = output_mapping.get("raw_output")
    if isinstance(raw_output, Mapping):
        return raw_output
    if isinstance(raw_output, str):
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            return output_mapping
        return parsed if isinstance(parsed, Mapping) else output_mapping

    return output_mapping


def _is_success_status(status: Any) -> bool:
    return str(status or "").lower() in {"completed", "complete", "success", "succeeded"}


def _payload_has_error(payload: Mapping[str, Any]) -> bool:
    return bool(payload.get("error") or payload.get("exception"))


def _error_message(payload: Mapping[str, Any]) -> str | None:
    error = payload.get("error") or payload.get("exception")
    if error is None:
        return None
    return str(error)


def _query_text(arguments: Mapping[str, Any]) -> str:
    query = arguments.get("query_text") or arguments.get("query") or arguments.get("text") or ""
    return str(query)


def _requested_top_n(arguments: Mapping[str, Any]) -> int | None:
    value = arguments.get("top_n")
    if value is None:
        value = arguments.get("top_k")
    if value is None:
        value = arguments.get("limit")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _metadata_tables(payload: Mapping[str, Any]) -> list[str]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, list):
        return []

    tables: list[str] = []
    for item in metadata:
        row = _as_mapping(item)
        table_name = _table_identifier(row)
        if table_name:
            tables.append(table_name)
    return _ordered_unique(tables)


def _table_identifier(row: Mapping[str, Any]) -> str:
    identifier = row.get("identifier")
    if identifier:
        return str(identifier)

    parts = [
        row.get("catalog_name"),
        row.get("database_name"),
        row.get("schema_name"),
        row.get("table_name"),
    ]
    cleaned = [str(part) for part in parts if part]
    return ".".join(cleaned)


def _duration_ms(start_time: Any, end_time: Any) -> float | None:
    start = _parse_datetime(start_time)
    end = _parse_datetime(end_time)
    if start is None or end is None:
        return None
    duration = (end - start).total_seconds() * 1000
    if duration < 0:
        return None
    return round(duration, 3)


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _ordered_unique(values: Iterable[str]) -> list[str]:
    """Normalize names: strip whitespace, drop empties/dupes, keep first-seen order.

    Order stability matters because ranking metrics slice [:k] on this list.
    """
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _recall_at_k(
    retrieved_tables: list[str],
    expected_tables: list[str],
    k: int,
    table_matcher: TableMatcher,
) -> float:
    """|expected ∩ top-k retrieved| / |expected|.

    Classic IR recall@k, but the 'relevance' test is delegated to table_matcher
    so `schema.users` can still count as a hit for expected `users`.
    """
    matched = table_matcher(retrieved_tables[:k], expected_tables)
    return len(_ordered_unique(matched)) / len(expected_tables)


def _first_relevant_rank(
    retrieved_tables: list[str],
    expected_tables: list[str],
    table_matcher: TableMatcher,
) -> int | None:
    """1-based position of the first retrieved table that matches any gold table.

    None means the entire ranked list missed every expected table.
    Useful as a latency-to-relevance signal (lower is better).
    """
    for index, table in enumerate(retrieved_tables, start=1):
        if table_matcher([table], expected_tables):
            return index
    return None


def _diagnose(full_recall: bool, result_match: bool | None) -> Diagnosis:
    """Turn (did we find all gold tables?, was the final answer correct?) into a bucket.

    This is the product's main debugging lever:
      retrieval_likely_bottleneck     → invest in search / embeddings / top_n
      downstream_reasoning_failure    → invest in SQL gen / planner / tools
      recovered_without_full_retrieval→ gold tables may be over-strict, or agent
                                        found an alternate valid path
    """
    if result_match is None:
        return "outcome_not_evaluable"
    if full_recall and result_match:
        return "success"
    if not full_recall and not result_match:
        return "retrieval_likely_bottleneck"
    if full_recall and not result_match:
        return "downstream_reasoning_failure"
    return "recovered_without_full_retrieval"


def _pct(numerator: int, denominator: int) -> float | None:
    """Safe percent; None when the denominator is empty (avoids fake 0%)."""
    if denominator == 0:
        return None
    return round((numerator / denominator) * 100, 2)


def _mean_pct(values: Iterable[float | None]) -> float | None:
    """Average of unit-interval ratios, reported as percent. Skips None entries."""
    present = [value for value in values if value is not None]
    if not present:
        return None
    return round((sum(present) / len(present)) * 100, 2)
