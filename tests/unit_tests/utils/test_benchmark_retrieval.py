from __future__ import annotations

from datetime import datetime, timezone

from datus.utils.benchmark_retrieval import (
    RetrievalEvent,
    evaluate_retrieval,
    extract_retrieval_events,
    summarize_retrieval,
)
from datus.utils.benchmark_utils import compute_table_matches


def test_evaluate_retrieval_scores_ranked_candidates() -> None:
    event = RetrievalEvent(
        action_id="complete_call_1",
        query_text="school meal eligibility by school",
        requested_top_n=5,
        retrieved_tables=[
            "california_schools.schools",
            "california_schools.frpm",
            "california_schools.satscores",
        ],
        duration_ms=12.5,
        success=True,
        error=None,
    )

    result = evaluate_retrieval(
        events=[event],
        expected_tables=["schools", "frpm"],
        expected_tables_source="explicit",
        result_match=True,
        table_matcher=compute_table_matches,
    )

    assert result.status == "evaluated"
    assert result.expected_tables == ["schools", "frpm"]
    assert result.retrieved_tables == [
        "california_schools.schools",
        "california_schools.frpm",
        "california_schools.satscores",
    ]
    assert result.matched_tables == ["schools", "frpm"]
    assert result.missing_tables == []
    assert result.search_call_count == 1
    assert result.failed_search_call_count == 0
    assert result.recall_at_1 == 0.5
    assert result.recall_at_3 == 1.0
    assert result.recall_at_5 == 1.0
    assert result.first_relevant_rank == 1
    assert result.table_recall == 1.0
    assert result.table_precision == 2 / 3
    assert result.full_recall is True
    assert result.result_match is True
    assert result.diagnosis == "success"


def test_evaluate_retrieval_handles_missing_gold_tables() -> None:
    result = evaluate_retrieval(
        events=[],
        expected_tables=[],
        expected_tables_source="missing",
        result_match=None,
        table_matcher=compute_table_matches,
    )

    assert result.status == "not_evaluable"
    assert result.expected_tables_source == "missing"
    assert result.table_recall is None
    assert result.full_recall is None
    assert result.diagnosis == "outcome_not_evaluable"


def test_evaluate_retrieval_handles_unobserved_search_table() -> None:
    result = evaluate_retrieval(
        events=[],
        expected_tables=["schools"],
        expected_tables_source="explicit",
        result_match=False,
        table_matcher=compute_table_matches,
    )

    assert result.status == "not_observed"
    assert result.search_call_count == 0
    assert result.table_recall is None
    assert result.full_recall is None
    assert result.diagnosis == "outcome_not_evaluable"


def test_evaluate_retrieval_diagnoses_retrieval_bottleneck() -> None:
    event = RetrievalEvent(
        action_id="complete_call_2",
        query_text="district demographics",
        requested_top_n=3,
        retrieved_tables=["california_schools.districts"],
        duration_ms=None,
        success=True,
        error=None,
    )

    result = evaluate_retrieval(
        events=[event],
        expected_tables=["schools", "frpm"],
        expected_tables_source="explicit",
        result_match=False,
        table_matcher=compute_table_matches,
    )

    assert result.status == "evaluated"
    assert result.matched_tables == []
    assert result.missing_tables == ["schools", "frpm"]
    assert result.recall_at_1 == 0.0
    assert result.recall_at_3 == 0.0
    assert result.recall_at_5 == 0.0
    assert result.first_relevant_rank is None
    assert result.table_recall == 0.0
    assert result.table_precision == 0.0
    assert result.full_recall is False
    assert result.diagnosis == "retrieval_likely_bottleneck"


def test_summarize_retrieval_counts_only_evaluated_tasks_in_metric_denominators() -> None:
    evaluated = evaluate_retrieval(
        events=[
            RetrievalEvent(
                action_id="complete_call_3",
                query_text="school facts",
                requested_top_n=5,
                retrieved_tables=["schools"],
                duration_ms=1.0,
                success=True,
                error=None,
            )
        ],
        expected_tables=["schools", "frpm"],
        expected_tables_source="explicit",
        result_match=False,
        table_matcher=compute_table_matches,
    )
    not_observed = evaluate_retrieval(
        events=[],
        expected_tables=["schools"],
        expected_tables_source="explicit",
        result_match=True,
        table_matcher=compute_table_matches,
    )
    not_evaluable = evaluate_retrieval(
        events=[],
        expected_tables=[],
        expected_tables_source="missing",
        result_match=True,
        table_matcher=compute_table_matches,
    )

    summary = summarize_retrieval([evaluated, not_observed, not_evaluable])

    assert summary == {
        "total_tasks": 3,
        "grounded_tasks": 2,
        "observed_tasks": 1,
        "scored_tasks": 1,
        "not_observed_tasks": 1,
        "not_evaluable_tasks": 1,
        "observation_coverage_pct": 50.0,
        "full_recall_count": 0,
        "full_recall_rate_pct": 0.0,
        "mean_table_recall_pct": 50.0,
        "mean_table_precision_pct": 100.0,
        "mean_recall_at_1_pct": 50.0,
        "mean_recall_at_3_pct": 50.0,
        "mean_recall_at_5_pct": 50.0,
        "total_search_calls": 1,
        "failed_search_calls": 0,
        "diagnosis_counts": {
            "retrieval_likely_bottleneck": 1,
            "outcome_not_evaluable": 2,
        },
    }



def test_extract_retrieval_events_reads_structured_search_table_result() -> None:
    action_history = [
        {
            "action_id": "call_1",
            "action_type": "tool",
            "input": {
                "function_name": "search_table",
                "arguments": {"query_text": "school meals", "top_n": 5},
            },
            "output": None,
            "status": "processing",
            "start_time": "2026-07-31T10:00:00+00:00",
            "end_time": None,
        },
        {
            "action_id": "complete_call_1",
            "action_type": "tool",
            "input": {
                "function_name": "search_table",
                "arguments": {"query_text": "school meals", "top_n": 5},
            },
            "output": {
                "result": {
                    "metadata": [
                        {"identifier": "california_schools.schools", "_distance": 0.1},
                        {"identifier": "california_schools.frpm", "_distance": 0.2},
                    ],
                    "sample_data": [],
                }
            },
            "status": "completed",
            "start_time": "2026-07-31T10:00:00+00:00",
            "end_time": "2026-07-31T10:00:00.250000+00:00",
        },
    ]

    events = extract_retrieval_events(action_history)

    assert len(events) == 1
    assert events[0].action_id == "complete_call_1"
    assert events[0].query_text == "school meals"
    assert events[0].requested_top_n == 5
    assert events[0].retrieved_tables == [
        "california_schools.schools",
        "california_schools.frpm",
    ]
    assert events[0].duration_ms == 250.0
    assert events[0].success is True
    assert events[0].error is None


def test_extract_retrieval_events_accepts_prefixed_tool_name_and_json_raw_output() -> None:
    action_history = [
        {
            "action_id": "complete_call_2",
            "action_type": "tool",
            "input": {
                "function_name": "db_tools.search_table",
                "arguments": {"query": "school tests", "top_k": 3},
            },
            "output": {
                "raw_output": (
                    "{\"metadata\":["
                    "{\"database_name\":\"california_schools\","
                    "\"table_name\":\"satscores\"}"
                    "]}"
                )
            },
            "status": "success",
            "start_time": None,
            "end_time": None,
        },
    ]

    events = extract_retrieval_events(action_history)

    assert len(events) == 1
    assert events[0].query_text == "school tests"
    assert events[0].requested_top_n == 3
    assert events[0].retrieved_tables == ["california_schools.satscores"]
    assert events[0].duration_ms is None
    assert events[0].success is True


def test_extract_retrieval_events_preserves_failed_terminal_calls() -> None:
    action_history = [
        {
            "action_id": "complete_call_3",
            "action_type": "tool",
            "input": {
                "function_name": "search_table",
                "arguments": {"query_text": "bad query", "top_n": 5},
            },
            "output": {"error": "database connection failed"},
            "status": "failed",
            "start_time": datetime(2026, 7, 31, 10, 0, 0, tzinfo=timezone.utc),
            "end_time": datetime(2026, 7, 31, 10, 0, 1, tzinfo=timezone.utc),
        },
    ]

    events = extract_retrieval_events(action_history)

    assert len(events) == 1
    assert events[0].action_id == "complete_call_3"
    assert events[0].success is False
    assert events[0].error == "database connection failed"
    assert events[0].retrieved_tables == []
    assert events[0].duration_ms == 1000.0
