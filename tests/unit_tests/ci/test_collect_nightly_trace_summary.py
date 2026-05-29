import json

from ci.collect_nightly_trace_summary import build_process_diagnostics, dedupe_trace_refs, summarize_observations


def test_summarize_observations_reports_counts_tokens_and_findings():
    observations = [
        {
            "id": "agent-1",
            "type": "AGENT",
            "name": "Agent workflow",
            "startTime": "2026-05-28T00:00:00Z",
            "endTime": "2026-05-28T00:00:10Z",
        },
        {
            "id": "gen-1",
            "type": "GENERATION",
            "name": "generation",
            "startTime": "2026-05-28T00:00:01Z",
            "endTime": "2026-05-28T00:00:40Z",
            "usageDetails": {"input": 12, "output": 8, "total": 20},
        },
        {
            "id": "tool-1",
            "type": "TOOL",
            "name": "read_query",
            "level": "ERROR",
            "statusMessage": "query failed",
            "startTime": "2026-05-28T00:00:03Z",
            "endTime": "2026-05-28T00:00:04Z",
        },
    ]

    summary = summarize_observations(observations, slow_span_threshold_seconds=30.0)

    assert summary["observation_count"] == 3
    assert summary["agent_span_count"] == 1
    assert summary["generation_span_count"] == 1
    assert summary["tool_span_count"] == 1
    assert summary["failed_span_count"] == 1
    assert summary["token_usage"] == {"input": 12, "output": 8, "total": 20}
    assert summary["finding_type_counts"] == {"failed_span": 1, "slow_span": 1}


def test_build_process_diagnostics_groups_by_suite():
    diagnostics = build_process_diagnostics(
        [
            {
                "suite": "Gen Agent Tests",
                "nodeid": "tests/test_agent.py::test_a",
                "trace_id": "trace-a",
                "trace_fetch_status": "fetched",
                "observation_count": 3,
                "tool_span_count": 1,
                "generation_span_count": 1,
                "agent_span_count": 1,
                "failed_span_count": 0,
                "duration_seconds": 10.0,
                "finding_type_counts": {"slow_span": 1},
                "token_usage": {"total": 100},
            },
            {
                "suite": "Gen Agent Tests",
                "nodeid": "tests/test_agent.py::test_b",
                "trace_fetch_status": "missing_trace_reference",
                "observation_count": 0,
                "finding_type_counts": {},
                "token_usage": {},
            },
        ]
    )

    assert diagnostics["summary"]["case_count"] == 2
    assert diagnostics["summary"]["trace_reference_count"] == 1
    assert diagnostics["summary"]["trace_fetch_status_counts"] == {
        "fetched": 1,
        "missing_trace_reference": 1,
    }
    assert diagnostics["summary"]["finding_type_counts"] == {"slow_span": 1}
    assert diagnostics["summary"]["token_usage"] == {"total": 100}
    assert diagnostics["suites"][0]["suite"] == "Gen Agent Tests"


def test_dedupe_trace_refs_keeps_latest_row_for_same_case():
    rows = [
        {"suite": "S", "nodeid": "n", "trace_id": "t", "outcome": "failed"},
        {"suite": "S", "nodeid": "n", "trace_id": "t", "outcome": "passed"},
        {"suite": "S", "nodeid": "n", "outcome": "passed"},
    ]

    deduped = dedupe_trace_refs(json.loads(json.dumps(rows)))

    assert deduped == [
        {"suite": "S", "nodeid": "n", "outcome": "passed"},
        {"suite": "S", "nodeid": "n", "trace_id": "t", "outcome": "passed"},
    ]
