import json
import urllib.parse

from ci.collect_nightly_trace_summary import (
    _collect_token_usage,
    build_process_diagnostics,
    dedupe_trace_refs,
    fetch_observations,
    summarize_observations,
)


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
    assert summary["finding_type_counts"] == {"failed_span": 1}


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


def test_fetch_observations_uses_v2_cursor_from_meta(monkeypatch):
    requested_cursors: list[str] = []
    requested_fields: list[str] = []

    class FakeResponse:
        def __init__(self, payload: dict):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    responses = [
        {"data": [{"id": "first"}], "meta": {"cursor": "cursor-2"}},
        {"data": [{"id": "second"}], "meta": {}},
    ]

    def fake_urlopen(request, timeout):
        parsed = urllib.parse.urlparse(request.full_url)
        query = urllib.parse.parse_qs(parsed.query)
        requested_cursors.append(query.get("cursor", [""])[0])
        requested_fields.append(query.get("fields", [""])[0])
        return FakeResponse(responses.pop(0))

    monkeypatch.setattr("ci.collect_nightly_trace_summary.urllib.request.urlopen", fake_urlopen)

    observations = fetch_observations(
        trace_id="trace-1",
        base_url="https://langfuse.test",
        public_key="pk",
        secret_key="sk",
        from_start_time="2026-05-28T00:00:00Z",
        to_start_time="2026-05-28T01:00:00Z",
        timeout=1.0,
        max_pages=3,
    )

    assert [item["id"] for item in observations] == ["first", "second"]
    assert requested_cursors == ["", "cursor-2"]
    assert requested_fields == ["core,basic,usage,metrics", "core,basic,usage,metrics"]


def test_collect_token_usage_prefers_one_usage_payload_over_top_level_fields():
    usage = _collect_token_usage(
        [
            {
                "usageDetails": {"input": 10, "output": 5, "total": 15},
                "usage": {"input": 99, "output": 99, "total": 198},
                "promptTokens": 10,
                "completionTokens": 5,
                "totalTokens": 15,
            },
            {"promptTokens": 3, "completionTokens": 2, "totalTokens": 5},
        ]
    )

    assert usage == {"input": 10, "output": 5, "total": 15, "input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
