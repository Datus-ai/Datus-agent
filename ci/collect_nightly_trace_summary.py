#!/usr/bin/env python3
"""Collect optional Langfuse trace diagnostics for Datus nightly runs.

Nightly pass/fail must not depend on Langfuse availability. This script writes
diagnostic artifacts and exits successfully when credentials or trace data are
missing.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "https://us.cloud.langfuse.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect nightly Langfuse trace diagnostics")
    parser.add_argument("--trace-references-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--diagnostics-json", required=True)
    parser.add_argument("--from-start-time", default="")
    parser.add_argument("--to-start-time", default="")
    parser.add_argument(
        "--base-url", default=os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or DEFAULT_BASE_URL
    )
    parser.add_argument("--public-key", default=os.getenv("LANGFUSE_PUBLIC_KEY", ""))
    parser.add_argument("--secret-key", default=os.getenv("LANGFUSE_SECRET_KEY", ""))
    parser.add_argument("--project-id", default=os.getenv("LANGFUSE_PROJECT_ID", ""))
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--slow-span-threshold-seconds", type=float, default=30.0)
    parser.add_argument("--max-pages", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    refs = load_jsonl(Path(args.trace_references_jsonl))
    from_start_time, to_start_time = default_time_bounds(args.from_start_time, args.to_start_time)
    has_credentials = bool(args.public_key and args.secret_key)

    rows: list[dict[str, Any]] = []
    for ref in refs:
        trace_id = str(ref.get("trace_id") or "")
        if not trace_id:
            rows.append(build_missing_row(ref, "missing_trace_reference"))
            continue
        if not has_credentials:
            rows.append(build_missing_row(ref, "no_credentials"))
            continue
        try:
            observations = fetch_observations(
                trace_id=trace_id,
                base_url=args.base_url,
                public_key=args.public_key,
                secret_key=args.secret_key,
                from_start_time=from_start_time,
                to_start_time=to_start_time,
                timeout=args.timeout,
                max_pages=args.max_pages,
            )
            rows.append(
                summarize_trace(
                    ref,
                    observations,
                    base_url=args.base_url,
                    project_id=args.project_id,
                    slow_span_threshold_seconds=args.slow_span_threshold_seconds,
                )
            )
        except Exception as exc:  # noqa: BLE001 - non-blocking diagnostics.
            rows.append(build_missing_row(ref, "api_error", str(exc)))

    write_jsonl(Path(args.output_jsonl), rows)
    diagnostics = build_process_diagnostics(rows)
    write_json(Path(args.diagnostics_json), diagnostics)
    print(f"Generated nightly trace diagnostics: {args.output_jsonl} ({len(rows)} rows), {args.diagnostics_json}")
    return 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return dedupe_trace_refs(rows)


def dedupe_trace_refs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the latest row per suite/nodeid/trace-id combination."""

    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        suite = str(row.get("suite") or "")
        nodeid = str(row.get("nodeid") or "")
        trace_id = str(row.get("trace_id") or "")
        key = (suite, nodeid, trace_id)
        selected[key] = row
    return [selected[key] for key in sorted(selected, key=lambda item: (item[0], item[1], item[2]))]


def default_time_bounds(from_start_time: str, to_start_time: str) -> tuple[str, str]:
    end = _parse_datetime(to_start_time) or datetime.now(UTC) + timedelta(minutes=5)
    start = _parse_datetime(from_start_time) or (end - timedelta(days=14))
    return _format_iso(start), _format_iso(end)


def fetch_observations(
    *,
    trace_id: str,
    base_url: str,
    public_key: str,
    secret_key: str,
    from_start_time: str,
    to_start_time: str,
    timeout: float,
    max_pages: int,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    cursor = ""
    for _ in range(max_pages):
        query = {
            "traceId": trace_id,
            "fromStartTime": from_start_time,
            "toStartTime": to_start_time,
            "limit": "1000",
            "fields": "core,basic,usage,metrics",
        }
        if cursor:
            query["cursor"] = cursor
        url = f"{base_url.rstrip('/')}/api/public/v2/observations?{urllib.parse.urlencode(query)}"
        request = urllib.request.Request(url, headers=_auth_headers(public_key, secret_key))
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL is configured by CI.
            payload = json.loads(response.read().decode("utf-8"))
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, list):
            observations.extend(item for item in data if isinstance(item, dict))
        meta = payload.get("meta") if isinstance(payload, dict) else None
        next_cursor = ""
        if isinstance(meta, dict):
            next_cursor = str(meta.get("cursor") or meta.get("nextCursor") or meta.get("next_cursor") or "")
        if not next_cursor:
            break
        cursor = next_cursor
    return observations


def build_missing_row(ref: dict[str, Any], status: str, error: str = "") -> dict[str, Any]:
    return {
        **_base_trace_row(ref),
        "trace_fetch_status": status,
        "observation_count": 0,
        "duration_seconds": None,
        "tool_span_count": 0,
        "generation_span_count": 0,
        "agent_span_count": 0,
        "failed_span_count": 0,
        "token_usage": {},
        "slowest_spans": [],
        "finding_type_counts": {},
        "diagnostic_findings": [],
        "error": error or None,
    }


def summarize_trace(
    ref: dict[str, Any],
    observations: list[dict[str, Any]],
    *,
    base_url: str,
    project_id: str,
    slow_span_threshold_seconds: float,
) -> dict[str, Any]:
    row = {
        **_base_trace_row(ref),
        "trace_fetch_status": "fetched",
        **summarize_observations(observations, slow_span_threshold_seconds=slow_span_threshold_seconds),
    }
    if not row.get("trace_url") and row.get("trace_id") and project_id:
        row["trace_url"] = f"{base_url.rstrip('/')}/project/{project_id}/traces/{row['trace_id']}"
    return row


def summarize_observations(
    observations: list[dict[str, Any]],
    *,
    slow_span_threshold_seconds: float = 30.0,
) -> dict[str, Any]:
    type_counts = Counter(_observation_type(item) for item in observations)
    failed_spans = [item for item in observations if _is_failed_observation(item)]
    durations = [(_observation_duration(item), item) for item in observations]
    durations = [(duration, item) for duration, item in durations if duration is not None]
    slowest = sorted(durations, key=lambda item: item[0], reverse=True)[:5]
    start_times = [_parse_datetime(_observation_start(item)) for item in observations]
    end_times = [_parse_datetime(_observation_end(item)) for item in observations]
    start_times = [item for item in start_times if item is not None]
    end_times = [item for item in end_times if item is not None]

    findings: list[dict[str, Any]] = []
    for item in failed_spans[:10]:
        findings.append(
            {
                "type": "failed_span",
                "severity": "warning",
                "name": item.get("name"),
                "span_type": _observation_type(item),
                "message": str(item.get("statusMessage") or item.get("error") or "Trace span failed"),
            }
        )
    if observations and type_counts.get("GENERATION", 0) == 0:
        findings.append(
            {
                "type": "missing_generation_span",
                "severity": "warning",
                "message": "Trace has no generation span.",
            }
        )

    return {
        "observation_count": len(observations),
        "duration_seconds": _duration_between(min(start_times), max(end_times)) if start_times and end_times else None,
        "tool_span_count": type_counts.get("TOOL", 0),
        "generation_span_count": type_counts.get("GENERATION", 0),
        "agent_span_count": type_counts.get("AGENT", 0),
        "failed_span_count": len(failed_spans),
        "token_usage": _collect_token_usage(observations),
        "slowest_spans": [_span_summary(item, duration) for duration, item in slowest],
        "finding_type_counts": dict(Counter(str(finding.get("type")) for finding in findings)),
        "diagnostic_findings": findings,
        "error": None,
    }


def build_process_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    suite_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        suite_rows[str(row.get("suite") or "unknown")].append(row)

    suite_summaries = []
    for suite_name in sorted(suite_rows):
        suite_summaries.append({"suite": suite_name, **summarize_rows(suite_rows[suite_name])})

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "summary": summarize_rows(rows),
        "suites": suite_summaries,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fetch_status_counts: Counter[str] = Counter()
    finding_type_counts: Counter[str] = Counter()
    token_usage: Counter[str] = Counter()
    durations: list[float] = []
    tool_spans = generation_spans = agent_spans = failed_spans = observations = 0
    trace_refs = 0

    for row in rows:
        status = str(row.get("trace_fetch_status") or "unknown")
        fetch_status_counts[status] += 1
        if row.get("trace_id"):
            trace_refs += 1
        observations += int(row.get("observation_count") or 0)
        tool_spans += int(row.get("tool_span_count") or 0)
        generation_spans += int(row.get("generation_span_count") or 0)
        agent_spans += int(row.get("agent_span_count") or 0)
        failed_spans += int(row.get("failed_span_count") or 0)
        duration = row.get("duration_seconds")
        if isinstance(duration, (int, float)) and not isinstance(duration, bool):
            durations.append(float(duration))
        for key, value in (row.get("finding_type_counts") or {}).items():
            finding_type_counts[str(key)] += int(value or 0)
        for key, value in (row.get("token_usage") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                token_usage[str(key)] += int(value)

    return {
        "case_count": len(rows),
        "trace_reference_count": trace_refs,
        "trace_fetch_status_counts": dict(sorted(fetch_status_counts.items())),
        "finding_type_counts": dict(sorted(finding_type_counts.items())),
        "observation_count": observations,
        "tool_span_count": tool_spans,
        "generation_span_count": generation_spans,
        "agent_span_count": agent_spans,
        "failed_span_count": failed_spans,
        "avg_duration_seconds": round(sum(durations) / len(durations), 3) if durations else None,
        "token_usage": dict(sorted(token_usage.items())),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _base_trace_row(ref: dict[str, Any]) -> dict[str, Any]:
    return {
        "suite": ref.get("suite") or "unknown",
        "nodeid": ref.get("nodeid"),
        "outcome": ref.get("outcome"),
        "pytest_duration_seconds": ref.get("duration_seconds"),
        "trace_expected": bool(ref.get("trace_expected")),
        "trace_id": ref.get("trace_id"),
        "trace_url": ref.get("trace_url"),
        "trace_provider": ref.get("trace_provider"),
        "trace_run_id": ref.get("trace_run_id"),
        "trace_span_id": ref.get("trace_span_id"),
    }


def _auth_headers(public_key: str, secret_key: str) -> dict[str, str]:
    token = base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode("ascii")
    return {"Accept": "application/json", "Authorization": f"Basic {token}"}


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _format_iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _duration_between(start: datetime, end: datetime) -> float | None:
    if end < start:
        return None
    return (end - start).total_seconds()


def _duration_from_values(start: Any, end: Any) -> float | None:
    parsed_start = _parse_datetime(start)
    parsed_end = _parse_datetime(end)
    if not parsed_start or not parsed_end:
        return None
    return _duration_between(parsed_start, parsed_end)


def _observation_type(item: dict[str, Any]) -> str:
    return str(item.get("type") or item.get("observationType") or "UNKNOWN").upper()


def _observation_start(item: dict[str, Any]) -> Any:
    return item.get("startTime") or item.get("start_time")


def _observation_end(item: dict[str, Any]) -> Any:
    return item.get("endTime") or item.get("end_time")


def _observation_duration(item: dict[str, Any]) -> float | None:
    for key in ["duration", "latency", "durationSeconds", "duration_seconds"]:
        value = item.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return _duration_from_values(_observation_start(item), _observation_end(item))


def _is_failed_observation(item: dict[str, Any]) -> bool:
    level = str(item.get("level") or item.get("status") or "").upper()
    if level in {"ERROR", "FAILED"}:
        return True
    return bool(item.get("statusMessage") or item.get("error"))


def _span_summary(item: dict[str, Any], duration: float | None) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "name": item.get("name"),
        "span_type": _observation_type(item),
        "duration_seconds": duration,
    }


def _collect_token_usage(observations: list[dict[str, Any]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for item in observations:
        usage_payload: dict[str, Any] | None = None
        for payload_key in ["usageDetails", "usage_details", "usage", "usageMetadata"]:
            payload = item.get(payload_key)
            if isinstance(payload, dict):
                usage_payload = payload
                break
        if usage_payload is not None:
            _merge_numeric_usage(totals, usage_payload)
        else:
            for key in ["promptTokens", "completionTokens", "totalTokens", "inputTokens", "outputTokens"]:
                value = item.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    totals[_normalize_usage_key(key)] += int(value)
    return dict(totals)


def _merge_numeric_usage(target: Counter[str], payload: dict[str, Any]) -> None:
    for key, value in payload.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            target[_normalize_usage_key(str(key))] += int(value)


def _normalize_usage_key(key: str) -> str:
    aliases = {
        "promptTokens": "input_tokens",
        "inputTokens": "input_tokens",
        "completionTokens": "output_tokens",
        "outputTokens": "output_tokens",
        "totalTokens": "total_tokens",
    }
    return aliases.get(key, key)


if __name__ == "__main__":
    raise SystemExit(main())
