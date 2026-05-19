#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Build a PR comment that summarizes a failed merge queue run."""

from __future__ import annotations

import argparse
import json
import os
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

COMMENT_MARKER = "<!-- datus-merge-queue-failure-comment -->"
DEFAULT_MAX_FAILURES = 5
DEFAULT_MAX_DETAIL_CHARS = 1800


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _truncate(value: str, max_chars: int = DEFAULT_MAX_DETAIL_CHARS) -> str:
    value = value.strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 40].rstrip() + "\n... truncated ..."


def _fence(value: str) -> str:
    sanitized = value.replace("```", "` ` `")
    return f"```text\n{sanitized}\n```"


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_suite_results(artifacts_dir: Path) -> list[dict[str, Any]]:
    suite_results: list[dict[str, Any]] = []
    for path in sorted(artifacts_dir.rglob("merge-queue-results.json")):
        data = _load_json(path)
        if not data:
            continue
        results = data.get("results", [])
        if isinstance(results, list):
            suite_results.extend(result for result in results if isinstance(result, dict))
    return suite_results


def load_junit_failures(artifacts_dir: Path, *, max_failures: int = DEFAULT_MAX_FAILURES) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for path in sorted(artifacts_dir.rglob("test-results-merge-*.xml")):
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError):
            continue

        for testcase in root.iter():
            if _local_name(testcase.tag) != "testcase":
                continue
            fault = None
            for child in testcase:
                if _local_name(child.tag) in {"failure", "error"}:
                    fault = child
                    break
            if fault is None:
                continue

            failures.append(
                {
                    "file": path.name,
                    "classname": testcase.attrib.get("classname", ""),
                    "name": testcase.attrib.get("name", ""),
                    "message": fault.attrib.get("message", ""),
                    "details": fault.text or "",
                    "kind": _local_name(fault.tag),
                }
            )
            if len(failures) >= max_failures:
                return failures
    return failures


def failed_jobs_from_needs(needs_json: str | None) -> str:
    if not needs_json:
        return "unknown"
    try:
        needs = json.loads(needs_json)
    except json.JSONDecodeError:
        return "unknown"
    if not isinstance(needs, dict):
        return "unknown"
    failed = [name for name, job in needs.items() if isinstance(job, dict) and job.get("result") == "failure"]
    return ", ".join(failed) if failed else "unknown"


def _format_suite_results(results: Sequence[dict[str, Any]]) -> list[str]:
    if not results:
        return ["- No merge queue suite summary artifact was found."]

    lines: list[str] = []
    for result in results:
        suite = result.get("suite", "unknown")
        exit_code = result.get("exit_code", "unknown")
        targets = result.get("targets", [])
        target_count = len(targets) if isinstance(targets, list) else "unknown"
        status = "failed" if exit_code != 0 else "passed"
        lines.append(f"- `{suite}`: {status}, exit code `{exit_code}`, targets `{target_count}`")
    return lines


def _format_failures(failures: Sequence[dict[str, str]]) -> list[str]:
    if not failures:
        return ["No JUnit failure details were found in the merge queue artifacts. Use the linked run logs."]

    lines: list[str] = []
    for index, failure in enumerate(failures, 1):
        classname = failure["classname"]
        name = failure["name"]
        test_id = f"{classname}::{name}" if classname else name
        lines.append(f"{index}. `{test_id}`")
        if failure["message"]:
            lines.append(f"   - {failure['kind']}: `{_truncate(failure['message'], 240)}`")
        if failure["details"]:
            lines.append("")
            lines.append(_fence(_truncate(failure["details"])))
    return lines


def build_comment(
    *,
    artifacts_dir: Path,
    repository: str,
    workflow: str,
    event_name: str,
    ref: str,
    sha: str,
    run_number: str,
    run_url: str,
    failed_jobs: str,
    max_failures: int = DEFAULT_MAX_FAILURES,
) -> str:
    suite_results = load_suite_results(artifacts_dir)
    failures = load_junit_failures(artifacts_dir, max_failures=max_failures)

    lines = [
        COMMENT_MARKER,
        "## Merge Queue Failure",
        "",
        "This PR was removed from the merge queue because the real `merge_group` run failed. "
        "PR head checks can still look green because this ran on GitHub's synthetic queue ref.",
        "",
        f"- Repository: `{repository}`",
        f"- Workflow: `{workflow}`",
        f"- Event: `{event_name}`",
        f"- Queue ref: `{ref}`",
        f"- Queue SHA: `{sha}`",
        f"- Failed jobs: `{failed_jobs}`",
        f"- Run: [#{run_number}]({run_url})",
        "",
        "### Suite Summary",
        *_format_suite_results(suite_results),
        "",
        "### Failure Details",
        *_format_failures(failures),
    ]
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("ci/merge-queue-artifacts"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", "unknown"))
    parser.add_argument("--workflow", default=os.environ.get("GITHUB_WORKFLOW", "unknown"))
    parser.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", "unknown"))
    parser.add_argument("--ref", default=os.environ.get("GITHUB_REF", "unknown"))
    parser.add_argument("--sha", default=os.environ.get("GITHUB_SHA", "unknown"))
    parser.add_argument("--run-number", default=os.environ.get("GITHUB_RUN_NUMBER", "unknown"))
    parser.add_argument(
        "--run-url",
        default=(
            f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
            f"{os.environ.get('GITHUB_REPOSITORY', 'unknown')}/actions/runs/"
            f"{os.environ.get('GITHUB_RUN_ID', 'unknown')}"
        ),
    )
    parser.add_argument("--failed-jobs", default=failed_jobs_from_needs(os.environ.get("NEEDS_JSON")))
    parser.add_argument("--max-failures", type=int, default=DEFAULT_MAX_FAILURES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    comment = build_comment(
        artifacts_dir=args.artifacts_dir,
        repository=args.repository,
        workflow=args.workflow,
        event_name=args.event,
        ref=args.ref,
        sha=args.sha,
        run_number=args.run_number,
        run_url=args.run_url,
        failed_jobs=args.failed_jobs,
        max_failures=args.max_failures,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(textwrap.dedent(comment), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
