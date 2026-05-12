#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Run deterministic suites for GitHub merge queue."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "ci"
DEFAULT_REPORT = OUT_DIR / "merge-queue-results.json"
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("MERGE_QUEUE_TEST_TIMEOUT", "1800"))

FULL_UNIT_TARGETS = ("tests/unit_tests/",)
FULL_UNIT_MARK_EXPR = "not nightly and not quarantine"
ACCEPTANCE_MARK_EXPR = "acceptance"


def log(message: str) -> None:
    print(f"[merge-queue] {message}", flush=True)


def load_pr_acceptance_targets() -> list[str]:
    """Reuse the PR harness target list so PR and merge-queue coverage stay aligned."""
    module_path = REPO_ROOT / "ci" / "run-pr-tests.py"
    module_spec = importlib.util.spec_from_file_location("_run_pr_tests_for_merge_queue", module_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Unable to load PR harness targets from {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    targets = getattr(module, "PR_ACCEPTANCE_TARGETS", None)
    if not isinstance(targets, list) or not all(isinstance(target, str) for target in targets):
        raise RuntimeError("ci/run-pr-tests.py PR_ACCEPTANCE_TARGETS must be a list of strings")
    return targets


def acceptance_integration_targets(targets: Sequence[str]) -> list[str]:
    return [target for target in targets if target.startswith("tests/integration/")]


def existing_paths(paths: Sequence[str]) -> list[str]:
    existing: list[str] = []
    for item in paths:
        path = REPO_ROOT / item
        if path.exists():
            existing.append(item)
        else:
            log(f"Skipping missing path: {item}")
    return existing


def build_pytest_command(
    targets: Sequence[str],
    *,
    mark_expr: str,
    junit_xml: Path,
    extra_args: Sequence[str] = (),
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        "-m",
        mark_expr,
        "--tb=short",
        "--showlocals",
        "--disable-warnings",
        f"--junitxml={junit_xml}",
        *extra_args,
    ]


def run_command(command: Sequence[str], *, suite_name: str, timeout: int) -> int:
    log(f"Running {suite_name}: {' '.join(command)}")
    try:
        completed = subprocess.run(list(command), cwd=REPO_ROOT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"{suite_name} timed out after {timeout}s")
        return 1
    log(f"{suite_name} exited with code {completed.returncode}")
    return completed.returncode


def suite_definitions() -> dict[str, dict[str, Any]]:
    acceptance_targets = acceptance_integration_targets(load_pr_acceptance_targets())
    return {
        "full-unit": {
            "description": "Full deterministic unit suite excluding nightly and quarantined tests.",
            "targets": list(FULL_UNIT_TARGETS),
            "mark_expr": FULL_UNIT_MARK_EXPR,
            "junit_xml": OUT_DIR / "test-results-merge-full-unit.xml",
            "extra_args": ["--timeout=300", "--dist=loadscope", "-n", "auto"],
        },
        "acceptance-integration": {
            "description": "Deterministic acceptance integration coverage reused from the PR harness.",
            "targets": acceptance_targets,
            "mark_expr": ACCEPTANCE_MARK_EXPR,
            "junit_xml": OUT_DIR / "test-results-merge-acceptance.xml",
            "extra_args": ["--timeout=300"],
        },
    }


def run_suite(name: str, suite: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    targets = existing_paths(suite["targets"])
    if not targets:
        log(f"{name} has no existing targets")
        return {"suite": name, "exit_code": 1, "targets": []}

    command = build_pytest_command(
        targets,
        mark_expr=suite["mark_expr"],
        junit_xml=suite["junit_xml"],
        extra_args=suite["extra_args"],
    )
    exit_code = run_command(command, suite_name=name, timeout=timeout)
    return {
        "suite": name,
        "exit_code": exit_code,
        "targets": targets,
        "junit_xml": str(suite["junit_xml"].relative_to(REPO_ROOT)),
    }


def write_report(results: Sequence[dict[str, Any]], path: Path | None = None) -> None:
    report_path = path or DEFAULT_REPORT
    payload = {
        "status": "success" if all(result["exit_code"] == 0 for result in results) else "failure",
        "results": list(results),
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"Wrote report to {report_path.relative_to(REPO_ROOT)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic merge queue suites")
    parser.add_argument(
        "--suite",
        action="append",
        choices=("full-unit", "acceptance-integration"),
        help="Run one suite. Defaults to all merge queue suites.",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="Per-suite timeout in seconds")
    args = parser.parse_args(argv)

    suites = suite_definitions()
    selected_names = args.suite or list(suites)
    results = [run_suite(name, suites[name], timeout=args.timeout) for name in selected_names]
    write_report(results)
    return 0 if all(result["exit_code"] == 0 for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
