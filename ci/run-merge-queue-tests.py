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
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "ci"
DEFAULT_REPORT = OUT_DIR / "merge-queue-results.json"
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("MERGE_QUEUE_TEST_TIMEOUT", "600"))
PYTEST_BASETEMP_ENV = "DATUS_CI_PYTEST_BASETEMP"

ACCEPTANCE_MARK_EXPR = "acceptance and not quarantine"
DEFAULT_PR_HARNESS_MARK_EXPR = "(acceptance or component or llm_harness) and not quarantine"


def log(message: str) -> None:
    print(f"[merge-queue] {message}", flush=True)


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return slug or "default"


def default_pytest_basetemp() -> Path:
    configured = os.environ.get(PYTEST_BASETEMP_ENV)
    if configured:
        return Path(configured)

    temp_root = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir())
    run_id = slugify(os.environ.get("GITHUB_RUN_ID", "local"))
    run_attempt = slugify(os.environ.get("GITHUB_RUN_ATTEMPT", "0"))
    return temp_root / f"datus-agent-pytest-{run_id}-{run_attempt}-{os.getpid()}"


PYTEST_BASETEMP = default_pytest_basetemp()


def suite_pytest_basetemp(suite_name: str) -> Path:
    return PYTEST_BASETEMP / slugify(suite_name)


def prepare_suite_pytest_basetemp(basetemp: Path) -> None:
    basetemp.parent.mkdir(parents=True, exist_ok=True)


def path_is_under(path: Path, parent: Path) -> bool:
    if not str(parent):
        return False

    path_abs = path.resolve(strict=False)
    parent_abs = parent.resolve(strict=False)
    if path_abs == parent_abs:
        return False

    try:
        path_abs.relative_to(parent_abs)
    except ValueError:
        return False
    return True


def is_safe_pytest_basetemp(path: Path) -> bool:
    if not str(path):
        return False

    path_abs = path.resolve(strict=False)
    unsafe_roots = {
        Path("/").resolve(strict=False),
        REPO_ROOT.resolve(strict=False),
        OUT_DIR.resolve(strict=False),
        Path.home().resolve(strict=False),
    }
    if path_abs in unsafe_roots:
        return False

    allowed_roots = [
        Path(os.environ["RUNNER_TEMP"]) if os.environ.get("RUNNER_TEMP") else None,
        Path(tempfile.gettempdir()),
        Path("/tmp"),
    ]
    return any(root is not None and path_is_under(path_abs, root) for root in allowed_roots)


def cleanup_pytest_basetemp() -> None:
    if not PYTEST_BASETEMP.exists():
        return

    if not is_safe_pytest_basetemp(PYTEST_BASETEMP):
        log(f"Refusing to remove unsafe pytest basetemp: {PYTEST_BASETEMP}")
        return

    shutil.rmtree(PYTEST_BASETEMP, ignore_errors=True)
    log(f"Cleaned pytest basetemp: {PYTEST_BASETEMP}")


def load_pr_harness_config() -> tuple[list[str], str]:
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
    mark_expr = getattr(module, "PR_HARNESS_MARK_EXPR", DEFAULT_PR_HARNESS_MARK_EXPR)
    if not isinstance(mark_expr, str) or not mark_expr:
        raise RuntimeError("ci/run-pr-tests.py PR_HARNESS_MARK_EXPR must be a non-empty string")
    return targets, mark_expr


def acceptance_unit_targets(targets: Sequence[str]) -> list[str]:
    return [target for target in targets if target.startswith("tests/unit_tests/")]


def acceptance_integration_targets(targets: Sequence[str]) -> list[str]:
    return [target for target in targets if target.startswith("tests/integration/")]


def missing_paths(paths: Sequence[str]) -> list[str]:
    """Return targets that do not exist in the checkout.

    A stale entry must fail loudly: silently skipping it would drop merge-queue
    coverage without any signal in the results.
    """
    return [item for item in paths if not (REPO_ROOT / item).exists()]


def build_pytest_command(
    targets: Sequence[str],
    *,
    mark_expr: str,
    junit_xml: Path,
    extra_args: Sequence[str] = (),
    basetemp: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
    ]
    if basetemp is not None:
        command.append(f"--basetemp={basetemp}")

    command.extend(
        [
            "-m",
            mark_expr,
            "--tb=short",
            "--showlocals",
            "--disable-warnings",
            f"--junitxml={junit_xml}",
            *extra_args,
        ]
    )
    return command


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
    pr_acceptance_targets, pr_harness_mark_expr = load_pr_harness_config()
    unit_targets = acceptance_unit_targets(pr_acceptance_targets)
    integration_targets = acceptance_integration_targets(pr_acceptance_targets)
    return {
        "acceptance-unit": {
            "description": "Deterministic unit harness targets reused from the PR acceptance list.",
            "targets": unit_targets,
            "mark_expr": pr_harness_mark_expr,
            "junit_xml": OUT_DIR / "test-results-merge-acceptance-unit.xml",
            "extra_args": ["--timeout=120", "--dist=loadscope", "-n", "auto"],
        },
        "acceptance-integration": {
            "description": "Deterministic acceptance integration coverage reused from the PR harness.",
            "targets": integration_targets,
            "mark_expr": ACCEPTANCE_MARK_EXPR,
            "junit_xml": OUT_DIR / "test-results-merge-acceptance.xml",
            "extra_args": ["--timeout=120"],
        },
    }


def run_suite(name: str, suite: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    targets = list(suite["targets"])
    stale = missing_paths(targets)
    if stale:
        for item in stale:
            log(f"{name} target missing from checkout: {item}")
        log("Update PR_ACCEPTANCE_TARGETS in ci/run-pr-tests.py to match the tree")
        return {"suite": name, "exit_code": 1, "targets": [], "missing": stale}
    if not targets:
        log(f"{name} has no existing targets")
        return {"suite": name, "exit_code": 1, "targets": []}

    basetemp = suite_pytest_basetemp(name)
    prepare_suite_pytest_basetemp(basetemp)
    command = build_pytest_command(
        targets,
        mark_expr=suite["mark_expr"],
        junit_xml=suite["junit_xml"],
        extra_args=suite["extra_args"],
        basetemp=basetemp,
    )
    log(f"{name} pytest basetemp: {basetemp}")
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
        choices=("acceptance-unit", "acceptance-integration"),
        help="Run one suite. Defaults to all merge queue suites.",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="Per-suite timeout in seconds")
    args = parser.parse_args(argv)

    try:
        suites = suite_definitions()
        selected_names = args.suite or list(suites)
        results = [run_suite(name, suites[name], timeout=args.timeout) for name in selected_names]
        write_report(results)
        return 0 if all(result["exit_code"] == 0 for result in results) else 1
    finally:
        cleanup_pytest_basetemp()


if __name__ == "__main__":
    sys.exit(main())
