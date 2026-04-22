#!/usr/bin/env python3
"""Run PR acceptance plus impacted unit tests with coverage and diff reporting.

Usage:
    python3 ci/run-pr-tests.py [base_ref]

Outputs (written to GITHUB_OUTPUT if available, otherwise stdout):
    overall       - Overall line coverage percentage
    diff          - Diff coverage percentage for the PR's changed lines
    test_total    - Total test count
    test_passed   - Passed test count
    test_failed   - Failed test count (failures + errors)
    test_skipped  - Skipped test count
    test_outcome  - "success" or "failure"

Generated files (all written to ci/ directory):
    ci/coverage.xml             - Cobertura coverage report
    ci/htmlcov/                 - Full HTML coverage report
    ci/test-results.xml         - Combined JUnit test results
    ci/test-results-*.xml       - Per-suite JUnit results for acceptance / impacted unit suites
    ci/diff-cover.json          - Diff coverage data
    ci/diff-cover-report.md     - Diff coverage markdown report
    ci/test-report.md           - Test failure markdown report
    ci/pytest-coverage.txt      - Full pytest output
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import xml.etree.ElementTree as XMLTree
from typing import Any, TextIO

import defusedxml.ElementTree as ET

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JUNIT_XML = os.path.join(OUT_DIR, "test-results.xml")
DEFAULT_COVERAGE_XML = os.path.join(OUT_DIR, "coverage.xml")
DEFAULT_COVERAGE_HTML = os.path.join(OUT_DIR, "htmlcov")
DEFAULT_COVERAGE_DB = os.path.join(OUT_DIR, ".coverage")
DEFAULT_PYTEST_LOG = os.path.join(OUT_DIR, "pytest-coverage.txt")
PR_ACCEPTANCE_TARGETS = [
    "tests/unit_tests/agent/node/test_chat_agentic_node.py",
    "tests/unit_tests/agent/node/test_gen_sql_agentic_node.py",
    "tests/unit_tests/agent/node/test_compare_agentic_node.py",
    "tests/unit_tests/agent/node/test_explore_agentic_node.py",
    "tests/unit_tests/agent/node/test_feedback_agentic_node.py",
    "tests/unit_tests/agent/node/test_gen_report_agentic_node.py",
    "tests/unit_tests/agent/node/test_gen_ext_knowledge_agentic_node.py",
    "tests/unit_tests/agent/node/test_gen_metrics_agentic_node.py",
    "tests/unit_tests/agent/node/test_gen_semantic_model_agentic_node.py",
    "tests/unit_tests/agent/node/test_gen_table_agentic_node.py",
    "tests/unit_tests/agent/node/test_sql_summary_agentic_node.py",
    "tests/unit_tests/agent/node/test_skill_creator_agentic_node.py",
    "tests/unit_tests/agent/node/test_gen_job_agentic_node.py",
    "tests/unit_tests/agent/node/test_migration_agentic_node.py",
    "tests/integration/api/test_api.py",
    "tests/integration/cli/test_cli_commands.py",
    "tests/integration/cli/test_cli_textual.py",
    "tests/integration/storage/test_storage_layout.py",
    "tests/integration/storage/test_doc_search.py",
    "tests/integration/tools/test_func_tools_db.py",
    "tests/integration/tools/db_tools/test_connector_duckdb.py",
]
IMPACTED_TEST_MAPPING = [
    ("datus/agent/", "tests/unit_tests/agent/"),
    ("datus/api/", "tests/unit_tests/api/"),
    ("datus/cli/", "tests/unit_tests/cli/"),
    ("datus/configuration/", "tests/unit_tests/configuration/"),
    ("datus/models/", "tests/unit_tests/models/"),
    ("datus/storage/", "tests/unit_tests/storage/"),
    ("datus/tools/", "tests/unit_tests/tools/"),
    ("datus/utils/", "tests/unit_tests/utils/"),
    ("ci/", "tests/unit_tests/ci/"),
    # Catch top-level datus/ changes (for example __init__.py or future shared modules)
    # with the full unit suite as a safety net when no narrower prefix applies.
    ("datus/", "tests/unit_tests/"),
]


def log(msg: str) -> None:
    print(f"[ci] {msg}", flush=True)


TEST_CMD_TIMEOUT = int(os.environ.get("TEST_CMD_TIMEOUT", "1800"))
GIT_CMD_TIMEOUT = int(os.environ.get("GIT_CMD_TIMEOUT", "60"))
DIFF_COVER_TIMEOUT = int(os.environ.get("DIFF_COVER_TIMEOUT", "300"))
_COMPARE_BRANCH_CACHE: dict[str, str | None] = {}
READER_JOIN_TIMEOUT_SECONDS = 5


def _run_cmd(
    cmd: list[str],
    timeout: int,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str] | None:
    """Run a command with timeout. Returns CompletedProcess or None on timeout."""
    try:
        return subprocess.run(cmd, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired:
        log(f"Command timed out after {timeout}s: {' '.join(cmd)}")
        return None


def _stream_process_output(proc: subprocess.Popen[str], log_file: TextIO) -> None:
    """Stream process output to stdout and the log file until EOF."""
    if not proc.stdout:
        return
    for line in proc.stdout:
        sys.stdout.write(line)
        log_file.write(line)


def _normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _dedupe_preserve(items: list[str]) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _remove_output_path(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        os.remove(path)


def _reset_report_outputs() -> None:
    for path in [
        DEFAULT_COVERAGE_XML,
        DEFAULT_COVERAGE_HTML,
        DEFAULT_COVERAGE_DB,
        DEFAULT_JUNIT_XML,
        os.path.join(OUT_DIR, "test-results-acceptance.xml"),
        os.path.join(OUT_DIR, "test-results-impacted-unit.xml"),
        os.path.join(OUT_DIR, "diff-cover.json"),
        os.path.join(OUT_DIR, "diff-cover-report.md"),
        os.path.join(OUT_DIR, "test-report.md"),
        DEFAULT_PYTEST_LOG,
    ]:
        _remove_output_path(path)


def _build_pytest_command(
    targets: list[str],
    junit_xml: str,
    *,
    mark_expr: str | None = None,
    append: bool = False,
    emit_reports: bool = True,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
    ]
    if mark_expr:
        cmd.extend(["-m", mark_expr])

    cmd.append("--cov=datus")
    if append:
        cmd.append("--cov-append")

    if emit_reports:
        cmd.extend(
            [
                f"--cov-report=xml:{DEFAULT_COVERAGE_XML}",
                f"--cov-report=html:{DEFAULT_COVERAGE_HTML}",
                "--cov-report=term-missing",
            ]
        )
    else:
        cmd.append("--cov-report=")

    cmd.extend(
        [
            f"--junitxml={junit_xml}",
            "-q",
            "--disable-warnings",
            "--log-level=CRITICAL",
            "--log-cli-level=CRITICAL",
            "--showlocals",
        ]
    )
    return cmd


def _run_pytest_suite(
    targets: list[str],
    junit_xml: str,
    log_file: TextIO,
    *,
    suite_name: str,
    mark_expr: str | None = None,
    append: bool = False,
    emit_reports: bool = True,
) -> int:
    """Run one pytest suite and stream logs to stdout and the CI log file."""
    cmd = _build_pytest_command(
        targets,
        junit_xml,
        mark_expr=mark_expr,
        append=append,
        emit_reports=emit_reports,
    )
    log(f"Running {suite_name}: {' '.join(cmd)}")

    env = os.environ.copy()
    env["COVERAGE_FILE"] = DEFAULT_COVERAGE_DB

    banner = f"\n=== {suite_name.upper()} ===\n"
    sys.stdout.write(banner)
    log_file.write(banner)
    log_file.flush()

    popen_kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "env": env,
    }
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)
    reader = threading.Thread(target=_stream_process_output, args=(proc, log_file), daemon=True)
    reader.start()

    try:
        exit_code = proc.wait(timeout=TEST_CMD_TIMEOUT)
    except subprocess.TimeoutExpired:
        log(f"{suite_name} timed out after {TEST_CMD_TIMEOUT}s, killing process")
        if os.name != "nt":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        proc.wait()
        timeout_msg = f"\n[ci] TIMEOUT: {suite_name} killed after {TEST_CMD_TIMEOUT}s\n"
        sys.stdout.write(timeout_msg)
        log_file.write(timeout_msg)
        exit_code = 1
    finally:
        reader.join(timeout=READER_JOIN_TIMEOUT_SECONDS)
        if reader.is_alive():
            log(f"{suite_name} output reader did not exit within {READER_JOIN_TIMEOUT_SECONDS}s")

    log(f"{suite_name} exited with code {exit_code}")
    return exit_code


def _normalize_suite_exit_code(
    exit_code: int,
    *,
    suite_name: str,
    allow_empty_collection: bool = False,
) -> int:
    if exit_code == 5 and allow_empty_collection:
        log(f"{suite_name} collected no tests (pytest rc=5); treating as success")
        return 0
    return exit_code


def select_impacted_unit_tests(changed_files: list[str]) -> list[str]:
    """Map changed source files to the unit-test paths that should run."""
    impacted: list[str] = []
    for path in changed_files:
        normalized = _normalize_path(path)
        if not normalized:
            continue

        if normalized.startswith("tests/unit_tests/"):
            if normalized.endswith(".py"):
                impacted.append(normalized)
            else:
                parent = normalized.rstrip("/").rsplit("/", 1)[0]
                impacted.append(f"{parent}/" if parent else "tests/unit_tests/")
            continue

        for prefix, test_target in IMPACTED_TEST_MAPPING:
            if normalized.startswith(prefix):
                impacted.append(test_target)
                break

    return _dedupe_preserve(impacted)


def find_compare_branch(base_ref: str) -> str | None:
    """Determine the compare branch for diff-cover."""
    if base_ref in _COMPARE_BRANCH_CACHE:
        return _COMPARE_BRANCH_CACHE[base_ref]

    if base_ref:
        log(f"Using explicit base_ref: origin/{base_ref}")
        resolved_ref = f"origin/{base_ref}"
        _COMPARE_BRANCH_CACHE[base_ref] = resolved_ref
        return resolved_ref

    log("No base_ref provided, auto-detecting compare branch...")

    current = _run_cmd(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        GIT_CMD_TIMEOUT,
        capture_output=True,
        text=True,
    )
    current_branch = current.stdout.strip() if current and current.returncode == 0 else ""
    log(f"Current branch: {current_branch}")

    result = _run_cmd(
        ["git", "branch", "-r", "--format=%(refname:short)"],
        GIT_CMD_TIMEOUT,
        capture_output=True,
        text=True,
    )
    if not result or result.returncode != 0:
        log("Failed to list remote branches")
        _COMPARE_BRANCH_CACHE[base_ref] = None
        return None

    branches = [
        b.strip()
        for b in result.stdout.splitlines()
        if b.strip() and not b.strip().endswith("/HEAD") and b.strip() != f"origin/{current_branch}"
    ]
    log(f"Candidate remote branches: {len(branches)}")

    head = _run_cmd(
        ["git", "rev-parse", "HEAD"],
        GIT_CMD_TIMEOUT,
        capture_output=True,
        text=True,
    )
    head_commit = head.stdout.strip() if head and head.returncode == 0 else ""
    log(f"HEAD commit: {head_commit[:12]}")

    best_commit = None
    best_branch = None
    best_timestamp = -1
    skipped_same_as_head = 0
    for branch in branches:
        mb = _run_cmd(
            ["git", "merge-base", "HEAD", branch],
            GIT_CMD_TIMEOUT,
            capture_output=True,
            text=True,
        )
        if not mb or mb.returncode != 0:
            continue
        commit = mb.stdout.strip()
        if commit == head_commit:
            skipped_same_as_head += 1
            continue
        ts = _run_cmd(
            ["git", "log", "-1", "--format=%ct", commit],
            GIT_CMD_TIMEOUT,
            capture_output=True,
            text=True,
        )
        if ts and ts.returncode == 0:
            timestamp = int(ts.stdout.strip())
            if timestamp > best_timestamp:
                best_timestamp = timestamp
                best_commit = commit
                best_branch = branch

    log(f"Skipped {skipped_same_as_head} branches (merge-base == HEAD)")
    if best_commit:
        count = _run_cmd(
            ["git", "rev-list", "--count", f"{best_commit}..HEAD"],
            GIT_CMD_TIMEOUT,
            capture_output=True,
            text=True,
        )
        commits_ahead = count.stdout.strip() if count and count.returncode == 0 else "?"
        log(f"Selected merge-base: {best_commit[:12]} (branch: {best_branch}, {commits_ahead} commits ahead)")
    else:
        log("No suitable merge-base found")

    _COMPARE_BRANCH_CACHE[base_ref] = best_commit
    return best_commit


def list_changed_files(base_ref: str) -> list[str]:
    """Return repository paths changed against the PR base."""
    compare_ref = find_compare_branch(base_ref)
    if not compare_ref:
        log("No compare branch available; skipping impacted unit-test selection")
        return []

    proc = _run_cmd(
        ["git", "diff", "--name-only", f"{compare_ref}...HEAD"],
        GIT_CMD_TIMEOUT,
        capture_output=True,
        text=True,
    )
    if not proc or proc.returncode != 0:
        err = proc.stderr.strip() if proc else "timed out"
        log(f"Failed to enumerate changed files against {compare_ref}: {err}")
        return []

    changed = [_normalize_path(line.strip()) for line in proc.stdout.splitlines() if line.strip()]
    log(f"Detected {len(changed)} changed files against {compare_ref}")
    return changed


def resolve_impacted_unit_tests(base_ref: str) -> list[str]:
    changed_files = list_changed_files(base_ref)
    impacted = select_impacted_unit_tests(changed_files)
    if impacted:
        log(f"Impacted unit-test targets: {', '.join(impacted)}")
    else:
        log("No impacted unit-test targets selected")
    return impacted


def run_tests(base_ref: str = "") -> tuple[int, list[str]]:
    """Run PR acceptance plus impacted unit tests and return the exit code and JUnit XML paths."""
    _reset_report_outputs()

    impacted_targets = resolve_impacted_unit_tests(base_ref)
    junit_xml_paths: list[str] = []
    exit_codes: list[int] = []

    with open(DEFAULT_PYTEST_LOG, "w", encoding="utf-8") as log_file:
        acceptance_xml = os.path.join(OUT_DIR, "test-results-acceptance.xml")
        acceptance_rc = _run_pytest_suite(
            PR_ACCEPTANCE_TARGETS,
            acceptance_xml,
            log_file,
            suite_name="acceptance",
            mark_expr="acceptance",
            emit_reports=not impacted_targets,
        )
        exit_codes.append(acceptance_rc)
        junit_xml_paths.append(acceptance_xml)

        if impacted_targets:
            impacted_xml = os.path.join(OUT_DIR, "test-results-impacted-unit.xml")
            impacted_rc = _run_pytest_suite(
                impacted_targets,
                impacted_xml,
                log_file,
                suite_name="impacted unit tests",
                mark_expr="not acceptance and not nightly",
                append=True,
                emit_reports=True,
            )
            exit_codes.append(
                _normalize_suite_exit_code(
                    impacted_rc,
                    suite_name="impacted unit tests",
                    allow_empty_collection=True,
                )
            )
            junit_xml_paths.append(impacted_xml)

    merge_junit_results(junit_xml_paths)

    if os.path.exists(DEFAULT_COVERAGE_DB):
        os.remove(DEFAULT_COVERAGE_DB)

    exit_code = 0 if all(code == 0 for code in exit_codes) else 1
    log(f"Overall pytest exit code: {exit_code}")
    return exit_code, junit_xml_paths


def merge_junit_results(junit_xml_paths: list[str], output_path: str | None = None) -> str:
    """Merge one or more JUnit XML files into ci/test-results.xml."""
    if output_path is None:
        output_path = DEFAULT_JUNIT_XML

    root = XMLTree.Element("testsuites")
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    total_time = 0.0
    suite_count = 0

    for junit_xml_path in junit_xml_paths:
        if not junit_xml_path or not os.path.exists(junit_xml_path):
            log(f"Skipping missing JUnit XML during merge: {junit_xml_path}")
            continue

        try:
            tree = ET.parse(junit_xml_path)
            source_root = tree.getroot()
        except Exception as e:
            log(f"Failed to parse {junit_xml_path} during merge: {e}")
            continue

        suites = source_root.findall("testsuite") if source_root.tag == "testsuites" else [source_root]
        for suite in suites:
            root.append(copy.deepcopy(suite))
            suite_count += 1
            for attr in totals:
                totals[attr] += int(suite.attrib.get(attr, 0))
            total_time += float(suite.attrib.get("time", 0) or 0)

    for attr, value in totals.items():
        root.set(attr, str(value))
    root.set("time", f"{total_time:.3f}")

    XMLTree.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    log(f"Wrote merged JUnit report to {output_path} ({suite_count} suites)")
    return output_path


def parse_test_results(junit_xml_paths: str | list[str] | None = None) -> dict[str, Any]:
    """Parse one or more JUnit XML files to extract test counts and failures."""
    if junit_xml_paths is None:
        junit_xml_paths = [DEFAULT_JUNIT_XML]
    elif isinstance(junit_xml_paths, str):
        junit_xml_paths = [junit_xml_paths]

    total = passed = failed = errors = skipped = 0
    failures: list[dict[str, str]] = []
    parsed_any = False

    for junit_xml_path in junit_xml_paths:
        if not os.path.exists(junit_xml_path):
            log(f"JUnit XML not found, skipping parse: {junit_xml_path}")
            continue

        try:
            tree = ET.parse(junit_xml_path)
            root = tree.getroot()
        except Exception as e:
            log(f"Failed to parse {junit_xml_path}: {e}")
            continue

        parsed_any = True
        suites = root.findall("testsuite") if root.tag == "testsuites" else [root]

        for suite in suites:
            total += int(suite.attrib.get("tests", 0))
            errors += int(suite.attrib.get("errors", 0))
            failed += int(suite.attrib.get("failures", 0))
            skipped += int(suite.attrib.get("skipped", 0))

            for testcase in suite.findall("testcase"):
                failure = testcase.find("failure")
                error = testcase.find("error")
                fault = failure if failure is not None else error
                if fault is not None:
                    failures.append(
                        {
                            "name": testcase.attrib.get("name", "unknown"),
                            "classname": testcase.attrib.get("classname", ""),
                            "message": fault.attrib.get("message", ""),
                            "text": (fault.text or "").strip(),
                        }
                    )

    if parsed_any:
        passed = total - failed - errors - skipped
        log(f"Test results: {passed} passed, {failed} failed, {errors} errors, {skipped} skipped (total: {total})")
    else:
        log("No JUnit XML results were parsed")

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "failures": failures,
    }


def write_test_report(test_results: dict[str, Any], output_path: str | None = None) -> str:
    """Write a markdown report of test failures."""
    if output_path is None:
        output_path = os.path.join(OUT_DIR, "test-report.md")
    lines = []
    failures = test_results["failures"]
    total = test_results["total"]
    passed = test_results["passed"]
    failed = test_results["failed"] + test_results["errors"]
    skipped = test_results["skipped"]

    lines.append(f"**{passed}/{total}** tests passed")
    if skipped:
        lines.append(f", {skipped} skipped")
    if failed:
        lines.append(f", **{failed} failed**")
    lines.append("\n")

    if failures:
        lines.append("\n### Failed Tests\n\n")
        for i, failure in enumerate(failures, 1):
            test_id = f"{failure['classname']}::{failure['name']}" if failure["classname"] else failure["name"]
            lines.append(f"{i}. `{test_id}`\n")

        lines.append("\n### Failure Details\n\n")
        for failure in failures:
            test_id = f"{failure['classname']}::{failure['name']}" if failure["classname"] else failure["name"]
            lines.append(f"<details><summary><code>{test_id}</code></summary>\n\n")
            if failure["message"]:
                lines.append(f"**Message:** {failure['message']}\n\n")
            if failure["text"]:
                lines.append(f"```\n{failure['text']}\n```\n\n")
            lines.append("</details>\n\n")

    report = "".join(lines)
    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(report)
        log(f"Wrote test report to {output_path}")
    except Exception as e:
        log(f"Failed to write test report: {e}")

    return report


def extract_coverage(base_ref: str) -> tuple[float, float]:
    """Extract overall and diff coverage metrics."""
    diff_json = os.path.join(OUT_DIR, "diff-cover.json")
    diff_report = os.path.join(OUT_DIR, "diff-cover-report.md")

    try:
        tree = ET.parse(DEFAULT_COVERAGE_XML)
        overall = float(tree.getroot().attrib.get("line-rate", 0)) * 100
        log(f"Overall coverage: {overall:.2f}%")
    except Exception as e:
        log(f"Failed to parse {DEFAULT_COVERAGE_XML}: {e}")
        return 0.0, 0.0

    compare_branch = find_compare_branch(base_ref)
    if compare_branch:
        log(f"Running diff-cover --compare-branch={compare_branch}")
        proc = _run_cmd(
            [
                "diff-cover",
                DEFAULT_COVERAGE_XML,
                f"--compare-branch={compare_branch}",
                "--json-report",
                diff_json,
                "--markdown-report",
                diff_report,
                "--fail-under=0",
                "--exclude",
                "datus/cli/web/chatbot.py",
                "--exclude",
                "datus/cli/screen/subject_screen.py",
            ],
            DIFF_COVER_TIMEOUT,
            capture_output=True,
            text=True,
        )
        if not proc or proc.returncode != 0:
            err = proc.stderr.strip() if proc else "timed out"
            code = proc.returncode if proc else "timeout"
            log(f"diff-cover failed (exit {code}): {err}")
        else:
            log("diff-cover completed successfully")
    else:
        log("Skipping diff-cover (no compare branch)")

    try:
        with open(diff_json, encoding="utf-8") as f:
            diff_pct = json.load(f).get("total_percent_covered", 0)
        log(f"Diff coverage: {diff_pct:.2f}%")
    except Exception as e:
        diff_pct = 0
        log(f"Failed to read {diff_json}: {e}")

    return overall, diff_pct


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PR acceptance plus impacted unit tests with coverage and diff reporting.",
    )
    parser.add_argument(
        "base_ref",
        nargs="?",
        default="",
        help="Base branch reference for diff-cover comparison (e.g. 'main').",
    )
    args = parser.parse_args()

    base_ref = args.base_ref
    log(f"Starting (base_ref={base_ref!r})")

    test_exit_code, junit_xml_paths = run_tests(base_ref=base_ref)
    test_outcome = "success" if test_exit_code == 0 else "failure"

    test_results = parse_test_results(junit_xml_paths)
    write_test_report(test_results)
    overall, diff_pct = extract_coverage(base_ref)

    outputs = {
        "overall": f"{overall:.2f}",
        "diff": f"{diff_pct:.2f}",
        "test_total": str(test_results["total"]),
        "test_passed": str(test_results["passed"]),
        "test_failed": str(test_results["failed"] + test_results["errors"]),
        "test_skipped": str(test_results["skipped"]),
        "test_outcome": test_outcome,
    }

    github_output = os.environ.get("GITHUB_OUTPUT", "")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            for key, val in outputs.items():
                f.write(f"{key}={val}\n")
        log(f"Wrote outputs to GITHUB_OUTPUT: {outputs}")
    else:
        log("GITHUB_OUTPUT not set, printing to stdout")
        for key, val in outputs.items():
            print(f"{key}={val}")

    log("Done")
    return test_exit_code


if __name__ == "__main__":
    sys.exit(main())
