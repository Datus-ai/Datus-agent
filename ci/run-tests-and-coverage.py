#!/usr/bin/env python3
"""Run unit tests with coverage, extract metrics, and generate reports.

Usage:
    python3 ci/run-tests-and-coverage.py [base_ref]
    python3 ci/run-tests-and-coverage.py [base_ref] --test-paths t1.py t2.py
    python3 ci/run-tests-and-coverage.py --source-dir datus/tools/ --test-paths t1.py t2.py

Outputs (written to GITHUB_OUTPUT if available, otherwise stdout):
    overall       - Overall line coverage percentage
    diff          - Diff coverage percentage (or directory coverage in --source-dir mode)
    test_total    - Total test count
    test_passed   - Passed test count
    test_failed   - Failed test count (failures + errors)
    test_skipped  - Skipped test count
    test_outcome  - "success" or "failure"

Generated files (all written to ci/ directory):
    ci/coverage.xml        - Cobertura coverage report
    ci/htmlcov/            - Full HTML coverage report
    ci/test-results.xml    - JUnit test results
    ci/diff-cover.json     - Diff coverage data (diff mode)
    ci/dir-cover.json      - Directory coverage data (--source-dir mode)
    ci/diff-cover-report.md - Diff coverage markdown report
    ci/test-report.md      - Test failure markdown report
    ci/pytest-coverage.txt  - Full pytest output
"""

import argparse
import json
import os
import subprocess
import sys
import defusedxml.ElementTree as ET

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def log(msg):
    print(f"[ci] {msg}", flush=True)


# Timeout configuration (seconds), configurable via environment variables
TEST_CMD_TIMEOUT = int(os.environ.get("TEST_CMD_TIMEOUT", "1800"))    # 30 min
GIT_CMD_TIMEOUT = int(os.environ.get("GIT_CMD_TIMEOUT", "60"))        # 60 sec
DIFF_COVER_TIMEOUT = int(os.environ.get("DIFF_COVER_TIMEOUT", "300")) # 5 min


def _run_cmd(cmd, timeout, **kwargs):
    """Run a command with timeout. Returns CompletedProcess or None on timeout."""
    try:
        return subprocess.run(cmd, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired:
        log(f"Command timed out after {timeout}s: {' '.join(cmd)}")
        return None


# ---------------------------------------------------------------------------
# 1. Run pytest
# ---------------------------------------------------------------------------


def run_tests(test_paths=None):
    """Run pytest and return the exit code.

    Args:
        test_paths: Optional list of specific test file paths to run.
                    When None or empty, runs the full tests/unit_tests/ suite.
    """
    log("Running pytest...")
    coverage_xml = os.path.join(OUT_DIR, "coverage.xml")
    coverage_html = os.path.join(OUT_DIR, "htmlcov")
    coverage_db = os.path.join(OUT_DIR, ".coverage")
    results_xml = os.path.join(OUT_DIR, "test-results.xml")
    targets = test_paths if test_paths else ["tests/unit_tests/"]
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        "--cov=datus",
        f"--cov-report=xml:{coverage_xml}",
        f"--cov-report=html:{coverage_html}",
        "--cov-report=term-missing",
        f"--junitxml={results_xml}",
        "-q",
        "--disable-warnings",
        "--log-level=CRITICAL",
        "--log-cli-level=CRITICAL",
        "--showlocals",
    ]
    log(f"Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["COVERAGE_FILE"] = coverage_db

    with open(os.path.join(OUT_DIR, "pytest-coverage.txt"), "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        if proc.stdout:
            for line in proc.stdout:
                sys.stdout.write(line)
                log_file.write(line)
        try:
            exit_code = proc.wait(timeout=TEST_CMD_TIMEOUT)
        except subprocess.TimeoutExpired:
            log(f"pytest timed out after {TEST_CMD_TIMEOUT}s, killing process")
            proc.kill()
            if proc.stdout:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    log_file.write(line)
            proc.wait()
            log_file.write(f"\n[ci] TIMEOUT: pytest killed after {TEST_CMD_TIMEOUT}s\n")
            exit_code = 1

    log(f"pytest exited with code {exit_code}")

    # Clean up .coverage sqlite file after reports are generated
    if os.path.exists(coverage_db):
        os.remove(coverage_db)

    return exit_code


# ---------------------------------------------------------------------------
# 2. Parse test results
# ---------------------------------------------------------------------------


def parse_test_results(junit_xml_path=None):
    """Parse JUnit XML to extract test counts and failure details."""
    if junit_xml_path is None:
        junit_xml_path = os.path.join(OUT_DIR, "test-results.xml")
    total = passed = failed = errors = skipped = 0
    failures = []

    try:
        tree = ET.parse(junit_xml_path)
        root = tree.getroot()

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

        passed = total - failed - errors - skipped
        log(f"Test results: {passed} passed, {failed} failed, {errors} errors, {skipped} skipped (total: {total})")
    except Exception as e:
        log(f"Failed to parse {junit_xml_path}: {e}")
        sys.exit(1)

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "failures": failures,
    }


def write_test_report(test_results, output_path=None):
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
        for i, f in enumerate(failures, 1):
            test_id = f"{f['classname']}::{f['name']}" if f["classname"] else f["name"]
            lines.append(f"{i}. `{test_id}`\n")

        lines.append("\n### Failure Details\n\n")
        for f in failures:
            test_id = f"{f['classname']}::{f['name']}" if f["classname"] else f["name"]
            lines.append(f"<details><summary><code>{test_id}</code></summary>\n\n")
            if f["message"]:
                lines.append(f"**Message:** {f['message']}\n\n")
            if f["text"]:
                lines.append(f"```\n{f['text']}\n```\n\n")
            lines.append("</details>\n\n")

    report = "".join(lines)
    try:
        with open(output_path, "w") as fh:
            fh.write(report)
        log(f"Wrote test report to {output_path}")
    except Exception as e:
        log(f"Failed to write test report: {e}")

    return report


# ---------------------------------------------------------------------------
# 3. Coverage metrics
# ---------------------------------------------------------------------------


def find_compare_branch(base_ref):
    """Determine the compare branch for diff-cover.

    Priority:
    1. Explicit base_ref argument (e.g. from PR event)
    2. Most recent merge-base with any remote branch
    """
    if base_ref:
        log(f"Using explicit base_ref: origin/{base_ref}")
        return f"origin/{base_ref}"

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

    return best_commit


def extract_coverage(base_ref):
    """Extract overall and diff coverage metrics."""
    coverage_xml = os.path.join(OUT_DIR, "coverage.xml")
    diff_json = os.path.join(OUT_DIR, "diff-cover.json")
    diff_report = os.path.join(OUT_DIR, "diff-cover-report.md")

    # Overall coverage
    try:
        tree = ET.parse(coverage_xml)
        overall = float(tree.getroot().attrib.get("line-rate", 0)) * 100
        log(f"Overall coverage: {overall:.2f}%")
    except Exception as e:
        log(f"Failed to parse {coverage_xml}: {e}")
        sys.exit(1)

    # Diff coverage
    compare_branch = find_compare_branch(base_ref)
    if compare_branch:
        log(f"Running diff-cover --compare-branch={compare_branch}")
        proc = _run_cmd(
            [
                "diff-cover",
                coverage_xml,
                f"--compare-branch={compare_branch}",
                "--json-report",
                diff_json,
                "--markdown-report",
                diff_report,
                "--fail-under=0",
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
        with open(diff_json) as f:
            diff_pct = json.load(f).get("total_percent_covered", 0)
        log(f"Diff coverage: {diff_pct:.2f}%")
    except Exception as e:
        diff_pct = 0
        log(f"Failed to read {diff_json}: {e}")

    return overall, diff_pct


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------


def extract_directory_coverage(source_dir):
    """Extract coverage metrics for all files under a source directory.

    Args:
        source_dir: Source directory path relative to repo root
                    (e.g. "datus/tools/" or "datus/utils/").

    Returns:
        (overall_pct, dir_pct) tuple.
    """
    coverage_xml = os.path.join(OUT_DIR, "coverage.xml")
    dir_json = os.path.join(OUT_DIR, "dir-cover.json")

    # Overall coverage
    try:
        tree = ET.parse(coverage_xml)
        root = tree.getroot()
        overall = float(root.attrib.get("line-rate", 0)) * 100
        log(f"Overall coverage: {overall:.2f}%")
    except Exception as e:
        log(f"Failed to parse {coverage_xml}: {e}")
        sys.exit(1)

    # Determine the directory prefix relative to the source root.
    # coverage.xml <source> points to e.g. ".../datus", and filenames are
    # relative to that (e.g. "tools/some_tool.py").  We strip the leading
    # "datus/" from source_dir to get the prefix to match.
    normalized = source_dir.rstrip("/") + "/"
    if normalized.startswith("datus/"):
        prefix = normalized[len("datus/") :]
    else:
        prefix = normalized

    log(f"Directory coverage mode: source_dir={source_dir}, prefix={prefix}")

    src_stats = {}
    total_covered = 0
    total_violations = 0

    try:
        tree = ET.parse(coverage_xml)
        root = tree.getroot()

        for cls in root.iter("class"):
            filename = cls.attrib.get("filename", "")
            if not filename.startswith(prefix):
                continue

            covered_lines = []
            violation_lines = []
            for line in cls.findall(".//line"):
                line_num = int(line.attrib["number"])
                hits = int(line.attrib["hits"])
                if hits > 0:
                    covered_lines.append(line_num)
                else:
                    violation_lines.append(line_num)

            total = len(covered_lines) + len(violation_lines)
            pct = (len(covered_lines) / total * 100) if total > 0 else 100.0

            # Use full path with datus/ prefix as key
            full_path = f"datus/{filename}"
            src_stats[full_path] = {
                "covered_lines": covered_lines,
                "violation_lines": violation_lines,
                "percent_covered": round(pct, 1),
            }
            total_covered += len(covered_lines)
            total_violations += len(violation_lines)

    except Exception as e:
        log(f"Failed to parse {coverage_xml} for directory coverage: {e}")
        sys.exit(1)

    total_lines = total_covered + total_violations
    dir_pct = (total_covered / total_lines * 100) if total_lines > 0 else 100.0

    report = {
        "report_name": "Directory",
        "src_stats": src_stats,
        "total_num_lines": total_lines,
        "total_num_violations": total_violations,
        "total_percent_covered": round(dir_pct, 2),
    }

    try:
        with open(dir_json, "w") as f:
            json.dump(report, f, indent=2)
        log(f"Wrote directory coverage to {dir_json}")
    except Exception as e:
        log(f"Failed to write {dir_json}: {e}")

    log(f"Directory coverage: {dir_pct:.2f}% ({total_covered}/{total_lines} lines)")
    return overall, dir_pct


def main():
    parser = argparse.ArgumentParser(
        description="Run unit tests with coverage and generate reports.",
    )
    parser.add_argument(
        "base_ref",
        nargs="?",
        default="",
        help="Base branch reference for diff-cover comparison (e.g. 'main').",
    )
    parser.add_argument(
        "--test-paths",
        nargs="*",
        default=None,
        help="Specific test file paths to run. If omitted, runs all tests/unit_tests/.",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Source directory for directory coverage mode (e.g. 'datus/tools/').",
    )
    args = parser.parse_args()

    base_ref = args.base_ref
    test_paths = args.test_paths
    source_dir = args.source_dir
    log(f"Starting (base_ref={base_ref!r}, test_paths={test_paths!r}, source_dir={source_dir!r})")

    # Run tests
    test_exit_code = run_tests(test_paths=test_paths)
    test_outcome = "success" if test_exit_code == 0 else "failure"

    # Parse test results
    test_results = parse_test_results()
    write_test_report(test_results)

    # Extract coverage
    if source_dir:
        overall, dir_pct = extract_directory_coverage(source_dir)
        diff_pct = dir_pct
    else:
        overall, diff_pct = extract_coverage(base_ref)

    # Write outputs
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
        with open(github_output, "a") as f:
            for key, val in outputs.items():
                f.write(f"{key}={val}\n")
        log(f"Wrote outputs to GITHUB_OUTPUT: {outputs}")
    else:
        log("GITHUB_OUTPUT not set, printing to stdout")
        for key, val in outputs.items():
            print(f"{key}={val}")

    log("Done")


if __name__ == "__main__":
    main()
