#!/usr/bin/env python3
"""Run unit tests with coverage, extract metrics, and generate reports.

Usage:
    python3 ci/run-tests-and-coverage.py [base_ref]

Outputs (written to GITHUB_OUTPUT if available, otherwise stdout):
    overall       - Overall line coverage percentage
    diff          - Diff coverage percentage
    test_total    - Total test count
    test_passed   - Passed test count
    test_failed   - Failed test count (failures + errors)
    test_skipped  - Skipped test count
    test_outcome  - "success" or "failure"

Generated files:
    coverage.xml        - Cobertura coverage report
    test-results.xml    - JUnit test results
    diff-cover.json     - Diff coverage data
    diff-cover-report.md - Diff coverage markdown report
    test-report.md      - Test failure markdown report
    pytest-coverage.txt  - Full pytest output
"""

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET


def log(msg):
    print(f"[ci] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Run pytest
# ---------------------------------------------------------------------------

def run_tests():
    """Run pytest and return the exit code."""
    log("Running pytest...")
    cmd = [
        sys.executable, "-m", "pytest", "tests/unit_tests/",
        "--cov=datus",
        "--cov-report=xml:coverage.xml",
        "--cov-report=term-missing",
        "--junitxml=test-results.xml",
        "-s", "-vv", "--tb=short", "--showlocals",
    ]
    log(f"Command: {' '.join(cmd)}")

    with open("pytest-coverage.txt", "w") as log_file:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if proc.stdout:
            for line in proc.stdout:
                sys.stdout.write(line)
                log_file.write(line)
        exit_code = proc.wait()

    log(f"pytest exited with code {exit_code}")
    return exit_code


# ---------------------------------------------------------------------------
# 2. Parse test results
# ---------------------------------------------------------------------------

def parse_test_results(junit_xml_path="test-results.xml"):
    """Parse JUnit XML to extract test counts and failure details."""
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
                    failures.append({
                        "name": testcase.attrib.get("name", "unknown"),
                        "classname": testcase.attrib.get("classname", ""),
                        "message": fault.attrib.get("message", ""),
                        "text": (fault.text or "").strip(),
                    })

        passed = total - failed - errors - skipped
        log(f"Test results: {passed} passed, {failed} failed, {errors} errors, {skipped} skipped (total: {total})")
    except Exception as e:
        log(f"Failed to parse {junit_xml_path}: {e}")

    return {
        "total": total, "passed": passed, "failed": failed,
        "errors": errors, "skipped": skipped, "failures": failures,
    }


def write_test_report(test_results, output_path="test-report.md"):
    """Write a markdown report of test failures."""
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

    current = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True,
    )
    current_branch = current.stdout.strip() if current.returncode == 0 else ""
    log(f"Current branch: {current_branch}")

    result = subprocess.run(
        ["git", "branch", "-r", "--format=%(refname:short)"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log("Failed to list remote branches")
        return None

    branches = [
        b.strip() for b in result.stdout.splitlines()
        if b.strip()
        and not b.strip().endswith("/HEAD")
        and b.strip() != f"origin/{current_branch}"
    ]
    log(f"Candidate remote branches: {len(branches)}")

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    head_commit = head.stdout.strip() if head.returncode == 0 else ""
    log(f"HEAD commit: {head_commit[:12]}")

    best_commit = None
    best_branch = None
    best_timestamp = -1
    skipped_same_as_head = 0
    for branch in branches:
        mb = subprocess.run(
            ["git", "merge-base", "HEAD", branch],
            capture_output=True, text=True,
        )
        if mb.returncode != 0:
            continue
        commit = mb.stdout.strip()
        if commit == head_commit:
            skipped_same_as_head += 1
            continue
        ts = subprocess.run(
            ["git", "log", "-1", "--format=%ct", commit],
            capture_output=True, text=True,
        )
        if ts.returncode == 0:
            timestamp = int(ts.stdout.strip())
            if timestamp > best_timestamp:
                best_timestamp = timestamp
                best_commit = commit
                best_branch = branch

    log(f"Skipped {skipped_same_as_head} branches (merge-base == HEAD)")
    if best_commit:
        count = subprocess.run(
            ["git", "rev-list", "--count", f"{best_commit}..HEAD"],
            capture_output=True, text=True,
        )
        commits_ahead = count.stdout.strip() if count.returncode == 0 else "?"
        log(f"Selected merge-base: {best_commit[:12]} (branch: {best_branch}, {commits_ahead} commits ahead)")
    else:
        log("No suitable merge-base found")

    return best_commit


def extract_coverage(base_ref):
    """Extract overall and diff coverage metrics."""
    # Overall coverage
    try:
        tree = ET.parse("coverage.xml")
        overall = float(tree.getroot().attrib.get("line-rate", 0)) * 100
        log(f"Overall coverage: {overall:.2f}%")
    except Exception as e:
        overall = 0
        log(f"Failed to parse coverage.xml: {e}")

    # Diff coverage
    compare_branch = find_compare_branch(base_ref)
    if compare_branch:
        log(f"Running diff-cover --compare-branch={compare_branch}")
        proc = subprocess.run(
            [
                "diff-cover", "coverage.xml",
                f"--compare-branch={compare_branch}",
                "--json-report", "diff-cover.json",
                "--markdown-report", "diff-cover-report.md",
                "--fail-under=0",
            ],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            log(f"diff-cover failed (exit {proc.returncode}): {proc.stderr.strip()}")
        else:
            log("diff-cover completed successfully")
    else:
        log("Skipping diff-cover (no compare branch)")

    try:
        with open("diff-cover.json") as f:
            diff_pct = json.load(f).get("total_percent_covered", 0)
        log(f"Diff coverage: {diff_pct:.2f}%")
    except Exception as e:
        diff_pct = 0
        log(f"Failed to read diff-cover.json: {e}")

    return overall, diff_pct


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    base_ref = sys.argv[1] if len(sys.argv) > 1 else ""
    log(f"Starting (base_ref={base_ref!r})")

    # Run tests
    test_exit_code = run_tests()
    test_outcome = "success" if test_exit_code == 0 else "failure"

    # Parse test results
    test_results = parse_test_results()
    write_test_report(test_results)

    # Extract coverage
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
