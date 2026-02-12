#!/usr/bin/env python3
"""Extract coverage metrics from coverage.xml and diff-cover, write to GITHUB_OUTPUT."""

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET


def log(msg):
    print(f"[coverage-metrics] {msg}", flush=True)


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

    # Get current branch name to exclude it
    current = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True,
    )
    current_branch = current.stdout.strip() if current.returncode == 0 else ""
    log(f"Current branch: {current_branch}")

    # List all remote branches, exclude current and HEAD
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

    # Get current HEAD commit to skip branches that haven't diverged
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    head_commit = head.stdout.strip() if head.returncode == 0 else ""
    log(f"HEAD commit: {head_commit[:12]}")

    # Find the merge-base closest to HEAD (most recent fork point)
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
        # Skip if merge-base is HEAD itself (no divergence, diff would be empty)
        if commit == head_commit:
            skipped_same_as_head += 1
            continue
        # Get commit timestamp for comparison
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


def main():
    base_ref = sys.argv[1] if len(sys.argv) > 1 else ""
    log(f"Starting coverage metrics extraction (base_ref={base_ref!r})")

    # Overall coverage: parse coverage.xml
    try:
        tree = ET.parse("coverage.xml")
        overall = float(tree.getroot().attrib.get("line-rate", 0)) * 100
        log(f"Overall coverage from coverage.xml: {overall:.2f}%")
    except Exception as e:
        overall = 0
        log(f"Failed to parse coverage.xml: {e}")

    # Diff coverage: run diff-cover and generate JSON
    compare_branch = find_compare_branch(base_ref)
    if compare_branch:
        log(f"Running diff-cover --compare-branch={compare_branch}")
        proc = subprocess.run(
            [
                "diff-cover",
                "coverage.xml",
                f"--compare-branch={compare_branch}",
                "--json-report",
                "diff-cover.json",
                "--markdown-report",
                "diff-cover-report.md",
                "--fail-under=0",
            ],
            capture_output=True,
            text=True,
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

    github_output = os.environ.get("GITHUB_OUTPUT", "")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"overall={overall:.2f}\n")
            f.write(f"diff={diff_pct:.2f}\n")
        log(f"Wrote outputs to GITHUB_OUTPUT: overall={overall:.2f}, diff={diff_pct:.2f}")
    else:
        log("GITHUB_OUTPUT not set, printing to stdout")
        print(f"overall={overall:.2f}")
        print(f"diff={diff_pct:.2f}")

    log("Done")


if __name__ == "__main__":
    main()
