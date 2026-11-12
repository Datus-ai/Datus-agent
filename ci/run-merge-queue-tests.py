#!/usr/bin/env python3
"""Check generated contracts and coverage-map consistency on the merged tree."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO_ROOT / "ci/merge-queue-results.json"


def run_checks(root: Path, timeout: int) -> dict:
    checks = [
        ("generated-catalog", [sys.executable, "datus/tools/registry/catalog_builder.py", "--check"]),
        (
            "coverage-map",
            [
                sys.executable,
                "ci/harness/validate_coverage_map.py",
                "--strict",
                "--report-json",
                "ci/harness/coverage-map-report.json",
                "--report-md",
                "ci/harness/coverage-map-report.md",
            ],
        ),
    ]
    results = []
    for name, command in checks:
        try:
            process = subprocess.run(command, cwd=root, timeout=timeout, check=False)
            result = {"name": name, "exit_code": process.returncode}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result = {"name": name, "exit_code": 1, "error": str(exc)}
        results.append(result)
    return {"status": "success" if all(r["exit_code"] == 0 for r in results) else "failure", "checks": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    result = run_checks(REPO_ROOT, args.timeout)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return int(result["status"] != "success")


if __name__ == "__main__":
    raise SystemExit(main())
