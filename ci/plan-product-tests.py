#!/usr/bin/env python3
"""Print and persist the impact-based live product test plan."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.harness.test_plan import changed_files, check_product_execution, impact_plan  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-execution", type=Path, help="Verify pytest execution reports in this directory")
    parser.add_argument("--plan", type=Path, default=Path("ci/product-test-plan.json"))
    args = parser.parse_args()
    if args.verify_execution:
        plan = json.loads(args.plan.read_text())
        result = check_product_execution(plan, sorted(args.verify_execution.glob("*.json")))
        output = args.output or Path("ci/product-execution-coverage.json")
        output.write_text(json.dumps(result, indent=2) + "\n")
        for error in result["errors"]:
            print(f"[product-execution] {error}", flush=True)
        return int(bool(result["errors"]))
    plan = impact_plan(changed_files(args.base_ref))
    output = args.output or Path("ci/product-test-plan.json")
    output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as handle:
            handle.write(f"required={str(bool(plan['live_targets'])).lower()}\n")
            handle.write(f"targets={json.dumps(plan['live_targets'])}\n")


if __name__ == "__main__":
    raise SystemExit(main())
