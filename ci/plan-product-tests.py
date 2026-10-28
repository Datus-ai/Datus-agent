#!/usr/bin/env python3
"""Print and persist the impact-based live product test plan."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.harness.test_plan import changed_files, impact_plan  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref")
    parser.add_argument("--output", type=Path, default=Path("ci/product-test-plan.json"))
    args = parser.parse_args()
    plan = impact_plan(changed_files(args.base_ref))
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as handle:
            handle.write(f"required={str(bool(plan['live_targets'])).lower()}\n")
            handle.write(f"targets={json.dumps(plan['live_targets'])}\n")


if __name__ == "__main__":
    main()
