"""Expose the shared merge-group classification to workflow preparation steps."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ci.harness.gate_profile import repository_profile  # noqa: E402
from ci.harness.pr_test_plan import pr_test_plan  # noqa: E402
from ci.harness.test_plan import changed_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    changed = changed_files(args.base_ref)
    result = repository_profile(changed, args.base_ref)
    plan = pr_test_plan(changed, profile=result)
    result["run_tests"] = bool(plan["pytest_targets"])
    result["needs_excel"] = plan["needs_excel"]
    result["needs_runtime_fixtures"] = plan["needs_runtime_fixtures"]
    print(json.dumps(result, indent=2))
    if args.github_output:
        with args.github_output.open("a") as output:
            output.write(f"profile={result['profile']}\nci_tests={str(result['ci_tests']).lower()}\n")
            output.write(f"run_tests={str(result['run_tests']).lower()}\n")
            output.write(f"needs_excel={str(result['needs_excel']).lower()}\n")
            for key in ("coverage_applicable", "needs_runtime_fixtures"):
                output.write(f"{key}={str(result[key]).lower()}\n")


if __name__ == "__main__":
    main()
