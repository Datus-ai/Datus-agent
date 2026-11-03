"""Expose the shared merge-group classification to workflow preparation steps."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ci.harness.gate_profile import repository_profile  # noqa: E402
from ci.harness.test_plan import changed_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    result = repository_profile(changed_files(args.base_ref), args.base_ref)
    print(json.dumps(result, indent=2))
    if args.github_output:
        with args.github_output.open("a") as output:
            output.write(f"profile={result['profile']}\nci_tests={str(result['ci_tests']).lower()}\n")


if __name__ == "__main__":
    main()
