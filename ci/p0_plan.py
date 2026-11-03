"""Explicit P0 selection and blocking execution evidence, shared with Nightly."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ci.harness.test_plan import check_execution, load_map  # noqa: E402


def load_profile(path: Path | None = None) -> dict:
    profile = yaml.safe_load((path or ROOT / "ci/harness/p0.yml").read_text())
    core = load_map()["execution"]["core_nodeids"]
    for group in profile["groups"].values():
        group["nodeids"] = list(
            dict.fromkeys([*(core if group.get("include_core_nodeids") else []), *group.get("nodeids", [])])
        )
        if not group["nodeids"]:
            raise ValueError("Every P0 group must have required nodeids")
    return profile


def selected_groups(profile: dict, pattern: str = "") -> dict:
    groups = {name: group for name, group in profile["groups"].items() if not pattern or re.search(pattern, name)}
    if not groups:
        raise ValueError("The filter selected no P0 groups")
    return groups


def check_p0_execution(reports: list[Path], groups: dict) -> dict:
    if not groups:
        raise ValueError("P0 requires at least one group")
    nodeids = [nodeid for group in groups.values() for nodeid in group["nodeids"]]
    return check_execution(reports, "p0", {"execution": {"core_nodeids": nodeids}, "flow_groups": {}})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group")
    parser.add_argument("--filter", default="")
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "ci/p0-execution-coverage.json")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    profile = load_profile()
    if args.github_output:
        with args.github_output.open("a") as output:
            for name, sha in profile["release_sources"].items():
                output.write(f"{name}={sha}\n")
    if args.group:
        print("\n".join(profile["groups"][args.group]["nodeids"]))
        return 0
    groups = selected_groups(profile, args.filter)
    if args.verify:
        result = check_p0_execution(sorted(args.verify.glob("*.json")), groups)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        for error in result["errors"]:
            print(error)
        return int(result["status"] != "success")
    print(json.dumps({**profile, "groups": groups}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
