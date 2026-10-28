"""Shared impact selection and execution evidence for PR and merge-queue gates."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "ci/harness/coverage-map.yml"


def load_map(path: Path = MAP) -> dict:
    return yaml.safe_load(path.read_text())


def core_targets(data: dict | None = None) -> list[str]:
    return list(dict.fromkeys(n.split("::", 1)[0] for n in (data or load_map())["execution"]["core_nodeids"]))


def impact_plan(changed: Iterable[str], data: dict | None = None) -> dict:
    files = sorted(set(changed))
    rules = (data or load_map())["execution"]["impact_rules"]
    matched = [
        r
        for r in rules
        if any(
            any(p == prefix or (prefix.endswith("/") and p.startswith(prefix)) for prefix in r["paths"]) for p in files
        )
    ]
    return {
        "changed_files": files,
        "reasons": [r["id"] for r in matched],
        "unit_targets": sorted({p for r in matched for p in r.get("unit_targets", [])}),
        "live_targets": sorted({p for r in matched for p in r.get("live_targets", [])}),
        "nightly_groups": sorted({p for r in matched for p in r.get("nightly_groups", [])}),
    }


def changed_files(base_ref: str | None = None) -> list[str]:
    """Use the event base SHA, including every PR in a merge group."""
    if base_ref is None:
        event_file = os.environ.get("GITHUB_EVENT_PATH")
        event = json.loads(Path(event_file).read_text()) if event_file else {}
        base_ref = event.get("merge_group", {}).get("base_sha") or event.get("pull_request", {}).get("base", {}).get(
            "sha"
        )
    if not base_ref:
        base_ref = "upstream/main"
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return result.stdout.splitlines()


def covers(reference: str, nodeid: str) -> bool:
    return nodeid == reference or nodeid.startswith(reference + "::") or nodeid.startswith(reference + "[")


def check_execution(reports: list[Path], layer: str, data: dict | None = None) -> dict:
    data = data or load_map()
    selected, outcomes = set(), {}
    errors = []
    for path in reports:
        if not path.is_file():
            errors.append(f"Missing execution report: {path}")
            continue
        try:
            report = json.loads(path.read_text())
            if report.get("exit_code", 0) not in (0, 5):
                errors.append(f"Unsuccessful pytest session: {path}: exit {report['exit_code']}")
            if not isinstance(report["collected"], list) or not isinstance(report["outcomes"], dict):
                raise ValueError("Invalid collected/outcomes fields")
        except (ValueError, KeyError) as exc:
            errors.append(f"Invalid execution report: {path}: {exc}")
            continue
        selected.update(report["collected"])
        for nodeid, outcome in report["outcomes"].items():
            if outcomes.get(nodeid, "passed") == "passed":
                outcomes[nodeid] = outcome
    required = list(data["execution"]["core_nodeids"])
    for group in data["flow_groups"].values():
        for flow in group.get("flows", []):
            claim = flow.get("coverage", {}).get(layer, {})
            if claim.get("status") == "covered":
                required.extend(claim.get("nodeids", []))
    evidence = {}
    for reference in dict.fromkeys(required):
        matches = sorted(n for n in selected if covers(reference, n))
        unhealthy = {n: outcomes.get(n, "not_executed") for n in matches if outcomes.get(n) != "passed"}
        evidence[reference] = {"collected": matches, "unhealthy": unhealthy}
        if not matches:
            errors.append(f"Required test was not collected: {reference}")
        elif unhealthy:
            errors.append(f"Required test did not pass: {reference}: {unhealthy}")
    return {"layer": layer, "status": "failure" if errors else "success", "errors": errors, "evidence": evidence}


def verify_gate(reports: list[Path], layer: str, output: Path) -> int:
    result = check_execution(reports, layer)
    output.write_text(json.dumps(result, indent=2) + "\n")
    for error in result["errors"]:
        print(f"[gate-coverage] {error}", flush=True)
    return int(bool(result["errors"]))
