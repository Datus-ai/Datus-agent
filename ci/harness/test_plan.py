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


class _UniqueKeySafeLoader(yaml.SafeLoader):
    def construct_document(self, node: yaml.Node) -> dict:
        self._check_unique_keys(node, set())
        return super().construct_document(node)

    def _check_unique_keys(self, node: yaml.Node, visited: set[int]) -> None:
        if id(node) in visited:
            return
        visited.add(id(node))
        if isinstance(node, yaml.MappingNode):
            keys = set()
            for key_node, value_node in node.value:
                # Check explicit keys before SafeLoader flattens YAML merges, so
                # an override of an inherited key remains valid.
                key = "<<" if key_node.tag == "tag:yaml.org,2002:merge" else self.construct_object(key_node, deep=True)
                try:
                    duplicate = key in keys
                    keys.add(key)
                except TypeError as exc:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping", node.start_mark, "found unhashable key", key_node.start_mark
                    ) from exc
                if duplicate:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"found duplicate key {key!r}",
                        key_node.start_mark,
                    )
                self._check_unique_keys(value_node, visited)
        elif isinstance(node, yaml.SequenceNode):
            for child in node.value:
                self._check_unique_keys(child, visited)


def load_map(path: Path = MAP) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=_UniqueKeySafeLoader)  # noqa: S506 - custom safe loader.


def core_targets(data: dict | None = None) -> list[str]:
    return list(dict.fromkeys(n.split("::", 1)[0] for n in (data or load_map())["execution"]["core_nodeids"]))


def normalize_path(path: str) -> str:
    return path.replace("\\", "/").removeprefix("./")


def collapse_targets(targets: Iterable[str]) -> list[str]:
    """Keep parent directories without narrowing pytest collection to explicit children."""
    unique = list(dict.fromkeys(targets))
    directories = [target for target in unique if target.endswith("/")]
    return [
        target for target in unique if not any(target != parent and target.startswith(parent) for parent in directories)
    ]


def validate_unit_rules(data: dict, repo_root: Path = ROOT) -> list[str]:
    """Reject ambiguous rules and stale targets before silently losing test coverage."""
    rules = data.get("execution", {}).get("unit_rules")
    if not isinstance(rules, list) or not rules:
        return ["execution.unit_rules: expected a non-empty list of ownership rules"]
    errors, ids, paths = [], set(), set()
    for rule in rules:
        if not isinstance(rule, dict):
            errors.append("unit_rules: expected a rule mapping")
            continue
        rule_id = rule.get("id")
        if not isinstance(rule_id, str) or not rule_id:
            errors.append(f"unit_rules: missing rule id: {rule_id!r}")
            continue
        if rule_id in ids:
            errors.append(f"unit_rules: missing or duplicate rule id: {rule_id!r}")
        ids.add(rule_id)
        for field in ("paths", "targets"):
            values = rule.get(field)
            if not isinstance(values, list) or not values or not all(isinstance(p, str) and p for p in values):
                errors.append(f"unit_rules.{rule_id}.{field}: expected a non-empty list of paths")
                continue
            for path in values:
                if path.startswith("/") or ".." in Path(path).parts:
                    errors.append(f"unit_rules.{rule_id}.{field}: unsafe path: {path}")
                if field == "paths":
                    if path in paths:
                        errors.append(f"unit_rules.{rule_id}: duplicate source path: {path}")
                    paths.add(path)
                elif not path.startswith("tests/unit_tests/") or not (repo_root / path.split("::", 1)[0]).exists():
                    errors.append(f"unit_rules.{rule_id}: unit target does not exist: {path}")
                elif (repo_root / path).is_dir() and not path.endswith("/"):
                    errors.append(f"unit_rules.{rule_id}: directory target needs a trailing slash: {path}")
        if (
            isinstance(rule.get("targets"), list)
            and "tests/unit_tests/" in rule["targets"]
            and not rule.get("rationale")
        ):
            errors.append(f"unit_rules.{rule_id}: full-suite selection requires a rationale")
    for rule in data.get("execution", {}).get("impact_rules", []):
        if "unit_targets" in rule:
            errors.append(f"impact_rules.{rule['id']}: move unit_targets into unit_rules")
    return errors


def unit_owner(path: str, rules: list[dict]) -> tuple[str, dict] | None:
    """Resolve the same exact-file / longest-directory owner for planning and audits."""
    matches = [
        (prefix, rule)
        for rule in rules
        for prefix in rule["paths"]
        if path == prefix or (prefix.endswith("/") and path.startswith(prefix))
    ]
    return max(matches, key=lambda match: len(match[0])) if matches else None


def unit_test_plan(changed: Iterable[str], data: dict | None = None) -> dict:
    """Choose the most specific owner plus its explicit consumer tests for each path.

    Product/nightly rules remain additive in ``impact_plan``. They must not
    broaden a leaf module's unit selection merely because they share a flow.
    """
    data = data or load_map()
    errors = validate_unit_rules(data)
    if errors:
        raise ValueError("Invalid unit ownership rules:\n" + "\n".join(errors))
    rules = data["execution"]["unit_rules"]
    selections = []
    for path in sorted({normalize_path(p) for p in changed}):
        if path.startswith("tests/unit_tests/"):
            if Path(path).name.startswith("test_") and path.endswith(".py"):
                targets, rule_id = [path], "changed-test"
            else:
                # conftest and ordinary helper modules affect their subtree.
                # Fixture assets can live below a fixtures/ directory with no
                # tests of its own; select the closest enclosing test package.
                parent = Path(path).parent
                while (
                    Path(path).name != "conftest.py"
                    and parent != Path("tests/unit_tests")
                    and not any((ROOT / parent).glob("test_*.py"))
                ):
                    parent = parent.parent
                targets, rule_id = [parent.as_posix() + "/"], "test-support"
            selections.append({"path": path, "rule": rule_id, "targets": targets, "fallback": False})
            continue
        match = unit_owner(path, rules)
        if match:
            prefix, rule = match
            selections.append(
                {
                    "path": path,
                    "rule": rule["id"],
                    "matched_path": prefix,
                    "targets": rule["targets"],
                    "fallback": rule.get("fallback", False),
                }
            )
        else:
            selections.append({"path": path, "rule": None, "targets": [], "fallback": False})
    return {
        "targets": collapse_targets(sorted({t for s in selections for t in s["targets"]})),
        "selections": selections,
    }


def impact_plan(changed: Iterable[str], data: dict | None = None) -> dict:
    data = data or load_map()
    files = sorted({normalize_path(p) for p in changed})
    rules = data["execution"]["impact_rules"]
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
        "unit_targets": unit_test_plan(files, data)["targets"],
        "live_targets": sorted({p for r in matched for p in r.get("live_targets", [])}),
        "nightly_groups": sorted({p for r in matched for p in r.get("nightly_groups", [])}),
    }


def changed_files(base_ref: str | None = None) -> list[str]:
    """Include every PR in a merge group and both sides of renamed paths."""
    if base_ref is None:
        event_file = os.environ.get("GITHUB_EVENT_PATH")
        event = json.loads(Path(event_file).read_text()) if event_file else {}
        base_ref = event.get("merge_group", {}).get("base_sha") or event.get("pull_request", {}).get("base", {}).get(
            "sha"
        )
    if not base_ref:
        base_ref = "origin/main" if os.getenv("GITHUB_ACTIONS") == "true" else "upstream/main"
    result = subprocess.run(
        ["git", "diff", "--no-renames", "--name-only", f"{base_ref}...HEAD"],
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


def check_product_execution(plan: dict, reports: list[Path]) -> dict:
    """Every planned file/nodeid must contribute successfully executed tests."""
    if not plan.get("live_targets"):
        return {"status": "failure", "errors": ["The product plan has no live targets"], "evidence": {}}
    return check_execution(
        reports,
        "product_e2e",
        {"execution": {"core_nodeids": plan["live_targets"]}, "flow_groups": {}},
    )
