#!/usr/bin/env python3
"""Validate the docs-driven core-flow harness coverage map."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COVERAGE_MAP = Path("ci/harness/coverage-map.yml")
DEFAULT_REPORT_MD = Path("ci/harness/coverage-map-report.md")
DEFAULT_REPORT_JSON = Path("ci/harness/coverage-map-report.json")

PR_LAYER = "pr_acceptance"
MQ_LAYER = "merge_queue"
NIGHTLY_LAYER = "nightly"
WEEKLY_LAYER = "weekly_benchmark"
REQUIRED_LAYERS = (PR_LAYER, MQ_LAYER, NIGHTLY_LAYER, WEEKLY_LAYER)
VALID_PRIORITIES = {"p0", "p1", "p2"}
VALID_STATUSES = {"covered", "partial", "gap", "external", "manual", "not_applicable"}
COVERED_STATUSES = {"covered", "partial", "external", "manual"}
DETERMINISTIC_COVERED_STATUSES = {"covered", "partial"}
ISSUE_REF_RE = re.compile(
    r"^(?:"
    r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/issues/[1-9][0-9]*"
    r"|[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#[1-9][0-9]*"
    r"|#[1-9][0-9]*"
    r")$"
)
FLOW_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")


class FlowRef:
    def __init__(self, group_id: str, flow: dict[str, Any]) -> None:
        self.group_id = group_id
        self.flow = flow

    @property
    def flow_id(self) -> str:
        return str(self.flow.get("id") or "<missing-id>")


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_str_list(value: Any, errors: list[str], field: str, *, allow_empty: bool = False) -> list[str]:
    if value is None:
        if allow_empty:
            return []
        errors.append(f"{field}: required list[str] is missing")
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        errors.append(f"{field}: expected list[str], got {value!r}")
        return []
    if not value and not allow_empty:
        errors.append(f"{field}: must not be empty")
    return value


def split_nodeid(nodeid: str) -> tuple[Path, str]:
    if "::" not in nodeid:
        return Path(nodeid), ""
    path_text, suffix = nodeid.split("::", 1)
    return Path(path_text), suffix


def class_or_function_exists(repo_root: Path, nodeid: str) -> bool:
    path, suffix = split_nodeid(nodeid)
    full_path = repo_root / path
    if not full_path.exists():
        return False
    if not suffix:
        return True
    first = suffix.split("::", 1)[0]
    if not first:
        return True
    text = full_path.read_text(encoding="utf-8", errors="replace")
    escaped = re.escape(first)
    return bool(re.search(rf"^\s*(class|(?:async\s+)?def)\s+{escaped}\b", text, flags=re.MULTILINE))


def iter_flows(coverage_map: dict[str, Any], errors: list[str]) -> list[FlowRef]:
    groups = coverage_map.get("flow_groups")
    if not isinstance(groups, dict):
        errors.append("flow_groups: required mapping is missing")
        return []

    flows: list[FlowRef] = []
    for group_id, group in groups.items():
        if not isinstance(group_id, str) or not group_id:
            errors.append("flow_groups: group id must be a non-empty string")
            continue
        if not isinstance(group, dict):
            errors.append(f"{group_id}: group declaration must be a mapping")
            continue
        group_flows = group.get("flows")
        if not isinstance(group_flows, list) or not group_flows:
            errors.append(f"{group_id}.flows: must be a non-empty list")
            continue
        for index, flow in enumerate(group_flows):
            if not isinstance(flow, dict):
                errors.append(f"{group_id}.flows[{index}]: flow declaration must be a mapping")
                continue
            flows.append(FlowRef(group_id=group_id, flow=flow))
    return flows


def validate_issue_ref(value: str, errors: list[str], field: str) -> None:
    if not ISSUE_REF_RE.fullmatch(value):
        errors.append(f"{field}: invalid GitHub issue reference: {value}")


def parse_issue_ref(value: str, default_owner: str, default_repo: str) -> tuple[str, str, int] | None:
    if value.startswith("#"):
        return default_owner, default_repo, int(value[1:])
    if "#" in value and not value.startswith("http"):
        repo_ref, number = value.rsplit("#", 1)
        owner, repo = repo_ref.split("/", 1)
        return owner, repo, int(number)
    match = re.fullmatch(r"https://github\.com/([^/]+)/([^/]+)/issues/([1-9][0-9]*)", value)
    if not match:
        return None
    return match.group(1), match.group(2), int(match.group(3))


def github_issue_exists(owner: str, repo: str, issue_number: int, token: str | None) -> tuple[bool, str]:
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - fixed GitHub API host.
            return response.status == 200, ""
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return False, str(exc.reason)


def validate_docs(repo_root: Path, flow_ref: FlowRef, errors: list[str]) -> None:
    docs = normalize_str_list(flow_ref.flow.get("docs"), errors, f"{flow_ref.flow_id}.docs")
    for doc_path in docs:
        if not (repo_root / doc_path).exists():
            errors.append(f"{flow_ref.flow_id}.docs: docs path does not exist: {doc_path}")


def validate_nodeids(repo_root: Path, flow_id: str, layer_name: str, layer: dict[str, Any], errors: list[str]) -> None:
    nodeids = normalize_str_list(
        layer.get("nodeids"), errors, f"{flow_id}.coverage.{layer_name}.nodeids", allow_empty=True
    )
    for nodeid in nodeids:
        if not class_or_function_exists(repo_root, nodeid):
            errors.append(f"{flow_id}.coverage.{layer_name}: nodeid target does not exist: {nodeid}")


def validate_layer(flow_id: str, layer_name: str, coverage: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    layer = coverage.get(layer_name)
    if not isinstance(layer, dict):
        errors.append(f"{flow_id}.coverage.{layer_name}: required mapping is missing")
        return {}
    status = layer.get("status")
    if status not in VALID_STATUSES:
        errors.append(f"{flow_id}.coverage.{layer_name}.status: invalid status {status!r}")
    return layer


def flow_has_deterministic_coverage(flow: dict[str, Any]) -> bool:
    coverage = flow.get("coverage") or {}
    for layer_name in (PR_LAYER, MQ_LAYER):
        layer = coverage.get(layer_name) or {}
        if layer.get("status") in DETERMINISTIC_COVERED_STATUSES:
            return True
    return False


def layer_is_covered(flow: dict[str, Any], layer_name: str) -> bool:
    layer = (flow.get("coverage") or {}).get(layer_name) or {}
    return layer.get("status") in COVERED_STATUSES


def validate_flow(repo_root: Path, flow_ref: FlowRef, errors: list[str]) -> None:
    flow = flow_ref.flow
    flow_id = flow_ref.flow_id
    if not isinstance(flow.get("id"), str) or not FLOW_ID_RE.fullmatch(flow_id):
        errors.append(f"{flow_id}: id must match {FLOW_ID_RE.pattern}")
    if not flow_id.startswith(f"{flow_ref.group_id}."):
        errors.append(f"{flow_id}: id must start with group prefix {flow_ref.group_id}.")
    if not isinstance(flow.get("title"), str) or not flow.get("title"):
        errors.append(f"{flow_id}.title: required non-empty string is missing")

    priority = flow.get("priority")
    if priority not in VALID_PRIORITIES:
        errors.append(f"{flow_id}.priority: invalid priority {priority!r}")

    quality_sensitive = flow.get("quality_sensitive")
    if not isinstance(quality_sensitive, bool):
        errors.append(f"{flow_id}.quality_sensitive: expected boolean")

    validate_docs(repo_root, flow_ref, errors)
    normalize_str_list(flow.get("entrypoints"), errors, f"{flow_id}.entrypoints")
    normalize_str_list(flow.get("contracts"), errors, f"{flow_id}.contracts")
    gaps = normalize_str_list(flow.get("gaps"), errors, f"{flow_id}.gaps", allow_empty=True)
    for index, gap in enumerate(gaps):
        validate_issue_ref(gap, errors, f"{flow_id}.gaps[{index}]")

    coverage = flow.get("coverage")
    if not isinstance(coverage, dict):
        errors.append(f"{flow_id}.coverage: required mapping is missing")
        return

    for layer_name in REQUIRED_LAYERS:
        layer = validate_layer(flow_id, layer_name, coverage, errors)
        if layer:
            validate_nodeids(repo_root, flow_id, layer_name, layer, errors)

    if priority == "p0" and not flow_has_deterministic_coverage(flow):
        errors.append(f"{flow_id}: p0 flow must have deterministic PR or merge queue coverage")
    if priority == "p0" and not layer_is_covered(flow, NIGHTLY_LAYER) and not gaps:
        errors.append(f"{flow_id}: p0 flow must have nightly coverage or an explicit gap issue")
    if quality_sensitive and not layer_is_covered(flow, WEEKLY_LAYER) and not gaps:
        errors.append(f"{flow_id}: quality-sensitive flow must have benchmark coverage or an explicit gap issue")


def validate_coverage_map(
    repo_root: Path,
    coverage_map: dict[str, Any],
    *,
    check_github_issues: bool = False,
    github_token: str | None = None,
    default_owner: str = "Datus-ai",
    default_repo: str = "Datus-agent",
) -> tuple[list[str], list[FlowRef]]:
    errors: list[str] = []
    if coverage_map.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    source = coverage_map.get("source")
    if not isinstance(source, dict):
        errors.append("source: required mapping is missing")
    else:
        docs_navigation = source.get("docs_navigation")
        if not isinstance(docs_navigation, str) or not (repo_root / docs_navigation).exists():
            errors.append("source.docs_navigation: must point to an existing file")

    layers = coverage_map.get("layers")
    if not isinstance(layers, dict):
        errors.append("layers: required mapping is missing")
    else:
        for layer_name in REQUIRED_LAYERS:
            if layer_name not in layers:
                errors.append(f"layers.{layer_name}: required layer declaration is missing")

    gap_issue = coverage_map.get("gap_issue")
    if not isinstance(gap_issue, str):
        errors.append("gap_issue: required GitHub issue reference is missing")
    else:
        validate_issue_ref(gap_issue, errors, "gap_issue")

    flows = iter_flows(coverage_map, errors)
    seen_ids: set[str] = set()
    for flow_ref in flows:
        if flow_ref.flow_id in seen_ids:
            errors.append(f"{flow_ref.flow_id}: duplicate flow id")
        seen_ids.add(flow_ref.flow_id)
        validate_flow(repo_root, flow_ref, errors)

    if check_github_issues:
        issue_refs = sorted(
            {
                ref
                for ref in [gap_issue] + [gap for flow_ref in flows for gap in (flow_ref.flow.get("gaps") or [])]
                if isinstance(ref, str) and ISSUE_REF_RE.fullmatch(ref)
            }
        )
        for issue_ref in issue_refs:
            parsed = parse_issue_ref(issue_ref, default_owner, default_repo)
            if not parsed:
                continue
            owner, repo, issue_number = parsed
            exists, reason = github_issue_exists(owner, repo, issue_number, github_token)
            if not exists:
                errors.append(f"{issue_ref}: GitHub issue could not be verified ({reason})")

    return errors, flows


def coverage_status(flow: dict[str, Any], layer_name: str) -> str:
    return str(((flow.get("coverage") or {}).get(layer_name) or {}).get("status") or "")


def build_summary(flows: list[FlowRef], errors: list[str]) -> dict[str, Any]:
    priorities = Counter(str(flow_ref.flow.get("priority") or "unknown") for flow_ref in flows)
    layer_statuses = {
        layer_name: dict(Counter(coverage_status(flow_ref.flow, layer_name) for flow_ref in flows))
        for layer_name in REQUIRED_LAYERS
    }
    gap_flows = [flow_ref.flow_id for flow_ref in flows if flow_ref.flow.get("gaps")]
    return {
        "flow_count": len(flows),
        "priority_counts": dict(sorted(priorities.items())),
        "layer_status_counts": layer_statuses,
        "gap_flow_count": len(gap_flows),
        "gap_flows": gap_flows,
        "error_count": len(errors),
    }


def build_report_markdown(flows: list[FlowRef], errors: list[str], source_path: Path) -> str:
    summary = build_summary(flows, errors)
    lines = [
        "# Harness Coverage Map Report",
        "",
        f"- Source: `{source_path}`",
        f"- Generated at: `{utc_now()}`",
        f"- Flows: {summary['flow_count']}",
        f"- Validation errors: {summary['error_count']}",
        "",
        "## Coverage",
        "",
        "| Group | Flow | Priority | PR | Merge Queue | Nightly | Weekly Benchmark | Gaps |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for flow_ref in flows:
        flow = flow_ref.flow
        gaps = flow.get("gaps") or []
        gap_text = "<br>".join(gaps) if gaps else ""
        lines.append(
            "| {group} | `{flow_id}` {title} | {priority} | {pr} | {mq} | {nightly} | {weekly} | {gaps} |".format(
                group=flow_ref.group_id,
                flow_id=flow_ref.flow_id,
                title=flow.get("title", ""),
                priority=flow.get("priority", ""),
                pr=coverage_status(flow, PR_LAYER),
                mq=coverage_status(flow, MQ_LAYER),
                nightly=coverage_status(flow, NIGHTLY_LAYER),
                weekly=coverage_status(flow, WEEKLY_LAYER),
                gaps=gap_text,
            )
        )
    if errors:
        lines.extend(["", "## Validation Errors", ""])
        lines.extend(f"- {error}" for error in errors)
    return "\n".join(lines) + "\n"


def build_manifest(source_path: Path, flows: list[FlowRef], errors: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "source": str(source_path),
        "summary": build_summary(flows, errors),
        "flows": [
            {
                "group": flow_ref.group_id,
                "id": flow_ref.flow_id,
                "title": flow_ref.flow.get("title", ""),
                "priority": flow_ref.flow.get("priority", ""),
                "quality_sensitive": bool(flow_ref.flow.get("quality_sensitive", False)),
                "docs": flow_ref.flow.get("docs") or [],
                "coverage": {
                    layer_name: {
                        "status": coverage_status(flow_ref.flow, layer_name),
                        "nodeids": ((flow_ref.flow.get("coverage") or {}).get(layer_name) or {}).get("nodeids") or [],
                    }
                    for layer_name in REQUIRED_LAYERS
                },
                "gaps": flow_ref.flow.get("gaps") or [],
            }
            for flow_ref in flows
        ],
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--coverage-map", default=str(DEFAULT_COVERAGE_MAP))
    parser.add_argument("--report-md", default="")
    parser.add_argument("--report-json", default="")
    parser.add_argument("--strict", action="store_true", help="Fail when validation errors are present.")
    parser.add_argument(
        "--check-github-issues",
        action="store_true",
        help="Verify gap issue references through the GitHub REST API. Not required for deterministic CI.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    coverage_map_path = (repo_root / args.coverage_map).resolve()
    coverage_map = read_yaml(coverage_map_path)
    errors, flows = validate_coverage_map(
        repo_root,
        coverage_map,
        check_github_issues=args.check_github_issues,
        github_token=os.environ.get("GITHUB_TOKEN"),
    )
    manifest = build_manifest(coverage_map_path.relative_to(repo_root), flows, errors)

    report_md = Path(args.report_md) if args.report_md else None
    if report_md:
        report_md.write_text(
            build_report_markdown(flows, errors, coverage_map_path.relative_to(repo_root)), encoding="utf-8"
        )

    report_json = Path(args.report_json) if args.report_json else None
    if report_json:
        write_json(report_json, manifest)

    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    if args.strict and errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
