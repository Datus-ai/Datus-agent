#!/usr/bin/env python3
"""Generate and update the Datus nightly manifest artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

NODEID_RE = re.compile(r"^(?P<nodeid>(?:tests|ci)/.*?\.py(?:::[^\s]+)+)(?:\s|$)")
MANIFEST_NODEID_RE = re.compile(r"^DATUS_MANIFEST_NODEID\s+(?P<nodeid>.+)$")
SUMMARY_RE = re.compile(r"=+\s+(?P<summary>.+?)\s+=+$")
COUNT_RE = re.compile(
    r"(?P<count>\d+)\s+"
    r"(?P<kind>passed|failed|error|errors|skipped|xfailed|xpassed|deselected|rerun|reruns|selected)"
)

PACKAGE_NAMES = (
    "datus-agent",
    "datus-db-core",
    "datus-sqlalchemy",
    "datus-postgresql",
    "datus-mysql",
    "datus-clickhouse",
    "datus-starrocks",
    "datus-trino",
    "datus-greenplum",
    "datus-hive",
    "datus-spark",
    "datus-bi-core",
    "datus-bi-superset",
    "datus-bi-grafana",
    "datus-scheduler-core",
    "datus-scheduler-airflow",
    "datus-semantic-core",
    "datus-semantic-metricflow",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "suites": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid manifest JSON at {path}: {exc}") from exc


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_output(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def repo_info(path: Path, name: str | None = None) -> dict[str, Any]:
    exists = path.exists()
    return {
        "name": name or path.name,
        "path": str(path),
        "exists": exists,
        "commit": git_output(path, "rev-parse", "HEAD") if exists else "",
        "branch": git_output(path, "rev-parse", "--abbrev-ref", "HEAD") if exists else "",
        "remote": git_output(path, "config", "--get", "remote.origin.url") if exists else "",
        "dirty": bool(git_output(path, "status", "--porcelain")) if exists else False,
    }


def package_info(name: str) -> dict[str, Any]:
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return {"name": name, "installed": False}

    direct_url = None
    direct_url_text = dist.read_text("direct_url.json")
    if direct_url_text:
        try:
            direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError:
            direct_url = {"raw": direct_url_text}

    return {
        "name": name,
        "installed": True,
        "version": dist.version,
        "location": str(Path(dist.locate_file(""))),
        "direct_url": direct_url,
    }


def file_sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def env_value(name: str) -> str:
    return os.environ.get(name, "")


def init_manifest(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    external_root = Path(args.external_repos_root).resolve()
    manifest = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "run": {
            "repository": env_value("GITHUB_REPOSITORY"),
            "workflow": env_value("GITHUB_WORKFLOW"),
            "run_id": env_value("GITHUB_RUN_ID"),
            "run_number": env_value("GITHUB_RUN_NUMBER"),
            "run_attempt": env_value("GITHUB_RUN_ATTEMPT"),
            "ref": env_value("GITHUB_REF"),
            "sha": env_value("GITHUB_SHA") or git_output(repo_root, "rev-parse", "HEAD"),
            "event_name": env_value("GITHUB_EVENT_NAME"),
            "datus_agent_ref": env_value("DATUS_AGENT_REF"),
            "nightly_group_filter": env_value("NIGHTLY_GROUP_FILTER"),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "runner_name": env_value("RUNNER_NAME"),
            "runner_os": env_value("RUNNER_OS"),
            "runner_arch": env_value("RUNNER_ARCH"),
        },
        "paths": {
            "repo_root": str(repo_root),
            "external_repos_root": str(external_root),
            "nightly_home": env_value("NIGHTLY_HOME") or env_value("DATUS_TEST_HOME"),
            "nightly_project_root": env_value("NIGHTLY_PROJECT_ROOT"),
            "nightly_pytest_basetemp": env_value("NIGHTLY_PYTEST_BASETEMP"),
            "unit_test_home": env_value("UNIT_TEST_HOME") or env_value("NIGHTLY_UNIT_TEST_HOME"),
            "log_file": env_value("LOG_FILE") or env_value("NIGHTLY_LOG_FILE"),
        },
        "repositories": [
            repo_info(repo_root, "Datus-agent"),
            repo_info(external_root / "datus-db-adapters"),
            repo_info(external_root / "datus-bi-adapters"),
            repo_info(external_root / "datus-scheduler-adapters"),
            repo_info(external_root / "datus-semantic-adapter"),
        ],
        "packages": [package_info(name) for name in PACKAGE_NAMES],
        "compose_projects": [],
        "suites": [],
    }
    write_manifest(Path(args.output), manifest)


def find_suite(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    suites = manifest.setdefault("suites", [])
    for suite in suites:
        if suite.get("name") == name:
            return suite
    suite = {"name": name}
    suites.append(suite)
    return suite


def parse_command(raw: str) -> list[str]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return [raw]
    return parsed if isinstance(parsed, list) else [str(parsed)]


def record_suite(args: argparse.Namespace) -> None:
    path = Path(args.output)
    manifest = load_manifest(path)
    suite = find_suite(manifest, args.name)
    suite.update(
        {
            "mode": args.mode,
            "kind": args.kind,
            "status": args.status,
            "exit_code": args.exit_code,
            "started_at": args.started_at,
            "ended_at": args.ended_at,
            "command": parse_command(args.command_json),
        }
    )
    if args.compose_file or args.compose_project:
        compose_path = Path(args.compose_file) if args.compose_file else Path()
        suite["compose"] = {
            "project": args.compose_project,
            "file": args.compose_file,
            "file_sha256": file_sha256(compose_path) if args.compose_file else "",
            "host_ports": parse_host_ports(args.host_ports),
        }
    write_manifest(path, manifest)


def parse_host_ports(raw: str) -> list[dict[str, str]]:
    if not raw:
        return []
    ports = []
    for line in raw.splitlines():
        if ":" not in line:
            continue
        label, port = line.rsplit(":", 1)
        ports.append({"label": label.strip(), "port": port.strip()})
    return ports


def parse_collection_output(text: str) -> dict[str, Any]:
    nodeids: list[str] = []
    summaries: list[str] = []
    counts: dict[str, int] = {}

    for raw_line in text.replace("\r", "").splitlines():
        line = strip_ansi(raw_line).strip()
        if not line:
            continue

        manifest_nodeid_match = MANIFEST_NODEID_RE.match(line)
        if manifest_nodeid_match:
            nodeid = manifest_nodeid_match.group("nodeid")
            if nodeid not in nodeids:
                nodeids.append(nodeid)
            continue

        nodeid_match = NODEID_RE.match(line)
        if nodeid_match:
            nodeid = nodeid_match.group("nodeid")
            if nodeid not in nodeids:
                nodeids.append(nodeid)
            continue

        summary_match = SUMMARY_RE.match(line)
        if summary_match:
            summary = summary_match.group("summary").strip()
            summaries.append(summary)
            for count_match in COUNT_RE.finditer(summary):
                kind = count_match.group("kind")
                normalized = "errors" if kind == "error" else "reruns" if kind == "rerun" else kind
                counts[normalized] = counts.get(normalized, 0) + int(count_match.group("count"))

    if nodeids:
        counts.setdefault("selected", len(nodeids))

    return {
        "nodeids": nodeids,
        "counts": counts,
        "summaries": summaries,
        "raw_tail": tail_lines(text, 80),
    }


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)


def tail_lines(text: str, limit: int) -> list[str]:
    lines = strip_ansi(text).replace("\r", "").splitlines()
    return lines[-limit:]


def record_collection(args: argparse.Namespace) -> None:
    path = Path(args.output)
    manifest = load_manifest(path)
    suite = find_suite(manifest, args.name)
    output_path = Path(args.collection_output)
    text = output_path.read_text(encoding="utf-8", errors="replace") if output_path.exists() else ""
    collection = parse_collection_output(text)
    collection.update(
        {
            "exit_code": args.exit_code,
            "collected_at": utc_now(),
            "status": "passed" if args.exit_code == 0 else "failed",
        }
    )
    suite["collection"] = collection
    write_manifest(path, manifest)


def record_compose_project(args: argparse.Namespace) -> None:
    path = Path(args.output)
    manifest = load_manifest(path)
    projects = manifest.setdefault("compose_projects", [])
    entry = {
        "group": args.group,
        "project": args.project,
        "compose_file": args.compose_file,
        "compose_file_sha256": file_sha256(Path(args.compose_file)),
        "host_ports": parse_host_ports(args.host_ports),
        "recorded_at": utc_now(),
    }
    projects[:] = [project for project in projects if project.get("group") != args.group]
    projects.append(entry)
    write_manifest(path, manifest)


def finalize_manifest(args: argparse.Namespace) -> None:
    path = Path(args.output)
    manifest = load_manifest(path)
    manifest["completed_at"] = utc_now()
    manifest["exit_code"] = args.exit_code
    manifest["summary"] = summarize_manifest(manifest)
    write_manifest(path, manifest)


def summarize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    suites = manifest.get("suites", [])
    status_counts: dict[str, int] = {}
    collected_nodeids = 0
    failed_suites = []
    for suite in suites:
        status = str(suite.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        collection = suite.get("collection") or {}
        collected_nodeids += len(collection.get("nodeids") or [])
        if status not in {"passed", "skipped"}:
            failed_suites.append(suite.get("name"))

    return {
        "suite_count": len(suites),
        "status_counts": status_counts,
        "collected_nodeid_count": collected_nodeids,
        "failed_suites": failed_suites,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--output", required=True)
    init_parser.add_argument("--repo-root", required=True)
    init_parser.add_argument("--external-repos-root", required=True)
    init_parser.set_defaults(func=init_manifest)

    suite_parser = subparsers.add_parser("record-suite")
    suite_parser.add_argument("--output", required=True)
    suite_parser.add_argument("--name", required=True)
    suite_parser.add_argument("--mode", required=True, choices=("blocking", "warn-only"))
    suite_parser.add_argument("--kind", required=True)
    suite_parser.add_argument("--status", required=True)
    suite_parser.add_argument("--exit-code", required=True, type=int)
    suite_parser.add_argument("--started-at", required=True)
    suite_parser.add_argument("--ended-at", required=True)
    suite_parser.add_argument("--command-json", required=True)
    suite_parser.add_argument("--compose-file", default="")
    suite_parser.add_argument("--compose-project", default="")
    suite_parser.add_argument("--host-ports", default="")
    suite_parser.set_defaults(func=record_suite)

    collection_parser = subparsers.add_parser("record-collection")
    collection_parser.add_argument("--output", required=True)
    collection_parser.add_argument("--name", required=True)
    collection_parser.add_argument("--exit-code", required=True, type=int)
    collection_parser.add_argument("--collection-output", required=True)
    collection_parser.set_defaults(func=record_collection)

    compose_parser = subparsers.add_parser("record-compose-project")
    compose_parser.add_argument("--output", required=True)
    compose_parser.add_argument("--group", required=True)
    compose_parser.add_argument("--project", required=True)
    compose_parser.add_argument("--compose-file", required=True)
    compose_parser.add_argument("--host-ports", default="")
    compose_parser.set_defaults(func=record_compose_project)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--output", required=True)
    finalize_parser.add_argument("--exit-code", required=True, type=int)
    finalize_parser.set_defaults(func=finalize_manifest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
