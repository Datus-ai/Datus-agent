#!/usr/bin/env python3
"""Classify nightly failures into routing-friendly structured categories."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import check_flaky_registry

CATEGORIES = (
    "product_regression",
    "provider_outage",
    "docker_infra",
    "known_flaky",
    "unknown_flaky",
    "harness_setup",
    "credential_or_config",
    "unknown_failure",
)

PROVIDER_SUITE_PATTERNS = (re.compile(r"\b(provider_health|Provider Health Tests)\b", re.IGNORECASE),)
PROVIDER_ERROR_PATTERNS = (
    re.compile(
        r"\b(429|529|rate limit|quota|overload|ProviderError|APITimeout|APIConnectionError)\b",
        re.IGNORECASE,
    ),
)
DOCKER_SUITE_PATTERNS = (
    re.compile(r"\b(compose|container|service health|readiness|unhealthy|timed out waiting)\b", re.IGNORECASE),
)
DOCKER_ERROR_PATTERNS = (
    re.compile(r"\b(Host port is already in use|No container found|Docker .* required)\b", re.IGNORECASE),
    re.compile(r"\b(service health|readiness|unhealthy|timed out waiting)\b", re.IGNORECASE),
)
CREDENTIAL_PATTERNS = (
    re.compile(r"\b(unauthorized|forbidden|authentication failed|invalid api key)\b", re.IGNORECASE),
    re.compile(r"\b(missing|not set|empty|invalid)\s+(api key|credential|secret|token|password)\b", re.IGNORECASE),
    re.compile(r"\b[A-Z0-9_]+_(KEY|TOKEN|SECRET|PASSWORD)\b.*\b(missing|not set|empty|invalid)\b", re.IGNORECASE),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def read_text(path: Path | None) -> str:
    if not path or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path | None) -> tuple[dict[str, Any] | None, str]:
    if not path or not path.exists():
        return None, "missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    if not isinstance(data, dict):
        return None, f"expected json object, got {type(data).__name__}"
    return data, ""


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)


def tail_lines(text: str, limit: int = 80) -> list[str]:
    lines = strip_ansi(text).replace("\r", "").splitlines()
    return lines[-limit:]


def command_text(suite: dict[str, Any]) -> str:
    command = suite.get("command") or []
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return str(command)


def add_finding(
    findings: list[dict[str, Any]],
    *,
    category: str,
    title: str,
    severity: str,
    blocking: bool,
    source: str,
    details: dict[str, Any] | None = None,
) -> None:
    if category not in CATEGORIES:
        category = "unknown_failure"
    finding = {
        "category": category,
        "title": title,
        "severity": severity,
        "blocking": blocking,
        "source": source,
    }
    if details:
        finding["details"] = details
    findings.append(finding)


def match_any(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def suite_category(suite: dict[str, Any]) -> str:
    name = str(suite.get("name") or "")
    kind = str(suite.get("kind") or "")
    mode = str(suite.get("mode") or "")
    text = " ".join([name, kind, mode, command_text(suite)])

    if name in {"Flaky Registry Check", "Flaky Log Classification"}:
        return "harness_setup"
    if match_any(PROVIDER_SUITE_PATTERNS, text):
        return "provider_outage"
    if kind == "compose" or match_any(DOCKER_SUITE_PATTERNS, text):
        return "docker_infra"
    return "product_regression"


def classify_flaky_registry(
    registry_path: Path,
    log_file: Path | None,
    findings: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if not log_file or not log_file.exists():
        return errors

    try:
        registry = check_flaky_registry.validate_registry(
            check_flaky_registry.load_registry(registry_path), strict=False
        )
        registered, unregistered, patterns = check_flaky_registry.classify_log(log_file, registry)
    except Exception as exc:  # noqa: BLE001 - classifier must preserve failure details without failing nightly.
        errors.append(str(exc))
        add_finding(
            findings,
            category="harness_setup",
            title="Flaky registry classification failed",
            severity="warning",
            blocking=False,
            source="flaky_registry",
            details={"error": str(exc), "registry": str(registry_path)},
        )
        return errors

    for nodeid in sorted(set(registered)):
        add_finding(
            findings,
            category="known_flaky",
            title="Registered rerun observed",
            severity="warning",
            blocking=False,
            source="flaky_registry",
            details={"nodeid": nodeid},
        )
    for entry_id in sorted(set(patterns)):
        add_finding(
            findings,
            category="known_flaky",
            title="Registered log warning observed",
            severity="warning",
            blocking=False,
            source="flaky_registry",
            details={"entry_id": entry_id},
        )
    for nodeid in sorted(set(unregistered)):
        add_finding(
            findings,
            category="unknown_flaky",
            title="Unregistered rerun observed",
            severity="warning",
            blocking=False,
            source="flaky_registry",
            details={"nodeid": nodeid},
        )
    return errors


def classify_suites(manifest: dict[str, Any] | None, log_text: str, findings: list[dict[str, Any]]) -> None:
    if not manifest:
        return

    for suite in manifest.get("suites") or []:
        name = str(suite.get("name") or "unknown suite")
        mode = str(suite.get("mode") or "blocking")
        status = str(suite.get("status") or "unknown")
        exit_code = suite.get("exit_code")
        collection = suite.get("collection") or {}
        collection_status = collection.get("status")

        if collection_status == "failed":
            add_finding(
                findings,
                category="harness_setup",
                title=f"{name} collection failed",
                severity="error" if mode == "blocking" else "warning",
                blocking=mode == "blocking",
                source="manifest.collection",
                details={
                    "suite": name,
                    "collection_exit_code": collection.get("exit_code"),
                    "raw_tail": collection.get("raw_tail") or [],
                },
            )
            if status not in {"passed", "skipped"}:
                continue

        if status in {"passed", "skipped"}:
            continue

        category = suite_category(suite)
        blocking = mode == "blocking"
        add_finding(
            findings,
            category=category,
            title=f"{name} failed",
            severity="error" if blocking else "warning",
            blocking=blocking,
            source="manifest.suite",
            details={
                "suite": name,
                "mode": mode,
                "kind": suite.get("kind"),
                "exit_code": exit_code,
                "started_at": suite.get("started_at"),
                "ended_at": suite.get("ended_at"),
                "command": suite.get("command") or [],
                "compose": suite.get("compose"),
            },
        )


def classify_log_patterns(log_text: str, findings: list[dict[str, Any]]) -> None:
    if not log_text:
        return
    if match_any(CREDENTIAL_PATTERNS, log_text):
        add_finding(
            findings,
            category="credential_or_config",
            title="Credential or configuration error pattern observed",
            severity="error",
            blocking=True,
            source="log.pattern",
            details={"log_tail": tail_lines(log_text, 40)},
        )
    if match_any(DOCKER_ERROR_PATTERNS, log_text):
        add_finding(
            findings,
            category="docker_infra",
            title="Docker or service readiness error pattern observed",
            severity="error",
            blocking=True,
            source="log.pattern",
            details={"log_tail": tail_lines(log_text, 40)},
        )
    if match_any(PROVIDER_ERROR_PATTERNS, log_text):
        add_finding(
            findings,
            category="provider_outage",
            title="Provider failure pattern observed",
            severity="warning",
            blocking=False,
            source="log.pattern",
            details={"log_tail": tail_lines(log_text, 40)},
        )


def summarize_findings(findings: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts = Counter(str(finding.get("category")) for finding in findings)
    blocking_category_counts = Counter(
        str(finding.get("category")) for finding in findings if bool(finding.get("blocking"))
    )
    severity_counts = Counter(str(finding.get("severity")) for finding in findings)
    return {
        "finding_count": len(findings),
        "category_counts": dict(sorted(category_counts.items())),
        "blocking_category_counts": dict(sorted(blocking_category_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
    }


def build_classification(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest) if args.manifest else None
    log_file = Path(args.log_file) if args.log_file else None
    registry_path = Path(args.registry)
    manifest, manifest_error = read_json(manifest_path)
    log_text = read_text(log_file)
    findings: list[dict[str, Any]] = []
    classifier_errors: list[str] = []

    if manifest_error:
        add_finding(
            findings,
            category="harness_setup",
            title="Nightly manifest unavailable",
            severity="error",
            blocking=True,
            source="manifest",
            details={"manifest": str(manifest_path), "error": manifest_error},
        )

    classifier_errors.extend(classify_flaky_registry(registry_path, log_file, findings))
    classify_suites(manifest, log_text, findings)
    classify_log_patterns(log_text, findings)

    if args.exit_code != 0 and not any(finding.get("blocking") for finding in findings):
        add_finding(
            findings,
            category="unknown_failure",
            title="Nightly failed without a classified blocking finding",
            severity="error",
            blocking=True,
            source="exit_code",
            details={"exit_code": args.exit_code, "log_tail": tail_lines(log_text, 80)},
        )

    status = "passed" if args.exit_code == 0 else "failed"
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "status": status,
        "exit_code": args.exit_code,
        "sources": {
            "manifest": str(manifest_path) if manifest_path else "",
            "log_file": str(log_file) if log_file else "",
            "registry": str(registry_path),
        },
        "summary": summarize_findings(findings),
        "findings": findings,
        "classifier_errors": classifier_errors,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="nightly-manifest.json")
    parser.add_argument("--log-file", default="")
    parser.add_argument("--registry", default="ci/flaky-registry.yml")
    parser.add_argument("--output", default="nightly-failure-classification.json")
    parser.add_argument("--exit-code", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    classification = build_classification(args)
    write_json(Path(args.output), classification)
    return 0


if __name__ == "__main__":
    sys.exit(main())
