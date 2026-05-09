#!/usr/bin/env python3
"""Validate flaky ownership and classify nightly reruns/log warnings."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = REPO_ROOT / "ci" / "flaky-registry.yml"
VALID_TYPES = {"test", "log_pattern"}
VALID_LAYERS = {
    "adapter",
    "benchmark",
    "component",
    "harness_correctness",
    "known_flaky",
    "llm_harness",
    "product_e2e",
    "provider_health",
}
REQUIRED_FIELDS = {"id", "type", "owner", "layer", "reason", "allowed_until"}
RERUN_RE = re.compile(r"\bRERUN\b\s+(?P<nodeid>tests/[^\s]+)")


class Registry(NamedTuple):
    entries: list[dict[str, Any]]
    test_nodeids: set[str]
    log_patterns: list[tuple[str, re.Pattern[str]]]


def load_registry(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Flaky registry must be a mapping: {path}")
    return data


def parse_date(value: Any, field_name: str) -> date:
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date string: {value!r}") from exc


def _entry_path_exists(nodeid: str) -> bool:
    test_path = nodeid.split("::", 1)[0]
    return (REPO_ROOT / test_path).exists()


def validate_registry(data: dict[str, Any], *, today: date | None = None, strict: bool = False) -> Registry:
    today = today or date.today()
    if data.get("version") != 1:
        raise ValueError("Flaky registry version must be 1")

    raw_entries = data.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("Flaky registry 'entries' must be a list")

    ids: set[str] = set()
    test_nodeids: set[str] = set()
    log_patterns: list[tuple[str, re.Pattern[str]]] = []

    for idx, entry in enumerate(raw_entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry #{idx} must be a mapping")
        missing = REQUIRED_FIELDS - set(entry)
        if missing:
            raise ValueError(f"Entry #{idx} missing required field(s): {', '.join(sorted(missing))}")

        entry_id = str(entry["id"])
        if entry_id in ids:
            raise ValueError(f"Duplicate flaky entry id: {entry_id}")
        ids.add(entry_id)

        entry_type = entry["type"]
        if entry_type not in VALID_TYPES:
            raise ValueError(f"{entry_id}: type must be one of {sorted(VALID_TYPES)}")
        if entry["layer"] not in VALID_LAYERS:
            raise ValueError(f"{entry_id}: layer must be one of {sorted(VALID_LAYERS)}")

        expires = parse_date(entry["allowed_until"], f"{entry_id}.allowed_until")
        if expires < today:
            raise ValueError(f"{entry_id}: flaky entry expired on {expires.isoformat()}")

        allowed_in = entry.get("allowed_in", [])
        if not isinstance(allowed_in, list) or not allowed_in or not all(isinstance(item, str) for item in allowed_in):
            raise ValueError(f"{entry_id}: allowed_in must be a non-empty list of strings")

        if entry_type == "test":
            nodeid = entry.get("nodeid")
            if not isinstance(nodeid, str) or "::" not in nodeid:
                raise ValueError(f"{entry_id}: test entries require a pytest nodeid")
            if strict and not _entry_path_exists(nodeid):
                raise ValueError(f"{entry_id}: test path does not exist for nodeid {nodeid}")
            test_nodeids.add(nodeid)
        else:
            pattern = entry.get("pattern")
            if not isinstance(pattern, str) or not pattern:
                raise ValueError(f"{entry_id}: log_pattern entries require pattern")
            try:
                compiled = re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"{entry_id}: invalid regex pattern: {exc}") from exc
            log_patterns.append((entry_id, compiled))

    return Registry(entries=raw_entries, test_nodeids=test_nodeids, log_patterns=log_patterns)


def _matches_registered_nodeid(nodeid: str, registered: set[str]) -> bool:
    if nodeid in registered:
        return True
    normalized = re.sub(r"\[[^\]]+\]$", "", nodeid)
    return normalized in registered


def classify_log(log_file: Path, registry: Registry) -> tuple[list[str], list[str], list[str]]:
    text = log_file.read_text(encoding="utf-8", errors="replace")

    unregistered_reruns: list[str] = []
    registered_reruns: list[str] = []
    for match in RERUN_RE.finditer(text):
        nodeid = match.group("nodeid")
        if _matches_registered_nodeid(nodeid, registry.test_nodeids):
            registered_reruns.append(nodeid)
        else:
            unregistered_reruns.append(nodeid)

    matched_patterns: list[str] = []
    for entry_id, pattern in registry.log_patterns:
        if pattern.search(text):
            matched_patterns.append(entry_id)

    return registered_reruns, unregistered_reruns, matched_patterns


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Path to ci/flaky-registry.yml")
    parser.add_argument("--log-file", help="Nightly log file to classify after tests finish")
    parser.add_argument("--strict", action="store_true", help="Validate test nodeid paths exist")
    args = parser.parse_args(argv)

    try:
        registry_path = Path(args.registry)
        registry = validate_registry(load_registry(registry_path), strict=args.strict)
        print(f"Validated {len(registry.entries)} flaky registry entries from {registry_path}")

        if args.log_file:
            registered, unregistered, patterns = classify_log(Path(args.log_file), registry)
            if registered:
                print("Registered reruns:")
                for nodeid in sorted(set(registered)):
                    print(f"  - {nodeid}")
            if patterns:
                print("Registered log warning patterns observed:")
                for entry_id in sorted(set(patterns)):
                    print(f"  - {entry_id}")
            if unregistered:
                print("Unregistered reruns detected:")
                for nodeid in sorted(set(unregistered)):
                    print(f"  - {nodeid}")
                return 1
        return 0
    except Exception as exc:
        print(f"Flaky registry check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
