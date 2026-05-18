#!/usr/bin/env python3
"""Generate the Datus provider coverage manifest."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def read_json(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError(f"expected list[str], got {value!r}")


def append_error(errors: list[str], message: str) -> None:
    if message not in errors:
        errors.append(message)


def normalize_list_or_error(value: Any, errors: list[str], field_name: str) -> list[str]:
    try:
        return normalize_list(value)
    except ValueError as exc:
        append_error(errors, f"{field_name}: {exc}")
        return []


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


def validate_coverage_config(repo_root: Path, catalog: dict[str, Any], coverage: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if coverage.get("version") != 1:
        errors.append("provider coverage version must be 1")

    catalog_providers = set((catalog.get("providers") or {}).keys())
    declarations = coverage.get("providers")
    if not isinstance(declarations, dict):
        errors.append("provider coverage 'providers' must be a mapping")
        return errors

    for provider_id, declaration in declarations.items():
        if provider_id not in catalog_providers:
            errors.append(f"{provider_id}: declared provider is not present in conf/providers.yml")
            continue
        if not isinstance(declaration, dict):
            errors.append(f"{provider_id}: provider declaration must be a mapping")
            continue
        for section_name in ("deterministic", "live_provider_health"):
            section = declaration.get(section_name) or {}
            if not isinstance(section, dict):
                errors.append(f"{provider_id}.{section_name}: must be a mapping")
                continue
            nodeids = normalize_list_or_error(
                section.get("nodeids"),
                errors,
                f"{provider_id}.{section_name}.nodeids",
            )
            for nodeid in nodeids:
                if not class_or_function_exists(repo_root, nodeid):
                    errors.append(f"{provider_id}.{section_name}: nodeid target does not exist: {nodeid}")
        required_env = normalize_list_or_error(declaration.get("required_env"), errors, f"{provider_id}.required_env")
        for env_name in required_env:
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", env_name):
                errors.append(f"{provider_id}.required_env: invalid env var name: {env_name}")

    return errors


def provider_health_suite(manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    if not manifest:
        return None
    for suite in manifest.get("suites") or []:
        if suite.get("name") == "Provider Health Tests":
            return suite
    return None


def collected_nodeids_for(suite: dict[str, Any] | None, errors: list[str] | None = None) -> list[str]:
    if not suite:
        return []
    collection = suite.get("collection") or {}
    if errors is None:
        return normalize_list(collection.get("nodeids"))
    return normalize_list_or_error(collection.get("nodeids"), errors, "nightly.provider_health.collection.nodeids")


def match_declared_nodeids(declared: list[str], collected: list[str]) -> list[str]:
    matched: list[str] = []
    for nodeid in collected:
        if any(nodeid == declared_nodeid or nodeid.startswith(f"{declared_nodeid}::") for declared_nodeid in declared):
            matched.append(nodeid)
    return matched


def provider_required_env(
    provider_catalog: dict[str, Any],
    declaration: dict[str, Any],
    coverage_errors: list[str],
    provider_id: str,
) -> list[str]:
    env_names = []
    catalog_env = provider_catalog.get("api_key_env")
    if isinstance(catalog_env, str) and catalog_env:
        env_names.append(catalog_env)
    env_names.extend(
        normalize_list_or_error(declaration.get("required_env"), coverage_errors, f"{provider_id}.required_env")
    )
    return sorted(set(env_names))


def coverage_status(nodeids: list[str], inherited_from: str = "") -> str:
    if inherited_from:
        return "inherited"
    if nodeids:
        return "covered"
    return "missing"


def build_provider_entry(
    provider_id: str,
    provider_catalog: dict[str, Any],
    declaration: dict[str, Any],
    provider_suite: dict[str, Any] | None,
    collected_provider_nodeids: list[str],
    coverage_errors: list[str],
) -> dict[str, Any]:
    deterministic = declaration.get("deterministic") or {}
    live = declaration.get("live_provider_health") or {}
    deterministic_nodeids = normalize_list_or_error(
        deterministic.get("nodeids"),
        coverage_errors,
        f"{provider_id}.deterministic.nodeids",
    )
    live_nodeids = normalize_list_or_error(
        live.get("nodeids"),
        coverage_errors,
        f"{provider_id}.live_provider_health.nodeids",
    )
    matched_live_nodeids = match_declared_nodeids(live_nodeids, collected_provider_nodeids)
    live_mode = str(live.get("mode") or ("warn-only" if live_nodeids else "not-tested"))

    live_status = "not_declared"
    if live_nodeids:
        live_status = "declared"
        if provider_suite:
            live_status = "collected" if matched_live_nodeids else "declared_not_collected"
    elif live_mode == "not-tested":
        live_status = "not_tested"

    return {
        "id": provider_id,
        "type": provider_catalog.get("type", ""),
        "auth_type": provider_catalog.get("auth_type", "api_key"),
        "base_url": provider_catalog.get("base_url", ""),
        "default_model": provider_catalog.get("default_model", ""),
        "models": normalize_list_or_error(provider_catalog.get("models"), coverage_errors, f"{provider_id}.models"),
        "models_from": provider_catalog.get("models_from", ""),
        "required_env": provider_required_env(provider_catalog, declaration, coverage_errors, provider_id),
        "coverage": {
            "deterministic": {
                "status": coverage_status(deterministic_nodeids, str(deterministic.get("inherits_from") or "")),
                "inherits_from": deterministic.get("inherits_from", ""),
                "nodeids": deterministic_nodeids,
            },
            "live_provider_health": {
                "status": live_status,
                "mode": live_mode,
                "nodeids": live_nodeids,
                "collected_nodeids": matched_live_nodeids,
            },
        },
    }


def summarize(entries: list[dict[str, Any]], coverage_errors: list[str]) -> dict[str, Any]:
    deterministic_covered = [
        entry["id"] for entry in entries if entry["coverage"]["deterministic"]["status"] in {"covered", "inherited"}
    ]
    live_declared = [
        entry["id"]
        for entry in entries
        if entry["coverage"]["live_provider_health"]["status"] in {"declared", "collected", "declared_not_collected"}
    ]
    live_collected = [
        entry["id"] for entry in entries if entry["coverage"]["live_provider_health"]["status"] == "collected"
    ]
    missing_declared = [
        entry["id"]
        for entry in entries
        if entry["coverage"]["deterministic"]["status"] == "missing"
        and entry["coverage"]["live_provider_health"]["status"] in {"not_declared", "not_tested"}
    ]
    deterministic_missing = [
        entry["id"] for entry in entries if entry["coverage"]["deterministic"]["status"] == "missing"
    ]
    live_missing = [
        entry["id"]
        for entry in entries
        if entry["coverage"]["live_provider_health"]["status"] in {"not_declared", "not_tested"}
    ]
    return {
        "providers_total": len(entries),
        "deterministic_covered": len(deterministic_covered),
        "live_provider_health_declared": len(live_declared),
        "live_provider_health_collected": len(live_collected),
        "deterministic_missing": deterministic_missing,
        "live_provider_health_missing": live_missing,
        "missing_declared_coverage": missing_declared,
        "coverage_error_count": len(coverage_errors),
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    provider_catalog_path = Path(args.provider_catalog)
    coverage_config_path = Path(args.coverage_config)
    nightly_manifest = read_json(Path(args.nightly_manifest)) if args.nightly_manifest else None
    catalog = read_yaml(provider_catalog_path)
    coverage = read_yaml(coverage_config_path)
    coverage_errors = validate_coverage_config(repo_root, catalog, coverage)
    catalog_providers = catalog.get("providers") or {}
    declarations = coverage.get("providers") or {}
    suite = provider_health_suite(nightly_manifest)
    collected_provider_nodeids = collected_nodeids_for(suite, coverage_errors)

    providers = []
    for provider_id in sorted(catalog_providers):
        provider_catalog = catalog_providers[provider_id] or {}
        declaration = declarations.get(provider_id) or {}
        providers.append(
            build_provider_entry(
                provider_id,
                provider_catalog,
                declaration,
                suite,
                collected_provider_nodeids,
                coverage_errors,
            )
        )

    missing_declarations = sorted(set(catalog_providers) - set(declarations))
    for provider_id in missing_declarations:
        coverage_errors.append(f"{provider_id}: provider has no coverage declaration")

    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "catalog": {
            "source": str(provider_catalog_path),
            "provider_count": len(catalog_providers),
        },
        "coverage_config": {
            "source": str(coverage_config_path),
            "declared_provider_count": len(declarations),
        },
        "nightly": {
            "manifest": str(args.nightly_manifest or ""),
            "provider_health_suite_status": suite.get("status") if suite else "",
            "provider_health_suite_mode": suite.get("mode") if suite else "",
            "provider_health_collected_nodeids": len(collected_provider_nodeids),
        },
        "providers": providers,
        "summary": summarize(providers, coverage_errors),
        "coverage_errors": coverage_errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--provider-catalog", default="conf/providers.yml")
    parser.add_argument("--coverage-config", default="ci/provider-coverage.yml")
    parser.add_argument("--nightly-manifest", default="nightly-manifest.json")
    parser.add_argument("--output", default="provider-coverage-manifest.json")
    parser.add_argument("--strict", action="store_true", help="Fail when coverage errors are present")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_manifest(args)
    write_json(Path(args.output), manifest)
    if args.strict and manifest["coverage_errors"]:
        for error in manifest["coverage_errors"]:
            print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
