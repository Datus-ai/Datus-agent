#!/usr/bin/env python3
"""Verify that nightly adapter packages came from the checked-out repositories."""

from __future__ import annotations

import argparse
import importlib
import json
from importlib import metadata
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

EXPECTED_LOCAL_PACKAGES = {
    "datus-db-core": "datus-db-adapters/datus-db-core",
    "datus-sqlalchemy": "datus-db-adapters/datus-sqlalchemy",
    "datus-snowflake": "datus-db-adapters/datus-snowflake",
    "datus-postgresql": "datus-db-adapters/datus-postgresql",
    "datus-mysql": "datus-db-adapters/datus-mysql",
    "datus-clickhouse": "datus-db-adapters/datus-clickhouse",
    "datus-starrocks": "datus-db-adapters/datus-starrocks",
    "datus-trino": "datus-db-adapters/datus-trino",
    "datus-greenplum": "datus-db-adapters/datus-greenplum",
    "datus-hive": "datus-db-adapters/datus-hive",
    "datus-spark": "datus-db-adapters/datus-spark",
    "datus-bi-core": "datus-bi-adapters/datus-bi-core",
    "datus-bi-superset": "datus-bi-adapters/datus-bi-superset",
    "datus-bi-grafana": "datus-bi-adapters/datus-bi-grafana",
    "datus-scheduler-core": "datus-scheduler-adapters/datus-scheduler-core",
    "datus-scheduler-airflow": "datus-scheduler-adapters/datus-scheduler-airflow",
    "datus-semantic-core": "datus-semantic-adapter/datus-semantic-core",
    "datus-semantic-metricflow": "datus-semantic-adapter/datus-semantic-metricflow",
    "datus-storage-base": "datus-storage-adapters/datus-storage-base",
    "datus-storage-postgresql": "datus-storage-adapters/datus-storage-postgresql",
}


def distribution_source_path(package_name: str) -> tuple[Path | None, str | None]:
    try:
        dist = metadata.distribution(package_name)
    except metadata.PackageNotFoundError:
        return None, "package is not installed"

    direct_url_text = dist.read_text("direct_url.json")
    if not direct_url_text:
        return None, "package has no direct_url.json and was likely installed from a registry"

    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as exc:
        return None, f"package has invalid direct_url.json: {exc}"

    source_url = direct_url.get("url")
    if not isinstance(source_url, str) or not source_url:
        return None, "package direct_url.json has no source URL"

    parsed = urlparse(source_url)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        return None, f"package source is not a local checkout: {source_url}"

    return Path(url2pathname(parsed.path)).resolve(), None


def verify_local_sources(external_repos_root: Path) -> list[str]:
    external_repos_root = external_repos_root.resolve()
    errors: list[str] = []

    for package_name, relative_path in EXPECTED_LOCAL_PACKAGES.items():
        expected_path = (external_repos_root / relative_path).resolve()
        source_path, error = distribution_source_path(package_name)
        if error:
            errors.append(f"{package_name}: {error}")
        elif source_path != expected_path:
            errors.append(f"{package_name}: expected checkout source {expected_path}, got {source_path}")

    return errors


def verify_semantic_adapter_imports() -> list[str]:
    errors: list[str] = []
    try:
        core_models = importlib.import_module("datus_semantic_core.models")
    except Exception as exc:  # noqa: BLE001 - report the actual nightly import failure.
        errors.append(f"datus-semantic-core import failed: {exc}")
    else:
        if not hasattr(core_models, "SemanticValidationError"):
            errors.append("datus-semantic-core is missing SemanticValidationError")

    try:
        importlib.import_module("datus_semantic_metricflow")
    except Exception as exc:  # noqa: BLE001 - report the actual nightly import failure.
        errors.append(f"datus-semantic-metricflow import failed: {exc}")

    return errors


def verify_storage_adapter_imports() -> list[str]:
    errors: list[str] = []
    required_names = ("FtsField", "FtsIndexStatus", "FtsSpec", "normalize_fts_spec")
    fts_contract = None
    try:
        fts_contract = importlib.import_module("datus_storage_base.vector.fts")
    except Exception as exc:  # noqa: BLE001 - report the actual nightly import failure.
        errors.append(f"datus-storage-base FTS import failed: {exc}")
    else:
        missing_names = [name for name in required_names if not hasattr(fts_contract, name)]
        if missing_names:
            errors.append(f"datus-storage-base FTS contract is missing: {', '.join(missing_names)}")

    try:
        agent_fts = importlib.import_module("datus.storage.fts")
    except Exception as exc:  # noqa: BLE001 - report the actual nightly import failure.
        errors.append(f"Datus Agent FTS import failed: {exc}")
    else:
        mismatched_names = [
            name
            for name in required_names
            if fts_contract is not None
            and hasattr(fts_contract, name)
            and getattr(agent_fts, name, None) is not getattr(fts_contract, name)
        ]
        if mismatched_names:
            errors.append(
                f"Datus Agent FTS contract does not re-export datus-storage-base: {', '.join(mismatched_names)}"
            )

    try:
        vector_adapter = importlib.import_module("datus_storage_postgresql.vector")
    except Exception as exc:  # noqa: BLE001 - report the actual nightly import failure.
        errors.append(f"datus-storage-postgresql vector import failed: {exc}")
    else:
        if not hasattr(vector_adapter, "PgvectorBackend"):
            errors.append("datus-storage-postgresql vector adapter is missing PgvectorBackend")

    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-repos-root",
        type=Path,
        required=True,
        help="Directory containing the checked-out adapter repositories.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    errors = verify_local_sources(args.external_repos_root)
    errors.extend(verify_semantic_adapter_imports())
    errors.extend(verify_storage_adapter_imports())
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"Verified {len(EXPECTED_LOCAL_PACKAGES)} adapter packages from {args.external_repos_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
