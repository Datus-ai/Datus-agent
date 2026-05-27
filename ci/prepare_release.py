#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Prepare source metadata changes for a datus-agent release PR."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from packaging.version import Version

CI_DIR = Path(__file__).resolve().parent
if str(CI_DIR) not in sys.path:
    sys.path.insert(0, str(CI_DIR))

from check_release_readiness import (  # noqa: E402
    ADAPTER_CORE_PACKAGES,
    REPO_ROOT,
    check_tag_available,
    fetch_latest_pypi_version,
    read_pyproject_version,
)


def parse_canonical_version(raw_version: str) -> Version:
    version = Version(raw_version)
    if str(version) != raw_version:
        raise ValueError(f"Use canonical PEP 440 version {version!s} instead of {raw_version!r}")
    return version


def is_same_final_prerelease(target_version: Version, current_version: Version) -> bool:
    return (
        target_version.is_prerelease
        and not current_version.is_prerelease
        and Version(target_version.base_version) == current_version
    )


def ensure_version_can_advance(
    repo_root: Path,
    target_version: Version,
    *,
    allow_current_version: bool = False,
    allow_same_final_prerelease: bool = False,
) -> None:
    current_version = read_pyproject_version(repo_root)
    same_final_prerelease = is_same_final_prerelease(target_version, current_version)
    if target_version < current_version:
        if not (allow_same_final_prerelease and same_final_prerelease):
            raise ValueError(f"Release version must advance current version {current_version}; got {target_version}")
        stable_tag_errors = check_tag_available(repo_root, Version(target_version.base_version))
        if stable_tag_errors:
            raise ValueError(
                f"Cannot prepare prerelease {target_version} from finalized {current_version}: {stable_tag_errors[0]}"
            )
    elif target_version == current_version and not allow_current_version:
        raise ValueError(f"Release version must advance current version {current_version}; got {target_version}")

    tag_errors = check_tag_available(repo_root, target_version)
    if tag_errors:
        raise ValueError(tag_errors[0])


def update_project_version(pyproject_path: Path, version: Version) -> bool:
    lines = pyproject_path.read_text(encoding="utf-8").splitlines(keepends=True)
    in_project = False
    changed = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "[project]":
            in_project = True
            continue
        if in_project and stripped.startswith("[") and stripped.endswith("]"):
            break
        if in_project and line.startswith("version = "):
            replacement = f'version = "{version}"\n'
            if line != replacement:
                lines[index] = replacement
                changed = True
            break
    else:
        raise ValueError(f"Unable to find [project] version in {pyproject_path}")

    if changed:
        pyproject_path.write_text("".join(lines), encoding="utf-8")
    return changed


def update_dependency_lower_bounds(path: Path, bounds: dict[str, Version], *, quoted: bool) -> bool:
    content = path.read_text(encoding="utf-8")
    updated = content
    for package_name, version in bounds.items():
        if quoted:
            pattern = rf'("{re.escape(package_name)})(?:[^"]*)(")'
            replacement = rf"\1>={version}\2"
        else:
            pattern = rf"(?m)^({re.escape(package_name)})(?:[^\n#]*)$"
            replacement = rf"\1>={version}"
        updated, count = re.subn(pattern, replacement, updated, count=1)
        if count != 1:
            raise ValueError(f"Unable to update dependency lower bound for {package_name} in {path}")

    if updated != content:
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def latest_adapter_bounds(timeout: float, allow_prerelease: bool) -> dict[str, Version]:
    return {
        package_name: fetch_latest_pypi_version(package_name, timeout=timeout, allow_prerelease=allow_prerelease)
        for package_name in ADAPTER_CORE_PACKAGES
    }


def prepare_release(
    repo_root: Path,
    version: Version,
    *,
    update_adapter_bounds: bool,
    pypi_timeout: float = 10.0,
    allow_prerelease: bool = False,
) -> list[Path]:
    changed: list[Path] = []

    pyproject_path = repo_root / "pyproject.toml"
    requirements_path = repo_root / "requirements.txt"

    if update_project_version(pyproject_path, version):
        changed.append(pyproject_path)

    if update_adapter_bounds:
        bounds = latest_adapter_bounds(timeout=pypi_timeout, allow_prerelease=allow_prerelease)
        if update_dependency_lower_bounds(pyproject_path, bounds, quoted=True):
            changed.append(pyproject_path)
        if update_dependency_lower_bounds(requirements_path, bounds, quoted=False):
            changed.append(requirements_path)

    return sorted(set(changed))


def git_has_diff(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet"],
        cwd=repo_root,
        check=False,
    )
    return result.returncode == 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root to update")
    parser.add_argument("--version", required=True, help="New canonical PEP 440 datus-agent version")
    parser.add_argument(
        "--update-adapter-bounds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update adapter core dependency lower bounds to latest PyPI releases",
    )
    parser.add_argument(
        "--allow-prerelease",
        action="store_true",
        help="Allow prerelease adapter versions when updating adapter core lower bounds",
    )
    parser.add_argument(
        "--allow-current-version",
        action="store_true",
        help="Allow preparing a release branch that already has the target version",
    )
    parser.add_argument(
        "--allow-same-final-prerelease",
        action="store_true",
        help="Allow preparing 0.3.2rcN from current 0.3.2 when the stable tag is not present",
    )
    parser.add_argument(
        "--allow-no-changes",
        action="store_true",
        help="Exit successfully when release metadata is already prepared",
    )
    parser.add_argument("--pypi-timeout", type=float, default=10.0, help="PyPI request timeout in seconds")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        version = parse_canonical_version(args.version)
        ensure_version_can_advance(
            repo_root,
            version,
            allow_current_version=args.allow_current_version,
            allow_same_final_prerelease=args.allow_same_final_prerelease,
        )
        changed = prepare_release(
            repo_root,
            version,
            update_adapter_bounds=args.update_adapter_bounds,
            pypi_timeout=args.pypi_timeout,
            allow_prerelease=args.allow_prerelease,
        )
    except Exception as exc:
        print(f"Release preparation failed: {exc}", file=sys.stderr)
        return 1

    if not changed or not git_has_diff(repo_root):
        if args.allow_no_changes:
            print(f"Release metadata is already prepared for datus-agent {version}")
            return 0
        print(f"Release preparation produced no changes for {version}", file=sys.stderr)
        return 1

    print(f"Prepared release metadata for datus-agent {version}")
    print("Changed files:")
    for path in changed:
        print(f"  - {path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
