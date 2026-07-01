#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Validate release metadata that must stay correct for weekly releases."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import subprocess
import sys
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_CORE_PACKAGES = (
    "datus-db-core",
    "datus-semantic-core",
    "datus-bi-core",
    "datus-scheduler-core",
)
CI_DEPENDENCY_GROUP = "ci"
CI_ADAPTER_PACKAGES = (
    "datus-metricflow",
    "datus-semantic-metricflow",
)
CONSOLE_VERSION_COMMANDS = ("datus", "datus-agent", "datus-api", "datus-gateway")


@dataclass(frozen=True)
class DependencyCheck:
    name: str
    pyproject_lower_bound: Version
    requirements_lower_bound: Version
    latest_version: Version | None = None


def _read_pyproject(repo_root: Path) -> dict:
    with (repo_root / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)


def read_pyproject_version(repo_root: Path) -> Version:
    version = _read_pyproject(repo_root)["project"]["version"]
    return Version(version)


def parse_dependency_list(requirement_lines: Iterable[str]) -> dict[str, Requirement]:
    requirements: dict[str, Requirement] = {}
    for raw_line in requirement_lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        requirements[canonicalize_name(requirement.name)] = requirement
    return requirements


def read_pyproject_dependencies(repo_root: Path) -> dict[str, Requirement]:
    dependencies = _read_pyproject(repo_root)["project"].get("dependencies", [])
    return parse_dependency_list(dependencies)


def read_pyproject_dependency_group_dependencies(repo_root: Path, group_name: str) -> dict[str, Requirement]:
    dependency_groups = _read_pyproject(repo_root).get("dependency-groups", {})
    dependencies = dependency_groups.get(group_name)
    if dependencies is None:
        raise ValueError(f"pyproject.toml is missing dependency-groups.{group_name}")
    return parse_dependency_list(dependencies)


def read_requirements_dependencies(repo_root: Path) -> dict[str, Requirement]:
    requirements_path = repo_root / "requirements.txt"
    return parse_dependency_list(requirements_path.read_text(encoding="utf-8").splitlines())


def lower_bound(requirement: Requirement) -> Version | None:
    bounds = []
    for specifier in requirement.specifier:
        if specifier.operator != ">=":
            continue
        try:
            bounds.append(Version(specifier.version))
        except InvalidVersion as exc:
            raise ValueError(f"Invalid version in {requirement}: {specifier.version}") from exc
    return max(bounds) if bounds else None


def check_source_version_consistency(repo_root: Path, expected_version: str | None = None) -> list[str]:
    errors: list[str] = []
    pyproject_version = read_pyproject_version(repo_root)

    if expected_version is not None and pyproject_version != Version(expected_version):
        errors.append(f"Expected release version {expected_version}, but pyproject.toml has {pyproject_version}")

    return errors


def check_tag_available(repo_root: Path, version: Version) -> list[str]:
    tag = f"v{version}"
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}"],
        cwd=repo_root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        return [f"Release tag {tag} already exists"]
    return []


def check_installed_distribution_version(version: Version, distribution_name: str = "datus-agent") -> list[str]:
    try:
        installed_version = Version(metadata.version(distribution_name))
    except metadata.PackageNotFoundError:
        return [f"Installed distribution {distribution_name!r} was not found"]

    if installed_version != version:
        return [f"Installed distribution {distribution_name!r} has version {installed_version}, expected {version}"]
    return []


def check_console_script_versions(
    version: Version,
    commands: Iterable[str] = CONSOLE_VERSION_COMMANDS,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[str]:
    errors: list[str] = []
    for command in commands:
        try:
            result = runner(
                [command, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except OSError as exc:
            errors.append(f"{command} --version failed to execute: {exc}")
            continue
        output = f"{result.stdout}\n{result.stderr}".strip()
        if result.returncode != 0:
            errors.append(f"{command} --version failed with exit code {result.returncode}: {output}")
            continue
        if str(version) not in output:
            errors.append(f"{command} --version output {output!r} does not contain expected version {version}")
    return errors


def check_adapter_dependency_consistency(
    repo_root: Path,
    package_names: Iterable[str] = ADAPTER_CORE_PACKAGES,
) -> tuple[list[DependencyCheck], list[str]]:
    pyproject_deps = read_pyproject_dependencies(repo_root)
    requirements_deps = read_requirements_dependencies(repo_root)
    checks: list[DependencyCheck] = []
    errors: list[str] = []

    for package_name in package_names:
        normalized = canonicalize_name(package_name)
        pyproject_requirement = pyproject_deps.get(normalized)
        requirements_requirement = requirements_deps.get(normalized)
        if pyproject_requirement is None:
            errors.append(f"{package_name} is missing from pyproject.toml project.dependencies")
            continue
        if requirements_requirement is None:
            errors.append(f"{package_name} is missing from requirements.txt")
            continue

        pyproject_lower = lower_bound(pyproject_requirement)
        requirements_lower = lower_bound(requirements_requirement)
        if pyproject_lower is None:
            errors.append(f"{package_name} in pyproject.toml must declare a >= lower bound")
            continue
        if requirements_lower is None:
            errors.append(f"{package_name} in requirements.txt must declare a >= lower bound")
            continue
        if pyproject_lower != requirements_lower:
            errors.append(
                f"{package_name} lower bound mismatch: pyproject.toml has {pyproject_lower}, "
                f"requirements.txt has {requirements_lower}"
            )
            continue

        checks.append(
            DependencyCheck(
                name=package_name,
                pyproject_lower_bound=pyproject_lower,
                requirements_lower_bound=requirements_lower,
            )
        )

    return checks, errors


def check_ci_adapter_dependency_consistency(
    repo_root: Path,
    package_names: Iterable[str] = CI_ADAPTER_PACKAGES,
    group_name: str = CI_DEPENDENCY_GROUP,
) -> tuple[list[DependencyCheck], list[str]]:
    checks: list[DependencyCheck] = []
    errors: list[str] = []

    try:
        group_deps = read_pyproject_dependency_group_dependencies(repo_root, group_name)
    except ValueError as exc:
        return checks, [str(exc)]

    for package_name in package_names:
        normalized = canonicalize_name(package_name)
        requirement = group_deps.get(normalized)
        if requirement is None:
            errors.append(f"{package_name} is missing from pyproject.toml dependency-groups.{group_name}")
            continue

        group_lower = lower_bound(requirement)
        if group_lower is None:
            errors.append(
                f"{package_name} in pyproject.toml dependency-groups.{group_name} must declare a >= lower bound"
            )
            continue

        checks.append(
            DependencyCheck(
                name=package_name,
                pyproject_lower_bound=group_lower,
                requirements_lower_bound=group_lower,
            )
        )

    return checks, errors


def fetch_latest_pypi_version(package_name: str, timeout: float = 10.0, allow_prerelease: bool = False) -> Version:
    request = urllib.request.Request(
        f"https://pypi.org/pypi/{package_name}/json",
        headers={"User-Agent": "datus-release-readiness"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)

    versions = []
    for raw_version, files in payload.get("releases", {}).items():
        try:
            version = Version(raw_version)
        except InvalidVersion:
            continue
        if version.is_prerelease and not allow_prerelease:
            continue
        if not files or all(file.get("yanked", False) for file in files):
            continue
        versions.append(version)

    if not versions:
        raise ValueError(f"No usable PyPI releases found for {package_name}")
    return max(versions)


def check_adapter_dependencies_are_latest(
    dependency_checks: Iterable[DependencyCheck],
    fetch_latest: Callable[[str], Version],
) -> tuple[list[DependencyCheck], list[str]]:
    refreshed: list[DependencyCheck] = []
    errors: list[str] = []

    for check in dependency_checks:
        latest = fetch_latest(check.name)
        refreshed_check = DependencyCheck(
            name=check.name,
            pyproject_lower_bound=check.pyproject_lower_bound,
            requirements_lower_bound=check.requirements_lower_bound,
            latest_version=latest,
        )
        refreshed.append(refreshed_check)
        if check.pyproject_lower_bound != latest:
            errors.append(
                f"{check.name} lower bound must match latest PyPI release: "
                f"declared {check.pyproject_lower_bound}, latest {latest}"
            )

    return refreshed, errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root to validate")
    parser.add_argument("--expected-version", help="Expected datus-agent version for release validation")
    parser.add_argument("--check-tag-available", action="store_true", help="Fail if v<version> already exists")
    parser.add_argument(
        "--check-installed-metadata",
        action="store_true",
        help="Require the installed datus-agent distribution metadata version to match source version",
    )
    parser.add_argument(
        "--check-console-versions",
        action="store_true",
        help="Require installed console scripts to report the source version",
    )
    parser.add_argument(
        "--check-adapter-latest",
        action="store_true",
        help="Fetch PyPI metadata and require adapter lower bounds to match latest published versions",
    )
    parser.add_argument(
        "--allow-prerelease",
        action="store_true",
        help="Allow prerelease adapter versions when selecting latest PyPI releases",
    )
    parser.add_argument("--pypi-timeout", type=float, default=10.0, help="PyPI request timeout in seconds")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    errors: list[str] = []

    errors.extend(check_source_version_consistency(repo_root, expected_version=args.expected_version))
    version = read_pyproject_version(repo_root)
    if args.check_tag_available:
        errors.extend(check_tag_available(repo_root, version))
    if args.check_installed_metadata:
        errors.extend(check_installed_distribution_version(version))
    if args.check_console_versions:
        errors.extend(check_console_script_versions(version))

    core_dependency_checks, core_dependency_errors = check_adapter_dependency_consistency(repo_root)
    ci_dependency_checks, ci_dependency_errors = check_ci_adapter_dependency_consistency(repo_root)
    dependency_checks = [*core_dependency_checks, *ci_dependency_checks]
    dependency_errors = [*core_dependency_errors, *ci_dependency_errors]
    errors.extend(dependency_errors)

    if args.check_adapter_latest and not dependency_errors:
        latest_checks, latest_errors = check_adapter_dependencies_are_latest(
            dependency_checks,
            lambda package_name: fetch_latest_pypi_version(
                package_name,
                timeout=args.pypi_timeout,
                allow_prerelease=args.allow_prerelease,
            ),
        )
        dependency_checks = latest_checks
        errors.extend(latest_errors)

    if errors:
        print("Release readiness checks failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"Release readiness checks passed for datus-agent {version}")
    print("Adapter dependency lower bounds:")
    for check in dependency_checks:
        latest = f", latest={check.latest_version}" if check.latest_version is not None else ""
        print(f"  - {check.name}>={check.pyproject_lower_bound}{latest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
