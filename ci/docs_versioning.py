#!/usr/bin/env python3
"""Resolve and maintain Datus docs versions for MkDocs/mike deployment."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, NamedTuple

from packaging.version import InvalidVersion, Version

FULL_RELEASE_RE = re.compile(r"^v?(?P<version>\d+\.\d+\.\d+[0-9A-Za-z.+-]*)$")
STABLE_PATCH_RE = re.compile(r"^\d+\.\d+\.\d+$")
MINOR_RE = re.compile(r"^\d+\.\d+$")


class DocsVersionMetadata(NamedTuple):
    skip_deploy: bool
    build_config: str = "mkdocs.yml"
    source_ref: str = ""
    edit_uri: str = "edit/main/docs/"
    docs_version: str = ""
    aliases: tuple[str, ...] = ()
    set_default: str = ""

    def to_github_env(self) -> str:
        return "\n".join(
            [
                f"DOCS_SKIP_DEPLOY={'true' if self.skip_deploy else 'false'}",
                f"DOCS_BUILD_CONFIG={self.build_config}",
                f"DOCS_SOURCE_REF={self.source_ref}",
                f"DOCS_EDIT_URI={self.edit_uri}",
                f"DOCS_VERSION={self.docs_version}",
                f"DOCS_ALIASES={' '.join(self.aliases)}",
                f"DOCS_SET_DEFAULT={self.set_default}",
            ]
        )


def parse_release_version(raw_version: str) -> Version:
    match = FULL_RELEASE_RE.fullmatch(raw_version.strip())
    if not match:
        raise ValueError(f"Invalid release version format: {raw_version}")
    try:
        return Version(match.group("version"))
    except InvalidVersion as exc:
        raise ValueError(f"Invalid release version: {raw_version}") from exc


def is_stable_patch(version: Version) -> bool:
    return STABLE_PATCH_RE.fullmatch(str(version)) is not None


def docs_minor_label(version: Version) -> str:
    return f"{version.major}.{version.minor}"


def git_tag_exists(tag: str, repo_root: Path = Path(".")) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
        cwd=repo_root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def resolve_source_tag(version: Version, tag_exists: Callable[[str], bool]) -> str:
    version_text = str(version)
    for candidate in (f"v{version_text}", version_text):
        if tag_exists(candidate):
            return candidate
    raise ValueError(f"Tag v{version_text} or {version_text} not found")


def parse_published_versions(raw_json: str) -> list[Version]:
    try:
        items = json.loads(raw_json or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid published versions JSON") from exc
    if not isinstance(items, list):
        raise ValueError("Invalid published versions JSON: expected a list")

    versions: list[Version] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_version = str(item.get("version") or "")
        if not (MINOR_RE.fullmatch(raw_version) or STABLE_PATCH_RE.fullmatch(raw_version)):
            continue
        try:
            versions.append(Version(raw_version))
        except InvalidVersion:
            continue
    return versions


def should_promote_latest(current_version: Version, published_versions_json: str) -> bool:
    stable_versions = parse_published_versions(published_versions_json)
    return not stable_versions or current_version >= max(stable_versions)


def resolve_docs_metadata(
    *,
    event_name: str,
    github_ref_type: str,
    github_ref_name: str,
    input_version: str = "",
    published_versions_json: str = "[]",
    tag_exists: Callable[[str], bool] = git_tag_exists,
) -> DocsVersionMetadata:
    if event_name == "workflow_dispatch" and input_version.strip():
        version = parse_release_version(input_version)
        if not is_stable_patch(version):
            raise ValueError("Versioned docs deploy only supports stable patch versions, for example v0.3.7")
        source_ref = resolve_source_tag(version, tag_exists)
        return version_metadata_for_stable_tag(source_ref, version, published_versions_json)

    if github_ref_type == "tag":
        try:
            version = parse_release_version(github_ref_name)
        except ValueError:
            return DocsVersionMetadata(skip_deploy=True, edit_uri=f"edit/{github_ref_name}/docs/")
        if not is_stable_patch(version):
            return DocsVersionMetadata(skip_deploy=True, edit_uri=f"edit/{github_ref_name}/docs/")
        return version_metadata_for_stable_tag(github_ref_name, version, published_versions_json)

    return DocsVersionMetadata(skip_deploy=False, docs_version="dev")


def version_metadata_for_stable_tag(
    source_ref: str,
    version: Version,
    published_versions_json: str,
) -> DocsVersionMetadata:
    aliases: tuple[str, ...] = ()
    set_default = ""
    if should_promote_latest(version, published_versions_json):
        aliases = ("latest",)
        set_default = "latest"
    return DocsVersionMetadata(
        skip_deploy=False,
        source_ref=source_ref,
        edit_uri=f"edit/{source_ref}/docs/",
        docs_version=docs_minor_label(version),
        aliases=aliases,
        set_default=set_default,
    )


def parse_minor(raw_minor: str) -> tuple[int, int]:
    if not MINOR_RE.fullmatch(raw_minor):
        raise ValueError(f"Invalid minor docs version: {raw_minor}")
    major, minor = raw_minor.split(".", maxsplit=1)
    return int(major), int(minor)


def should_hide_docs_version(version: str, min_visible_minor: tuple[int, int]) -> bool:
    if version == "dev":
        return False
    if MINOR_RE.fullmatch(version):
        major, minor = parse_minor(version)
        return (major, minor) < min_visible_minor
    if FULL_RELEASE_RE.fullmatch(version):
        return True
    return False


def hide_legacy_docs_versions(min_visible_minor: tuple[int, int]) -> bool:
    raw = subprocess.check_output(["mike", "list", "-j"], text=True)
    versions = json.loads(raw or "[]")
    changed = False

    for item in versions:
        if not isinstance(item, dict):
            continue
        version = str(item.get("version") or "")
        props = item.get("properties") or {}
        hidden = bool(props.get("hidden", False))
        target_hidden = should_hide_docs_version(version, min_visible_minor)
        if hidden == target_hidden:
            continue

        subprocess.run(
            [
                "mike",
                "props",
                version,
                "--set",
                f"hidden={'true' if target_hidden else 'false'}",
                "--allow-empty",
            ],
            check=True,
        )
        changed = True

    if changed:
        subprocess.run(["git", "push", "--", "origin", "gh-pages"], check=True)
    return changed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve = subparsers.add_parser("resolve", help="Resolve docs deployment metadata and print GitHub env lines")
    resolve.add_argument("--input-version", default="")
    resolve.add_argument("--event-name", required=True)
    resolve.add_argument("--github-ref-type", required=True)
    resolve.add_argument("--github-ref-name", required=True)
    resolve.add_argument("--published-versions-json", default=None)

    hide = subparsers.add_parser("hide-legacy", help="Hide patch/pre-release docs versions from mike version picker")
    hide.add_argument("--min-visible-minor", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "resolve":
        try:
            metadata = resolve_docs_metadata(
                event_name=args.event_name,
                github_ref_type=args.github_ref_type,
                github_ref_name=args.github_ref_name,
                input_version=args.input_version,
                published_versions_json=args.published_versions_json
                if args.published_versions_json is not None
                else os.environ.get("PUBLISHED_DOCS_JSON", "[]"),
            )
        except ValueError as exc:
            print(f"::error::{exc}", file=sys.stderr)
            return 1
        print(metadata.to_github_env())
        return 0

    if args.command == "hide-legacy":
        try:
            min_visible_minor = parse_minor(args.min_visible_minor)
        except ValueError as exc:
            print(f"::error::Invalid --min-visible-minor: {exc}", file=sys.stderr)
            return 1
        changed = hide_legacy_docs_versions(min_visible_minor)
        if not changed:
            print("No docs version visibility changes needed.")
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
