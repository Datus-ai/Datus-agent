"""Select a smaller documentation gate only for explicitly safe changes."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path, PurePosixPath

import yaml

from ci.harness.test_plan import MAP, ROOT

MAP_PATH = MAP.relative_to(ROOT).as_posix()
DOC_SUFFIXES = {".md", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico"}


def compare_base(base_ref: str | None = None, *, root: Path = ROOT) -> str:
    if not base_ref:
        event_path = os.getenv("GITHUB_EVENT_PATH")
        event = json.loads(Path(event_path).read_text()) if event_path else {}
        base_ref = event.get("merge_group", {}).get("base_sha") or event.get("pull_request", {}).get("base", {}).get(
            "sha"
        )
    result = subprocess.run(
        [
            "git",
            "merge-base",
            base_ref or ("origin/main" if os.getenv("GITHUB_ACTIONS") == "true" else "upstream/main"),
            "HEAD",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return result.stdout.strip()


def _without_docs(data: dict) -> dict:
    result = copy.deepcopy(data)
    for group in result.get("flow_groups", {}).values():
        for flow in group.get("flows", []):
            flow.pop("docs", None)
    return result


def documentation_map_change(before: str, after: str) -> bool:
    """Allow only flow docs lists; execution rules and coverage claims stay gated."""
    old, new = yaml.safe_load(before), yaml.safe_load(after)
    if not isinstance(old, dict) or not isinstance(new, dict):
        return False
    return _without_docs(old) == _without_docs(new)


def select_profile(changed: list[str], *, map_before: str | None = None, map_after: str | None = None) -> dict:
    profile = {"profile": "code", "changed_files": sorted(set(changed)), "ci_tests": False}
    if not changed:
        return {**profile, "reason": "No changes were identified; retain the full gate"}
    for name in changed:
        path = PurePosixPath(name)
        if name == MAP_PATH:
            if map_before is None or map_after is None or not documentation_map_change(map_before, map_after):
                return {**profile, "reason": "Coverage or execution configuration changed"}
        elif name not in {"mkdocs.yml", "README.md", "README.zh.md"} and not (
            path.parts[0] == "docs" and ".." not in path.parts and path.suffix.lower() in DOC_SUFFIXES
        ):
            return {**profile, "reason": f"Full gate required by {name}"}
    return {
        **profile,
        "profile": "docs",
        "ci_tests": MAP_PATH in changed,
        "reason": "Only documentation, navigation, or flow documentation references changed",
    }


def repository_profile(changed: list[str], base_ref: str | None = None, *, root: Path = ROOT) -> dict:
    before = after = None
    if MAP_PATH in changed:
        base = compare_base(base_ref, root=root)
        before = subprocess.run(
            ["git", "show", f"{base}:{MAP_PATH}"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout
        after = (root / MAP_PATH).read_text() if (root / MAP_PATH).is_file() else None
    return select_profile(changed, map_before=before, map_after=after)
