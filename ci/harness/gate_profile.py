"""Classify PR changes before selecting tests and application coverage."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import tomllib
from pathlib import Path, PurePosixPath

import yaml

from ci.harness.test_plan import MAP, ROOT

MAP_PATH = MAP.relative_to(ROOT).as_posix()
DOC_SUFFIXES = {".md", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico"}
VERSION_FILES = {"pyproject.toml", "uv.lock", "requirements.txt", "requirements-test.txt"}
GATE_PATHS = (
    MAP_PATH,
    "ci/harness/",
    "ci/run-pr-tests.py",
    "ci/run-tests-and-coverage.py",
    "ci/select_gate_profile.py",
    "ci/pytest_gate_plugin.py",
    ".github/workflows/run-ut-and-coverage.yml",
    "pytest.ini",
    "tests/conftest.py",
    "tests/integration/conftest.py",
    "tests/helpers/",
    "tests/data/",
    "tests/conf/",
)


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or (prefix.endswith("/") and name.startswith(prefix)) for prefix in prefixes)


def _documentation(name: str) -> bool:
    path = PurePosixPath(name)
    return name in {"mkdocs.yml", "README.md", "README.zh.md"} or (
        path.parts and path.parts[0] == "docs" and ".." not in path.parts and path.suffix.lower() in DOC_SUFFIXES
    )


def version_metadata_change(name: str, before: str | None, after: str | None) -> bool:
    """Only the project's own version may change without dependency regression."""
    if before is None or after is None:
        return False
    if name in {"requirements.txt", "requirements-test.txt"}:

        def requirements(text: str) -> list[str]:
            return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]

        return requirements(before) == requirements(after)
    try:
        old, new = tomllib.loads(before), tomllib.loads(after)
        for document in (old, new):
            if name == "pyproject.toml":
                del document["project"]["version"]
            elif name == "uv.lock":
                own = [
                    p
                    for p in document["package"]
                    if p["name"] == "datus-agent" and p.get("source") == {"editable": "."}
                ]
                if len(own) != 1:
                    return False
                del own[0]["version"]
            else:
                return False
        return old == new
    except (AttributeError, KeyError, TypeError, tomllib.TOMLDecodeError):
        return False


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


def select_profile(
    changed: list[str],
    *,
    map_before: str | None = None,
    map_after: str | None = None,
    version_changes: dict[str, tuple[str | None, str | None]] | None = None,
) -> dict:
    profile = {"profile": "code", "changed_files": sorted(set(changed)), "ci_tests": False}
    if not changed:
        return {
            **profile,
            "run_core": True,
            "coverage_applicable": True,
            "metadata_paths": [],
            "gate_changes": [],
            "reason": "No changes were identified; retain the full gate",
        }
    metadata = [
        name
        for name, texts in (version_changes or {}).items()
        if name in changed and version_metadata_change(name, *texts)
    ]
    map_docs = (
        MAP_PATH in changed
        and map_before is not None
        and map_after is not None
        and documentation_map_change(map_before, map_after)
    )
    effective = [
        name
        for name in changed
        if name not in metadata and not _documentation(name) and not (name == MAP_PATH and map_docs)
    ]
    critical = [name for name in effective if _matches(name, GATE_PATHS)]
    runtime = [name for name in effective if not name.startswith(("ci/", ".github/workflows/", "tests/"))]
    return {
        **profile,
        "profile": "code" if effective else ("metadata" if metadata else "docs"),
        "ci_tests": map_docs,
        "metadata_paths": metadata,
        "gate_changes": critical,
        "run_core": bool(runtime or critical),
        "coverage_applicable": any(name.startswith("datus/") and name.endswith(".py") for name in effective),
        "reason": (
            f"Runtime or shared gate changes: {', '.join(runtime + critical)}"
            if runtime or critical
            else "Only scoped CI/tests, documentation, or project version metadata changed"
        ),
    }


def repository_profile(changed: list[str], base_ref: str | None = None, *, root: Path = ROOT) -> dict:
    before = after = None
    version_changes = {}
    if MAP_PATH in changed or VERSION_FILES.intersection(changed):
        base = compare_base(base_ref, root=root)
        for name in sorted(({MAP_PATH} | VERSION_FILES).intersection(changed)):
            old = subprocess.run(
                ["git", "show", f"{base}:{name}"], cwd=root, capture_output=True, text=True, check=False, timeout=60
            )
            old_text = old.stdout if old.returncode == 0 else None
            new_text = (root / name).read_text() if (root / name).is_file() else None
            if name == MAP_PATH:
                before, after = old_text, new_text
            else:
                version_changes[name] = (old_text, new_text)
    return select_profile(changed, map_before=before, map_after=after, version_changes=version_changes)
