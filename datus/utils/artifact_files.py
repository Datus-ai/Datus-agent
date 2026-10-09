# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Canonical file selection for artifact API bundles and saved query hashes."""

from pathlib import Path

from datus.schemas.artifact_manifest import ArtifactKind

ARTIFACT_DIRS = {
    "report": {
        "render": ((".jsx", ".js", ".css", ".json", ".md"), True),
        "queries": ((".sql", ".json"), False),
        "analysis": ((".md", ".json"), False),
    },
    "dashboard": {
        "render": ((".jsx", ".js", ".css", ".json", ".md"), True),
        "queries": ((".sql.j2", ".params.json", ".brief.json"), False),
        "analysis": ((".md", ".json"), False),
    },
}


def iter_artifact_files(artifact_dir: Path, kind: ArtifactKind, *, queries_only: bool = False) -> list[Path]:
    """Walk ``artifact_dir`` and return allowed files sorted by slug-relative path.

    Honours the per-prefix allowlist; files under any other directory or
    whose name doesn't end in one of the listed suffix patterns are
    silently dropped so a stray scratch file doesn't trip detail.

    Each candidate is resolved before being kept so a symlink under
    ``render/`` / ``queries/`` / ``analysis/`` cannot exfiltrate a file
    from outside the artifact directory into the inline bundle — the LLM
    controls these paths and a stray ``ln -s /etc/passwd render/foo.jsx``
    would otherwise survive the ``is_file()`` probe (which follows
    symlinks).
    """
    artifact_dir_resolved = artifact_dir.resolve()
    found: list[Path] = []
    for sub, (allowed_suffixes, recursive) in ARTIFACT_DIRS[kind].items():
        if queries_only and sub != "queries":
            continue
        root = artifact_dir / sub
        if not root.is_dir():
            continue
        iterator = root.rglob("*") if recursive else root.iterdir()
        for path in iterator:
            if not path.is_file():
                continue
            name_lower = path.name.lower()
            if not any(name_lower.endswith(suffix) for suffix in allowed_suffixes):
                continue
            resolved = path.resolve()
            if not resolved.is_file():
                continue
            try:
                resolved.relative_to(artifact_dir_resolved)
            except ValueError:
                # Symlink (or other indirection) escapes the artifact root —
                # drop silently rather than leak content from outside.
                continue
            found.append(path)
    found.sort(key=lambda p: p.relative_to(artifact_dir).as_posix())
    return found
