# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Path normalization and sandbox resolution for KB artifact files."""

import os
from typing import Optional

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Maps generation kind → top-level KB subdir name beneath the project's subject/
# directory (e.g. ``{project_root}/subject/semantic_models``).
KIND_TO_SUBDIR = {
    "semantic": "semantic_models",
    "metric": "semantic_models",
    "sql_summary": "sql_summaries",
}


def normalize_kb_relative_path(path: str, kind: Optional[str]) -> str:
    """
    Silently normalize a relative path so that it lands under the typed
    sub-directory of the project's ``subject/`` tree, even when the caller
    forgets the ``{subdir}/`` prefix.

    Rules:
      * Empty / absolute paths → unchanged.
      * "." / "./" → unchanged (workspace-root directory operations).
      * Path starts with a parent-traversal segment (``..``) → unchanged so
        the downstream sandbox check decides whether to reject.
      * Unknown ``kind`` → unchanged.
      * Path already starts with any known KB subdir (semantic_models /
        sql_summaries) → unchanged (caller is being explicit).
      * Otherwise → prepend ``{subdir}/``.
    """
    if not path or os.path.isabs(path):
        return path
    if path in (".", "./"):
        return path
    parts = [p for p in path.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts:
        return path
    if parts[0] == "..":
        return path
    # After the ``subject/`` relocation, prompts emit fully-qualified paths
    # like ``subject/semantic_models/orders.yml``. Callers join the result
    # with ``kb_home`` (``{project_root}/subject``), so strip a leading
    # ``subject/`` segment first to avoid ``subject/subject/...`` drift.
    if parts[0] == "subject":
        parts = parts[1:]
        if not parts:
            return path
    subdir = KIND_TO_SUBDIR.get(kind or "")
    if not subdir:
        return "/".join(parts)
    head = parts[0]
    if head in set(KIND_TO_SUBDIR.values()):
        return "/".join(parts)
    return f"{subdir}/{'/'.join(parts)}"


def resolve_kb_sandbox_path(
    raw_path: str,
    kind: str,
    knowledge_base_dir: str,
) -> Optional[str]:
    """
    Resolve an LLM-reported file path to an absolute path under the sandbox
    ``{knowledge_base_dir}/{kind_subdir}/`` for the given ``kind``.

    Used where the path comes from the model's final JSON (not from a
    ``write_file`` tool result), so it must be validated against the per-kind
    sandbox before syncing — otherwise a fabricated response could cause an
    arbitrary file on disk to be imported. Returns ``None`` when the path
    escapes the sandbox so callers can skip it.
    """
    if not raw_path:
        return None
    if os.path.isabs(raw_path):
        candidate = os.path.normpath(raw_path)
    else:
        normalized = normalize_kb_relative_path(raw_path, kind)
        candidate = os.path.normpath(os.path.join(knowledge_base_dir, normalized))
    subdir = KIND_TO_SUBDIR.get(kind or "")
    if not subdir:
        return candidate
    try:
        sandbox = os.path.realpath(os.path.join(knowledge_base_dir, subdir))
        candidate_real = os.path.realpath(candidate)
        if os.path.commonpath([sandbox, candidate_real]) != sandbox:
            logger.warning(
                f"Rejected path {raw_path!r} for kind={kind}: resolved {candidate_real!r} escapes sandbox {sandbox!r}."
            )
            return None
    except ValueError:
        logger.warning(f"Rejected path {raw_path!r} for kind={kind}: cannot verify containment under sandbox.")
        return None
    return candidate
