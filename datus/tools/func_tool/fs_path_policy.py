# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Path classification policy shared between ``FilesystemFuncTool`` and the
permission / generation hooks.

A single ``classify_path()`` pure function assigns every filesystem path the
tool sees to one of four zones (``INTERNAL``/``WHITELIST``/``HIDDEN``/
``EXTERNAL``). The same classification is then enforced in two places:

1. ``FilesystemFuncTool`` — visibility and early-return for ``HIDDEN``/bounds.
2. ``PermissionHooks`` — force ``ASK`` for ``EXTERNAL`` and short-circuit
   ``INTERNAL``/``WHITELIST`` against any ``check_permission`` verdict.

Keeping the function pure (no IO beyond ``Path.resolve(strict=False)``) lets
both layers share behavior without coupling them to each other.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class PathZone(str, Enum):
    """Classification of a filesystem path relative to the project root."""

    INTERNAL = "internal"
    WHITELIST = "whitelist"
    HIDDEN = "hidden"
    EXTERNAL = "external"


@dataclass(frozen=True)
class ResolvedPath:
    """Result of ``classify_path``.

    Attributes:
        raw: Caller-supplied path string, unmodified.
        resolved: Absolute, symlink-resolved ``Path`` (``strict=False`` so the
            target may not exist yet — needed for write ops).
        zone: Which ``PathZone`` the resolved path belongs to.
        display: Human/LLM-friendly rendering. Relative to ``root_path`` for
            ``INTERNAL``/``WHITELIST``-in-project, ``~``-prefixed for home
            whitelist, absolute for ``EXTERNAL``/``HIDDEN``.
        read_only: True when the path is whitelisted only for reading (e.g. the
            current session's compact-archive directory). Reads are allowed but
            writes/edits must be rejected at the tool layer.
    """

    raw: str
    resolved: Path
    zone: PathZone
    display: str
    read_only: bool = False


def _normalize_allow_roots(raw: Any) -> Tuple[Path, ...]:
    """Coerce an ``allow_read``/``allow_write`` config value into anchor paths.

    Accepts a single string or a list of strings; non-absolute entries (after
    ``~`` expansion) are dropped because resolving them against the process CWD
    would silently grant an unrelated directory. Existence is NOT required —
    the anchor may be created later (e.g. a DAGs folder the scheduler pod
    provisions on first submit).
    """
    if raw is None:
        return ()
    candidates: Iterable[Any] = [raw] if isinstance(raw, (str, Path)) else raw
    try:
        candidates = list(candidates)
    except TypeError:
        logger.warning("filesystem allowlist must be a path or list of paths; ignoring %r.", raw)
        return ()

    roots: List[Path] = []
    for candidate in candidates:
        if not isinstance(candidate, (str, Path)):
            continue
        text = str(candidate).strip()
        if not text:
            continue
        expanded = Path(os.path.expanduser(text))
        if not expanded.is_absolute():
            logger.warning("Ignoring relative filesystem allowlist entry %r — absolute paths only.", text)
            continue
        resolved = expanded.resolve(strict=False)
        if resolved not in roots:
            roots.append(resolved)
    return tuple(roots)


class PathAllowlist(BaseModel):
    """Extra roots outside the project root that stay reachable to fs tools.

    Parsed from ``agent.filesystem.allow_read`` / ``allow_write``: a path under
    a ``write`` anchor classifies as a writable ``WHITELIST``, one under a
    ``read`` anchor as a read-only ``WHITELIST``. Intentionally generic — the
    first consumer is the Airflow DAGs folder mounted next to (not inside) the
    project workspace, but any deployment-specific shared directory plugs in
    through the same config keys with no code change.

    Anchors only ever *upgrade* an otherwise ``EXTERNAL`` path: project-side
    zones (``INTERNAL``, the built-in whitelist, ``HIDDEN``) are decided first,
    so an allowlist entry can never expose ``.datus/`` internals.

    Frozen: the tool, the permission hook and the scheduler tools all read the
    same instance off ``AgentConfig``, so it must never be mutated underneath
    one of them. Build via :meth:`from_dict` — direct construction skips the
    absolute-path normalization that keeps a relative entry from resolving
    against the process CWD.
    """

    model_config = ConfigDict(frozen=True)

    read: Tuple[Path, ...] = ()
    write: Tuple[Path, ...] = ()

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> "PathAllowlist":
        """Build from the ``agent.filesystem`` config section (``None`` → empty)."""
        if not isinstance(raw, dict):
            return cls()
        return cls(
            read=_normalize_allow_roots(raw.get("allow_read")),
            write=_normalize_allow_roots(raw.get("allow_write")),
        )

    def anchors(self) -> List[Path]:
        """All anchors, write-first (mirrors ``classify_path`` precedence)."""
        return [*self.write, *self.read]

    def __bool__(self) -> bool:
        return bool(self.read or self.write)


def _is_relative_to(candidate: Path, anchor: Path) -> bool:
    """Python 3.12 has ``Path.is_relative_to`` but guarded with a try/except for
    non-Path comparisons. Kept as a small helper for readability and so tests
    can mock or extend it without touching ``Path``.
    """
    try:
        candidate.relative_to(anchor)
        return True
    except ValueError:
        return False


# Keeps a pathological allowlist from bloating a tool-result the LLM has to read.
_MAX_LISTED_ROOTS = 8


def _join_roots(roots: List[Path]) -> str:
    shown = [str(root) for root in roots[:_MAX_LISTED_ROOTS]]
    if len(roots) > _MAX_LISTED_ROOTS:
        shown.append(f"... (+{len(roots) - _MAX_LISTED_ROOTS} more)")
    return ", ".join(shown)


def strict_mode_rejection_message(
    display: str,
    *,
    root_path: Path,
    allowlist: Optional[PathAllowlist] = None,
) -> str:
    """Error text for an ``EXTERNAL`` path rejected by strict mode.

    Naming the reachable roots is what keeps the model from probing: with a bare
    rejection it walks parent directories and shell-outs looking for a file it
    wrote through a proxied (differently anchored) fs op, and every probe is
    another rejected turn.

    The text up to and including ``display`` is a tool-result contract matched by
    the permission hooks' tests and by the LLM — only ever append to it.
    """
    root_resolved = Path(root_path).expanduser().resolve(strict=False)
    allowed = [root_resolved]
    read_only: List[Path] = []
    if allowlist is not None:
        # Anchors under the project root add nothing — the root already covers them.
        for anchor in allowlist.write:
            if not _is_relative_to(anchor, root_resolved) and anchor not in allowed:
                allowed.append(anchor)
        for anchor in allowlist.read:
            if _is_relative_to(anchor, root_resolved) or anchor in allowed or anchor in read_only:
                continue
            read_only.append(anchor)

    hints = [f"allowed roots: {_join_roots(allowed)}"]
    if read_only:
        hints.append(f"read-only roots: {_join_roots(read_only)}")
    return f"Path outside workspace is not allowed in strict mode: {display} ({'; '.join(hints)})"


def _resolve_home(datus_home: Optional[Path]) -> Path:
    """Resolve the effective ``~/.datus`` root for whitelist anchors."""
    if datus_home is not None:
        return Path(datus_home).expanduser().resolve(strict=False)
    return (Path.home() / ".datus").resolve(strict=False)


def classify_path(
    path: str,
    *,
    root_path: Path,
    current_node: Optional[str] = None,
    datus_home: Optional[Path] = None,
    session_data_dir: Optional[Path] = None,
    allowlist: Optional[PathAllowlist] = None,
) -> ResolvedPath:
    """Classify ``path`` into a ``PathZone`` relative to ``root_path``.

    Args:
        path: The raw path string as reported by the LLM / caller. May be
            relative, absolute, or ``~``-prefixed. Empty or ``.`` is treated
            as the project root itself.
        root_path: The project root (used as the base for relative paths and
            as the anchor for ``INTERNAL`` / project-side whitelist checks).
            Will be ``resolve()``'d internally, so symlinks in the root are
            normalized up front.
        current_node: Name of the currently executing node. Accepted for call-
            site compatibility; the whole ``.datus/memory/**`` subtree is
            ``HIDDEN`` to filesystem tools regardless (persistent memory is
            written exclusively through the dedicated ``add_memory`` /
            ``edit_memory`` tools), so this no longer influences classification.
        datus_home: Override for ``~/.datus``. Primarily a test hook; when
            ``None`` the classifier uses the real home directory.
        session_data_dir: When provided, the current session's compact-archive
            directory (``path_manager.session_data_dir(session_id)``) qualifies
            as a read-only ``WHITELIST`` anchor. Lets the model ``read_file``
            archived tool I/O without prompting; writes are denied so the
            compact pass owns the directory contents. Other session_ids'
            directories stay ``EXTERNAL`` even though they live under the
            same ``sessions/`` root.
        allowlist: Operator-configured extra roots (``agent.filesystem
            .allow_read`` / ``allow_write``). Matching paths become
            ``WHITELIST`` (read-only for ``read``-only anchors) instead of
            ``EXTERNAL``, which is what lets strict-mode deployments reach
            shared directories mounted outside the project workspace.

    Returns:
        A ``ResolvedPath`` with the computed zone and display form. Never
        raises — callers can rely on getting some ``PathZone`` back.
    """
    raw = path
    root_resolved = Path(root_path).expanduser().resolve(strict=False)
    home_resolved = _resolve_home(datus_home)

    candidate_input = path.strip() if path else ""
    if candidate_input in ("", ".", "./"):
        expanded = root_resolved
    else:
        expanded = Path(os.path.expanduser(candidate_input))
        if not expanded.is_absolute():
            expanded = root_resolved / expanded

    resolved = expanded.resolve(strict=False)

    # Anchor order matters: project-side anchors are matched before the home
    # anchor so a project that happens to live under ``~/.datus`` still
    # classifies ``{project_root}/.datus/skills/x`` as the project's own skill
    # rather than the global one. See Decision Order step 4 in the plan.
    # ``current_node`` no longer creates a writable memory anchor: the entire
    # ``.datus/memory/**`` subtree is HIDDEN to filesystem tools so the
    # dedicated memory tools own it exclusively.
    del current_node
    project_dot_datus = (root_resolved / ".datus").resolve(strict=False)
    project_skills = (project_dot_datus / "skills").resolve(strict=False)
    project_plans = (project_dot_datus / "plans").resolve(strict=False)
    global_skills = (home_resolved / "skills").resolve(strict=False)

    # ``project_plans`` hosts the LLM's plan-mode markdown files and must be
    # both readable and writable from any agentic node.
    writable_whitelist = [project_skills, project_plans, global_skills]

    session_data_anchor: Optional[Path] = None
    if session_data_dir is not None:
        session_data_anchor = Path(session_data_dir).expanduser().resolve(strict=False)

    zone: PathZone
    display: str
    read_only = False
    # Relative displays use ``.as_posix()`` instead of ``str()`` so the forward-
    # slash-shaped scope globs used downstream (e.g. ``subject/**`` matched
    # with ``wcmatch.globmatch``) still work on Windows. ``str(Path)`` yields
    # backslashes on Windows, which ``wcmatch`` does not normalize — it would
    # silently reject valid writes.
    matched_writable = any(_is_relative_to(resolved, anchor) for anchor in writable_whitelist)
    matched_session_data = session_data_anchor is not None and _is_relative_to(resolved, session_data_anchor)
    if matched_writable:
        zone = PathZone.WHITELIST
        if _is_relative_to(resolved, root_resolved):
            display = resolved.relative_to(root_resolved).as_posix()
        elif _is_relative_to(resolved, home_resolved):
            # ``home_resolved`` already *is* the ``.datus`` directory, so we
            # rebuild the display with the canonical ``~/.datus/`` prefix the
            # LLM can feed back unambiguously. Without this we would show
            # ``~/skills/foo`` and lose which home we meant.
            display = "~/.datus/" + resolved.relative_to(home_resolved).as_posix()
        else:
            display = str(resolved)
    elif matched_session_data:
        # Compact archive: read-only access for the current session only.
        # ``session_data_anchor`` lives under ``home_resolved`` (which is
        # ``~/.datus``), so render it with the ``~/.datus/`` prefix the LLM
        # already understands instead of a raw absolute path.
        zone = PathZone.WHITELIST
        read_only = True
        if _is_relative_to(resolved, home_resolved):
            display = "~/.datus/" + resolved.relative_to(home_resolved).as_posix()
        else:
            display = str(resolved)
    elif _is_relative_to(resolved, project_dot_datus):
        zone = PathZone.HIDDEN
        if _is_relative_to(resolved, root_resolved):
            display = resolved.relative_to(root_resolved).as_posix()
        else:
            display = str(resolved)
    elif _is_relative_to(resolved, root_resolved):
        zone = PathZone.INTERNAL
        display = resolved.relative_to(root_resolved).as_posix() or "."
    elif allowlist is not None and any(_is_relative_to(resolved, anchor) for anchor in allowlist.write):
        # Configured shared root outside the project — absolute display, since
        # nothing project-relative would round-trip back to the same file.
        zone = PathZone.WHITELIST
        display = str(resolved)
    elif allowlist is not None and any(_is_relative_to(resolved, anchor) for anchor in allowlist.read):
        zone = PathZone.WHITELIST
        read_only = True
        display = str(resolved)
    else:
        zone = PathZone.EXTERNAL
        display = str(resolved)

    return ResolvedPath(raw=raw, resolved=resolved, zone=zone, display=display, read_only=read_only)


def whitelist_anchors(
    *,
    root_path: Path,
    current_node: Optional[str] = None,
    datus_home: Optional[Path] = None,
    session_data_dir: Optional[Path] = None,
    allowlist: Optional[PathAllowlist] = None,
) -> list[Path]:
    """Return the resolved whitelist anchor directories for a given node.

    Used by walkers that need to answer "is there a whitelisted subtree
    underneath this HIDDEN directory?" without re-running ``classify_path``
    for every descendant. The order mirrors ``classify_path`` (project-side
    first) so longer prefixes win. ``current_node`` is accepted for call-site
    compatibility but no longer contributes a memory anchor — the
    ``.datus/memory/**`` subtree is HIDDEN to filesystem tools. Configured
    ``allowlist`` roots are appended last (they live outside the project, so
    they never shadow a project-side anchor).
    """
    del current_node
    root_resolved = Path(root_path).expanduser().resolve(strict=False)
    home_resolved = _resolve_home(datus_home)
    project_dot_datus = (root_resolved / ".datus").resolve(strict=False)
    anchors = [
        (project_dot_datus / "skills").resolve(strict=False),
        (project_dot_datus / "plans").resolve(strict=False),
        (home_resolved / "skills").resolve(strict=False),
    ]
    if session_data_dir is not None:
        anchors.append(Path(session_data_dir).expanduser().resolve(strict=False))
    if allowlist is not None:
        anchors.extend(allowlist.anchors())
    return anchors


def build_walk_patterns(
    *,
    root_path: Path,
    current_node: Optional[str] = None,
) -> tuple[list[str], list[str]]:
    """Build the (exclude, re-include) glob pattern pair used by the walker.

    The tool feeds these into ``wcmatch`` so ``HIDDEN`` subtrees are pruned
    before any per-entry work, mirroring Claude Code's ``--glob !{pattern}``
    ripgrep trick. Re-includes win over excludes, so the whitelisted subtrees
    under ``.datus/`` stay visible. The ``.datus/memory/**`` subtree is never
    re-included — it is HIDDEN to filesystem tools.

    Args:
        root_path: Project root (unused beyond documentation today; accepted
            for forward-compat so callers that later want root-relative
            patterns don't need to change signature).
        current_node: Accepted for call-site compatibility; no longer affects
            the pattern set.

    Returns:
        ``(excludes, re_includes)`` — both are glob patterns rooted at
        ``root_path`` (no leading ``/``).
    """
    del root_path  # reserved for future use; keeps API stable.
    del current_node
    excludes = [".datus", ".datus/**"]
    re_includes = [".datus/skills/**", ".datus/plans/**"]
    return excludes, re_includes
