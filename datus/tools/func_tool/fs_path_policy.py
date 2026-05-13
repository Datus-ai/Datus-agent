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
from typing import Optional


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
        read_only: True when the path is whitelisted only because the calling
            node inherited a parent's memory directory. Reads are allowed but
            writes/edits must be rejected at the tool layer.
    """

    raw: str
    resolved: Path
    zone: PathZone
    display: str
    read_only: bool = False


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


def _resolve_home(datus_home: Optional[Path]) -> Path:
    """Resolve the effective ``~/.datus`` root for whitelist anchors."""
    if datus_home is not None:
        return Path(datus_home).expanduser().resolve(strict=False)
    return (Path.home() / ".datus").resolve(strict=False)


def classify_path(
    path: str,
    *,
    root_path: Path,
    current_node: Optional[str],
    datus_home: Optional[Path] = None,
    inherited_memory_node: Optional[str] = None,
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
        current_node: Name of the currently executing node. Only then does
            ``{root_path}/.datus/memory/{current_node}/**`` qualify as
            ``WHITELIST``. ``None`` causes the memory anchor to be dropped,
            which demotes every ``.datus/memory/**`` path to ``HIDDEN``.
        datus_home: Override for ``~/.datus``. Primarily a test hook; when
            ``None`` the classifier uses the real home directory.
        inherited_memory_node: When a built-in sub-agent inherits its parent's
            memory (read-only), the parent's
            ``{root_path}/.datus/memory/{inherited_memory_node}/**`` subtree
            also qualifies as ``WHITELIST``, but the resulting ``ResolvedPath``
            carries ``read_only=True`` so the tool layer can deny writes.

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
    project_dot_datus = (root_resolved / ".datus").resolve(strict=False)
    project_skills = (project_dot_datus / "skills").resolve(strict=False)
    project_plans = (project_dot_datus / "plans").resolve(strict=False)
    project_memory_node: Optional[Path] = None
    if current_node:
        project_memory_node = (project_dot_datus / "memory" / current_node).resolve(strict=False)
    global_skills = (home_resolved / "skills").resolve(strict=False)

    inherited_memory_dir: Optional[Path] = None
    if inherited_memory_node and inherited_memory_node != current_node:
        inherited_memory_dir = (project_dot_datus / "memory" / inherited_memory_node).resolve(strict=False)

    # Owned (writable) anchors first; the inherited anchor is checked last and
    # tagged ``read_only=True`` so the tool layer can deny writes there.
    # ``project_plans`` hosts the LLM's plan-mode markdown files and must be
    # both readable and writable from any agentic node.
    writable_whitelist = [project_skills, project_plans]
    if project_memory_node is not None:
        writable_whitelist.append(project_memory_node)
    writable_whitelist.append(global_skills)

    zone: PathZone
    display: str
    read_only = False
    # Relative displays use ``.as_posix()`` instead of ``str()`` so the forward-
    # slash-shaped scope globs used downstream (e.g. ``subject/**`` matched
    # with ``wcmatch.globmatch``) still work on Windows. ``str(Path)`` yields
    # backslashes on Windows, which ``wcmatch`` does not normalize — it would
    # silently reject valid writes.
    matched_writable = any(_is_relative_to(resolved, anchor) for anchor in writable_whitelist)
    matched_inherited = inherited_memory_dir is not None and _is_relative_to(resolved, inherited_memory_dir)
    if matched_writable or matched_inherited:
        zone = PathZone.WHITELIST
        # Owned matches always win; the read-only flag is only set when the
        # path is exclusively reachable via the inherited anchor.
        read_only = matched_inherited and not matched_writable
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
    elif _is_relative_to(resolved, project_dot_datus):
        zone = PathZone.HIDDEN
        if _is_relative_to(resolved, root_resolved):
            display = resolved.relative_to(root_resolved).as_posix()
        else:
            display = str(resolved)
    elif _is_relative_to(resolved, root_resolved):
        zone = PathZone.INTERNAL
        display = resolved.relative_to(root_resolved).as_posix() or "."
    else:
        zone = PathZone.EXTERNAL
        display = str(resolved)

    return ResolvedPath(raw=raw, resolved=resolved, zone=zone, display=display, read_only=read_only)


def whitelist_anchors(
    *,
    root_path: Path,
    current_node: Optional[str],
    datus_home: Optional[Path] = None,
    inherited_memory_node: Optional[str] = None,
) -> list[Path]:
    """Return the resolved whitelist anchor directories for a given node.

    Used by walkers that need to answer "is there a whitelisted subtree
    underneath this HIDDEN directory?" without re-running ``classify_path``
    for every descendant. The order mirrors ``classify_path`` (project-side
    first) so longer prefixes win. ``inherited_memory_node`` adds the parent
    sub-agent's memory dir so a built-in child can ``Read``/``Glob`` it; write
    enforcement is the tool layer's job.
    """
    root_resolved = Path(root_path).expanduser().resolve(strict=False)
    home_resolved = _resolve_home(datus_home)
    project_dot_datus = (root_resolved / ".datus").resolve(strict=False)
    anchors = [
        (project_dot_datus / "skills").resolve(strict=False),
        (project_dot_datus / "plans").resolve(strict=False),
    ]
    if current_node:
        anchors.append((project_dot_datus / "memory" / current_node).resolve(strict=False))
    if inherited_memory_node and inherited_memory_node != current_node:
        anchors.append((project_dot_datus / "memory" / inherited_memory_node).resolve(strict=False))
    anchors.append((home_resolved / "skills").resolve(strict=False))
    return anchors


def build_walk_patterns(
    *,
    root_path: Path,
    current_node: Optional[str],
    inherited_memory_node: Optional[str] = None,
) -> tuple[list[str], list[str]]:
    """Build the (exclude, re-include) glob pattern pair used by the walker.

    The tool feeds these into ``wcmatch`` so ``HIDDEN`` subtrees are pruned
    before any per-entry work, mirroring Claude Code's ``--glob !{pattern}``
    ripgrep trick. Re-includes win over excludes, so the two whitelisted
    subtrees under ``.datus/`` stay visible.

    Args:
        root_path: Project root (unused beyond documentation today; accepted
            for forward-compat so callers that later want root-relative
            patterns don't need to change signature).
        current_node: Same semantics as ``classify_path``; ``None`` leaves
            the memory subtree excluded.
        inherited_memory_node: When set and distinct from ``current_node``,
            the parent's memory subtree is also re-included so ``Glob`` /
            ``Grep`` can surface inherited topic files (read-only).

    Returns:
        ``(excludes, re_includes)`` — both are glob patterns rooted at
        ``root_path`` (no leading ``/``).
    """
    del root_path  # reserved for future use; keeps API stable.
    excludes = [".datus", ".datus/**"]
    re_includes = [".datus/skills/**", ".datus/plans/**"]
    if current_node:
        re_includes.append(f".datus/memory/{current_node}/**")
    if inherited_memory_node and inherited_memory_node != current_node:
        re_includes.append(f".datus/memory/{inherited_memory_node}/**")
    return excludes, re_includes
