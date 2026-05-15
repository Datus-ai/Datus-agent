# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared base for the artifact-tree-aware filesystem tools.

Both visual-artifact subagents (report + dashboard) wrap the standard
:class:`FilesystemFuncTool` with two policies:

1. ``<root>/<id>/queries/*`` is read-only via the filesystem layer —
   writes must go through the artifact-specific ``save_query`` /
   ``save_query_template`` so the SQL is actually executed and the
   metadata is well-formed.
2. ``<root>/<id>/render/*`` is writable, but only for ``.jsx`` / ``.js`` /
   ``.css`` files; data files must live under ``queries/``.

The wrappers used to be a near-identical 95-line copy each. This module
hosts the common logic so the concrete subclasses just declare a few
class constants.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar, Optional

from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

# Extensions allowed under ``<root>/<id>/render/``. JSON / data files are
# intentionally excluded — those belong under ``queries/`` and only the
# artifact-specific ``save_query*`` tool should produce them.
RENDER_ALLOWED_SUFFIXES: frozenset[str] = frozenset({".jsx", ".js", ".css"})


class ArtifactFilesystemFuncTool(FilesystemFuncTool):
    """Common base for ``ReportFilesystemFuncTool`` / ``DashboardFilesystemFuncTool``.

    Subclasses declare:

    * :attr:`ARTIFACT_ROOT_DIR_NAME` — ``"reports"`` or ``"dashboards"``;
      shapes the path regexes and the error messages.
    * :attr:`SAVE_QUERY_TOOL_NAME` — ``"save_query"`` or
      ``"save_query_template"``; named in error messages so the LLM
      gets pointed at the right tool.
    * :attr:`ARTIFACT_KIND` — ``"report"`` or ``"dashboard"``; appears in
      humanized error messages (e.g. "the dashboard directory").
    """

    # ── Subclass-supplied class variables ─────────────────────────────────

    ARTIFACT_ROOT_DIR_NAME: ClassVar[str] = ""
    SAVE_QUERY_TOOL_NAME: ClassVar[str] = ""
    ARTIFACT_KIND: ClassVar[str] = ""

    # ── Derived regexes (computed via __init_subclass__) ─────────────────

    _RENDER_PATH_RE: ClassVar[re.Pattern[str]] = re.compile(r"^$")
    _QUERIES_PATH_RE: ClassVar[re.Pattern[str]] = re.compile(r"^$")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Allow abstract intermediates that haven't set the root yet.
        if not cls.ARTIFACT_ROOT_DIR_NAME:
            return
        root = re.escape(cls.ARTIFACT_ROOT_DIR_NAME)
        # Slug is constrained to ``[a-z0-9_]{1,80}`` — matches
        # ARTIFACT_SLUG_PATTERN in datus.schemas.artifact_manifest.
        slug = r"[a-z0-9_]{1,80}"
        cls._RENDER_PATH_RE = re.compile(rf"^{root}/{slug}/render(?:/.+)?$")
        cls._QUERIES_PATH_RE = re.compile(rf"^{root}/{slug}/queries/.+$")

    # ── Path classification ──────────────────────────────────────────────

    def _is_queries_path(self, path: str) -> bool:
        try:
            resolved = self._classify(path)
        except Exception:  # pragma: no cover - defensive
            return False
        try:
            rel = resolved.resolved.relative_to(self._root_resolved)
        except ValueError:
            return False
        return bool(self._QUERIES_PATH_RE.match(rel.as_posix()))

    def _classify_render_path(self, path: str) -> Optional[str]:
        """Return ``"render"`` when path lives under the artifact's render dir."""
        try:
            resolved = self._classify(path)
        except Exception:  # pragma: no cover - defensive
            return None
        try:
            rel = resolved.resolved.relative_to(self._root_resolved)
        except ValueError:
            return None
        if not self._RENDER_PATH_RE.match(rel.as_posix()):
            return None
        return "render"

    # ── Error templates (override-friendly) ──────────────────────────────

    def _queries_write_reject(self) -> str:
        return (
            f"Files under {self.ARTIFACT_ROOT_DIR_NAME}/<id>/queries/ must not be written directly. "
            f"Use the `{self.SAVE_QUERY_TOOL_NAME}` tool — it runs the SQL and persists the artifact files."
        )

    def _queries_edit_reject(self) -> str:
        return (
            "Query artifact files cannot be edited in place. "
            f"Re-run `{self.SAVE_QUERY_TOOL_NAME}` with the same name to regenerate them."
        )

    def _queries_delete_reject(self) -> str:
        return (
            "Query artifact files cannot be deleted via delete_file. "
            f"Re-run {self.SAVE_QUERY_TOOL_NAME} for the desired final state, or remove the "
            f"{self.ARTIFACT_KIND} directory via the filesystem outside the agent."
        )

    def _render_extension_reject(self, display: str) -> FuncToolResult:
        return FuncToolResult(
            success=0,
            error=(
                f"{display}: only .jsx / .js / .css files are allowed under "
                f"{self.ARTIFACT_ROOT_DIR_NAME}/<id>/render/. Data files belong under "
                f"queries/ and must be produced via {self.SAVE_QUERY_TOOL_NAME}."
            ),
        )

    # ── Overrides ────────────────────────────────────────────────────────

    def write_file(self, path: str, content: str, file_type: str = "") -> FuncToolResult:  # type: ignore[override]
        if self._is_queries_path(path):
            return FuncToolResult(success=0, error=self._queries_write_reject())
        if self._classify_render_path(path) == "render":
            suffix = Path(path).suffix.lower()
            if suffix not in RENDER_ALLOWED_SUFFIXES:
                return self._render_extension_reject(path)
        return super().write_file(path, content, file_type)

    def edit_file(self, path: str, old_string: str, new_string: str) -> FuncToolResult:  # type: ignore[override]
        if self._is_queries_path(path):
            return FuncToolResult(success=0, error=self._queries_edit_reject())
        if self._classify_render_path(path) == "render":
            suffix = Path(path).suffix.lower()
            if suffix not in RENDER_ALLOWED_SUFFIXES:
                return self._render_extension_reject(path)
        return super().edit_file(path, old_string, new_string)

    def delete_file(self, path: str) -> FuncToolResult:  # type: ignore[override]
        if self._is_queries_path(path):
            return FuncToolResult(success=0, error=self._queries_delete_reject())
        if self._classify_render_path(path) == "render":
            suffix = Path(path).suffix.lower()
            if suffix not in RENDER_ALLOWED_SUFFIXES:
                return self._render_extension_reject(path)
        return super().delete_file(path)
