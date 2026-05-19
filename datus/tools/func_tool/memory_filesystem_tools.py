# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""In-memory read-only filesystem tool.

Backed by a ``{path: content}`` dict instead of a real directory. Mirrors the
LLM-facing API of :class:`FilesystemFuncTool` (``read_file`` / ``glob`` /
``grep``) so the same tool surface works against a published artifact bundle
loaded from the database, without materializing files to disk.

Write operations (``write_file`` / ``edit_file`` / ``delete_file``) are
intentionally absent — this tool is designed for the ``ask_report`` /
``ask_dashboard`` follow-up subagents, which are read-only by contract.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

from agents import Tool
from wcmatch import glob as wc_glob

from datus.tools import BaseTool
from datus.tools.func_tool import FuncToolResult
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# Match FilesystemFuncTool's per-read cap so large files behave the same way
# whether they're on disk or in memory.
_MAX_READ_BYTES = 200 * 1024
_MAX_GLOB_RESULTS = 200
_MAX_GREP_MATCHES = 100


class MemoryFilesystemFuncTool(BaseTool):
    """Read-only filesystem tool over an in-memory ``{path: content}`` bundle.

    Paths in ``files`` are slug-relative POSIX strings (e.g. ``"manifest.json"``,
    ``"queries/q1.sql"``, ``"analysis/intent.md"``). The tool accepts the same
    relative path forms that the LLM uses against the disk-backed tool
    (``"analysis/intent.md"``, ``"./queries/q1.sql"``); a leading ``"./"`` or
    ``"/"`` is stripped, and ``"."`` is treated as the bundle root.
    """

    tool_name = "memory_filesystem"
    tool_description = "Read-only filesystem operations over an in-memory bundle."

    def __init__(
        self,
        files: Dict[str, str],
        *,
        root_label: str = "in-memory",
        **kwargs,
    ):
        """
        Args:
            files: Bundle contents. Keys are slug-relative POSIX paths, values
                are the file bodies as strings.
            root_label: Human-readable identifier used in log messages and the
                ``root_path`` attribute (consumed by callers that log
                ``self.filesystem_func_tool.root_path``). Has no effect on
                lookup semantics.
        """
        super().__init__(**kwargs)
        self._files: Dict[str, str] = {self._normalize(p): content for p, content in (files or {}).items()}
        self._root_label = root_label

    # ``root_path`` is read by ChatAgenticNode._setup_filesystem_tools for a
    # debug log line. Expose a label that's recognizable in logs but not a
    # filesystem path — there is no disk root.
    @property
    def root_path(self) -> str:
        return self._root_label

    def available_tools(self) -> List[Tool]:
        from datus.tools.func_tool import trans_to_function_tool

        return [
            trans_to_function_tool(self.read_file),
            trans_to_function_tool(self.glob),
            trans_to_function_tool(self.grep),
        ]

    @staticmethod
    def all_tools_name() -> List[str]:
        return ["read_file", "glob", "grep"]

    # ------------------------------------------------------------- read/walk

    def read_file(self, path: str, offset: int = 0, limit: int = 0) -> FuncToolResult:
        """
        Read the contents of a file from the in-memory bundle.

        Args:
            path: Slug-relative path. ``"./prefix"`` and ``"/prefix"`` are
                accepted and normalized; ``"."`` is the bundle root and not a
                file.
            offset: 1-based line number to start reading from. 0 reads from
                the beginning.
            limit: Maximum number of lines to read. 0 reads everything.

        Returns:
            FuncToolResult with the file body (or numbered slice in
            ``"N: line"`` form when offset/limit are set). Size cap matches
            the disk-backed tool so the LLM sees identical truncation
            behavior.
        """
        try:
            key = self._normalize(path)
            if key not in self._files:
                return FuncToolResult(success=0, error=f"File not found: {path}")

            content = self._files[key]
            use_slice = offset > 0 or limit > 0

            if use_slice:
                lines = content.split("\n")
                start = max(0, offset - 1) if offset > 0 else 0
                end = start + limit if limit > 0 else len(lines)
                selected = lines[start:end]
                numbered = [f"{start + i + 1}: {line}" for i, line in enumerate(selected)]
                result = "\n".join(numbered)
                if len(result.encode("utf-8")) > _MAX_READ_BYTES:
                    return FuncToolResult(
                        success=0,
                        error=(
                            f"Read slice too large: {path} "
                            f"(limit={_MAX_READ_BYTES} bytes; reduce 'limit' to read a smaller range)"
                        ),
                    )
                return FuncToolResult(result=result)

            if len(content.encode("utf-8")) > _MAX_READ_BYTES:
                return FuncToolResult(
                    success=0,
                    error=(
                        f"File too large: {path} "
                        f"(limit={_MAX_READ_BYTES} bytes; use 'offset'/'limit' to read in chunks)"
                    ),
                )
            return FuncToolResult(result=content)
        except Exception as exc:
            logger.error(f"MemoryFilesystemFuncTool.read_file failed for {path}: {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def glob(self, pattern: str, path: str = ".") -> FuncToolResult:
        """
        Find files matching a glob pattern within the bundle.

        Args:
            pattern: Glob pattern, e.g. ``"*.sql"``, ``"queries/*.json"``,
                ``"**/*.md"``. Matched against bundle-relative paths.
            path: Subdirectory to scope the search to. ``"."`` is the bundle
                root. Pattern is applied to paths relative to ``path``.

        Returns:
            FuncToolResult with ``{"files": [...], "truncated": bool}``,
            shape-matched to the disk-backed tool. Files are reported in
            slug-relative form (consistent with what the LLM passes to
            ``read_file``).
        """
        try:
            prefix = self._normalize_dir(path)
            scoped = list(self._scoped_paths(prefix))

            # Collect one match past the cap so we can distinguish
            # "exactly the cap" (not truncated) from "actually had more"
            # (truncated). Breaking at ``>= cap`` would conflate the two
            # and falsely flag a bundle with exactly ``_MAX_GLOB_RESULTS``
            # hits as truncated.
            matches: List[str] = []
            for full_key in scoped:
                rel = full_key[len(prefix) :].lstrip("/") if prefix else full_key
                try:
                    matched = wc_glob.globmatch(rel, pattern, flags=wc_glob.DOTGLOB | wc_glob.GLOBSTAR)
                except Exception:
                    matched = rel == pattern
                if matched:
                    matches.append(full_key)
                    if len(matches) > _MAX_GLOB_RESULTS:
                        break

            truncated = len(matches) > _MAX_GLOB_RESULTS
            if truncated:
                matches = matches[:_MAX_GLOB_RESULTS]
            result: dict = {"files": matches, "truncated": truncated}
            if truncated:
                result["message"] = (
                    f"Results truncated to {_MAX_GLOB_RESULTS}. Use a more specific pattern to narrow results."
                )
            return FuncToolResult(result=result)
        except Exception as exc:
            logger.exception(f"MemoryFilesystemFuncTool.glob failed for {pattern} in {path}")
            return FuncToolResult(success=0, error=str(exc))

    def grep(
        self,
        pattern: str,
        path: str = ".",
        include: str = "",
        case_sensitive: bool = True,
    ) -> FuncToolResult:
        """
        Search bundle file contents using a regular expression.

        Args:
            pattern: Python regex applied per line.
            path: Subdirectory to scope the search to. ``"."`` is the bundle
                root.
            include: Optional glob filter on file names (e.g. ``"*.sql"``).
            case_sensitive: Whether the regex is case-sensitive.

        Returns:
            FuncToolResult with ``{"matches": [{"file", "line", "content"}],
            "truncated": bool}``, shape-matched to the disk-backed tool.
        """
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                compiled = re.compile(pattern, flags)
            except re.error as exc:
                return FuncToolResult(success=0, error=f"Invalid regex pattern: {exc}")

            prefix = self._normalize_dir(path)
            scoped = list(self._scoped_paths(prefix))

            matches: List[dict] = []
            for full_key in scoped:
                if include:
                    name = full_key.rsplit("/", 1)[-1]
                    try:
                        if not wc_glob.globmatch(name, include, flags=wc_glob.DOTGLOB | wc_glob.GLOBSTAR):
                            continue
                    except Exception:
                        if name != include:
                            continue

                body = self._files[full_key]
                if len(body.encode("utf-8")) > _MAX_READ_BYTES:
                    continue
                # Same overshoot-by-one strategy as ``glob`` so a bundle
                # with exactly ``_MAX_GREP_MATCHES`` hits isn't falsely
                # flagged truncated. We break inside the line loop AND
                # the outer loop once the overshoot is reached.
                for line_num, line in enumerate(body.split("\n"), start=1):
                    if compiled.search(line):
                        matches.append({"file": full_key, "line": line_num, "content": line.rstrip()})
                        if len(matches) > _MAX_GREP_MATCHES:
                            break
                if len(matches) > _MAX_GREP_MATCHES:
                    break

            truncated = len(matches) > _MAX_GREP_MATCHES
            if truncated:
                matches = matches[:_MAX_GREP_MATCHES]
            return FuncToolResult(result={"matches": matches, "truncated": truncated})
        except Exception as exc:
            logger.exception(f"MemoryFilesystemFuncTool.grep failed for {pattern} in {path}")
            return FuncToolResult(success=0, error=str(exc))

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _normalize(path: str) -> str:
        if not path:
            return ""
        normalized = path.replace("\\", "/").strip()
        while normalized.startswith("./"):
            normalized = normalized[2:]
        normalized = normalized.lstrip("/")
        return normalized

    def _normalize_dir(self, path: str) -> str:
        """Return a directory key (no trailing slash) usable as a path prefix.

        ``"."`` and ``""`` map to ``""`` (root). Other inputs are normalized
        the same way as file paths and trailing slashes are stripped.
        """
        if not path or path == ".":
            return ""
        return self._normalize(path).rstrip("/")

    def _scoped_paths(self, prefix: str) -> Iterable[str]:
        """Yield bundle paths under ``prefix`` (``""`` selects everything)."""
        if not prefix:
            yield from self._files.keys()
            return
        prefix_with_sep = prefix + "/"
        for key in self._files.keys():
            if key == prefix or key.startswith(prefix_with_sep):
                yield key
