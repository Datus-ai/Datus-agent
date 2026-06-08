# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
import re
from pathlib import Path
from typing import Iterator, List, Optional

from agents import Tool
from wcmatch import glob as wc_glob

from datus.tools import BaseTool
from datus.tools.func_tool import FuncToolResult
from datus.tools.func_tool.fs_path_policy import (
    PathZone,
    ResolvedPath,
    classify_path,
    whitelist_anchors,
)
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class FilesystemConfig:
    """Configuration for filesystem operations"""

    def __init__(
        self,
        root_path: str = None,
        allowed_extensions: List[str] = None,
        max_file_size: int = 200 * 1024,
    ):
        self.root_path = root_path or os.getcwd()
        self.allowed_extensions = allowed_extensions or [
            ".txt",
            ".md",
            ".py",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".json",
            ".yaml",
            ".yml",
            ".csv",
            ".sql",
            ".j2",
            ".html",
            ".css",
            ".xml",
        ]
        self.max_file_size = max_file_size


class FilesystemFuncTool(BaseTool):
    """Function tool wrapper for filesystem operations.

    Path resolution is centralized in :func:`classify_path`. Every operation
    runs the same classifier and then branches on the resulting zone:

    * ``INTERNAL`` / ``WHITELIST`` — proceed.
    * ``HIDDEN`` — return a ``File not found`` style error for reads/writes
      and prune the subtree in walks, so ``.datus/sessions`` etc. stay
      invisible to the LLM.
    * ``EXTERNAL`` — proceed at the tool level; the ``PermissionHooks`` layer
      is responsible for asking the user first.
    """

    def __init__(
        self,
        root_path: str = None,
        *,
        current_node: Optional[str] = None,
        datus_home: Optional[str] = None,
        strict: bool = False,
        inherited_memory_node: Optional[str] = None,
        session_data_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            strict: When ``True``, ``EXTERNAL`` paths (anything outside the
                project root and its whitelist) are rejected at the tool layer
                with the same "not found" semantics as ``HIDDEN``. This is the
                mode the API / gateway surfaces run in — the agent has no
                interactive broker to confirm external access, so we fail
                closed instead of ever touching the host filesystem.
                ``False`` (the CLI default) lets ``PermissionHooks`` prompt
                the user.
            inherited_memory_node: When set, the path policy treats
                ``.datus/memory/{inherited_memory_node}/**`` as a read-only
                whitelisted subtree so the calling node can ``Read``/``Glob``
                its parent's memory. Writes/edits to that subtree are rejected
                at the tool layer with a clear error message.
            session_data_dir: When set, the compact-archive directory for the
                current session (``path_manager.session_data_dir(session_id)``)
                qualifies as a read-only WHITELIST anchor so the LLM can
                ``read_file`` archived tool I/O without triggering a permission
                prompt. Other sessions' archive directories stay EXTERNAL.
        """
        super().__init__(**kwargs)
        self.root_path = root_path or os.getcwd()
        self.config = FilesystemConfig(root_path=self.root_path)
        self._current_node = current_node
        self._datus_home = Path(datus_home).expanduser().resolve(strict=False) if datus_home else None
        self._root_resolved = Path(self.root_path).expanduser().resolve(strict=False)
        self._strict = strict
        self._inherited_memory_node = inherited_memory_node
        self._session_data_dir = Path(session_data_dir).expanduser().resolve(strict=False) if session_data_dir else None

    @property
    def strict(self) -> bool:
        return self._strict

    def available_tools(self) -> List[Tool]:
        """Get all available filesystem tools"""
        from datus.tools.func_tool import trans_to_function_tool

        bound_tools = []
        methods_to_convert = [
            self.read_file,
            self.write_file,
            self.edit_file,
            self.delete_file,
            self.glob,
            self.grep,
        ]

        for bound_method in methods_to_convert:
            bound_tools.append(trans_to_function_tool(bound_method))
        return bound_tools

    # ``BaseTool`` framework methods that aren't tools themselves — must be
    # filtered out of the introspection result so the saas tool catalog only
    # carries actual filesystem operations.
    _BASE_TOOL_FRAMEWORK_METHODS: frozenset = frozenset({"call_action", "get_actions", "set_tool_context"})

    @staticmethod
    def all_tools_name() -> List[str]:
        """Return every public tool method on this class.

        Mirrors ``DBFuncTool.all_tools_name()`` so ``VALID_TOOL_METHODS``
        in ``datus.api.services.agent_service`` can derive the
        ``filesystem_tools`` set by introspection instead of hand-curating
        it — the latter has drifted from the runtime in the past
        (``delete_file`` was missing for a while).
        """
        from datus.utils.class_utils import get_public_instance_methods

        return [
            name
            for name in get_public_instance_methods(FilesystemFuncTool).keys()
            if name != "available_tools" and name not in FilesystemFuncTool._BASE_TOOL_FRAMEWORK_METHODS
        ]

    # ------------------------------------------------------------------ zones

    def _classify(self, path: str) -> ResolvedPath:
        return classify_path(
            path,
            root_path=self._root_resolved,
            current_node=self._current_node,
            datus_home=self._datus_home,
            inherited_memory_node=self._inherited_memory_node,
            session_data_dir=self._session_data_dir,
        )

    def _read_only_reject(self, resolved: ResolvedPath) -> FuncToolResult:
        """Reject writes to a read-only whitelist (inherited parent memory)."""
        return FuncToolResult(
            success=0,
            error=(
                f"Read-only path: {resolved.display} is inherited from the parent "
                "agent's memory and cannot be modified by this sub-agent."
            ),
        )

    def _not_found(self, resolved: ResolvedPath) -> FuncToolResult:
        """Uniform ``File not found`` response for hidden zones.

        We deliberately do not distinguish between "really missing" and
        "hidden by policy" — leaking that ``.datus/sessions`` exists would
        defeat the invisibility guarantee.
        """
        return FuncToolResult(success=0, error=f"File not found: {resolved.display}")

    def _strict_reject(self, resolved: ResolvedPath) -> FuncToolResult:
        """Error response for ``EXTERNAL`` paths in strict mode.

        Unlike ``_not_found``, this is explicit: the caller **asked** for a
        path outside the workspace, so hiding the rejection would be
        confusing. The error message names the path so the LLM can fix it
        on the next turn. Used by the API / gateway surfaces that have no
        interactive broker to prompt the user.
        """
        return FuncToolResult(
            success=0,
            error=f"Path outside workspace is not allowed in strict mode: {resolved.display}",
        )

    def _get_safe_path(self, path: str) -> Optional[Path]:
        """Deprecated sandbox helper kept for backward compat.

        Delegates to :meth:`_classify`; returns ``None`` for ``HIDDEN`` or
        ``EXTERNAL`` to preserve the historical "reject out-of-sandbox"
        semantics used by a handful of callers outside this class.
        """
        resolved = self._classify(path)
        if resolved.zone in (PathZone.HIDDEN, PathZone.EXTERNAL):
            return None
        return resolved.resolved

    def _is_allowed_file(self, file_path: Path) -> bool:
        """Check if file extension is allowed"""
        if not self.config.allowed_extensions:
            return True
        return file_path.suffix.lower() in self.config.allowed_extensions

    # ------------------------------------------------------------- read/write

    def read_file(self, path: str, offset: int = 0, limit: int = 0) -> FuncToolResult:
        """
        Read the contents of a file.

        Args:
            path: Path to the file. Relative paths are resolved under the
                project root; absolute paths are permitted but the permission
                layer will prompt the user when they fall outside the project.
            offset: Line number to start reading from (1-based). 0 means start from beginning.
            limit: Maximum number of lines to read. 0 means read all lines.

        ``config.max_file_size`` bounds a single read: when neither ``offset``
        nor ``limit`` is set it caps the whole-file size; otherwise it caps the
        sliced output, so large files can still be read in chunks.

        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure.
                  - 'error' (Optional[str]): Error message on failure.
                  - 'result' (Optional[str]): File contents on success. When offset/limit are set,
                    returns numbered lines in "N: line content" format.
        """
        try:
            resolved = self._classify(path)
            if resolved.zone == PathZone.HIDDEN:
                return self._not_found(resolved)
            if self._strict and resolved.zone == PathZone.EXTERNAL:
                return self._strict_reject(resolved)

            target_path = resolved.resolved
            if not target_path.exists():
                return FuncToolResult(success=0, error=f"File not found: {resolved.display}")

            if not target_path.is_file():
                return FuncToolResult(success=0, error=f"Path is not a file: {resolved.display}")

            if not self._is_allowed_file(target_path):
                return FuncToolResult(success=0, error=f"File type not allowed: {resolved.display}")

            max_bytes = self.config.max_file_size
            use_slice = offset > 0 or limit > 0

            try:
                if use_slice:
                    start = max(0, offset - 1) if offset > 0 else 0
                    with target_path.open("r", encoding="utf-8") as fh:
                        selected: List[str] = []
                        for idx, line in enumerate(fh):
                            if idx < start:
                                continue
                            if limit > 0 and len(selected) >= limit:
                                break
                            selected.append(line.rstrip("\n"))
                    numbered = [f"{start + i + 1}: {line}" for i, line in enumerate(selected)]
                    result = "\n".join(numbered)
                    if len(result.encode("utf-8")) > max_bytes:
                        return FuncToolResult(
                            success=0,
                            error=(
                                f"Read slice too large: {resolved.display} "
                                f"(limit={max_bytes} bytes; reduce 'limit' to read a smaller range)"
                            ),
                        )
                    return FuncToolResult(result=result)

                if target_path.stat().st_size > max_bytes:
                    return FuncToolResult(
                        success=0,
                        error=(
                            f"File too large: {resolved.display} "
                            f"(limit={max_bytes} bytes; use 'offset'/'limit' to read in chunks)"
                        ),
                    )
                content = target_path.read_text(encoding="utf-8")
                return FuncToolResult(result=content)
            except UnicodeDecodeError:
                return FuncToolResult(success=0, error=f"Cannot read binary file: {resolved.display}")
            except PermissionError:
                return FuncToolResult(success=0, error=f"Permission denied: {resolved.display}")

        except Exception as e:
            logger.error(f"Error reading file {path}: {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def write_file(self, path: str, content: str, file_type: str = "") -> FuncToolResult:
        """
        Create a new file or overwrite an existing file.

        Args:
            path: Target path. Relative paths are resolved under the project
                root. Absolute paths require user confirmation via the
                permission hook.
            content: The content to write to the file.
            file_type: Optional tag consumed by ``GenerationHooks`` for
                post-write sync-to-DB routing; has no effect on where the
                file lands. The prompt is responsible for supplying the
                correct directory in ``path``.

        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure.
                  - 'error' (Optional[str]): Error message on failure.
                  - 'result' (Optional[str]): Success message on success.
        """
        del file_type
        try:
            resolved = self._classify(path)
            if resolved.zone == PathZone.HIDDEN:
                return self._not_found(resolved)
            if self._strict and resolved.zone == PathZone.EXTERNAL:
                return self._strict_reject(resolved)
            if resolved.read_only:
                return self._read_only_reject(resolved)

            target_path = resolved.resolved
            if not self._is_allowed_file(target_path):
                return FuncToolResult(success=0, error=f"File type not allowed: {resolved.display}")

            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(content, encoding="utf-8")
                return FuncToolResult(result=f"File written successfully: {resolved.display}")
            except PermissionError:
                return FuncToolResult(success=0, error=f"Permission denied: {resolved.display}")

        except Exception as e:
            logger.error(f"Error writing file {path}: {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def delete_file(self, path: str) -> FuncToolResult:
        """
        Delete a single regular file.

        Use this to remove a file that's no longer needed (e.g. a previously
        emitted component that the report no longer references). The path
        rules mirror ``write_file`` / ``edit_file``: same zone classification,
        same read-only / external / hidden gating.

        Args:
            path: Target path. Relative paths are resolved under the project
                root. Refusing to delete directories is intentional — bulk
                cleanup is a different operation and we don't surface it.

        Returns:
            dict: A dictionary with the execution result, containing:
                  - 'success' (int): 1 for success, 0 for failure.
                  - 'error' (Optional[str]): Error message on failure.
                  - 'result' (Optional[str]): Success message on success.
        """
        try:
            resolved = self._classify(path)
            if resolved.zone == PathZone.HIDDEN:
                return self._not_found(resolved)
            if self._strict and resolved.zone == PathZone.EXTERNAL:
                return self._strict_reject(resolved)
            if resolved.read_only:
                return self._read_only_reject(resolved)

            target_path = resolved.resolved
            if not target_path.exists():
                return FuncToolResult(success=0, error=f"File not found: {resolved.display}")
            if target_path.is_dir():
                return FuncToolResult(
                    success=0,
                    error=(
                        f"Path is a directory, not a file: {resolved.display}. "
                        "delete_file refuses to remove directories — handle them explicitly."
                    ),
                )
            if not target_path.is_file():
                return FuncToolResult(success=0, error=f"Path is not a regular file: {resolved.display}")

            if not self._is_allowed_file(target_path):
                return FuncToolResult(success=0, error=f"File type not allowed: {resolved.display}")

            try:
                target_path.unlink()
                return FuncToolResult(result=f"File deleted successfully: {resolved.display}")
            except PermissionError:
                return FuncToolResult(success=0, error=f"Permission denied: {resolved.display}")
            except OSError as exc:
                return FuncToolResult(success=0, error=f"Failed to delete file: {exc}")

        except Exception as e:
            logger.error(f"Error deleting file {path}: {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def edit_file(self, path: str, old_string: str, new_string: str) -> FuncToolResult:
        """
        Make a single edit to a file by replacing old_string with new_string.

        Args:
            path: Target path, resolved the same way as ``write_file``.
            old_string: The text to find and replace. Must match exactly once in the file.
            new_string: The text to replace old_string with.

        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure.
                  - 'error' (Optional[str]): Error message on failure.
                  - 'result' (Optional[str]): Success message on success.
        """
        try:
            if not old_string:
                return FuncToolResult(success=0, error="old_string must not be empty")

            resolved = self._classify(path)
            if resolved.zone == PathZone.HIDDEN:
                return self._not_found(resolved)
            if self._strict and resolved.zone == PathZone.EXTERNAL:
                return self._strict_reject(resolved)
            if resolved.read_only:
                return self._read_only_reject(resolved)

            target_path = resolved.resolved
            if not target_path.exists():
                return FuncToolResult(success=0, error=f"File not found: {resolved.display}")

            if not target_path.is_file():
                return FuncToolResult(success=0, error=f"Path is not a file: {resolved.display}")

            if not self._is_allowed_file(target_path):
                return FuncToolResult(success=0, error=f"File type not allowed: {resolved.display}")

            try:
                content = target_path.read_text(encoding="utf-8")
                match_count = content.count(old_string)

                if match_count == 0:
                    preview = old_string[:100] + "..." if len(old_string) > 100 else old_string
                    return FuncToolResult(
                        success=0,
                        error=f"old_string not found in file. Looking for: {preview}",
                    )

                if match_count > 1:
                    return FuncToolResult(
                        success=0,
                        error=f"old_string matches {match_count} times in file. It must match exactly once. "
                        "Provide more surrounding context to make the match unique.",
                    )

                content = content.replace(old_string, new_string, 1)
                target_path.write_text(content, encoding="utf-8")
                return FuncToolResult(result=f"File edited successfully: {resolved.display}")
            except UnicodeDecodeError:
                return FuncToolResult(success=0, error=f"Cannot edit binary file: {resolved.display}")
            except PermissionError:
                return FuncToolResult(success=0, error=f"Permission denied: {resolved.display}")

        except Exception as e:
            logger.error(f"Error editing file {path}: {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    # ------------------------------------------------------------------ walks

    # Minimal fallback excludes when no .gitignore is found
    _FALLBACK_EXCLUDE_DIRS = {".git", "__pycache__", "node_modules"}

    def _load_gitignore_patterns(self, search_root: Path) -> List[str]:
        """Load exclude patterns from .gitignore in the search root or its ancestors.

        Walks up from search_root to self.config.root_path looking for .gitignore.
        Parses non-comment, non-empty lines and converts to glob patterns.
        Always excludes .git directory.
        """
        patterns = [".git", ".git/**", "**/.git/**"]

        root_resolved = Path(self.config.root_path).resolve(strict=False)
        current = search_root.resolve(strict=False)
        gitignore_path = None
        while True:
            candidate = current / ".gitignore"
            if candidate.is_file():
                gitignore_path = candidate
                break
            if current == root_resolved or current == current.parent:
                break
            current = current.parent

        if not gitignore_path:
            for d in self._FALLBACK_EXCLUDE_DIRS:
                patterns.extend([d, f"{d}/**", f"**/{d}/**"])
            return patterns

        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("!"):
                        continue
                    entry = line.lstrip("/")
                    if entry.endswith("/"):
                        dir_name = entry.rstrip("/")
                        patterns.append(dir_name)
                        patterns.append(f"**/{dir_name}")
                    patterns.append(entry)
                    if not entry.endswith("/**"):
                        patterns.append(f"{entry}/**")
                    if not entry.startswith("**/"):
                        patterns.append(f"**/{entry}")
        except Exception as e:
            logger.warning(f"Failed to fully parse .gitignore at {gitignore_path}: {e}")

        return patterns

    def _walk_files(self, seed: ResolvedPath, include_pattern: str = "") -> Iterator[Path]:
        """Walk a directory tree yielding files.

        Traversal honors:

        * Gitignore entries under the seed.
        * Symlink safety: a symlink target outside the project root is only
          followed when the seed itself was ``EXTERNAL`` (the hook confirmed
          it). For ``INTERNAL``/``WHITELIST`` seeds we re-classify each
          resolved item so a symlink pointing at ``~/secrets`` is skipped.
        * ``HIDDEN`` prune: every entry is classified; ``HIDDEN`` subtrees
          are skipped entirely, which keeps ``.datus/sessions`` etc. invisible
          even if the LLM happens to search from project root.
        """
        target_path = seed.resolved
        if seed.zone == PathZone.HIDDEN or not target_path.exists() or not target_path.is_dir():
            return

        seed_is_project_relative = seed.zone in (PathZone.INTERNAL, PathZone.WHITELIST)
        exclude_patterns = self._load_gitignore_patterns(target_path)
        visited_inodes = set()
        anchors = whitelist_anchors(
            root_path=self._root_resolved,
            current_node=self._current_node,
            datus_home=self._datus_home,
            inherited_memory_node=self._inherited_memory_node,
            session_data_dir=self._session_data_dir,
        )

        def has_whitelisted_descendant(directory: Path) -> bool:
            """Is any whitelist anchor strictly underneath ``directory``?

            Needed because ``.datus/`` itself classifies HIDDEN even though
            ``.datus/skills/`` inside it is WHITELIST — we still have to
            descend into the HIDDEN parent to reach the visible subtree.
            """
            try:
                for anchor in anchors:
                    if anchor == directory:
                        return True
                    try:
                        anchor.relative_to(directory)
                        return True
                    except ValueError:
                        continue
            except Exception:
                return False
            return False

        def should_gitignore_exclude(file_path: Path) -> bool:
            try:
                relative_path = str(file_path.relative_to(target_path))
            except ValueError:
                return False
            for exclude_pattern in exclude_patterns:
                try:
                    if wc_glob.globmatch(relative_path, exclude_pattern, flags=wc_glob.DOTGLOB | wc_glob.GLOBSTAR):
                        return True
                except Exception:
                    continue
            return False

        def walk_recursive(current_path: Path):
            try:
                try:
                    current_inode = current_path.stat().st_ino
                except OSError:
                    return

                if current_inode in visited_inodes:
                    return
                visited_inodes.add(current_inode)

                for item in current_path.iterdir():
                    try:
                        if should_gitignore_exclude(item):
                            continue

                        item_resolved = item.resolve(strict=False)

                        # Classify every resolved item; HIDDEN subtrees are
                        # pruned regardless of where the walk started, unless
                        # they contain a whitelist anchor further inside (in
                        # which case we descend but never yield at the HIDDEN
                        # level itself).
                        item_is_hidden = False
                        if seed_is_project_relative:
                            item_zone = classify_path(
                                str(item_resolved),
                                root_path=self._root_resolved,
                                current_node=self._current_node,
                                datus_home=self._datus_home,
                                inherited_memory_node=self._inherited_memory_node,
                                session_data_dir=self._session_data_dir,
                            ).zone
                            if item_zone == PathZone.EXTERNAL:
                                # Symlink escape from project tree; skip.
                                continue
                            if item_zone == PathZone.HIDDEN:
                                if item_resolved.is_dir() and has_whitelisted_descendant(item_resolved):
                                    item_is_hidden = True  # descend but don't yield files here
                                else:
                                    continue

                        if item_resolved.is_dir():
                            yield from walk_recursive(item_resolved)
                        elif item_resolved.is_file() and not item_is_hidden:
                            if include_pattern:
                                if not wc_glob.globmatch(
                                    item.name, include_pattern, flags=wc_glob.DOTGLOB | wc_glob.GLOBSTAR
                                ):
                                    continue
                            yield item_resolved
                    except OSError:
                        continue
            except OSError:
                return

        yield from walk_recursive(target_path)

    @staticmethod
    def _glob_part_has_magic(part: str) -> bool:
        return any(ch in part for ch in "*?[{")

    def _rescope_path_pattern_glob(self, pattern: str, path: str) -> tuple[str, str]:
        """Treat path-like glob patterns as an explicit seed directory.

        ``glob(pattern="subject/foo/*.yml", path=".")`` should search
        ``subject/foo`` even when ``subject/`` is gitignored from a root walk.
        """
        if not pattern or os.path.isabs(pattern):
            return pattern, path

        normalized_pattern = pattern.replace("\\", "/")
        if "/" not in normalized_pattern:
            return pattern, path

        parts = [part for part in normalized_pattern.split("/") if part not in ("", ".")]
        if len(parts) <= 1:
            return pattern, path

        fixed_parts: List[str] = []
        remainder_parts: List[str] = []
        for idx, part in enumerate(parts):
            if self._glob_part_has_magic(part):
                remainder_parts = parts[idx:]
                break
            fixed_parts.append(part)
        else:
            fixed_parts = parts[:-1]
            remainder_parts = parts[-1:]

        if not fixed_parts or not remainder_parts:
            return pattern, path

        fixed_prefix = "/".join(fixed_parts)
        remainder = "/".join(remainder_parts)
        candidates = []
        if path not in ("", ".", "./"):
            candidates.append(os.path.join(path, fixed_prefix))
        candidates.append(fixed_prefix)

        for candidate in candidates:
            resolved = self._classify(candidate)
            if resolved.zone == PathZone.HIDDEN:
                continue
            if resolved.resolved.exists() and resolved.resolved.is_dir():
                return remainder, candidate
        return pattern, path

    def glob(self, pattern: str, path: str = ".") -> FuncToolResult:
        """
        Find files matching a glob pattern.

        Args:
            pattern: Glob pattern to match (e.g., "*.py", "**/*.yaml", "src/**/*.ts").
            path: Starting directory for the search. Defaults to workspace root ".".

        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure.
                  - 'error' (Optional[str]): Error message on failure.
                  - 'result' (Optional[dict]): Dict with 'files' (list of paths), 'truncated' (bool),
                    and — when the search seed fell outside the project root — ``'external': True``
                    so the caller knows reported paths are absolute.
        """
        max_results = 200
        try:
            pattern, path = self._rescope_path_pattern_glob(pattern, path)
            seed = self._classify(path)
            if seed.zone == PathZone.HIDDEN:
                return FuncToolResult(result={"files": [], "truncated": False})
            if self._strict and seed.zone == PathZone.EXTERNAL:
                return self._strict_reject(seed)

            target_path = seed.resolved
            if not target_path.exists():
                return FuncToolResult(success=0, error=f"Directory not found: {seed.display}")
            if not target_path.is_dir():
                return FuncToolResult(success=0, error=f"Path is not a directory: {seed.display}")

            report_relative_to: Optional[Path] = None
            if seed.zone in (PathZone.INTERNAL, PathZone.WHITELIST):
                report_relative_to = self._root_resolved

            matches: List[str] = []
            for file_path in self._walk_files(seed):
                try:
                    match_rel = str(file_path.relative_to(target_path))
                except ValueError:
                    match_rel = str(file_path)

                try:
                    matched = wc_glob.globmatch(match_rel, pattern, flags=wc_glob.DOTGLOB | wc_glob.GLOBSTAR)
                except Exception:
                    matched = file_path.name == pattern

                if not matched:
                    continue

                if report_relative_to is not None:
                    try:
                        reported = str(file_path.relative_to(report_relative_to))
                    except ValueError:
                        reported = str(file_path)
                else:
                    reported = str(file_path)
                matches.append(reported)
                if len(matches) >= max_results:
                    break

            truncated = len(matches) >= max_results
            result_data: dict = {
                "files": matches,
                "truncated": truncated,
            }
            if seed.zone == PathZone.EXTERNAL:
                result_data["external"] = True
            if truncated:
                result_data["message"] = (
                    f"Results truncated to {max_results}. Use a more specific pattern to narrow results."
                )
            return FuncToolResult(result=result_data)

        except Exception as e:
            logger.exception(f"Error in glob search for {pattern} in {path}")
            return FuncToolResult(success=0, error=str(e))

    def grep(self, pattern: str, path: str = ".", include: str = "", case_sensitive: bool = True) -> FuncToolResult:
        """
        Search file contents using a regular expression pattern.

        Args:
            pattern: Regular expression pattern to search for.
            path: Starting directory for the search. Defaults to workspace root ".".
            include: Optional glob pattern to filter files (e.g., "*.py", "*.sql").
            case_sensitive: Whether the search is case-sensitive. Defaults to True.

        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure.
                  - 'error' (Optional[str]): Error message on failure.
                  - 'result' (Optional[dict]): Dict with 'matches' (list of {file, line, content}) and 'truncated'.
        """
        max_matches = 100
        try:
            seed = self._classify(path)
            if seed.zone == PathZone.HIDDEN:
                return FuncToolResult(result={"matches": [], "truncated": False})
            if self._strict and seed.zone == PathZone.EXTERNAL:
                return self._strict_reject(seed)

            target_path = seed.resolved
            if not target_path.exists():
                return FuncToolResult(success=0, error=f"Path not found: {seed.display}")

            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                compiled = re.compile(pattern, flags)
            except re.error as e:
                return FuncToolResult(success=0, error=f"Invalid regex pattern: {str(e)}")

            report_relative_to: Optional[Path] = None
            if seed.zone in (PathZone.INTERNAL, PathZone.WHITELIST):
                report_relative_to = self._root_resolved

            def _search_file(file_path: Path, reported_file: str) -> List[dict]:
                if not self._is_allowed_file(file_path):
                    return []
                try:
                    if file_path.stat().st_size > self.config.max_file_size:
                        return []
                except OSError:
                    return []
                try:
                    content = file_path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, PermissionError, OSError):
                    return []
                found = []
                for line_num, line in enumerate(content.split("\n"), start=1):
                    if compiled.search(line):
                        found.append({"file": reported_file, "line": line_num, "content": line.rstrip()})
                return found

            matches: List[dict] = []

            if target_path.is_file():
                if report_relative_to is not None:
                    try:
                        reported_file = str(target_path.relative_to(report_relative_to))
                    except ValueError:
                        reported_file = str(target_path)
                else:
                    reported_file = str(target_path)
                matches = _search_file(target_path, reported_file)[:max_matches]
            elif target_path.is_dir():
                for file_path in self._walk_files(seed, include_pattern=include):
                    if report_relative_to is not None:
                        try:
                            reported_file = str(file_path.relative_to(report_relative_to))
                        except ValueError:
                            reported_file = str(file_path)
                    else:
                        reported_file = str(file_path)

                    for m in _search_file(file_path, reported_file):
                        matches.append(m)
                        if len(matches) >= max_matches:
                            break
                    if len(matches) >= max_matches:
                        break
            else:
                return FuncToolResult(success=0, error=f"Path is not a file or directory: {seed.display}")

            truncated = len(matches) >= max_matches
            result_data: dict = {
                "matches": matches,
                "truncated": truncated,
            }
            if seed.zone == PathZone.EXTERNAL:
                result_data["external"] = True
            return FuncToolResult(result=result_data)

        except Exception as e:
            logger.exception(f"Error in grep search for {pattern} in {path}")
            return FuncToolResult(success=0, error=str(e))


def filesystem_function_tools(
    root_path: str = None,
    *,
    current_node: Optional[str] = None,
    strict: bool = False,
    inherited_memory_node: Optional[str] = None,
    session_data_dir: Optional[str] = None,
) -> List[Tool]:
    """Get filesystem function tools"""
    return FilesystemFuncTool(
        root_path=root_path,
        current_node=current_node,
        strict=strict,
        inherited_memory_node=inherited_memory_node,
        session_data_dir=session_data_dir,
    ).available_tools()
