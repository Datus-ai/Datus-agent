# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import inspect
import os
import re
from pathlib import Path
from typing import Callable, Iterator, List, Optional

from agents import Tool
from wcmatch import glob as wc_glob

from datus.tools import BaseTool
from datus.tools.func_tool import FuncToolResult
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class FilesystemConfig:
    """Configuration for filesystem operations"""

    def __init__(
        self,
        root_path: str = None,
        allowed_extensions: List[str] = None,
        max_file_size: int = 1024 * 1024,
    ):
        self.root_path = root_path or os.getenv("FILESYSTEM_MCP_PATH", os.path.expanduser("~"))
        self.allowed_extensions = allowed_extensions or [
            ".txt",
            ".md",
            ".py",
            ".js",
            ".ts",
            ".json",
            ".yaml",
            ".yml",
            ".csv",
            ".sql",
            ".html",
            ".css",
            ".xml",
        ]
        self.max_file_size = max_file_size


PathNormalizer = Callable[[str, Optional[str]], str]


class FilesystemFuncTool(BaseTool):
    """Function tool wrapper for filesystem operations"""

    def __init__(
        self,
        root_path: str = None,
        *,
        path_normalizer: Optional[PathNormalizer] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.root_path = root_path or os.getenv("FILESYSTEM_MCP_PATH", os.path.expanduser("~"))
        self.config = FilesystemConfig(root_path=root_path)
        self._path_normalizer = path_normalizer
        # Detect strict_kind support via signature inspection up front so a
        # TypeError raised *inside* the normalizer can't be mistaken for a
        # legacy 2-arg signature and silently drop the strict flag.
        self._normalizer_accepts_strict_kind = False
        if path_normalizer is not None:
            try:
                params = inspect.signature(path_normalizer).parameters
                self._normalizer_accepts_strict_kind = "strict_kind" in params or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
            except (TypeError, ValueError):
                pass

    def _normalize(self, path: str, file_type: Optional[str] = None, *, strict: bool = False) -> str:
        """
        Apply the configured path normalizer (if any) before sandbox resolution.

        With ``strict=False`` (default, read-side), normalizer errors are logged
        and the original path is returned so the downstream sandbox check can
        fail naturally. With ``strict=True`` (write-side), the exception is
        re-raised so callers don't silently land a mutation at a mis-normalized
        location. ``strict=True`` is also forwarded to the normalizer as the
        ``strict_kind`` kwarg so KB normalizers can enforce cross-kind write
        restrictions on mutating ops while keeping reads lax.
        """
        if self._path_normalizer is None or not path:
            return path
        try:
            if self._normalizer_accepts_strict_kind:
                return self._path_normalizer(path, file_type, strict_kind=strict)
            return self._path_normalizer(path, file_type)
        except Exception as e:
            logger.warning(f"path_normalizer raised on path={path!r} file_type={file_type!r}: {e}")
            if strict:
                raise
            return path

    def available_tools(self) -> List[Tool]:
        """Get all available filesystem tools"""
        from datus.tools.func_tool import trans_to_function_tool

        bound_tools = []
        methods_to_convert = [
            self.read_file,
            self.write_file,
            self.edit_file,
            self.glob,
            self.grep,
        ]

        for bound_method in methods_to_convert:
            bound_tools.append(trans_to_function_tool(bound_method))
        return bound_tools

    def _get_safe_path(self, path: str) -> Optional[Path]:
        """Get a safe path within the root directory.

        Uses ``Path.relative_to`` instead of string ``startswith`` so that
        sibling directories whose names share the root's prefix (e.g. a
        ``knowledge_base_home_backup`` sitting next to ``knowledge_base_home``)
        can't be mistaken for an in-sandbox path via ``../`` traversal.
        """
        try:
            root = Path(self.config.root_path).resolve()
            target = (root / path).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                return None
            return target
        except Exception:
            return None

    def _is_allowed_file(self, file_path: Path) -> bool:
        """Check if file extension is allowed"""
        if not self.config.allowed_extensions:
            return True
        return file_path.suffix.lower() in self.config.allowed_extensions

    def read_file(self, path: str, offset: int = 0, limit: int = 0) -> FuncToolResult:
        """
        Read the contents of a file.

        Args:
            path: Path to the file. Absolute paths are permitted for read-only operations.
            offset: Line number to start reading from (1-based). 0 means start from beginning.
            limit: Maximum number of lines to read. 0 means read all lines.

        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure.
                  - 'error' (Optional[str]): Error message on failure.
                  - 'result' (Optional[str]): File contents on success. When offset/limit are set,
                    returns numbered lines in "N: line content" format.
        """
        try:
            path = self._normalize(path)
            target_path = self._get_safe_path(path)

            if not target_path or not target_path.exists():
                return FuncToolResult(success=0, error=f"File not found: {path}")

            if not target_path.is_file():
                return FuncToolResult(success=0, error=f"Path is not a file: {path}")

            if not self._is_allowed_file(target_path):
                return FuncToolResult(success=0, error=f"File type not allowed: {path}")

            if target_path.stat().st_size > self.config.max_file_size:
                return FuncToolResult(success=0, error=f"File too large: {path}")

            try:
                content = target_path.read_text(encoding="utf-8")

                if offset > 0 or limit > 0:
                    lines = content.split("\n")
                    start = max(0, offset - 1) if offset > 0 else 0
                    end = start + limit if limit > 0 else len(lines)
                    selected = lines[start:end]
                    numbered = [f"{start + i + 1}: {line}" for i, line in enumerate(selected)]
                    return FuncToolResult(result="\n".join(numbered))

                return FuncToolResult(result=content)
            except UnicodeDecodeError:
                return FuncToolResult(success=0, error=f"Cannot read binary file: {path}")
            except PermissionError:
                return FuncToolResult(success=0, error=f"Permission denied: {path}")

        except Exception as e:
            logger.error(f"Error reading file {path}: {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def write_file(self, path: str, content: str, file_type: str = "") -> FuncToolResult:
        """
        Create a new file or overwrite an existing file.

        Args:
            path: Relative path within the workspace directory. Do NOT use absolute paths.
            content: The content to write to the file
            file_type: Type of file being written (e.g., "reference_sql", "semantic_model").
                       Used by hooks for special handling.

        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure.
                  - 'error' (Optional[str]): Error message on failure.
                  - 'result' (Optional[str]): Success message on success.
        """
        try:
            try:
                path = self._normalize(path, file_type, strict=True)
            except Exception as e:
                return FuncToolResult(success=0, error=f"Path normalization failed: {e}")
            target_path = self._get_safe_path(path)

            if not target_path:
                return FuncToolResult(success=0, error=f"Invalid path: {path}")

            if not self._is_allowed_file(target_path):
                return FuncToolResult(success=0, error=f"File type not allowed: {path}")

            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(content, encoding="utf-8")
                return FuncToolResult(result=f"File written successfully: {str(path)}")
            except PermissionError:
                return FuncToolResult(success=0, error=f"Permission denied: {path}")

        except Exception as e:
            logger.error(f"Error writing file {path}: {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def edit_file(self, path: str, old_string: str, new_string: str) -> FuncToolResult:
        """
        Make a single edit to a file by replacing old_string with new_string.

        Args:
            path: Relative path within the workspace directory. Do NOT use absolute paths.
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

            try:
                path = self._normalize(path, strict=True)
            except Exception as e:
                return FuncToolResult(success=0, error=f"Path normalization failed: {e}")
            target_path = self._get_safe_path(path)

            if not target_path or not target_path.exists():
                return FuncToolResult(success=0, error=f"File not found: {path}")

            if not target_path.is_file():
                return FuncToolResult(success=0, error=f"Path is not a file: {path}")

            if not self._is_allowed_file(target_path):
                return FuncToolResult(success=0, error=f"File type not allowed: {path}")

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
                return FuncToolResult(result=f"File edited successfully: {str(path)}")
            except UnicodeDecodeError:
                return FuncToolResult(success=0, error=f"Cannot edit binary file: {path}")
            except PermissionError:
                return FuncToolResult(success=0, error=f"Permission denied: {path}")

        except Exception as e:
            logger.error(f"Error editing file {path}: {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    # Minimal fallback excludes when no .gitignore is found
    _FALLBACK_EXCLUDE_DIRS = {".git", "__pycache__", "node_modules"}

    def _load_gitignore_patterns(self, search_root: Path) -> List[str]:
        """Load exclude patterns from .gitignore in the search root or its ancestors.

        Walks up from search_root to self.config.root_path looking for .gitignore.
        Parses non-comment, non-empty lines and converts to glob patterns.
        Always excludes .git directory.
        """
        patterns = [".git", ".git/**", "**/.git/**"]

        # Search for .gitignore from search_root up to root_path
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
            # No .gitignore found, use fallback
            for d in self._FALLBACK_EXCLUDE_DIRS:
                patterns.extend([d, f"{d}/**", f"**/{d}/**"])
            return patterns

        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # Skip negation patterns (not supported in this simplified parser)
                    if line.startswith("!"):
                        continue
                    # Strip leading / (gitignore root-relative marker)
                    entry = line.lstrip("/")
                    # Handle trailing-slash directory entries: also match the dir name itself
                    if entry.endswith("/"):
                        dir_name = entry.rstrip("/")
                        patterns.append(dir_name)
                        patterns.append(f"**/{dir_name}")
                    patterns.append(entry)
                    # Ensure directory entries also match contents
                    if not entry.endswith("/**"):
                        patterns.append(f"{entry}/**")
                    # Ensure patterns match at any depth unless already prefixed
                    if not entry.startswith("**/"):
                        patterns.append(f"**/{entry}")
        except Exception as e:
            logger.warning(f"Failed to fully parse .gitignore at {gitignore_path}: {e}")

        return patterns

    def _walk_files(self, path: str, include_pattern: str = "") -> Iterator[Path]:
        """Walk directory tree yielding files, respecting gitignore, symlink safety, and sandbox.

        Args:
            path: Starting directory path (relative to root).
            include_pattern: Optional glob pattern to filter filenames (e.g., "*.py").

        Yields:
            Path objects for each matching file.
        """
        target_path = self._get_safe_path(path)
        if not target_path or not target_path.exists() or not target_path.is_dir():
            return

        root_path_resolved = Path(self.config.root_path).resolve(strict=False)
        target_path_resolved = target_path.resolve(strict=False)

        try:
            target_path_resolved.relative_to(root_path_resolved)
        except ValueError:
            return

        exclude_patterns = self._load_gitignore_patterns(target_path)
        visited_inodes = set()

        def should_exclude(file_path: Path) -> bool:
            relative_path = str(file_path.relative_to(target_path_resolved))
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
                        if item.is_dir() and item.name == ".git":
                            continue

                        if should_exclude(item):
                            continue

                        item_resolved = item.resolve(strict=False)

                        # Security: ensure resolved path stays within sandbox
                        try:
                            item_resolved.relative_to(root_path_resolved)
                        except ValueError:
                            continue

                        if item.is_dir():
                            yield from walk_recursive(item_resolved)
                        elif item.is_file():
                            if include_pattern:
                                if not wc_glob.globmatch(
                                    item.name, include_pattern, flags=wc_glob.DOTGLOB | wc_glob.GLOBSTAR
                                ):
                                    continue
                            yield item
                    except OSError:
                        continue
            except OSError:
                return

        yield from walk_recursive(target_path_resolved)

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
                  - 'result' (Optional[dict]): Dict with 'files' (list of paths relative to
                    ``root_path`` so callers can feed them back to ``read_file`` /
                    ``write_file`` without leaking absolute paths) and 'truncated' (bool).
        """
        max_results = 200
        try:
            target_path = self._get_safe_path(path)

            if not target_path or not target_path.exists():
                return FuncToolResult(success=0, error=f"Directory not found: {path}")

            if not target_path.is_dir():
                return FuncToolResult(success=0, error=f"Path is not a directory: {path}")

            target_path_resolved = target_path.resolve(strict=False)
            root_path_resolved = Path(self.config.root_path).resolve(strict=False)
            matches = []

            for file_path in self._walk_files(path):
                relative_path = str(file_path.relative_to(target_path_resolved))
                # Report paths relative to root_path so the LLM can feed them
                # back to read_file/write_file without leaking absolute paths.
                try:
                    reported_path = str(file_path.relative_to(root_path_resolved))
                except ValueError:
                    reported_path = str(file_path)
                try:
                    if wc_glob.globmatch(relative_path, pattern, flags=wc_glob.DOTGLOB | wc_glob.GLOBSTAR):
                        matches.append(reported_path)
                        if len(matches) >= max_results:
                            break
                except Exception:
                    if file_path.name == pattern:
                        matches.append(reported_path)
                        if len(matches) >= max_results:
                            break

            truncated = len(matches) >= max_results
            result_data = {
                "files": matches,
                "truncated": truncated,
            }
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
            target_path = self._get_safe_path(path)

            if not target_path or not target_path.exists():
                return FuncToolResult(success=0, error=f"Directory not found: {path}")

            if not target_path.is_dir():
                return FuncToolResult(success=0, error=f"Path is not a directory: {path}")

            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                compiled = re.compile(pattern, flags)
            except re.error as e:
                return FuncToolResult(success=0, error=f"Invalid regex pattern: {str(e)}")

            matches = []

            for file_path in self._walk_files(path, include_pattern=include):
                if not self._is_allowed_file(file_path):
                    continue

                try:
                    if file_path.stat().st_size > self.config.max_file_size:
                        continue
                except OSError:
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, PermissionError, OSError):
                    continue

                for line_num, line in enumerate(content.split("\n"), start=1):
                    if compiled.search(line):
                        matches.append(
                            {
                                "file": str(file_path),
                                "line": line_num,
                                "content": line.rstrip(),
                            }
                        )
                        if len(matches) >= max_matches:
                            break

                if len(matches) >= max_matches:
                    break

            truncated = len(matches) >= max_matches
            return FuncToolResult(
                result={
                    "matches": matches,
                    "truncated": truncated,
                }
            )

        except Exception as e:
            logger.exception(f"Error in grep search for {pattern} in {path}")
            return FuncToolResult(success=0, error=str(e))


def filesystem_function_tools(root_path: str = None) -> List[Tool]:
    """Get filesystem function tools"""
    return FilesystemFuncTool(root_path=root_path).available_tools()
