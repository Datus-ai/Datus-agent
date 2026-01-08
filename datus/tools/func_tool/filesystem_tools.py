# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
from pathlib import Path
from typing import List, Optional

from agents import Tool
from pydantic import BaseModel, Field
from wcmatch import glob

from datus.tools import BaseTool
from datus.tools.func_tool import FuncToolResult
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class EditOperation(BaseModel):
    """Single edit operation for file editing"""

    oldText: str = Field(description="The text to be replaced")
    newText: str = Field(description="The text to replace with")


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


class FilesystemFuncTool(BaseTool):
    """Function tool wrapper for filesystem operations"""

    def __init__(self, root_path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.root_path = root_path or os.getenv("FILESYSTEM_MCP_PATH", os.path.expanduser("~"))
        self.config = FilesystemConfig(root_path=root_path)

    def available_tools(self) -> List[Tool]:
        """Get all available filesystem tools"""
        from datus.tools.func_tool import trans_to_function_tool

        bound_tools = []
        methods_to_convert = [
            self.read_file,
            self.read_multiple_files,
            self.write_file,
            self.edit_file,
            self.create_directory,
            self.list_directory,
            self.directory_tree,
            self.move_file,
            self.search_files,
        ]

        for bound_method in methods_to_convert:
            bound_tools.append(trans_to_function_tool(bound_method))
        return bound_tools

    def _get_safe_path(self, path: str) -> Optional[Path]:
        """Get a safe path within the root directory"""
        try:
            root = Path(self.config.root_path).resolve()
            target = (root / path).resolve()
            if not str(target).startswith(str(root)):
                return None

            return target
        except Exception:
            return None

    def _is_allowed_file(self, file_path: Path) -> bool:
        """Check if file extension is allowed"""
        if not self.config.allowed_extensions:
            return True
        return file_path.suffix.lower() in self.config.allowed_extensions

    def read_file(self, path: str) -> FuncToolResult:
        """
        Read the contents of a file.

        Args:
            path: The path of the file to read

        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure.
                  - 'error' (Optional[str]): Error message on failure.
                  - 'result' (Optional[str]): File contents on success.
        """
        try:
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
                return FuncToolResult(result=content)
            except UnicodeDecodeError:
                return FuncToolResult(success=0, error=f"Cannot read binary file: {path}")
            except PermissionError:
                return FuncToolResult(success=0, error=f"Permission denied: {path}")

        except Exception as e:
            logger.error(f"Error reading file {path}: {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def read_multiple_files(self, paths: List[str]) -> FuncToolResult:
        """
        Read the contents of multiple files.

        Args:
            paths: List of file paths to read

        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure.
                  - 'error' (Optional[str]): Error message on failure.
                  - 'result' (Optional[dict]): Dictionary mapping paths to their contents on success.
        """
        try:
            results = {}

            for path in paths:
                target_path = self._get_safe_path(path)
                if not target_path or not target_path.exists():
                    results[path] = f"File not found: {path}"
                    continue

                if not target_path.is_file():
                    results[path] = f"Path is not a file: {path}"
                    continue

                if not self._is_allowed_file(target_path):
                    results[path] = f"File type not allowed: {path}"
                    continue

                try:
                    content = target_path.read_text(encoding="utf-8")
                    results[path] = content
                except UnicodeDecodeError:
                    results[path] = f"Cannot read binary file: {path}"
                except PermissionError:
                    results[path] = f"Permission denied: {path}"

            return FuncToolResult(result=results)

        except Exception as e:
            logger.error(f"Error reading multiple files: {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def write_file(self, path: str, content: str, file_type: str = "") -> FuncToolResult:
        """
        Create a new file or overwrite an existing file.

        Args:
            path: The path of the file to write
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
            target_path = self._get_safe_path(path)

            if not target_path:
                return FuncToolResult(success=0, error=f"Invalid path: {path}")

            if not self._is_allowed_file(target_path):
                return FuncToolResult(success=0, error=f"File type not allowed: {path}")

            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(content, encoding="utf-8")
                return FuncToolResult(result=f"File written successfully: {str(target_path)}")
            except PermissionError:
                return FuncToolResult(success=0, error=f"Permission denied: {path}")

        except Exception as e:
            logger.error(f"Error writing file {path}: {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def edit_file(self, path: str, edits: List[EditOperation]) -> FuncToolResult:
        """
        Make selective edits to a file.

        Args:
            path: The path of the file to edit
            edits: List of edit operations. Each edit must be an object (not a string) with two properties
                   Example: [{"oldText": "text to find", "newText": "text to replace"}]
        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure.
                  - 'error' (Optional[str]): Error message on failure.
                  - 'result' (Optional[str]): Success message on success.
        """
        try:
            # Handle edits passed as JSON string from LLM
            if isinstance(edits, str):
                try:
                    import json

                    edits = json.loads(edits)
                except json.JSONDecodeError as e:
                    return FuncToolResult(success=0, error=f"Invalid JSON in edits parameter: {str(e)}")

            target_path = self._get_safe_path(path)

            if not target_path or not target_path.exists():
                return FuncToolResult(success=0, error=f"File not found: {path}")

            if not target_path.is_file():
                return FuncToolResult(success=0, error=f"Path is not a file: {path}")

            if not self._is_allowed_file(target_path):
                return FuncToolResult(success=0, error=f"File type not allowed: {path}")

            try:
                content = target_path.read_text(encoding="utf-8")
                edits_applied = 0

                for edit in edits:
                    # Handle both EditOperation objects and dictionaries
                    if isinstance(edit, dict):
                        old_text = edit.get("oldText", "")
                        new_text = edit.get("newText", "")
                    else:
                        old_text = edit.oldText
                        new_text = edit.newText

                    # Check if old_text exists in content before replacing
                    if old_text and old_text in content:
                        content = content.replace(old_text, new_text)
                        edits_applied += 1
                    elif old_text:
                        # old_text not found in file
                        preview = old_text[:100] + "..." if len(old_text) > 100 else old_text
                        return FuncToolResult(
                            success=0,
                            error=f"Text not found in file, cannot apply edit. Looking for: {preview}",
                        )

                if edits_applied == 0:
                    return FuncToolResult(success=0, error="No edits were applied")

                target_path.write_text(content, encoding="utf-8")
                return FuncToolResult(
                    result=f"File edited successfully: {str(target_path)} ({edits_applied} edit(s) applied)"
                )
            except UnicodeDecodeError:
                return FuncToolResult(success=0, error=f"Cannot edit binary file: {path}")
            except PermissionError:
                return FuncToolResult(success=0, error=f"Permission denied: {path}")

        except Exception as e:
            logger.error(f"Error editing file {path}: {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def create_directory(self, path: str) -> FuncToolResult:
        """
        Create a new directory or ensure it exists.

        Args:
            path: The path of the directory to create

        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure.
                  - 'error' (Optional[str]): Error message on failure.
                  - 'result' (Optional[str]): Success message on success.
        """
        try:
            target_path = self._get_safe_path(path)

            if not target_path:
                return FuncToolResult(success=0, error=f"Invalid path: {path}")

            try:
                target_path.mkdir(parents=True, exist_ok=True)
                return FuncToolResult(result=f"Directory created: {path}")
            except PermissionError:
                return FuncToolResult(success=0, error=f"Permission denied: {path}")

        except Exception as e:
            logger.error(f"Error creating directory {path}: {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def list_directory(self, path: str) -> FuncToolResult:
        """
        List the contents of a directory.

        Args:
            path: The path of the directory to list. Use "." to list the workspace root directory.
                  Note: Only relative paths are allowed for security.

        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure.
                  - 'error' (Optional[str]): Error message on failure.
                  - 'result' (Optional[List[Dict]]): List of items with 'name' and 'type' on success.
        """
        try:
            target_path = self._get_safe_path(path)
            logger.debug(f"target_path: {target_path}")

            if not target_path or not target_path.exists():
                return FuncToolResult(success=0, error=f"Directory not found: {path}")

            if not target_path.is_dir():
                return FuncToolResult(success=0, error=f"Path is not a directory: {path}")

            try:
                items = []
                for item in sorted(target_path.iterdir()):
                    item_info = {"name": item.name, "type": "directory" if item.is_dir() else "file"}
                    items.append(item_info)

                return FuncToolResult(result=items)
            except PermissionError:
                return FuncToolResult(success=0, error=f"Permission denied: {path}")

        except Exception as e:
            logger.error(f"Error listing directory {path}: {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def directory_tree(self, path: str, max_depth: int = 3, max_items: int = 1000) -> FuncToolResult:
        """
        Get a tree view of a directory with depth and item limits.

        Args:
            path: The path of the directory to analyze. Use "." for workspace root.
            max_depth: Maximum depth to traverse (default: 3). Use -1 for unlimited depth.
            max_items: Maximum number of items to display (default: 1000). Prevents context overflow.

        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure.
                  - 'error' (Optional[str]): Error message on failure.
                  - 'result' (Optional[str]): Tree view string on success.
        """
        try:
            target_path = self._get_safe_path(path)

            if not target_path or not target_path.exists():
                return FuncToolResult(success=0, error=f"Directory not found: {path}")

            if not target_path.is_dir():
                return FuncToolResult(success=0, error=f"Path is not a directory: {path}")

            try:
                item_count = {"count": 0}  # Mutable counter to track across recursion
                truncated = {"flag": False}  # Track if output was truncated

                def build_tree(dir_path: Path, prefix: str = "", depth: int = 0) -> List[str]:
                    # Check depth limit
                    if max_depth >= 0 and depth >= max_depth:
                        return [f"{prefix}    ... (max depth {max_depth} reached)"]

                    # Check item limit
                    if item_count["count"] >= max_items:
                        truncated["flag"] = True
                        return [f"{prefix}    ... (max items {max_items} reached)"]

                    lines = []
                    try:
                        items = sorted(dir_path.iterdir())
                    except PermissionError:
                        return [f"{prefix}    ... (permission denied)"]

                    for i, item in enumerate(items):
                        # Check item limit before processing
                        if item_count["count"] >= max_items:
                            truncated["flag"] = True
                            lines.append(f"{prefix}    ... (truncated at {max_items} items)")
                            break

                        is_last = i == len(items) - 1
                        current_prefix = "└── " if is_last else "├── "

                        if item.is_dir():
                            lines.append(f"{prefix}{current_prefix}{item.name}/")
                            item_count["count"] += 1
                            next_prefix = prefix + ("    " if is_last else "│   ")
                            lines.extend(build_tree(item, next_prefix, depth + 1))
                        else:
                            try:
                                size = item.stat().st_size
                                lines.append(f"{prefix}{current_prefix}{item.name} ({size} bytes)")
                            except Exception:
                                lines.append(f"{prefix}{current_prefix}{item.name}")
                            item_count["count"] += 1

                    return lines

                tree_lines = [f"{target_path.name}/"]
                tree_lines.extend(build_tree(target_path))
                tree_output = "\n".join(tree_lines)

                # Add warning if truncated
                if truncated["flag"]:
                    tree_output += (
                        f"\n\nOutput truncated: Reached limit of {max_items} items. "
                        "Use smaller max_items or specify a subdirectory."
                    )

                return FuncToolResult(result=tree_output)
            except PermissionError:
                return FuncToolResult(success=0, error=f"Permission denied: {path}")

        except Exception as e:
            logger.error(f"Error building directory tree {path}: {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def move_file(self, source: str, destination: str) -> FuncToolResult:
        """
        Move or rename a file or directory.

        Args:
            source: The current path of the file or directory
            destination: The new path for the file or directory

        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure.
                  - 'error' (Optional[str]): Error message on failure.
                  - 'result' (Optional[str]): Success message on success.
        """
        try:
            source_path = self._get_safe_path(source)
            dest_path = self._get_safe_path(destination)

            if not source_path or not source_path.exists():
                return FuncToolResult(success=0, error=f"Source not found: {source}")

            if not dest_path:
                return FuncToolResult(success=0, error=f"Invalid destination: {destination}")

            try:
                source_path.rename(dest_path)
                return FuncToolResult(result=f"Moved {source} to {destination}")
            except PermissionError:
                return FuncToolResult(success=0, error="Permission denied")
            except OSError as e:
                return FuncToolResult(success=0, error=f"Move failed: {str(e)}")

        except Exception as e:
            logger.error(f"Error moving file from {source} to {destination}: {str(e)}")
            return FuncToolResult(success=0, error=str(e))

    def search_files(self, path: str, pattern: str, exclude_patterns: Optional[List[str]] = None) -> FuncToolResult:
        """
        Recursively search for files and directories matching a pattern.

        Args:
            path: Starting directory to begin search
            pattern: Glob-style pattern to match (e.g., "*.py", "**/*.yaml")
            exclude_patterns: List of glob-style patterns to exclude from results

        Returns:
            dict: A dictionary with the execution result, containing these keys:
                  - 'success' (int): 1 for success, 0 for failure.
                  - 'error' (Optional[str]): Error message on failure.
                  - 'result' (Optional[List[str]]): List of matching absolute file paths on success.
        """
        try:
            target_path = self._get_safe_path(path)

            if not target_path or not target_path.exists():
                return FuncToolResult(success=0, error=f"Directory not found: {path}")

            if not target_path.is_dir():
                return FuncToolResult(success=0, error=f"Path is not a directory: {path}")

            exclude_patterns = exclude_patterns or []

            try:
                matches = []
                root_path_resolved = Path(self.config.root_path).resolve(strict=False)
                target_path_resolved = target_path.resolve(strict=False)

                # Ensure target path is within root path sandbox
                try:
                    target_path_resolved.relative_to(root_path_resolved)
                except ValueError:
                    return FuncToolResult(success=0, error=f"Path {path} is outside the allowed directory")

                # Track visited inodes to prevent symlink loops
                visited_inodes = set()

                def should_exclude(file_path: Path) -> bool:
                    relative_path = str(file_path.relative_to(target_path_resolved))
                    for exclude_pattern in exclude_patterns:
                        try:
                            # globmatch: minimatch-compatible with DOTGLOB (hidden files) and GLOBSTAR (**)
                            if glob.globmatch(relative_path, exclude_pattern, flags=glob.DOTGLOB | glob.GLOBSTAR):
                                return True
                        except Exception:
                            continue
                    return False

                def search_recursive(current_path: Path):
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
                                if should_exclude(item):
                                    continue

                                item_resolved = item.resolve(strict=False)

                                # Security: ensure resolved path stays within sandbox
                                try:
                                    item_resolved.relative_to(root_path_resolved)
                                except ValueError:
                                    continue

                                relative_path = str(item.relative_to(target_path_resolved))
                                try:
                                    if glob.globmatch(relative_path, pattern, flags=glob.DOTGLOB | glob.GLOBSTAR):
                                        matches.append(str(item_resolved))
                                except Exception:
                                    if item.name == pattern:
                                        matches.append(str(item_resolved))

                                if item.is_dir():
                                    search_recursive(item_resolved)

                            except OSError:
                                continue

                    except OSError:
                        return

                search_recursive(target_path_resolved)

                return FuncToolResult(result=matches)

            except PermissionError:
                return FuncToolResult(success=0, error=f"Permission denied: {path}")

        except Exception as e:
            logger.exception(f"Error searching files in {path}")
            return FuncToolResult(success=0, error=str(e))


def filesystem_function_tools(root_path: str = None) -> List[Tool]:
    """Get filesystem function tools"""
    return FilesystemFuncTool(root_path=root_path).available_tools()
