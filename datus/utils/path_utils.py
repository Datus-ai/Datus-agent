# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import glob
import os.path
from pathlib import Path
from typing import Dict, List

from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def has_glob_pattern(path: str) -> bool:
    """Check if a path contains glob patterns.

    Args:
        path: Path string to check

    Returns:
        bool: True if path contains any glob pattern characters (* ? [ ] **)
    """
    glob_chars = ["*", "?", "[", "]"]
    return any(char in path for char in glob_chars)


def get_files_from_glob_pattern(path_pattern: str, dialect: str | DBType = DBType.SQLITE) -> List[Dict[str, str]]:
    """Get files from glob pattern

    Args:
        path_pattern (str): glob pattern
        dialect (str, optional): dialect of the database. Defaults to DBType.SQLITE.

    Returns:
        List[Dict[str, str]]: list of dicts with keys logic_name, database_name, and uri
    """
    if not has_glob_pattern(path_pattern):
        return []
    if isinstance(dialect, DBType):
        dialect = dialect.value
    path_pattern = os.path.expanduser(path_pattern)
    normalized_pattern = path_pattern.replace("\\", "/")

    # Detect whether the directory part contains any wildcard
    if "/" in normalized_pattern:
        dir_pattern, _ = normalized_pattern.rsplit("/", 1)
    else:
        dir_pattern, _ = "", normalized_pattern
    dir_has_wildcard = any(ch in dir_pattern for ch in ("*", "?", "["))

    files = glob.glob(path_pattern, recursive=True)
    result: List[Dict[str, str]] = []

    for file_path in files:
        path = Path(file_path)
        if not path.is_file():
            continue

        database_name = path.stem  # 文件名（去扩展名）
        # logic_name 使用父目录名称（当目录中存在通配符时）
        if dir_has_wildcard:
            logic_name = path.parent.name
        else:
            logic_name = database_name

        uri = f"{dialect}:///{path.as_posix()}"
        result.append(
            {
                "logic_name": logic_name,
                "name": database_name,
                "uri": uri,
            }
        )
    return result


def get_file_name(path: str) -> str:
    path = Path(path)
    suffix = path.suffix
    if not suffix:
        return path.name
    return path.name[: -len(path.suffix)]


def get_file_fuzzy_matches(text: str, path: str = ".", max_matches: int = 5) -> List[str]:
    """Get fuzzy matches for files.

    Args:
        text: Text to match
        path: Root path to search from
        max_matches: Maximum number of matches to return

    Returns:
        List of relative file paths that match
    """
    results = []

    root_path = Path(path)
    if not root_path.exists():
        return results

    # Use recursive glob pattern to search all subdirectories
    patterns = [
        f"*{text}*",  # Files in current directory
        f"**/*{text}*",  # Files in any subdirectory containing text
        f"*{text}*/**/*",  # Files in subdirectories of folders containing text
    ]

    seen_files = set()  # To avoid duplicates

    for pattern in patterns:
        try:
            for file_path in root_path.glob(pattern):
                if file_path.is_file():
                    relative_path = str(file_path.relative_to(root_path))

                    # Check if text matches (case-insensitive)
                    if text.lower() in relative_path.lower() and relative_path not in seen_files:
                        results.append(relative_path)
                        seen_files.add(relative_path)

                        if len(results) >= max_matches:
                            return results
        except Exception as e:
            logger.debug(f"Error with pattern {pattern}: {e}")
            continue

    return results
