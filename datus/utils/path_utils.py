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


def get_files_from_glob_pattern(path_pattern: str, dialect: str = DBType.SQLITE) -> List[Dict[str, str]]:
    """Get files from glob pattern

    Args:
        path_pattern (str): glob pattern
        dialect (str, optional): dialect of the database. Defaults to DBType.SQLITE.

    Returns:
        List[Dict[str, str]]: list of files with name and uri
    """
    if not has_glob_pattern(path_pattern):
        return []
    path_pattern = os.path.expanduser(path_pattern)
    paths = path_pattern.split("/")
    if len(paths) == 1:
        name_index = -1
    else:
        if "*" in paths[-2] or "?" in paths[-2]:
            name_index = -2
        else:
            name_index = -1
    files = glob.glob(path_pattern, recursive=True)
    result = []
    for file_path in files:
        path = Path(file_path)
        file_name = path.parts[name_index]
        if name_index == -1 and path.suffix:
            file_name = file_name.rsplit(".", 1)[0]
        uri = f"{dialect}:///{path.as_posix()}"
        result.append({"name": file_name, "uri": uri})
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
