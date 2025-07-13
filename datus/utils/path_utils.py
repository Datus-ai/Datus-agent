import glob
from typing import Dict, List

from datus.utils.constants import DBType


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
        file_name = file_path.split("/")[name_index]
        if name_index == -1 and "." in file_name:
            file_name = file_name[: file_name.rfind(".")]
        result.append({"name": file_name, "uri": f"{dialect}:///{file_path}"})
    return result
