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
    # Use pathlib
    pattern = Path(path_pattern)
    parts = pattern.parts
    if len(parts) == 1:
        name_index = -1
    else:
        if "*" in parts[-2] or "?" in parts[-2]:
            name_index = -2
        else:
            name_index = -1

    # Transfer to Path type
    files = glob.glob(str(pattern), recursive=True)
    result = []
    for file_path in files:
        path = Path(file_path)
        file_name = path.parts[name_index]
        if name_index == -1 and "." in file_name:
            file_name = file_name.rsplit(".", 1)[0]
        uri = f"{dialect}:///{path.as_posix()}"  # 使用 as_posix 保证 URI 格式
        result.append({"name": file_name, "uri": uri})
    return result
