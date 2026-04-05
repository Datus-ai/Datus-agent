"""Path safety utilities — prevent path traversal attacks."""

from pathlib import Path


def safe_resolve(base: Path, user_path: str) -> Path:
    """Resolve user_path relative to base; raise ValueError if it escapes base.

    Args:
        base: The base directory path
        user_path: The user-provided path (may contain .., /, etc.)

    Returns:
        The safely resolved Path object

    Raises:
        ValueError: If the resolved path escapes the base directory
    """
    resolved = (base / user_path).resolve()
    if not str(resolved).startswith(str(base.resolve())):
        raise ValueError(f"Path '{user_path}' escapes the project root")
    return resolved
