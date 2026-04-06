"""Path safety utilities — prevent path traversal attacks."""

from pathlib import Path

from datus.utils.exceptions import DatusException, ErrorCode


def safe_resolve(base: Path, user_path: str) -> Path:
    """Resolve user_path relative to base; raise DatusException if it escapes base.

    Args:
        base: The base directory path
        user_path: The user-provided path (may contain .., /, etc.)

    Returns:
        The safely resolved Path object

    Raises:
        DatusException: If the resolved path escapes the base directory
    """
    resolved = (base / user_path).resolve()
    base_resolved = base.resolve()
    if not resolved.is_relative_to(base_resolved):
        raise DatusException(
            ErrorCode.COMMON_VALIDATION_FAILED,
            message=f"Path '{user_path}' escapes the project root",
        )
    return resolved
