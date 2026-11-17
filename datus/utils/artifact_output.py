"""Host-supplied artifact output, independent of the project's context and KB."""

from pathlib import Path
from typing import Any


def artifact_output_root(config: Any) -> Path:
    return Path(getattr(config, "artifact_output_root", None) or config.project_root).expanduser().resolve()


def artifact_mutation_guard(config: Any, path: Path) -> None:
    """Protect query/model files while allowing artifact presentation edits."""
    root = artifact_output_root(config)
    try:
        parts = path.resolve().relative_to(root).parts
    except ValueError:
        raise ValueError("Artifact writes must stay inside the supplied output root") from None
    if len(parts) < 4 or parts[0] not in {"reports", "dashboards"} or parts[2] not in {"render", "analysis"}:
        raise ValueError("Use dedicated artifact tools for manifests, queries and models")
    if parts[2] == "render" and path.suffix not in {".jsx", ".js", ".css"}:
        raise ValueError("Render files must be JSX, JavaScript or CSS")


def session_user_scope(config: Any, user_scope: str | None) -> str | None:
    """A host may supply a session directory already isolated for this caller."""
    return None if getattr(config, "session_dir_scoped", False) else user_scope
