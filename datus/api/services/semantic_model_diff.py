"""Immutable, display-only semantic YAML snapshots for one complete chat turn."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@dataclass
class SemanticModelSnapshot:
    """Retain a turn's original UTF-8 YAML contents for comparison with its final files."""

    project_root: Path
    files: dict[str, str]

    @classmethod
    def capture(cls, project_root: str | Path) -> Optional["SemanticModelSnapshot"]:
        """Read YAML bytes rather than mtimes, including files not yet indexed in the KB."""
        try:
            project = Path(project_root).resolve()
            root = project / "subject" / "semantic_models"
            files = {}
            if root.is_symlink():
                return None
            if root.exists():
                for path in root.rglob("*"):
                    if path.suffix.lower() not in {".yml", ".yaml"} or not path.is_file():
                        continue
                    if path.is_symlink() or not path.resolve().is_relative_to(root):
                        continue
                    files[path.relative_to(project).as_posix()] = path.read_bytes().decode("utf-8")
            return cls(project_root=project, files=files)
        except (OSError, UnicodeError, ValueError):
            # A partial scan must never turn an unreadable file into a fake deletion.
            logger.warning("Cannot snapshot semantic model files for this chat turn", exc_info=True)
            return None

    def changes(self) -> list[dict[str, Any]]:
        """Return sorted final additions, modifications and deletions, omitting net-zero writes."""
        current = self.capture(self.project_root)
        if current is None:
            return []
        changes = []
        for path in sorted(self.files.keys() | current.files.keys()):
            old = self.files.get(path)
            new = current.files.get(path)
            if old == new:
                continue
            changes.append(
                {
                    "filePath": path,
                    "changeType": "added" if old is None else "deleted" if new is None else "modified",
                    "oldText": old,
                    "newText": new,
                }
            )
        return changes
