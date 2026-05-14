"""Per-session plan-mode state persistence.

Stores the three plan-mode fields (``plan_mode_active``,
``plan_file_path``, ``workflow_prompt_sent``) under
``~/.datus/data/{project_name}/state/{session_id}.json`` so an
``AgenticNode`` can be reconstructed on resume.

Decoupled from :class:`SessionManager` (SQLite) on purpose: tests can
exercise round-trip behaviour without spinning up the agents-library DB.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@dataclass
class PlanModeState:
    plan_mode_active: bool = False
    plan_file_path: Optional[str] = None
    workflow_prompt_sent: bool = False

    @classmethod
    def load(cls, path: Path) -> "PlanModeState":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                plan_mode_active=bool(data.get("plan_mode_active", False)),
                plan_file_path=data.get("plan_file_path"),
                workflow_prompt_sent=bool(data.get("workflow_prompt_sent", False)),
            )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load PlanModeState from %s: %s", path, exc)
            return cls()

    def save(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(asdict(self), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to persist PlanModeState to %s: %s", path, exc)
