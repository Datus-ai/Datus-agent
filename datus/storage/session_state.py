"""Per-session agent state persistence.

Stores plan-mode state for a single session under
``~/.datus/data/{project_name}/state/{session_id}.json`` so an
``AgenticNode`` rebuilt by an API resume / CLI re-attach can recover the
plan-mode flag, plan file path, and workflow-prompt-sent flag.

The file layout is nested under a ``plan_mode`` key:

    {
      "plan_mode": {
        "plan_mode_active": bool,
        "plan_file_path": str | null,
        "workflow_prompt_sent": bool
      }
    }

For backward compatibility, files written by older code in the flat layout
(``plan_mode_active`` at top level) are still readable. Compact-subsystem
state was previously persisted alongside plan-mode under a ``compact`` key;
that section was removed because the minor-compact pass is idempotent via
the in-message ``[DATUS_ARCHIVED]`` marker, so persistence added no
correctness value. Legacy files carrying the ``compact`` key are simply
ignored on load.

Decoupled from :class:`SessionManager` (SQLite) on purpose: tests can
exercise round-trip behaviour without spinning up the agents-library DB.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


@dataclass
class PlanModeState:
    plan_mode_active: bool = False
    plan_file_path: Optional[str] = None
    workflow_prompt_sent: bool = False

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PlanModeState":
        """Build a state from a dict, defaulting any malformed field.

        Strict type checks: ``bool(x)`` happily accepts the literal string
        ``"false"`` (truthy because non-empty), which would mis-restore
        plan-mode state from corrupted / legacy payloads.
        """
        if not isinstance(data, dict):
            return cls()
        raw_active = data.get("plan_mode_active", False)
        raw_path = data.get("plan_file_path")
        raw_prompt_sent = data.get("workflow_prompt_sent", False)
        return cls(
            plan_mode_active=raw_active if isinstance(raw_active, bool) else False,
            plan_file_path=raw_path if isinstance(raw_path, str) else None,
            workflow_prompt_sent=raw_prompt_sent if isinstance(raw_prompt_sent, bool) else False,
        )

    @classmethod
    def load(cls, path: Path) -> "PlanModeState":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load PlanModeState from %s: %s", path, exc)
            return cls()
        if not isinstance(data, dict):
            return cls()
        # Nested layout (current).
        if "plan_mode" in data:
            return cls.from_dict(data.get("plan_mode"))
        # Legacy flat layout — read the plan-mode keys at top level.
        return cls.from_dict(data)

    def save(self, path: Path) -> None:
        """Write the plan-mode section under the nested ``plan_mode`` key.

        Always wraps the payload in ``{"plan_mode": ...}`` so any reader
        expecting the nested layout (current code path) finds it without
        falling back to legacy-flat detection.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"plan_mode": asdict(self)}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to persist PlanModeState to %s: %s", path, exc)
