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

# Sections written by older code that no longer persist state. They are
# dropped on every save so stale data never lingers on disk (see the
# module docstring's note about the removed ``compact`` section).
_LEGACY_SECTIONS = ("compact",)


def _load_raw(path: Path) -> Dict[str, Any]:
    """Read the whole state file as a dict, tolerating absence / corruption."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read session state from %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _save_section(path: Path, key: str, payload: Dict[str, Any]) -> None:
    """Merge one section into the state file, preserving sibling sections.

    The file holds independent sections (``plan_mode``, ``context_state``, …)
    written at different times by different subsystems. A naive whole-file
    overwrite would clobber the other sections, so we read-modify-write and
    only replace ``key``. Legacy sections are explicitly dropped.
    """
    try:
        data = _load_raw(path)
        # Drop the legacy flat layout (plan-mode keys at top level) and any
        # retired sections so a re-save never round-trips stale state.
        data = {k: v for k, v in data.items() if k not in _LEGACY_SECTIONS}
        data.pop("plan_mode_active", None)
        data.pop("plan_file_path", None)
        data.pop("workflow_prompt_sent", None)
        data[key] = payload
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to persist session state to %s: %s", path, exc)


def _remove_section(path: Path, key: str) -> None:
    """Drop one section from the state file, preserving sibling sections.

    Used by session cleanup (``clear``/``delete``) so a persisted mirror does
    not survive a reset and leak the previous turn's state. No-op when the
    file or section is absent.
    """
    if not path.exists():
        return
    try:
        data = _load_raw(path)
        if key not in data:
            return
        data.pop(key, None)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to remove section %r from session state %s: %s", key, path, exc)


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

        Merges into the file via :func:`_save_section` so sibling sections
        (e.g. ``context_state``) survive, while the legacy flat layout and
        retired sections are dropped.
        """
        _save_section(path, "plan_mode", asdict(self))


@dataclass
class ContextState:
    """Most recent LLM call's context-window occupancy, for resume.

    Persisted separately from the SQLite usage tables: ``turn_usage`` has no
    per-call occupancy column, and ``running_turn_usage`` is cleared at turn
    end (to avoid double-counting cumulative totals). This section is never
    cleared, so a freshly resumed process can render the context-window bar
    before the next LLM call repopulates it.

    ``last_call_input_tokens`` is the real context-window usage of the last
    call (input + cache_read + cache_creation); ``context_length`` is the
    model's maximum context (the bar's denominator).
    """

    last_call_input_tokens: int = 0
    context_length: int = 0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ContextState":
        """Build from a dict, coercing non-int fields to the safe default."""
        if not isinstance(data, dict):
            return cls()

        def _as_int(value: Any) -> int:
            # ``bool`` is an ``int`` subclass but is never a valid token count;
            # reject it (and any non-int) so corrupted payloads fall back to 0.
            if isinstance(value, bool) or not isinstance(value, int):
                return 0
            return max(0, value)

        return cls(
            last_call_input_tokens=_as_int(data.get("last_call_input_tokens", 0)),
            context_length=_as_int(data.get("context_length", 0)),
        )

    @classmethod
    def load(cls, path: Path) -> "ContextState":
        data = _load_raw(path)
        if not data:
            return cls()
        return cls.from_dict(data.get("context_state"))

    def save(self, path: Path) -> None:
        """Merge the context-state section into the state file."""
        _save_section(path, "context_state", asdict(self))

    @classmethod
    def clear(cls, path: Path) -> None:
        """Remove the persisted context-state mirror (session reset/delete)."""
        _remove_section(path, "context_state")
