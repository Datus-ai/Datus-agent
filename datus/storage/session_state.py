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

    def to_json(self) -> str:
        """Serialize for the session database's metadata row.

        The three fields are always written together, so they travel as one
        JSON value rather than three rows that could land out of step.
        """
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: Optional[str]) -> Optional["PlanModeState"]:
        """Parse a stored payload; ``None`` means "nothing recorded".

        Distinct from :meth:`from_dict`, which defaults a malformed field but
        still returns a state. Here the caller needs to tell "no plan-mode row"
        from "plan mode is off", because only the former should fall through to
        the legacy JSON file.
        """
        if not payload:
            return None
        try:
            data = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable plan-mode payload: %s", exc)
            return None
        return cls.from_dict(data) if isinstance(data, dict) else None


@dataclass
class ContextState:
    """Context-window occupancy measured on the last model call.

    Persisted separately from the usage tables: ``turn_usage`` records billed
    consumption, which occupancy is not (cache reads cost little but fill the
    window just the same), and ``running_turn_usage`` is cleared at turn end to
    avoid double-counting cumulative totals.

    Zero means "no measurement": a history rewrite invalidates the reading
    until another model response arrives, and a session that has not called the
    model yet has nothing to report. Both cases render as an empty bar and hold
    compaction off, which is the intended behaviour for an unknown occupancy.

    The denominator is not stored — ``AgenticNode.context_length`` resolves the
    active model's window on every access.
    """

    last_call_input_tokens: int = 0

    def __post_init__(self) -> None:
        """Keep a negative occupancy unrepresentable.

        ``save_context_state`` takes a caller-supplied state and writes the
        number through verbatim; a negative value would surface as a negative
        occupancy in the status bar and a negative ratio in the compaction
        gate. Clamping here means no call site has to remember to.
        """
        if self.last_call_input_tokens < 0:
            self.last_call_input_tokens = 0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ContextState":
        """Build from a dict, coercing anything unusable to zero.

        A pre-migration payload carrying ``valid: false`` reads as zero. Those
        records were written when an explicit flag was the only way to mark a
        reading stale, and their token count may be a text estimate rather than
        a measurement. Nothing writes that flag any more.
        """
        if not isinstance(data, dict):
            return cls()
        if data.get("valid") is False:
            return cls()
        value = data.get("last_call_input_tokens", 0)
        # ``bool`` is an ``int`` subclass but is never a valid token count;
        # reject it (and any non-int) so corrupted payloads fall back to 0.
        if isinstance(value, bool) or not isinstance(value, int):
            return cls()
        return cls(max(0, value))

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


def read_context_state(node: Any) -> ContextState:
    """Read one node's measured occupancy without consulting billing history.

    ``running_turn_usage.session_total_tokens`` is already zeroed by
    :class:`~datus.schemas.token_usage.TokenUsage` whenever that snapshot is
    marked invalid, so a stale reading never reaches here as a live one.
    """
    state = getattr(node, "_context_state", None)
    if isinstance(state, ContextState):
        return state
    running = getattr(node, "running_turn_usage", None)
    if running is not None:
        return ContextState.from_dict({"last_call_input_tokens": getattr(running, "session_total_tokens", 0)})
    return ContextState.from_dict({"last_call_input_tokens": getattr(node, "_restored_context_used", 0)})
