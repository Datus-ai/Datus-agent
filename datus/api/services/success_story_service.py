"""Service for persisting success stories to the benchmark CSV.

The CSV layout is ``{benchmark_dir}/{subagent_name}/success_story.csv`` —
unchanged from the legacy chatbot so that ``datus.cli.tutorial`` and
``init_success_story_*`` consumers keep working.
"""

import csv
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from datus.api.constants import BUILTIN_SUBAGENTS
from datus.api.models.success_story_models import SuccessStoryData, SuccessStoryInput
from datus.configuration.agent_config import AgentConfig
from datus.utils.csv_utils import sanitize_csv_field
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_CSV_FIELDS = ("session_link", "session_id", "subagent_name", "user_message", "sql", "timestamp")
_EXTRA_BUILTIN_SUBAGENTS = {"feedback"}
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_.\-]")
_DEFAULT_SUBAGENT_NAME = "default"


class SubagentNotFoundError(ValueError):
    """Raised when subagent_id cannot be resolved to a canonical name."""


class SuccessStoryService:
    """Append-only CSV writer for success stories."""

    def __init__(self, agent_config: AgentConfig):
        self.agent_config = agent_config

    def save(self, payload: SuccessStoryInput) -> SuccessStoryData:
        """Persist *payload* as a new CSV row and return metadata.

        Raises:
            SubagentNotFoundError: when ``subagent_id`` does not resolve.
            OSError: when the CSV write fails.
        """
        subagent_name = self._resolve_subagent_name(payload.subagent_id)
        target_dir = self._resolve_target_dir(subagent_name)
        target_dir.mkdir(parents=True, exist_ok=True)
        csv_path = target_dir / "success_story.csv"

        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        row = {
            "session_link": sanitize_csv_field(payload.session_link),
            "session_id": payload.session_id,
            "subagent_name": sanitize_csv_field(subagent_name),
            "user_message": sanitize_csv_field(payload.user_message),
            "sql": sanitize_csv_field(payload.sql),
            "timestamp": timestamp,
        }

        self._append_row(csv_path, row)
        logger.info("Saved success story for session %s under %s", payload.session_id, csv_path)

        return SuccessStoryData(
            csv_path=str(csv_path),
            subagent_name=subagent_name,
            session_id=payload.session_id,
            session_link=payload.session_link,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_subagent_name(self, subagent_id: Optional[str]) -> str:
        """Map a ``subagent_id`` to a safe directory name.

        Acceptable forms: builtin names, ``agentic_nodes`` keys, or the custom
        sub-agent DB UUID stored under ``entry["id"]``. UUID form is resolved
        back to the sanitized key so CSV layout stays human-readable.
        """
        if not subagent_id:
            return _DEFAULT_SUBAGENT_NAME

        if subagent_id in BUILTIN_SUBAGENTS or subagent_id in _EXTRA_BUILTIN_SUBAGENTS:
            return subagent_id

        agentic_nodes = getattr(self.agent_config, "agentic_nodes", None) or {}
        if subagent_id in agentic_nodes:
            return subagent_id

        for key, entry in agentic_nodes.items():
            if isinstance(entry, dict) and entry.get("id") == subagent_id:
                return key

        raise SubagentNotFoundError(f"Subagent '{subagent_id}' not found")

    def _resolve_target_dir(self, subagent_name: str) -> Path:
        """Compute the per-subagent output directory inside benchmark_dir.

        Defense-in-depth: even though ``_resolve_subagent_name`` only returns
        keys validated by ``_SAFE_NAME_RE``, we sanitize and re-check the path
        escapes ``benchmark_dir``.
        """
        base_dir = self.agent_config.path_manager.benchmark_dir.resolve()
        safe_name = subagent_name if _SAFE_NAME_RE.match(subagent_name) else _UNSAFE_CHARS.sub("_", subagent_name)
        target_dir = (base_dir / safe_name).resolve()
        try:
            target_dir.relative_to(base_dir)
        except ValueError as e:
            raise SubagentNotFoundError(f"Unsafe subagent name: {subagent_name!r}") from e
        return target_dir

    @staticmethod
    def _append_row(csv_path: Path, row: dict) -> None:
        """Append *row* to *csv_path*, writing the header if the file is new."""
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            try:
                _lock_file(f)
                writer = csv.DictWriter(f, fieldnames=list(_CSV_FIELDS))
                if os.fstat(f.fileno()).st_size == 0:
                    writer.writeheader()
                writer.writerow(row)
            finally:
                _unlock_file(f)


def _lock_file(f) -> None:
    """Best-effort advisory lock; no-op on platforms without fcntl."""
    try:
        import fcntl

        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    except (ImportError, OSError):
        pass


def _unlock_file(f) -> None:
    try:
        import fcntl

        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except (ImportError, OSError):
        pass
