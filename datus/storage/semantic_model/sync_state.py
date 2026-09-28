# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""The content digest each semantic YAML file had when it was last projected.

A file edited outside the validating save path — an agent's ``edit_file``, a
shell ``sed``, a ``git pull`` — is only noticed by comparing what is on disk
with what the Knowledge Base was built from. Only a full-file projection is
recorded, and any path that changes the KB otherwise forgets the entry, so a
stale entry can only cause one redundant re-sync, never a missed one.

With no entries at all (first run, or the state file lost) a reconcile trusts
only the files whose projection still matches what they declare — by metric
names and dataset presence — and re-projects the rest. An expression edited in
place while no state existed is the one change that goes unnoticed.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Optional

from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig

logger = get_logger(__name__)

_STATE_FILE = "semantic_sync_state.json"
_lock = threading.Lock()


def file_digest(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def load_digests(agent_config: "AgentConfig", datasource: str) -> Optional[Dict[str, str]]:
    """``{yaml_path: digest}`` for ``datasource``, or ``None`` when never recorded."""
    state = _read(agent_config)
    if state is None or datasource not in state:
        return None
    return dict(state[datasource])


def record_digests(agent_config: "AgentConfig", datasource: str, digests: Dict[str, str]) -> None:
    normalized = {state_key(path): digest for path, digest in digests.items()}
    _update(agent_config, datasource, lambda entries: entries.update(normalized))


def forget_digests(agent_config: "AgentConfig", datasource: str, yaml_paths: Iterable[str]) -> None:
    paths = {state_key(path) for path in yaml_paths}
    if paths:
        _update(agent_config, datasource, lambda entries: [entries.pop(path, None) for path in paths])


def state_key(path) -> str:
    """Callers pass the same file resolved or not; key it one way."""
    return str(Path(str(path)).expanduser().resolve(strict=False))


def _state_path(agent_config: "AgentConfig") -> Optional[Path]:
    try:
        return Path(agent_config.path_manager.project_data_dir) / _STATE_FILE
    except Exception:  # noqa: BLE001 - no project dir means nothing to track
        return None


def _read(agent_config: "AgentConfig") -> Optional[Dict[str, Dict[str, str]]]:
    path = _state_path(agent_config)
    if path is None or not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning(f"Ignoring unreadable semantic sync state at {path}")
        return None
    return state if isinstance(state, dict) else None


def _update(agent_config: "AgentConfig", datasource: str, change) -> None:
    from datus.storage.semantic_model.artifact_file import atomic_write_text

    path = _state_path(agent_config)
    if path is None:
        return
    # Cross-process writers can still interleave; that only costs a re-sync.
    with _lock:
        state = _read(agent_config) or {}
        entries = dict(state.get(datasource) or {})
        change(entries)
        state[datasource] = entries
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(path, json.dumps(state, indent=2, sort_keys=True))
        except OSError:
            logger.warning(f"Failed to write semantic sync state at {path}", exc_info=True)
