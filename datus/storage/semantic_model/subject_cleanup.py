# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Retryable cleanup of directories emptied by an explicit model projection."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from datus.storage.semantic_model.artifact_file import atomic_write_text
from datus.storage.semantic_model.sync_state import _exclusive, state_key

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig


def _path(config: "AgentConfig") -> Path:
    return Path(config.path_manager.project_data_dir) / "semantic_subject_cleanup.json"


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def remember_cleanup(config: "AgentConfig", datasource: str, yaml_path: str, nodes: Iterable[int]) -> set[int]:
    """Persist occupied nodes before replacing rows, including earlier failed cleanup."""
    path = _path(config)
    key = state_key(yaml_path)
    with _exclusive(path):
        state = _read(path)
        files = state.setdefault(datasource, {})
        pending = set(files.get(key, [])) | set(nodes)
        if pending:
            files[key] = sorted(pending)
            atomic_write_text(path, json.dumps(state, sort_keys=True))
        return pending


def finish_cleanup(config: "AgentConfig", datasource: str, yaml_path: str, nodes: set[int]) -> None:
    """Remove only previously occupied, now-empty nodes; retain work on failure."""
    from datus.storage.semantic_model.reconcile import _DatasourceStores

    if not nodes:
        return
    stores = _DatasourceStores(config, datasource)
    stores.remove_emptied_nodes(nodes)
    path = _path(config)
    with _exclusive(path):
        state = _read(path)
        files = state.get(datasource, {})
        key = state_key(yaml_path)
        remaining = set(files.get(key, [])) - nodes
        if remaining:
            files[key] = sorted(remaining)
        else:
            files.pop(key, None)
        atomic_write_text(path, json.dumps(state, sort_keys=True))
