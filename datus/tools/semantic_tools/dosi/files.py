# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Normal native-engine access for independent model file consumers."""

from pathlib import Path
from typing import Any

from datus.tools.semantic_tools.dosi.config import DosiConfig
from datus.tools.semantic_tools.dosi.engine import shared_engine_handle
from datus.tools.semantic_tools.exceptions import SemanticCoreException
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def file_engine(
    path: str | Path, agent_config: Any = None, *, config: DosiConfig | None = None, allow_invalid: bool = False
):
    """Load and reuse a file's engine. Invalid models remain available to raw editors.

    A failed load is never converted into a successful native validation result.
    Structured readers must still check the flat document shape before using it.
    """
    from datus.storage.semantic_model.artifact_file import model_lock_held

    if model_lock_held(path):
        # The outer transaction loaded the file before acquiring the native lock.
        return None
    if config is None and agent_config is not None:
        config = _file_config(path, agent_config)
    config = (config or DosiConfig()).model_copy(update={"semantic_model_path": str(path)})
    try:
        return shared_engine_handle(config).get()
    except SemanticCoreException as exc:
        if not allow_invalid:
            raise
        logger.warning("Model %s requires repair: %s", path, exc)
        return None


def _file_config(path: str | Path, agent_config: Any) -> DosiConfig:
    """Use the file's datasource without mutating the request's active context.

    File-only consumers can supply path metadata without a connection profile.
    Execution-capable AgentConfig instances retain their database coordinates.
    """
    from datus.configuration.agent_config import (
        _apply_runtime_db_context_to_dosi_profile,
        _db_config_to_dosi_profile,
    )

    datasource = getattr(agent_config, "current_datasource", None)
    context_provider = getattr(agent_config, "runtime_db_context", None)
    context = context_provider() if callable(context_provider) else {}
    datasource = context.get("datasource") or datasource
    manager = getattr(agent_config, "path_manager", None)
    root = getattr(manager, "semantic_models_dir", None)
    if isinstance(root, (str, Path)):
        try:
            relative = Path(path).resolve().relative_to(Path(root).resolve())
            if len(relative.parts) > 1 and relative.parts[0] != datasource:
                datasource = relative.parts[0]
                context = {}
        except ValueError:
            pass
    profile_provider = getattr(agent_config, "current_db_config", None)
    profile = None
    if callable(profile_provider):
        profile = _apply_runtime_db_context_to_dosi_profile(
            _db_config_to_dosi_profile(profile_provider(datasource)), context
        )
    return DosiConfig(
        datasource=datasource,
        db_config=profile,
        timeout_seconds=float((profile or {}).get("timeout_seconds", 30)),
    )
