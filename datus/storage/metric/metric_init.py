# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Metric bootstrap compatibility aliases and non-LLM YAML import."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any, Callable, Optional

from datus.agent.node.semantic_authoring import AUTHORING_FORMAT_OSI
from datus.configuration.agent_config import AgentConfig
from datus.schemas.batch_events import BatchEventEmitter
from datus.utils.loggings import get_logger
from datus.utils.terminal_utils import suppress_keyboard_input

if TYPE_CHECKING:
    from datus.schemas.action_history import ActionHistory

logger = get_logger(__name__)

BIZ_NAME = "metric_init"
DEFAULT_METRICS_BATCH_SIZE = 5


async def init_success_story_metrics_async(
    agent_config: AgentConfig,
    success_story: str,
    subject_tree: Optional[list] = None,
    emit: Optional[BatchEventEmitter] = None,
    extra_instructions: Optional[str] = None,
    *,
    build_mode: str = "overwrite",
    action_callback: Optional[Callable[["ActionHistory"], None]] = None,
    batch_size: int = DEFAULT_METRICS_BATCH_SIZE,
) -> tuple[bool, str, Optional[dict[str, Any]]]:
    """Route the historical metrics bootstrap API to full semantic_modeling."""
    from datus.storage.semantic_model.semantic_modeling_init import init_success_story_semantic_modeling_async

    if extra_instructions:
        logger.warning(
            "extra_instructions is deprecated for metrics bootstrap; use semantic_modeling request context instead"
        )
    return await init_success_story_semantic_modeling_async(
        agent_config,
        success_story,
        subject_tree,
        emit,
        build_mode=build_mode,
        action_callback=action_callback,
        batch_size=batch_size,
        authoring_scope="full",
    )


def init_success_story_metrics(
    agent_config: AgentConfig,
    success_story: str,
    subject_tree: Optional[list] = None,
    emit: Optional[BatchEventEmitter] = None,
    extra_instructions: Optional[str] = None,
    *,
    build_mode: str = "overwrite",
    batch_size: int = DEFAULT_METRICS_BATCH_SIZE,
) -> tuple[bool, str, Optional[dict[str, Any]]]:
    """Synchronous compatibility wrapper for full semantic_modeling."""
    with suppress_keyboard_input():
        return asyncio.run(
            init_success_story_metrics_async(
                agent_config,
                success_story,
                subject_tree,
                emit,
                extra_instructions,
                build_mode=build_mode,
                batch_size=batch_size,
            )
        )


def init_semantic_yaml_metrics(
    yaml_file_path: str,
    agent_config: AgentConfig,
) -> tuple[bool, str]:
    """Import only metric projections from an existing semantic YAML file."""
    if not os.path.exists(yaml_file_path):
        logger.error("Semantic YAML file %s not found", yaml_file_path)
        return False, f"Semantic YAML file {yaml_file_path} not found"

    from datus.storage.semantic_model.semantic_model_init import reject_non_dosi_semantic_yaml

    error = reject_non_dosi_semantic_yaml(yaml_file_path, agent_config)
    if error:
        return False, error

    from datus.tools.func_tool.generation_tools import GenerationTools

    result = GenerationTools(agent_config=agent_config, authoring_format=AUTHORING_FORMAT_OSI).sync_osi_to_db(
        yaml_file_path,
        include_semantic_objects=False,
        include_metrics=True,
    )
    if result.get("success"):
        return True, result.get("message", "")
    return False, result.get("error", "Unknown error")


__all__ = [
    "DEFAULT_METRICS_BATCH_SIZE",
    "init_semantic_yaml_metrics",
    "init_success_story_metrics",
    "init_success_story_metrics_async",
]
