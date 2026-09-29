# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Configuration for Agent's embedded Dosi runtime."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class DosiConfig(BaseModel):
    """Model files and the current datasource connection for the native engine."""

    model_config = ConfigDict(extra="allow")

    datasource: Optional[str] = None
    timeout_seconds: int = Field(default=30)
    service_type: str = "dosi"
    # Path to the OSI semantic model file (.yaml/.yml/.json). Takes precedence
    # over semantic_models_path.
    semantic_model_path: Optional[str] = None
    # Directory of OSI models (Datus convention, e.g. subject/semantic_models/
    # <datasource>). Used when semantic_model_path is unset: discovery and
    # queries route across every top-level YAML/YML/JSON model in the directory.
    semantic_models_path: Optional[str] = None
    # Named connection profile; falls back to the base-class `datasource`.
    connection: Optional[str] = None
    # Inline datasource entry (agent.yml vocabulary: type/host/port/...).
    db_config: Optional[Dict[str, Any]] = None
    # Explicit SQL dialect for dry-run compilation without a connection.
    dialect: Optional[str] = None
    # Per-profile connection-pool cap inside the engine.
    pool_size: int = 8
