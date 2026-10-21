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
    # Apache Ossie ontology loaded beside the model (model-first). Set per
    # engine handle once the registry pairs a model file with its ontology.
    ontology_path: Optional[str] = None
    # Directory of ontology files (Datus convention: subject/semantic_models/
    # ontology/<datasource>). Unset means ontologies are not loaded at all,
    # which keeps every non-ontology agent on the plain model.
    ontology_dir: Optional[str] = None
    # An Ossie mapping document (apache/ossie#458) binding the model to its
    # ontology. Set per engine handle by the registry, like ontology_path.
    mapping_path: Optional[str] = None
    # Local files that mapping reaches by relative iri; the engine reads them,
    # so their edits rebuild it. Set by the registry with mapping_path.
    mapping_references: list[str] = Field(default_factory=list)
