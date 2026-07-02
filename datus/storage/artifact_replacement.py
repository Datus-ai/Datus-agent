# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

ArtifactReplacementPlan = Tuple[Any, str, List[Dict[str, Any]]]
ArtifactSnapshot = Tuple[Any, str, List[Dict[str, Any]]]


def snapshot_artifact_replacements(replacement_plans: List[ArtifactReplacementPlan]) -> List[ArtifactSnapshot]:
    snapshots = []
    for rag, yaml_path, _rows in replacement_plans:
        snapshots.append((rag, yaml_path, rag.list_artifact_rows(yaml_path)))
    return snapshots


def restore_artifact_replacements(snapshots: List[ArtifactSnapshot]) -> None:
    for rag, yaml_path, rows in reversed(snapshots):
        try:
            rag.restore_artifact_rows(yaml_path, rows)
        except Exception as restore_exc:
            logger.warning("Failed to restore artifact rows for %s after sync failure: %s", yaml_path, restore_exc)


def delete_stale_artifact_rows(replacement_plans: List[ArtifactReplacementPlan]) -> None:
    for rag, yaml_path, rows in replacement_plans:
        rag.delete_artifact_rows_except(yaml_path, [row.get("id", "") for row in rows])
