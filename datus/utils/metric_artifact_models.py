# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Save and resolve frozen metric models without reading the live project."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from datus.schemas.metric_artifact_query import MODEL_SNAPSHOT_HASH_LENGTHS, MetricQueryFile


def save_metric_query_snapshot(query: MetricQueryFile, content: str, *, artifact_dir: Path) -> MetricQueryFile:
    """Save under a short hash, extending it on collision without replacing any file.

    Publish complete bytes with an exclusive hard link so concurrent writers
    cannot overwrite another model or observe a partially written snapshot.
    The full SHA-256 remains authoritative at every filename length.
    """
    if query.model_snapshot_path is None:
        raise ValueError("saving a metric model requires model_snapshot_path")
    resolve_metric_query_snapshot(query, files={query.model_snapshot_path: content})
    root = artifact_dir.resolve()
    models = root / "models"
    if not models.resolve().is_relative_to(root):
        raise ValueError("metric model snapshot escapes the artifact directory")
    models.mkdir(parents=True, exist_ok=True)
    suffix = Path(query.model_snapshot_path).suffix
    with tempfile.NamedTemporaryFile(dir=models, prefix=".model-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content.encode("utf-8"))
            stream.close()
            for length in MODEL_SNAPSHOT_HASH_LENGTHS:
                relative = f"models/{query.model_revision[:length]}{suffix}"
                target = root / relative
                if not target.resolve().is_relative_to(root):
                    raise ValueError("metric model snapshot escapes the artifact directory")
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    if hashlib.sha256(target.read_bytes()).hexdigest() != query.model_revision:
                        continue
                return query.model_copy(update={"model_snapshot_path": relative})
            raise ValueError("metric model snapshot conflicts with an existing full SHA-256 filename")
        finally:
            temporary.unlink(missing_ok=True)


def resolve_metric_query_snapshot(
    query: MetricQueryFile,
    *,
    files: Mapping[str, str] | None = None,
    artifact_dir: Path | None = None,
) -> MetricQueryFile:
    """Hydrate execution-only model bytes while retaining the saved reference.

    A reference is authoritative, even when inline bytes are also present. A
    broken reference must not fall back to an embedded or current Studio model.
    Legacy 1.0 recipes continue to execute their inline snapshot unchanged.
    """
    content = query.model_snapshot
    if query.model_snapshot_path is not None:
        relative = query.model_snapshot_path
        if files is not None:
            content = files.get(relative)
        elif artifact_dir is not None:
            root = artifact_dir.resolve()
            path = root / relative
            if not path.resolve().is_relative_to(root):
                raise ValueError("metric model snapshot escapes the artifact directory")
            try:
                content = path.read_bytes().decode("utf-8")
            except OSError as exc:
                raise ValueError(f"metric model snapshot missing: {relative}") from exc
        else:
            content = None
    if content is None:
        raise ValueError("metric model snapshot missing; resave the metric query before publishing")
    if hashlib.sha256(content.encode("utf-8")).hexdigest() != query.model_revision:
        raise ValueError("metric model snapshot must match model_revision")
    return query.model_copy(update={"model_snapshot": content})
