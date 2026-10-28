# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""File-level metric authoring over OSI model documents.

``MetricAuthor`` reads/mutates a single metric *in place* inside its owning
``semantic_model.metrics`` list, preserving the rest of the document (datasets,
relationships, sibling metrics) and DATUS ``custom_extensions``. It operates on
the raw core-schema dicts — not any compiled IR — so round-trips keep
OSI-native fields (``expression.dialects``, ``ai_context``) intact.

Agent uses this only for Explorer metric editing. ``DosiMetricAuthor`` supplies
native document validation after a mutation. This editor is not an LLM tool.
"""

from __future__ import annotations

import glob
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from .authoring import MetricMutationResult, MetricSource
from .exceptions import SemanticCoreException
from .models import ValidationIssue, ValidationResult

# OSI-format constants (the on-disk authoring format these helpers operate on).
DATUS_VENDOR = "DATUS"
DEFAULT_SCHEMA_VERSION = "0.2.0.dev0"

# MetricSource / MetricMutationResult carry this as their ``format``.
FORMAT = "osi"


class MetricAuthoringError(SemanticCoreException):
    """Raised for structural authoring failures (metric not found, name
    mismatch, unresolvable target, invalid source, ...)."""


@dataclass
class _MetricLocation:
    file_path: str
    docs: List[Any]  # every YAML document in the file
    doc_index: int
    model_index: int  # index within docs[doc_index]["semantic_model"]
    metric_index: int  # index within model["metrics"]

    @property
    def model(self) -> Dict[str, Any]:
        return self.docs[self.doc_index]["semantic_model"][self.model_index]

    @property
    def node(self) -> Dict[str, Any]:
        return self.model["metrics"][self.metric_index]


def _model_files(root: str) -> List[str]:
    # A file root pins authoring to exactly that model file; a directory root
    # scans it recursively for OSI YAML and JSON.
    if os.path.isfile(root):
        return [root]
    return sorted(
        glob.glob(os.path.join(root, "**", "*.yaml"), recursive=True)
        + glob.glob(os.path.join(root, "**", "*.yml"), recursive=True)
        + glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
    )


def _read_docs(file_path: str) -> List[Any]:
    with open(file_path, encoding="utf-8") as fh:
        if file_path.lower().endswith(".json"):
            return [json.load(fh)]
        return list(yaml.safe_load_all(fh.read())) or []


def _iter_core_models(docs: List[Any]):
    """Yield (doc_index, model_index, model_dict) for every core semantic model."""
    for doc_index, doc in enumerate(docs):
        if not isinstance(doc, dict):
            continue
        models = doc.get("semantic_model")
        if not isinstance(models, list):
            continue
        for model_index, model in enumerate(models):
            if isinstance(model, dict):
                yield doc_index, model_index, model


def _dump_yaml(obj: Any) -> str:
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True, default_flow_style=False)


def _datus_hints(node: Dict[str, Any]) -> Dict[str, Any]:
    """Merge every DATUS ``custom_extensions`` payload on ``node`` into one dict."""
    merged: Dict[str, Any] = {}
    for ext in node.get("custom_extensions") or []:
        if not isinstance(ext, dict) or str(ext.get("vendor_name", "")).upper() != DATUS_VENDOR:
            continue
        raw = ext.get("data")
        if isinstance(raw, dict):
            merged.update(raw)
        elif isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                merged.update(parsed)
    return merged


def _set_datus_hints(node: Dict[str, Any], updates: Dict[str, Any]) -> None:
    """Write ``updates`` into node's DATUS custom extension, preserving JSON-string style."""
    extensions = node.setdefault("custom_extensions", [])
    for ext in extensions:
        if isinstance(ext, dict) and str(ext.get("vendor_name", "")).upper() == DATUS_VENDOR:
            raw = ext.get("data")
            if isinstance(raw, str):
                try:
                    data = json.loads(raw) if raw.strip() else {}
                except json.JSONDecodeError:
                    data = {}
                data.update(updates)
                ext["data"] = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            else:
                data = dict(raw) if isinstance(raw, dict) else {}
                data.update(updates)
                ext["data"] = data
            return
    # No DATUS entry yet — add one using the JSON-string convention.
    extensions.append(
        {
            "vendor_name": DATUS_VENDOR,
            "data": json.dumps(updates, ensure_ascii=False, separators=(",", ":")),
        }
    )


def json_clone(value: Any) -> Any:
    """Deep-copy plain YAML/JSON data without importing copy for every call site."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def default_validate_document(doc: Dict[str, Any]) -> None:
    """Lightweight structural check of a full OSI core document; raises on error.

    Dosi injects its full document validator. This structural check keeps
    ``MetricAuthor`` usable on its own.
    """
    models = doc.get("semantic_model")
    if not isinstance(models, list) or not models:
        raise MetricAuthoringError("document must contain a non-empty 'semantic_model' list")
    for model in models:
        if not isinstance(model, dict):
            raise MetricAuthoringError("each 'semantic_model' entry must be a mapping")
        for metric in model.get("metrics") or []:
            if not isinstance(metric, dict) or not str(metric.get("name") or "").strip():
                raise MetricAuthoringError("each metric must be a mapping with a non-empty 'name'")


def _atomic_write(file_path: str, docs: List[Any]) -> None:
    directory = os.path.dirname(file_path) or "."
    if file_path.lower().endswith(".json"):
        if len(docs) != 1:
            raise MetricAuthoringError("JSON model files must contain exactly one document")
        payload = json.dumps(docs[0], ensure_ascii=False, indent=2) + "\n"
    else:
        payload = yaml.safe_dump_all(docs, sort_keys=False, allow_unicode=True, default_flow_style=False)
    # Preserve the existing file's permission bits; mkstemp creates 0600, which
    # would otherwise strip group/other read access on first edit.
    mode = stat.S_IMODE(os.stat(file_path).st_mode) if os.path.exists(file_path) else 0o644
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, file_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


class MetricAuthor:
    """Read/write/delete/validate a single metric in its OSI source file.

    Assumes a single writer: the read-modify-write cycle is not file-locked, so
    concurrent writes to the same model file can lose updates. Callers that
    allow concurrent authoring must serialize it upstream.

    Args:
        semantic_models_path: a directory scanned for OSI model YAML/JSON files, or a
            single model file to pin authoring to (siblings are never touched).
        validate_document: optional ``(doc_dict) -> None`` that raises on an
            invalid full document; defaults to :func:`default_validate_document`.
        schema_version: ``version`` stamped on the synthetic document built for
            :meth:`validate` when a metric has no resolvable model context.
        error_cls: exception type raised for structural failures.
    """

    def __init__(
        self,
        semantic_models_path: str,
        *,
        validate_document: Optional[Callable[[Dict[str, Any]], None]] = None,
        schema_version: str = DEFAULT_SCHEMA_VERSION,
        error_cls: type[Exception] = MetricAuthoringError,
    ):
        self._root = semantic_models_path
        self._validate_document = validate_document or default_validate_document
        self._schema_version = schema_version
        self._error_cls = error_cls

    # ---- location -------------------------------------------------------

    def _require_root(self) -> str:
        if not self._root or not (os.path.isdir(self._root) or os.path.isfile(self._root)):
            raise self._error_cls(f"semantic_models_path is not a directory or file: {self._root}")
        return self._root

    def _locate(self, metric_name: str) -> Optional[_MetricLocation]:
        for file_path in _model_files(self._require_root()):
            try:
                docs = _read_docs(file_path)
            except (yaml.YAMLError, json.JSONDecodeError):
                continue
            for doc_index, model_index, model in _iter_core_models(docs):
                for metric_index, metric in enumerate(model.get("metrics") or []):
                    if isinstance(metric, dict) and metric.get("name") == metric_name:
                        return _MetricLocation(file_path, docs, doc_index, model_index, metric_index)
        return None

    def _model_owning_dataset(self, dataset: Optional[str]) -> Optional[Tuple[str, List[Any], int, int]]:
        """Find (file_path, docs, doc_index, model_index) whose model declares ``dataset``."""
        if not dataset:
            return None
        for file_path in _model_files(self._require_root()):
            try:
                docs = _read_docs(file_path)
            except (yaml.YAMLError, json.JSONDecodeError):
                continue
            for doc_index, model_index, model in _iter_core_models(docs):
                for ds in model.get("datasets") or []:
                    if isinstance(ds, dict) and ds.get("name") == dataset:
                        return file_path, docs, doc_index, model_index
        return None

    def _sole_model(self) -> Optional[Tuple[str, List[Any], int, int]]:
        found: List[Tuple[str, List[Any], int, int]] = []
        for file_path in _model_files(self._require_root()):
            try:
                docs = _read_docs(file_path)
            except (yaml.YAMLError, json.JSONDecodeError):
                continue
            for doc_index, model_index, _model in _iter_core_models(docs):
                found.append((file_path, docs, doc_index, model_index))
                if len(found) > 1:
                    return None
        return found[0] if len(found) == 1 else None

    # ---- read -----------------------------------------------------------

    def read(self, metric_name: str) -> MetricSource:
        loc = self._locate(metric_name)
        if loc is None:
            raise self._error_cls(f"Metric `{metric_name}` was not found in {self._root}.")
        return MetricSource(
            name=metric_name,
            format=FORMAT,
            text=_dump_yaml(loc.node),
            semantic_model=str(loc.model.get("name") or "") or None,
            file_path=loc.file_path,
        )

    # ---- write ----------------------------------------------------------

    def _parse_node(self, source: str) -> Dict[str, Any]:
        try:
            node = yaml.safe_load(source)
        except yaml.YAMLError as exc:
            raise self._error_cls(f"Invalid YAML: {exc}") from exc
        if not isinstance(node, dict) or not node.get("name"):
            raise self._error_cls("Metric source must be a mapping with a `name` field.")
        return node

    def write(
        self,
        metric_name: str,
        source: str,
        *,
        subject_path: Optional[List[str]] = None,
        create: bool = False,
    ) -> MetricMutationResult:
        node = self._parse_node(source)
        if node.get("name") != metric_name:
            raise self._error_cls(f"Metric name in source (`{node.get('name')}`) does not match `{metric_name}`.")
        if subject_path is not None:
            _set_datus_hints(node, {"subject_path": list(subject_path)})

        existing = self._locate(metric_name)
        if create and existing is not None:
            raise self._error_cls(f"Metric `{metric_name}` already exists at {existing.file_path}.")
        if not create and existing is None:
            raise self._error_cls(f"Metric `{metric_name}` does not exist; use create=True.")

        if existing is not None:
            existing.model["metrics"][existing.metric_index] = node
            file_path, docs, doc_index = (
                existing.file_path,
                existing.docs,
                existing.doc_index,
            )
            model_index = existing.model_index
        else:
            target = self._model_owning_dataset(_datus_hints(node).get("dataset")) or self._sole_model()
            if target is None:
                raise self._error_cls(
                    "Cannot resolve a target semantic model for the new metric. "
                    "Set the metric's DATUS `dataset` to a dataset declared by an existing "
                    "semantic model, or ensure exactly one semantic model exists."
                )
            file_path, docs, doc_index, model_index = target
            docs[doc_index]["semantic_model"][model_index].setdefault("metrics", []).append(node)

        self._validate_document(docs[doc_index])
        _atomic_write(file_path, docs)
        return MetricMutationResult(
            name=metric_name,
            format=FORMAT,
            file_path=file_path,
            semantic_model=str(docs[doc_index]["semantic_model"][model_index].get("name") or "") or None,
            created=create,
            affected_paths=[file_path],
        )

    # ---- delete ---------------------------------------------------------

    def delete(self, metric_name: str) -> MetricMutationResult:
        loc = self._locate(metric_name)
        if loc is None:
            raise self._error_cls(f"Metric `{metric_name}` was not found in {self._root}.")
        model_name = str(loc.model.get("name") or "") or None
        del loc.model["metrics"][loc.metric_index]
        self._validate_document(loc.docs[loc.doc_index])
        _atomic_write(loc.file_path, loc.docs)
        return MetricMutationResult(
            name=metric_name,
            format=FORMAT,
            file_path=loc.file_path,
            semantic_model=model_name,
            deleted=True,
            affected_paths=[loc.file_path],
        )

    # ---- validate -------------------------------------------------------

    def validate(self, source: str, *, metric_name: Optional[str] = None) -> ValidationResult:
        try:
            node = self._parse_node(source)
            name = metric_name or str(node.get("name"))
            # Validate the metric inside its real model context when we can
            # resolve one, else wrap it in a minimal single-model document.
            existing = self._locate(name) if name else None
            if existing is not None:
                model = json_clone(existing.model)
                replaced = False
                for idx, metric in enumerate(model.get("metrics") or []):
                    if isinstance(metric, dict) and metric.get("name") == name:
                        model["metrics"][idx] = node
                        replaced = True
                        break
                if not replaced:
                    model.setdefault("metrics", []).append(node)
            else:
                target = self._model_owning_dataset(_datus_hints(node).get("dataset")) or self._sole_model()
                if target is not None:
                    _file_path, docs, doc_index, model_index = target
                    model = json_clone(docs[doc_index]["semantic_model"][model_index])
                    model.setdefault("metrics", []).append(node)
                else:
                    model = {
                        "name": "authoring_preview",
                        "datasets": [],
                        "metrics": [node],
                    }

            doc = {"version": self._schema_version, "semantic_model": [model]}
            self._validate_document(doc)
        except Exception as exc:  # noqa: BLE001 - surface any validation failure to the caller
            return ValidationResult(
                valid=False,
                issues=[ValidationIssue(severity="error", message=str(exc))],
            )
        return ValidationResult(valid=True, issues=[])
