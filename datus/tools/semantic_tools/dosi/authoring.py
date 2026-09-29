# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Native-Dosi metric authoring over the shared OSI file editor."""

from __future__ import annotations

from typing import Any, Optional

import yaml

from datus.tools.semantic_tools.dosi.engine import load_binding
from datus.tools.semantic_tools.dosi.model import (
    legacy_window_error,
    legacy_window_findings,
    validation_messages,
)
from datus.tools.semantic_tools.exceptions import SemanticCoreException
from datus.tools.semantic_tools.metric_author import MetricAuthor


class DosiAuthoringError(SemanticCoreException):
    """A model mutation rejected by the native Dosi contract."""


def _invalid_document_payload(code: str, message: str) -> dict[str, Any]:
    return {
        "valid": False,
        "issues": [
            {
                "severity": "error",
                "code": code,
                "location": "semantic_model",
                "message": message,
            }
        ],
        "compile_errors": [],
    }


def dosi_validation_text_payload(
    model_text: str,
    metric_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Validate source text and always return structured document issues."""

    try:
        document = yaml.safe_load(model_text)
    except yaml.YAMLError as exc:
        return _invalid_document_payload("invalid_yaml", str(exc))
    if not isinstance(document, dict):
        return _invalid_document_payload(
            "invalid_document_root",
            "OSI model must contain one object",
        )

    findings = legacy_window_findings(document)
    if findings:
        return _invalid_document_payload(
            "legacy_window_hint",
            legacy_window_error(findings),
        )

    if metric_names is None:
        return dict(load_binding().validate(model_text))
    return dict(load_binding().validate(model_text, metric_names=metric_names))


def dosi_validation_payload(document: dict[str, Any]) -> dict[str, Any]:
    """Validate a parsed OSI document through every native-Dosi preflight."""

    try:
        model_text = yaml.safe_dump(document, sort_keys=False)
    except yaml.YAMLError as exc:
        raise DosiAuthoringError(f"cannot serialize OSI document: {exc}") from exc
    return dosi_validation_text_payload(model_text)


def validate_dosi_document(document: dict[str, Any]) -> None:
    """Reject a raw OSI document unless every native-Dosi preflight passes."""

    payload = dosi_validation_payload(document)
    if payload.get("valid"):
        return
    messages = validation_messages(payload) or ["native Dosi validation failed"]
    raise DosiAuthoringError("; ".join(messages))


class DosiMetricAuthor(MetricAuthor):
    """Shared file mutations guarded by the native engine's model contract."""

    def __init__(self, semantic_models_path: str):
        binding = load_binding()
        super().__init__(
            semantic_models_path,
            validate_document=validate_dosi_document,
            schema_version=str(binding.SPEC_VERSION),
            error_cls=DosiAuthoringError,
        )
