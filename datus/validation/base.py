# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ValidationIssue(BaseModel):
    """Single validation finding produced by a validator."""

    code: str = Field(..., description="Stable issue code")
    severity: Literal["error", "warning"] = Field(default="error", description="Issue severity")
    message: str = Field(..., description="Human-readable validation message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Structured validator details")


class ValidationResult(BaseModel):
    """Normalized output of a validation pass."""

    passed: bool = Field(..., description="Whether validation passed")
    issues: List[ValidationIssue] = Field(default_factory=list, description="Validation issues")
    retry_hint: Optional[str] = Field(default=None, description="Suggested retry guidance")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional validation metadata")


class BaseValidator(ABC):
    """Abstract validator interface for node-level output validation."""

    validator_id: str

    def __init__(self, validator_id: str):
        self.validator_id = validator_id

    @abstractmethod
    def validate(self, sql: str, context: Dict[str, Any]) -> ValidationResult:
        """Validate generated SQL against the provided context."""
