# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import sqlglot
from sqlglot import expressions

from datus.tools.skill_tools.skill_manager import SkillManager
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import parse_read_dialect
from datus.validation.base import BaseValidator, ValidationIssue, ValidationResult
from datus.validation.models import SkillValidatorSpec

logger = get_logger(__name__)


class SqlParseableValidator(BaseValidator):
    """Ensure the generated SQL can be parsed for the current dialect."""

    def __init__(self):
        super().__init__("sql_parseable")

    def validate(self, sql: str, context: Dict[str, Any]) -> ValidationResult:
        dialect = parse_read_dialect(str(context.get("db_type") or "").lower())
        try:
            sqlglot.parse_one(sql, read=dialect or None, error_level=sqlglot.ErrorLevel.RAISE)
        except Exception as exc:
            issue = ValidationIssue(
                code=self.validator_id,
                severity="error",
                message=f"SQL is not parseable: {exc}",
                details={"dialect": dialect or None},
            )
            return ValidationResult(
                passed=False,
                issues=[issue],
                retry_hint="Fix the SQL syntax while preserving the intended output columns.",
            )
        return ValidationResult(passed=True)


class OutputColumnsMatchContractValidator(BaseValidator):
    """Validate that final SELECT columns match the expected output contract exactly."""

    def __init__(self):
        super().__init__("output_columns_match_contract")

    def validate(self, sql: str, context: Dict[str, Any]) -> ValidationResult:
        expected_columns = context.get("expected_output_columns") or []
        if not expected_columns:
            return ValidationResult(passed=True, metadata={"skipped": True, "reason": "missing_expected_columns"})

        dialect = parse_read_dialect(str(context.get("db_type") or "").lower())
        try:
            parsed = sqlglot.parse_one(sql, read=dialect or None, error_level=sqlglot.ErrorLevel.RAISE)
        except Exception as exc:
            issue = ValidationIssue(
                code=self.validator_id,
                severity="error",
                message=f"Unable to inspect output columns because SQL failed to parse: {exc}",
            )
            return ValidationResult(passed=False, issues=[issue])

        select_nodes = list(parsed.find_all(expressions.Select))
        final_select = select_nodes[-1] if select_nodes else None
        if final_select is None:
            issue = ValidationIssue(
                code=self.validator_id,
                severity="error",
                message="Unable to find a final SELECT statement to validate output columns.",
            )
            return ValidationResult(passed=False, issues=[issue])

        actual_columns = []
        for expr in final_select.expressions:
            name = expr.alias_or_name or getattr(expr, "output_name", "") or expr.sql(dialect=dialect or None)
            actual_columns.append(name)

        issues: List[ValidationIssue] = []
        missing = [col for col in expected_columns if col not in actual_columns]
        extra = [col for col in actual_columns if col not in expected_columns]
        if missing:
            issues.append(
                ValidationIssue(
                    code=self.validator_id,
                    severity="error",
                    message=f"Final SELECT is missing expected columns: {', '.join(missing)}",
                    details={"missing_columns": missing},
                )
            )
        if extra:
            issues.append(
                ValidationIssue(
                    code=self.validator_id,
                    severity="error",
                    message=f"Final SELECT has unexpected columns: {', '.join(extra)}",
                    details={"extra_columns": extra},
                )
            )
        if not issues and actual_columns != expected_columns:
            issues.append(
                ValidationIssue(
                    code=self.validator_id,
                    severity="error",
                    message=(
                        "Final SELECT column order does not match the expected contract order. "
                        f"Expected: {expected_columns}; actual: {actual_columns}"
                    ),
                    details={"expected_columns": expected_columns, "actual_columns": actual_columns},
                )
            )

        return ValidationResult(
            passed=not issues,
            issues=issues,
            metadata={"expected_columns": expected_columns, "actual_columns": actual_columns},
        )


class SqlPatternValidator(BaseValidator):
    """Validate SQL against required/forbidden string patterns."""

    def __init__(
        self,
        validator_id: str,
        must_contain: Optional[List[str]] = None,
        must_not_contain: Optional[List[str]] = None,
    ):
        super().__init__(validator_id)
        self.must_contain = must_contain or []
        self.must_not_contain = must_not_contain or []

    def validate(self, sql: str, context: Dict[str, Any]) -> ValidationResult:
        lowered_sql = sql.lower()
        issues: List[ValidationIssue] = []

        missing_patterns = [pattern for pattern in self.must_contain if pattern.lower() not in lowered_sql]
        forbidden_patterns = [pattern for pattern in self.must_not_contain if pattern.lower() in lowered_sql]

        if missing_patterns:
            issues.append(
                ValidationIssue(
                    code=self.validator_id,
                    severity="error",
                    message=f"SQL is missing required patterns: {', '.join(missing_patterns)}",
                    details={"missing_patterns": missing_patterns},
                )
            )
        if forbidden_patterns:
            issues.append(
                ValidationIssue(
                    code=self.validator_id,
                    severity="error",
                    message=f"SQL contains forbidden patterns: {', '.join(forbidden_patterns)}",
                    details={"forbidden_patterns": forbidden_patterns},
                )
            )

        return ValidationResult(
            passed=not issues,
            issues=issues,
            metadata={
                "must_contain": self.must_contain,
                "must_not_contain": self.must_not_contain,
            },
        )


class PinnedDateLiteralValidator(BaseValidator):
    """Require deterministic pinned dates instead of dynamic current-date functions."""

    def __init__(
        self,
        validator_id: str,
        forbidden_patterns: Optional[List[str]] = None,
    ):
        super().__init__(validator_id)
        self.forbidden_patterns = forbidden_patterns or ["current_date", "current_timestamp", "now()"]

    def validate(self, sql: str, context: Dict[str, Any]) -> ValidationResult:
        reference_date = context.get("reference_date")
        if not reference_date:
            return ValidationResult(passed=True, metadata={"skipped": True, "reason": "missing_reference_date"})

        lowered_sql = sql.lower()
        issues: List[ValidationIssue] = []
        forbidden_patterns = [pattern for pattern in self.forbidden_patterns if pattern.lower() in lowered_sql]
        if forbidden_patterns:
            issues.append(
                ValidationIssue(
                    code=self.validator_id,
                    severity="error",
                    message=(
                        "SQL uses dynamic current-date functions and must be pinned to the provided reference date "
                        f"{reference_date}. Forbidden patterns: {', '.join(forbidden_patterns)}"
                    ),
                    details={"forbidden_patterns": forbidden_patterns, "reference_date": reference_date},
                )
            )

        if reference_date not in sql:
            issues.append(
                ValidationIssue(
                    code=self.validator_id,
                    severity="error",
                    message=(
                        f"SQL must include the pinned reference date literal {reference_date}. "
                        f"Use DATE '{reference_date}' instead of dynamic current-date functions."
                    ),
                    details={"reference_date": reference_date},
                )
            )

        return ValidationResult(
            passed=not issues,
            issues=issues,
            metadata={"reference_date": reference_date, "forbidden_patterns": self.forbidden_patterns},
        )


class ValidatorRegistry:
    """Collect built-in and skill-declared validators for a node invocation."""

    def __init__(self, skill_manager: Optional[SkillManager] = None):
        self.skill_manager = skill_manager

    def build_validators(
        self,
        node_name: str,
        skill_patterns: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[BaseValidator]:
        validators: List[BaseValidator] = [SqlParseableValidator()]
        if not self.skill_manager:
            return validators

        context = context or {}
        available_skills = self.skill_manager.get_available_skills(node_name=node_name, patterns=skill_patterns)
        for skill in available_skills:
            specs = self._load_skill_validator_specs(skill.location / "assets" / "validators.json")
            for spec in specs:
                if spec.applies_to.node and spec.applies_to.node != node_name:
                    continue
                if spec.requires_context and any(not context.get(key) for key in spec.requires_context):
                    continue
                validator = self._instantiate_validator(spec)
                if validator:
                    validators.append(validator)
        return validators

    def _load_skill_validator_specs(self, path: Path) -> List[SkillValidatorSpec]:
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError("validators.json must contain a list")
            return [SkillValidatorSpec.model_validate(item) for item in payload]
        except Exception as exc:
            logger.warning(f"Failed to load skill validators from {path}: {exc}")
            return []

    def _instantiate_validator(self, spec: SkillValidatorSpec) -> Optional[BaseValidator]:
        if spec.type == "output_columns_match_contract":
            return OutputColumnsMatchContractValidator()
        if spec.type == "sql_pattern":
            return SqlPatternValidator(
                validator_id=spec.id,
                must_contain=spec.config.get("must_contain", []),
                must_not_contain=spec.config.get("must_not_contain", []),
            )
        if spec.type == "pinned_date_literal":
            return PinnedDateLiteralValidator(
                validator_id=spec.id,
                forbidden_patterns=spec.config.get(
                    "forbidden_patterns",
                    ["current_date", "current_timestamp", "now()"],
                ),
            )
        logger.debug(f"Skipping unsupported validator type '{spec.type}' from skill asset")
        return None


def run_validators(validators: List[BaseValidator], sql: str, context: Dict[str, Any]) -> ValidationResult:
    """Run validators sequentially and combine their issues."""

    combined_issues: List[ValidationIssue] = []
    metadata: Dict[str, Any] = {}
    retry_hints: List[str] = []
    for validator in validators:
        result = validator.validate(sql, context)
        if result.metadata:
            metadata[validator.validator_id] = result.metadata
        if not result.passed:
            combined_issues.extend(result.issues)
            if result.retry_hint:
                retry_hints.append(result.retry_hint)
    return ValidationResult(
        passed=not combined_issues,
        issues=combined_issues,
        retry_hint="\n".join(retry_hints) if retry_hints else None,
        metadata=metadata or None,
    )
