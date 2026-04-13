# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

from typing import Iterable

from datus.validation.base import ValidationIssue


def format_validation_retry_feedback(issues: Iterable[ValidationIssue]) -> str:
    """Format validation issues into a short retry prompt."""

    lines = ["Validation failed:"]
    for idx, issue in enumerate(issues, start=1):
        lines.append(f"{idx}. {issue.code}: {issue.message}")
    lines.append("")
    lines.append("Regenerate the SQL. Preserve correct parts and fix only the issues above.")
    return "\n".join(lines)
