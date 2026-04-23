# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Validation-specific exceptions.

:class:`ValidationBlockingException` is raised by :class:`ValidationHook` when a
deliverable check fails at blocking severity. The owning node's
``execute_stream`` catches it and injects the report back into the agent loop
for retry.
"""

from __future__ import annotations

from datus.utils.exceptions import DatusException, ErrorCode
from datus.validation.report import ValidationReport


class ValidationBlockingException(DatusException):
    """Raised by :class:`ValidationHook` when validation fails (blocking).

    Carries the full :class:`ValidationReport` so the outer retry loop can
    render it back into a synthetic user message for the agent.
    """

    def __init__(self, report: ValidationReport):
        self.report = report
        failures = [c.name for c in report.checks if not c.passed and c.severity == "blocking"]
        super().__init__(
            code=ErrorCode.VALIDATION_BLOCKING_FAILURE,
            message_args={"failures": ", ".join(failures) if failures else "unknown"},
        )
