# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Validation infrastructure for table-producing subagents.

This module provides hook-driven validation that fires automatically after
mutating tool calls (execute_ddl, execute_write, transfer_query_result) and
at the end of an agent run. It separates validation into two layers:

- Layer A (builtin_checks): code-level invariants (table exists, row count,
  cross-DB row count parity). Always enforced.
- Layer B (llm_runner): LLM-interpreted validator skills. Gated by
  ``agent.validation.skill_validators_enabled``.

Blocking failures raise ``ValidationBlockingException`` which is caught in the
owning node's ``execute_stream`` and used to drive retries with the failure
report injected as a user message. The retry budget is configurable via
``agent.validation.max_retries`` (default 3).
"""

from datus.validation.exceptions import ValidationBlockingException
from datus.validation.hook import ValidationHook
from datus.validation.report import (
    CheckResult,
    DBRef,
    DeliverableTarget,
    SessionTarget,
    TableTarget,
    TargetFilter,
    TransferTarget,
    ValidationReport,
)

__all__ = [
    "CheckResult",
    "DBRef",
    "DeliverableTarget",
    "SessionTarget",
    "TableTarget",
    "TargetFilter",
    "TransferTarget",
    "ValidationBlockingException",
    "ValidationHook",
    "ValidationReport",
]
