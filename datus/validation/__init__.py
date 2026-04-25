# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Validation infrastructure for table-, chart-, dashboard-, dataset-, and
scheduler job-producing subagents.

This module provides hook-driven validation that fires automatically after
mutating tool calls (execute_ddl, execute_write, transfer_query_result,
create_chart, create_dashboard, create_dataset, submit_sql_job, update_job) and
at the end of an agent run. It separates validation into two layers:

- Layer A (builtin_checks): code-level invariants (table exists, row count,
  cross-DB row count parity, BI resources exist, scheduler job exists/status).
  Always enforced.
- Layer B (llm_runner): LLM-interpreted validator skills. Gated by
  ``agent.validation.skill_validators_enabled``.

Blocking failures are recorded in ``ValidationHook.final_report``; the owning
node's ``execute_stream`` reads it after the stream ends and drives retries by
injecting the failure report as a user message. The retry budget is
configurable via ``agent.validation.max_retries`` (default 3).
"""

from datus.validation.hook import ValidationHook
from datus.validation.report import (
    ChartTarget,
    CheckResult,
    DashboardTarget,
    DatasetTarget,
    DBRef,
    DeliverableTarget,
    SchedulerJobTarget,
    SessionTarget,
    TableTarget,
    TargetFilter,
    TransferTarget,
    ValidationReport,
)

__all__ = [
    "ChartTarget",
    "CheckResult",
    "DashboardTarget",
    "DatasetTarget",
    "DBRef",
    "DeliverableTarget",
    "SchedulerJobTarget",
    "SessionTarget",
    "TableTarget",
    "TargetFilter",
    "TransferTarget",
    "ValidationHook",
    "ValidationReport",
]
