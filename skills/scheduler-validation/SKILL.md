---
name: scheduler-validation
description: Validate scheduled ETL jobs by checking submission, triggerability, run status, and scheduler-side health before a pipeline change is considered complete
tags:
  - scheduler
  - orchestration
  - validation
  - data-engineering
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
---

# Scheduler Validation

Use this skill after a pipeline job is created or updated and you need to confirm it can run in the scheduler.

## Preferred tools

- `list_scheduler_connections`
- `list_scheduler_jobs`
- `get_scheduler_job`
- `trigger_scheduler_job`
- `pause_job`
- `resume_job`

## Core workflow

1. Confirm the target job exists and points at the intended connection or platform.
2. Trigger a test run if that is safe.
3. Poll scheduler status until success or failure is clear.
4. If the run fails, report the scheduler-side failure point and stop promotion.
5. If the run succeeds, return a compact verification summary.

## Checklist

Use the core workflow steps above as the checklist. No external reference document is needed — the preferred tools provide the verification surface directly.

## Output expectations

- job identifier
- scheduler platform
- latest run status
- blocking failure reason when present
- pass / fail decision
