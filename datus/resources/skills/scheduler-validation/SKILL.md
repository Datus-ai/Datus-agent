---
name: scheduler-validation
description: Scheduler validator driven by ValidationHook — read-only static verification of scheduled jobs (schedule correctness, configuration, most recent run outcome). Does not trigger test runs.
tags:
  - scheduler
  - orchestration
  - validation
  - data-engineering
version: "2.0.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - scheduler
kind: validator
severity: blocking
mode: llm
targets:
  - type: scheduler_job
---

# Scheduler Validation

Driven by `ValidationHook.on_end` for `scheduler` runs. The hook passes a
`SessionTarget` containing every `SchedulerJobTarget` the run delivered.
Iterate each target and run the checks below.

## Target shape

You receive `SessionTarget.targets` — loop over `SchedulerJobTarget` entries.
Each carries `platform` + `job_id` + optional `job_name`.

Layer A (the builtin hook) has already confirmed each job **exists** and
**is not in a failed status**. Your job is to verify schedule correctness and
recent runtime outcome when run history is available.

## Core workflow

Read-only static verification. Never trigger a test run — scheduled jobs may
be expensive / long-running / have downstream side effects. Runtime
verification is the user's call.

For every `SchedulerJobTarget` in the session:

1. **Verify schedule** — call `get_scheduler_job(job_id)` and confirm the
   `schedule` expression matches the intended cron (e.g. user asked for
   "daily" but job submitted `0 * * * *`).
2. **Verify configuration** — check other fields returned by
   `get_scheduler_job` (description, SQL path, job_type, connection id) match
   the intended setup.
3. **Inspect most recent run, if any** — call `list_job_runs(job_id, limit=1)`.
   - No runs yet → advisory note "no runtime history; user-initiated trigger
     required to verify runtime health". Not a blocking failure.
   - Latest run `success` → report the success.
   - Latest run `failed` → call `get_run_log(job_id, run_id)`, surface the
     error as a blocking finding so the user can fix before the next schedule
     window hits.
   - Latest run `running` → advisory (in-progress; can't judge yet).

## Validation checklist (per job)

| Check | Tool | What to verify |
|-------|------|----------------|
| Job exists | `get_scheduler_job` | Status is active, schedule is correct |
| Latest run inspected | `list_job_runs` | Latest run status is `success`, or no run history is advisory |
| Run log clean | `get_run_log` | No errors in output (only check on failure) |

## Output format

Report a table with these columns:

| Column | Description |
|--------|-------------|
| Job ID | The scheduler job identifier |
| Schedule | Cron expression |
| Latest Run | PASS if latest run succeeded, FAIL + error summary if it failed, ADVISORY if absent/running |
| Overall | PASS or FAIL |

If the latest run failed, include the error message from `get_run_log` so the user can diagnose the issue. Never call `trigger_scheduler_job`.
