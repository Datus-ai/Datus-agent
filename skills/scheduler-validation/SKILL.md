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
allowed_agents:
  - scheduler
---

# Scheduler Validation

Use this skill after a pipeline job is created or updated. You MUST complete ALL steps below before reporting success.

## Core workflow

**IMPORTANT**: Every step is mandatory. Do NOT skip any step or report success without completing all checks.

1. **Confirm job exists** — call `get_scheduler_job(job_id)` and verify:
   - Job status is `active` (not paused or errored)
   - Schedule expression matches the intended cron
2. **Trigger a test run** — call `trigger_scheduler_job(job_id)` to start a manual run
3. **Poll until completion** — call `list_job_runs(job_id, limit=1)` repeatedly (up to 5 attempts, 10s apart) until the run status is `success` or `failed`
4. **If the run failed** — call `get_run_log(job_id, run_id)` to retrieve the error log, report the failure reason, and STOP (do not report success)
5. **If the run succeeded** — report a compact verification summary

## Validation checklist (per job)

| Check | Tool | What to verify |
|-------|------|----------------|
| Job exists | `get_scheduler_job` | Status is active, schedule is correct |
| Test run triggered | `trigger_scheduler_job` | Returns a run_id |
| Run completes | `list_job_runs` | Latest run status is `success` |
| Run log clean | `get_run_log` | No errors in output (only check on failure) |

## Output format

Report a table with these columns:

| Column | Description |
|--------|-------------|
| Job ID | The scheduler job identifier |
| Schedule | Cron expression |
| Test Run | PASS if run succeeded, FAIL + error summary otherwise |
| Overall | PASS or FAIL |

If the test run fails, include the error message from `get_run_log` so the user can diagnose the issue.
