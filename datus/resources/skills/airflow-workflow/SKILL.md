---
name: airflow-workflow
description: Execution guide for Airflow scheduled jobs — troubleshooting, updating, conn_id conventions, and cron references
tags:
  - scheduler
  - airflow
  - workflow
version: 1.0.0
user_invocable: false
allowed_agents:
  - scheduler
---

# Airflow Workflow

Execution guide for the scheduler subagent working with Airflow.

## Troubleshoot a Failed Job

1. **Check job status** — `get_scheduler_job(job_id)`
2. **List recent runs** — `list_job_runs(job_id, limit=5)` to find the failed run
3. **Get error log** — `get_run_log(job_id, run_id)` for the failed run_id
4. **Analyze the error** — common failure categories:
   - SQL syntax error → fix SQL and `update_job()`
   - Connection failure → check conn_id in Airflow Connections (Admin > Connections), verify host is reachable from the scheduler worker
   - Timeout → optimize the query or increase resources
   - Permission denied → verify DB credentials in Airflow Connections (Admin > Connections)
5. **Fix and re-run**:
   - Update SQL: `update_job(job_id, sql_file_path=..., job_name=..., conn_id=...)`
   - Manual trigger to verify: `trigger_scheduler_job(job_id)`
   - Confirm success: `list_job_runs(job_id, limit=1)`

## Update an Existing Job

1. **Check current state** — `get_scheduler_job(job_id)` to see existing config
2. **Pause the job** — `pause_job(job_id)` to prevent runs during update
3. **Write SQL** — use `write_file` or `edit_file` to save the new SQL under
   `jobs/<job_name>.sql`
4. **Update** — `update_job(job_id, sql_file_path=..., job_name=..., conn_id=...)`
5. **Resume** — `resume_job(job_id)` to re-enable scheduling
6. **Do not manually trigger** after a normal create/update unless the user
   explicitly asks for an immediate run. Deterministic validation triggers and
   polls deliverable scheduler jobs after the agent returns the target.

## Delete an Existing Job

1. **Confirm with the user** — deletion is destructive.
2. **Delete** — call `delete_job(job_id)`.
3. **Honor the tool result** — if `delete_job` returns `success=0`, report the
   deletion as failed or incomplete. Do not claim completion or success.
4. **Verify only with direct lookup** — use `get_scheduler_job(job_id)` if you
   need a follow-up check. For Airflow, scheduling deletion is complete when
   the job is not found or is inactive/deleted.
5. **Do not rely on list output** — `list_scheduler_jobs` may omit an Airflow DAG
   after its file is removed even while Airflow metadata still exists and blocks
   re-creation with the same job id.
6. **Use precise wording for partial cleanup** — if metadata still exists but
   the DAG is inactive/deleted, say scheduling has been removed and metadata
   cleanup is pending. The same `dag_id` may not be immediately reusable via
   submit; use update or retry cleanup if needed.
7. **Use explicit file deletion only** — `delete_job` owns Airflow DAG file
   removal. For other files, use a dedicated delete-file tool if one is
   available; otherwise report that file deletion is unavailable. Do not
   overwrite or empty files as a substitute for deletion.

## DB Connection (`conn_id`)

`submit_sql_job` and `update_job` require `conn_id` — the Airflow Connection ID for the target database.
The connection is managed entirely by Airflow (Admin > Connections) and resolved at runtime by the scheduler worker.

Available conn_id values are shown in the `submit_sql_job` and `update_job` tool descriptions (from `scheduler.connections` in agent.yml).

## Naming Conventions

- `job_name`: `<frequency>_<domain>_<description>`, e.g. `daily_sales_summary`, `hourly_order_count`
- SQL file: `jobs/<job_name>.sql`

Before calling `submit_sql_job` or `update_job`, create or update that SQL
file with `write_file` / `edit_file`. Do not ask the user to create the file
when filesystem tools are available.

## Common Cron Expressions

| Schedule | Cron |
|----------|------|
| Every day at 8am | `0 8 * * *` |
| Every hour | `0 * * * *` |
| Every 2 hours | `0 */2 * * *` |
| Monday at 9am | `0 9 * * 1` |
| 1st of month at midnight | `0 0 1 * *` |

## Quick Reference

| Goal | Tool |
|------|------|
| Create SQL file | `write_file(path="jobs/<job_name>.sql", content=...)` |
| Submit SQL job | `submit_sql_job(job_name, sql_file_path, conn_id)` |
| Submit SparkSQL job | `submit_sparksql_job(job_name, sql_file_path)` |
| Check job status | `get_scheduler_job(job_id)` |
| List all jobs | `list_scheduler_jobs(limit=20)` |
| Trigger manual run | `trigger_scheduler_job(job_id)` only when explicitly requested or troubleshooting |
| View run history | `list_job_runs(job_id)` |
| View run log | `get_run_log(job_id, run_id)` |
| Pause / Resume | `pause_job(job_id)` / `resume_job(job_id)` |
| Update job | `update_job(job_id, sql_file_path, job_name, conn_id)` |
| Delete job | `delete_job(job_id)` |
