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
5. **Resume and verify** — `resume_job(job_id)`, then `trigger_scheduler_job(job_id)` to test

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
| Trigger manual run | `trigger_scheduler_job(job_id)` |
| View run history | `list_job_runs(job_id)` |
| View run log | `get_run_log(job_id, run_id)` |
| Pause / Resume | `pause_job(job_id)` / `resume_job(job_id)` |
| Update job | `update_job(job_id, sql_file_path, job_name, conn_id)` |
| Delete job | `delete_job(job_id)` |
