---
name: scheduler-workflow
description: Workflows for submitting, monitoring, and troubleshooting Airflow scheduled jobs
tags:
  - scheduler
  - airflow
  - workflow
version: 1.0.0
user_invocable: true
---

# Scheduler Workflow

Standard operating procedures for managing scheduled jobs via the Datus scheduler tools.
Load this skill when you need to perform multi-step scheduler operations.

For single-step operations (check status, list jobs, pause/resume), call the scheduler tools directly without loading this skill.

## Workflow 1: Submit a SQL Job

Use when the user wants to schedule a SQL query for periodic execution.

### Steps

1. **Validate SQL first** — run `read_query()` against the target namespace to confirm the SQL executes correctly
2. **Save SQL to file** — `write_file(path="jobs/<job_name>.sql", content=<sql>)`
3. **Submit the job**:
   - Standard SQL: `submit_sql_job(job_name=..., sql_file_path=..., conn_id=..., schedule=..., description=...)`
   - SparkSQL: `submit_sparksql_job(job_name=..., sql_file_path=..., schedule=..., description=...)`
4. **Verify submission** — `get_scheduler_job(job_id)` to confirm status is active

### DB connection (`conn_id`)

`submit_sql_job` and `update_job` require `conn_id` — the Airflow Connection ID for the target database.
The connection is managed entirely by Airflow (Admin > Connections) and resolved at runtime by the scheduler worker.

Available conn_id values are shown in the `submit_sql_job` and `update_job` tool descriptions (from `scheduler.connections` in agent.yml).

### Naming conventions

- `job_name`: `<frequency>_<domain>_<description>`, e.g. `daily_sales_summary`, `hourly_order_count`
- SQL file: `jobs/<job_name>.sql`

### Common cron expressions

| Schedule | Cron |
|----------|------|
| Every day at 8am | `0 8 * * *` |
| Every hour | `0 * * * *` |
| Every 2 hours | `0 */2 * * *` |
| Monday at 9am | `0 9 * * 1` |
| 1st of month at midnight | `0 0 1 * *` |

## Workflow 2: Troubleshoot a Failed Job

Use when the user reports a job failure or wants to diagnose issues.

### Steps

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

## Workflow 3: Update an Existing Job

Use when the user wants to modify a scheduled job's SQL or configuration.

### Steps

1. **Check current state** — `get_scheduler_job(job_id)` to see existing config
2. **Pause the job** — `pause_job(job_id)` to prevent runs during update
3. **Validate new SQL** — `read_query()` to test the updated SQL
4. **Save and update** — write new SQL file, then `update_job(job_id, sql_file_path=..., job_name=..., conn_id=...)`
5. **Resume and verify** — `resume_job(job_id)`, optionally `trigger_scheduler_job(job_id)` to test

## Quick Reference

| Goal | Tool |
|------|------|
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
