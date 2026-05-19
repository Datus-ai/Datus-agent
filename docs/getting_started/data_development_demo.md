# Data Development Demo

This guide is a hands-on demo for building a data mart with Datus skills. It
uses the `product_adoption` dataset, which models Pendo feature usage events
for account-level product adoption analysis. The demo package includes a
business requirements document under `docs/` for the target mart.

## Tutorial Overview

This tutorial demonstrates an end-to-end data development workflow driven by Datus skills. You will start from raw Pendo data, initialize project knowledge from reference SQL, create an ETL implementation plan, review the generated SQL, execute the jobs, and reconcile the final mart against a trusted expected-result table.

The business scenario is product adoption analysis. Product, customer success, and growth teams need to understand how customer accounts use product features across applications. The target mart summarizes feature usage at the account and application level so downstream users can identify trial usage, regular usage, heavy adoption, and low-adoption accounts that may need enablement.

The package includes two inputs for the workflow:

| Input | Purpose |
|---|---|
| `docs/pendo_product_adoption_summary_requirements.md` | Business requirements for the target product adoption summary mart. |
| `ref_sql/` | Historical SQL references used to initialize project knowledge, extract lineage, and ground implementation decisions. |

The main source data is Pendo feature interaction data:

| Source | Purpose |
|---|---|
| `raw.feature_event` | Feature usage events, including visitor, account, application, feature, event count, minutes, and timestamp. |
| `raw.feature_history` | Feature metadata used by reference SQL and project knowledge initialization. |
| `raw.page_history` | Page metadata used by reference SQL and feature enrichment examples. |

The implementation target for this tutorial is:

```text
marts.pendo__product_adoption_summary
```

The target mart grain is:

```text
feature_id + account_id + app_id
```

The mart calculates adoption metrics such as total users, sessions, events, minutes, active days, average usage per session, adoption level, and feature health score.

The expected result table is already stored in DuckDB and loaded into PostgreSQL during initialization:

```text
marts.pendo__product_adoption_summary_expected
```

The goal is to produce `marts.pendo__product_adoption_summary` so that it reconciles exactly with the expected table.

Workflow summary:

| Step | What You Do | Outcome |
|---|---|---|
| 1 | Start PostgreSQL | Local database is available. |
| 2 | Load DuckDB data | `raw` source tables and the `marts` expected table are copied into PostgreSQL. |
| 3 | Start Datus | Datus is ready with configured model and datasource. |
| 4 | Run `project-set-up` | Project knowledge docs are generated from reference SQL. |
| 5 | Run `etl-plan` | An implementation plan is created and approved. |
| 6 | Run `sql-review` | Generated SQL is reviewed and corrected before execution. |
| 7 | Run `execute-job` | Staging and mart jobs are executed. |
| 8 | Run `data-compare` | Final mart is reconciled against the expected table. |

Successful completion means:

- `marts.pendo__product_adoption_summary` exists in PostgreSQL.
- The mart has 24,995 rows.
- All 13 output columns match `marts.pendo__product_adoption_summary_expected`.
- The comparison passes at sub-1e-9 numeric tolerance.
- No further SQL corrections are needed.

## Step 0: Download the Demo Package

Download the package: [product_adoption.zip](../assets/product_adoption.zip).

After extraction, keep this directory structure intact:

```text
product_adoption/
  README.md
  docker-compose.yml
  pendo_start.duckdb
  docs/
    pendo_product_adoption_summary_requirements.md
  docker/
    duckdb-loader/
      Dockerfile
      requirements.txt
      load_duckdb_to_postgres.py
  ref_sql/
    staging/
      stg_pendo__feature_event.sql
      stg_pendo__feature_history.sql
      stg_pendo__page_history.sql
    intermediate/
      int_pendo__latest_feature.sql
      int_pendo__latest_page.sql
      int_pendo__feature_info.sql
      int_pendo__feature_daily_metrics.sql
    marts/
      feature.sql
      feature_event.sql
      feature_daily_metrics.sql
  .datus/
    skills/
```

The `docs/pendo_product_adoption_summary_requirements.md` file is the business
requirement used by the planning step.

## Step 1: Start PostgreSQL

From the `product_adoption` directory, start PostgreSQL:

```bash
cd product_adoption
docker compose up -d postgres
```

PostgreSQL connection values:

| Setting | Value |
|---|---|
| Host | `127.0.0.1` |
| Port | `5432` |
| Database | `pendo` |
| Username | `pendo` |
| Password | `pendo` |
| Default schema | `raw` |

## Step 2: Load DuckDB Data Into PostgreSQL

Run the one-time migration:

```bash
docker compose --profile migration run --rm duckdb-loader
```

The loader copies these DuckDB schemas into PostgreSQL:

```text
raw
marts
```

Expected baseline table after migration:

```text
marts.pendo__product_adoption_summary_expected
```

Expected row count:

```text
24995
```

## Step 3: Start Datus

Start Datus from the same directory:

```bash
datus
```

After Datus opens, configure the model in the Datus interface.

Then configure the datasource with these values:

| Setting | Value |
|---|---|
| Datasource name | `pendo_pg` |
| Type | `PostgreSQL` |
| Host | `127.0.0.1` |
| Port | `5432` |
| Database | `pendo` |
| Username | `pendo` |
| Password | `pendo` |
| Default schema | `raw` |

## Step 4: Initialize Project Knowledge

Use the `project-set-up` skill to initialize the project knowledge base.

Enter this prompt in Datus:

```text
Initialize this project using skill project-set-up
```

Expected output documents:

| Document | Purpose |
|---|---|
| `AGENTS.md` | Project overview, architecture, core asset index, and key decisions. |
| `docs/business_knowledge.md` | Business rules, mandatory filtering, SCD semantics, daily metrics, first-time/return logic, and divide-by-zero guards. |
| `docs/technical_standards.md` | SQL conventions for full reloads, schema bootstrap, timestamp parsing, naming, CTEs, window deduplication, and NULL handling. |
| `docs/table_lineage.md` | DAG and field lineage across the retained staging, intermediate, and mart reference SQL. |
| `docs/ref_sql_inventory.md` | Per-file purpose, source tables, target tables, and SQL evidence. |

Expected analysis scope:

| Layer | Files |
|---|---|
| Staging | `stg_pendo__feature_event`, `stg_pendo__feature_history`, `stg_pendo__page_history` |
| Intermediate | `int_pendo__latest_feature`, `int_pendo__latest_page`, `int_pendo__feature_info`, `int_pendo__feature_daily_metrics` |
| Marts | `feature`, `feature_event`, `feature_daily_metrics` |

Important findings to expect:

1. All reference SQL uses full reloads.
2. The reference SQL uses DuckDB-style syntax.
3. Latest-record logic uses `ROW_NUMBER()` over business keys ordered by `last_updated_at`.
4. Event data is sanitized, while metadata text is mostly pass-through.
5. Daily ratios are rounded to 3 decimals and divide-by-zero returns NULL.
6. Previous-feature sequencing has no tie-breaker when timestamps are equal.

## Step 5: Create the ETL Plan

Use the `etl-plan` skill to create the implementation plan.

Enter this prompt in Datus:

```text
Please create an ETL plan using skill etl-plan
```

Expected plan:

| Item | Detail |
|---|---|
| Plan file | `plans/build_product_adoption_summary.md` |
| Goal | Build `marts.pendo__product_adoption_summary` in PostgreSQL and reconcile it against `marts.pendo__product_adoption_summary_expected`. |
| Expected baseline | `marts.pendo__product_adoption_summary_expected` contains 24,995 rows. |
| Out of scope | `pendo__product_adoption_analytics` is not built in this tutorial. |

Expected planned jobs:

| Job | Purpose |
|---|---|
| `jobs/stg_pendo__feature_event.sql` | Materialize `staging.stg_pendo__feature_event` from `raw.feature_event`. |
| `jobs/pendo__product_adoption_summary.sql` | Build the product adoption summary mart. |

After reviewing the plan, approve implementation with this input:

```text
Approve, start implementation the plan
```

This approval step generates the SQL jobs.

## Step 6: Review the Generated SQL

Use the `sql-review` skill before execution.

Enter this prompt in Datus:

```text
Please review the ETL SQL using skill sql-review
```

Expected review result:

| Item | Expected Status |
|---|---|
| Modified file | `jobs/pendo__product_adoption_summary.sql` |
| Main fix | Add explicit `CASE WHEN avg_events_per_session IS NULL THEN NULL` before `LEAST(...)`. |
| Type check | `feature_health_score` remains `double precision`. |
| Remaining low risks | Timestamp regex guard and missing inline explanation for no `WHERE` filter. |

## Step 7: Execute the SQL Jobs

After review passes, execute the jobs with the `execute-job` skill.

Enter this prompt in Datus:

```text
Please execute the SQL jobs using skill execute-job
```

The `execute-job` skill may use project execution tools such as `gen_table` and `gen_job` when DDL or job generation is required.

Expected generated tables:

```text
staging.stg_pendo__feature_event
marts.pendo__product_adoption_summary
```

## Step 8: Compare Results

After the jobs finish, compare the generated mart with the expected table using the `data-compare` skill.

Enter this prompt in Datus:

```text
Please compare the job result with the expected table using skill data-compare
```

Expected comparison result:

```text
marts.pendo__product_adoption_summary reconciles perfectly with marts.pendo__product_adoption_summary_expected.
```

Successful validation means:

- 24,995 rows match.
- All 13 columns match.
- Numeric comparison passes at sub-1e-9 tolerance.
- Bidirectional `EXCEPT` checks pass.
- No SQL corrections are needed.

## Skill Reference

The project includes Datus skills under:

```text
.datus/skills/
```

| Skill | Use |
|---|---|
| `project-set-up` | Initialize project knowledge from SQL, docs, lineage, and business rules. |
| `etl-plan` | Create and confirm an implementation plan before generating SQL. |
| `sql-review` | Review generated ETL SQL against the approved plan. |
| `execute-job` | Execute SQL jobs and DDL-oriented table/job operations. |
| `data-compare` | Compare generated results against expected data and explain any differences. |

## Daily Startup

After the environment has already been initialized, start PostgreSQL and Datus with:

```bash
cd product_adoption
docker compose up -d postgres
datus
```

The DuckDB file is only needed for initialization or a full rebuild.
