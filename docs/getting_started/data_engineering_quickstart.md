# Data Engineering Quickstart

This guide walks through a complete Datus workflow using the open DAComp data-engineering dataset. You will inspect the warehouse design, build layered tables interactively, generate ETL jobs, produce marts data, submit a daily Airflow job, and publish the result to Superset.

## Step 0: Download DAComp First

DAComp is **not bundled** with `datus-agent`. Download the dataset from Hugging Face first:

- <https://huggingface.co/datasets/DAComp/dacomp-de>

This tutorial uses the `dacomp-de-impl-001` example from that dataset.

If you installed Datus only from PyPI, also clone the Datus repo so you can use the local helper compose files shipped in `quickstart/data_engineering/`:

```bash
git clone https://github.com/Datus-ai/Datus-agent.git
export DATUS_AGENT_REPO=/absolute/path/to/Datus-agent
```

Point `DACOMP_HOME` at the extracted example directory and create a writable DuckDB workbench:

```bash
export DACOMP_HOME=/absolute/path/to/dacomp-de-impl-001
cp "$DACOMP_HOME/lever_start.duckdb" "$DACOMP_HOME/lever_workbench.duckdb"
cd "$DACOMP_HOME"
```

The rest of this guide assumes the example directory contains:

- `docs/data_contract.yaml`
- `config/layer_dependencies.yaml`
- `sql/staging/`
- `sql/intermediate/`
- `sql/marts/`
- `lever_start.duckdb`

## Step 1: Understand the Warehouse Layers

The DAComp example already encodes a classic warehouse layout:

| Layer | Tables | Purpose |
|---|---:|---|
| `staging` | 24 | Clean raw ATS records and normalize types and formats |
| `intermediate` | 17 | Join entities and apply reusable business logic |
| `marts` | 14 | Publish analytics-ready outputs for dashboards and metrics |

The two files that drive the design are:

- `docs/data_contract.yaml` - row-level cleanup, validation, and normalization rules
- `config/layer_dependencies.yaml` - layer order and table dependencies

Read those first so the prompts you give to `gen_table` and `gen_job` stay aligned with the intended warehouse design.

## Step 2: Start the Local Quickstart Stack

The repo now includes the two local demo environments used by this walkthrough under `quickstart/data_engineering/`.

Start Superset:

```bash
cd "$DATUS_AGENT_REPO/quickstart/data_engineering/superset"
export SUPERSET_DB_PASSWORD='superset'
export SUPERSET_SECRET_KEY='datus-test-secret-key-not-for-prod'
export SUPERSET_ADMIN_PASSWORD='admin'
docker compose up -d
```

Start Airflow:

```bash
cd "$DATUS_AGENT_REPO/quickstart/data_engineering/airflow"
docker compose up -d
```

Default local endpoints:

- Superset: `http://127.0.0.1:8088`, username `admin`, password `admin`
- Airflow: `http://127.0.0.1:8080`, username `admin`, password `admin`

For this quickstart, the Superset compose file reads its database password, admin password, and secret key from environment variables. The three `export` commands above intentionally use local demo values so the walkthrough stays copy-pasteable.

The Airflow compose file mounts `${DACOMP_HOME}` into the container and preloads an Airflow connection named `duckdb_dacomp_lever`, which points to `/workspace/lever_workbench.duckdb`.

## Step 3: Install the Required Packages

Install Datus plus the adapters used in this walkthrough:

```bash
pip install datus-bi-superset datus-postgresql datus-semantic-metricflow datus-scheduler-airflow
```

Package roles in this quickstart:

- `datus-bi-superset` - Superset BI adapter used by `gen_dashboard`
- `datus-postgresql` - PostgreSQL driver support for the Superset materialization target
- `datus-semantic-metricflow` - semantic model and metric generation
- `datus-scheduler-airflow` - Airflow scheduler adapter; it installs `datus-scheduler-core` transitively

## Step 4: Configure `agent.yml`

Add the following minimum configuration to your `~/.datus/conf/agent.yml`:

```yaml
agent:
  services:
    databases:
      lever_duckdb:
        type: duckdb
        uri: "duckdb:///${DACOMP_HOME}/lever_workbench.duckdb"
        default: true

    semantic_layer:
      metricflow: {}

    bi_tools:
      superset:
        type: superset
        api_url: http://127.0.0.1:8088
        username: admin
        password: admin
        dataset_db:
          uri: "postgresql+psycopg2://superset:superset@127.0.0.1:5433/superset_examples"
          schema: public

    schedulers:
      airflow_prod:
        type: airflow
        api_base_url: http://127.0.0.1:8080/api/v1
        username: admin
        password: admin
        dags_folder: "${DATUS_AGENT_REPO}/quickstart/data_engineering/airflow/dags"
        connections:
          duckdb_dacomp_lever: DAComp Lever DuckDB

  agentic_nodes:
    gen_semantic_model:
      semantic_adapter: metricflow
    gen_metrics:
      semantic_adapter: metricflow
    gen_dashboard:
      bi_platform: superset
    scheduler:
      scheduler_service: airflow_prod
```

Then start Datus on the workbench database:

```bash
cd "$DACOMP_HOME"
datus-cli --database lever_duckdb
```

Here `dags_folder` is the host-side directory where Datus writes generated DAG files. `quickstart/data_engineering/airflow/docker-compose.yml` mounts that directory into the Airflow container as `/opt/airflow/dags`, so newly generated DAGs are picked up automatically.

## Step 5: Create the Layered Warehouse Interactively

Use `gen_table` when you want explicit DDL review before execution. The agent will show the DDL and wait for confirmation.

Create the target schemas:

```text
/gen_table Create schemas staging, intermediate, and marts in the current DuckDB database. Keep the existing raw schema unchanged.
```

Create a representative staging table from the contract:

```text
/gen_table Read ./docs/data_contract.yaml and create staging.stg_lever__requisition from raw.requisition. Apply the validation and normalization rules from the contract and show me the DDL before execution.
```

Use the same pattern for the other staging tables you want to inspect manually. For large batches, let `gen_job` take over.

## Step 6: Generate ETL Jobs and Produce Marts Data

Use `gen_job` to operationalize the SQL assets from the DAComp example inside the current DuckDB workbench.

Build a representative intermediate table:

```text
/gen_job Create intermediate.int_lever__requisition_users in the current database using ./sql/intermediate/int_lever__requisition_users.sql.
```

Then build a marts table that is ready for downstream analytics:

```text
/gen_job Create marts.lever__recruitment_analytics_dashboard in the current database using ./sql/marts/lever__recruitment_analytics_dashboard.sql.
```

The intended order is always:

```text
staging -> intermediate -> marts
```

After the marts table is built, validate it directly:

```sql
SELECT COUNT(*) FROM marts.lever__recruitment_analytics_dashboard;
```

## Step 7: Generate Semantic Assets for the Marts Layer

Now turn the marts output into reusable semantic assets.

Generate a semantic model:

```text
/gen_semantic_model Generate a semantic model for marts.lever__recruitment_analytics_dashboard.
```

Generate metrics from a dashboard-oriented KPI prompt:

```text
/gen_metrics Create metrics for open requisitions, applications, interviews, offers, and hires from marts.lever__recruitment_analytics_dashboard. subject_tree: recruiting/hiring/dashboard
```

These files are typically written under your project semantic-model storage and become reusable context for future analysis.

## Step 8: Submit a Daily Airflow Job

Use the scheduler subagent to operationalize a daily marts refresh. The Airflow quickstart environment already exposes the `duckdb_dacomp_lever` connection.

Submit a daily SQL job at 8 AM:

```text
/scheduler Submit a daily SQL job named daily_lever_recruitment_dashboard from ./sql/marts/lever__recruitment_analytics_dashboard.sql at 8am every day using the duckdb_dacomp_lever connection
```

Then trigger it once for validation:

```text
/scheduler Trigger daily_lever_recruitment_dashboard once now and show me the latest run status
```

What to expect:

- a DAG file appears under `${DATUS_AGENT_REPO}/quickstart/data_engineering/airflow/dags`
- the same file is visible inside the Airflow container as `/opt/airflow/dags/<dag_id>.py`
- Airflow returns a `job_id`
- the job becomes visible in the Airflow UI

## Step 9: Write Into Superset and Create a Dashboard

Once the marts table exists, hand it to the BI subagent.

```text
/gen_dashboard Create a recruiting operations dashboard in Superset from marts.lever__recruitment_analytics_dashboard. Include KPI tiles for requisitions, applications, interviews, offers, and hires, plus a funnel chart and a weekly trend chart.
```

Under the hood, the Superset workflow is:

```text
write_query -> create_dataset -> create_chart -> create_dashboard -> add_chart_to_dashboard
```

Because `bi_tools.superset.dataset_db` points to the local Superset PostgreSQL service, Datus can materialize the query result and register it as a Superset dataset automatically.

## Step 10: Verify the End-to-End Result

You should now have:

- `staging`, `intermediate`, and `marts` schemas in `lever_workbench.duckdb`
- at least one marts table that queries successfully
- generated semantic model and metric YAML
- a daily Airflow job visible in the scheduler UI
- a Superset dashboard URL returned by `gen_dashboard`

For deeper reference, continue with:

- [Database Adapters](../adapters/db_adapters.md)
- [Semantic Adapters](../adapters/semantic_adapters.md)
- [Table Generation](../subagent/gen_table.md)
- [ETL Job](../subagent/gen_job.md)
- [Scheduler](../subagent/scheduler.md)
- [Generate Dashboard](../subagent/gen_dashboard.md)
