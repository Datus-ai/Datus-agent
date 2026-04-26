# Data Engineering Quickstart

This guide walks through a complete Datus workflow using the open DAComp data-engineering dataset. You will inspect the warehouse design, build layered tables interactively, generate ETL jobs, produce marts data, submit a daily Airflow job, and publish the result to Superset.

## Step 0: Download the Quickstart Data

DAComp is **not bundled** with `datus-agent`. This tutorial uses a small
quickstart package derived from the DAComp Lever example, so you do not need to
download the full DAComp archive.

Download and unpack the quickstart data and local Docker stack:

```bash
mkdir -p ~/datus-quickstart-data
cd ~/datus-quickstart-data

curl -L -o datus-de-lever-quickstart-v1.zip \
  https://github.com/Datus-ai/datus-quickstart-data/releases/download/data-engineering-v1/datus-de-lever-quickstart-v1.zip
curl -L -o datus-data-engineering-quickstart-stack-v1.zip \
  https://github.com/Datus-ai/datus-quickstart-data/releases/download/data-engineering-v1/datus-data-engineering-quickstart-stack-v1.zip

unzip datus-de-lever-quickstart-v1.zip
unzip datus-data-engineering-quickstart-stack-v1.zip
```

Point `DACOMP_HOME` at the extracted data directory, point
`DATUS_QUICKSTART_STACK` at the extracted Docker stack, and create a writable
DuckDB workbench:

```bash
export DACOMP_HOME=/absolute/path/to/datus-de-lever-quickstart
export DATUS_QUICKSTART_STACK=/absolute/path/to/datus-data-engineering-quickstart-stack
cp "$DACOMP_HOME/lever_start.duckdb" "$DACOMP_HOME/lever_workbench.duckdb"
cd "$DACOMP_HOME"
```

The rest of this guide assumes the example directory contains:

- `docs/data_contract.yaml`
- `config/layer_dependencies.yaml`
- `lever_start.duckdb`

## Step 1: Understand the Warehouse Layers

The DAComp example already encodes a classic warehouse layout:

| Layer | Tables | Purpose |
|---|---:|---|
| `staging` | 24 | Clean raw ATS records and normalize types and formats |
| `intermediate` | 17 | Join entities and apply reusable business logic |
| `marts` | 14 | Publish analytics-ready outputs for dashboards and reporting |

The two files that drive the design are:

- `docs/data_contract.yaml` - row-level cleanup, validation, and normalization rules
- `config/layer_dependencies.yaml` - layer order and table dependencies

Read those first so the prompts you give to the agent stay aligned with the intended warehouse design.

## Step 2: Start the Local Quickstart Stack

The downloaded stack includes the two local demo environments used by this walkthrough.

Start Superset:

```bash
cd "$DATUS_QUICKSTART_STACK/superset"
docker compose up -d
```

Start Airflow:

```bash
cd "$DATUS_QUICKSTART_STACK/airflow"
docker compose up -d
```

Default local endpoints:

- Superset: `http://127.0.0.1:8088`, username `admin`, password `admin`
- Airflow: `http://127.0.0.1:8080`, username `admin`, password `admin`

For this quickstart, the Superset compose file uses local demo defaults for the
metadata database and admin user.

The Airflow compose file mounts `${DACOMP_HOME}` into the container and exposes
an Airflow connection named `duckdb_dacomp_lever`, which points to
`/workspace/lever_workbench.duckdb`.

## Step 3: Install the Required Packages

Install Datus plus the adapters used in this walkthrough:

```bash
pip install datus-agent datus-bi-superset datus-postgresql datus-scheduler-airflow
```

## Step 4: Configure `agent.yml`

Merge the following service configuration into the existing `agent:` section in
`~/.datus/conf/agent.yml`. Keep any existing `agent.providers` settings; the
`/model` command uses those credentials. The paths use the `DACOMP_HOME` and `DATUS_QUICKSTART_STACK`
environment variables from Step 0.

```yaml
agent:
  services:
    datasources:
      lever_duckdb:
        type: duckdb
        uri: "duckdb:///${DACOMP_HOME}/lever_workbench.duckdb"
        default: true
      superset_serving:
        type: postgresql
        host: 127.0.0.1
        port: 5433
        database: superset_examples
        schema: public
        username: superset
        password: superset

    bi_platforms:
      superset:
        type: superset
        api_base_url: http://127.0.0.1:8088
        username: admin
        password: admin
        dataset_db:
          datasource_ref: superset_serving
          bi_database_name: examples

    schedulers:
      airflow_prod:
        type: airflow
        api_base_url: http://127.0.0.1:8080/api/v1
        username: admin
        password: admin
        dags_folder: "${DATUS_QUICKSTART_STACK}/airflow/dags"
        connections:
          duckdb_dacomp_lever: DAComp Lever DuckDB

  agentic_nodes:
    gen_dashboard:
      bi_platform: superset
    scheduler:
      scheduler_service: airflow_prod
```

Then start Datus with the `lever_duckdb` datasource, which points at the
workbench DuckDB file:

```bash
cd "$DACOMP_HOME"
datus-cli --datasource lever_duckdb
```

If the CLI says no model is configured, configure one before continuing:

```text
/model
```

Choose a provider/model and enter credentials if prompted. `/model` writes
provider credentials under `agent.providers` in `~/.datus/conf/agent.yml` and
writes the active provider/model for this project to `./.datus/config.yml`.

Here `dags_folder` is the host-side directory where Datus writes generated DAG files. The Airflow compose file mounts that directory into the Airflow container as `/opt/airflow/dags`, so newly generated DAGs are picked up automatically.

## Step 5: Create the Required Staging Tables

For natural-language agent tasks, avoid starting the message with a raw SQL verb
such as `CREATE` or `COPY`; the CLI uses those leading keywords to detect direct
SQL.

Ask the agent to create the target schemas:

```text
Please set up the target schemas staging, intermediate, and marts in the current DuckDB database. Keep the existing raw schema unchanged.
```

This walkthrough builds a narrow but complete dependency chain for
`marts.lever__requisition_enhanced`. Use `docs/data_contract.yaml` as the source
of truth for field selection, renames, and business logic.

Ask the agent to create the staging tables required by the `source_models`
listed for `lever__requisition_enhanced` and
`intermediate.int_lever__requisition_users`. The agent will route the request to
the table-generation workflow:

```text
Read ./docs/data_contract.yaml and create the staging tables needed for marts.lever__requisition_enhanced: staging.stg_lever__requisition from raw.requisition, staging.stg_lever__user from raw.user, staging.stg_lever__requisition_posting from raw.requisition_posting, and staging.stg_lever__requisition_offer from raw.requisition_offer. Use the field design and source-to-target mapping from the contract.
```

These four staging tables are the minimum raw-to-staging inputs for the
requisition-enhancement example.

## Step 6: Build the Intermediate and Marts Tables

Build the intermediate model first. It should combine requisition fields with
user fields according to the `int_lever__requisition_users` entry in
`docs/data_contract.yaml`.

Create the intermediate table:

```text
Read ./docs/data_contract.yaml and create intermediate.int_lever__requisition_users from staging.stg_lever__requisition and staging.stg_lever__user. Use the contract's field design, joins, and source-to-target mapping.
```

Then create the marts table that is ready for downstream analytics. The contract
defines `marts.lever__requisition_enhanced` as one row per `requisition_id`,
using:

- `intermediate.int_lever__requisition_users`
- `staging.stg_lever__requisition_posting`
- `staging.stg_lever__requisition_offer`

Create the marts table:

```text
Read ./docs/data_contract.yaml and create marts.lever__requisition_enhanced from intermediate.int_lever__requisition_users, staging.stg_lever__requisition_posting, and staging.stg_lever__requisition_offer. Use the contract's business logic: keep all base requisition rows, count posting and offer links by requisition_id, fill missing counts with 0, and add has_posting and has_offer flags.
```

The intended order is always:

```text
staging -> intermediate -> marts
```

After the marts table is built, validate it directly:

```sql
SELECT COUNT(*) FROM marts.lever__requisition_enhanced;
```

## Step 7: Submit a Daily Airflow Job

Ask the agent to operationalize a daily marts refresh. The Airflow quickstart environment already exposes the `duckdb_dacomp_lever` connection.

Submit a daily SQL job at 8 AM that rebuilds the same contract-derived chain:

```text
Submit a daily SQL job named daily_lever_requisition_enhanced that refreshes staging.stg_lever__requisition, staging.stg_lever__user, staging.stg_lever__requisition_posting, staging.stg_lever__requisition_offer, intermediate.int_lever__requisition_users, and marts.lever__requisition_enhanced at 8am every day using the duckdb_dacomp_lever connection. Use the SQL generated and validated from docs/data_contract.yaml in the previous steps.
```

Then trigger it once for validation:

```text
Trigger daily_lever_requisition_enhanced once now and show me the latest run status
```

What to expect:

- a DAG file appears under `${DATUS_QUICKSTART_STACK}/airflow/dags`
- the same file is visible inside the Airflow container as `/opt/airflow/dags/<dag_id>.py`
- Airflow returns a `job_id`
- the job becomes visible in the Airflow UI

## Step 8: Promote the Marts Table to the Superset Serving DB

The marts table above was built through the `lever_duckdb` datasource. Before
dashboard generation can create Superset assets, copy that table into the
BI-registered `superset_serving` Postgres datasource referenced by
`dataset_db.datasource_ref`. These names are Datus datasource names from
`agent.yml`, not physical database or catalog names inside DuckDB or Postgres.

```text
Please copy the source table marts.lever__requisition_enhanced from the lever_duckdb datasource into the superset_serving datasource as public.lever__requisition_enhanced, replacing the target table if it already exists. Then verify the source and target row counts.
```

After this step, the table exists in the same database Superset knows as
`bi_database_name: examples`.

## Step 9: Create a Superset Dashboard

Once the marts table exists in `superset_serving`, ask the agent to build the dashboard.

```text
Please create a requisition operations dashboard in Superset from public.lever__requisition_enhanced. Include KPI tiles for total requisitions, open requisitions, requisitions with postings, requisitions with offers, and total requested headcount. Add charts by status, team, location, employment_status, count_postings, and count_offers.
```

Data preparation is a separate ETL / scheduler step. Dashboard generation
expects the table or SQL dataset to already be available in the BI-registered
database.

## Step 10: Verify the End-to-End Result

You should now have:

- `staging`, `intermediate`, and `marts` schemas in `lever_workbench.duckdb`
- `marts.lever__requisition_enhanced` built from raw data through staging and intermediate layers
- a daily Airflow job visible in the scheduler UI
- a Superset dashboard URL returned by the dashboard generation flow
