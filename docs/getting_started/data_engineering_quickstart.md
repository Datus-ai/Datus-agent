# Data Engineering Quickstart

This guide walks through a complete Datus workflow using the open DAComp data-engineering dataset. You will inspect the warehouse design, build layered tables interactively, generate ETL jobs, produce marts data, submit a daily Airflow job, and publish the result to Superset.

## Step 0: Download the Quickstart Data

DAComp is **not bundled** with `datus-agent`. This tutorial uses a small
quickstart package derived from the DAComp Lever example, so you do not need to
download the full DAComp archive.

First create and enter the working directory:

```bash
mkdir -p ~/datus-quickstart-data
cd ~/datus-quickstart-data
```

Run the bash block below — it downloads and unpacks the quickstart data and
local Docker stack, exports `DACOMP_HOME` / `DATUS_QUICKSTART_STACK`, and
finally prints the two `export` statements so you can paste them into another
shell:

```bash
curl -L -o datus-de-lever-quickstart-v2.zip \
  https://github.com/Datus-ai/datus-quickstart-data/releases/download/data-engineering-v2/datus-de-lever-quickstart-v2.zip
curl -L -o datus-data-engineering-quickstart-stack-v2.zip \
  https://github.com/Datus-ai/datus-quickstart-data/releases/download/data-engineering-v2/datus-data-engineering-quickstart-stack-v2.zip

unzip -o datus-de-lever-quickstart-v2.zip
unzip -o datus-data-engineering-quickstart-stack-v2.zip

export DACOMP_HOME="$(pwd)/datus-de-lever-quickstart"
export DATUS_QUICKSTART_STACK="$(pwd)/datus-data-engineering-quickstart-stack"
cd "$DACOMP_HOME"

echo "export DACOMP_HOME=$DACOMP_HOME"
echo "export DATUS_QUICKSTART_STACK=$DATUS_QUICKSTART_STACK"
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

The downloaded stack includes the local demo services used by this walkthrough.

Start Superset:

```bash
cd "$DATUS_QUICKSTART_STACK/superset"
docker compose up -d
```

Start the local lakehouse stack used by both Datus and Airflow:

```bash
cd "$DATUS_QUICKSTART_STACK/lakehouse"
docker compose up -d
docker logs datus-quickstart-lakehouse-seed
```

The seed log should end with:

```text
Seeded lake.demo_raw tables: requisition, user, requisition_posting, requisition_offer
```

Start Airflow after the lakehouse network exists:

```bash
cd "$DATUS_QUICKSTART_STACK/airflow"
docker compose up -d
```

Default local endpoints:

- Superset: `http://127.0.0.1:8088`, username `admin`, password `admin`
- Airflow: `http://127.0.0.1:8080`, username `admin`, password `admin`
- MinIO Console: `http://127.0.0.1:9001`, username `admin`, password `password`
- Iceberg REST catalog: `http://127.0.0.1:8181`

For this quickstart, the Superset compose file uses local demo defaults for the
metadata database and admin user.

The lakehouse compose file seeds the required Lever source tables into the
shared demo raw namespace `lake.demo_raw`. Airflow exposes a shared connection named
`lakehouse_demo`; scheduled jobs use in-memory DuckDB attached to the same
Iceberg REST catalog, so Datus and Airflow no longer compete for a local
`.duckdb` file lock.

In production or SaaS deployments, initialize `lake.demo_raw` with a system account
that can write raw data, and run Datus/Airflow with a separate account that can
read `lake.demo_raw` and write non-raw workspace namespaces such as
`lake.ws_lever_demo`. Datus does not enforce namespace-specific rules itself;
the Iceberg catalog and storage permissions are the boundary.

If you need to reset the raw demo tables, rerun only the seed service:

```bash
cd "$DATUS_QUICKSTART_STACK/lakehouse"
docker compose up -d --force-recreate seed-demo-raw
docker logs datus-quickstart-lakehouse-seed
```

## Step 3: Configure `agent.yml`

Merge the following service configuration into the existing `agent:` section in
`~/.datus/conf/agent.yml`. Keep any existing `agent.providers` settings; the
`/model` command uses those credentials. The paths use the `DACOMP_HOME` and `DATUS_QUICKSTART_STACK`
environment variables from Step 0.

```yaml
agent:
  project_name: lever_demo

  services:
    datasources:
      demo_lakehouse:
        type: duckdb
        uri: "duckdb:///:memory:"
        default: true
        iceberg:
          catalog_alias: lake
          catalog_uri: http://127.0.0.1:8181
          warehouse: s3://warehouse/
          s3_region: us-east-1
          s3_endpoint: http://127.0.0.1:9000
          s3_access_key_id: admin
          s3_secret_access_key: password
          s3_url_style: path
          authorization_type: none
          access_delegation_mode: none
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
          lakehouse_demo:
            description: Shared DuckDB Iceberg demo lakehouse
            type: duckdb_iceberg
            default: true
            capabilities:
              - sql
              - lakehouse

    semantic_layer:
      metricflow:
        type: metricflow

  agentic_nodes:
    gen_dashboard:
      bi_platform: superset
    scheduler:
      scheduler_service: airflow_prod
```

The YAML above is for the local demo stack, whose Iceberg REST catalog does not
authenticate. For a shared public catalog, use the runtime account credentials
instead, for example `client_id`, `client_secret`, `oauth2_server_uri`, and
`access_delegation_mode: vended_credentials`. Configure the Airflow connection
with the same runtime account; reserve the system/admin account for raw-data
initialization outside Datus.

Then start Datus with the `demo_lakehouse` datasource:

```bash
cd "$DACOMP_HOME"
datus-cli --datasource demo_lakehouse
```

If this workspace was used with an older version of the quickstart, update
`./.datus/config.yml` before starting Datus:

```yaml
default_datasource: demo_lakehouse
```

If the CLI says no model is configured, configure one before continuing:

```text
/model
```

Choose a provider/model and enter credentials if prompted. `/model` writes
provider credentials under `agent.providers` in `~/.datus/conf/agent.yml` and
writes the active provider/model for this project to `./.datus/config.yml`.

Here `dags_folder` is the host-side directory where Datus writes generated DAG files. The Airflow compose file mounts that directory into the Airflow container as `/opt/airflow/dags`, so newly generated DAGs are picked up automatically.

Before continuing, verify that Datus can read the seeded lakehouse data:

```sql
SELECT COUNT(*) FROM lake.demo_raw.requisition;
```

The quickstart data should return `200` rows for `requisition`.

## Step 4: Create the Required Staging Tables

For natural-language agent tasks, avoid starting the message with a raw SQL verb
such as `CREATE` or `COPY`; the CLI uses those leading keywords to detect direct
SQL.

This quickstart uses:

- shared demo source namespace: `lake.demo_raw`
- current workspace output namespace: `lake.ws_lever_demo`

Ask the agent to create the workspace output schema:

```text
Please set up the current workspace output schema lake.ws_lever_demo. Treat lake.demo_raw as read-only source data.
```

This walkthrough builds a narrow but complete dependency chain for
`marts_lever__requisition_enhanced`. Use `docs/data_contract.yaml` as the source
of truth for field selection, renames, and business logic.

Ask the agent to create the staging tables required by the `source_models`
listed for `lever__requisition_enhanced` and
`intermediate.int_lever__requisition_users`. The agent will route the request to
the table-generation workflow:

```text
Read ./docs/data_contract.yaml and create the staging tables needed for marts_lever__requisition_enhanced in lake.ws_lever_demo: stg_lever__requisition from lake.demo_raw.requisition, stg_lever__user from lake.demo_raw.user, stg_lever__requisition_posting from lake.demo_raw.requisition_posting, and stg_lever__requisition_offer from lake.demo_raw.requisition_offer. Use the field design and source-to-target mapping from the contract.
```

These four staging tables are the minimum raw-to-staging inputs for the
requisition-enhancement example.

## Step 5: Build the Intermediate and Marts Tables

Build the intermediate model first. It should combine requisition fields with
user fields according to the `int_lever__requisition_users` entry in
`docs/data_contract.yaml`.

Create the intermediate table:

```text
Read ./docs/data_contract.yaml and create lake.ws_lever_demo.int_lever__requisition_users from lake.ws_lever_demo.stg_lever__requisition and lake.ws_lever_demo.stg_lever__user. Use the contract's field design, joins, and source-to-target mapping.
```

Then create the marts table that is ready for downstream analytics. The contract
defines `marts_lever__requisition_enhanced` as one row per `requisition_id`,
using:

- `lake.ws_lever_demo.int_lever__requisition_users`
- `lake.ws_lever_demo.stg_lever__requisition_posting`
- `lake.ws_lever_demo.stg_lever__requisition_offer`

Create the marts table:

```text
Read ./docs/data_contract.yaml and create lake.ws_lever_demo.marts_lever__requisition_enhanced from lake.ws_lever_demo.int_lever__requisition_users, lake.ws_lever_demo.stg_lever__requisition_posting, and lake.ws_lever_demo.stg_lever__requisition_offer. Use the contract's business logic: keep all base requisition rows, count posting and offer links by requisition_id, fill missing counts with 0, and add has_posting and has_offer flags.
```

The intended order is always:

```text
staging -> intermediate -> marts
```

After the marts table is built, validate it directly:

```sql
SELECT COUNT(*) FROM lake.ws_lever_demo.marts_lever__requisition_enhanced;
```

## Step 6: Submit a Daily Airflow Job

Ask the agent to operationalize a daily marts refresh. The Airflow quickstart environment already exposes the `lakehouse_demo` connection.

Submit a daily SQL job at 8 AM that rebuilds the same contract-derived chain:

```text
Submit a daily SQL job named daily_lever_requisition_enhanced that refreshes lake.ws_lever_demo.stg_lever__requisition, lake.ws_lever_demo.stg_lever__user, lake.ws_lever_demo.stg_lever__requisition_posting, lake.ws_lever_demo.stg_lever__requisition_offer, lake.ws_lever_demo.int_lever__requisition_users, and lake.ws_lever_demo.marts_lever__requisition_enhanced at 8am every day using the lakehouse_demo connection. Use the SQL generated and validated from docs/data_contract.yaml in the previous steps.
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

## Step 7: Promote the Marts Table to the Superset Serving DB

The marts table above was built through the `demo_lakehouse` datasource. Before
dashboard generation can create Superset assets, copy that table into the
BI-registered `superset_serving` Postgres datasource referenced by
`dataset_db.datasource_ref`. These names are Datus datasource names from
`agent.yml`, not physical database or catalog names inside DuckDB or Postgres.

```text
Please copy the source table lake.ws_lever_demo.marts_lever__requisition_enhanced from the demo_lakehouse datasource into the superset_serving datasource as public.lever__requisition_enhanced, replacing the target table if it already exists. Then verify the source and target row counts.
```

The transfer tool creates `public.lever__requisition_enhanced` from the source
result columns if it does not already exist.

After this step, the table exists in the same database Superset knows as
`bi_database_name: examples`.

## Step 8: Create a Superset Dashboard

Once the marts table exists in `superset_serving`, ask the agent to build the dashboard.

```text
Please create a requisition operations dashboard in Superset from public.lever__requisition_enhanced. Include KPI tiles for total requisitions, open requisitions, requisitions with postings, requisitions with offers, and total requested headcount. Add charts by status, team, location, employment_status, count_postings, and count_offers.
```

Data preparation is a separate ETL / scheduler step. Dashboard generation
expects the table or SQL dataset to already be available in the BI-registered
database.

## Step 9: Verify the End-to-End Result

You should now have:

- `lake.demo_raw` seeded as the shared demo source namespace
- `lake.ws_lever_demo.marts_lever__requisition_enhanced` built from raw data through staging and intermediate tables
- a daily Airflow job visible in the scheduler UI
- a Superset dashboard URL returned by the dashboard generation flow
