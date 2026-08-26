# End-to-End Data Engineering

This scenario tutorial walks through a complete local Datus workflow using the open DAComp
data-engineering dataset. You will inspect the warehouse design, build layered
tables interactively in a local DuckDB workbench file, generate ETL jobs,
produce marts data, submit a daily Airflow job, and publish the result to
Superset.

!!! info "What this tutorial starts from"
    This guide starts with source data and creates a new pipeline and dashboard. If you already have a Superset dashboard and want to turn it into analysis agents, follow [Turn a Dashboard into a Copilot](dashboard_copilot.md). If this is your first time using Datus, complete [Install and First Query](Quickstart.md) first.

This guide uses Datus plugins to work with Airflow and Superset. Datus
datasources handle SQL execution and data movement, while the plugins discover,
create, run, and inspect resources through the Airflow and Superset APIs.

The local open-source quickstart does **not** require Iceberg, MinIO, or S3.
The SaaS Studio tour uses a managed DuckDB + Iceberg lakehouse instead; see
[SaaS Studio Tour Variant](#saas-studio-tour-variant) for the namespace model.

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
local Docker stack, creates a writable DuckDB workbench, exports `DACOMP_HOME`
/ `DATUS_QUICKSTART_STACK`, and finally prints the two `export` statements so
you can paste them into another shell:

```bash
curl -L -o datus-de-lever-quickstart-v1.zip \
  https://github.com/Datus-ai/datus-quickstart-data/releases/download/data-engineering-v1/datus-de-lever-quickstart-v1.zip
curl -L -o datus-data-engineering-quickstart-stack-v1.zip \
  https://github.com/Datus-ai/datus-quickstart-data/releases/download/data-engineering-v1/datus-data-engineering-quickstart-stack-v1.zip

unzip -o datus-de-lever-quickstart-v1.zip
unzip -o datus-data-engineering-quickstart-stack-v1.zip

export DACOMP_HOME="$(pwd)/datus-de-lever-quickstart"
export DATUS_QUICKSTART_STACK="$(pwd)/datus-data-engineering-quickstart-stack"
cp "$DACOMP_HOME/lever_start.duckdb" "$DACOMP_HOME/lever_workbench.duckdb"
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

The Superset Database named `examples` connects to PostgreSQL as
`postgres:5432/superset_examples`. The Superset plugin uses that
credential-free connection identity to find the matching Datus datasource.
Expose the endpoint to the host and make the Compose service name resolvable
before starting Superset:

```bash
cd "$DATUS_QUICKSTART_STACK/superset"

cat > docker-compose.override.yml <<'YAML'
services:
  postgres:
    ports:
      - "5432:5432"
YAML

if grep -qE '(^|[[:space:]])postgres([[:space:]]|$)' /etc/hosts && \
   ! grep -qxE '[[:space:]]*127\.0\.0\.1[[:space:]]+postgres[[:space:]]*' /etc/hosts; then
  echo 'Conflicting /etc/hosts entry for postgres; replace it with: 127.0.0.1 postgres' >&2
  exit 1
fi
grep -qxE '[[:space:]]*127\.0\.0\.1[[:space:]]+postgres[[:space:]]*' /etc/hosts || \
  echo '127.0.0.1 postgres' | sudo tee -a /etc/hosts

docker compose up -d
```

The host's port 5432 must be available. This walkthrough uses
`postgres:5432` so Datus and Superset report the same connection identity.

Start Airflow:

```bash
cd "$DATUS_QUICKSTART_STACK/airflow"
docker compose up -d
```

Default local endpoints:

- Superset: `http://127.0.0.1:8088`, username `admin`, password `admin`
- Airflow: `http://127.0.0.1:8080`, username `admin`, password `admin`
- PostgreSQL serving database: `postgres:5432/superset_examples`, username/password `superset/superset`

For this quickstart, the Superset compose file uses local demo defaults for the
metadata database and admin user.

The Airflow compose file mounts `${DACOMP_HOME}` into the container and exposes
an Airflow connection named `duckdb_dacomp_lever`, which points to
`/workspace/lever_workbench.duckdb`.

Keep the local demo credentials out of `agent.yml` even though they are public
defaults. Export them in every shell that runs Datus:

```bash
export AIRFLOW_PASSWORD=admin
export SUPERSET_PASSWORD=admin
export SUPERSET_PG_PASSWORD=superset
```

## Step 3: Install and Configure the Plugins

Install both published plugins into the same environment as Datus. A bare
package name uses the default pip/PyPI install source; do not add a `pip:`
prefix. The Superset plugin includes the `superset-dashboard-authoring` skill
used later in this guide:

```bash
datus plugin install datus-airflow-plugin
datus plugin install datus-superset-plugin
datus plugin info airflow
datus plugin info superset
```

To replace an existing installation with the latest published package:

```bash
datus plugin install datus-airflow-plugin --force
datus plugin install datus-superset-plugin --force
```

Merge the following configuration into the existing `agent:` section in
`~/.datus/conf/agent.yml`. Keep any existing `agent.providers` settings; the
`/model` command uses those credentials. The paths use the `DACOMP_HOME` and `DATUS_QUICKSTART_STACK`
environment variables from Step 0.

```yaml
agent:
  filesystem:
    allow_write:
      - "${DATUS_QUICKSTART_STACK}/airflow/dags"

  services:
    datasources:
      lever_duckdb:
        type: duckdb
        uri: "duckdb:///${DACOMP_HOME}/lever_workbench.duckdb"
        default: true
      superset_serving:
        type: postgresql
        host: postgres
        port: 5432
        database: superset_examples
        schema: public
        username: superset
        password: ${SUPERSET_PG_PASSWORD}

  plugins:
    airflow:
      local:
        default: true
        api_base_url: http://127.0.0.1:8080/api/v1
        api_version: auto
        username: admin
        password: ${AIRFLOW_PASSWORD}
        verify_ssl: true
        timeout: 30
        dags_folder: "${DATUS_QUICKSTART_STACK}/airflow/dags"
        dag_id_prefix: daily_lever_
        allow_commands: dags,tasks,version,health

    superset:
      local:
        default: true
        api_base_url: http://127.0.0.1:8088
        auth_mode: login
        username: admin
        password: ${SUPERSET_PASSWORD}
        provider: db
        verify_ssl: "true"
        timeout: "30"
```

`filesystem.allow_write` authorizes the agent to publish a DAG into the host
directory mounted by Airflow. The `dags_folder` value tells the agent where to
publish the runtime copy. DAG discovery, triggering, run inspection, and log
retrieval go through the Airflow plugin under main-agent control.

Enable both profiles for this project, then start the chat session:

```bash
cd "$DACOMP_HOME"
datus plugin enable airflow --profile local
datus plugin enable superset --profile local
datus --datasource lever_duckdb
```

Verify both services through the main agent rather than running plugin commands
yourself:

```text
Using the enabled local profiles, ask the Airflow plugin for its server version and health, then ask the Superset plugin for health and the available databases. Perform read-only checks only and report any connectivity or authentication error.
```

Always configure and enable plugins before launching Datus. Plugin skills and
environment context are prepared when a session starts; restart the session
after changing a profile. The selected `lever_duckdb` datasource points at the
writable workbench file.

The quickstart injects `duckdb_dacomp_lever` through Airflow's
`AIRFLOW_CONN_DUCKDB_DACOMP_LEVER` environment variable. Environment-provided
connections are available to `BaseHook` at task runtime but are not returned by
Airflow's REST connection endpoint. Step 6 validates this connection by running
the DAG.

If the CLI says no model is configured, configure one before continuing:

```text
/model
```

Choose a provider/model and enter credentials if prompted. `/model` writes
provider credentials under `agent.providers` in `~/.datus/conf/agent.yml` and
writes the active provider/model for this project to `./.datus/config.yml`.

## Step 4: Create the Required Staging Tables

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

First ask the agent to inspect the physical source columns. This prevents a
source-to-target rename from being mistaken for a missing field:

```text
Inspect the schemas and sample rows for raw.requisition, raw.user, raw.requisition_posting, and raw.requisition_offer. Before generating SQL, confirm these source-to-target renames from the physical columns: raw.requisition.id to requisition_id, name to requisition_name, creator_id to creator_user_id, owner_id to owner_user_id, and hiring_manager_id to hiring_manager_user_id; raw.user.id to user_id, name to user_name, and external_directory_id to external_directory_user_id. Do not create NULL placeholders for columns that exist in the source tables.
```

Then ask the agent to create the staging tables required by the `source_models`
listed for `lever__requisition_enhanced` and
`intermediate.int_lever__requisition_users`. The agent will route the request to
the table-generation workflow:

```text
Read ./docs/data_contract.yaml and create the staging tables needed for marts.lever__requisition_enhanced: staging.stg_lever__requisition from raw.requisition, staging.stg_lever__user from raw.user, staging.stg_lever__requisition_posting from raw.requisition_posting, and staging.stg_lever__requisition_offer from raw.requisition_offer. Use the field design and source-to-target mapping from the contract.
```

These four staging tables are the minimum raw-to-staging inputs for the
requisition-enhancement example.

## Step 5: Build the Intermediate and Marts Tables

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

After the marts table is built, validate every layer and the dashboard
dimensions:

```sql
SELECT 'stg_user' AS model, COUNT(*) AS row_count FROM staging.stg_lever__user
UNION ALL
SELECT 'stg_requisition', COUNT(*) FROM staging.stg_lever__requisition
UNION ALL
SELECT 'stg_requisition_posting', COUNT(*) FROM staging.stg_lever__requisition_posting
UNION ALL
SELECT 'stg_requisition_offer', COUNT(*) FROM staging.stg_lever__requisition_offer
UNION ALL
SELECT 'int_requisition_users', COUNT(*) FROM intermediate.int_lever__requisition_users
UNION ALL
SELECT 'marts_requisition_enhanced', COUNT(*) FROM marts.lever__requisition_enhanced;

SELECT
  COUNT(*) AS total_rows,
  COUNT(status) AS rows_with_status,
  COUNT(team) AS rows_with_team,
  COUNT(location) AS rows_with_location,
  SUM(count_postings) AS posting_links,
  SUM(count_offers) AS offer_links
FROM marts.lever__requisition_enhanced;
```

Every model must be non-empty, and `rows_with_status`, `rows_with_team`,
`rows_with_location`, `posting_links`, and `offer_links` must all be greater
than zero. With the version 1 quickstart package, the marts table contains 146
rows. If dimensions are unexpectedly all NULL, return to the schema inspection
above and correct the source-column mappings before continuing.

Finally, preserve the exact statements that succeeded. The scheduled DAG will
read this file from the `/workspace` mount instead of regenerating SQL:

```text
Collect the exact SQL statements that successfully created the staging, intermediate, and marts schemas and built the four staging tables, intermediate.int_lever__requisition_users, and marts.lever__requisition_enhanced. Keep them in dependency order and write them to ./jobs/daily_lever_requisition_enhanced.sql. Do not replace validated statements with newly invented SQL. Execute the saved file once against lever_duckdb and confirm it reproduces the same non-zero validation results.
```

## Step 6: Publish and Run a Daily Airflow DAG

The Airflow plugin lists DAGs, inspects source and import errors, triggers runs,
and retrieves run state, task state, and logs. The agent publishes the new DAG
through its filesystem tools and then uses the plugin to validate and run it.

For this local stack, publishing means writing the generated file into the
allowlisted host directory mounted at `/opt/airflow/dags`. Ask the agent to
author, publish, and validate the DAG:

```text
Use the Airflow plugin with profile local and follow its airflow skill. Create ./dags/daily_lever_requisition_enhanced.py for DAG id daily_lever_requisition_enhanced with schedule 0 8 * * *, catchup disabled, and a fixed timezone-aware start date. At runtime, read /workspace/jobs/daily_lever_requisition_enhanced.sql, resolve the duckdb_dacomp_lever Airflow connection with BaseHook, reconstruct the DuckDB SQLAlchemy URL from the connection schema or host, and execute the validated SQL inside a committed transaction. Keep the project source file, then use the filesystem tools to write identical content to the local profile's configured dags_folder. Confirm the two files are identical. Wait until the Airflow plugin reports the DAG, check import errors and DAG details, then trigger it once and wait for completion. After the wait finishes, read the latest run again and show the final dag_run_id and state. If it fails, inspect task states and logs before reporting the error.
```

The publish and trigger operations may require confirmation. The same agent
prompt performs the required read-back checks; if you want to repeat them, ask
the agent to list the matching DAG, import errors, DAG details, and latest run
through the Airflow plugin.

What to expect:

- the maintained source is `$DACOMP_HOME/dags/daily_lever_requisition_enhanced.py`
- an identical runtime copy appears under `${DATUS_QUICKSTART_STACK}/airflow/dags`
- the same file is visible inside Airflow as `/opt/airflow/dags/daily_lever_requisition_enhanced.py`
- Airflow reports the `dag_id`, a successful `dag_run_id`, and run state

## Step 7: Promote the Marts Table to the Superset Serving DB

The marts table above was built through the `lever_duckdb` datasource. Before
the Superset plugin can create assets, copy that table into the
`superset_serving` Postgres datasource. These names are Datus datasource names from
`agent.yml`, not physical database or catalog names inside DuckDB or Postgres.

```text
Please copy the source table marts.lever__requisition_enhanced from the lever_duckdb datasource into the superset_serving datasource as public.lever__requisition_enhanced, replacing the target table if it already exists. Then verify the source and target row counts.
```

The transfer tool creates `public.lever__requisition_enhanced` from the source
result columns if it does not already exist. For the version 1 package, both
the source and target should report 146 rows.

After this step, the table exists in the PostgreSQL database registered in
Superset as `examples`. Because both sides identify it as
`postgres:5432/superset_examples`, the plugin can resolve it uniquely to the
`superset_serving` Datus datasource.

## Step 8: Create a Superset Dashboard with the Plugin

Once the marts table exists in `superset_serving`, ask the agent to use the
plugin's authoring skill.

```text
Use the Superset plugin with profile local and follow the superset-dashboard-authoring skill. Discover the Superset Database named examples and resolve its credential-free connection identity uniquely to the superset_serving Datus datasource. Validate public.lever__requisition_enhanced and the planned queries on that Datus datasource first. Register it as a physical Superset Dataset, then create a requisition operations dashboard with KPI tiles for total requisitions, open requisitions, requisitions with postings, requisitions with offers, and total requested headcount. Add charts by status, team, location, employment_status, count_postings, and count_offers. Store only non-sensitive Database, Dataset, Dashboard, and Chart resource request payloads in project-local JSON files. Never persist authentication or login request bodies, tokens, cookies, passwords, or other secrets, and redact sensitive fields before writing any payload. Every chart must contain matching params and query_context JSON strings. Attach all charts and update a complete position_json layout so the dashboard is not blank. Read the Database, Dataset, Dashboard, and Charts back, confirm that the Database connection still identifies postgres:5432/superset_examples, and run representative chart data queries. Return the Database, Dataset, Dashboard, and Chart IDs plus the dashboard URL.
```

Data preparation is a separate ETL / scheduled-workflow step. Dashboard generation
expects the table or SQL dataset to already be available in the database known
to Superset. Superset create/update/query operations are confirmation-gated by
the plugin.

The same agent prompt reads the created resources back and runs representative
chart queries. To repeat verification, give the returned IDs back to the main
agent and ask it to inspect the Database, table metadata, Dashboard, Charts,
and chart data through the Superset plugin. Never copy example IDs from this
page.

The dashboard should contain 11 charts. A representative total-requisitions
chart query should return 146, category queries should return more than one
group, and the Database connection should identify
`postgres:5432/superset_examples`.

## Step 9: Verify the End-to-End Result

You should now have:

- `staging`, `intermediate`, and `marts` schemas in `lever_workbench.duckdb`
- `marts.lever__requisition_enhanced` built from raw data through staging and intermediate layers
- the validated SQL and maintained DAG source under `$DACOMP_HOME/jobs` and `$DACOMP_HOME/dags`
- a successful daily Airflow DAG run visible in the Airflow UI
- Superset Database, Dataset, Dashboard, and Chart IDs plus a dashboard URL

## SaaS Studio Tour Variant

The hosted SaaS tour uses the same Lever workflow, but it does not use the
local `lever_workbench.duckdb` file. Instead, the platform provides a shared
DuckDB + Iceberg lakehouse:

- shared read-only raw namespace: `lake.demo_raw`
- per-workspace writable namespace: `lake.ws_<workspace_id>`
- SaaS Airflow connection: `duckdb_lever_workbench`

The hosted platform supplies managed Airflow/Superset plugin profiles and its
own DAG deployment channel. Do not carry the local `filesystem.allow_write` or
Compose-mounted DAG path into SaaS. Plugin operations still discover, trigger,
and verify the managed resources; the namespace rules below remain unchanged.

Every user should run the tour in a separate workspace. The backend renders the
seeded `docs/data_contract.yaml` for that workspace, so outputs target
`lake.ws_<workspace_id>` while sources stay in `lake.demo_raw`. Prompts and SQL
should use fully qualified table names such as:

```text
lake.demo_raw.requisition
lake.ws_<workspace_id>.stg_lever__requisition
lake.ws_<workspace_id>.int_lever__requisition_users
lake.ws_<workspace_id>.marts_lever__requisition_enhanced
```

Do not use unqualified physical schemas such as `raw.*`, `staging.*`,
`intermediate.*`, or `marts.*` in the SaaS tour. Those names are logical layers
only; the physical write boundary is the workspace namespace.

When the workspace namespace changes, recreate the demo project and regenerate
the DAG so it uses the current `lake.ws_<workspace_id>` namespace.

## Next Steps

- [Turn a Dashboard into a Copilot](dashboard_copilot.md) — build analysis subagents from an existing Superset dashboard.
- [Build a Context-Rich Agent](contextual_data_engineering.md) — build reusable context and compare answer quality.
- [Plugins](../plugin/introduction.md) — configure Airflow, Superset, and other integrations.
- [Choose Your Getting Started Path](index.md) — compare all getting-started guides.
