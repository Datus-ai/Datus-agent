# Airflow Plugin

The Airflow plugin (`datus-airflow-plugin`) connects the Datus agent to a
remote Apache Airflow 2.x or 3.x deployment through Airflow's stable REST API.
Nothing Airflow-related is installed where Datus runs. Once a profile is
configured, the agent can inspect, trigger, and troubleshoot DAGs, manage
variables, connections, and pools, and export or deploy DAG source code — all
driven by natural-language requests in the chat.

## Installation

```bash
datus plugin install datus-airflow-plugin
```

Requires datus-agent >= 0.3.8. After installation the bundled skills appear in
`/skill list`, and the agent discovers the plugin and its configured
environments when a session starts. See [Plugins](introduction.md) for other
install sources and profile management.

## Skills

| Skill | Purpose |
|---|---|
| `airflow` | Day-to-day operations against a configured Airflow environment |
| `airflow-setup` | Create an environment profile through a guided conversation |
| `airflow-dag-export` | Export or deploy DAG source with an explicitly confirmed scope |

### airflow

The core operations skill. With it the agent can:

- list and inspect DAGs, runs, task states, and task logs;
- trigger a DAG run — optionally waiting for the result — and clear failed
  runs or tasks for a retry;
- pause and unpause DAGs and check import errors after a deployment;
- manage instance-wide variables, connections, and pools;
- create and monitor backfills and read server version and health (assets and
  the backfill API require Airflow 3).

A profile can carry scope guardrails: `dag_id_prefix` restricts every
DAG-scoped action to matching DAG ids, and `allow_commands` restricts the
available command groups. The agent sees these limits in its context and works
inside them instead of discovering them by failing. They guard against
mistakes; they are not a security boundary — tenant isolation still belongs on
the Airflow server.

### airflow-setup

Configuration by conversation. Ask the agent to set up the plugin and the
skill collects the Airflow web server URL, the auth method (a static API
token, or a username and password), and optional settings such as a
`dags_folder` deployment URI or the scope guardrails above. Secrets are always
written as `${ENV_VAR}` references, never literals, and the skill verifies the
new profile with a cheap read-only call before finishing. A resulting profile
looks like this:

```yaml
agent:
  plugins:
    airflow:
      prod:
        default: true
        api_base_url: https://airflow.example.com/api/v1  # /api/v1 = Airflow 2, /api/v2 = Airflow 3
        username: admin
        password: ${AIRFLOW_PASSWORD}
        dags_folder: s3://my-bucket/dags/  # optional deployment URI;
                                           # storage credentials belong to the s3 plugin
```

### airflow-dag-export

A confirmation-gated workflow for exporting, backing up, migrating, or
deploying DAG source:

1. **Discover** — the live Airflow API is the only source of truth. The skill
   lists the active DAG set and fetches each Python file through the API; it
   never scans the `dags_folder` storage behind the scheduler.
2. **Propose and refine** — it presents a complete proposal: environment,
   selected DAGs, files, destination, and transfer method. You adjust the
   scope in natural language — by DAG id, glob or regex, owner, tag, paused
   state, source keyword, or referenced connection id — and it recomputes the
   proposal after every change.
3. **Confirm and write** — nothing is written to the destination until you
   confirm the exact current proposal. The export ships with a
   `dag-export-manifest.json` recording every DAG, file, and checksum, and
   never contains credentials.
4. **Upload** — the destination URI selects the transfer: local paths are
   copied directly, `s3://` goes through the [S3 plugin](s3.md), `gs://` and
   `abfs://` through the GCS and ADLS plugins. The Airflow plugin itself
   contains no object-storage client.

## Using with the Agent

Some example requests, once a profile exists:

- **"Which DAGs failed last night in prod?"** — the agent lists recent failed
  runs, drills into task states, and pulls the failing task's logs.
- **"Trigger sales_daily and wait for it to finish."** — the agent asks for
  confirmation, starts the run, and polls until it succeeds or fails.
- **"Clear the failed tasks of yesterday's sales_daily run and retry them."**
- **"Backfill sales_daily for the first week of January — dry-run first."**
- **"Set up the airflow plugin for our staging cluster."** — runs
  `airflow-setup`.

Commands the agent runs go through the Datus
[permission system](introduction.md#permissions): read-only operations run
without confirmation; routine reversible ones (pausing, clearing runs, setting
variables) ask once under the `normal` mode; anything that starts a run,
deletes an object, imports files in bulk, or touches connection secrets always
asks first.

## Orchestrating Workflows

### Deploy DAGs through the S3 plugin

When the scheduler syncs its DAGs from object storage — the usual production
setup — pair this plugin with the [S3 plugin](s3.md):

1. Point the profile's `dags_folder` at the bucket the scheduler reads, e.g.
   `s3://my-bucket/dags/`. Storage credentials live in the S3 plugin's own
   profile; the Airflow plugin never receives them.
2. Ask the agent to deploy. It uploads the DAG file through the S3 plugin,
   then polls the DAG list and import errors through the Airflow API until
   the new DAG is parsed.
3. Continue in the same conversation: trigger the DAG and watch the run.

A single end-to-end request works too:

```text
Create dags/hello_world.py with one task that prints the run date, upload it
to the Airflow DAG root with the s3 plugin, verify the DAG appears, trigger
it, and wait for completion.
```

The same pairing runs in reverse for backup and migration — "export all
active prod DAGs to s3://backup/airflow/2026-08-24/" drives
`airflow-dag-export` with the S3 plugin as the transfer layer.

### Airflow on Amazon MWAA

For Airflow hosted on Amazon MWAA, the [MWAA plugin](mwaa.md) manages the
environment itself — login links, tokens, environment details — and this
plugin can be pointed at the environment's web server for the fine-grained,
permission-classified DAG operations described above.

## Related Docs

- [Plugins](introduction.md) — install sources, profiles, activation, and permissions
- [S3 plugin](s3.md) — the transfer layer behind `s3://` DAG deployment
- [MWAA plugin](mwaa.md) — Amazon-managed Airflow environments
- [Data Engineering Quickstart](../getting_started/data_engineering_quickstart.md) — an end-to-end pipeline that publishes a daily Airflow DAG
- [Skills](../skills/introduction.md) — how skills are discovered and loaded
