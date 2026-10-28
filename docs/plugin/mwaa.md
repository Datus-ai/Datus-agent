# MWAA Plugin

The MWAA plugin (`datus-mwaa-plugin`) connects the Datus agent to Amazon
Managed Workflows for Apache Airflow. It works at the environment level:
inspecting MWAA environments, minting Airflow UI login links and CLI tokens,
listing the DAGs an environment currently runs and reading their source, and
passing Airflow CLI commands through MWAA's REST endpoint. Creating, updating,
or deleting environments is out of scope.

## Installation

```bash
datus plugin install datus-mwaa-plugin
```

Requires datus-agent >= 0.3.8. AWS credentials resolve through the standard
boto3 chain, or through the profile fields collected by `mwaa-setup` below.
See [Plugins](introduction.md) for other install sources and profile
management.

## Skills

| Skill | Purpose |
|---|---|
| `mwaa` | Inspect environments, mint tokens, read current DAGs and their source |
| `mwaa-setup` | Create a profile (region, credentials, default environment) |
| `mwaa-dag-export` | Export the environment's DAG source with an explicitly confirmed scope |

### mwaa

With this skill the agent can:

- list MWAA environments and describe one — the details include the Airflow
  version, status, web server URL, and the S3 bucket and DAG path the
  environment reads;
- mint a one-time Airflow UI login URL, or a CLI token together with the web
  server hostname;
- list the DAGs the environment currently runs and read a DAG's source. Both
  go through a short-lived MWAA web session straight to the environment's
  Airflow REST API — the S3 bucket is never read;
- run an Airflow CLI command over MWAA's REST endpoint. This is an opaque
  passthrough — the wrapped command could be destructive — so the agent
  always asks for confirmation first, and MWAA does not support every Airflow
  subcommand there. For day-to-day DAG operations prefer the
  [Airflow plugin](airflow.md) pointed at the environment.

### mwaa-setup

Ask the agent to set up the plugin and the skill collects the AWS region, the
credential source (the default AWS chain, a named profile, keys as
`${ENV_VAR}` references, or a role to assume), and an optional default
environment name, then verifies the profile by listing environments. The IAM
principal needs `airflow:ListEnvironments`, `airflow:GetEnvironment`,
`airflow:CreateWebLoginToken`, and `airflow:CreateCliToken`. A resulting
profile looks like this:

```yaml
agent:
  plugins:
    mwaa:
      prod:
        default: true
        region: us-east-1
        environment: prod-airflow  # optional default environment
        # credentials: standard AWS chain, or profile / keys / role_arn
```

### mwaa-dag-export

The MWAA counterpart of the Airflow plugin's export workflow, with the same
guarantees: the environment's Airflow API is the only source of truth (the
skill never enumerates the MWAA S3 DAG prefix), the scope is adjustable in
natural language and recomputed after every change, nothing is written or
uploaded before you confirm the exact proposal, and the export carries a
checksummed `dag-export-manifest.json` that never contains credentials or
tokens. Uploads route by destination URI — `s3://` goes through the
[S3 plugin](s3.md).

## Using with the Agent

Some example requests, once a profile exists:

- **"Give me a login link for the prod MWAA UI."** — a one-time web-login URL.
- **"Which DAGs run in the prod environment? Show me sales_daily's source."**
- **"Describe the analytics-airflow environment — which bucket does it read
  DAGs from?"**
- **"Set up the mwaa plugin for us-east-1."** — runs `mwaa-setup`.

These inspection operations run without confirmation. Only the Airflow CLI
passthrough always asks first, because its payload is opaque to the
[permission system](introduction.md#permissions).

## Orchestrating Workflows

### Fine-grained DAG operations through the Airflow plugin

MWAA manages the environment; the [Airflow plugin](airflow.md) manages the
DAGs. For triggering, pausing, clearing, log reading, and the rest of its
permission-classified operation set, configure an Airflow-plugin profile
against the MWAA environment's web server — this plugin supplies the hostname
and token. Ask the agent to wire the two together:

```text
Point the airflow plugin at the prod MWAA environment, then trigger
sales_daily and wait for the run.
```

### Deploy DAGs through the S3 plugin

An MWAA environment reads its DAGs from the S3 bucket and prefix shown in its
environment details. The MWAA plugin contains no S3 transfer, so deployment
is a combination: the agent uploads the file through the [S3 plugin](s3.md),
then verifies through this plugin that the DAG appears:

```text
Upload dags/sales_daily.py to the prod MWAA environment's DAG folder and
confirm the DAG shows up.
```

In the other direction, "export every DAG of the prod MWAA environment to
s3://backup/mwaa/2026-08-24/" runs `mwaa-dag-export` — the source always
comes through the Airflow API, and the upload goes through the S3 plugin.

## Related Docs

- [Plugins](introduction.md) — install sources, profiles, activation, and permissions
- [Airflow plugin](airflow.md) — fine-grained DAG operations for the same environment
- [S3 plugin](s3.md) — DAG uploads into the environment's bucket
- [Skills](../skills/introduction.md) — how skills are discovered and loaded
