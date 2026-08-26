# Choose Your Getting Started Path

Datus supports several workflows, but you do not need to complete every tutorial in order. Start with the short installation guide, then choose the path that matches what you want to build.

## Start here

| Your goal | Recommended guide | What you will build |
| --- | --- | --- |
| Install Datus, connect a datasource, and ask the first question | [Install and First Query](Quickstart.md) | A working local Datus REPL with a configured model and datasource |
| Learn how metadata, semantic models, metrics, Reference SQL, and scoped subagents work together | [Build a Context-Rich Agent](contextual_data_engineering.md) | A Knowledge Base and two subagents over the bundled California Schools dataset |
| Build a layered warehouse workflow from source data to a scheduled pipeline and BI dashboard | [End-to-End Data Engineering](data_engineering_quickstart.md) | DuckDB staging/intermediate/marts tables, an Airflow DAG, and a Superset dashboard |
| Turn an existing BI dashboard into reusable SQL, metrics, and analysis subagents | [Turn a Dashboard into a Copilot](dashboard_copilot.md) | Reference SQL, a Dosi semantic model, and two subagents scoped to a Superset dashboard |

If this is your first time using Datus, complete [Install and First Query](Quickstart.md) first. The other three guides are independent paths; choose one rather than reading them all front to back.

## How the paths differ

```text
Install and First Query
├── Build a Context-Rich Agent
│   └── Learn the core context-building workflow on bundled sample data
├── End-to-End Data Engineering
│   └── Build data → ETL → Airflow → Superset dashboard
└── Turn a Dashboard into a Copilot
    └── Use an existing dashboard → SQL and metric evidence → analysis subagents
```

The two scenario tutorials use Superset for different purposes:

- **End-to-End Data Engineering** creates and publishes a new dashboard after building the data pipeline.
- **Dashboard Copilot** starts from an existing dashboard and converts its query evidence into reusable context and subagents.

## What to learn after the tutorials

- [CLI](../cli/introduction.md): commands, input modes, sessions, and agent selection
- [Knowledge Base](../knowledge_base/introduction.md): metadata, semantic models, metrics, and Reference SQL
- [Subagents](../subagent/introduction.md): built-in and customized agents with scoped context
- [Skills](../skills/introduction.md): reusable workflows used by agents and plugins
- [Configuration](../configuration/introduction.md): datasources, semantic adapters, storage, and nodes
