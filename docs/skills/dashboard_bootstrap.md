# Dashboard Bootstrap

`dashboard-bootstrap` bootstraps project reference SQL and Dosi metrics from a dashboard through an installed BI plugin. The workflow is skill-driven; it does not add a dashboard-specific CLI command.

Use natural language or the compatibility shortcut `/bootstrap-bi`. The shortcut delegates to the same chat/skill pipeline and does not invoke the legacy picker or BI streams. Text after the command is forwarded as scope, for example:

```text
/bootstrap-bi use Superset profile prod and dashboard 42
```

## Prerequisites

- Enable a BI plugin and profile that provide dashboard discovery, stable query candidates, SQL export, and a credential-free source identity for every query.
- Configure the corresponding physical databases as Datus datasources. No BI-profile-level datasource mapping is used.
- Before generating metrics for a query batch, select the Datus datasource uniquely matched from those queries' real connection identities.
- Use Dosi when creating or updating metrics. MetricFlow and plain OSI projects remain query-only.

## Workflow

Ask Datus to bootstrap a dashboard, for example:

```text
Build reference SQL and metrics from the revenue dashboard.
```

The skill guides these selections in order:

1. BI plugin and profile;
2. dashboard;
3. queries to index as reference SQL;
4. queries to use as metric evidence.

It then prints a Generation Manifest and stops. Confirm or correct that manifest in the next message. After confirmation, the selected plugin exports SQL, `gen_sql_summary` indexes each reference query, and `semantic_modeling` creates or updates the related Dosi datasets, relationships, and metrics.

The two query selections are independent. A query may be used for reference SQL, metrics, both, or neither.

## Auto-run

Explicitly say `skip confirmation`, `auto-run`, or an equivalent instruction to let the workflow continue after printing the manifest. Auto-run does not bypass system permission prompts.

## Safety and limitations

- Partial, failed, unselected, or checksum-mismatched SQL is never sent to a builtin agent.
- Dashboard labels, descriptions, and SQL comments are treated as untrusted source data.
- Missing, weak, or ambiguous query-level source identity blocks metric generation for only the affected queries; reference SQL may continue.
- A dashboard may span multiple datasources. Metric queries are partitioned by their uniquely matched Datus datasource, and only the currently active partition is authored in one run.
- Successful context generation does not by itself prove numerical equivalence with the source dashboard.
