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

It then prints a Generation Manifest and stops. Confirm or correct that manifest in the next message. After confirmation, the selected plugin exports SQL, `gen_sql_summary` indexes each reference query, and `semantic_modeling` creates or updates the related Dosi datasets, relationships, and metrics. As the final step, `dashboard-bootstrap` loads `create-subagent` when that skill is available and persists the dashboard's main and attribution nodes in the loaded `agent.yml`.

The two query selections are independent. A query may be used for reference SQL, metrics, both, or neither.

## Dashboard subagents

When Agent configuration is mutable, the final step creates or updates two nodes using the legacy naming and tool pattern:

- `<platform>_<dashboard>` uses `gen_sql` with database and context-search tools.
- `<platform>_<dashboard>_attribution` uses `gen_report` with semantic attribution tools.

Both nodes are scoped only to successful tables and exact metric/reference-SQL subject references for the active datasource. Metric scope uses `<metric.subject_path>.<metric.name>` from the synchronized Dosi model; reference-SQL scope uses `<subject_tree>.<name>` from the generated SQL summary. Bare subject paths select whole subtrees and are not used for an exact Dashboard selection. The generic `create-subagent` skill resolves these references against the post-sync subject trees, edits `agent.agentic_nodes` without replacing sibling entries, and verifies the YAML after writing. When the runtime marks configuration read-only, that skill is not discoverable and the workflow skips persistence without failing context construction.

## Auto-run

Explicitly say `skip confirmation`, `auto-run`, or an equivalent instruction to let the workflow continue after printing the manifest. Auto-run does not bypass system permission prompts.

## Safety and limitations

- Partial, failed, unselected, or checksum-mismatched SQL is never sent to a builtin agent.
- Dashboard labels, descriptions, and SQL comments are treated as untrusted source data.
- Missing, weak, or ambiguous query-level source identity blocks metric generation for only the affected queries; reference SQL may continue.
- A dashboard may span multiple datasources. Metric queries are partitioned by their uniquely matched Datus datasource, and only the currently active partition is authored in one run.
- Successful context generation does not by itself prove numerical equivalence with the source dashboard.
- Subagent creation failure does not invalidate successfully built context and can be retried independently.
