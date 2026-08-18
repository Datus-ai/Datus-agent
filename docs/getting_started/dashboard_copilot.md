# Dashboard Copilot

Dashboard Copilot builds project reference SQL and metrics from a BI dashboard. The workflow is driven by the bundled `dashboard-bootstrap` skill and the selected BI plugin; Datus does not hard-code a BI vendor's dashboard API or SQL compiler.

`/bootstrap-bi` remains available as a compatibility shortcut. It delegates to the normal chat/skill pipeline and no longer starts the legacy picker or BI streams. At the end, the workflow can persist the same main and attribution subagent pair through the generic `create-subagent` skill.

## Prerequisites

Before starting, configure:

1. A BI plugin and profile whose bundled export skill can discover dashboards and query candidates, then export SQL with stable IDs, files, checksums, statuses, and credential-free query-level source identities.
2. Each physical database used by the dashboard configured as a Datus datasource. No profile-level mapping is required; Datus matches each query's real BI connection identity and authors only the partition for the active datasource.
3. An LLM model selected with `/model`.
4. A writable Dosi semantic project when metrics must be authored. MetricFlow and plain OSI projects are query-only in this workflow.
5. A mutable loaded `agent.yml` when dashboard-specific subagents should be persisted. Read-only runtimes hide `create-subagent`; context bootstrap still works and skips only persistence.

The Superset plugin implements the required export contract. Other BI plugins can participate without changing Datus Agent when they expose the same capabilities through their own bundled skill.

## Start the workflow

Use natural language:

```text
Build reference SQL and metrics from the World Bank dashboard in the Superset prod profile.
```

Or use the compatibility shortcut and append the same scope as free text:

```text
/bootstrap-bi use Superset profile prod and the World Bank dashboard
```

Both forms enter the same workflow. The shortcut asks the agent to load `dashboard-bootstrap`; it does not run a separate implementation.

## Selection and confirmation

The skill guides the agent through these stages:

1. Select an installed BI plugin and a named profile.
2. Select a dashboard by stable ID, URL, or unambiguous title.
3. Select queries for reference SQL.
4. Independently select queries for metric generation. A query may be in either set, both sets, or neither set.
5. Review a Generation Manifest containing the plugin/profile, dashboard, selected query IDs, exclusions, datasource match, export mode, and ambiguities.

Aggregation is only a recommendation signal. It does not automatically make a dashboard query a metric.

By default, the agent stops after displaying the Generation Manifest. Confirm or correct it in the next message. No SQL is exported and no Knowledge Base artifact is generated before confirmation.

To intentionally skip this turn boundary, say so explicitly, for example:

```text
/bootstrap-bi use Superset profile prod and dashboard 42; run automatically without confirmation
```

This does not bypass system permission prompts.

## Export and context construction

After confirmation, the selected BI plugin exports the SQL. Datus verifies query identity, status, file location, and checksum against the confirmed manifest.

- Each confirmed reference query is sent as complete original SQL to `gen_sql_summary`.
- Confirmed metric queries are first partitioned by their uniquely matched Datus datasource, then grouped by business domain and sent as complete original SQL to `semantic_modeling`. One run authors only the active-datasource partition.
- `semantic_modeling` inspects the live schema, updates the Dosi semantic model and metrics, validates them, and reconciles the Knowledge Base.
- A failure in one path does not implicitly block the other path.
- After context construction, `dashboard-bootstrap` loads `create-subagent` when available and adds or updates `<platform>_<dashboard>` plus `<platform>_<dashboard>_attribution` under `agent.agentic_nodes`. Their scope contains only successful active-datasource tables and exact metric/reference-SQL subject references.

The plugin owns BI access and SQL fidelity. The main agent and plugin do not hand-write reference SQL index entries or Dosi artifacts.

## Result

The final report lists:

- plugin/profile and dashboard identity;
- export directory and manifest;
- succeeded, failed, and skipped reference queries;
- succeeded, failed, and skipped semantic domains;
- generated reference SQL identifiers, semantic model files, and metric names returned by their owning builtin agents;
- subagents created, updated, unchanged, failed, or skipped;
- the smallest safe retry scope.

Subagent creation is an optional final step and does not change whether the Knowledge Base context succeeded. The generic skill edits only the loaded `agent.yml`; it does not copy custom prompt files because the nodes fall back to the builtin `gen_sql` and `gen_report` templates. If the configuration is immutable or the skill cannot be loaded, the workflow reports the skip and preserves all successfully built context.

For exact Knowledge Base scope, `metrics` and `sqls` are canonical dotted references in the form `<subject_path>.<item_name>`. They are derived from the synchronized Dosi metric or SQL-summary artifact, not from metric storage IDs, bare adapter metric names, YAML paths, or plugin query IDs. A bare subject path selects its whole subtree and is used only when that broader scope was explicitly requested.

## Failure behavior

- Missing plugin export capabilities: install or update a compatible BI plugin.
- Multiple matching profiles or dashboards: select a stable profile name and dashboard ID.
- Failed, partial, unselected, or checksum-mismatched SQL: that query is rejected rather than reconstructed by the LLM.
- Missing, weak, or ambiguous query source identity: reference SQL may continue, but metric authoring stops for the affected query.
- Query matched to a non-active datasource: its metric partition is deferred until that datasource is active; the workflow never silently switches it.
- Query-only semantic adapter: migrate the project to Dosi before authoring metrics.
- Immutable or unavailable Agent configuration: subagent creation is skipped; reference SQL and metric context results remain valid.

See [Dashboard Bootstrap](../skills/dashboard_bootstrap.md) for the full workflow contract.
