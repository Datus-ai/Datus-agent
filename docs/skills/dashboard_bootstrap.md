# Dashboard Bootstrap

`dashboard-bootstrap` is the single main-agent entry for building reference SQL,
DOSI semantic models/metrics and optionally reconstructing BI dashboards as Datus
visual dashboards. Source access, native interpretation and commands remain in
the installed BI plugin; platform-specific guidance ships with that plugin.

Use natural language or `/bootstrap-bi`; the shortcut forwards scope to the same
chat/skill pipeline, without the legacy picker or BI streams:

```text
/bootstrap-bi use Superset profile prod and dashboard 42
```

An unspecified shortcut request keeps the reference-SQL and metric context default.
A visual-only request creates only the selected visual output, using maintained
metrics by default. No separate migration skill/command is introduced.

## Select outputs and policy

Examples:

```text
Build reference SQL and metrics from the revenue dashboard.
Rebuild the selected BI dashboard in Datus using existing metrics.
Sync the selected project dashboards, generate missing metrics, and rebuild them in Datus.
```

Reference SQL, metric evidence, visual output and optional agent persistence are
independent selections. `reuse_only` is the default for reconstruction; an explicit
request to generate metrics authorizes `create_missing`, which reuses existing
matching definitions and authors only selected missing ones. Editing conflicting
business definitions needs explicit edit scope. SQL fallback requires explicit
scope and support from the source plugin and target artifact tools.

## Source prerequisites

Enable an appropriate BI plugin/profile and configure physical databases as Datus
datasources. Source skills may provide query exports, or verified native asset
bundles with stable chart/dataset IDs, definitions, dependencies and coverage.
The workflow does not require every native chart to expose runtime SQL. Dataset
Custom SQL is not automatically the chart's aggregate query.

Metric authoring requires strong per-evidence physical connection matching to the
active datasource. A profile can front many sources; no BI-profile-level datasource
mapping or automatic shared-datasource switch is used. Reuse of an independently
complete curated DOSI target uses its actual semantic/datasource binding instead.

## Workflow and on-demand references

The skill records plugin/profile/site, dashboard/chart scope, selected outputs,
metric policy, source identity evidence or pending inspection, exclusions and target
intent in a Generation Manifest. It stops for confirmation before export/generation
unless the user explicitly requests `skip confirmation` or `auto-run`. Existing
confirmation and unambiguous user scope are retained; auto-run does not bypass
system permissions.

The packaged skill loads only relevant supporting documents:

- `references/source-contract.md`: query exports/native bundles, identity matching,
  checksum verification, provenance and coverage.
- `references/context.md`: selected reference indexing and semantic/metric authoring.
- `references/visual-dashboard.md`: analytical mapping, metric-before-visual dependency,
  exact `gen_visual_dashboard` handoff, actual metric query saves and validation.
- `references/subagents.md`: optional persistence of the legacy scoped agent pair.
- `references/validation.md`: separate phase/chart evidence, completion and retries.

Plugin skills provide native syntax, commands and limitations; vendor guides are
not copied into the common skill. Use CLI, prompts and skills from one plugin
release. An observed release/location mismatch stops source consumption.

After confirmed source export/inspection, `gen_sql_summary` owns reference indexing
and `semantic_modeling` owns DOSI datasets, relationships and metrics. Reference
indexing can proceed independently. A dashboard that requires new metrics waits
for metric validation/reconciliation and rediscovery before `gen_visual_dashboard`
runs. The visual node discovers actual metric capabilities and saves matching
panels via `save_metric_query_template`, including real trial results/model
snapshots. Newly created metrics next to SQL templates do not establish binding.

Fresh child tasks do not inherit parent-loaded skill bodies. Handoffs include exact
source IDs/checksum, datasource/model binding, metric identities, analytical intent,
blockers, approved differences and readable supporting skill/reference locations.
The child loads the required skills to register its own supporting-file read roots
before reading the selected references; it does not restart orchestration or
create missing metrics itself.

## Optional dashboard subagents

Legacy context bootstrap retains the final optional pair when `create-subagent`
is available: `<platform>_<dashboard>` (`gen_sql`) and its `_attribution` node
(`gen_visual_report`). Visual-only requests create this pair only when requested.
Both use successful active-datasource artifacts and exact post-sync metric and
reference-SQL subject references. The generic creation skill preserves unrelated
configuration and verifies writes. In immutable API deployments this phase is
skipped; its failure does not invalidate completed context or dashboard artifacts.

## Completion and limitations

Report source coverage, reference index resolution, metrics, actual per-chart query
binding, render/interactions, publication and optional agents separately. Preserve
partial/blocked objects and retry only the smallest affected set.

- Retrieved, derived and target SQL keep distinct provenance. Identical SQL is
  deduplicated downstream only with matching physical source, dialect and execution
  context, preserving all original references and immutable source snapshots.
- A saved reference YAML is only `written` until its exact subject/search reference
  resolves; do not presume an unexplained indexing delay.
- Unsupported native semantics and missing required upstream processing block
  dependent charts; generated SELECT models do not deploy or execute ETL.
- Changes to calculations, grain, filters, security or interactions must remain
  visible. A successful render, preflight or target-only reconciliation does not
  prove numerical equivalence with the original BI dashboard.
- Preview and publication are distinct; use the existing publication flow only
  when requested.
