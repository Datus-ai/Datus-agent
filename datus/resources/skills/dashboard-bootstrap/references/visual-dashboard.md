# Datus Visual Dashboard Reconstruction

Read when the requested output includes a Datus visual dashboard. The main agent
owns this flow. Load the selected plugin's available rebuild skill for native
interpretation and any deterministic source preflight; treat it as supporting
guidance, not a second orchestration loop. No source-specific skill may broaden
the confirmed scope or silently replace existing DOSI definitions.

## 1. Establish analytical mappings and metric policy

Start from the verified evidence in [source-contract.md](source-contract.md).
For every selected chart compare analytical intent with actual discovered metric
details: aggregation/distinctness, numerator/denominator, dimensions, required
dimensions, time grain/timezone, NULL handling, filter stage, LOD/window behavior
and security. Similar labels are insufficient. Preserve unsupported interactions
and requested redesign differences explicitly; visual similarity is not proof
of analytical equivalence.

Use only metric discovery tools available in the current node. If the main agent
has discovery, build a preliminary mapping. Otherwise, when missing-metric
creation is selected, use `semantic_modeling` to inspect/reuse and author only
missing definitions; require its structured validation/reconciliation results.
With `reuse_only`, let the visual node perform its own discovery and block missing
metrics; do not dispatch an authoring task just to discover definitions.
The destination node always rechecks actual metric capabilities itself.

A request to convert an external dashboard into a Datus project dashboard selects
`create_missing` unless the user explicitly limits it to existing metrics. This
covers necessary additive definitions, not unrelated measures or changes to an
existing conflicting business definition.

- `reuse_only`: reuse maintained DOSI definitions; report missing or incompatible
  metrics/dimensions as blocked. No automatic metric authoring.
- `create_missing`: reuse matching definitions, then run the selected domain's
  `semantic_modeling` task using [context.md](context.md). Preserve existing
  definitions; conflicting edits require explicit scope. Do not add plausible
  extra KPIs absent from the selected analytical intent.
- Any SQL-backed exception must be explicitly authorized, permitted by the
  plugin and target tools, and recorded with considered metric candidates and
  `fallback_reason`. A discovery/model-load/permission/query failure is not a
  semantic gap. Never treat an error as an empty successful metric registry.

## 2. Complete metric work before destination authoring

Wait for all metric authoring required by the chosen charts. Check actual model
files and structured validation/reconciliation evidence; a successful prose
summary alone is insufficient. Rediscover full metric identities as
`{path: [...], name: ...}`, their supported dimensions/parameters and target
model binding after writes. Keep the real model SHA256 when required by the
plugin's evaluator, rather than inventing a revision.

If main-agent discovery is unavailable, the completed semantic task must return
its verified identities/model evidence and the visual node must rediscover them
before saving queries. Failed or stale discovery keeps the affected charts
pending. Do not invoke visual generation concurrently with the metric task it
depends on, or generate SQL first with a promise to attach metrics later.
Unrelated ready charts may continue with explicitly partial scope.

## 3. Delegate one exact destination handoff

Invoke `task(type="gen_visual_dashboard", ...)` only after required metric work.
Parent-loaded skills are not automatically inherited by a fresh child task.
Include all of the following in its prompt, or a workspace handoff file it must
read before writing artifacts:

```text
source: plugin/release, profile/site, verified bundle path + manifest SHA256
selection: exact dashboard/chart IDs and selected evidence/dependency references
target: active datasource, database/schema, new slug intent or authorized edit scope
policy: reuse_only or create_missing completed; permitted SQL exceptions, if any
metrics: exact discovered {path, name}, supported dimensions/parameters, model file/revision
mapping: per-chart analytical intent, required filters/grain/interactions, differences
coverage: per-chart blockers/exclusions and any documented source-gap resolutions
instructions: load the available dashboard-bootstrap and plugin rebuild skills
              to register their supporting-file roots in this child; then read
              only the visual/validation and needed platform references at their
              returned skill_locations; do not restart main-agent orchestration
validation: real metric trial calls, parameter checks, render checks, comparison scope
```

Pass concrete paths and verified values, never the example's placeholder labels.
Loading in the parent does not register supporting-file read roots in the child;
passing an absolute path alone does not grant access. If a child cannot load the
needed skill or read a reference, include its actual relevant constraints explicitly
in a retry handoff; never assume it inherited parent context or bypass file access.
Do not give it a missing-metric creation instruction: the main agent must finish
that phase first. The main agent must not write dashboard query/render artifacts
as an alternate path around a failed task.

## 4. Bind queries and validate inside the visual node

- Require real metric discovery and `save_metric_query_template` for matching
  metrics. Copy identities from discovery exactly; the tool owns compilation,
  warehouse-policy checks, trial results, `.metric.json` and model snapshots.
- Run a plugin-provided binding evaluator when documented, after actual metric
  discovery and before query saves. Its report is only a preflight. Recheck model
  revision and source checksum; changed inputs invalidate the mapping/preflight.
- Do not copy proposed calls into artifact files. Do not manufacture SQL, trial
  results or model snapshots. Verify saved query slugs and returned metadata.
- For several metrics use separate metric queries. Combine results only with
  proven equal datasource, keys, grain, filter context and unique keys; independent
  KPI values need no join. Do not add additive totals across incompatible grains.
- Use the runtime's `useQuery` and real saved query references. Preserve required
  global/local filters, time bounds, drill-down and supported interactions.
- Exercise default and alternate parameters, empty results and supported drill-downs,
  then use `validate_render()` and the existing finalization path. Keep any
  untested interaction explicit; structural render validation is not a browser test.

## 5. Reconcile and return

Use [validation.md](validation.md). Confirm the generated dashboard actually binds
the intended metrics; having both SQL templates and newly created metrics in the
project is insufficient. Distinguish source-vs-target comparisons from target-only
cross-chart checks and metric-vs-derived-SQL checks. Without source results, report
source equivalence as `unverified`, even if all target checks pass.

Return source ID → metric identity → saved target query → chart/block ID mappings,
actual validation outcomes, semantic/visual differences and blocked charts. Publish
only through the existing publication flow when requested; preview is not publication.
