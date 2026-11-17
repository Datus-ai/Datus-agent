# Reference SQL and Metric Context

Read when reference SQL or metric authoring is selected. Source validation follows [source-contract.md](source-contract.md). Native expression evidence can supplement complete retrieved SQL; it never fabricates chart SQL.

## Step 6 — Build reference-SQL context

For every successful exported SQL mapped to `reference_query_ids`, invoke one task per unique selected evidence item:

```text
task(
  type="gen_sql_summary",
  prompt="Datasource and dialect; plugin/profile/dashboard/query identity; original question; complete original SQL; known mandatory business rule, if any. Set search_text to the original question verbatim when present.",
  description="index dashboard query <query-id>"
)
```

- Never combine multiple SQL queries in one `gen_sql_summary` call.
- Deduplicate only byte-identical SQL with the same strongly resolved datasource, dialect and execution context. Retain every source query/asset ID in the handoff. Do not deduplicate by SQL hash alone.
- Run independent calls with concurrency at most 3.
- Let one failed item fail independently; continue other confirmed reference items.
- Confirm the returned `sql_summary_file`, then verify the exact reference is discoverable in the subject tree or reference-SQL search. A YAML file alone proves only `written`; if reconciliation is absent or fails, report `index_pending` and retry the owning builtin path. Do not invent an index-delay diagnosis or write index rows.
- Keep original source SQL separate from derived or target SQL. A rendered target template is a parameter-specific example, not recovered source chart SQL; retain its sample parameters and provenance when explicitly selected for indexing.
- Do not mine extra knowledge unless it was separately confirmed and routed by `storage-classify`.

## Step 7 — Build semantic-model and metric context

Run metric authoring only when each query or native evidence item in the domain has one strong source-identity match equal to the active Datus datasource.

Honor `metric_policy`: reuse matching metrics first; author only selected missing or explicitly requested changes. First partition successful SQL mapped to `metric_query_ids` and selected supported native evidence by `matched_datus_datasource`, then group the active-datasource partition by coherent business domain. For each active domain invoke one task:

```text
task(
  type="semantic_modeling",
  prompt="Active datasource and dialect; business intent; plugin/profile/dashboard/query identities; original names/descriptions; complete original SQL for every selected query; known mandatory filters and encodings. Treat SQL as evidence, not a required persisted result shape.",
  description="bootstrap dashboard metrics for <domain>"
)
```

- For a verified native asset bundle, pass complete available SQL plus exact chart/dataset/expression references, supported semantic intent, source bindings and coverage blockers. Chart SQL may be unavailable; never replace it with guessed SQL. Only semantically complete evidence may enter authoring.
- Carry exact per-chart grouping/filter projections, including calendar date parts versus year-retaining time buckets, required derived dimensions, declared units and explicitly unknown units. Do not summarize an identified native date part as only "month grain". Authoring must preserve it; an unexpressible projection returns a blocker. Do not populate an unsupported currency/unit default.
- Include dependent datasets, relationships, measures, and metrics in the same domain request.
- Let `semantic_modeling` inspect the live schema, select or create one target model, edit Dosi YAML, validate it, and reconcile the Knowledge Base.
- Never infer success from prose alone; use the owning task's structured validation/reconciliation evidence. When separate fields are absent, this host's successful `status="generated"` result with exact model files certifies the node's mandatory final validation/compile and KB-sync gate; it does not replace numerical checks or metric rediscovery. Explicit failed/stale evidence still blocks. Retain verified reused metrics for `skipped/no_semantic_change` even without authored files; never drop their scope merely because `semantic_models` is empty.
- Compare authoring output back to the pinned native requirements before visual handoff. A passing trial against a newly chosen baseline cannot justify changed grouping keys or invented unit metadata; correct affected definitions through the owning task first.
- Do not make reference-SQL success a prerequisite for metrics, or metric success a prerequisite for reference SQL.
- Report metric partitions matched to other Datus datasources as pending; process them only in a later run after the user activates that datasource. Never silently switch the shared datasource.

For visual output, complete semantic validation and reconciliation, then rediscover exact metric identities and the active model revision before reading [visual-dashboard.md](visual-dashboard.md). Never launch visual generation concurrently with a metric task it depends on. An empty or failed discovery after authoring is a pending/failed metric phase, not permission to fall back to SQL.
