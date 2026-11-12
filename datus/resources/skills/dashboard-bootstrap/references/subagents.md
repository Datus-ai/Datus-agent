# Optional Dashboard Subagents

Read only when subagent persistence is selected. For a visual-only request, do not create additional agent configuration unless requested. Legacy context bootstrap retains this final step when available.

## Step 8 — Create dashboard subagents when supported

Run this optional step only after all reference-SQL and semantic-modeling tasks for the active datasource have finished.

1. Inspect the skills available to the current main agent for `create-subagent`.
2. If it is absent, report `subagent creation skipped: mutable configuration skill unavailable`. Do not edit configuration by another mechanism, and do not change the success status of context construction.
3. If it is present, load `create-subagent`. If loading is refused, skip creation and report the refusal without attempting a direct write.
4. Build one scoped context per selected dashboard and datasource partition from that dashboard's selected successful artifacts only. Retain the source-dashboard-to-metric mapping from planning/discovery, including verified reused metrics; a shared domain model is not that dashboard's selection:
   - `datasource`: the active, uniquely matched Datus datasource;
   - `tables`: complete identifiers of the tables required by that dashboard's selected metrics/queries, verified through `semantic_modeling`, rather than every table in a shared model;
   - `metrics`: exact canonical subject references, each formed from the synchronized metric's stored `subject_path` plus its `name`;
   - `sqls`: exact canonical subject references, each formed from the synchronized reference SQL's `subject_tree` plus its `name`.
5. Derive and validate the subject references rather than treating task outputs as ready-to-store identifiers:
   - include only metrics selected for this dashboard or explicitly confirmed in its scope, plus their required dependencies; do not enumerate every metric in a shared domain model;
   - for each included authored or reused metric, verify its actual post-sync subject path and name through maintained discovery/subject-tree evidence, using returned model files when applicable; write `<metric.subject_path>.<metric.name>`;
   - a verified `skipped/no_semantic_change` result can supply existing metrics even when `semantic_models=[]`. Keep the verified chart coverage and resolve those exact maintained metric references; an empty authored-file list does not mean an empty metric scope;
   - for each successful `gen_sql_summary` result, read its returned `sql_summary_file` and write `<subject_tree>.<name>`;
   - apply Datus reference-path quoting to every segment and confirm each exact leaf resolves in the matching post-sync subject tree;
   - never store metric IDs, bare metric names, SQL summary IDs, artifact file paths, checksums, or plugin query IDs in `metrics` or `sqls`;
   - never use a bare subject path unless the confirmed scope explicitly requests the entire subtree.
6. Exclude failed, pending, unselected, unconfirmed, unresolved or ambiguous artifacts, and skipped work that supplies no verified reusable artifact. If `tables`, `metrics`, and `sqls` are all empty, skip creation. If any selected metric or exact subject reference remains unresolved, leave an existing pair unchanged and report persistence blocked. Successful reference SQL must not replace a selected or verified reused metric scope with SQL-only visibility; SQL-only scope requires an explicitly reference-only selection.
7. Derive the base name with the legacy Dashboard convention:
   - extract ASCII alphanumeric or contiguous CJK tokens, lowercase ASCII tokens, and join them with underscores;
   - use `bi` when the plugin/platform produces no token and `dashboard` when the title produces no token;
   - keep at most the first three dashboard-title tokens;
   - join them as `<platform>_<dashboard>`;
   - if the result does not start with a letter, prefix it with `dashboard_`.
8. Before writing either node of any pair, resolve names for the complete selected
   dashboard/datasource-partition set against its stable dashboard IDs and the active configuration's
   existing node names. Reserve both `<base-name>` and `<base-name>_attribution`:
   check title-normalization collisions, title-vs-ID/fallback aliases and collisions
   between either member of different pairs. Never use a title alone as ownership
   evidence for an existing node.
   - Keep the legacy name unchanged when neither member collides.
   - For a collision, append a deterministic identifier suffix derived from the
     full plugin/profile/site/dashboard ID **and datasource partition** to the legacy base. Extend the suffix
     until both members are unique against the whole selection, its dashboard-ID
     aliases, reserved builtin names and unrelated existing nodes. The suffix does
     not change the three-token normalization or prefix rules for the base.
   - Reuse an existing pair only when a persisted mapping or verified provenance
     identifies both the same source dashboard and the same datasource partition.
     A pair owned by another datasource is not reusable, even for the same dashboard;
     choose a distinct name and never overwrite that partition's `scoped_context`.
     If ownership is unknown, choose fresh
     names rather than replacing that node's scope. Retain and report the exact
     (source ID, datasource partition)-to-pair mapping so retries resolve to the same pair.
   - Build each pair's scope from that dashboard's own successful selected artifacts
     and active-datasource partition; never reuse the last processed dashboard's
     scope for another pair. Validate the complete mapping before either write.
9. Ask `create-subagent` to create or update both scoped nodes:

```text
name: <base-name>
node_class: gen_sql
agent_description: <dashboard description, falling back to its title>
tools: context_search_tools,db_tools.search_table,db_tools.describe_table,db_tools.execute_sql
scoped_context: <successful active-datasource artifacts>

name: <base-name>_attribution
node_class: gen_visual_report
agent_description: Attribution analysis for <dashboard description, falling back to its title>
tools: semantic_tools.*,db_tools.*,context_search_tools.*
scoped_context: <the same successful active-datasource artifacts>
```

For a dashboard spanning multiple datasources, create or update nodes only for the partition processed against the active datasource in this run. Never mix context from another datasource into these nodes. A creation failure does not invalidate context already built; retry only the configuration step after correcting the reported cause.
