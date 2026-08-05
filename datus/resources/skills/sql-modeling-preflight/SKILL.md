---
name: sql-modeling-preflight
description: Prepare one request-local SQL modeling plan before semantic authoring
tags:
  - semantic-model
  - metrics
  - sql
version: "1.3.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_semantic_model
  - gen_metrics
---

# SQL Modeling Preflight

Run this phase when the current request contains SQL or explicitly names a
readable SQL file. Existing-artifact maintenance and natural-language-only
authoring skip this skill's tool call.

1. Inspect the complete current user request. If it explicitly names one or more SQL files, call `read_file` for those paths before preflight and use their complete contents. This is the only artifact read allowed before preflight; do not discover unrelated SQL files.
2. If the request neither contains SQL nor explicitly names a SQL file, do not call `prepare_sql_modeling_plan`; continue with the active authoring workflow.
3. If SQL was provided directly or loaded from a named file, submit one entry per statement to `prepare_sql_modeling_plan`:
   - `source_index`: its 1-based position in the request.
   - `name`: an optional business label. Omit it when no clear name is available; the planner generates a stable name from `source_index` and disambiguates repeated labels.
   - `question`: preserve the supplied business question verbatim. Infer a concise question only when none was provided.
   - `sql`: copy the complete statement verbatim from the current request or `read_file` result, including comments, hints, whitespace, and the statement terminator when present. Do not reformat, reconstruct, truncate, or invent SQL.
   - Prefer one call. For a large input, use a few complete-statement batches with `finalize=false`, then set `finalize=true` on the last batch. Never split one SQL statement across batches.
   - Keep `source_index` unique across batches and continue numbering from the complete input. Repeating the same index is accepted only when the complete entry is identical.
   - Submit every statement once in source order. If the tool rejects an entry, re-check the source content; never add a placeholder entry merely to satisfy a count.
4. Continue submitting batches while the tool returns `collecting`; proceed when it returns `ready` or `partial`. `finalize=true` closes source collection, not plan correction. A partial plan exposes failed sources in `unresolved_sources`. Fix an `unresolved` submission or report the blocker.
5. Use the compact plan as an editable draft:
   - `sources` retain the original SQL and question as evidence.
   - `outputs` contain stable `output_id`, source, name, expression, and an editable `role` (`metric`, `dimension`, or `non_metric`). Every original output ID must remain, but its modeling decision may change.
   - `queryability_contracts` contain only `contract_id`, related metric output IDs, the complete semantic dimension combination, and optional time grain. Dimensions may be changed to qualified names such as `activity_monthly_mom_detail.start_month`.
   - Optional `generated_sql` contains corrected query-backed SQL keyed by source ID. Original `source_sql` remains unchanged.
6. When authoring, compiler feedback, or warehouse dry-run reveals a wrong role, expression, SQL, dataset choice, qualified dimension, or time grain, call `update_sql_modeling_plan` with the complete corrected compact plan. Updating invalidates previous validation and dry-run evidence. Successful publication locks further changes.
7. Bind every metric-role output ID to a generated or reused metric. Equivalent outputs may share one metric. Publish validates every retained GROUP BY combination using the current qualified dimensions.
8. Query the live semantic model, schema, and metric catalog with their discovery tools when needed; they are not frozen into the plan.
9. Follow only the active format's authoring skill. If the active format cannot execute the corrected plan, return a concrete blocker.

The preflight is request-local. Do not depend on another node run or a cache created by `gen_semantic_model`, `gen_metrics`, `/build-kb`, CLI, or bootstrap.
