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
   - `name`: a concise, meaningful English snake_case business name inferred from the question and SQL.
   - `question`: preserve the supplied business question verbatim. Infer a concise question only when none was provided.
   - `sql`: copy the complete statement verbatim from the current request or `read_file` result, including comments, hints, whitespace, and the statement terminator when present. Do not reformat, reconstruct, truncate, or invent SQL.
   - Prefer one call. For a large input, use a few complete-statement batches with `finalize=false`, then set `finalize=true` on the last batch. Never split one SQL statement across batches.
   - Keep `source_index` unique across batches and continue numbering from the complete input. Repeating the same index is accepted only when the complete entry is identical.
   - Submit every statement once in source order. If the tool rejects an entry, re-check the source content; never add a placeholder entry merely to satisfy a count.
4. Continue submitting batches while the tool returns `collecting`; proceed only when it returns `ready`. Fix an `unresolved` submission or report the blocker. Natural-language-only requests skip this phase entirely.
5. Treat the returned `candidate_plan` as authoritative:
   - `output_contracts` define every final query output and classify it as `direct`, `query_backed`, `dimension`, or `non_metric`.
   - `metric_requirements` define the output-id-scoped metric completeness contract. Evaluate each requirement independently; one SQL statement may contain both direct and query-backed metric outputs.
   - `dataset_requirements` define query-backed datasets. Their exact SQL stays request-local inside the tool layer.
   - `queryability_contracts` define the complete source `GROUP BY` combinations that generated metrics must compile and execute with.
   - Query-level classifications are summaries only. They never override an output contract or force a directly lowerable sibling output through a query-backed dataset.
   - Reusable candidates and the existing metric catalog may reduce duplicate dependencies, but never remove a required final output.
   - For SQL-backed authoring, use returned `semantic_source_evidence` as the combined physical schema, relationship, and request-SQL field-usage inspection. Call `inspect_semantic_sources` only when this evidence is partial or additional physical tables are required.
6. Use authored business names for datasets and metrics. Fingerprints and requirement identifiers are internal identities and must never become artifact names.
7. For a query-backed requirement, pass its `dataset_requirement_id` to `upsert_osi_datasets` and omit `source`; the tool injects the exact SQL. Prefer reusing an existing dataset whose complete source SQL exactly matches. A valid current-request requirement may update a same-named query-backed dataset to the planner-owned SQL; without that evidence, different SQL remains a naming conflict.
8. Follow only the active format's authoring skill. If the plan requires a capability that the active format cannot execute, return a concrete blocker instead of emitting instructions or artifacts for another format.

The preflight is request-local. Do not depend on another node run or a cache created by `gen_semantic_model`, `gen_metrics`, `/build-kb`, CLI, or bootstrap.
