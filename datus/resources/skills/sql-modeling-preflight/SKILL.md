---
name: sql-modeling-preflight
description: Prepare one request-local SQL modeling plan before semantic authoring
tags:
  - semantic-model
  - metrics
  - sql
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_semantic_model
  - gen_metrics
---

# SQL Modeling Preflight

Run this phase only when the current request contains SQL. Existing-artifact
maintenance and natural-language-only authoring skip this skill's tool call.

1. Inspect the complete current user request and count every SQL statement in request order.
2. If the request contains no SQL, do not call `prepare_sql_modeling_plan`; continue with the active authoring workflow.
3. If the request contains SQL, call `prepare_sql_modeling_plan` exactly once with one metadata entry per statement:
   - `source_index`: its 1-based position in the request.
   - `name`: a concise, meaningful English snake_case business name inferred from the question and SQL.
   - `question`: preserve the supplied business question verbatim. Infer a concise question only when none was provided.
   - Do not copy SQL into the tool call. The tool extracts and owns the exact request text.
4. Do not write or edit files while the returned status is `pending` or `unresolved`. Fix the submitted evidence or report the blocker.
5. Treat the returned `candidate_plan` as authoritative:
   - `output_contracts` define every final query output and classify it as `direct`, `query_backed`, `dimension`, or `non_metric`.
   - `metric_requirements` define the output-id-scoped metric completeness contract. Evaluate each requirement independently; one SQL statement may contain both direct and query-backed metric outputs.
   - `dataset_requirements` define query-backed datasets. Their exact SQL stays request-local inside the tool layer.
   - `queryability_contracts` define the complete source `GROUP BY` combinations that generated metrics must compile and execute with.
   - Query-level classifications are summaries only. They never override an output contract or force a directly lowerable sibling output through a query-backed dataset.
   - Reusable candidates and the existing metric catalog may reduce duplicate dependencies, but never remove a required final output.
   - For SQL-backed authoring, use returned `semantic_source_evidence` as the combined physical schema, relationship, and request-SQL field-usage inspection. Call `inspect_semantic_sources` only when this evidence is partial or additional physical tables are required.
6. Use authored business names for datasets and metrics. Fingerprints and requirement identifiers are internal identities and must never become artifact names.
7. For a query-backed requirement, pass its `dataset_requirement_id` to `upsert_osi_datasets` and omit `source`; the tool injects the exact SQL. Reuse an existing dataset only when its complete source SQL exactly matches. A same-named dataset with different SQL is a conflict; choose another meaningful name rather than overwriting it.
8. Follow only the active format's authoring skill. If the plan requires a capability that the active format cannot execute, return a concrete blocker instead of emitting instructions or artifacts for another format.

The preflight is request-local. Do not depend on another node run or a cache created by `gen_semantic_model`, `gen_metrics`, `/build-kb`, CLI, or bootstrap.
