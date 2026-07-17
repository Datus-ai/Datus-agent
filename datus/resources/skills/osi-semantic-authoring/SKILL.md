---
name: osi-semantic-authoring
description: OSI core schema semantic model authoring specification — field roles, structural keys, Datus extension hints, relationships, and validation
tags:
  - semantic-model
  - osi
version: "2.0.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_semantic_model
  - gen_metrics
---

# OSI Semantic Authoring

Describe tables as strict **OSI (Open Semantic Interchange) core schema** documents plus Datus business hints.

CRITICAL BOUNDARY: You author **OSI core semantics only**. You do NOT write MetricFlow `data_source:`, `measures:`, `identifiers:`, `agg_time_dimension`, `create_metric`, or any execution-engine YAML. The Datus OSI compiler lowers OSI core documents to the configured backend.

The OSI expression dialect, target semantic model name, and target semantic model file for the current run are shown in the system prompt Workspace section — use those exact values (`<osi_dialect>` below stands for that dialect).

## Field roles — the three-way decision

OSI core separates three column roles by **structure**, not by type labels:

| Role | How it is declared | Examples |
|------|--------------------|----------|
| **Dimension** — used for GROUP BY / filtering | A field **with** a `dimension:` block | code columns, names, statuses, dates |
| **Key** — identifies rows / joins | Dataset `primary_key` / `unique_keys`, relationship `from_columns`/`to_columns`. NOT a field type. | declared PK columns, FK join columns |
| **Measure source** — only aggregated | A field **without** a `dimension:` block (just name/expression/description) | balances, amounts, quantities, precomputed rates |

The presence of the `dimension:` block IS the dimension declaration. A field without the block is a plain row-level expression: it documents the column and may back metric expressions, but it is not exposed for grouping (`get_dimensions` will not list it). **NEVER write `dimension: {is_time: false}` to mean "this is not a time column" — omit the entire block for non-dimension fields.**

## What you produce

One valid OSI core document for the current business domain / semantic model scope. The authoritative document shape — object structures, field lists, `version`, and the Datus execution subset notes — is the **OSI Core Authoring Specification** section of the system prompt; author against it exactly. This skill adds the decision rules the specification cannot express: which columns become which role, and how to fix validation conflicts.

## Authoring rules

1. **Root schema is fixed.** Root keys are only `version` and `semantic_model`. `semantic_model` is a list. Do NOT write top-level `datasets:`, `relationships:`, or `metrics:`.
2. **Use OSI core dataset shape.** Dataset `source` is a string, not `{table: ...}`. Dataset columns are `fields`, not `dimensions`. Field expressions are `expression.dialects[]`, not `expr`. Use the exact OSI expression dialect from the system prompt in every `expression.dialects[].dialect`.
3. **Datus-only hints go into `custom_extensions`.** The only field-level hint is `time_granularity` (on the time field). Dataset-level: `source_type: "query"` for query sources. Do NOT emit field `type` hints — roles are expressed structurally per the table above.
4. **Semantic model boundary.** One OSI `semantic_model` represents the current business domain. Put all related logical datasets needed by the provided SQL history in this semantic model, with relationships declared once under the semantic model object.
5. **Canonical logical datasets.** For the same source and row grain, create one canonical dataset that metrics can reference by name. Create a separate dataset only when the logical row grain or fixed business scope is genuinely different.
6. **Dataset `description` and `ai_context` are required.** `description`: one concise human sentence with the business entity and row grain. `ai_context.instructions`: when to use the dataset, the row grain (spell out the full grain explicitly — this is where grain lives when no primary key is declared), the primary time field, important row-selection columns, relationship caveats.
7. **Keys are transcribed, never inferred.** Write `primary_key` ONLY when the source metadata explicitly declares one: a `PRIMARY KEY` in the DDL, or `pk: true` columns in `describe_table` output. If the source declares no key — the normal case for warehouse tables — **omit `primary_key` entirely**; do not guess from column names, comments, or data. The same applies to `unique_keys` (unique constraints/indexes only). Exceptions: ClickHouse `PRIMARY KEY`/`ORDER BY` and StarRocks `DUPLICATE KEY` in DDL are **sort keys, not uniqueness** — never transcribe them (a StarRocks `PRIMARY KEY` table model is a true upsert key and may be transcribed).
8. **Field selection — decide by role, not by listing every column:**
   - Code / name / status / label columns the SQL groups or filters by → field **with** `dimension: {}` block.
   - The primary date/time column → field with `dimension: {is_time: true}` plus `{"time_granularity":"day|week|month|quarter|year"}` hint. Point it at a real date/time column, never a numeric surrogate key.
   - Columns whose comments/usage indicate measured quantities (balance, amount, quantity and their equivalents in the comment language) and that are only aggregated → field **without** a `dimension:` block (name/expression/description only); metric expressions reference them by physical column name.
   - Precomputed ratio columns (rate, ratio, percent and equivalents) → field **without** a `dimension:` block; note in its `description` that the metrics workflow recomputes weighted ratios from the numerator/denominator columns instead of aggregating this column.
   - Declared key columns (rule 7) → they live in `primary_key`/`unique_keys`/relationships; also add a `dimension: {}` field ONLY when queries genuinely group by them.
   - Columns no provided SQL uses and that carry no key/time role → omit.
   - Populate `description` for all non-obvious fields from column comments, sample values, and profiler evidence; keep original language, do not translate.
9. **Time dimension**: exactly one primary time field per dataset. When several date columns exist and the primary one is ambiguous, ASK before generating. **Verify `time_granularity` with data**: run one query such as `SELECT COUNT(DISTINCT <time_col>), MIN(<time_col>), MAX(<time_col>) FROM <table>` and derive the snapshot interval (e.g. month-end dates spanning months → `month`; consecutive dates → `day`). When the data is indeterminate (a single distinct date), fall back to the table/column comments (e.g. a "monthly statistics" table comment → `month`), else default to `day`.
10. **Validation conflicts are fixed structurally.** If `validate_semantic` reports an element lowering to multiple types, follow the structural fix in the message: move the column into `primary_key`/a relationship everywhere, or give it a `dimension:` block everywhere, or drop the `dimension:` block in datasets that only aggregate it. Never bounce a column between roles across validation attempts, and never falsify keys to silence the validator (e.g. do not delete a snapshot date from a declared composite key — the compiler resolves that case automatically).
11. **Relationships** live inside the semantic model object, never inside a dataset. Use OSI core fields `from`, `to`, `from_columns`, `to_columns`. Do NOT use non-core fields such as `from_dataset`, `from_identifier`, `join_on`, `from_column`, or `to_column`.
12. Do NOT add metrics in the semantic-model step. Metrics are added by the metrics workflow under `semantic_model[0].metrics`.
13. Preserve literal values and column names exactly; do not invent columns. Keep column comments in their original language — do not translate.

## Worked example — monthly snapshot table

Input (describe_table): `branch_loan_quality_monthly` — a **monthly** loan-quality statistics table, no declared primary key, columns: `snapshot_date` (date), `branch_no`/`assess_dim_code`/`scope_code` (varchar code columns), `branch_name` (varchar), `loan_balance`/`npl_balance`/`overdue_balance` (numeric balances), `npl_rate`/`overdue_rate` (numeric precomputed rates).

Correct field layout:

```yaml
        # no primary_key: the source declares none — the grain ("one row per month per branch per assessment dimension per scope") goes in ai_context.instructions
        fields:
          - name: snapshot_date
            expression: {dialects: [{dialect: <osi_dialect>, expression: snapshot_date}]}
            dimension: {is_time: true}
            custom_extensions: [{vendor_name: DATUS, data: '{"time_granularity":"month"}'}]
          - name: branch_no
            expression: {dialects: [{dialect: <osi_dialect>, expression: branch_no}]}
            dimension: {}
          - name: assess_dim_code
            expression: {dialects: [{dialect: <osi_dialect>, expression: assess_dim_code}]}
            dimension: {}
          - name: scope_code
            expression: {dialects: [{dialect: <osi_dialect>, expression: scope_code}]}
            dimension: {}
          - name: branch_name
            expression: {dialects: [{dialect: <osi_dialect>, expression: branch_name}]}
            dimension: {}
          - name: loan_balance                    # aggregation-only: field WITHOUT dimension block
            expression: {dialects: [{dialect: <osi_dialect>, expression: loan_balance}]}
            description: "Loan principal balance; aggregated by metrics"
          # npl_balance / overdue_balance: same plain-field shape as loan_balance
          - name: npl_rate                        # precomputed ratio: plain field, never aggregated directly
            expression: {dialects: [{dialect: <osi_dialect>, expression: npl_rate}]}
            description: "Precomputed row-level NPL ratio; metrics recompute the weighted ratio from npl_balance / loan_balance"
          # overdue_rate: same plain-field shape as npl_rate
```

WRONG (do not do this): declaring `loan_balance` or `npl_rate` as fields **with** a `dimension:` block (or `dimension: {is_time: false}`); inventing `primary_key: [branch_no, ...]` when the DDL declares none; adding `{"type":"numeric"}` hints.

## Workflow notes

- Write OSI core YAML under the semantic model directory shown in the system prompt only, at the target semantic model file path.
- Inspect the table schema and comments (`describe_table` reports `pk`/`nullable` facts when the database declares them); map columns to roles per the table above.
- When a critical modeling choice is ambiguous (which column set is the grain, which is the primary time dimension), ASK before generating.
- Call `validate_semantic(scope="semantic_model")` after writing the OSI semantic model and fix errors until it passes; treat warnings about "aggregates column X which is also a dimension" as instructions to drop that field's `dimension:` block or the field itself.
- After validation passes, call `end_semantic_model_generation(semantic_model_files=[...])`. In OSI mode this syncs OSI datasets to the Knowledge Base without using MetricFlow YAML.
