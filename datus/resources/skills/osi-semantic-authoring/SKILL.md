---
name: osi-semantic-authoring
description: OSI core schema semantic model authoring specification — field roles, structural keys, Datus extension hints, relationships, and validation
tags:
  - semantic-model
  - osi
version: "2.1.0"
user_invocable: false
disable_model_invocation: false
allowed_agents:
  - gen_semantic_model
  - gen_metrics
---

# OSI Semantic Authoring

Describe tables as strict **OSI (Open Semantic Interchange) core schema** documents plus Datus business hints.

CRITICAL BOUNDARY: You author **OSI core semantics only**. You do NOT write MetricFlow `data_source:`, `measures:`, `identifiers:`, `agg_time_dimension`, `create_metric`, or any execution-engine YAML. The Datus OSI compiler lowers OSI core documents to the configured backend.

The OSI expression dialect is shown in the system prompt Workspace section. Plan the target model name and file with `plan_osi_semantic_model_target` after identifying the fact and dimension tables; use the returned values exactly (`<osi_dialect>` below stands for that dialect).

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
3. **Datus-only hints go into `custom_extensions`.** The only field-level hint is `time_granularity` (on the time field). Dataset-level: `source_type: "query"` for query sources. Relationships use the native OSI core `name` as the stable joined-dimension path prefix; do not add a relationship alias extension. Do NOT emit field `type` hints — roles are expressed structurally per the table above.
4. **Semantic model boundary.** One OSI `semantic_model` represents the current business domain. Put all related logical datasets needed by the provided SQL history in this semantic model, with relationships declared once under the semantic model object.
5. **Canonical logical datasets.** For the same source and row grain, create one canonical dataset that metrics can reference by name. Create a separate dataset only when the logical row grain or fixed business scope is genuinely different.
6. **Dataset `description` and `ai_context` are required.** `description`: one concise human sentence with the business entity and row grain. `ai_context.instructions`: when to use the dataset, the row grain (spell out the full grain explicitly — this is where grain lives when no primary key is declared), the primary time field, important row-selection columns, relationship caveats.
7. **Separate physical primary keys from verified logical keys.**
   - Write `primary_key` ONLY when source metadata or an explicit data contract declares it: a `PRIMARY KEY` in DDL, or `pk: true` columns in `describe_table`. Historical SQL and profiling must never manufacture a physical primary key.
   - Transcribe declared unique constraints/indexes directly into `unique_keys`.
   - A request-SQL JOIN, ETL pattern, column name, or stated row grain may propose an **ordered candidate logical key**. It is not yet a key. Submit every candidate you intend to author in one `validate_semantic_key_candidates(candidates=[...])` call. Add an ordered list as one `unique_keys` entry only when its result has `verification_scope: full_table`, `is_valid_logical_key: true`, and `recommended_osi_declaration: unique_keys`. The scan covers rows visible to the current datasource principal; if row-level policy limits that principal, require an explicit data contract or an unrestricted verification before declaring a global key. Mention the verification evidence in `ai_context.instructions`.
   - If the table is empty, any key component is NULL, any duplicate group exists, or verification fails/cannot run, omit the candidate from both `primary_key` and `unique_keys`. Do not validate only one snapshot/partition and generalize it to the whole table.
   - ClickHouse `PRIMARY KEY`/`ORDER BY` and StarRocks `DUPLICATE KEY` are sort keys, not uniqueness. Never transcribe them. A StarRocks `PRIMARY KEY` table model is a true upsert key and may be transcribed.
8. **Field selection — decide by role, not by listing every column:**
   - Code / name / status / label columns the SQL groups or filters by → field **with** `dimension: {}` block.
   - Consume every non-time `queryability_contracts.dimension_expr_hints` entry from preflight. For a derived `expr`, author a dimension whose `name` is the hint `alias` and whose executable field expression preserves that SQL expression. For a direct column or pure column alias, an equivalent dimension over that source column is sufficient. A raw dependency column is not equivalent to a derived expression, and mentioning an expression in `description` does not make it queryable.
   - The primary date/time column → field with `dimension: {is_time: true}` plus `{"time_granularity":"day|week|month|quarter|year"}` hint. Point it at a real date/time column, never a numeric surrogate key.
   - Columns whose comments/usage indicate measured quantities (balance, amount, quantity and their equivalents in the comment language) and that are only aggregated → field **without** a `dimension:` block (name/expression/description only); metric expressions reference them by physical column name.
   - Precomputed ratio columns (rate, ratio, percent and equivalents) → field **without** a `dimension:` block; note in its `description` that the metrics workflow recomputes weighted ratios from the numerator/denominator columns instead of aggregating this column.
   - Declared key columns (rule 7) → they live in `primary_key`/`unique_keys`/relationships; also add a `dimension: {}` field ONLY when queries genuinely group by them.
   - Columns no provided SQL uses and that carry no key/time role → omit.
   - Populate `description` for all non-obvious fields from column comments, sample values, and profiler evidence; keep original language, do not translate.
9. **Time dimension**: exactly one primary time field per dataset. When several date columns exist and the primary one is ambiguous, ASK before generating. **Verify `time_granularity` with data**: run one query such as `SELECT COUNT(DISTINCT <time_col>), MIN(<time_col>), MAX(<time_col>) FROM <table>` and derive the snapshot interval (e.g. month-end dates spanning months → `month`; consecutive dates → `day`). When the data is indeterminate (a single distinct date), fall back to the table/column comments (e.g. a "monthly statistics" table comment → `month`), else default to `day`.
10. **Validation conflicts are fixed structurally.** If `validate_semantic` reports an element lowering to multiple types, follow the structural fix in the message: move the column into a verified key/relationship everywhere, give it a `dimension:` block everywhere, or drop the `dimension:` block in datasets that only aggregate it. Never bounce a column between roles across validation attempts, and never falsify keys to silence the validator.
11. **Relationships** live inside the semantic model object, never inside a dataset. Use OSI core fields `from`, `to`, `from_columns`, `to_columns`. The lists may contain one or more columns; they must have equal lengths and their order defines component correspondence. `to_columns` must exactly equal the target dataset's complete `primary_key` or one complete `unique_keys` entry — never join to a subset of a composite key. Collect every unverified target candidate returned by `inspect_semantic_sources` and verify all candidates you intend to use in one `validate_semantic_key_candidates` call before authoring relationships. Do NOT use non-core fields such as `from_dataset`, `from_identifier`, `join_on`, `from_column`, or `to_column`.
12. Do NOT add metrics in the semantic-model step. Metrics are added by the metrics workflow under `semantic_model[0].metrics`.
13. Preserve literal values and column names exactly; do not invent columns. Keep column comments in their original language — do not translate.
14. **Honor output-level query-backed requirements.** When the request-local SQL plan contains `dataset_requirements`, create only those datasets required by `query_backed_metric` outputs; directly lowerable sibling outputs do not belong in them. Match an existing dataset only when its `source` fingerprint matches the requirement. Otherwise choose a meaningful business name and call `upsert_osi_datasets` with `dataset_requirement_id` on that dataset object while omitting `source`; the tool injects the exact original SQL and DATUS query-source extension. Fingerprints and requirement IDs are internal identities and must never become authored names. Never overwrite a same-named query-backed dataset whose source SQL differs. Use `output_grain` fields as dimensions and required metric output fields as plain fields. Do not expose internal CTE columns that are absent from `output_contracts`.
15. **Use the narrow dataset mutation tool.** After planning the target, call `upsert_osi_datasets` with the first non-empty dataset batch; it creates a missing target as a complete valid document. For an existing model, use the same upsert so unrelated datasets, relationships, metrics, and model metadata are preserved. Use `edit_file` only for relationships or model metadata that the dataset upsert does not own. Never create an empty semantic-model shell.

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

WRONG (do not do this): declaring `loan_balance` or `npl_rate` as fields **with** a `dimension:` block (or `dimension: {is_time: false}`); inventing `primary_key: [branch_no, ...]` when the DDL declares none; promoting JOIN columns to `unique_keys` without a passing full-table candidate-key verification; adding `{"type":"numeric"}` hints.

## Workflow notes

- Resolve the target before writing. Priority: an explicit user-provided semantic model name; an existing model containing the core fact table; an inferred business domain; the core fact table as fallback.
- Put the core fact table first in `fact_tables`. Pass dimensions separately; dimension tables never participate in naming.
- When the resolver returns an existing file, preserve its semantic model name permanently, even when adding dimensions. Read and update that file instead of creating a renamed model.
- For a new resolved target, call `upsert_osi_datasets` directly; the first non-empty batch creates the document atomically. For an existing target, read it first and use narrow mutations.
- For an existing target, keep its current semantic model name and preserve all unrelated datasets, relationships, and metrics. Add or update datasets with `upsert_osi_datasets`; never replace the file with a partial document containing only the requested objects.
- Use preflight `semantic_source_evidence` as the normal combined inspection result. Call `inspect_semantic_sources` at most once only for partial/no-SQL inspection or additional physical tables. Its schema records include declared `pk`/`nullable` facts and its SQL usage comes only from the exact request captured by preflight.
- Treat `source_columns` / `target_columns` returned by relationship discovery as one ordered composite when `key_arity > 1`. Put every complete target list you intend to use into the single batch `validate_semantic_key_candidates` call; never validate or author its components independently.
- When a critical modeling choice is ambiguous (which column set is the grain, which is the primary time dimension), ASK before generating.
- Call `validate_semantic(scope="semantic_model", semantic_model_name="<planned semantic model name>")` without a custom `checks` subset after writing or editing the OSI semantic model, and fix errors with `edit_file` until the adapter's complete default profile passes; treat warnings about "aggregates column X which is also a dimension" as instructions to drop that field's `dimension:` block or the field itself.
- After validation passes, call `publish_semantic_model(semantic_model_files=[...])`. In OSI mode this syncs OSI datasets to the Knowledge Base without using MetricFlow YAML.
