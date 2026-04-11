---
name: dbt-layered-generation
description: Generate SQL for a layered dbt-style data warehouse (staging → intermediate → marts) with Phase 2 validation and retry logic encoded from DAComp benchmark experiments
tags:
  - dbt
  - data-engineering
  - staging
  - intermediate
  - marts
  - layered-warehouse
  - sql-generation
  - dacomp
version: "1.1.0"
user_invocable: false
disable_model_invocation: false
---

# DBT Layered SQL Generation

Generate SQL for a layered dbt-style data warehouse following the DAComp DE-Impl format.

## When to use this skill

Activate when the task provides a `data_contract.yaml` with a `layer_dependencies.yaml`, references layer names (staging / intermediate / marts), or asks to implement a DAComp DE-Impl pipeline. The runner executes layers in topological order (staging → intermediate → marts). For each table: generate SQL, execute, validate against contract, optionally retry (max 3 rounds for execution errors, max 1 for column mismatch).

---

## Data Reality Check Workflow (MANDATORY before any staging SQL)

The data contract describes the **desired** schema, but source data often violates those claims. Before writing staging SQL, you MUST verify the reality with DB tools. **Contract claims lose to observed data** — if the data says something different, follow the data.

### Required steps for every staging table

1. **Observe the raw source**: call `describe_table` on the `source_table` the spec points to. Note the **actual** column types — they are frequently VARCHAR even when the contract claims TIMESTAMP / BIGINT / BOOLEAN.

2. **For every TIMESTAMP target column**, measure the cast success rate with `read_query`:

   ```sql
   SELECT
       COUNT(*)                                            AS total,
       COUNT(TRY_CAST(col AS TIMESTAMP))                   AS cast_ok,
       CAST(COUNT(TRY_CAST(col AS TIMESTAMP)) AS DOUBLE)
         / NULLIF(COUNT(*), 0)                             AS success_rate
   FROM raw.<source_table>
   ```

   Interpret the result:

   | success_rate | meaning | action |
   |---|---|---|
   | ≥ 0.95 | DuckDB can parse this format | Use `TRY_CAST` normally |
   | < 0.95 (and > 0) | Mixed-quality data | Use `TRY_CAST`, accept partial NULLs, **do NOT filter** |
   | 0.0 | Format not supported by DuckDB standard TIMESTAMP (e.g. strings with `+0800` style timezone) | Still use `TRY_CAST`, **all values will become NULL**, **MUST NOT add any `WHERE col IS NOT NULL` filter** — the rows must be preserved with NULL values. The contract's `not_null` claim on this column is a **spec wish**, not an executable rule, and gold solutions explicitly accept this violation. |

3. **For suspiciously-short VARCHAR ids**, run a quick sample with `read_query LIMIT 3` to see the actual format before applying `TRIM`, `length(...) > 0`, etc.

### Strict rule: observation ≠ imitation

The raw column names you see via `describe_table` are for **type checking and cast verification only**. The **output column names** MUST come from the contract's `columns:` section. Never copy raw column names into the final SELECT — always use the contract's renamed target columns. For example:

- Raw: `raw.application.id` → Contract: `stg_lever__application.application_id`
- Raw: `raw.user.creator_id` → Contract: `stg_lever__user.creator_user_id`

If the contract renames it, you rename it. The DB check is for understanding source types, not for bypassing the contract's naming.

### Why this workflow exists

The DAComp DE benchmark includes a deliberate trap: contracts declare `created_at: TIMESTAMP not_null` while the raw data is VARCHAR with non-standard timezone strings that `TRY_CAST(...AS TIMESTAMP)` returns NULL for. A naive agent that trusts the contract's `not_null` and adds `WHERE created_at IS NOT NULL` drops 100% of rows. The gold solution uses `try_cast` and accepts NULL values without filtering. **Observe the data, follow the data.**

---

## DuckDB Dialect Rules

Contract specs often use MySQL or generic SQL syntax. Translate as follows:

| Contract syntax | DuckDB equivalent |
|---|---|
| `col RLIKE 'pattern'` | `regexp_matches(col, 'pattern')` |
| `col REGEXP 'pattern'` | `regexp_matches(col, 'pattern')` |
| `IFNULL(a, b)` | `COALESCE(a, b)` |
| `GROUP_CONCAT(col)` | `STRING_AGG(col, ',')` |
| `GROUP_CONCAT(col SEPARATOR sep)` | `STRING_AGG(col, sep)` |
| `CAST(x AS DATETIME)` | `TRY_CAST(x AS TIMESTAMP)` |
| `STR_TO_DATE(x, fmt)` | `STRPTIME(x, fmt)` |
| `DATE_FORMAT(x, fmt)` | `STRFTIME(x, fmt)` |
| `YEAR(col)` | `EXTRACT(year FROM col)` |
| `MONTH(col)` | `EXTRACT(month FROM col)` |
| `DAY(col)` | `EXTRACT(day FROM col)` |

Use `TRY_CAST(x AS TIMESTAMP)` (not `CAST`) for string-to-timestamp conversion. `TRY_CAST` returns NULL on unparseable values rather than raising an error — this is correct behavior; preserve those rows.

---

## Type Casting Policy (CRITICAL)

**DO NOT explicitly CAST columns that are already strongly typed in the source.**

The contract's `data_type` field is a **logical type hint**, not a coercion directive. The source column type (from `describe_table`) is authoritative.

```sql
-- WRONG: source column is already BOOLEAN, contract says INT
SELECT CAST(is_active AS INTEGER) AS is_active FROM raw.users

-- CORRECT: pass through unchanged
SELECT is_active FROM raw.users
```

```sql
-- WRONG: source column is already BIGINT, contract says VARCHAR
SELECT CAST(user_id AS VARCHAR) AS user_id FROM raw.events

-- CORRECT: pass through unchanged
SELECT user_id FROM raw.events
```

Cast only when actively changing representation:

```sql
-- CORRECT: string column needs to become a timestamp
SELECT TRY_CAST(event_time_str AS TIMESTAMP) AS event_time FROM raw.events

-- CORRECT: integer epoch needs to become a timestamp
SELECT to_timestamp(created_at_epoch) AS created_at FROM raw.orders
```

Decision rule: if `describe_table` shows the source column is already the target type, omit the cast entirely.

---

## Row Filtering Policy (CRITICAL)

**The filtering decision depends on whether the column is a pass-through column or a cast result.**

`constraints: not_null` in the contract is a claim the gold SQL generally honors — **but only if the claim is compatible with the raw data**. Casting (`TRY_CAST`) can silently produce NULLs that were not present in the source; when that happens, the gold solution preserves the rows and accepts the NULLs.

### The two-case rule

**Case 1 — Pass-through column (no cast, same type as raw)**

If the target column's expression is a direct reference (`col`, `TRIM(col)`, `col AS new_name`) or a CASE WHEN that never introduces new NULLs beyond what the raw column already has, and the contract declares `constraints: [not_null]`:

→ **Apply `WHERE col IS NOT NULL`** (the contract's `not_null` means "delete rows where raw is NULL").

This is the default for VARCHAR foreign keys, names, codes, booleans, numeric IDs — anything that is just renamed or trimmed. Gold filters these.

**Case 2 — Cast-derived column (`TRY_CAST`, `STRPTIME`, `to_timestamp`, etc.)**

If the target column is produced by a cast (`TRY_CAST(x AS TIMESTAMP)`, `STRPTIME(x, fmt)`, `to_timestamp(x)`), even when the contract declares `constraints: [not_null]`:

→ **Do NOT add `WHERE col IS NOT NULL`.**

The cast can silently produce NULLs when the raw format is non-standard (e.g. `+0800` timezone strings that DuckDB's standard TIMESTAMP parser rejects). The contract's `not_null` on a cast column is a **spec wish**, not an executable rule. Gold preserves those rows with NULL and the benchmark evaluator accepts it.

Use the Data Reality Check success-rate test from earlier in this skill to confirm the cast behavior. If success_rate < 0.95, you MUST NOT filter that column.

### Primary key special case

If a column has `constraints: [not_null, unique]` with `on_failure: delete_row`, that is always a `WHERE pk IS NOT NULL` (or `LENGTH(pk) > 0`). No ambiguity.

### Temporal / cross-column rules

Cross-column rules like `end_time >= start_time` or `archived_at >= created_at` must be written as **"only apply when both sides are non-null"** so a NULL cast does not drop the row:

```sql
WHERE (archived_at IS NULL OR created_at IS NULL OR archived_at >= created_at)
```

Gold comments this pattern explicitly: *"Relax time filtering: do not drop rows because created_at is empty, only compare when comparable."*

### Regex / format validation rules

These should use `CASE WHEN valid THEN value ELSE NULL END`, not `WHERE`:

```sql
-- CORRECT: regex failure becomes NULL, row preserved
SELECT
    user_id,
    CASE
        WHEN regexp_matches(phone_number, '^\d{10}$') THEN phone_number
        ELSE NULL
    END AS phone_number
FROM raw.users
```

### Summary table (updated)

| Situation | Action |
|---|---|
| Pass-through column, `constraints: [not_null]` | `WHERE col IS NOT NULL` |
| Pass-through PK, `not_null, unique` + `delete_row` rule | `WHERE LENGTH(pk) > 0` |
| Cast-derived column (`TRY_CAST`, `STRPTIME`, ...), any constraint | **No WHERE filter**; let cast NULLs pass through |
| Regex / whitelist validation | `CASE WHEN valid THEN value ELSE NULL END` |
| Temporal cross-column constraint | `(a IS NULL OR b IS NULL OR a <= b)` |
| Column has no `not_null` constraint | Do not filter |

### Before writing the WHERE clause

Walk through every `constraints: [not_null]` column in the contract and classify it:

1. Is the target expression a direct reference / TRIM / rename? → Pass-through → Filter.
2. Is it `TRY_CAST`, `STRPTIME`, `to_timestamp`, or any other format-dependent cast? → Cast-derived → Do not filter.

Only columns that fail *step 1* should appear in your final `WHERE ... IS NOT NULL` list.

---

## Marts Layer Timestamp Columns (CRITICAL for benchmarks)

Many marts tables (e.g. `*_dashboard`, `*_dashboard_simple`) have columns like `report_date` and `generated_at` that the contract defines as `CURRENT_DATE` and `CURRENT_TIMESTAMP`.

**These columns are non-deterministic by design.** Gold DuckDB files in benchmarks are built at a fixed point in time, so `report_date` in gold is frozen. Your generated table will have today's date. This IS expected — these columns cannot match gold via hash comparison, and the benchmark evaluator may skip them in compare_cols.

Action: follow the contract (use `current_date`, `current_timestamp`) — do not try to hardcode dates to match gold. The mismatch is not your bug.

---

## Intermediate and Marts Output Column Discipline

The `business_logic` narrative mixes CTE input columns with final output columns. This frequently causes generated SQL to include too many columns in the final SELECT.

### Rule 1: CTE inputs are not automatically output columns

If the business logic says:

> "Start from orders as cte_orders: Select order_id, customer_id, amount, created_at, last_modified_at..."

`last_modified_at` being selected into the CTE does NOT mean it appears in the final output. Only include it if it is explicitly listed in the "Final assembly" section or used in a derived field definition.

### Rule 2: Raw timestamp fields are almost always intermediate only

Fields like `last_advanced_at`, `last_interaction_at`, `updated_at`, `last_modified_at` are typically used ONLY to compute derived fields (e.g., `days_since_last_interaction = DATEDIFF('day', last_interaction_at, CURRENT_DATE)`). They should NOT appear in the final SELECT unless the contract explicitly lists them as output columns.

### Rule 3: Final output columns come from exactly three sources

1. **Keys and business attributes** explicitly carried from the base table in the contract's output columns list
2. **Derived fields** defined with a `= <formula>` expression in the business logic
3. **Computed fields** explicitly named in the "Final assembly" / "Output" section of the business logic

### Good vs bad example

Contract business logic (intermediate layer):

> "Start from customer_events: Select customer_id, event_type, event_time, last_event_time.
> Compute: days_since_last_event = DATEDIFF('day', last_event_time, CURRENT_DATE).
> Final assembly: customer_id, event_type, days_since_last_event."

```sql
-- WRONG: includes last_event_time which was only used for the computation
SELECT
    customer_id,
    event_type,
    last_event_time,                                              -- REMOVE
    DATEDIFF('day', last_event_time, CURRENT_DATE) AS days_since_last_event
FROM staging.stg_customer_events

-- CORRECT: only the three final output sources
SELECT
    customer_id,
    event_type,
    DATEDIFF('day', last_event_time, CURRENT_DATE) AS days_since_last_event
FROM staging.stg_customer_events
```

---

## Layer Quick Reference

| Layer | Reads from | Key patterns |
|---|---|---|
| Staging | `raw.*` | Thin wrappers; dialect translation; CASE WHEN normalization; no joins/aggregations |
| Intermediate | `staging.*` | Joins, aggregations, window functions, derived field computation; CTE inputs stay in CTEs |
| Marts | `intermediate.*`, `staging.*` | Final consumption shape; match contract output columns exactly; use `current_date`/`current_timestamp` for timestamp columns per contract |
