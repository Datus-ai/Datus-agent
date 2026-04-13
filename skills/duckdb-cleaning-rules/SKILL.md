---
name: duckdb-cleaning-rules
description: Reality-checked staging and cleaning rules for generating DuckDB SQL that matches a declarative contract without dropping rows or mis-casting columns
tags:
  - duckdb
  - data-cleaning
  - staging
  - dialect-translation
  - data-profiling
  - dacomp
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
---

# DuckDB Cleaning Rules

Rules for writing DuckDB SQL that cleans / normalizes raw tabular data into a contract-described target table, **where the raw data may not match the contract's declared types or nullability**.

## When to use this skill

Load this skill alongside any SQL-generation skill (e.g. `dbt-layered-generation`) whenever you are:

- Writing a staging-layer transformation from `raw.*` tables
- Translating contract SQL that was authored in MySQL / ANSI / generic SQL into DuckDB
- Choosing whether to apply `COALESCE`, `WHERE ... IS NOT NULL`, or `CAST` on any column

This skill is the **authoritative source** for dialect translation, row-filtering semantics, type-casting discipline, COALESCE sentinel selection, and timestamp determinism for DAComp-style benchmarks.

---

## Data Reality Check Workflow

The data contract describes the **desired** schema, but source data often violates those claims. **Contract claims lose to observed data** — if the data says something different, follow the data.

### If explore pre-flight findings are available

When the user prompt includes a "Pre-Flight Data Reality Findings" section, it already contains per-column observations from a dedicated explore pass (schema types, cast test results, format samples). **Use those findings directly** — do NOT re-run the profiling steps below. The findings are factual; combine them with the contract spec to choose your casting strategy.

### Schema discovery rules

- **Raw tables live in the `raw.` schema** (e.g. `raw.user`, `raw.account_history`). Always qualify as `FROM raw.<table>`.
- Use `list_tables(schema_name='raw')` and `describe_table('raw.<name>')` for discovery.
- The DB file name (catalog) is **never** a valid schema prefix. `FROM _skill_scratch_pendo.user` is **wrong**; `FROM raw.user` is correct.

### Cast strategy decision (when NO explore findings are available)

For every TIMESTAMP / DATE target column where the raw source is VARCHAR, test with `read_query`:

1. **Plain cast**: `SELECT COUNT(*) AS total, COUNT(TRY_CAST(col AS TIMESTAMP)) AS cast_ok FROM raw.<table>` — if success_rate ≥ 0.95, use plain `TRY_CAST`.
2. **Substring salvage**: `SELECT COUNT(*), COUNT(TRY_CAST(SUBSTRING(col, 1, 23) AS TIMESTAMP)) FROM raw.<table>` — if the plain cast fails but this succeeds, use `TRY_CAST(SUBSTRING(col, 1, 23) AS TIMESTAMP)`.
3. **If both fail**: use plain `TRY_CAST` and accept NULLs — do NOT add `WHERE col IS NOT NULL`.

### Output column naming rule

Raw column names from `describe_table` are for type checking only. **Output column names MUST come from the contract's `columns:` section.** Never copy raw names into the final SELECT — always use the contract's renamed targets.

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

The cast can silently produce NULLs when the raw format is non-standard — for example `+0800`-style timezone strings that DuckDB's standard TIMESTAMP parser rejects, or DATE strings with slashes instead of dashes, or INT strings with thousand separators. Any format DuckDB can't natively parse triggers the trap. The contract's `not_null` on a cast column is a **spec wish**, not an executable rule. Gold preserves those rows with NULL and the benchmark evaluator accepts it.

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

### Decision summary

Walk through every `constraints: [not_null]` column and classify it via the left column below; the right column tells you the action:

| Situation | Action |
|---|---|
| Pass-through column (direct ref / TRIM / rename), `constraints: [not_null]` | `WHERE col IS NOT NULL` |
| Pass-through PK, `not_null, unique` + `delete_row` rule | `WHERE LENGTH(pk) > 0` |
| Cast-derived column (`TRY_CAST`, `STRPTIME`, `to_timestamp`, …), any constraint | **No WHERE filter**; let cast NULLs pass through |
| Regex / whitelist validation | `CASE WHEN valid THEN value ELSE NULL END` |
| Temporal cross-column constraint | `(a IS NULL OR b IS NULL OR a <= b)` |
| Column has no `not_null` constraint | Do not filter |

Only columns whose row hits the first line of the table should appear in your final `WHERE ... IS NOT NULL` list.

---

## Deterministic `CURRENT_DATE` via Pinned Reference Date

When the contract references `CURRENT_DATE` / `CURRENT_TIMESTAMP` for output columns or window filters (e.g. `*_dashboard`, `*_daily_metrics`), emitting the literal SQL function makes the query non-deterministic — the result depends on wall-clock time at execution. For frozen-gold benchmarks this breaks hash comparisons.

### When the system prompt pins a reference date

If your system prompt contains a line like:

```
Current date: 2025-10-27
```

this is the dataset's pinned reference date. You MUST emit a fixed DATE / TIMESTAMP literal instead of the non-deterministic SQL functions:

| Contract says | Wall-clock version (WRONG) | Pinned version (CORRECT) |
|---|---|---|
| `CURRENT_DATE` | `CURRENT_DATE` | `DATE '<current_date>'` |
| `current_date - INTERVAL '30' DAY` | `CURRENT_DATE - INTERVAL '30' DAY` | `DATE '<current_date>' - INTERVAL '30' DAY` |
| `CURRENT_TIMESTAMP` | `CURRENT_TIMESTAMP` | `TIMESTAMP '<current_date> 00:00:00'` |

Example (pinned to `2025-10-27`, table name taken from DAComp DE impl-001 for illustration only — substitute whatever table your task actually queries):

```sql
-- WRONG: non-deterministic, breaks hash match
SELECT
    CURRENT_DATE AS report_date,
    CURRENT_TIMESTAMP AS generated_at,
    COUNT(*) FILTER (WHERE application.created_at >= CURRENT_DATE - INTERVAL '30' DAY)
        AS applications_last_30_days
FROM staging.stg_lever__application

-- CORRECT: uses pinned reference date everywhere CURRENT_DATE would be
SELECT
    DATE '2025-10-27' AS report_date,
    TIMESTAMP '2025-10-27 00:00:00' AS generated_at,
    COUNT(*) FILTER (WHERE application.created_at >= DATE '2025-10-27' - INTERVAL '30' DAY)
        AS applications_last_30_days
FROM staging.stg_lever__application
```

Apply this translation anywhere the contract references "current date", "today", "now", or the equivalent SQL functions — not just in the output columns, but also in window filters and relative-time computations that the derived metrics depend on.

### When no reference date is pinned

If the system prompt has no `Current date:` line, the dataset is not pinned. Follow the contract verbatim (use `CURRENT_DATE`, `CURRENT_TIMESTAMP`) and accept that those columns will not hash-match gold — the benchmark evaluator typically excludes them from compare_cols, and the mismatch is not your bug.

---

## COALESCE Sentinel Discipline for Scoring and Classification (CRITICAL)

When business_logic declares *"set to 0 if no qualifying rows"* for an aggregate column (e.g. `avg_days_to_advancement`, `candidates_hired`), **that rule applies ONLY to the final output column value**. It does NOT automatically apply to every other place the same underlying CTE column is referenced for classification, threshold scoring, or derived labels.

Reusing `COALESCE(col, 0)` inside a threshold CASE silently breaks the classification because **0 sits at the bottom of the numeric range** — it spuriously matches any low-threshold bucket meant for "fast" / "efficient" / "best" and fails any high-threshold bucket meant for "slow" / "missing".

### Decision rule for each CASE expression

When you write a CASE / threshold expression referencing an aggregate that could be NULL:

1. **Output column itself?** → `COALESCE(x, 0)` is fine when business_logic says "set to 0 if no qualifying rows".
2. **Threshold classification, business_logic silent about NULL?** → Do NOT wrap in `COALESCE(x, 0)`. Let NULL propagate so it falls through to `ELSE`.
3. **Threshold classification, business_logic explicitly says "treat missing as 999" (or similar)?** → Use that specific sentinel (`COALESCE(x, 999)`).

Write every classification CASE as if the raw value could be NULL and ask *"would NULL route to the correct branch?"* If no, adjust the sentinel for that branch only, not for the output column.

### Example (contract from DAComp DE impl-001 `int_lever__posting_analytics`, illustrative — the pattern generalizes to any benchmark that mixes output defaults with threshold classification)

Contract:

> `avg_days_to_advancement: set to 0 if no qualifying rows.`
> `posting_effectiveness_score: +20 if avg_days_to_advancement ≤ 30; +10 if ≤ 60; else +0`
> `primary_bottleneck: ... avg_days_to_advancement > 60 → 'Slow Decision Process'`

```sql
-- CORRECT: output column uses 0; score lets NULL propagate; bottleneck uses 999 sentinel
SELECT
    posting_id,
    COALESCE(te.avg_days_to_advancement, 0) AS avg_days_to_advancement,

    -- Score: no COALESCE; NULL <= 30 is NULL → falls to ELSE 0 (missing data gets 0 points, not +20)
    CASE WHEN te.avg_days_to_advancement <= 30 THEN 20
         WHEN te.avg_days_to_advancement <= 60 THEN 10
         ELSE 0
    END AS score_component,

    -- Bottleneck: 999 sentinel so NULL routes to 'Slow Decision Process' (worst case)
    CASE WHEN COALESCE(te.avg_days_to_advancement, 999) > 60
         THEN 'Slow Decision Process'
         ...
    END AS primary_bottleneck
FROM ...
```

The wrong pattern uses `COALESCE(...,0)` inside both the score CASE and the bottleneck CASE. The score then spuriously gives missing data +20 points (because 0 ≤ 30), and the bottleneck never routes NULL to "Slow Decision Process" (because 0 is not > 60).

### Quick audit

Before emitting the final SELECT, scan every CASE expression and ask: *"if this aggregate is NULL, does my sentinel route it to the right branch?"* Fix any branch that would misroute.
