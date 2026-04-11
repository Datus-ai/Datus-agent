---
name: dbt-layered-generation
description: Orchestrate SQL generation for a layered dbt-style data warehouse (staging → intermediate → marts); handles planning, output-column discipline, and cross-layer contract compliance. Pairs with `duckdb-cleaning-rules` for SQL dialect/cleaning details.
tags:
  - dbt
  - data-engineering
  - staging
  - intermediate
  - marts
  - layered-warehouse
  - sql-generation
  - dacomp
version: "2.0.0"
user_invocable: false
disable_model_invocation: false
---

# DBT Layered SQL Generation

Orchestrate SQL generation for a layered dbt-style data warehouse following the DAComp DE-Impl format. This skill focuses on **cross-layer structure**: task planning, output-column discipline, and strict contract compliance. Low-level cleaning mechanics (DuckDB dialect, type casting, row filtering, COALESCE semantics, CURRENT_DATE handling) are maintained separately in the `duckdb-cleaning-rules` skill — load it alongside this one whenever you are generating DuckDB SQL.

## When to use this skill

Activate when the task provides a `data_contract.yaml` with a `layer_dependencies.yaml`, references layer names (staging / intermediate / marts), or asks to implement a DAComp DE-Impl pipeline. The runner executes layers in topological order (staging → intermediate → marts). For each table: generate SQL, execute, validate against contract, optionally retry (max 3 rounds for execution errors, max 1 for column mismatch).

## Skill dependencies

This skill is a **structural orchestrator**. It does NOT contain:

- DuckDB dialect translation tables (see `duckdb-cleaning-rules` → DuckDB Dialect Rules)
- The Data Reality Check workflow for raw-source profiling (see `duckdb-cleaning-rules` → Data Reality Check)
- Type casting / row filtering / COALESCE sentinel decisions (see `duckdb-cleaning-rules` → Row Filtering Policy + COALESCE Sentinel Discipline)
- `CURRENT_DATE` pinning for deterministic dashboards (see `duckdb-cleaning-rules` → Deterministic CURRENT_DATE)

For any staging table and for intermediate / marts tables that do cleaning or classification, **load `duckdb-cleaning-rules` in addition** to this skill.

---

## Planning Workflow (MANDATORY for complex marts / intermediate tables)

For any table with more than ~5 output columns or any non-trivial business logic, you MUST use `todo_write` to record a short plan **before writing SQL**. This prevents two recurring failure modes:

1. **MaxTurnsExceeded**: complex marts (e.g. `lever__hiring_manager_scorecard`, `lever__departmental_hiring_trends`) where the agent keeps exploring without ever finalizing — a plan enforces closure.
2. **Over-production**: 20+ column outputs where the contract only asks for 10-15, because the agent accumulates CTE columns without ever pruning (see `lever__opportunity_stage_history`).

### When to invoke todo_write

Call `todo_write` at the start of any table that matches **any** of:
- Contract `columns:` section has > 5 entries
- `business_logic` includes derived fields, CASE classification, or multi-CTE joins
- Output depends on 3+ upstream tables
- You're about to look up something (call `describe_table` / `read_query`) and continue generating

### Required plan shape

The plan must be concrete and SQL-oriented, not narrative. Each item should correspond to a concrete deliverable in the final SQL:

```json
[
  {"content": "Read contract columns: exact list of target output columns, with types", "status": "pending"},
  {"content": "Describe upstream stg_lever__opportunity for created_at / stage_id types", "status": "pending"},
  {"content": "Write CTE `opportunity_base` with the 6 base columns", "status": "pending"},
  {"content": "Write CTE `stage_join` aggregating stage_name and archive_reason", "status": "pending"},
  {"content": "Write final SELECT matching contract columns EXACTLY (no extras)", "status": "pending"},
  {"content": "Verify: final SELECT columns === contract columns:, same order, same names", "status": "pending"}
]
```

Mark items `completed` with `todo_update` as you finish them. When every item is `completed`, emit the JSON response with the SQL.

### Why this works

- Forces the agent to enumerate the final output column list FIRST, preventing CTE column leakage into the output.
- Makes the "verify" step explicit, so the agent self-checks column names against the contract before returning.
- Bounds exploration: each `describe_table` call is tied to a specific plan item, not open-ended.

Do NOT skip the plan for complex tables. The benchmark has measured the failure rate for un-planned complex marts at 100% — even with all other rules followed, the agent drifts or over-produces without a written plan.

---

## Strict Column Name Adherence (CRITICAL)

The contract's `columns:` section is the ONLY source of truth for the final output column names. **Never invent plausible alternatives**, even when the inferred names seem more natural.

### Common failure patterns observed in benchmarks

| Pattern | Agent wrote | Contract wants |
|---|---|---|
| "count" suffix drift | `basic_quality_count`, `exceptional_count` | `basic_quality_candidates`, `exceptional_candidates` |
| "avg" prefix collapse | `avg_quality_score` | `overall_avg_quality_score` |
| Redundant name doubling | `stage_name` (when the stage CTE was aliased) | `stage` |
| Missing business-domain rename | `contact_name` | `opportunity_contact_name` |
| Unrelated extras dragged from intermediate | `archived_at`, `archived_reason_id`, `posting_id`, `emails`, `phones`, `linkedin_link`, `github_link`, `tags` | (contract lists none of these) |

### Mandatory cross-check before returning SQL

After writing the final SELECT, run this mental diff:

1. Extract the `columns:` list from the contract `columns:` section (or `business_logic` "Final assembly" / "Output" block).
2. Extract the column names your final SELECT produces, in order.
3. They must be **identical sets** — no extras, no renames, no omissions.

If they differ:
- **Extras in your SELECT**: remove them. CTE input columns are NOT final outputs.
- **Missing from your SELECT**: add them, deriving them from upstream if necessary.
- **Renames**: rename to match the contract EXACTLY (case-sensitive, underscore-sensitive).

### Rule of thumb

If the contract field is spelled `basic_quality_candidates`, write `AS basic_quality_candidates`. If it's `overall_avg_quality_score`, write `AS overall_avg_quality_score`. Your own sense of "better" naming is irrelevant — the benchmark scorer compares by literal column name after lowering.

If you genuinely cannot decide what name to use for a derived field, fall back to reading the **exact** `columns:` list via `filesystem_tools` on the contract YAML, or via the per-table spec the runner already injected into your user prompt.

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

### Rule 4: "Set missing aggregates to 0" applies to EVERY aggregate column uniformly

When the contract's business_logic says something like *"Set missing interview and feedback aggregates to 0"*, you MUST apply `COALESCE(..., 0)` to **every** aggregate column in that group, not just the counts. Agents frequently wrap `total_interviews`, `completed_interviews`, `canceled_interviews` in COALESCE but forget `avg_interview_duration` and `total_interview_time` because "avg and sum feel different from count".

```sql
-- WRONG: avg_interview_duration is an aggregate too; contract said "set missing aggregates to 0"
LEFT JOIN interview_summary is2 USING (opportunity_id)
SELECT
    COALESCE(is2.total_interviews, 0)     AS total_interviews,
    COALESCE(is2.completed_interviews, 0) AS completed_interviews,
    COALESCE(is2.canceled_interviews, 0)  AS canceled_interviews,
    is2.avg_interview_duration            AS avg_interview_duration,  -- MISSING COALESCE
    COALESCE(is2.total_interview_time, 0) AS total_interview_time

-- CORRECT: every aggregate column in the group gets the same treatment
LEFT JOIN interview_summary is2 USING (opportunity_id)
SELECT
    COALESCE(is2.total_interviews, 0)        AS total_interviews,
    COALESCE(is2.completed_interviews, 0)    AS completed_interviews,
    COALESCE(is2.canceled_interviews, 0)     AS canceled_interviews,
    COALESCE(is2.avg_interview_duration, 0)  AS avg_interview_duration,
    COALESCE(is2.total_interview_time, 0)    AS total_interview_time
```

**Before emitting the final SELECT, do a cross-check**: for every aggregate CTE you joined with LEFT JOIN, list every column you're selecting from it. If the contract said to default missing aggregates to 0, every one of those columns needs `COALESCE(..., 0)`. No exceptions for averages, sums, or medians.

This rule is about *which aggregates to COALESCE in the final SELECT*. For the distinct question of *what sentinel to use inside a threshold CASE*, see `duckdb-cleaning-rules` → COALESCE Sentinel Discipline for Scoring and Classification.

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

## Window / Aggregation Key: "per X" is authoritative

When business_logic uses phrasing like *"most recent … per candidate"*, *"total … per user"*, *"rank … per department"*, the noun after **"per"** is the `PARTITION BY` (or `GROUP BY`) key — not whichever foreign-key column looks convenient.

### Common confusion

A table may contain several foreign keys (`candidate_id`, `opportunity_id`, `user_id`) and the agent is tempted to partition by the join target that will later be used. This is frequently wrong.

### Rule

1. Read the business_logic sentence that describes the window operation verbatim.
2. Find the **"per X"** (or "by X", "for each X") phrase that qualifies the aggregation.
3. `PARTITION BY X` in your ROW_NUMBER / RANK / FIRST_VALUE window.
4. The subsequent `JOIN` target (e.g. "join … on opportunity_id") is **unrelated** to the partition key — it's the join column, not the aggregation key.

### Example (from DAComp lever)

Contract says:

> "Determine the most recent offer **per candidate** by ordering each candidate's offers by created_at descending and selecting the single most recent record. Left join this most recent offer to opportunities **on opportunity_id**."

```sql
-- WRONG: partitioned by opportunity_id (the later join target), not candidate_id
WITH ranked_offers AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY opportunity_id
        ORDER BY created_at DESC
    ) AS rn
    FROM staging.stg_lever__offer
)
-- This computes "most recent offer per opportunity" which is a different set of rows.

-- CORRECT: partitioned by candidate_id as the business_logic explicitly says
WITH ranked_offers AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY candidate_id
        ORDER BY created_at DESC
    ) AS rn
    FROM staging.stg_lever__offer
)
-- Then the join is still `ON opportunity.opportunity_id = ranked_offers.opportunity_id`;
-- the partition key and the join key are different columns, by design.
```

---

## Layer Quick Reference

| Layer | Reads from | Key patterns |
|---|---|---|
| Staging | `raw.*` | Thin wrappers; dialect translation; CASE WHEN normalization; no joins/aggregations |
| Intermediate | `staging.*` | Joins, aggregations, window functions, derived field computation; CTE inputs stay in CTEs |
| Marts | `intermediate.*`, `staging.*` | Final consumption shape; match contract output columns exactly; use pinned `current_date` for timestamp columns per contract |
