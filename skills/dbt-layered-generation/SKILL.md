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

1. **MaxTurnsExceeded**: complex marts (example: `lever__hiring_manager_scorecard`, `lever__departmental_hiring_trends` from DAComp DE impl-001 — the pattern generalizes to **any** mart whose contract has a multi-CTE scoring / classification spec) where the agent keeps exploring without ever finalizing — a plan enforces closure.
2. **Over-production**: 20+ column outputs where the contract only asks for 10-15, because the agent accumulates CTE columns without ever pruning (example: `lever__opportunity_stage_history` in DAComp DE impl-001 — the pattern generalizes to **any** mart whose business_logic walks through CTEs verbosely before the Final Assembly section).

### When to invoke todo_write

Call `todo_write` at the start of any table that matches **any** of:
- Contract `columns:` section has > 5 entries
- `business_logic` includes derived fields, CASE classification, or multi-CTE joins
- Output depends on 3+ upstream tables
- You're about to look up something (call `describe_table` / `read_query`) and continue generating

### Required plan shape

The plan must be concrete and SQL-oriented, not narrative. Each item should correspond to a concrete deliverable in the final SQL. Example shape (table names are illustrative — substitute your task's real upstream tables):

```json
[
  {"content": "Read contract columns: exact list of target output columns, with types", "status": "pending"},
  {"content": "Describe upstream <base_table> for key column types (e.g. timestamps, enums)", "status": "pending"},
  {"content": "Write CTE `<base_name>` with the base columns needed downstream", "status": "pending"},
  {"content": "Write CTE `<join_name>` aggregating / joining the auxiliary lookups", "status": "pending"},
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

## Output Column Compliance (CRITICAL)

The contract's `columns:` section is the **only** source of truth for final output column names and their presence. **Never invent plausible alternatives**; never leak CTE-internal columns; never omit a listed column.

### The two-step final check

After writing the final SELECT, run a mental diff:

1. List the contract's expected columns (from `columns:` or `business_logic` "Final assembly" / "Output" block).
2. List your final SELECT's columns, in order.
3. They must be **identical sets** — case-sensitive, underscore-sensitive, no extras, no renames, no omissions.

If the sets differ: **remove extras, add missing, rename to match exactly**. Your sense of "better" naming is irrelevant — the benchmark scorer compares by literal column name after lowering.

### Where extras come from

Most extras are CTE input columns that the agent dragged into the final SELECT out of habit:

- **CTE inputs are not auto-outputs.** If the business_logic says *"Start from orders: select order_id, customer_id, amount, created_at, last_modified_at"*, those are the CTE's inputs. Only include a column in the final output if it is (a) explicitly listed in the contract's `columns:` section, (b) used as a key carried from the base table, or (c) named in a derived-field `= <formula>` definition.
- **Raw timestamp fields** (`last_advanced_at`, `last_interaction_at`, `updated_at`, `last_modified_at`) are almost always intermediate-only. They feed into `DATEDIFF(...)` derived fields but rarely appear in the final SELECT themselves.

### Common rename drift patterns

> *The agent-wrote / contract-wants pairs below come from DAComp DE impl-001 observations (lever / pendo domains). The **patterns** — suffix drift, prefix collapse, unrelated leakage — generalize to any contract-driven task.*

| Pattern | Agent wrote | Contract wants |
|---|---|---|
| "count" suffix drift | `basic_quality_count`, `exceptional_count` | `basic_quality_candidates`, `exceptional_candidates` |
| "avg" prefix collapse | `avg_quality_score` | `overall_avg_quality_score` |
| Redundant name doubling | `stage_name` (when a stage CTE was aliased) | `stage` |
| Missing business-domain rename | `contact_name` | `opportunity_contact_name` |
| Unrelated intermediate leakage | `archived_at`, `posting_id`, `emails`, `phones`, `linkedin_link`, `github_link`, `tags` | (contract lists none) |

If the contract field is spelled `basic_quality_candidates`, write `AS basic_quality_candidates`. If you cannot decide what name to use for a derived field, read the exact `columns:` list via `filesystem_tools` on the contract YAML.

### Uniform COALESCE across aggregate groups

When business_logic says *"Set missing interview and feedback aggregates to 0"* (or similar), you MUST apply `COALESCE(..., 0)` to **every** aggregate column in that group, not just the counts. Agents frequently wrap `total_*` and `completed_*` columns in COALESCE but forget `avg_*` and `total_*_time` because "avg and sum feel different from count".

```sql
-- WRONG: avg_interview_duration is an aggregate too; contract said "set missing aggregates to 0"
LEFT JOIN interview_summary is2 USING (opportunity_id)
SELECT
    COALESCE(is2.total_interviews, 0)     AS total_interviews,
    COALESCE(is2.completed_interviews, 0) AS completed_interviews,
    is2.avg_interview_duration            AS avg_interview_duration,  -- MISSING COALESCE
    COALESCE(is2.total_interview_time, 0) AS total_interview_time

-- CORRECT: every aggregate column in the group gets the same treatment
LEFT JOIN interview_summary is2 USING (opportunity_id)
SELECT
    COALESCE(is2.total_interviews, 0)        AS total_interviews,
    COALESCE(is2.completed_interviews, 0)    AS completed_interviews,
    COALESCE(is2.avg_interview_duration, 0)  AS avg_interview_duration,
    COALESCE(is2.total_interview_time, 0)    AS total_interview_time
```

> *Example table names are from DAComp DE impl-001 → `int_lever__candidate_insights`; the rule is universal.*

This rule governs **which aggregates to COALESCE for output values**. For the separate question of *what sentinel to use inside a threshold CASE / classification expression*, see `duckdb-cleaning-rules` → COALESCE Sentinel Discipline.

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

### Example (from DAComp DE impl-001, lever domain)

> The rule is generic — the example below uses DAComp lever tables, but the trap ("partition by the join target instead of the 'per X' noun") appears anywhere business_logic names a window operation in natural language. Replace table names with whatever your task uses.

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
