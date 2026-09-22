---
name: gen-datasource-v2
description: Generate a reproducible DuckDB sample database from business requirements and DDL for any industry, preserving schema constraints and correlated business data. Uses set-based SQL and validation before import.
---

# Generate a Datasource v2

Build the requested database directly from the business description and DDL. Make
reasonable assumptions and continue; do not add a question-discovery phase or ask
the user to design the data. Business-specific choices come from this request,
not from a fixed industry template.

You write business SQL and assertions. The bundled runner creates the original
schema, executes SQL and checks the result. Use SQL `range()`, temporary tables,
joins, expressions and window functions for bulk generation. Small Python helpers
are appropriate when a requested structure is awkward in SQL.

## Start

1. Save the supplied DDL to `data/schema.sql`, unless it is already supplied there.
   Preserve every table, column, type, NOT NULL, PRIMARY KEY (including composite),
   UNIQUE and FOREIGN KEY. Normalize unsupported dialect syntax only when needed
   and record the change. Inferred business joins do not change the user's schema.
2. Use the `<skill_location>` directory returned by `load_skill` to initialize:

   ```text
   python3 <skill_location>/scripts/datasource.py init --ddl data/schema.sql --directory data --rows 80000 --min-rows 50000 --max-rows 100000 --months 17 --seed 42
   ```

   Adjust row bounds to the request. This saves `gen.py`, `schema.json`, and
   `settings.json` and prints dependency order. Boilerplate and DDL already exist;
   do not copy them into another generator. The date window is fixed in settings.
3. Write `data/generate.sql` and `data/checks.json`; run `python3 data/gen.py`.
   It writes `data/_build/datasource.duckdb` and `data/quality.json`. On SQL failure
   it reports the statement number and error. Read `schema.json` only if needed.

## First executable version

Briefly choose root counts, parent-child fan-outs, units, states and shared causes
from the supplied business description. Put decisions in SQL comments while
writing; avoid drafting a separate long design. Aim to run within 6–8 minutes.

- Insert into existing tables in dependency order. `INSERT INTO t BY NAME SELECT`
  avoids column-position errors. TEMP staging tables may hold helper variables;
  persistent tables must exactly match the supplied DDL.
- Allocate the **whole database** budget: choose root counts and child fan-outs
  together, summing dimensions and facts. Scale detail explicitly for demo size.
- Sample each FK from an existing parent, selecting composite key tuples together.
  Pick a parent once and inherit all related values. Reuse upstream business codes
  for undeclared relationships. Composite-grain rows need unique combinations.
- Generate base quantities; derive dependent totals, rates and statuses from them.
  Build staged children first when parents need their aggregates, then insert final
  parents before children. This avoids updating referenced keys in DuckDB.
- Define the grain of every detail table before choosing counts. If a detail table
  is a sample, store a matching sample denominator or derive the parent aggregate
  from exactly that sample; never compare a sampled child count with a full-population
  parent total. Reconcile counts, sums and rates at the same grain used by the DDL.
- Persist the values used by a flag and derive the flag from those persisted values.
  For OOS, status, SLA and lifecycle flags, the stored boolean must equal the same
  rounded value-versus-limit or timestamp predicate checked by the assertion. A
  correlated latent cause may change the measurement, but must not override a
  mathematically contradictory flag.
- Use meaningful units and enum domains from the request. Ordinary descriptions
  need short defaults, not extensive catalogs. Fill every NOT NULL field.
- Model time relationships explicitly. Actual events follow their causes; planned
  deadlines can be future-dated. An ID or sequence number is not an event time.
- Where attribution matters, a shared scoped cause influences related measurements
  and outcomes across tables, with independent noise. Keep helper variables TEMP.
- Apply seasonality, long tails and lifecycle rules only when appropriate to the
  business. Physical measurements need not have revenue-like trends or promotions.

The runner provides deterministic macros:

```sql
CREATE TEMP TABLE seeds AS
SELECT i + 1 AS entity_id, u01(i, 1) AS latent_factor,
       data_start() + floor(u01(i, 2) * (data_end() - data_start() + 1))::INTEGER AS event_date
FROM range(200) AS r(i);
```

`u01(integer_key, integer_stream)` is uniform in [0,1), reproducible for the saved
seed and DuckDB version. Different stream numbers give distinct draws.
`data_start()` / `data_end()` return fixed DATEs. Avoid unseeded random and `now()`.
For fixed fan-out k: parent index is `1 + floor((child_id-1)/k)::BIGINT`, and
within-parent index is `1 + (child_id-1)%k`. Use floor: integer casts round.

## Validate and repair

Write roughly 8–12 scalar assertions with the first SQL version, covering the
request's cross-table agreement, plausible values, status/time relationships and
attributable effects. This is executable validation, not question planning.

```json
{"assertions": [
  {"name": "derived quantities agree", "expect": "zero",
   "sql": "SELECT count(*) FROM items WHERE abs(total - quantity*unit_price)>0.01"},
  {"name": "ratio in business range", "expect": {"min": 0.6, "max": 0.98},
   "sql": "SELECT sum(good_qty)*1.0/nullif(sum(total_qty),0) FROM batches"}
]}
```

The runner checks exact schema, all declared constraints, populated tables, total
row bounds and business assertions. NULL/NaN, missing assertions and invalid SQL
fail. Inspect `quality.json` for all failures; batch local repairs and rerun.
Keep acceptance criteria stable; do not delete or widen assertions just to pass.
Normally two repair rounds suffice. Report persistent limitations honestly rather
than dropping constraints. Use at most three extra inspection queries.

## Import and deliver

When `quality.json` says `ok: true`, call:

```text
import_database_file(path="data/_build/datasource.duckdb", mode="replace", keep_constraints=true)
```

Inspect the import result: constraint degradation is a failure. The active datasource
is held open by Datus; never generate into, overwrite or delete its
`data/datasource.duckdb`. Keep the validated build and report for inspection.
Write `data/README.md` in roughly 40–80 lines: actual counts, dates, business
relationships, anomalies, assumptions, quality and limitations. Deliver once done.
The runner is this skill's quality gate; v1's inferred roles and heuristic checks
are not requirements for this independent path.
