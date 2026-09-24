# Semantic contract and reusable generation patterns

Read this before the first generation. `init` writes an incomplete `semantics.json`:
fill its CHOOSE fields and add the business rules. It already copies declared PK/FK
tuples; add undeclared logical relationships without changing the physical DDL.
All identifiers below refer to actual supplied tables/columns. Unknown fields,
unsupported aggregates, missing columns and unmeasurable rules fail the gate.

## Contract

Example shape (adapt identifiers and business choices to this request):

```json
{
  "quality_contract_version": 1,
  "tables": {
    "entities": {
      "role": "dimension", "grain": "one business entity", "logical_key": ["id"],
      "min_rows": 20, "dates": {"planned_end": "planned"}
    },
    "events": {
      "role": "event", "grain": "one entity event", "logical_key": ["id"],
      "min_rows": 10000, "dates": {"occurred": "actual", "finished": "actual"}
    }
  },
  "relationships": [
    {"table": "events", "columns": ["entity"], "parent": "entities", "parent_columns": ["id"], "nullable": false}
  ],
  "series": [
    {"name": "activity", "table": "events", "date_column": "occurred",
     "measure": {"aggregate": "sum", "column": "value", "unit": "usage units"},
     "monthly": "growth", "weekly": "weekday_heavy", "reason": "Expanding business usage"}
  ],
  "distributions": [
    {"name": "entity activity", "table": "events", "entity": ["entity"],
     "measure": {"aggregate": "sum", "column": "value", "unit": "usage units"},
     "shape": "long_tail", "reason": "Different entity sizes"}
  ],
  "sequences": [
    {"table": "events", "partition_by": ["entity"], "order_by": ["step"],
     "timestamp": "occurred", "end_timestamp": "finished"}
  ],
  "anomalies": [
    {"name": "temporary demand surge", "series": "activity", "start": "2026-02-09", "end": "2026-02-22",
     "direction": "up", "explanation": "A bounded demand event affects volume and downstream outcomes"}
  ],
  "not_applicable": {}
}
```

- Tables: exactly the supplied tables; roles are `dimension`, `fact`, `event`,
  `summary`. Declare each table's grain and logical key, including composite tuples.
  For genuinely keyless data use `logical_key: []` and a nonempty `keyless_reason`;
  this emits an unassessed warning. Set `min_rows` from required analytical coverage,
  before spending the remaining total budget on leaf/detail tables.
- Dates: classify **every** DATE/TIMESTAMP column as `actual`, `planned`, or `audit`.
  Only actual occurrences must be on/before the saved observation end date. A planned
  expiry is allowed in the future even in a dimension table. Never relabel an actual
  occurrence as planned merely to make a failure disappear.
- Relationships: sample parent tuples together; parent tuples must be unique/non-NULL.
  `nullable: true` uses MATCH SIMPLE (a child tuple with a NULL component is exempt);
  otherwise every child tuple must be present. No physical FK is needed for this check.
- Measures: `count` is row count and omits `column`; `sum`/`avg` require a numeric
  column. Always state the unit. Do not sum unrelated sensor units or average IDs.
  A series can restrict its cohort with `"scope": {"region": ["west"], "entity": [12, 18]}`;
  scope columns and values are validated and parameterized, never interpolated SQL.
  Time-shape checks currently require positive monthly aggregates; signed/net metrics
  need separate business assertions and a suitable positive activity series.
- Sequences: select the actual entity partition, event order and start/end timestamps.
  Partition + order must be unique; add tie breakers for repeated steps/rework.
  Ordered events may touch but must not overlap when `end_timestamp` is declared.
  Omit that field when only start-time monotonicity is meaningful. An inspection's
  row number or arbitrary defect number is not automatically an event sequence.
- Every rule category needs rules or a `not_applicable` explanation, exclusively.
  Example: `"sequences": "Independent point events with no ordered lifecycle"`.
  Missing categories fail; explanations emit WARN, not PASS. Ordinary activity demos
  should include time shapes, entity distributions and at least one bounded anomaly.
  Pure reference data or an explicitly requested steady-state simulation may differ.
- The runner saves versioned build metadata automatically. Do not hand-write its
  schema snapshot or remove declared keys to satisfy the post-import check.

## Weighted calendar: seasonality and weekday structure

Construct a TEMP calendar over the saved window. Separate long-term trend, recurring
seasonality, weekday multipliers and bounded shocks rather than embedding one giant
CASE expression per table. One generic positive weight is:

```sql
exp(growth_per_month * month_index)
* (1 + seasonal_amplitude * sin(2*pi()*day_index/365.25 + phase))
* weekday_multiplier
* shock_multiplier
```

Keep `abs(seasonal_amplitude) < 1`. Choose a realistic phase and amplitude from the
business; do not put demand curves on a physical measurement that should be stable.
For weekly shape, choose heavier weekdays/weekends explicitly; always-on operations
can use 1.0 every day with an explanation. Derive related downstream timestamps
from upstream events rather than resampling the calendar independently.

Sample with a deterministic cumulative weight table. Compute
`hi = sum(weight) over (order by day)`, `lo = hi-weight`, `total=sum(weight) over ()`;
for each root ID use `draw=u01(id, stream)*total` and join `lo<=draw AND draw<hi`.
CAST day offsets to INTEGER before adding to DATE. Normalize probabilities once;
do not use DuckDB integer casts as floor. Uniform `u01` draws become nonuniform
business dates through this CDF. A preaggregated daily-count allocation also works.

The gate observes the **declared measure**, not whichever numeric column looks big.
It requires at least three complete months. Additive series include zero-activity
days; averages exclude days with no observations. Monthly `growth` requires last/
first >=1.15; `decline` <=0.85; `stable` max/min <=1.3; `seasonal` max/min >=1.3.
Weekly weekend/weekday mean must be >=1.12 (`weekend_heavy`), <=0.88
(`weekday_heavy`), or between 0.88 and 1.12 (`flat`). The seasonal check establishes
variation, not repeated-year periodicity; keep causal/recurrence assertions when needed.
If the requested window is too short, record time-shape coverage as unassessed with
that reason instead of changing the user's date range.

## Persistent entity weights: long tails without contradictory dimensions

Assign a stable rank independently of business IDs. A generic smoothed rank weight
is `pow(rank + offset, -alpha)`; choose alpha/offset for the entity count, then use
the same cumulative-weight sampling as dates. Reuse the sampled parent and inherit
its attributes. Customer size, demand and equipment utilization are different
concepts: do not apply one distribution to every FK.

For intentional balance, assign near-equal weights. For long tails, generate a few
busy entities and many smaller ones, while retaining analytical coverage for the
tail. Avoid applying the same popularity weight again to quantity and price unless
the business actually requires that compound effect.

The gate uses a declared additive measure (`count` or `sum`) by a declared entity
tuple. Long tails need >=10 entities, Top ceil(10%) share >=30% (<60 entities),
>=40% (<200) or >=50% (otherwise), <=92%, and single-entity share <=50%.
Balanced allocations require positive totals for observed entities and head share
<=2/entity_count. Small categorical domains should normally declare balance, not
Zipf. These checks do not establish realism for unobserved parent entities; add
coverage assertions where all parents should participate.

## Scoped anomalies: cause first, consequences afterward

Create a small TEMP cause table with start/end, affected entity/region/group, and
effect parameters. Join roots to this scope and carry the cause through related
tables: a demand shock can change activity and workload; a process excursion can
change measurements, outcomes and alerts. Keep background noise independent and
derive flags from persisted measurements/limits, including rounding.

Use the user's existing description/reason/code columns to make the cause queryable;
never add schema columns silently. If the DDL has no cause field, document exact
scope and mechanism in table comments and README. For a localized effect define a
series with `scope` on the affected existing entity/group, and supplement with cross-table
assertions where needed. The independent anomaly rule compares
daily means in the named window to the immediately preceding equally long window:
`up` >=1.2x, `down` <=0.8x. Both windows must fit inside the observation period.
Prefer windows that span whole weeks to avoid confusing weekday effects with shocks.
Checking the effect alone does not prove causal attribution; retain cross-table
assertions for lineage and the shared cause's predicted outcomes.

## Comments and contract stability

Emit catalog metadata in `generate.sql`, for example:

```sql
COMMENT ON TABLE events IS 'One observed event per entity and sequence position';
COMMENT ON COLUMN events.entity IS 'Logical FK to entities.id; inherited from parent';
COMMENT ON COLUMN events.value IS 'Usage units; affected by the declared demand window';
```

Do this for every table and column, retaining supplied comments. The independent
gate checks nonempty comments both before and after import; it cannot judge their
semantic truth. Comments, contracts and assertions should describe the same grain.

`semantics.json` is locked on first execution. Fix generated data to satisfy it;
do not add exceptions after seeing failed statistics. A genuine contract correction
must use `datasource.py run --directory data --contract-change-reason "..."`, which
records both versions and the reason. Report corrections and all warnings at delivery.
