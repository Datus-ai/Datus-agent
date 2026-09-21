# Profile reference

**Reading this file is enough to write a profile. You do not need the engine source.**

The engine (`scripts/ddl_engine.py`) owns *how* data is produced; the profile states *what the
business rules are*. Every field is optional - an empty `{}` still produces structurally correct
data. The profile is what makes it look real.

---

## 1. What the engine does on its own

| Capability | Notes |
|---|---|
| Table roles | date_dim / dim / fact / detail / downstream / event / metric_daily / snapshot, inferred structurally, never from name prefixes. `date_dim` and `metric_daily` share a shape - a date grain, no foreign key, a few measures - and are told apart by whether every column is a calendar attribute (`year_num`, `quarter_cd`) or the table carries something the date does not determine (`channel`, `gmv`). Force either with `roles` |
| Primary and foreign keys | **Declared PRIMARY KEY / FOREIGN KEY / UNIQUE win**; inference only fills gaps. A foreign key may reference a UNIQUE column, not only a primary key, and the child is sampled from the column the DDL names. Renamed keys (`deal.buyer -> cust.cid`) still connect. **A COMPOSITE `PRIMARY KEY (a, b)` is parsed but not honoured end to end**: generation does not know about it (`pk_of` falls back to the first column), so the combination is unique only by luck, and the quality check verifies that first column alone |
| Enum domains | **Extracted from DDL inline comments** (`order_status VARCHAR, -- pending / paid / shipped`). `report()` lists which columns were extracted and which end in `...` (incomplete) |
| Column semantics | id / date / ts / amount / count / ratio / enum / flag / name / seq / measure, from name + type |
| Row allocation | Fact layer 65-75%, dimensions derived from business density, two-pass total calibration (within 6%) |
| Denormalised copies | A column this table declares that its PARENT also carries is copied down from the parent row, not sampled again - so `flights.origin_code` agrees with the route it points at |
| The 17 invariants | Weighted calendar sampling, derived-from-base quantities, child events anchored to parents, monotonic sequences, complete terminal states, FKs sampled from upstream only, layer backfill, stock baseline, zero header/detail amount drift |
| Type contract | Tables are created with the declared types (a BIGINT key stays BIGINT; DECIMAL(18,2) does not become DOUBLE) |
| Table/column comments | Written per role; overridable in the profile |
| Structural metadata | Written as `.<db stem>.meta.json` **beside the database**, so a build at `data/_build/` puts it there. It is derived from `gen.py`, not a deliverable: the quality check reads it during the same run and it goes when `_build/` does |

**Run `eng.report()` first and override only what is wrong.**

### 1.1 Row allocation: the engine does it, you do not

`_plan_rows` runs during inference and `report()` prints the result per table. **Never compute the
split by hand** - it is the single most expensive way to burn a run. The rule it applies:

| Step | What the engine does |
|---|---|
| Budget | `rows * 0.94`; the remaining 6% is headroom for dimensions |
| Fact layer | Facts and their details share one block of the budget: `0.52 + 0.28`, or `0.32 + 0.28` when an event or snapshot table exists. Within that block a fact weighs 1 and a detail weighs its lines per parent - `1.8` for a detail table, `0.6` for a downstream fact - so a detail table always comes out above its parent, in the 1.4-2.2 band `references/design-from-scratch.md` section 1.2 states. Event stream `0.34`, snapshot `0.20`. Shares are renormalised over the roles that actually exist, so a DDL with no detail table gives its share to the main fact |
| Dimensions | `main_fact_rows / DIM_DENSITY[kind]`, then clamped to `[4, 8% of rows]` |
| Daily metric table | `max(number of days, 7% of rows)`; the real count is `days x dimension combinations`, and `report()` prints the combinations it will build |
| Date dimension | Exactly the number of days |
| Pinned | `table_rows` / `dim_rows` win over all of the above |

`DIM_DENSITY` is fact rows per entity, keyed by dimension kind. These are the exact constants;
the kind is guessed from the table name and corrected with `dim_kinds`:

| kind | density | Meaning |
|---|---|---|
| `enum` | 1500 | Carrier / channel / payment / plan / product line - the business has only a few |
| `org` | 600 | Seller / store / warehouse / line / department / team. **Also the fallback when the name matches nothing** |
| `staff` | 300 | Driver / doctor / support rep / agent / teacher |
| `item` | 60 | SKU / course / item / procedure |
| `customer` | 4 | B2C buyer / patient / student - this number *is* the repurchase rate |
| `customer_b2b` | 10 | B2B account |

The total lands within 6% of `rows` after two-pass calibration. If a table's count is wrong, pin
it or fix its kind - do not go looking for the formula.

### 1.2 Constructor arguments

These are arguments to `DDLEngine(...)`, not profile fields. The profile cannot set them. The
built-in `plan_datasource(ddl=, rows=, months=, end_date=)` tool takes the first four and prints
the resulting plan without a generator existing, which is the cheapest way to check them:

| Argument | Default | Meaning |
|---|---|---|
| `ddl` | - | The CREATE TABLE statements, DuckDB syntax |
| `rows` | `80_000` | Total row budget across all tables; see 1.1 |
| `profile` | `{}` | Everything in section 2 |
| `months` | `17` | Length of the window, counted back in whole months from the month containing `end_date` |
| `end_date` | **yesterday** | Last day of data, a `datetime.date`. The window is `[first day of the month, months-1 months back .. end_date]`. Pass it when the data must end on a fixed day - a demo that has to look the same next week, or a business description that names a cut-off |
| `seed` | `42` | Fixed seed; same inputs produce the same database |
| `extra_tables` | `"none"` | `"none"` builds exactly the DDL tables. `"date_dim"` adds a date dimension, `"summary"` a dws/ads layer, `"all"` both. Extra tables carry no user DDL constraints and give the agent competing definitions, hence off by default |

`report()`'s first line prints the resolved window, so you can check `months` / `end_date` without
computing dates: `knobs: months=17 (502 days, 2025-05-01~2026-09-14) seed=42`.

### 1.3 Engine defaults you do not have to look up

Every value below is what the engine does with no profile at all. A production run went into
`ddl_engine.py` twelve times for exactly these, one question at a time, and the file being readable
on disk made the "never read the engine" rule unenforceable - so here they are.

| Question | Answer |
|---|---|
| Amount identity on a fact row | `paid = gross - discount + shipping + tax`. Discount is 0 for 38% of rows, otherwise 2-22% of gross, capped at 90%. Shipping is 0 for 42% of rows, otherwise a flat 3-22 (**additive, never a fraction of the goods**). Tax is 0-8.5% of `gross - discount`. **Coupon is not part of the identity** - it is 0 for 58% of rows, otherwise 30-85% *of the discount* |
| Cost vs price | A cost column is never allowed above the price column. A `columns` bound `<= 1` reads as a cost *ratio*; `> 1` as an absolute range, still clamped to the price |
| Flag columns | True with p = 0.93 on a dimension, p = 0.88 on a fact. Override per column with `columns: {"t.col": {"p": 0.7}}` |
| Lines per document | The row allocation decides the average (1.8 by default, see 1.1), and the spread around it is 1-5 lines with weights 0.52 / 0.26 / 0.12 / 0.06 / 0.04 scaled to that average. **A document always has at least one line, and that beats the plan**: pin a detail table below its parent and the plan keeps your number - `_plan_rows` and `precheck` both honour an explicit pin - while generation still produces one line per parent, so the table comes out larger than the printed plan. Pin the parent instead and the lines follow it |
| Detail line quantities | A detail row draws 1-4 with weights 0.71 / 0.19 / 0.07 / 0.03. **Not the same as a generic `count` column on a fact table**, which draws 1-5 with 0.52 / 0.26 / 0.12 / 0.06 / 0.04. Either is overridden by `derive` |
| Daily metric grid width | `days x dimension combinations`, capped at `row budget / number of days`, and capped again by what the schema allows. `report()` prints both the plan and the binding cap |
| Same-named columns on a fact | Inherited from the FK'd dimension row rather than redrawn, so `category` on an order line matches the product's |
| Effective dates | `before_start` and `baseline_share` apply to a **dimension's** date columns (invariant 16: 45% of entities exist before the window). They do not apply to fact dates |
| Name column order | Within a dimension row, name columns are filled **last**, so a `naming` template can reference the other columns of that row (see `naming` below) |

### 1.4 Business codes

A column whose name ends in `no`, `code`, `sn`, `serial`, `sku`, `number`, `barcode` or `ref` (on a
word boundary) can be filled with a generated code rather than a random string: the first three
letters of the column name upper-cased, then the row's date as `yymmdd` when the table has one, then
a 1-based 6-digit counter - `ORD260718000001`, or `SKU000117` on a table with no date.

**It is a fallback, not a rule.** The code fills a column only when nothing more specific claimed
it: the primary key, a foreign key, a joint group, and every column carrying a `name` / `enum` /
`date` / `ts` / `amount` / `count` / `ratio` / `flag` / `measure` semantic are all filled by their
own branch first. So `sku_code` classified as an enum gets enum values, and only a column left as
free text ends up with a code.

`report()` lists exactly the columns that will get one (`generated business codes: ...`) - it uses
the same predicate the generators do, so trust the line rather than the name pattern. There is no
knob: to force a different shape, set the column's semantic to `name` in `semantics` and template it
with `naming`; to force a code onto an enum-looking column, set it to `text` in `semantics`.

---

## 2. Field index

| Field | Purpose | When it is required |
|---|---|---|
| `calendar` | Promotion / trough / anomaly calendar | **Almost always** - without it the data has no explainable movement |
| `conditional` | Different enum weights or numeric ranges per group | Whenever dimensions must differ (refund rate by category, conversion by channel) |
| `formulas` | Arithmetic between columns | Accounting identities, cost/margin and other derived columns |
| `enums` | Enum domains and weights | Only when the DDL comments are incomplete |
| `event_seq` | Status sequence of an event stream | Whenever there is an event table |
| `table_rows` / `dim_rows` | Pin a table's row count | The user asked for a specific row count |
| `dim_kinds` | Dimension kind (drives cardinality) | Inference is wrong |
| `columns` | Per-column value ranges | Amount/quantity magnitudes are unreasonable |
| `naming` | Name templates, per table | A specific naming style is needed |
| `vocab` | Dataset-wide name vocabulary | **Any non-retail industry** - the built-in words are brands and store suffixes |
| `head_share` | Top-10% concentration target | Default 0.74; change for more/less concentration |
| `tier_cols` / `tier_bands` | Tier column and bands | **Required when a dimension has more than one column containing tier/level** (see pitfalls) |
| `roles` | Force a table role | `report()` got it wrong and cannot self-correct |
| `semantics` | Force a column's semantic | `report()` classified a column wrong - common outside retail naming |
| `joint` | Joint distributions (region x city) | Combinations must stay legal |
| `lifecycle` | Lifecycle timestamp chain | Multiple timestamps (created/paid/shipped/...) |
| `derive` | Derive a quantity from a base quantity | Funnel metric tables |
| `summary_dims` | Dimensions of the summary layer | `extra_tables="summary"` is on and the defaults do not fit |
| `refund_rate` | Probability a detail line carries a refund | The default 5.5% is wrong for the industry |
| `effective_col` | Which column makes an entity usable | Inference picked the wrong date, or there is none to find |
| `no_date_dim` / `date_dim_name` | Suppress or rename the auto-built date dimension | Only with `extra_tables` including `date_dim` |
| `pre_sql` / `extra_sql` | Business post-processing SQL. **One SQL string, or a list of statements** (`["UPDATE ...", "UPDATE ..."]`); anything else is refused by `precheck()` before generating. `pre_sql` runs before the summary layer, `extra_sql` after | **Last resort**, when nothing above can express it |
| `weekly_shape` | How a week looks: `weekend_heavy` (default, consumer retail), `weekday_heavy` (B2B, payroll, clinics, booking desks) or `flat` (metering, sensors, always-on). **Set it** - the default is a consumer shop, so a B2B dataset left alone has its busiest days on the weekend. `check_datasource_quality` reads it from the generator metadata and verifies *that* shape, so `flat` passes as flat rather than failing for having no cycle - and declaring a shape the data does not show still fails | an unsupported value is refused by `precheck`, not swapped for the default; `weekend_lift` overrides the number outright |
| `trend_mom` | Month-over-month growth | Default 0.031 |
| `table_comments` / `column_comments` | Comments | Recommended wherever a definition is not obvious |

---

## 3. Field details

### calendar - time signal (most important)

```python
"weekly_shape": "weekday_heavy",    # weekend_heavy (default) / weekday_heavy / flat
"calendar": {
  # (start MM-DD, end MM-DD, intensity multiplier, name); expanded across years automatically
  "promos": [("11-27", "11-30", 5.2, "Black Friday / Cyber Monday"), ("06-16", "06-18", 3.2, "618")],
  "slows":  [("02-10", "02-20", 0.52, "Spring Festival shutdown")],
  # Anomalies: `at` is a relative position 0-1 in the range; `scope` limits the
  # attribution dimension, and an empty {} means global
  "disruptions": [
    {"at": .30, "days": 26, "factor": .45, "scope": {"site_cd": "JP"}, "name": "JP payment outage"},
    {"at": .62, "days": 30, "factor": 1.9, "scope": {"courier_id": "CR003"}, "name": "Hub relocation"},
  ],
}
```

- **Multiplier guidance**: top-tier promotion 4-5x, mid-tier 2-3x, shutdown 0.5-0.6x. For anomaly
  multipliers see the believable-range table under invariant 12 in `pitfalls.md`
- **`scope` keys must be real column names** - writing `site` when the column is `site_cd` means the
  anomaly never lands. The pre-check warns about this
- Anomalies on fact tables currently only work in the **suppressing** direction (`factor < 1` drops
  rows probabilistically). `factor > 1` has no effect on fact tables; use `conditional` or `pre_sql`
  for amplifying anomalies

### conditional - business differentiation

```python
"conditional": {
  # Enum column: different value weights per group (channel is a column of orders itself)
  "orders.order_status": {"__by__": "channel",
                          "social":      {"paid": .78, "refunded": .17, "cancelled": .05},
                          "paid_search": {"paid": .88, "refunded": .07, "cancelled": .05},
                          "__default__": {"paid": .85, "refunded": .10, "cancelled": .05}},
  # Numeric column: different [lo, hi] per group (category is a column of products itself)
  "products.list_price": {"__by__": "category",
                          "Electronics": [200, 900], "Clothing": [20, 150],
                          "__default__": [30, 300]},
}
```

#### Two forms of `__by__`: same-table and cross-table

```python
"conditional": {
  "orders.order_status":     {"__by__": "channel", ...},           # same table
  "order_items.unit_price":  {"__by__": "products.category", ...}, # cross-table: "upstream_table.column"
}
```

Cross-table grouping (refund rate by category, margin by category - the most important
differentiations in e-commerce) is **supported directly**: write `upstream_table.column` and the
engine follows this table's foreign key to that table to read the value. It requires a **foreign-key
path** (declared or inferred); without one the pre-check errors out.

Measured: `item.unit_price` grouped by `prod.category` puts 3C in [500,900] and apparel in [30,90],
cleanly separated.

| Case | How to write it |
|---|---|
| Grouping column is on this table (channel, platform, status) | `"__by__": "channel"` |
| Grouping column is on an upstream dimension (category, brand, region) | `"__by__": "products.category"` |
| A dimension grouped by its own upstream | Not supported yet; the pre-check warns and `__default__` applies |

- A same-table `__by__` column must be generated before the target column (enum columns always come
  before amount columns, so an enum is the safest grouping key). Cross-table lookups have no such
  ordering constraint
- Without `__default__`, unlisted values fall back to the engine default - the pre-check warns
- **On a fact table**: the numeric range applies to the **first amount column** only; the remaining
  amount columns are derived from it by accounting role
- **On a dimension table**: **every amount column takes its own** `conditional`, so price band and
  cost ratio by category can be configured separately
- Cost columns on a dimension have one extra rule: an upper bound **<= 1 is read as a cost ratio**
  (a fraction of price), **> 1 as an absolute amount** (clamped to not exceed the price). That is how
  a per-category margin gradient is configured.
  The fraction is of the **list** price, and the line is sold at a discount off that, so the margin
  the data ends up showing is *narrower* than `1 - ratio`. A production run set `[.80, .88]` on one
  category expecting a 12-20% margin and measured 1.8%, because a ~15% line discount came off after:
  its per-category margin gradient came out 30x and failed its own assertion. Leave room for the
  discount - `[.65, .75]` on that category measured 4.2x across categories and passed.

```python
"products.list_price": {"__by__": "category", "3C": [200, 900], "Beauty": [30, 200]},
"products.cost_price": {"__by__": "category", "3C": [.80, .88],   # cost ratio -> 12-20% margin
                        "Beauty": [.35, .45], "__default__": [.5, .65]},
```

- **Supported on dimension, fact and detail tables alike**, with two limits.
  *Grouping across tables* (`"__by__": "products.category"`, reading the value from the row a foreign
  key points at) works on **fact and detail** tables, which carry the key. On a **dimension** the
  grouping column must be one of that table's own columns - `products.cost_price` grouped by
  `category` is fine, grouped by anything outside `products` it silently falls through to
  `__default__`.
  *The cost-ratio reading* is the other: it belongs to `_gen_dim`, so `products.cost_price` reads
  `[.80, .88]` as a ratio while `order_items.unit_cost` does not. A detail table's cost column needs no conditional at all -
  it already follows the cost of the product the line references, so the per-category gradient you
  configure on the dimension arrives on the detail rows on its own.

### semantics - correcting what a column is

```python
"semantics": {
    "encounters.insurance_paid": "amount",
    "encounters.self_paid":      "amount",
    "encounters.diagnosis_code": "enum",
},
```

Legal values: `id`, `date_pk`, `date`, `ts`, `amount`, `count`, `ratio`, `enum`, `flag`, `name`,
`seq`, `measure`, `text`.

The engine classifies a column from its name, which is a naming convention, and naming conventions
are per-industry. It reads `paid_amount` as money but not `insurance_paid`, `copay`,
`premium_received` or `principal` - measured on a hospital schema it got 32 of 34 columns right, and
the two it missed were both money. Rather than growing the pattern list one industry at a time, the
caller states the ones it disagrees with; an LLM reading the DDL judges this far better than a
regex, and freezing that judgement in the profile keeps the run reproducible and the decision
auditable.

- `report()` prints the full inferred mapping - read it and override only what is wrong
- Applied **before** role detection, so a corrected amount column also counts towards choosing the
  main fact table
- A key declared in the DDL keeps `id` unless explicitly overridden (the pre-check warns if you do,
  because foreign keys to it stop resolving)
- The pre-check rejects an unknown semantic or a missing column. Without that a typo would leave the
  column on its wrong guess and produce plausible-looking wrong data that no quality check can catch

### formulas - arithmetic between columns

```python
"formulas": {
  "orders.paid_amount":       "original_amount - discount_amount + shipping_amount + tax_amount",
  "order_items.total_cost":   "unit_cost * quantity",
  "order_items.gross_profit": "sales_amount - total_cost",     # depends on total_cost; ordered after it
}
```

- Expressions may only reference columns of **the same row of the same table**; `+ - * / ( )` and
  `min/max/abs/round` are supported
- **Topologically sorted automatically** by reference; circular dependencies fail the pre-check
- Enforced row by row, so there is no drift

### derive - a quantity derived from a base quantity (the core of funnel metric tables)

```python
"derive": {
  # "table.target": {"from": base column, "ratio": (lo, hi), "min": floor (optional, integer columns)}
  "daily_channel_metrics.clicks":         {"from": "impressions",   "ratio": (.010, .030)},
  "daily_channel_metrics.sessions":       {"from": "clicks",        "ratio": (.80, .95)},
  "daily_channel_metrics.product_views":  {"from": "sessions",      "ratio": (2.5, 4.0)},
  "daily_channel_metrics.add_to_carts":   {"from": "sessions",      "ratio": (.08, .15)},
  "daily_channel_metrics.checkout_users": {"from": "add_to_carts",  "ratio": (.30, .45)},
  "daily_channel_metrics.purchasers":     {"from": "checkout_users","ratio": (.45, .60)},
  "daily_channel_metrics.attributed_orders": {"from": "purchasers", "ratio": (.95, 1.05)},
  "daily_channel_metrics.ad_spend":       {"from": "clicks",        "ratio": (2.5, 6.0)},
  "daily_channel_metrics.attributed_revenue": {"from": "attributed_orders", "ratio": (180, 320)},
}
```

- `from` must be a column of **the same table** generated before the target. Amount and ratio columns
  keep two decimals; count columns are rounded to integers
- `ratio` is a multiplier range and may be < 1 (converging) or > 1 (expanding, e.g.
  sessions -> product_views)
- **`ratio` can itself vary by group** (per-channel CTR/CVR differences, which previously required SQL):

```python
"metrics.clicks": {"from": "impressions",
                   "ratio": {"__by__": "channel",              # same-table column or "upstream.column"
                             "paid_search": (.04, .06),        # search ads CTR 4-6%
                             "social":      (.010, .015),      # social 1-1.5%
                             "email":       (.08, .12),
                             "__default__": (.02, .03)}},
```

Measured: email CTR 9.92% / paid_search 5.0% / social 1.25%, each inside its configured band.

- **Chained derivation follows column order in the DDL** - write the funnel columns in business order
  and the chain forms automatically
- Without `derive`, every count column is randomised independently and the funnel does not converge
  at all (measured: CTR reaching 43% under defaults)
- **Do not back-solve the top of the funnel.** A chain is relative: it says clicks are 1-3% of
  impressions, not what impressions should be so that `attributed_revenue` matches the order
  table's GMV. That is a cross-table constraint the chain cannot express, and arithmetic in your
  head is the expensive way to approximate it. Generate once and read the achieved ratio out of an
  assertion - a 9-second run answers it exactly
- **To move that ratio, turn the knob at the *end* of the chain**, not the head. Widening or
  narrowing `impressions` barely moves it: the calendar multipliers and the day x dimension
  weights scale the whole chain together, so the head's range mostly cancels out. The ratio is set
  by the last conversion step - `attributed_orders`' `ratio` against `purchasers`. A production
  run followed the head instead, cut `impressions` from `(2000, 8500)` to `(1800, 6800)`, watched
  attributed/actual go the **wrong way** (1.22 -> 1.32), and spent three turns assuming a stale
  cache before fixing it at `attributed_orders` - where `(0.95, 1.05)` -> `(0.72, 0.80)` landed it
  first try

### refund_rate, effective_col, no_date_dim, date_dim_name

Four scalars the engine reads and cannot infer. They are small, but an undocumented knob is worse
than a missing one: a measured production run spent ten minutes disassembling the engine's bytecode
to work out what `refund_rate` did.

```python
"refund_rate": 0.055,                            # default 0.055
"effective_col": {"products": "launched_at"},    # per table; "__auto__" (the default) infers it
"no_date_dim": False,                            # skip the auto-built date dimension entirely
"date_dim_name": "dim_date",                     # name it something else
```

- **`refund_rate`** is the probability that one detail line carries a refund quantity, applied when
  generating the detail table. **It is a single global value - there is no per-group form.** A
  parent already in a refund status refunds at 0.82 regardless. To vary the rate by category or
  region, restate it afterwards with `pre_sql`; `conditional` does not reach this field.
- **`effective_col`** names the column that makes an entity usable - registered / hired / listed /
  opened - so facts referencing it cannot predate it (invariant 3). The engine infers it from the
  column name; set this when it picks the wrong date, or when the name is one it does not know.
  `"__auto__"` restores inference for a single table.
- **`no_date_dim`** and **`date_dim_name`** only matter when `extra_tables` includes `date_dim`:
  the first suppresses the auto-built dimension, the second renames it from `dim_date`.

### joint - joint distributions (keeping combinations legal)

```python
"joint": {
  "customers": [                                   # a table may carry several groups
    {"cols": ["country", "province", "city"],
     "values": [["China", "Guangdong", "Shenzhen", 12],   # trailing number is the weight (default 1)
                ["China", "Zhejiang", "Hangzhou", 8],
                ["USA", "California", "Los Angeles", 5]]},
  ],
  "daily_channel_metrics": [
    {"cols": ["channel", "source", "campaign_name", "platform"],
     "values": [["paid_search", "Google", "brand_always_on", "web", 20],
                ["social", "TikTok", "spring_launch", "ios", 15]]},
  ],
}
```

- Prevents dirty combinations such as `country=China + province=California`, or
  `organic_search + TikTok ad`
- **Fact tables inherit same-named columns from the foreign-key entity automatically**: once
  `customers` carries the geography combination, `orders.country/province/city` need no configuration
- **A daily metric table (`metric_daily`) uses only the first group**, and the combination count is
  truncated to `rows / days` - to get 16 combinations x 518 days, `table_rows` must be >= 8288 or the
  combinations get cut

### lifecycle - multi-stage business timestamps (created -> paid -> shipped -> completed)

```python
"lifecycle": {
  "orders": {
    # One entry per business timestamp, in DDL column order. Entry [0] belongs to the first
    # timestamp, which is the anchor and has nothing before it - write 0 and it is ignored. So four
    # timestamps take four numbers, of which three are real gaps.
    "gap_hours": [0, 6, 30, 72],          # order->paid 6h, paid->shipped 30h, shipped->completed 72h
    # Status -> how many stages it reaches (decides which later timestamps are NULL)
    "stages": {"pending": 1, "paid": 2, "shipped": 3, "completed": 4,
               "cancelled": 1, "refunded": 4},
    # In-flight statuses can only occur near the cut-off date (a year-old order cannot still be pending)
    "in_flight": ["pending", "paid", "shipped"],
  },
}
```

- Timestamps are strictly increasing and truncated by `stages`: `cancelled` keeps only `order_time`
  and the rest are NULL
- When the cut-off truncates a row, **the status is downgraded automatically** - there is never a
  "completed with an empty completion time"
- A status listed in `in_flight` that lands on an early date is rewritten to a terminal status.
  Without this you get "an order from a year ago is still pending", which reads as fake instantly
- Without `lifecycle`, every timestamp is randomised independently: out of order within a day and
  inconsistent with the status

### columns - per-column parameters (full key list)

```python
"columns": {
  "dim_product.list_price":   {"range": (9, 320)},        # numeric range (amount/count/measure)
  # A measure has no inferable units, so without a range it falls back to 0.1-40 - a plausible
  # weight or duration and nonsense for a score or a rate. precheck names every measure still on
  # it. A lower bound of 0 is fine: those draw from a bell centred in the range, not a lognormal.
  "dim_customer.reg_dt":      {"before_start": (30, 900)},# days before the range start (dimension attribute date)
  "dim_customer.is_vip":      {"p": 0.08},                # probability a boolean/flag column is 1
  "dim_product.cost_price":   {"cost_ratio": (.4, .65)},  # cost as a fraction of price
  "ods_order.channel":        {"values": [...], "weights": [...]},  # enum domain and weights for this column only
  "dim_seller.seller_name":   {"vocab": {...}},           # name vocabulary (also configurable via `naming`)
  "customers.reg_dt":         {"baseline_share": 0.45},   # share of entities existing before the range start (stock baseline)
}
```

### enums / event_seq

```python
"enums": {                          # matched by column name across tables; no need to repeat DDL comments
  "order_status": {"paid": .845, "refunded": .08, "cancelled": .05, "pending": .025},
  "event_type_cd": ["PICKUP", "ARRIVE_HUB", "CUSTOMS_CLEAR", "DELIVERED"],   # a list means equal weights
},
"event_seq": {"ods_tracking_event": ["PICKUP", "ARRIVE_HUB", "CUSTOMS_CLEAR", "DELIVERED"]},
```

### Row counts and cardinality

```python
"table_rows": {"customers": 12000, "orders": 30000},   # pinned tables skip two-pass calibration
"dim_kinds": {"agent_master": "staff", "product_catalog": "enum"},
#   enum (1500 facts per entity) / org (600) / staff (300) / item (60) / customer (4) / customer_b2b (10)
"columns": {"dim_product.list_price_usd": {"range": (9, 320)},
            "dim_customer.reg_dt": {"before_start": (30, 900)}},
"head_share": {"dim_seller": 0.74},                    # Top-10% share target
```

### Naming and tiering

```python
# Dataset-wide vocabulary: the built-in words are retail-flavoured, so any other industry should
# replace them once here rather than per table. Keys: brand / org_suffix / person / given / item,
# plus name_sep and name_order for scripts that do not write "Given Family".
"vocab": {"brand": ["Cardiology", "Neurology", "Oncology"], "org_suffix": ["Ward", "Unit", "Clinic"],
          "item": ["Standard", "Extended", "Follow-up"]},
"naming": {"dim_seller": {"tpl": "{brand} {org_suffix}",
                          "vocab": {"brand": ["Lumora", "Nordvik"]}}},   # per-table override
"tier_cols": {"customers": "member_level"},            # see pitfall 1 below
"tier_bands": [(.05, "platinum"), (.2, "gold"), (.5, "silver"), (1.0, "normal")],
```

**Vocabulary resolution** is three layers, later wins: the engine's built-in words, then
`profile["vocab"]` (one override for the whole dataset), then `naming[table]["vocab"]`. Override at
the widest layer that is correct - a hospital dataset wants `profile["vocab"]`, not one entry per
table.

**Without `tpl`** the shape follows the table's dimension kind: `customer` / `staff` produce a
person name (`person` + `given`, joined per `name_sep` / `name_order`), `item` produces
`{brand} {item}`, and everything else `{brand} {org_suffix}`, with a `(#2)`, `(#3)` suffix once the
brand list is exhausted so names stay unique.

**With `tpl`** it is a `str.format` template, and the fields come from two places:

| Source | What it contributes |
|---|---|
| The merged vocabulary | One random pick per list-valued key. Scalars (`name_sep`, `name_order`) are not offered as fields |
| The row being built (`ent`) | Every string column already generated on that row, **overriding the vocabulary under the same key** |
| The engine | `{i}` and `{n}`, both the 1-based row number |

That override is the useful part: name columns are generated *last* within a dimension row, so
`"tpl": "{brand} {item}"` on a table that also has a `brand` column produces a name carrying that
row's actual brand, not an unrelated draw. If the table has no such column, the same template
still works and draws `brand` from the vocabulary. A key missing from both raises `KeyError` - the
template is not tolerant of typos.

---

## 4. Believable metric ranges (follow this and half the rework disappears)

The engine does not own metric magnitudes - under defaults ROAS reaches 26, CTR reaches 43%, and
attributed orders come out 68x actual orders. **These numbers can only be calibrated by hand.** Use
this table when configuring `derive` / `conditional`:

| Metric | Believable range | Note |
|---|---|---|
| Display ad CTR | 0.5% - 3% | Search ads reach 3-6%, feeds 0.8-2% |
| Click -> session | 80% - 95% | Some bounce |
| Session -> product view | 2.5 - 4.0 per session | This expands, it does not converge |
| Session -> add to cart | 8% - 15% | |
| Add to cart -> checkout | 30% - 45% | |
| Checkout -> payment | 45% - 65% | |
| **Site-wide CVR** (orders/sessions) | **1% - 3%** | Direct reaches 3-4%, social 1-1.5% |
| CPC | 2 - 8 (currency units) | Brand terms low, competitive terms high |
| **ROAS** | **2 - 8** | Affiliate/email 6-9, paid search 3-6, social 3-5; organic has no spend -> NULL |
| **CAC** | 0.2 - 0.4 x average order value | Above 1x means the model is wrong |
| Marketing cost ratio (spend/GMV) | 8% - 15% | |
| Attributed orders / actual orders | 0.85 - 1.15 | Paid channels over-attribute, direct under-attributes |
| E-commerce refund rate | 5% - 12% | Apparel 15-18%, 3C 4-5% |
| Gross margin | 15% - 65% | Beauty 55-70%, 3C 10-20%, FMCG 15-25% |
| Repurchase | 2 - 6 orders per customer | B2B 8-15 |
| Top-10% customers' GMV share | 50% - 70% | Above 80% means Zipf is over-concentrated |
| Order status mix | completed 75-85%, cancelled 8-13%, refunded 5-12% | |

**Calibration order**: fix the outermost base quantity (impressions/visitors) first, configure the
`derive` ratios level by level, then run a question-validation pass reading CTR/CVR/ROAS back out.
Do not wait until everything is configured to check.

---

## 5. Six easy mistakes

**1. Without `tier_cols` the wrong column gets picked.** The engine finds the tier column by matching
`tier|level|grade|segment`, so `city_tier` wins first and then gets overwritten with
platinum/gold by GMV rank - business nonsense. **When a dimension has more than one column containing
tier/level, name it explicitly.**

**2. `scope` keys are column names, not concepts.** Writing `{"site": "JP"}` when the column is
`site_cd` means that anomaly never happens, and the quality check cannot detect it (the data is
perfectly legal). The pre-check warns; do not ignore it.

**3. Cross-table grouping needs the qualified name.** `"__by__": "category"` on `order_items` is
rejected by the pre-check (that table has no such column); the correct form is
`"__by__": "products.category"`. `derive`'s `from` still accepts same-table columns only (the base
quantity must be on the same row), but its `ratio` can be grouped by a cross-table column.

**4. Do not let a promotion window cross the cut-off date.** When `calendar` promos are expanded
across years, **a window entirely outside the data range is dropped, not truncated**. Configuring
`("08-25", "09-05", 1.6, "Back to school")` with data ending 08-31 makes the 2026 occurrence vanish
completely, and the final month's YoY can collapse from +40% to +3%. The data is legal and the
quality check cannot see it - only a YoY question exposes it. Either pull the window inside the
cut-off, or move the cut-off past the window.

**5. `pre_sql` cannot UPDATE a foreign-key column.** DuckDB refuses to rewrite a column that
children reference, so `UPDATE flights SET origin_code = ...` fails once anything points at it -
and a run that met this spent rounds working out why a statement its pre-check had validated
still would not execute. The pre-check plans against an empty schema, where no child rows exist
yet, so it cannot see this coming.

Two ways out, and the first is usually right: **do not restate what the engine already aligns.**
A denormalised column that also exists on a parent table is copied down from the row it belongs
to, so `flights.origin_code` already agrees with its route without any SQL. When the value really
is yours to compute, drop the `REFERENCES` on that column - a denormalised copy does not need its
own constraint, the parent's key already enforces the domain.

**6. `pre_sql` is a last resort.** It runs before the automatic summary layer and can restate metrics
and backfill across tables. But anything `conditional` / `formulas` can express should not be SQL -
SQL bypasses the engine's consistency guarantees and is not reusable.

---

## 6. Performance and re-runs

- **Calibration results are reused**: the first generation may run two calibration passes, but as long
  as the DDL and the row-affecting parameters are unchanged (`rows` / `table_rows` / `dim_rows` /
  `dim_kinds` / `roles` / date range / seed), the previous calibration is reused and generation is
  **single-pass**. The output says "reusing previous calibration". This matters most while iterating -
  measured 3.85s -> 1.41s
- Changing any of those parameters changes the fingerprint and triggers recalibration (two passes).
  That is expected
- A row-count deviation beyond +-25% prints a loud warning - usually the main fact table was
  classified as a dimension or metric table. Override with `profile["roles"]`; do not ship it as-is

---

## 7. Pre-check

`generate()` **runs the pre-check before generating** so misconfiguration is caught up front:

| Level | Checks |
|---|---|
| Error (blocks generation) | Reference to a non-existent table/column (`columns`/`conditional`/`formulas`/`table_rows`), a formula referencing a missing column, self-reference or a dependency cycle, `conditional` missing `__by__`, anomaly `at` outside 0-1 |
| Warning (generation continues) | A comment referencing a missing object, `conditional` missing `__default__`, an anomaly `scope` column that does not exist, a dimension exceeding 8%/15% of total rows |

It can also be run alone: `eng.precheck(strict=False)` prints without raising.

---

## 8. Minimal working example

```python
import pathlib, sys
sys.path.insert(0, "<skill>/scripts")
from ddl_engine import DDLEngine

HERE = pathlib.Path(__file__).resolve().parent          # data/
DDL = (HERE.parent / "schema.sql").read_text()

PROFILE = {
    "calendar": {
        "promos": [("11-27", "11-30", 5.2, "Black Friday"), ("06-16", "06-18", 3.2, "618")],
        "slows":  [("02-10", "02-20", 0.52, "Spring Festival")],
        "disruptions": [{"at": .35, "days": 25, "factor": .45,
                         "scope": {"channel": "paid_search"}, "name": "Ad account suspended"}],
    },
    "conditional": {
        # configured on the upstream dimension; the gradient propagates to details
        "products.list_price": {"__by__": "category",
                                "Electronics": [200, 900], "Clothing": [20, 150],
                                "__default__": [30, 300]},
    },
    "formulas": {
        "orders.paid_amount": "original_amount - discount_amount + shipping_amount + tax_amount",
        "order_items.total_cost": "unit_cost * quantity",
        "order_items.gross_profit": "sales_amount - total_cost",
    },
    "table_rows": {"customers": 12000, "products": 2000, "orders": 30000},
    "tier_cols": {"customers": "member_level"},
}

eng = DDLEngine(DDL, rows=110_000, profile=PROFILE, months=17)
eng.report()                                            # inspect inference first
eng.generate(str(HERE / "_build" / "datasource.duckdb"))   # never the datasource's own file
```
