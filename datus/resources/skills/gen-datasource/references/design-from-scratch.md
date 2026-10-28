# Designing a schema from scratch

**You are almost certainly on Path A and do not need this file.** Path A starts from a DDL, and the
engine allocates rows, sizes dimensions and enforces the invariants on its own - `plan_datasource`
prints what it decided and `profile_skeleton` hands back a profile that already runs.

This file is for the two cases where there is no DDL to start from: designing the schema yourself
(Path B) or writing a generator by hand (Path C). The row-allocation percentages and dimension
formulas below are a **design aid for inventing a schema**, never a worksheet for a schema that
already exists - computing them by hand for a DDL the engine has already planned is the single most
expensive mistake measured on this skill.

---

## Path B: no DDL

Write the DDL directly from the user's business description - that is far faster than writing a generator - confirm table and column names with the user, then follow path A.

Order: 1. find the main fact (the core business action of the industry: order / policy / loan / visit / work order) -> 2. split out details (is the action made of multiple lines?) -> 3. find the event stream (status transitions or high-frequency behaviour; this is where the row count lives) -> 4. add dimensions (one per entity the action touches) -> 5. pick metrics (the top 6 on this industry's dashboard) -> 6. pick signals (peak/trough seasons, which dimension breaks; see the industry event calendar in `references/pitfalls.md`).

**Put enum values in DDL inline comments** (`order_status VARCHAR, -- pending / paid / shipped`); the engine extracts them as the value domain, so the profile need not repeat them.

---

## Path C: hand-written generator (fallback)

Only when the engine genuinely cannot express the structure - for example multi-period state-migration snapshots (credit M1->M2->M3 delinquency) or deeply nested parent-child facts. Use the primitives in `scripts/genlib.py` (weighted calendar, Zipf, event chains, CSV->DuckDB) and write your own loop; see "hand-written generator reference" at the end of `references/pitfalls.md`.

**Phases 0, 1, 2 and 4 are below; 3, 5 and 6 stayed in SKILL.md** because they apply to every path - the invariants, the quality check and question validation. On path A the modelling here is given by the DDL and the build is done by the engine, so none of this file is needed.

---

## Phase 0: parse the request (do not interrogate the user)

Extract the following from one sentence. **Anything missing takes its default and you start work**; only ask back when the industry is genuinely undeterminable.

| Element | How to parse | Default |
|---|---|---|
| Total rows | `50k`/`100k`/`1M` -> number | **80,000** (midpoint of 50k-100k; generates in ~2s, fastest to iterate, still enough for dashboards and Q&A) |
| Industry | From the business description and DDL table/column names | Model it from the description |
| Table/column names | If given, **follow exactly**; add missing dimensions/summaries around them | Derive from the industry |
| Time span | Use if specified | **17 months** (must be >= 13 so YoY is computable) |
| Cut-off date | Use if specified | **Yesterday**; for a complete final month use the last day of last month. Never use the end of the current month - that creates future-dated rows and a dashboard showing GMV that has not happened yet |
| Language | The user's | Column names English snake_case; names and enum labels in the user's language |
| Base currency | Required whenever multiple regions are involved | USD |

**Rows to scale** (total includes dim + ods + dws + ads). Look it up, or call `genlib.plan_scale(total_rows)`:

This is what `plan_scale` returns, not a target to aim at by hand - the fact layer takes 70% of the
total (main 18.2%, detail 27.3%, event stream 24.5%), the summary layer 24% and the dimensions 6%.
The figures below are its output for `dims={"seller": "org", "sku": "item", "buyer": "customer"}`:

| Total | Main fact | Detail | Event stream | Summary | Buyers/users | Sellers/stores | SKUs |
|---|---|---|---|---|---|---|---|
| 10k | 1.8k | 2.7k | 2.5k | 2.4k | 455 | 3 | 30 |
| **50k** | 9.1k | 13.7k | 12.2k | 12k | 2.3k | 15 | 152 |
| **80k (default)** | 14.6k | 21.8k | 19.6k | 19.2k | 3.6k | 24 | 243 |
| **100k** | 18.2k | 27.3k | 24.5k | 24k | 4.5k | 30 | 303 |
| 500k | 91k | 136.5k | 122.5k | 120k | 22.8k | 152 | 1.5k |
| 1M | 182k | 273k | 245k | 240k | 45.5k | 303 | 3k |
| 5M | 910k | 1365k | 1225k | 1200k | 227.5k | 1.5k | 15.2k |

Dimension counts follow the kinds you pass, so they move with `dims=`; the fact and summary shares
do not. **Call it rather than reproducing it** - the one call also validates the hard constraints:

```python
from genlib import plan_scale
plan = plan_scale(100_000, dims={"seller": "org", "sku": "item",
                                 "buyer": "customer", "courier": "enum"})
# -> fact_main 18,200 / fact_detail 27,300 / event_stream 24,500 / summary_layer 24,000
#    dims {'date': 516, 'seller': 30, 'sku': 303, 'buyer': 4550, 'courier': 12}
assert not plan["warnings"]        # fix the dimension kinds before forcing it through
```

---

## Phase 1: modelling

### 1.1 Four layers (table counts scale with the row budget; merge dws/ads on small sets)

```
dim_*   dimensions: 3-7   entity master data + dim_date (mandatory)
ods_*   detail:     4-8   1 main fact + detail children + 1 high-frequency event stream
dws_*   summary:    3-5   pre-aggregated on the main analysis grain (day/month x core dimension)
ads_*   application:1-2   cross-domain daily report, one row per day, preferred entry for Q&A
```

### 1.2 Row allocation

> **On Path A you do not compute this.** The engine allocates every table from `rows=` on its
> own - `_plan_rows` runs during inference, before you see anything - and
> `plan_datasource(ddl=...)` prints the result per table without a generator existing yet. **Do not do this arithmetic by hand.** A measured production run tried to
> divide an 80,000-row budget across five tables using the percentages below, reached 51,000,
> could not reconcile the gap, opened `ddl_engine.py` to find the allocator - and spent the
> remaining 30 turns in the source without generating a single row. The numbers below are for
> *designing* a schema on Path B, and for *judging* the allocation `report()` prints. They are
> not a worksheet.
>
> To change the allocation, do not reverse-engineer it: pin the table with
> `profile["table_rows"]` / `profile["dim_rows"]`, or correct a dimension's kind with
> `profile["dim_kinds"]`. Both override the engine outright. `references/profile-spec.md` §1.1
> states the allocation rule the engine actually applies.

**Core principle: the bulk of the data is the fact tables; a dimension table is only a list of entities.** Dimension row counts are decided by how many entities the business actually has and **do not scale with the total** - putting 25,000 users in a 100,000-row database leaves no room for facts.

These are the shares `plan_scale` applies, so the table above and this one are the same contract:

| Layer | Share of total | Note |
|---|---|---|
| All dimensions | **6%** | No single dimension above 8% of total; cardinality derived per 1.3, not guessed |
| Main fact | 18.2% | Orders / work orders / policies / subscriptions / visits |
| Fact detail | 27.3% | 1.4-2.2 rows per main-fact row |
| Event stream | 24.5% | Tracking / events / status transitions / instalments |
| Summary layer | 24% | Produced by SQL; row count = number of aggregation cells, driven by dimension cardinality |

> The fact layer (main + detail + events) totals **70%**. If dimensions crowd it out, cut the user dimension first (it runs away most easily), then SKUs.

### 1.3 Dimension cardinality: derive it from the business (**the easiest thing to get wrong**)

The number of entities per dimension follows from how many fact rows one entity should produce. Look up the density, then `cardinality = main_fact_rows / density`:

| Dimension kind | Fact rows per entity | Typical cardinality in a 100k DB | Note |
|---|---|---|---|
| `dim_date` | = number of days | 516 (fixed) | Does not scale; a high share in a small DB is normal |
| Enum-like (carrier/channel/payment/plan/product line) | 1000+ | **4-12** | The business only has a few; never scale these |
| Organisational (seller/store/warehouse/line/department/team) | 200-3000 | **30-60** | Too many and the head cells go sparse; every dashboard drill-down shows 1 |
| Staff (driver/doctor/agent/rep) | 100-1000 | **20-120** | |
| Item (SKU/course/item/procedure) | 20-200 | **200-400** | Driven by sell-through; long-tail SKUs may have zero sales |
| Customer (buyer/patient/student/account) | **3-15** | **3k-4k** | This number *is* the repurchase rate; B2C 3-6, B2B 8-15 |

**Hard constraints**: one dimension <= 8% of total; all dimensions together <= 15% (excluding `dim_date`). The customer dimension blows the budget most often - in a 100k database a `dim_user` above 8,000 rows means fewer than 2 orders per customer, which no business would show.

Then validate the density of the **main analysis grain** (usually `entity x day` or `entity x site x day`) is >= 20:

```
cardinality = main_fact_rows / (days x secondary_cardinality x target_density)

e.g. 250,000 orders / (516 days x 6 sites x 20 orders) = 4   <- only 4 sellers? not realistic
```

When the two disagree (the business wants 40 sellers, the density formula wants 4), **the business wins** and the density problem is solved in this order - never by distorting the distribution:

1. Reduce the number of dimensions in the grain (seller x day, not seller x site x day)
2. Concentrate the secondary dimension (78% of a seller's orders land on their main site, which lowers effective cardinality naturally)
3. Add a coarser summary table (`dws_xxx_month`)
4. Raise the fact row count
5. If it is still sparse, **accept it honestly** - real warehouses are sparse at the finest grain. Record per-grain density in the "Definitions and trade-offs" section of `README.md`

### 1.4 Cross-domain joins (mandatory for multi-domain sets)

A downstream entity's foreign key may **only be sampled from the already-generated upstream set**; never generate the ID independently.

```
dim_user ──┬─→ ods_order.buyer_id
           └─→ ods_interaction.user_id      one user master, supports "content drives GMV"
ods_order ──→ ods_shipment.order_id          only paid orders ship
ods_content ─→ ods_order.source_content_id   content attribution of GMV
```

---

## Phase 2: signal design (the soul of demo data; do not skip)

Data without signal is a noisy flat line and every dashboard and question built on it is useless. All four kinds are required.

### 2.1 Time intensity model

Every fact row's date must be sampled from a **weighted calendar**, never `rng.choice(DAYS)`:

```python
intensity(d) = trend(+2-4% MoM) x season(sine +-10%) x weekday(weekend 1.3x) x event multiplier
```

### 2.2 Event calendar (promotions / shutdowns)

Pick 5-8 for the industry and write them into `dim_date.event_name` so anomalies are **explainable**:

| Type | Multiplier | Example |
|---|---|---|
| Top-tier promotion | 4-5x | Black Friday / Cyber Monday, Singles' Day |
| Mid-tier promotion | 2-3x | 618, Christmas season, Mother's Day |
| Shutdown / trough | 0.5-0.6x | Spring Festival, long holidays |

### 2.3 Attributable anomalies (the source of the agent's exam questions)

Inject 3-5 anomaly windows, each **scoped to one attribution dimension**, so "why did it drop" has exactly one answer:

| Attribution | Example | Degradation |
|---|---|---|
| Global | Holiday capacity backlog -> network-wide delays | 1.9-2.2x |
| Single region | Typhoon over South China | 1.8-2.0x |
| Single supplier | A carrier relocated its sorting hub | 1.7-1.9x |

**The multiplier must leave the metric inside a believable range**: on-time rate 85% -> 28% is believable, -> 8% is a cliff that reads as fake at a glance.

### 2.4 Long tail

Entity popularity follows Zipf; **validate the resulting share**, do not just set alpha:

| Object | Top 10% should hold | alpha starting point |
|---|---|---|
| Sellers / stores / customers | 70-85% | 1.35 |
| SKUs | 60-75% | 1.05 |
| Content / creators | 55-70% | 1.15 (square-root the sales weight to smooth) |

## Phase 4: build (nothing to do on path A)

`eng.generate()` does all of the following in one call. **You write no DDL, no INSERT and no COMMENT SQL.**

| What the engine does | Implementation |
|---|---|
| Creates tables with the declared types (a BIGINT key stays BIGINT; DECIMAL(18,2) does not become DOUBLE) | `build_db(types=...)` |
| Generates and bulk-loads (via CSV, two orders of magnitude faster than row-wise INSERT) | `read_csv_auto` |
| Two-pass row calibration (within 6% of target) | inside `generate()` |
| Runs `pre_sql` -> auto summary layer -> `extra_sql`, in that order | |
| Writes `COMMENT` on every table and key column | `_comments()`, overridable in the profile |
| Cleans up intermediate CSVs, leaving only the database | |

**Your two jobs**: state the business rules in the profile (calendar, enums, differentiation dictionaries, anomalies), and check `report()`'s inference, overriding in the profile where it is wrong.

Comments are the core difference between a demo dataset and raw data - the agent reads them to understand semantics. The engine writes them per role; add the non-obvious definitions:

```python
"column_comments": {
    "ods_order.gross_amt_usd": "Order goods total (USD, converted via fx_rate_to_usd)",
    "ods_shipment.is_on_time": "Delivered on time: deliver_dt<=promise_dt; NULL when undelivered",
}
```

---
