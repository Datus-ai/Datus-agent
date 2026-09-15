# The 17 invariants in detail, and a distortion root-cause table

Every invariant here came out of a real rework. The left column is **what gives the data away**; the
right column is **what to do instead**.

---

## 1. Fact dates must be sampled from a weighted calendar

Uniformly random dates produce a completely flat monthly curve (measured: monthly GMV
12.4/10.9/12.1/12.0/12.0M, with February low only because it is shorter). The line chart is noise,
and YoY/MoM comparison and anomaly attribution - the core of data Q&A - all stop working.

```python
# WRONG - flat curve, the demo is dead on arrival
d = START + timedelta(days=rng.randint(0, span))

# RIGHT - trend + season + weekday + events
weights = day_weights(DAYS, calendar=cal, trend_mom=0.031, weekend_lift=1.33)
order_days = rng.choices(DAYS, weights, k=N_ORDERS)   # batch sampling, two orders of magnitude faster
```

**Acceptance**: monthly max/min >= 1.8x; weekend/weekday >= 1.15x; promotion day vs normal day >= 2x.

---

## 2. Derived quantities come from base quantities, never from independent randomness

Randomising impressions and clicks independently gave average CTR 28.5%, max **363.8**, and 367 rows
with more clicks than impressions. Any conversion dashboard exposes this the moment it opens.

```python
# WRONG
impression = rng.randint(10, 100000); click = rng.randint(1, 5000)

# RIGHT - fix the base quantity and a plausible ratio, then derive
impression = max(120, int(heat * SCALE * rng.uniform(0.4, 2.2)))
ctr = bounded_gauss(rng, 0.041, 0.013, lo=0.008, hi=0.092)
click = max(1, int(impression * ctr))          # max(1, ...) keeps int() from truncating small values to 0
```

Same family: paid amount <= amount due, refund <= amount paid, used credit <= credit limit, courses
completed <= courses enrolled.

---

## 3. Child event time = parent time + a non-negative offset

Sampling child dates independently put **50% of interactions before the content was published**
(measured 25946/52000). Causality is broken and every retention/decay analysis is wrong.

```python
# WRONG
action_dt = rng.choice(DAYS)

# RIGHT - anchor to the parent, power-law decay offset, clamp to the cut-off
decay = [1.0 / ((k + 1) ** 1.45) for k in range(31)]
offset = rng.choices(range(31), decay)[0]
action_dt = min(content["publish_dt"] + timedelta(days=offset), END_DT)
```

---

## 4. Event sequences are forced monotonic

Even when every step is a positive offset, one clamp (to the cut-off or the delivery date) can make
the next row earlier than the previous one - measured 184 rows going backwards in time.

```python
# RIGHT - always backstop after any clamp
if j > 0 and t <= prev_ts:
    t = prev_ts + timedelta(hours=1)
prev_ts = t
```

**Acceptance**: compare with `lag(ts) OVER (PARTITION BY entity ORDER BY seq)`; backwards steps must
be zero.

---

## 5. Status labels are derived from facts, never pre-set

Labelling an entity `ON_TIME` and then randomising dates produced **933 shipments marked on time but
delivered after the promised date**. The agent gets two different answers depending on whether it
counts labels or dates.

```python
# WRONG
scenario = "ON_TIME"; deliver_dt = ship_dt + timedelta(days=rng.randint(1, 3))

# RIGHT - compute the fact, then label it
transit = max(1, round(sla * rng.lognormvariate(-0.24, 0.34) * (disrupt ** 0.72)))
deliver_dt = ship_dt + timedelta(days=transit)
status = "DELIVERED" if deliver_dt <= END_DT else "IN_TRANSIT"
is_on_time = 1 if status == "DELIVERED" and deliver_dt <= promise_dt else 0
```

The same goes for tier labels (S/A/B/C), risk grades and membership levels: backfill them **from
actual contribution after generation**, or the agent immediately spots contradictions like "a C-tier
customer ranks 5th by GMV".

---

## 6. Cross-table foreign keys are sampled from the upstream set only

Randomly composed IDs left **59** of 28,000 shipments joinable to an order (pure collision), and every
cross-domain analysis broke.

```python
# WRONG
order_id = f"ORD{rng.randrange(10**7):010d}"

# RIGHT - take from the upstream result set and inherit its business attributes
for (oid, seller, odt, site, weight, amt) in paid_orders:   # only paid orders ship
    ship_dt = odt + timedelta(days=rng.choices([0,1,2,3], [.42,.34,.18,.06])[0])
```

**Acceptance**: 100% hit rate on every foreign-key path.

---

## 7. Multi-currency always stores three columns

Summing local amounts across sites gave every currency a mean of 2400 - JPY and USD treated as the
same unit, and the site ranking was meaningless.

```python
# RIGHT - local amount + rate + base amount; summaries use the base only
row = [..., currency_cd, fx_rate_to_usd, gross_amt_local, gross_amt_usd, ...]
```

**Acceptance**: `avg(local)/avg(base)` should be ~150 for JPY and exactly 1 for USD.

---

## 8. Popularity is Zipf-weighted exactly once

Product weight = `product Zipf x seller Zipf`, while seller sampling also used the seller Zipf, put
**99.8% of GMV in the top 10% of sellers**. The long tail became a dictatorship and 90% of the seller
dimension had no rows at all.

```python
# WRONG - double weighting
product["w"] = p_zipf[i] * seller["w"]
picks = rng.choices(products, [p["w"] for p in products], k=n)

# RIGHT - separate concerns: pick the seller by seller weight, then a SKU within that seller
product["w"] = p_zipf[i]
seller = rng.choices(sellers, [s["w"] for s in sellers])[0]
picks = rng.choices(sku_by_seller[seller["id"]], [p["w"] for p in pool], k=n)
```

A side effect of the wrong version: an order spanning several sellers has to be filtered out
afterwards, cutting detail rows ~30% below plan. **Picking the seller first** fixes that too.

**Acceptance**: the top-10% share lands in the target band from Phase 2.4; above 90% means weights
were stacked.

**Small cardinalities need a flatter target.** A `head_share` of 0.74 is meaningful for hundreds of
entities; with 21 products the top 10% is 2 rows, and forcing those 2 to hold 74% produces an
absurdly dominant head. The engine flattens the target automatically below 200 distinct values
(n=21 -> 0.35, n=103 -> 0.53, n>=200 -> 0.74), and the quality check relaxes its long-tail floor to
match (30% / 40% / 50%). Keep the two in step - if you raise one, raise the other.

---

## 9. Terminal states must appear

Truncating the event sequence at a fixed length (`EVENT_SEQ[:n]`) means a delivered shipment never
reaches `DELIVERED`: `final_flag` is all zeros, the funnel chart breaks, and downstream metrics are
permanently 0.

```python
# WRONG
for k in range(min(per_ship_ev, len(EVENT_SEQ))): ...

# RIGHT - the terminal state follows the entity's status
seq = FULL_SEQ if status == "DELIVERED" else FULL_SEQ[:rng.randint(2, 5)]
```

**Acceptance**: zero completed entities missing their terminal event.

---

## 10. Dimension cardinality comes from the formula

1,000 sellers x 151 days x 6 sites = 900,000 cells holding 25,000 orders - an average of **1.01
orders per cell**. The summary table is as wide as the detail table and every drill-down is a list of
ones.

Derive it with the Phase 1.3 formulas in SKILL.md, and record the measured density per grain in the
data-dictionary section of `README.md`.

---

## 11. Business differences must be modelled explicitly

One global random rate gave all six categories a refund rate of 6.0-6.8%. "Which category has the
highest refund rate" - a very common question - has no signal and the answer carries no insight.

```python
# RIGHT - explicit differences that match business intuition
REFUND_RATE = {"APPAREL": 0.168, "BEAUTY": 0.112, "HOME": 0.081,
               "PET": 0.068, "OUTDOOR": 0.059, "3C": 0.047}
rr = REFUND_RATE[main_category]
```

Same family: delivery time by region, average order value by channel, SLA by service tier, activity
by age band. **Measured effect**: the refund-rate gradient went from "all 6%" to 11.24% -> 3.67%, and
the ranking matches business intuition exactly.

### Gradient baselines (use these when writing a profile instead of guessing)

| Case | Direction and magnitude |
|---|---|
| E-commerce refund rate | Apparel 15-18% > beauty 10-12% > home 7-9% > pet/outdoor 5-7% > 3C 4-5% (max/min 3-4x) |
| E-commerce order value | 3C > home > outdoor > beauty > apparel > FMCG (max/min 5-10x) |
| Channel conversion | Direct 3-4% > organic 2.5-3% > paid search 2-2.5% > affiliate 1.8-2% > social 1-1.5% (~3x) |
| Channel ROAS | Affiliate 6-8 > paid search 5-7 > social 4-6; organic has no ad spend, so ROAS is NULL |
| Delivery time | Express 3 days < standard 6-7 < economy 10-11; cross-border adds 2-4 days of customs |
| SaaS churn | SMB 3-3.5%/month > mid-market 1.2-1.5% > enterprise 0.5-0.8% (~5x) |
| Credit default rate | Cash loans > instalments > mortgage-backed; approval rate correlates strongly with score bands |
| Insurance loss ratio | Auto 60-68% > health 50-58% > accident 30-38% |
| Course completion | 1-on-1 > small class > large class; refund rate higher on low-price funnel courses |
| Gaming spend | Whales at 0.5% of users drive 40%+ of revenue (Zipf alpha up to 1.8, more extreme than e-commerce) |
| Tiered ARPU | Top vs bottom tier differ 20-60x (more when non-purchasers are included) |

### Industry event calendars (reference when writing `calendar`)

| Industry | Peak / promotions | Trough / shutdown |
|---|---|---|
| E-commerce retail | 618 2.5-3.5x, Singles' Day 4-5x, Black Friday/Cyber Monday 4.5-5.5x, Christmas 2-2.5x | Spring Festival 0.5-0.6x, new-year lull 0.8x |
| Content community | Spring Festival session time 1.6x, school holidays 1.3x, trending events 2-3x | - |
| SaaS / B2B | Quarter-end push 1.8x (B2B specific), year-end renewal peak | Summer 0.8x |
| Insurance | Jan-Feb campaign 3x (insurance specific) | Spring Festival |
| Education | School holidays 2.5x, back-to-school 1.8x | Post-exam |
| Logistics | 1.6x backlog for 3 days after a promotion | Spring Festival capacity 2.1x (delivery time worsens; volume does not rise) |
| Mobility | Morning/evening peaks (**hour-level signal is mandatory**), holidays 2x, rain/snow 1.5x | - |
| Offline retail | Weekends 1.5x (stronger than online), holidays 2x | Severe weather 0.7x |

---

## 12. Anomaly multipliers stay in a believable range

A degradation multiplier of 3.1x dropped the on-time rate from 85% straight to **8%** - a cliff to
zero that reads as synthetic at a glance. Dialled back to 2.1x it lands at 28%: visible and credible.

| Metric | Believable anomaly range | Cliff (not credible) |
|---|---|---|
| On-time delivery | 85% -> 25-50% | -> < 15% |
| Conversion rate | 3% -> 1.2-2% | -> 0.1% |
| Refund rate | 6% -> 12-20% | -> 60% |
| DAU | 100% -> 60-80% | -> 10% |

After tuning, plot the curve and check: the anomaly window should be a **step**, not a drop to zero.

---

## 13. Names must be semantic

`seller_0` / `SKU000003847` / `wh_3` makes the agent answer "the best seller is SKU000003847" and the
demo's credibility collapses.

```python
BRANDS = ["Lumora", "Nordvik", "Kaizen Field", "Volta Ridge", "Marisol", ...]  # fictional brands
name = f"{brand} {rng.choice(['Flagship Store', 'Overseas Store', 'Select Store', 'Direct Store'])}"
sku_name = f"{brand} {rng.choice(CAT_ITEMS[cat])}{rng.choice(['', ' Pro', ' Gen 2', ' Flagship'])}"
wh_name  = "Shenzhen Bonded Warehouse" / "Ningbo Forward Warehouse" / "Los Angeles Overseas Warehouse"
```

**Use fictional brand names.** Do not attach a real company name to a negative metric such as "the
carrier with the worst delivery time".

---

## 14. Fix sparsity by adding a grain, not by distorting the distribution

When the finest grain misses its density target, do **not** go back and make the long tail a
dictatorship to hit the number - that is exactly the trap in invariant 8. The right order is: reduce
the number of dimensions in the grain -> concentrate the secondary dimension -> add `dws_xxx_month`
-> raise the row count -> accept it honestly and write it into the data dictionary.

Real warehouses are sparse at the finest grain. Measured final state: all-domain x day 485 orders,
category x day 20.4, seller x month 33.1, seller x site x day 4.6. The last one misses, every main
analysis grain passes - that is the correct trade-off.

---

## 15. Audit summary-layer definitions column by column

To squeeze a SKU count into a `GROUP BY ALL`, someone wrote `count(DISTINCT min(sku_id))` - valid
syntax, always equal to 1, and **the quality check cannot catch it** (non-zero, non-null, correct
type).

After writing each dws/ads table, go column by column asking "what is the numerator and denominator
of this definition". Watch especially for:

- `count(DISTINCT ...)` inflated by a JOIN
- `sum()` double-counting after a multi-table JOIN (pre-aggregate to the grain, then JOIN)
- A `COALESCE`d key after a `FULL JOIN` silently dropping one side
- A ratio with a zero denominator (always `nullif(x, 0)`)

---

## 16. There must be a stock baseline; entities cannot all start inside the window

Starting every subscription/customer/asset from zero inside the data range makes the first month
near-zero, monthly variation **104x**, first-to-last month **+7820%** - YoY comes out astronomical and
the dashboard breaks on the first screen. A real business already has stock before the observation
window opens.

```python
# WRONG - everything is new
start = rng.choices(DAYS, DAY_W)[0]

# RIGHT - 40-50% of entities already exist before the range starts
if rng.random() < 0.45:
    start = START_DT - timedelta(days=rng.randint(30, 900))   # stock
else:
    start = rng.choices(DAYS, DAY_W)[0]                        # new inside the range
# Transactions generated outside the range (bills, instalments) are not stored;
# only their contribution to the stock is kept.
```

After the fix: monthly variation 2.7x, first-to-last month +161% - the curve went from "exponential
explosion" to "healthy growth".

**Applies to**: subscriptions, loans, policies, memberships, active headcount - anything with a
lifetime. Transactional facts (orders, visits) are unaffected.

**Acceptance**: first month >= 35% of the second month; monthly max/min <= 12x.

---

## 17. No O(n) lookups inside the generation loop

`DAYS.index(x)` inside a 55,000-iteration loop made 110,000 rows take **20.5 seconds**; a pre-built
dict brought it to **0.7 seconds**, 30x faster. For comparison, a correct implementation generates
3.42M rows in 19.8 seconds.

```python
# WRONG - list.index() / in list / a comprehension over all days, inside the loop
span = [d for d in DAYS if st <= d <= en]
d = rng.choices(span, [DAY_W[DAYS.index(x)] for x in span])[0]

# RIGHT - pre-built index, slice the weights
DAY_IDX = {d: i for i, d in enumerate(DAYS)}
lo = DAY_IDX.get(max(st, START_DT), 0)
hi = DAY_IDX.get(min(en, END_DT), len(DAYS) - 1)
d = rng.choices(DAYS[lo:hi + 1], DAY_W[lo:hi + 1])[0]
```

When generation is slow, check three things: `list.index()` in the loop, `x in list` in the loop, and
comprehensions over the full set.

---

# Distortion root-cause table

| Symptom | Root cause | Invariant |
|---|---|---|
| The line chart is flat, no trend | Uniformly random dates | 1 |
| Conversion rate > 100% / absurd ratios | Numerator and denominator randomised independently | 2 |
| Child event earlier than parent | Time not anchored to the parent | 3 |
| Event times go backwards | No monotonic backstop after a clamp | 4 |
| Label contradicts the fact | Labelled before generating | 5 |
| Cross-domain JOIN is nearly empty | Foreign keys randomised independently | 6 |
| Metrics across regions are incomparable | Multi-currency not converted | 7 |
| Top 10% share > 90% | Popularity weighted twice | 8 |
| The last funnel step is always 0 | Sequence truncated at a fixed length | 9 |
| One row per cell in the aggregate table | Dimension cardinality too high | 10 |
| Every dimension shows the same metric | No differentiation dictionary | 11 |
| The metric goes to zero during the anomaly | Degradation multiplier too large | 12 |
| Answers are full of IDs and codes | Names not semantic | 13 |
| Distribution made a dictatorship to hit density | Wrong fix applied | 14 |
| A column is always 1 / always the total | Summary definition is wrong | 15 |
| First month near zero, YoY up tens of times | No stock baseline; all entities new in-range | 16 |
| Generation below 50k rows/second | O(n) lookup inside the loop | 17 |

---

# Appendix: hand-written generator reference (path C only)

On path A (DDL-driven) the engine does all of this and none of this section is needed. Correct
examples for each invariant are above; this appendix adds the three things the invariants do not
cover: the overall skeleton, the summary-layer SQL, and bulk comments.

## Skeleton

```python
"""<industry> demo warehouse: <domain 1> + <domain 2>, single database output."""
import random, shutil, time
from datetime import date, timedelta
from pathlib import Path
from genlib import (Calendar, date_range, day_weights, zipf_weights, tune_alpha, head_share,
                    bounded_gauss, lognorm_between, EventChain, CsvOut, day_ts, build_db,
                    plan_scale)

ROOT, SEED = Path(__file__).parent, 42
CSV_DIR, DB = ROOT / "_csv", ROOT / "data" / "datasource.duckdb"
TOTAL_ROWS = 80_000                               # default when the user did not specify
END_DT   = date(2026, 5, 31)                      # cut-off: yesterday or last month's end, never the future
START_DT = date(2025, 1, 1)                       # >= 13 months, otherwise YoY is impossible
DAYS     = date_range(START_DT, END_DT)

PLAN = plan_scale(TOTAL_ROWS, dims={"seller": "org", "sku": "item", "buyer": "customer"})
assert not PLAN["warnings"], PLAN["warnings"]     # clear the hard constraints before writing code
N_ORDERS = PLAN["fact_main"]                      # facts are the bulk of the data
N_SELLERS = PLAN["dims"]["seller"]                # dimensions are entity lists; they do not scale with the total
N_SKUS, N_USERS = PLAN["dims"]["sku"], PLAN["dims"]["buyer"]

CAL = Calendar(promos=[...], slows=[...], disruptions=[...])
DAY_W = day_weights(DAYS, CAL, trend_mom=0.031, weekend_lift=1.33)

def generate():
    rng, out = random.Random(SEED), CsvOut(CSV_DIR)
    dims  = gen_dims(rng, out)                     # dimensions first
    facts = gen_facts(rng, out, dims)              # facts reference dimensions
    gen_downstream(rng, out, facts)                # downstream domains reference upstream facts (invariant 6)
    backfill_tiers(out, facts)                     # tier labels backfilled afterwards (invariant 5)
    return out

if __name__ == "__main__":
    t0 = time.perf_counter(); out = generate(); t1 = time.perf_counter()
    sizes = build_db(DB, CSV_DIR, list(out.counts), DWS_SQL, COMMENTS)
    shutil.rmtree(CSV_DIR)
    print(f"{len(sizes)} tables / {sum(sizes.values()):,} rows | generate {t1-t0:.1f}s "
          f"build {time.perf_counter()-t1:.1f}s | {DB.stat().st_size/1e6:.1f} MB")
```

## Summary-layer SQL

```sql
-- Pre-aggregate to the grain before joining, so sum() is not inflated
CREATE OR REPLACE TABLE dws_sales_seller_day AS
WITH oi AS (SELECT order_id, sum(item_cost_usd) cost_usd, sum(qty) qty
            FROM ods_order_item GROUP BY 1),
     sk AS (SELECT o.seller_id, o.order_dt, o.site_cd, count(DISTINCT i.sku_id) sku_cnt
            FROM ods_order_item i JOIN ods_order o USING (order_id) GROUP BY 1,2,3)
SELECT o.order_dt AS stat_dt, o.seller_id, s.seller_name, s.tier_cd, o.site_cd,
       d.day_type_cd, d.event_name, d.is_weekend,
       count(*) AS order_cnt, count(DISTINCT o.buyer_id) AS buyer_cnt,
       round(sum(o.gross_amt_usd), 2) AS gmv_usd,
       round(sum(o.paid_amt_usd) - sum(o.refund_amt_usd), 2) AS net_revenue_usd,
       round(sum(o.paid_amt_usd) - sum(o.refund_amt_usd) - sum(oi.cost_usd), 2) AS gross_profit_usd,
       any_value(sk.sku_cnt) AS sku_cnt              -- never a count(DISTINCT min(...)) hack
FROM ods_order o
JOIN dim_seller s USING (seller_id)
JOIN dim_date d ON d.date_key = o.order_dt
JOIN oi USING (order_id)
LEFT JOIN sk ON sk.seller_id=o.seller_id AND sk.order_dt=o.order_dt AND sk.site_cd=o.site_cd
GROUP BY ALL;

-- ads layer: one row per day, the preferred Q&A entry point. Ratios always guard against divide-by-zero
CREATE OR REPLACE TABLE ads_business_daily AS
SELECT d.date_key AS stat_dt, d.month_key, d.quarter_cd, d.is_weekend, d.day_type_cd, d.event_name,
       COALESCE(s.order_cnt, 0) AS order_cnt, COALESCE(s.gmv_usd, 0) AS gmv_usd,
       round(COALESCE(s.gmv_usd, 0) / nullif(s.order_cnt, 0), 2) AS avg_order_value_usd,
       round(100.0 * f.on_time_cnt / nullif(f.delivered_cnt, 0), 2) AS on_time_rate_pct
FROM dim_date d
LEFT JOIN (SELECT stat_dt, sum(order_cnt) order_cnt, sum(gmv_usd) gmv_usd
           FROM dws_sales_seller_day GROUP BY 1) s ON s.stat_dt = d.date_key
LEFT JOIN (...) f ON f.stat_dt = d.date_key;
```

## Bulk comments

```python
COMMENTS = [
    ("TABLE ads_business_daily", "Cross-domain daily report: sales + community + fulfilment in one table; preferred Q&A entry"),
    ("TABLE dim_date", "Date dimension with promotion/shutdown event flags, for YoY/MoM and anomaly attribution"),
    ("COLUMN ods_order.gross_amt_usd", "Order goods total (USD, converted via fx_rate_to_usd)"),
    ("COLUMN ods_shipment.is_on_time", "On time: deliver_dt<=promise_dt; NULL when undelivered"),
    ("COLUMN dim_date.event_name", "Name of the day's promotion or shutdown event, used to explain metric movements"),
]   # at least one per table; any column whose definition is not obvious must have one
```
