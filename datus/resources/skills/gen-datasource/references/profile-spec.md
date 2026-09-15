# Profile reference

**Reading this file is enough to write a profile. You do not need the engine source.**

The engine (`scripts/ddl_engine.py`) owns *how* data is produced; the profile states *what the
business rules are*. Every field is optional - an empty `{}` still produces structurally correct
data. The profile is what makes it look real.

---

## 1. What the engine does on its own

| Capability | Notes |
|---|---|
| Table roles | date_dim / dim / fact / detail / downstream / event / metric_daily / snapshot, inferred structurally, never from name prefixes |
| Primary and foreign keys | **Declared PRIMARY KEY / FOREIGN KEY win**; inference only fills gaps. Renamed keys (`deal.buyer -> cust.cid`) still connect |
| Enum domains | **Extracted from DDL inline comments** (`order_status VARCHAR, -- pending / paid / shipped`). `report()` lists which columns were extracted and which end in `...` (incomplete) |
| Column semantics | id / date / ts / amount / count / ratio / enum / flag / name / seq / measure, from name + type |
| Row allocation | Fact layer 65-75%, dimensions derived from business density, two-pass total calibration (within 6%) |
| The 17 invariants | Weighted calendar sampling, derived-from-base quantities, child events anchored to parents, monotonic sequences, complete terminal states, FKs sampled from upstream only, layer backfill, stock baseline, zero header/detail amount drift |
| Type contract | Tables are created with the declared types (a BIGINT key stays BIGINT; DECIMAL(18,2) does not become DOUBLE) |
| Table/column comments | Written per role; overridable in the profile |
| Structural metadata | Written to `data/.datasource.meta.json`; the quality check reuses it instead of re-inferring |

**Run `eng.report()` first and override only what is wrong.**

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
| `naming` | Name templates | A specific naming style is needed |
| `head_share` | Top-10% concentration target | Default 0.74; change for more/less concentration |
| `tier_cols` / `tier_bands` | Tier column and bands | **Required when a dimension has more than one column containing tier/level** (see pitfalls) |
| `roles` | Force a table role | `report()` got it wrong and cannot self-correct |
| `joint` | Joint distributions (region x city) | Combinations must stay legal |
| `lifecycle` | Lifecycle timestamp chain | Multiple timestamps (created/paid/shipped/...) |
| `derive` | Derive a quantity from a base quantity | Funnel metric tables |
| `summary_dims` | Dimensions of the summary layer | `extra_tables="summary"` is on and the defaults do not fit |
| `pre_sql` / `extra_sql` | Business post-processing SQL | **Last resort**, when nothing above can express it |
| `trend_mom` / `weekend_lift` | Trend and weekend coefficients | Defaults 0.031 / 1.33; B2B needs different values |
| `table_comments` / `column_comments` | Comments | Recommended wherever a definition is not obvious |

---

## 3. Field details

### calendar - time signal (most important)

```python
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
  a per-category margin gradient is configured:

```python
"products.list_price": {"__by__": "category", "3C": [200, 900], "Beauty": [30, 200]},
"products.cost_price": {"__by__": "category", "3C": [.80, .88],   # cost ratio -> 12-20% margin
                        "Beauty": [.35, .45], "__default__": [.5, .65]},
```

- **Supported on dimension, fact and detail tables alike.**

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
    # Average gap in hours between adjacent timestamps, matching timestamp column order in the DDL
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
"naming": {"dim_seller": {"tpl": "{brand} {org_suffix}",
                          "vocab": {"brand": ["Lumora", "Nordvik"]}}},
"tier_cols": {"customers": "member_level"},            # see pitfall 1 below
"tier_bands": [(.05, "platinum"), (.2, "gold"), (.5, "silver"), (1.0, "normal")],
```

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

## 5. Five easy mistakes

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

**5. `pre_sql` is a last resort.** It runs before the automatic summary layer and can restate metrics
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
eng.generate(str(HERE / "datasource.duckdb"))
```
