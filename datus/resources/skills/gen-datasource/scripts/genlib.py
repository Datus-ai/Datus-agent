"""
Shared primitives for generating a synthetic demo warehouse.

Usage: put this file next to the generation script, then
    from genlib import (Calendar, day_weights, zipf_weights, head_share, pick,
                        bounded_gauss, lognorm_between, EventChain, CsvOut,
                        suggest_cardinality, HOUR_PROFILE, day_ts, build_db)

The design rules are the 17 invariants in ../references/pitfalls.md; these helpers exist so you
do not have to hand-write your way around them.
"""

from __future__ import annotations

import csv
import math
from datetime import date, datetime, timedelta
from pathlib import Path

__all__ = [
    "Calendar",
    "day_weights",
    "date_range",
    "zipf_weights",
    "head_share",
    "pick",
    "bounded_gauss",
    "lognorm_between",
    "EventChain",
    "CsvOut",
    "suggest_cardinality",
    "HOUR_PROFILE",
    "day_ts",
    "build_db",
    "split_rows",
    "plan_scale",
    "DIM_DENSITY",
    "tune_alpha",
]


# ---------------------------------------------------------------- calendar and time intensity
class Calendar:
    """Event calendar: promotion lifts, shutdown suppression, fulfilment anomaly windows.

    promos / slows: [(start, end, multiplier, name), ...]
    disruptions:    [(start, end, factor, scope, name), ...]
        scope looks like {} / {"region": "CN_SOUTH"} / {"courier": "CR03"}; empty means global
    """

    def __init__(self, promos=None, slows=None, disruptions=None):
        self.promos = list(promos or [])
        self.slows = list(slows or [])
        self.disruptions = list(disruptions or [])

    def volume_factor(self, d: date) -> float:
        f = 1.0
        for s, e, m, _ in self.promos:
            if s <= d <= e:
                f *= m
        for s, e, m, _ in self.slows:
            if s <= d <= e:
                f *= m
        return f

    def day_type(self, d: date):
        """Return (day_type_cd, event_name), written straight into dim_date."""
        for s, e, _, nm in self.promos:
            if s <= d <= e:
                return "PROMO", nm
        for s, e, _, nm in self.slows:
            if s <= d <= e:
                return "HOLIDAY_SLOW", nm
        return "NORMAL", ""

    def disrupt_factor(self, d: date, **attrs) -> float:
        """Degradation multiplier for an anomaly window. Pass the current entity dimensions in attrs,
        e.g. region='CN_SOUTH', courier='CR03'.

        Every key declared in scope must match attrs for the window to apply; an empty scope is global.
        """
        f = 1.0
        for s, e, m, scope, _ in self.disruptions:
            if not (s <= d <= e):
                continue
            if all(attrs.get(k) == v for k, v in scope.items()):
                f *= m
        return f

    def disrupt_name(self, d: date, **attrs):
        for s, e, _, scope, nm in self.disruptions:
            if s <= d <= e and all(attrs.get(k) == v for k, v in scope.items()):
                return nm
        return ""


def date_range(start: date, end: date):
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def day_weights(
    days,
    calendar: Calendar = None,
    trend_mom=0.03,
    weekend_lift=1.33,
    seasonal_amp=0.10,
    seasonal_peak_yday=300,
    dow_overrides=None,
):
    """Invariant 1: sampling weights for fact dates.

    trend_mom      month-over-month growth (0.03 = +3% per month)
    weekend_lift   Saturday multiplier versus a weekday; Sunday takes 0.96 of it
    seasonal_amp   amplitude of the within-year sinusoidal season
    Returns a weight list the same length as days, ready for rng.choices(days, weights, k=N).
    """
    if not days:
        return []
    base = days[0]
    dow_mult = {5: weekend_lift, 6: weekend_lift * 0.96, 0: 0.94, 4: 1.08}
    if dow_overrides:
        dow_mult.update(dow_overrides)
    out = []
    for d in days:
        months = (d.year - base.year) * 12 + d.month - base.month
        w = (1.0 + trend_mom) ** months if trend_mom else 1.0
        if seasonal_amp:
            phase = (d.timetuple().tm_yday - seasonal_peak_yday) / 365.0 * 2 * math.pi
            w *= 1.0 + seasonal_amp * math.cos(phase)
        w *= dow_mult.get(d.weekday(), 1.0)
        if calendar:
            w *= calendar.volume_factor(d)
        out.append(w)
    return out


HOUR_PROFILE = [1, 0.6, 0.4, 0.3, 0.3, 0.5, 1, 2, 2.5, 2.2, 2, 2.2, 2.6, 2.2, 2, 2, 2.3, 3, 4, 5, 5.5, 4.5, 3, 2]


def day_ts(rng, d: date, profile=None) -> datetime:
    """Expand a date into a timestamp with a realistic hour distribution (low at night, midday and evening peaks)."""
    hours = list(range(24))
    h = rng.choices(hours, profile or HOUR_PROFILE)[0]
    return datetime.combine(d, datetime.min.time()) + timedelta(
        hours=h, minutes=rng.randint(0, 59), seconds=rng.randint(0, 59)
    )


# ---------------------------------------------------------------- long tail
def zipf_rank_shift(n):
    """Rank offset q for Zipf-Mandelbrot.

    Plain Zipf (1/i^alpha) only constrains the combined Top X% share; it says nothing about the
    number-one entity itself. With 11,000 customers and 30,000 orders at a Top-10% share of 58%,
    the top customer takes 1,100 orders - two orders a day for 17 months, which no business has.
    Adding a rank offset proportional to the cardinality leaves both the tail shape and the Top X%
    target intact while flattening the head substantially: measured, Top-1 fell from 26% to 20% at
    n=103 and from 7.8% to 4.6% at n=532. Small dimensions (a few dozen agents or stores) need the
    floor offset most, or the leader takes nearly half.
    """
    return max(4.0, n * 0.04)


def zipf_weights(n, alpha=1.3, shuffle_with=None, q=None):
    """Invariant 8: Zipf(-Mandelbrot) popularity weights. Pass an rng as shuffle_with to decorrelate rank from id order."""
    if q is None:
        q = zipf_rank_shift(n)
    w = [1.0 / ((i + 1 + q) ** alpha) for i in range(n)]
    if shuffle_with is not None:
        shuffle_with.shuffle(w)
    return w


def head_share(weights, top_pct=0.10):
    """Check what share of the total the Top X% of entities hold - always run this after setting alpha."""
    s = sorted(weights, reverse=True)
    k = max(1, int(len(s) * top_pct))
    return sum(s[:k]) / sum(s)


def tune_alpha(n, target_share=0.78, top_pct=0.10, lo=0.3, hi=6.0):
    """Binary-search the alpha that makes the Top X% hold exactly target_share."""
    for _ in range(40):
        mid = (lo + hi) / 2
        if head_share(zipf_weights(n, mid), top_pct) < target_share:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 3)


def pick(rng, population, weights=None, k=1):
    """Batch weighted sampling (performance-critical: draw k at once; never call this once per loop iteration)."""
    return rng.choices(population, weights, k=k)


# ---------------------------------------------------------------- numeric distributions
def bounded_gauss(rng, mu, sigma, lo, hi):
    """Invariant 2: bounded normal, for ratio fields such as CTR, discount rate and completion rate."""
    for _ in range(12):
        v = rng.gauss(mu, sigma)
        if lo <= v <= hi:
            return v
    return min(hi, max(lo, mu))


def lognorm_between(rng, lo, hi):
    """Log-uniform distribution for prices, weights and durations - far more realistic than uniform."""
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def split_rows(total_rows, weights: dict):
    """Split a total row count across tables by share; weights look like {'ods_order': 0.10, ...}."""
    s = sum(weights.values())
    return {k: max(1, int(total_rows * v / s)) for k, v in weights.items()}


def suggest_cardinality(fact_rows, days, secondary_card=1, target_density=20):
    """Invariant 10: derive dimension cardinality. Returns the suggested cardinality (at least 1)."""
    return max(1, int(fact_rows / max(1, days * secondary_card * target_density)))


# How many fact rows an entity of each dimension kind should carry on average (invariant 10 / skill Phase 1.3)
DIM_DENSITY = {
    "enum": 1500,  # carrier/channel/payment/plan/product line - the business has only a few; never scale them
    "org": 600,  # seller/store/warehouse/line/department/team
    "staff": 300,  # driver/doctor/support rep/agent/teacher
    "item": 60,  # SKU/course/item/procedure
    "customer": 4,  # B2C buyer/patient/student - this number is the repurchase rate
    "customer_b2b": 10,  # B2B account
}

# Split within the fact layer (share of the fact-layer total)
_FACT_SPLIT = {"fact_main": 0.26, "fact_detail": 0.39, "event_stream": 0.35}


def plan_scale(total_rows, days=516, dims=None, fact_share=0.70, dim_share=0.06):
    """Plan per-table sizes from the total row count and the business characteristics.

        plan = plan_scale(100_000, dims={"seller": "org", "sku": "item",
                                         "buyer": "customer", "courier": "enum"})
        plan["fact_main"]      -> main fact table rows
        plan["dims"]["seller"] -> number of sellers
        plan["warnings"]       -> hard-constraint violations (must be empty before writing any code)

    Values of dims are keys of DIM_DENSITY, or an integer for a fixed cardinality.
    """
    fact_total = int(total_rows * fact_share)
    out = {k: int(fact_total * v) for k, v in _FACT_SPLIT.items()}
    out["summary_layer"] = int(total_rows * (1 - fact_share - dim_share))
    out["fact_total"] = fact_total

    cards, warns = {"date": days}, []
    for name, kind in (dims or {}).items():
        if isinstance(kind, int):
            cards[name] = kind
            continue
        if kind not in DIM_DENSITY:
            warns.append(f"unknown dimension kind {kind} (choose from: {', '.join(DIM_DENSITY)})")
            continue
        cards[name] = max(3, round(out["fact_main"] / DIM_DENSITY[kind]))

    for name, n in cards.items():
        if name != "date" and n > total_rows * 0.08:
            warns.append(
                f"dim_{name} has {n:,} rows, over 8% of the total; shrink it per invariant 10 or use a coarser dimension"
            )
    dim_total = sum(n for k, n in cards.items() if k != "date")
    if dim_total > total_rows * 0.15:
        warns.append(f"dimensions total {dim_total:,} rows, over 15% of the total; the fact layer gets squeezed")
    if "customer" in str(dims or {}):
        for name, kind in (dims or {}).items():
            if kind == "customer" and cards.get(name, 0) > total_rows * 0.08:
                warns.append(f"dim_{name} is so large that repurchase falls below 2, which no business would show")

    out["dims"] = cards
    out["dim_total"] = dim_total
    out["warnings"] = warns
    return out


# ---------------------------------------------------------------- event chain
class EventChain:
    """Invariants 3/4/9: event sequences with strictly monotonic times and no missing terminal state.

    chain = EventChain(rng, start_ts, min_gap_hours=1)
    for name in seq:
        ts = chain.step(span_hours / len(seq), cap=deliver_ts, pin=deliver_ts if terminal else None)
    """

    def __init__(self, rng, start_ts: datetime, min_gap_hours=1):
        self.rng = rng
        self.ts = start_ts
        self.min_gap = timedelta(hours=min_gap_hours)
        self.first = True

    def step(self, avg_hours=6.0, cap: datetime = None, pin: datetime = None) -> datetime:
        if self.first:
            self.first = False
            if pin is not None:
                self.ts = max(pin, self.ts)
            return self.ts
        if pin is not None:
            t = pin
        else:
            t = self.ts + timedelta(hours=max(1.0, avg_hours * self.rng.uniform(0.5, 1.5)))
            if cap is not None and t > cap:
                t = cap
        if t <= self.ts:  # monotonic backstop after a clamp
            t = self.ts + self.min_gap
            if cap is not None and t > cap:  # the backstop must not breach the hard cap; fall back to minute-level gaps
                t = min(cap, self.ts + timedelta(minutes=1))
                if t <= self.ts:
                    t = self.ts + timedelta(seconds=1)
        self.ts = t
        return t


# ---------------------------------------------------------------- output
class CsvOut:
    """CSV writer: overwrites, idempotent, counts rows. An empty string means NULL (paired with nullstr='')."""

    def __init__(self, out_dir, verbose=True):
        self.dir = Path(out_dir)
        if self.dir.exists():
            for f in self.dir.glob("*.csv"):
                f.unlink()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.counts = {}
        self.verbose = verbose

    def write(self, table, header, rows):
        with open(self.dir / f"{table}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        self.counts[table] = len(rows)
        if self.verbose:
            print(f"  {table:<26}{len(rows):>10,}")
        return len(rows)

    @property
    def total(self):
        return sum(self.counts.values())


def build_db(db_path, csv_dir, tables, dws_sql="", comments=(), drop_existing=True, types=None, create_sql=None):
    """CSV -> create DuckDB tables -> run the summary-layer SQL -> write comments. Returns {table: rows}.

    With types={table: [(column, type), ...]} the tables are created with the declared DDL types
    instead of letting read_csv guess - otherwise a BIGINT key becomes VARCHAR and DECIMAL(18,2)
    becomes DOUBLE, contradicting the user DDL.

    With create_sql={table: "CREATE TABLE ..."} the table is created from the declared statement and
    the rows are INSERTed, so PRIMARY KEY / UNIQUE / FOREIGN KEY survive into the database - an agent
    reads relationships off those keys. DuckDB has no ALTER TABLE ADD CONSTRAINT, so the constraints
    have to exist at creation time. `tables` must be in dependency order (parents first), which is
    what the engine's topological order already gives. If a statement or an insert is rejected (a
    generated value violating UNIQUE, or a parent that had to fall back), that one table falls back
    to the constraint-free CREATE TABLE AS SELECT rather than failing the whole build.
    """
    import duckdb

    db_path, csv_dir = Path(db_path), Path(csv_dir)
    if drop_existing and db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    degraded = []
    for t in tables:
        spec = (types or {}).get(t)
        sel = ", ".join(f'TRY_CAST("{c}" AS {ty}) AS "{c}"' for c, ty in spec) if spec else "*"
        read = (
            f"read_csv_auto('{csv_dir}/{t}.csv', header=true, nullstr='', sample_size=-1, all_varchar=true)"
            if spec
            else f"read_csv_auto('{csv_dir}/{t}.csv', header=true, nullstr='', sample_size=-1)"
        )
        stmt = (create_sql or {}).get(t)
        if stmt:
            try:
                con.execute(f"DROP TABLE IF EXISTS {t}")
                con.execute(stmt)
                cols = ", ".join(f'"{c}"' for c, _ in spec) if spec else "*"
                con.execute(
                    f"INSERT INTO {t} ({cols}) SELECT {sel} FROM {read}"
                    if spec
                    else f"INSERT INTO {t} SELECT * FROM {read}"
                )
                continue
            except Exception as e:  # noqa: BLE001 - fall back, never fail the build
                degraded.append(f"{t}: {str(e).splitlines()[0][:110]}")
                con.execute(f"DROP TABLE IF EXISTS {t}")
        con.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT {sel} FROM {read}")
    if degraded:
        print(
            "  ! constraints dropped on "
            + str(len(degraded))
            + " table(s) because the generated data or a parent table did not satisfy them:"
        )
        for d in degraded:
            print("    - " + d)
    # Returned as well as printed: generate() surfaces it to the caller, who would otherwise only
    # learn about a silently constraint-free table by inspecting the database.
    build_db.last_degraded = list(degraded)
    if dws_sql:
        con.execute(dws_sql)
    for obj, txt in comments:
        con.execute(f"COMMENT ON {obj} IS '{txt.replace(chr(39), chr(39) * 2)}'")
    con.close()
    # Reopen read-only to count: this reflects the persisted state, unaffected by intermediate tables or estimates on the same connection
    con = duckdb.connect(str(db_path), read_only=True)
    names = [
        r[0]
        for r in con.execute("SELECT table_name FROM duckdb_tables() ORDER BY 1").fetchall()
        if not r[0].startswith("_")
    ]  # a leading underscore marks an intermediate table; not part of the deliverable
    sizes = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in names}
    con.close()
    return sizes
