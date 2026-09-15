# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Quality checks for a generated demo datasource (DuckDB).

Deterministic validation of a synthetic warehouse: is it layered, do foreign keys resolve,
does the time series carry trend/weekday/event signal, are derived ratios in range, is the
long tail believable, is the aggregation dense enough to drill into.

This is the engine-side counterpart of the ``gen-datasource`` skill. It lives here rather
than in the skill bundle so the checks version with the tool that runs them, and so a
private deployment picks them up by upgrading the package.

The checker works on an open DuckDB connection, so it can run against a live datasource
after the data has been imported - validating what will actually be queried rather than a
build artifact.
"""

from __future__ import annotations

import re
from datetime import date as _date
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

RATIO_HINTS = ("_rate", "_pct", "_ratio", "ctr", "cvr", "_share")
# FX rates and SLA targets are not bounded ratios despite the naming.
RATIO_EXCLUDE = ("fx_", "exchange", "_to_usd", "_to_cny", "sla_")
COUNT_HINTS = ("_cnt", "_qty", "_num", "_count")
# Metric preference: amounts beat counts. These are identifier/calendar columns and can never be metrics.
METRIC_PREFER = ("amt", "gmv", "revenue", "sales", "value", "profit", "premium", "balance")
AUDIT_COLS = ("created_at", "updated_at", "etl_", "_load_", "insert_", "modify_")
METRIC_EXCLUDE = (
    "_key",
    "_id",
    "year",
    "month",
    "week",
    "day_of",
    "quarter",
    "_seq",
    "flag",
    "is_",
    "_rate",
    "_pct",
    "_ratio",
    "sla",
    "_dt",
    "score",
    "level",
)
NUMERIC = ("BIGINT", "DOUBLE", "INTEGER", "HUGEINT", "SMALLINT", "FLOAT", "DECIMAL")
# Future dates that are legitimate business semantics rather than a defect.
FUTURE_OK_SUFFIX = ("promise_dt", "expire_dt", "due_dt", "end_dt", "renew_dt")

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"

# Only the datasource's own catalog is in scope. A connector may ATTACH others (an Iceberg REST
# catalog, an import staging database), and counting those tables would corrupt every ratio here.
_OWN_CATALOG = "database_name = current_database()"


def _q(name: str) -> str:
    """Quote an identifier. Table and column names come from a live user datasource, where a
    reserved word, mixed case or a hyphen is ordinary - unquoted they make the query fail, and a
    failed query silently degrades into a default value and a wrong verdict."""
    return '"' + str(name).replace('"', '""') + '"'


def _lit(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _show_ratio(ratio: float) -> str:
    """Print a monthly max/min ratio without rounding it across its own threshold.

    ``{:.1f}`` printed 1.79 as "1.8x" and then failed it against the 1.8 floor, so the report
    argued with itself. Near the boundary the extra digit is the difference between a verdict a
    reader can act on and one they have to distrust; away from it, it is noise.
    """
    return f"{ratio:.1f}" if ratio >= 2 or ratio < 1.5 else f"{ratio:.2f}"


class QualityChecker:
    """Run the demo-datasource checks against an open DuckDB connection.

    Args:
        con: An open DuckDB connection (or any object exposing ``execute(...).fetchall()``).
        config: Optional business assertions - ``fk``, ``assertions``, ``skip``, ``strict_ddl``.
        meta: Optional structural metadata written by the generator
            (``.<db stem>.meta.json``): table roles, declared keys, strict-DDL flag. Supplying
            it keeps the checker from inferring a second, conflicting view of the schema.
    """

    ROLE2KIND = {
        "date_dim": "date",
        "dim": "dim",
        "fact": "fact",
        "detail": "fact",
        "downstream": "fact",
        "metric_daily": "fact",
        "event": "event",
        "snapshot": "fact",
    }

    def __init__(self, con: Any, config: Optional[Dict[str, Any]] = None, meta: Optional[Dict[str, Any]] = None):
        self.con = con
        self.cfg = config or {}
        self.meta = meta
        self.results: List[Tuple[str, str, str]] = []
        self.skip = set(self.cfg.get("skip", []))
        self.kind: Dict[str, str] = {}
        self.date_table: Optional[str] = None
        self.query_errors: List[str] = []

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _num(v):
        """DECIMAL columns come back as Decimal and raise TypeError when mixed with float.

        This normalises the type only; it never changes a threshold.
        """
        return float(v) if isinstance(v, Decimal) else v

    def q(self, sql: str) -> List[tuple]:
        try:
            return [tuple(self._num(v) for v in r) for r in self.con.execute(sql).fetchall()]
        except Exception as e:  # noqa: BLE001 - probing queries legitimately fail on some shapes
            # Returning [] keeps a probe that does not apply from aborting the run, but every
            # caller then reads the failure as "no violations found". Recording it means a
            # degraded run is reported instead of silently passing - see check_query_health.
            logger.debug("quality check query failed: %s | %s", e, sql)
            self.query_errors.append(f"{str(e).splitlines()[0][:120]} | {sql[:120]}")
            return []

    def one(self, sql: str, default=None):
        r = self.q(sql)
        return r[0][0] if r and r[0] and r[0][0] is not None else default

    def add(self, name: str, ok: bool, detail: str, warn: bool = False) -> None:
        """With warn=True a failure is recorded as WARN rather than FAIL (acceptable but worth flagging)."""
        if name in self.skip:
            return
        self.results.append((name, PASS if ok else (WARN if warn else FAIL), detail))

    def tables(self) -> List[str]:
        return [r[0] for r in self.q(f"SELECT table_name FROM duckdb_tables() WHERE {_OWN_CATALOG} ORDER BY 1")]

    def cols(self, t: str) -> List[tuple]:
        return self.q(
            f"SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_catalog = current_database() AND table_name = {_lit(t)} "
            f"ORDER BY ordinal_position"
        )

    def numeric_cols(self, t: str) -> List[str]:
        return [c for c, d in self.cols(t) if any(d.upper().startswith(n) for n in NUMERIC)]

    def date_cols(self, t: str) -> List[str]:
        return [c for c, d in self.cols(t) if d.upper() in ("DATE", "TIMESTAMP")]

    # ------------------------------------------------------------------ checks

    def check_structure(self) -> List[str]:
        ts = self.tables()
        rows = self.one(f"SELECT sum(estimated_size) FROM duckdb_tables() WHERE {_OWN_CATALOG}", 0)
        # Layering is inferred structurally, never from a dim_/ods_ prefix - user DDL names vary wildly.
        if self.meta and self.meta.get("roles"):
            syn = set(self.meta.get("synthetic_tables", []))
            for t in ts:
                r = self.meta["roles"].get(t)
                self.kind[t] = (
                    "summary"
                    if ((t in syn and not r) or re.match(r"^(dws|ads|agg|summary)", t))
                    else self.ROLE2KIND.get(r, "fact")
                )
        for t in ts:
            if t in self.kind:
                continue
            n = self.one(f"SELECT count(*) FROM {_q(t)}", 0)
            cs = [c for c, _ in self.cols(t)]
            if (
                any(c in ("date_key", "stat_dt", "dt") for c in cs)
                and n <= 1200
                and not any(c.endswith("_id") for c in cs)
            ):
                self.kind[t] = "date"
            elif any(c.endswith("_seq") for c in cs):
                self.kind[t] = "event"
            elif n <= max(2000, rows * 0.08) and sum(1 for c in cs if c.endswith("_id")) <= 2:
                self.kind[t] = "dim"
            elif re.search(r"^(dws|ads|agg|summary)", t):
                self.kind[t] = "summary"
            else:
                self.kind[t] = "fact"
        for t in ts:
            if re.search(r"^(dws|ads|agg|summary)", t):
                self.kind[t] = "summary"

        cnt = {k: sum(1 for v in self.kind.values() if v == k) for k in ("date", "dim", "fact", "event", "summary")}
        strict = self.cfg.get("strict_ddl", bool(self.meta and self.meta.get("strict_ddl")))
        no_sum = not cnt["summary"]
        note = ""
        if no_sum:
            note = " (source tables only, no summary layer" + (
                "; strict_ddl declared)" if strict else "; set extra_tables='summary' if a dashboard rollup is needed)"
            )
        self.add(
            "layering",
            bool(cnt["dim"] and cnt["fact"] and (not no_sum or strict)),
            f"{len(ts)} tables / {rows:,.0f} rows - " + " ".join(f"{k}:{v}" for k, v in cnt.items() if v) + note,
            warn=no_sum and not strict,
        )

        date_t = [t for t, k in self.kind.items() if k == "date"]
        self.add(
            "date dimension present",
            bool(date_t),
            f"date dimension {date_t[0]}"
            if date_t
            else (
                "user DDL has no date dimension; event names are documented in the data dictionary (strict_ddl)"
                if strict
                else "no date dimension, so anomaly attribution has no anchor"
            ),
            warn=strict,
        )
        self.date_table = date_t[0] if date_t else None
        return ts

    def check_fk(self, ts: Sequence[str]) -> None:
        """Declared foreign keys are authoritative; name-based inference only fills the gaps."""
        pk_owner: Dict[str, str] = {}
        for t in ts:
            cs = [c for c, _ in self.cols(t)]
            if not cs:
                continue
            head = cs[0]
            if head.endswith("_id") and t.startswith(("dim_", "ods_")):
                n = self.one(f"SELECT count(*) FROM {_q(t)}", 0)
                d = self.one(f"SELECT count(DISTINCT {_q(head)}) FROM {_q(t)}", 0)
                if n and n == d:
                    pk_owner.setdefault(head, t)

        pairs = [
            tuple(p.split(".")) + tuple(q.split("."))
            for p, q in (x if isinstance(x, (list, tuple)) else (x["from"], x["to"]) for x in self.cfg.get("fk", []))
        ]
        declared = []
        for src, (rt, rc) in ((self.meta or {}).get("declared", {}).get("fk", {}) or {}).items():
            if "." in src:
                st, sc = src.split(".", 1)
                if st in ts and rt in ts:
                    declared.append((st, sc, rt, rc))
        pairs += declared

        auto = []
        for t in ts:
            if self.kind.get(t) in ("date", "dim"):
                continue
            for c, _ in self.cols(t):
                if c.endswith("_id") and c in pk_owner and pk_owner[c] != t:
                    auto.append((t, c, pk_owner[c], c))

        bad, detail = [], []
        for st, sc, tt, tc in pairs + auto:
            miss = self.one(
                f"SELECT count(*) FROM {_q(st)} a LEFT JOIN {_q(tt)} b ON a.{_q(sc)}=b.{_q(tc)} "
                f"WHERE a.{_q(sc)} IS NOT NULL AND b.{_q(tc)} IS NULL",
                0,
            )
            tot = self.one(f"SELECT count(*) FROM {_q(st)} WHERE {_q(sc)} IS NOT NULL", 0) or 1
            if miss:
                bad.append(f"{st}.{sc}->{tt} {miss:,} orphans ({100.0 * miss / tot:.1f}%)")
            else:
                detail.append(f"{st}.{sc}->{tt}")
        n_decl = len(declared)
        self.add(
            "foreign key integrity",
            bool(detail) and not bad,
            "; ".join(bad)
            if bad
            else (
                f"{len(detail)} path(s) resolve 100%" + (f" ({n_decl} from declared DDL)" if n_decl else "")
                if detail
                else "no foreign key relationship found - nothing was verified. Declare REFERENCES "
                "in the DDL, or pass the generator metadata so the declared keys are known"
            ),
            # WARN only for "there was nothing to check". Orphans are a FAIL even when they are
            # the ONLY finding - with every path broken, `detail` is empty and `warn=not detail`
            # alone downgraded a broken database to a warning that `ok` still passes.
            warn=not detail and not bad,
        )

        # A primary key that is entirely NULL while foreign keys report a 100% hit rate is the
        # most dangerous false positive this checker can produce, so it is checked explicitly.
        pk_bad = []
        pks = (self.meta or {}).get("pks", {})
        for t in ts:
            col = pks.get(t)
            if not col or col not in [c for c, _ in self.cols(t)]:
                continue
            n = self.one(f"SELECT count(*) FROM {_q(t)}", 0) or 0
            if not n:
                continue
            nul = self.one(f"SELECT count(*) FROM {_q(t)} WHERE {_q(col)} IS NULL", 0) or 0
            dup = n - (self.one(f"SELECT count(DISTINCT {_q(col)}) FROM {_q(t)}", 0) or 0)
            if nul:
                pk_bad.append(f"{t}.{col} has {nul:,}/{n:,} NULL keys")
            if dup:
                pk_bad.append(f"{t}.{col} has {dup:,} duplicate keys")
        self.add(
            "primary key non-null and unique",
            bool(pks) and not pk_bad,
            "; ".join(pk_bad)
            if pk_bad
            else (
                f"{len(pks)} table(s) have a non-null unique key"
                if pks
                else "no generator metadata, so no key was verified - see generator_meta in the result"
            ),
            warn=not pks,
        )

    # ------------------------------------------------------------------ time signal

    def _main_daily(self):
        """Pick day-grain tables plus a main amount/count column to observe the time signal on."""
        out = []
        # ROLE2KIND folds metric_daily into "fact" so the layering counts stay a closed set;
        # everything fact-shaped is therefore reachable through "fact" alone.
        cands = [t for t, k in self.kind.items() if k == "summary"] + sorted(
            [t for t, k in self.kind.items() if k == "fact"],
            key=lambda t: -(self.one(f"SELECT count(*) FROM {_q(t)}", 0) or 0),
        )
        for t in cands[:4]:
            dcs = []
            for c, d in self.cols(t):
                if d.upper() not in ("DATE", "TIMESTAMP") or any(a in c for a in AUDIT_COLS):
                    continue  # audit timestamps are not business dates
                expr = _q(c) if d.upper() == "DATE" else f"CAST({_q(c)} AS DATE)"
                if (self.one(f"SELECT count(DISTINCT {expr}) FROM {_q(t)}", 0) or 0) > 60:
                    dcs.append(expr)
            if not dcs:
                continue
            ncs = [
                c
                for c in self.numeric_cols(t)
                if not any(h in c for h in METRIC_EXCLUDE)
                and (self.one(f"SELECT sum({_q(c)}) FROM {_q(t)}", 0) or 0) > 0
            ]
            if not ncs:
                continue
            pref = [c for c in ncs if any(h in c for h in METRIC_PREFER)]
            pool = sorted(pref or ncs, key=lambda c: -(self.one(f"SELECT sum({_q(c)}) FROM {_q(t)}", 0) or 0))
            out.append((t, dcs[:2], pool[:4]))
        return out

    def _weekly_dev(self, t: str, dc: str, c: str) -> float:
        we = self.one(
            f"SELECT avg(v) FROM (SELECT {dc} d, sum({_q(c)}) v FROM {_q(t)} GROUP BY 1) WHERE dayofweek(d) IN (0,6)", 0
        )
        wd = self.one(
            f"SELECT avg(v) FROM (SELECT {dc} d, sum({_q(c)}) v FROM {_q(t)} GROUP BY 1) WHERE dayofweek(d) NOT IN (0,6)",
            0,
        )
        return (we / wd) if (we and wd) else 1.0

    def check_time(self) -> None:
        groups = self._main_daily()
        if not groups:
            self.add("time signal", False, "no table with a business date; trend/cycle cannot be assessed")
            return
        # Span, trend, stock baseline and event attribution describe the BUSINESS, so they observe
        # the headline series: ``_main_daily`` already ranks tables by row count and columns by
        # magnitude with preferred names first, so the head of the first group is it.
        #
        # They used to observe whichever (table, date, metric) had the strongest weekday deviation,
        # which is the right question for the weekday check alone and a lottery for the rest. A
        # production run measured the trend on a discount column, then on ad revenue, then on
        # discounts again - the winner moved whenever the data did - and got PASS, FAIL and PASS on
        # what was substantially the same database. It spent four rounds chasing the flip.
        headline = next(((t, dcs[0], mcs[0]) for t, dcs, mcs in groups if dcs and mcs), None)
        if not headline:
            self.add("time signal", False, "no observable numeric metric found")
            return

        t, dc, mc = headline
        span_rows = self.q(f"SELECT min({dc}), max({dc}), count(DISTINCT {dc}) FROM {_q(t)}")
        if not span_rows or span_rows[0][0] is None or span_rows[0][1] is None:
            self.add("time span", False, f"{t}: the date column holds no usable value")
            return
        span = span_rows[0]
        months = (span[1].year - span[0].year) * 12 + span[1].month - span[0].month + 1
        self.add(
            "time span",
            months >= 13,
            f"{span[0]} ~ {span[1]}, {months} months / {span[2]} days"
            + ("" if months >= 13 else " (< 13 months, year-over-year is impossible)"),
        )

        mm = self.q(
            f"SELECT date_trunc('month', {dc}) m, sum({_q(mc)}) v FROM {_q(t)} GROUP BY 1 HAVING sum({_q(mc)})>0 ORDER BY 1"
        )
        if len(mm) >= 3:
            vals = [r[1] for r in mm[1:-1]] or [r[1] for r in mm]  # drop partial first/last months
            ratio = max(vals) / min(vals) if min(vals) else 0
            first, last = vals[0], vals[-1]
            shown = _show_ratio(ratio)
            self.add(
                "time trend",
                1.8 <= ratio <= 12,
                f"observing {t}.{mc}: monthly {min(vals):,.0f} ~ {max(vals):,.0f} ({shown}x), "
                f"first to last month {100.0 * (last / first - 1):+.0f}%"
                + (
                    ""
                    if 1.8 <= ratio <= 12
                    else "  <- too volatile, usually a missing stock baseline (cold start)"
                    if ratio > 12
                    else "  <- flat: max/min across months must be >= 1.8x. Raise profile['trend_mom'] "
                    "or add promotion windows to profile['calendar']['promos']"
                ),
            )
            allv = [r[1] for r in mm]
            if len(allv) >= 4 and allv[1]:
                self.add(
                    "stock baseline",
                    allv[0] >= allv[1] * 0.35,
                    f"first month {allv[0]:,.0f} vs second {allv[1]:,.0f}"
                    + (
                        ""
                        if allv[0] >= allv[1] * 0.35
                        else "  <- first month too low: every entity is new in-range, distorting YoY (invariant 16)"
                    ),
                )

        # The weekday check is the one that legitimately hunts: a downstream fact (ship/settle date)
        # smooths the weekly shape away and a cumulative stock column never had one, so a flat
        # reading on the headline series says nothing until the alternatives have been tried.
        wt, wmc, r = t, mc, self._weekly_dev(t, dc, mc)
        if abs(r - 1.0) < 0.12:
            for ct, cdcs, cmcs in groups:
                for cdc in cdcs:
                    for cmc in cmcs:
                        cand = self._weekly_dev(ct, cdc, cmc)
                        if abs(cand - 1.0) > abs(r - 1.0):
                            wt, wmc, r = ct, cmc, cand
        if r:
            self.add(
                "weekday cycle",
                r >= 1.12 or r <= 0.88,
                f"observing {wt}.{wmc}: weekend/weekday = {r:.2f}x"
                + (
                    " (B2C shape: weekends busier)"
                    if r >= 1.12
                    else " (B2B shape: weekends quieter)"
                    if r <= 0.88
                    else "  <- no weekday signal, the curve is flat"
                ),
            )

        if self.date_table:
            dtbl = self.date_table
            ev = self.q(
                f"SELECT d.event_name, avg(x.v) FROM (SELECT {dc} d, sum({_q(mc)}) v "
                f"FROM {_q(t)} GROUP BY 1) x JOIN {_q(dtbl)} d ON d.date_key=x.d "
                f"WHERE d.day_type_cd='PROMO' AND d.event_name<>'' GROUP BY 1 ORDER BY 2 DESC LIMIT 3"
            )
            base = self.one(
                f"SELECT avg(x.v) FROM (SELECT {dc} d, sum({_q(mc)}) v FROM {_q(t)} GROUP BY 1) x "
                f"JOIN {_q(dtbl)} d ON d.date_key=x.d WHERE d.day_type_cd='NORMAL'",
                0,
            )
            if ev and base:
                self.add(
                    "events are explainable",
                    ev[0][1] / base >= 2.0,
                    "; ".join(f"{n} {v / base:.1f}x" for n, v in ev) + f" (normal-day baseline {base:,.0f})",
                )

    def check_future(self, ts: Sequence[str]) -> None:
        today = _date.today()
        bad = []
        for t in ts:
            for c in self.date_cols(t):
                if c.endswith(FUTURE_OK_SUFFIX):
                    continue  # a promise/expiry/renewal date in the future is business semantics
                n = self.one(f"SELECT count(*) FROM {_q(t)} WHERE {_q(c)} > DATE '{today}'", 0)
                if n:
                    mx = self.one(f"SELECT max({_q(c)}) FROM {_q(t)}")
                    bad.append(f"{t}.{c} has {n:,} rows after today (max {mx})")
        self.add("no future-dated rows", not bad, "; ".join(bad) if bad else f"every fact date is on or before {today}")

    def check_derived(self, ts: Sequence[str]) -> None:
        bad = []
        for t in ts:
            for c in self.numeric_cols(t):
                if any(h in c for h in RATIO_HINTS) and not any(x in c for x in RATIO_EXCLUDE):
                    mn = self.one(f"SELECT min({_q(c)}) FROM {_q(t)}")
                    mx = self.one(f"SELECT max({_q(c)}) FROM {_q(t)}")
                    if mn is None:
                        continue
                    hi = 100.0 if ("_pct" in c or "_rate_pct" in c) else 1.0
                    # Margin/growth/YoY style ratios are legitimately negative; only cap the top.
                    signed = any(
                        k in c for k in ("margin", "profit", "growth", "yoy", "mom", "change", "diff", "delta")
                    )
                    if (mn < 0 and not signed) or mx > hi * 1.0001:
                        bad.append(
                            f"{t}.{c} in [{mn:.4g},{mx:.4g}] out of range (max {hi:g}"
                            + ("" if signed else ", min 0")
                            + ")"
                        )
                if any(h in c for h in COUNT_HINTS):
                    neg = self.one(f"SELECT count(*) FROM {_q(t)} WHERE {_q(c)} < 0", 0)
                    if neg:
                        bad.append(f"{t}.{c} has {neg:,} negative values")
        self.add(
            "derived quantities sane", not bad, "; ".join(bad) if bad else "ratio columns in range, no negative counts"
        )

    def check_dead_cols(self, ts: Sequence[str]) -> None:
        zero, const = [], []
        for t in ts:
            n = self.one(f"SELECT count(*) FROM {_q(t)}", 0)
            if not n:
                continue
            for c in self.numeric_cols(t):
                s = self.one(f"SELECT sum({_q(c)}) FROM {_q(t)}")
                if s is not None and s == 0:
                    zero.append(f"{t}.{c}")
                elif n > 50 and self.one(f"SELECT count(DISTINCT {_q(c)}) FROM {_q(t)}", 0) == 1:
                    const.append(f"{t}.{c}")
        self.add("no dead columns", not zero, f"all-zero numeric columns: {zero if zero else 'none'}")
        self.add(
            "no constant columns",
            not const,
            f"constant numeric columns: {const if const else 'none'} (may be a wrong definition)",
            warn=True,
        )

    def check_monotonic(self, ts: Sequence[str]) -> None:
        bad = []
        for t in ts:
            cs = [c for c, _ in self.cols(t)]
            seq = next((c for c in cs if c.endswith("_seq")), None)
            key = next((c for c in cs if c.endswith("_id") and c != seq), None)
            tsc = next((c for c, d in self.cols(t) if d.upper() == "TIMESTAMP"), None)
            if not (seq and key and tsc):
                continue
            gk = next((c for c in cs if c.endswith("_id") and c not in (key,)), key)
            n = self.one(
                f"SELECT count(*) FROM (SELECT {_q(tsc)} ts, "
                f"lag({_q(tsc)}) OVER (PARTITION BY {_q(gk)} ORDER BY {_q(seq)}) p FROM {_q(t)}) "
                f"WHERE p IS NOT NULL AND ts < p",
                0,
            )
            if n:
                bad.append(f"{t} has {n:,} backwards steps")
        self.add("event sequence monotonic", not bad, "; ".join(bad) if bad else "event chains increase strictly")

    @staticmethod
    def _head_cap(k: int) -> float:
        """Single-entity share cap, banded by cardinality.

        This observes an amount (popularity weight x unit price x quantity), and unit-price
        variance inflates concentration to roughly twice the pure-weight theoretical value,
        so the cap leaves headroom.
        """
        return 30.0 if k < 50 else 22.0 if k < 200 else 12.0 if k < 1000 else 8.0

    def check_longtail(self, ts: Sequence[str]) -> None:
        best, best_rows = None, 0
        for t in ts:
            if self.kind.get(t) not in ("fact", "event"):
                continue
            n = self.one(f"SELECT count(*) FROM {_q(t)}", 0)
            if n < 1000 or n < best_rows:
                continue
            ncs = [
                c
                for c in self.numeric_cols(t)
                if any(h in c for h in METRIC_PREFER)
                and not any(x in c for x in ("refund", "cost", "promo", "fee", "claim"))
            ]
            # A usable dimension has a cardinality far below the row count; otherwise every key
            # holds one record and concentration is meaningless.
            fks = [
                c
                for c, _ in self.cols(t)
                if c.endswith("_id") and (self.one(f"SELECT count(DISTINCT {_q(c)}) FROM {_q(t)}", 0) or n) < n * 0.2
            ]
            if not ncs or not fks:
                continue
            best, best_rows = (t, fks[0], ncs[0]), n
        if not best:
            self.add("long-tail concentration", True, "no assessable fact table (skipped)", warn=True)
            return

        t, dim, m = best
        share = self.one(
            f"WITH x AS (SELECT {_q(dim)} k, sum({_q(m)}) v, "
            f"row_number() OVER (ORDER BY sum({_q(m)}) DESC) rn, count(*) OVER () n FROM {_q(t)} GROUP BY 1) "
            f"SELECT 100.0*sum(CASE WHEN rn<=greatest(1,n/10) THEN v END)/sum(v) FROM x"
        )

        # Check every dimension, not just one: an insurance dataset with a healthy holder_id and an
        # agent_id holding 40% of premium is a real case this caught.
        head_bad, head_ok = [], []
        total_rows = self.one(f"SELECT count(*) FROM {_q(t)}", 0) or 1
        for d in [c for c, _ in self.cols(t) if c.endswith("_id")]:
            k = self.one(f"SELECT count(DISTINCT {_q(d)}) FROM {_q(t)}", 0) or 0
            if k < 3 or k > total_rows * 0.2:
                continue  # a near-unique column is not a dimension
            v = self.one(
                f"WITH x AS (SELECT {_q(d)} k, sum({_q(m)}) v FROM {_q(t)} GROUP BY 1) SELECT 100.0*max(v)/sum(v) FROM x"
            )
            if v is None:
                continue
            (head_bad if v > self._head_cap(k) else head_ok).append(f"{d} top entity {v:.1f}% of {k:,}")
        self.add(
            "head not dominant",
            not head_bad,
            (
                f"{t}: "
                + "; ".join(head_bad)
                + "  <- a passing Top-10% total does not make the head reasonable; check the Zipf alpha and rank offset"
            )
            if head_bad
            else f"{t}: " + "; ".join(head_ok or ["no assessable dimension"]),
        )

        # The floor relaxes with cardinality: the Top 10% of a small dimension is two or three
        # entities, and forcing them to hold 50% produces an extreme head. The engine flattens its
        # long-tail target below 200 distinct values, and this must use the same standard.
        n_dim = self.one(f"SELECT count(DISTINCT {_q(dim)}) FROM {_q(t)}", 0) or 1
        lo = 30.0 if n_dim < 60 else 40.0 if n_dim < 200 else 50.0
        if share is None:
            # The metric sums to NULL/0 - exactly the degenerate data this check exists to catch,
            # so it must report a FAIL rather than crash the whole run on a format string.
            self.add(
                "long-tail concentration",
                False,
                f"{t} grouped by {dim}: {m} sums to NULL/0, so concentration cannot be measured "
                f"(a dead metric column is itself the defect)",
            )
        else:
            self.add(
                "long-tail concentration",
                lo <= share <= 92,
                f"{t} grouped by {dim} on {m}: Top 10% holds {share:.1f}% ({n_dim:,} entities, floor {lo:.0f}%)"
                + ("" if lo <= share <= 92 else "  <- too low = no head, too high = stacked Zipf weights"),
            )

    def check_density(self, ts: Sequence[str]) -> None:
        rows = []
        for t in ts:
            if self.kind.get(t) == "summary":
                cc = next((c for c in self.numeric_cols(t) if c.endswith("_cnt")), None)
                if cc:
                    rows.append((t, cc, self.one(f"SELECT avg({_q(cc)}) FROM {_q(t)} WHERE {_q(cc)}>0", 0) or 0))
            elif self.kind.get(t) == "fact":
                # Without a summary layer, measure the fact table's own per-day density.
                expr = None
                for c, d in self.cols(t):
                    if d.upper() not in ("DATE", "TIMESTAMP") or any(a in c for a in AUDIT_COLS):
                        continue
                    expr = _q(c) if d.upper() == "DATE" else f"CAST({_q(c)} AS DATE)"
                    break
                if expr:
                    v = self.one(
                        f"SELECT avg(n) FROM (SELECT {expr} d, count(*) n FROM {_q(t)} GROUP BY 1) WHERE n>0", 0
                    )
                    rows.append((f"{t} (by day)", "rows/day", v or 0))
        total = self.one(f"SELECT sum(estimated_size) FROM duckdb_tables() WHERE {_OWN_CATALOG}", 0) or 0
        thr = 20 if total >= 500_000 else 8  # a small dataset cannot reach 20 per cell at a fine grain
        ok = any(r[2] >= thr for r in rows)
        self.add(
            "aggregation density",
            ok,
            f"threshold {thr}/cell (database {total:,.0f} rows): "
            + "; ".join(f"{t}.{c} {v:.1f}" for t, c, v in sorted(rows, key=lambda x: -x[2])[:5])
            + ("" if ok else "  <- no grain qualifies; reduce dimensions or add a coarser summary (invariant 14)"),
        )

    def check_semantics(self, ts: Sequence[str]) -> None:
        nt = self.one(f"SELECT count(*) FROM duckdb_tables() WHERE {_OWN_CATALOG} AND comment IS NOT NULL", 0)
        tt = len(ts)
        self.add(
            "metadata comments",
            nt >= tt * 0.9,
            f"{nt}/{tt} tables carry a comment (the agent reads them to understand semantics)",
        )
        bad = []
        for t in ts:
            if self.kind.get(t) != "dim":
                continue
            for c, d in self.cols(t):
                if c.endswith("_name") and d.upper().startswith("VARCHAR"):
                    n = self.one(
                        f"SELECT count(*) FROM {_q(t)} WHERE regexp_matches({_q(c)}, '^[a-zA-Z_]+[_ ]?[0-9]+$')", 0
                    )
                    tot = self.one(f"SELECT count(*) FROM {_q(t)}", 1)
                    if n and n > tot * 0.5:
                        bad.append(f"{t}.{c} {n}/{tot} look like xxx_123")
        self.add(
            "semantic naming",
            not bad,
            "; ".join(bad) if bad else "dimension names are not placeholders (e.g. seller_0 / wh_3)",
        )

    def check_config(self) -> None:
        from datus.utils.sql_utils import validate_read_only_sql

        for i, a in enumerate(self.cfg.get("assertions", []) or []):
            # Assertions are hand-written JSON. A missing key or an unknown `expect` is a
            # configuration mistake and must be reported as one, not crash every other check.
            name = (a or {}).get("name") or f"assertion #{i + 1}"
            sql = (a or {}).get("sql") if isinstance(a, dict) else None
            if not isinstance(sql, str) or not sql.strip():
                # The validators below assume a string; a number or a list would raise out of
                # this loop and take every remaining check with it.
                self.add(name, False, "malformed assertion: 'sql' must be a non-empty string")
                continue
            # The tool layer gates this too; repeated here so the checker cannot be handed a
            # write through another caller. parse_sql_type reads only the first statement, so the
            # multi-statement rule this validator carries is the part that actually matters.
            violation, _ = validate_read_only_sql(sql, "duckdb")
            if violation:
                self.add(name, False, f"assertion must be a single read-only query ({violation})")
                continue
            exp = a.get("expect", "zero")
            v = self.one(sql)
            if exp == "zero":
                ok, d = (v == 0), f"actual {v}"
            elif exp == "nonzero":
                ok, d = bool(v), f"actual {v}"
            elif isinstance(exp, dict):
                lo, hi = exp.get("min", float("-inf")), exp.get("max", float("inf"))
                ok = v is not None and lo <= v <= hi
                d = f"actual {v:.4g} (expected {lo}~{hi})" if v is not None else "query returned nothing"
            else:
                ok, d = False, f"unknown expect {exp!r}: use 'zero', 'nonzero' or {{'min':x,'max':y}}"
            self.add(name, ok, d)

    # ------------------------------------------------------------------ entry point

    def run(self) -> List[Dict[str, str]]:
        ts = self.check_structure()
        self.check_fk(ts)
        self.check_time()
        self.check_future(ts)
        self.check_derived(ts)
        self.check_dead_cols(ts)
        self.check_monotonic(ts)
        self.check_longtail(ts)
        self.check_density(ts)
        self.check_semantics(ts)
        self.check_config()
        self.check_query_health()
        return [{"check": n, "status": s, "detail": d} for n, s, d in self.results]

    def check_query_health(self) -> None:
        """Surface queries that failed, so a run degraded by them is not read as a clean pass.

        A failed probe yields [] and reads as "nothing wrong", so without this a schema the
        checker cannot address (an unsupported type, a view it cannot aggregate) would come back
        all-PASS. WARN rather than FAIL: some probes do not apply to some shapes by design.
        """
        if not self.query_errors:
            return
        self.add(
            "all checks could run",
            False,
            f"{len(self.query_errors)} query(ies) failed, so the checks that depend on them "
            f"proved nothing: " + "; ".join(self.query_errors[:3]) + (" ..." if len(self.query_errors) > 3 else ""),
            warn=True,
        )


def summarize(results: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    """Aggregate a run into counts plus the failing/warning subsets."""
    n_pass = sum(1 for r in results if r["status"] == PASS)
    n_fail = sum(1 for r in results if r["status"] == FAIL)
    n_warn = sum(1 for r in results if r["status"] == WARN)
    return {
        "total": len(results),
        "passed": n_pass,
        "failed": n_fail,
        "warned": n_warn,
        "ok": n_fail == 0,
        "failures": [r for r in results if r["status"] == FAIL],
        "warnings": [r for r in results if r["status"] == WARN],
    }
