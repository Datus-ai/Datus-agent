"""DDL-driven synthetic data engine: give it a CREATE TABLE script, get semantically correct demo data.

    from ddl_engine import DDLEngine
    eng = DDLEngine(ddl_sql, rows=100_000, profile=PROFILE)
    eng.report()                 # inspect the inference first; override in the profile if wrong
    eng.generate("demo.duckdb")

Design: the DDL already states table names, column names, types and keys - an LLM need not re-derive them.
The engine owns everything generic: table roles, column semantics, topological order, the 17 invariants.
The profile only adds what DDL cannot express: enum domains, status mixes, differentiation dicts, calendars, vocabularies.
"""

from __future__ import annotations

import bisect
import itertools
import random
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
from genlib import (
    DIM_DENSITY,
    Calendar,
    CsvOut,
    EventChain,
    bounded_gauss,
    date_range,
    day_ts,
    day_weights,
    lognorm_between,
    tune_alpha,
    zipf_weights,
)

NUM_T = ("DECIMAL", "DOUBLE", "FLOAT", "REAL", "NUMERIC")
INT_T = ("INTEGER", "BIGINT", "SMALLINT", "TINYINT", "HUGEINT", "INT")
STR_T = ("VARCHAR", "CHAR", "TEXT", "STRING")

# Column-semantics rules: (regex, allowed type prefixes, semantic). Order matters - first match wins.
# Many columns in user DDL are bare words (channel / quantity / clicks / city) with no _cd/_cnt suffix,
# so a vocabulary rule set is needed too, or they degrade to text/measure and get filled with junk.
BARE_ENUM = (
    r"^(status|state|channel|source|medium|platform|device|os|category|subcategory|"
    r"class|kind|type|level|tier|grade|segment|gender|sex|country|nation|region|"
    r"province|state_cd|city|district|currency|payment|method|brand|supplier|vendor|"
    r"store|shop|warehouse|carrier|courier|department|industry|size|color|unit)$"
)
BARE_COUNT = (
    r"^(qty|quantity|units|pieces|headcount)$|(_qty|_quantity|_units|_pieces)$|"
    r"(^|_)(impressions|clicks|sessions|visits|visitors|views|users|orders|purchasers|"
    r"buyers|carts|checkouts|installs|signups|leads|conversions|opens|plays|likes|"
    r"shares|comments|followers|subscribers|tickets|calls|claims)$"
)
BARE_AMOUNT = (
    r"^(spend|budget|revenue|sales|gmv|turnover|income|expense|freight|shipping|"
    r"discount|tax|subsidy|deposit)$|(_spend|_budget|_sales|_gmv|_income|_expense|"
    r"_charge|_subsidy|_deposit|_payout)$"
)

SEMANTIC = [
    (r"^(date_key|stat_dt|dt)$", ("DATE",), "date_pk"),
    (BARE_COUNT, INT_T + NUM_T, "count"),
    (BARE_AMOUNT, NUM_T + INT_T, "amount"),
    (BARE_ENUM, STR_T, "enum"),
    # campaign / activity / group are categorical attributes, not entity names - send them to enum, not the name template
    (r"(campaign|promotion|program|ad_group|adgroup)(_name|_cd|_code)?$", STR_T, "enum"),
    (
        r"(_reason|_cause|_result|_mode|_device|_source|_medium|_category|_brand|_supplier|"
        r"_currency|_region|_country|_province|_city|_group|_gender|_platform|_class)$",
        STR_T,
        "enum",
    ),
    (r"(_dt|_date|_day)$", ("DATE",), "date"),
    (r"(_ts|_time|_at)$", ("TIMESTAMP",), "ts"),
    (
        r"(_amt|_amount|_price|_fee|_cost|_gmv|_revenue|_profit|_balance|_premium|_usd|_value)(_\w+)?$",
        NUM_T + INT_T,
        "amount",
    ),
    (r"(_rate|_pct|_ratio|_share|_margin)$|^(ctr|cvr|roi)$", NUM_T, "ratio"),
    (r"(_cnt|_qty|_num|_count|_times|_seats|_days|_hours|_sec|_min)$", INT_T + NUM_T, "count"),
    (r"^(is_|has_)|(_flag)$", INT_T + ("BOOLEAN",) + STR_T, "flag"),
    (r"(_seq|_no|_order_num)$", INT_T, "seq"),
    (r"(_status|_cd|_code|_type|_level|_tier|_seg|_channel|_method)(_\w+)?$", STR_T, "enum"),
    (r"(_name|_title|_desc|_label)$", STR_T, "name"),
    (r"(_id|_key)$", STR_T + INT_T, "id"),
    (r"(_weight|_kg|_size|_length|_area|_score)$", NUM_T + INT_T, "measure"),
]

# Every semantic a column may carry. The regex table above is a zero-cost default that is right
# most of the time; anything it gets wrong is corrected per run through profile["semantics"],
# which is why the table never needs to grow another industry's vocabulary.
SEMANTICS = (
    "id",
    "date_pk",
    "date",
    "ts",
    "amount",
    "count",
    "ratio",
    "enum",
    "flag",
    "name",
    "seq",
    "measure",
    "text",
)

ROLE_DATE, ROLE_DIM, ROLE_FACT, ROLE_DETAIL, ROLE_EVENT, ROLE_SNAPSHOT, ROLE_DOWNSTREAM = (
    "date_dim",
    "dim",
    "fact",
    "detail",
    "event",
    "snapshot",
    "downstream",
)
ROLE_METRIC = "metric_daily"  # daily metric table with no FK (traffic/spend/headline); a day x dimension grid

AUDIT_TS = re.compile(
    r"^(created|create|updated|update|modified|inserted|insert|loaded|load|"
    r"etl|sync|dw)_(at|ts|time|dt)$"
)

# A dimension entity's effective date: registered / hired / listed / opened - facts referencing it cannot predate it (invariant 3).
# Attribute dates such as birth/expire/valid_to are deliberately excluded; they say nothing about availability.
EFFECTIVE_DATE = re.compile(
    r"(^|_)(reg|register|registered|signup|sign_up|enroll|join|joined|onboard|onboarding|hire|hired|"
    r"launch|launched|list|listed|shelf|publish|published|activate|activated|effective|"
    r"open|opened|start|issue|issued|found|founded|entry)(_|$)"
)

# Columns a date dimension is allowed to carry: every one of them is a function of the date itself.
# A daily metric table has the same shape - a date grain, no foreign key, a handful of measures -
# and is told apart only by carrying something the date does not determine (channel, impressions,
# gmv). Without that test the three most idiomatic names for a metric table's date column
# (``stat_dt`` / ``dt`` / ``date_key``, all classified ``date_pk``) routed the whole table through
# the date-dimension generator and every business column came out NULL.
CALENDAR_ATTR = re.compile(
    r"(^|_)(year|yr|month|mon|quarter|qtr|week|wk|day|dow|doy|date|dt|"
    r"weekend|workday|holiday|season|period|fiscal|half|decade|epoch)(_|$)"
)

# Standard dim_date columns (added when the DDL has no date dimension: anomaly attribution and period comparison anchor on it)
DATE_DIM_COLS = [
    ("date_key", "DATE"),
    ("year_num", "INTEGER"),
    ("month_key", "INTEGER"),
    ("quarter_cd", "VARCHAR"),
    ("week_of_year", "INTEGER"),
    ("day_of_week", "INTEGER"),
    ("day_of_month", "INTEGER"),
    ("is_weekend", "INTEGER"),
    ("is_workday", "INTEGER"),
    ("day_type_cd", "VARCHAR"),
    ("event_name", "VARCHAR"),
]


def _semantic(col: str, dtype: str) -> str:
    dtype = dtype.upper()
    for pat, types, sem in SEMANTIC:
        if re.search(pat, col) and any(dtype.startswith(t) for t in types):
            return sem
    if dtype.startswith("DATE"):
        return "date"
    if dtype.startswith("TIMESTAMP"):
        return "ts"
    if any(dtype.startswith(t) for t in NUM_T + INT_T):
        return "measure"
    return "text"


class DDLEngine:
    def __init__(self, ddl, rows=80_000, profile=None, months=17, end_date=None, seed=42, extra_tables="none"):
        """extra_tables decides whether anything is built beyond the user DDL (default none = follow the DDL exactly):
            "none"     only the tables written in the DDL
            "date_dim" plus a date dimension (the event-name anchor for promotions/anomalies)
            "summary"  plus a summary layer (dws/ads at day grain)
            "all"      both
        Extra tables give the agent two competing definitions and carry no user DDL constraints, hence off by default."""
        assert extra_tables in ("none", "date_dim", "summary", "all"), extra_tables
        self.extra_tables = extra_tables
        self.ddl = ddl
        self.rows = rows
        self.profile = profile or {}
        self.seed = seed
        self.rng = random.Random(seed)
        self.end = end_date or (date.today() - timedelta(days=1))
        start = self.end.replace(day=1)
        for _ in range(months - 1):
            start = (start - timedelta(days=1)).replace(day=1)
        self.start = start
        self.months = months
        self.days = date_range(start, self.end)
        self.cal = self._calendar()
        self.day_w = day_weights(
            self.days,
            self.cal,
            trend_mom=self.profile.get("trend_mom", 0.031),
            weekend_lift=self.profile.get("weekend_lift", 1.33),
        )
        self._cum_w = list(itertools.accumulate(self.day_w))
        self.pools, self.refs = {}, {}
        self._enum_cache = {}
        self._fact_rows, self._pending_dim_rows, self._id_base = {}, {}, {}
        self._code_seq = {}  # (table, column) -> codes handed out, so _fill_generic never repeats one
        self.schema = self._parse()
        self.synthetic = set()
        if self.extra_tables in ("date_dim", "all"):
            self._ensure_date_dim()
        self._apply_semantics()
        self._infer()

    def _ensure_date_dim(self):
        """Add a date dimension when the DDL has none: period comparison, promotion attribution and the calendar all anchor on it."""
        if self.profile.get("no_date_dim"):
            return
        for t, cols in self.schema.items():
            names = [c["name"] for c in cols]
            if (
                any(n in ("date_key", "stat_dt", "dt") for n in names)
                and len(cols) <= 12
                and not any(n.endswith("_id") for n in names)
                # A daily metric table has all of the above and is not a date dimension. Without
                # this a caller who asked for one with extra_tables="date_dim" silently got none,
                # because `stat_dt` on the metric table was mistaken for a calendar already there.
                and not self._carries_non_calendar_data(cols)
            ):
                return
        name = self.profile.get("date_dim_name", "dim_date")
        self.schema[name] = [{"name": c, "type": d, "sem": _semantic(c, d)} for c, d in DATE_DIM_COLS]
        self.synthetic.add(name)

    def _apply_semantics(self):
        """Apply profile["semantics"] = {"table.column": semantic} over the inferred semantics.

        The regex table is a naming heuristic, and naming conventions are per-industry: `paid_amount`
        is recognised as money, `insurance_paid` and `copay` are not. Rather than growing the table
        one vocabulary at a time, the caller states the semantics it disagrees with - an LLM reading
        the DDL judges this far better than a regex, and freezing its judgement here in the profile
        keeps the result reproducible and auditable instead of hidden in a pattern list.

        Applied before inference so a corrected semantic also feeds role detection (main-fact choice
        counts amount columns, fact detection looks for business dates). Explicitly set columns are
        then exempt from the inference-time adjustments in ``_infer``, the same way profile["roles"]
        beats role inference.
        """
        self._sem_override = {}
        for key, sem in (self.profile.get("semantics") or {}).items():
            if "." not in key:
                continue
            t, c = key.split(".", 1)
            for col in self.schema.get(t, []):
                if col["name"] == c and sem in SEMANTICS:
                    col["sem"] = sem
                    self._sem_override[(t, c)] = sem

    # ---------------------------------------------------------------- parsing
    def _parse(self):
        """Let DuckDB parse the DDL - no SQL parser to write, and common dialects just work."""
        con = duckdb.connect(":memory:")
        for stmt in [s.strip() for s in self.ddl.split(";") if s.strip()]:
            try:
                con.execute(stmt)
            except Exception as e:
                cleaned = re.sub(r"\b(ENGINE|CHARSET|COLLATE|COMMENT)\s*=?\s*'[^']*'", "", stmt)
                cleaned = re.sub(r"\bAUTO_INCREMENT\b|\bUNSIGNED\b|\bCOMMENT\s+'[^']*'", "", cleaned, flags=re.I)
                try:
                    con.execute(cleaned)
                except Exception:
                    raise ValueError(
                        f"Cannot parse DDL: {e}\nStatement: {stmt[:200]}\n"
                        f"The engine parses DuckDB syntax. This looks like another dialect "
                        f"(PostgreSQL/MySQL/StarRocks/Oracle/...): rewrite it to DuckDB first - drop the "
                        f"schema prefix and backticks, map the types, strip PARTITION BY / DISTRIBUTED BY / "
                        f"PROPERTIES / ENGINE / storage clauses and CREATE INDEX, and keep the inline "
                        f"comments (the engine extracts enum domains from them)."
                    )
        self.ddl_enums = self._scan_ddl_comments()
        self.decl_pk, self.decl_uniq, self.decl_fk = self._scan_constraints(con)
        # Normalised CREATE TABLE text, so the built database can carry the declared
        # PRIMARY KEY / UNIQUE / FOREIGN KEY instead of the constraint-free shape that
        # CREATE TABLE AS SELECT produces. An agent reads those keys off the schema.
        self.decl_sql = {
            t: sql for t, sql in con.execute("SELECT table_name, sql FROM duckdb_tables()").fetchall() if sql
        }
        schema = {}
        for (t,) in con.execute("SELECT table_name FROM duckdb_tables() ORDER BY 1").fetchall():
            cols = con.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                f"WHERE table_name='{t}' ORDER BY ordinal_position"
            ).fetchall()
            schema[t] = [{"name": c, "type": d, "sem": _semantic(c, d)} for c, d in cols]
        con.close()
        if not schema:
            raise ValueError("No table was parsed out of the DDL")
        return schema

    def _scan_constraints(self, con):
        """Read the PRIMARY KEY / FOREIGN KEY / UNIQUE declared in the DDL. A relationship the user wrote down
        is authoritative and beats guesses such as "the first id column". DuckDB normalises the dialects."""
        pk, uniq, fk = {}, {}, {}
        try:
            rows = con.execute(
                "SELECT table_name, constraint_type, constraint_column_names, constraint_text FROM duckdb_constraints()"
            ).fetchall()
        except Exception:
            return pk, uniq, fk
        for t, ctype, cols, txt in rows:
            cols = list(cols or [])
            if ctype == "PRIMARY KEY" and cols:
                pk[t] = cols
            elif ctype == "UNIQUE" and cols:
                uniq.setdefault(t, []).append(cols)
            elif ctype == "FOREIGN KEY" and cols:
                m = re.search(r'REFERENCES\s+["`]?([\w.]+)["`]?\s*\(\s*["`]?(\w+)', txt or "", re.I)
                if m:
                    fk[(t, cols[0])] = (m.group(1).split(".")[-1], m.group(2))
        return pk, uniq, fk

    def _scan_ddl_comments(self):
        """Extract enum domains from DDL inline comments - people usually list the values there already, e.g.
        `order_status VARCHAR,  -- pending / paid / shipped / completed`.
        Once extracted, the profile need not repeat them under enums."""
        out = {}
        for line in self.ddl.splitlines():
            m = re.match(r"\s*[`\"']?(\w+)[`\"']?\s+[\w()., ]+?[^-]*?--\s*(.+?)\s*$", line)
            if not m:
                continue
            col, note = m.group(1), m.group(2)
            # Real DDL usually labels the column before listing its values
            # (`-- order status: pending / paid / shipped`). Keep only what follows the last
            # colon, or the label joins the first value and the whole domain is discarded.
            if re.search(r"[:\uff1a]", note):
                tail = re.split(r"[:\uff1a]", note)[-1].strip()
                if tail:
                    note = tail
            if re.search(r"[\u4e00-\u9fff]{4,}", note) and not re.search(r"[/|\u3001]", note):
                continue  # prose description, not a value domain
            partial = "..." in note or "…" in note
            # `/` is the documented separator; `|` and the CJK enumeration comma are accepted
            # because hand-written DDL uses them just as often. `,` is deliberately not one -
            # it is ambiguous with the column separator and appears inside values.
            parts = [x.strip().rstrip(".").strip() for x in re.split(r"\s*[/|\u3001]\s*", note) if x.strip()]
            parts = [x for x in parts if x and len(x) <= 24]
            if len(parts) >= 2 and all(re.match(r"^[\w\u4e00-\u9fff +\-']{1,24}$", x) for x in parts):
                out[col] = parts
                if partial:  # comment ends in "..." (incomplete); flag it so it can be completed
                    self._partial_enums = getattr(self, "_partial_enums", set()) | {col}
        return out

    # ---------------------------------------------------------------- inference
    def _infer(self):
        ov = self.profile.get("roles", {})
        pk, self.roles, self.fks = {}, {}, {}
        for t, cols in self.decl_pk.items():  # 1. keys declared in the DDL
            if len(cols) == 1 and t in self.schema:
                pk[cols[0]] = t
        for (t, col), (ref_t, _rc) in self.decl_fk.items():  # 2. FK targets declared in the DDL
            if ref_t in self.schema:
                pk.setdefault(col, ref_t)
        for t, cols in self.schema.items():  # 3. only guess when nothing is declared: first id column
            ids = [c["name"] for c in cols if c["sem"] == "id"]
            if ids:
                pk.setdefault(ids[0], t)
        ovr = getattr(self, "_sem_override", {})
        # Columns whose domain is written in a DDL comment are always generated as enums (declared info beats naming conventions)
        for t, cols in self.schema.items():
            for c in cols:
                if (
                    c["name"] in getattr(self, "ddl_enums", {})
                    and c["sem"] in ("text", "name")
                    and (t, c["name"]) not in ovr
                ):
                    c["sem"] = "enum"
        # A declared PK/FK column must be generated with id semantics even when it is not named *_id (else it gets text placeholders)
        for t, col in list(self.decl_fk) + [(t, c[0]) for t, c in self.decl_pk.items() if len(c) == 1]:
            for c in self.schema.get(t, []):
                if c["name"] == col and (t, col) not in ovr:
                    c["sem"] = "id"
        for t, cols in self.schema.items():
            declared = [col for (tt, col) in self.decl_fk if tt == t]
            own_pk = self.decl_pk.get(t, [cols[0]["name"]])[0]
            inferred = [
                c["name"]
                for c in cols
                if c["name"] != own_pk and c["sem"] == "id" and c["name"] in pk and pk[c["name"]] != t
            ]
            self.fks[t] = declared + [c for c in inferred if c not in declared]
        # Attribute dates (registered/hired/listed) are not business dates; business dates are what mark a fact table
        attr_date = re.compile(
            r"(reg|onboard|hire|launch|first|join|create|birth|open|expire|"
            r"valid|found|entry)"
        )

        def _traits(t):
            cols = self.schema[t]
            sems = [c["sem"] for c in cols]
            # ``date_pk`` counts: it is the strongest possible business date - the table's own grain.
            # Leaving it out sent a demoted daily metric table to ROLE_DIM instead of ROLE_METRIC.
            biz_date = any(
                c["sem"] in ("date", "ts", "date_pk")
                and not attr_date.search(c["name"])
                and not AUDIT_TS.match(c["name"])
                for c in cols
            )
            return biz_date, sems.count("amount"), "seq" in sems and "ts" in sems

        def _measures(t):
            return sum(1 for c in self.schema[t] if c["sem"] in ("amount", "count", "ratio", "measure"))

        for t, cols in self.schema.items():
            sems = [c["sem"] for c in cols]
            if t in ov:
                self.roles[t] = ov[t]
            elif self._is_date_dim(t, cols, sems):
                self.roles[t] = ROLE_DATE
            elif _traits(t)[2]:
                self.roles[t] = ROLE_EVENT
            elif re.search(r"snapshot|_snap|balance_daily|_hist$", t):
                self.roles[t] = ROLE_SNAPSHOT
            elif not self.fks[t]:
                biz, namt, _ = _traits(t)
                # Structural decision, never the table-name prefix (the skill states "no prefix dependency"):
                # business date + metric columns = fact; if the date column IS the grain and there is no FK at all, it is a daily metric table
                if biz and (namt >= 1 or _measures(t) >= 3):
                    has_date_grain = any(
                        c["sem"] in ("date", "date_pk") and not attr_date.search(c["name"]) for c in cols
                    )
                    self.roles[t] = ROLE_METRIC if (has_date_grain and _measures(t) >= 3) else ROLE_FACT
                else:
                    self.roles[t] = ROLE_DIM
        # Iterate: a table referencing only dimensions is a fact; a table referencing facts is a detail/downstream
        for _ in range(len(self.schema)):
            for t in self.schema:
                if t in self.roles:
                    continue
                deps = [pk[f] for f in self.fks[t] if f in pk]
                known = [self.roles.get(d) for d in deps]
                if any(k is None for k in known):
                    continue
                if any(k in (ROLE_FACT, ROLE_DOWNSTREAM, ROLE_DETAIL) for k in known):
                    self.roles[t] = ROLE_DETAIL
                else:
                    biz, namt, _ = _traits(t)
                    # Referencing only dimensions: it is a fact only if it has a business date. An attribute table like
                    # dim_product ("has price/cost but only a listing date") holds amounts as attributes, not measures - still a dimension.
                    self.roles[t] = ROLE_FACT if biz else ROLE_DIM
        for t in self.schema:
            self.roles.setdefault(t, ROLE_FACT)
        # Main fact: the fact table with the most foreign keys that also carries amounts
        facts = [t for t, r in self.roles.items() if r == ROLE_FACT]
        if facts:
            self.main_fact = max(
                facts, key=lambda t: (sum(1 for c in self.schema[t] if c["sem"] == "amount"), len(self.fks[t]))
            )
            for t in facts:
                if t != self.main_fact and self.main_fact in [pk.get(f) for f in self.fks[t]]:
                    self.roles[t] = ROLE_DETAIL
        else:
            self.main_fact = None
        for t, r in list(self.roles.items()):
            if r != ROLE_DETAIL:
                continue
            par = next(
                (
                    pk[f]
                    for f in self.fks[t]
                    if pk.get(f) in self.roles and self.roles[pk[f]] in (ROLE_FACT, ROLE_DOWNSTREAM)
                ),
                None,
            )
            if not par:
                continue
            pdates = {c["name"] for c in self.schema[par] if c["sem"] == "date"}
            odates = {c["name"] for c in self.schema[t] if c["sem"] == "date"}
            if odates and not (odates & pdates):  # own dates, differently named from the parent -> downstream fact
                self.roles[t] = ROLE_DOWNSTREAM
        self.pk_owner = pk
        self._plan_rows()

    def _is_date_dim(self, t, cols, sems):
        """Is this a date dimension, or a daily metric table wearing the same shape?

        Both have a ``date_pk``, no foreign key and a small column list, so the original test
        matched either. The difference is what the table is keyed by: a date dimension is keyed by
        the date alone and every attribute follows from it, while a daily metric table is keyed by
        date x dimension and carries measures the calendar cannot produce. So any enum, amount,
        count, ratio or measure column that is not a calendar attribute rules a date dimension out.

        Flags (``is_weekend``) and names (``event_name``) are not tested: they carry no grain and a
        real date dimension has them.
        """
        if "date_pk" not in sems or len(cols) > 12 or self.fks[t]:
            return False
        return not self._carries_non_calendar_data(cols)

    @staticmethod
    def _carries_non_calendar_data(cols):
        """Does this table hold anything the date alone does not determine?

        The one test that separates a date dimension from a daily metric table, and the only part
        of that decision `_ensure_date_dim` can make - it runs before `_infer`, so foreign keys are
        not known yet. Flags and names are not tested: `is_weekend` and `event_name` belong to a
        real date dimension and carry no grain.
        """
        return any(
            c["sem"] in ("enum", "amount", "count", "ratio", "measure") and not CALENDAR_ATTR.search(c["name"])
            for c in cols
        )

    def _plan_rows(self):
        """Allocate rows per skill Phase 1.2/1.3: the fact layer takes the bulk, dimensions size by business density."""
        roles = self.roles
        dims = [t for t, r in roles.items() if r == ROLE_DIM]
        facts = [t for t, r in roles.items() if r == ROLE_FACT]
        details = [t for t, r in roles.items() if r in (ROLE_DETAIL, ROLE_DOWNSTREAM)]
        events = [t for t, r in roles.items() if r == ROLE_EVENT]
        snaps = [t for t, r in roles.items() if r == ROLE_SNAPSHOT]
        metrics = [t for t, r in roles.items() if r == ROLE_METRIC]
        pinned = {**self.profile.get("dim_rows", {}), **self.profile.get("table_rows", {})}
        self.pinned_rows = pinned
        share = {"fact": 0.32 if (events or snaps) else 0.52, "detail": 0.28, "event": 0.34, "snapshot": 0.20}
        if not details:
            share["fact"] += share.pop("detail", 0)
        if not events:
            share["event"] = 0
        if not snaps:
            share["snapshot"] = 0
        tot = sum(v for v in share.values() if v) or 1
        budget = self.rows * 0.94  # leave 6% for dimensions
        n = {}
        for t in facts:
            n[t] = max(50, int(budget * share["fact"] / tot / len(facts)))
        main_n = n.get(self.main_fact, max(50, int(budget * 0.5)))
        for t in details:
            n[t] = max(50, int(budget * share["detail"] / tot / max(1, len(details))))
        for t in events:
            n[t] = max(50, int(budget * share["event"] / tot / max(1, len(events))))
        for t in snaps:
            n[t] = max(50, int(budget * share["snapshot"] / tot / max(1, len(snaps))))
        # Dimension cardinality: by business density, capped by the hard constraints
        kinds = self.profile.get("dim_kinds", {})
        for t in dims:
            if t in pinned:
                continue
            kind = kinds.get(t) or self._guess_kind(t)
            n[t] = max(4, min(int(self.rows * 0.08), round(main_n / DIM_DENSITY[kind])))
        for t in metrics:  # daily metric table: rows = days x number of dimension combinations
            n[t] = max(len(self.days), int(self.rows * 0.07))
        for t, r in roles.items():
            if r == ROLE_DATE:
                n[t] = len(self.days)
        n.update({t: v for t, v in pinned.items() if t in self.schema})  # an explicit user value wins over everything
        self.nrows = n

    def _guess_kind(self, t):
        s = t.lower()
        if re.search(r"courier|carrier|channel|plan|method|payment_type|product_line|policy_type", s):
            return "enum"
        if re.search(r"seller|store|shop|merchant|warehouse|branch|dept|line|factory|team|org", s):
            return "org"
        if re.search(r"staff|driver|doctor|agent|teacher|employee|courier_man|nurse", s):
            return "staff"
        if re.search(r"product|sku|item|course|goods|material|drug", s):
            return "item"
        if re.search(r"user|customer|buyer|member|patient|student|account|player|policyholder", s):
            return "customer"
        return "org"

    def _calendar(self):
        c = self.profile.get("calendar", {})
        promos, slows, disrupts = [], [], []
        for y in range(self.start.year, self.end.year + 1):
            for md0, md1, mult, nm in c.get("promos", []):
                try:
                    s = date(y, *map(int, md0.split("-")))
                    e = date(y, *map(int, md1.split("-")))
                except ValueError:
                    continue
                if self.start <= s and e <= self.end:
                    promos.append((s, e, mult, f"{y} {nm}"))
            for md0, md1, mult, nm in c.get("slows", []):
                try:
                    s = date(y, *map(int, md0.split("-")))
                    e = date(y, *map(int, md1.split("-")))
                except ValueError:
                    continue
                if self.start <= s and e <= self.end:
                    slows.append((s, e, mult, f"{y} {nm}"))
        span = (self.end - self.start).days
        for d in c.get("disruptions", []):
            s = self.start + timedelta(days=int(span * d.get("at", 0.5)))
            disrupts.append((s, s + timedelta(days=d.get("days", 28)), d["factor"], d.get("scope", {}), d["name"]))
        return Calendar(promos, slows, disrupts)

    # ---------------------------------------------------------------- structural metadata
    def _dump_meta(self, db_path, sizes):
        """Persist the engine inference so the quality check reuses it, instead of re-deriving a second, conflicting view."""
        import json

        made = set(getattr(self, "_made", {})) | set(getattr(self, "synthetic", []))
        meta = {
            "generator": "gen-datasource/ddl_engine",
            "date_range": [self.start.isoformat(), self.end.isoformat()],
            "extra_tables": self.extra_tables,
            "strict_ddl": self.extra_tables == "none",
            "declared": {
                "pk": self.decl_pk,
                "fk": {f"{t}.{c}": list(v) for (t, c), v in self.decl_fk.items()},
                "unique": self.decl_uniq,
            },
            "roles": dict(self.roles),
            "synthetic_tables": sorted(made),
            "fks": {t: v for t, v in self.fks.items() if v},
            "pks": {t: self.pk_of(t) for t in self.schema},
            "col_sem": {f"{t}.{c['name']}": c["sem"] for t, cols in self.schema.items() for c in cols},
            "rows": sizes,
            "fingerprint": self._fingerprint(),
            "calibrated_rows": {t: n for t, n in self.nrows.items()},
        }
        try:
            f = Path(db_path).resolve()
            (
                f.parent / f".{f.stem}.meta.json"
            ).write_text(  # bound to the database name so sibling databases do not clash
                json.dumps(meta, ensure_ascii=False, indent=2)
            )
        except Exception:
            pass

    # ---------------------------------------------------------------- configuration pre-check
    def _future_tables(self):
        """Tables that only appear during generation: the auto-built date dim / summary layer, plus tables the profile CREATEs itself."""
        out = set()
        if self.extra_tables in ("date_dim", "all"):
            out.add("dim_date")
        if self.extra_tables in ("summary", "all"):
            out.add("ads_business_daily")
            if self.main_fact:
                stem = self.main_fact.split("_", 1)[-1]
                out |= {f"dws_{stem}_subject_day", f"dws_{stem}_channel_day"}
        for key in ("pre_sql", "extra_sql"):
            out |= set(
                re.findall(r"CREATE\s+(?:OR\s+REPLACE\s+)?TABLE\s+[\"`]?(\w+)", str(self.profile.get(key, "")), re.I)
            )
        return out

    def precheck(self, strict=True):
        """Validate the profile before generating: do the referenced tables/columns exist, do formulas cycle, are dimensions oversized.

        Returns (errors, warnings). With strict=True an error raises immediately - turning "found out after the run"
        into "known before the run", which saves a whole regeneration cycle.
        """
        err, warn = [], []
        tabs = set(self.schema) | self._future_tables()  # includes the tables the engine will add
        colof = {t: {c["name"] for c in cols} for t, cols in self.schema.items()}

        def chk_ref(where, key, need_col=True, fatal=True):
            """fatal=False: a misconfiguration only disables that one setting (e.g. a comment); it does not block generation."""
            bag = err if fatal else warn
            if "." not in key:
                if key not in tabs:
                    bag.append(f"{where}: table `{key}` is not in the DDL")
                return
            t, c = key.split(".", 1)
            if t not in tabs:
                bag.append(f"{where}: table `{t}` is not in the DDL ({key})")
            elif need_col and t in colof and c not in colof[t]:
                near = [x for x in colof[t] if c.lower() in x.lower() or x.lower() in c.lower()]
                bag.append(f"{where}: `{t}` has no column `{c}`" + (f"; did you mean {near[0]}?" if near else ""))

        sql_types_ok = True
        for k in ("pre_sql", "extra_sql"):  # a type error must surface before generating
            v = self.profile.get(k)
            if v is not None and not isinstance(v, (str, list, tuple)):
                err.append(f"{k}: must be a str or a list of str, got {type(v).__name__}")
                sql_types_ok = False
            elif isinstance(v, (list, tuple)) and any(not isinstance(x, str) for x in v):
                err.append(f"{k}: every list element must be a str")
                sql_types_ok = False
        if sql_types_ok:
            err.extend(self._validate_sql_blocks())

        for k in self.profile.get("columns", {}):
            chk_ref("columns", k)

        # A misspelled semantics key must never pass quietly: it would leave the column on its
        # inferred semantic and the data would look plausible while being wrong, which no quality
        # check can detect. Same reasoning as conditional silently doing nothing on a detail table.
        for k, v in (self.profile.get("semantics") or {}).items():
            chk_ref("semantics", k)
            if v not in SEMANTICS:
                err.append(f"semantics[{k}]: `{v}` is not a semantic; choose one of {', '.join(SEMANTICS)}")
            elif "." in k:
                t, c = k.split(".", 1)
                if (t, c) in getattr(self, "_sem_override", {}):
                    declared_key = c in self.decl_pk.get(t, []) or (t, c) in self.decl_fk
                    if declared_key and v != "id":
                        warn.append(
                            f"semantics[{k}]: overriding a key column declared in the DDL to `{v}`; "
                            f"foreign keys to it will not resolve"
                        )
        for k in self.profile.get("column_comments", {}):
            chk_ref("column_comments", k, fatal=False)  # a bad comment only fails to be written; not fatal
        for k in self.profile.get("table_comments", {}):
            chk_ref("table_comments", k, need_col=False, fatal=False)
        for k in (
            list(self.profile.get("table_rows", {}))
            + list(self.profile.get("dim_kinds", {}))
            + list(self.profile.get("naming", {}))
            + list(self.profile.get("roles", {}))
            + list(self.profile.get("event_seq", {}))
        ):
            chk_ref("table-level config", k, need_col=False)

        # conditional: target and grouping columns must exist, and the grouping column must be generated first
        for key, spec in self.profile.get("conditional", {}).items():
            chk_ref("conditional", key)
            if "." not in key or not isinstance(spec, dict):
                continue
            t, c = key.split(".", 1)
            by = spec.get("__by__")
            if not by:
                err.append(f"conditional[{key}]: missing __by__ (the grouping column)")
            elif "." in by:  # cross-table grouping: validate the FK path and the upstream column
                up_t, up_c = by.split(".", 1)
                if up_t not in tabs:
                    err.append(f"conditional[{key}]: upstream table `{up_t}` is not in the DDL")
                elif up_c not in colof.get(up_t, set()):
                    err.append(f"conditional[{key}]: `{up_t}` has no column `{up_c}`")
                elif not any(self.pk_owner.get(f) == up_t for f in self.fks.get(t, [])):
                    err.append(
                        f"conditional[{key}]: `{t}` has no foreign key to `{up_t}`; "
                        f"cross-table grouping needs an FK path"
                    )
                elif self.roles.get(t) == ROLE_DIM:
                    warn.append(
                        f"conditional[{key}]: cross-table grouping is not supported on a dimension yet; __default__ applies"
                    )
            elif t in colof and by not in colof[t]:
                near = [x for x in colof[t] if by.lower() in x.lower()]
                err.append(
                    f"conditional[{key}]: grouping column `{by}` is not on `{t}`"
                    + (
                        f"; did you mean {near[0]}?"
                        if near
                        else "; for cross-table grouping write `upstream_table.column`"
                    )
                )
            elif t in colof:
                sem = {x["name"]: x["sem"] for x in self.schema[t]}
                if sem.get(by) not in ("enum", "id") and sem.get(c) == "amount":
                    warn.append(
                        f"conditional[{key}]: grouping column `{by}` is not an enum/FK; the bands may never match"
                    )
            if "__default__" not in spec:
                warn.append(f"conditional[{key}]: no __default__; unlisted values fall back to the engine default")

        # formulas: referenced columns exist, no self-reference, no cycles
        fml = self.profile.get("formulas", {})
        for key, expr in fml.items():
            chk_ref("formulas", key)
            if "." not in key:
                continue
            t, c = key.split(".", 1)
            if t not in colof:
                continue
            refs = {w for w in re.findall(r"[A-Za-z_]\w*", str(expr)) if w not in ("min", "max", "abs", "round")}
            for r in refs:
                if r not in colof[t]:
                    err.append(f"formulas[{key}]: expression references a non-existent column `{r}`")
            if c in refs:
                err.append(f"formulas[{key}]: formula references itself `{c}`")
        for t in {k.split(".", 1)[0] for k in fml if "." in k} & tabs:
            try:
                self._formulas(t)
            except ValueError as e:
                err.append(f"formulas: {e}")

        for key, d in (self.profile.get("derive", {}) or {}).items():
            chk_ref("derive", key)
            if "." not in key or not isinstance(d, dict):
                continue
            t, _c = key.split(".", 1)
            if t in colof and d.get("from") not in colof[t]:
                err.append(f"derive[{key}]: base column `{d.get('from')}` is not on `{t}`")
            r = d.get("ratio")
            if isinstance(r, dict):
                by = r.get("__by__")
                if not by:
                    err.append(f"derive[{key}]: grouped ratio is missing __by__")
                elif "." not in by and t in colof and by not in colof[t]:
                    err.append(f"derive[{key}]: grouping column `{by}` is not on `{t}`")

        # enums and calendar
        for c in self.profile.get("enums", {}):
            if not any(c in cs for cs in colof.values()):
                warn.append(f"enums[{c}]: no table has a column with this name")
        cal = self.profile.get("calendar", {})
        for kind in ("promos", "slows"):
            for item in cal.get(kind, []):
                if len(item) < 4:
                    continue
                md0, md1, _mult, nm = item[0], item[1], item[2], item[3]
                kept = 0
                for y in range(self.start.year, self.end.year + 1):
                    try:
                        a = date(y, *map(int, str(md0).split("-")))
                        b = date(y, *map(int, str(md1).split("-")))
                    except ValueError:
                        continue
                    if self.start <= a and b <= self.end:
                        kept += 1
                if kept == 0:
                    err.append(
                        f"calendar.{kind}[{nm}]: window {md0}-{md1} never lands inside the data range; "
                        f"the whole window is dropped"
                    )
                else:
                    # How many years the window could theoretically cover: years whose window start falls inside the range
                    try:
                        m0 = int(str(md0).split("-")[0])
                    except ValueError:
                        m0 = 1
                    expect = sum(
                        1 for y in range(self.start.year, self.end.year + 1) if self.start <= date(y, m0, 1) <= self.end
                    )
                    if kept < expect:
                        warn.append(
                            f"calendar.{kind}[{nm}]: window {md0}-{md1} lands in only {kept}/{expect} "
                            f"years - a year crossing the data boundary is dropped whole (not truncated), distorting YoY"
                        )
        for d in cal.get("disruptions", []):
            if not (0 <= d.get("at", 0.5) <= 1):
                err.append(f"calendar.disruptions[{d.get('name')}]: at must be between 0 and 1")
            for sk in d.get("scope", {}):
                if not any(sk in cs for cs in colof.values()):
                    warn.append(f"disruption `{d.get('name')}` scope column `{sk}` does not exist; it will never match")

        # row-count constraints (invariant 10)
        for t, n in self.nrows.items():
            if self.roles.get(t) == ROLE_DIM and n > self.rows * 0.08:
                warn.append(f"dimension {t} has {n:,} rows, over 8% of the total; the fact layer gets squeezed")
        dim_total = sum(n for t, n in self.nrows.items() if self.roles.get(t) == ROLE_DIM)
        if dim_total > self.rows * 0.15:
            warn.append(f"dimensions total {dim_total:,} rows, over 15% of the total")

        if err:
            msg = "configuration pre-check failed:\n  " + "\n  ".join(f"x {e}" for e in err)
            if warn:
                msg += "\n  " + "\n  ".join(f"! {w}" for w in warn)
            if strict:
                raise ValueError(msg)
            print(msg)
        elif warn:
            print("configuration pre-check passed with %d warning(s):" % len(warn))
            for w in warn:
                print(f"  ! {w}")
        return err, warn

    # ---------------------------------------------------------------- report
    def report(self):
        if self.extra_tables != "none":
            print(
                f"! extra_tables='{self.extra_tables}': tables will be built beyond your DDL"
                f" ({'date dimension' if self.extra_tables == 'date_dim' else ''}"
                f"{'summary layer' if self.extra_tables == 'summary' else ''}"
                f"{'date dimension + summary layer' if self.extra_tables == 'all' else ''})"
            )
        print(
            f"data range {self.start} ~ {self.end} ({len(self.days)} days)"
            f" | target {self.rows:,} rows | main fact table {self.main_fact}"
        )
        nd_pk, nd_fk = len(self.decl_pk), len(self.decl_fk)
        if nd_pk or nd_fk:
            print(f"declared in the DDL: {nd_pk} primary key(s), {nd_fk} foreign key(s) (they win over inference)")
        print(f"{'table':<26}{'role':<14}{'rows':>10}  foreign keys")
        for t in sorted(self.schema, key=lambda x: -self.nrows.get(x, 0)):
            print(f"{t:<26}{self.roles[t]:<10}{self.nrows.get(t, 0):>10,}  {','.join(self.fks.get(t, [])) or '-'}")
        # These are pre-calibration figures and they do not add up to the target - generate() scales
        # them and re-runs toward it. Saying so costs one line; not saying it cost a production run
        # a long stretch of reasoning spent reconciling the sum. It is a target and not a promise:
        # calibration gets at most three attempts and can only move tables that are neither pinned
        # by ``table_rows`` nor fixed by their role, so a fully pinned schema lands where it lands
        # and generate() says so.
        planned = sum(self.nrows.get(t, 0) for t in self.schema)
        print(
            f"{'':<26}{'':<10}{planned:>10,}  planned total before calibration; generate() aims for "
            f"{self.rows:,} (+/-6%, up to 3 passes) and reports the deviation it reached"
        )
        if getattr(self, "ddl_enums", None):
            part = getattr(self, "_partial_enums", set())
            print(
                f"extracted value domains for {len(self.ddl_enums)} column(s) from DDL comments:"
                + " "
                + ", ".join(f"{k}({len(v)})" for k, v in list(self.ddl_enums.items())[:8])
                + (" ..." if len(self.ddl_enums) > 8 else "")
            )
            if part:
                print(
                    f"  {', '.join(sorted(part))} end in '...' (incomplete);"
                    f" list them fully in profile['enums'] if you need the whole domain"
                )
        self._print_semantics()
        self._print_plan()
        return self

    def _print_plan(self):
        """Print what the engine is about to do, not how it does it.

        A measured production run spent 66% of its wall clock reading this file - eight greps for
        `VOCAB`, `_joint_plan`, `_code_val`, `_enum_values`, `ROLE_METRIC`, `_cond_pick` and the
        constructor - because it was trying to predict the output before spending a generation pass.
        Every line below answers one of those greps directly. Showing the decision is far cheaper
        than making the caller reconstruct it from the implementation.
        """
        import random as _r

        p = self.profile
        print(
            f"knobs: months={self.months} ({len(self.days)} days, {self.start}~{self.end})  seed={self.seed}  "
            f"extra_tables={self.extra_tables!r}  "
            f"(override via DDLEngine(months=, end_date=, seed=, extra_tables=))"
        )

        # Sample names: answers VOCAB / _name_for / naming without reading either.
        rng = _r.Random(self.seed)
        named = []
        for t in sorted(self.schema):
            if self.roles.get(t) != ROLE_DIM:
                continue
            col = next((c["name"] for c in self.schema[t] if c["sem"] == "name"), None)
            if col:
                samples = ", ".join(self._name_for(t, i, rng) for i in range(2))
                named.append(f"{t}.{col} -> {samples}")
        if named:
            print("name samples (change with profile['vocab'] or profile['naming']):")
            for line in named[:8]:
                print(f"  {line}")

        # Generated business codes: answers CODE_COL / _code_val.
        codes = [
            f"{t}.{c['name']}" for t in sorted(self.schema) for c in self.schema[t] if self._is_code_col(t, c["name"])
        ]
        if codes:
            print(f"generated business codes: {', '.join(codes[:10])}{' ...' if len(codes) > 10 else ''}")

        # Daily metric grid: answers ROLE_METRIC / row allocation for those tables.
        for t, role in sorted(self.roles.items()):
            if role != ROLE_METRIC:
                continue
            planned, _ = self._metric_combos(t)
            budget = max(1, round(self.nrows[t] / max(1, len(self.days))))
            limited = (
                " (all the schema allows; the row budget had room for %d)" % budget if len(planned) < budget else ""
            )
            print(
                f"{t}: {len(self.days)} days x {len(planned)} dimension combos = "
                f"{len(self.days) * len(planned):,} rows{limited}"
                + ("" if limited else f" (raise profile['table_rows']['{t}'] for more combos)")
            )

        # Which declarative blocks actually resolved: answers _cond_pick / _joint_plan / derive.
        applied = []
        for key, spec in (p.get("conditional") or {}).items():
            if isinstance(spec, dict):
                groups = [k for k in spec if not k.startswith("__")]
                dflt = " + default" if "__default__" in spec else " (NO default: unlisted values use the engine's)"
                applied.append(f"conditional {key} by {spec.get('__by__')} ({len(groups)} groups{dflt})")
        for key, spec in (p.get("derive") or {}).items():
            ratio = (spec or {}).get("ratio")
            by = ratio.get("__by__") if isinstance(ratio, dict) else None
            applied.append(f"derive {key} from {(spec or {}).get('from')}" + (f" by {by}" if by else ""))
        for t, groups in (p.get("joint") or {}).items():
            for grp in groups if isinstance(groups, list) else []:
                applied.append(f"joint {t}({', '.join(grp.get('cols', []))}) {len(grp.get('values', []))} combos")
        if applied:
            print("declarative rules in effect:")
            for line in applied[:12]:
                print(f"  {line}")
            if len(applied) > 12:
                print(f"  ... and {len(applied) - 12} more")
        planned_sql = False
        for key in ("pre_sql", "extra_sql"):
            block = self._sql_block(key)
            if block.strip():
                planned_sql = True
                print(
                    f"{key}: {block.count(';')} statement(s) will run "
                    f"({'before' if key == 'pre_sql' else 'after'} the summary layer)"
                )
        if planned_sql:
            # Say that DuckDB has already planned them. Otherwise the only way to know a statement
            # is sound is to reason it through by hand, which one production run did at length.
            print("  every statement above was planned against the schema by precheck(); columns and types check out")

    def _print_semantics(self):
        """Print the inferred semantic of every column, so the caller can correct what is wrong.

        The regex table is a naming heuristic and naming is per-industry: it reads `paid_amount` as
        money but not `insurance_paid` or `copay`. Printing the whole mapping turns that into a cheap
        diff - the caller overrides the handful it disagrees with in profile["semantics"] instead of
        the engine carrying another industry's vocabulary.
        """
        ovr = getattr(self, "_sem_override", {})
        print("column semantics (override the wrong ones in profile['semantics']):")
        for t in sorted(self.schema):
            groups = {}
            for c in self.schema[t]:
                groups.setdefault(c["sem"], []).append(c["name"] + ("*" if (t, c["name"]) in ovr else ""))
            body = "  ".join(f"{sem}: {', '.join(cs)}" for sem, cs in sorted(groups.items()))
            print(f"  {t:<24}{body}")
        if ovr:
            print(f"  (* = set by profile['semantics'], {len(ovr)} column(s))")
        # A text column matching CODE_COL is not unrecognised - it gets a business code, and the
        # line below already says so. Listing it here too contradicted that line in the same report.
        unknown = [
            f"{t}.{c['name']}"
            for t in self.schema
            for c in self.schema[t]
            if c["sem"] == "text" and not self._is_code_col(t, c["name"])
        ]
        if unknown:
            print(f"  unrecognised, will be filled as free text: {unknown[:12]}{' ...' if len(unknown) > 12 else ''}")

    # ---------------------------------------------------------------- generation
    DEFAULT_VOCAB = {
        "brand": [
            "Lumora",
            "Nordvik",
            "Kaizen Field",
            "Volta Ridge",
            "Marisol",
            "Ferncrest",
            "Auralin",
            "Bastion Works",
            "Selva",
            "Northwind Lab",
            "Petrichor",
            "Okuda",
        ],
        "org_suffix": [
            "Flagship Store",
            "Overseas Store",
            "Select Store",
            "Direct Store",
            "Fulfilment Hub",
            "Service Centre",
        ],
        # Family and given names. `name_sep` joins them and `name_order` decides which comes
        # first, so a CJK vocabulary override ("", "family_first") reads naturally with no code change.
        "person": [
            "Miller",
            "Clarke",
            "Okafor",
            "Navarro",
            "Iversen",
            "Bianchi",
            "Haddad",
            "Petrova",
            "Silva",
            "Novak",
            "Reyes",
            "Aoki",
        ],
        "given": ["James", "Ada", "Noor", "Elias", "Mira", "Tomas", "Leah", "Ravi", "Sofia", "Kenji", "Anna", "Owen"],
        "name_sep": " ",
        "name_order": "given_first",
        "item": ["Standard", "Pro", "Lite", "Flagship", "Starter", "Plus"],
    }

    def _topo(self):
        order, seen = [], set()
        buckets = [ROLE_DATE, ROLE_DIM, ROLE_FACT, ROLE_DETAIL, ROLE_DOWNSTREAM, ROLE_EVENT, ROLE_SNAPSHOT, ROLE_METRIC]
        for role in buckets:
            group = [t for t, r in self.roles.items() if r == role]
            # dimensions can reference each other (product -> seller); order by dependency count
            group.sort(key=lambda t: len([f for f in self.fks[t] if self.pk_owner.get(f) in group]))
            for t in group:
                if t not in seen:
                    order.append(t)
                    seen.add(t)
        return order

    # ---------------------------------------------------------------- conditional distributions / column formulas
    def _fingerprint(self):
        """Configuration fingerprint: unchanged DDL and row-affecting parameters mean the last calibration can be reused."""
        import hashlib
        import json as _j

        key = _j.dumps(
            {
                "ddl": self.ddl,
                "rows": self.rows,
                "seed": self.seed,
                "start": self.start.isoformat(),
                "end": self.end.isoformat(),
                "extra": self.extra_tables,
                "table_rows": self.profile.get("table_rows"),
                "dim_rows": self.profile.get("dim_rows"),
                "dim_kinds": self.profile.get("dim_kinds"),
                "roles": self.profile.get("roles"),
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _load_calibration(self, out):
        """Reuse the previously calibrated row counts - when iterating, every run after the first is single-pass."""
        import json as _j

        try:
            f = Path(out).resolve()
            m = _j.loads((f.parent / f".{f.stem}.meta.json").read_text())
            if m.get("fingerprint") == self._fingerprint() and m.get("calibrated_rows"):
                cal = {k: v for k, v in m["calibrated_rows"].items() if k in self.nrows}
                if cal:
                    self.nrows.update(cal)
                    return True
        except Exception:
            pass
        return False

    def _pool_cum(self, tbl):
        """Cumulative weights of an entity pool, cached per table - rebuilding the list inside the loop is O(rows x pool)."""
        if not hasattr(self, "_cum_cache"):
            self._cum_cache = {}
        c = self._cum_cache.get(tbl)
        if c is None:
            pool = self.pools.get(tbl) or []
            c = self._cum_cache[tbl] = list(itertools.accumulate(e["__w__"] for e in pool))
        return c

    def _pick_enum(self, t, col, row, ents, rng):
        """Single entry point for enum values: consult conditional first (same-table or `upstream.column`
        grouping), otherwise sample the value domain by weight.

        Facts and dimensions always went through conditional; detail tables used to call _enum_values
        directly, so a conditional configured on a detail table silently did nothing. All three agree now.
        """
        spec = self._cond_spec(t, col)
        cw = self._cond_pick(spec, row, ents, t) if spec else None
        if isinstance(cw, dict) and cw:
            return rng.choices(list(cw), list(cw.values()))[0]
        vals, ws = self._enum_values(t, col)
        return rng.choices(vals, ws)[0]

    def _cond_spec(self, t, col):
        """profile["conditional"]["table.column"] = {"__by__": grouping column, value: params, "__default__": params}

        An enum column takes a weight dict (status mix differs by category); a numeric column takes [lo, hi].
        This is the configuration entry point for invariant 11 (business differentiation) - no SQL post-processing.
        """
        return self.profile.get("conditional", {}).get(f"{t}.{col}")

    def _resolve_by(self, t, by):
        """__by__ accepts two forms: `column` (same table) and `upstream_table.column` (across an FK).
        For the cross-table form it returns (local FK column, upstream column) and reads the value off that entity."""
        if "." not in by:
            return None, by
        up_t, up_c = by.split(".", 1)
        for f in self.fks.get(t, []):
            if self.pk_owner.get(f) == up_t:
                return f, up_c
        return None, None  # no FK path; the pre-check reports it

    def _cond_pick(self, spec, row, ents=None, t=None):
        by = spec.get("__by__")
        if not by:
            return spec.get("__default__")
        if "." in by and t:
            fk_col, up_col = self._resolve_by(t, by)
            e = (ents or {}).get(fk_col)
            key = e.get(up_col) if isinstance(e, dict) else None
        else:
            key = row.get(by)
        if key is not None and key in spec:
            return spec[key]
        return spec.get("__default__")

    def _formulas(self, t):
        """profile["formulas"]["table.column"] = "an arithmetic expression over other columns", evaluated in dependency order.

        "orders.paid_amount": "original_amount - discount_amount + shipping_amount + tax_amount"
        "order_items.total_cost": "unit_cost * quantity"
        """
        if not hasattr(self, "_fml_cache"):
            self._fml_cache = {}
        if t in self._fml_cache:
            return self._fml_cache[t]
        raw = {k.split(".", 1)[1]: v for k, v in self.profile.get("formulas", {}).items() if k.startswith(f"{t}.")}
        cols = {c["name"] for c in self.schema.get(t, [])}
        deps = {c: {w for w in re.findall(r"[A-Za-z_]\w*", expr) if w in cols and w != c} for c, expr in raw.items()}
        order, seen = [], set()

        def visit(c, trail=()):
            if c in seen:
                return
            if c in trail:
                raise ValueError(f"{t} has a circular formula dependency: {' -> '.join(trail + (c,))}")
            for d in deps.get(c, ()):
                if d in raw:
                    visit(d, trail + (c,))
            seen.add(c)
            order.append(c)

        for c in raw:
            visit(c)
        out = [(c, compile(raw[c], f"<formula {t}.{c}>", "eval")) for c in order]
        self._fml_cache[t] = out
        return out

    def _apply_formulas(self, t, row):
        for col, code in self._formulas(t):
            try:
                v = eval(code, {"__builtins__": {}, "min": min, "max": max, "abs": abs, "round": round}, row)
            except Exception:
                continue  # a dependency column is missing; keep the original value
            row[col] = round(v, 2) if isinstance(v, float) else v

    def _col_profile(self, t, c):
        return self.profile.get("columns", {}).get(f"{t}.{c}", {})

    def _enum_values(self, t, c):
        hit = self._enum_cache.get((t, c))
        if hit is not None:
            return hit
        v = self._enum_values_raw(t, c)
        self._enum_cache[(t, c)] = v
        return v

    def _enum_values_raw(self, t, c):
        p = self._col_profile(t, c)
        if "values" in p:
            vals = p["values"]
            w = p.get("weights") or [1] * len(vals)
            return vals, w
        g = self.profile.get("enums", {}).get(c)
        if g:
            return (g, [1] * len(g)) if isinstance(g, list) else (list(g), list(g.values()))
        d = getattr(self, "ddl_enums", {}).get(c)
        if d:  # domain written in a DDL comment; give it default Zipf weights
            return d, [1.0 / (i + 1) ** 0.6 for i in range(len(d))]
        n = 4 if c.endswith(("_status", "_type")) else 5
        return [f"{c.rsplit('_', 1)[0].upper()[:6]}{i + 1}" for i in range(n)], [1] * n

    def _person_name(self, vocab, rng):
        """Join a family and a given name. `name_sep` and `name_order` let a vocabulary override
        produce natural names in any script - CJK passes ("", "family_first") and needs no code change."""
        family, given = rng.choice(vocab["person"]), rng.choice(vocab["given"])
        sep = vocab.get("name_sep", " ")
        if vocab.get("name_order") == "family_first":
            return f"{family}{sep}{given}"
        return f"{given}{sep}{family}"

    def _name_for(self, t, i, rng, ent=None):
        p = self._col_profile(t, "__name__") or self.profile.get("naming", {}).get(t, {})
        # Three layers: the built-in vocabulary, a dataset-wide profile["vocab"] (one override for
        # every table - a hospital does not want retail brand names anywhere), then the per-table one.
        vocab = {**self.DEFAULT_VOCAB, **(self.profile.get("vocab") or {}), **p.get("vocab", {})}
        kind = self.profile.get("dim_kinds", {}).get(t) or self._guess_kind(t)
        if "tpl" in p:
            # Only list-valued entries are drawn from; name_sep / name_order are scalars.
            fields = {k: rng.choice(v) for k, v in vocab.items() if isinstance(v, (list, tuple))}
            fields.update({k: v for k, v in (ent or {}).items() if isinstance(v, str)})
            return p["tpl"].format(**fields, i=i + 1, n=i + 1)
        if kind in ("customer", "staff"):
            return self._person_name(vocab, rng)
        if kind == "item":
            return f"{rng.choice(vocab['brand'])} {rng.choice(vocab['item'])}"
        base = f"{rng.choice(vocab['brand'])} {rng.choice(vocab['org_suffix'])}"
        return base + (f" (#{i // len(vocab['brand']) + 1})" if i >= len(vocab["brand"]) else "")

    # ------------------------------------------------------------ small helpers
    def _is_int_col(self, t, col):
        d = next((c["type"].upper() for c in self.schema[t] if c["name"] == col), "")
        return any(d.startswith(x) for x in INT_T)

    def _pk_val(self, t, i, d=None):
        """Primary-key values honour the declared type: an integer PK gets integers, only VARCHAR gets a prefixed business code."""
        pk = self.pk_of(t)
        if self._is_int_col(t, pk):
            return self._id_base.setdefault(t, (len(self._id_base) + 1) * 10_000_000 + 1) + i
        prefix = re.sub(r"_id$|_key$|_no$", "", pk).upper()[:3] or "ENT"
        return f"{prefix}{d.strftime('%y%m%d')}{i + 1:07d}" if d is not None else f"{prefix}{i + 1:07d}"

    CODE_COL = re.compile(r"(^|_)(no|code|sn|serial|sku|number|barcode|ref)$")
    # Semantics whose own branch fills the column before the code fallback is reached, in both
    # ``_gen_dim``'s elif chain and ``_gen_fact``'s per-semantic buckets. A code only ever fills
    # what nothing more specific claimed.
    CODE_OWNED_SEM = ("name", "enum", "date", "ts", "amount", "count", "ratio", "flag", "measure")

    def _is_code_col(self, t, col):
        """Will this column actually be filled with a generated business code?

        ``CODE_COL`` alone is not the answer: it is the *last* branch both generators try, so a
        ``sku_code`` that inference classified as an enum gets enum values and never sees a code.
        Shared by the generators and ``report()`` so the two cannot disagree - a report that
        predicts a code where enum values land is worse than no report at all, because it is the
        surface the agent is told to trust instead of reading this file.
        """
        if col == self.pk_of(t) or not self.CODE_COL.search(col):
            return False
        sem = next((c["sem"] for c in self.schema[t] if c["name"] == col), None)
        if sem is None or sem in self.CODE_OWNED_SEM:
            return False
        # Joint groups are written into the row before the per-column chain runs at all, so this
        # test comes first: it holds whatever semantic the column carries, ``id`` included.
        if any(col in g.get("cols", ()) for g in (self.profile.get("joint", {}) or {}).get(t, [])):
            return False
        if sem == "id":
            # A foreign key is sampled from the parent pool; only a non-referencing id falls through.
            return self.pk_owner.get(col, t) == t
        return True

    def _code_val(self, t, col, i, d=None):
        pre = re.sub(r"[^A-Za-z]", "", col).upper()[:3] or "CD"
        return f"{pre}{d.strftime('%y%m%d')}{i + 1:06d}" if d is not None else f"{pre}{i + 1:06d}"

    def _joint_plan(self, t, n, rng):
        """Joint sampling: related enum columns in one row must be picked as a group (channel/source/campaign, province/city/tier).
        Sampling them independently creates combinations the business does not have, such as 'organic_search + TikTok ad'."""
        out = []
        for g in (self.profile.get("joint", {}) or {}).get(t, []):
            vals = [tuple(v[: len(g["cols"])]) for v in g["values"]]
            w = [float(v[len(g["cols"])]) if len(v) > len(g["cols"]) else 1.0 for v in g["values"]]
            out.append((g["cols"], rng.choices(vals, w, k=n)))
        return out

    AMT_ROLE = [
        ("refund", r"refund|chargeback|return"),
        ("cost", r"cost|cogs"),
        ("profit", r"profit|margin"),
        ("coupon", r"coupon|voucher"),
        ("discount", r"discount|promo|rebate|reduction|deduct"),
        ("ship", r"ship|freight|delivery|postage|logistic"),
        ("tax", r"tax|vat|duty"),
        ("paid", r"paid|net_|settle|actual|final|payable|received|pay_"),
        ("unit", r"unit|price"),
        ("gross", r"."),
    ]

    _ROLE_CACHE = {}

    @classmethod
    def _amt_role(cls, col):
        """Column name -> amount role. A pure function called hundreds of thousands of times, so it must be cached."""
        r = cls._ROLE_CACHE.get(col)
        if r is None:
            c = col.lower()
            r = cls._ROLE_CACHE[col] = next(x for x, pat in cls.AMT_ROLE if re.search(pat, c))
        return r

    def _settle_amounts(self, cols, base, rng, parts=None):
        """Generic accounting identities: paid = original - discount + shipping + tax; margin = revenue - cost.
        Shipping and tax are additive and not proportional to the goods amount - never randomise them at 0.6-1.0x."""
        parts, out = parts or {}, {}
        disc = parts.get("discount")
        if disc is None:
            disc = round(base * (rng.uniform(0.02, 0.22) if rng.random() < 0.62 else 0), 2)
        disc = min(disc, round(base * 0.9, 2))
        cost = parts.get("cost", round(base * rng.uniform(0.42, 0.66), 2))
        ship = 0.0 if rng.random() < 0.42 else round(rng.uniform(3, 22), 2)
        tax = round((base - disc) * rng.uniform(0, 0.085), 2)
        for c in cols:
            r = self._amt_role(c)
            if r == "gross" or r == "unit":
                out[c] = base
            elif r == "discount":
                out[c] = disc
            elif r == "coupon":
                out[c] = 0.0 if rng.random() < 0.58 else round(disc * rng.uniform(0.3, 0.85), 2)
            elif r == "ship":
                out[c] = ship
            elif r == "tax":
                out[c] = tax
            elif r == "cost":
                out[c] = cost
            elif r == "profit":
                out[c] = round(base - disc - cost, 2)
            elif r == "refund":
                out[c] = parts.get("refund", 0.0)
            elif r == "paid":
                out[c] = round(base - disc + ship + tax, 2)
        return out

    def _scratch_schema(self):
        """An empty in-memory copy of the schema this run will build, for planning SQL against.

        Declared tables keep their real CREATE text so constraints and exact types are the ones
        the statements will actually meet; synthetic tables (date dimension, summary layer) are
        rebuilt from the inferred column list. Foreign keys force an order, so creation retries
        once after everything else exists.
        """
        import duckdb

        con = duckdb.connect(":memory:")
        pending = []
        for t, cols in self.schema.items():
            sql = getattr(self, "decl_sql", {}).get(t)
            if not sql:
                body = ", ".join(f'"{c["name"]}" {c["type"]}' for c in cols)
                sql = f'CREATE TABLE "{t}" ({body})'
            try:
                con.execute(sql)
            except Exception:  # noqa: BLE001 - almost always a not-yet-created FK target
                pending.append(sql)
        for sql in pending:
            try:
                con.execute(sql)
            except Exception as e:  # noqa: BLE001 - a table we cannot build is one we cannot check
                logger_msg = str(e).splitlines()[0]
                print(f"  ! pre-check could not stage a table for SQL validation: {logger_msg}")
        return con

    def _validate_sql_blocks(self):
        """Plan every pre_sql / extra_sql statement against the empty schema.

        A mistake in these costs a whole generate-import-check cycle to discover, so a measured
        production run hand-verified them instead: 221,000 characters of reasoning in one turn,
        a third of it walking DuckDB's type rules (``hash(...) % 100 is UINT64, ::BIGINT safe``).
        DuckDB answers the same question in milliseconds, and EXPLAIN needs no data - only the
        schema, which is known before a single row exists.

        Returns a list of error lines, one per statement that cannot be planned.
        """
        blocks = [(k, self.profile.get(k)) for k in ("pre_sql", "extra_sql")]
        if not any(v for _, v in blocks):
            return []
        try:
            con = self._scratch_schema()
        except Exception as e:  # noqa: BLE001 - validation is a convenience, never a gate on generating
            print(f"  ! pre-check could not stage the schema for SQL validation: {str(e).splitlines()[0]}")
            return []

        problems = []
        try:
            for key, value in blocks:
                if not value:
                    continue
                if isinstance(value, (list, tuple)):
                    statements = [str(x).strip().rstrip(";") for x in value if str(x).strip()]
                else:
                    try:
                        statements = [st.query.strip().rstrip(";") for st in con.extract_statements(str(value))]
                    except Exception as e:  # noqa: BLE001 - a parse error IS the finding
                        problems.append(f"{key}: cannot be parsed as SQL - {str(e).splitlines()[0]}")
                        continue
                for i, stmt in enumerate(statements, 1):
                    if not stmt:
                        continue
                    # A statement that creates or drops something has to actually run, or the
                    # statements after it are planned against a schema that never existed.
                    mutates_schema = re.match(r"\s*(CREATE|DROP|ALTER)\b", stmt, re.I)
                    try:
                        con.execute(stmt if mutates_schema else "EXPLAIN " + stmt)
                    except Exception as e:  # noqa: BLE001 - this is what we came for
                        head = " ".join(stmt.split())[:80]
                        problems.append(f"{key}[{i}]: {str(e).splitlines()[0]}  <-  {head}")
        finally:
            con.close()
        return problems

    def _sql_block(self, key):
        """pre_sql / extra_sql may be one SQL string or a list of statements (easier to read and maintain).
        List elements get a semicolon appended and are concatenated, so a `list + str` type error cannot blow up mid-run."""
        v = self.profile.get(key) or ""
        if isinstance(v, (list, tuple)):
            v = "\n".join(s.rstrip().rstrip(";") + ";" for s in v if str(s).strip())
        elif not isinstance(v, str):
            raise TypeError(f"profile[{key!r}] must be a str or a list of str, got {type(v).__name__}")
        return (v + "\n") if v.strip() else ""

    def _derive(self, t, col, row, rng):
        """Invariant 2: derive a quantity from a base quantity. profile['derive']['table.column'] = {'from': base, 'ratio': (lo, hi)}"""
        d = (self.profile.get("derive", {}) or {}).get(f"{t}.{col}")
        if not d or d["from"] not in row:
            return None
        ratio = d.get("ratio", (0.2, 0.6))
        if isinstance(ratio, dict):  # different coefficients per group: {"__by__": column, value: (lo, hi)}
            ratio = self._cond_pick(ratio, row, None, t) or (0.2, 0.6)
        lo, hi = ratio
        v = float(row[d["from"]]) * rng.uniform(lo, hi)
        sem = next((c["sem"] for c in self.schema[t] if c["name"] == col), "count")
        return round(v, 2) if sem in ("amount", "ratio", "measure") else max(int(d.get("min", 0)), round(v))

    def _gen_date(self, t, o):
        cols = self.schema[t]
        rows = []
        for d in self.days:
            dt_cd, ev = self.cal.day_type(d)
            v = {
                "date_key": d.isoformat(),
                "year_num": d.year,
                "month_key": d.year * 100 + d.month,
                "quarter_cd": f"{d.year}-Q{(d.month - 1) // 3 + 1}",
                "week_of_year": d.isocalendar()[1],
                "day_of_week": d.weekday() + 1,
                "day_of_month": d.day,
                "is_weekend": int(d.weekday() >= 5),
                "is_workday": int(d.weekday() < 5),
                "day_type_cd": dt_cd,
                "event_name": ev,
                "stat_dt": d.isoformat(),
                "dt": d.isoformat(),
            }
            rows.append([v.get(c["name"], "") for c in cols])
        o.write(t, [c["name"] for c in cols], rows)
        self.pools[t] = None

    def _gen_dim(self, t, o):
        rng, cols = self.rng, self.schema[t]
        n = self.nrows[t]
        # A Top-10% target does not apply to small cardinalities: with 21 products the top 10% is
        # 2 rows, and forcing those 2 to hold 74% guarantees an extreme head (measured: #1 took 52%).
        # The smaller the cardinality, the flatter the target has to be.
        target = self.profile.get("head_share", {}).get(t, 0.74)
        if n < 200:
            target = min(target, 0.30 + n * 0.0022)  # n=21 -> 0.35, n=103 -> 0.53, n=200 -> 0.74
        alpha = tune_alpha(n, target) if n >= 12 else 1.0
        w = zipf_weights(n, alpha, shuffle_with=rng) if n >= 12 else [1.0] * n
        pk = self.pk_of(t)  # declared PK wins; never assume it is the first column
        joint = self._joint_plan(t, n, rng)
        eff_c = self._eff_col(t)
        # Audit timestamps (created_at/updated_at) are not independent business dates; they are aligned to the effective date at the end
        audit_cs = [
            c["name"] for c in cols if c["name"] != pk if c["sem"] in ("date", "ts") and AUDIT_TS.match(c["name"])
        ]
        rows, pool = [], []
        for i in range(n):
            ent = {pk: self._pk_val(t, i), "__w__": w[i], "__i__": i}
            for jcols, jvals in joint:
                ent.update(dict(zip(jcols, jvals[i])))
            for c in sorted((x for x in cols if x["name"] != pk), key=lambda x: x["sem"] == "name"):
                name, sem = c["name"], c["sem"]
                if name in ent:
                    continue
                if sem == "id" and name in self.pk_owner and self.pk_owner[name] != t:
                    up = self.pools.get(self.pk_owner[name])
                    ent[name] = (
                        rng.choices([e[self.pk_of(self.pk_owner[name])] for e in up], [e["__w__"] ** 0.55 for e in up])[
                            0
                        ]
                        if up
                        else ""
                    )
                elif sem == "name":
                    ent[name] = self._name_for(t, i, rng, ent)
                elif sem == "enum":
                    # conditional applies to dimensions too (category -> status mix, region -> membership mix)
                    cw = self._cond_pick(self._cond_spec(t, name) or {}, ent)
                    if isinstance(cw, dict) and cw:
                        ent[name] = rng.choices(list(cw), list(cw.values()))[0]
                    else:
                        vals, ws = self._enum_values(t, name)
                        ent[name] = rng.choices(vals, ws)[0]
                elif sem in ("date", "ts"):
                    # Invariant 16: entities cannot all be new in-range; 45% are stock existing before the range by default
                    cp = self._col_profile(t, name)
                    lo, hi = cp.get("before_start", (1, 900))
                    if rng.random() < cp.get("baseline_share", 0.45):
                        d0 = self.start - timedelta(days=rng.randint(lo, hi))
                    else:
                        d0 = rng.choices(self.days, self.day_w)[0]
                    ent[name] = d0.isoformat() if sem == "date" else day_ts(rng, d0).strftime("%Y-%m-%d %H:%M:%S")
                elif sem == "amount":
                    lo, hi = self._col_profile(t, name).get("range", (8, 400))
                    cr = self._cond_pick(self._cond_spec(t, name) or {}, ent)
                    if isinstance(cr, (list, tuple)) and len(cr) == 2:
                        lo, hi = cr  # conditional: amount range banded by the grouping column
                    ent[name] = round(lognorm_between(rng, lo, hi), 2)
                elif sem == "count":
                    lo, hi = self._col_profile(t, name).get("range", (1, 60))
                    cr = self._cond_pick(self._cond_spec(t, name) or {}, ent)
                    if isinstance(cr, (list, tuple)) and len(cr) == 2:
                        lo, hi = cr
                    ent[name] = max(lo, int(lognorm_between(rng, max(1, lo), hi)))
                elif sem == "ratio":
                    ent[name] = round(bounded_gauss(rng, 0.8, 0.12, 0.4, 0.99), 3)
                elif sem == "flag":
                    ent[name] = 1 if rng.random() < self._col_profile(t, name).get("p", 0.93) else 0
                elif sem == "measure":
                    ent[name] = round(lognorm_between(rng, 0.1, 50), 3)
                elif self._is_code_col(t, name):
                    ent[name] = self._code_val(t, name, i)
                else:
                    ent[name] = f"{name}_{i + 1}"
            # Effective-date index, so facts/details referencing this entity can enforce "not before it" (invariant 3)
            ent["__eff__"] = self._day_index(ent[eff_c]) if (eff_c and ent.get(eff_c)) else 0
            ent["__eff_ts__"] = None
            if eff_c and ent.get(eff_c) and len(str(ent[eff_c])) > 10:
                try:
                    ent["__eff_ts__"] = datetime.fromisoformat(str(ent[eff_c])[:19])
                except ValueError:
                    pass
            for ac in audit_cs:  # audit columns align to the effective date instead of being random
                anchor = ent.get(eff_c) if eff_c else None
                if not anchor:
                    continue
                sem_a = next(c["sem"] for c in cols if c["name"] == ac)
                if sem_a == "date":
                    ent[ac] = str(anchor)[:10]
                else:
                    ent[ac] = (
                        str(anchor)
                        if len(str(anchor)) > 10
                        else day_ts(rng, date.fromisoformat(str(anchor)[:10])).strftime("%Y-%m-%d %H:%M:%S")
                    )
            pool.append(ent)
            rows.append([ent[c["name"]] for c in cols])
        # A cost column must never exceed the price column (derived-quantity constraint)
        price_c = next((c["name"] for c in cols if re.search(r"price|list", c["name"])), None)
        cost_c = next((c["name"] for c in cols if "cost" in c["name"]), None)
        if price_c and cost_c:
            # The cost ratio can be grouped via conditional (margin differs by category). An upper
            # bound <=1 is read as a cost ratio; >1 as an absolute range clamped to the price.
            # Unset, the default .38-.68 applies.
            cidx = [c["name"] for c in cols].index(cost_c)
            dflt = self._col_profile(t, cost_c).get("cost_ratio", (0.38, 0.68))
            cspec = self._cond_spec(t, cost_c)
            for e, r in zip(pool, rows):
                cr = self._cond_pick(cspec, e) if cspec else None
                lo, hi = cr if (isinstance(cr, (list, tuple)) and len(cr) == 2) else dflt
                e[cost_c] = (
                    round(e[price_c] * rng.uniform(lo, hi), 2)
                    if hi <= 1
                    else min(round(lognorm_between(rng, lo, hi), 2), round(e[price_c] * 0.95, 2))
                )
                r[cidx] = e[cost_c]
        self.pools[t] = pool
        self._pending_dim_rows[t] = (([c["name"] for c in cols]), rows)
        o.write(t, [c["name"] for c in cols], rows)

    def pk_of(self, t):
        d = getattr(self, "decl_pk", {}).get(t)
        if d and len(d) == 1:
            return d[0]  # a declared PK beats the first-column guess
        return self.schema[t][0]["name"]

    # ---------------------------------------------------- effective date (generic implementation of invariant 3)
    def _eff_col(self, t):
        """Name of a dimension's effective-date column: registered / listed / hired. None when absent."""
        if t in getattr(self, "_eff_cache", {}):
            return self._eff_cache[t]
        self._eff_cache = getattr(self, "_eff_cache", {})
        ov = (self.profile.get("effective_col", {}) or {}).get(t, "__auto__")
        if ov != "__auto__":
            self._eff_cache[t] = ov
            return ov
        c = next(
            (
                c["name"]
                for c in self.schema[t]
                if c["sem"] in ("date", "ts") and not AUDIT_TS.match(c["name"]) and EFFECTIVE_DATE.search(c["name"])
            ),
            None,
        )
        self._eff_cache[t] = c
        return c

    def _day_index(self, v):
        """Convert an entity's effective date to an index into days; 0 when before the range start (no constraint) or unparsable."""
        try:
            d = date.fromisoformat(str(v)[:10])
        except Exception:
            return 0
        return min(max((d - self.start).days, 0), len(self.days) - 1)

    def _pick_day_ge(self, lo_idx, rng):
        """Sample one day from days[lo_idx:] by calendar weight (bisect, O(log n))."""
        total = self._cum_w[-1]
        base = self._cum_w[lo_idx - 1] if lo_idx else 0.0
        if base >= total:
            return len(self.days) - 1
        return min(bisect.bisect_left(self._cum_w, rng.uniform(base, total)), len(self.days) - 1)

    def _pick_subject(self, t):
        """Subject dimension: the lowest-cardinality foreign key (seller/store/warehouse); it shapes the long tail."""
        cands = [
            (f, self.pk_owner[f])
            for f in self.fks[t]
            if self.pk_owner.get(f) in self.pools and self.pools[self.pk_owner[f]]
        ]
        if not cands:
            return None, None
        return min(cands, key=lambda x: len(self.pools[x[1]]))

    def _gen_fact(self, t, o, parent=None):
        rng, cols = self.rng, self.schema[t]
        n = self.nrows[t]
        names = [c["name"] for c in cols]
        pk = self.pk_of(t)  # declared PK wins; never assume it is the first column
        date_cols = [c["name"] for c in cols if c["sem"] == "date"]
        ts_cols = [c["name"] for c in cols if c["sem"] == "ts"]
        biz_ts = [c for c in ts_cols if not AUDIT_TS.match(c)]
        audit_ts = [c for c in ts_cols if AUDIT_TS.match(c)]
        lc = (self.profile.get("lifecycle", {}) or {}).get(t, {})
        gaps = lc.get("gap_hours") or [0] + [1.5 * 4**j for j in range(len(biz_ts))]
        stages = lc.get("stages", {})
        # In-flight statuses (mid-lifecycle) can only occur near the cut-off date -
        # an order from a year ago cannot still be pending/shipped
        inflight = set(lc.get("in_flight", []))
        span_h = sum(gaps[: max(1, len(biz_ts))])
        hard_cap = datetime.combine(self.end, datetime.min.time()) + timedelta(hours=23, minutes=59, seconds=59)
        amt_cols = [c["name"] for c in cols if c["sem"] == "amount"]
        # Measures (downtime_minutes, weight_kg) and names on a fact table have no dedicated branch
        # below; without this they fall through to setdefault(c, "") and the whole column lands NULL.
        other_cols = [c for c in cols if c["sem"] in ("measure", "name")]
        cnt_cols = [c["name"] for c in cols if c["sem"] == "count"]
        enum_cols = [c["name"] for c in cols if c["sem"] == "enum"]
        flag_cols = [c["name"] for c in cols if c["sem"] == "flag"]
        ratio_cols = [c["name"] for c in cols if c["sem"] == "ratio"]

        subj_col, subj_tbl = self._pick_subject(t)
        days = rng.choices(self.days, self.day_w, k=n)
        fk_samples, fk_pools = {}, {}
        for f in self.fks[t]:
            up = self.pools.get(self.pk_owner[f])
            if not up:
                continue
            if any(e.get("__eff__") for e in up):
                # Upstream has an effective date: sort by it, build cumulative weights, and sample only from the already-effective prefix
                ps = sorted(up, key=lambda e: e.get("__eff__") or 0)
                fk_pools[f] = (
                    ps,
                    [e.get("__eff__") or 0 for e in ps],
                    list(itertools.accumulate(e["__w__"] for e in ps)),
                )
            else:
                fk_samples[f] = rng.choices(up, [e["__w__"] for e in up], k=n)
        status_col = next((c for c in enum_cols if "status" in c), None)
        st_vals, st_w = self._enum_values(t, status_col) if status_col else ([], [])
        joint = self._joint_plan(t, n, rng)

        rows, refs = [], []
        for i in range(n):
            d = days[i]
            ent = {f: fk_samples[f][i] for f in fk_samples}
            # Invariant 3: only reference upstream entities already effective that day (no order before registration or before listing)
            for f, (ps, es, cw) in fk_pools.items():
                ent[f] = self._pick_items(ps, es, cw, d, 1, rng)[0]
            lo_i = max((e.get("__eff__") or 0) for e in ent.values()) if ent else 0
            if lo_i and (d - self.start).days < lo_i:  # fallback for a day with no effective entity at all
                d = self.days[self._pick_day_ge(lo_i, rng)]
            row = {pk: self._pk_val(t, i, d)}
            for f, e in ent.items():
                row[f] = e[self.pk_of(self.pk_owner[f])]
            for jcols, jvals in joint:
                row.update(dict(zip(jcols, jvals[i])))
            for c in date_cols:
                row[c] = d.isoformat()
            for c in enum_cols:
                if c in row:
                    continue
                inherited = next((e[c] for e in ent.values() if c in e), None)
                if inherited is not None:
                    row[c] = inherited
                else:
                    spec = self._cond_spec(t, c)
                    cw = self._cond_pick(spec, row, ent, t) if spec else None
                    if isinstance(cw, dict) and cw:
                        row[c] = rng.choices(list(cw), list(cw.values()))[0]
                    else:
                        vals, ws = self._enum_values(t, c)
                        row[c] = rng.choices(vals, ws)[0]
            # Lifecycle timestamps (invariants 3/4): multiple time columns on one row must increase
            # monotonically and be truncated by status; audit columns such as created_at stay out of
            # the chain and align to the first business timestamp.
            t0 = day_ts(rng, d)
            if inflight and status_col and row.get(status_col) in inflight and t0 + timedelta(hours=span_h) <= hard_cap:
                pool_v = [(v, w) for v, w in zip(st_vals, st_w) if v not in inflight]
                if pool_v:
                    row[status_col] = rng.choices([v for v, _ in pool_v], [w for _, w in pool_v])[0]
            ets = max((e["__eff_ts__"] for e in ent.values() if e.get("__eff_ts__")), default=None)
            if ets is not None and t0 < ets:  # even same-day, it cannot precede the exact registration/listing moment
                t0 = min(ets + timedelta(minutes=rng.randint(2, 720)), hard_cap)
            cur, kmax = t0, stages.get(str(row.get(status_col, "")), len(biz_ts))
            reached = 0
            for j, c in enumerate(biz_ts):
                if j >= kmax:
                    row[c] = ""
                    continue
                if j:
                    cur = cur + timedelta(hours=max(0.05, gaps[min(j, len(gaps) - 1)] * rng.uniform(0.35, 1.9)))
                if cur <= hard_cap:
                    row[c] = cur.strftime("%Y-%m-%d %H:%M:%S")
                    reached = j + 1
                else:
                    row[c] = ""
            if stages and status_col and reached < kmax:
                back = next((k for k, v in stages.items() if v == reached), None)
                if back is not None:
                    row[status_col] = back
            for c in audit_ts:
                row[c] = t0.strftime("%Y-%m-%d %H:%M:%S")
            # Disruption filter: inside the window with factor<1, drop the row probabilistically
            # (suppressing anomaly). This must run after enum/joint assignment, otherwise scope can
            # only match FK dimension attributes and never the fact's own channel/source/site columns.
            attrs = {}
            for f, e in ent.items():
                for k, v in e.items():
                    if isinstance(v, str) and not k.startswith("__"):
                        attrs.setdefault(k, v)  # real column names, matching the profile scope contract
            for k, v in row.items():
                if isinstance(v, str):
                    attrs[k.replace("_cd", "")] = v
            fct = self.cal.disrupt_factor(d, **attrs)
            if fct < 1 and rng.random() > fct:
                continue
            base = 0.0
            if amt_cols:
                lo, hi = self._col_profile(t, amt_cols[0]).get("range", (12, 900))
                cr = self._cond_pick(self._cond_spec(t, amt_cols[0]) or {}, row, ent, t)
                if isinstance(cr, (list, tuple)) and len(cr) == 2:
                    lo, hi = cr  # conditional: amount range banded by the grouping column
                base = round(lognorm_between(rng, lo, hi), 2)
                row.update(self._settle_amounts(amt_cols, base, rng))
            for c in cnt_cols:
                row[c] = self._derive(t, c, row, rng) or rng.choices([1, 2, 3, 4, 5], [0.52, 0.26, 0.12, 0.06, 0.04])[0]
            for c in ratio_cols:
                row[c] = round(bounded_gauss(rng, 0.04, 0.013, 0.008, 0.092), 4)
            for c in flag_cols:
                row[c] = 1 if rng.random() < self._col_profile(t, c).get("p", 0.88) else 0
            for c in other_cols:
                row.setdefault(c["name"], self._fill_generic(t, c, rng, {"dt": d, "ts": t0}))
            for c in names:
                if c not in row and self._is_code_col(t, c):
                    row[c] = self._code_val(t, c, i, d)
                row.setdefault(c, "")
            self._apply_formulas(t, row)
            rows.append([row[c] for c in names])
            refs.append(
                {
                    "pk": row[pk],
                    "dt": d,
                    "ts": t0,
                    "status": row.get(status_col, ""),
                    "amt": base,
                    "fks": {f: row[f] for f in self.fks[t]},
                    "subj": ent.get(subj_col, {}).get("__w__", 1.0) if subj_col else 1.0,
                }
            )
        o.write(t, names, rows)
        self.refs[t] = refs
        self._fact_rows[t] = (names, rows)

    @staticmethod
    def _amt_ratio(col, rng, status=None):
        if "refund" in col:
            return rng.uniform(0.15, 0.9) if rng.random() < 0.12 else 0.0
        if "promo" in col or "discount" in col or "tax" in col or "fee" in col:
            return rng.uniform(0.01, 0.12)
        if "cost" in col:
            return rng.uniform(0.38, 0.68)
        if "profit" in col:
            return rng.uniform(0.12, 0.45)
        if "paid" in col or "net" in col or "settle" in col:
            return rng.uniform(0.88, 1.0)
        return rng.uniform(0.6, 1.0)

    # ---------------------------------------------------------------- detail / downstream / event
    def _parent_of(self, t):
        for f in self.fks[t]:
            par = self.pk_owner.get(f)
            if par and par in self.refs:
                return f, par
        return None, None

    def _gen_detail(self, t, o):
        """Detail rows inherit the parent date and keep amounts self-consistent within the row
        (price x qty = line amount, cost from the product cost, margin = revenue - cost), then are
        summed back into the parent per document (invariant 2 / zero header-detail drift)."""
        rng, cols = self.rng, self.schema[t]
        names = [c["name"] for c in cols]
        pk, (fk, par) = self.pk_of(t), self._parent_of(t)
        if not par:
            return self._gen_fact(t, o)
        prefs = self.refs[par]
        n_per = max(1, round(self.nrows[t] / max(1, len(prefs))))
        item_col = next(
            (
                c["name"]
                for c in cols
                if c["name"] != pk
                and c["sem"] == "id"
                and self.pk_owner.get(c["name"]) in self.pools
                and self.pools[self.pk_owner[c["name"]]]
            ),
            None,
        )
        pool = self.pools[self.pk_owner[item_col]] if item_col else None
        pool_pk = self.pk_of(self.pk_owner[item_col]) if item_col else None
        # Sort products by effective (listing) date and pre-build cumulative weights - a detail row may only reference an entity effective by then
        pool_cum = list(itertools.accumulate(e["__w__"] for e in pool)) if pool else None
        eff_sorted, eff_cum = [], []
        if pool and any(e.get("__eff__") for e in pool):
            pool = sorted(pool, key=lambda e: e.get("__eff__") or 0)
            eff_sorted = [e.get("__eff__") or 0 for e in pool]
            eff_cum = list(itertools.accumulate(e["__w__"] for e in pool))
            pool_cum = eff_cum  # cumulative weights follow the sorted pool
        p_cost = p_list = None
        if pool:
            pcols = [c["name"] for c in self.schema[self.pk_owner[item_col]] if c["sem"] == "amount"]
            p_cost = next((c for c in pcols if re.search(r"cost", c)), None)
            p_list = next((c for c in pcols if c != p_cost), None)
        amt_cols = [c["name"] for c in cols if c["sem"] == "amount"]
        unit_col = next((c for c in amt_cols if self._amt_role(c) == "unit"), None)
        cnt_cols = [c["name"] for c in cols if c["sem"] == "count"]
        qty_col = next((c for c in cnt_cols if not re.search(r"refund|return", c)), None)
        rfq_col = next((c for c in cnt_cols if re.search(r"refund|return", c)), None)
        reason_cols = [c["name"] for c in cols if re.search(r"reason|cause", c["name"])]
        refund_p = self.profile.get("refund_rate", 0.055)
        rows, agg, no = [], {}, 0
        for pi, pr in enumerate(prefs):
            k = max(1, min(6, rng.choices([1, 2, 3, 4, 5], [0.52, 0.26, 0.12, 0.06, 0.04])[0] if n_per <= 2 else n_per))
            if pool and eff_cum:
                picks = self._pick_items(pool, eff_sorted, eff_cum, pr["dt"], k, rng)
            else:
                picks = rng.choices(pool, cum_weights=pool_cum, k=k) if pool else [None] * k
            if pool:  # one entity may appear only once per parent document (otherwise the same SKU appears twice at different prices)
                seen, uniq = set(), []
                for e in picks:
                    if e[pool_pk] in seen:
                        continue
                    seen.add(e[pool_pk])
                    uniq.append(e)
                picks = uniq
            st = str(pr["status"]).upper()
            dead = bool(re.search(r"CANCEL|PENDING|FAIL", st))
            back = bool(re.search(r"REFUND|RETURN", st))
            tot = {"gross": 0.0, "net": 0.0, "discount": 0.0, "cost": 0.0, "refund": 0.0}
            for j, e in enumerate(picks):
                no += 1
                row = {pk: self._pk_val(t, no - 1), fk: pr["pk"]}
                ents = {item_col: e} if (item_col and e) else None
                if item_col and e:
                    row[item_col] = e[pool_pk]
                qty = rng.choices([1, 2, 3, 4], [0.71, 0.19, 0.07, 0.03])[0]
                lp = float(e[p_list]) if (e and p_list) else round(lognorm_between(rng, 9, 320), 2)
                drate = rng.uniform(0.05, 0.42) if rng.random() < 0.58 else 0.0
                up = round(lp * (1 - drate), 2)
                # conditional constrains the value actually written, then the list price is back-solved to keep the discount relation
                cr = self._cond_pick(self._cond_spec(t, unit_col) or {}, row, ents, t) if unit_col else None
                if isinstance(cr, (list, tuple)) and len(cr) == 2:
                    up = round(lognorm_between(rng, cr[0], cr[1]), 2)
                    lp = round(up / (1 - drate), 2) if drate < 0.99 else up
                uc = float(e[p_cost]) if (e and p_cost) else round(lp * rng.uniform(0.42, 0.62), 2)
                sales, disc, tcost = round(up * qty, 2), round((lp - up) * qty, 2), round(uc * qty, 2)
                rq = 0 if dead else (rng.randint(1, qty) if rng.random() < (0.82 if back else refund_p) else 0)
                ramt = round(up * rq, 2)
                for c in cols:
                    if c["name"] == pk:
                        continue
                    nm, sem = c["name"], c["sem"]
                    if nm in row:
                        continue
                    if sem == "amount":
                        role = self._amt_role(nm)
                        row[nm] = {
                            "unit": up,
                            "cost": uc if re.search(r"unit|avg", nm) else tcost,
                            "discount": disc,
                            "coupon": 0.0,
                            "ship": 0.0,
                            "tax": 0.0,
                            "profit": round(sales - tcost, 2),
                            "refund": ramt,
                            "gross": sales,
                            "paid": sales,
                        }[role]
                        if re.search(r"list|msrp|tag", nm):
                            row[nm] = lp
                    elif nm == qty_col:
                        row[nm] = qty
                    elif nm == rfq_col:
                        row[nm] = rq
                    elif nm in reason_cols:
                        row[nm] = self._pick_enum(t, nm, row, ents, rng) if rq else ""
                    elif sem == "enum" and self._cond_spec(t, nm):
                        row[nm] = self._pick_enum(t, nm, row, ents, rng)
                    elif item_col and e and nm in e and not nm.startswith("__") and sem not in ("ts", "date"):
                        row[nm] = e[nm]  # inherit product attributes, but never time columns:
                        # a detail row's time comes from the parent fact only (else created_at becomes the listing date)
                    else:
                        row[nm] = self._fill_generic(t, c, rng, pr)
                tot["gross"] += (
                    sales + disc
                )  # pre-discount goods amount (list price x qty), feeds the parent "original" column
                tot["net"] += sales  # post-discount goods amount = original - discount
                tot["discount"] += disc
                tot["cost"] += tcost
                tot["refund"] += ramt
                self._apply_formulas(t, row)
                rows.append([row[c] for c in names])
            agg[pr["pk"]] = {k2: round(v, 2) for k2, v in tot.items()}
        o.write(t, names, rows)
        self._realign_parent(par, agg)

    def _pick_items(self, pool, eff_sorted, eff_cum, d, k, rng):
        """Sample k entities by popularity from the "effective date <= d" prefix (bisect; no O(n) filtering in the loop, invariant 17)."""
        di = min(max((d - self.start).days, 0), len(self.days) - 1)
        hi = bisect.bisect_right(eff_sorted, di)
        if hi <= 0:  # nothing effective that day; fall back to the whole pool
            hi = len(pool)
        top = eff_cum[hi - 1]
        if top <= 0:
            return rng.choices(pool[:hi], k=k)
        return [pool[min(bisect.bisect_left(eff_cum, rng.uniform(0, top)), hi - 1)] for _ in range(k)]

    def _realign_parent(self, par, agg):
        """Backfill the parent fact's amount columns from the detail totals - zero header/detail drift and a self-consistent accounting identity."""
        names, rows = self._fact_rows[par]
        amt_cols = [c["name"] for c in self.schema[par] if c["sem"] == "amount"]
        if not amt_cols or not agg:
            return
        idx = {c: names.index(c) for c in amt_cols}
        rng = self.rng
        # When the parent has a discount column, its "original" column must be the pre-discount
        # amount: original - discount = post-discount goods amount. Without a discount column the
        # single amount column should be the post-discount amount actually received.
        has_disc = any(self._amt_role(c) in ("discount", "coupon") for c in amt_cols)
        for r in rows:
            parts = agg.get(r[0])
            if parts is None:
                continue
            base = parts["gross"] if has_disc else parts.get("net", parts["gross"])
            for c, v in self._settle_amounts(amt_cols, base, rng, parts).items():
                r[idx[c]] = v
            if self._formulas(par):  # formulas outrank the default accounting rules; recompute after backfill
                d = dict(zip(names, r))
                self._apply_formulas(par, d)
                for k, v in d.items():
                    r[names.index(k)] = v
        self._csv.write(par, names, rows)  # rewrite the parent table in place
        for ref, r in zip(self.refs[par], rows):
            ref["amt"] = r[idx[amt_cols[0]]]

    def _fill_generic(self, t, c, rng, pr=None):
        sem, name = c["sem"], c["name"]
        if sem == "enum":
            vals, ws = self._enum_values(t, name)
            return rng.choices(vals, ws)[0]
        if sem == "count":
            return rng.randint(1, 20)
        if sem == "ratio":
            return round(bounded_gauss(rng, 0.04, 0.013, 0.008, 0.092), 4)
        if sem == "flag":
            return 1 if rng.random() > 0.12 else 0
        if sem == "amount":
            return round(lognorm_between(rng, 5, 300), 2)
        if sem == "measure":
            return round(lognorm_between(rng, 0.1, 40), 3)
        if sem == "ts":
            # Invariant 3: a child timestamp is the parent moment plus a non-negative offset, clamped
            # to the cut-off. "A random moment on the same day" is not enough - about half the detail
            # rows would land before the order was created.
            anchor = (pr or {}).get("ts")
            if anchor is not None:
                cap = datetime.combine(self.end, datetime.min.time()) + timedelta(hours=23, minutes=59, seconds=59)
                return min(anchor + timedelta(seconds=rng.randint(30, 5400)), cap).strftime("%Y-%m-%d %H:%M:%S")
            return day_ts(rng, pr["dt"] if pr else rng.choice(self.days)).strftime("%Y-%m-%d %H:%M:%S")
        if sem == "date":
            return (pr["dt"] if pr else rng.choice(self.days)).isoformat()
        if sem == "name":
            return self._name_for(t, rng.randint(0, 99), rng)
        if self._is_code_col(t, name):
            # _gen_dim and _gen_fact call _code_val directly; the detail / downstream / event /
            # metric generators reach a column only through here, so without this branch a
            # code column on any of them landed NULL while report() promised a business code.
            # The counter is per (table, column) rather than a loop index: these generators nest
            # loops (a detail row per parent, an event row per stage), and a reused index would
            # hand out duplicate codes.
            seq = self._code_seq[(t, name)] = self._code_seq.get((t, name), 0) + 1
            return self._code_val(t, name, seq - 1, (pr or {}).get("dt"))
        return ""

    def _gen_downstream(self, t, o):
        """Downstream facts (shipment/claim/repayment): dated after the parent, status derived from the facts, promise dates may be in the future."""
        rng, cols = self.rng, self.schema[t]
        names = [c["name"] for c in cols]
        pk, (fk, par) = self.pk_of(t), self._parent_of(t)
        if not par:
            return self._gen_fact(t, o)

        prefs = self.refs[par]
        keep = [p for p in prefs if not p["status"] or "CANCEL" not in str(p["status"]).upper()]
        n = min(self.nrows[t], len(keep))
        picks = rng.sample(keep, n) if n < len(keep) else keep
        date_cols = [c["name"] for c in cols if c["sem"] == "date"]
        start_c = next(
            (c for c in date_cols if re.search(r"ship|start|open|begin|create", c)), date_cols[0] if date_cols else None
        )
        promise_c = next((c for c in date_cols if re.search(r"promise|due|expect|sla", c)), None)
        end_c = next((c for c in date_cols if re.search(r"deliver|end|close|finish|complete|settle", c)), None)
        status_col = next((c["name"] for c in cols if c["sem"] == "enum" and "status" in c["name"]), None)
        ontime_col = next(
            (c["name"] for c in cols if c["sem"] == "flag" and re.search(r"on_time|ontime|success|is_ok", c["name"])),
            None,
        )
        rows, refs = [], []
        for i, pr in enumerate(picks):
            row = {pk: self._pk_val(t, i), fk: pr["pk"]}
            for f in self.fks[t]:
                if f == fk or f not in self.pk_owner:
                    continue
                up = self.pools.get(self.pk_owner[f])
                if up:
                    e = rng.choices(up, cum_weights=self._pool_cum(self.pk_owner[f]))[0]
                    row[f] = e[self.pk_of(self.pk_owner[f])]
                    row["__sla__"] = next((v for k, v in e.items() if "sla" in k and isinstance(v, (int, float))), 6)
            sla = int(row.pop("__sla__", 6)) or 6
            s_dt = pr["dt"] + timedelta(days=rng.choices([0, 1, 2, 3], [0.42, 0.34, 0.18, 0.06])[0])
            if s_dt > self.end:
                continue
            attrs = {k.replace("_cd", ""): v for k, v in row.items() if isinstance(v, str)}
            f_dis = self.cal.disrupt_factor(s_dt, **attrs)
            transit = max(1, round(sla * rng.lognormvariate(-0.24, 0.34) * (f_dis**0.72)))
            e_dt = s_dt + timedelta(days=transit)
            done = e_dt <= self.end and rng.random() > 0.004 * f_dis
            if start_c:
                row[start_c] = s_dt.isoformat()
            if promise_c:
                row[promise_c] = (s_dt + timedelta(days=sla)).isoformat()
            if end_c:
                row[end_c] = e_dt.isoformat() if done else ""
            for c in date_cols:
                row.setdefault(c, s_dt.isoformat())
            if status_col:
                vals, ws = self._enum_values(t, status_col)
                done_v = next(
                    (v for v in vals if re.search(r"DELIVER|DONE|COMPLETE|CLOSED|SUCCESS", str(v).upper())), vals[0]
                )
                open_v = next((v for v in vals if re.search(r"TRANSIT|OPEN|PROCESS|PENDING", str(v).upper())), vals[-1])
                row[status_col] = done_v if done else open_v
            if ontime_col:
                row[ontime_col] = (
                    (1 if (done and promise_c and e_dt <= s_dt + timedelta(days=sla)) else 0) if done else ""
                )
            for c in cols:
                if c["name"] in row:
                    continue
                row[c["name"]] = self._fill_generic(t, c, rng, {"dt": s_dt})
            self._apply_formulas(t, row)
            rows.append([row.get(c, "") for c in names])
            refs.append(
                {
                    "pk": row[pk],
                    "dt": s_dt,
                    "end": e_dt if done else None,
                    "status": row.get(status_col, ""),
                    "amt": 0,
                    "fks": {},
                    "done": done,
                    "span": transit,
                }
            )
        o.write(t, names, rows)
        self.refs[t] = refs
        self._fact_rows[t] = (names, rows)

    def _gen_event(self, t, o):
        """Event stream: sequence times strictly monotonic, and a completed entity must reach its terminal state (invariants 4 and 9)."""
        rng, cols = self.rng, self.schema[t]
        names = [c["name"] for c in cols]
        pk, (fk, par) = self.pk_of(t), self._parent_of(t)
        if not par:
            return self._gen_fact(t, o)
        prefs = self.refs[par]
        type_col = next((c["name"] for c in cols if c["sem"] == "enum"), None)
        seq_col = next((c["name"] for c in cols if c["sem"] == "seq"), None)
        ts_col = next((c["name"] for c in cols if c["sem"] == "ts"), None)
        date_cols = [c["name"] for c in cols if c["sem"] == "date"]
        flag_col = next((c["name"] for c in cols if c["sem"] == "flag"), None)
        seq_vals = (self.profile.get("event_seq", {}) or {}).get(t)
        if not seq_vals:
            v, _ = self._enum_values(t, type_col) if type_col else ([], [])
            seq_vals = list(v) or ["STEP1", "STEP2", "STEP3", "DONE"]
        per = max(1, round(self.nrows[t] / max(1, len(prefs))))
        rows, no = [], 0
        for pr in prefs:
            done = pr.get("done", True)
            seq = seq_vals if done else seq_vals[: max(1, min(len(seq_vals) - 1, rng.randint(1, len(seq_vals) - 1)))]
            if per < len(seq) and not done:
                seq = seq[: max(1, per)]
            span = max(1, pr.get("span", 3))
            chain = EventChain(rng, day_ts(rng, pr["dt"]), min_gap_hours=1)
            cap = datetime.combine(min(pr.get("end") or self.end, self.end), datetime.min.time()) + timedelta(hours=20)
            hard = datetime.combine(self.end, datetime.min.time()) + timedelta(hours=23, minutes=59)
            cap = min(cap, hard)
            for j, ev in enumerate(seq):
                no += 1
                pin = (
                    (datetime.combine(pr["end"], datetime.min.time()) + timedelta(hours=rng.randint(9, 20)))
                    if (j == len(seq) - 1 and done and pr.get("end"))
                    else None
                )
                ts = chain.step(avg_hours=span * 24 / max(1, len(seq) - 1), cap=cap, pin=pin)
                row = {pk: self._pk_val(t, no - 1), fk: pr["pk"]}
                if seq_col:
                    row[seq_col] = j + 1
                if type_col:
                    row[type_col] = ev
                if ts_col:
                    row[ts_col] = ts.strftime("%Y-%m-%d %H:%M:%S")
                for c in date_cols:
                    row[c] = ts.date().isoformat()
                if flag_col:
                    row[flag_col] = 1 if j == len(seq) - 1 else 0
                for c in cols:
                    if c["name"] in row:
                        continue
                    row[c["name"]] = self._fill_generic(t, c, rng, {"dt": ts.date()})
                self._apply_formulas(t, row)
                rows.append([row[c] for c in names])
        o.write(t, names, rows)

    def _metric_combos(self, t):
        """Plan the dimension combinations of a daily metric table: (combos, weights).

        Shared by ``_gen_metric`` and ``report()`` so the projection cannot drift from what is
        actually built. The row budget is only a cap: when the business has fewer legal
        combinations than the budget allows - three channels against room for eleven - the grid is
        that much smaller, and reporting the budget would over-state both the combinations and the
        resulting row count.
        """
        cols = self.schema[t]
        enum_cols = [c["name"] for c in cols if c["sem"] == "enum"]
        # The profile's joint distribution wins (combinations the business really has); without one,
        # fall back to the cartesian product of the enum domains.
        combos, weights = [], []
        groups = (self.profile.get("joint", {}) or {}).get(t, [])
        if groups:
            g = groups[0]
            for v in g["values"]:
                combos.append(dict(zip(g["cols"], v[: len(g["cols"])])))
                weights.append(float(v[len(g["cols"])]) if len(v) > len(g["cols"]) else 1.0)
        else:
            vals = [self._enum_values(t, c)[0] for c in enum_cols] or [[""]]
            for combo in itertools.product(*vals):
                combos.append(dict(zip(enum_cols, combo)))
                weights.append(1.0)
        cap = max(1, round(self.nrows[t] / max(1, len(self.days))))
        order = sorted(range(len(combos)), key=lambda i: -weights[i])[:cap]
        return [combos[i] for i in order], [weights[i] for i in order]

    def _gen_metric(self, t, o):
        """Daily metric table (traffic/spend/headline table with no FK): fills a day x dimension-combination
        grid so the grain is unique with no gaps or duplicates. Magnitudes follow calendar intensity
        (trend/weekend/promotion/anomaly) and derived columns come from base quantities (invariants 1 and 2)."""
        rng, cols = self.rng, self.schema[t]
        names = [c["name"] for c in cols]
        pk = self.pk_of(t)  # declared PK wins; never assume it is the first column
        date_c = next((c["name"] for c in cols if c["sem"] in ("date", "date_pk")), None)
        cnt_cols = [c["name"] for c in cols if c["sem"] == "count"]
        amt_cols = [c["name"] for c in cols if c["sem"] == "amount"]
        ratio_cols = [c["name"] for c in cols if c["sem"] == "ratio"]
        ts_cols = [c["name"] for c in cols if c["sem"] == "ts"]
        combos, weights = self._metric_combos(t)
        wm = sum(weights) / len(weights)
        dwm = sum(self.day_w) / len(self.day_w)
        lo, hi = self._col_profile(t, cnt_cols[0] if cnt_cols else "").get("range", (400, 9000))
        rows, i = [], 0
        for di, d in enumerate(self.days):
            for ci, cb in enumerate(combos):
                i += 1
                row = {pk: self._pk_val(t, i - 1), **cb}
                if date_c:
                    row[date_c] = d.isoformat()
                inten = (
                    (self.day_w[di] / dwm)
                    * (weights[ci] / wm)
                    * self.cal.disrupt_factor(d, **{k.replace("_cd", ""): v for k, v in cb.items()})
                )
                for j, c in enumerate(cnt_cols):
                    v = self._derive(t, c, row, rng)
                    if v is None:
                        v = (
                            round(lognorm_between(rng, lo, hi) * inten)
                            if j == 0
                            else round(float(row[cnt_cols[j - 1]]) * rng.uniform(0.25, 0.62))
                        )
                    row[c] = max(0, int(v))
                for c in amt_cols:
                    v = self._derive(t, c, row, rng)
                    row[c] = v if v is not None else round(lognorm_between(rng, 20, 900) * inten, 2)
                for c in ratio_cols:
                    row[c] = round(bounded_gauss(rng, 0.03, 0.012, 0.004, 0.09), 4)
                for c in ts_cols:
                    row[c] = day_ts(rng, d).strftime("%Y-%m-%d %H:%M:%S")
                for c in cols:
                    if c["name"] in row:
                        continue
                    row[c["name"]] = self._fill_generic(t, c, rng, {"dt": d})
                self._apply_formulas(t, row)
                rows.append([row[c] for c in names])
        o.write(t, names, rows)
        self.refs[t] = []

    # ---------------------------------------------------------------- main flow
    def generate(self, out="data/datasource.duckdb", verbose=True, tolerance=0.06, _attempt=1, _t_start=None):
        import shutil
        import time

        if _attempt == 1:
            self.precheck()  # validate the config first so a mistake costs nothing instead of a full pass
            self._reused_cal = self._load_calibration(out)
        t0 = time.perf_counter()
        t_start = _t_start if _t_start is not None else t0  # end-to-end start, preserved across retries
        Path(out).resolve().parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(out).resolve().parent / f"_csv_{Path(out).stem}"
        o = CsvOut(tmp, verbose=False)
        self._csv = o
        for t in self._topo():
            r = self.roles[t]
            {
                ROLE_DATE: self._gen_date,
                ROLE_DIM: self._gen_dim,
                ROLE_FACT: self._gen_fact,
                ROLE_DETAIL: self._gen_detail,
                ROLE_DOWNSTREAM: self._gen_downstream,
                ROLE_EVENT: self._gen_event,
                ROLE_SNAPSHOT: self._gen_fact,
                ROLE_METRIC: self._gen_metric,
            }[r](t, o)
        self._backfill_tiers(o)
        t_gen = time.perf_counter() - t0
        t1 = time.perf_counter()
        from genlib import build_db

        # pre_sql runs before the summary layer: business post-processing (restatement, cross-table
        # backfill) must happen after the fact tables are final and before summaries are built,
        # otherwise dws/ads will not reconcile with the source tables
        extra = (
            self._sql_block("pre_sql")
            + (self._auto_summary_sql() if self.extra_tables in ("summary", "all") else "")
            + self._sql_block("extra_sql")
        )
        sizes, degraded = build_db(
            out,
            tmp,
            list(o.counts),
            extra,
            self._comments(),
            types={t: [(c["name"], c["type"]) for c in cols] for t, cols in self.schema.items()},
            create_sql=getattr(self, "decl_sql", None),
        )
        shutil.rmtree(tmp, ignore_errors=True)
        self._dump_meta(out, sizes)
        total = sum(sizes.values())
        dev = total / self.rows - 1
        free = [
            t
            for t in self.nrows
            if self.roles[t] not in (ROLE_DATE, ROLE_DIM, ROLE_METRIC) and t not in getattr(self, "pinned_rows", {})
        ]
        if abs(dev) > tolerance and _attempt < 3 and free:
            k = self.rows / total
            for t in free:
                self.nrows[t] = max(50, int(self.nrows[t] * k))
            self.rng = random.Random(self.seed)
            self.pools, self.refs, self._fact_rows, self._pending_dim_rows = {}, {}, {}, {}
            return self.generate(out, verbose, tolerance, _attempt + 1, t_start)
        res = {
            "tables": sizes,
            "degraded": list(degraded),
            "rows": total,
            "deviation": dev,
            "attempts": _attempt,
            "t_gen": t_gen,
            "t_db": time.perf_counter() - t1,
            "t_last_pass": time.perf_counter() - t0,
            "t_total": time.perf_counter() - t_start,
        }  # end-to-end, including every retry
        if verbose:
            extra = f" (last pass {res['t_last_pass']:.2f}s)" if _attempt > 1 else ""
            if getattr(self, "_reused_cal", False) and _attempt == 1:
                extra += " (reused previous calibration)"
            if abs(dev) > 0.25:  # badly off: usually a wrong role inference - do not ship it as a success
                print(
                    f"  ! actual rows differ from target by {100 * dev:+.0f}%, far outside tolerance. "
                    f"Usual causes: the main fact table was classified as a dimension/metric table, "
                    f"or pinned table_rows values conflict."
                    f"\n    Check the roles in report() and override with profile['roles'] if needed."
                )
            print(
                f"  {len(sizes)} tables / {res['rows']:,} rows (target {self.rows:,}, deviation "
                f"{100 * dev:+.1f}%, {_attempt} pass(es)) | generate {t_gen:.2f}s "
                f"build {res['t_db']:.2f}s total {res['t_total']:.2f}s{extra} | "
                f"{Path(out).stat().st_size / 1e6:.1f} MB | {self.start} ~ {self.end}"
            )
        return res

    def _backfill_tiers(self, o):
        """Backfill the tier column from actual contribution (invariant 5): compute the facts first, label second."""
        if not self.main_fact:
            return
        for t, pool in self.pools.items():
            if not pool:
                continue
            want = (self.profile.get("tier_cols", {}) or {}).get(t)
            cands = [
                c["name"]
                for c in self.schema[t]
                if c["sem"] == "enum" and re.search(r"tier|level|grade|segment", c["name"])
            ]
            tier_c = next((c for c in cands if not want or c in want), None) if cands else None
            if not tier_c or (want is not None and tier_c not in want):
                continue
            fk = next((f for f in self.fks.get(self.main_fact, []) if self.pk_owner.get(f) == t), None)
            if not fk:
                continue
            agg = {}
            for r in self.refs[self.main_fact]:
                k = r["fks"].get(fk)
                agg[k] = agg.get(k, 0.0) + (r["amt"] or 0)
            pkc = self.pk_of(t)
            ranked = sorted(pool, key=lambda e: -agg.get(e[pkc], 0.0))
            # Prefer the enum values declared in the profile (business order, high to low) so member_level does not become S/A/B/C
            declared = self.profile.get("enums", {}).get(tier_c)
            lbls = list(declared) if declared else None
            bands = (
                self.profile.get("tier_bands", {}).get(tier_c)
                if isinstance(self.profile.get("tier_bands"), dict)
                else self.profile.get("tier_bands")
            )
            if not bands:
                q = [0.05, 0.2, 0.5, 1.0][-len(lbls) :] if lbls else [0.05, 0.2, 0.5, 1.0]
                if lbls and len(lbls) != len(q):
                    q = [round((i + 1) / len(lbls), 3) for i in range(len(lbls))]
                bands = list(zip(q, lbls or ["S", "A", "B", "C"]))
            for rk, e in enumerate(ranked):
                e[tier_c] = next(lbl for q, lbl in bands if rk < len(ranked) * q or lbl == bands[-1][1])
            names, rows = self._pending_dim_rows[t]
            ci, pi = names.index(tier_c), names.index(pkc)
            m = {e[pkc]: e[tier_c] for e in pool}
            for r in rows:
                r[ci] = m[r[pi]]
            o.write(t, names, rows)

    def _auto_summary_sql(self):
        """User DDL usually only has source tables. Q&A and dashboards need day-grain summaries, derived here from the main fact."""
        mf = self.main_fact
        self._made = {}
        if not mf or any(t.startswith(("dws_", "ads_")) for t in self.schema):
            return ""
        cols = self.schema[mf]
        # A fact table may carry only TIMESTAMP and no DATE (order_time/paid_time); then truncate the
        # first business timestamp to a day, otherwise the whole summary layer is missing and there is
        # no day-grain entry point for dashboards or Q&A
        date_c = next((c["name"] for c in cols if c["sem"] == "date"), None)
        date_x = date_c
        if not date_c:
            ts_c = next((c["name"] for c in cols if c["sem"] == "ts" and not AUDIT_TS.match(c["name"])), None)
            if not ts_c:
                return ""
            date_x = f"CAST({ts_c} AS DATE)"
        date_f = f"f.{date_c}" if date_c else f"CAST(f.{ts_c} AS DATE)"
        amts = [c["name"] for c in cols if c["sem"] == "amount"]
        fks = [f for f in self.fks[mf] if self.pk_owner.get(f) in self.pools]
        status_c = next((c["name"] for c in cols if c["sem"] == "enum" and "status" in c["name"]), None)
        date_tbl = next((t for t, r in self.roles.items() if r == ROLE_DATE), None)
        subj_col, subj_tbl = self._pick_subject(mf)
        subj_col, subj_tbl = self._pick_subject(mf)

        def _agg(pfx="", skip=()):
            a = ["count(*) AS row_cnt"]
            a += [f"count(DISTINCT {pfx}{f}) AS {f.replace('_id', '')}_cnt" for f in fks if f not in skip]
            a += [f"round(sum({pfx}{x}), 2) AS {x}" for x in amts]
            if status_c:
                vals, _ = self._enum_values(mf, status_c)
                a += [
                    f"sum(CASE WHEN {pfx}{status_c}='{v}' THEN 1 ELSE 0 END) "
                    f"AS {re.sub(r'[^a-z0-9]', '_', str(v).lower())}_cnt"
                    for v in list(vals)[:4]
                ]
            return a

        sql = ""
        dense = (
            bool(subj_tbl)
            and (self.nrows.get(mf, 0) / max(1, len(self.days) * len(self.pools.get(subj_tbl) or [1]))) >= 3
        )
        if subj_tbl and subj_col and not dense:
            subj_tbl = subj_col = (
                None  # subject cardinality too high (e.g. buyer x day): one row per cell, no analytical value
            )
        # A low-cardinality enum column on the fact itself (channel/payment method/category) is a better summary grain
        want = (self.profile.get("summary_dims", {}) or {}).get(mf)
        enum_dims = list(want) if want else []
        if not enum_dims:
            for c in cols:
                if c["sem"] != "enum" or (status_c and c["name"] == status_c) or self.CODE_COL.search(c["name"]):
                    continue
                if 2 <= len(self._enum_values(mf, c["name"])[0]) <= 30:
                    enum_dims = [c["name"]]
                    break
        for ed in enum_dims:
            self._made[f"dws_{mf.split('_', 1)[-1]}_{ed}_day"] = f"{ed} x day summary (generated by the engine)"
            sql += f"""
CREATE OR REPLACE TABLE dws_{mf.split("_", 1)[-1]}_{ed}_day AS
SELECT {date_x} AS stat_dt, {ed}, {", ".join(_agg())}
FROM {mf} GROUP BY ALL;
"""
        if subj_tbl and subj_col:
            self._made[f"dws_{mf.split('_', 1)[-1]}_subject_day"] = "subject x day summary (generated by the engine)"
            pk_s = self.pk_of(subj_tbl)
            name_c = next((c["name"] for c in self.schema[subj_tbl] if c["sem"] == "name"), pk_s)
            tier_c = next(
                (
                    c["name"]
                    for c in self.schema[subj_tbl]
                    if c["sem"] == "enum" and re.search(r"tier|level|grade", c["name"])
                ),
                None,
            )
            extra = f", s.{tier_c}" if tier_c else ""
            sql += f"""
CREATE OR REPLACE TABLE dws_{mf.split("_", 1)[-1]}_subject_day AS
SELECT {date_f} AS stat_dt, f.{subj_col}, s.{name_c}{extra},
       {", ".join(_agg("f.", skip=(subj_col,)))}
FROM {mf} f JOIN {subj_tbl} s ON s.{pk_s}=f.{subj_col}
GROUP BY ALL;
"""
        d_join = f"FROM {date_tbl} d LEFT JOIN x ON x.stat_dt=d.{self.pk_of(date_tbl)}" if date_tbl else "FROM x"
        d_cols = (
            "d.date_key AS stat_dt, d.month_key, d.quarter_cd, d.is_weekend, d.day_type_cd, d.event_name,"
            if date_tbl
            else "x.stat_dt,"
        )
        metrics = ", ".join(
            ["COALESCE(x.row_cnt, 0) AS row_cnt"]
            + [f"COALESCE(x.{f.replace('_id', '')}_cnt, 0) AS {f.replace('_id', '')}_cnt" for f in fks]
            + [f"COALESCE(x.{a}, 0) AS {a}" for a in amts]
        )
        ratio = ""
        if len(amts) >= 2:
            ratio = (
                f", round(100.0*COALESCE(x.{amts[-1]},0)/nullif(x.{amts[0]},0), 2) "
                f"AS {amts[-1].rsplit('_', 1)[0]}_rate_pct"
            )
            ratio += f", round(COALESCE(x.{amts[0]},0)/nullif(x.row_cnt,0), 2) AS avg_{amts[0]}_per_row"
        sql += f"""
CREATE OR REPLACE TABLE ads_business_daily AS
WITH x AS (SELECT {date_x} AS stat_dt, {", ".join(_agg())} FROM {mf} GROUP BY 1)
SELECT {d_cols} {metrics}{ratio}
{d_join};
"""
        return sql

    def _comments(self):
        c = []
        role_txt = {
            ROLE_DATE: "Date dimension with promotion/shutdown event flags, for period comparison and anomaly attribution",
            ROLE_DIM: "Dimension table (entity master data)",
            ROLE_FACT: "Main fact table",
            ROLE_DETAIL: "Fact detail; amounts align exactly with the header per document",
            ROLE_DOWNSTREAM: "Downstream fact; dated after the upstream, status derived from the facts",
            ROLE_EVENT: "Event stream; time strictly increasing by sequence number, completed entities carry the terminal state",
            ROLE_SNAPSHOT: "Snapshot table",
            ROLE_METRIC: "Daily metric table: one row = one day x one dimension combination, unique grain",
        }
        for t, r in self.roles.items():
            c.append((f"TABLE {t}", self.profile.get("table_comments", {}).get(t, role_txt[r])))
        for t in self.synthetic:
            c.append(
                (
                    f"TABLE {t}",
                    self.profile.get("table_comments", {}).get(
                        t,
                        "Date dimension (added by the engine): promotion/shutdown event flags, for period comparison and anomaly attribution",
                    ),
                )
            )
        for k, v in self.profile.get("column_comments", {}).items():
            c.append((f"COLUMN {k}", v))
        if self.extra_tables in ("summary", "all"):
            c.append(
                (
                    "TABLE ads_business_daily",
                    "Cross-domain daily report (generated by the engine); one row per day, the preferred entry for Q&A and dashboards",
                )
            )
        for t, txt in getattr(self, "_made", {}).items():
            c.append((f"TABLE {t}", self.profile.get("table_comments", {}).get(t, txt)))
        return c
