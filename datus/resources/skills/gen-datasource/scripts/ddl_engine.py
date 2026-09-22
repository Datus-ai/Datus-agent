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
    r"store|shop|warehouse|carrier|courier|department|industry|size|color|unit|"
    r"priority|severity|urgency)$"
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
# The fallback when a measure column has no declared range. A measure's units cannot be read off a
# DDL, so this is a plausible weight or duration and nonsense for anything bounded - it is written
# into the skeleton and named by precheck so the caller corrects it rather than meeting it in the data.
MEASURE_RANGE = (0.1, 40)

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


# A key named after the thing it identifies, for a table that declared none. `_id` / `_key` are
# already classified `id`; these are the families that are not, and that `_semantic` sends to
# `enum` or `text` because on a FACT table that is what they are - `orders.currency_code` is an
# attribute, `currencies.currency_code` is a key. The suffix cannot tell them apart, so the test
# is whether the column is named after ITS OWN table (see `DDLEngine._natural_keys`).
NATURAL_KEY_SUFFIX = re.compile(r"(^|_)(code|cd|uuid|guid|slug|handle)$")


def _stem(word: str) -> str:
    """`airports` -> `airport`, `countries` -> `countrie`, `routes` -> `route`.

    Deliberately not a real singulariser: it only has to make two names comparable by prefix, and
    `countrie` still prefixes `country_code` once both sides are stemmed. Guard 4 in
    `_natural_keys` is what stops a loose match from becoming a key.
    """
    w = word.lower()
    for suffix in ("ies", "es", "s"):
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def _named_after(table: str, col: str) -> bool:
    """Is ``col`` named after ``table`` - `airports.airport_code`, `skus.sku`, `countries.country`?"""
    t, c = _stem(table), _stem(col)
    # The table name can carry a qualifier the column drops (`flight_sensors` -> `sensor_id`), so
    # the last token of the table is enough on that side.
    tails = {t, _stem(table.split("_")[-1])}
    return any(c == x or c.startswith(x) or x.startswith(c) for x in tails if x)


def _semantic(col: str, dtype: str) -> str:
    dtype = dtype.upper()
    for pat, types, sem in SEMANTIC:
        if re.search(pat, col) and any(dtype.startswith(t) for t in types):
            return sem
    # A BOOLEAN is a flag whatever it is called: `in_service BOOLEAN` fell through to free text
    # and a run paid a turn to say so in profile["semantics"].
    if dtype.startswith("BOOL"):
        return "flag"
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
            weekend_lift=self._weekly()[0],
            dow_overrides=self._weekly()[1],
        )
        self._cum_w = list(itertools.accumulate(self.day_w))
        self.pools, self.refs = {}, {}
        self._enum_cache = {}
        self._fact_rows, self._pending_dim_rows, self._id_base = {}, {}, {}
        self._code_seq = {}  # (table, column) -> codes handed out, so _fill_generic never repeats one
        self._alt_seq = {}  # (table, column) -> alternate-key values handed out, same reason
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
    @staticmethod
    def _ddl_parse_error(stmt, exc, declared=frozenset()):
        """The sentence a reader acts on when a CREATE will not parse.

        Two failures reach here and they need opposite fixes, so the message has to tell them
        apart. A missing TABLE survives the retry pass only when no statement declares it - a
        typo, or a table the DDL forgot - and answering that with "rewrite the dialect" sends
        the reader to edit syntax that parses fine. Everything else really is a dialect the
        engine does not speak.

        ⚠️ Matched on "Table with name", not on "Catalog Error": a foreign dialect raises the
        same error class for its TYPES (`NUMBER(19)` -> "Catalog Error: Type with name NUMBER
        does not exist"), and the looser test claimed Oracle and PostgreSQL DDL referenced a
        missing table. The suite caught it; the two spellings differ only in that one word.
        """
        text = str(exc)
        missing = DDLEngine._MISSING.search(text)
        if missing and missing.group(1).lower() in declared:
            # The target IS declared, so nothing is misspelled: these statements could not be
            # ordered into a sequence that resolves, which in practice means a cycle.
            return (
                f"Cannot parse DDL: {exc}\nStatement: {stmt[:200]}\n"
                f"`{missing.group(1)}` IS declared in this DDL, so the name is right - but no order "
                f"of the CREATE statements resolves them all, which means they reference each other "
                f"in a cycle. Break it: drop one REFERENCES and re-add it as a plain column."
            )
        if missing and "does not exist" in text:
            return (
                f"Cannot parse DDL: {exc}\nStatement: {stmt[:200]}\n"
                f"This statement references a table that no CREATE in this DDL declares - the "
                f"engine retried it after every other table existed and it still could not "
                f"resolve. Check the referenced name for a typo, or add the missing table. "
                f"Declaration ORDER is not the problem: forward references are retried."
            )
        return (
            f"Cannot parse DDL: {exc}\nStatement: {stmt[:200]}\n"
            f"The engine parses DuckDB syntax. This looks like another dialect "
            f"(PostgreSQL/MySQL/StarRocks/Oracle/...): rewrite it to DuckDB first - drop the "
            f"schema prefix and backticks, map the types, strip PARTITION BY / DISTRIBUTED BY / "
            f"PROPERTIES / ENGINE / storage clauses and CREATE INDEX, and keep the inline "
            f"comments (the engine extracts enum domains from them)."
        )

    _DECLARES = re.compile(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"']?([\w.]+)[`\"']?", re.I)
    _MISSING = re.compile(r"Table with name (\w+) does not exist", re.I)

    @classmethod
    def _declared_names(cls, stmts):
        """Every table this DDL creates, whether or not it parsed."""
        return {m.group(1).split(".")[-1].lower() for s in stmts for m in [cls._DECLARES.search(s)] if m}

    @classmethod
    def _blame(cls, leftover, stmts):
        """Which unparsed statement to report, out of a set that failed together.

        ⚠️ NOT simply the first. When `a -> b` and `b -> xxx` both fail, `a` is first in file order
        and its error says "Table with name b does not exist" - about a table declared on the very
        next line. Reporting that sends the reader hunting a typo in `a`, and never mentions `xxx`,
        which is the only thing actually wrong. Prefer a statement whose missing target no CREATE
        declares; that is the one the reader has to fix, and the rest fail only because of it.
        """
        declared = cls._declared_names(stmts)
        for stmt, first, _last in leftover:
            m = cls._MISSING.search(str(first))
            if m and m.group(1).lower() not in declared:
                return stmt, first
        # Nothing names an undeclared table, so the remaining failures are either a cycle or a
        # second problem that was hidden behind a dependency. A LAST failure that is no longer
        # about a missing table is the latter: everything it needed got created and it still will
        # not parse, which makes it the statement to report and the reason to report.
        for stmt, _first, last in leftover:
            if not cls._MISSING.search(str(last)):
                return stmt, last
        return leftover[0][0], leftover[0][1]

    @staticmethod
    def _create_until_stuck(items, attempt):
        """Retry ``attempt`` over ``items`` until a whole pass adds nothing.

        ⚠️ ONE RETRY PASS ONLY RESOLVES A REFERENCE ONE LEVEL DEEP. With ``a -> b -> c`` declared
        in that order, the retry meets ``a`` again before ``b`` exists and gives up on a schema
        that is perfectly valid - and, worse, reports it as a table nothing declares, which sends
        the reader looking for a typo in a name that is right there. Passes repeat while any
        statement lands; that terminates because every pass either shrinks the list or ends it.

        Returns ``(item, first failure, last failure)``. Both are needed: the first names what was
        missing when nothing else existed, which is the state the reader's DDL describes, while the
        last is what still blocks the statement after everything that COULD be created has been -
        and those differ when a statement has a second problem hiding behind its dependency.
        """
        pending = [(item, None, None) for item in items]
        while pending:
            rest, progressed = [], False
            for item, first, _last in pending:
                failure = attempt(item)
                if failure is None:
                    progressed = True
                else:
                    rest.append((item, first if first is not None else failure, failure))
            pending = rest
            if not progressed:
                break
        return pending

    @staticmethod
    def _try_ddl(con, stmt):
        """Execute one CREATE, retrying once with the foreign-dialect noise stripped.

        Returns the failure, or ``None`` when the statement landed.
        """
        try:
            con.execute(stmt)
            return None
        except Exception as e:  # noqa: BLE001 - the caller decides whether this is fatal
            cleaned = re.sub(r"\b(ENGINE|CHARSET|COLLATE|COMMENT)\s*=?\s*'[^']*'", "", stmt)
            cleaned = re.sub(r"\bAUTO_INCREMENT\b|\bUNSIGNED\b|\bCOMMENT\s+'[^']*'", "", cleaned, flags=re.I)
            try:
                con.execute(cleaned)
                return None
            except Exception:  # noqa: BLE001
                return e

    def _parse(self):
        """Let DuckDB parse the DDL - no SQL parser to write, and common dialects just work."""
        con = duckdb.connect(":memory:")
        # ⚠️ A FAILED CREATE IS NOT AN ERROR YET, and treating it as one rejected valid schemas.
        #
        # DuckDB resolves a REFERENCES target at CREATE time, so a foreign key pointing at a table
        # declared FURTHER DOWN the file fails on the first pass and succeeds once the rest exists.
        # Nothing requires a schema to be written parent-first - a DDL that opens with the fact
        # table and lists its dimensions after it is the ordinary shape - yet that DDL used to come
        # back as "This looks like another dialect", which sends the reader off rewriting syntax
        # that was never wrong. The same file parsed if you reordered it.
        stmts = [s.strip() for s in self.ddl.split(";") if s.strip()]
        leftover = self._create_until_stuck(stmts, lambda s: self._try_ddl(con, s))
        if leftover:
            stmt, failure = self._blame(leftover, stmts)
            raise ValueError(self._ddl_parse_error(stmt, failure, self._declared_names(stmts))) from None
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

    def _natural_keys(self):
        """Tables keyed by a name rather than by an `_id`, where the DDL declared nothing.

        ⚠️ Without this the key is invisible to every downstream step. `_semantic` sends `_code` to
        `enum` and `slug` / `uuid` to `text`, and both the key guess and the foreign-key inference
        read `sem == "id"` - so on a bare DDL keyed by `airport_code`, `routes.origin_airport_code`
        pointed at nothing. Measured on a production DDL with no keys at all: 5 of the 12 foreign
        keys a reader would draw were inferred, the `airports` dimension was an island, and the
        plan said none of it.

        Four guards, because the suffix alone is not the signal - `orders.currency_code` is an
        attribute and `currencies.currency_code` is a key:

        1. the table declares no PRIMARY KEY (a declaration always wins);
        2. the table has no `_id` / `_key` column - `flights.flight_number` matches the naming and
           is NOT unique, one flight number per day, so anything more key-shaped takes precedence;
        3. the column is named after its own table;
        4. something references it, or it is the first column. A key nothing points at buys
           nothing and only risks being wrong.
        """
        out = {}
        referenced = {rc for (rt, rc) in (getattr(self, "decl_fk", None) or {}).values() if rc}
        for t, cols in self.schema.items():
            if self.decl_pk.get(t):
                continue
            names = [c["name"] for c in cols]
            if any(re.search(r"(_id|_key)$", n) for n in names):
                continue
            for n in names:
                if not (NATURAL_KEY_SUFFIX.search(n) or _named_after(t, n)):
                    continue
                if not _named_after(t, n):
                    continue
                if n in referenced or n == names[0]:
                    out[n] = t
                    break
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
        # 2b. A single-column UNIQUE is a key too, and SQL lets a foreign key reference one.
        #
        # ⚠️ WITHOUT THIS EVERY SUCH CHILD ROW IS AN ORPHAN. `decl_uniq` was parsed, written into
        # the metadata, and read by nothing else - so a child whose FK pointed at a UNIQUE column
        # found no parent pool here, fell through to the generic filler and invented its own
        # values: measured at 100% orphans against 0% for the same DDL with the target declared
        # PRIMARY KEY. That is invariant 6 ("FKs sampled from upstream only") broken by a schema
        # the engine accepted without a word.
        #
        # ⚠️ AFTER the declared foreign keys, not before. A UNIQUE column on an unrelated table
        # shares nothing but a name, and running first let it take that name: `users.email UNIQUE`
        # + `contacts.email UNIQUE` + `tickets.email REFERENCES users(email)` handed `email` to
        # `contacts`. An explicit REFERENCES states which table is meant; a UNIQUE elsewhere does
        # not, so it fills gaps rather than claiming.
        for t, keys in (getattr(self, "decl_uniq", None) or {}).items():
            if t not in self.schema:
                continue
            for cols in keys:
                if len(cols) == 1:
                    pk.setdefault(cols[0], t)
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
        # A declared PK/FK/UNIQUE column must be generated with id semantics even when it is not
        # named *_id (else it gets text placeholders).
        #
        # ⚠️ UNIQUE belongs in this list, and leaving it out is what made a unique column collide
        # with itself. `carriers.iata_code UNIQUE` was classified `enum` by name, so it drew from a
        # 4-value built-in vocabulary - five rows, three distinct codes, and the UNIQUE constraint
        # dropped for the whole table at build time. The column says every value differs; enum
        # semantics say pick from a short list. Only one of those can be honoured.
        # ⚠️ UNIQUE ONLY UPGRADES A COLUMN THAT IS NOT ALREADY SOMETHING. A primary or foreign key
        # IS an identifier whatever it is called, but UNIQUE lands on natural keys of every type -
        # `report_date DATE UNIQUE` on a calendar table is ordinary, and forcing id semantics onto
        # it stopped the date generator running and left the whole column NULL. Only a column with
        # no better classification is promoted, which still covers the `iata_code VARCHAR UNIQUE`
        # case this was added for.
        #
        # `name` is excluded for a different reason: promoting it DID satisfy the constraint, by
        # replacing 1,200 brand names with `brand_name_1 ... brand_name_1200`. A demo database is
        # read by an agent and shown to people; a unique column of slugs is not better than a
        # readable one that collides. `_unique_name` keeps the names and makes them unique.
        _TYPED_SEM = ("date", "ts", "amount", "count", "ratio", "measure", "flag", "name", "seq")
        _unique_cols = [
            (t, cols[0])
            for t, keys in (getattr(self, "decl_uniq", None) or {}).items()
            for cols in keys
            if len(cols) == 1
            and any(c["name"] == cols[0] and c["sem"] not in _TYPED_SEM for c in self.schema.get(t, ()))
        ]
        for t, col in list(self.decl_fk) + [(t, c[0]) for t, c in self.decl_pk.items() if len(c) == 1] + _unique_cols:
            for c in self.schema.get(t, []):
                if c["name"] == col and (t, col) not in ovr:
                    c["sem"] = "id"
        # A name-keyed table, and every column anywhere that carries that name. Forcing the
        # semantic - rather than special-casing each sampling site - is what makes the existing
        # `sem == "id"` tests see it: the key guess, the foreign-key inference and all three
        # sampling paths then need no change at all.
        self.natural_keys = self._natural_keys()
        for col, owner in self.natural_keys.items():
            for t, cols in self.schema.items():
                for c in cols:
                    if c["name"] == col and (t, col) not in ovr:
                        c["sem"] = "id"
            pk.setdefault(col, owner)
        # A key referenced under a longer name. `routes.origin_airport_code` is the `airports` key
        # with a qualifier in front, and nothing inferred it: `pk_owner` is indexed by exact column
        # name, so on a bare DDL the whole `airports` dimension was an island - 8 of the 12 edges a
        # reader would draw. Measured on the two DDLs this work is based on, requiring a `_`
        # boundary before the key name adds exactly those 4 edges and not one anywhere else.
        #
        # Declared keys still win: `setdefault`, and `decl_fk` was written into `pk` first.
        #
        # Resolved against a SNAPSHOT of the keys, then applied everywhere, in two passes. One pass
        # made the result depend on table order: the first table to claim `origin_airport_code` put
        # it in `pk`, and every later table with the same column was skipped by the "already a key"
        # guard - so `flights` linked and `routes`, two lines further down the DDL, kept 20/20
        # orphans.
        known = dict(pk)
        self.renamed_keys = {}
        for t, cols in self.schema.items():
            owned = {c for c, owner in known.items() if owner == t} | set(self.decl_pk.get(t, ()))
            for c in cols:
                n = c["name"]
                if n in known or n in owned:
                    continue
                for key, owner in known.items():
                    if owner != t and n.endswith(f"_{key}"):
                        self.renamed_keys[(t, n)] = (owner, key)
                        break
        for (t, n), (owner, _key) in self.renamed_keys.items():
            for c in self.schema.get(t, ()):
                if c["name"] == n and (t, n) not in ovr:
                    c["sem"] = "id"
            pk.setdefault(n, owner)
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
        # Anchored at a token start. Unanchored, `aggregate_dt` matched "reg", `invalidated_at`
        # matched "valid", `reopened_at` matched "open" and `reentry_time` matched "entry" - so the
        # table's only business date read as an entity attribute and the whole table was demoted to
        # a dimension, every measure on it generated as a static attribute with no time signal.
        attr_date = re.compile(
            # `reg` spelled out: a token start alone still matched `regression_dt` and `regional_dt`.
            r"(?:^|_)(?:reg(?:_|ist)|onboard|hire|launch|first|join|create|birth|open|expire|valid|found|entry)"
        )

        self._demoted = {}

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
                    self._blame_date(t, attr_date, _measures(t))
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
                    if not biz:
                        self._blame_date(t, attr_date, namt or _measures(t))
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

    # Lines per parent document. A production run was handed 26,320 order_items under 48,880 orders
    # - 0.54 lines per order, which is not a schema that can exist and is also not a plan the engine
    # can carry out: ``_gen_detail`` rounds lines-per-parent up to one, so it would have produced
    # 48,880 rows against a printed plan of 26,320. The caller saw the contradiction, stopped
    # trusting the allocation, and pinned ``table_rows`` on every table by hand.
    DETAIL_PER_PARENT = 1.8
    DOWNSTREAM_PER_PARENT = 0.6  # a shipment / claim / repayment does not follow every document

    def _fanout_parent(self, t):
        """The table a fixed fan-out counts from. `_doc_parent` first, then any declared parent,
        so this works on a table the engine classified as something other than a detail."""
        par = self._doc_parent(t)
        if par:
            return par
        for f in self.fks.get(t, []):
            owner = self.pk_owner.get(f)
            if owner and owner != t:
                return owner
        return None

    def _apply_fanout(self, n):
        """``profile['per_parent']``: this table has exactly N rows for each row of its parent.

        ⚠️ There was no way to say this, and for a whole class of schema the fan-out IS the grain:
        a lot has exactly 16 wafers and a wafer exactly 10 probed die, an invoice has one line per
        ordered item, a sensor channel one reading per interval. The engine offers a stochastic
        1.8 lines per parent, which destroys that grain, and pinning `table_rows` instead expresses
        the counts but leaves calibration nothing to scale - measured on a 12-table foundry DDL,
        the pins implied a plan 24% over budget that three passes could not recover. Faced with
        those two, a production run abandoned the engine and hand-wrote 886 lines of generator,
        which took 24 minutes of its 44.

        Applied to a fresh allocation AND after every calibration rescale, so scaling the root
        fact carries the whole tree with it coherently - which is why these tables are derived
        rather than pinned: pinning them would freeze the tree and make the budget unreachable
        again.
        """
        fixed = getattr(self, "fixed_fanout", None) or {}
        if not fixed:
            return n
        for _ in range(len(fixed) + 1):  # a fan-out of a fan-out settles on the next pass
            for t, k in fixed.items():
                par = self._fanout_parent(t)
                if par and par in n:
                    n[t] = max(1, int(n[par] * k))
        return n

    def _doc_parent(self, t):
        """The document a detail row belongs to, resolved from declared keys at planning time.

        ``_parent_of`` answers the same question during generation, but it filters on ``self.refs``,
        which fills up only as tables are generated - asked before that it calls every detail an
        orphan, and orphans would all be sized as if they were facts.
        """
        for f in self.fks.get(t, []):
            par = self.pk_owner.get(f)
            if par and self.roles.get(par) in (ROLE_FACT, ROLE_DETAIL, ROLE_DOWNSTREAM):
                return par
        return None

    def _doc_weight(self, t, details, seen=()):
        """A detail's weight within the document block: its parent's, times its lines per parent."""
        if t in seen:  # a key cycle in the DDL must not recurse forever
            return 1.0
        per = self.DOWNSTREAM_PER_PARENT if self.roles.get(t) == ROLE_DOWNSTREAM else self.DETAIL_PER_PARENT
        par = self._doc_parent(t)
        return per * (self._doc_weight(par, details, seen + (t,)) if par in details else 1.0)

    # How a week looks, by business shape rather than by assuming retail. The engine defaulted to
    # `weekend_lift=1.33` - Saturday a third busier than a weekday - which is a consumer shop and
    # nothing else. A B2B schema generated that way has its busiest days on the weekend, and the
    # quality check then confirms the "B2C shape" it was handed.
    WEEKLY_SHAPE = {
        "weekend_heavy": (1.33, None),  # consumer retail, food delivery, entertainment
        "weekday_heavy": (0.35, {0: 1.05, 4: 0.95}),  # B2B, payroll, clinics, logistics booking
        "flat": (1.0, {0: 1.0, 4: 1.0}),  # metering, sensors, always-on services
    }

    def _weekly(self):
        """(weekend_lift, dow_overrides) for the declared weekly shape."""
        shape = self.profile.get("weekly_shape", "weekend_heavy")
        lift, overrides = self.WEEKLY_SHAPE.get(shape, self.WEEKLY_SHAPE["weekend_heavy"])
        return self.profile.get("weekend_lift", lift), overrides

    def _plan_rows(self):
        """Allocate rows: the fact layer takes the bulk, dimensions size by business density.

        The shares are stated in ``references/profile-spec.md`` section 1.1.
        """
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
        # The fact layer and its details share one block of the budget, split by lines per parent
        # rather than by a share each. Independent shares gave `detail` less than `fact`, which no
        # amount of tuning fixes: a detail row exists only as a line of its parent document, so its
        # count is the parent's count times the lines per document and can never be below it.
        #
        # Resolve in three steps, and in this order: the pins, then the details those pins
        # determine, then a share of what is left for whatever nothing has fixed. Applying the pins
        # last - as a plain overwrite once the shares were computed - left a detail table on the
        # number its share gave it and never looked at the parent again: pinning `orders` to 12,000
        # or to 1,500 produced the same 12,085 `order_items`, so one line per order or eight, both
        # outside the documented 1.4-2.2 band, both a plan `_gen_detail` then carried out faithfully.
        block = share["fact"] + share.get("detail", 0)
        weights = {t: 1.0 for t in facts}
        weights.update({t: self._doc_weight(t, details) for t in details})
        settled = {t: pinned[t] for t in facts + details if t in pinned}

        def per_parent(t):
            return self.DOWNSTREAM_PER_PARENT if roles[t] == ROLE_DOWNSTREAM else self.DETAIL_PER_PARENT

        progressed = True
        while progressed:  # a detail of a detail settles on the pass after its parent
            progressed = False
            for t in details:
                par = self._doc_parent(t)
                if par not in (facts + details):
                    continue
                if t not in settled and par in settled:  # pinned parent -> its lines follow
                    settled[t] = max(50, int(settled[par] * per_parent(t)))
                elif par not in settled and t in settled:  # pinned lines -> the documents under them
                    settled[par] = max(50, int(settled[t] / per_parent(t)))
                else:
                    continue
                progressed = True
        rest = [t for t in facts + details if t not in settled]
        left = max(0.0, budget * block / tot - sum(settled.values()))
        unit = left / (sum(weights[t] for t in rest) or 1)
        n.update(settled)
        for t in rest:
            n[t] = max(50, int(unit * weights[t]))
        main_n = n.get(self.main_fact, max(50, int(budget * 0.5)))
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
        self.fixed_fanout = {
            t: int(k)
            for t, k in (self.profile.get("per_parent", {}) or {}).items()
            if t in self.schema and isinstance(k, (int, float)) and int(k) >= 1
        }
        self._apply_fanout(n)
        # A key-bearing `joint` group is the row count: ten real airports means ten rows, and
        # asking for more can only be answered with a duplicate key. Recorded so the pre-check can
        # say it happened rather than leaving the caller to wonder why `dim_rows` was ignored.
        self._joint_clamped = {}
        for t in list(n):
            limit = self._joint_key_limit(t)
            if limit is not None and n[t] > limit:
                self._joint_clamped[t] = (n[t], limit)
                n[t] = limit
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
    def _meta_key(self, t):
        """The column the quality check may verify as this table's key, or None when it has none.

        ⚠️ This used to be ``pk_of`` for every table, and ``pk_of`` always answers - it falls back
        to the first column. A table left deliberately keyless therefore shipped its FIRST COLUMN
        as a primary key, and on a time series that column is the foreign key: a measured run
        failed with "sensor_readings.series_id has 16,377 duplicate keys" for a table whose DDL
        declares no key at all. The advice that went with the limitation - leave such a table
        keyless and say so - could not be followed, because following it still produced a hard
        failure, so the run instead invented a surrogate primary key and altered the user's schema.

        A declared composite key is reported as none for the same reason: no single column of it
        is unique, so there is nothing a one-column check can verify.
        """
        declared = getattr(self, "decl_pk", {}).get(t)
        if declared:
            return declared[0] if len(declared) == 1 else None
        col = self.pk_of(t)  # a guess, so it has to earn the name
        if col in self.fks.get(t, ()) or (t, col) in (getattr(self, "decl_fk", None) or {}):
            return None  # a foreign key is the parent's key, never this table's
        return col

    def _dump_meta(self, db_path, sizes):
        """Persist the engine inference so the quality check reuses it, instead of re-deriving a second, conflicting view."""
        import json

        made = set(getattr(self, "_made", {})) | set(getattr(self, "synthetic", []))
        meta = {
            "generator": "gen-datasource/ddl_engine",
            "date_range": [self.start.isoformat(), self.end.isoformat()],
            "extra_tables": self.extra_tables,
            # The weekday check asserted a B2C shape on every dataset. It has to know what was
            # asked for: `flat` is a legitimate answer for metering or an always-on service, and
            # without this the caller who declares it can never reach ok: true.
            "weekly_shape": self.profile.get("weekly_shape", "weekend_heavy"),
            "strict_ddl": self.extra_tables == "none",
            "declared": {
                "pk": self.decl_pk,
                "fk": {f"{t}.{c}": list(v) for (t, c), v in self.decl_fk.items()},
                "unique": self.decl_uniq,
            },
            "roles": dict(self.roles),
            "synthetic_tables": sorted(made),
            "fks": {t: v for t, v in self.fks.items() if v},
            "pks": {t: c for t in self.schema for c in [self._meta_key(t)] if c},
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

    def _calibration_lands_at(self, planned, free_rows):
        """Where generate()'s calibration would land, by replaying its own update rule.

        Three passes of ``k = rows / total`` applied to the scalable tables only, with the same
        ``max(50, int(...))`` floor. Shared with the pre-check so a warning cannot promise one
        outcome while generation produces another.
        """
        free = dict(free_rows)
        fixed = planned - sum(free.values())
        total = planned
        # Two updates, not three: `generate()` rescales before attempts 2 and 3 and never after
        # attempt 3 (`_attempt < 3`). A third update here brings the simulation inside tolerance
        # for a run that will finish outside it, which suppresses the warning this exists to give.
        for _ in range(2):
            if not total or abs(total / self.rows - 1) <= 0.06:
                break
            k = self.rows / total
            free = {t: max(50, int(n * k)) for t, n in free.items()}
            total = fixed + sum(free.values())
        return total

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
            err.extend(self._sql_problems())

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
            + list(self.profile.get("lifecycle", {}) or {})
        ):
            chk_ref("table-level config", k, need_col=False)

        # lifecycle: stages count BUSINESS timestamps. `created_at` and its kin are audit columns,
        # aligned to the first business timestamp and not counted, so (created_at, resolved_at)
        # is one stage, not two - and a status mapped to stage 2 was downgraded on every row
        # without a word. A run declared exactly that and read the result as "lifecycle ignored".
        for t, lc in (self.profile.get("lifecycle") or {}).items():
            if t not in self.schema:
                if t in tabs:  # a summary or pre_sql table: the reference check let it through
                    err.append(f"lifecycle[{t}]: `{t}` is not a DDL table; lifecycle applies to generated tables only")
                continue
            if not isinstance(lc, dict):
                continue
            ts_cols = [c["name"] for c in self.schema[t] if c["sem"] == "ts"]
            biz = [c for c in ts_cols if not AUDIT_TS.match(c)]
            audit = [c for c in ts_cols if AUDIT_TS.match(c)]
            over = {k: v for k, v in (lc.get("stages") or {}).items() if isinstance(v, int) and v > len(biz)}
            if over:
                warn.append(
                    f"lifecycle[{t}]: stages {over} reach past the {len(biz)} business timestamp(s) {biz}"
                    + (
                        f" - {', '.join(audit)} are audit columns, aligned to the first one and not counted"
                        if audit
                        else ""
                    )
                    + "; every row with such a status is downgraded to the last stage that exists"
                )

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

        # joint: on a fan-out table the group is drawn per row WITH replacement (the row count is
        # not known up front), so a key column in it would repeat and the table would lose its
        # constraints at build time. Real values for a key belong on the parent or the dimension.
        for t, groups in (self.profile.get("joint") or {}).items():
            if t not in self.schema or self.roles.get(t) not in (ROLE_DETAIL, ROLE_DOWNSTREAM):
                continue
            for g in groups if isinstance(groups, list) else []:
                keys = sorted(set(g.get("cols", ())) & self._key_cols(t))
                if keys:
                    err.append(
                        f"joint[{t}]: group ({', '.join(g.get('cols', ()))}) includes key column(s) "
                        f"{', '.join(keys)}; a {self.roles[t]} table is drawn per row with replacement, so the "
                        f"key would repeat and the table would ship without its constraints - put the real "
                        f"values on the table that owns the key"
                    )

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
            # `_derive` multiplies the base by the ratio, so (0.0, 0.0) turns the whole chain to
            # zeros - visible only after a generate, an import and a check. The skeleton no longer
            # emits it (it prefills believable ratios), but it is never a legitimate ratio, so a
            # hand-written one is still refused here.
            if isinstance(d, dict):
                ratios = [d.get("ratio")] if not isinstance(d.get("ratio"), dict) else list(d["ratio"].values())
                for r in ratios:
                    if isinstance(r, (list, tuple)) and len(r) == 2 and r[0] == 0 and r[1] == 0:
                        err.append(
                            f"derive[{key}]: a ratio of (0.0, 0.0) generates a column of zeros. Fill in the real range"
                        )
                        break
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

        # Measure units cannot be inferred from a DDL, so the engine falls back to a 0.1-40
        # lognormal - a plausible weight or duration and nonsense for anything bounded. A production
        # schema shipped a GPA of 11.79 on a DECIMAL(4,2). The skeleton writes the fallback out as a
        # value to correct, but deleting the entry puts the column right back on it silently, so the
        # columns still sitting on it are named here. A warning, not an error: the skeleton has to
        # stay runnable as copied, and 0.1-40 is genuinely right for some measures.
        cfg = self.profile.get("columns") or {}
        unranged = sorted(
            f"{t}.{c['name']}"
            for t in self.schema
            if self.roles.get(t) != ROLE_DATE
            for c in self.schema[t]
            if c["sem"] == "measure"
            and tuple(cfg.get(f"{t}.{c['name']}", {}).get("range", ()) or ()) in ((), MEASURE_RANGE)
        )
        if unranged:
            warn.append(
                f"measure columns on the engine's 0.1-40 fallback: {', '.join(unranged[:8])}"
                + (f" and {len(unranged) - 8} more" if len(unranged) > 8 else "")
                + ". Set columns['t.col']['range'] for any whose units are not a 0.1-40 quantity"
            )

        shape = self.profile.get("weekly_shape")
        if shape is not None and shape not in self.WEEKLY_SHAPE:
            # `.get(shape, weekend_heavy)` swallowed a typo, and the two surfaces then disagreed:
            # the report and the metadata echoed what was asked for while the data was generated as
            # a consumer shop - so the quality check was handed a label the rows did not carry.
            err.append(f"weekly_shape: {shape!r} is not a shape; choose one of {', '.join(sorted(self.WEEKLY_SHAPE))}")

        pin_keys = [k for k in ("table_rows", "dim_rows") if self.profile.get(k)]
        pinned_names = set(self.profile.get("table_rows", {}) or {}) | set(self.profile.get("dim_rows", {}) or {})
        # What calibration could scale if nothing were pinned. A schema of dimensions and metric
        # tables has none of its own, and refusing a pin there would blame the caller for the DDL.
        calibratable = [t for t in self.nrows if self.roles.get(t) not in (ROLE_DATE, ROLE_DIM, ROLE_METRIC)]
        if calibratable and pinned_names and not [t for t in calibratable if t not in pinned_names]:
            err.append(
                f"{' and '.join(pin_keys)} pins every table calibration could scale "
                f"({', '.join(sorted(calibratable))}), so the total is whatever the pins add up to and "
                f"rows= stops meaning anything. A production run shipped 90,004 rows against a requested "
                f"80,000 this way. Leave one of them unpinned."
            )

        pinned = self.profile.get("table_rows", {}) or {}
        pinned_total = sum(v for v in pinned.values() if isinstance(v, int))
        if pinned_total > self.rows * 1.06:
            free = [
                t for t in self.nrows if self.roles.get(t) not in (ROLE_DATE, ROLE_DIM, ROLE_METRIC) and t not in pinned
            ]
            err.append(
                f"table_rows pins {pinned_total:,} rows against a budget of {self.rows:,} "
                f"({100.0 * (pinned_total / self.rows - 1):+.0f}%). Calibration can only scale tables that are "
                f"neither pinned nor role-fixed, and {'none are left' if not free else 'only ' + ', '.join(sorted(free)) + ' remain'}"
                f" - the total cannot come back to target. Lower the pinned counts or raise rows="
            )

        # row-count constraints (invariant 10)
        for t, n in self.nrows.items():
            if self.roles.get(t) == ROLE_DIM and n > self.rows * 0.08:
                warn.append(f"dimension {t} has {n:,} rows, over 8% of the total; the fact layer gets squeezed")
        # ⚠️ The pins can each look modest and still make the target unreachable, and neither
        # check above sees it. A pinned DETAIL fixes its parent (pin / lines-per-parent) and
        # therefore every sibling detail of that parent, so the plan grows far past what was
        # pinned. Measured: `flight_sensors: 21,000` + `sensor_readings: 35,000` against
        # rows=80,000 pins 70% of the budget - too little to trip either check - while the plan
        # those two imply is 109,727 (+37%). Calibration, which may only scale the three unpinned
        # tables, reached +11% after its three passes, and the run spent two whole build cycles
        # chasing the total before giving up.
        #
        # Replays calibration's own update rule rather than modelling it, so this cannot drift
        # from what generate() will actually do.
        for t, k in (self.profile.get("per_parent", {}) or {}).items():
            if t not in self.schema:
                err.append(f"per_parent: table `{t}` is not in the DDL")
                continue
            if not isinstance(k, (int, float)) or int(k) < 1:
                err.append(f"per_parent[{t}]: {k!r} is not a row count; give a whole number >= 1")
                continue
            if not self._fanout_parent(t):
                err.append(
                    f"per_parent[{t}]: {t} has no parent to count from - it references no other table, "
                    f"so 'N rows per parent row' has no meaning here"
                )
            elif self.roles.get(self._fanout_parent(t)) in (ROLE_DATE, ROLE_DIM, ROLE_METRIC):
                # A dimension is sized by business density and calibration never scales it, so a
                # tree rooted in one cannot grow toward `rows=` - and `_gen_detail` needs its
                # parent in `refs`, which a dimension never enters, so the exact grain is lost too.
                err.append(
                    f"per_parent[{t}]: its parent {self._fanout_parent(t)} is planned as a "
                    f"{self.roles.get(self._fanout_parent(t))}, which calibration never scales and the "
                    f"detail generator cannot read rows from. A fan-out has to hang off a fact or a "
                    f"detail - set profile['roles'] = {{{self._fanout_parent(t)!r}: 'fact'}} if that is "
                    f"what it is"
                )
            elif self.roles.get(t) not in (ROLE_DETAIL, ROLE_DOWNSTREAM):
                # The row COUNT is honoured for every role, because every generator reads nrows.
                # Exact lines per parent are the detail generator's mechanism, so a table the
                # engine reads as something else gets the count and not the grain.
                warn.append(
                    f"per_parent[{t}]: {t} is planned as a {self.roles.get(t)}, so it gets the row "
                    f"count ({k} x its parent) but not one row per parent - the exact fan-out is the "
                    f"detail generator's. Set profile['roles'] = {{{t!r}: 'detail'}} if the grain matters"
                )

        for t, (asked, limit) in (getattr(self, "_joint_clamped", None) or {}).items():
            warn.append(
                f"{t} is capped at {limit:,} rows, not {asked:,}: a `joint` group on its key supplies "
                f"{limit:,} distinct combination(s), and a key cannot repeat. Supply more combinations "
                f"if the table needs more rows."
            )

        planned = sum(self.nrows.get(t, 0) for t in self.schema)
        free_rows = {
            t: self.nrows[t]
            for t in self.nrows
            if self.roles.get(t) not in (ROLE_DATE, ROLE_DIM, ROLE_METRIC) and t not in pinned_names
        }
        if pinned_names and free_rows and self.rows:
            reached = self._calibration_lands_at(planned, free_rows)
            if abs(reached / self.rows - 1) > 0.06:
                warn.append(
                    f"the pins imply a plan of {planned:,} rows against rows={self.rows:,} "
                    f"({100.0 * (planned / self.rows - 1):+.0f}%), and calibration can only scale "
                    f"{', '.join(sorted(free_rows))} - replaying its three passes lands at "
                    f"~{reached:,} ({100.0 * (reached / self.rows - 1):+.0f}%), outside the 6% "
                    f"tolerance. A pinned detail table also fixes its parent and every sibling "
                    f"detail of that parent, which is where the extra rows come from. Unpin the "
                    f"detail tables and let rows= do the work, or raise rows= to match."
                )

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

    # Where each funnel stage sits relative to the top of the funnel, recognised by what the stage
    # is called. A step's ratio is the quotient of the two stages it connects, so a schema that
    # skips stages still gets a believable number: purchasers taken straight off sessions is a 2%
    # site conversion, where "purchasers per checkout user" would have said 55%. Keyed on the step
    # name alone, that is exactly the mistake it made. These are not claims about the caller's
    # business - they are numbers that make a first run converge, so the quality check can say which
    # one to move instead of the caller designing ten of them against a blank page. An empty slot is
    # a decision, and deriving these ten by hand took 16% of a measured 676-second turn.
    FUNNEL_STAGE = (
        (r"impression|exposure|imp_cnt", 1.0),
        (r"click", 0.035),
        (r"unique|visitor|uv", 0.027),
        (r"session|visit", 0.030),
        (r"view|browse|detail|pv", 0.088),  # the one fan-out: one visit browses several items
        (r"cart|wish|favou?rite", 0.0031),
        (r"checkout|submit", 0.0012),
        (r"purchas|buyer|convert|paid_user", 0.00065),
        (r"order", 0.00072),
    )
    FUNNEL_DEFAULT = (0.25, 0.45)  # neither end recognised: narrow rather than stall
    FUNNEL_BAND = 0.12  # the spread around the point estimate; wide enough to vary, tight enough to stay monotone
    # Money about a step, per unit of the count it follows: a cost per acquisition, an order value.
    MONEY_RATIO = ((r"spend|cost|budget|fee", (20.0, 60.0)), (r"revenue|gmv|sales|amount|value", (60.0, 260.0)))
    MONEY_DEFAULT = (30.0, 120.0)

    @classmethod
    def _stage_level(cls, name):
        """Match a stage word at the start of one of the column name's `_`-separated tokens.

        A bare substring search read `review_count` and `interview_count` as page views, because
        both contain "view" - so a review table got the one fan-out ratio in the table and its
        counts went up instead of down. Anchoring to a token start keeps `page_view`, `view_count`
        and `pv_cnt` and drops the accidents.
        """
        for pattern, level in cls.FUNNEL_STAGE:
            if re.search(rf"(?:^|_)(?:{pattern})", name, re.I):
                return level
        return None

    @classmethod
    def _default_ratio(cls, src, dst, is_money=False):
        """The prefilled ratio for one derive step, and whether the engine recognised both ends."""
        if is_money:
            for pattern, ratio in cls.MONEY_RATIO:
                if re.search(rf"(?:^|_)(?:{pattern})", dst, re.I):
                    return ratio, True
            return cls.MONEY_DEFAULT, False
        lo_stage, hi_stage = cls._stage_level(src), cls._stage_level(dst)
        if not lo_stage or not hi_stage:
            return cls.FUNNEL_DEFAULT, False
        point = hi_stage / lo_stage
        lo, hi = point * (1 - cls.FUNNEL_BAND), point * (1 + cls.FUNNEL_BAND)
        if point < 1:  # a narrowing step must stay narrowing on every row
            hi = min(hi, 0.99)
        return (round(lo, 4), round(hi, 4)), True

    def profile_skeleton(self):
        """A copy-paste PROFILE with everything the engine already knows filled in.

        Measured on a production run: 40% of one 272,000-character turn went on enumerating
        enum domains, joint combinations, per-channel `conditional` dictionaries and naming
        vocabularies in reasoning - then re-emitting them as a file in the next turn. It was
        composing from a blank page because that is what the skill handed it.

        Copied out unedited it produces a database that passes its own funnel assertions: an empty
        slot is a decision, and ten of them were what the caller met first. Everything is inferred
        from the DDL or prefilled with a believable default; the only blank left is ``calendar``,
        which is a business fact the engine has no way to know and must not invent. Fill that, run,
        and let the quality check say which default to move.
        """
        enum_cols = {
            (t, c["name"])
            for t in self.schema
            for c in self.schema[t]
            if c["sem"] == "enum" and self.roles.get(t) != ROLE_DATE
        }
        known = getattr(self, "ddl_enums", {}) or {}
        partial = getattr(self, "_partial_enums", set())
        undefined = sorted({c for _t, c in enum_cols if c not in known} | set(partial))
        fact_enums = sorted({c for t, c in enum_cols if self.roles.get(t) in (ROLE_FACT, ROLE_DETAIL)})

        import json as _json

        def lit(value):
            """Emit an identifier as a Python string literal.

            Table and column names come from the caller's DDL, where DuckDB allows quotes and
            newlines inside a quoted identifier. Interpolating them raw produced a skeleton that
            would not parse - and the skeleton exists to be copied into `gen.py` and run.

            ``json.dumps`` rather than ``repr``: its output is a valid Python string literal too,
            and it keeps the double quotes the rest of the skeleton uses instead of switching to
            single ones on the entries that happen to need escaping.
            """
            return _json.dumps(str(value))

        out = ["PROFILE = {"]
        out.append("    # --- The only section the engine cannot infer at all. Write it first. ---")
        out.append('    "calendar": {')
        out.append(f"        # Windows must fall inside {self.start} ~ {self.end} or they are dropped whole.")
        out.append('        "promos": [  # ("MM-DD", "MM-DD", multiplier, "name")')
        out.append("        ],")
        out.append('        "slows": [],')
        out.append('        "disruptions": [  # {"at": 0.0-1.0, "days": N, "factor": <1, "name": "", "scope": {}}')
        out.append("        ],")
        out.append("    },")
        out.append(f'    "trend_mom": {self.profile.get("trend_mom", 0.031)},   # month-over-month growth')

        if known:
            out.append("    # Domains below came from your DDL comments and are already applied:")
            for col, vals in list(known.items())[:12]:
                flag = "  <- looks incomplete, finish it in enums" if col in partial else ""
                out.append(f"    #   {col}: {', '.join(str(v) for v in vals[:6])}{flag}")
        if undefined:
            out.append('    "enums": {   # no domain found, or the DDL comment looked truncated.')
            out.append("        #   An empty list changes nothing: the DDL domain or the default still applies.")
            out.append("        #   Fill one in only to override what is listed above.")
            for col in undefined[:12]:
                out.append(f"        {lit(col)}: [],")
            out.append("    },")

        if fact_enums:
            out.append('    "conditional": {   # one dimension behaving differently from another')
            out.append(f"        # Grouping columns available on the fact layer: {', '.join(fact_enums[:8])}")
            out.append("    },")

        # Every metric table, not just the first: one "derive" block holding all of them. Emitting
        # a second block would be a duplicate key that silently overwrites the first, and stopping
        # after one dropped the second table's funnel entirely - a schema with a channel table and
        # a campaign table got a chain for one and zeros for the other.
        derive_lines = []
        for t in sorted(t for t, r in self.roles.items() if r == ROLE_METRIC):
            # Counts form the funnel and chain to each other; amounts are money *about* a step, so
            # each takes the nearest count before it. Chaining them by raw column order produced
            # "attributed_revenue from ad_spend", which means nothing - and a cap of 8 cut off
            # attributed_orders, the one ratio profile-spec names as the tuning point.
            counts = [c["name"] for c in self.schema[t] if c["sem"] == "count"]
            if len(counts) < 3:
                continue
            derive_lines.append(f"        # {t}: the funnel, prefilled. Tune what the quality check flags.")
            for i in range(1, len(counts)):
                ratio, known_step = self._default_ratio(counts[i - 1], counts[i])
                derive_lines.append(
                    f'        {lit(f"{t}.{counts[i]}")}: {{"from": {lit(counts[i - 1])}, "ratio": {ratio}}},'
                    + ("" if known_step else "  # <- step not recognised; check this one")
                )
            ordered = [c["name"] for c in self.schema[t] if c["sem"] in ("count", "amount")]
            for name in [c["name"] for c in self.schema[t] if c["sem"] == "amount"]:
                before = [c for c in ordered[: ordered.index(name)] if c in counts]
                if before:
                    ratio, known_step = self._default_ratio(before[-1], name, is_money=True)
                    derive_lines.append(
                        f'        {lit(f"{t}.{name}")}: {{"from": {lit(before[-1])}, "ratio": {ratio}}},'
                        + ("  # money per step" if known_step else "  # <- amount not recognised; check this one")
                    )
        if derive_lines:
            out.append('    "derive": {')
            out.extend(derive_lines)
            out.append("    },")

        # Measure columns: the engine's fallback is a 0.1-40 lognormal, which is a plausible weight
        # or duration and nonsense for anything bounded - a production schema got a GPA of 11.79 on
        # a DECIMAL(4,2). It cannot be inferred from the DDL, so the default is written out here as
        # a value to correct rather than left hidden behind the generator.
        measures = sorted(
            f"{t}.{c['name']}"
            for t in self.schema
            if self.roles.get(t) != ROLE_DATE
            for c in self.schema[t]
            if c["sem"] == "measure" and f"{t}.{c['name']}" not in (self.profile.get("columns") or {})
        )
        if measures:
            out.append(
                '    "columns": {   # the engine cannot infer a measure\'s units. This is its fallback, not a reading'
            )
            out.append("        #   of your schema - correct any that is not a 0.1-40 quantity.")
            for key in measures[:12]:
                out.append(f'        {lit(key)}: {{"range": {MEASURE_RANGE}}},')
            out.append("    },")
        out.append('    "semantics": {},   # only the columns report() got wrong, above')
        out.append('    "vocab": {},       # the built-in words are retail-shaped (brands, "Works"/"Labs" suffixes).')
        out.append("        #   Any other industry replaces them here: brand / org_suffix / person / given / item.")
        out.append("}")
        out.append("")
        out.append("# The ratios above are defaults, not guesses at your business: run first, then move")
        out.append("# the ones check_datasource_quality flags. Nothing here has to be decided up front.")
        out.append("# Not in this skeleton on purpose:")
        out.append("#   formulas   - the amount identities above are already enforced; do not restate them")
        out.append("#   table_rows - pinning is what stops calibration reaching your row budget")
        return "\n".join(out)

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
        # What was INFERRED has to say so. A silent guess is worse than no guess: the engine's
        # inference on a keyless DDL used to be wrong in three ways - 8 of 12 edges, one ownership
        # reversed, a whole dimension unlinked - and the plan mentioned none of it, so the reader
        # had no way to correct what it could not see.
        fan = getattr(self, "fixed_fanout", None) or {}
        if fan:
            print(
                "fixed fan-out: "
                + ", ".join(f"{t} = {k} x {self._fanout_parent(t) or '?'}" for t, k in sorted(fan.items()))
                + " (derived from the parent, so calibration scales the parent and these follow)"
            )
        nat = getattr(self, "natural_keys", None) or {}
        ren = getattr(self, "renamed_keys", None) or {}
        if nat or ren:
            print("INFERRED keys - nothing declared these, so check them and declare any that are wrong:")
            for col, owner in sorted(nat.items(), key=lambda kv: kv[1]):
                print(f"  {owner}.{col} reads as the key of {owner} (the column is named after the table)")
            for (t, col), (owner, key) in sorted(ren.items()):
                print(f"  {t}.{col} -> {owner}.{key} (the key under a longer name)")
        for t, col in sorted(getattr(self, "_demoted", {}).items()):
            print(
                f"! {t} carries measures but is planned as a dimension, because "
                f"`{col}` reads as an entity attribute date (when the entity came into being) "
                f"rather than a business event. If `{col}` is when something happened, set "
                f"profile['roles'] = {{{t!r}: 'fact'}} - as a dimension its measures carry no "
                f"time signal at all."
            )
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
        # The same validation generate() runs, at the point the skill actually looks. A production
        # run pinned table_rows on every table, ran `gen.py report`, read a plan with nothing wrong
        # in it and moved on: the "pins every table calibration could scale" error existed the whole
        # time and had no surface to appear on until generate(), long after the profile was written.
        # precheck() prints its own findings; only the clean case needs a line here, and only when
        # there is a profile to have validated - plan_datasource runs report() on a bare DDL, where
        # "validated" would be a green light for work nobody has done yet.
        if not any(self.precheck(strict=False)) and self.profile:
            print("profile validated: no errors, no warnings")
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
            f"weekly shape: {self.profile.get('weekly_shape', 'weekend_heavy')} "
            f"(Saturday x{self._weekly()[0]} vs a weekday; "
            f"set profile['weekly_shape'] to {' / '.join(self.WEEKLY_SHAPE)})"
        )
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
                try:
                    samples = ", ".join(self._name_for(t, i, rng, self._preview_ent(t, rng)) for i in range(2))
                except KeyError as e:
                    samples = (
                        f"<naming[{t!r}]['tpl'] asks for {e}, which is neither a vocabulary key "
                        f"nor a generated column of {t}>"
                    )
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
        for t in sorted(self.schema):
            amts = [c["name"] for c in self.schema[t] if c["sem"] == "amount"]
            # An allow-list, not an exclusion list: only ``_gen_fact`` (fact and snapshot) and
            # ``_gen_detail``'s backfill call ``_settle_amounts``. Metric, event, downstream, dim
            # and date tables fill amounts independently, and claiming an identity they do not hold
            # is the failure this line was added to prevent - measured on a metric table whose
            # "paid_revenue = gross - discount + tax" was off by hundreds per row. A role added
            # later defaults to silence, which is the safe direction.
            if len(amts) < 2 or self.roles.get(t) not in (ROLE_FACT, ROLE_SNAPSHOT, ROLE_DETAIL):
                continue
            # The roles drive how `_settle_amounts` splits the money, whether or not an identity
            # comes out of them, and a role read wrong is the whole failure: `scholarship_amount`
            # matched the shipping pattern once and was *added* to the amount paid rather than
            # deducted. So they are printed for every table that settles amounts, not only for the
            # tables that end up announcing an identity.
            print(f"amount roles on {t}: " + ", ".join(f"{c}={self._amt_role(c)}" for c in amts))
            roles = {self._amt_role(c): c for c in amts}
            if not ({"discount", "tax", "ship", "cost", "profit"} & set(roles)):
                continue
            # A production run restated exactly this in profile['formulas'] and spent 39,000
            # characters of reasoning deriving it. The engine has always done it; nothing said so.
            parts = [roles.get("gross") or roles.get("unit") or amts[0]]
            for role, sign in (("discount", "-"), ("ship", "+"), ("tax", "+")):
                if role in roles:
                    parts.append(f"{sign} {roles[role]}")
            said = False
            if "paid" in roles and len(parts) > 1:
                print(f"amount identity enforced on {t}: {roles['paid']} = {' '.join(parts)}")
                said = True
            if "cost" in roles and "profit" in roles:
                revenue = roles.get("paid") or roles.get("gross") or roles.get("unit") or amts[0]
                print(f"amount identity enforced on {t}: {roles['profit']} = {revenue} - {roles['cost']}")
                said = True
            if said:
                print(
                    "  (coupon is not part of it. A role read wrong is corrected with "
                    "profile['formulas'], not by restating the identity)"
                )

        planned_sql = False
        for key in ("pre_sql", "extra_sql"):
            if not isinstance(self.profile.get(key) or "", (str, list, tuple)):
                # `_sql_block` raises TypeError on anything else, and it did so from here - halfway
                # through the report, as a traceback, for a mistake the pre-check below names in a
                # sentence. A report must not be the thing that breaks on a bad profile.
                continue
            block = self._sql_block(key)
            if block.strip():
                planned_sql = True
                print(
                    f"{key}: {block.count(';')} statement(s) will run "
                    f"({'before' if key == 'pre_sql' else 'after'} the summary layer)"
                )
        if planned_sql and not self._sql_problems():
            # The failures are listed by the pre-check below, which report() always runs, so saying
            # them here too would print the same line twice in the one report that needs to be
            # unambiguous. What stays here is the claim that they were planned - the alternative is
            # reasoning them through by hand, which one production run did at length.
            print("  every statement above planned against the schema; columns and types check out")

        # Last line of the report, because it is the question the reader has once they have read it.
        # Both traced runs called this before writing `gen.py` and then spent 30+ minutes inventing
        # their own verification; naming the loop here puts it in front of them at the moment the
        # plan becomes a file. `generate()` repeats it after every build - see `_next_step`.
        print(
            "then: write gen.py from profile_skeleton (fill calendar), run it once, and go straight to"
            "\n  import_database_file + check_datasource_quality. The check is the verification step;"
            "\n  correcting a default is what it is for, and designing one up front is not."
        )

    def _preview_ent(self, t, rng):
        """A stand-in row for the name preview.

        A ``tpl`` draws from the row being built, so ``"{brand} {subcategory}"`` has nothing to draw
        from before generation starts. Filling the table's enum columns here shows what the run will
        really produce, and keeps a valid template from raising KeyError out of ``report()`` - which
        is where a production run met it: a traceback in place of the plan, no row generated yet,
        and the workaround it reached for was a fake ``vocab["subcategory"]`` list that made the
        preview print names the run would never produce.
        """
        ent = {}
        for c in self.schema[t]:
            if c["sem"] != "enum":
                continue
            vals, ws = self._enum_values(t, c["name"])
            if vals:
                ent[c["name"]] = rng.choices(vals, ws)[0]
        return ent

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
        # ⚠️ THE MIDDLE GROUND, and it was the only one of the three nobody reported.
        #
        # The report already names what it EXTRACTED a domain for (from DDL comments) and what it
        # did not recognise at all (free text). Between them sits a column recognised as an enum
        # whose values nothing declared - the engine fills it from a built-in vocabulary, which is
        # a business decision made by a default. Silent, and it surfaces much later as a quality
        # failure about a distribution the reader never chose: a measured run wrote an assertion
        # over a unit column whose domain had been guessed, and got back "... differentiated by
        # unit: actual 0.9994 (expected 1000~100000)".
        # Everything that already pins a domain, in `_enum_values_raw`'s own precedence order:
        # profile['columns'][t.c]['values'] > profile['enums'][c] > ddl_enums[c] > built-in vocab.
        # Naming a column the reader has already set is worse than saying nothing - the line tells
        # them to do the thing they did, and repeats itself on the next report.
        pinned_cols = {
            key
            for key, spec in (self.profile.get("columns") or {}).items()
            if isinstance(spec, dict) and "values" in spec
        }
        # NON-EMPTY only, matching `_enum_values_raw`'s `if g:` - `profile_skeleton` emits
        # `"col": []` as a placeholder, so treating a key as a decision would silence the
        # line for exactly the reader who pasted the skeleton and filled nothing in.
        pinned_names = set(getattr(self, "ddl_enums", {})) | {
            col for col, vals in (self.profile.get("enums") or {}).items() if vals
        }
        guessed = [
            f"{t}.{c['name']}"
            for t in sorted(self.schema)
            # A date dimension's enums come from the calendar, never from a vocabulary, so
            # profile['enums'] is a no-op there - `profile_skeleton` already excludes them.
            if self.roles.get(t) != ROLE_DATE
            for c in self.schema[t]
            if c["sem"] == "enum" and c["name"] not in pinned_names and f"{t}.{c['name']}" not in pinned_cols
        ]
        if guessed:
            print(
                f"  value domains GUESSED for {len(guessed)} enum column(s) - nothing declared them: "
                f"{guessed[:12]}{' ...' if len(guessed) > 12 else ''}\n"
                f"    Set profile['enums'] for any whose real values you know, or put them in a DDL "
                f"comment. An assertion about a guessed domain is measuring the default, not the business."
            )

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
        """Generation order: by role, and within a role by dependency.

        ⚠️ Dependency COUNT is not a topological order, and the difference is not cosmetic. In a
        chain A -> B -> C inside one bucket, B and C both have one dependency, so the tie broke on
        dict order and the child could be generated before its parent. Its parent then had no refs
        yet, `_parent_of` called it an orphan, and `_gen_detail` fell through to `_gen_fact`, which
        invents the key: measured on a 12-table foundry DDL, `probe_die_result` came out
        **100% orphaned against `wafer`** on the stock engine with no profile at all, and with the
        roles set so the whole chain was one bucket the order came out exactly reversed -
        probe, then wafer, then wafer_lot.

        Kahn's algorithm, seeded with the old count order so that genuinely independent tables
        keep the order they had and nothing else moves.
        """
        order, seen = [], set()
        buckets = [ROLE_DATE, ROLE_DIM, ROLE_FACT, ROLE_DETAIL, ROLE_DOWNSTREAM, ROLE_EVENT, ROLE_SNAPSHOT, ROLE_METRIC]
        for role in buckets:
            group = [t for t, r in self.roles.items() if r == role]
            group.sort(key=lambda t: len([f for f in self.fks[t] if self.pk_owner.get(f) in group]))
            inside = set(group)
            deps = {
                t: {self.pk_owner[f] for f in self.fks[t] if self.pk_owner.get(f) in inside and self.pk_owner[f] != t}
                for t in group
            }
            remaining = list(group)
            while remaining:
                ready = [t for t in remaining if not (deps[t] - seen)]
                if not ready:
                    # A cycle within the bucket. Emit what is left in the old order rather than
                    # hanging: a key cycle is the DDL's to fix and generation still has to finish.
                    ready = remaining[:]
                for t in ready:
                    order.append(t)
                    seen.add(t)
                remaining = [t for t in remaining if t not in seen]
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

    def _ref_cols_needed(self, t):
        """Columns of ``t`` that a child's cross-table ``conditional`` groups by.

        A parent fact is not in ``pools``, so a child row can only see it through ``refs`` - and
        those carried the key, the date and the status, never the column a ``__by__`` named.
        ``delays.delay_minutes by flights.status`` therefore resolved to nothing and fell back to
        ``__default__`` on every row, while ``report()`` listed the rule as in effect. The columns
        travel with the ref only when something asks for them, so an unconfigured run pays nothing.
        """
        cache = self.__dict__.setdefault("_need_cols_cache", {})
        if t not in cache:
            need = set()
            for spec in (self.profile.get("conditional") or {}).values():
                by = spec.get("__by__") if isinstance(spec, dict) else None
                if isinstance(by, str) and "." in by and by.split(".", 1)[0] == t:
                    need.add(by.split(".", 1)[1])
            cache[t] = need
        return cache[t]

    def _ref_extra(self, t, row):
        need = self._ref_cols_needed(t)
        return {"cols": {c: row.get(c) for c in need}} if need else {}

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

    def _unique_name(self, t, col, candidate):
        """Make ``candidate`` unique within ``t.col``, keeping it readable.

        Only for a column the DDL declared UNIQUE. The vocabulary is finite and combinations
        repeat, so a 1,200-row dimension drew 1,101 distinct brand names and the constraint was
        dropped for the whole table at build time. Suffixing the few that collide keeps 1,200
        readable names; classifying the column as an identifier instead produced 1,200 unique
        `brand_name_N` slugs, which satisfies the constraint by discarding what the column is for.
        """
        seen = self.__dict__.setdefault("_name_seen", {}).setdefault((t, col), set())
        name, n = candidate, 1
        while name in seen:
            n += 1
            name = f"{candidate} {n}"
        seen.add(name)
        return name

    def _is_declared_unique(self, t, col):
        return any(cols == [col] for cols in (getattr(self, "decl_uniq", None) or {}).get(t, ()))

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
    def _blame_date(self, t, attr_date, has_measures):
        """Record the date column that demoted a table to a dimension, if one did.

        The date-name test is a naming heuristic and naming is per-industry: `opened_at` is when an
        account was opened on a dimension and when a ticket was raised on a fact. Getting it wrong
        costs the whole table - every measure comes out as a static attribute with no time signal -
        so the column that decided it is named in the report rather than left to be found in the
        data. Both classification paths record it: a table reaches ROLE_DIM with foreign keys and
        without them, and only the first was covered.

        Only when a column actually matched. A table with no date at all is a dimension for the
        ordinary reason, and reporting it as demoted by `` - an empty name - pointed the caller at a
        fact table it should not build.
        """
        if not has_measures:
            return
        blamed = next(
            (
                c["name"]
                for c in self.schema[t]
                if c["sem"] in ("date", "ts", "date_pk") and attr_date.search(c["name"])
            ),
            "",
        )
        if blamed:
            self._demoted[t] = blamed

    def _measure_val(self, t, name, rng, row=None, ents=None):
        """A measure, honouring `columns['t.col']['range']`.

        Both measure paths ignored the range, so the slot the skeleton writes and the warning
        precheck prints asked the caller to set a value that changed nothing - a GPA declared
        (0.0, 4.0) still came out at 49.02. They also disagreed on the fallback, 0.1-50 against
        0.1-40, so neither matched what was documented.

        A lower bound of zero is the ordinary case for a bounded measure (a score, a rate, a
        backlog) and ``lognorm_between`` cannot take it - log(0). Those draw from a bell centred in
        the range instead, which is closer to how bounded measures actually sit than a uniform.
        """
        lo, hi = self._col_profile(t, name).get("range", MEASURE_RANGE)
        # ⚠️ `conditional` used to stop at the amount columns, and nothing said so.
        #
        # profile-spec documents the numeric form as "Numeric column: different [lo, hi] per
        # group" - no mention that it means amount only. A measure column took the banding
        # silently and generated the fallback range anyway: a run that banded `delay_minutes` at
        # [200, 400] for weather got 0-8, and went into this file to find out why. The grouping
        # column is resolved exactly as `_gen_fact` resolves it for an amount.
        cr = self._cond_pick(self._cond_spec(t, name) or {}, row or {}, ents, t)
        if isinstance(cr, (list, tuple)) and len(cr) == 2:
            lo, hi = cr
        if lo > 0:
            return round(lognorm_between(rng, lo, hi), 3)
        return round(bounded_gauss(rng, (lo + hi) / 2, (hi - lo) / 4, lo, hi), 3)

    def _count_val(self, t, name, rng, row=None, ents=None):
        """A count under a declared ``columns[...]['range']`` or a ``conditional`` band, else None.

        The fact and generic fillers drew a count from a fixed 1-5 / 1-20 with no look at the
        profile, so ``passenger_count: {"range": (70, 250)}`` - the slot the skeleton itself
        offers - averaged 1.85 and a run reclassified the column as a measure to get around it.
        Same shape as the dimension path; None means "nothing declared, keep your default".
        """
        rg = self._col_profile(t, name).get("range")
        cr = self._cond_pick(self._cond_spec(t, name) or {}, row or {}, ents, t)
        if isinstance(cr, (list, tuple)) and len(cr) == 2:
            rg = cr
        if not rg:
            return None
        lo, hi = rg
        return max(int(lo), int(lognorm_between(rng, max(1, lo), hi)))

    def _lifecycle_prep(self, t):
        """Everything the lifecycle chain of ``t`` needs, computed once per table.

        Shared by the fact and detail generators: `lifecycle` was read by `_gen_fact` alone, so a
        detail table (an alert per flight) declaring stages kept every timestamp filled - OPEN
        alerts with a `resolved_at` - and nothing said the block had been ignored.
        """
        cols = self.schema[t]
        ts_cols = [c["name"] for c in cols if c["sem"] == "ts"]
        biz = [c for c in ts_cols if not AUDIT_TS.match(c)]
        lc = (self.profile.get("lifecycle", {}) or {}).get(t, {})
        gaps = lc.get("gap_hours") or [0] + [1.5 * 4**j for j in range(len(biz))]
        return {
            "declared": bool(lc),
            "biz": biz,
            "audit": [c for c in ts_cols if AUDIT_TS.match(c)],
            "gaps": gaps,
            "stages": lc.get("stages", {}),
            "inflight": set(lc.get("in_flight", [])),
            "span_h": sum(gaps[: max(1, len(biz))]),
            "cap": datetime.combine(self.end, datetime.min.time()) + timedelta(hours=23, minutes=59, seconds=59),
        }

    def _lifecycle_settle_status(self, lp, row, t0, status_col, st_vals, st_w, rng):
        """An in-flight status on a row old enough to have finished is rewritten to a terminal one."""
        if lp["inflight"] and status_col and row.get(status_col) in lp["inflight"]:
            if t0 + timedelta(hours=lp["span_h"]) <= lp["cap"]:
                pool_v = [(v, w) for v, w in zip(st_vals, st_w) if v not in lp["inflight"]]
                if pool_v:
                    row[status_col] = rng.choices([v for v, _ in pool_v], [w for _, w in pool_v])[0]

    def _lifecycle_chain(self, lp, row, t0, status_col, rng):
        """Business timestamps increase from ``t0`` by the declared gaps, truncated by the stage the
        status reaches and by the cut-off; a status the cut-off truncated is downgraded to the stage
        actually reached; audit columns align to ``t0``."""
        biz, gaps, stages, cap = lp["biz"], lp["gaps"], lp["stages"], lp["cap"]
        cur, kmax = t0, stages.get(str(row.get(status_col, "")), len(biz))
        reached = 0
        for j, c in enumerate(biz):
            if j >= kmax:
                row[c] = ""
                continue
            if j:
                cur = cur + timedelta(hours=max(0.05, gaps[min(j, len(gaps) - 1)] * rng.uniform(0.35, 1.9)))
            if cur <= cap:
                row[c] = cur.strftime("%Y-%m-%d %H:%M:%S")
                reached = j + 1
            else:
                row[c] = ""
        if stages and status_col and reached < kmax:
            back = next((k for k, v in stages.items() if v == reached), None)
            if back is not None:
                row[status_col] = back
        for c in lp["audit"]:
            row[c] = t0.strftime("%Y-%m-%d %H:%M:%S")

    def _is_int_col(self, t, col):
        d = next((c["type"].upper() for c in self.schema[t] if c["name"] == col), "")
        return any(d.startswith(x) for x in INT_T)

    def _pk_val(self, t, i, d=None):
        """Primary-key values honour the declared type: an integer PK gets integers, only VARCHAR gets a prefixed business code."""
        return self._key_val(t, self.pk_of(t), i, d)

    def _key_val(self, t, col, i, d=None):
        """One key column's i-th value. Shared so an alternate key is typed like a primary one."""
        if self._is_int_col(t, col):
            return self._id_base.setdefault((t, col), (len(self._id_base) + 1) * 10_000_000 + 1) + i
        prefix = re.sub(r"_id$|_key$|_no$", "", col).upper()[:3] or "ENT"
        return f"{prefix}{d.strftime('%y%m%d')}{i + 1:07d}" if d is not None else f"{prefix}{i + 1:07d}"

    def _alt_key_cols(self, t):
        """Columns of ``t`` that some declared foreign key points at, other than its primary key.

        ⚠️ A parent's UNIQUE column is a real key to its children, and no generator filled one.
        ``_gen_dim`` reached it through a catch-all that produced ``series_id_1``; every other
        generator reached it through ``_fill_generic``, which has no branch for an id column the
        table owns - so on a fact / detail / downstream / event parent the column came out empty
        and every child pointing at it was an orphan against NULL.

        ⚠️ A column this table DECLARES as its own foreign key is excluded, however much it is
        also a target. A 1:1 extension keyed by its parent's natural key is exactly that shape
        (`accounts.user_id UNIQUE REFERENCES users(user_id)`, with `tx` pointing at
        `accounts.user_id`), and filling it here overwrote a value sampled from `users` with a
        sequence of this table's own: 1,600/1,600 orphans upward and 17,090/17,090 downward, a new
        break of exactly the kind this method exists to close.

        Declared, not inferred: ``self.fks`` also holds columns inference merely guessed are
        foreign keys, on nothing more than a name another table happens to key. Letting a guess
        veto a REFERENCES the DDL actually states inverts "declared wins over inferred" - measured
        on `stores.region_code UNIQUE` referenced by `visits`, where the guess pointed at an
        unrelated `regions` table and cost the UNIQUE constraint on the parent.

        ⚠️ Only a column ``_key_val`` can actually produce: an integer or a string. A key value is
        a prefixed business code otherwise, and writing one into a DATE, TIMESTAMP or DOUBLE made
        the whole column NULL and the child 100% orphaned - shapes that were correct before, since
        those columns are filled by their own semantic branch and only needed carrying. The same
        reasoning as the typed-semantic gate on the UNIQUE promotion: a typed column keeps its
        type, and uniqueness on it is a hole to close elsewhere, not by retyping the column.
        """
        cache = self.__dict__.setdefault("_alt_key_cache", {})
        if t not in cache:
            pk = self.pk_of(t)
            mine = {c for (tt, c) in (getattr(self, "decl_fk", None) or {}) if tt == t}
            typed = {c["name"]: c["type"].upper() for c in self.schema.get(t, ())}
            cache[t] = tuple(
                sorted(
                    rc
                    for rc in self._alt_ref_cols(t)
                    if rc != pk and rc not in mine and typed.get(rc, "").startswith(INT_T + STR_T)
                )
            )
        return cache[t]

    def _alt_ref_cols(self, t):
        """Every column of ``t`` a declared foreign key points at, other than its primary key.

        Wider than ``_alt_key_cols`` on purpose. What a child must read and what this engine may
        overwrite are two questions, and answering them with one set broke a shape that worked:
        a DATE / TIMESTAMP / DOUBLE target is filled correctly by its own semantic branch, so it
        only ever needed carrying to the child - claiming it as well wrote a business code into a
        typed column and the whole column landed NULL.
        """
        cache = self.__dict__.setdefault("_alt_ref_cache", {})
        if t not in cache:
            own = {c["name"] for c in self.schema.get(t, ())}
            pk = self.pk_of(t)
            cache[t] = tuple(
                sorted(
                    {
                        rc
                        for (rt, rc) in (getattr(self, "decl_fk", None) or {}).values()
                        if rt == t and rc and rc != pk and rc in own
                    }
                )
            )
        return cache[t]

    def _set_alt_keys(self, t, row):
        """Give each of those columns a unique value of its own.

        Counted per (table, column) rather than by a loop index: these generators nest loops - a
        detail row per parent, an event row per stage - and a reused index hands out duplicates,
        the same reason the code-column branch keeps its own ``_code_seq``.
        """
        for col in self._alt_key_cols(t):
            seq = self._alt_seq[(t, col)] = self._alt_seq.get((t, col), 0) + 1
            row[col] = self._key_val(t, col, seq - 1)

    def _ref_alt(self, t, row):
        """The alternate-key values a child may need, carried on the parent's ref record."""
        return {c: row[c] for c in self._alt_ref_cols(t) if c in row}

    def _parent_key_val(self, t, fk_col, par, pr):
        """The parent value this child's foreign key must carry.

        ⚠️ ``pr["pk"]`` is the parent's PRIMARY KEY, and the detail / downstream / event
        generators wrote it into the child unconditionally - correct only when the foreign key
        happens to point at the primary key. Pointed at a UNIQUE column instead, the child was
        filled with primary-key values and matched nothing: measured at 100% orphans on a
        three-level DDL (fact -> detail -> detail), which is the shape ``_gen_fact``'s own
        sampling site - the one ``ref_col_of`` was wired into - never reaches.

        Falls back to the primary key when the parent's record does not carry the column. Every
        record built today does, so this is unreachable as the code stands; it stays because the
        alternative on a shape nobody anticipated is a KeyError that kills the whole generate.
        """
        ref_c = self.ref_col_of(t, fk_col)
        if ref_c == self.pk_of(par):
            return pr["pk"]
        return (pr.get("alt") or {}).get(ref_c, pr["pk"])

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

    def _key_cols(self, t):
        """Columns of ``t`` that something relies on being unique: its key, its declared UNIQUEs,
        and any column a declared foreign key points at."""
        cache = self.__dict__.setdefault("_key_cols_cache", {})
        if t not in cache:
            # ⚠️ `_meta_key`, not `pk_of`. `pk_of` always answers - it falls back to the first
            # column - so on a keyless time series it named the FOREIGN KEY as the key, and a
            # `joint` group on that column would then be drawn without replacement and the table
            # clamped to the number of combinations, for a column that repeats by design.
            cols = set()
            primary = self._meta_key(t)
            if primary:
                cols.add(primary)
            for keys in (getattr(self, "decl_uniq", None) or {}).get(t, ()):
                if len(keys) == 1:
                    cols.add(keys[0])
            cols.update(self._alt_ref_cols(t))
            cache[t] = frozenset(c for c in cols if c)
        return cache[t]

    def _joint_key_rows(self, t, g):
        """The value rows a key-bearing ``joint`` group may actually use.

        ⚠️ Distinct TUPLES are not enough: `[["ATL", "Atlanta"], ["ATL", "Atlanta Metro"]]` is two
        distinct combinations and one duplicate key, which drops the PRIMARY KEY at build time and
        with it every foreign key pointing at the table. At most one row per value of each key
        column in the group, in the order supplied so the result is stable.
        """
        cols = list(g.get("cols", ()))
        keys = [c for c in cols if c in self._key_cols(t)]
        out, seen = [], {c: set() for c in keys}
        for v in g.get("values", ()) or ():
            tup = tuple(v[: len(cols)])
            if len(tup) < len(cols):
                continue
            picked = {c: tup[cols.index(c)] for c in keys}
            if any(picked[c] in seen[c] for c in keys):
                continue
            for c in keys:
                seen[c].add(picked[c])
            out.append(tup)
        return out

    def _joint_key_limit(self, t):
        """How many rows ``t`` can have before a key-bearing ``joint`` group has to repeat itself.

        ⚠️ ``joint`` is the only mechanism that can give a key column real-world values - an
        `airport_code` of `ATL` rather than `AIR0000001` - and it sampled WITH replacement, so on a
        key it silently produced duplicates: 10 rows, 6 distinct, and the PRIMARY KEY was dropped
        at build time. That cascades - every foreign key pointing at the table is refused for want
        of a unique constraint, so one natural-key dimension took the constraints off three tables.
        A measured run spent ten minutes discovering there was no legal way to do this: `joint`
        broke the key, and at the time `pre_sql` ran after the constraints were on, so it could
        not UPDATE a key that was referenced either.
        """
        limit = None
        for g in (self.profile.get("joint", {}) or {}).get(t, []):
            if set(g.get("cols", ())) & self._key_cols(t):
                n = len(self._joint_key_rows(t, g))
                limit = n if limit is None else min(limit, n)
        return limit

    def _joint_groups(self, t):
        """The ``joint`` groups of ``t`` as (cols, value tuples, weights), parsed once."""
        cache = self.__dict__.setdefault("_joint_groups_cache", {})
        if t not in cache:
            out = []
            for g in (self.profile.get("joint", {}) or {}).get(t, []):
                vals = [tuple(v[: len(g["cols"])]) for v in g["values"]]
                w = [float(v[len(g["cols"])]) if len(v) > len(g["cols"]) else 1.0 for v in g["values"]]
                out.append((list(g["cols"]), vals, w))
            cache[t] = out
        return cache[t]

    def _joint_draw(self, t, rng):
        """One row's worth of ``joint`` values, drawn with replacement.

        ``_joint_plan`` sizes its draw to a row count known up front, which a detail or downstream
        table does not have - its rows fan out from the parent. Those generators never called it,
        so a group declared on `flight_sensors(series_name, unit)` was reported as in effect and
        the table came out `SERIES5 / UNIT3`. A per-row draw needs no count.
        """
        out = {}
        for cols, vals, w in self._joint_groups(t):
            out.update(dict(zip(cols, rng.choices(vals, w)[0])))
        return out

    def _joint_plan(self, t, n, rng):
        """Joint sampling: related enum columns in one row must be picked as a group (channel/source/campaign, province/city/tier).
        Sampling them independently creates combinations the business does not have, such as 'organic_search + TikTok ad'."""
        out = []
        for g in (self.profile.get("joint", {}) or {}).get(t, []):
            vals = [tuple(v[: len(g["cols"])]) for v in g["values"]]
            w = [float(v[len(g["cols"])]) if len(v) > len(g["cols"]) else 1.0 for v in g["values"]]
            if set(g.get("cols", ())) & self._key_cols(t):
                # Without replacement, because a key cannot repeat. Weights are dropped with it:
                # a key column has one row per value, so there is no distribution left to shape.
                # The same projection-unique subset ``_plan_rows`` sized the table from, or the
                # plan and the rows would disagree about which combinations exist.
                uniq = self._joint_key_rows(t, g)
                out.append((g["cols"], rng.sample(uniq, k=min(n, len(uniq)))))
            else:
                out.append((g["cols"], rng.choices(vals, w, k=n)))
        return out

    AMT_ROLE = [
        ("refund", r"refund|chargeback|return"),
        ("cost", r"cost|cogs"),
        ("profit", r"profit|margin"),
        ("coupon", r"coupon|voucher"),
        ("discount", r"discount|promo|rebate|reduction|deduct"),
        ("ship", r"ship|freight|delivery|postage|logistic"),
        # Anchored at both ends: these are whole words, and a token-start match alone read
        # `taxonomy_value` as a tax. The longer words below need only the leading anchor.
        # `duty` only in a monetary form - as the last token, or followed by one. Anchored at the
        # front alone it read `duty_roster_allowance` as a tax, and `duty_free_price` as one too.
        ("tax", r"tax(?:es)?(?:_|$)|vat(?:_|$)|duty(?:_(?:amount|amt|fee|charge|cost|value|paid))?$|duties(?:_|$)"),
        ("paid", r"paid|net_|settle|actual|final|payable|received|pay_|payment"),
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
            # Anchored at a token start, not anywhere in the string. `scholarship_amount` matched
            # the shipping pattern and was settled as a delivery charge - added to the paid amount
            # rather than deducted from it. The words below are English business vocabulary and
            # they collide across industries: "ship" in scholarship, "tax" in taxonomy, "duty" in
            # duty_roster. A token start keeps shipping_amount and freight_fee and drops those.
            r = cls._ROLE_CACHE[col] = next(x for x, pat in cls.AMT_ROLE if re.search(rf"(?:^|_)(?:{pat})", c))
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

    #: Appended to every "could not stage" line, and it is not padding.
    #:
    #: All three staging failures are already swallowed - each `except` below says so in its own
    #: comment ("never block generating on a staging problem", "validation is a convenience, never
    #: a gate"). That fact lived only in the comments. What reached the reader was a sentence
    #: naming an internal step and a DuckDB error, with a `!` for severity and nothing to say
    #: whether it mattered.
    #:
    #: A measured run spent a full turn on it, hypothesising five different mechanisms in a row
    #: ("maybe staging executes CREATE TABLE in resolvable order", "maybe it validates
    #: incrementally", "maybe the validator has trouble with my statement's complexity"), then
    #: gave up and went into this file to read the parser - which is exactly what SKILL.md's
    #: budget section lists first among the ways to lose a run. The answer it was looking for is
    #: one clause long and belongs in the message.
    #: ⚠️ No cause is named here on purpose. It used to say "usually foreign-key order", which
    #: `_create_until_stuck` has since made false - ordering is retried until it stops helping, so
    #: a failure that survives to this line is NOT an ordering problem. It is also appended to
    #: `_stage_summary_layer`, where the subject is extra_sql and foreign keys are irrelevant.
    #: What both calls can honestly say is that it does not block anything.
    _STAGING_IS_ADVISORY = "\n    (advisory only: generation is unaffected and the profile is not necessarily wrong.)"

    def _scratch_schema(self):
        """An empty in-memory copy of the schema this run will build, for planning SQL against.

        Declared tables keep their real CREATE text so constraints and exact types are the ones
        the statements will actually meet; synthetic tables (date dimension, summary layer) are
        rebuilt from the inferred column list. Foreign keys force an order, so creation retries
        once after everything else exists.
        """
        con = duckdb.connect(":memory:")
        # The table NAME travels with the statement. Without it the warning below could only quote
        # DuckDB's error, which names whichever table the failing one REFERENCED - so it pointed at
        # the neighbour rather than at the table that failed.
        items = []
        for t, cols in self.schema.items():
            sql = getattr(self, "decl_sql", {}).get(t)
            if not sql:
                body = ", ".join(f'"{c["name"]}" {c["type"]}' for c in cols)
                sql = f'CREATE TABLE "{t}" ({body})'
            items.append((t, sql))

        def _stage(item):  # noqa: ANN001 - local helper
            try:
                con.execute(item[1])
                return None
            except Exception as e:  # noqa: BLE001 - almost always a not-yet-created FK target
                return e

        # Same repeated-pass rule as `_parse`: a foreign-key chain more than one level deep needs
        # more than one retry, and a table left unstaged here silently stops validating the SQL
        # that touches it.
        for (t, _sql), _first, failure in self._create_until_stuck(items, _stage):
            logger_msg = (str(failure).splitlines() or [""])[0]
            print(
                f"  ! pre-check could not stage `{t}`, so statements touching it went "
                f"unvalidated: {logger_msg}{self._STAGING_IS_ADVISORY}"
            )
        return con

    def _stage_summary_layer(self, con):
        """Build the auto summary layer on the scratch database, so ``extra_sql`` can reference it.

        ``extra_sql`` runs *after* the summary layer and post-processing those tables is its main
        use, but they are created during ``build_db`` and never appear in ``self.schema`` - so
        validating against the schema alone reported every legitimate reference as "table does not
        exist", and ``generate()`` calls ``precheck`` in strict mode. That made ``extra_sql``
        unusable on the ``summary`` / ``all`` paths, which is worse than not validating it.

        Returns True when ``extra_sql`` can be validated. On any failure it says so and returns
        False: a statement this cannot stage is one it has no business judging.
        """
        if self.extra_tables not in ("summary", "all"):
            return True
        made = getattr(self, "_made", None)
        try:
            # ``_auto_summary_sql`` rewrites ``self._made``; generate() recomputes it, but precheck
            # must not leave a trace either way.
            sql = self._auto_summary_sql()
            for statement in con.extract_statements(sql) if sql.strip() else []:
                con.execute(statement.query)
            return True
        except Exception as e:  # noqa: BLE001 - never block generating on a staging problem
            print(
                "  ! pre-check could not stage the summary layer, so extra_sql was not validated: "
                + (str(e).splitlines() or [""])[0]
                + self._STAGING_IS_ADVISORY
            )
            return False
        finally:
            if made is None:
                self.__dict__.pop("_made", None)
            else:
                self._made = made

    def _sql_problems(self):
        """Validate the SQL blocks once per engine and remember the answer.

        ``precheck()`` and ``report()`` both need it and must not disagree, and ``report()`` is
        the surface the skill tells the agent to run first - a claim there that validation
        happened, printed by an engine whose ``__init__`` stops at ``_infer()``, would be a
        statement about work nobody had done.
        """
        if not hasattr(self, "_sql_problem_cache"):
            self._sql_problem_cache = self._validate_sql_blocks()
        return self._sql_problem_cache

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
            print(
                "  ! pre-check could not stage the schema, so no statement was validated: "
                + (str(e).splitlines() or [""])[0]
                + self._STAGING_IS_ADVISORY
            )
            return []

        problems = []
        try:
            for key, value in blocks:
                if key == "extra_sql" and not self._stage_summary_layer(con):
                    # The summary layer could not be staged, so every reference to it would read as
                    # "table does not exist". Skipping beats blocking a configuration that is
                    # actually fine. ``continue`` rather than ``break``: skipping this block should
                    # not depend on it happening to be the last one in the list.
                    continue
                if not value:
                    continue
                if isinstance(value, (list, tuple)):
                    statements = [str(x).strip().rstrip(";") for x in value if str(x).strip()]
                else:
                    try:
                        statements = [st.query.strip().rstrip(";") for st in con.extract_statements(str(value))]
                    except Exception as e:  # noqa: BLE001 - a parse error IS the finding
                        problems.append(f"{key}: cannot be parsed as SQL - " + (str(e).splitlines() or [""])[0])
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
                        problems.append(f"{key}[{i}]: " + (str(e).splitlines() or [""])[0] + f"  <-  {head}")
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
                        rng.choices([e[self.ref_col_of(t, name)] for e in up], [e["__w__"] ** 0.55 for e in up])[0]
                        if up
                        else ""
                    )
                elif sem == "name":
                    ent[name] = self._name_for(t, i, rng, ent)
                    if self._is_declared_unique(t, name):
                        ent[name] = self._unique_name(t, name, ent[name])
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
                    # Reads `columns['t.col']['range']` like amount and count do. It did not, so the
                    # slot the skeleton writes and the warning precheck prints both asked the caller
                    # to set a value that changed nothing: a GPA declared (0.0, 4.0) still came out
                    # at 49.02. The two measure paths also disagreed on the fallback - 0.1-50 here
                    # against 0.1-40 in `_fill_generic` - so neither matched what was documented.
                    ent[name] = self._measure_val(t, name, rng, ent)
                elif self._is_code_col(t, name):
                    ent[name] = self._code_val(t, name, i)
                else:
                    ent[name] = f"{name}_{i + 1}"
            self._set_alt_keys(t, ent)
            self._split_opposed_values(t, ent, rng)
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

    def _pool_entity(self, table, col, value):
        """The pooled parent row whose ``col`` holds ``value``, or None.

        Indexed once per (table, col) - `_inherit_denormalised` needs to repoint an entity for
        every fact row, and a linear scan of a dimension pool per row is not affordable. The pools
        a fact samples are built before it runs, so the index cannot go stale within a pass; it is
        dropped whenever generation restarts (see `generate`'s calibration retry).
        """
        idx = self.__dict__.setdefault("_pool_key_idx", {})
        key = (table, col)
        if key not in idx:
            idx[key] = {e[col]: e for e in self.pools.get(table, ()) if col in e}
        return idx[key].get(value)

    def _inherit_denormalised(self, t, row, ent):
        """Copy a denormalised column down from the parent row it belongs to.

        ⚠️ A FACT THAT CARRIES BOTH `route_id` AND `origin_code` IS STATING THAT THEY AGREE. Each
        was sampled independently - the flight drew a route, then drew an airport that had nothing
        to do with it - so the row said its route starts at one airport and the flight at another.
        Measured: 95% of rows disagreed with the route they pointed at. Every join through the two
        keys returns a different answer, and no assertion in `checks.json` can express the
        contradiction because each column is individually legal.

        ⚠️ THE ENTITY MOVES WITH THE VALUE. Rewriting `row` alone leaves `ent[other]` pointing at
        the airport that was just discarded, and everything downstream reads attributes off that
        entity - `_gen_fact`'s enum backfill, and `_cond_pick`'s cross-table `__by__`. Measured
        after a first attempt that only rewrote `row`: the route/airport contradiction was gone and
        a new one had grown in its place, 4,168 of 5,640 rows carrying a `region` belonging to the
        abandoned airport. Trading one inconsistency for another is not a fix.

        Deliberately narrow, because the alternative is clobbering columns that only happen to
        share a name. All three must hold: the column is a DECLARED foreign key of this table, the
        table also has a declared foreign key to some parent P, and P carries a column of the same
        name that was populated.
        """
        declared = {col for (tt, col) in (getattr(self, "decl_fk", None) or {}) if tt == t}
        # ``list(...)`` on BOTH loops: the body can drop an entry, and CPython validates the dict's
        # size on every ``__next__`` including the one that would raise StopIteration - so mutating
        # it here is not a race that sometimes survives, it is a guaranteed RuntimeError.
        for fk_col, parent_row in list(ent.items()):
            if fk_col not in declared:
                continue
            parent = self.pk_owner.get(fk_col)
            if not parent:
                continue
            parent_cols = {c["name"] for c in self.schema.get(parent, ())}
            for other in list(ent):
                if other == fk_col or other not in declared or other not in parent_cols:
                    continue
                inherited = parent_row.get(other)
                if inherited is None:
                    continue
                # ⚠️ RESOLVE FIRST, WRITE SECOND. The parent's copy of this column is not always a
                # value THIS column may hold: a parent and child can reference different columns of
                # the same table, so `p.x` holds `a.code` while `f.x` must hold `a.alt`. Writing
                # the inherited value and then finding no entity for it leaves `row[other]` naming
                # a row that does not exist - the exact thing these tests assert against - and the
                # independently sampled value it replaced was at least self-consistent. When the
                # inheritance cannot be resolved, the right move is to not inherit.
                moved = self._pool_entity(self.pk_owner.get(other), self.ref_col_of(t, other), inherited)
                if moved is None:
                    continue
                row[other] = inherited
                ent[other] = moved

    # The two ends of a directed edge. A pair of foreign keys to the SAME parent whose names read
    # like these must not land on the same entity: a route from an airport to itself is not a
    # route. Deliberately a short list of opposed words rather than "any two keys to one parent" -
    # `orders(billing_address_id, shipping_address_id)` points twice at `addresses` and being equal
    # there is the common case, not a defect.
    EDGE_FROM = re.compile(r"(^|_)(origin|orig|from|source|src|depart|departure|start|sender)(_|$)")
    EDGE_TO = re.compile(r"(^|_)(destination|dest|dst|to|target|tgt|arrive|arrival|end|receiver)(_|$)")

    def _opposed_fk_pairs(self, t):
        """Foreign-key column pairs on ``t`` that are the two ends of one directed edge.

        ⚠️ Each end is sampled independently, so they collide: measured on the flights DDL, 1 route
        in 13 had ``origin_airport_code == destination_airport_code`` and 403 flights were booked
        on it. A production run found it, spent three attempts trying to repair it with ad-hoc
        UPDATEs - all three refused, twice by the foreign key still being referenced - and the
        self-loop shipped anyway.
        """
        cache = self.__dict__.setdefault("_opposed_cache", {})
        if t not in cache:
            own = [c["name"] for c in self.schema.get(t, ())]
            pairs = []
            for a in own:
                if not self.EDGE_FROM.search(a):
                    continue
                par = self.pk_owner.get(a)
                if not par or par == t:
                    continue
                for b in own:
                    if b != a and self.EDGE_TO.search(b) and self.pk_owner.get(b) == par:
                        pairs.append((a, b, par))
            cache[t] = tuple(pairs)
        return cache[t]

    def _split_opposed_values(self, t, row, rng):
        """Redraw the far end when both ends of a directed edge landed on the same PARENT.

        ⚠️ Compared as entities, not as values. The two ends may reference different unique columns
        of the same parent - `origin_port_id` at `ports.port_id`, `destination_iata` at
        `ports.iata` - and then two values that name the SAME port are never equal, so a value
        comparison finds nothing to fix and the self-loop ships. Excluding the resolved parent also
        removes the retry loop that could, on a two-row pool, return the same end every time.
        """
        for a, b, par in self._opposed_fk_pairs(t):
            up = self.pools.get(par)
            if not up or len(up) < 2 or a not in row or b not in row:
                continue
            near = self._pool_entity(par, self.ref_col_of(t, a), row[a])
            far = self._pool_entity(par, self.ref_col_of(t, b), row[b])
            if near is None or far is not near:
                continue
            other = [e for e in up if e is not near]
            if other:
                row[b] = rng.choice(other)[self.ref_col_of(t, b)]

    def _split_opposed_entities(self, t, ent, rng):
        """Same, where the row carries resolved parent entities rather than bare values.

        The entity is redrawn, not just the value written into the row: ``ent`` feeds the
        effective-date floor, `conditional` grouping and denormalised inheritance, and rewriting
        one without the other is how a previous fix put a value in the row that belonged to no
        parent at all.
        """
        for a, b, par in self._opposed_fk_pairs(t):
            up = self.pools.get(par)
            if not up or len(up) < 2 or a not in ent or b not in ent:
                continue
            if ent[b] is not ent[a]:
                continue
            other = [e for e in up if e is not ent[a]]
            if other:
                ent[b] = rng.choice(other)

    def ref_col_of(self, t, fk_col):
        """The parent column this foreign key actually points at.

        ⚠️ NOT the parent's PRIMARY KEY, which is what every sampling site used to read. SQL lets a
        foreign key reference any unique column, `_scan_constraints` has always captured which one
        as the second half of its `decl_fk` value, and nothing consumed it - so a child pointing at
        a UNIQUE non-PK column was filled from the parent's PK instead and matched nothing:
        measured at 100% orphans, against 0% for the same DDL with the target declared PRIMARY KEY.

        Falls back to the PK, which is both the overwhelmingly common case and what an INFERRED
        (undeclared) foreign key means.
        """
        owner = self.pk_owner.get(fk_col)
        declared = (getattr(self, "decl_fk", None) or {}).get((t, fk_col))
        # ⚠️ ONLY when the declared parent IS the pool being sampled. `pk_owner` is keyed by column
        # name across the whole schema, so a column named `code` can be owned by `airports` while
        # this table's FK declares `countries(iso2)`. Returning the declared column then indexes an
        # `airports` row by `iso2` - KeyError, and the whole generate dies. Picking the wrong
        # parent is a pre-existing inaccuracy; crashing on it would be a new failure.
        if declared and declared[0] == owner:
            ref_c = declared[1]
            if ref_c and any(c["name"] == ref_c for c in self.schema.get(owner, ())):
                return ref_c
        return self.pk_of(owner) if owner else fk_col

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
        # In-flight statuses (mid-lifecycle) can only occur near the cut-off date -
        # an order from a year ago cannot still be pending/shipped
        life = self._lifecycle_prep(t)
        hard_cap = life["cap"]
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
            self._split_opposed_entities(t, ent, rng)
            lo_i = max((e.get("__eff__") or 0) for e in ent.values()) if ent else 0
            if lo_i and (d - self.start).days < lo_i:  # fallback for a day with no effective entity at all
                d = self.days[self._pick_day_ge(lo_i, rng)]
            row = {pk: self._pk_val(t, i, d)}
            for f, e in ent.items():
                row[f] = e[self.ref_col_of(t, f)]
            self._inherit_denormalised(t, row, ent)
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
            self._lifecycle_settle_status(life, row, t0, status_col, st_vals, st_w, rng)
            ets = max((e["__eff_ts__"] for e in ent.values() if e.get("__eff_ts__")), default=None)
            if ets is not None and t0 < ets:  # even same-day, it cannot precede the exact registration/listing moment
                t0 = min(ets + timedelta(minutes=rng.randint(2, 720)), hard_cap)
            self._lifecycle_chain(life, row, t0, status_col, rng)
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
                row[c] = (
                    self._derive(t, c, row, rng)
                    or self._count_val(t, c, rng, row, ent)
                    or rng.choices([1, 2, 3, 4, 5], [0.52, 0.26, 0.12, 0.06, 0.04])[0]
                )
            for c in ratio_cols:
                row[c] = round(bounded_gauss(rng, 0.04, 0.013, 0.008, 0.092), 4)
            for c in flag_cols:
                row[c] = 1 if rng.random() < self._col_profile(t, c).get("p", 0.88) else 0
            for c in other_cols:
                row.setdefault(c["name"], self._fill_generic(t, c, rng, {"dt": d, "ts": t0}, row, ent))
            for c in names:
                if c not in row and self._is_code_col(t, c):
                    row[c] = self._code_val(t, c, i, d)
                row.setdefault(c, "")
            self._set_alt_keys(t, row)
            self._apply_formulas(t, row)
            rows.append([row[c] for c in names])
            refs.append(
                {
                    "pk": row[pk],
                    "alt": self._ref_alt(t, row),
                    "dt": d,
                    "ts": t0,
                    "status": row.get(status_col, ""),
                    "amt": base,
                    "fks": {f: row[f] for f in self.fks[t]},
                    "subj": ent.get(subj_col, {}).get("__w__", 1.0) if subj_col else 1.0,
                    **self._ref_extra(t, row),
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

    # How many lines a document gets. The spread is what makes the long tail - most orders have one
    # line, a few have five - and its mean is 1.84.
    LINE_SPREAD = ([1, 2, 3, 4, 5], [0.52, 0.26, 0.12, 0.06, 0.04])
    LINE_SPREAD_MEAN = 1.84

    def _lines_for(self, lines_per_doc, rng):
        """Lines on this document, for a requested average of ``lines_per_doc``.

        The draw used to ignore the request entirely below two lines per document: it returned the
        raw spread, mean 1.84, whatever the plan said. That made a detail table the one table
        calibration could not move - it rescales ``nrows`` between passes, and this generator was
        not reading it. Two production runs pinned their other tables, left the detail table free to
        absorb the difference, and shipped 90,004 and 90,073 rows against a requested 80,000, the
        same total on all three passes because every pass produced identical rows.

        The spread is kept and its *extra* lines are scaled to the requested mean, so the first line
        is never scaled away: a document has at least one line by construction, rather than by a
        clamp that would push the mean back above what was asked for. The fractional part is
        resolved by a coin flip so the mean is exact rather than rounded off. At 1.84 lines per
        document this is the raw spread again. A plan below one line per parent is not reachable -
        ``precheck`` says so.
        """
        spread = rng.choices(*self.LINE_SPREAD)[0] - 1
        extra = max(0.0, lines_per_doc - 1.0) * spread / (self.LINE_SPREAD_MEAN - 1)
        # No upper clamp: the draw is already bounded by the spread's own top (five lines scaled),
        # and a ceiling only truncated the tail - it cost 2% of the requested mean at three lines
        # per document and 3% at four, on the tables where the caller had asked for the most.
        return 1 + int(extra) + (1 if rng.random() < extra - int(extra) else 0)

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
        lines_per_doc = self.nrows[t] / max(1, len(prefs))  # fractional: calibration moves it in small steps
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
        # A declared `lifecycle` takes over the row's timestamps and status, exactly as on a fact;
        # without one the timestamps stay anchored to the parent by `_fill_generic` as before.
        life = self._lifecycle_prep(t)
        status_col = next((c["name"] for c in cols if c["sem"] == "enum" and "status" in c["name"]), None)
        st_vals, st_w = self._enum_values(t, status_col) if (life["declared"] and status_col) else ([], [])
        rows, agg, no, refs = [], {}, 0, []
        exact = (getattr(self, "fixed_fanout", None) or {}).get(t)
        for pi, pr in enumerate(prefs):
            # A declared fan-out is a count, not a mean: `_lines_for` spreads around it, which is
            # right for order lines and wrong for the 16 wafers in a lot.
            k = exact if exact else self._lines_for(lines_per_doc, rng)
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
                row = {pk: self._pk_val(t, no - 1), fk: self._parent_key_val(t, fk, par, pr)}
                ents = {item_col: e} if (item_col and e) else {}
                if pr.get("cols"):
                    ents[fk] = pr["cols"]  # the parent row, for a `__by__` that crosses to it
                ents = ents or None
                if item_col and e:
                    row[item_col] = e[pool_pk]
                for k, v in self._joint_draw(t, rng).items():
                    row.setdefault(k, v)
                own_t0 = None  # set when this row runs its own lifecycle; its children anchor to it
                if life["declared"]:
                    if status_col and status_col not in row:  # a joint group may have supplied it
                        row[status_col] = self._pick_enum(t, status_col, row, ents, rng)
                    anchor = pr.get("ts")
                    own_t0 = (
                        min(anchor + timedelta(seconds=rng.randint(30, 5400)), life["cap"])
                        if anchor is not None
                        else day_ts(rng, pr["dt"])
                    )
                    self._lifecycle_settle_status(life, row, own_t0, status_col, st_vals, st_w, rng)
                    self._lifecycle_chain(life, row, own_t0, status_col, rng)
                qty = (self._count_val(t, qty_col, rng, row, ents) if qty_col else None) or rng.choices(
                    [1, 2, 3, 4], [0.71, 0.19, 0.07, 0.03]
                )[0]
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
                        row[nm] = self._fill_generic(t, c, rng, pr, row, ents)
                tot["gross"] += (
                    sales + disc
                )  # pre-discount goods amount (list price x qty), feeds the parent "original" column
                tot["net"] += sales  # post-discount goods amount = original - discount
                tot["discount"] += disc
                tot["cost"] += tcost
                tot["refund"] += ramt
                self._set_alt_keys(t, row)
                self._apply_formulas(t, row)
                rows.append([row[c] for c in names])
                refs.append(
                    {
                        "pk": row[pk],
                        "alt": self._ref_alt(t, row),
                        # A row that ran its own lifecycle is the anchor for ITS children: a note
                        # on a claim follows the claim's timestamps and status, not the order's.
                        "dt": own_t0.date() if own_t0 else pr["dt"],
                        "ts": own_t0 if own_t0 else pr.get("ts"),
                        "status": row.get(status_col, "") if (own_t0 and status_col) else pr.get("status", ""),
                        "amt": sales,
                        "fks": {f: row.get(f, "") for f in self.fks[t]},
                        "subj": 1.0,
                        **self._ref_extra(t, row),
                    }
                )
            agg[pr["pk"]] = {k2: round(v, 2) for k2, v in tot.items()}
        o.write(t, names, rows)
        # A detail table is a legitimate parent - order_items has serials, a claim has line items.
        # Without these two lines `_parent_of` could not see it, so `_gen_detail` fell through to
        # `_gen_fact` for the nested table and generated it as an independent fact: 10,754 rows with
        # the foreign key NULL on every one of them, which the FK check then reports as resolving.
        self.refs[t] = refs
        self._fact_rows[t] = (names, rows)
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

    def _fill_generic(self, t, c, rng, pr=None, row=None, ents=None):
        """``row`` / ``ents`` are the grouping context `conditional` needs - the row being built
        and its resolved parents. Optional because most callers fill a column that does not read
        them; a measure column does (see `_measure_val`)."""
        sem, name = c["sem"], c["name"]
        if sem == "enum":
            vals, ws = self._enum_values(t, name)
            return rng.choices(vals, ws)[0]
        if sem == "count":
            v = self._count_val(t, name, rng, row, ents)
            return v if v is not None else rng.randint(1, 20)
        if sem == "ratio":
            return round(bounded_gauss(rng, 0.04, 0.013, 0.008, 0.092), 4)
        if sem == "flag":
            return 1 if rng.random() > 0.12 else 0
        if sem == "amount":
            return round(lognorm_between(rng, 5, 300), 2)
        if sem == "measure":
            return self._measure_val(t, name, rng, row, ents)
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
            row = {pk: self._pk_val(t, i), fk: self._parent_key_val(t, fk, par, pr)}
            for f in self.fks[t]:
                if f == fk or f not in self.pk_owner:
                    continue
                up = self.pools.get(self.pk_owner[f])
                if up:
                    e = rng.choices(up, cum_weights=self._pool_cum(self.pk_owner[f]))[0]
                    row[f] = e[self.ref_col_of(t, f)]
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
            for k, v in self._joint_draw(t, rng).items():
                row.setdefault(k, v)
            for c in cols:
                if c["name"] in row:
                    continue
                row[c["name"]] = self._fill_generic(t, c, rng, {"dt": s_dt}, row)
            self._set_alt_keys(t, row)
            self._apply_formulas(t, row)
            rows.append([row.get(c, "") for c in names])
            refs.append(
                {
                    "pk": row[pk],
                    "alt": self._ref_alt(t, row),
                    "dt": s_dt,
                    "end": e_dt if done else None,
                    "status": row.get(status_col, ""),
                    "amt": 0,
                    "fks": {},
                    "done": done,
                    "span": transit,
                    **self._ref_extra(t, row),
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
                row = {pk: self._pk_val(t, no - 1), fk: self._parent_key_val(t, fk, par, pr)}
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
                    row[c["name"]] = self._fill_generic(t, c, rng, {"dt": ts.date()}, row)
                self._set_alt_keys(t, row)
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
                    row[c["name"]] = self._fill_generic(t, c, rng, {"dt": d}, row)
                self._set_alt_keys(t, row)
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
        fixed_fanout = getattr(self, "fixed_fanout", None) or {}
        free = [
            t
            for t in self.nrows
            if self.roles[t] not in (ROLE_DATE, ROLE_DIM, ROLE_METRIC)
            and t not in getattr(self, "pinned_rows", {})
            and t not in fixed_fanout  # derived from its parent, so it follows rather than scales
        ]
        if abs(dev) > tolerance and _attempt < 3 and free:
            k = self.rows / total
            for t in free:
                self.nrows[t] = max(50, int(self.nrows[t] * k))
            self._apply_fanout(self.nrows)  # the tree follows its root
            self.rng = random.Random(self.seed)
            self.pools, self.refs, self._fact_rows, self._pending_dim_rows = {}, {}, {}, {}
            self.__dict__.pop("_pool_key_idx", None)
            self.__dict__.pop("_name_seen", None)
            self._alt_seq = {}  # or the retry hands out values continuing from the discarded pass
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
                extra += " (reused the previous row allocation; every row was generated fresh)"
            # Warn from the tolerance, not from 25%: a production run shipped 90,004 rows against a
            # requested 80,000 (+12.5%) and nothing said a word, because four of five tables were
            # pinned and calibration had nothing left to scale.
            if abs(dev) > tolerance:
                print(
                    f"  ! actual rows differ from target by {100 * dev:+.0f}%, outside the "
                    f"{100 * tolerance:.0f}% tolerance, after {_attempt} pass(es). "
                    f"Usual causes: the main fact table was classified as a dimension/metric table, "
                    f"or pinned table_rows values leave calibration nothing to scale."
                    f"\n    Check the roles in report() and override with profile['roles'] if needed."
                )
            print(
                f"  {len(sizes)} tables / {res['rows']:,} rows (target {self.rows:,}, deviation "
                f"{100 * dev:+.1f}%, {_attempt} pass(es)) | generate {t_gen:.2f}s "
                f"build {res['t_db']:.2f}s total {res['t_total']:.2f}s{extra} | "
                f"{Path(out).stat().st_size / 1e6:.1f} MB | {self.start} ~ {self.end}"
            )
            print(self._next_step(out))
        return res

    @staticmethod
    def _next_step(out):
        """What to do with the database that was just built.

        ⚠️ THE ONE PLACE THIS INSTRUCTION IS READ AT THE MOMENT IT APPLIES. SKILL.md carries the
        same rule, and SKILL.md is loaded once, at the top of a session that then runs for dozens
        of turns - by the time this decision is made it is tens of thousands of tokens back. This
        line is printed by every build, immediately before the reader picks what to do next.

        And what they pick is the whole cost. Two measured runs, two models, same shape: 79% of the
        wall clock went to the stretch BEFORE the first `import_database_file`, and once the check
        was finally run it converged in three rounds either way (13 and 7 minutes). Neither was
        stuck - both were verifying, by hand. One wrote duckdb queries against successive
        `raw2/raw3/raw4` builds for 43 minutes; the other read the engine's source for 33. The
        check answers the same questions in one call and its assertions re-run for free on the next
        build, which is why SKILL.md lists hand-written verification among the three ways to lose a
        run.
        """
        return (
            f"  NEXT: import_database_file(path={out!r}, mode='replace')\n"
            f"        then check_datasource_quality(config_path='data/checks.json')\n"
            f"  Run these now, before inspecting the build. Hand-written queries against it answer"
            f" one question each and are paid for every time; the check answers all of them in one"
            f" call and re-runs for free."
        )

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
