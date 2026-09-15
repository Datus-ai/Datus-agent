---
name: gen-datasource
description: Generate a realistic synthetic data warehouse (single DuckDB database) from a business description and a row budget, then load it into the active datasource. The output is meant to be queried - by an AI agent answering data questions, by a business report, or by a dashboard - so it carries trend/seasonality/promotion signals, attributable anomalies, Zipf long tails and cross-domain key joins, and it ships with a data dictionary and a quality report. Use when the user asks to build a demo database, seed a datasource, create sample/mock/test business data, or generate data from DDL.
tags:
  - datasource
  - synthetic-data
  - demo
  - duckdb
version: "1.0.0"
user_invocable: true
disable_model_invocation: false
# Claude Code compatibility metadata. Datus does not wire this into BashTool - it mounts bash with
# agent_config.bash_allowed_patterns - so on Datus the real gate is PermissionManager. Declared so
# the skill behaves the same on a host that does enforce it, and kept in step with the commands this
# file actually asks for. Both spellings: the matcher compares argv[0] literally, so a "python:"
# rule alone would never match the "python3 ..." an agent types.
allowed_commands:
  - "python:data/gen.py"
  - "python3:data/gen.py"
---

# Generate a Synthetic Datasource

**Goal**: given an industry (optionally with table/column names or DDL) and a row budget, produce one DuckDB database whose structure looks like a real warehouse, whose curves look like a real business, and whose anomalies can be explained - then load it into the active Datus datasource so the agent can query it immediately.

---

## Step 0: you do not need to locate this skill

`load_skill` returned this file behind a `<skill_location>` line naming the directory it came from.
**That path is the answer to "where is the engine / where are the references".** Use it verbatim.
Never reconstruct it and never guess the interpreter version - a production run guessed
`python3.11` on a `python3.12` install and paid a rejected `read_file` plus a `bash` round-trip to
learn what the first line of its own context already said.

For `gen.py` you do not need even that: the skill ships inside the `datus` package, so the file
resolves the engine itself at import time. Copy these four lines and never search the filesystem
for the skill directory:

```python
import pathlib, sys
import datus
SKILL = pathlib.Path(datus.__file__).resolve().parent / "resources" / "skills" / "gen-datasource"
sys.path.insert(0, str(SKILL / "scripts"))
from ddl_engine import DDLEngine
```

### When this file does not answer your question

This file is the contract, and it is complete for writing a profile - it arrives whole when the
skill is loaded, so the normal path opens nothing. **Before opening anything, call
`plan_datasource(ddl=...)`**: it prints the engine's plan from the DDL alone, which is what most
questions about the engine are really asking, and it needs no generator on disk.

Two facts about the environment shape what to do when you still need more:

- **A packaged deployment strips Python source to `.pyc`.** `ddl_engine.py` may simply not exist on
  disk; the engine still imports and runs, but there is nothing to read.
- The long-form documents next to the engine survive that strip and are **readable directly with
  `read_file`** - the skill bundle is a read-only whitelist anchor, so no copying is needed:

  ```
  read_file("<skill_location>/references/profile-spec.md")
  ```

  `references/profile-spec.md` is every profile field in full; `references/pitfalls.md` is the 17
  invariants and the distortion root-cause table. Use `read_file`, not `cat` - a file this size
  comes back from bash as an archived-output stub.

**Never disassemble the engine.** A measured production run spent **10 minutes - 39% of its wall
clock - running `marshal` and `dis` over `ddl_engine.pyc`** to work out one undocumented profile
key. If a knob you need is not in this file or in `references/profile-spec.md`, it is not a knob:
express the rule with `conditional` / `formulas` / `derive`, or fall back to `pre_sql`, and say so
in the delivery summary. Reverse-engineering bytecode is never the answer, and neither is guessing.

---

## Your time budget

**Your own output tokens are the clock.** Tool execution is ~2% of this job: generation takes a
couple of seconds, import about one, the quality check under one. Everything else is you writing.
At the measured throughput of a production run, **every 4,000 tokens you emit costs a minute**.

A complete run fits in roughly 40,000 output tokens:

| | budget |
|---|---|
| Read this file, decide the path, design the profile | ~12,000 |
| Write `data/gen.py` (DDL + profile + main) | ~8,000 |
| Write `data/checks.json` and `data/README.md` | ~10,000 |
| Inspect the report, fix what the checks flag, deliver | ~10,000 |

The three things that blow the budget, all measured on real runs:

1. **Reading the engine to predict its behaviour.** `plan_datasource(ddl=...)` already prints every
   decision - roles, row allocation, column semantics, name samples, the metric grid, which
   declarative rules resolved - from the DDL alone, before `gen.py` exists. Read that output
   instead. Reconstructing the same facts from the implementation cost
   one run 66% of its wall clock, and a second run **all** of it: it tried to divide the row budget
   across the tables by hand, could not make the total come out, went into the engine to find the
   allocator and never came back - 36 turns, zero rows. **Row allocation is the engine's job**
   (Phase 1.2). The cost of one `gen.py report` is 0.3 seconds; the cost of predicting it is your
   whole budget.
2. **Continuing after the checks pass.** See "Stop as soon as it passes" in Phase 5 - 32% of one
   run's wall clock went into rounds that changed nothing.
3. **Ad-hoc verification queries.** Assertions in `checks.json` re-run for free; a hand-written
   query is paid for every time. See the budget in Phase 6.

---

## Hard constraint: data comes from the engine

**The only legal way to generate**: write one `data/gen.py` and run it. This is the whole file -
copy it, replace the DDL and the profile, change nothing else:

```python
#!/usr/bin/env python3
"""<business scenario> demo datasource. Fixed seed, re-runnable, all rules in PROFILE."""
import pathlib
import sys

import datus

SKILL = pathlib.Path(datus.__file__).resolve().parent / "resources" / "skills" / "gen-datasource"
sys.path.insert(0, str(SKILL / "scripts"))
from ddl_engine import DDLEngine  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent            # data/
ARG = sys.argv[1] if len(sys.argv) > 1 else ""
OUT = HERE / "_build" / "datasource.duckdb" if ARG in ("", "report") else pathlib.Path(ARG)

DDL = """
CREATE TABLE ...;                                          -- DuckDB syntax; see Step 1
"""

PROFILE = {
    "calendar": {...},                                     # always
    "semantics": {...},                                    # whatever report() got wrong
    "conditional": {...}, "derive": {...}, "formulas": {...},
}

eng = DDLEngine(DDL, rows=80_000, profile=PROFILE, months=17, seed=42)
eng.report()                                               # always print the plan
if ARG == "report":                                        # `gen.py report` stops here
    raise SystemExit(0)
OUT.parent.mkdir(parents=True, exist_ok=True)
result = eng.generate(str(OUT))
print(result["rows"], "rows |", len(result["tables"]), "tables |", result["degraded"] or "constraints kept")
```

**`gen.py report` prints the engine's whole plan** - roles, row allocation per table, column
semantics, sample generated names, which columns get business codes, the daily-metric grid, and
which `conditional` / `derive` / `joint` rules resolved. **Read that instead of reading the engine.**
It is the answer to "what will this actually produce", and it costs 0.3 seconds.

**Forbidden**:

| Forbidden | Why |
|---|---|
| Hand-written `INSERT` / `CREATE TABLE AS SELECT ... FROM range()` to produce rows | Bypasses all 17 invariants: no trend/weekday/promotion signal in dates, no Zipf tail on entities, event chains not monotonic, foreign keys not sampled from upstream, no stock baseline. The result queries fine and looks fake at a glance |
| Splitting generation across numbered SQL files (`01_xxx.sql`, `02_xxx.sql`, ...) | Same as above, plus not reproducible (no fixed seed), not parameterizable (changing the row count means rewriting every file), and no two-pass row calibration |
| Pushing rows in through a SQL tool | Same as above |

The **only** place SQL is allowed is the profile's `pre_sql` / `extra_sql`, and only for business post-processing (metric restatement, cross-table backfill, derived summary tables) - never to produce primary data.

The test is simple: **if you are writing SQL that decides how one row comes into existence, you are on the wrong path.** The engine makes the data; your job is to tell it the business rules through the profile.

---

## Running inside Datus: how the database reaches the datasource

The active datasource's DuckDB file is held open by the agent process, so a second
writer cannot take the lock:

```
IOException: Could not set lock on file "...": Conflicting lock is held in python (PID N)
```

**Never try to generate directly into the datasource file.** The flow is:

0. **Plan before you write anything**, as soon as the DDL is in DuckDB syntax:
   ```
   plan_datasource(ddl=<the normalised DDL>, rows=80000, months=17)
   ```
   It returns the engine's whole plan - row allocation per table, table roles, column semantics,
   the resolved date window, sample names, business codes, the daily-metric grid - and generates
   nothing. **This is the answer to every "what will the engine do with my DDL" question, and it
   is one call.** Read it, then write `gen.py` with a profile that corrects what it got wrong.
1. Generate to a build path inside the workspace: `python3 data/gen.py data/_build/datasource.duckdb`
   (`gen.py` takes the output path as its first argument and creates the parent itself, so the
   command stays a single invocation with no shell chaining)
2. Load it into the datasource with the built-in tool, which runs through the
   connection that already holds the lock and replays the source DDL so primary
   keys, unique and foreign keys survive:
   ```
   import_database_file(path="data/_build/datasource.duckdb", mode="replace")
   ```
3. Verify with `check_datasource_quality(config_path="data/checks.json")` - it searches
   `data/` and `data/_build/` for `.datasource.meta.json` itself and reports which one it used,
   so do not pass `meta_path`
4. Keep the metadata, then delete the build directory - the engine writes
   `.datasource.meta.json` **next to the database**, so it is inside `_build/`, and
   `check_datasource_quality` needs it to verify the declared keys:
   ```
   cp data/_build/.datasource.meta.json data/.datasource.meta.json && rm -rf data/_build
   ```

**Do not generate a second copy at `data/datasource.duckdb`.** The datasource already holds the
data after step 2, and in a Datus deployment that path is often the datasource's own file - this
process has it open, so writing it corrupts the handle and can strand the old copy on disk as an
`.nfs*` orphan. A production run did exactly that, then failed importing it back with
`Unique file handle conflict`, having spent a full extra generate cycle on it. **`data/gen.py` is
what makes the database reproducible**, not a second copy of the bytes.

`import_database_file` copies table and column comments too, so the comments the
engine wrote arrive with the data - do not re-issue `COMMENT ON` by hand.

---

## Deliverables

```
./
└── data/
    ├── README.md               <=150 lines: what the schema cannot say about itself
    ├── gen.py                  generator incl. profile; fixed seed, re-runnable
    ├── checks.json             business assertions for check_datasource_quality
    └── .datasource.meta.json   structural metadata written by the engine; the
                                quality check reads it - never hand-write it
```

Everything lives under `data/` so the directory can be handed over whole: database, manual, generator and assertions together - runnable, reproducible, re-parameterizable. Leave nothing in the workspace root except the user's own DDL file.

**All generation logic goes in the single file `data/gen.py`.** Do not emit `sql/`, `steps/` or any multi-file generation flow.

Because `gen.py` sits next to the database, anchor paths to its own directory:

```python
HERE = pathlib.Path(__file__).resolve().parent            # data/
ARG = sys.argv[1] if len(sys.argv) > 1 else ""
OUT = HERE / "_build" / "datasource.duckdb" if ARG in ("", "report") else pathlib.Path(ARG)

eng = DDLEngine(DDL, rows=ROWS, profile=PROFILE)
eng.report()                                              # always print the inference
if ARG == "report":                                       # `gen.py report` stops here
    raise SystemExit(0)
OUT.parent.mkdir(parents=True, exist_ok=True)             # never chain `mkdir && python`
eng.generate(str(OUT))                                    # absolute; never depends on cwd
```

`python3 data/gen.py report` prints the inference without generating - that is the cheap first
call. Any other argument is the output path.

`README.md` carries **only what the database cannot say about itself**, and it has a hard budget:
**150 lines.** Do not emit a separate `DATA_DICT.md`.

The engine writes `COMMENT ON` for every table and every non-obvious column, and
`import_database_file` copies those into the datasource - so `describe_table` already returns the
field-level dictionary. **Repeating it in the README is duplication that goes stale.** Leave out
anything a reader can get from the schema:

| Leave out | Why |
|---|---|
| A "Connect" section | The tables are already in the datasource; the reader queries them with `execute_sql` |
| A "Regenerate" section | `python3 data/gen.py` is one obvious line and `gen.py` is sitting right there |
| Per-column tables | Already in the database as comments; `describe_table` returns them |
| Row counts per table restated in prose | Already in the table above them |

What is left is the part only this run knows: which table to start from, which questions were
verified, the signal and anomaly calendar, and the definitions and trade-offs that no column comment
can carry.

### data/README.md template

Target ~60 lines for a five-table schema; **150 is the hard ceiling**.

```markdown
# <business scenario> datasource

DuckDB in datasource `<name>`: <N> tables / <M> rows, YYYY-MM-DD to YYYY-MM-DD.
Built for agent Q&A, business reports and dashboards. Column meanings are in the
table/column comments - use `describe_table`.

## Where to start

| Table | Rows | Use it for |
|---|---|---|
| <entry table> | | Headline trend, YoY/MoM - **start here** |
| <main fact> | | Document-grain drill-down |
| <dimension> | | |

## Verified questions

Each of these was checked against the data and returns a sensible answer:

1. <YoY>   2. <promotion contribution>   3. <anomaly attribution>
4. <Top-N concentration>   5. <dimension differences>   6. <cross-domain>

## Built-in signals

| Type | When | Effect | Attribute it with |
|---|---|---|---|
| Promotion | 11-27~11-30 | GMV x5.2 | `dim_date.event_name` |
| Trough | 02-10~02-20 | GMV x0.52 | same |
| Anomaly | <window> | <metric> A% -> B% | `<column>='<value>'`, others unaffected |

## Differentiated by design

| Dimension | Metric | Range | So that |
|---|---|---|---|
| category | refund rate | 6.1% ~ 14.7% | "which category refunds most" has an answer |

## Definitions and trade-offs

- <currency, timezone, what a status means, how a derived column is computed>
- <anything below target and why - e.g. seller x site x day is sparse at 4.6 rows/cell>
- Grain density: <all-domain x day N> | <category x day N> | <finest grain N>
```

> **This is not ETL test data.** ETL tests only require "the metric is non-zero", so random distributions suffice. Demo data requires "the metric is explainable" - a chart must have a trend and an inflection point, and when the agent is asked "why did February drop" there has to be an answer. The two are generated in almost opposite ways.

---

## Step 1: normalise the DDL to DuckDB syntax

The engine parses the DDL with DuckDB itself, so **a DDL in any other dialect fails outright** -
PostgreSQL, MySQL, StarRocks and Oracle all do, and the error is a parser error with no hint. Most
users paste DDL exported from their real warehouse, so expect to rewrite it. This is the step where
an LLM is genuinely better than the engine: you read the dialect, the engine only executes.

Rewrite it yourself, then put the **rewritten** DDL into `gen.py` (not the original) so the result is
reproducible:

| Dialect | What to strip or map |
|---|---|
| All | `schema.table` -> `table`; backticks and `[brackets]` -> nothing or `"quotes"`; `CREATE INDEX` / `CREATE SEQUENCE` -> delete |
| MySQL | `ENGINE=` / `DEFAULT CHARSET=` / `AUTO_INCREMENT` / `KEY idx (...)` -> delete; `UNSIGNED` -> delete; `DATETIME` -> `TIMESTAMP`; `TINYINT(1)` -> `BOOLEAN` |
| StarRocks / Doris | `DUPLICATE KEY` / `PARTITION BY` / `DISTRIBUTED BY` / `PROPERTIES (...)` -> delete |
| PostgreSQL | `BIGSERIAL` -> `BIGINT`; `TEXT[]` / `JSONB` -> `VARCHAR`; `USING btree` -> delete |
| Oracle | `NUMBER(19)` -> `BIGINT`; `NUMBER(18,2)` -> `DECIMAL(18,2)`; `VARCHAR2(n)` -> `VARCHAR` |

**Keep the keys.** `PRIMARY KEY`, `UNIQUE` and `REFERENCES` are the engine's strongest signal and they
end up in the delivered database - do not drop them while cleaning up.

### Enum comments have a format, and it is what the engine reads

The engine extracts a column's value domain from its **inline comment**, which saves writing
`profile["enums"]` by hand. Dialects that carry comments as a `COMMENT 'x'` clause must be converted,
or the domain is lost silently:

```sql
-- WRONG (MySQL/StarRocks clause; DuckDB rejects it and the domain is lost)
order_status VARCHAR(20) COMMENT 'pending / paid / shipped',

-- RIGHT
order_status VARCHAR, -- pending / paid / shipped
```

The accepted shape, exactly:

| Rule | Accepted | Rejected |
|---|---|---|
| One column per line, comment on that same line | `st VARCHAR, -- a / b / c` | a `/* ... */` block, a comment on its own line, or two columns sharing a line (the first identifier on the line wins) |
| Separator | `/`, `\|`, or the CJK enumeration comma | `,` (ambiguous with the column separator) |
| A label before the values | `-- order status: a / b / c` (text before the last colon is dropped) | |
| At least two values, each <= 24 chars | `-- a / b / c` | `-- some prose description` |
| Trailing `...` marks the list incomplete | `-- a / b / ...` | |

`report()` prints which columns yielded a domain and which ended in `...`.

---

## Path selection (do this first)

| What the user gave you | Path | What you write | Cost |
|---|---|---|---|
| **DDL** (~90% of cases) | **A: DDL-driven** | one profile (~40 lines of declarations) | **~1s to run** |
| Only an industry / table-name description | B: write the DDL first, then A | DDL + profile | same |
| Structure the engine cannot express | C: hand-written generator (rare fallback) | full generator | see below |

**Default to A.** Fall back to C only when `report()` is clearly wrong *and* profile overrides cannot fix it.

What the three paths share: **data is always produced by a Python generator; no path hand-writes SQL to create rows.**

---

## Path A: DDL-driven (main path)

The DDL already states table names, column names, types and keys - none of that needs to be inferred again. `scripts/ddl_engine.py` owns everything generic: table-role detection, column semantics, topological generation order, and enforcement of the 17 invariants. You supply only the business knowledge DDL cannot express.

```python
from ddl_engine import DDLEngine
eng = DDLEngine(DDL_SQL, rows=80_000, profile=PROFILE)
eng.report()           # 1. inspect inference: roles / row allocation / FK chains
eng.generate(str(OUT)) # 2. generate, two-pass calibration, total within 6% of target
```

### What the engine does for you

| Capability | Notes |
|---|---|
| Table roles | Iterative structural inference: `date_dim`/`dim`/`fact`/`detail`/`downstream`/`event`/`snapshot`. **Not based on name prefixes** - it reads what a table references, so `policy` or `ods_order` are classified correctly either way |
| Column semantics | Name + type mapped to id/date/ts/amount/count/ratio/enum/flag/name/seq/measure via 12 rules |
| Row allocation | **Automatic** - you pass `rows=` and the engine splits it; `report()` prints the per-table result. Fact layer 65-75%; dimensions derived from business density (see 1.3); hard cap of 8% of total per table. Override with `table_rows` / `dim_rows` / `dim_kinds`, never by hand-computing |
| Relationships | **Declared PRIMARY KEY / FOREIGN KEY / UNIQUE win**; inference only fills gaps. Renamed keys (`deal.buyer -> cust.cid`) still connect |
| Differentiation | `conditional` gives a column different enum weights or numeric ranges per group - no SQL post-processing |
| Column formulas | `formulas` declares arithmetic identities (accounting identities, cost/margin); dependency-sorted and enforced row by row |
| Invariants | Weighted calendar sampling, derived-from-base quantities, child events anchored to parents, monotonic sequences, complete terminal states, FKs sampled from upstream only, layer backfill, stock baseline, zero header/detail amount drift |
| Summary layer | **Off by default** (`extra_tables="none"` builds exactly the DDL tables). Pass `"summary"` for daily rollups, `"date_dim"` for a date dimension |
| Comments | Table comments written per role; overridable in the profile |

### The profile: the only thing you write

```python
PROFILE = {
  # 1. Event calendar (month-day, expanded across years) and attributable anomalies
  "calendar": {
    "promos": [("11-27", "11-30", 5.2, "Black Friday / Cyber Monday"),
               ("06-16", "06-18", 3.2, "Mid-year sale")],
    "slows":  [("02-10", "02-20", 0.52, "Spring Festival shutdown")],
    "disruptions": [                       # `at` is a relative position 0-1 in the range
      {"at": .30, "days": 26, "factor": .45, "scope": {"site": "JP"}, "name": "JP payment outage"},
      {"at": .62, "days": 30, "factor": .55, "scope": {"courier": "CR003"}, "name": "Hub relocation"}],
  },
  #    Anomalies on a fact table only work in the suppressing direction: `factor < 1` drops rows
  #    probabilistically. `factor > 1` has no effect there - express an amplifying anomaly with
  #    `conditional` or, as a last resort, `pre_sql`.
  # 2. Real value domains and distributions for enum columns (matched by column name, cross-table)
  "enums": {
    "order_status": {"PAID": .845, "REFUNDED": .08, "CANCELLED": .05, "PENDING": .025},
    "category_cd":  {"3C": .24, "APPAREL": .22, "HOME": .18, "BEAUTY": .16},
    "event_type_cd": ["PICKUP", "ARRIVE_HUB", "CUSTOMS_CLEAR", "DELIVERED"],
  },
  "event_seq": {"ods_tracking_event": ["PICKUP", "ARRIVE_HUB", "CUSTOMS_CLEAR", "DELIVERED"]},
  # 3. Conditional distributions: one column, different parameters per group.
  #    Enum columns take a weight dict, numeric columns take [lo, hi].
  #    `__by__` names the grouping column; `__default__` is the fallback.
  "conditional": {
    "ods_order.order_status": {"__by__": "category_cd",
                               "APPAREL": {"PAID": .74, "REFUNDED": .21, "CANCELLED": .05},
                               "3C":      {"PAID": .90, "REFUNDED": .05, "CANCELLED": .05},
                               "__default__": {"PAID": .85, "REFUNDED": .10, "CANCELLED": .05}},
    # `__by__` may also cross a foreign key: "upstream_table.column".
    # Configuring it on the upstream dimension propagates the gradient to details.
    "dim_product.list_price_usd": {"__by__": "category_cd",
                                   "3C": [200, 900], "APPAREL": [20, 150],
                                   "__default__": [30, 300]},
  },
  # 4. Column formulas: arithmetic between columns, topologically sorted (cost before profit)
  "formulas": {
    "orders.paid_amount":       "original_amount - discount_amount + shipping_amount + tax_amount",
    "order_items.total_cost":   "unit_cost * quantity",
    "order_items.gross_profit": "sales_amount - total_cost",
  },
  # 5. Dimension kinds (override when inference is wrong) and value ranges
  "dim_kinds": {"agent_master": "staff", "product_catalog": "enum"},
  "columns": {"dim_product.list_price_usd": {"range": (9, 320)}},
  # 6. Names. The built-in vocabulary is retail-flavoured (brands, store suffixes), so any other
  #    industry must replace it once at dataset level or every name reads as a shop.
  "vocab": {"brand": ["Cardiology", "Neurology"], "org_suffix": ["Ward", "Clinic"]},
  "naming": {"dim_seller": {"tpl": "{brand} {org_suffix}"}},   # per-table template
  # 7. Odds and ends the engine reads but cannot infer
  "refund_rate": 0.055,        # P(a detail line carries a refund); one global value, no per-group form
  "effective_col": {"products": "launched_at"},   # which column makes an entity usable, when inference misses it
  # 8. Optional: role and column-semantic overrides, table/column comments, extra SQL
  "roles": {"some_table": "dim"},
  "semantics": {"encounters.insurance_paid": "amount"},
  "table_comments": {"ods_order": "Order header; amounts in USD"},
}
```

### Correcting column semantics

Every column is classified as one of `id / date / date_pk / ts / amount / count / ratio / enum /
flag / name / seq / measure`, and that classification decides how it is filled. The engine guesses
from the column name, which is a naming convention, and naming conventions are per-industry: it
reads `paid_amount` as money but not `insurance_paid`, `copay`, `premium_received` or `principal`.

`report()` prints the full mapping. **Read it and correct what is wrong** - this is the other place
where you are better than the engine:

```python
"semantics": {
    "encounters.insurance_paid": "amount",     # would otherwise be filled as a generic measure
    "encounters.self_paid":      "amount",
    "encounters.diagnosis_code": "enum",
},
```

Only list the ones you disagree with; a correct guess costs nothing to leave alone. The override is
applied before role detection, so a corrected amount column also counts towards picking the main
fact table. A key declared in the DDL keeps `id` unless you say otherwise, and the pre-check rejects
an unknown semantic or a column that does not exist - a typo here would otherwise leave the column
on its wrong guess and produce plausible-looking wrong data that no quality check can detect.

**The complete field reference is `references/profile-spec.md`** - reading that one file is enough to write a profile. You do not need the engine source.

**Order of work**: run `report()` -> override only what is wrong -> add the calendar and differentiation (`conditional`) -> declare identities with `formulas` -> generate -> run the quality check.

**The first `gen.py` is a first draft, not a finished answer.** Put the calendar, the corrected
semantics, the enums, the identities and *rough* `derive` bands in it, then generate and read the
numbers off `checks.json`. Do not compute in your head what one 9-second run will tell you: a
measured production run emitted **70,000 output tokens in a single turn** - 59% of its wall clock -
and spent it back-solving a marketing funnel so a ratio would land on target, and hand-checking SQL
that `precheck()` now plans for you in milliseconds. Two extra generate-check rounds cost about
100 seconds. Predicting them cost five minutes.

Enum domains usually need no configuration: **values in DDL inline comments are extracted automatically** (`order_status VARCHAR, -- pending / paid / shipped`), and `report()` lists which columns were extracted and which look incomplete.

### Believable metric ranges

The engine does not own metric magnitudes: under defaults ROAS reaches 26, CTR reaches 43% and
attributed orders come out 68x actual orders. **These can only be calibrated by hand.** Use this
when configuring `derive` and `conditional` - following it removes most of the rework.

| Metric | Believable range | Note |
|---|---|---|
| Display ad CTR | 0.5% - 3% | Search ads 3-6%, feeds 0.8-2% |
| Click -> session | 80% - 95% | |
| Session -> product view | 2.5 - 4.0 per session | Expands, does not converge |
| Session -> add to cart | 8% - 15% | |
| Add to cart -> checkout | 30% - 45% | |
| Checkout -> payment | 45% - 65% | |
| **Site-wide CVR** (orders/sessions) | **1% - 3%** | Direct 3-4%, social 1-1.5% |
| **ROAS** | **2 - 8** | Affiliate/email 6-9, paid search 3-6, social 3-5; organic has no spend -> NULL |
| **CAC** | 0.2 - 0.4 x average order value | Above 1x means the model is wrong |
| Marketing cost ratio (spend/GMV) | 8% - 15% | |
| Attributed / actual orders | 0.85 - 1.15 | |
| E-commerce refund rate | 5% - 12% | Apparel 15-18%, 3C 4-5% |
| Gross margin | 15% - 65% | Beauty 55-70%, 3C 10-20%, FMCG 15-25% |
| Repurchase | 2 - 6 orders per customer | B2B 8-15 |
| Top-10% customers' GMV share | 50% - 70% | Above 80% means Zipf is over-concentrated |
| Order status mix | completed 75-85%, cancelled 8-13%, refunded 5-12% | |

Calibrate outside-in: fix the outermost base quantity (impressions/visitors), configure the `derive`
ratios level by level, then read CTR/CVR/ROAS back out. Do not wait until everything is configured.

### Measured

| Scenario | Tables | Rows | Time | Quality |
|---|---|---|---|---|
| E-commerce (9-table DDL, `dim_`/`ods_` naming) | 11 (2 summaries added) | 102,565 | **0.88s** | 17/17 |
| Insurance (8-table DDL, unprefixed mixed naming) | 10 (2 summaries added) | 105,332 | **0.92s** | 17/17 |

---

## Path B: no DDL

Write the DDL directly from the user's business description - that is far faster than writing a generator - confirm table and column names with the user, then follow path A.

Order: 1. find the main fact (the core business action of the industry: order / policy / loan / visit / work order) -> 2. split out details (is the action made of multiple lines?) -> 3. find the event stream (status transitions or high-frequency behaviour; this is where the row count lives) -> 4. add dimensions (one per entity the action touches) -> 5. pick metrics (the top 6 on this industry's dashboard) -> 6. pick signals (peak/trough seasons, which dimension breaks; see the industry event calendar in `references/pitfalls.md`).

**Put enum values in DDL inline comments** (`order_status VARCHAR, -- pending / paid / shipped`); the engine extracts them as the value domain, so the profile need not repeat them.

---

## Path C: hand-written generator (fallback)

Only when the engine genuinely cannot express the structure - for example multi-period state-migration snapshots (credit M1->M2->M3 delinquency) or deeply nested parent-child facts. Use the primitives in `scripts/genlib.py` (weighted calendar, Zipf, event chains, CSV->DuckDB) and write your own loop; see "hand-written generator reference" at the end of `references/pitfalls.md`.

**Phases 0-6 below are shared by all three paths.** On path A the modelling is given by the DDL and the build is done by the engine; every other step still applies.

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

| Total | Main fact | Detail | Event stream | Buyers/users | Sellers/stores | SKUs | Time |
|---|---|---|---|---|---|---|---|
| 10k | 1.2k | 1.8k | 4k | 400 | 15 | 80 | < 1s |
| **50k** | 6k | 9k | 20k | 2k | 30 | 200 | ~1s |
| **80k (default)** | 9.6k | 14k | 32k | 3.2k | 35 | 250 | ~2s |
| **100k** | 12k | 18k | 40k | 4k | 40 | 300 | ~2s |
| 500k | 60k | 90k | 200k | 18k | 70 | 500 | ~6s |
| 1M | 120k | 180k | 400k | 35k | 90 | 700 | ~12s |
| 5M | 600k | 900k | 2M | 150k | 150 | 1200 | ~50s |

**Fix the main fact table first (12% of total), expand the rest by the ratios in 1.2, and derive dimension cardinality from business characteristics per 1.3.** One call gives you all of it and validates the hard constraints:

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

| Layer | Share of total | Note |
|---|---|---|
| All dimensions | **3-8%** | No single dimension above 8% of total; cardinality derived per 1.3, not guessed |
| Main fact | 10-15% | Orders / work orders / policies / subscriptions / visits |
| Fact detail | 15-20% | 1.4-2.2 rows per main-fact row |
| Event stream | **35-50%** | Tracking / events / status transitions / instalments; 3-5 rows per main-fact row |
| Summary layer | 20-28% | Produced by SQL; row count = number of aggregation cells, driven by dimension cardinality |

> The fact layer (main + detail + events) should total **65-75%**. If dimensions crowd it out, cut the user dimension first (it runs away most easily), then SKUs.

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

---

## Phase 3: the invariants (must hold on every path)

| # | Invariant |
|---|---|
| 1 | Fact dates are always sampled from the weighted calendar; `rng.choice(DAYS)` is forbidden |
| 2 | Derived quantities come from base quantities: `click = int(impression * ctr)`; never randomise both independently |
| 3 | Child event time = parent time + a **non-negative** offset, clamped to the cut-off date |
| 4 | An entity's event sequence is forced monotonic: `t = max(t, prev + min_gap)` |
| 5 | Status labels are derived from facts, never pre-set before generating (otherwise a row is labelled ON_TIME while delivered after the promise) |
| 6 | Cross-table foreign keys are sampled only from the already-generated upstream set |
| 7 | Multi-currency always stores three columns: local amount + rate + base amount; aggregate only on the base |
| 8 | An entity's popularity is Zipf-weighted **exactly once** (stacking pushes Top 10% to 99.8%) |
| 9 | Terminal events must exist on completed entities; never truncate sequences at a fixed length |
| 10 | Dimension cardinality is derived by the 1.3 formulas |
| 11 | Business differences (refund rate by category, delivery time by region) must be modelled explicitly, or every dimension looks identical |
| 12 | Anomaly multipliers stay in a believable range; no cliffs |
| 13 | Names are semantic: `Lumora Overseas Flagship Store`, not `seller_0` |
| 14 | Sparsity is fixed by adding a coarser grain, **never by distorting the distribution** |
| 15 | Summary-layer definitions are audited column by column; `count(DISTINCT)` must not be approximated with `min()`-style hacks |
| 16 | Entities with a lifetime (subscription/loan/policy/membership) need a 40-50% stock baseline; they cannot all start inside the window |
| 17 | No `list.index()` / `x in list` / whole-list comprehensions inside the generation loop; build index dicts up front |

On path A the engine guarantees all of these; on path C check them yourself. Details in `references/pitfalls.md`.

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

## Phase 5: quality check (mandatory; no delivery without it)

After `import_database_file`, run the built-in tool against the datasource:

```
check_datasource_quality(config_path="data/checks.json")
```

It runs 19 automatic checks: layering, mandatory date dimension, FK orphan rate, time span and YoY feasibility, time trend, stock baseline, weekday cycle (auto-detecting B2C weekend-heavy vs B2B weekend-light), event explainability, derived-ratio range and negative counts, dead and constant columns, event monotonicity, long-tail concentration, aggregation density (thresholds adapt to database size), comment coverage, semantic naming, primary-key non-null uniqueness, no single entity dominating the head, and no future-dated rows.

Trend, span, stock baseline and event attribution observe the **headline** series - the largest fact table's highest-magnitude business metric (identifier columns like `month_key` and cumulative stock columns like `billed_usd` are excluded). Only the weekday check searches for the strongest signal, and it says which column it used. It finds the `.datasource.meta.json` the engine wrote (roles, keys, strict-DDL mode) instead of re-inferring, and reports which copy it used.

Business-specific rules (header/detail amount alignment, causal ordering, label self-consistency, dimension gradients) go in `data/checks.json`:

```json
{
  "assertions": [
    {"name": "header/detail amount alignment", "expect": "zero",
     "sql": "SELECT count(*) FROM ods_order o JOIN (SELECT order_id, round(sum(item_amt_usd),2) s FROM ods_order_item GROUP BY 1) x USING(order_id) WHERE abs(o.gross_amt_usd-x.s)>0.02"},
    {"name": "child event after parent", "expect": "zero",
     "sql": "SELECT count(*) FROM ods_interaction i JOIN ods_content c USING(content_id) WHERE i.action_dt < c.publish_dt"},
    {"name": "status label self-consistent", "expect": "zero",
     "sql": "SELECT count(*) FROM ods_shipment WHERE ship_status='DELIVERED' AND ((is_on_time=1 AND deliver_dt>promise_dt) OR (is_on_time=0 AND deliver_dt<=promise_dt))"},
    {"name": "terminal event present", "expect": "zero",
     "sql": "WITH t AS (SELECT shipment_id, max(CASE WHEN event_type_cd='DELIVERED' THEN 1 ELSE 0 END) d FROM ods_tracking_event GROUP BY 1) SELECT count(*) FROM ods_shipment s JOIN t USING(shipment_id) WHERE s.ship_status='DELIVERED' AND t.d=0"},
    {"name": "core ratio in believable range", "expect": {"min": 60, "max": 95},
     "sql": "SELECT 100.0*sum(on_time_cnt)/sum(delivered_cnt) FROM dws_fulfillment_day"},
    {"name": "dimensions differ", "expect": {"min": 2.0, "max": 6.0},
     "sql": "WITH t AS (SELECT category_cd, 100.0*sum(refund_amt_usd)/sum(gmv_usd) r FROM dws_sales_category_day GROUP BY 1) SELECT max(r)/min(r) FROM t"}
  ]
}
```

`expect` is `zero` | `nonzero` | `{"min": x, "max": y}`.

**Pass criterion: `ok: true` in the result (zero FAIL). WARN is acceptable but every WARN must be
explained in the delivery report.**

### Write the plausibility checks *before* the first check call

The 19 built-in checks prove the data is structurally sound. They say nothing about whether the
numbers are *believable* - whether ROAS is 5 or 26, whether attributed orders track actual orders,
whether apparel refunds more than electronics. Those belong in `checks.json`, as assertions, and
they must be there **before** you call `check_datasource_quality` the first time.

Put every number from the believable-ranges table you care about into an assertion:

```json
{"name": "attributed orders track actual", "expect": {"min": 0.85, "max": 1.15},
 "sql": "SELECT sum(attributed_orders)*1.0/(SELECT count(*) FROM orders) FROM daily_channel_metrics"},
{"name": "paid-channel ROAS believable", "expect": {"min": 2, "max": 8},
 "sql": "SELECT sum(attributed_revenue)/sum(ad_spend) FROM daily_channel_metrics WHERE ad_spend > 0"},
{"name": "refund gradient by category", "expect": {"min": 1.5, "max": 4.0},
 "sql": "WITH t AS (SELECT category, 1.0*sum(refund_quantity)/sum(quantity) r FROM order_items JOIN products USING(product_id) GROUP BY 1) SELECT max(r)/min(r) FROM t"}
```

**Then `ok: true` is the finish line.** Write `data/README.md` and deliver.

Why this matters, measured: a run reached `ok: true` at 22 minutes and then spent **11 more minutes
(32% of its wall clock)** hand-checking margins, refund gradients, ROAS and the attributed/actual
ratio. It found a real defect - an impressions range three times too high - and fixed it. But the
check result was **identical before and after**, so nothing recorded what had been verified or
repaired. The work was worth doing; doing it as ad-hoc queries after the gate was what cost the time.

If you do find something after a pass, **add the assertion first, then fix** - never fix without
recording, or the next run repeats the same eleven minutes.

On FAIL, locate the cause with the root-cause table in `references/pitfalls.md` and fix the
generator - **never relax a threshold to make the data pass**.

Reasonable WARNs (no fix needed, just explain):

| WARN | When it is reasonable |
|---|---|
| Missing date dimension | The user's DDL has none and `extra_tables="none"` is honouring it - put the event calendar in `data/README.md` |
| No summary layer | Same: the user wanted source tables only |
| Dimension exceeds 8% of total | The user specified that row count explicitly (user intent beats invariant 10) |
| Constant numeric column | The column really is constant in the business (e.g. `is_active` always 1) |

---

## Phase 6: question validation

**Your `checks.json` assertions are this phase.** Write the business questions you care about as
assertions, and `check_datasource_quality` answers all of them in one call - that is what the file
is for. Aim for 8-12, covering:

1. YoY growth of the core metric
2. Promotion contribution versus a normal day
3. An anomaly window that drills down to one dimension
4. Top-N concentration and whether a tiering matches contribution
5. Differentiated metrics by category/region ranking the way the business would expect
6. Any cross-domain question the schema supports

**Budget: at most 5 ad-hoc `execute_sql` calls** beyond the assertions, and only to read something
an assertion cannot express. A measured run issued 43; the extra 38 produced no change to the
dataset. An assertion is cheaper than a query because it re-runs for free on the next check, and it
ships with the dataset as documentation of what was verified.

If an assertion fails, go back to Phase 2 and retune the signal - not the assertion.

---

## Delivery

Confirm every artifact exists under `data/`: `README.md` (<=150 lines), `gen.py`, `checks.json`, `.datasource.meta.json`. The data itself lives in the datasource, not in a file beside them. Confirm there is **nothing extra**: no `_build/` left behind, no `sql/`, no `steps/`, no `DATA_DICT.md`; all generation logic in `data/gen.py` alone.

> **If `data/datasource.duckdb` exists, leave it alone.** It is not yours and it is not a leftover:
> in a Datus deployment that is the datasource's own file, bound when the project was created, and
> this process has it open. **Never delete it, never overwrite it, never pass it to
> `import_database_file`.** You did not create it - your output went to `data/_build/`, which step 4
> removed. A production run spent a turn deciding what to do about it; the answer is nothing.

Report to the user: the datasource the tables were loaded into, table list with row counts, **the built-in business signals and anomaly calendar** (this is what the user needs to design dashboards and questions), the quality-check result, and an honest account of anything below target with the trade-off taken. If you built tables beyond the user's DDL (`extra_tables` other than `none`), list exactly which and why.

Reference documents:

- `references/profile-spec.md` - **complete profile field reference; the only thing to read before writing a profile**
- `references/pitfalls.md` - the 17 invariants in detail, a distortion root-cause table, and the hand-written generator reference
- `scripts/genlib.py` - shared generation primitives (path C only)
