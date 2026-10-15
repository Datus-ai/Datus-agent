---
name: init
<<<<<<< HEAD
description: Lightweight project initialization — optionally scoped to specific files / tables / datasources / domains. Infer the project goal and in-scope datasources, scan the in-scope file tree and database metadata (db/table/desc/sample), classify into business domains, then write an AGENTS.md inventory skeleton plus the cheap file-based stores (atomic facts to ./knowledge/*.md via lite extract-knowledge, durable preferences to memory). Stops short of the expensive vector-indexed stores (semantic_models / metrics / reference_sql). Single confirmation-free pass, low token cost.
=======
description: Lightweight project initialization — optionally scoped to specific files / tables / datasources / domains. Reads the project's SQL (through the persisted table lineage graph, upsert_lineage / query_lineage), human-written docs and the database schema without querying table data, then writes an AGENTS.md project map and per-domain ./knowledge/*.md files holding only statements a source confirms; inferred conclusions are returned as pending questions and written only after the user confirms them. Stops short of the vector-indexed stores (semantic_models / metrics / reference_sql). Single confirmation-free pass.
>>>>>>> ce07705 ([Enhancement] Rewrite init as a goal-driven skill that reads schema only and writes confirmed statements only (#1504))
tags:
  - init
  - workspace
  - project
<<<<<<< HEAD
  - classify
version: 3.1.0
=======
  - lineage
version: 4.13.1
>>>>>>> ce07705 ([Enhancement] Rewrite init as a goal-driven skill that reads schema only and writes confirmed statements only (#1504))
user_invocable: true
---

# Lightweight Project Initialization

<<<<<<< HEAD
You are initializing a project workspace **lightly**. The goal is a fast, low-cost first pass that gives a downstream agent a usable map of the project without paying for the heavy vector-store generation. You scan the project (files + database metadata), classify it into business domains, then write:

1. an **`AGENTS.md` inventory skeleton** (the project map injected into every node's `<project_context>`), and
2. the **cheap file-based stores** — atomic business facts to `./knowledge/*.md` (via lite `extract-knowledge`) and durable cross-session preferences to `memory` (via `add_memory`).

You **do not** build the vector-indexed stores (`semantic_models`, `metrics`, `reference_sql`) here — those cost tokens, fan out `explore` subagents, and write LanceDB, so they are out of scope for this lightweight pass. There is **no confirmation gate** in this skill: knowledge/memory/AGENTS.md are cheap markdown writes, so just do them.

You run in the main agent context, so you may call `add_memory`/`edit_memory`, the filesystem tools (`glob`, `grep`, `read_file`, `write_file`, `edit_file`), the database tools (`list_databases`, `list_tables`, `describe_table`, `get_table_ddl`, `search_table`, `execute_sql` for read-only `SELECT` / `SHOW` / `EXPLAIN` probes only), and `load_skill` (to run `extract-knowledge` lite and consult `storage-classify`). Do **NOT** use the todo tools (`todo_write` / `todo_update` / `todo_list` / `todo_read`): this is a single-pass, single-turn skill with no subagent fan-out — no later step consumes todos, so writing them only adds sidebar noise.

**Routing authority is `storage-classify`.** This skill decides *what to scan*; for which content lands in `knowledge` vs `memory` vs `AGENTS.md`, follow `storage-classify`'s decision tree (branches 1, 5, 6) — do not re-invent routing here.
=======
## Goal

You are initializing a data project for a downstream agent that answers questions with SQL or changes ETL. Write down what that agent would get wrong and cannot find out with one or two tool calls. For every line, ask: without it, would the agent pick the wrong table, double count, join wrongly, misread a code or miss a filter? If not, leave it out.

What usually passes that test:
- which table to use when several hold similar data;
- grain and keys;
- how tables join: keys and cardinality;
- metric definitions and business rules: mandatory filters, code meanings, term → column mappings;
- time: partition and date columns, load mode, snapshot semantics;
- lineage of the core tables;
- known issues: data gaps, workarounds in SQL, disagreements between docs, SQL and the database.

What does not: column lists and types, tool lists, generic SQL advice, anything the schema already says.

The downstream agent cannot scan tables and rarely re-reads every script, so these facts are expensive for it to recover. In knowledge files, coverage matters more than brevity: a missing rule costs a wrong answer, an extra line costs little. Keep each line short.

## Evidence

- **Confirmed statements only.** Project files hold only what a source states directly: DDL and schema declarations, doc sections, SQL comments, what a script literally does (tables read and written, joins, filters, mappings, formulas), lineage, scheduler config, and what the user confirmed. A conclusion drawn from those sources is an inference: a grain read off a `GROUP BY` or `ROW_NUMBER`, a cardinality read off join syntax, a cadence read off names or parameters, a table judged current without a doc naming it, a layer meaning read off a prefix, a rule generalized from repeated SQL. Inferences go to the pending questions of your final reply, never into a file.
- **Cite** grain, keys, cardinality, status and time as `(DDL)`, `(doc: <name> §<section>)`, `(<file>:<line>)` or `(confirmed by user, <YYYY-MM-DD>)`. Never invent a table, column, code value or number.
- **Where inference hides:**
  - Grain comes only from a declared primary or unique key or a doc. A key that allows duplicates, a sort key or a distribution key does not make rows unique.
  - Write `N:1` only when the join equates every column of a declared primary or unique key on the "one" side; a join on part of a composite key can still fan out.
  - Cadence and load mode come only from scheduler config or docs. Otherwise write what the load statement literally does, with its location, e.g. overwrites the partition of the run date (`<file>:<line>`).
  - "Authoritative" or "legacy" only when a doc or the user says so. Otherwise write the facts on each candidate's card (which tables read it in lineage, which job builds it, what the docs say) and ask which one is current.
- **Who wins.** For business definitions, docs before SQL comments before SQL. For what exists, the database schema. Published data follows its build script.
- **Disagreements are findings.** Compare what the docs say with what the scripts do and what the schema holds. Record every disagreement under the domain's `## Known Issues` with both sides and what the downstream agent should do; never silently pick one.

## Limits
>>>>>>> ce07705 ([Enhancement] Rewrite init as a goal-driven skill that reads schema only and writes confirmed statements only (#1504))

- **No table data.** Use schema metadata only: `list_tables`, `describe_table`, and `execute_sql` only for the dialect's statement that returns a table's DDL. No `SELECT` against a table, not even with `LIMIT` or a partition filter, not even to settle a conflict. Do not delegate to subagents: their tools can query table data.
- **No questions during the run.** Do not call `ask_user`, except before replacing an existing `AGENTS.md` wholesale; otherwise edit the sections you own in place.
- **Scope.** Hints after `/init` arrive as "Additional context from the user" and name files, datasources, tables or domains. With hints, read and write only in scope and keep every other section verbatim. Without hints, cover the whole project on the active datasource.
- **Lightweight.** Do not build semantic models, metrics or reference SQL (`/build-kb` owns them): no `semantic_modeling`, no `gen_sql_summary`, no generation manifest.

<<<<<<< HEAD
## Step 0 — Resolve Scope & Project Context (inferred, no questions)

**Do NOT call `ask_user` in this step.** The user may invoke this skill as `/init <free-text scope hints>` — any hints arrive as an "Additional context from the user" block. Parse them into a concrete scope, then infer the project goal and the in-scope datasources from the repository and configuration. When no hints are given, default to the **whole project**.

1. **Parse the scope hints** into any of: in-scope **files** (globs / paths, e.g. `queries/*.sql`), **datasources**, **tables** (e.g. `orders`, `order_items`), and **business domains** (e.g. "only the sales domain"). Free text — interpret generously, e.g. `/init orders + order_items tables and queries/*.sql, sales domain only`. When no hints are given, the scope is the whole project.
2. **Infer the goal.** Read `README.md` if it exists (first 3000 chars); otherwise derive a 1-2 sentence goal from the directory name, top-level files, and the datasource/table names. State it as an explicit assumption you surface in `AGENTS.md`.
3. **Infer datasource scope from configuration — default to the single active datasource.** Use the **currently active datasource** of the session (the one pinned via `--datasource` / `/datasource`, i.e. the project's `default_datasource`). **When multiple datasources are configured, scope to that active one only — do NOT initialize all of them.** Fall back to the sole configured datasource when only one exists, or to a specific one the project files clearly reference. Broaden beyond the active datasource only if item 1's hints explicitly name other datasources. Do not invent unconfigured services — only mention an extra service (Airflow, Superset, …) if a project file explicitly references it.
4. Use `glob` to scan the **in-scope** directory tree (top 3 levels). Skip hidden dirs and `__pycache__` / `node_modules` / `.venv`. When the scope names specific files / dirs, narrow the scan to that subset.

Record the resolved **goal**, **in-scope datasources**, and the **file / table / domain scope** — this scope governs every later step, **including which `AGENTS.md` sections you write** (Step 3 touches only in-scope sections, via a scoped `edit_file`). If the scope is empty after parsing, state plainly that you are defaulting to the whole project.
=======
## Where to Look

Read the highest-value sources first and let what you learn decide what to read next:

1. Human docs and instruction files: `README`, an existing `AGENTS.md`, business rules, metric definitions, data dictionaries.
2. The SQL. `upsert_lineage(paths=[<in-scope SQL globs>], prune_missing=True)` builds the lineage graph; `query_lineage()` without tables returns the whole table graph, and with tables the statements around them. The scripts that build the most downstream tables define the published metrics. SQL comments are where authors explain their filters and workarounds.
3. Scheduler or pipeline config.
4. The database schema: check that every table and column you write exists, and read declared keys and partitioning.
>>>>>>> ce07705 ([Enhancement] Rewrite init as a goal-driven skill that reads schema only and writes confirmed statements only (#1504))

When the script headers pose business questions answered by the SQL below them, the scripts are a **question→SQL corpus**: they define the project's business vocabulary (see the glossary rule below).

<<<<<<< HEAD
## Step 1 — Scan & Classify

Gather the raw material **inside the resolved scope**, then classify it into a **multi-level taxonomy of business domains / subtopics** (e.g. `sales/orders`, `sales/refunds`, `infra/etl`). This taxonomy feeds the `AGENTS.md` inventory and helps you locate atomic facts worth filing as knowledge.

**File side:**
- Use `glob` / `grep` to collect candidate files **within scope** — scan **all text files**, judged by content rather than extension. **Skip binary files** (images, executables, archives, compiled artifacts, parquet/db blobs, etc.) and **skip oversized files** (> ~1 MB) — if a large text file is clearly relevant, read its head/batches rather than the whole thing. When unsure whether a file is text, peek at the first bytes (`read_file` head) before committing to it.
- **Note any validated-query corpus but do NOT enumerate it here.** A corpus of validated `(question, SQL)` pairs (a queries file, a golden/benchmark set, a saved-query catalog, a dbt/analysis SQL folder) feeds the vector-indexed `reference_sql` store, which is out of scope for this lightweight pass. You may mention its location under `## Data Assets` as a project asset, but do not generate `reference_sql` entries or a dedicated index section.

**Database side (for each in-scope datasource):**
- `list_databases` → `list_tables` to enumerate tables/views; restrict to in-scope tables when the scope names them.
- For representative in-scope tables: `describe_table` (or `get_table_ddl`) for **desc** (column names/types/comments) and `search_table` (its `sample_data`) or `execute_sql("SELECT * FROM <t> LIMIT 5")` for **sample**. Sampling desc/sample is enough for the inventory — you do not need exhaustive statistics here.
- For large databases (>50 tables), sample representative tables per naming pattern rather than describing every table.

**Classify** every in-scope file and table into the domain taxonomy. A single domain may contain both files and tables.

Keep the taxonomy in your working context — do **not** record it with todos (see the tool note above); no later step reads them.

---

## Step 2 — File Cheap Stores (knowledge + memory)

While scanning the in-scope material, you will encounter **atomic business facts** that a downstream agent cannot infer from `INFORMATION_SCHEMA` / column comments alone — field encodings / enum / status codes, mandatory constant filters, join traps, business-term→field mappings. These are cheap to file (plain markdown, no vector index) and high-value.

1. **Atomic facts → `knowledge`.** Run `extract-knowledge` in **lite** mode (do NOT trigger its deep blind-SQL flow): pass the **source** (a table's comments/sample, a doc, a config) and the **specific fact to mine**, plus the datasource it applies to. It writes `./knowledge/<domain-slug>.md`. Only file facts that are genuinely non-inferable — skip anything mechanically derivable from the schema.
   - **Do not enumerate the validated-query corpus for knowledge here.** Mining the `(question, SQL)` corpus pair-by-pair is out of scope for this lightweight pass. Here, only file facts you can read directly off table metadata / docs.
2. **Durable preferences/context → `memory`.** A user/team habit or a default the agent should remember next session goes to `add_memory` (≤ 2000 bytes). Skip session-specific or one-shot content.

No confirmation gate — these are cheap, reversible writes. If a write would destructively overwrite an existing `./knowledge/*.md`, prefer a scoped `edit_file` and do not clobber unrelated entries.

---

## Step 3 — Write the AGENTS.md Inventory Skeleton

Write or update `./AGENTS.md`, following the *AGENTS.md Section Ownership* from `storage-classify`. The canonical section order is:

`# <project name>` · `## Architecture` · `## Directory Map` · `## Services` · `## Data Assets` · `## Recommended Tools` · `## SQL Conventions` · `## Knowledge`

**Honor the resolved scope here too.** When Step 0 resolved a narrower scope (specific files / tables / datasources / domains), write **only the in-scope content** and leave every out-of-scope section untouched: use a **scoped `edit_file`** that updates just the rows/bullets the in-scope material affects (e.g. add only the in-scope directories to `## Directory Map`, only the in-scope tables/domains to `## Data Assets`, only the knowledge files you wrote this run to `## Knowledge`). **Never collapse the whole map down to the scope** — an existing `AGENTS.md` keeps its out-of-scope sections verbatim. A full whole-project skeleton is written only when the scope is the whole project (no hints).

**AGENTS.md is the project's KB entry point.** Because the agentic runtime injects AGENTS.md's first ~200 lines into every node's `<project_context>`, this is the one place that reliably tells a downstream agent *what exists and how to reach it*.

You **own and fill** the inventory sections below — but **write only the sections that have real content**, in the canonical order:

- `# <project name>` — one-line description (the inferred goal).
- `## Architecture` — brief data flow / stack; ASCII diagram only if complex.
- `## Directory Map` — main dirs only (Directory / Purpose / Key Entry Point).
- `## Services` — configured datasources + any user-mentioned services.
- `## Data Assets` — **never enumerate every table.** Summarize per database by domain + table count, naming 3-5 representative tables; note "Use `list_tables` / `search_table` to explore details at runtime."
- `## Recommended Tools` — runtime tools per configured service type.
- `## SQL Conventions` — if the project has a validated-SQL corpus, **induce** its recurring output conventions as a short bullet list (this rides in `<project_context>` and nudges downstream `gen_sql` toward the project's answer shape):
  - **Induce, do not hardcode.** Read a representative slice of the corpus; state only patterns you actually observe, phrased schema-free (no specific table/column/code names).
  - **Every rule carries its trigger phrasing** — write each as *"when the question says/asks ⟨observable phrasing⟩ → ⟨output shape⟩"*. A rule whose trigger you cannot state as question wording is not a rule — drop it.
  - **Counter-example scan before persisting** — for each candidate rule, scan for pairs whose question matches the trigger but whose SQL differs; any counter-example means narrow or drop it.
  - If the corpus is absent or too small to induce any convention, **omit the `## SQL Conventions` section entirely** — do not write a placeholder.
- `## Knowledge` — index of the `./knowledge/*.md` files you wrote in Step 2. One bullet per file: `- [<Domain>](knowledge/<slug>.md) — <one-line scope>`. **Write the scope line to convey unguessable specifics** — name the kinds of concrete values inside (exact thresholds, literal filter codes, enum spellings, term→column mappings), not just the topic. If you wrote no knowledge files this run, **omit the `## Knowledge` section** — do not write a placeholder.

**Only write sections that have real content.** The vector-index sections (`## Semantic Models`, `## Metrics`, `## Reference SQL`) are out of scope for this lightweight pass, so **do not write them at all** — neither the section header nor a "none yet" placeholder. Likewise omit any inventory section you have nothing concrete to put in. Never emit empty placeholder lines like `_No metrics yet._`.

Hard constraints:
- **AGENTS.md is a top-level overview, not a data dictionary.** Target **≤ 200 lines**.
- If `AGENTS.md` already exists, prefer a scoped `edit_file` over a full rewrite, and **ask via `ask_user` before overwriting an existing file wholesale** (this is the only `ask_user` this skill may make). Never touch `## Knowledge` entries you did not write this run.

---

## Step 4 — Wrap Up

Close by telling the user what was written: the `AGENTS.md` inventory, any `./knowledge/*.md` files, and any memory entries. Keep it to a short summary — do not propose follow-up actions.

---

## Important Notes

- **Routing lives in `storage-classify`, not here.** Knowledge vs memory vs AGENTS.md follows its decision tree (branches 1, 5, 6).
- **Honor the resolved scope in every step.** When Step 0 resolved scope hints, never scan, file knowledge/memory, or write AGENTS.md sections outside that scope — only default to whole-project when no hints were given.
- **This is the lightweight pass — stay in scope.** Do NOT call `semantic_modeling` / `gen_sql_summary`, do NOT fan out `explore` subagents, and do NOT emit a Generation Manifest — the vector-indexed stores are out of scope for this skill.
- Use placeholder comments (e.g. `<!-- Describe your architecture here -->`) when you cannot determine something rather than inventing facts.
- **Do not ask the user anything except before wholesale-overwriting an existing `AGENTS.md`.** Goal, datasource scope, knowledge, and memory are all inferred and written directly.
=======
## Output

**`./knowledge/<domain>.md`**: one file per business domain. Prefer a few wide domains; shared dimension tables go into `common`. Keep a file to about 400 lines and split a larger domain; extend an existing file for the same domain instead of starting a parallel one. Suggested layout, sections only when they have content, adapted to the project:

```markdown
# <Domain Title>

> **Domain:** <what this file covers and what it does not>. Sources: <script dirs / doc names>.

## Tables

### <table_name>
- **Use for:** <purpose a doc states, or the facts that show it>. **Status** (only when a doc or the user states it): authoritative, or legacy — use <other>
- **Grain:** one row per <entity × time>; key `(<cols>)` (<source>)
- **Time:** <partition / date column> (DDL); <load mode as stated, or the literal load statement with `<file>:<line>`>; snapshot semantics
- **Dimensions:** non-obvious codes `<col>`: <value>=<label>, …
- **Measures:** `<col>` — <meaning>, <additivity>, <unit>
- **Filters:** <predicates docs require or the relevant SQL applies, with their scope and exceptions>

## Relationships
| Left | Right | Keys | Cardinality | Note |
|---|---|---|---|---|

## Lineage
| Table | Upstream | Built by | Downstream |
|---|---|---|---|

## Metric Definitions
- **<metric>** = <formula in columns>; source `<table>`; filters <…>; precomputed in `<table>.<column>`

## <Business Rule Topic>
- <one atomic fact per bullet>

## Known Issues
- <gap / bug / conflict> — <what to do about it>
```

- Write each definition itself, never "see `<file>`": the downstream agent may not have the source files.
- **Metrics.** Take the business name and intent from docs and the exact computation from the script that publishes it, with `<table>.<column>`. Cover every metric the docs define and every metric a published table computes. Copy each formula exactly, with every branch, condition and threshold.
- **Glossary.** For a question→SQL corpus, write every term the headers and SQL define with its exact definition (predicate, code list, table or date convention), one bullet per term, deduplicated across files, under a glossary topic per domain. Also record per-question-type conventions, citing the pairs they come from: which table answers which kind of question, the output grain, default filters.
- Lineage tables list core tables only; the complete graph stays in `lineage/lineage.json`.

**`./AGENTS.md`**: the project map. Only its first 200 lines are injected into every session, so keep it within 200 lines and leave details to the knowledge files. Suggested sections, each only when it has real content:

```markdown
# <Project> — <one-line goal, only when README or a doc states it>

## Data Architecture
- Datasource `<name>` (<dialect>), database `<db>`; layers and naming as a doc states them; main flows

## Directory Map
| Path | Contents |

## Core Tables
| Table | Layer | Grain | Use for | Details |

## Global Rules
- <few cross-domain rules, each stated by a doc or confirmed by the user>

## SQL Conventions
- <only for a question→SQL corpus>

## Knowledge
- [<Domain>](knowledge/<domain>.md) — <the tables, metrics and code tables it holds>
```

- **SQL Conventions** come only from validated `(question, SQL)` pairs, never from ETL scripts. Each is *"when the question says ⟨phrasing⟩ → ⟨output shape⟩"*, schema-free, checked for counter-examples, and cites its pairs. These and the glossary's per-question-type conventions are the one exception to *Confirmed statements only*: write them directly; the user reviews them in the file.
- Do not write `## Semantic Models` / `## Metrics` / `## Reference SQL`; `/build-kb` owns them.

**memory** (`add_memory`): only durable preferences about how the agent should work; usually nothing during init.

## Done When

- Every table you wrote exists in `list_tables`, or is listed as missing under Known Issues.
- Every core table has a card with at least **Use for**.
- `AGENTS.md` is within 200 lines, has no empty sections or placeholders, and every `knowledge/...` link in it resolves.
- Nothing inferred was written: every grain, key, cardinality, status, cadence and snapshot claim cites its source.
- For a question→SQL corpus, every term the headers define appears with its exact definition.

Then reply briefly: the files written and the number of core tables, relationships, rules and known issues. End with **Pending questions**: at most 10 inferred conclusions, most consequential first, each with its evidence (`<file>:<line>`, doc section, lineage facts) and the exact line you would write if it is confirmed; say how many more you left out. They live only in this reply.

When the user later confirms, corrects or rejects pending questions in this conversation, write each confirmed or corrected item into the matching knowledge file or `AGENTS.md` section with `(confirmed by user, <YYYY-MM-DD>)`, and drop the rejected ones.
>>>>>>> ce07705 ([Enhancement] Rewrite init as a goal-driven skill that reads schema only and writes confirmed statements only (#1504))
