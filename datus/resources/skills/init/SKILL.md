---
name: init
description: Lightweight project initialization — optionally scoped to specific files / tables / datasources / domains. Reads the project's SQL (through the persisted table lineage graph, upsert_lineage / query_lineage), human-written docs and the database schema without querying table data, then writes an AGENTS.md project map and per-domain ./knowledge/*.md files holding only statements a source confirms; inferred conclusions are returned as pending questions and written only after the user confirms them. Stops short of the vector-indexed stores (semantic_models / metrics / reference_sql). Single confirmation-free pass.
tags:
  - init
  - workspace
  - project
  - lineage
version: 4.13.1
user_invocable: true
---

# Lightweight Project Initialization

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

- **No table data.** Use schema metadata only: `list_tables`, `describe_table`, and `execute_sql` only for the dialect's statement that returns a table's DDL. No `SELECT` against a table, not even with `LIMIT` or a partition filter, not even to settle a conflict. Do not delegate to subagents: their tools can query table data.
- **No questions during the run.** Do not call `ask_user`, except before replacing an existing `AGENTS.md` wholesale; otherwise edit the sections you own in place.
- **Scope.** Hints after `/init` arrive as "Additional context from the user" and name files, datasources, tables or domains. With hints, read and write only in scope and keep every other section verbatim. Without hints, cover the whole project on the active datasource.
- **Lightweight.** Do not build semantic models, metrics or reference SQL (`/build-kb` owns them): no `semantic_modeling`, no `gen_sql_summary`, no generation manifest.

## Where to Look

Read the highest-value sources first and let what you learn decide what to read next:

1. Human docs and instruction files: `README`, an existing `AGENTS.md`, business rules, metric definitions, data dictionaries.
2. The SQL. `upsert_lineage(paths=[<in-scope SQL globs>], prune_missing=True)` builds the lineage graph; `query_lineage()` without tables returns the whole table graph, and with tables the statements around them. The scripts that build the most downstream tables define the published metrics. SQL comments are where authors explain their filters and workarounds.
3. Scheduler or pipeline config.
4. The database schema: check that every table and column you write exists, and read declared keys and partitioning.

When the script headers pose business questions answered by the SQL below them, the scripts are a **question→SQL corpus**: they define the project's business vocabulary (see the glossary rule below).

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
