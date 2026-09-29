---
name: init
description: Lightweight project initialization — optionally scoped to specific files / tables / datasources / domains. Statically analyzes the project's SQL (lineage, joins, constant rules, author comments) with extract_sql_lineage, reads human-written docs, verifies the findings against the database with cheap probes, then writes an AGENTS.md project map (data architecture, core tables, global rules, knowledge index) and per-domain ./knowledge/*.md files (table cards, relationships, lineage, business rules, known issues). Stops short of the expensive vector-indexed stores (semantic_models / metrics / reference_sql). Single confirmation-free pass.
tags:
  - init
  - workspace
  - project
  - lineage
version: 4.7.0
user_invocable: true
---

# Lightweight Project Initialization

You are initializing a data project so that a downstream agent — one that answers questions with SQL or changes ETL — gets the answers right. Write down what that agent **cannot get from the schema** and would otherwise get wrong:

1. **Which table to use** — the authoritative table among layers, versions, backups and copies.
2. **Grain and keys** — what one row is; the most common cause of double counting.
3. **How tables join** — keys, cardinality, fan-out traps, keys that need a transform.
4. **Business rules** — metric definitions, mandatory filters, code/enum meanings, term→column mappings.
5. **Time** — partition / date columns, load mode and latency, snapshot semantics, rule change dates.
6. **Lineage** — which table is built from which, by which script.
7. **Known issues** — data gaps, bugs worked around in SQL, doc/SQL/database disagreements.

Do **not** write what the agent already knows or can fetch in one call: tool lists or "recommended tools", full column lists, column types, generic SQL advice.

Outputs:
- **`./AGENTS.md`** — the project map, injected (first ~200 lines) into every node's `<project_context>`.
- **`./knowledge/<domain>.md`** — per-domain detail, read on demand through the `## Knowledge` index.
- **memory** (`add_memory`) — only durable, agent-bound preferences; usually nothing during init.

You do **not** build `semantic_models` / `metrics` / `reference_sql` (that is `/build-kb`). No confirmation gate: these are cheap markdown writes.

Tools: `extract_sql_lineage`, filesystem (`glob`, `grep`, `read_file`, `write_file`, `edit_file`), database (`list_databases`, `list_tables`, `describe_table`, `get_table_ddl`, `search_table`, `execute_sql` for read-only `SELECT` / `SHOW` / `EXPLAIN`), `add_memory`, `load_skill` (`storage-classify` for routing edge cases). Do **not** use todo tools or subagents.

## Evidence Rules (apply to every step)

- **Every statement you write needs a source**: a script, a doc section, a comment, or a database probe. If you are inferring, say so (`inferred from naming`). Never invent a table, column, code value or number, and do not add caveats or interpretations the sources do not support.
- **Authority order.** Human-written material states *intent*: docs written for people or agents, and SQL comments (the only place script authors explain themselves). The database states *reality*: what exists and what values actually occur. SQL states *practice*: what the pipelines really do. For business definitions prefer docs → comments → SQL; for existence, names and values prefer the database.
- **Disagreements are findings, not noise.** A doc naming a table the database lacks, a doc rule the SQL does not apply, two script versions filtering differently — record each under the domain's `## Known Issues` with both sides, and tell the downstream agent what to do.
- **Frequency identifies candidates.** Compare distinct statements against the table-read denominator and check clause context, business scope and counterexamples. Neither high frequency nor a single occurrence establishes a mandatory table rule; confirm it with authoritative documentation or independent consumers before promoting it.

---

## Step 0 — Scope & Inventory (no questions)

The user may invoke `/init <free-text hints>`; hints arrive as "Additional context from the user". Parse them into files / datasources / tables / domains. No hints → whole project. Do **not** call `ask_user`.

1. **Datasource**: use the active datasource (`default_datasource` / `--datasource`); broaden only if hints name others. Note its dialect and database.
2. **Files**: `glob` the in-scope tree (skip hidden dirs, `__pycache__`, `node_modules`, `.venv`, binaries, files > ~1 MB). Sort them into:
   - **SQL scripts** — ETL (`INSERT` / `CREATE ... AS`) and query corpora (`.sql`, SQL inside `.py` DAGs);
   - **human docs** — `.md` / `.txt` / wiki exports: business rules, metric definitions, agent instructions, data dictionaries;
   - **config** — scheduler / dbt / pipeline definitions.
3. **Goal**: first ~3000 chars of `README.md` if present, else infer 1–2 sentences from directory, doc titles and table names (mark it as inferred).

---

## Step 1 — Static Analysis of the SQL

Skip this step when there are no SQL scripts in scope.

1. Call `extract_sql_lineage(paths=[<in-scope SQL globs>])` once — it returns every section. Pass `dialect` when the scripts target a different engine than the datasource. Inspect `pagination` and follow every needed `next_offset` with `result_path` and `offset`, keeping paths, dialect and sections unchanged. Fetch all `comments.file_headers` pages for a question-to-SQL corpus. Narrow `paths` for source-level detail; use `result_path="statements"` or `"raw_lineage"` for uncollapsed evidence. If `files_truncated` is true, split the input paths. For `detail_omitted` or `text_truncated`, read the referenced source file.
2. **Database name mapping.** `stats.databases_referenced` lists the databases the scripts write to and read from. If they differ from the datasource's database, or table-name case differs, match script names against `list_tables` (case-insensitive) and record the mapping once in AGENTS.md.

Read the result as leads, then open only what it points at:

- `tables` + `lineage` + `roots` → tables read/written in the scanned corpus and candidate flows, including pure SELECT queries. A root only means no write was observed here, not external ownership. Several `scripts` for one target may be versions or distinct jobs; inspect their evidence before deciding; several similarly named targets (version, backup or date suffixes) = a version family to resolve in Step 3.
- `joins` → candidate relationships with direction relative to the returned table order. `transforms` describe key expressions; `evidence` locates the full conditions. Self-joins retain aliases. Check unresolved relationship conditions before assuming a relationship is absent.
- `rules.filters` → observed constant predicates, ranked by `distinct_statements` (comment-free normalized SQL, so copies do not inflate support). Compare with `table_read_statements`, the independent statements reading that table. Frequency is a lead, never proof of a mandatory rule. WHERE, JOIN, HAVING and CASE contexts are separate; inspect `conditions` for full branch/compound logic and branch order.
- `rules.value_mappings` → partial observed code-to-label mappings; inspect evidence for ELSE branches, evaluation order and conflicting labels before writing business definitions.
- `rules.window_functions` → observed ROW_NUMBER windows, including numbering without filtering and Top N. `rules.dedup` only reports a traced selection retaining at most one row per partition at that query stage. Neither proves source-table uniqueness or final output grain; verify in Step 3. `parameterized_predicates` on lineage are query parameters, not proof of incremental loading.
- `comments.file_headers` → the full comment block each script opens with. In an ETL project it describes the script; when headers carry a business question with its definitions, notes or expected output, the scripts form a **question→SQL corpus** and the headers are the richest human input you have — read every one (see Step 5).
- `comments.metric_notes` → the author's name for a computed column plus its expression: map metric names to where they are computed.
- `comments.notes` → intent and caveats; notes on `WHERE` / `JOIN` lines often explain a rule or a workaround (bug, exclusion, temporary fix). Read the surrounding code for the ones that matter.
- `comments.column_labels` → a glossary for non-obvious column names.
- `unresolved` → files the tool could not analyze; read them directly if they build core tables.

Then **read the build scripts of the published tables in full** — the most downstream targets in `lineage` and the summary tables they read. Their final `SELECT` is where official metrics are defined: formulas, ratios and their denominators, scoring / ranking / banding rules, thresholds, caps, weights. Record each one; these are the definitions a question-answering agent needs most and the tool cannot interpret for you. When several versions of such a script exist, read the one that builds the authoritative table (Step 3) and note material differences in the others.

---

## Step 2 — Read the Human Docs

Docs usually hold the business meaning SQL cannot: metric definitions, the rule behind a filter, which table is authoritative.

- Read every in-scope doc. For a large doc (> ~60 KB) list its headings first (`grep -n "^#"`), then read it section by section — do not skip sections that define tables, rules or metrics.
- Extract: metric definitions (name → formula → source columns / tables), mandatory rules, table catalogs and their stated purpose, source-priority rules, example SQL, caveats. Also list every **constant the docs prescribe** — default filter values, fixed values, named categories, value lists — for checking in Step 3.
- **Compare every metric the docs define with the script that computes it** (from Step 1). Where they differ — source table, grain, denominator, threshold, direction — record the conflict in Known Issues, state which one the published data follows (the script's), and verify with one query when cheap.
- A doc written as agent instructions is **input, not output**: distill it into the structures below; do not paste it into AGENTS.md.

---

## Step 3 — Verify Against the Database

Budget roughly 30–60 cheap, read-only queries. Always use aggregates or `LIMIT`; add a partition / date filter on large tables. Record what each probe showed; write unverified claims as `unverified`. Check three things:

1. **Existence and versions.** Every table or column you take from docs or scripts exists (`list_tables` / `describe_table`, matched case-insensitively and across version suffixes before you call it missing — a wrong "missing" claim sends the agent away from a table it needs). For each version family, compare freshness (`MAX(<date column>)`, row count): the authoritative one exists, is fresh, is consumed downstream in lineage, and/or is named by the docs.
2. **Grain and time.** For each core table: `COUNT(*)` vs `COUNT(DISTINCT <candidate key>)`; its date range; and, for any snapshot / partition date, `SELECT <date col>, COUNT(*) ... GROUP BY 1 ORDER BY 1` over the range to see the real cadence, gaps and extra snapshots inside one period (they double count when a query filters by a longer period). Name suffixes are conventions, not evidence — write the cadence you observed. For the top relationships, check the side you believe is "one" the same way.
3. **Values.** `SELECT <col>, COUNT(*) ... GROUP BY 1 LIMIT 20` for: code mappings from docs / SQL (note values that occur but are undocumented); every doc-prescribed constant from Step 2 (a value that matches no rows, or a "fixed" value that varies, is a Known Issue — write what the data holds, never the doc's value as a working rule); columns that look like variants of one name (which one is populated); and dimensions of published tables whose values differ from the detail tables (merged or renamed members — write the mapping).

---

## Step 4 — Choose Core Tables and Domains

- **Core tables** (≤ 30): rank by downstream use in lineage, join-hub position, doc mentions, and being the authoritative version. Include the tables a question-answering agent should query directly (published / summary tables) and the shared dimension tables.
- **Domains**: a few wide business areas (usually matching lineage chains or doc sections), one knowledge file each. Prefer fewer, wider domains; shared dimension tables go into a `common` domain.

---

## Step 5 — Write `./knowledge/<domain>.md`

One file per domain, target ≤ ~400 lines; split a domain that grows past it (e.g. `<domain>-glossary.md` for a large term list). Reuse and extend an existing file for the same domain instead of creating a parallel one. Sections appear in this order and only when they have content:

```markdown
# <Domain Title>

> **Domain:** <what this file covers and what it does not>. Sources: <script dirs / doc names>.

## Tables

### <table_name>
- **Use for:** <when to pick it>. **Status:** authoritative | legacy — use <other> | realtime | staging
- **Grain:** one row per <entity × time>; key `(<cols>)` (verified: count = distinct)
- **Time:** <date / partition column>; data from <start>; <observed cadence, load mode and latency>
- **Dimensions:** only non-obvious ones — `<col>`: <value>=<label>, …; snapshot / latest-record semantics
- **Measures:** `<col>` — <meaning>, <additivity>, <unit>; <conditions the build script already applies inside it>
- **Filters:** predicates the scripts consistently apply when reading this table (from `rules.filters`, checked against its `table_read_statements` denominator and source evidence) or that docs require

## Relationships
| Left | Right | Keys | Cardinality | Note |
|---|---|---|---|---|
| a | b | `a.x = b.y` | N:1 (verified) | <transform / dedup first / fan-out trap> |

## Lineage
| Table | Upstream | Built by | Downstream |
|---|---|---|---|

## Metric Definitions
- **<metric name>** = <formula in columns>; source `<table>`; filters <…>; <where it is precomputed, e.g. column `x` of `<summary table>`>; <scoring / ranking / threshold rule when the metric is scored>

## <Business Rule Topic>
- <one atomic fact per bullet>

## Known Issues
- <gap / bug / conflict> — <what to do about it>
```

Content rules:
- **Only what the schema does not say.** A table card line is written only when it adds something beyond column names and types. No column lists.
- **Atomic facts.** Business-rule topics follow the fact rules of `extract-knowledge` (its *Worth-Writing Test* and *Knowledge File Layout*): shortest atomic facts, no filler, no derivable facts. Do **not** run the `extract-knowledge` workflow itself during init.
- **Self-contained.** Write the definition itself, never "see `<file>`": the downstream agent may not have the source files.
- **Exact computations.** Take a metric's business name and intent from docs and its exact computation from the script that publishes it (Step 1), with its location (`<table>.<column>`). Cover every metric the docs define and every scored / ranked item of a published table. Transcribe scoring, ranking and banding rules exactly — every branch in evaluation order (first match wins), each threshold with its operator, what counts are relative to, partition and sort direction, tie handling (`RANK` vs `ROW_NUMBER`), and how NULL / zero / empty groups are treated — as a compact ordered rule such as `ratio ≥ 0.9 → A; ratio ≥ 0.7 → B; else C`. On measure lines, state the conditions the build script already applies inside the column, so consumers neither apply them twice nor miss them.
- **Question→SQL corpus.** When the scripts are validated answers to business questions (Step 1 `file_headers`, or `queries_without_target` ≈ all statements), they specify the business vocabulary. Extract **every** term, segment, cohort, code list, fixed filter and metric the headers and SQL define — `**<term>**: <exact predicate / code list / table / date convention>`, one bullet per term, deduplicated across files — into a glossary topic per domain. The glossary should hold about as many bullets as there are distinct terms in the headers; a glossary much shorter than the term count means you summarized — go back and write the missing terms out. Also record per-question-type conventions: which table answers which kind of question, the output grain, default filters, and how date ranges and identifiers are handled.
- **Coverage over brevity.** Knowledge files are read on demand: a missing rule costs a wrong answer, an extra line costs little. Brevity applies to each line, not to coverage.
- Lineage tables list core tables only; do not paste the whole graph.

---

## Step 6 — Write `./AGENTS.md`

Hard limit: **≤ 200 lines** (only the first 200 are injected). It is a map with the few rules that apply everywhere — details live in knowledge files. Canonical section order; write a section only when it has real content, never a placeholder:

```markdown
# <Project> — <one-line goal>

## Data Architecture
- Datasource `<name>` (<dialect>), database `<db>`; <N> tables by layer: <counts>
- Layers & naming: <prefix → meaning>
- Loading: <cadence / mode / latency>
- Main flows: `<source> → <detail> → <summary> → <published>` (one line per domain)
- <name mapping when scripts reference a different database or name case than the datasource>

## Directory Map
| Path | Contents |
(only directories a downstream agent will open: scripts by layer, docs)

## Core Tables
| Table | Layer | Grain | Use for | Details |
|---|---|---|---|---|
| <table> | <layer> | <one row per …> | <when to query it; "legacy — use X"> | [<domain>](knowledge/<domain>.md) |

## Global Rules
- <rules that hold across domains: mandatory filters on shared dimension tables, snapshot handling, version choice, source priority, time conventions>

## SQL Conventions
- <only when the project has a validated question→SQL corpus; see below>

## Knowledge
- [<Domain>](knowledge/<domain>.md) — <scope naming the concrete contents: which tables, metric names, code tables>
```

- **Global Rules** are few (≤ ~12) and each must be supported by authoritative docs or verified against independent SQL consumers and counterexamples. High file frequency alone is insufficient.
- **SQL Conventions**: only from a corpus of validated `(question, SQL)` pairs — ETL scripts are not such a corpus. Each rule is *"when the question says ⟨phrasing⟩ → ⟨output shape⟩"*, schema-free, checked for counter-examples. No corpus → omit the section.
- Do not write `## Semantic Models` / `## Metrics` / `## Reference SQL` — `/build-kb` owns them.
- `AGENTS.md` already exists → scoped `edit_file` of the sections you own; ask via `ask_user` only before replacing the whole file (the only question this skill may ask). With scope hints, edit only in-scope rows and keep every other section verbatim.

---

## Step 7 — Self-Check, Then Wrap Up

Before finishing, check and fix:
1. `AGENTS.md` ≤ 200 lines; no empty sections or placeholder lines.
2. Every table name you wrote exists in `list_tables`, or is marked missing under Known Issues after the variant check in Step 3.
3. Every `knowledge/...` link in `AGENTS.md` resolves to a file you wrote.
4. Every core table has a card with at least **Use for** and **Grain**.
5. For a question→SQL corpus: count the distinct terms the file headers define and the glossary bullets you wrote; every term must appear with its exact definition — add the missing ones. The summary **must** state `terms defined / terms written`.

Then reply with a short summary: files written, the number of core tables / relationships / rules / known issues, and anything you could not verify. Do not propose follow-up actions.

## Important Notes

- **Stay in scope.** With scope hints, never scan, write knowledge for, or edit AGENTS.md sections outside that scope.
- **Lightweight pass.** Do NOT call `semantic_modeling` / `gen_sql_summary`, do NOT fan out subagents, do NOT emit a Generation Manifest.
- **Do not ask the user anything** except before wholesale-overwriting an existing `AGENTS.md`.
