---
name: build-kb
description: Build the project's vector-indexed knowledge base from files plus database metadata — optionally scoped to specific files / tables / datasources / domains. Scan the in-scope material, classify it into business domains, explore each domain in parallel with explore subagents, then (after the user confirms a generation manifest) route every artifact to its store via storage-classify, generating semantic_models / metrics / reference_sql (and mining any extra knowledge), and refresh AGENTS.md's KB index. The lightweight /init handles the AGENTS.md inventory plus file-based knowledge/memory; this skill owns the heavy vector-store generation.
tags:
  - build-kb
  - knowledge-base
  - semantic-models
  - metrics
  - reference-sql
  - classify
version: 1.0.0
user_invocable: true
---

# Build Knowledge Base

You are building the project's **vector-indexed knowledge base** — `semantic_models`, `metrics`, and `reference_sql` (the LanceDB-backed stores) — from the project's files and database metadata. This is the heavy companion to the lightweight `/init`: `/init` already produces the `AGENTS.md` inventory and the file-based stores (`./knowledge/*.md`, `memory`); **this skill owns the expensive generation** that writes the vector stores and then refreshes `AGENTS.md`'s KB index sections.

This is an orchestration skill running in the main agent context, so you may call `task`, `todo_write`/`todo_list`/`todo_read`/`todo_update`, `ask_user`, `add_memory`/`edit_memory`, the filesystem tools (`glob`, `grep`, `read_file`, `write_file`, `edit_file`), and the database tools (`list_databases`, `list_tables`, `describe_table`, `search_table`, `read_query`).

**Routing authority is `storage-classify`.** This skill decides *what to scan and explore within scope*; the `storage-classify` skill owns *which content goes to which store, written with which mechanism*. Do NOT re-invent storage routing rules here — load and follow `storage-classify` in Step 3.

**Two-phase contract (important).** Heavy generation (`gen_semantic_model` / `gen_metrics` / `gen_sql_summary` / `gen_skill` / `extract-knowledge`) costs tokens and writes real artifacts. So there is a hard **turn boundary** after exploration: you produce a Generation Manifest and then **end your turn**. Do NOT call `ask_user` for this confirmation — just present the manifest and stop. The user confirms or corrects it in their next message; only then do you run Step 3 and Step 4.

---

## Step 0 — Resolve Scope (inferred, no questions)

**Do NOT call `ask_user` in this step.** The user invokes this skill as `/build-kb <free-text scope hints>` — the hints arrive as an "Additional context from the user" block. Parse them into a concrete scope; when no hints are given, default to the **whole project**.

1. **Parse the scope hints** into any of: in-scope **files** (globs / paths, e.g. `queries/*.sql`), **datasources**, **tables** (e.g. `orders`, `order_items`), and **business domains** (e.g. "only the sales domain"). Free text — interpret generously, e.g. `/build-kb orders + order_items tables and queries/*.sql, sales domain only`.
2. **Infer the goal and datasource defaults** the same way `/init` does: read `README.md` (first 3000 chars) or derive a 1-2 sentence goal. For datasources, **default to the currently active datasource** of the session (the one pinned via `--datasource` / `/datasource`, i.e. the project's `default_datasource`); **when multiple datasources are configured, scope to that active one only — do NOT cover all of them** unless the Step 0.1 hints explicitly name other datasources. Hints from Step 0.1 **override** these defaults.
3. **Reuse `/init`'s inventory when present.** If `./AGENTS.md` exists, read it to reuse the directory map / data-assets inventory rather than re-scanning the whole tree — narrow your scan to the in-scope subset.
4. Use `glob` to scan the in-scope directory tree (top 3 levels). Skip hidden dirs and `__pycache__` / `node_modules` / `.venv`.

Record the resolved **goal**, **in-scope datasources**, and the **file/table/domain scope** (with a one-line reason for each) — they become the first rows of the Generation Manifest so the user can correct them at the single confirmation point. If the scope is empty after parsing, state plainly that you are defaulting to the whole project.

---

## Step 1 — Scan & Classify (within scope)

Gather the raw material **inside the resolved scope**, then classify it into a **multi-level taxonomy of business domains / subtopics** (e.g. `sales/orders`, `sales/refunds`, `infra/etl`).

**File side:**
- Use `glob` / `grep` to collect candidate files **within scope**: `*.sql`, `*.md`, `*.yml` / `*.yaml`, scripts, configs, notebooks.
- **Validated-query corpus is special — treat it as enumerable, never as a sample.** When the in-scope material holds a corpus of validated `(question, SQL)` pairs (a queries file, a golden/benchmark set, a saved-query catalog, a dbt/analysis SQL folder), count the total and plan to index **every** pair. Unlike tables (where representative sampling is fine), each validated query is a future few-shot example: one omitted pair is one the runtime can never retrieve. Do **not** curate "representative" patterns here.

**Database side (for each in-scope datasource):**
- `list_databases` → `list_tables` to enumerate tables/views; restrict to in-scope tables when the scope names them.
- For representative in-scope tables: `describe_table` for **desc** (column names/types/comments), `search_table` (its `sample_data`) or `read_query("SELECT * FROM <t> LIMIT 5")` for **sample**, and `read_query("SELECT COUNT(*) AS rows, COUNT(DISTINCT <key>) AS card FROM <t>")` for key **statistics** (row count, key-column cardinality). There is no dedicated statistics tool — compute it with `read_query`.
- For large databases (>50 in-scope tables), sample representative tables per naming pattern rather than describing every table.

**Classify** every in-scope file and table into the domain taxonomy. A single domain may contain both files and tables.

**Record with todos when the taxonomy is large (> 3 domains):** call `todo_write` with one todo per domain — `title` = domain name (≤ 8 words), `content` = the files + tables it covers plus exploration focus notes. Track each domain's progress with `todo_update` (`pending` → `in_progress` → `completed` / `failed`) in the next steps. Skip todos for small scopes (≤ 3 domains) to avoid noise.

---

## Step 2 — Explore Each Domain in Parallel (concurrency ≤ 3)

For each domain, delegate a read-only exploration to an `explore` subagent:

`task(type="explore", prompt=..., description="explore <domain>")`

The `explore` subagent runs in an isolated context — it sees nothing you gathered unless you inline it. The prompt must carry: the **inferred project goal** and the **datasource (+ dialect)** the domain lives in, the domain's file list + table list, and the already-gathered desc / sample / statistic summaries. It must instruct the subagent to **explore read-only and summarize**, returning a **structured result in the storage-classify taxonomy**:

```
subject (the domain)
  → store: one of semantic_models | metrics | reference_sql | knowledge | skills | memory | AGENTS.md | none
    → ref: file path / table name / column name
       rationale: one line — why this store (cite the storage-classify decision-tree branch)
       prompt-seed: the self-contained seed to hand the downstream generator — not just a bare ref but the context it needs (e.g. table names + the column encodings/intent for gen_semantic_model; the full SQL + the business question + any mandatory filter for gen_sql_summary)
```

Coverage focuses on **semantic_models, metrics, reference_sql, and knowledge** — `/init` already wrote the AGENTS.md inventory and the initial knowledge/memory, so here the explorer should surface vector-store candidates plus any **additional** knowledge atoms the corpus reveals (do not re-propose facts `/init` already filed; do not propose other stores beyond memory/AGENTS.md notes).

**For a validated-query corpus, instruct the explorer to enumerate EVERY `(question, SQL)` pair as its own `reference_sql` ref** (not a representative handful), and to carry each pair's **original natural-language question** in the `prompt-seed` (it is the best retrieval key for future questions). Large corpora: have the explorer return the full list in batches rather than truncating.

**Concurrency rule:** issue at most **3** `task` calls per batch (3 tool calls in one message), wait for the batch to return, then start the next batch. Set the domain's todo to `in_progress` when you launch it and `completed` when it returns. Tell the user (briefly) how many domains were dropped if you cap anything.

---

## Turn Boundary — Emit the Generation Manifest, then STOP

This manifest is the **single user confirmation point** — it must lead with the resolved scope (Step 0) so the user can correct the goal or scope here, not via an earlier question.

1. **Lead with the resolved scope** so it is explicitly confirmable:

   > **Inferred goal:** <1-2 sentences> — *(from `README.md` / directory name / table names)*
   > **In-scope datasources:** <names> — *(from hints / `agent.yml`)*
   > **Scope:** <files / tables / domains, or "whole project (no scope hints given)">

2. Aggregate every explore result, **dedupe** refs that appear under multiple domains, and build a **Generation Manifest** grouped by store, rendered as a Markdown table:

   | Subject | Store | Refs | Mechanism | Summary |
   |---------|-------|------|-----------|---------|
   | sales/orders | semantic_models | `orders`, `order_items` | `task(gen_semantic_model)` | core order facts |
   | sales/orders | metrics | GMV, AOV | `task(gen_metrics)` | built on orders measures |
   | … | reference_sql | `queries/top_skus.sql` | `task(gen_sql_summary)` | reusable ranking query |
   | … | knowledge | `status` enum on `orders` | `extract-knowledge` (lite) | atomic field-encoding fact |

3. **STOP here.** After printing the resolved scope + manifest, **end your turn**. Do **NOT** call any generation `task`, `extract-knowledge`, `gen_skill`, `add_memory`, or write any store yet. Do **NOT** call `ask_user`. State plainly: *"Reply to confirm, or correct the goal / scope / any manifest row, and I'll run the generation."* Wait for the user's next message.

---

## Step 3 — Route & Generate (next turn, after confirmation; concurrency ≤ 3)

Once the user confirms or corrects the manifest:

1. `load_skill("storage-classify")` and treat its **Decision Tree** + **Per-Store Reference** + **Context Handoff to Subagents** as the routing authority for every item.
2. **Make every delegated prompt self-contained (see storage-classify's *Context Handoff*).** The `explore` subagents that produced these refs are gone, and each generator runs in a fresh context — so inline the **datasource (+ dialect)**, the **business intent**, the **`prompt-seed` the explorer returned**, and the **rules/encodings already gathered** that the artifact must honor. Route each manifest item to its store with the prescribed mechanism:
   - **Light items** → write directly: `memory` via `add_memory` (≤ 2000 bytes); small AGENTS.md notes via `write_file` / `edit_file`.
   - **Heavy items** → delegate (the placeholders below are the *minimum* each prompt must carry):
     - semantic_models → `task(type="gen_semantic_model", prompt="<datasource> · <table name(s)> · intent · known column encodings / join-key traps>")`
     - metrics → `task(type="gen_metrics", prompt="<datasource> · metric name + definition · the base semantic model / measure it builds on · any mandatory filter>")`
     - reference_sql → `task(type="gen_sql_summary", prompt="<datasource/dialect> · the original natural-language question (if known) · the complete SQL · why it is written this way>")` — **one call per SQL, enumerate the whole corpus**: each query becomes its own `reference_sql` entry; index **every** `(question, SQL)` pair, do not select representatives (recall is driven by coverage). If a manifest row lists several SQLs, expand it into one `gen_sql_summary` call each; never pass multiple queries in a single prompt (they collapse into one mixed, unsearchable entry). Always pass the **original question** when the example came from one — it is the retrieval key future questions match against. The prompt must also instruct the generator explicitly: **"set `search_text` to the original natural-language question verbatim** (trim whitespace, keep its language); only fall back to keyword phrases when no original question exists" — `search_text` is the vector key the runtime embeds, and a user's question matches another question far better than it matches SQL keywords.
     - skills → `task(type="gen_skill", prompt="<skill intent + the concrete steps observed>")`
     - knowledge → run `extract-knowledge` in **lite** mode (do NOT trigger its deep blind-SQL flow); pass the **source (the SQL/doc/table) and the specific fact to mine**, plus the datasource it applies to. Only mine atoms `/init` did not already file — do not duplicate existing `./knowledge/*.md` entries.
3. **Ordering:** metrics build on semantic models — generate all `semantic_models` items **before** their dependent `metrics` items.
   - **Dual-route every `(question, SQL)` pair — this is required, not optional.** For each pair: (a) send it to `gen_sql_summary` so the example (with its original question) lands in `reference_sql`, AND (b) feed the same pair to `extract-knowledge` (lite) to mine the non-inferable rule. The example teaches *answer shape* (retrieved later for few-shot); the mined atom teaches *why* (encodings, mandatory filters, term→column mappings). One source, two stores — neither replaces the other.
4. **Concurrency ≤ 3:** dispatch heavy `task` calls in batches of at most 3, waiting for each batch. Update each item's todo (`in_progress` → `completed` / `failed`) as you go.

Do not hand-write semantic_models / metrics / reference_sql YAML yourself — always go through the matching subagent (per storage-classify's Forbidden rules).

---

## Step 4 — Refresh the AGENTS.md KB Index

After all generation completes, update `./AGENTS.md` **last**, following the *AGENTS.md Section Ownership* from `storage-classify`. **Do not rewrite the inventory sections `/init` owns** (`# title` · `## Architecture` · `## Directory Map` · `## Services` · `## Data Assets` · `## Recommended Tools` · `## SQL Conventions`). Use a scoped `edit_file` that touches only the KB index:

- **`## Semantic Models` / `## Metrics` / `## Reference SQL` — the vector-index sections you just populated.** `/init` does not write these sections, so **insert each one** (in canonical order) with **what it covers + how many + which tool retrieves it** (these stores are queried by retrieval, not read as files). **Only insert a section if you actually generated content for it** — never write a "none yet" placeholder for a store you produced nothing for:
  - `## Semantic Models` — `N` models (`schools`, `satscores`, `frpm`); retrieve with `search_semantic_model`.
  - `## Metrics` — `N` metrics (`county_avg_sat_math`, `avg_frpm_rate`, …); retrieve with `search_metrics`.
  - `## Reference SQL` — `N` validated queries; retrieve similar `(question → SQL)` examples with `search_reference_sql` before writing new SQL.
- **`## Knowledge` — append only.** This section is **owned by `extract-knowledge`** and `/init` already filed the initial atoms. If Step 3 mined *additional* knowledge, append one bullet per new `./knowledge/*.md` (`- [<Domain>](knowledge/<slug>.md) — <one-line scope>`), writing the scope line to convey unguessable specifics (exact thresholds, literal filter codes, enum spellings, term→column mappings). **Never overwrite existing entries.**

If `./AGENTS.md` does not exist (the user never ran `/init`), create the full file per the *AGENTS.md Section Ownership* — the same skeleton `/init` would have written — then fill the KB index as above. In that fallback only, also induce a `## SQL Conventions` section from the validated-SQL corpus (see below).

### `## SQL Conventions` (only when creating AGENTS.md from scratch)

If `/init` already wrote `## SQL Conventions`, leave it untouched. Only when you are creating `AGENTS.md` from scratch and the project has a validated-SQL corpus, **induce** its recurring output conventions and write them as a short bullet list. This section rides in `<project_context>`, so it nudges every downstream `gen_sql` toward this project's answer shape.

- **Induce, do not hardcode.** Read a representative slice of the corpus and state only patterns you actually observe, phrased schema-free (no specific table/column/code names).
- **Every rule carries its trigger phrasing** — write each as *"when the question says/asks ⟨observable phrasing⟩ → ⟨output shape⟩"*. A rule whose trigger you cannot state as question wording is not a rule — drop it.
- **Counter-example scan before persisting.** For each candidate rule, scan the corpus for pairs whose question matches the trigger but whose SQL has a *different* shape; any counter-example means narrow or drop the rule.
- If the corpus is absent or too small to induce any convention, **omit the `## SQL Conventions` section entirely** — do not write a placeholder.

### Make the KB reachable at runtime

The KB you just built is useless if `gen_sql` can't reach its retrieval tools. With the default tool-permission behavior, a `gen_sql` node whose `agentic_nodes.gen_sql` block omits `tools:` inherits node defaults (which include `context_search_tools.*`), so **no per-project `tools:` list is required**. Only if the project's `agent.yml` pins an explicit `tools:` for `gen_sql` must it include `context_search_tools.*`. Mention this in your closing note only when the config visibly restricts tools; otherwise leave config untouched.

Hard constraints:
- **AGENTS.md is a top-level overview, not a data dictionary.** Target **≤ 200 lines** (only the first ~200 lines are injected into `<project_context>`).
- For the semantic/metric/reference_sql index lines, state the **count and the retrieval tool** so a downstream agent knows the KB exists and how to consult it — do NOT inline their contents.
- Prefer a scoped `edit_file` over a full rewrite; ask via `ask_user` before overwriting an existing file wholesale.

Tell the user the KB is built and AGENTS.md's index is refreshed.

---

## Important Notes

- **Routing lives in `storage-classify`, not here.** When in doubt about which store an item belongs to, defer to its decision tree and disambiguation table.
- The explore subagent's `subject → store → ref` output is exactly `storage-classify`'s input contract — they dovetail.
- **Scope is the whole point of this skill vs `/init`.** Honor the resolved scope in every step — never scan, explore, or generate outside it unless the user gave no hints (whole-project default).
- **`/init` owns the file-based stores; this skill owns the vector stores.** Do not duplicate the AGENTS.md inventory or re-file knowledge/memory `/init` already wrote — only add what the heavy generation produces.
- Use placeholder comments when you cannot determine something rather than inventing facts.
- **Do not ask the user anything before the manifest.** Scope is resolved in Step 0 and confirmed at the turn boundary; the manifest is the single confirmation gate, not an `ask_user`. (The only later `ask_user` allowed is the Step 4 guard before wholesale-overwriting an existing `AGENTS.md`.)
