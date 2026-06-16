---
name: init
description: Lightweight project initialization — infer the project goal and in-scope datasources, scan the file tree and database metadata (db/table/desc/sample), classify into business domains, then write an AGENTS.md inventory skeleton plus the cheap file-based stores (atomic facts to ./knowledge/*.md via lite extract-knowledge, durable preferences to memory). Stops short of the expensive vector-indexed stores (semantic_models / metrics / reference_sql). Single confirmation-free pass, low token cost.
tags:
  - init
  - workspace
  - project
  - classify
version: 3.0.0
user_invocable: true
---

# Lightweight Project Initialization

You are initializing a project workspace **lightly**. The goal is a fast, low-cost first pass that gives a downstream agent a usable map of the project without paying for the heavy vector-store generation. You scan the project (files + database metadata), classify it into business domains, then write:

1. an **`AGENTS.md` inventory skeleton** (the project map injected into every node's `<project_context>`), and
2. the **cheap file-based stores** — atomic business facts to `./knowledge/*.md` (via lite `extract-knowledge`) and durable cross-session preferences to `memory` (via `add_memory`).

You **do not** build the vector-indexed stores (`semantic_models`, `metrics`, `reference_sql`) here — those cost tokens, fan out `explore` subagents, and write LanceDB, so they are out of scope for this lightweight pass. There is **no confirmation gate** in this skill: knowledge/memory/AGENTS.md are cheap markdown writes, so just do them.

You run in the main agent context, so you may call `todo_write`/`todo_list`/`todo_read`/`todo_update`, `add_memory`/`edit_memory`, the filesystem tools (`glob`, `grep`, `read_file`, `write_file`, `edit_file`), the database tools (`list_databases`, `list_tables`, `describe_table`, `get_table_ddl`, `search_table`, `read_query`), and `load_skill` (to run `extract-knowledge` lite and consult `storage-classify`).

**Routing authority is `storage-classify`.** This skill decides *what to scan*; for which content lands in `knowledge` vs `memory` vs `AGENTS.md`, follow `storage-classify`'s decision tree (branches 1, 5, 6) — do not re-invent routing here.

---

## Step 0 — Project Context (inferred, no questions)

**Do NOT call `ask_user` in this step.** Infer the project goal and the in-scope datasources from the repository and configuration.

1. **Infer the goal.** Read `README.md` if it exists (first 3000 chars); otherwise derive a 1-2 sentence goal from the directory name, top-level files, and the datasource/table names. State it as an explicit assumption you surface in `AGENTS.md`.
2. **Infer datasource scope from configuration — default to the single active datasource.** Use the **currently active datasource** of the session (the one pinned via `--datasource` / `/datasource`, i.e. the project's `default_datasource`). **When multiple datasources are configured, scope to that active one only — do NOT initialize all of them.** Fall back to the sole configured datasource when only one exists, or to a specific one the project files clearly reference. Broaden beyond the active datasource only if the user explicitly asks (via the free-text hints). Do not invent unconfigured services — only mention an extra service (Airflow, Superset, …) if a project file explicitly references it.
3. Use `glob` to scan the directory tree (top 3 levels). Skip hidden dirs and `__pycache__` / `node_modules` / `.venv`.

---

## Step 1 — Scan & Classify

Gather the raw material, then classify it into a **multi-level taxonomy of business domains / subtopics** (e.g. `sales/orders`, `sales/refunds`, `infra/etl`). This taxonomy feeds the `AGENTS.md` inventory and helps you locate atomic facts worth filing as knowledge.

**File side:**
- Use `glob` / `grep` to collect candidate files: `*.sql`, `*.md`, `*.yml` / `*.yaml`, scripts, configs, notebooks.
- **Note any validated-query corpus but do NOT enumerate it here.** A corpus of validated `(question, SQL)` pairs (a queries file, a golden/benchmark set, a saved-query catalog, a dbt/analysis SQL folder) feeds the vector-indexed `reference_sql` store, which is out of scope for this lightweight pass. You may mention its location under `## Data Assets` as a project asset, but do not generate `reference_sql` entries or a dedicated index section.

**Database side (for each in-scope datasource):**
- `list_databases` → `list_tables` to enumerate tables/views.
- For representative tables: `describe_table` (or `get_table_ddl`) for **desc** (column names/types/comments) and `search_table` (its `sample_data`) or `read_query("SELECT * FROM <t> LIMIT 5")` for **sample**. Sampling desc/sample is enough for the inventory — you do not need exhaustive statistics here.
- For large databases (>50 tables), sample representative tables per naming pattern rather than describing every table.

**Classify** every file and table into the domain taxonomy. A single domain may contain both files and tables.

**Record with todos when the taxonomy is large (> 3 domains):** call `todo_write` with one todo per domain — `title` = domain name (≤ 8 words), `content` = the files + tables it covers. Skip todos for small projects (≤ 3 domains) to avoid noise.

---

## Step 2 — File Cheap Stores (knowledge + memory)

While scanning, you will encounter **atomic business facts** that a downstream agent cannot infer from `INFORMATION_SCHEMA` / column comments alone — field encodings / enum / status codes, mandatory constant filters, join traps, business-term→field mappings. These are cheap to file (plain markdown, no vector index) and high-value.

1. **Atomic facts → `knowledge`.** Run `extract-knowledge` in **lite** mode (do NOT trigger its deep blind-SQL flow): pass the **source** (a table's comments/sample, a doc, a config) and the **specific fact to mine**, plus the datasource it applies to. It writes `./knowledge/<domain-slug>.md`. Only file facts that are genuinely non-inferable — skip anything mechanically derivable from the schema.
   - **Do not enumerate the validated-query corpus for knowledge here.** Mining the `(question, SQL)` corpus pair-by-pair is out of scope for this lightweight pass. Here, only file facts you can read directly off table metadata / docs.
2. **Durable preferences/context → `memory`.** A user/team habit or a default the agent should remember next session goes to `add_memory` (≤ 2000 bytes). Skip session-specific or one-shot content.

No confirmation gate — these are cheap, reversible writes. If a write would destructively overwrite an existing `./knowledge/*.md`, prefer a scoped `edit_file` and do not clobber unrelated entries.

---

## Step 3 — Write the AGENTS.md Inventory Skeleton

Write or update `./AGENTS.md`, following the *AGENTS.md Section Ownership* from `storage-classify`. The canonical section order is:

`# <project name>` · `## Architecture` · `## Directory Map` · `## Services` · `## Data Assets` · `## Recommended Tools` · `## SQL Conventions` · `## Knowledge`

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
- **This is the lightweight pass — stay in scope.** Do NOT call `gen_semantic_model` / `gen_metrics` / `gen_sql_summary`, do NOT fan out `explore` subagents, and do NOT emit a Generation Manifest — the vector-indexed stores are out of scope for this skill.
- Use placeholder comments (e.g. `<!-- Describe your architecture here -->`) when you cannot determine something rather than inventing facts.
- **Do not ask the user anything except before wholesale-overwriting an existing `AGENTS.md`.** Goal, datasource scope, knowledge, and memory are all inferred and written directly.
