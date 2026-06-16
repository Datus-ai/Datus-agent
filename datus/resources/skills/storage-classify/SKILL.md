---
name: storage-classify
description: Decide where a produced artifact must be persisted before writing it, then route it the prescribed way — semantic_models / metrics / reference_sql via the matching task() subagent, knowledge via extract-knowledge (lite), memory via add_memory, skills via create-skill, and AGENTS.md edited directly. Load before persisting any business fact, validated SQL, metric/model definition, session preference, project convention, or reusable workflow.
tags:
  - storage
  - classification
  - routing
  - persistence
version: "1.3.0"
user_invocable: false
disable_model_invocation: false
---

# Classify and Route Persistent Storage

You receive one or more pieces of content that *might* be worth persisting (a business fact, a validated SQL, a metric or model definition, a session preference, a project convention, a reusable workflow). For each piece you must:

1. **Classify** it into exactly one of the seven persistent stores — or decide it should NOT be persisted.
2. **Route** it the prescribed way for that store (delegate to a `task` subagent / another skill / a memory tool, or edit `AGENTS.md` directly).

You are a router and curator, not a generator. The goal is that every artifact lands in the one store that fits its shape, lifetime, and reuse pattern — and is written with the mechanism that owns that store.

## Role & Boundary

- **`semantic_models` / `metrics` / `reference_sql` MUST go through a `task` subagent.** Never hand-write their YAML files or vector-store rows yourself — the subagents own schema, validation, and indexing.
- **`knowledge` goes through `extract-knowledge` in lite mode** — summarize the atomic fact directly against the source; do NOT trigger the deep blind-SQL-iteration flow.
- **`memory` goes through `add_memory` / `edit_memory`** — the only writers; they enforce the 2000-byte cap.
- **`skills` goes through `create-skill`.**
- **`AGENTS.md` is the only file you edit yourself** (`write_file` / `edit_file`).
- When content does not clearly belong to any store, or routing would overwrite something destructively, call `ask_user`. If `ask_user` is unavailable, report the gap and do **not** write by default.
- **This skill only routes — it does not own persistence mechanics.** Whether a store is a flat file, a vector index, or both, and the durability of that write, is the responsibility of the target subagent/skill (e.g. `gen_sql_summary` must persist *and index* its `reference_sql`). Do not add file-vs-index verification logic here.

## Context Handoff to Subagents

**Every `task()` subagent and `extract-knowledge` run in an isolated context.** They do NOT inherit the current session/conversation, the project goal, the active datasource, the inventory you built, or the facts you just gathered. A bare table name or a SQL string with no surrounding context yields a generic, poorly-indexed artifact (wrong dimensions, empty `search_text`, a summary that ignores a mandatory filter). So whoever dispatches the `task` MUST inline everything the generator needs. At minimum pass:

- **Datasource** name (and dialect) the artifact targets — so paths, identifiers, and YAML `data_source` resolve to the right place.
- **Business intent / the question** the artifact answers — what it is for, in one or two lines.
- **Rules and encodings you already discovered** that the generator must honor (a mandatory constant filter like `rtype='D'`, a column encoding, a join-key trap), so it does not rediscover, ignore, or contradict them.
- For `reference_sql` / `metrics`: the **knowledge atoms mined from the same source**, so the summary / `search_text` / measure expression stay consistent with the rule that explains the SQL.

The dispatching skill owns this handoff — the prompt seeds in the decision tree below (`<table names + intent>`, `<complete SQL + business context>`, …) are minimums, not the whole prompt.

## Decision Tree

Walk top to bottom; the first match wins. Each item routes to exactly one store.

1. **Reusable atomic business fact or rule** — field encoding / enum / status code, mandatory constant filter, join trap (must go through a mapping table), boundary trap (strict vs non-strict inequality), business-term-to-field mapping, same-name-field divergence.
   → **knowledge** — delegate to `extract-knowledge` (lite mode).
2. **Structured semantic definition of a table** — its identifiers, measures, dimensions (the structure, not a single metric).
   → **semantic_models** — `task(type="gen_semantic_model", prompt=<table name(s) + intent>)`.
3. **A reusable metric built on a measure** — a named business metric, its aggregation expr, base measures, available dimensions.
   → **metrics** — `task(type="gen_metrics", prompt=<metric description or SQL>)`.
4. **A complete, validated SQL worth indexing for semantic search / future reuse** (plus a human summary).
   → **reference_sql** — `task(type="gen_sql_summary", prompt=<the complete SQL + business context>)`. **One SQL = one call = one entry.** `gen_sql_summary` produces exactly one `reference_sql` row per invocation, so dispatch a **separate** `task(gen_sql_summary)` call for every distinct query. NEVER bundle multiple SQLs into one prompt — that collapses them into a single useless entry whose `sql`/`search_text` mixes unrelated queries and breaks few-shot retrieval. When the query came with an original natural-language question, pass it along and instruct the generator to use it **verbatim as `search_text`** — a future user question matches a stored question far better than it matches SQL keywords.
5. **A high-level project overview** — architecture, directory map, services, data assets, or the knowledge index.
   → **AGENTS.md** — edit the file directly (see *AGENTS.md Section Ownership*).
6. **A lightweight, cross-session preference or context bound to one agent** (≤ 2000 bytes) — a user/team habit, a default the agent should remember next session.
   → **memory** — `add_memory` / `edit_memory`.
7. **A reusable, multi-step workflow or operating procedure** that will be invoked repeatedly.
   → **skills** — delegate to `create-skill`.
8. **None of the above** — one-shot, inferable from `INFORMATION_SCHEMA` / table comments / column names, or not reusable.
   → **Do NOT persist.** Say so and stop.

## Per-Store Reference

| Store | Path | Format | Route (mechanism) |
|-------|------|--------|-------------------|
| semantic_models | `./subject/semantic_models/{datasource}/{name}.yml` (anchored to project root) | YAML — `data_source` with `identifiers` / `measures` / `dimensions`; also mirrored into LanceDB `semantic_model` | `task(type="gen_semantic_model", …)` — prompt MUST name the table(s) |
| metrics | LanceDB `metrics` table (`~/.datus/data/{project}/datus_db/`) | Vector rows — `measure_expr`, `metric_type`, `base_measures`, dimensions | `task(type="gen_metrics", …)` — built on a semantic model's measures |
| reference_sql | `./subject/sql_summaries/{id}.yaml` + LanceDB `reference_sql` | YAML — `id` / `name` / `sql` / `summary` / `search_text` / `tags` | `task(type="gen_sql_summary", …)` — prompt MUST carry **one** complete SQL; **one call per query** (never batch multiple SQLs into a single entry) |
| knowledge | `./knowledge/<domain-slug>.md`, indexed under `AGENTS.md ## Knowledge` | Markdown atomic facts (no longer a vector store) | `extract-knowledge` (**lite** mode) |
| memory | `{workspace_root}/.datus/memory/{node}/MEMORY.md` (only `chat` and custom subagents) | Markdown — **hard 2000-byte cap** | `add_memory` / `edit_memory` (the only writers) |
| AGENTS.md | `./AGENTS.md` (project root; first ~200 lines injected into `<project_context>`) | Markdown — Architecture / Directory / Services / Knowledge … | **edit directly** (`write_file` / `edit_file`) |
| skills | `./.datus/skills/` (project) > `~/.datus/skills/` (user) > `datus/resources/skills/` (builtin); first-wins | `SKILL.md` + YAML frontmatter | `create-skill` |

**When NOT to pick a store:**

- *Not semantic_models* — a single metric (that's metrics); a one-table fact already in the column comment.
- *Not metrics* — structural table/column/measure definitions (that's semantic_models).
- *Not reference_sql* — a one-shot query never to be reused; or the *reason* a query is written a certain way (that atomic rule is knowledge).
- *Not knowledge* — anything inferable from `INFORMATION_SCHEMA` / comments, generic SQL knowledge, or a fact that mechanically composes from existing facts.
- *Not memory* — a team-level long-lived business fact (that's knowledge); anything that would exceed 2000 bytes.
- *Not AGENTS.md* — fine-grained atomic facts (those are knowledge; AGENTS.md only holds their index).
- *Not skills* — a declarative "what is true" fact (that's knowledge); skills capture executable "how to do" procedures.

## AGENTS.md Section Ownership

Because you edit `AGENTS.md` yourself, keep its sections consistent and minimal. Sections and their intent:

| Section | Purpose | Edit constraint |
|---------|---------|-----------------|
| `# <project name>` | One-line project description | Keep one line |
| `## Architecture` | Data flow / stack / how services connect | Brief; ASCII diagram only if complex |
| `## Directory Map` | Directory / Purpose / Key Entry Point / Consumer | Table; main dirs only |
| `## Services` | Service / Type / Connection / Description | Configured DBs + user-mentioned services |
| `## Data Assets` | Per-database high-level summary | **Never enumerate every table** — categorize + count |
| `## Recommended Tools` | Runtime tools per service type | Only configured/mentioned services |
| `## SQL Conventions` | Project's induced SQL output discipline (answer-shape rules) | **Induced from the project's validated-SQL corpus** by `/init`; schema-free bullets; project-level presentation guidance, NOT knowledge |
| `## Semantic Models` | KB index: how many models, which tables | Count + `search_semantic_model`; do NOT inline definitions |
| `## Metrics` | KB index: how many metrics, key names | Count + `search_metrics`; do NOT inline definitions |
| `## Reference SQL` | KB index: how many validated queries | Count + `search_reference_sql`; retrieved for few-shot, not read as files |
| `## Knowledge` | Index of `./knowledge/*.md` (mapped to files) | **Maintained by `extract-knowledge`** — do not overwrite its entries |

**AGENTS.md is the KB entry point.** It is injected into every node's `<project_context>`, so it is the one reliable place that tells a downstream agent *what KB exists and how to reach it*. Index retrieval-backed stores (semantic_models/metrics/reference_sql) by **count + which search tool**; index `knowledge` by **file links**. Never inline the stores' contents.

**Write only sections that have real content.** Omit any section you have nothing concrete to put in — never write empty "none yet" / "not built yet" placeholder lines. A section is added later (per the handling order below) when content for it actually exists.

Handling order:

1. **`AGENTS.md` missing** → `write_file` a minimal skeleton (`# <project dir name>` plus the relevant section); leave the inventory + knowledge/memory sections for `/init` to fill in and the vector-index sections (`## Semantic Models` / `## Metrics` / `## Reference SQL`) for `/build-kb`. Do not block on the user running `/init` first.
2. **Exists but missing the target section** → insert it in the canonical order above (Knowledge goes last).
3. **Exists with the section** → prefer a minimal `edit_file`. Never rewrite the whole file when a scoped edit suffices, and never touch `## Knowledge` entries owned by `extract-knowledge`.

## Disambiguation (overlapping boundaries)

- **knowledge vs memory** — a team-level, long-lived business fact → knowledge; a session-level, agent-bound preference or working context → memory.
- **reference_sql vs knowledge — not exclusive, they feed each other.** The complete reusable SQL → reference_sql (an example that teaches *answer shape*, retrieved later for few-shot); the atomic business *rule* mined from why that SQL is written that way → knowledge (teaches *why*). A single `(question, gold_sql)` pair routes to BOTH: store the example, and run `extract-knowledge` to mine the rule.
- **output / answer conventions are NOT knowledge.** "Return only the ratio column", "match the exact column projection", "don't add helper columns", "count questions return a single scalar" — these are *presentation conventions*, not facts about the data. They live in two project-level places: as `reference_sql` examples (few-shot, the strongest teacher) and, when a recurring pattern can be induced from the corpus, as schema-free bullets in `AGENTS.md ## SQL Conventions` (rides in `<project_context>`). They are NOT knowledge atoms and NOT a global prompt rule — answer shape is project/context-specific (strict-eval wants minimal projection; interactive BI often wants extra context columns). Only durable data-model facts (field encodings, term→field mappings, mandatory filters) are knowledge. Never encode a benchmark/eval quirk as knowledge.
- **semantic_models vs metrics** — the structure (table/column/measure/dimension definitions) → semantic_models; a concrete metric built on a measure → metrics.
- **AGENTS.md vs knowledge** — a high-level overview or index → AGENTS.md; a fine-grained atomic fact → knowledge (AGENTS.md only indexes it).
- **skills vs knowledge** — an executable, repeatable procedure ("how to do") → skills; a declarative statement ("what is true") → knowledge.

## Output / Behavior

For each piece of input:

1. State the verdict: `{ target store, the decision-tree branch that matched and why }`.
2. **Execute the routing** for that store — call the prescribed `task` subagent / skill / memory tool, or directly edit `AGENTS.md`.
3. If the content does not clearly fit any store, or routing would destructively overwrite existing content, call `ask_user` with numbered options before acting.

Process multiple pieces one at a time. When `ask_user` is unavailable and a verdict is ambiguous, report what is missing and route nothing by default.

End with a short human-readable summary: how many pieces were classified, where each was routed (or why it was not persisted), and any items still awaiting user confirmation. Do not return a JSON envelope.

## Forbidden

- Do not hand-write `semantic_models` / `metrics` / `reference_sql` YAML or vector-store rows — always delegate to the matching `task` subagent.
- Do not bundle multiple SQLs into one `gen_sql_summary` call — one query per call, one entry per query.
- Do not run `extract-knowledge` in deep mode — use lite.
- Do not exceed the 2000-byte `memory` cap, and do not write `memory` with any tool other than `add_memory` / `edit_memory`.
- Do not overwrite `## Knowledge` entries in `AGENTS.md` — those are owned by `extract-knowledge`.
- Do not persist one-shot content, generic SQL knowledge, or anything inferable from `INFORMATION_SCHEMA` / table comments / column names.
- Do not generate the artifact's content yourself (e.g. write the SQL, invent the metric) — your job is classification and routing, not authoring.
- Do not resolve a destructive overwrite silently — go through `ask_user`.
