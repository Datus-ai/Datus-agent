# Init

`init` is the lightweight way to bootstrap a project workspace. It reads your SQL scripts, docs and database, then writes a concise picture of the project — which tables to use, how they join, which rules apply — so future sessions start with the context a SQL-writing agent needs, without the heavier vector knowledge base.

Run it with the `/init` command inside the REPL.

## What it does

- Statically analyzes the project's SQL (ETL scripts and SQL embedded in Python DAGs) with the `extract_sql_lineage` tool: table dependencies, join keys, directions and expressions. Reads filters, mappings, windows and author comments directly from the source SQL. Nothing is executed.
- Reads the human-written docs in the project (business rules, metric definitions, data dictionaries).
- Verifies what it found with a few dozen cheap, read-only database queries: which table versions exist and are fresh, table grain, join cardinality, actual code values.
- Writes an `AGENTS.md` project map at the project root: data architecture, core tables, cross-domain rules, and a knowledge index.
- Writes one `knowledge/<domain>.md` file per business domain: table cards (use for, grain, time, codes, required filters), relationships, lineage, metric definitions, business rules, and known issues such as docs that disagree with the database.

It is the **lightweight** tier: no vector index and no confirmation gate. It takes a few minutes on a project with a hundred or so scripts. For the vector-indexed knowledge base, use [`/build-kb`](build_kb.md).

The analyzer returns `schema_version=5`: the complete dependency graph and resolved joins in one response, without `sections`, pagination or result-size clipping. `lineage` includes SELECT-only reads (`target=null`), separating different SQL and write operations. Each join contains only `expression` and `evidence`: the expression combines relations, aliases, join direction and conditions. Physical tables appear as `tables` IDs (`#3 AS o`); CTE/derived relations keep their scope name followed by the physical tables they read (`x{#1,#2}`) — read their projections in the original SQL. `files` stores paths once, and `=f3` marks a byte-identical copy of `f3` that is analyzed and cited only once. Join evidence uses `file_id:start-end` (or a single line) for the actual condition; `@statement` explicitly marks a statement-start fallback. Lineage evidence still locates statement starts. Detailed `issues` are omitted; `complete=false` signals inputs or relationships that need source inspection.

**Breaking change from schema v2:** remove `sections`, `max_items`, `max_files`, `offset`, `result_path` and `max_output_chars` from calls. The tool no longer returns rules, comments, table/root inventories or detailed occurrence collections. Read source files for business rules and explanations. Initialization inventories filters, CASE/IF, windows and comments across the scripts, then reads core definitions in context. Parameterized predicates do not prove incremental loading; ROW_NUMBER alone does not prove table grain.

## When to use it

- A new project workspace has no `AGENTS.md` yet.
- You explicitly want to initialize or re-scan the project.
- Scripts, docs or schemas changed materially and the project map is stale.

You can skip it when an up-to-date `AGENTS.md` already exists and nothing material has changed.

## How to use it

```text
/init
```

You can add optional free-text hints after the command — a goal, a scope, or specific files or tables to focus on:

```text
/init focus on the sales and finance schemas
```

With no hints, `init` covers the whole project and the active datasource. With hints, it only updates the in-scope parts of `AGENTS.md` and the knowledge files.

## Init vs. Build KB { #init-vs-build-kb }

| | `init` | [`/build-kb`](build_kb.md) |
|---|---|---|
| Speed | Minutes | Longer |
| Output | `AGENTS.md` + `knowledge/*.md` | Vector KB: semantic models, metrics, reference SQL |
| Confirmation | None | Manifest confirmation gate |
| Vector index | No | Yes |
| When | First, always | After init, when you want semantic search |

Typical flow: run `/init` first for immediate context, then `/build-kb` when you want the vector-indexed knowledge base.

## Notes

- Knowledge is stored as markdown files under `knowledge/`, indexed from `AGENTS.md`; only the first 200 lines of `AGENTS.md` are loaded into every session, so details live in the knowledge files.
- Where docs, SQL and the database disagree, `init` records the conflict under the domain's **Known Issues** instead of silently picking one side.
- `init` runs only read-only queries (`SELECT` / `SHOW` / `EXPLAIN`).
- Generating semantic models, metrics, and reference SQL is the job of [`/build-kb`](build_kb.md).
