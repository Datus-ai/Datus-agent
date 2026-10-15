# Init

`init` is the lightweight way to bootstrap a project workspace. It reads your SQL scripts, docs and database schema, then writes a concise picture of the project — which tables to use, how they join, which rules apply — so future sessions start with the context a SQL-writing agent needs, without the heavier vector knowledge base.

Run it with the `/init` command inside the REPL.

## What it does

- Statically analyzes the project's SQL (ETL scripts and SQL embedded in Python DAGs) into a persisted table lineage graph with the `upsert_lineage` tool, then reads it with `query_lineage`. Reads joins, filters, mappings, windows and author comments directly from the source SQL. Nothing is executed.
- Reads the human-written docs in the project (business rules, metric definitions, data dictionaries).
- Checks the tables and columns it found against the database schema (table lists, column descriptions, DDL for declared keys and partitioning). It never queries table data, so row counts, freshness and actual code values are not measured.
- Writes only what a source states — DDL, docs, SQL comments, what a script literally does, scheduler config — and cites the source of grain, keys, cardinality and status. Conclusions it would have to infer, such as a grain read off a `GROUP BY` or which table version is current, are not written: init lists them as pending questions at the end of its reply. Conventions generalized from a validated question→SQL corpus are the exception; they are written directly, citing the pairs they come from, for you to review.
- Writes an `AGENTS.md` project map at the project root: data architecture, core tables, cross-domain rules, and a knowledge index.
- Writes one `knowledge/<domain>.md` file per business domain: table cards (use for, grain, time, codes, required filters), relationships, lineage, metric definitions, business rules, and known issues such as docs that disagree with the database.

It is the **lightweight** tier: no vector index and no confirmation gate. It takes a few minutes on a project with a hundred or so scripts. For the vector-indexed knowledge base, use [`/build-kb`](build_kb.md).

The lineage graph is saved in `lineage/lineage.json`, one file for the whole project. Every analyzed file (or saved inline SQL) is a *source*: `upsert_lineage` re-analyzes only sources whose content, dialect, default database or analyzer version changed, and replaces everything a source contributed before, so lineage from deleted statements disappears. The dialect and default database are saved in the file as project settings, so a later call that omits them analyzes the same way. `prune_missing=True` also removes sources whose files were deleted, and `delete_lineage` removes sources by ID or glob. The file holds four parts: `settings` (dialect and default database), `sources` (content hash, dialect, default database, parse issues), `nodes` (tables, views and `query:<hash>` nodes for pure queries, with the author's comment as `label`) and `edges` (upstream → downstream, each with the statements that produced it: source, line, operation and folded temporary tables; a write that reads no table has no upstream). Source IDs are relative to the project root. The file holds only facts derived from the SQL, with no timestamps or counters, so it can be committed; if a merge leaves it unreadable, `upsert_lineage(paths=[...], rebuild=True)` rebuilds it from the SQL files.

`query_lineage` returns the subgraph around the requested tables (or the whole table graph) with one record per statement, `file_id:line` evidence, each table's role and connected component, and how many saved queries read it. `complete=false` means a source that may involve the requested tables, even one no returned statement cites, changed since it was analyzed, was analyzed under other settings, or has statements that failed to parse.

Under the `normal` permission profile, `upsert_lineage` and `delete_lineage` ask before writing `lineage/lineage.json`; `auto` runs `upsert_lineage` without asking and still asks before `delete_lineage`.

**Breaking change:** `extract_sql_lineage` is replaced by `upsert_lineage`, `delete_lineage` and `query_lineage`. The lineage tools no longer return joins: initialization reads join conditions from the source SQL and states a cardinality only when the join covers every column of a declared primary or unique key; a cardinality the scripts merely suggest becomes a pending question. Parameterized predicates do not prove incremental loading; ROW_NUMBER alone does not prove table grain.

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
- `init` reads only schema metadata (table lists, column descriptions, table DDL); it never runs `SELECT` against table data.
- Answer the pending questions in the same chat. Confirmed answers are written to the knowledge files or `AGENTS.md`, marked as confirmed by you; rejected ones are dropped. Unanswered questions are not saved anywhere.
- Generating semantic models, metrics, and reference SQL is the job of [`/build-kb`](build_kb.md).
