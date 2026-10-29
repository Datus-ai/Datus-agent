# Init

`init` is the lightweight way to bootstrap a project workspace. It reads your SQL scripts, docs and database schema, then writes a concise picture of the project — which tables to use, how they join, which rules apply — so future sessions start with the context a SQL-writing agent needs, without the heavier vector knowledge base.

Run it with the `/init` command inside the REPL.

## What it does

- Statically analyzes the project's SQL (ETL scripts and SQL embedded in Python DAGs) into a persisted table lineage graph with the `upsert_lineage` tool, then reads it with `query_lineage`. Reads joins, filters, mappings, windows and author comments directly from the source SQL. Nothing is executed.
- Reads the human-written docs in the project (business rules, metric definitions, data dictionaries).
- Checks the tables and columns it found against the database schema (table lists, column descriptions, DDL for declared keys and partitioning). It never queries table data, so row counts, freshness and actual code values are not measured.
- Writes established knowledge only from what a source states — DDL, docs, SQL comments, what a script literally does, scheduler config — and cites the source of grain, keys, cardinality and status. Conclusions it would have to infer, such as a grain read off a `GROUP BY` or which table version is current, are saved separately as unconfirmed questions in `knowledge/pending-questions.md`; the reply highlights important ones. Question→SQL examples can supply explicit terms and column mappings with source references; broader interpretations remain pending questions.
- Writes an `AGENTS.md` project map at the project root: data architecture, core tables, cross-domain rules, and a knowledge index.
- Writes one `knowledge/<domain>.md` file per business domain: table cards (use for, grain, time, codes, required filters), relationships, lineage, metric definitions, business rules, and known issues such as docs that disagree with the database.

It is the **lightweight** tier. REPL `/init` reuses the current chat session and completes initialization in one ordinary agentic loop, subject to context capacity and run budgets. It reuses an existing metadata KB for table search when available and never automatically runs metadata bootstrap. For the vector-indexed knowledge base, use [`/build-kb`](build_kb.md).

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

With no hints, `init` first explores core tables selected from docs, SQL outputs and related lineage on the active datasource. With hints, it only updates the in-scope parts of `AGENTS.md` and the knowledge files. Completion reports distinguish selected scope from the database as a whole.

`/init` accepts only an optional free-text scope:

```text
/init orders domain, focus on orders and customers
/init
```

Each `/init` runs within one ordinary Agentic loop in the current chat session. The model advances through business topics; the host does not start new chats per domain/table group or reset the budget after a limit. There is no fixed table quota. The run uses the configured `max_turns` and existing tool timeouts. Successful metadata reads are cached for this invocation. Failed calls can be corrected and retried within the same run budget and tool timeouts. The full init skill is injected into system instructions for this invocation, survives compaction, and is removed when normal chat resumes.

Read existing AGENTS.md, relevant knowledge, pending questions, docs, SQL and lineage first. For a large catalog without useful scope clues, use `ask_user` to request a schema, domain or table list. Mid-run clarification preserves existing results and other progress; later user corrections take precedence. Init does not bootstrap metadata. Existing KB search is optional and hides sample rows. Use metadata only, reading DDL when needed; never query business rows.

Use `todo_list/read/write/update` for business topics and work steps, adjusting them as needed rather than creating one task per table. Todo use neither activates plan mode nor requires plan confirmation. The runtime does not maintain a separate selected-table registry or per-table completion state.

The model incrementally edits AGENTS.md and knowledge/*.md through ordinary file tools, preserving human text, unrelated scopes and useful existing definitions. Write details before navigation and read back changes. Resolve structural questions with available tools before asking the user. Reserve pending questions for business ambiguities or unavailable information. Shared purpose, grain or similar SQL does not prove identical columns or keys: scope claims to actual evidence. Store unresolved questions in knowledge/pending-questions.md and do not use them as facts for answers, SQL or build-kb generation.

The final reply explains the current scope, outcomes, changed files, open questions and unfinished work. The CLI reports only whether execution returned or stopped and why. Metadata success, completed todos, file-only work and a normal reply do not certify business completion or full-database coverage. `knowledge/pending-questions.md` holds unresolved questions; todos and conversation summaries hold work progress.

Large results are archived intact and read with `read_tool_result`. Page budgets scale with the available model window, within bounded limits. Compaction retains a bounded set of exact recent archive identifiers, sections and cursors alongside the summary, so recovery does not depend on the model reproducing them. Automatic mid-turn compaction preserves the current goal, latest corrections, outcomes and next step, then continues the same loop. Read todo details on demand rather than injecting entire table lists. Long user clarifications use the same archive paging mechanism for complete recovery. Capacity checks include full input, output reserve and a safety margin; compaction does not reset the run budget. Interruption, timeouts, model errors or exhausted budgets preserve documents and report the concrete reason and unfinished scope. Ordinary tools and interactions are restored on exit, including errors; user confirmations continue in the same conversation.

There is no init task ID, persistent task queue or .datus/init/ checkpoint directory. Later invocations read existing documents and supplement them. Explicit user-selected plan mode remains read-only. Existing file permissions and concurrent-edit checks remain in force, without an init-specific content validator or multi-file transaction.

## Init vs. Build KB { #init-vs-build-kb }

| | `init` | [`/build-kb`](build_kb.md) |
|---|---|---|
| Speed | Minutes | Longer |
| Output | `AGENTS.md` + `knowledge/*.md` | Vector KB: semantic models, metrics, reference SQL |
| Confirmation | Scope clarification when needed | Manifest confirmation gate |
| Vector index | No | Yes |
| When | First, always | After init, when you want semantic search |

Typical flow: run `/init` first for immediate context, then `/build-kb` when you want the vector-indexed knowledge base.

## Notes

- Knowledge is stored as markdown files under `knowledge/`, indexed from `AGENTS.md`; only the first 200 lines of `AGENTS.md` are loaded into every session, so details live in the knowledge files.
- Where docs, SQL and the database disagree, `init` records the conflict under the domain's **Known Issues** instead of silently picking one side.
- `init` reads only schema metadata (table lists, column descriptions, table DDL); it never runs `SELECT` against table data.
- Continue answering pending questions in chat. Confirmed answers are written to the knowledge files or `AGENTS.md`, marked as confirmed by you; rejected ones are dropped. Unconfirmed questions are kept in `knowledge/pending-questions.md`, grouped by domain with affected tables and sources. The model maintains this list with ordinary file tools as questions arise, preserving other domains and human notes. The current run and later chats can read relevant entries through the link in `AGENTS.md`; long lists use search and bounded reads. After confirmation, the model updates established knowledge and removes or resolves only the matching question. The list is unconfirmed material, not facts for answers, SQL or build-kb generation; it is created only when needed.
- Generating semantic models, metrics, and reference SQL is the job of [`/build-kb`](build_kb.md).
