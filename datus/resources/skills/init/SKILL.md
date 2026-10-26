---
name: init
description: Lightweight initialization of a project's core scope, optionally limited to files, tables or domains. Reads SQL and persisted lineage, human docs and live schema without querying table data. Writes confirmed knowledge to AGENTS.md and knowledge/*.md; unconfirmed inferences are kept separately in knowledge/pending-questions.md. Each call handles its current scope using existing documents and one ordinary chat loop. Reuses an existing metadata KB without bootstrapping one; semantic models, metrics and reference SQL belong to build-kb.
tags:
  - init
  - workspace
  - project
  - lineage
version: 4.25.0
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

- **Confirmed statements only.** Established knowledge holds only what a source states directly: DDL and schema declarations, doc sections, SQL comments, what a script literally does (tables read and written, joins, filters, mappings, formulas), lineage, scheduler config, and what the user confirmed. A conclusion drawn from those sources is an inference: a grain read off a `GROUP BY` or `ROW_NUMBER`, a cardinality read off join syntax, a cadence read off names or parameters, a table judged current without a doc naming it, a layer meaning read off a prefix, a rule generalized from repeated SQL. Keep inferences as explicitly unconfirmed questions in `knowledge/pending-questions.md`, never as established facts in domain knowledge or AGENTS.md.
- **Cite** grain, keys, cardinality, status and time as `(DDL)`, `(doc: <name> §<section>)`, `(<file>:<line>)` or `(confirmed by user, <YYYY-MM-DD>)`. Never invent a table, column, code value or number.
- **Preserve definitions.** Record metric expressions, filters and join predicates with their source locations. Keep each definition within its evidenced scope; possible risks are not confirmed business rules.
- **Where inference hides:**
  - Grain comes only from a declared primary or unique key or a doc. A key that allows duplicates, a sort key or a distribution key does not make rows unique.
  - Write `N:1` only when the join equates every column of a declared primary or unique key on the "one" side; a join on part of a composite key can still fan out.
  - Cadence and load mode come only from scheduler config or docs. Otherwise write what the load statement literally does, with its location, e.g. overwrites the partition of the run date (`<file>:<line>`).
  - "Authoritative" or "legacy" only when a doc or the user says so. Otherwise write the facts on each candidate's card (which tables read it in lineage, which job builds it, what the docs say) and ask which one is current.
- **Who wins.** For business definitions, docs before SQL comments before SQL. For what exists, the database schema. Published data follows its build script.
- **Disagreements are findings.** Compare what the docs say with what the scripts do and what the schema holds. Record every disagreement under the domain's `## Known Issues` with both sides and what the downstream agent should do; never silently pick one.

## Limits

- **No table data.** Use schema metadata only: `list_tables`, `describe_table`, and `get_table_ddl` when a specific important claim needs more evidence. No business-row queries, samples or counts. The bounded REPL run exposes metadata-only DB tools; general SQL, shell, MCP and delegation tools are not part of it.
- **Clarify scope when needed.** If many tables are visible and project docs, SQL, existing metadata and user hints do not establish a useful business scope, use `ask_user`. Ask for a schema, business domain or explicit table list. Wait for the answer; if cancelled, stop and preserve progress. You may also ask during analysis when an answer is needed to proceed. Do not guess core tables from names or fill a table-count quota. Save other unresolved business questions in the pending-question file during analysis and summarize important ones in the final reply. Update only the sections you own in existing output files.
- **Scope.** Hints after `/init` arrive as "Additional context from the user" and name files, datasources, tables or domains. With hints, read and write only in scope and keep every other section verbatim. Without hints, select core tables from docs, SQL outputs and related lineage on the active datasource. Prioritize important business questions. Core-scope completion never means exhaustive database coverage. Do not generalize facts from representative tables to unchecked tables.
- **Lightweight.** Do not build semantic models, metrics or reference SQL (`/build-kb` owns them): no `semantic_modeling`, no `gen_sql_summary`, no generation manifest.
- **Existing metadata only.** If `search_table` is available, use it to find a few candidates; empty search is not proof of absence. Otherwise follow docs, SQL and scoped table lists. Never bootstrap metadata as an init prerequisite. Verify important structural claims against the live database.

## Current-Scope Execution

When the host provides `<init_context>`, complete this invocation within the current ordinary chat loop. Move through useful business topics at your discretion. Pagination and automatic mid-turn compaction let the same run continue; compaction does not reset `max_turns` or start another session. Each later `/init` supplements the existing documents. There is no init task ID, persistent queue or resume state.

- **Explore scope**: read existing knowledge, relevant pending questions, project docs, SQL and optional metadata search. Pending questions are not evidence. For a large database without reliable scope clues, use `ask_user` for a schema, domain or table list. For overly broad work, clarify the useful area first. Continue here after the answer; preserve existing progress and apply later user corrections.
- **Plan work with existing todos**: use `todo_list`, `todo_read`, `todo_write` and `todo_update` for business topics and next steps. These tools do not activate plan mode or require plan approval. Preserve unrelated existing todos. Use `plan_init_scope` only to register selected tables and relevant sources or remove names from current coverage. It does not schedule work, impose a table-count quota or require ending the chat. Do not maintain a second execution plan in init records.
- **Analyze progressively**: decide which related tables, metadata searches, source reads and SQL lineage updates are useful. Dependency lookups do not enlarge selected coverage. Use live metadata to check important structural claims, and read DDL only when useful. Match each claim to evidence for those exact tables: shared purpose, grain, names or SQL do not establish identical columns or keys. Before writing "verified" or "identical", check the supporting evidence; otherwise narrow the claim and mark remaining structures unverified.
- **Resolve questions with tools first**: within the chosen scope, use metadata and project sources to resolve structural questions before asking the user or saving pending questions. An unchecked schema is a next lookup. Continue useful feasible checks, without exhaustive checks that add no value. Keep genuine business ambiguities or unavailable information for user confirmation. Never query business rows to fill a knowledge gap.
- **Edit incrementally**: use ordinary `read_file`, `edit_file` and `write_file` tools. Preserve unrelated scope, human text and useful prior definitions. Combine repeated definitions and citations; explain conflicting definitions with sources. A missing finding does not erase earlier knowledge. Write details before navigation and read back the changes. Save unresolved business questions in `knowledge/pending-questions.md` as they arise; link it from AGENTS.md as unconfirmed material. Existing knowledge is context, not independent evidence for a new claim.
- **Record outcomes and continue**: use `report_init_progress` to record a short summary, explicitly completed tables, remaining work and next step. Metadata success alone is not knowledge completion. The report does not end this chat; choose the next useful action yourself. Use `[]` for file-only work. Set `scope_complete=true` only when the requested current scope is complete, explaining the completed file work when no tables are involved. An empty selection or normal reply alone is not a completion report. `get_init_status` provides bounded pages of coverage, sources and user clarifications; todo tools hold work steps. The program records reports without certifying business content or full-database coverage.
- **Large or interrupted work**: read archived results through `read_tool_result`, search and read long files in bounded sections, and continue the current task after automatic compaction. Recover current goals, latest user corrections, recorded outcomes and next steps from the runtime reminder, todos and relevant knowledge; do not reload entire archived results. Tools retain their own timeouts and the run retains its configured `max_turns`. When interrupted, budget-exhausted or unable to recover capacity, stop and report the concrete reason and remaining work. Never restart in a fresh chat to reset the budget. Explicit user-selected plan mode remains read-only; ordinary todo use does not enable it.

Copy archive identifiers, sections and pagination cursors exactly from tool results or the runtime's retained references. A preview, captured count or archived full result does not establish that you inspected every record. Describe the claims actually checked and keep unfinished lookups visible in progress and the final reply.

Before finishing, read back the written files and keep the verified scope consistent across the overview, details, progress and final reply.

Without `<init_context>`, perform the same scoped exploration and editing through the available ordinary tools.

## Where to Look

Read the highest-value sources first and let what you learn decide what to read next:

1. Human docs and instruction files: `README`, an existing `AGENTS.md`, business rules, metric definitions, data dictionaries.
2. The SQL. `upsert_lineage(paths=[<in-scope SQL globs>], prune_missing=True)` builds the lineage graph; use `query_lineage(tables=[<relevant tables>])` for related statements. Read source SQL for formulas and filters. Published output scripts are useful starting points; graph degree alone does not establish business authority. SQL comments explain filters and workarounds.
3. Scheduler or pipeline config.
4. The database schema: check that every table and column you write exists, and read declared keys and partitioning.

When the script headers pose business questions answered by the SQL below them, the scripts are a **question→SQL corpus**: use their explicit terms and column mappings as sourced examples, without generalizing beyond those examples.

## Output

Maintain these public artifacts using the existing file tools:

- `knowledge/<domain>.md`: useful table guidance, relationships, business definitions, source references and known issues. Use additional detail files when needed; preserve the project's existing organization.
- `knowledge/<domain>-index.md`: optional navigation and overview when a domain has several detail files. Small domains can link directly from `AGENTS.md`.
- `AGENTS.md`: concise project context and knowledge navigation, within 200 lines, preserving human content and links to other domains. When the pending-question file exists, link it within this navigation and explicitly state that it is unconfirmed material, not facts to use for answers, SQL or KB generation.
- `knowledge/pending-questions.md`: a shared Markdown list of unresolved business questions or information the available tools cannot establish, grouped by domain. Resolve questions answerable by metadata or project files before adding them here; report deferred lookups as unfinished work rather than user questions. For each question name the affected tables and source references, and explain what needs confirmation. Read before editing; combine duplicates and preserve unrelated questions and human text. Use search and bounded reads when it grows. Create it only when questions exist; do not overwrite an existing list with an empty file. No question IDs or structured submission are required.
- `lineage/lineage.json`: shared SQL lineage, updated only through the lineage tools. Prefer querying relevant named tables to keep exploration focused.

Do not create `.datus/init/` task metadata. The ordinary file tools handle edits; there is no init-specific renderer, content validator or multi-file transaction. After interruption, later scoped calls read the existing files and supplement unfinished documentation.

Reply with the current scope, files changed, checked tables, important unknowns, source differences and unfinished areas. Do not claim full-database completion. End with at most 10 consequential pending questions, each citing its evidence and the statement that would be written if confirmed. The file retains every unresolved question. Proposed interpretations in that file are not confirmed definitions and must not be used as defaults for SQL, answers or vector-KB generation.

When the user later confirms or corrects a pending interpretation, read the matching entry, update the relevant knowledge with user attribution, then remove that entry or mark it resolved. Discard rejected interpretations and preserve unrelated pending questions. A new source may resolve a question only when it directly establishes the fact; an inference remains unconfirmed.
