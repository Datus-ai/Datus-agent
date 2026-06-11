---
name: session-summarize
description: Review the current chat session and persist its valuable takeaways — business facts/rules, validated SQL, metric/model definitions, durable preferences, project conventions, reusable workflows — by classifying each via storage-classify and routing it to the right store. Present a Summary Manifest and STOP for confirmation before any heavy generation. Use at the end of a working session or when the user asks to capture what was learned.
tags:
  - session
  - summarization
  - persistence
  - classify
version: "1.1.0"
user_invocable: true
---

# Session Summarize

You are wrapping up the **current chat session**: harvest everything worth keeping from the conversation, classify each piece, and persist it into the right store. You are a curator of session takeaways, not a generator.

You run in the main agent context, so you may call `task`, `load_skill`, `todo_write`/`todo_list`/`todo_update`, `add_memory`/`edit_memory`, the filesystem tools (`glob`, `grep`, `read_file`, `write_file`, `edit_file`), and the search tools (`search_semantic_model`, `search_metrics`, `search_reference_sql`).

**Routing authority is `storage-classify`.** This skill decides *what to harvest from the session*; the `storage-classify` skill owns *which content goes to which store, written with which mechanism*. Do NOT re-invent storage routing rules here — load and follow `storage-classify` in Step 2.

**Two-phase contract (important).** Heavy generation (`gen_semantic_model` / `gen_metrics` / `gen_sql_summary` / `gen_skill` / `extract-knowledge`) costs tokens and writes real files. So there is a hard **turn boundary** after classification: you produce a Summary Manifest and then **end your turn**. Do NOT call `ask_user` for this confirmation — just present the manifest and stop. The user confirms or corrects it in their next message; only then do you run Step 3.

---

## Step 1 — Harvest Candidates from the Session

Review the **current session's conversation** (the transcript already in your context) and collect every piece that *might* be worth persisting. Look for:

- **Business facts / rules** surfaced during the session — field encodings / enums / status codes, mandatory constant filters, join traps (must go through a mapping table), boundary traps (strict vs non-strict inequality), business-term-to-field mappings, same-name-field divergence.
- **Validated SQL** — a complete query the user ran and accepted (plus the business question it answered), worth indexing for future few-shot reuse.
- **Metric / model definitions** produced or refined this session — a named metric and its measure, or a table's identifiers / measures / dimensions.
- **Durable preferences or working context** the agent should remember next session (≤ 2000 bytes, bound to one agent).
- **Project conventions / overview** facts that belong in `AGENTS.md`.
- **Reusable, multi-step workflows** the session repeated and that will recur.

**Exclude (do NOT persist):** one-shot exploration, anything inferable from `INFORMATION_SCHEMA` / table comments / column names, generic SQL knowledge, presentation/answer conventions (those are taught by `reference_sql` examples, not knowledge), or facts that mechanically compose from existing facts. When in doubt, drop it — Step 2 re-checks against `storage-classify`'s "When NOT to pick a store".

**Record with todos when the harvest is large (> 3 candidates):** `todo_write` one todo per candidate (`title` = a short label, `content` = the raw content + where it came from). Skip todos for small sessions.

---

## Step 2 — Classify Each Candidate

`load_skill("storage-classify")` and treat its **Decision Tree** + **Per-Store Reference** + **Disambiguation** as the routing authority. For each candidate, walk the decision tree top-to-bottom (first match wins) and record a verdict:

```
candidate
  → store: semantic_models | metrics | reference_sql | knowledge | skills | memory | AGENTS.md | none
    → ref: the table / SQL / file / fact it refers to
       rationale: one line — the decision-tree branch that matched and why
       prompt-seed: the self-contained seed to hand the downstream mechanism — the ref PLUS the context the generator needs (datasource, the business question, the rule that explains it): e.g. table names + column encodings for gen_semantic_model; the full SQL + the question it answered for gen_sql_summary; the atomic fact + its source for extract-knowledge
```

A single `(question, gold_sql)` pair may route to **both** `reference_sql` (the example) and `knowledge` (the atomic rule mined from *why* the SQL is written that way) — record both.

---

## Turn Boundary — Emit the Summary Manifest, then STOP

Aggregate every verdict, **dedupe** refs that appear more than once, and build a **Summary Manifest** grouped by store, rendered as a Markdown table:

| Item | Store | Refs | Mechanism | Summary |
|------|-------|------|-----------|---------|
| order status encoding | knowledge | `orders.status` enum | `extract-knowledge` (lite) | atomic field-encoding fact |
| top-SKU ranking query | reference_sql | the validated SQL | `task(gen_sql_summary)` | reusable ranking example |
| GMV metric | metrics | GMV on `orders` | `task(gen_metrics)` | built on orders measures |
| prefers DuckDB dialect | memory | — | `add_memory` | session preference (≤ 2000 B) |

**STOP here.** After printing the manifest, **end your turn**. Do **NOT** call any generation `task`, `extract-knowledge`, `gen_skill`, `add_memory`, or write any store yet. Do **NOT** call `ask_user`. State plainly: *"Reply to confirm, or correct / drop any row, and I'll persist the rest."* Wait for the user's next message.

If the harvest produced **nothing** worth persisting, say so plainly (*"Nothing in this session needs persisting — it was all one-shot / inferable."*) and stop. No manifest, no next step.

---

## Step 3 — Route & Generate (next turn, after confirmation; concurrency ≤ 3)

Once the user confirms or corrects the manifest:

1. Treat `storage-classify`'s **Decision Tree** + **Per-Store Reference** + **Context Handoff to Subagents** as the routing authority for every item.
2. **Make every delegated prompt self-contained (see storage-classify's *Context Handoff*).** Each generator runs in a fresh context and sees none of this session — so inline the **datasource (+ dialect)**, the **business intent / question** the session established, and the **rules/encodings surfaced this session** that the artifact must honor. Use the `prompt-seed` you recorded in Step 2 as the carrier, not a bare ref. Route each item with the prescribed mechanism:
   - **Light items** → write directly: `memory` via `add_memory` / `edit_memory` (≤ 2000 bytes); small `AGENTS.md` notes via `edit_file` (never overwrite `## Knowledge` entries owned by `extract-knowledge`).
   - **Heavy items** → delegate (placeholders are the *minimum* each prompt must carry):
     - semantic_models → `task(type="gen_semantic_model", prompt="<datasource> · <table name(s)> · intent · known column encodings / join-key traps>")`
     - metrics → `task(type="gen_metrics", prompt="<datasource> · metric name + definition · the base semantic model / measure it builds on · any mandatory filter>")`
     - reference_sql → `task(type="gen_sql_summary", prompt="<datasource/dialect> · the complete SQL · the business question it answered · why it is written this way>")` — one call per SQL (instruct the generator: when the original question is known, use it verbatim as `search_text`)
     - skills → `task(type="gen_skill", prompt="<skill intent + the concrete steps the session repeated>")`
     - knowledge → run `extract-knowledge` in **lite** mode (do NOT trigger its deep blind-SQL flow); pass the **source and the specific fact to mine**, plus the datasource it applies to
3. **Ordering:** metrics build on semantic models — generate all `semantic_models` items **before** their dependent `metrics` items. When a `reference_sql` item comes from a `(question, gold_sql)` pair, also feed that pair to `extract-knowledge` (lite): one source, two stores (the example teaches answer shape; the mined rule teaches *why*).
4. **Concurrency ≤ 3:** dispatch heavy `task` calls in batches of at most 3, waiting for each batch. Update each item's todo (`in_progress` → `completed` / `failed`) as you go.

End with a short human-readable summary: how many items were persisted, where each was routed (or why it was dropped), and any item still awaiting user confirmation.

---

## Forbidden

- Do not hand-write `semantic_models` / `metrics` / `reference_sql` YAML or vector-store rows — always delegate to the matching `task` subagent.
- Do not run `extract-knowledge` in deep mode — use lite.
- Do not exceed the 2000-byte `memory` cap, and do not write `memory` with any tool other than `add_memory` / `edit_memory`.
- Do not overwrite `## Knowledge` entries in `AGENTS.md` — those are owned by `extract-knowledge`.
- Do not persist one-shot content, generic SQL knowledge, presentation/answer conventions, or anything inferable from `INFORMATION_SCHEMA` / table comments / column names.
- Do not generate the artifact's content yourself — your job is to harvest, classify, and route, not to author.
- Do not run any generation before the user confirms the manifest.
