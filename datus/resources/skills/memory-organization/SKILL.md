---
name: memory-organization
description: Audit and reorganize every persistent store — semantic_models, metrics, reference_sql, knowledge, memory, AGENTS.md, skills — verifying each item sits in the correct store per storage-classify, and surfacing duplicates, misclassifications, conflicts, and stale/erroneous entries. Produce a Remediation Plan, STOP for confirmation, then execute. Use ask_user only for genuine decisions during analysis. If nothing needs fixing, report it and stop.
tags:
  - memory
  - organization
  - persistence
  - classify
  - audit
version: "1.1.0"
user_invocable: true
---

# Memory Organization

You give the project's persistent stores a health check: confirm every item is classified into the correct store, and surface duplicates, misclassifications, conflicts, and stale/erroneous entries so the structure stays clean. You are an auditor and curator, not a generator.

You run in the main agent context, so you may call `task`, `load_skill`, `todo_write`/`todo_list`/`todo_update`, `ask_user`, `add_memory`/`edit_memory`, the filesystem tools (`glob`, `grep`, `read_file`, `write_file`, `edit_file`), and the search tools (`search_semantic_model`, `search_metrics`, `search_reference_sql`).

**Routing authority is `storage-classify`.** This skill decides *what to inspect and how to detect problems*; the `storage-classify` skill owns *which store an item belongs to and how it is rewritten*. Do NOT re-invent classification rules here — load and follow `storage-classify` in Step 2.

**Two-phase contract (important).** Reorganizing rewrites and deletes real files. So there is a hard **turn boundary** after analysis: you produce a Remediation Plan and then **end your turn**. Do NOT call `ask_user` for that confirmation gate — just present the plan and stop. The user confirms or corrects it in their next message; only then do you run Step 3. (You *may* call `ask_user` earlier, during analysis, but only for a genuine human judgement — see Step 2.)

---

## Step 1 — Inventory Every Store (read-only)

Enumerate the current contents of each store. Do not modify anything in this step.

- **semantic_models** — `glob ./subject/semantic_models/**/*.yml`; cross-check with `search_semantic_model`.
- **reference_sql** — `glob ./subject/sql_summaries/*.yaml`; cross-check with `search_reference_sql`.
- **metrics** — `search_metrics` (LanceDB-backed; there are no flat files to glob).
- **knowledge** — `glob ./knowledge/*.md`, and read the `## Knowledge` index in `AGENTS.md`.
- **memory** — read `./.datus/memory/{node}/MEMORY.md` for each node present (`chat` and any custom subagents).
- **AGENTS.md** — read the top-level file (its sections and index counts).
- **skills** — list `./.datus/skills/`, `~/.datus/skills/`, and the bundled `datus/resources/skills/` (note first-wins overrides).

**Record with todos when the inventory is large:** `todo_write` one todo per store so progress through analysis is trackable.

---

## Step 2 — Analyze Against the Classification Rules

`load_skill("storage-classify")` and treat its **Decision Tree** + **Per-Store Reference** + **Disambiguation** + **When NOT to pick a store** as the rule set. Scan within and across stores for:

- **Misclassification** — an item in the wrong store per the decision tree. E.g. a single metric stored as a `semantic_model`; structural table/column definitions stored as `metrics`; a presentation/answer convention written as `knowledge`; a team-level long-lived business fact (or anything > 2000 bytes) parked in `memory`; a fine-grained atomic fact inlined into `AGENTS.md` instead of `knowledge`.
- **Duplicates** — the same atomic fact appearing twice in `knowledge`; the same SQL indexed twice in `reference_sql`; the same fact living in both `memory` and `knowledge`.
- **Conflicts** — two entries asserting contradictory facts (e.g. two `knowledge` atoms disagreeing on a status-code encoding).
- **Errors / stale** — `knowledge` that contradicts the current schema; `## Knowledge` index links pointing to missing `./knowledge/*.md` files; `AGENTS.md` index counts that no longer match the stores; `memory` over the 2000-byte cap.
- **Structural issues** — `AGENTS.md` sections out of canonical order, or inlining store *contents* where it should only carry an index (count + retrieval tool, or file links for knowledge).

Use `ask_user` **only** when the resolution genuinely needs a human judgement that the data cannot settle — e.g. which of two conflicting facts is correct, or whether to merge two near-duplicate entries or keep both. Do not use `ask_user` for the plan-confirmation gate (that is the turn boundary below).

---

## Turn Boundary — Emit the Remediation Plan (or report clean), then STOP

**If issues were found:** render a **Remediation Plan** as a Markdown table, then **end your turn**:

| Issue | Store(s) | Type | Proposed Action | Mechanism |
|-------|----------|------|-----------------|-----------|
| single metric stored as semantic_model | semantic_models → metrics | misclassification | move to metrics | `task(gen_metrics)` + remove stale model |
| duplicate status-code fact | knowledge | duplicate | merge into one atom | `edit_file` |
| memory holds team-level fact | memory → knowledge | misclassification | re-route to knowledge | `extract-knowledge` (lite) + `edit_memory` |
| `## Knowledge` link to deleted file | AGENTS.md | stale | drop the dead index line | `edit_file` |

Do **NOT** execute any fix, run any generation `task`, or call `ask_user` at this gate. State plainly: *"Reply to confirm, or correct / drop any row, and I'll apply the remediation."* Wait for the user's next message.

**If nothing needs fixing:** report it plainly — *"All stores are correctly classified and consistent: no duplicates, conflicts, misclassifications, or stale entries. Nothing to reorganize."* — and **stop. No plan, no next step.**

---

## Step 3 — Execute Remediation (next turn, after confirmation; concurrency ≤ 3)

Once the user confirms or corrects the plan, apply each action with the store's prescribed mechanism (per `storage-classify`):

**Make every re-route prompt self-contained (see storage-classify's *Context Handoff*).** A regenerating subagent runs in a fresh context and cannot see the stale entry you are replacing — so lift the content out of the old store during Step 1's inventory and inline it: the **datasource (+ dialect)**, the misplaced item's **actual content** (the SQL, the metric definition, the fact), the **business intent**, and the **rule/encoding** it must honor. Re-routing a metric with only its name produces a different metric, not a faithful move.

- **Misclassified items** → re-route through the correct owner: regenerate via the matching `task` subagent (`gen_semantic_model` / `gen_metrics` / `gen_sql_summary` / `gen_skill`) or `extract-knowledge` (lite) — passing the lifted content + context above — then remove the stale copy from the wrong store.
- **Duplicates** → consolidate via `edit_file` (knowledge / reference_sql YAML index) or `edit_memory` (memory).
- **Conflicts** → keep the resolved entry (per the Step 2 `ask_user` decision) and remove the other.
- **AGENTS.md index / structure** → fix with a scoped `edit_file`; never rewrite the whole file when a scoped edit suffices, and never touch `## Knowledge` entries owned by `extract-knowledge` beyond removing provably dead links.
- **Stale / erroneous entries** → remove them.

For any destructive overwrite or deletion that the plan did not already spell out, go through `ask_user` first. Never hand-write `semantic_models` / `metrics` / `reference_sql` YAML — always go through the matching subagent. **Concurrency ≤ 3** for heavy `task` calls; update each todo (`in_progress` → `completed` / `failed`) as you go.

End with a short human-readable summary: how many issues were remediated, what changed in each store, and anything still awaiting user confirmation.

---

## Forbidden

- Do not hand-write `semantic_models` / `metrics` / `reference_sql` YAML or vector-store rows — always delegate to the matching `task` subagent.
- Do not run `extract-knowledge` in deep mode — use lite.
- Do not exceed the 2000-byte `memory` cap, and do not write `memory` with any tool other than `add_memory` / `edit_memory`.
- Do not overwrite `## Knowledge` entries in `AGENTS.md` beyond removing provably dead index links — those are owned by `extract-knowledge`.
- Do not execute any remediation before the user confirms the plan.
- Do not delete or overwrite content destructively without an explicit plan row or an `ask_user` confirmation.
- Do not generate replacement content yourself — your job is to audit, classify, and re-route, not to author.
