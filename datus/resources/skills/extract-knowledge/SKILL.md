---
name: extract-knowledge
description: Mine shortest atomic facts from (question + gold_sql) pairs into ./knowledge/*.md; either by simulating SQL drafting (lite) or by driving the gen_sql subagent in blind iteration (deep)
tags:
  - knowledge
  - sql
  - gold-sql
  - iteration
version: "1.3.0"
user_invocable: true
disable_model_invocation: false
---

# Extract Business Knowledge from Gold SQL Pairs

You receive one or more `(question, gold_sql)` pairs. For each pair you must:

1. Produce a SQL candidate whose result set matches `gold_sql` **exactly** — either by simulating the drafting yourself (lite mode) or by driving a blind `gen_sql` subagent (deep mode). In deep mode, **never expose the gold SQL or any gold result values to the subagent**.
2. Diff the candidate against `gold_sql` to surface the *business knowledge gap* — the rules, joins, filters, granularity, or business definitions that a generic SQL agent would not know.
3. Filter the gap through the "worth-writing" test, then persist each surviving atom as the **shortest possible fact** into `./knowledge/<domain-slug>.md`, and refresh the `## Knowledge` index in `./AGENTS.md`.

The goal is **the shortest description that lets an LLM answer correctly**. Matching the SQL is only the means; reusable atomic facts are the deliverable.

## Critical Rules

- **Never expose `gold_sql` to the subagent.** Not the full SQL, not snippets, not concrete result values, not column-level data. The subagent must stay "blind" — only then do its mistakes reveal what knowledge is missing.
- **Never quote gold's concrete values when talking to the subagent.** Follow-up prompts may only describe *symptoms* and *qualitative directions*.
- Across retries, always reuse the subagent's `session_id` via the `task` tool — that's the only way the subagent remembers prior attempts.

## Input

Resolve the pair source in this priority order (use the first one that works, do not keep looking):

1. **Explicitly supplied in the current user message** — single pair, or multiple (list, CSV file path, YAML/JSON).
2. **Files / paths attached to the current message** — if the user passes a path, `read_file` first, then parse.
3. **Recovered from recent conversation context** — when the user invokes `/extract-knowledge` without explicit pairs:
   - Scan backward from the latest message and locate the most recent user-approved `(question, SQL)` — i.e. a data question the user asked, followed by an assistant SQL the user continued to refine, accepted, or did not reject.
   - Use the user's original question as `question`, and the final adopted SQL as `gold_sql`.
   - **Confirm via `ask_user`** by showing the recovered `question` and a prefix of `gold_sql`. **Do not run silently** — context recovery has ambiguity risk (you may have latched onto an intermediate draft).

If none of the above yields a clear `(question, SQL)`, call `ask_user` to request the pair from the user.

## Mode Selection (lite / deep)

Once the input is resolved, decide which mode to run before entering the workflow:

- **lite (default)** — do not call the `gen_sql` subagent. The main agent itself simulates "given this question + the current datasource schema, how would I draft the SQL," diffs that mental draft against `gold_sql`, and mines the gap directly. **Pros:** fast, zero subagent calls, single-pass. **Cons:** no independent blind-generator validation; depends on the main agent's discipline to "pretend not to know the answer."
- **deep** — full pipeline. Drive the `gen_sql` subagent through multi-round blind iteration (≤5 rounds) until result match, then mine facts from the final diff. **Pros:** independent blind validator, iteration surfaces finer-grained boundary facts. **Cons:** consumes subagent tokens, requires multiple `execute_sql` executions.

**Decision rule:**
1. When the `ask_user` tool is available, call it once and ask the user to choose lite or deep; include the one-line tradeoff. Default suggestion = lite.
2. When `ask_user` is unavailable (no such tool, or running non-interactively), **default to lite** and note in the final output: "defaulted to lite; rerun and choose deep for stricter validation."

Record the chosen `mode` and route to the right workflow branch.

## Workflow (one pair at a time)

### Step 1 — Validate `gold_sql` (both modes)

Run `execute_sql(sql=<gold_sql>)`.
- If it errors: report to the user and **skip this pair**. Do not invent questions to salvage a broken gold.
- Otherwise: cache the result (row count, column names, small preview). This is the factual baseline for later comparison.

### Steps 2–4 (lite mode) — Main agent simulates and diffs

Take this branch only when `mode = lite`; skip the deep branch entirely.

1. **Temporarily ignore `gold_sql`.** Mentally switch to a state where you only know `question` + the current datasource schema. **Do not** let the specific shape of `gold_sql` influence this draft.
2. **Write a draft SQL** (no need to `execute_sql`-execute it; the draft is only a comparison baseline) — this is the version a SQL agent unfamiliar with this project's business conventions would produce given that schema.
3. **Diff the draft against `gold_sql` along these axes:**
   - Table choice, table aliases, missing mapping tables
   - Join type and join keys
   - WHERE filters (especially constant filters, status codes, tenant / platform predicates)
   - Group-by granularity, aggregate functions, deduplication strategy
   - Output columns, column order, column aliases
   - Boundary conditions, strict vs non-strict inequalities
   - Business-term-to-field mapping (e.g. which expression encodes "active" / "retained")
4. **Treat each diff item as a candidate fact** and pass it into Step 5's "worth-writing" filter.

There is no match / mismatch verdict in lite mode — diff items drive extraction directly. Lite mode does not enter the "5-round exhaustion" branch and does not write Open Gaps (no iteration history exists).

### Step 2 (deep mode) — First subagent call

Take this branch only when `mode = deep` and continue with Steps 3 and 4 below.

Invoke:

```
task(
  type="gen_sql",
  prompt=<question>,         # natural-language question only
  description="extract-knowledge: initial attempt for <short topic>"
)
```

The returned envelope contains `result.sql` (or `result.sql_file_path` for long SQL), `result.response`, and `result.session_id`. **Save the `session_id`** — every later retry must reuse it.

### Step 3 (deep mode) — Execute and compare

Run the subagent's SQL via `execute_sql` and compare against the cached gold result:

- Row count match?
- Column names / count match at the semantic level (aliases may differ)?
- After sorting both sides by the same key, do sampled rows match?
- For precise verdicts, use `execute_sql` to run difference probes (two-way `EXCEPT`, key-column `SUM` / `COUNT(DISTINCT ...)` consistency checks, etc.)

Verdict: **match** or **mismatch**.

### Step 4 (deep mode) — Mismatch: diagnose and retry on the same session

Diagnose qualitatively. Common symptoms → likely missing knowledge:

| Symptom | Likely missing knowledge |
|---------|--------------------------|
| Too many rows | Missing filter, missing join, wrong granularity |
| Too few rows | Extra filter, join condition too strict |
| Aggregate value off | Wrong measure column, missing dedup, wrong unit |
| Extra / missing columns | Output shape mismatch, projection rule missing |
| Time bucketing differs | Granularity / business calendar / fiscal-year rule |
| Unexpected NULLs | Wrong join type (LEFT vs INNER), missing COALESCE rule |

When designing the follow-up prompt, obey:

- Describe symptoms and direction only. Never leak gold values. Never restate gold SQL's exact wording to the subagent.
- **The "direction" portion of your prompt should already read like a draft of the future knowledge fact.** I.e. when you write the prompt you have mentally passed it through the "worth-writing" filter: every business rule / mandatory filter / field trap you point the subagent at must be transcribable as a knowledge fact in Step 5 by simply switching from imperative to declarative voice. Benefits: prompts are shorter and more focused; Step 5 becomes near-zero-cost (cherry-pick validated directions from prompt history, strip imperative voice, write to file); prompts and knowledge content stay aligned.
- Avoid shotgun guessing ("maybe the join is wrong? or the aggregation? or null handling?") — such prompts cannot be transcribed into any fact and waste tokens.

Then call:

```
task(
  type="gen_sql",
  session_id=<saved session_id>,    # must be the previous round's id
  prompt=<hint only>,
  description="extract-knowledge: refine #<n>"
)
```

Loop Step 3 → Step 4 for **at most 5 rounds (including the first)**. If still mismatched after round 5: stop retrying, but **do not discard the process information** — promote every confirmed knowledge gap (the rules the subagent finally got right each round, the business rules / field traps / mandatory filters you've already pinpointed in follow-up prompts) into facts via Step 5, then log the remaining un-aligned differences in Open Gaps via Step 7.

## "Worth-Writing" Test (read before Step 5)

For every candidate diff item, ask these 4 questions in order. **If any answer is "yes," drop it.**

1. Without this knowledge, could a SQL agent with the question + schema still get it right?
2. Can this information be inferred directly from `INFORMATION_SCHEMA` / table comments / column names?
3. Is this generic SQL knowledge (not specific to this business / dataset)?
4. **Can this fact be mechanically composed from other facts already in the file?** (If two existing facts combine to produce it, the combination is derivative — do not write it.)

**Record atomic facts only** — anything else counts as derivative and is not written. Common atomic categories worth writing:

- **Field encoding** — bit-position meanings, enum mappings, status codes, special-value semantics
- **Business measure definitions** — the criteria behind business terms
- **Mandatory constant filters** — constant predicates whose omission corrupts results
- **Boundary traps** — strict inequalities, interval endpoint open/closed conventions
- **Implicit table joins** — relationships that must go through a mapping table; direct joins forbidden
- **Same-name field divergence** — the same field name has different semantics in different contexts
- **Required timing / parameter constraints** — business-mandated fixed parameters

**Self-check:** after writing a group of facts, look back — can you drop any single one and still let the LLM answer correctly? If yes, that one was derivative. Drop it.

### Step 5 — Mine atomic facts (both modes)

**Trigger:**
- **lite mode** — enter immediately after Steps 2–4 (lite) produce the draft-vs-`gold_sql` diff.
- **deep mode** — match achieved *or* 5 rounds exhausted. Both cases enter — partial-alignment progress still carries knowledge value.

**lite mode:** every diff item from Steps 2–4 (lite) is a candidate fact.
**deep mode:** diff the subagent's final SQL against `gold_sql` (also against the subagent's *first* attempt — the middle is informative). On failure, additionally walk through every follow-up prompt you sent; the "directions you've already qualitatively named" are confirmed fact sources.

**Systematic corpus scan (both modes, additive) — mine encodings the per-pair diff misses.** The diff only surfaces facts a generic agent got *wrong on this question*; a literal that the draft happened to guess, or that no pair exercised, slips through. So, when you have access to the validated-SQL corpus (not just the single pair), also scan its `WHERE` / `SELECT` clauses **structurally** for non-inferable atoms, independent of any single diff:
- **Literal filter values / coded enumerations** — every constant predicate (`col = '<literal>'`, `col IN (...)`, bit/flag tests) where the literal carries business meaning → a *term ↔ column ↔ value* fact (e.g. a business term maps to a specific column holding a specific code).
- **Business-term → column / expression mappings** — when a question's term resolves to a particular column or computed expression in `SELECT`/`WHERE` → record the mapping.
- **Question-word → column trigger mappings** — when a recurring *interrogative phrasing* (not just a noun) consistently resolves to one column or expression across pairs (e.g. one wording always hits column A while a near-synonym wording hits column B), record the trigger fact: *question says ⟨phrasing⟩ → use ⟨column/expression⟩*. These disambiguate near-duplicate columns that schema inspection cannot.
- **Recurring mandatory constant filters** — a constant predicate that appears across many pairs and whose omission corrupts results → a required-filter fact.
- **Threshold facts always carry the comparison operator's strictness.** A threshold without its boundary semantics is half a fact: record `> 500` vs `>= 500` (and `BETWEEN`'s inclusivity) exactly as the validated SQL has it — strict-vs-inclusive mismatches flip rows at the boundary and are among the most common silent errors.

This pass extracts **structure** (literals in `WHERE`, term→column in `SELECT`), never hardcoded domain vocabulary — it works for any dataset. Every candidate it produces is just another candidate: funnel it through the same "worth-writing" test (so generic/inferable facts are still dropped) and the same Step 6 dedup/conflict gates (so it never duplicates what the diff already wrote).

Pass every candidate through the "worth-writing" test. Only survivors become facts.

Each fact contains only:

- `statement` — one sentence stating the fact itself. **Do not explain "why it matters."** Critical warnings (boundary values, easy-to-miss traps) go in inline parenthetical notes.
- `example` (optional) — **only when the one-sentence statement cannot remove ambiguity** by itself; then attach a minimal SQL snippet.

**Forbidden:**
- Restating the rule in SQL (don't say "use `> 7`, not `>= 7`" and then paste a `WHERE x > 7` snippet)
- Writing "violating this would be wrong" as a standalone field — pure filler
- Writing as a standalone fact anything that mechanically composes from other facts

### Knowledge File Layout

```
business domain   →   topic   →   facts
   (one .md)         (## heading)   (list items / table rows / paragraphs / optional ### subblock)
```

- **Business domain = one file** — a slice of business broad enough that its rules co-evolve. Path: `./knowledge/<domain-slug>.md`. Prefer fewer, wider domains over many narrow ones.
- **Topic (`##`) = a group of facts sharing context** — typically the constraints of one table, a set of measure definitions, or a related mapping group.
- **Fact representation, in priority order:**
  1. **Bulleted list item** — multiple independent facts under the same topic (most common)
  2. **Table row** — several parallel field / enum / branch mappings
  3. **Paragraph** — a conceptual fact a sentence or two can explain
  4. **`### <one-sentence rule>` subblock** — only when a SQL example must be attached

**Hard constraints:**
- A topic block must hold ≥2 independent facts (otherwise merge into a wider topic)
- Heading depth never exceeds `###`
- Do not write `Derived from` / `Why it matters` / `When it applies` label fields
- The file must contain no question text (trace via git history instead)
- **Strong preference: reuse first, create new last.** Before persisting anything, list `./knowledge/` and read each file's Domain intro. Prefer adding a new topic or fact under an existing domain over opening a new file.

**Template for a new business-domain file:**

```markdown
# <Domain Title>

> **Domain:** <one-sentence scope statement — what this file covers, what it doesn't>.

## <Topic Title>

<optional 1–2 sentence intro describing what this topic covers>

- fact 1
- fact 2 (note ...)

| Field / Branch | Value / Verdict |
|---------------|-----------------|
| ... | ... |

### <rule that needs a SQL example>

<one-sentence statement, may carry an inline warning>

\`\`\`sql
<minimal snippet that resolves ambiguity>
\`\`\`
```

### Step 6 — Persist (fact-level dedup / conflict gates)

For every classified fact, walk the gates **in order**. Do not skip steps.

#### 6.1 Resolve target domain file

- If `./knowledge/` does not yet exist, just `write_file` the first file (no `ask_user` needed for an empty-directory case).
- If the fact does not belong to any existing domain, **before** creating a new file you must `ask_user`: list existing domains, propose a new slug, let the user choose reuse / new / custom.
- Never silently open a new file when a reasonable reuse path exists.

#### 6.2 Detect duplicates and conflicts inside the file

`read_file` the target. Extract existing facts per topic (list items + table rows + paragraph facts + ### subblock headings). Compare each candidate against existing facts **semantically** (not by string):

| Outcome | Definition | Action |
|---------|-----------|--------|
| **Duplicate** | Same fact, same scope, same direction. | **Silent skip** (idempotent). Record in final report. |
| **Refinement** | Existing fact is a strict subset of the new one (lower precision, missing a condition, narrower scope). | **`ask_user`**: replace / merge / keep both with a scope qualifier. Default suggestion = merge. |
| **Conflict** | Same scope, opposite direction / mutually contradictory. | **`ask_user` is mandatory**. Show both side-by-side. Options: keep old / replace with new / keep both with an explicit conditional gate (you must spell out the condition). **Never resolve a conflict silently.** |
| **Complementary** | Same topic, different facet. | **Append** a same-form fact under the same topic (list item / table row / paragraph). |
| **Derivable** | The candidate mechanically composes from existing facts. | **Drop, do not write**. The "worth-writing" test (question 4) is supposed to catch this earlier — this is a safety net. |
| **New topic / new domain** | No relevant existing topic. | Create the heading via 6.3. |

#### 6.3 Write or edit the file

- Prefer `edit_file` for minimum diff.
- When a new fact changes the appropriate representation of an existing topic, **whole-block rewriting that topic is allowed** (only touch that block); avoid letting persistent append accumulate redundancy.
- Heading conventions:
  - New domain → `write_file` the Domain intro followed by the first topic heading
  - New top-level topic → insert a `## <topic title>` block
  - New fact into existing topic → append as list item / table row / paragraph; only add a `### <one-sentence rule>` subblock when a SQL example is needed

**Do not** insert `---` separators between facts — markdown headings / lists already separate them, and `---` makes `edit_file` insertion points ambiguous.

### Step 7 — Update the AGENTS.md index

Maintain the `## Knowledge` section of `./AGENTS.md`, so that later main agents handling related business-domain tasks can enter from AGENTS.md, see what `knowledge/` holds, and `read_file` the relevant domain file as context.

**AGENTS.md handling order:**

1. **`./AGENTS.md` missing** → `write_file` a minimal skeleton (just `# <project directory name>` + the `## Knowledge` section; leave the rest for `/init` to fill in later). Do not wait for the user to run `/init` first.
2. **AGENTS.md exists but lacks the `## Knowledge` section** → insert after `## Artifacts` (or at end of file if Artifacts is also absent).
3. **Rescan `./knowledge/`** and **rewrite the entire `## Knowledge` section** sorted alphabetically by domain title — guarantees the index stays consistent across runs.

**Fixed structure of the `## Knowledge` section:**

```markdown
## Knowledge

`./knowledge/` holds this project's atomic business facts, organized per business domain (maintained by `/extract-knowledge`).
Before handling a task that touches a domain, `read_file` the corresponding file for context.

- [<Domain Title A>](knowledge/<domain-slug-a>.md) — <one-sentence scope>
- [<Domain Title B>](knowledge/<domain-slug-b>.md) — <one-sentence scope>

### Open Gaps

- <question summary> — <remaining un-aligned difference>
```

- One index line = one business-domain file (not per topic, not per fact). Scope sentence comes from that file's `> Domain:` intro.
- When `./knowledge/` is empty, omit the index list and `### Open Gaps`; keep only the section intro paragraph.
- `### Open Gaps` omits entirely when there are no entries; when present, group by domain.

**5-round mismatch handling (deep mode only):** keep the facts Step 5–6 already wrote in their domain files, then append a line `- <question summary> — <remaining un-aligned difference>` under `### Open Gaps`, and continue with the next pair. Lite mode has no iteration history and does not write Open Gaps.

## Final Output

Return a single human-readable summary covering: the mode used (lite / deep); how many pairs were processed; how many matched / failed (deep) or were aligned (lite); which `knowledge/*.md` files were created / edited; how many facts were added; any conflicts still needing attention. When lite was used because `ask_user` was unavailable, add one line: "defaulted to lite; rerun and choose deep for stricter validation." **Do not return a JSON envelope.**

## Tools You'll Use

- `execute_sql(sql=...)` — execute gold SQL and difference probes (deep mode additionally executes subagent SQL). In lite mode it is only used for Step 1's gold validation.
- `task(type="gen_sql", prompt=..., session_id=...)` — **deep mode only**; delegate SQL generation, reuse session across retries.
- `read_file`, `write_file`, `edit_file` — manage `./knowledge/*.md` and `./AGENTS.md`. **Always** `read_file` the target domain file before edits (6.2 depends on it).
- `ask_user` — required for: ambiguous input parsing; confirming a context-recovered pair; mode selection (lite / deep); new-domain confirmation (6.1); refinement decisions (6.2); **every fact conflict** (6.2). Phrase questions with numbered options.

## Forbidden

- In lite mode, do not call `task(type="gen_sql", ...)` — the mode selection already declared no subagent.
- In deep mode, do not call `task(type="gen_sql", ...)` before confirming `gold_sql` actually runs.
- In deep mode, do not open a new `gen_sql` session for retries — always pass the previous round's `session_id`.
- Do not write SQL on behalf of the user's question — that's the subagent's job (deep mode). Your role is orchestrator and knowledge curator.
- Do not put gold SQL into `./knowledge/*.md`. Knowledge is *atomic facts mined from the gap*, not the answer itself.
- Do not write derivative facts — anything that mechanically composes from other facts is dropped.
- Do not write `Derived from` / `Why it matters` / `When it applies` label fields; do not include any question text in the files.
- Do not attach a SQL example to every fact — only when the one-sentence statement cannot remove ambiguity by itself.
- When an existing domain can plausibly cover the topic, do not create a new file. Reuse + slightly extending the intro is almost always right.
- Do not resolve fact conflicts silently. Conflicts **must** go through `ask_user` — the user decides replace / coexist / scope-gate.
- Do not insert `---` separators between facts.
- Do not give rules generic names (`### Rule 1`, `### Note`). The one-sentence rule itself is the heading.
- Topic heading depth must not exceed `####`. Deeper means the domain was sliced wrong — fold the levels or split the file.
- A single-fact topic must not be its own `##` — merge it into a wider topic.
