# Init Command `/init`

## Overview

`/init` is a thin shortcut that asks the active chat agent to follow the
bundled **`init` skill** and perform a **lightweight** project
initialization. It produces a fast, low-cost first pass — an `AGENTS.md`
inventory plus the cheap file-based stores — without paying for the heavy
vector-store generation. The skill walks the agent through:

1. Inferring the project goal and in-scope datasources from `README.md`,
   the directory tree, and `agent.yml` — no upfront questions.
2. Scanning files plus database metadata (directory tree, table
   descriptions, sample rows) and classifying everything into business
   domains.
3. Filing the **cheap file-based stores** directly: atomic, non-inferable
   business facts go to `./knowledge/*.md` via lite `extract-knowledge`,
   and durable cross-session preferences go to `memory`.
4. Writing an **`AGENTS.md` inventory skeleton** to `./AGENTS.md` —
   Architecture, Directory Map, Services, Data Assets, Recommended Tools,
   SQL Conventions, and a Knowledge index — with the vector-index sections
   (`## Semantic Models` / `## Metrics` / `## Reference SQL`) left as
   pointers to `/build-kb`.

There is **no confirmation gate**: knowledge, memory, and `AGENTS.md` are
cheap, reversible markdown writes, so the skill just does them (it only
prompts before overwriting an existing `AGENTS.md` wholesale).

`/init` deliberately **stops short** of the expensive vector-indexed stores
(`semantic_models` / `metrics` / `reference_sql`). To build those —
including indexing a validated-query corpus for few-shot retrieval — run
[`/build-kb`](build_kb_command.md) afterwards.

Because `/init` runs through the standard chat pipeline, you get all the
usual UX automatically — streamed `ActionHistory` events, **Ctrl+O** to
expand trace details, **ESC** to interrupt, and the ability to keep
chatting after generation to refine specific sections.

The skill source lives at `datus/resources/skills/init/SKILL.md` and is
loaded directly from the installed package — no copy to `~/.datus/skills`
is performed. To customize behavior without changing code, drop an
override at `./.datus/skills/init/SKILL.md` (project-local, takes
precedence) or `~/.datus/skills/init/SKILL.md` (user-global). Project
and user overrides shadow the packaged copy by skill name.

---

## Basic Usage

```text
> /init
> /init this is a sales analytics warehouse, focus on the orders domain
```

`/init` accepts an optional free-text description. Anything after the
command is forwarded verbatim to the skill as extra goal/scope hints,
which it folds into the inferred context. With no arguments, the skill
infers everything itself. The datasource scope defaults to whichever one
the REPL is currently pinned to (set at launch with `--datasource` or
switched via `/datasource`). Even when several datasources are configured,
`/init` initializes only that active one — switch with `/datasource <name>`
first to target a different one.

When the agent runs, you'll see the standard chat trace: `load_skill`,
filesystem scans (`glob` / `grep` / `read_file`), database metadata calls
(`list_tables`, `describe_table`), an `extract-knowledge` pass for any
atomic facts, and finally `filesystem_tools.write_file` on `AGENTS.md`.
No `explore` fan-out and no Generation Manifest — those belong to
`/build-kb`.

---

## Prerequisites

- A configured LLM. Run `/model` first if no model is active — `/init`
  needs the agent to drive each step.
- A non-empty `~/.datus/conf/agent.yml`. Populate datasources via
  `/datasource` so they appear in the skill's "Services" table.

If you want to target a different datasource, switch with
`/datasource <name>` first, then run `/init`.

---

## Relationship to `/build-kb`

| | `/init` (lightweight) | [`/build-kb`](build_kb_command.md) (heavy) |
|---|---|---|
| Cost | Low, single confirmation-free pass | High, fans out subagents + confirmation gate |
| Writes | `AGENTS.md` inventory, `./knowledge/*.md`, `memory` | `semantic_models`, `metrics`, `reference_sql` (vector/LanceDB), refreshes the `AGENTS.md` KB index |
| Scope | Whole project | Optional file/table/domain scope |
| Confirmation | None (except AGENTS.md overwrite) | Generation Manifest, then stops for your confirmation |

A typical flow is `/init` first for the map, then `/build-kb` (optionally
scoped) to build the retrieval-backed knowledge base.

---

## Customizing Output

The skill is the single source of truth for what `AGENTS.md` looks like.
To change section structure, table columns, or summary style, edit:

- **Project-local override** (preferred for one-off tweaks):
  `./.datus/skills/init/SKILL.md`
- **User-global override** (applies to every project):
  `~/.datus/skills/init/SKILL.md`
- **Built-in fallback** (always available, ships with the package):
  `datus/resources/skills/init/SKILL.md`

Project-local skills shadow user-global skills, which in turn shadow the
packaged built-in.

---

## Iterative Refinement

Because `/init` is just a chat turn, you can keep chatting afterwards to
refine the result, e.g.:

```text
> /init
… <streaming output, AGENTS.md written> …
> Rewrite the Architecture section to emphasize the data warehouse layer.
```

The agent will edit `AGENTS.md` in place using `filesystem_tools.write_file`.

See also: [`/build-kb`](build_kb_command.md), [`/model`](model_command.md), [`/datasource` (in the slash command reference)](reference.md), [Skills Integration](../integration/skills.md).
