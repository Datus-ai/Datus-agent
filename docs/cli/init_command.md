# Init Command `/init`

## Overview

`/init` is a thin shortcut that asks the active chat agent to follow the
bundled **`init` skill** and produce an `AGENTS.md` for the current
project. The skill itself walks the agent through:

1. Asking what the project is about (`ask_user`).
2. Scanning the directory tree (`filesystem_tools`).
3. Reading `README.md` if present.
4. Asking which configured services to include and adding any extras the
   user names.
5. Categorizing tables in each chosen datasource (without enumerating
   every one).
6. Generating the markdown and writing it to `./AGENTS.md`
   (with an overwrite prompt when the file already exists).

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
```

`/init` takes no arguments. The datasource passed to `db_tools.list_tables`
is whichever one the REPL is currently pinned to (set at launch with
`--datasource` or switched via `/datasource`).

When the agent runs, you'll see the standard chat trace: `load_skill`,
followed by interleaved `ask_user` interactions, `filesystem_tools.*`
calls, `db_tools.list_tables` calls, and finally `filesystem_tools.write_file`.

---

## Prerequisites

- A configured LLM. Run `/model` first if no model is active — `/init`
  needs the agent to drive each step.
- A non-empty `~/.datus/conf/agent.yml`. Populate datasources via
  `/datasource` so they appear in the skill's "Services" table.

If you want to target a different datasource, switch with
`/datasource <name>` first, then run `/init`.

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

See also: [`/model`](model_command.md), [`/datasource` (in the slash command reference)](reference.md), [Skills Integration](../integration/skills.md).
