# Init Command `/init`

## Overview

`/init` generates an `AGENTS.md` file in the current project directory. The
file describes the project's architecture, directory layout, configured
services, and data assets of the current datasource — content that downstream
`datus` sub-agents and external coding agents (Claude Code, Cursor, …)
consume as context.

`/init` runs entirely inside the REPL: it reuses the active LLM (`/model`)
and the agent configuration that's already loaded in the session, so there's
no separate setup step. It also takes **no arguments** — the datasource used
to enrich the prompt is whichever one the REPL is currently pinned to (set
at launch with `--datasource` or switched via `/datasource`).

---

## Basic Usage

```text
Datus> /init
```

The handler:

1. Reads `agent.yml` (must already exist; configure datasources via
   `/datasource` and an LLM via `/model` first).
2. Scans the current working directory (up to 3 levels deep), skipping
   common noise (`.git`, `node_modules`, `__pycache__`, …).
3. Detects project type from indicator files (`pyproject.toml`,
   `package.json`, `Dockerfile`, `dbt_project.yml`, …).
4. Reads `README.md` (or `README.rst` / `README` / `readme.md`) if present.
5. Probes the **current datasource** (when one is selected) for its table
   list and adds it to the LLM context.
6. Calls the active LLM with the gathered context to generate the
   `AGENTS.md` content. Falls back to a template skeleton if the LLM call
   fails.
7. Writes `AGENTS.md` to the project root.

If `AGENTS.md` already exists, you'll be asked whether to **overwrite** or
**cancel**.

To target a different datasource, switch with `/datasource <name>` first,
then run `/init`.

---

## Generated Sections

The generated file follows this structure:

| Section | Source |
|---------|--------|
| `# <project-name>` | Directory basename + LLM one-liner description |
| `## Architecture` | LLM, anchored on directory tree, project type, and README excerpt |
| `## Directory Map` | LLM, table mapping directories to purpose / entry point / consumer |
| `## Services` | Pulled from `agent.services.datasources` in `agent.yml` |
| `## Data Tables` | Only when the REPL has a datasource selected |
| `## Artifacts` | LLM, e.g., data catalogs, semantic models, SQL files, configs |

---

## Prerequisites

- A configured LLM. Run `/model` first if no model is active.
- A non-empty `~/.datus/conf/agent.yml`. The CLI auto-creates a minimal
  `.datus/config.yml` on first launch; populate datasources with
  `/datasource` before running `/init` if you want them to appear in the
  Services section.

If `agent.yml` is missing, `/init` prints a hint and exits without writing
anything.

---

## Examples

```bash
# Generate AGENTS.md using the active model and currently selected datasource
Datus> /init

# Pick a different datasource first, then re-run
Datus> /datasource duckdb-demo
Datus> /init
```

See also: [`/model`](model_command.md), [`/datasource` (in the slash command reference)](reference.md).
