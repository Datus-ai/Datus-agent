# Tool / Plugin Commands `!`

## 1. Overview

The `!` prefix is a power-user escape hatch that runs, directly from the chat REPL, either one of the agent's own **tools** or an installed **plugin's** CLI — without asking the model to do it for you. It is available in **chat mode** only (in SQL/bash mode a leading `!` is part of the statement, e.g. shell history expansion).

```bash
!<tool> [args...]        # run an agent tool directly
!<plugin> <args...>      # run an installed plugin's CLI (datus <plugin> ...)
```

**Tools are matched first.** If the first token names a live tool it runs as a tool; otherwise, if it names an installed + activated plugin, it dispatches to that plugin's CLI; otherwise the input is rejected with a usage hint.

Type `!` on its own to list the available tools and plugins.

## 2. Running a tool

```bash
!list_tables
!search_table user purchase --top_n=5
!describe_table orders
```

- Arguments use a simple grammar: positional values in schema order, plus `--key=value` named overrides (bare `--flag` means `--flag=true`). Lists accept `--items=a,b,c` or `--items=['a','b']`.
- `!<tool> --help` prints the tool's parameter schema (name / type / required / description).
- Every call goes through the **same permission pipeline** an LLM-driven tool call would: read-only tools run without a prompt, while writes (`execute_sql` with INSERT/DDL, `bash`, file writes, …) require confirmation under the active permission profile. Denied calls never execute (and are not sent to the model).
- The call + result render as an **execution turn** (a styled block, like SQL/bash modes) and are fed to the model: the run **enters the conversation context and triggers a reply**, so you can immediately ask follow-up questions about it.

### Autocomplete

- Typing `!` opens a completion menu listing tools (first) then plugins, each tagged in the description column.
- After a tool name, `--` completes the tool's parameter flags plus `--help`.
- Once a tool/plugin name is chosen, a dim `<required> [--optional]` hint is shown after the input, naming the remaining arguments as you type.

## 3. Running a plugin CLI

```bash
!hello sync orders --limit=100
!hello status
```

- `!<plugin> <args...>` runs `datus <plugin> <args...>` as a subprocess. The command is permission-gated like any bash command, and the plugin's own CLI permissions apply inside the child process. Its output is fed to the model as an execution turn (same as `!<tool>`), so it enters the conversation and triggers a reply.
- When a plugin provides command metadata, the `!<plugin>` completer lists its
  commands and hints their arguments. Completion follows nested command groups,
  so typing `!airflow dags ` can offer commands such as `list` and `trigger`.

## 4. Notes

- `!` runs only in chat mode. Use `Tab` to cycle input modes (chat → sql → bash) for SQL / shell execution instead.
- Tool commands run with the same privileges as the Datus-CLI process; the permission profile (`/permission`) governs which tool actions prompt or are blocked.
- Plugin execution is gated by `agent.plugins_enabled` and the project's plugin activation — only installed and active plugins are reachable via `!`.
