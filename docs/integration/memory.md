# Auto Memory

Auto Memory is a persistent memory system for Datus-agent that enables agents to automatically retain valuable information across conversations. It is purely file-based and prompt-driven -- no vector database or embedding is required.

## Overview

When users interact with the agent, the agent can recognize valuable information and persist it to a single Markdown file stored under the workspace. On subsequent conversations, this memory is loaded automatically so the agent can recall prior context.

**Key characteristics:**

- **File-based**: Memory is stored as a single plain Markdown file (`MEMORY.md`)
- **Dedicated tools**: Memory is written exclusively through `add_memory` / `edit_memory` -- generic filesystem tools cannot touch the memory subtree
- **20 KB soft limit, 25 KB hard limit**. Above 20 KB the agent is asked to consolidate. Agent writes that would grow the file past 25 KB are rejected, so the file stays within what the system prompt shows. The exception is a file enlarged past 25 KB by hand: edits that shrink it are still allowed, but they may change content beyond the excerpt the prompt shows.
- **Per-subagent isolation**: Each subagent has its own memory directory
- **Zero configuration**: No setup needed -- eligible agents are automatically enabled

## Memory Directory

Memory is stored under `.datus/memory/` in the workspace, with each agent having its own subdirectory holding a single `MEMORY.md`:

```text
{workspace_root}/
└── .datus/
    └── memory/
        ├── chat/                       # Built-in chat agent
        │   └── MEMORY.md              # Single file, automatically loaded
        └── my_custom_agent/           # Custom subagent
            └── MEMORY.md
```

> The memory directory is created automatically on the agent's first write -- no manual setup needed.

## Which Agents Have Memory

| Agent Type | Memory Enabled |
|-----------|---------------|
| `chat` (built-in main agent) | Yes |
| Custom subagents | Yes |
| Built-in system subagents (`gen_sql`, `gen_visual_report`, etc.) | No |
| `explore` | No |

Only interactive, user-facing agents have memory. Built-in system subagents that perform specific pipeline tasks do not; when launched via `task`, they receive their parent agent's memory inlined as read-only background.

## Single-File Memory

Memory is one flat `MEMORY.md` file per agent:

- **Automatically loaded** when the system prompt is built. Files at or below **25 KB** (25,000 UTF-8 bytes) are included in full.
- Files over **20 KB** (20,000 UTF-8 bytes) carry a warning that asks the agent to consolidate stale or duplicate entries until the file is below 20 KB. This also applies after a manual edit.
- Only a manual edit can make the file larger than 25 KB. The prompt then shows the leading complete lines that fit and marks the excerpt as partial.
- Best for concise, durable facts: user preferences, key project decisions, references to external systems

Keep entries short to reduce the amount of memory injected into every conversation.

## Memory Tools

The agent maintains memory through two dedicated tools:

| Tool | Purpose |
|------|---------|
| `add_memory(content)` | Append one concise fact to memory |
| `edit_memory(old_string, new_string)` | Update an entry, or delete it by passing an empty `new_string` |

Both tools serialize writes to the same file across sessions and processes. An edit whose target text has changed or become ambiguous returns an error and leaves the file intact. Results report `used_bytes` and `remaining_budget`, not the file content. Writes past 20 KB succeed with a `warning` in the tool result, and `remaining_budget` is zero at or above that threshold. A write that would grow the file past 25 KB is rejected and nothing is written; edits that shrink the file are always allowed.

## Usage

### Ask the Agent to Remember

Use natural language:

```text
> Remember that I prefer DuckDB
> Remember the project uses snake_case naming convention
> Remember the default report format is Markdown
```

The agent will save the information via `add_memory`, and it will take effect in the next conversation.

### Ask the Agent to Forget

```text
> Forget my DuckDB preference
> Stop remembering the naming convention
```

The agent will find and remove the corresponding entry with `edit_memory`.

### Correct a Memory

When the agent gives a wrong answer based on memory, simply correct it:

```text
> That's wrong, our project uses PostgreSQL, not DuckDB
```

The agent will immediately update the incorrect entry with `edit_memory`.

### View Current Memory

The memory file is plain Markdown -- you can view or manually edit it:

```bash
cat {workspace_root}/.datus/memory/chat/MEMORY.md
```

Or ask the agent:

```text
> Read your current memory
```

## Agent Memory Behavior

The agent automatically leverages memory in these scenarios:

- **New conversation starts**: Reviews memory for user preferences and prior context
- **Answering project questions**: Checks memory for relevant decisions or conventions
- **User references a past discussion**: Looks up related memory entries
- **Suggesting tools, databases, or workflows**: Respects stated preferences

The agent automatically decides what is worth saving:

| Should Save | Should NOT Save |
|------------|----------------|
| Stable patterns confirmed across interactions | Temporary details of current task |
| Key decisions and project structure | Incomplete, unverified information |
| User preferences and workflow habits | Speculative conclusions from one interaction |
| Solutions to recurring problems | In-progress work state |

## Configuration

Auto Memory requires **no explicit configuration** -- eligible agents are automatically enabled.

The memory directory location follows the resolved workspace root:

| Priority | Source |
|----------|--------|
| 1 | Node-specific `workspace_root` in `agentic_nodes` config |
| 2 | `agent.project_root` in `agent.yml` (defaults to the launch CWD) |

For example, when `agent.project_root` is set to `~/my_project`, the chat agent's memory file is at:

```text
~/my_project/.datus/memory/chat/MEMORY.md
```

## Best Practices

1. **Keep entries concise**. Aim for one short line per fact.
2. **Prune regularly**: Ask the agent to delete or correct outdated or incorrect memories to free space
3. **Use explicit requests**: For important information, explicitly say "remember this" to ensure persistence
4. **Manual editing works too**. Edit the Markdown file between agent writes so an editor cannot overwrite a concurrent update.
