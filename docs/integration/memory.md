# Auto Memory

Auto Memory is a persistent memory system for Datus-agent that enables agents to automatically retain valuable information across conversations. It is purely file-based and prompt-driven -- no vector database or embedding is required.

## Overview

When users interact with the agent, the agent can recognize valuable information and persist it to Markdown files stored under the workspace. On subsequent conversations, this memory is automatically loaded, allowing the agent to recall prior context.

**Key characteristics:**

- **File-based**: Memory is stored as plain Markdown files, managed via existing `read_file` / `write_file` / `edit_file` tools
- **Two-layer structure**: A concise `MEMORY.md` main file auto-loaded into context, plus optional topic sub-files read on demand
- **Per-subagent isolation**: Each subagent has its own memory directory
- **Zero configuration**: No setup needed -- eligible agents are automatically enabled

## Memory Directory

Memory files are stored under `.datus/memory/` in the workspace, with each agent having its own subdirectory:

```text
{workspace_root}/
└── .datus/
    └── memory/
        ├── chat/                       # Built-in chat agent
        │   ├── MEMORY.md              # Main file: auto-loaded (≤200 lines)
        │   ├── patterns.md            # Sub-file: read on demand
        │   └── conventions.md
        └── my_custom_agent/           # Custom subagent
            ├── MEMORY.md
            └── domain.md
```

> The memory directory is created automatically on the agent's first write -- no manual setup needed.

## Which Agents Have Memory

| Agent Type | Memory Enabled |
|-----------|---------------|
| `chat` (built-in main agent) | Yes |
| Custom subagents | Yes |
| Built-in system subagents (`gen_sql`, `gen_report`, etc.) | No |
| `explore` | No |

Only interactive, user-facing agents have memory. Built-in system subagents that perform specific pipeline tasks do not.

## Two-Layer Memory

### L1: MEMORY.md (Main File)

- **Automatically loaded** into agent context at the start of every conversation
- Capped at **200 lines** -- content beyond this is truncated
- Best for concise key information and links to L2 files

### L2: Topic Sub-files

- Read by the agent **on demand** via `read_file`
- No line limit -- suitable for detailed content
- Examples: `patterns.md`, `conventions.md`, `domain.md`

**Best stored in L1:**

- User preferences and workflow habits
- Key project structure and file paths
- Frequently referenced conventions
- Links to L2 topic files

**Best stored in L2:**

- Detailed debugging notes
- Complex domain patterns and business rules
- Extended decision records

## Usage

### Ask the Agent to Remember

Use natural language:

```text
> Remember that I prefer DuckDB
> Remember the project uses snake_case naming convention
> Remember the default report format is Markdown
```

The agent will write the information to `MEMORY.md`, and it will take effect in the next conversation.

### Ask the Agent to Forget

```text
> Forget my DuckDB preference
> Stop remembering the naming convention
```

The agent will find and remove the corresponding memory entry.

### Correct a Memory

When the agent gives a wrong answer based on memory, simply correct it:

```text
> That's wrong, our project uses PostgreSQL, not DuckDB
```

The agent will immediately update the incorrect memory entry.

### View Current Memory

Memory files are plain Markdown -- you can view or manually edit them:

```bash
cat {workspace_root}/.datus/memory/chat/MEMORY.md
```

Or ask the agent:

```text
> Read your current memory file
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

The memory directory location follows the `workspace_root` setting:

| Priority | Source |
|----------|--------|
| 1 | Node-specific `workspace_root` in `agentic_nodes` config |
| 2 | `storage.workspace_root` in `agent.yml` |
| 3 | Top-level `workspace_root` in `agent.yml` |
| 4 | Current directory (`.`) |

For example, when `workspace_root` is set to `~/my_project`, the chat agent's memory file is at:

```text
~/my_project/.datus/memory/chat/MEMORY.md
```

## Best Practices

1. **Keep MEMORY.md concise**: Stay within 200 lines -- move detailed content to L2 sub-files
2. **Organize by topic**: Use semantic sub-file names (e.g., `db_conventions.md`) rather than chronological logs
3. **Clean up regularly**: Ask the agent to delete or correct outdated or incorrect memories
4. **Use explicit requests**: For important information, explicitly say "remember this" to ensure persistence
5. **Manual editing works too**: Memory files are plain Markdown -- feel free to view and edit them directly
