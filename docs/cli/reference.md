# Slash Command Reference

All slash commands available in Datus-CLI, grouped by category.

## Session

| Command | Aliases | Description |
|---------|---------|-------------|
| `/help` | | Display help for all slash commands |
| `/exit` | `/quit` | Exit the CLI |
| `/clear` | | Clear console and chat session |
| `/chat_info` | | Show current chat session information |
| `/compact` | | Compact chat session by summarizing history |
| `/resume` | | List and resume a previous chat session |
| `/rewind` | | Rewind current session to a specific turn |

## Metadata

| Command | Description |
|---------|-------------|
| `/databases` | List all databases |
| `/database` | Switch the current database |
| `/tables` | List all tables |
| `/schemas` | List all schemas or show schema details |
| `/schema` | Switch the current schema |
| `/table_schema` | Show table field details |
| `/indexes` | Show indexes for a table |

## Context

| Command | Description |
|---------|-------------|
| `/catalog` | Display database catalog explorer |
| `/subject` | Display semantic models, metrics, and references |

## Agents

| Command | Description |
|---------|-------------|
| `/agent` | Open the unified agent manager (built-in overrides / default). `/agent <name>` still sets the default directly. |
| `/subagent` | Open the unified agent manager, seeded on the Custom tab |
| `/datasource` | Switch the current datasource |
| `/language` | Pin the response language for every agentic node ([Language Command](other_commands.md#language)) |

## System

| Command | Aliases | Description | Details |
|---------|---------|-------------|---------|
| `/model` | `/models` | Switch LLM provider/model at runtime | [Model Command](other_commands.md#model) |
| `/effort` | | Set reasoning effort level (off/minimal/low/medium/high) | [Effort Command](other_commands.md#effort) |
| `/init` | | Lightweight init: scan + write `AGENTS.md` inventory and file-based knowledge/memory (no vector KB); optional free text adds goal/scope hints | [Init Command](../skills/init.md) |
| `/build-kb` | | Build the vector-indexed KB (semantic models / metrics / reference SQL) and refresh the `AGENTS.md` index; optional free text scopes files/tables/domains | [Build KB Command](../skills/build_kb.md) |
| `/session-summarize` | | Summarize the current session into persistent stores (manifest, then confirm); optional free text adds focus hints | |
| `/memory-organize` | | Audit and reorganize all persistent stores (remediation plan, then confirm); optional free text adds focus hints | |
| `/mcp` | | Manage MCP servers (list/add/remove/check/call/filter) | [MCP Extensions](mcp_extensions.md) |
| `/skill` | | Manage skills and marketplace | [Skill Command](skill_command.md) |
| `/bootstrap-bi` | | Extract BI dashboard assets for sub-agent context | |
| `/services` | | List configured service platforms and their read-only methods | |
| `/permission` | | Switch the active CLI / agent permission profile | |
| `/profile` | | Deprecated alias for `/permission` | |
