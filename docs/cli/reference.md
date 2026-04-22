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
| `/agent` | Select or inspect the default agent |
| `/subagent` | Manage sub-agents (list/add/remove/update) |
| `/datasource` | Switch the current datasource |

## System

| Command | Aliases | Description | Details |
|---------|---------|-------------|---------|
| `/model` | `/models` | Switch LLM provider/model at runtime | [Model Command](model_command.md) |
| `/mcp` | | Manage MCP servers (list/add/remove/check/call/filter) | [MCP Extensions](mcp_extensions.md) |
| `/skill` | | Manage skills and marketplace | [Skill Command](skill_command.md) |
| `/bootstrap-bi` | | Extract BI dashboard assets for sub-agent context | |
| `/services` | | List configured service platforms and their read-only methods | |
