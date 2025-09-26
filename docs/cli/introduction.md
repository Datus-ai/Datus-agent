# Introduction

Datus-CLI is an AI-native, terminal-first SQL client built for data engineers. It combines an agentic chat experience, precise context management, and reproducible execution into one lightweight workflow, enabling you to reason about data, generate and refine SQL, and run tasks—all without leaving your terminal.

## Core Magic Commands

At its core are three "magic commands" that keep things simple and fast:

- **[`/` Chat command](chat_command.md)** - For multi-turn conversation
- **[`@` Context command](context_command.md)** - For on-demand recall of metadata/metrics/SQL history
- **[`!` Execution command](execution_command.md)** - For deterministic, scriptable actions

## Additional Features

In addition to the three magic commands, you can also:

- [Run SQL directly](sql_execution.md) like a traditional SQL client
- Use [system commands](chat_command.md#session-commands) such as `.compact` and `.clear` to manage or reset sessions
- [Explore metadata](sql_execution.md#explore-metadata) with `.tables` and `.databases`
- Use [`.mcp`](mcp_extensions.md) to add, remove, or test MCP servers to extend functionality

## What is Vibe SQL?

Vibe SQL is the Datus philosophy that you don't need to write every SQL line yourself. Instead of manually crafting queries, you describe your intent and data "vibe" in natural language, and Datus:

- Generates working SQL behind the scenes
- Builds tables, metrics, and views for you automatically
- Can wire those results into a ready-made chatbot or analytics interface for your stakeholders

## What Makes Datus-CLI Different

- **Context, not control** - Datus treats the workflow context as short-term memory (task description, executed SQL + results/reflections, selected tables/metrics) and builds long-term memory from user preferences and the surrounding data environment (lineage, metrics, docs)

- **Explainable, reliable flows** - Workflows can pause, edit, and resume at any node, introducing human-in-the-loop (HITL) when confidence is low—so you get both exploration and control

- **Two-way context engineering** - Datus builds durable memory from Tree + Vector recall across catalogs, metrics, success-story SQL, and external rules—so useful context is always one "@" away

## Why a CLI?

A CLI provides tighter control over context (not just code), making it easier to keep diffs small and reproducible while collaborating with an AI agent. This is especially useful when you want reliable changes instead of opaque, large edits typical of GUI tools.

## Getting Started

You first need to install and initialize Datus-CLI through the installation process, and then you can start Datus-CLI by running:

```bash
datus-cli --namespace duckdb-demo
```