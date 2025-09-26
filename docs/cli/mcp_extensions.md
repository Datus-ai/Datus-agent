# MCP Extensions

## 1. Overview

MCP (Model Context Protocol) is how Datus-CLI connects to external tool servers — enabling your agent to gain new capabilities beyond what's built in.

With `.mcp`, you can:

- Add local or remote MCP servers (stdio, HTTP, SSE)
- List all available servers and their status
- Check a server's connection and available tools
- Call a tool exposed by any MCP server directly inside your CLI

This makes Datus-CLI infinitely extensible: you can plug in SQLite, Snowflake, MetricFlow, Filesystem, or even your own custom MCP server without changing the Datus core.

## 2. Basic Usage

### Add a New MCP Server

```bash
.mcp add <name> <command> [args...]
```

**Examples:**

```bash
# Local stdio-based server
.mcp add --transport stdio sqlite /Users/me/bin/uv -- -m mcp_sqlite_server

# SSE server with authentication header
.mcp add --transport sse api-server https://api.example.com/mcp/sse \
  --header "Authorization: Bearer token" --timeout 30.0

# HTTP stream server
.mcp add --transport http metricflow https://localhost:9000/mcp
```

---

### List Existing Servers

```bash
.mcp list
```

Shows all configured MCP servers and their status:

```
1. duckdb-mftutorial  ✘ failed
2. filesystem         ✔ connected
3. metricflow         ✔ connected
4. snowflake_local    ✔ connected
5. sqlite             ✔ connected
```

You can press Enter to view server details, such as command, args, environment, and available tools.

---

### Check a Specific Server

```bash
.mcp check <mcp_name>
```

Verifies connectivity and prints available tools from that server.

---

### Call a Tool from a Server

```bash
.mcp call <mcp_name>.<tool_name> <args>
```

**Examples:**

```bash
.mcp call sqlite.list_tables
.mcp call metricflow.list_metrics
```

---

## 3. Advanced Features

### Storage & Configuration

- All MCP server definitions are stored in a single JSON file: `~/.datus/conf/.mcp.json`
- Each entry includes command, args, optional env, and headers
- Environment variables can be expanded with `${VAR}` or `${VAR:-default}` inside command, args, env, url, or headers

**Example configuration:**

```json
{
  "mcpServers": {
    "metricflow": {
      "command": "/Users/me/miniconda3/bin/uv",
      "args": [
        "--directory", "/Users/me/src/mcp-metricflow-server",
        "run", "mcp-metricflow-server"
      ],
      "env": {
        "MF_PROJECT_DIR": "/Users/me/.metricflow/sample_models"
      }
    }
  }
}
```

---

### Server Types

- **stdio** — starts a local subprocess and communicates via stdin/stdout
- **http** — connects to a streamable HTTP endpoint
- **sse** — connects to a Server-Sent Events endpoint

Use stdio for local dev tools (fastest), http for persistent servers, and sse for remote cloud services.

### Environment Variable Expansion

MCP configurations support environment variable expansion:

```json
{
  "mcpServers": {
    "snowflake": {
      "command": "${PYTHON_PATH:-python}",
      "args": ["-m", "mcp_snowflake_server"],
      "env": {
        "SNOWFLAKE_ACCOUNT": "${SNOWFLAKE_ACCOUNT}",
        "SNOWFLAKE_USER": "${SNOWFLAKE_USER}",
        "SNOWFLAKE_PASSWORD": "${SNOWFLAKE_PASSWORD:-}"
      }
    }
  }
}
```

### Error Handling and Debugging

- Server connection failures are clearly reported
- Tool call errors include detailed error messages
- Debug mode available for troubleshooting connectivity issues
- Automatic retry logic for transient connection failures

### Security Considerations

- Environment variables containing secrets are handled securely
- Authentication headers are stored encrypted
- Local stdio servers run with user permissions
- Remote connections use secure protocols (HTTPS/WSS)

## Integration Examples

### Database Integration

```bash
# Add SQLite server for local development
.mcp add sqlite uvx mcp-server-sqlite --db-path ./data/sample.db

# Query tables through MCP
.mcp call sqlite.list_tables
.mcp call sqlite.read_query "SELECT * FROM customers LIMIT 5"
```

### MetricFlow Integration

```bash
# Add MetricFlow server for semantic layer
.mcp add metricflow python -m mcp_metricflow_server

# Access business metrics
.mcp call metricflow.list_metrics
.mcp call metricflow.query_metrics --metrics revenue --dimensions customer_segment
```

### Filesystem Integration

```bash
# Add filesystem server for file operations
.mcp add filesystem python -m mcp_filesystem_server --base-path /project/data

# File operations through MCP
.mcp call filesystem.list_directory /reports
.mcp call filesystem.read_file /reports/summary.md
```