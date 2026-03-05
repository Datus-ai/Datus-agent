# Datus MCP Server

Expose Datus's database and context search tools via the **Model Context Protocol (MCP)**, enabling integration with
Claude Desktop, Claude Code, and other MCP-compatible clients.

**Server Modes:**

- **Static Mode**: Single namespace, suitable for Claude Desktop, CLI tools, or single-tenant HTTP/SSE server
- **Dynamic Mode**: Multi-namespace HTTP/SSE server, supports all namespaces via URL path

**Supported Transport Modes:**

- `http`: Streamable HTTP (bidirectional, default)
- `sse`: Server-Sent Events over HTTP (for web clients)
- `stdio`: Standard input/output (for Claude Desktop and CLI tools)

## Quick Start

- Install Datus:

```bash
pip install datus-agent
```

- Start the MCP server:

```bash
# Static Mode: Single namespace
uvx --from datus-agent datus-mcp --namespace <your namespace>
uvx --from datus-agent datus-mcp --namespace <your namespace> --transport http --host 127.0.0.1 --port 8000
# Dynamic Mode: Multi-namespace HTTP/SSE server
uvx --from datus-agent datus-mcp --dynamic --transport http --host 127.0.0.1 --port 8000
uvx --from datus-agent datus-mcp --dynamic --transport sse --host 127.0.0.1 --port 8000

# Or run directly
datus-mcp --namespace <your namespace>
# Dynamic Mode: Multi-namespace HTTP/SSE server
datus-mcp --dynamic --transport http --host 127.0.0.1 --port 8000
datus-mcp --dynamic --transport sse --host 127.0.0.1 --port 8000
```

## Client Integration

### Claude Code

Start the Datus MCP server, then add it to Claude Code:

```bash
# Start server (SSE mode)
datus-mcp --dynamic --transport sse --port 8000

# Add to Claude Code
claude mcp add --transport sse datus http://127.0.0.1:8000/sse/<your namespace>
```

### Claude Desktop

Claude Desktop requires [mcp-remote](https://www.npmjs.com/package/mcp-remote) to connect to a remote MCP server.

Start the Datus MCP server, then configure Claude Desktop using the following script:

```bash
# npx may not be found directly by Claude Desktop, so we resolve the full paths
NODE_BIN_DIR=$(dirname $(which node))
NPX_PATH=$(which npx)

cat > ~/Library/Application\ Support/Claude/claude_desktop_config.json << EOF
{
  "mcpServers": {
    "datus-agent": {
      "command": "$NPX_PATH",
      "args": [
        "mcp-remote@latest",
        "http://127.0.0.1:8000/sse/<your namespace>",
        "--transport",
        "sse-only"
      ],
      "env": {
        "PATH": "$NODE_BIN_DIR:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin"
      }
    }
  }
}
EOF
```

Or manually add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "datus-agent": {
      "command": "npx",
      "args": [
        "mcp-remote@latest",
        "http://127.0.0.1:8000/sse/<your namespace>",
        "--transport",
        "sse-only"
      ]
    }
  }
}
```

!!! tip
    Claude Desktop also supports stdio transport directly. See the [Other MCP Clients](#other-mcp-clients) section for stdio configuration.

### Other MCP Clients

For MCP clients that support stdio transport:

- **Using uvx:**

```json
{
  "mcpServers": {
    "datus": {
      "command": "uvx",
      "args": [
        "--from",
        "datus-agent",
        "datus-mcp",
        "--namespace",
        "<your namespace>",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

- **Using python directly:**

```json
{
  "mcpServers": {
    "datus": {
      "command": "python",
      "args": [
        "-m",
        "datus.mcp_server",
        "--namespace",
        "<your namespace>",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

For MCP clients that support HTTP/SSE transport:

- **SSE:**
```json
{
  "mcpServers": {
    "DatusServer": {
      "url": "http://127.0.0.1:8000/sse/<your namespace>",
      "transport": "sse"
    }
  }
}
```

- **Streamable HTTP:**
```json
{
  "mcpServers": {
    "DatusServer": {
      "url": "http://127.0.0.1:8000/mcp/<your namespace>",
      "transport": "http"
    }
  }
}
```

## HTTP Server Mode

**Static Mode (Single Namespace):**

```bash
# Streamable HTTP (default, bidirectional)
datus-mcp --namespace <your namespace> --transport http --host 0.0.0.0 --port 8000

# SSE mode (for web clients)
datus-mcp --namespace <your namespace> --transport sse --port 8000
```

Connect to:

- Streamable HTTP: `http://localhost:8000/mcp`
- SSE: `http://localhost:8000/sse`

**Dynamic Mode (Multi-Namespace):**

Run a single server that supports all configured namespaces via URL path:

```bash
# Start dynamic server with sse mode
datus-mcp --dynamic --host 0.0.0.0 --port 8000 --transport sse

# Start dynamic server with streamable HTTP mode
datus-mcp --dynamic --host 0.0.0.0 --port 8000 --transport http
```

Connect to specific namespace:

- HTTP: `http://localhost:8000/mcp/{namespace}`
- SSE: `http://localhost:8000/sse/{namespace}`
- With subagent: `http://localhost:8000/mcp/{namespace}?subagent={subagent_name}`

Example:

- Streamable HTTP, namespace `bird_sqlite`: `http://localhost:8000/mcp/bird_sqlite`
- SSE, namespace `bird_sqlite`: `http://localhost:8000/sse/bird_sqlite`
- Streamable HTTP, namespace `superset`, subagent `sales_dashboard`: `http://localhost:8000/mcp/superset?subagent=sales_dashboard`

Info endpoints:

- `http://localhost:8000/` - Server info and available namespaces
- `http://localhost:8000/health` - Health check

## Available Tools

The MCP server exposes the following tools:

| Category           | Tools                                                                                                                                                             |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Database**       | `list_databases`, `list_schemas`, `list_tables`, `search_table`, `describe_table`, `get_table_ddl`, `read_query`                                                  |
| **Context Search** | `list_subject_tree`, `search_metrics`, `get_metrics`, `search_reference_sql`, `get_reference_sql`, `search_semantic_objects`, `search_knowledge`, `get_knowledge` |

## Command Line Options

```bash
datus-mcp --help

Mode Selection (mutually exclusive, one required):
  --dynamic            Run in dynamic mode: support all namespaces via /mcp/{namespace} URL
  --namespace, -n      Run in static mode with specified namespace

Static Mode Options:
  --sub-agent, -s      Sub-agent name for scoped context
  --database, -d       Database name override
  --transport, -t      Transport type: http (default), sse, stdio

Dynamic Mode Options:
  --transport, -t      Transport type: http (default), sse:
                       http: via /mcp/{namespace} URL
                       sse: via /sse/{namespace} URL

Common Options:
  --config, -c         Path to agent configuration file
  --host               Host to bind for HTTP transports (default: 0.0.0.0)
  --port, -p           Port to bind for HTTP transports (default: 8000)
  --debug              Enable debug logging
```
