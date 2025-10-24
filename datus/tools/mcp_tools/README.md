# MCP Tools - Model Context Protocol Server Management

This package provides comprehensive functionality for managing MCP servers including stdio, sse, and http communication types.

## Features

- **Multi-Protocol Support**: Support for stdio, sse (Server-Sent Events), and http communication protocols
- **Config Management**: Full CRUD operations for MCP server configs
- **Tool Filtering**: Fine-grained control over which tools are accessible from MCP servers
- **Config-Only Management**: Focus on configuration without server lifecycle management
- **Config Validation**: Pydantic-based configuration validation
- **Env Variable Support**: Support for `${VAR}` and `${VAR:-default}` variable expansion
- **Type Safety**: Full Pydantic validation for all configs
- **Standardized Tool Interface**: Follows the BaseTool pattern for integration

## Quick Start

```python
from datus.tools.mcp_tools import MCPTool

# Initialize the MCP tool
mcp_tool = MCPTool()

# Add a stdio MCP server
result = mcp_tool.add_server(
    name="my-filesystem-server",
    type="stdio",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    env={"NODE_OPTIONS": "--no-warnings"}
)

# List all servers
result = mcp_tool.list_servers()
servers = result.result['servers']

# Get server config
result = mcp_tool.get_server("my-filesystem-server")

# Check server connectivity
result = mcp_tool.check_connectivity("my-filesystem-server")

# Add an SSE server with env variables
result = mcp_tool.add_server(
    name="api-server",
    type="sse",
    url="${API_BASE_URL:-https://api.example.com}/mcp",
    headers={"Authorization": "Bearer ${API_KEY}"}
)

# List available tools from a server
result = mcp_tool.list_tools("my-filesystem-server")
if result.success:
    tools = result.result.get("tools", [])
    for tool in tools:
        print(f"Tool: {tool['name']} - {tool['description']}")

# Call a tool
result = mcp_tool.call_tool(
    "my-filesystem-server",
    "list_directory",
    {"path": "/tmp"}
)

# Set up tool filtering (allow only specific tools)
result = mcp_tool.set_tool_filter(
    "my-filesystem-server",
    allowed_tools=["read_file", "write_file", "list_directory"],
    blocked_tools=None,
    enabled=True
)

# List tools with filtering applied
result = mcp_tool.list_filtered_tools("my-filesystem-server")
```

## Server Types

### 1. Stdio (Standard Input/Output)
For local command execution with process communication via stdin/stdout.

```python
mcp_tool.add_server(
    name="stdio-server",
    type="stdio",
    command="python",
    args=["-m", "my_mcp_server"],
    env={"DEBUG": "1"}
)
```

### 2. SSE (Server-Sent Events)
For HTTP-based event streaming communication.

```python
mcp_tool.add_server(
    name="sse-server",
    type="sse",
    url="https://api.example.com/mcp/sse",
    headers={"Authorization": "Bearer token"},
    timeout=30.0
)
```

### 3. HTTP
For HTTP-based communication.

```python
mcp_tool.add_server(
    name="http-server",
    type="http",
    url="https://localhost:9000/mcp",
    headers={"Content-Type": "application/json"},
    timeout=120.0
)
```

## Config File

The MCP tools use a JSON config file to persist server configs. The configuration path is fixed at `{agent.home}/conf/.mcp.json` (default: `~/.datus/conf/.mcp.json`).

You can customize the root directory by configuring `agent.home` in `agent.yml`.

```json
{
  "mcpServers": {
    "metricflow": {
      "type": "stdio",
      "command": "/usr/local/bin/uv",
      "args": [
        "--directory", "/path/to/mcp-metricflow-server",
        "run", "mcp-metricflow-server"
      ],
      "env": {
        "MF_PROJECT_DIR": "/path/to/models"
      },
      "tool_filter": {
        "allowed_tool_names": ["get_dimensions", "get_metrics", "query_metrics"],
        "enabled": true
      }
    },
    "api_server": {
      "type": "sse",
      "url": "${API_BASE_URL:-https://api.example.com}/mcp",
      "headers": {
        "Authorization": "Bearer ${API_KEY}"
      },
      "tool_filter": {
        "blocked_tool_names": ["delete_resource", "admin_action"],
        "enabled": true
      }
    },
    "http_server": {
      "type": "http",
      "url": "${HTTP_URL:-https://localhost:8001/mcp}",
      "headers": {
        "Content-Type": "application/json"
      },
      "timeout": 60.0
    }
  }
}
```

## Available Operations

### Config Management
- `add_server()` - Add new server config
- `remove_server()` - Remove server config
- `list_servers()` - List servers with filtering
- `get_server()` - Get specific server config
- `check_connectivity()` - Check server connectivity

### Server Operations
- `list_tools()` - List available tools from a server (with optional filtering)
- `list_filtered_tools()` - List tools with filtering applied
- `call_tool()` - Call a specific tool on a server (respects tool filters)

### Tool Filtering
- `set_tool_filter()` - Configure tool filtering for a server
- `get_tool_filter()` - Retrieve current tool filter configuration
- `remove_tool_filter()` - Remove/disable tool filtering for a server

## Tool Filtering

Tool filtering provides fine-grained control over which tools are accessible from MCP servers, following the OpenAI MCP specification. This is useful for security, limiting functionality, or creating specialized tool subsets.

### Filter Types

#### 1. Allowlist Filtering (Whitelist)
Only specified tools are permitted. All other tools are blocked.

```python
# Allow only specific tools
mcp_tool.set_tool_filter(
    "my-server",
    allowed_tools=["read_file", "write_file", "list_directory"],
    blocked_tools=None,  # Not used with allowlist
    enabled=True
)
```

#### 2. Blocklist Filtering (Blacklist)
All tools are permitted except those specifically blocked.

```python
# Block dangerous tools
mcp_tool.set_tool_filter(
    "my-server",
    allowed_tools=None,  # Not used with blocklist
    blocked_tools=["delete_file", "execute_command", "system_shutdown"],
    enabled=True
)
```

#### 3. Combined Filtering
When both allowlist and blocklist are specified, **allowlist takes precedence**.

```python
# Allowlist wins - only read_file and write_file are allowed
mcp_tool.set_tool_filter(
    "my-server",
    allowed_tools=["read_file", "write_file", "delete_file"],
    blocked_tools=["delete_file"],  # This is ignored due to allowlist precedence
    enabled=True
)
```

### Filter Management

#### Setting Tool Filters
```python
# Basic allowlist
result = mcp_tool.set_tool_filter(
    server_name="filesystem-server",
    allowed_tools=["read_file", "write_file", "list_directory"],
    enabled=True
)

# Basic blocklist
result = mcp_tool.set_tool_filter(
    server_name="api-server",
    blocked_tools=["admin_delete", "system_reset"],
    enabled=True
)

# Disable filtering
result = mcp_tool.set_tool_filter(
    server_name="trusted-server",
    enabled=False
)
```

#### Getting Filter Configuration
```python
result = mcp_tool.get_tool_filter("my-server")
if result.success:
    if result.result["has_filter"]:
        filter_config = result.result["filter_config"]
        print(f"Allowed tools: {filter_config.get('allowed_tool_names')}")
        print(f"Blocked tools: {filter_config.get('blocked_tool_names')}")
        print(f"Enabled: {filter_config.get('enabled')}")
    else:
        print("No tool filter configured")
```

#### Removing Filters
```python
# Disable filtering completely
result = mcp_tool.remove_tool_filter("my-server")
```

### Using Filtered Tools

#### Listing Tools with Filtering
```python
# List all available tools (filtered if filter is configured)
result = mcp_tool.list_tools("my-server", apply_filter=True)

# List tools without applying filters (shows all server tools)
result = mcp_tool.list_tools("my-server", apply_filter=False)

# Explicitly list filtered tools
result = mcp_tool.list_filtered_tools("my-server")

if result.success:
    tools = result.result["tools"]
    print(f"Found {len(tools)} tools (filtered: {result.result['filtered']})")
```

#### Tool Call Enforcement
Tool filters are automatically enforced when calling tools:

```python
# This will succeed if 'read_file' is allowed
result = mcp_tool.call_tool("my-server", "read_file", {"path": "/tmp/file.txt"})

# This will fail if 'delete_file' is blocked
result = mcp_tool.call_tool("my-server", "delete_file", {"path": "/tmp/file.txt"})
if not result.success:
    print(f"Tool call blocked: {result.message}")
```

### Configuration File Integration

Tool filters are persisted in the configuration file:

```json
{
  "mcpServers": {
    "secure-server": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "my_server"],
      "tool_filter": {
        "allowed_tool_names": ["read_file", "search_files"],
        "enabled": true
      }
    },
    "restricted-server": {
      "type": "sse",
      "url": "https://api.example.com/mcp",
      "tool_filter": {
        "blocked_tool_names": ["admin_delete", "system_config"],
        "enabled": true
      }
    }
  }
}
```

### Use Cases

1. **Security**: Restrict access to dangerous operations
2. **Role-based Access**: Different tool sets for different user roles
3. **Testing**: Limit tools during development/testing
4. **Specialized Workflows**: Create focused tool subsets for specific tasks
5. **Compliance**: Ensure only approved tools are accessible

### Best Practices

- **Use Allowlists for High-Security Scenarios**: More secure than blocklists
- **Document Filter Rationale**: Clearly document why specific tools are filtered
- **Test Filter Configurations**: Verify filters work as expected
- **Monitor Filter Usage**: Track which tools are being blocked
- **Regular Filter Reviews**: Periodically review and update filters

## Architecture

```
datus/tools/mcp_tools/
├── __init__.py              # Package exports
├── mcp_config.py           # Pydantic data models
# (mcp_server_types.py merged into mcp_config.py)
├── mcp_manager.py          # Core business logic
└── mcp_tool.py             # BaseTool interface
```

## Integration with CLI

This backend implementation provides the foundation for CLI commands like:

```bash
.mcp list                    # List all MCP servers
.mcp add <name> <type>       # Add new MCP server
.mcp remove <name>           # Remove MCP server
.mcp get <name>              # Get server config
.mcp check <name>            # Check server connectivity
.mcp tools <name>            # List tools from a server
.mcp filter set <name>       # Set tool filter for server
.mcp filter get <name>       # Get tool filter configuration
.mcp filter remove <name>    # Remove tool filter
```

## Error Handling

All operations return structured results with success/failure status and descriptive messages:

```python
result = mcp_tool.add_server(...)
if result.success:
    print(f"Success: {result.message}")
    server_config = result.result
else:
    print(f"Error: {result.message}")
```