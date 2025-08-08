# MCP Tools - Model Context Protocol Server Management

This package provides comprehensive functionality for managing MCP servers including stdio, sse, and http communication types.

## Features

- **Multi-Protocol Support**: Support for stdio, sse (Server-Sent Events), and http communication protocols
- **Config Management**: Full CRUD operations for MCP server configs
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
    server_type="stdio",
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
    server_type="sse", 
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
```

## Server Types

### 1. Stdio (Standard Input/Output)
For local command execution with process communication via stdin/stdout.

```python
mcp_tool.add_server(
    name="stdio-server",
    server_type="stdio",
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
    server_type="sse",
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
    server_type="http",
    url="https://localhost:9000/mcp",
    headers={"Content-Type": "application/json"},
    timeout=120.0
)
```

## Config File

The MCP tools use a JSON config file to persist server configs. The configuration is loaded in the following order (first found wins):
1. Explicit config path (if provided)
2. `conf/.mcp.json` (current directory)
3. `~/.datus/conf/.mcp.json` (user home directory)

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
      }
    },
    "api_server": {
      "type": "sse",
      "url": "${API_BASE_URL:-https://api.example.com}/mcp",
      "headers": {
        "Authorization": "Bearer ${API_KEY}"
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
- `list_tools()` - List available tools from a server
- `call_tool()` - Call a specific tool on a server

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
@mcp list                    # List all MCP servers
@mcp add <name> <type>       # Add new MCP server
@mcp remove <name>           # Remove MCP server  
@mcp get <name>              # Get server config
@mcp check <name>            # Check server connectivity
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

## Future Enhancements

- Server discovery and auto-config
- Plugin system for custom server types
- Advanced config validation
- Integration with external config management systems