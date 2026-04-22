# Datus MCP Server

通过 **Model Context Protocol (MCP)** 暴露 Datus 的数据库查询与上下文搜索工具，支持与
Claude Desktop、Claude Code 及其他 MCP 兼容客户端集成。

**服务模式：**

- **静态模式（Static Mode）**：单数据源，适用于 Claude Desktop、CLI 工具或单租户 HTTP/SSE 服务
- **动态模式（Dynamic Mode）**：多数据源 HTTP/SSE 服务，通过 URL 路径访问所有数据源

**支持的传输方式：**

- `http`：Streamable HTTP（双向通信，默认）
- `sse`：Server-Sent Events over HTTP（适用于 Web 客户端）
- `stdio`：标准输入/输出（适用于 Claude Desktop 和 CLI 工具）

## 快速开始

- 安装 Datus：

```bash
pip install datus-agent
```

- 启动 MCP 服务：

```bash
# 静态模式：单数据源
uvx --from datus-agent datus-mcp --datasource <your datasource>
uvx --from datus-agent datus-mcp --datasource <your datasource> --transport http --host 127.0.0.1 --port 8000
# 动态模式：多数据源 HTTP/SSE 服务
uvx --from datus-agent datus-mcp --dynamic --transport http --host 127.0.0.1 --port 8000
uvx --from datus-agent datus-mcp --dynamic --transport sse --host 127.0.0.1 --port 8000

# 或者直接运行
datus-mcp --datasource <your datasource>
# 动态模式：多数据源 HTTP/SSE 服务
datus-mcp --dynamic --transport http --host 127.0.0.1 --port 8000
datus-mcp --dynamic --transport sse --host 127.0.0.1 --port 8000
```

## 客户端集成

### Claude Code

启动 Datus MCP 服务后，将其添加到 Claude Code：

```bash
# 启动服务（SSE 模式）
datus-mcp --dynamic --transport sse --port 8000

# 添加到 Claude Code
claude mcp add --transport sse datus http://127.0.0.1:8000/sse/<your datasource>
```

### Claude Desktop

Claude Desktop 需要通过 [mcp-remote](https://www.npmjs.com/package/mcp-remote) 连接远程 MCP 服务。

启动 Datus MCP 服务后，使用以下脚本配置 Claude Desktop：

```bash
# Claude Desktop 可能无法直接找到 npx，因此需要解析完整路径
NODE_BIN_DIR=$(dirname $(which node))
NPX_PATH=$(which npx)

cat > ~/Library/Application\ Support/Claude/claude_desktop_config.json << EOF
{
  "mcpServers": {
    "datus-agent": {
      "command": "$NPX_PATH",
      "args": [
        "mcp-remote@latest",
        "http://127.0.0.1:8000/sse/<your datasource>",
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

或者手动将以下内容添加到 `claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "datus-agent": {
      "command": "npx",
      "args": [
        "mcp-remote@latest",
        "http://127.0.0.1:8000/sse/<your datasource>",
        "--transport",
        "sse-only"
      ]
    }
  }
}
```

!!! tip
    Claude Desktop 也支持 stdio 直连方式，参见[其他 MCP 客户端](#其他-mcp-客户端)章节。

### 其他 MCP 客户端 {#其他-mcp-客户端}

支持 stdio 传输的 MCP 客户端：

- **使用 uvx：**

```json
{
  "mcpServers": {
    "datus": {
      "command": "uvx",
      "args": [
        "--from",
        "datus-agent",
        "datus-mcp",
        "--datasource",
        "<your datasource>",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

- **直接使用 python：**

```json
{
  "mcpServers": {
    "datus": {
      "command": "python",
      "args": [
        "-m",
        "datus.mcp_server",
        "--datasource",
        "<your datasource>",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

支持 HTTP/SSE 传输的 MCP 客户端：

- **SSE：**
```json
{
  "mcpServers": {
    "DatusServer": {
      "url": "http://127.0.0.1:8000/sse/<your datasource>",
      "transport": "sse"
    }
  }
}
```

- **Streamable HTTP：**
```json
{
  "mcpServers": {
    "DatusServer": {
      "url": "http://127.0.0.1:8000/mcp/<your datasource>",
      "transport": "http"
    }
  }
}
```

## HTTP 服务模式

**静态模式（单数据源）：**

```bash
# Streamable HTTP（默认，双向通信）
datus-mcp --datasource <your datasource> --transport http --host 0.0.0.0 --port 8000

# SSE 模式（适用于 Web 客户端）
datus-mcp --datasource <your datasource> --transport sse --port 8000
```

连接地址：

- Streamable HTTP：`http://localhost:8000/mcp`
- SSE：`http://localhost:8000/sse`

**动态模式（多数据源）：**

启动单个服务即可通过 URL 路径访问所有已配置的数据源：

```bash
# 以 SSE 模式启动动态服务
datus-mcp --dynamic --host 0.0.0.0 --port 8000 --transport sse

# 以 HTTP Stream 模式启动动态服务
datus-mcp --dynamic --host 0.0.0.0 --port 8000 --transport http
```

连接指定数据源：

- Streamable HTTP：`http://localhost:8000/mcp/{datasource}`
- SSE：`http://localhost:8000/sse/{datasource}`
- 指定 subagent：`http://localhost:8000/mcp/{datasource}?subagent={subagent_name}`

示例：

- Streamable HTTP，datasource 为 `bird_sqlite`：`http://localhost:8000/mcp/bird_sqlite`
- SSE，datasource 为 `bird_sqlite`：`http://localhost:8000/sse/bird_sqlite`
- Streamable HTTP，datasource 为 `superset`，子代理为 `sales_dashboard`：`http://localhost:8000/mcp/superset?subagent=sales_dashboard`

信息端点：

- `http://localhost:8000/` - 服务信息及可用数据源
- `http://localhost:8000/health` - 健康检查

## 可用工具

MCP 服务暴露以下工具：

| 类别           | 工具                                                                                                                                                              |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **数据库**     | `list_databases`, `list_schemas`, `list_tables`, `search_table`, `describe_table`, `get_table_ddl`, `read_query`                                                  |
| **上下文搜索** | `list_subject_tree`, `search_metrics`, `get_metrics`, `search_reference_sql`, `get_reference_sql`, `search_semantic_objects`, `search_knowledge`, `get_knowledge` |

## 命令行参数

```bash
datus-mcp --help

模式选择（互斥，必选其一）：
  --dynamic            动态模式：通过 /mcp/{datasource} URL 支持所有数据源
  --datasource, -n      静态模式：指定单个数据源

静态模式选项：
  --sub-agent, -s      Sub-agent 名称，用于限定上下文范围
  --database, -d       覆盖默认数据库名称
  --transport, -t      传输方式：http（默认）、sse、stdio

动态模式选项：
  --transport, -t      传输方式：http（默认）、sse
                       http: 通过 /mcp/{datasource} URL 访问
                       sse: 通过 /sse/{datasource} URL 访问

通用选项：
  --config, -c         Agent 配置文件路径
  --host               HTTP 传输绑定地址（默认：0.0.0.0）
  --port, -p           HTTP 传输绑定端口（默认：8000）
  --debug              启用调试日志
```
