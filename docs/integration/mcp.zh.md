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

## 同步工具与容量限制

动态和静态模式均在线程池中执行同步工具、插件参数转换和结果格式化，
并传递请求的 ContextVars；异步工具在请求的事件循环中等待执行。
所有工具和数据源共享同一 MCP 服务的线程池，可在 `agent.yml` 配置：

```yaml
agent:
  mcp_tool_execution:
    max_workers: 8
    max_queue_size: 16
```

以上为默认值。工作线程数必须为正数，排队数必须非负。
运行和排队总数达到上限后，新调用立即返回 MCP 工具错误，客户端可稍后重试。
该线程池独立于 asyncio 默认线程池。

容量参数接受整数字符串；未知字段、布尔值和小数会产生配置错误。
`mcp_tool_execution` 不是映射时使用默认配置。
已缓存的数据源上下文在缓存锁可用时直接获取，不占用工具线程，
也不等待其他数据源的初始化锁。

取消请求不会终止已提交的阻塞数据库调用，也不会提前释放容量；
任务实际结束后才释放名额。还需配置数据源适配器的连接及 socket/read 超时，
HTTP 或 asyncio 超时本身无法停止 StarRocks/PyMySQL 读取。
应确认适配器将超时传递给数据库驱动：`timeout_seconds` 可能仅限制连接池等待，
并不限制 socket 读取。适配器层的超时修复需要在独立发布的数据库适配器包中完成。
关闭服务时停止接收任务，不在事件循环中等待阻塞读取结束。

HTTP/SSE 请求以及排队或运行中的工具会持有动态上下文引用。LRU 淘汰只标记上下文
待清理，最后一个使用者结束后才在后台关闭连接；调用者取消但线程仍在运行时也会保留引用。
关闭服务不等待数据源初始化完成；关闭后才完成初始化的上下文会被清理，不会重新进入缓存。

静态服务器也会保留连接，直到所有已提交的工具调用真正结束，包括请求已取消的调用。
`close()` 标记工具待清理，并在后台关闭连接，不会同步关闭仍在使用的连接。

嵌入后端仍可使用原有的异步 `get_or_create_context()` 和同步 `close_all()` manager 接口。
注入对象的身份及配置管理方式保持兼容。兼容层只串行处理同一数据源/子代理键的查询；
某个键的初始化或权限 I/O 卡住时，其他键仍可正常处理请求。
此前返回的上下文会在权限刷新前获得引用保护，避免并发淘汰提前关闭连接。
返回的 `ToolContext` 会继续由请求持有，淘汰时的 `close()` 延迟到最后一个使用者结束后清理。
manager 应从空缓存启动，所有查询通过此服务器执行，并在初始化及权限 I/O 完成后写入新缓存项，
写入到返回之间不得再次让出事件循环。
尚未完成的查询结束后，才在清理线程执行 `close_all()`。
此类 manager 使用默认的有界工具线程池；其异步查询方法仍需自行将阻塞初始化移出事件循环。

嵌入此库的后端需要更新 Datus-agent 依赖或子模块，并重新构建镜像才能交付修复。

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

MCP 服务注册以下工具。实际可用情况取决于数据源和已配置的能力；请以当前连接的 `tools/list` 为准。

| 类别 | 工具 |
| --- | --- |
| **数据库** | `check_datasource_quality`, `describe_table`, `execute_sql`, `get_migration_capabilities`, `import_database_file`, `list_databases`, `list_schemas`, `list_tables`, `load_file_as_table`, `plan_datasource`, `search_table`, `suggest_table_layout`, `validate_ddl` |
| **上下文搜索** | `get_reference_sql`, `list_subject_tree`, `search_metrics`, `search_reference_sql`, `search_semantic_objects` |
| **语义模型** | `attribution_analyze`, `get_metric`, `get_query_metrics_result`, `list_metrics`, `query_metrics`, `query_rows`, `validate_semantic` |
| **参考模板** | `execute_reference_template`, `get_reference_template`, `render_reference_template`, `search_reference_template` |
| **日期解析** | `parse_temporal_expressions` |
| **平台文档** | `get_document`, `list_document_nav`, `search_document` |
| **文件系统** | `glob`, `grep`, `read_file`, `read_image` |

## 命令行参数

```bash
datus-mcp --help

模式选择（互斥，必选其一）：
  --dynamic            动态模式：通过 /mcp/{datasource} URL 支持所有数据源
  --datasource, -n      静态模式：指定单个数据源

静态模式选项：
  --sub-agent, -s      Sub-agent 名称，用于限定上下文范围
  --config, -c         agent 配置文件路径
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
