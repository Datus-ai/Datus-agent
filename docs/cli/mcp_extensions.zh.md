# MCP 扩展

## 1. 概览

MCP（Model Context Protocol）是 Datus-CLI 连接外部工具服务器的方式，让你的智能体能力超越内置功能。

通过 `.mcp`，你可以：

- 添加本地或远程 MCP 服务器（stdio、HTTP、SSE）
- 列出所有可用服务器及其状态
- 检查服务器连接情况与可用工具
- 在 CLI 内直接调用任意 MCP 服务器暴露的工具

这让 Datus-CLI 拥有无限扩展性：无需改动核心即可接入 SQLite、Snowflake、MetricFlow、文件系统，甚至自建的 MCP 服务器。

## 2. 基础用法

### 添加新的 MCP 服务器

```bash
.mcp add <name> <command> [args...]
```

**示例：**

```bash
# 本地 stdio 服务器
.mcp add --transport stdio sqlite /Users/me/bin/uv -- -m mcp_sqlite_server

# 携带认证头的 SSE 服务器
.mcp add --transport sse api-server https://api.example.com/mcp/sse \
  --header "Authorization: Bearer token" --timeout 30.0

# HTTP 流式服务器
.mcp add --transport http metricflow https://localhost:9000/mcp
```

---

### 查看已有服务器

```bash
.mcp list
```

展示所有已配置的 MCP 服务器及其状态：

```
1. duckdb-mftutorial  ✘ failed
2. filesystem         ✔ connected
3. metricflow         ✔ connected
4. snowflake_local    ✔ connected
5. sqlite             ✔ connected
```

按 Enter 可查看详情，包括命令、参数、环境变量与可用工具。

---

### 检查指定服务器

```bash
.mcp check <mcp_name>
```

验证连接状况并打印该服务器提供的工具列表。

---

### 调用服务器工具

```bash
.mcp call <mcp_name>.<tool_name> <args>
```

**示例：**

```bash
.mcp call sqlite.list_tables
.mcp call metricflow.list_metrics
```

---

## 3. 高级特性

### 存储与配置

- 所有 MCP 服务器定义存放在单一 JSON 文件：`~/.datus/conf/.mcp.json`
- 每条记录包含命令、参数、可选的环境变量与 Header
- 可以在 command、args、env、url、headers 中使用 `${VAR}` 或 `${VAR:-default}` 展开环境变量

**配置示例：**

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

### 服务器类型

- **stdio** —— 本地启动子进程，通过 stdin/stdout 通信
- **http** —— 连接到可流式响应的 HTTP 端点
- **sse** —— 连接到 Server-Sent Events 端点

开发本地工具优先使用 stdio（最快）；http 适合常驻服务；sse 适用于远端云服务。

### 环境变量展开

MCP 配置支持环境变量展开：

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

### 错误处理与调试

- 清晰报告服务器连接失败原因
- 工具调用错误会附带详细信息
- 提供调试模式诊断连接问题
- 针对瞬时失败具备自动重试逻辑

### 安全性考量

- 含密钥的环境变量会安全处理
- 认证 Header 会加密存储
- 本地 stdio 服务器以用户权限运行
- 远程连接采用安全协议（HTTPS/WSS）

## 集成示例

### 数据库集成

```bash
# 为本地开发添加 SQLite 服务器
.mcp add sqlite uvx mcp-server-sqlite --db-path ./data/sample.db

# 通过 MCP 查询数据表
.mcp call sqlite.list_tables
.mcp call sqlite.read_query "SELECT * FROM customers LIMIT 5"
```

### MetricFlow 集成

```bash
# 添加 MetricFlow 服务器以访问语义层
.mcp add metricflow python -m mcp_metricflow_server

# 调用业务指标
.mcp call metricflow.list_metrics
.mcp call metricflow.query_metrics --metrics revenue --dimensions customer_segment
```

### 文件系统集成

```bash
# 添加文件系统服务器以执行文件操作
.mcp add filesystem python -m mcp_filesystem_server --base-path /project/data

# 通过 MCP 执行文件操作
.mcp call filesystem.list_directory /reports
.mcp call filesystem.read_file /reports/summary.md
```

