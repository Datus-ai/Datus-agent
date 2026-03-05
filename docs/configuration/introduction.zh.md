# 配置概览

Agent 是 Datus 的核心配置单元。它决定智能体的行为方式、所连接的模型与数据库，以及工作流如何执行。通过自定义 Agent，你可以让 Datus 适配不同环境与业务场景。

## 要点速览

- **弹性与扩展性**：可配置多家 LLM、连接多种数据库，并定义节点策略与工作流规则
- **统一配置文件**：`agent.yml` 同时用于 datus-agent 与 datus-cli
- **启动优先级**：

    1. 命令行 `-f` 指定的文件
    2. `./conf/agent.yml`
    3. `~/.datus/conf/agent.yml`

- **职责分离**：MCP（Model Context Protocol）配置独立存放在 `.mcp.json`，不混入 `agent.yml`

这种结构让 Agent 模块化、可移植、易维护，便于在不同环境中保持一致的运行方式。

## 配置总览

Datus Agent 配置决定系统“如何工作”——选用哪些模型、组件如何连接（节点、工作流、存储、命名空间、基准测试），以及查询如何端到端处理。

| 模块 | 作用 | 关键概念/职责 |
|---|---|---|
| **[Agent](agent.md)** | 全局设置与模型提供方 | 定义默认目标 LLM，以及受支持的提供方（类型、Base URL、API Key、模型名） |
| **[Nodes](nodes.md)** | 任务级处理单元 | 每个“节点”负责一环（结构关联、SQL 生成、推理、反思、输出、聊天、实用工具） |
| **[Workflow](workflow.md)** | 节点编排 | 定义顺序/并行/子工作流/反思路径，描述回答用户问题的执行链 |
| **[Storage](storage.md)** | 向量与嵌入配置 | 管理嵌入模型、设备、存储路径，以及元数据/文档/指标的嵌入与检索 |
| **[Namespace](namespace.md)** | 数据库连接抽象 | 以命名空间封装多数据库连接（Snowflake、StarRocks、SQLite、DuckDB 等） |
| **[Benchmark](benchmark.md)** | 评测与测试 | 配置基准数据集（如 BIRD-DEV、Spider2、Semantic Layer）评估 SQL 生成效果 |

## 配置结构

```yaml
# 模型
models:
  default: "openai:gpt-4"
  providers:
    openai:
      type: "openai"
      base_url: "https://api.openai.com/v1"
      api_key: "${OPENAI_API_KEY}"

# 数据库连接
namespace:
  production:
    type: snowflake
    account: "${SNOWFLAKE_ACCOUNT}"
    username: "${SNOWFLAKE_USER}"
    password: "${SNOWFLAKE_PASSWORD}"

# 工作流
workflow:
  default: "reflection"
  plans:
    reflection:
      - schema_linking
      - generate_sql
      - execute_sql
      - reflect
      - output

# 节点
nodes:
  schema_linking:
    model: "openai:gpt-3.5-turbo"
    similarity_threshold: 0.7

# 存储与嵌入
storage:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  vector_store_path: "./data/vector_store"

# 基准数据集
benchmark:
  bird_dev:
    path: "./benchmark/bird/dev_20240627"
    database_pattern: "**/*.sqlite"
```

## 环境变量支持

所有配置项均支持环境变量与默认值展开：

```yaml
# 直接引用环境变量
api_key: ${OPENAI_API_KEY}

# 带默认值
timeout: ${API_TIMEOUT:-30}

# 复杂插值
connection_string: "postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT:-5432}/${DB_NAME}"
```

## 多环境配置

通过多份配置文件管理不同环境：

```text
conf/
├── agent.yml              # 主配置
├── agent.yml.dev          # 开发环境覆盖
├── agent.yml.staging      # 预发布环境
├── agent.yml.production   # 生产环境
└── .mcp.json              # MCP 服务器配置
```

## 下一步
- **[Agent 设置](agent.md)**：配置模型、提供方与全局设置
- **[数据库命名空间](namespace.md)**：接入多数据库
- **[数据库适配器](../adapters/db_adapters.md)**：安装额外的数据库连接器
- **[工作流定义](workflow.md)**：自定义执行路径
- **[节点配置](nodes.md)**：微调各节点行为
- **[存储设置](storage.md)**：配置知识库/向量存储
- **[基准测试](benchmark.md)**：评测与验证
