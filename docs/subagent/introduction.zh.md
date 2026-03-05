# Subagent 指南

## 概览

Subagent 是 Datus 中专注于特定任务的专用 AI 助手。与处理通用 SQL 查询的默认聊天助手不同，subagent针对特定工作流进行了优化，例如生成语义模型、创建指标或分析 SQL 查询。

## 什么是subagent？

**subagent** 是面向特定任务的 AI 助手，具备以下特性：

- **专用系统提示**：针对特定任务优化的指令
- **自定义工具**：为任务定制的工具集（例如文件操作、验证）
- **范围化上下文**：可选，专用于该subagent的上下文（数据表、指标、Reference SQL）
- **独立会话**：与主聊天分离的对话历史
- **任务导向工作流**：完成特定目标的引导步骤

## 可用subagent

### 1. `gen_semantic_model`

**用途**：从数据库表生成 MetricFlow 语义模型。

**使用场景**：将数据库表结构转换为 YAML 语义模型定义。

**前置条件**：此subagent依赖 [datus-semantic-metricflow](../adapters/semantic_adapters.md)，请先运行 `pip install datus-semantic-metricflow` 安装。

**启动命令**：
```bash
/gen_semantic_model Generate a semantic model for the transactions table
```

**核心特性**：

- 自动获取表 DDL
- 识别度量、维度和标识符
- 使用 MetricFlow 验证
- 同步到知识库

**参考**：[语义模型生成指南](./gen_semantic_model.md)

---

### 2. `gen_metrics`

**用途**：将 SQL 查询转换为可复用的 MetricFlow 指标定义。

**使用场景**：将临时 SQL 计算转换为标准化指标。

**前置条件**：此subagent依赖 [datus-semantic-metricflow](../adapters/semantic_adapters.md)，请先运行 `pip install datus-semantic-metricflow` 安装。

**启动命令**：
```bash
/gen_metrics Generate a metric from this SQL: SELECT SUM(revenue) / COUNT(DISTINCT customer_id) FROM transactions
```

**核心特性**：

- 分析 SQL 业务逻辑
- 确定合适的指标类型（ratio、measure_proxy 等）
- 追加到现有语义模型文件
- 检查重复项

**参考**：[指标生成指南](./gen_metrics.md)

---

### 3. `gen_sql_summary`

**用途**：分析和分类 SQL 查询，用于知识复用。

**使用场景**：构建可搜索的 SQL 查询库，并进行语义分类。

**启动命令**：
```bash
/gen_sql_summary Analyze this SQL: SELECT region, SUM(revenue) FROM sales GROUP BY region
```

**核心特性**：

- 为 SQL 查询生成唯一 ID
- 按域/层级/标签分类
- 创建详细的摘要用于向量搜索
- 支持中文和英文

**参考**：[SQL 摘要指南](./gen_sql_summary.md)

---

### 4. `gen_ext_knowledge`

**用途**：生成和管理业务概念及领域特定定义。

**使用场景**：记录数据库 schema 中未存储的业务知识，例如业务规则、计算逻辑和领域特定概念。

**启动命令**：
```bash
/gen_ext_knowledge Extract knowledge from this sql
Question: What is the highest eligible free rate for K-12 students?
SQL: SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda'
```

**核心特性**：

- **知识差距发现**：Agent 首先尝试独立解决问题，然后与参考 SQL 对比，识别隐含的业务知识
- 生成带有唯一 ID 的结构化 YAML
- 支持主题路径分类（例如 `education/schools/data_integration`）
- 创建新条目前检查重复项
- 同步到知识库以供语义搜索

**参考**：[外部知识生成指南](./builtin_subagents.zh.md#gen_ext_knowledge)

---

### 5. 自定义subagent

你可以在 `agent.yml` 中定义自定义subagent，用于组织特定的工作流。

**配置示例**：
```yaml
agentic_nodes:
  my_custom_agent:
    model: claude
    system_prompt: my_custom_prompt
    prompt_version: "1.0"
    tools: db_tools.*, context_search_tools.*
    max_turns: 30
    agent_description: "Custom workflow assistant"
```

## 如何使用subagent

### 方法 1：CLI 命令（推荐）

使用斜杠命令启动subagent：

```bash
datus --namespace production

# 使用特定任务启动subagent
/gen_metrics Generate a revenue metric
```

**工作流程**：

1. 输入 `/[subagent_name]` 后跟你的请求
2. subagent使用专用工具处理任务
3. 审阅生成的输出（YAML、SQL 等）
4. 确认是否同步到知识库

### 方法 2：Web 界面

通过网页聊天机器人访问subagent：

```bash
datus web --namespace production
```

**步骤**：

1. 在主页面点击 "🔧 Access Specialized Subagents"
2. 选择需要的subagent（例如 "gen_metrics"）
3. 点击 "🚀 Use [subagent_name]"
4. 与专用助手对话

**直接 URL 访问**：
```text
http://localhost:8501/?subagent=gen_metrics
http://localhost:8501/?subagent=gen_semantic_model
http://localhost:8501/?subagent=gen_sql_summary
```

## subagent vs 默认聊天

| 方面 | 默认聊天 | subagent |
|--------|-------------|----------|
| **用途** | 通用 SQL 查询 | 特定任务工作流 |
| **工具** | 数据库工具、搜索工具 | 任务特定工具（文件操作、验证） |
| **会话** | 单一对话 | 每个subagent独立 |
| **提示** | 通用 SQL 辅助 | 任务优化的指令 |
| **输出** | SQL 查询 + 解释 | 结构化工件（YAML、文件） |
| **验证** | 可选 | 内置（例如 MetricFlow 验证） |

**何时使用默认聊天**：

- 临时 SQL 查询
- 数据探索
- 关于数据库的快速问题

**何时使用subagent**：

- 生成标准化工件（语义模型、指标）
- 遵循特定工作流（分类、验证）
- 构建知识库

## 配置

### 基础配置

在 `conf/agent.yml` 中定义subagent：

```yaml
agentic_nodes:
  gen_metrics:
    model: claude                          # LLM 模型
    system_prompt: gen_metrics             # 提示模板名称
    prompt_version: "1.0"                  # 模板版本
    tools: generation_tools.*, filesystem_tools.*, semantic_tools.*  # 可用工具
    hooks: generation_hooks                # 用户确认
    max_turns: 40                          # 最大对话轮数
    workspace_root: /path/to/workspace     # 文件工作空间
    agent_description: "Metric generation assistant"
    rules:                                 # 自定义规则
      - Use check_metric_exists to avoid duplicates
      - Validate with validate_semantic tool
```

### 关键参数

| 参数 | 必需 | 描述 | 示例 |
|-----------|----------|-------------|---------|
| `model` | 是 | LLM 模型名称 | `claude`、`deepseek`、`openai` |
| `system_prompt` | 是 | 提示模板标识符 | `gen_metrics`、`gen_semantic_model` |
| `prompt_version` | 否 | 模板版本 | `"1.0"`、`"2.0"` |
| `tools` | 是 | 逗号分隔的工具模式 | `db_tools.*, semantic_tools.*` |
| `hooks` | 否 | 启用确认工作流 | `generation_hooks` |
| `mcp` | 否 | MCP 服务器名称 | `filesystem_mcp` |
| `max_turns` | 否 | 最大对话轮数 | `30`、`40` |
| `workspace_root` | 否 | 文件操作目录 | `/path/to/workspace` |
| `agent_description` | 否 | 助手描述 | `"SQL analysis assistant"` |
| `rules` | 否 | 自定义行为规则 | 字符串列表 |

### 工具模式

**通配符模式**（所有方法）：
```yaml
tools: db_tools.*, generation_tools.*, filesystem_tools.*
```

**特定方法**：
```yaml
tools: db_tools.list_tables, db_tools.get_table_ddl, generation_tools.check_metric_exists
```

**可用工具类型**：

- `db_tools.*`：数据库操作（列出表、获取 DDL、执行查询）
- `generation_tools.*`：生成辅助工具（检查重复、上下文准备）
- `filesystem_tools.*`：文件操作（读取、写入、编辑文件）
- `context_search_tools.*`：知识库搜索（查找指标、语义模型）
- `semantic_tools.*`：语义层操作（列出指标、查询指标、验证）
- `date_parsing_tools.*`：日期/时间解析和规范化

### MCP 服务器

MCP（Model Context Protocol）服务器提供额外工具：

**内置 MCP 服务器**：

- `filesystem_mcp`：工作空间内的文件系统操作

**配置**：
```yaml
mcp: filesystem_mcp
```

> **注意**：MetricFlow 集成现在通过 [datus-semantic-metricflow](../adapters/semantic_adapters.md) 适配器提供的原生 `semantic_tools.*` 工具实现，不再使用 MCP 服务器。

## 总结

subagent提供**专用的、工作流优化的 AI 助手**，用于特定任务：

- **任务导向**：针对特定工作流优化的提示和工具
- **独立会话**：每个subagent拥有独立的对话历史
- **工件生成**：创建标准化文件（YAML、文档）
- **内置验证**：自动检查和验证（例如 MetricFlow）
- **知识库集成**：同步生成的工件以供复用
- **灵活配置**：自定义工具、提示和行为

