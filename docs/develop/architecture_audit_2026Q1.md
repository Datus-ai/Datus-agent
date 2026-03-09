# Datus-Agent Architecture Audit Report (2026 Q1)

## 审计范围

本次审计覆盖 Datus-Agent 全项目架构设计与实现，重点深入 `datus/tools/func_tool/` 模块（LLM 可调用工具层）和测试基础设施。

---

## 一、项目概览

Datus-Agent 是一个 AI 驱动的数据分析 agent：自然语言 → SQL，支持多数据库、RAG 知识库、MCP 协议。核心技术栈：Python 3.12+、OpenAI Agents SDK + LiteLLM、LanceDB、FastAPI、FastMCP、Streamlit。

**代码规模：** ~200+ 源文件，核心模块 9 个，26 个 Node 实现，6 个 LLM 提供商，13 个存储子系统。

---

## 二、架构优点（值得保持）

| 方面 | 评价 |
|------|------|
| **Node 接口契约** | `setup_input / execute / update_context` 三步协议清晰，职责分明 |
| **DB Connector 抽象** | `BaseSqlConnector` + `ConnectorRegistry` + 插件式发现，扩展性好 |
| **LLM 模型抽象** | Factory 模式 + LiteLLM 统一层，新增 Provider 只需继承实现 |
| **权限系统** | 规则化 ALLOW/DENY/ASK，支持 Node 级别覆盖，设计完整 |
| **错误码体系** | `DatusException` + `ErrorCode` 分段枚举，结构化错误处理 |
| **配置系统** | YAML 驱动 + 环境变量替换，工作流可声明式定义 |
| **Prompt 管理** | Jinja2 模板 + 版本化 + 目录回退，灵活可控 |

---

## 三、全项目架构问题（按严重度排序）

### 1. [严重] Agent 类是 "God Object"

**文件：** `datus/agent/agent.py` (53KB, 927 行)

Agent 类承担了过多职责：工作流编排、存储初始化、模型创建、Benchmark 运行、会话管理等全部集中在一个类里，导入了 7 个主要模块。

**影响：** 难以单独测试任何子功能，修改任一子系统都可能影响 Agent 类，新开发者理解成本高。

**建议：**
- 拆分为 `AgentOrchestrator`（工作流编排）、`StorageInitializer`（存储初始化）、`BenchmarkRunner`（基准测试）等独立类
- Agent 类仅保留高层编排逻辑，委托具体工作给子组件

### 2. [严重] AgenticNode 职责过重

**文件：** `datus/agent/node/agentic_node.py` (950 行)

一个基类混合了：会话管理 (SQLiteSession)、Token 计数、上下文压缩 (compaction)、权限系统、Skill 注入、Prompt 管理共 6 个关注点。

**Skill 注入时序问题尤其脆弱：**
- Skill tools 在 `_get_system_prompt()` 期间延迟注入
- 原因是子类 `setup_tools()` 会重置 `self.tools = []`
- 任何初始化顺序变更都可能导致 Skill 丢失

**建议：**
- 抽取 `SessionManager`、`TokenTracker`、`CompactionManager` 为独立组件
- Skill 注入改为不可变集合（与工具列表分离管理）
- `_parse_node_config()` 移到配置层，不应在 Node 内解析

### 3. [严重] 全局单例与隐式初始化

多个核心组件使用模块级全局单例：

| 文件 | 全局状态 | 问题 |
|------|----------|------|
| `datus/tools/db_tools/db_manager.py` | `_INSTANCE = None` | 被 20+ 文件导入，测试间状态泄漏 |
| `datus/storage/cache.py` | `_CACHE_INSTANCE = None` + `_scoped_storage_cache` | 多层缓存，失效难以保证 |
| `datus/configuration/agent_config_loader.py` | `global CONFIGURATION_MANAGER` | 可变配置被全局共享 |

**隐式初始化链：** `load_agent_config()` 内隐式调用 `get_storage_cache_instance()`，函数签名未体现副作用，测试必须了解初始化顺序。

**建议：**
- 引入显式的依赖注入容器（不一定是重框架，简单的工厂+注册即可）
- 配置对象应为不可变（frozen dataclass），组件接收副本而非共享引用
- 测试 fixtures 显式初始化/清理全局状态

### 4. [高] Node 反向依赖 CLI 层

**文件：** `datus/agent/node/agentic_node.py` 导入了 `datus.cli`

Agent Node 本应是纯逻辑层，不应知道 CLI 的存在。当前为了流式输出和状态更新而直接引用 CLI 组件。

**影响：** Node 无法脱离 CLI 使用（如纯 API 模式），违反分层架构原则。

**建议：**
- 定义抽象输出接口（`OutputSink` 或事件总线）
- CLI 实现该接口，Node 只依赖抽象
- API/MCP 服务提供各自的 `OutputSink` 实现

### 5. [高] Node 工厂使用 18 层 if-elif 链

**文件：** `datus/agent/node/node.py` — `Node.new_instance()`

每新增 Node 类型都要修改工厂方法，容易遗漏，违反开闭原则。

**建议：** 改为 Registry 模式 + 装饰器注册：
```python
@register_node(NodeType.TYPE_SCHEMA_LINKING)
class SchemaLinkingNode(Node): ...
```

### 6. [高] 工具层文件过于庞大

| 文件 | 行数 | 问题 |
|------|------|------|
| `datus/tools/func_tool/database.py` | 1027 | DB 操作 + 工具注册 + 命名空间管理混合 |
| `datus/agent/node/gen_sql_agentic_node.py` | 1071 | SQL 生成 Agent 职责过广 |
| `datus/agent/node/gen_ext_knowledge_agentic_node.py` | 1040 | 知识库生成逻辑庞大 |
| `datus/utils/benchmark_utils.py` | 2491 | "工具"文件导入了 configuration、tools、models |

**建议：** 按职责拆分，尤其是 `benchmark_utils.py` 应从 utils 层移到业务层（它不是通用工具）。

### 7. [中] 存储层 LanceDB 硬编码，无后端可替换性

所有 13 个存储子系统直接依赖 LanceDB，没有存储后端抽象层。`BaseEmbeddingStore` 直接调用 `lancedb.connect()`。

**影响：** 无法替换向量数据库（如切换到 Milvus、Qdrant、Chroma），也无法做存储层的单元测试 mock。

**建议：**
- 抽取 `VectorStoreBackend` 接口（connect / create_table / search / upsert）
- LanceDB 作为默认实现
- 至少提供 InMemory 实现用于测试

### 8. [中] Storage + RAG 双层模式职责不清

存储层存在两套并行抽象：
- `BaseEmbeddingStore` + 子类（`SchemaStorage`, `SchemaValueStorage` 等）
- `RAG` 类（`SchemaWithValueRAG`, `MetricRAG` 等）

再加上 `StorageCache` 居中调度（8 个不同存储工厂），Scope 过滤逻辑嵌入缓存而非可组合。

**建议：** 明确 Storage 层（数据持久化）和 RAG 层（检索增强）的职责边界，消除重复抽象。

### 9. [中] Node 间代码重复

多个 Node 的 `setup_input()` 和 `update_context()` 有高度相似的样板代码（重复出现在 GenerateSQLNode, ReasonSQLNode, FixNode 等）。流式执行也有相同的 try/yield/except 模板。

**建议：**
- 基类提供 `_build_common_input(workflow)` 辅助方法
- 流式执行用装饰器或 Mixin 消除模板代码

### 10. [中] MCP Server 文件过大

**文件：** `datus/mcp_server.py` (47.9KB, ~1030 行)

单文件包含 `ToolContext`、`ToolContextManager`、`LightweightDynamicMCPServer`、`DatusMCPServer` 四个类 + 工厂函数 + 入口点。

**建议：** 拆分为 `mcp/context.py`、`mcp/server.py`、`mcp/dynamic_server.py`、`mcp/__main__.py`。

### 11. [低] CLI 层代码膨胀

`datus/cli/repl.py` 等 CLI 文件行数过多，建议按子命令拆分。

---

## 四、Func Tool 层深度分析

### Tool Spec 层面问题（LLM 如何理解和使用这些 tool）

#### TS-1. [严重] 同名/同功能 Tool 重复暴露 [已修复]

`ContextSearchTools.search_metrics` 和 `SemanticTools.search_metrics` 功能高度重叠（都调用 `MetricRAG.search_metrics`），但返回格式不同。同时注册到 Agent 的 tool list 时，LLM 无法区分该调哪个，且"无结果"时一个返回 `success=1`，另一个返回 `success=0`，语义不一致。

**修复**: 从 `SemanticTools` 中移除 `search_metrics`，保留 `ContextSearchTools.search_metrics` 作为唯一入口。

#### TS-2. [严重] get_metrics docstring 与功能不符 [已修复]

`ContextSearchTools.get_metrics` 的 docstring 写着 "Search for business metrics using natural language queries"，实际是按 `subject_path + name` 精确查找。`get_reference_sql` 的返回值文档声称 `result` 是 list，实际返回单个 dict。

**修复**: 修正 docstring 准确描述实际行为。

#### TS-3. [严重] 参数命名不一致 [已修复]

`DBFuncTool.search_table` 使用 `database_name` 参数，而同类的 `list_tables`, `describe_table`, `read_query` 等 5 个方法都用 `database`。LLM 在生成参数时容易混用。

**修复**: `database_name` → `database`。

#### TS-4. [高] `read_query` 工具名误导

`database.py` 的 `read_query` 从字面意义上是 "读取一个 query"，但实际功能是执行 SQL 查询（只读 SELECT）。docstring 也过于简短，没有说明这是只读查询还是可以执行 DDL/DML，LLM 可能尝试用它执行 INSERT/UPDATE。

**建议**: 考虑改名为 `execute_sql` 或 `run_query`，并在 docstring 中明确标注 "SELECT only"。

#### TS-5. [高] `todo_write` 要求 LLM 传 JSON 字符串

`plan_tools.py` 的 `todo_write(todos_json: str)` 强制 LLM 把结构化数据序列化为 JSON 字符串再传入，增加了出错概率。LLM 的 function calling 天然支持结构化参数（list of objects）。同样的问题出现在 `generation_tools.py` 的 `end_metric_generation(metric_sqls_json: str)`。

**建议**: 直接接受 `List[Dict]` 类型参数。

#### TS-6. [高] `search_table` 的 docstring 过长（40+ 行）

`database.py` 的 `search_table` docstring 包含了完整的 "Database-specific parameter usage" 指南（PostgreSQL/MySQL/Snowflake/StarRocks/SQLite/DuckDB 各自的参数说明）。这些指南应放在 system prompt 中（一次性提供），而不是在每次 tool call 的 schema 中重复消耗 token。

**建议**: 缩短 docstring，将数据库特定指南移到 prompt 模板。

#### TS-7. [中] 返回值语义不一致

"无结果"时的 `success` 值在不同 tool 之间不统一：

| Tool | 无结果时 |
|------|----------|
| `search_table` | `success=0, error="No metadata rows found."` |
| `list_tables` | `success=1, result=[]` |
| `search_metrics` (ContextSearch) | `success=1, result=[]` (隐式) |
| `get_metrics` | `success=0, error="No matched result"` |

**建议**: 统一约定：查无数据 = `success=1` + 空结果，系统错误 = `success=0` + error message。

#### TS-8. [中] `edit_file` 类型标注与实际不符

`filesystem_tools.py` 的 `edit_file(edits: List[EditOperation])` 实际也接受 `str`（JSON 字符串）和 `List[Dict]`。类型标注应改为 `Union[str, List[Dict], List[EditOperation]]` 以匹配实际行为。

#### TS-9. [中] 部分 tool 缺少 "何时使用" 的引导

以下 tool 的 docstring 缺少使用场景指导，LLM 不知道何时该选择它们：

| Tool | 问题 |
|------|------|
| `generate_sql_summary_id` | 仅一行描述，LLM 不知道何时需要它 |
| `todo_read` / `todo_update` | 过于简短，缺少工作流上下文 |
| `list_databases` | 缺少何时用 `list_databases` vs `search_table` 的指引 |

### 实现层面问题

#### IM-1. [严重] `_normalize_null` 函数重复定义 [已修复]

完全相同的函数在 `context_search.py` 和 `semantic_tools.py` 中重复。

**修复**: 提取到 `base.py` 作为 `normalize_null`，两处改为引用。

#### IM-2. ContextSearchTools 是否应拆分 [评估结论: 暂不拆分]

评估了将 `ContextSearchTools` (511 行) 按 RAG 类型拆分为 `MetricSearchTools`、`ReferenceSqlSearchTools`、`KnowledgeSearchTools` 的方案。

**结论：当前不值得拆分**，理由：
1. 文件本身不算大（511行），各方法简单独立
2. `list_subject_tree` 是跨类型的聚合方法，拆分后无处安放
3. `available_tools()` 按数据可用性动态组装是核心价值，拆分后需额外协调层
4. `@mcp_tool_class` 装饰器绑定到整个类
5. Sub-agent 的 `_show_*` 权限控制依赖共享配置

**何时该拆：** 文件超过 800+ 行、频繁新增 RAG 类型、某个类型需要独特初始化逻辑时。

---

## 五、已完成修复 (PR #445 + review/architecture-audit)

### P0-1: SemanticTools.search_metrics 重复注册 [已修复]
- **影响文件**: `datus/tools/func_tool/semantic_tools.py`

### P0-2: get_metrics / get_reference_sql docstring 误导 [已修复]
- **影响文件**: `datus/tools/func_tool/context_search.py`

### P0-3: search_table 参数命名不一致 [已修复]
- **影响文件**: `datus/tools/func_tool/database.py`

### P0 附带: _normalize_null 重复定义 [已修复]
- **影响文件**: `datus/tools/func_tool/base.py`, `context_search.py`, `semantic_tools.py`

### UT 配置加载问题 [已修复, PR #445]
- **问题**: `test_mcp_server.py` 和 `test_tools_output.py` 未显式指定 `config_path`，导致加载项目根目录 `./conf/agent.yml`（无 `bird_sqlite` namespace）而非 `tests/conf/agent.yml`。在 CI 全量运行时因全局单例 `CONFIGURATION_MANAGER` 被其他测试先初始化而"碰巧"通过，但子目录单独运行时必然失败。
- **根因**: `configuration_manager()` 使用全局单例模式，首次调用决定后续所有调用的配置来源。
- **修复**: 所有 `create_server()`, `ToolContextManager()`, `LightweightDynamicMCPServer()`, `create_dynamic_app()` 调用显式传入 `TEST_CONF_DIR / "agent.yml"`。
- **影响文件**: `tests/unit_tests/tools/mcp_tools/test_mcp_server.py`, `tests/unit_tests/tools/test_tools_output.py`

---

## 六、后续改进计划

### P1: Tool docstring 系统性审查

**目标**: func_tool 的 docstring 是 LLM 看到的唯一文档，需要确保每个 tool 的描述与实际行为完全一致。

| Tool 类 | 待审查项 |
|---------|---------|
| `DBFuncTool` | 所有 6 个 tool 的参数描述、返回值格式、使用场景 |
| `ContextSearchTools` | `search_*` vs `get_*` 语义区分是否清晰 |
| `SemanticTools` | `query_metrics` 与 `list_metrics` 的区分说明 |

**行动**: 逐一对比 docstring 与实现，确保参数名/类型/默认值/返回格式完全匹配。

### P1: Tool 注册机制审查

**问题**: 不同 Node 通过手动调用 `available_tools()` 注册 tool，存在以下风险：
1. 同一 Node 同时注册多个 Tool 类时，可能产生同名 tool（已在 TS-1 中发现一例）
2. 缺少注册时的名称冲突检测

**建议**:
- 在 `trans_to_function_tool` 层或 Node 注册层增加 tool name 唯一性检查
- 考虑增加单元测试验证各 Node 注册的 tool 列表无重复

### P2: Tool 返回格式统一

**现状**: 部分 tool 返回 `FuncToolResult`，部分直接返回 dict。`FuncToolResult` 有 `success/error/result` 三字段，但 `result` 的内部结构因 tool 而异。

**建议**:
- 统一所有 tool 返回 `FuncToolResult`
- 在 docstring 中明确 `result` 字段的具体结构（如 dict 的 key 列表）

### P2: search_table 返回值优化

**现状**: `search_table` 返回 `{"metadata": [...], "sample_data": [...]}` 两个列表，LLM 需要自行关联。

**建议**: 考虑合并为单个列表，每项包含 metadata + 对应 sample_data，减少 LLM 理解成本。

---

## 七、测试基础设施问题

### 全局单例污染 (系统性风险)

**现状**: `CONFIGURATION_MANAGER` 和 `load_agent_config` 使用全局单例模式，测试间共享状态。

**影响**:
- 测试结果依赖执行顺序
- 子目录单独运行可能失败
- 不同 namespace 配置的测试互相污染

**建议**:
1. **短期**: 确保所有测试显式传入 `config_path` 和 `reload=True`（如 `load_acceptance_config` 的做法）
2. **长期**: 考虑为测试提供 `conftest.py` fixture 自动注入测试配置，或使用 `monkeypatch` 隔离全局状态

### ConnectorRegistry 加载警告

**现状**: 测试运行时始终输出 adapter 加载失败警告：
```
Failed to load adapter hive: ConnectorRegistry.register() got an unexpected keyword argument 'capabilities'
Failed to load adapter spark: ...
Failed to load adapter trino: ...
Failed to load adapter clickhouse: ...
```

**原因**: `ConnectorRegistry.register()` 接口变更后，部分 adapter 注册代码未同步更新。

**建议**: 统一 adapter 注册接口，或在这些 adapter 的注册代码中适配新接口。

### CI / Nightly / Regression 边界不清

**观察**: `test_tools_output.py` 标记为 CI 测试，但它：
- 依赖外部文件系统路径 (`~/benchmark/bird/dev_20240627`)
- 依赖 LLM API key（通过 `LLMBaseModel.create_model`）
- 依赖预构建的测试数据

**建议**: 审查所有 CI 标记的测试，确保严格符合 "零外部依赖、零网络访问、< 5s" 的标准。

---

## 八、修改清单

| 文件 | 修改类型 | PR |
|------|---------|-----|
| `datus/tools/func_tool/base.py` | 新增 `normalize_null` | review/architecture-audit |
| `datus/tools/func_tool/context_search.py` | 修复 docstring, 提取 `_normalize_null` | review/architecture-audit |
| `datus/tools/func_tool/database.py` | `database_name` → `database` | review/architecture-audit |
| `datus/tools/func_tool/semantic_tools.py` | 移除 `search_metrics`, 提取 `_normalize_null` | review/architecture-audit |
| `tests/unit_tests/tools/mcp_tools/test_mcp_server.py` | 显式传入 `TEST_CONF_PATH` | PR #445 |
| `tests/unit_tests/tools/test_tools_output.py` | 使用 `TEST_CONF_DIR` | PR #445 |
