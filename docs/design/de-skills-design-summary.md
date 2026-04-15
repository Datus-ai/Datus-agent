# DE 分支设计总结：Subagent、Skill 与 Tool

## 概览

本分支（`de-skills`）新增了面向数据工程（Data Engineering）场景的内置 Subagent、Skill 和 Tool，覆盖三大核心场景：

| 场景 | Subagent | Skill | 核心 Tool |
|------|----------|-------|-----------|
| 建表 | `gen_table` | `gen-table` | `execute_ddl` |
| 单库 ETL | `gen_job` | `gen-table` + `table-validation` | `execute_ddl` + `execute_write` |
| 跨库迁移 | `migration` | `data-migration` + `table-validation` | `execute_ddl` + `execute_write` + `transfer_query_result` |

---

## 1. 内置 Subagent

### 1.1 gen_table — 建表代理

- **职责**：从 SQL（CTAS）或自然语言描述创建数据库表
- **Node 类**：`GenTableAgenticNode`（`datus/agent/node/gen_table_agentic_node.py`）
- **Node Type**：`NodeType.TYPE_GEN_TABLE`
- **可用 Tool**：
  - 标准只读 DB 工具（`list_tables`、`describe_table`、`read_query` 等）
  - `execute_ddl` — 执行 DDL 语句
  - `ask_user` — 交互式确认（DDL 执行前必须确认）
  - 文件系统工具（读写 SQL 文件）
- **Skill 绑定**：`gen-table`
- **典型流程**：
  1. 解析用户输入（SQL 或自然语言）
  2. 探查源表 schema
  3. 生成 DDL 并通过 `ask_user` 确认
  4. 执行 DDL 并验证结果

### 1.2 gen_job — 单库 ETL 代理

- **职责**：在同一数据库内执行 ETL 任务（建表 + 写数据 + 验证）
- **Node 类**：`GenJobAgenticNode`（`datus/agent/node/gen_job_agentic_node.py`）
- **Node Type**：`NodeType.TYPE_GEN_JOB`
- **max_turns**：30
- **可用 Tool**：
  - 标准只读 DB 工具
  - `execute_ddl` — 执行 DDL（CREATE TABLE 等）
  - `execute_write` — 执行 DML（INSERT / UPDATE / DELETE）
  - 文件系统工具
  - `ask_user`（interactive 模式下）
- **Skill 绑定**：`gen-table` + `table-validation`
- **与 migration 的区别**：**不包含** `transfer_query_result`，仅限单库操作
- **典型流程**：
  1. 探查源表和目标表 schema
  2. 生成 DDL 创建目标表
  3. 通过 INSERT ... SELECT 写入数据
  4. 使用 table-validation skill 验证数据质量

### 1.3 migration — 跨库迁移代理

- **职责**：跨数据库引擎迁移数据（如 DuckDB → Greenplum / StarRocks）
- **Node 类**：`MigrationAgenticNode`（`datus/agent/node/migration_agentic_node.py`）
- **Node Type**：`NodeType.TYPE_MIGRATION`
- **max_turns**：40（迁移流程较长，轮次上限更高）
- **可用 Tool**：
  - 标准只读 DB 工具
  - `execute_ddl` — 在目标库执行 DDL
  - `execute_write` — 在目标库执行 DML
  - `transfer_query_result` — **核心跨库传输工具**
  - 文件系统工具
  - `ask_user`（interactive 模式下）
- **Skill 绑定**：`data-migration` + `table-validation`
- **安全机制**：
  - 目标数据库不可用时**立即报错**，禁止静默回退到其他库
  - `list_databases` 返回每个库的 `available` 状态，迁移前必须确认

---

## 2. Skill 设计

### 2.1 gen-table（建表）

- **触发场景**：用户提供 SQL 或自然语言描述建表需求
- **输入模式**：
  - **SQL 模式**：用户提供 SELECT SQL → CTAS 建表
  - **描述模式**：用户用自然语言描述表结构 → CREATE TABLE DDL
- **关键规则**：
  - DDL 执行前**必须**通过 `ask_user` 确认
  - 用户选择 Cancel 时立即停止
  - 不修改源表，只创建新表

### 2.2 table-validation（表验证）

- **触发场景**：表创建或写入后，验证 schema 和数据质量
- **验证项目**：
  - 对象存在性检查
  - 缺失 / 多余列检查
  - 类型和 nullable 不匹配
  - 行数门槛
  - Null 比率
  - 数值范围
  - 接受值集合
  - 正则 / 格式规则
  - 唯一性 / 重复键
- **输出**：紧凑的 pass/fail 报告

### 2.3 data-migration（跨库迁移）

- **触发场景**：需要将表从一个数据库迁移到另一个数据库
- **6 阶段工作流**：

| 阶段 | 动作 | 使用的 Tool |
|------|------|------------|
| Phase 1: 源库探查 | 获取源表 schema、行数、样本数据 | `describe_table(database=source)`, `read_query(database=source)` |
| Phase 2: 目标库探查 | 检查目标库是否已有同名表 | `list_tables(database=target)`, `describe_table(database=target)` |
| Phase 3: 生成目标 DDL | 跨方言类型映射 + DDL 生成 | `execute_ddl(sql, database=target)` |
| Phase 4: 数据传输 | 批量跨库传输 | `transfer_query_result(...)` |
| Phase 5: 对数校验 | 7 项 reconciliation 检查 | `read_query(database=source/target)` |
| Phase 6: 报告 | 汇总迁移结果 | — |

- **关键安全规则**：
  - 每个 tool 调用都必须显式指定 `database` 参数
  - 源库只读，所有写操作仅对目标库
  - 对数校验（reconciliation）是强制步骤，不可跳过
  - 目标库不可用时**禁止**回退到其他库

### 2.4 bi-validation（BI 发布验证）

- **触发场景**：BI 变更发布后验证
- **验证内容**：数据集/图表/仪表盘创建状态、刷新成功、关键指标一致性

---

## 3. Tool 层设计

### 3.1 DBFuncTool 新增方法

所有新增方法位于 `datus/tools/func_tool/database.py`：

#### execute_ddl(sql, database="")

- 执行 DDL 语句（CREATE TABLE、ALTER TABLE 等）
- `database` 参数支持指定目标数据库（跨库场景）
- 执行后显式 commit，避免 SQLAlchemy 事务锁

#### execute_write(sql, database="", min_rows=None, max_rows=None)

- 执行 DML 语句（INSERT / UPDATE / DELETE）
- 支持行数门槛校验（`min_rows` / `max_rows`）

#### transfer_query_result(source_sql, source_database, target_table, target_database, mode, batch_size)

- **核心跨库传输工具**
- 工作原理：
  1. 在 `source_database` 上执行 `source_sql`，获取 Pandas DataFrame
  2. 对 DataFrame 进行清洗（NaT/NaN → None）
  3. 在 `target_database` 上批量 INSERT（dialect-aware 引号处理）
- `mode`：`replace`（先 TRUNCATE 再插入）或 `append`（追加插入）
- **行数上限**：`_TRANSFER_MAX_ROWS`（防止超大数据集卡死）
- **方言感知列名引号**：
  - MySQL 系列（StarRocks、Hive 等）：使用反引号 `` ` ``
  - PostgreSQL 系列（Greenplum 等）：使用双引号 `"`
- **安全机制**：源库或目标库不可用时返回明确错误，禁止回退

#### list_databases（增强）

- 多连接器模式下返回每个数据库的 `{name, type, available, error}` 信息
- 供 LLM 在迁移前确认数据库可用性

### 3.2 多连接器路由

- `_get_connector(database)` 使用 `get_conn(db_name, db_name)` 路由到正确的数据库连接器
- `_needs_multi_connector()` 自动检测子代理是否配置了 `transfer_query_result`，决定是否启用多连接器模式
- 每个数据库在配置中是独立的 namespace

### 3.3 Migration Helper 层

位于 `datus/tools/migration/`，提供纯函数工具，**不依赖任何具体的数据库 adapter**：

#### type_mapping.py — 跨方言类型映射

- `map_columns_between_dialects(columns, source_dialect, target_dialect)` → 映射后的列定义列表
- 支持 DuckDB → Greenplum、DuckDB → StarRocks
- 参数保留：源 `DECIMAL(10,2)` 的参数优先于目标默认值
- 不支持的类型（LIST、STRUCT、MAP 等）抛 `UnsupportedTypeError`

#### target_profiles.py — 目标库 DDL 生成

- `GreenplumProfile`：schema 限定表名，无特殊 DDL 后缀
- `StarRocksProfile`：
  - 自动选择 DUPLICATE KEY 列（优先级：id 后缀 > 整数类型 > 非空列）
  - 生成 `DISTRIBUTED BY HASH(...)` 子句
- `build_target_ddl(columns, target_table, profile)` — 统一入口

#### reconciliation.py — 迁移后对数校验

生成 7 项 SQL 对（source 和 target 各一条）：

| 检查项 | 说明 |
|--------|------|
| `row_count` | 总行数比较 |
| `null_ratio` | 每列 NULL 数比较 |
| `min_max` | 数值/日期列范围比较 |
| `distinct_count` | 关键列基数比较 |
| `duplicate_key` | 目标表重复键检查 |
| `sample_diff` | 基于主键的样本行比对（top 10） |
| `numeric_aggregate` | 数值列 SUM/AVG 比较 |

---

## 4. 注册架构

每个内置 Subagent 需要在 **8 个位置** 注册：

| 位置 | 文件 | 说明 |
|------|------|------|
| 1 | `datus/configuration/node_type.py` | 添加 `TYPE_XXX` 常量 + ACTION_TYPES + 描述 |
| 2 | `datus/agent/node/node.py` | `new_instance()` 工厂分支 |
| 3 | `datus/agent/node/node_factory.py` | `create_interactive_node()` + `create_node_input()` |
| 4 | `datus/tools/func_tool/sub_agent_task_tool.py` | `NODE_CLASS_MAP` |
| 5 | 同上 | `BUILTIN_SUBAGENT_DESCRIPTIONS` |
| 6 | 同上 | `_create_builtin_node()` |
| 7 | 同上 | `_resolve_node_type()` |
| 8 | 同上 | `_build_node_input()` |

---

## 5. Prompt 模板

| 模板 | 服务对象 | 关键内容 |
|------|----------|----------|
| `gen_table_system_1.0.j2` | gen_table | 两种输入模式、DDL 确认流程 |
| `gen_job_system_1.0.j2` | gen_job | 单库 ETL 流程，不含跨库内容 |
| `migration_system_1.0.j2` | migration | 跨库迁移完整流程 + 安全硬规则 |

---

## 6. 测试覆盖

| 测试文件 | 覆盖范围 | 数量 |
|----------|----------|------|
| `test_type_mapping.py` | 跨方言类型映射 | 77 |
| `test_target_profiles.py` | DDL 生成 Profile | 28 |
| `test_reconciliation.py` | 对数校验 SQL 生成 | 13 |
| `test_database.py` | DBFuncTool 新方法 | ~20 |
| `test_gen_job_agentic_node.py` | gen_job subagent | 17 |
| `test_migration_agentic_node.py` | migration subagent | 17 |
| `test_migration_integration.py` | 集成测试（GP+SR） | 8 |
| `test_migration_subagent.py` | 子代理集成测试 | 6 |

---

## 7. 架构图

```
┌────────────────────────────────────────────────────────┐
│                    用户请求（自然语言）                    │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│              gen_sql (主代理 / 路由器)                    │
│         task(type="gen_table|gen_job|migration")        │
└──────┬──────────────┬──────────────┬──────────────────┘
       │              │              │
       ▼              ▼              ▼
┌─────────────┐ ┌───────────┐ ┌──────────────┐
│  gen_table   │ │  gen_job   │ │  migration   │
│  (建表)      │ │ (单库ETL)  │ │ (跨库迁移)    │
├─────────────┤ ├───────────┤ ├──────────────┤
│ Skills:     │ │ Skills:   │ │ Skills:      │
│ • gen-table │ │ • gen-table│ │ • data-      │
│             │ │ • table-  │ │   migration  │
│             │ │   validation│ │ • table-    │
│             │ │           │ │   validation │
├─────────────┤ ├───────────┤ ├──────────────┤
│ Tools:      │ │ Tools:    │ │ Tools:       │
│ • DB 只读   │ │ • DB 只读  │ │ • DB 只读    │
│ • execute_  │ │ • execute_│ │ • execute_   │
│   ddl       │ │   ddl     │ │   ddl        │
│ • ask_user  │ │ • execute_│ │ • execute_   │
│ • 文件系统   │ │   write   │ │   write      │
│             │ │ • ask_user│ │ • transfer_  │
│             │ │ • 文件系统 │ │   query_     │
│             │ │           │ │   result     │
│             │ │           │ │ • ask_user   │
│             │ │           │ │ • 文件系统    │
└─────────────┘ └───────────┘ └──────────────┘
                                     │
                     ┌───────────────┼───────────────┐
                     ▼               ▼               ▼
              ┌────────────┐ ┌────────────┐ ┌────────────┐
              │ type_      │ │ target_    │ │ reconcili- │
              │ mapping    │ │ profiles   │ │ ation      │
              │ (类型映射)  │ │ (DDL生成)   │ │ (对数校验)  │
              └────────────┘ └────────────┘ └────────────┘
```

---

## 8. 设计决策记录

| 决策 | 原因 |
|------|------|
| gen_job 和 migration 拆为两个独立 subagent | 单库 ETL 和跨库迁移职责不同，tool 集合不同（gen_job 不需要 transfer） |
| migration helper 层不依赖具体 adapter | adapter 是按需安装的可选依赖，migration 工具必须独立 |
| 方言感知列名引号放在 transfer_query_result 中 | 临时方案，已创建 issue #563 追踪迁移到 adapter 层 |
| 迁移前强制检查数据库可用性 | 防止 LLM 在目标库不可用时静默回退到其他库 |
| reconciliation 为强制步骤 | 跨库迁移存在类型转换和精度风险，必须校验 |
| max_turns: gen_job=30, migration=40 | 迁移流程包含更多阶段（探查→建表→传输→对数），需要更多轮次 |
