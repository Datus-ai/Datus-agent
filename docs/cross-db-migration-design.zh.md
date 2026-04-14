# 跨库迁移支持方案：DuckDB 到 Greenplum / StarRocks

## 背景

当前 Datus 已具备以下基础能力：

- 多数据库配置与 connector 管理
- 读取类工具：`read_query`、`describe_table`、`get_table_ddl`
- 写入类工具：`execute_ddl`、`execute_write`
- 单库 ETL 子 agent：`etl`
- 轻量校验 skill：`table-validation`

同时，`datus-db-adapters` 中已经提供了：

- `GreenplumConnector`
- `PostgreSQLConnector`
- `StarRocksConnector`

其中 PostgreSQL / Greenplum / StarRocks 均具备 DDL 与 DML 执行基础能力。

本方案目标是在此基础上，补齐 **DuckDB -> Greenplum / StarRocks** 的跨库迁移与对数闭环。

## 目标

支持用户在一次会话中完成以下流程：

1. 指定源库 DuckDB 中的源表或源 SQL
2. 指定目标库为 Greenplum 或 StarRocks
3. 自动分析源表结构
4. 生成目标表 DDL
5. 在目标库建表
6. 执行数据搬运
7. 完成迁移后对数
8. 输出迁移结果和校验报告

第一版聚焦：

- 单表迁移
- 全量迁移
- 报告型对数
- 不做自动阻断、自动回滚、CDC、并发迁移

## 总体设计

方案分为三层：

### 1. Core / Tool 层

负责跨库执行能力：

- 目标库 DDL 执行
- 数据搬运
- 类型映射
- 目标库 profile
- 对数 SQL 组织

### 2. Skill / Subagent 层

负责迁移流程编排：

- 新增 `data-migration` skill
- 新增 `migration` subagent
- 使用 skill 明确迁移与对数步骤

### 3. Target Profile 层

负责目标数据库差异收敛：

- Greenplum profile
- StarRocks profile

避免在 agent 逻辑里到处散落目标库分支判断。

## Core 改造

### 1. `execute_ddl` 增加 `database` 参数

#### 当前问题

`execute_write` 已支持：

- `database=...`

但 `execute_ddl` 仍只能作用于当前 connector。

#### 改造目标

将接口改为：

```python
execute_ddl(sql: str, database: str = "")
```

#### 行为

- 不传 `database`：保持兼容，使用当前数据库
- 传 `database`：DDL 明确打到指定目标库

#### 价值

迁移场景中，目标库 schema 和 table 创建必须可显式指定到 Greenplum / StarRocks。

### 2. 新增 `transfer_query_result` 工具

#### 职责

完成跨库数据搬运：

- 从源库执行查询
- 获取结果集
- 批量写入目标表

#### 最小接口

```python
transfer_query_result(
    source_sql: str,
    source_database: str,
    target_table: str,
    target_database: str = "",
    mode: str = "replace",
    batch_size: int = 5000
)
```

#### 语义

- `source_sql`：源端查询语句
- `source_database`：源库
- `target_table`：目标表全名
- `target_database`：目标库
- `mode`
  - `replace`
  - `append`
- `batch_size`：批量写入大小

#### 代码位置

`transfer_query_result` 作为 `DBFuncTool` 的新方法，放在：

```text
datus/tools/func_tool/database.py
```

原因：该方法需要调用 `self._get_connector(source_database)` 和 `self._get_connector(target_database)` 分别获取源端和目标端 connector，复用 `DBFuncTool` 现有的多 connector 路由和 LRU 缓存机制。

#### `replace` 模式语义

- `replace`：先对目标表执行 `TRUNCATE TABLE`，再批量 `INSERT`
- `append`：直接批量 `INSERT`，不清除已有数据

不使用 `DROP + CREATE` 方式，避免破坏已建好的表结构、索引和权限。

#### 第一版实现

- 源端 DuckDB：通过 connector 执行查询并取为 pandas DataFrame
- 目标端 Greenplum / StarRocks：批量 `INSERT`
- DataFrame → INSERT 转换路径：从 DataFrame 逐批生成参数化 INSERT 语句，使用目标 connector 底层 cursor 的 `executemany()` 写入，避免手工拼接 SQL 字符串带来的转义和性能问题

#### 数据量限制与内存预算

第一版对单次迁移的数据量有以下约束：

- 源端查询结果集整体加载到 pandas DataFrame，不做 cursor-based 分批读取
- **建议上限**：单次迁移不超过 **100 万行** 或 **500 MB** 内存占用
- 超过此上限时，工具应返回明确错误提示，建议用户添加 WHERE 条件分批迁移
- 后续版本可通过 cursor-based chunked read（`fetchmany(batch_size)`）支持更大数据量

#### 部分失败处理

第一版不支持事务回滚。已知限制：

- 如果写入过程中发生错误（例如第 3 批 INSERT 失败），目标表中已写入的前 2 批数据不会自动回滚
- 工具返回错误信息中包含：已成功写入行数、失败位置、错误原因
- 用户可通过重新执行 `replace` 模式来从头覆盖
- 后续版本可考虑引入 savepoint 或 staging table 模式实现原子性

#### 第一版不做

- `COPY`
- Greenplum 外部表装载
- StarRocks stream load
- Arrow / Parquet 中转优化
- cursor-based 分批读取（大数据量支持）
- 事务回滚 / staging table 原子写入

### 3. 新增类型映射能力

#### 建议位置

放在 `datus` 主仓，不放在 `datus-db-adapters`：

```text
datus/tools/migration/type_mapping.py
```

#### 原因

类型映射是跨方言能力，不属于单个 adapter 的职责。

#### 接口建议

```python
map_columns_between_dialects(
    columns,
    source_dialect: str,
    target_dialect: str,
    target_profile: str | None = None,
)
```

#### 输入

`columns` 为结构化列定义列表，例如：

- column name
- source type
- nullable

#### 输出

目标库列定义列表。

#### 第一版覆盖类型

- text / varchar
- integer / bigint
- double / decimal
- boolean
- date
- timestamp

#### 不支持的类型处理

以下 DuckDB 类型在第一版中 **不支持映射**，遇到时应明确报错而非静默丢弃：

- 嵌套类型：`LIST`、`STRUCT`、`MAP`、`UNION`
- 二进制类型：`BLOB`、`BYTEA`
- 特殊数值：`HUGEINT`（超出 BIGINT 范围时报错；可安全转为 BIGINT 时降级为 BIGINT）
- 空间类型：`GEOMETRY`、`POINT` 等

处理策略：

- `map_columns_between_dialects()` 遇到不支持的类型时，返回 `UnsupportedTypeError`，包含列名和源类型
- 调用方（`build_target_ddl`）汇总所有不支持的列，一次性报告给用户
- 用户可选择：排除这些列后继续迁移，或手动指定目标类型

### 4. 新增目标库 profile

#### 建议位置

```text
datus/tools/migration/target_profiles.py
```

#### 职责

将目标数据库的建表差异统一封装。

#### Greenplum profile

负责：

- schema 命名
- PostgreSQL / Greenplum 类型名
- 可选 distribution policy 占位
- 普通表 DDL 生成

第一版默认：

- 不自动生成复杂 distribution policy
- 未显式指定时生成普通 `CREATE TABLE`

#### StarRocks profile

负责：

- catalog / database / table 全限定名
- StarRocks 类型映射
- 建表语法要求
- key model
- distribution 策略

第一版默认：

- 使用 `DUPLICATE KEY`
- key 候选列选择规则（按优先级）：
  1. 列名包含 `id` 或 `_id` 后缀的列
  2. 类型为 `INT` / `BIGINT` 的列
  3. 非 nullable 的列优先
  4. 以上都不满足时，回退为第一列
- 取前 1-3 个满足条件的列作为 key
- 使用 `DISTRIBUTED BY HASH(<key cols>) BUCKETS 10`
- 不做分区、rollup、bitmap index、routine load

### 5. 新增迁移 helper

#### 建议位置

```text
datus/tools/migration/
  type_mapping.py
  target_profiles.py
  reconciliation.py
  inspect.py
```

#### helper 列表

##### `inspect_source_table(...)`

输出：

- 列定义
- row count
- 样例值
- 主键候选

##### `build_target_ddl(...)`

输入：

- source columns
- source dialect
- target dialect
- target profile
- target table

输出：

- 目标库建表 DDL

##### `build_reconciliation_checks(...)`

输出标准对数项对应的 SQL。

这些 helper 第一版做普通 Python 函数，不先暴露成 tool。

## Skill 设计

### 新增 `data-migration`

#### 建议目录

```text
skills/data-migration/
  SKILL.md
  references/checklist.md
```

#### 风格

沿用当前轻量 skill 风格：

- 无 `assets/`
- 无 `scripts/`
- 自然语言描述流程
- 用现有工具完成迁移与对数

#### skill 流程

1. Inspect source
2. Inspect target
3. Build target DDL
4. Create schema / table
5. Transfer data
6. Reconcile source vs target

#### skill 要求

必须明确：

- 先区分 `source_database` 与 `target_database`
- 源库只能读
- 目标库负责建表与写入
- 必须完成迁移后对数
- 对数顺序固定：
  1. row count
  2. null ratio
  3. min/max
  4. distinct count
  5. duplicate key
  6. key-based sample diff
  7. 数值聚合 compare

## Subagent 设计

### 新增 `migration`

在 `conf/agent.yml` 中增加：

```yaml
migration:
  model: claude_benchmark
  system_prompt: migration
  prompt_version: "1.1"
  agent_description: "Migrate a table from a source database to a target database, rebuild the target table, load data, and reconcile source vs target."
  max_turns: 40
  skills: "data-migration, table-validation"
  tools: db_tools.list_tables, db_tools.describe_table, db_tools.read_query, db_tools.get_table_ddl, db_tools.execute_ddl, db_tools.execute_write, db_tools.transfer_query_result, plan_tools, filesystem_tools
```

#### Scope 配置策略

迁移 subagent 需要同时访问源库和目标库的表。Scope 配置需注意：

- `scoped_context` 必须同时包含源端和目标端的表模式
- 源端表：只需读权限（`read_query`、`describe_table`、`get_table_ddl`）
- 目标端表：需要写权限（`execute_ddl`、`execute_write`、`transfer_query_result`）
- 若未配置 scope（`scoped_context` 为空），则默认允许访问所有已配置的 database，无需额外处理
- 若配置了 scope，调用方需确保 scope pattern 覆盖 `source_database.*` 和 `target_database.<target_table>`

## Prompt 设计

### 新增 `migration_system_1.1.j2`

#### 职责

明确这是一个跨库迁移 agent，必须：

1. Inspect source
2. Inspect target
3. Build target table
4. Transfer data
5. Reconcile

#### 关键约束

- 必须明确使用 `source_database` / `target_database`
- 不得把源库写操作和目标库读操作混淆
- 必须在迁移完成后做对数
- 最终输出迁移结果与对数摘要

## 对数方案

第一版不新增复杂 compare tool，继续复用：

- `read_query(database=...)`

### 固定对数项

1. row count compare
2. key null ratio compare
3. numeric / date min-max compare
4. distinct count compare
5. duplicate key compare
6. key-based sample compare
7. 数值聚合 compare
   - `sum`
   - `avg`
   - `min`
   - `max`

### 返回格式

每个 check 返回：

- `name`
- `source_query`
- `target_query`
- `source_value`
- `target_value`
- `status`
- `notes`

第一版是报告型，不自动阻断。

## 现有工具是否足够

### 已有且可复用的能力

- `read_query(database=...)`
- `describe_table`
- `get_table_ddl`
- `execute_write(database=...)`
- Greenplum / StarRocks adapter 的 DDL / DML 执行基础

### 需要新增的能力

- `execute_ddl(database=...)`
- `transfer_query_result`
- 类型映射 helper
- target profile
- `data-migration` skill
- `migration` subagent
- `migration_system_1.1.j2`

## 测试方案

### 1. Tool 单测

- `execute_ddl(database=...)` 命中指定 connector
- `transfer_query_result`
  - `replace`
  - `append`
  - 空结果集
  - 源查询失败
  - 目标写入失败

### 2. 类型映射单测

覆盖：

- DuckDB -> Greenplum
- DuckDB -> StarRocks

类型至少包括：

- varchar
- text
- int
- bigint
- decimal
- boolean
- date
- timestamp

### 3. Target profile 单测

- Greenplum DDL 生成
- StarRocks DDL 生成
- StarRocks 默认 key / distribution 逻辑

### 4. 集成测试

#### DuckDB -> Greenplum

- 建 schema
- 建表
- 导入
- 对数

#### DuckDB -> StarRocks

- 建 database/table
- 导入
- 对数

### 5. Skill / subagent smoke test

最小 case：

- `DuckDB raw.users -> Greenplum public.users_copy`
- `DuckDB raw.users -> StarRocks default_catalog.demo.users_copy`

## Assumptions

- Greenplum 与 StarRocks adapter 均已通过 adapter discovery 正常加载
- Greenplum / StarRocks 都支持当前仓库中的 DDL/DML 调用链
- 第一版只支持单表、全量迁移
- 第一版不做 CDC、双写、回滚、自动阻断
- 第一版优先正确性与完整性，不优先性能
- 后续如需性能优化，再分别补：
  - Greenplum `COPY`
  - StarRocks stream load
