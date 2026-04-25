# 数据管道 (gen_job)

## 概览

`gen_job` 是一个内置 subagent，同时覆盖 **单库 ETL** 和 **跨库传输**。从一个或多个源表构建或更新目标表，在源库与目标库不同时执行数据传输，并校验结果。

agent 会根据用户的 prompt 自动判断路径：源库和目标库相同则走单库 ETL（CREATE TABLE AS SELECT / INSERT FROM SELECT）；不同则走跨库传输（通过 `transfer_query_result`），并激活 `data-migration` skill 执行轻量对账校验。

## 快速开始

启动 Datus CLI 后使用 gen_job subagent：

```bash
# 单库 ETL
/gen_job 从 orders 和 customers 表构建一个汇总表

# 跨库传输
/gen_job 把 local_duckdb 里的 users 表传输到 greenplum
```

chat agent 检测到 ETL 或迁移任务时也会自动委派给 gen_job。

## 前置条件

### 数据库配置

所有涉及的数据源必须在 `agent.yml` 的 `services.datasources` 中配置：

```yaml
agent:
  services:
    datasources:
      local_duckdb:
        type: duckdb
        uri: duckdb:///./sample_data/duckdb-demo.duckdb
        name: demo
      greenplum:
        type: greenplum
        host: 127.0.0.1
        port: 15432
        username: gpadmin
        password: pivotal
        database: test
        schema_name: public
        sslmode: disable
      starrocks:
        type: starrocks
        host: 127.0.0.1
        port: 9030
        username: root
        password: ""
        database: test
        catalog: default_catalog
```

每个数据库条目有一个 **逻辑名**（YAML key，如 `local_duckdb`、`greenplum`）。指定源库和目标库时使用此逻辑名。

### 跨库传输额外要求

- 源库和目标库都必须可访问
- 源库需支持 pandas 查询执行（DuckDB、PostgreSQL 等）
- 目标库 adapter 需已安装（`datus-greenplum`、`datus-starrocks` 等）

## 内置工具

| 工具 | 用途 |
|------|------|
| `list_databases` | 列出可用数据库（多 connector 模式下返回类型信息） |
| `list_tables` | 列出数据库中的表 |
| `describe_table` | 获取列定义和元数据 |
| `read_query` | 执行只读 SQL 查询 |
| `get_table_ddl` | 获取建表语句 |
| `execute_ddl` | 执行 DDL（CREATE/ALTER/DROP TABLE/SCHEMA） |
| `execute_write` | 执行 DML（INSERT/UPDATE/DELETE） |
| `transfer_query_result` | 跨库数据传输 |
| `get_migration_capabilities` | 读取目标 adapter 的 `MigrationTargetMixin` 提示：requires / forbids / type_hints |
| `suggest_table_layout` | 让目标 adapter 根据源列建议分布键 / 分区 / 排序键 |
| `validate_ddl` | 针对目标方言做静态校验（可选再做 dry-run CREATE+DROP），执行 DDL 前先过一遍 |
| `read_file` / `write_file` | 读写 SQL 文件 |
| `ask_user` | 交互确认（仅 interactive 模式） |

所有数据库工具都支持 `database` 参数，用于显式路由到目标数据库。

## 工作流

### 单库 ETL

```
阶段 1：检查源表和目标表
阶段 2：生成并执行 DDL / DML
阶段 3：验证结果（内置存在性/行数检查，加上显式 schema 契约检查）
阶段 4：输出摘要
```

### 跨库传输

```
阶段 1：发现数据库（list_databases）并检查源表
阶段 2：对目标库调用 get_migration_capabilities() 获取方言提示
        + 对 OLAP 目标调用 suggest_table_layout()
阶段 3：草拟目标 DDL → validate_ddl() → execute_ddl()
阶段 4：传输数据（transfer_query_result）
阶段 5：行数对账和目标侧 sanity check
阶段 6：输出迁移报告
```

## 跨库传输详解

### 数据库发现

agent 首先调用 `list_databases()` 发现可用数据库：

```json
[
  {"name": "local_duckdb", "type": "duckdb"},
  {"name": "greenplum", "type": "greenplum"},
  {"name": "starrocks", "type": "starrocks"}
]
```

### 方言提示（由 adapter 驱动）

类型映射和 DDL 要求**不再硬编码在 agent 中**。每个目标 adapter 在 `datus-db-core` 中实现 `MigrationTargetMixin`，各自声明自身方言的契约。agent 通过三个包装工具消费：

- **`get_migration_capabilities(database=target)`** — 返回目标的 `dialect_family`、`requires`（DDL 必须包含的子句）、`forbids`（DDL 不能出现的模式）、`type_hints`（推荐映射）和参考 `example_ddl`。
- **`suggest_table_layout(database=target, columns_json=...)`** — 返回 OLAP 相关建议（StarRocks 的 `{duplicate_key, distributed_by, buckets}`、ClickHouse 的 `{engine, order_by}`、Trino/Hive 的 `{partitioned_by}` 等）。
- **`validate_ddl(database=target, ddl=..., target_table=...)`** — 做结构化校验（如 StarRocks 必须有 `DUPLICATE KEY` + `DISTRIBUTED BY`；ClickHouse 必须有 `ENGINE` + `ORDER BY`）；可选再跑 `dry_run_ddl`（在目标库 CREATE + DROP 临时表）。

本版本已实现 Mixin 的 adapter：StarRocks、Greenplum、PostgreSQL、MySQL、ClickHouse、Trino、Snowflake、Redshift、DuckDB、SQLite，以及基于 SQLAlchemy 基类的通用 OLTP fallback。其他未实装的 adapter（如 BigQuery、Hive、Spark、ClickZetta）回退到纯 LLM 模式——`get_migration_capabilities` 返回 `{"supported": false, "warning": ...}`，由 LLM 凭自身知识处理目标方言。

要为新方言加提示：在对应 adapter 实现 `MigrationTargetMixin` 即可，agent 不需改动。

### 数据传输

`transfer_query_result` 工具负责数据搬运：

```
transfer_query_result(
    source_sql="SELECT * FROM users",
    source_database="local_duckdb",
    target_table="public.users_copy",
    target_database="greenplum",
    mode="replace"        # replace（TRUNCATE+INSERT）或 append
)
```

限制：
- 单次传输最大 1,000,000 行
- 部分失败不支持事务回滚
- 通过批量 INSERT 写入（非 COPY 或 stream load）

### 对账校验

`transfer_query_result` 之后会自动执行轻量对账校验：

1. **工具返回的行数一致性** — 比较 `source_row_count` 与 `transferred_row_count`。
2. **目标侧行数** — 在目标表上执行一次 `COUNT(*)`，与 `transferred_row_count` 对比。
3. **目标侧样本** — 可选读取少量目标表样本，确认表可查询。

空值率、最值、去重计数、重复键、样本 diff、数值聚合等更重的检查应按项目需要写成项目级 validator skill。

## 可选配置

gen_job 开箱即用。可在 `agent.yml` 中自定义：

```yaml
agent:
  agentic_nodes:
    gen_job:
      max_turns: 40       # 默认：40（跨库场景需要更多 turn）
```

## 使用的 Skill

- **gen-table** — 建表和 DDL 决策
- **table-validation** — 显式 schema 契约检查
- **data-migration** — 跨库传输工作流和轻量对账校验

## 示例

### 示例 1：构建汇总表

```
用户：从 orders 表构建一个 daily_sales_summary 表，
      按日期聚合 total_amount 和 order_count

gen_job：
  1. 检查 orders 表结构
  2. 生成 CREATE TABLE daily_sales_summary DDL
  3. 执行 INSERT ... SELECT 聚合查询
  4. 验证行数和 schema
  5. 输出摘要
```

### 示例 2：DuckDB 迁移到 Greenplum

```
用户：把 local_duckdb 里的 users 表迁移到 greenplum，
      目标表 public.users_copy

gen_job：
  1. 调用 list_databases() → local_duckdb (duckdb) 和 greenplum (greenplum)
  2. describe_table("users", database="local_duckdb") → 6 列
  3. get_migration_capabilities(database="greenplum")
     → dialect_family="postgres-like"，建议 DISTRIBUTED BY
  4. 草拟 CREATE TABLE public.users_copy (...) DISTRIBUTED BY (id)
  5. validate_ddl(database="greenplum", ddl=..., target_table="users_copy")
     → errors=[]，通过
  6. execute_ddl(database="greenplum") 执行 DDL
  7. transfer_query_result(source_database="local_duckdb",
     target_database="greenplum", mode="replace")
  8. 激活 data-migration skill，执行行数和目标侧 sanity check
  9. 输出迁移报告，标注各检查项 pass/fail
```

### 示例 3：MySQL 迁移到 StarRocks

```text
用户：把 mysql_prod 的 orders 表迁移到 starrocks，目标 test.orders_copy

gen_job：
  1. get_migration_capabilities(database="starrocks")
     → 要求 DUPLICATE KEY + DISTRIBUTED BY HASH + BUCKETS
  2. suggest_table_layout(database="starrocks", columns_json=...)
     → {"duplicate_key": ["order_id"], "distributed_by": ["order_id"], "buckets": 10}
  3. 按建议 + type_hints 草拟 CREATE TABLE
  4. validate_ddl() 发现键列缺 NOT NULL → LLM 修正
  5. 在 starrocks 上执行 DDL
  6. transfer_query_result + 轻量对账校验
```

## 与 gen_table 的区别

| 能力 | gen_table | gen_job |
|------|-----------|---------|
| 建表 (CTAS) | 支持 | 支持 |
| 建表 (DDL) | 支持 | 支持 |
| 数据写入 (INSERT/UPDATE/DELETE) | 不支持 | 支持 |
| 跨库迁移 | 不支持 | 支持 |
| 用户确认 (ask_user) | 必须 | 可选 |
| 默认 max_turns | 20 | 40 |

需要交互式建表时用 **gen_table**，需要 ETL 流水线或跨库迁移时用 **gen_job**。
