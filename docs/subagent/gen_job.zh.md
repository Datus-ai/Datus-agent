# ETL 作业 (gen_job)

## 概览

`gen_job` 是一个内置 subagent，用于单库 ETL。在同一数据库内从源表构建目标表（CREATE TABLE AS SELECT、INSERT FROM SELECT 等）。

跨库迁移请使用 [migration](migration.zh.md) subagent。

## 快速开始

启动 Datus CLI 后使用 gen_job subagent：

```bash
# 单库 ETL
/gen_job 从 orders 和 customers 表构建一个汇总表

# 跨库迁移
/gen_job 把 local_duckdb 里的 users 表迁移到 greenplum
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

### 跨库迁移额外要求

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
| `read_file` / `write_file` | 读写 SQL 文件 |
| `ask_user` | 交互确认（仅 interactive 模式） |

所有数据库工具都支持 `database` 参数，用于显式路由到目标数据库。

## 工作流

### 单库 ETL

```
阶段 1：检查源表和目标表
阶段 2：生成并执行 DDL / DML
阶段 3：验证结果（行数、schema、数据质量）
阶段 4：输出摘要
```

### 跨库迁移

```
阶段 1：发现数据库（list_databases）并检查源表
阶段 2：生成目标表 DDL（跨库类型映射）
阶段 3：传输数据（transfer_query_result）
阶段 4：对数校验（7 项检查）
阶段 5：输出迁移报告
```

## 跨库迁移详解

### 数据库发现

agent 首先调用 `list_databases()` 发现可用数据库：

```json
[
  {"name": "local_duckdb", "type": "duckdb"},
  {"name": "greenplum", "type": "greenplum"},
  {"name": "starrocks", "type": "starrocks"}
]
```

### 类型映射

类型在不同方言间自动映射：

| DuckDB | Greenplum | StarRocks |
|--------|-----------|-----------|
| VARCHAR | VARCHAR | VARCHAR(65533) |
| VARCHAR(n) | VARCHAR(n) | VARCHAR(n) |
| TEXT | TEXT | STRING |
| INTEGER | INTEGER | INT |
| BIGINT | BIGINT | BIGINT |
| DOUBLE | DOUBLE PRECISION | DOUBLE |
| DECIMAL(p,s) | NUMERIC(p,s) | DECIMAL(p,s) |
| BOOLEAN | BOOLEAN | BOOLEAN |
| DATE | DATE | DATE |
| TIMESTAMP | TIMESTAMP | DATETIME |

不支持的类型（LIST、STRUCT、MAP、BLOB 等）会报错。

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

### 对数校验

迁移后自动执行 7 项对数检查：

1. **行数** — 源端与目标端总行数比较
2. **空值率** — 各列空值数量比较
3. **最值** — 数值/日期列的 MIN/MAX 比较
4. **去重计数** — 关键列基数比较
5. **重复键** — 目标表重复键检查
6. **样本对比** — 按键排序取前 10 行比较
7. **数值聚合** — SUM/AVG 比较

## 可选配置

gen_job 开箱即用。可在 `agent.yml` 中自定义：

```yaml
agent:
  agentic_nodes:
    gen_job:
      max_turns: 30       # 默认：30
```

## 使用的 Skill

- **gen-table** — 建表和 DDL 决策
- **table-validation** — Schema 和数据质量检查
- **data-migration** — 跨库迁移工作流和对数校验

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
  1. 调用 list_databases() → 发现 local_duckdb (duckdb) 和 greenplum (greenplum)
  2. 调用 describe_table("users", database="local_duckdb") → 获取 6 列信息
  3. 类型映射：INTEGER→INTEGER, VARCHAR→VARCHAR, DECIMAL→NUMERIC 等
  4. 在 greenplum 上执行 CREATE TABLE
  5. 调用 transfer_query_result(source_database="local_duckdb", target_database="greenplum")
  6. 执行 7 项对数检查，比较两端数据
  7. 输出迁移报告，标注各检查项 pass/fail
```

## 与 gen_table 的区别

| 能力 | gen_table | gen_job |
|------|-----------|---------|
| 建表 (CTAS) | 支持 | 支持 |
| 建表 (DDL) | 支持 | 支持 |
| 数据写入 (INSERT/UPDATE/DELETE) | 不支持 | 支持 |
| 跨库迁移 | 不支持 | 支持 |
| 用户确认 (ask_user) | 必须 | 可选 |
| 默认 max_turns | 20 | 30 |

需要交互式建表时用 **gen_table**，需要 ETL 流水线或跨库迁移时用 **gen_job**。
