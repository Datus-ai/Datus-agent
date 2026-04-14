# 跨库迁移 (migration)

## 概览

`migration` 是一个内置 subagent，专注于不同数据库引擎间的表迁移。完整生命周期：

1. 发现并检查源/目标数据库
2. 生成目标 DDL（跨库类型映射）
3. 通过 `transfer_query_result` 传输数据
4. 执行强制对数校验（7 项检查）

## 快速开始

```bash
# 从 DuckDB 迁移到 Greenplum
/migration 把 local_duckdb 里的 users 表迁移到 greenplum，目标表 public.users_copy

# 迁移到 StarRocks
/migration 把 local_duckdb 的 orders 迁移到 starrocks，目标表 test.orders_copy
```

## 前置条件

源库和目标库都必须在 `agent.yml` 中配置：

```yaml
agent:
  service:
    databases:
      local_duckdb:
        type: duckdb
        uri: duckdb:///./sample_data/duckdb-demo.duckdb
      greenplum:
        type: greenplum
        host: 127.0.0.1
        port: 15432
        username: gpadmin
        password: pivotal
        database: test
        sslmode: disable
```

工具调用中的 `database` 参数使用 **逻辑名**（YAML key），不是引擎内部的数据库名。

## 内置工具

| 工具 | 用途 |
|------|------|
| `list_databases` | 列出可用数据库（含类型信息） |
| `list_tables` | 列出表 |
| `describe_table` | 获取列定义 |
| `read_query` | 执行只读 SQL |
| `get_table_ddl` | 获取建表语句 |
| `execute_ddl` | 在目标库执行 DDL |
| `execute_write` | 在目标库执行 DML |
| `transfer_query_result` | 跨库数据传输 |
| `read_file` / `write_file` | SQL 文件读写 |

## 工作流

```
阶段 1：发现数据库（list_databases）→ 检查源表结构
阶段 2：生成目标 DDL（类型映射）→ 创建目标表
阶段 3：传输数据（transfer_query_result）
阶段 4：对数校验（7 项强制检查）
阶段 5：输出迁移报告
```

## 类型映射

| DuckDB | Greenplum | StarRocks |
|--------|-----------|-----------|
| VARCHAR | VARCHAR | VARCHAR(65533) |
| VARCHAR(n) | VARCHAR(n) | VARCHAR(n) |
| TEXT | TEXT | STRING |
| INTEGER | INTEGER | INT |
| BIGINT | BIGINT | BIGINT |
| DECIMAL(p,s) | NUMERIC(p,s) | DECIMAL(p,s) |
| BOOLEAN | BOOLEAN | BOOLEAN |
| DATE | DATE | DATE |
| TIMESTAMP | TIMESTAMP | DATETIME |

不支持的类型（LIST、STRUCT、MAP、BLOB）会报错。

## 对数校验

每次迁移后自动执行 7 项检查：

1. **行数** — 源端与目标端总行数比较
2. **空值率** — 各列空值数量比较
3. **最值** — 数值/日期列 MIN/MAX 比较
4. **去重计数** — 关键列基数比较
5. **重复键** — 目标表重复键检查
6. **样本对比** — 按键排序取前 10 行比较
7. **数值聚合** — SUM/AVG 比较

## 限制

- 单次传输最大 1,000,000 行
- 部分失败不支持事务回滚
- 通过批量 INSERT 写入（非 COPY 或 stream load）

## 可选配置

```yaml
agent:
  agentic_nodes:
    migration:
      max_turns: 40       # 默认：40
```

## 使用的 Skill

- **data-migration** — 迁移工作流和对数校验
- **table-validation** — Schema 和数据质量检查

## 与 gen_job 的区别

| 能力 | gen_job | migration |
|------|---------|-----------|
| 单库 ETL | 支持 | 不支持 |
| 跨库迁移 | 不支持 | 支持 |
| `transfer_query_result` | 无 | 有 |
| 对数校验 | 可选验证 | 强制 7 项 |
| 默认 max_turns | 30 | 40 |
