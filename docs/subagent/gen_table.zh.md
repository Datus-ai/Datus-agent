# 建表 (gen_table)

## 概览

`gen_table` 是一个内置 subagent，用于创建数据库表。支持两种输入模式：

1. **SQL 模式 (CTAS)**：提供 JOIN/SELECT SQL 语句，创建宽表用于查询加速
2. **描述模式**：用自然语言描述表结构，生成 CREATE TABLE DDL

agent 分析输入、生成 DDL、请求用户确认、执行 DDL 并验证结果。

## 快速开始

启动 Datus CLI 后使用 gen_table subagent：

```bash
# SQL 模式：从 JOIN 查询创建宽表
/gen_table CREATE TABLE wide_orders AS SELECT o.*, c.name FROM orders o JOIN customers c ON o.customer_id = c.id

# 描述模式：用自然语言描述
/gen_table 创建一个 user_profiles 表，包含 id (int), name (varchar), email (varchar), created_at (timestamp)
```

chat agent 检测到建表任务时也会自动委派给 gen_table。

## 内置工具

| 工具 | 用途 |
|------|------|
| `list_tables` | 列出数据库中的表 |
| `list_databases` | 列出可用数据库 |
| `list_schemas` | 列出 schema |
| `describe_table` | 获取列定义和元数据 |
| `read_query` | 执行只读 SQL（验证、行数统计） |
| `get_table_ddl` | 获取已有的建表语句 |
| `execute_ddl` | 执行 DDL（CREATE/ALTER/DROP TABLE/SCHEMA） |
| `read_file` / `write_file` | 读写 SQL 文件 |
| `ask_user` | 交互确认（仅 interactive 模式） |

gen_table 使用 `DBFuncTool.create_dynamic()` 支持多 connector，可在不同数据库上创建表。

## 工作流

### SQL 模式 (CTAS)

```
1. 解析输入 SQL，识别源表和列
2. 调用 describe_table 获取源表类型信息
3. 可选调用 read_query LIMIT 验证输出
4. 生成 CTAS DDL：CREATE TABLE schema.table AS (SELECT ...)
5. 通过 ask_user 展示完整 DDL → [执行 / 修改 / 取消]
6. 执行 DDL 并验证（行数 + schema 检查）
7. 输出摘要
```

### 描述模式 (CREATE TABLE)

```
1. 解析自然语言描述，提取表名、列、类型
2. 调用 describe_table 获取引用的已有表信息
3. 如关键信息缺失，通过 ask_user 询问
4. 生成 CREATE TABLE DDL
5. 通过 ask_user 展示完整 DDL → [执行 / 修改 / 取消]
6. 执行 DDL 并验证（schema 检查）
7. 输出摘要
```

## 用户确认

gen_table 在执行 DDL 前 **必须** 请求用户确认。完整的 DDL SQL 会在 `ask_user` 组件中展示：

- **执行**：执行 DDL
- **修改**：调整 DDL 后重新确认
- **取消**：立即停止，不创建任何表

DDL 执行失败时，agent 会解析错误、修复 SQL 并重新确认（最多重试 3 次）。

## 可选配置

gen_table 开箱即用。可在 `agent.yml` 中自定义：

```yaml
agent:
  agentic_nodes:
    gen_table:
      max_turns: 20       # 默认：20
```

## 使用的 Skill

- **gen-table** — 建表工作流：输入分析、DDL 生成、用户确认、执行、验证

## 重要规则

- DDL 执行需要用户确认 — 不会静默创建表
- 如果目标表已存在，agent 会警告并询问是否 DROP 后重建
- gen_table 只负责建表 — 语义模型生成请使用 `gen_semantic_model`
- 不会修改源表

## 与 gen_job 的区别

| 能力 | gen_table | gen_job |
|------|-----------|---------|
| 建表 (CTAS) | 支持 | 支持 |
| 建表 (DDL) | 支持 | 支持 |
| 数据写入 (INSERT/UPDATE/DELETE) | 不支持 | 支持 |
| 跨库迁移 | 不支持 | 支持 |
| 用户确认 (ask_user) | 必须 | 可选 |
| 默认 max_turns | 20 | 40 |

需要交互式建表时用 **gen_table**，单库 ETL 或跨库迁移都用 **gen_job**（gen_job 已覆盖两者）。
