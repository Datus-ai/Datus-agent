# SQL 执行

## 1. 概览

除了命令（`/`、`@`、`!`），Datus-CLI 也能像传统 SQL 客户端一样工作。你可以直接执行 SQL 查询，并通过内置的元数据命令（如 `/tables`、`/databases`、`/schemas` 等）探索数据库结构。

我们提供了一组内置的元数据命令，以统一方式访问不同数据库的元信息；当然，也可以继续使用原生命令，如 `SHOW DATABASES`、`DESCRIBE TABLES` 等。

---

## 2. 基础用法

### 执行 SQL

在 CLI 中直接运行 SQL 查询：

```sql
Datus-sql> SELECT T2.Zip FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`District Name` = 'Fresno County Office of Education' AND T1.`Charter School (Y/N)` = 1
```

**输出：**
```
┏━━━━━━━━━━━━━━━━━━┓
┃ Zip              ┃
┡━━━━━━━━━━━━━━━━━━┩
│ 93726-5309       │
│ 93628-9602       │
│ 93706-2611       │
│ 93726-5208       │
│ 93706-2819       │
└──────────────────┘
Returned 5 rows in 0.01 seconds
```

### 探索元数据 {#explore-metadata}

使用内置元数据命令查看数据库结构：

```bash
/databases                       # 列出所有数据库
/database <database_name>        # 切换当前数据库
/schemas                         # 列出所有 Schema（或展示详情）
/schema <schema_name>            # 切换当前 Schema
/tables                          # 列出当前 Schema 的所有表
/table_schema <table_name>       # 查看表结构详情
/indexes <table_name>            # 查看表索引
```

结合聊天或调试 SQL 时，这些命令能帮助你快速熟悉数据环境。

## 3. 高级能力

### 跨数据库兼容

元数据命令在不同数据库类型间保持一致体验：

- **SQLite**：本地文件数据库
- **DuckDB**：分析型工作负载
- **Snowflake**：云数仓
- **MySQL/PostgreSQL**：传统关系型数据库
- **StarRocks**：实时分析场景

### 丰富的结果展示

SQL 结果会带来：

- **表格化输出**：列对齐、带边框
- **类型感知**：数字、日期、字符串均按类型优化显示
- **性能指标**：执行耗时与返回行数
- **错误处理**：清晰的错误提示与建议

### 查询历史

- 所有执行过的查询都会自动保存
- 可通过上下方向键访问历史记录
- 查询历史在会话之间持久化
- 与 `@sql` 上下文系统深度集成

### 导出选项

查询结果可导出为多种格式：

- **CSV**：用于数据分析与报表
- **JSON**：用于 API 集成与数据交换
- **SQL**：用于查询导出与回放

## 最佳实践

### 查询优化

1. **充分利用 LIMIT**：对大表做探索性查询时先限制返回行数
2. **关注索引**：复杂 JOIN 前先用 `/indexes` 查看索引
3. **理解 Schema**：通过 `/table_schema` 明确字段类型
4. **监控性能**：关注执行时间提示

### 元数据探索流程

1. 使用 `/databases` 查看可用数据库
2. 通过 `/database <name>` 切换目标数据库
3. 使用 `/schemas` 浏览 Schema
4. 用 `/tables` 列出表
5. 通过 `/table_schema <table>` 查看特定表结构
6. 使用 `/indexes <table>` 查看性能相关索引

### 与聊天能力结合

将 SQL 执行与聊天命令结合，效率更高：

```bash
# 先用聊天生成 SQL
/ Show me the top 10 customers by revenue

# 直接执行生成的 SQL
SELECT customer_name, SUM(order_total) as revenue
FROM customers c JOIN orders o ON c.id = o.customer_id
GROUP BY customer_name
ORDER BY revenue DESC
LIMIT 10;

# 随后注入上下文
/catalog customers
/ Now show me their contact information
```

