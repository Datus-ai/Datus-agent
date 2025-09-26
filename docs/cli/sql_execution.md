# SQL Execution

## 1. Overview

Besides the magic commands (`/`, `@`, `!`), Datus-CLI can also act as a traditional SQL client. You can directly run SQL queries and explore database metadata with built-in metadata commands (`.tables`, `.databases`, `.schemas`, etc.).

We use a set of built-in Metadata Commands to provide a unified way to access metadata across all databases — but the native commands like `SHOW DATABASES` or `DESCRIBE TABLES` are still available and can be used as well.

---

## 2. Basic Usage

### Run SQL

Execute SQL queries directly in the CLI:

```sql
Datus-sql> SELECT T2.Zip FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`District Name` = 'Fresno County Office of Education' AND T1.`Charter School (Y/N)` = 1
```

**Output:**
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

### Explore Metadata

Use the built-in metadata commands to explore your database structure:

```bash
.databases                       # List all databases
.database <database_name>        # Switch current database
.schemas                         # List all schemas (or show details)
.schema <schema_name>            # Switch current schema
.tables                          # List all tables in current schema
.table_schema <table_name>       # Show table DDL
.indexes <table_name>            # Show table indexes
```

Use these commands to quickly inspect your data environment while chatting or debugging SQL.

## 3. Advanced Features

### Cross-Database Compatibility

The metadata commands work consistently across different database types:

- **SQLite**: Local file databases
- **DuckDB**: Analytical workloads
- **Snowflake**: Cloud data warehouse
- **MySQL/PostgreSQL**: Traditional RDBMS
- **StarRocks**: Real-time analytics

### Rich Output Formatting

SQL results are displayed with:

- **Table formatting**: Clean, aligned columns with borders
- **Data type awareness**: Proper formatting for numbers, dates, strings
- **Performance metrics**: Execution time and row count
- **Error handling**: Clear error messages with suggestions

### Query History

- All executed queries are automatically saved
- Access previous queries with up/down arrow keys
- Query history persists across sessions
- Integration with the `@sql` context system

### Export Options

Results can be exported in various formats:

- **CSV**: For data analysis and reporting
- **JSON**: For API integration and data exchange
- **Parquet**: For analytical workloads
- **Excel**: For business reporting

## Best Practices

### Query Optimization

1. **Use LIMIT** for exploratory queries on large tables
2. **Index awareness**: Check `.indexes` before writing complex JOINs
3. **Schema understanding**: Use `.table_schema` to understand data types
4. **Performance monitoring**: Pay attention to execution times

### Metadata Exploration Workflow

1. Start with `.databases` to see available databases
2. Switch to target database with `.database <name>`
3. Explore schemas with `.schemas`
4. List tables with `.tables`
5. Examine specific table structure with `.table_schema <table>`
6. Check indexes for performance with `.indexes <table>`

### Integration with Chat

Combine SQL execution with chat commands for maximum productivity:

```bash
# Use chat to generate SQL
/ Show me the top 10 customers by revenue

# Execute the generated SQL directly
SELECT customer_name, SUM(order_total) as revenue
FROM customers c JOIN orders o ON c.id = o.customer_id
GROUP BY customer_name
ORDER BY revenue DESC
LIMIT 10;

# Follow up with context injection
@catalog customers
/ Now show me their contact information
```