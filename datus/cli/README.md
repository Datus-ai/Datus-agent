# Datus-CLI User Manual

## Overview

Datus-CLI is a lightweight AI-powered SQL command-line interface designed for data engineers. It combines the power of AI with a modern TUI experience to enhance SQL development efficiency.

## Quick Start

```bash
# SQLite
#python cli/main.py --db_type sqlite --db_path benchmark/bird/dev_20240627/dev_databases/california_schools/california_schools.sqlite
python cli/main.py --namespace bird_sqlite  --config conf/agent.yml

# Snowflake
#python cli/main.py --db-type snowflake --sf-account your_account --sf-user your_user --sf-password your_password --sf-warehouse your_warehouse --sf-database your_database 
python cli/main.py --namespace spidersnow  --config conf/agent.yml
```
And then run sqls as you wish. 

## Examples to explore bird benchmark

```bash
python main.py bootstrap-kb --benchmark bird_dev --namespace bird_sqlite --kb_update_strategy overwrite

```

1. Natural Language Query:
```sql
Datus> !dastart
Creating a new SQL task
Enter task description (): How many schools in merged Alameda have number of test takers less than 100?
Enter database name (benchmark/bird/dev_20240627/dev_databases/california_schools/california_schools.sqlite): california_schools
Enter output directory (output): 
Enter external knowledge (optional) (): 
```

2. Table Exploration:
```sql
.tables
```

3. schema linking
```sql
Datus> !sl
Node Type: SCHEMA_LINKING
Node Input:
SchemaLinkingInput(
    input_text='How many schools in merged Alameda have number of test takers less than 100?',
    top_n=5,
    database_type='sqlite',
    database_name='california_schools',
    matching_rate='fast',
    sql_context=None
)
Do you want to execute this node? [y/n] (y): 
```

4. SQL Generation:
```sql
Datus> !gen
Node Type: GENERATE_SQL
Node Input:
GenerateSQLInput(
    database_type='sqlite',
    table_schemas=[
        TableSchema(
            table_name='schools',
            database_name='california_schools',
            schema_name='',
            schema_text="
... 670-4590,,www.acoe.org,1980-07-01,,0.0,,,00,County Office of Education (COE),10,County Community,COMM,County Community School,HS,High 
School,K-12,7-12,P,0.0,37.658212,-122.09713,Carolyn,Hobbs,chobbs@acoe.org,,,,,,,2016-10-06\n',
            created_at='2025-05-13T05:07:45.689786'
        )
    ],
    metrics=[],
    input_text='How many schools in merged Alameda have number of test takers less than 100?',
    contexts=[],
    external_knowledge=''
)
Do you want to execute this node? [y/n] (y): 
Executing GENERATE_SQL node...
Node Result:
GenerateSQLResult(
    success=True,
    error=None,
    sql_query="SELECT COUNT(*) AS NumberOfSchools FROM satscores WHERE cname = 'Alameda' AND NumTstTakr < 100 AND rtype = 'S';",
    tables=['satscores'],
    explanation="The question asks for the count of schools in Alameda with fewer than 100 test takers. The 'satscores' table contains the relevant data, including the county name 
('cname'), the number of test takers ('NumTstTakr'), and the record type ('rtype') which distinguishes between different types of records (e.g., 'S' for school). The query filters for 
records where the county is Alameda, the number of test takers is less than 100, and the record type is 'S' (school), then counts these records."
)
Context updated successfully
```

5. Run sql
```sql
Datus> !run
Running SQL: SELECT COUNT(*) AS NumberOfSchools FROM satscores WHERE cname = 'Alameda' AND NumTstTakr < 100 AND rtype = 'S';
┏━━━━━━━━━━━━━━━━━┓
┃ NumberOfSchools ┃
┡━━━━━━━━━━━━━━━━━┩
│ 45              │
└─────────────────┘
Returned 1 rows in 0.01 seconds
```

6. Chat with AI:
````sql
Datus> / Can you explain this query's performance?
# SQL Query Performance Explanation

The query you executed is performing well for several reasons:

## Efficient Query Structure
- **Simple COUNT operation**: Counting rows is generally efficient in SQLite as it can use indexes without retrieving full rows.
- **Clear filtering conditions**: The WHERE clause has three specific filters (`cname`, `NumTstTakr`, `rtype`).

## Performance Factors
1. **Index Utilization**: 
   - If there are indexes on `cname`, `NumTstTakr`, or `rtype`, the query will be very fast.
   - Without indexes, SQLite must scan the entire table but can still perform efficiently for moderate-sized tables.

2. **Filter Selectivity**:
   - The combination of filters likely reduces the result set significantly before counting.
   - `rtype = 'S'` probably filters to just schools (excluding districts or other types).

3. **Small Result Set**:
   - The query returns just one number (45), minimizing data transfer.

## Potential Improvements
- If this query runs frequently, ensure indexes exist on the filtered columns:
  ```sql
  CREATE INDEX idx_satscores_filter ON satscores(cname, NumTstTakr, rtype);
  ```
- For very large tables, consider materialized views if this count is needed often.

The query appears optimal for its purpose - it's simple, specific, and returns minimal data. The execution was successful with no errors reported.
````

7. End the workflow:
```sql
Datus> !save

Do you want to execute this node? [y/n] (y): 
Executing OUTPUT node...
Node Result:
OutputResult(success=True, error=None, output='02bd515a.csv', sql_query='', sql_result='')
Context updated successfully

Datus> !bash ls output
02bd515a.csv
Datus> !bash cat output/02bd515a.csv
...

Datus> !daend
Ending workflow session: <agent.workflow.Workflow object at 0x17f4af3b0>
```


## Command Types

Datus-CLI supports several types of commands, each with its own prefix:

### 1. Tool Commands (!)

Tool commands are used for AI-powered SQL generation and workflow execution:

| Command                | Description                                                        |
|------------------------|--------------------------------------------------------------------|
| `!darun <query>`       | Run a natural language query through the agentic way                |
| `!dastart <query>`     | Start a new workflow session with manual input                     |
| `!sl`                  | Schema linking: show list of recommended tables and values         |
| `!gen`                 | Generate SQL, optionally with table constraints                    |
| `!run`                 | Run the last generated SQL                                         |
| `!fix <description>`   | Fix the last SQL query                                             |
| `!reason`              | Run the full reasoning node to exploring                           |
| `!save`                | Save the last result to a file                                     |
| `!set <context_type>`  | Set the context type for the current workflow                      |
| &nbsp;&nbsp;&nbsp;&nbsp;`context_type: sql, lastsql, schema, schema_values, metrics, task` | |
| `!bash <command>`      | Execute a bash command (limited to safe commands)                  |
| `!daend`               | End the current agent session and save trajectory to file           |

### 2. Context Commands (@)

Context commands help explore database metadata:

| Command              | Description                          |
|----------------------|--------------------------------------|
| `@catalog`           | Display database catalogs            |
| `@subject`           | Display semantic models  and metrics |

### 3. Chat Commands (/)

Chat commands allow direct interaction with the AI:

| Command                | Description                        |
|------------------------|------------------------------------|
| `/<message>`           | Chat with the AI assistant          |

### 4. Internal Commands (.)

Internal commands for CLI control:

| Command                | Description                        |
|------------------------|------------------------------------|
| `.help`                | Display this help message           |
| `.exit`, `.quit`       | Exit the CLI                       |
| `.databases`           | List all databases                  |
| `.tables`              | List all tables                     |
| `.schemas table_name`  | Show schema information             |

## Features

### SQL Execution
- Interactive query input with syntax highlighting
- Multi-line SQL support
- Command history with search
- Tabular result display
- Streaming results for large datasets
- Query statistics and timing information

### Auto-completion
- SQL keyword completion
- Table and column name completion
- Function name completion
- Smart context-aware suggestions

### AI Integration
- Natural language to SQL conversion
- SQL query optimization
- Error analysis and fixing
- Intelligent table recommendations


## Tips

1. Use `!sl` to get table recommendations before writing complex queries
2. Use `@catalog` to explore database structure
3. Save successful queries and result with `!save` for future reference or result comparsion
